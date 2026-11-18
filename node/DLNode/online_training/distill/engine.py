#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Online distillation of a large YOLOX teacher into a small YOLOX student.

Three activities run independently (see docs/DISTILLATION_YOLOX.md):

* **inference**: ``detect()`` runs the deployed student with onnxruntime.  It
  never waits for the two others;
* **collection**: a few times per second, a frame is shown to the teacher and
  its dense predictions are stored on disk with the image;
* **training**: in the background, the student is trained on the stored
  samples by rounds.  After each round the trained student is compared with
  the deployed one on the most recent samples, which neither has been
  trained on, and replaces it only if it is better.

The engine can be driven by its two threads (``start()``, ``submit()``) or
step by step (``collect()``, ``train_round()``), which is what the command
line tool and the tests do.
"""
import copy
import dataclasses
import hashlib
import json
import logging
import os
import queue
import threading
import time

import cv2
import numpy as np

from node.DLNode.online_training.distill import yolox_ops
from node.DLNode.online_training.distill.store import SampleStore, pack_sample
from node.DLNode.online_training.distill.yolox_onnx import YoloxOnnx, default_providers

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DistillConfig:
    """Settings of the engine; ``0`` means "chosen from the hardware"."""

    device: str = "auto"            # 'auto', 'cpu' or 'cuda'
    use_gpu_inference: bool = True  # teacher / student onnxruntime sessions
    cpu_threads: int = 0            # threads used by training and by the teacher

    # Collection
    teacher_view: str = "student"   # 'student': same pixels as the student; 'full'
    sample_interval: float = 0.5    # seconds between two frames shown to the teacher
    min_disagreement: float = 0.03  # below: the student already agrees with the teacher
    keep_easy: float = 0.25         # share of "agreed" frames kept anyway
    max_train: int = 4000
    val_window: int = 96            # most recent samples, kept out of training

    # Training
    min_train: int = 48             # samples needed before the first round
    min_val: int = 12
    min_new_samples: int = 8        # new samples needed to train again after a failed round
    batch_size: int = 0
    steps_per_round: int = 0
    learning_rate: float = 5e-5
    warmup_steps: int = 20
    trainable_fraction: float = 0.3  # share of the layers trained, from the output
    focal_gamma: float = 2.0
    box_weight: float = 5.0
    ema_decay: float = 0.99
    augment: bool = True

    # Deployment gate
    min_gain: float = 0.01          # relative validation loss gain required to deploy
    ap_tolerance: float = 0.02      # AP50 drop tolerated when the loss improves
    patience: int = 3               # rounds without progress before rolling back
    keep_checkpoints: int = 5

    # Detection
    score_th: float = 0.3
    nms_th: float = 0.45
    seed: int = 0


def disagreement(teacher_obj, teacher_cls, student_obj, student_cls):
    """How differently teacher and student score the anchors, in [0, 1)."""
    teacher = teacher_obj * teacher_cls.max(axis=1)
    student = student_obj * student_cls.max(axis=1)
    return float(
        np.abs(teacher - student).sum() / (1.0 + np.maximum(teacher, student).sum())
    )


def should_deploy(candidate, deployed, min_gain=0.01, ap_tolerance=0.02):
    """Is the trained student better than the deployed one on validation data?

    The distillation loss is the main criterion: it is measured on every
    anchor of every validation image, so it is far less noisy than a count of
    detections.  AP50 against the teacher detections is a safeguard: a lower
    loss must not come with visibly worse detections.
    """
    if not np.isfinite(candidate["loss"]):
        return False
    return (
        candidate["loss"] <= deployed["loss"] * (1.0 - min_gain)
        and candidate["ap50"] >= deployed["ap50"] - ap_tolerance
    )


def _lower_thread_priority():
    """Run the calling thread below the interface and the pipeline.

    Collection and training must only use the processor time nobody else
    wants.  Best effort: silently does nothing where it is not permitted.
    """
    try:
        if os.name == "nt":
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.GetCurrentThread.restype = ctypes.c_void_p
            kernel32.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
            kernel32.SetThreadPriority(kernel32.GetCurrentThread(), -1)  # BELOW_NORMAL
        else:
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 5)
    except Exception:
        pass


def _atomic_write(path, data):
    tmp = path + ".tmp"
    with open(tmp, "wb") as fp:
        fp.write(data)
    os.replace(tmp, path)


class _Trainer:
    """Mutable training state: model, optimizer, averaged and deployed weights."""

    def __init__(self, model, config, device):
        import torch

        self.torch = torch
        self.model = model
        self.device = device
        self.base_lr = self.lr = config.learning_rate
        self.parameters = model.trainable_parameters(config.trainable_fraction)
        self.deployed = model.state()
        self.ema = copy.deepcopy(self.deployed)
        self.ema_updates = 0
        self.steps = 0
        self.stale_rounds = 0
        self.new_optimizer()

    def new_optimizer(self):
        self.optimizer = self.torch.optim.AdamW(self.parameters, lr=self.lr, weight_decay=0.0)

    def update_ema(self, decay):
        # Short warm-up so that the average follows the first steps
        self.ema_updates += 1
        decay = min(decay, (1.0 + self.ema_updates) / (10.0 + self.ema_updates))
        for name, parameter in self.model.module.named_parameters():
            if name in self.ema:
                self.ema[name].mul_(decay).add_(parameter.detach().cpu(), alpha=1.0 - decay)

    def restart_from_deployed(self):
        self.model.load_state(self.deployed)
        self.ema = copy.deepcopy(self.deployed)
        self.ema_updates = 0
        self.stale_rounds = 0
        self.new_optimizer()


class DistillationEngine:
    """Distils ``teacher_path`` into ``student_path``, both YOLOX ONNX files.

    Parameters
    ----------
    teacher_path, student_path : str
        YOLOX models exported without box decoding, with the same classes.
        The teacher input must be at least as large as the student input.
    workdir : str
        Where samples, checkpoints and the deployed student are kept.  An
        engine created on an existing ``workdir`` resumes from it.
    """

    def __init__(self, teacher_path, student_path, workdir, config=None):
        self.config = config or DistillConfig()
        self.teacher_path = teacher_path
        self.student_path = student_path
        self.workdir = workdir
        os.makedirs(os.path.join(workdir, "checkpoints"), exist_ok=True)

        self._providers = default_providers(self.config.use_gpu_inference)
        self._threads = self.config.cpu_threads or max(1, (os.cpu_count() or 2) // 2)
        with open(student_path, "rb") as fp:
            self._original = fp.read()

        self._lock = threading.Lock()
        self._rng = np.random.default_rng(self.config.seed)
        self._student = YoloxOnnx(self._original, self._providers)
        self.height, self.width = self._student.height, self._student.width
        self.num_classes = self._student.num_classes
        self._flip = yolox_ops.flip_index(self.height, self.width)

        self._version = 0
        self._history = []
        self._deployed_bytes = self._original
        self._resume()

        self.store = SampleStore(
            os.path.join(workdir, "samples"),
            signature={
                "teacher": os.path.basename(teacher_path),
                "teacher_bytes": os.path.getsize(teacher_path),
                "student_input": [self.height, self.width],
                "teacher_view": self.config.teacher_view,
                "classes": self.num_classes,
            },
            max_train=self.config.max_train,
            val_window=self.config.val_window,
            seed=self.config.seed,
        )
        self._store_lock = threading.Lock()

        self._teacher = None
        self._cell_maps = {}
        self._trainer = None
        self._last_thumb = None
        self._deployed_metrics = (None, None)
        self._trained_on = 0

        self._status = {
            "phase": "idle", "message": "", "device": "",
            "train_loss": None, "candidate": None, "deployed": None,
        }
        self._enabled = False
        self._reset_requested = False
        self._stop = threading.Event()
        self._frames = queue.Queue(maxsize=1)
        self._last_submit = 0.0
        self._threads_started = []

    # ------------------------------------------------------------------ #
    # Deployed student

    @property
    def _state_path(self):
        return os.path.join(self.workdir, "state.json")

    @property
    def deployed_path(self):
        return os.path.join(self.workdir, "deployed.onnx")

    def _resume(self):
        """Continue with the student deployed by a previous run, if any."""
        if not os.path.isfile(self.deployed_path):
            return
        try:
            with open(self.deployed_path, "rb") as fp:
                data = fp.read()
            student = YoloxOnnx(data, self._providers)
            if (student.height, student.width, student.num_classes) != (
                self.height, self.width, self.num_classes
            ):
                raise ValueError("deployed student does not match the student model")
            with open(self._state_path) as fp:
                state = json.load(fp)
            self._student, self._deployed_bytes = student, data
            self._version = int(state["version"])
            self._history = list(state["history"])
        except Exception as exc:
            logger.warning("distillation: cannot resume from %s (%s)", self.workdir, exc)

    def _save_state(self):
        data = json.dumps({"version": self._version, "history": self._history[-100:]}, indent=1)
        _atomic_write(self._state_path, data.encode())

    def detect(self, image, score_th=None, nms_th=None):
        """Detections of the deployed student: ``(boxes, scores, class_ids)``."""
        with self._lock:
            student = self._student
        return student.detect(
            image,
            self.config.score_th if score_th is None else score_th,
            self.config.nms_th if nms_th is None else nms_th,
        )

    def export(self, path):
        """Write the deployed student to ``path`` (a standard YOLOX ONNX file)."""
        with self._lock:
            data = self._deployed_bytes
        _atomic_write(path, data)
        return path

    @property
    def version(self):
        """Number of students deployed since the original one."""
        return self._version

    # ------------------------------------------------------------------ #
    # Collection

    def _ensure_teacher(self):
        if self._teacher is None:
            teacher = YoloxOnnx(self.teacher_path, self._providers, threads=self._threads)
            if teacher.num_classes != self.num_classes:
                raise ValueError(
                    f"teacher has {teacher.num_classes} classes, student {self.num_classes}"
                )
            if teacher.height < self.height or teacher.width < self.width:
                raise ValueError(
                    f"the teacher input ({teacher.width}x{teacher.height}) must be at "
                    f"least as large as the student input ({self.width}x{self.height})"
                )
            self._teacher = teacher
        return self._teacher

    def _teach(self, teacher, image, canvas, ratio):
        """Dense teacher predictions on the student grid.

        ``teacher_view`` selects what the teacher looks at:

        * ``'student'``: the very pixels given to the student, placed in the
          corner of the teacher input.  Each student anchor then has one
          teacher anchor looking at the same place at the same scale: the
          comparison is exact, and everything the teacher finds can be found
          by the student;
        * ``'full'``: the image at the teacher's own, higher resolution.  The
          teacher sees small objects better, but several of its cells fall
          into each student cell and part of what it finds is out of reach at
          the student resolution.
        """
        if (teacher.height, teacher.width) == (self.height, self.width):
            return teacher.dense(canvas)
        if self.config.teacher_view == "full":
            teacher_canvas, teacher_ratio = yolox_ops.letterbox(
                image, teacher.height, teacher.width
            )
            scale = ratio / teacher_ratio
        else:
            teacher_canvas = np.full(
                (teacher.height, teacher.width, 3), yolox_ops.PAD_VALUE, dtype=np.uint8
            )
            teacher_canvas[: self.height, : self.width] = canvas
            scale = 1.0
        key = round(scale, 5)
        if key not in self._cell_maps:
            self._cell_maps[key] = yolox_ops.cell_map(
                (teacher.height, teacher.width), (self.height, self.width), scale
            )
        return yolox_ops.project_dense(
            *teacher.dense(teacher_canvas), self._cell_maps[key],
            len(self._student.strides), scale,
        )

    def collect(self, image):
        """Show ``image`` to the teacher and store the sample if it is useful.

        Returns whether the sample was stored.
        """
        teacher = self._ensure_teacher()
        with self._lock:
            student = self._student

        canvas, ratio = yolox_ops.letterbox(image, self.height, self.width)
        # The student must be trained on the pixels that are stored
        _, jpeg = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        canvas = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
        boxes, objectness, class_scores = self._teach(teacher, image, canvas, ratio)
        _, student_obj, student_cls = student.dense(canvas)
        gap = disagreement(objectness, class_scores, student_obj, student_cls)

        thumb = cv2.resize(
            cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), (32, 32), interpolation=cv2.INTER_AREA
        ).astype(np.float32)
        agreed = gap < self.config.min_disagreement
        unchanged = (
            self._last_thumb is not None and np.abs(thumb - self._last_thumb).mean() < 1.5
        )
        # Nothing to learn from: the scene did not change, or the student
        # already agrees (a share of those is kept as easy examples)
        if agreed and (unchanged or self._rng.random() >= self.config.keep_easy):
            return False
        self._last_thumb = thumb
        with self._store_lock:
            self.store.add(
                pack_sample(jpeg.tobytes(), boxes, objectness, class_scores, ratio),
                priority=gap,
            )
        return True

    # ------------------------------------------------------------------ #
    # Training

    def _ensure_trainer(self):
        if self._trainer is None:
            from node.DLNode.online_training.distill import yolox_torch

            available, reason = yolox_torch.backend_status()
            if not available:
                raise RuntimeError(reason)
            import torch

            device = yolox_torch.pick_device(self.config.device)
            if device == "cpu":
                torch.set_num_threads(self._threads)
            model = yolox_torch.TrainableYolox(self.student_path, device)
            if self._deployed_bytes != self._original:
                model.load_state(model.onnx_state(self._deployed_bytes))
            self._trainer = _Trainer(model, self.config, device)
            self._set_status(device=device)
        return self._trainer

    def _hardware_defaults(self, device):
        on_gpu = str(device).startswith("cuda")
        return (
            self.config.batch_size or (16 if on_gpu else 4),
            self.config.steps_per_round or (200 if on_gpu else 40),
        )

    def _load(self, names):
        samples = []
        for name in names:
            try:
                samples.append(self.store.load(name))
            except (OSError, ValueError, KeyError):
                continue  # replaced by the collector in the meantime
        return samples

    def _augment(self, sample):
        """Flip and photometric changes; the teacher targets stay valid."""
        canvas, obj, pos, box = sample["canvas"], sample["obj"], sample["pos"], sample["box"]
        rng = self._rng
        if rng.random() < 0.5:
            canvas = canvas[:, ::-1]
            obj = obj[self._flip]
            pos = self._flip[pos]
            box = np.stack(
                [self.width - box[:, 2], box[:, 1], self.width - box[:, 0], box[:, 3]], axis=1
            )
        if rng.random() < 0.8:
            pixels = canvas.astype(np.float32)
            pixels = pixels * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25)
            pixels = pixels * rng.uniform(0.92, 1.08, size=3)
            if rng.random() < 0.2:
                pixels = pixels + rng.normal(0, rng.uniform(2, 8), pixels.shape)
            canvas = np.clip(pixels, 0, 255).astype(np.uint8)
        if rng.random() < 0.15:
            canvas = cv2.GaussianBlur(np.ascontiguousarray(canvas), (0, 0), rng.uniform(0.5, 1.5))
        return {**sample, "canvas": np.ascontiguousarray(canvas), "obj": obj, "pos": pos, "box": box}

    def _batch(self, samples, augment):
        torch = self._trainer.torch
        device = self._trainer.model.device
        if augment:
            samples = [self._augment(sample) for sample in samples]
        blob = torch.from_numpy(yolox_ops.to_blob([s["canvas"] for s in samples]))
        target = {
            "obj": torch.from_numpy(np.stack([s["obj"] for s in samples])),
            "pos_image": torch.from_numpy(
                np.concatenate([np.full(len(s["pos"]), i) for i, s in enumerate(samples)])
            ).long(),
            "pos_anchor": torch.from_numpy(np.concatenate([s["pos"] for s in samples])).long(),
            "cls": torch.from_numpy(np.concatenate([s["cls"] for s in samples])),
            "box": torch.from_numpy(np.concatenate([s["box"] for s in samples])),
        }
        return blob, {key: value.to(device) for key, value in target.items()}

    def _evaluate(self, state, names):
        """Validation loss and detection agreement with the teacher."""
        from node.DLNode.online_training.distill.yolox_torch import distillation_loss

        trainer, config = self._trainer, self.config
        torch = trainer.torch
        current = trainer.model.state()
        trainer.model.load_state(state)
        meter = yolox_ops.DetectionMeter()
        total, count = 0.0, 0
        batch_size = self._hardware_defaults(trainer.device)[0] * 2
        try:
            with torch.no_grad():
                for start in range(0, len(names), batch_size):
                    samples = self._load(names[start:start + batch_size])
                    if not samples:
                        continue
                    blob, target = self._batch(samples, augment=False)
                    boxes, obj, cls = trainer.model.predict(blob)
                    loss, _ = distillation_loss(
                        boxes, obj, cls, target, config.focal_gamma, config.box_weight
                    )
                    total += float(loss) * len(samples)
                    count += len(samples)
                    boxes = boxes.cpu().numpy()
                    obj = torch.sigmoid(obj).cpu().numpy()
                    cls = torch.sigmoid(cls).cpu().numpy()
                    for i, sample in enumerate(samples):
                        ref = yolox_ops.detections(
                            sample["box"], sample["obj"][sample["pos"]], sample["cls"],
                            config.score_th, config.nms_th,
                        )
                        det = yolox_ops.detections(
                            boxes[i], obj[i], cls[i], config.score_th, config.nms_th
                        )
                        meter.add(ref[0], ref[2], det[0], det[1], det[2])
        finally:
            trainer.model.load_state(current)
        result = meter.result()
        result["loss"] = total / count if count else float("inf")
        result["images"] = count
        return result

    def ready_to_train(self):
        with self._store_lock:
            return (
                len(self.store.train_names()) >= self.config.min_train
                and len(self.store.val_names()) >= self.config.min_val
            )

    def train_round(self, should_stop=lambda: False):
        """Train for one round, then deploy the result if it is better.

        Returns the record of the round (also appended to the history), or
        ``None`` when there are not enough samples yet.
        """
        from node.DLNode.online_training.distill.yolox_torch import distillation_loss

        if not self.ready_to_train():
            return None
        trainer, config = self._ensure_trainer(), self.config
        torch = trainer.torch
        batch_size, steps = self._hardware_defaults(trainer.device)
        started = time.time()
        with self._store_lock:
            self._trained_on = self.store.added
            # Fixed for the whole round: what is trained on now must not be
            # what the result is judged on
            val_names = self.store.val_names()
            train_count = len(self.store.train_names())

        held_out = set(val_names)
        self._set_status(phase="training")
        losses = []
        for _ in range(steps):
            if should_stop():
                break
            with self._store_lock:
                names = [
                    name for name in self.store.sample_train(batch_size)
                    if name not in held_out
                ]
            samples = self._load(names)
            if not samples:
                continue
            blob, target = self._batch(samples, config.augment)
            warmup = min(1.0, (trainer.steps + 1) / max(1, config.warmup_steps))
            for group in trainer.optimizer.param_groups:
                group["lr"] = trainer.lr * warmup
            boxes, obj, cls = trainer.model.predict(blob)
            loss, _ = distillation_loss(
                boxes, obj, cls, target, config.focal_gamma, config.box_weight
            )
            if not torch.isfinite(loss):
                losses = [float("inf")]
                break
            trainer.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.parameters, 10.0)
            trainer.optimizer.step()
            trainer.steps += 1
            trainer.update_ema(config.ema_decay)
            losses.append(float(loss.detach()))
            self._set_status(train_loss=float(np.mean(losses[-20:])))

        if should_stop():
            # Interrupted: the weights trained so far are kept for the next
            # round, nothing is evaluated or deployed now
            self._set_status(phase="idle")
            return None

        self._set_status(phase="evaluating")
        counts = (train_count, len(val_names))
        key = (self._version, hashlib.sha1("".join(val_names).encode()).hexdigest())
        if self._deployed_metrics[0] != key:
            self._deployed_metrics = (key, self._evaluate(trainer.deployed, val_names))
        deployed = self._deployed_metrics[1]
        diverged = not losses or not np.isfinite(losses).all()
        candidate = (
            {"loss": float("inf"), "ap50": 0.0, "f1": 0.0}
            if diverged else self._evaluate(trainer.ema, val_names)
        )

        record = {
            "round": len(self._history) + 1,
            "steps": trainer.steps,
            "train_loss": float(np.mean(losses)) if losses else None,
            "candidate": candidate,
            "deployed": deployed,
            "deployed_now": False,
            "rolled_back": False,
            "train": counts[0],
            "val": counts[1],
            "lr": trainer.lr,
        }
        if should_deploy(candidate, deployed, config.min_gain, config.ap_tolerance):
            self._deploy(trainer, candidate)
            record["deployed_now"] = True
        else:
            # A lower loss held back by the AP safeguard is still progress:
            # training goes on from where it is
            improving = candidate["loss"] <= deployed["loss"] * (1.0 - config.min_gain)
            trainer.stale_rounds = 0 if improving else trainer.stale_rounds + 1
            worse = candidate["loss"] > deployed["loss"] * 1.1
            if diverged or worse or trainer.stale_rounds >= config.patience:
                # Training went nowhere: start again from what is deployed,
                # more carefully
                trainer.lr = max(trainer.lr / 2.0, trainer.base_lr / 8.0)
                trainer.restart_from_deployed()
                record["rolled_back"] = True
        record["version"] = self._version
        record["seconds"] = round(time.time() - started, 1)
        self._history.append(record)
        self._save_state()
        self._set_status(
            phase="idle", candidate=candidate, deployed=self._deployed_metrics[1]
        )
        return record

    def _deploy(self, trainer, metrics):
        data = trainer.model.export(trainer.ema)
        student = YoloxOnnx(data, self._providers)
        version = self._version + 1
        checkpoints = os.path.join(self.workdir, "checkpoints")
        _atomic_write(os.path.join(checkpoints, f"v{version:05d}.onnx"), data)
        for old in sorted(os.listdir(checkpoints))[: -self.config.keep_checkpoints]:
            os.remove(os.path.join(checkpoints, old))
        _atomic_write(self.deployed_path, data)
        with self._lock:
            self._student, self._deployed_bytes, self._version = student, data, version
        trainer.deployed = copy.deepcopy(trainer.ema)
        trainer.stale_rounds = 0
        # Progress again: undo part of the caution taken after a rollback
        trainer.lr = min(trainer.base_lr, trainer.lr * 2.0)
        self._deployed_metrics = (None, metrics)
        logger.info(
            "distillation: student v%d deployed (val loss %.4f, AP50 %.3f)",
            version, metrics["loss"], metrics["ap50"],
        )

    def reset(self):
        """Go back to the original student.  Collected samples are kept."""
        if self._threads_started:
            self._reset_requested = True
        else:
            self._do_reset()

    def _do_reset(self):
        self._reset_requested = False
        for name in ("deployed.onnx", "state.json"):
            path = os.path.join(self.workdir, name)
            if os.path.isfile(path):
                os.remove(path)
        checkpoints = os.path.join(self.workdir, "checkpoints")
        for name in os.listdir(checkpoints):
            os.remove(os.path.join(checkpoints, name))
        student = YoloxOnnx(self._original, self._providers)
        with self._lock:
            self._student, self._deployed_bytes, self._version = student, self._original, 0
        self._history = []
        self._deployed_metrics = (None, None)
        self._trainer = None
        self._set_status(phase="idle", candidate=None, deployed=None, train_loss=None, message="")

    # ------------------------------------------------------------------ #
    # Background operation

    def start(self):
        if self._threads_started:
            return
        self._stop.clear()
        for target, name in ((self._collector_loop, "collect"), (self._trainer_loop, "train")):
            thread = threading.Thread(target=target, name=f"distill-{name}", daemon=True)
            thread.start()
            self._threads_started.append(thread)

    def stop(self, timeout=5.0):
        """Stop the background threads and wait for them.

        They must not be running when the interpreter exits.
        """
        self._stop.set()
        for thread in self._threads_started:
            thread.join(timeout)
        self._threads_started = []

    def set_training(self, enabled):
        """Enable or pause collection and training (inference always runs)."""
        self._enabled = bool(enabled)

    def submit(self, image):
        """Offer a live frame; returns immediately.  Most frames are ignored."""
        now = time.monotonic()
        if not self._enabled or now - self._last_submit < self.config.sample_interval:
            return
        try:
            self._frames.put_nowait(image.copy())
            self._last_submit = now
        except queue.Full:
            pass

    def _set_status(self, **values):
        with self._lock:
            self._status.update(values)

    def status(self):
        with self._lock:
            status = dict(self._status)
        with self._store_lock:
            status["train_samples"] = len(self.store.train_names())
            status["val_samples"] = len(self.store.val_names())
        status["version"] = self._version
        status["rounds"] = len(self._history)
        status["enabled"] = self._enabled
        return status

    def history(self):
        return list(self._history)

    def _collector_loop(self):
        _lower_thread_priority()
        while not self._stop.is_set():
            try:
                frame = self._frames.get(timeout=0.2)
            except queue.Empty:
                continue
            if not self._enabled:
                continue
            try:
                self.collect(frame)
            except ValueError as exc:
                # Incompatible models: retrying cannot help
                logger.error("distillation: %s", exc)
                self._enabled = False
                self._set_status(phase="error", message=str(exc))
            except Exception as exc:
                logger.exception("distillation: collection failed")
                self._set_status(phase="error", message=f"collection: {exc}")
                self._stop.wait(5.0)

    def _has_new_samples(self):
        with self._store_lock:
            new = self.store.added - self._trained_on
        last = self._history[-1] if self._history else None
        return last is None or last["deployed_now"] or new >= self.config.min_new_samples

    def _trainer_loop(self):
        def interrupted():
            return self._stop.is_set() or not self._enabled or self._reset_requested

        _lower_thread_priority()
        while not self._stop.is_set():
            try:
                if self._reset_requested:
                    self._do_reset()
                if not self._enabled or not self.ready_to_train() or not self._has_new_samples():
                    if self._status["phase"] not in ("error", "unavailable"):
                        self._set_status(phase="collecting" if self._enabled else "idle")
                    self._stop.wait(0.5)
                    continue
                self.train_round(should_stop=interrupted)
            except RuntimeError as exc:
                from node.DLNode.online_training.distill import yolox_torch

                if not yolox_torch.backend_status()[0]:
                    self._set_status(phase="unavailable", message=str(exc))
                    return
                logger.exception("distillation: training failed")
                self._set_status(phase="error", message=f"training: {exc}")
                self._stop.wait(10.0)
            except Exception as exc:
                logger.exception("distillation: training failed")
                self._set_status(phase="error", message=f"training: {exc}")
                self._stop.wait(10.0)
