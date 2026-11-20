#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Student model trainer for online knowledge distillation.

Real network backprop is performed through PyTorch (the student ONNX is
converted to a trainable ``torch.nn.Module`` via onnx2torch). When PyTorch /
onnx2torch are unavailable, or the student's decode format is unsupported, the
trainer falls back to a lightweight affine box-correction head (see
``online_adapter.py``).

The primary workflow:
1. Load student ONNX model
2. Run student inference on incoming frame
3. Compute distillation loss vs teacher predictions
4. If PyTorch backprop available: backpropagate and update weights
5. Export updated ONNX model on demand
"""

import collections
import copy
import logging
import os
import random
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime

from node.DLNode.object_detection.onnx_session_utils import make_session
from node.DLNode.object_detection.CustomONNX.custom_onnx import CustomONNX
from node.DLNode.online_training.agreement import RollingAgreement
from node.DLNode.online_training.distillation_loss import compute_distillation_score
from node.DLNode.online_training.online_adapter import BoxAffineAdapter
from node.DLNode.online_training.replay_buffer import ReservoirBuffer
from node.DLNode.online_training.torch_student import (
    TorchStudent,
    is_torch_backprop_available,
    is_format_supported as _torch_format_supported,
    probe_pytorch_compatibility,
)

logger = logging.getLogger(__name__)

# The student CNN is inference-only under plain onnxruntime, so the requested
# distillation loss is back-propagated through a small affine box-correction head
# (see online_adapter.py). The node learning-rate slider (~1e-5 .. 1e-2) is scaled
# into the normalised coordinate space the head works in so that improvement is
# actually visible within a reasonable number of frames.
_ADAPTER_LR_GAIN = 500.0
_ADAPTER_LR_MIN = 1e-3
_ADAPTER_LR_MAX = 0.5

# How many training frames between full mAP@0.5 evaluations on the replay
# buffer.  Evaluating every frame is too expensive; every 30 frames keeps the
# mAP display responsive without slowing down the pipeline.
_MAP_EVAL_INTERVAL = 30

# Agreement with the teacher is pooled over this many recent frames.
_AGREEMENT_WINDOW = 100

# With real network training, the original student is run again on one frame
# out of this many to tell whether the trained student is better than the
# one it started from (see ``StudentTrainer.metrics``).
_BASELINE_INTERVAL = 5


class StudentTrainer:
    """Manages the student model lifecycle: inference, scoring, and optional training.

    Parameters
    ----------
    model_path : str
        Path to the student ONNX model file.
    input_width : int
        Model input width.
    input_height : int
        Model input height.
    output_format : str
        Output format ('yolo11', 'yolox' or 'nanodet').
    num_classes : int
        Number of detection classes.
    learning_rate : float
        Learning rate for weight updates.
    providers : list[str], optional
        ONNX Runtime execution providers.
    train_scope : str
        When the PyTorch backprop path is active, which part of the network to
        train: ``'head'`` (detection heads only, default) or ``'all'`` (backbone
        and heads).
    """

    def __init__(
        self,
        model_path: str,
        input_width: int = 640,
        input_height: int = 640,
        output_format: str = "yolo11",
        num_classes: int = 80,
        learning_rate: float = 0.0001,
        score_threshold: float = 0.3,
        providers: Optional[List[str]] = None,
        train_scope: str = "head",
        min_teacher_confidence: float = 0.35,
        replay_buffer_size: int = 32,
        device: str = "cpu",
        backprop_depth: Optional[int] = None,
    ):
        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.model_path = model_path
        self.input_width = input_width
        self.input_height = input_height
        self.output_format = output_format
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.score_threshold = score_threshold
        self.providers = providers
        self.train_scope = train_scope
        self.backprop_depth = backprop_depth  # None = legacy train_scope behaviour
        self.min_teacher_confidence = float(min_teacher_confidence)
        self.replay_buffer_size = int(replay_buffer_size)
        self.device = device

        # Statistics
        self.frames_processed = 0
        self.total_score = 0.0
        self.current_score = 0.0
        self.best_score = 0.0
        # Distillation loss bookkeeping (lower is better). ``current_loss`` is the
        # requested set-based (DETR-style Hungarian) loss for the latest frame and
        # ``best_loss`` the lowest value seen so far. ``initial_loss`` is the very
        # first measured loss, used to report the student's improvement over time.
        self.current_loss = float('inf')
        self.best_loss = float('inf')
        self.initial_loss = None
        self.training_active = False
        self._last_loss = None
        self.last_train_loss = None
        self._nanodet_cls_pre_activated = None

        # Replay buffer (reservoir sampling + hard-example mining).
        # Stores (frame, teacher_bboxes, teacher_classes, teacher_scores)
        # payloads tagged with the distillation loss at insertion time.
        # Frames are stored (not blobs) so that mAP can re-run current
        # inference on buffered scenes.
        self._replay_buffer: ReservoirBuffer = ReservoirBuffer(self.replay_buffer_size)

        # mAP@0.5 tracking: recomputed every _MAP_EVAL_INTERVAL training steps.
        self._map_score: float = 0.0
        self._frames_since_map_update: int = 0

        # Agreement with the teacher on recent frames (see ``metrics``).
        self._agreement = RollingAgreement(_AGREEMENT_WINDOW)
        self._paired_agreement = RollingAgreement(_AGREEMENT_WINDOW)
        self._baseline_agreement = RollingAgreement(_AGREEMENT_WINDOW)
        self.last_inference_ms = None
        self.last_train_ms = None

        # Real, gradient-trained correction head for the requested loss. This is
        # what lets the student's *output* actually change (and improve) frame to
        # frame even without onnxruntime-training.
        self._adapter = BoxAffineAdapter(learning_rate=_ADAPTER_LR_MIN)
        self._adaptation_available = True

        # Load the student model for inference
        self._student_model = CustomONNX(
            model_path=model_path,
            input_width=input_width,
            input_height=input_height,
            output_format=output_format,
            num_classes=num_classes,
            providers=providers,
        )

        # Real network backprop path: convert the student ONNX into a trainable
        # PyTorch module so the requested distillation loss is back-propagated
        # through the actual backbone/heads (not only the correction head). This
        # is optional: when torch/onnx2torch are unavailable or the conversion
        # fails, we silently fall back to the affine adaptation head.
        self._torch = None
        self._torch_backprop = False

        # Probe compatibility first — provides a clear, actionable log message
        # regardless of the outcome so the user knows exactly why a particular
        # path was chosen.
        compat = probe_pytorch_compatibility(model_path, output_format)
        if compat['compatible']:
            logger.info(
                "[StudentTrainer] Compatibility probe: %s", compat['reason']
            )
        else:
            logger.warning(
                "[StudentTrainer] PyTorch backprop unavailable — %s",
                compat['reason'],
            )

        if is_torch_backprop_available() and _torch_format_supported(output_format):
            try:
                self._torch = TorchStudent(
                    model_path=model_path,
                    input_width=input_width,
                    input_height=input_height,
                    output_format=output_format,
                    num_classes=num_classes,
                    learning_rate=learning_rate,
                    train_scope=train_scope,
                    backprop_depth=backprop_depth,
                    device=device,
                )
                self._torch_backprop = True
                logger.info(
                    "[StudentTrainer] PyTorch backprop ENABLED — the student "
                    "network (%s) is trained with real gradient descent.",
                    train_scope,
                )
            except Exception as exc:
                logger.warning(
                    "[StudentTrainer] Could not enable PyTorch backprop (%s). "
                    "Falling back to the affine correction head.", exc,
                )
                self._torch = None
                self._torch_backprop = False
        elif is_torch_backprop_available():
            # PyTorch is installed but the student's decode format is not
            # supported by the differentiable TorchStudent path (e.g.
            # ``nanodet_multi``). Using it would yield no student boxes and no
            # weight updates, so keep ONNX inference (CustomONNX) and rely on the
            # affine-head path for learning instead.
            logger.info(
                "[StudentTrainer] PyTorch backprop not used for output_format "
                "'%s' (unsupported decode); using ONNX inference with "
                "affine-head updates.", output_format,
            )

        logger.info(
            f"[StudentTrainer] Initialized — model={os.path.basename(model_path)}, "
            f"input={input_width}x{input_height}, format={output_format}, "
            f"backprop={self.backprop_mode}"
        )

    @property
    def is_training_available(self) -> bool:
        """Whether a real weight/parameter update can be performed.

        True when the PyTorch backprop path is active (real network training) or
        via the built-in affine adaptation head (always the case).
        """
        return self._torch_backprop or self._adaptation_available

    @property
    def backprop_mode(self) -> str:
        """Human-readable description of the active learning path."""
        if self._torch_backprop:
            depth = getattr(self._torch, '_backprop_depth', -2)
            if depth == -1:
                return "pytorch-all"
            if depth >= 0:
                return f"pytorch-{depth}L"
            # _torch exists but depth attribute not yet set (should not happen)
            return f"pytorch-{self.train_scope}"
        return "affine-head"

    def set_backprop_depth(self, depth: int) -> None:
        """Change the backprop depth at runtime without reloading the model.

        Parameters
        ----------
        depth : int
            Number of trailing parameter tensors to train.
            ``-1`` trains all, ``0`` uses affine-head only.
        """
        self.backprop_depth = int(depth)
        if self._torch is not None:
            self._torch.set_backprop_depth(self.backprop_depth)

    @property
    def _network_update_count(self) -> int:
        """Number of real network backprop steps (PyTorch)."""
        if self._torch is not None:
            return int(self._torch.updates)
        return 0

    @property
    def _network_backprop_active(self) -> bool:
        """True when the real-network (PyTorch) backprop path is active.

        In that case ``infer`` already returns predictions from the updated
        network weights, so the affine correction head must not be applied on
        top of them.
        """
        return bool(self._torch_backprop)

    @property
    def improvement(self) -> float:
        """Absolute loss reduction since the first frame (>= 0 means better)."""
        if self.initial_loss is None or self.current_loss == float('inf'):
            return 0.0
        return max(0.0, float(self.initial_loss - self.current_loss))

    @property
    def improvement_pct(self) -> float:
        """Relative loss reduction since the first frame, in percent."""
        if self.initial_loss is None or self.initial_loss <= 0.0:
            return 0.0
        return 100.0 * self.improvement / float(self.initial_loss)

    @property
    def avg_score(self) -> float:
        """Average distillation score over all processed frames."""
        if self.frames_processed == 0:
            return 0.0
        return self.total_score / self.frames_processed

    def infer(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run student model inference on a frame.

        When the PyTorch backprop path is active, inference uses the *trained*
        torch weights. Otherwise the original onnxruntime session is used.

        Returns (bboxes, scores, class_ids).
        """
        if self._torch_backprop and self._torch is not None:
            try:
                return self._torch_infer(frame)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"[StudentTrainer] torch inference failed: {exc}")
        return self._original_infer(frame)

    def _original_infer(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detections of the network as loaded from the ONNX file (onnxruntime).

        A NanoDet output is decoded exactly as for the trained network, so
        that both can be compared: CustomONNX's own layout heuristic takes
        the official NanoDet-Plus export for another layout.
        """
        if self.output_format != "nanodet":
            return self._student_model(frame)
        model = self._student_model
        blob, ratio = model._preprocess(frame)
        raw = model.onnx_session.run(None, {model.input_name: blob})[0]
        return self._decode(raw, frame, ratio)

    def _torch_infer(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Inference through the trained torch weights, reusing CustomONNX decode."""
        blob, ratio = self._student_model._preprocess(frame)
        return self._decode(self._torch.forward_numpy(blob), frame, ratio)

    def _decode(self, raw, frame, ratio):
        """Detections from the raw output of the network."""
        orig_h, orig_w = self._frame_size(frame)
        if self.output_format == "yolox":
            return self._student_model._postprocess_yolox(raw, orig_w, orig_h, ratio)
        if self.output_format == "nanodet":
            # Force the exact channel layout / activation used by the
            # differentiable training decode (TorchStudent._decode_nanodet)
            # instead of relying on the statistical reg-first heuristic.
            # The heuristic inspects the *torch-converted* output and can
            # misdetect the layout, which scrambles the class/reg split and
            # floods the display with thousands of false-positive student
            # boxes (the "green blocks everywhere" symptom). Keeping the
            # inference layout in lock-step with the training decode avoids it.
            reg_first_override = bool(getattr(self._torch, "nanodet_reg_first", False))
            cls_pre_activated = self._infer_nanodet_cls_pre_activated(
                raw, reg_first_override
            )
            return self._student_model._postprocess_nanodet(
                raw, orig_w, orig_h,
                cls_pre_activated=cls_pre_activated,
                reg_first_override=reg_first_override,
            )
        return self._student_model._postprocess_yolo11(raw, orig_w, orig_h)

    def _infer_nanodet_cls_pre_activated(self, raw_output: np.ndarray, reg_first: bool) -> bool:
        """Detect whether NanoDet class channels are already sigmoid-activated.

        Uses the same bounded-range heuristic as CustomONNX nanodet_multi
        (scores in [0,1] => pre-activated), then caches the result per trainer.
        This keeps torch inference aligned with the existing NanoDet decode path.
        """
        cached = getattr(self, "_nanodet_cls_pre_activated", None)
        if cached is not None:
            return bool(cached)

        pre_activated = False
        output = np.squeeze(raw_output)
        num_classes = int(getattr(self, "num_classes", 0))
        if output.ndim == 2 and num_classes > 0 and output.shape[1] > num_classes:
            reg_channels = output.shape[1] - num_classes
            cls_block = output[:, reg_channels:] if reg_first else output[:, :num_classes]
            if cls_block.size > 0 and np.isfinite(cls_block).all():
                pre_activated = bool(cls_block.min() >= 0.0 and cls_block.max() <= 1.0)

        self._nanodet_cls_pre_activated = pre_activated
        return pre_activated

    def _update_last_loss(self, distillation: Dict) -> None:
        """Store the latest reported loss from the requested set-based metrics.

        Keeps a single source of truth for the loss across the inference and
        training paths.
        """
        self._last_loss = {
            'loss': distillation.get('loss', 0.0),
            'loss_box': distillation.get('loss_box', 0.0),
            'loss_class': distillation.get('loss_class', 0.0),
            'loss_iou': distillation.get('loss_iou', 0.0),
            'loss_cardinality': distillation.get('loss_cardinality', 0.0),
        }

    def train_step(
        self,
        frame: np.ndarray,
        teacher_bboxes: List,
        teacher_scores: List,
        teacher_class_ids: List,
        score_threshold: float = 0.3,
        teacher_valid: bool = True,
    ) -> Dict:
        """Perform one training step: inference + scoring + optional backprop.

        Parameters
        ----------
        frame : np.ndarray
            Input image (BGR).
        teacher_bboxes : list
            Teacher model bounding boxes [[x1,y1,x2,y2], ...].
        teacher_scores : list
            Teacher model confidence scores.
        teacher_class_ids : list
            Teacher model class IDs.
        score_threshold : float
            Minimum confidence to keep student predictions.
        teacher_valid : bool
            False when no teacher result is available for this frame (teacher
            not connected, or its result is stale).  The frame then does not
            count in the agreement metrics: an absent teacher is not a
            teacher that saw nothing.

        Returns
        -------
        dict with keys:
            - student_bboxes, student_scores, student_class_ids: student predictions
              **after** the correction head has been applied (so the values you
              see reflect the current trained state)
            - distillation: distillation score metrics (includes the requested
              set-based loss under the ``loss`` key, lower = closer to teacher)
            - training_step: True when a real parameter update was performed this
              frame (the correction head learned from the requested loss)
            - metrics: see :meth:`metrics`
        """
        # 1. Student inference (raw network output, before correction)
        started = time.perf_counter()
        s_bboxes, s_scores, s_class_ids = self.infer(frame)
        self.last_inference_ms = (time.perf_counter() - started) * 1000.0

        # Apply score threshold to student predictions
        if len(s_scores) > 0:
            mask = s_scores >= score_threshold
            s_bboxes = s_bboxes[mask]
            s_scores = s_scores[mask]
            s_class_ids = s_class_ids[mask]
        # Without network training, these are the detections of the original
        # student: the correction head is applied below
        original = (s_bboxes, s_scores, s_class_ids)

        # 1b. When training only the affine correction head (no real-network
        # backprop path), apply it so the returned predictions reflect what the
        # head has learned so far. With the PyTorch backprop path the network
        # weights themselves are updated, so ``infer`` already returns
        # the improved predictions and no extra correction is applied.
        frame_h, frame_w = self._frame_size(frame)
        if not self._network_backprop_active and len(s_bboxes) > 0:
            s_bboxes = self._adapter.apply(s_bboxes, frame_w, frame_h).astype(np.float32)

        # 2. Compute distillation score + the requested set-based loss
        distillation = compute_distillation_score(
            list(teacher_bboxes),
            list(teacher_scores),
            list(teacher_class_ids),
            s_bboxes.tolist() if len(s_bboxes) > 0 else [],
            s_scores.tolist() if len(s_scores) > 0 else [],
            s_class_ids.tolist() if len(s_class_ids) > 0 else [],
        )

        # The reported loss is the *requested* set-based distillation loss (single
        # source of truth, identical to the one shown by the IoU/Chart nodes), not
        # a separate approximation.
        self._update_last_loss(distillation)
        if teacher_valid:
            self._update_agreement(
                frame, teacher_bboxes, teacher_class_ids,
                (s_bboxes, s_scores, s_class_ids), original, score_threshold,
            )

        # 3. Update statistics
        self.frames_processed += 1
        self.current_score = distillation['score']
        self.total_score += self.current_score
        if self.current_score > self.best_score:
            self.best_score = self.current_score

        # Track the requested loss (lower = better student).
        self.current_loss = float(distillation.get('loss', 0.0))
        if self.initial_loss is None:
            self.initial_loss = self.current_loss
        if self.current_loss < self.best_loss:
            self.best_loss = self.current_loss

        # 4. Training step. ``training_step`` is True only when a real parameter
        # update actually happened this frame.
        training_performed = False
        self.last_train_loss = None
        started = time.perf_counter()
        if self.training_active:
            if self._torch_backprop and self._torch is not None:
                # Real network backprop: the requested distillation loss is
                # propagated through the student backbone/heads via PyTorch.
                training_performed = self._torch_train_step(
                    frame, teacher_bboxes, teacher_class_ids, frame_w, frame_h,
                    teacher_scores=teacher_scores,
                )
            else:
                # Fallback: sub-gradient descent on the affine correction head.
                training_performed = self._adapt_step(
                    frame_w, frame_h,
                    teacher_bboxes, teacher_class_ids,
                    s_bboxes, s_class_ids,
                )
        self.last_train_ms = (
            (time.perf_counter() - started) * 1000.0 if training_performed else 0.0
        )

        return {
            'student_bboxes': s_bboxes,
            'student_scores': s_scores,
            'student_class_ids': s_class_ids,
            'distillation': distillation,
            'training_step': training_performed,
            'metrics': self.metrics(),
        }

    def _update_agreement(self, frame, teacher_bboxes, teacher_class_ids,
                          student, original, score_threshold):
        """Score this frame against the teacher, before it is trained on."""
        self._agreement.add(teacher_bboxes, teacher_class_ids, *student)

        # Same frames for the student and for the one it started from, or the
        # comparison would only reflect how hard each set of frames is
        if self._network_backprop_active:
            if self.frames_processed % _BASELINE_INTERVAL:
                return
            if self._network_update_count > 0:
                try:
                    boxes, scores, class_ids = self._original_infer(frame)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug(f"[StudentTrainer] baseline inference failed: {exc}")
                    return
                if len(scores) > 0:
                    mask = scores >= score_threshold
                    boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]
                original = (boxes, scores, class_ids)
        self._paired_agreement.add(teacher_bboxes, teacher_class_ids, *student)
        self._baseline_agreement.add(teacher_bboxes, teacher_class_ids, *original)

    def metrics(self) -> Dict:
        """What the student is worth, measured against the teacher.

        Every frame is scored before it is trained on, so none of these
        figures is measured on a frame the student has already learned.

        precision / recall / f1 : agreement with the teacher over the last
            frames (same class, IoU >= 0.5).
        mean_iou : localisation quality of the detections that agree.
        baseline_f1 : the same F1 for the *original* student, on the same
            frames.
        f1_gain : ``f1 - baseline_f1`` on those frames; what training has
            brought so far.  Negative when training made the student worse.
        inference_ms / train_ms : duration of the student inference and of
            the training step for the last frame.

        Values are ``None`` until they have been measured.
        """
        result = self._agreement.result()
        trained = self._paired_agreement.result()["f1"]
        baseline = self._baseline_agreement.result()["f1"]
        result["baseline_f1"] = baseline
        result["f1_gain"] = None if baseline is None else trained - baseline
        result["inference_ms"] = self.last_inference_ms
        result["train_ms"] = self.last_train_ms
        return result

    def _torch_train_step(self, frame, teacher_bboxes, teacher_class_ids,
                          frame_w, frame_h, teacher_scores=None):
        """One real backprop step through the PyTorch student network.

        Applies a teacher-confidence quality filter before backpropagating:
        frames where the teacher is uncertain (mean confidence < threshold) are
        skipped so noisy pseudo-labels do not pollute the gradient.

        After training on the current frame, up to 2 additional steps are
        performed on frames sampled from the replay buffer with hard-example
        mining bias (harder frames are replayed more often) to prevent
        catastrophic forgetting when the scene changes.

        Returns True when at least one optimizer step updated the network weights.
        """
        # Teacher quality filter: skip low-confidence frames.
        if teacher_scores is not None and len(teacher_scores) > 0:
            mean_conf = float(np.mean(teacher_scores))
            if mean_conf < self.min_teacher_confidence:
                logger.debug(
                    "[StudentTrainer] Skipping torch train step — teacher mean "
                    "confidence %.3f < threshold %.3f.", mean_conf,
                    self.min_teacher_confidence,
                )
                return False

        try:
            blob, _ratio = self._student_model._preprocess(frame)
            loss_val = self._torch.train_step(
                blob, list(teacher_bboxes), list(teacher_class_ids),
                frame_w, frame_h,
                teacher_scores=list(teacher_scores) if teacher_scores is not None else None,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"[StudentTrainer] torch train step failed: {exc}")
            return False
        if loss_val is None:
            return False
        self.last_train_loss = float(loss_val)

        # Add this frame to the reservoir buffer tagged with the current loss
        # so hard-example mining can bias replay toward difficult frames.
        payload = (
            frame,
            list(teacher_bboxes),
            list(teacher_class_ids),
            list(teacher_scores) if teacher_scores is not None else [],
        )
        self._replay_buffer.add(payload, loss=self.last_train_loss)

        # Replay up to 2 past frames with hard-mining bias (70 % hard / 30 %
        # random) to reinforce knowledge of difficult scenes.
        if len(self._replay_buffer) > 1:
            replays = self._replay_buffer.sample(2, hard_mining_ratio=0.7)
            for r_frame, r_boxes, r_classes, r_scores in replays:
                try:
                    r_blob, _ = self._student_model._preprocess(r_frame)
                    r_h, r_w = r_frame.shape[:2]
                    self._torch.train_step(
                        r_blob, r_boxes, r_classes, r_w, r_h,
                        teacher_scores=r_scores if r_scores else None,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug(f"[StudentTrainer] replay step failed: {exc}")

        # Periodic mAP@0.5 re-evaluation on buffered frames.
        self._frames_since_map_update += 1
        if self._frames_since_map_update >= _MAP_EVAL_INTERVAL:
            self._map_score = self._compute_map_on_buffer()
            self._frames_since_map_update = 0

        return True

    @staticmethod
    def _frame_size(frame):
        """Return (height, width) of ``frame`` with a safe fallback."""
        if frame is not None and hasattr(frame, 'shape') and len(frame.shape) >= 2:
            return int(frame.shape[0]), int(frame.shape[1])
        return 1, 1

    def _compute_map_on_buffer(self) -> float:
        """Compute mAP@0.5 of the *current* student over all buffered frames.

        Runs inference on every frame stored in the replay buffer and compares
        the result against the buffered teacher annotations.  The evaluation is
        exact up to the current student weights (not stale predictions) and uses
        the same :func:`compute_map_at_50` implementation that is exported to
        callers.

        Returns 0.0 when the buffer is empty or inference fails.
        """
        from node.DLNode.online_training.distillation_loss import compute_map_at_50

        if len(self._replay_buffer) == 0:
            return 0.0

        t_boxes_list, s_boxes_list = [], []
        t_cls_list, s_cls_list = [], []

        for buf_frame, t_boxes, t_classes, _t_scores in self._replay_buffer:
            try:
                s_bboxes, s_scores, s_class_ids = self.infer(buf_frame)
                if len(s_scores) > 0:
                    mask = s_scores >= self.score_threshold
                    s_bboxes = s_bboxes[mask]
                    s_class_ids = s_class_ids[mask]
                t_boxes_list.append(list(t_boxes))
                s_boxes_list.append(s_bboxes.tolist() if len(s_bboxes) > 0 else [])
                t_cls_list.append(list(t_classes))
                s_cls_list.append(s_class_ids.tolist() if len(s_class_ids) > 0 else [])
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("[StudentTrainer] mAP eval step failed: %s", exc)

        if not t_boxes_list:
            return 0.0
        return compute_map_at_50(t_boxes_list, s_boxes_list, t_cls_list, s_cls_list)

    @property
    def map_score(self) -> float:
        """Cached mAP@0.5 on the replay buffer (recomputed every N frames)."""
        return self._map_score

    def _adapt_step(self, frame_w, frame_h, teacher_bboxes, teacher_class_ids,
                    student_bboxes, student_class_ids):
        """One sub-gradient descent step of the correction head (requested loss).

        Returns True when the head parameters were updated.
        """
        # Scale the user learning rate into the head's normalised space so the
        # improvement is visible within a reasonable number of frames.
        eff_lr = float(np.clip(
            self.learning_rate * _ADAPTER_LR_GAIN,
            _ADAPTER_LR_MIN, _ADAPTER_LR_MAX,
        ))
        updated, _loss_before = self._adapter.update(
            teacher_bboxes, student_bboxes, frame_w, frame_h,
            teacher_classes=teacher_class_ids,
            student_classes=student_class_ids,
            learning_rate=eff_lr,
        )
        return bool(updated)

    def reset(self):
        """Reset the student model to its original weights."""
        # The model file is never modified by training: reloading it is enough
        try:
            self._student_model = CustomONNX(
                model_path=self.model_path,
                input_width=self.input_width,
                input_height=self.input_height,
                output_format=self.output_format,
                num_classes=self.num_classes,
                providers=self.providers,
            )
            logger.info("[StudentTrainer] Model reset to original weights.")
        except Exception as exc:
            logger.error(f"[StudentTrainer] Reset failed: {exc}", exc_info=True)

        # Reset stats
        self.frames_processed = 0
        self.total_score = 0.0
        self.current_score = 0.0
        self.best_score = 0.0
        self.current_loss = float('inf')
        self.best_loss = float('inf')
        self.initial_loss = None
        self.last_train_loss = None
        # Reset mAP tracking.
        self._map_score = 0.0
        self._frames_since_map_update = 0
        self._agreement.clear()
        self._paired_agreement.clear()
        self._baseline_agreement.clear()
        # Restore the correction head to the identity (no learned correction).
        self._adapter.reset()
        # Clear the replay buffer so stale frames don't influence the fresh student.
        self._replay_buffer.clear()
        # Restore the trained PyTorch network weights to their original state.
        if self._torch is not None:
            try:
                self._torch.reset()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"[StudentTrainer] torch reset failed: {exc}")

    def export_onnx(self, output_path: str) -> str:
        """Export the current student model to an ONNX file.

        Parameters
        ----------
        output_path : str
            Destination path for the exported ONNX file.

        Returns
        -------
        str : The path where the model was saved.
        """
        # With the PyTorch backprop path the trained weights live in the torch
        # module, so export them (re-serialise to ONNX) to capture the learning.
        if self._torch_backprop and self._torch is not None:
            try:
                self._torch.export_onnx(output_path)
                logger.info(
                    f"[StudentTrainer] Exported trained PyTorch student to: {output_path}"
                )
                return output_path
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    f"[StudentTrainer] torch ONNX export failed ({exc}); "
                    "falling back to the original model bytes."
                )
        import shutil
        shutil.copy2(self.model_path, output_path)
        logger.info(f"[StudentTrainer] Exported student model to: {output_path}")
        return output_path

    def get_stats(self) -> Dict:
        """Get training statistics.

        Keys
        ----
        current_score / avg_score / best_score : distillation *score* in [0, 1]
            (higher = student agrees more with the teacher).
        current_loss / best_loss : requested set-based distillation *loss*
            (lower = student closer to the teacher). ``inf`` until the first frame.
        improvement / improvement_pct : loss reduction since the first frame
            (absolute and percentage; > 0 means the student got better).
        adapter_updates : number of correction-head gradient steps performed.
        network_updates : number of real network backprop steps performed.
        backprop_mode : active learning path ('pytorch-head'/'pytorch-all'/
            'affine-head').
        train_loss : differentiable training loss of the last network backprop.
        training_active / training_available : training flags.
        """
        return {
            'frames_processed': self.frames_processed,
            'current_score': self.current_score,
            'avg_score': self.avg_score,
            'best_score': self.best_score,
            'current_loss': self.current_loss,
            'best_loss': self.best_loss,
            'improvement': self.improvement,
            'improvement_pct': self.improvement_pct,
            'adapter_updates': self._adapter.updates,
            'network_updates': self._network_update_count,
            'backprop_mode': self.backprop_mode,
            'train_loss': self.last_train_loss,
            'training_active': self.training_active,
            'training_available': self.is_training_available,
            'map_score': self._map_score,
            'replay_buffer_size': len(self._replay_buffer),
        }
