#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the YOLOX distillation engine (node/DLNode/online_training/distill)."""
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

from node.DLNode.online_training.distill import students, yolox_ops
from node.DLNode.online_training.distill.engine import (
    DistillConfig,
    DistillationEngine,
    should_deploy,
)
from node.DLNode.online_training.distill.store import SampleStore, pack_sample

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS = os.path.join(_ROOT, "node", "DLNode", "object_detection", "YOLOX", "model")
_NANO = os.path.join(_MODELS, "yolox_nano.onnx")
_TINY = os.path.join(_MODELS, "yolox_tiny.onnx")
_VIDEO = os.path.join(
    _ROOT, "node", "DLNode", "semantic_segmentation",
    "road_segmentation_adas_0001", "road_sample.mp4",
)


def test_teacher_cells_are_mapped_to_the_student_cell_that_contains_them():
    teacher_cells, teacher_strides = yolox_ops.make_grid(640, 640)
    student_cells, student_strides = yolox_ops.make_grid(416, 416)
    scale = 416 / 640

    index = yolox_ops.cell_map((640, 640), (416, 416), scale)

    assert (index >= 0).all()
    # Same pyramid level, and the teacher cell centre lies in the student cell
    assert np.array_equal(student_strides[index], teacher_strides)
    centre = (teacher_cells + 0.5) * teacher_strides[:, None] * scale
    assert np.array_equal(np.floor(centre / teacher_strides[:, None]), student_cells[index])
    # Every student cell is taught by one to four teacher cells
    per_cell = np.bincount(index, minlength=len(student_strides))
    assert per_cell.min() >= 1 and per_cell.max() <= 4
    # Student image in the corner of the teacher input: one teacher cell per
    # student cell, the same one
    corner = yolox_ops.cell_map((640, 640), (416, 416), 1.0)
    inside = corner >= 0
    assert np.array_equal(np.sort(corner[inside]), np.arange(len(student_strides)))
    assert np.array_equal(teacher_cells[inside], student_cells[corner[inside]])


def test_projection_keeps_the_most_confident_teacher_cell_and_rescales_boxes():
    index = np.array([0, 0, 1, -1])
    boxes = np.array([[0, 0, 10, 10], [20, 20, 40, 40], [50, 50, 60, 60], [1, 1, 2, 2]], dtype=np.float32)
    objectness = np.array([0.2, 0.9, 0.0, 0.99], dtype=np.float32)
    class_scores = np.array([[1, 0], [0, 1], [0.5, 0.5], [1, 1]], dtype=np.float32)

    out_box, out_obj, out_cls = yolox_ops.project_dense(
        boxes, objectness, class_scores, index, anchors=3, scale=0.5
    )

    assert out_obj.tolist() == pytest.approx([0.9, 0.0, 0.0])
    assert out_cls[0].tolist() == [0, 1]
    assert out_box[0].tolist() == [10, 10, 20, 20]
    # A teacher cell outside the student grid teaches nothing
    assert out_obj[2] == 0 and not out_cls[2].any()


def test_flip_index_mirrors_the_columns_of_every_level():
    cells, strides = yolox_ops.make_grid(64, 96)
    index = yolox_ops.flip_index(64, 96)

    assert np.array_equal(index[index], np.arange(len(index)))
    assert np.array_equal(cells[index][:, 0], 96 / strides - 1 - cells[:, 0])
    assert np.array_equal(cells[index][:, 1], cells[:, 1])
    assert np.array_equal(strides[index], strides)


def test_detections_nms_is_class_aware_and_rescales_boxes():
    boxes = np.array(
        [[10, 10, 110, 110], [12, 12, 112, 112], [10, 10, 110, 110], [200, 200, 220, 220]],
        dtype=np.float32,
    )
    objectness = np.array([0.9, 0.8, 0.9, 0.9], dtype=np.float32)
    class_scores = np.zeros((4, 3), dtype=np.float32)
    class_scores[0, 1] = class_scores[1, 1] = 1.0  # same class, overlapping
    class_scores[2, 2] = 1.0                       # other class, same place
    class_scores[3, 0] = 0.2                       # below the threshold

    out_boxes, scores, class_ids = yolox_ops.detections(
        boxes, objectness, class_scores, score_th=0.3, nms_th=0.45, ratio=0.5
    )

    assert sorted(class_ids.tolist()) == [1, 2]
    assert np.allclose(sorted(scores.tolist()), [0.9, 0.9])
    assert np.allclose(out_boxes, [[20, 20, 220, 220]] * 2)


def test_detection_meter_counts_class_and_overlap():
    reference = np.array([[0, 0, 10, 10], [50, 50, 80, 80]], dtype=np.float32)

    perfect = yolox_ops.DetectionMeter()
    perfect.add(reference, [1, 2], reference, [0.9, 0.8], [1, 2])
    assert perfect.result()["f1"] == pytest.approx(1.0)
    assert perfect.result()["ap50"] == pytest.approx(1.0)

    flawed = yolox_ops.DetectionMeter()
    # Right box with the wrong class, and a duplicate of the first object
    flawed.add(reference, [1, 2], np.vstack([reference, reference[:1]]), [0.9, 0.8, 0.7], [1, 1, 1])
    result = flawed.result()
    assert (result["precision"], result["recall"]) == (pytest.approx(1 / 3), pytest.approx(0.5))

    empty = yolox_ops.DetectionMeter()
    empty.add(np.zeros((0, 4)), [], np.zeros((0, 4)), [], [])
    assert empty.result()["f1"] == 1.0


def _sample(seed=0, anchors=84, classes=3):
    rng = np.random.default_rng(seed)
    canvas = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    jpeg = cv2.imencode(".jpg", canvas)[1].tobytes()
    objectness = np.zeros(anchors, dtype=np.float32)
    objectness[[3, 40]] = [0.9, 0.2]
    return pack_sample(
        jpeg, rng.random((anchors, 4), dtype=np.float32), objectness,
        rng.random((anchors, classes), dtype=np.float32), 0.5,
    )


def test_store_validates_on_the_most_recent_samples_and_survives_a_restart(tmp_path):
    signature = {"teacher": "t", "classes": 3}
    store = SampleStore(str(tmp_path / "s"), signature, max_train=12, val_window=4)

    for i in range(8):
        store.add(_sample(i), priority=0.5)
    # A quarter of the samples at most, always the latest ones
    assert store.val_names() == sorted(store.val_names()) and len(store.val_names()) == 2
    assert max(store.train_names()) < min(store.val_names())
    first_window = store.val_names()

    for i in range(8, 30):
        store.add(_sample(i), priority=0.5)

    # Yesterday's validation samples are now trained on; the window moved on
    assert len(store.val_names()) == 4 and len(store.train_names()) == 12
    assert not set(first_window) & set(store.val_names())
    assert max(store.train_names()) < min(store.val_names())
    assert set(store.sample_train(20)) <= set(store.train_names())

    loaded = store.load(store.train_names()[0])
    assert loaded["canvas"].shape == (64, 64, 3)
    # Class scores and boxes are only kept where the teacher sees something
    assert loaded["pos"].tolist() == [3, 40] and loaded["cls"].shape == (2, 3)

    reopened = SampleStore(str(tmp_path / "s"), signature, max_train=12, val_window=4)
    assert reopened.val_names() == store.val_names()
    reopened.add(_sample(99))
    assert reopened.val_names()[-1] > store.val_names()[-1]
    assert len(reopened.train_names()) == 12

    # Samples made with another teacher must not be reused
    other = SampleStore(str(tmp_path / "s"), {"teacher": "other", "classes": 3})
    assert not other.train_names() and not other.val_names()


def test_a_student_is_deployed_only_when_it_is_better_on_validation():
    deployed = {"loss": 1.0, "ap50": 0.60}

    assert should_deploy({"loss": 0.9, "ap50": 0.62}, deployed)
    # Within the noise: not worth replacing the model in production
    assert not should_deploy({"loss": 0.995, "ap50": 0.70}, deployed)
    # A lower loss must not hide worse detections
    assert not should_deploy({"loss": 0.8, "ap50": 0.50}, deployed)
    assert not should_deploy({"loss": float("nan"), "ap50": 0.9}, deployed)
    assert not should_deploy({"loss": float("inf"), "ap50": 0.0}, deployed)


def test_student_catalogue_offers_small_students_built_from_the_shipped_files(tmp_path):
    from node.DLNode.online_training.distill.yolox_onnx import YoloxOnnx

    models = students.find_yolox_models()
    catalogue = students.student_catalogue(models)
    assert list(catalogue) == ["yolox_nano_320", "yolox_nano", "yolox_tiny_320", "yolox_tiny"]
    # The catalogue comes first, other YOLOX files stay selectable
    assert students.student_choices(models)[:4] == list(catalogue)
    assert "yolox_s" in students.student_choices(models)

    cache = str(tmp_path / "students")
    for key, spec in catalogue.items():
        path = students.resolve_student(key, cache, models)
        student = YoloxOnnx(path)
        assert (student.height, student.width, student.num_classes) == (spec.size, spec.size, 80)
        assert spec.native == (path == models[spec.base])
        assert spec.native or path.startswith(cache)

    # Resized variants are generated once and reused
    path = students.resolve_student("yolox_nano_320", cache, models)
    stamp = os.path.getmtime(path)
    assert students.resolve_student("yolox_nano_320", cache, models) == path
    assert os.path.getmtime(path) == stamp
    # Plain files are accepted by their name, unknown keys are refused
    assert students.resolve_student("yolox_s", cache, models) == models["yolox_s"]
    with pytest.raises(ValueError, match="yolox_nano_320"):
        students.resolve_student("yolox_xxl", cache, models)


def test_resizing_keeps_the_network_and_only_changes_the_declared_shapes():
    import onnx

    data = students.resize_yolox_onnx(_NANO, 320, 352)

    original, resized = onnx.load(_NANO), onnx.load_from_string(data)
    assert [n.op_type for n in resized.graph.node] == [n.op_type for n in original.graph.node]
    assert len(resized.graph.initializer) == len(original.graph.initializer)
    dims = [d.dim_value for d in resized.graph.input[0].type.tensor_type.shape.dim]
    assert dims == [1, 3, 320, 352]
    anchors = len(yolox_ops.make_grid(320, 352)[1])
    out = [d.dim_value for d in resized.graph.output[0].type.tensor_type.shape.dim]
    assert out == [1, anchors, 85]
    with pytest.raises(ValueError, match="multiple of 32"):
        students.resize_yolox_onnx(_NANO, 300, 300)


# ---------------------------------------------------------------------- #
# Tests below train a real model: they need PyTorch and onnx2torch.


@pytest.fixture(scope="module")
def student():
    pytest.importorskip("torch")
    pytest.importorskip("onnx2torch")
    from node.DLNode.online_training.distill.yolox_torch import TrainableYolox

    return TrainableYolox(_NANO)


def test_training_does_not_depend_on_torchvision_operators(student):
    """Packaged builds cannot always load them; YOLOX does not use them."""
    code = (
        "import sys; sys.modules['torchvision'] = None\n"
        "from node.DLNode.online_training.distill import yolox_torch\n"
        "assert yolox_torch.backend_status() == (True, ''), yolox_torch.backend_status()\n"
        f"yolox_torch.TrainableYolox({_NANO!r}).export()\n"
        "assert 'torchvision' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=_ROOT, capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr[-2000:]


def _own_predictions_as_target(student, blob):
    import torch

    with torch.no_grad():
        boxes, obj, cls = student.predict(blob)
    probabilities = torch.sigmoid(obj)
    image, anchor = torch.nonzero(probabilities >= 0.05, as_tuple=True)
    return {
        "obj": probabilities,
        "pos_image": image,
        "pos_anchor": anchor,
        "cls": torch.sigmoid(cls[image, anchor]),
        "box": boxes[image, anchor],
    }


def _gradient_norm(student, blob, target, gamma):
    import torch
    from node.DLNode.online_training.distill.yolox_torch import distillation_loss

    parameters = student.trainable_parameters(1.0)
    boxes, obj, cls = student.predict(blob)
    loss, _ = distillation_loss(boxes, obj, cls, target, gamma=gamma)
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    return float(sum((g ** 2).sum() for g in gradients if g is not None) ** 0.5), float(loss)


def _frames(count, step=12):
    capture = cv2.VideoCapture(_VIDEO)
    frames = []
    while len(frames) < count:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
        for _ in range(step - 1):
            capture.grab()
    capture.release()
    assert len(frames) == count
    return frames


def test_a_student_identical_to_its_teacher_is_left_unchanged(student):
    """The loss must be stationary when there is nothing to learn."""
    blob = yolox_ops.to_blob(yolox_ops.letterbox(_frames(1)[0], student.height, student.width)[0])
    target = _own_predictions_as_target(student, blob)
    shifted = dict(target, obj=(target["obj"] * 0.5), box=target["box"] + 8.0)

    for gamma in (0.0, 2.0):
        at_teacher, loss_at_teacher = _gradient_norm(student, blob, target, gamma)
        away, loss_away = _gradient_norm(student, blob, shifted, gamma)
        assert at_teacher < 1e-3 * away
        assert loss_away > loss_at_teacher
    # With the focal factor the loss itself vanishes
    assert loss_at_teacher < 1e-6


def test_trained_weights_are_written_back_into_the_original_graph(student, tmp_path):
    import onnx
    import torch
    from node.DLNode.online_training.distill.yolox_onnx import YoloxOnnx

    state = student.state()
    torch.manual_seed(0)
    changed = {k: v + 0.02 * v.abs().mean() * torch.randn_like(v) for k, v in state.items()}

    data = student.export(changed)  # checked against PyTorch by export()

    original = onnx.load(_NANO)
    exported = onnx.load_from_string(data)
    assert [n.op_type for n in exported.graph.node] == [n.op_type for n in original.graph.node]
    assert exported.graph.output[0] == original.graph.output[0]
    assert all(torch.equal(v, changed[k]) for k, v in student.onnx_state(data).items())

    canvas = yolox_ops.letterbox(_frames(1)[0], student.height, student.width)[0]
    before = YoloxOnnx(_NANO).raw(canvas)
    after = YoloxOnnx(data).raw(canvas)
    assert before.shape == after.shape and not np.allclose(before, after, atol=1e-3)
    # export() must leave the model being trained untouched
    assert all(torch.equal(v, state[k]) for k, v in student.state().items())


def test_engine_collects_trains_deploys_and_resumes(student, tmp_path):
    config = DistillConfig(
        device="cpu", min_train=8, min_val=2, steps_per_round=6, batch_size=2,
        learning_rate=5e-5, keep_easy=1.0, min_disagreement=0.0,
        # Deploy whatever is trained: this test is about the mechanics
        min_gain=-10.0, ap_tolerance=10.0,
    )
    workdir = str(tmp_path / "run")
    engine = DistillationEngine(_TINY, _NANO, workdir, config)
    frames = _frames(40)
    original = engine.detect(frames[0])

    assert engine.train_round() is None  # nothing collected yet
    assert all(engine.collect(frame) for frame in frames)
    status = engine.status()
    assert (status["train_samples"], status["val_samples"]) == (30, 10)

    record = engine.train_round()

    assert record["deployed_now"] and engine.version == 1
    assert record["steps"] == 6 and np.isfinite(record["train_loss"])
    assert record["candidate"]["images"] == 10
    assert os.path.isfile(engine.deployed_path)
    trained = engine.detect(frames[0])
    assert trained[0].shape[1] == 4 and len(trained[0]) == len(trained[1]) == len(trained[2])

    # A new engine on the same directory continues with the deployed student
    resumed = DistillationEngine(_TINY, _NANO, workdir, config)
    assert resumed.version == 1 and len(resumed.history()) == 1
    assert len(resumed.store.train_names()) == 30
    for a, b in zip(resumed.detect(frames[0]), trained):
        assert np.array_equal(a, b)

    resumed.reset()
    assert resumed.version == 0 and not os.path.isfile(resumed.deployed_path)
    for a, b in zip(resumed.detect(frames[0]), original):
        assert np.array_equal(a, b)


def test_a_diverging_round_never_reaches_production(student, tmp_path):
    config = DistillConfig(
        device="cpu", min_train=8, min_val=2, steps_per_round=4, batch_size=2,
        learning_rate=5.0, warmup_steps=1, keep_easy=1.0, min_disagreement=0.0,
    )
    engine = DistillationEngine(_TINY, _NANO, str(tmp_path / "run"), config)
    frames = _frames(40)
    for frame in frames:
        engine.collect(frame)
    before = engine.detect(frames[0])

    record = engine.train_round()

    assert not record["deployed_now"] and record["rolled_back"]
    assert engine.version == 0 and not os.path.isfile(engine.deployed_path)
    for a, b in zip(engine.detect(frames[0]), before):
        assert np.array_equal(a, b)
    # Training restarts from the deployed weights, with a smaller step
    assert record["lr"] == 5.0 and engine._trainer.lr == 2.5
    state = engine._trainer.model.state()
    assert all(np.array_equal(v, engine._trainer.deployed[k]) for k, v in state.items())


def test_a_resized_catalogue_student_is_trained_and_deployed_like_any_other(student, tmp_path):
    """The 320 px students only exist as generated files: the whole chain
    (onnxruntime inference, onnx2torch conversion, export) must accept them."""
    from node.DLNode.online_training.distill.yolox_onnx import YoloxOnnx

    models = students.find_yolox_models()
    path = students.resolve_student("yolox_nano_320", str(tmp_path / "students"), models)
    config = DistillConfig(
        device="cpu", min_train=8, min_val=2, steps_per_round=4, batch_size=2,
        keep_easy=1.0, min_disagreement=0.0, min_gain=-10.0, ap_tolerance=10.0,
    )
    engine = DistillationEngine(_TINY, path, str(tmp_path / "run"), config)
    assert (engine.height, engine.width, engine.num_classes) == (320, 320, 80)

    frames = _frames(40)
    for frame in frames:
        engine.collect(frame)
    record = engine.train_round()

    assert record["deployed_now"] and engine.version == 1
    exported = str(tmp_path / "student.onnx")
    engine.export(exported)
    deployed = YoloxOnnx(exported)
    assert (deployed.height, deployed.width) == (320, 320)
    boxes, scores, class_ids = deployed.detect(frames[0])
    assert len(boxes) == len(scores) == len(class_ids)
