#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for online training distillation loss module."""

import os

import cv2
import numpy as np
import pytest

from node.DLNode.online_training import student_models_registry
from node.DLNode.online_training.agreement import RollingAgreement, count_matches
from node.DLNode.online_training.student_trainer import StudentTrainer
from node.DLNode.online_training.distillation_loss import (
    compute_iou,
    match_detections,
    compute_distillation_score,
    hungarian_match_boxes,
    compute_set_distillation_loss,
    compute_map_at_50,
    _class_distribution_similarity,
    _count_ratio_score,
    _confidence_alignment,
    _spatial_coverage_score,
)
from node.DLNode.online_training.replay_buffer import ReservoirBuffer


class TestComputeIoU:
    def test_perfect_overlap(self):
        box = [0, 0, 100, 100]
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        box_a = [0, 0, 50, 50]
        box_b = [100, 100, 200, 200]
        assert compute_iou(box_a, box_b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        box_a = [0, 0, 100, 100]
        box_b = [50, 50, 150, 150]
        # Intersection: 50x50 = 2500, Union: 10000 + 10000 - 2500 = 17500
        assert compute_iou(box_a, box_b) == pytest.approx(2500.0 / 17500.0, rel=1e-3)

    def test_contained_box(self):
        box_a = [0, 0, 100, 100]
        box_b = [25, 25, 75, 75]
        # Intersection: 50x50 = 2500, Union: 10000 + 2500 - 2500 = 10000
        assert compute_iou(box_a, box_b) == pytest.approx(2500.0 / 10000.0)

    def test_zero_area_box(self):
        box_a = [0, 0, 0, 0]
        box_b = [0, 0, 100, 100]
        assert compute_iou(box_a, box_b) == pytest.approx(0.0)


class TestMatchDetections:
    def test_empty_teacher(self):
        matched, unmatched_t, unmatched_s = match_detections(
            [], [], [[0, 0, 50, 50]], [0.9]
        )
        assert matched == []
        assert unmatched_t == []
        assert unmatched_s == [0]

    def test_empty_student(self):
        matched, unmatched_t, unmatched_s = match_detections(
            [[0, 0, 50, 50]], [0.9], [], []
        )
        assert matched == []
        assert unmatched_t == [0]
        assert unmatched_s == []

    def test_perfect_match(self):
        boxes = [[0, 0, 100, 100]]
        scores = [0.9]
        matched, unmatched_t, unmatched_s = match_detections(
            boxes, scores, boxes, scores, iou_threshold=0.5
        )
        assert len(matched) == 1
        assert matched[0] == (0, 0)
        assert unmatched_t == []
        assert unmatched_s == []

    def test_no_match_low_iou(self):
        t_boxes = [[0, 0, 50, 50]]
        s_boxes = [[200, 200, 300, 300]]
        matched, unmatched_t, unmatched_s = match_detections(
            t_boxes, [0.9], s_boxes, [0.8], iou_threshold=0.5
        )
        assert matched == []
        assert unmatched_t == [0]
        assert unmatched_s == [0]


class TestClassDistributionSimilarity:
    def test_both_empty(self):
        assert _class_distribution_similarity([], []) == pytest.approx(1.0)

    def test_one_empty(self):
        assert _class_distribution_similarity([0, 1], []) == pytest.approx(0.0)

    def test_identical(self):
        assert _class_distribution_similarity([0, 1, 1], [0, 1, 1]) == pytest.approx(1.0)

    def test_same_distribution_different_counts(self):
        # Same proportions: 1 class-0, 2 class-1 vs 2 class-0, 4 class-1
        sim = _class_distribution_similarity([0, 1, 1], [0, 0, 1, 1, 1, 1])
        assert sim == pytest.approx(1.0, abs=0.01)

    def test_completely_different(self):
        sim = _class_distribution_similarity([0, 0, 0], [1, 1, 1])
        assert sim == pytest.approx(0.0)


class TestCountRatioScore:
    def test_equal_counts(self):
        assert _count_ratio_score(5, 5) == pytest.approx(1.0)

    def test_one_zero(self):
        assert _count_ratio_score(5, 0) == pytest.approx(0.0)

    def test_both_zero(self):
        assert _count_ratio_score(0, 0) == pytest.approx(1.0)

    def test_different_counts(self):
        assert _count_ratio_score(2, 4) == pytest.approx(0.5)
        assert _count_ratio_score(4, 2) == pytest.approx(0.5)

    def test_close_counts(self):
        assert _count_ratio_score(9, 10) == pytest.approx(0.9)


class TestConfidenceAlignment:
    def test_both_empty(self):
        assert _confidence_alignment([], []) == pytest.approx(1.0)

    def test_one_empty(self):
        assert _confidence_alignment([0.9], []) == pytest.approx(0.0)

    def test_identical(self):
        assert _confidence_alignment([0.9, 0.8], [0.9, 0.8]) == pytest.approx(1.0)

    def test_different_means(self):
        # Mean diff of 0.5 → mean_score = 0.5
        score = _confidence_alignment([0.9], [0.4])
        assert score < 0.8


class TestSpatialCoverageScore:
    def test_both_empty(self):
        assert _spatial_coverage_score([], []) == pytest.approx(1.0)

    def test_one_empty(self):
        assert _spatial_coverage_score([[0, 0, 50, 50]], []) == pytest.approx(0.0)

    def test_identical_boxes(self):
        boxes = [[0, 0, 100, 100]]
        assert _spatial_coverage_score(boxes, boxes) == pytest.approx(1.0)

    def test_non_overlapping(self):
        t_boxes = [[0, 0, 30, 30]]
        s_boxes = [[70, 70, 100, 100]]
        score = _spatial_coverage_score(t_boxes, s_boxes)
        assert score < 0.5


class TestDistillationScore:
    def test_both_empty(self):
        result = compute_distillation_score([], [], [], [], [], [])
        assert result['score'] == pytest.approx(1.0)

    def test_teacher_empty_student_has_detections(self):
        result = compute_distillation_score(
            [], [], [],
            [[0, 0, 50, 50]], [0.9], [0],
        )
        assert result['score'] == pytest.approx(0.0)
        assert result['student_count'] == 1

    def test_student_empty_teacher_has_detections(self):
        result = compute_distillation_score(
            [[0, 0, 50, 50]], [0.9], [0],
            [], [], [],
        )
        assert result['score'] == pytest.approx(0.0)
        assert result['teacher_count'] == 1

    def test_perfect_match(self):
        boxes = [[0, 0, 100, 100]]
        scores = [0.95]
        class_ids = [0]
        result = compute_distillation_score(
            boxes, scores, class_ids,
            boxes, scores, class_ids,
        )
        assert result['score'] > 0.9
        assert result['class_similarity'] == pytest.approx(1.0)
        assert result['count_ratio'] == pytest.approx(1.0)
        assert result['confidence_alignment'] == pytest.approx(1.0)
        assert result['spatial_coverage'] == pytest.approx(1.0)

    def test_different_counts_same_region(self):
        """Teacher has 2 detections, student has 1, but same class and region."""
        t_boxes = [[0, 0, 100, 100], [10, 10, 90, 90]]
        t_scores = [0.9, 0.8]
        t_classes = [0, 0]

        s_boxes = [[5, 5, 95, 95]]
        s_scores = [0.85]
        s_classes = [0]

        result = compute_distillation_score(
            t_boxes, t_scores, t_classes,
            s_boxes, s_scores, s_classes,
        )
        # Should still get a reasonable score because:
        # - same class distribution
        # - similar spatial coverage
        # - close confidences
        # - count ratio = 0.5 (penalty but not zero)
        assert result['score'] > 0.5
        assert result['class_similarity'] == pytest.approx(1.0)
        assert result['count_ratio'] == pytest.approx(0.5)
        assert result['spatial_coverage'] > 0.5

    def test_different_classes_penalized(self):
        """Same region and count but different classes → lower score."""
        boxes = [[0, 0, 100, 100]]
        result = compute_distillation_score(
            boxes, [0.9], [0],
            boxes, [0.9], [1],
        )
        assert result['score'] < 0.9
        assert result['class_similarity'] == pytest.approx(0.0)

    def test_many_student_few_teacher(self):
        """Student has many more detections than teacher."""
        t_boxes = [[50, 50, 150, 150]]
        t_scores = [0.9]
        t_classes = [0]

        s_boxes = [[0, 0, 50, 50], [50, 50, 100, 100], [100, 100, 200, 200],
                   [200, 200, 300, 300], [300, 300, 400, 400]]
        s_scores = [0.8, 0.7, 0.6, 0.5, 0.4]
        s_classes = [0, 0, 1, 1, 2]

        result = compute_distillation_score(
            t_boxes, t_scores, t_classes,
            s_boxes, s_scores, s_classes,
        )
        # Count ratio = 1/5 = 0.2, class similarity low → score should be low
        assert result['score'] < 0.5
        assert result['count_ratio'] == pytest.approx(0.2)


class TestHungarianMatchBoxes:
    def test_empty_sets(self):
        pairs, cost = hungarian_match_boxes([], [])
        assert pairs == []
        assert cost.shape == (0, 0)

    def test_one_empty(self):
        pairs, _ = hungarian_match_boxes([[0, 0, 10, 10]], [])
        assert pairs == []

    def test_optimal_assignment(self):
        # A clear best assignment: A0↔B1 (identical), A1↔B0 (identical).
        boxes_a = [[0, 0, 10, 10], [100, 100, 110, 110]]
        boxes_b = [[100, 100, 110, 110], [0, 0, 10, 10]]
        pairs, _ = hungarian_match_boxes(boxes_a, boxes_b)
        assert set(pairs) == {(0, 1), (1, 0)}

    def test_class_cost_influences_match(self):
        # Two overlapping candidates; class cost breaks the tie toward the
        # same-class student box.
        boxes_a = [[0, 0, 10, 10]]
        boxes_b = [[0, 0, 10, 10], [0, 0, 10, 10]]
        pairs, _ = hungarian_match_boxes(
            boxes_a, boxes_b, classes_a=[5], classes_b=[9, 5]
        )
        assert pairs == [(0, 1)]

    def test_returns_min_count_pairs(self):
        boxes_a = [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]]
        boxes_b = [[0, 0, 10, 10], [20, 20, 30, 30]]
        pairs, _ = hungarian_match_boxes(boxes_a, boxes_b)
        assert len(pairs) == 2


class TestComputeSetDistillationLoss:
    def test_both_empty(self):
        r = compute_set_distillation_loss([], [])
        assert r['loss'] == pytest.approx(0.0)
        assert r['loss_cardinality'] == pytest.approx(0.0)
        assert r['cardinality_error'] == 0

    def test_identical_sets_zero_loss(self):
        boxes = [[0, 0, 10, 10], [50, 50, 70, 70]]
        classes = [0, 1]
        r = compute_set_distillation_loss(boxes, boxes, classes, classes)
        assert r['loss'] == pytest.approx(0.0, abs=1e-6)
        assert r['loss_box'] == pytest.approx(0.0, abs=1e-6)
        assert r['loss_iou'] == pytest.approx(0.0, abs=1e-6)
        assert r['loss_cardinality'] == pytest.approx(0.0)
        assert r['loss_class'] == pytest.approx(0.0)
        assert r['fp_count'] == 0
        assert r['fn_count'] == 0
        assert r['iou_mean_matched'] == pytest.approx(1.0)

    def test_cardinality_is_absolute_count_difference(self):
        t = [[0, 0, 10, 10], [50, 50, 60, 60], [80, 80, 90, 90]]
        s = [[0, 0, 10, 10]]
        r = compute_set_distillation_loss(t, s)
        # |1 - 3| = 2
        assert r['loss_cardinality'] == pytest.approx(2.0)
        assert r['cardinality_error'] == 2
        # 2 teacher boxes go unmatched → false negatives.
        assert r['fn_count'] == 2
        assert r['fp_count'] == 0

    def test_false_positive_counted(self):
        t = [[0, 0, 10, 10]]
        s = [[0, 0, 10, 10], [500, 500, 510, 510]]
        r = compute_set_distillation_loss(t, s)
        assert r['fp_count'] == 1
        assert r['fn_count'] == 0
        # The extra student box has no teacher overlap → loss_fp ≈ 1.
        assert r['loss_fp'] == pytest.approx(1.0, abs=1e-6)

    def test_class_mismatch_scored(self):
        boxes = [[0, 0, 10, 10]]
        r_match = compute_set_distillation_loss(boxes, boxes, [0], [0])
        r_mismatch = compute_set_distillation_loss(boxes, boxes, [0], [1])
        assert r_match['loss_class'] == pytest.approx(0.0)
        assert r_match['class_mismatch_rate'] == pytest.approx(0.0)
        assert r_mismatch['loss_class'] == pytest.approx(1.0)
        assert r_mismatch['class_mismatch_rate'] == pytest.approx(1.0)
        assert r_mismatch['loss_cls_mismatch'] == pytest.approx(1.0)
        assert r_mismatch['loss'] > r_match['loss']

    def test_box_loss_combines_l1_and_iou(self):
        t = [[0, 0, 10, 10]]
        s = [[5, 5, 15, 15]]
        r = compute_set_distillation_loss(t, s)
        # boxes overlap partially → 0 < loss_iou < 1, positive box term
        assert 0.0 < r['loss_iou'] < 1.0
        assert r['loss_box'] > 0.0

    def test_detection_score_prefers_perfect_match(self):
        boxes = [[0, 0, 10, 10]]
        good = compute_set_distillation_loss(boxes, boxes, [0], [0])
        bad = compute_set_distillation_loss(
            boxes, [[0, 0, 10, 10], [99, 99, 100, 100]], [0], [0, 0]
        )
        assert good['detection_score'] > bad['detection_score']

    def test_soft_class_distillation_with_logits(self):
        boxes = [[0, 0, 10, 10]]
        # Matching teacher/student logits → low KL; divergent → higher KL.
        r_close = compute_set_distillation_loss(
            boxes, boxes, [0], [0],
            teacher_logits=[[2.0, 0.1, 0.1]], student_logits=[[2.0, 0.1, 0.1]],
        )
        r_far = compute_set_distillation_loss(
            boxes, boxes, [0], [0],
            teacher_logits=[[2.0, 0.1, 0.1]], student_logits=[[0.1, 0.1, 2.0]],
        )
        assert r_close['loss_class'] == pytest.approx(0.0, abs=1e-6)
        assert r_far['loss_class'] > r_close['loss_class']

    def test_one_empty_side(self):
        r = compute_set_distillation_loss([[0, 0, 10, 10]], [])
        assert r['loss_cardinality'] == pytest.approx(1.0)
        assert r['cardinality_error'] == 1
        assert r['fn_count'] == 1
        assert r['loss'] > 0.0


class TestDistillationScoreExposesLoss:
    def test_score_dict_has_loss_keys(self):
        boxes = [[0, 0, 10, 10]]
        scores = [0.9]
        classes = [0]
        r = compute_distillation_score(
            boxes, scores, classes, boxes, scores, classes
        )
        for key in ('loss', 'loss_total', 'loss_box', 'loss_iou',
                    'loss_cardinality', 'loss_class', 'fp_count', 'fn_count',
                    'cardinality_error', 'iou_mean_matched',
                    'class_mismatch_rate', 'detection_score'):
            assert key in r
        # Identical → near-zero loss
        assert r['loss'] == pytest.approx(0.0, abs=1e-6)


# ─── New: mAP@0.5 ─────────────────────────────────────────────────────────────


class TestComputeMapAt50:
    def test_empty_frames(self):
        assert compute_map_at_50([], []) == pytest.approx(0.0)

    def test_perfect_match_single_frame(self):
        boxes = [[0, 0, 10, 10]]
        result = compute_map_at_50([boxes], [boxes])
        assert result == pytest.approx(1.0, abs=0.01)

    def test_no_student_detections(self):
        boxes = [[0, 0, 10, 10]]
        result = compute_map_at_50([boxes], [[]])
        assert result == pytest.approx(0.0)

    def test_perfect_match_multiple_frames(self):
        boxes = [[0, 0, 50, 50], [100, 100, 150, 150]]
        result = compute_map_at_50([boxes, boxes], [boxes, boxes])
        assert result >= 0.99

    def test_partial_match_lower_than_perfect(self):
        t_boxes = [[0, 0, 50, 50], [100, 100, 150, 150]]
        s_boxes = [[0, 0, 50, 50]]   # student misses the second object
        result = compute_map_at_50([t_boxes], [s_boxes])
        assert 0.0 < result < 1.0

    def test_class_aware_mismatch(self):
        """Same location but different class → FP + FN → lower mAP."""
        t_boxes = [[0, 0, 10, 10]]
        s_boxes = [[0, 0, 10, 10]]
        t_cls = [0]
        s_cls = [1]   # wrong class
        result = compute_map_at_50([t_boxes], [s_boxes], [t_cls], [s_cls])
        assert result == pytest.approx(0.0)

    def test_class_aware_correct(self):
        """Same location and same class → perfect."""
        boxes = [[0, 0, 10, 10]]
        classes = [0]
        result = compute_map_at_50([boxes], [boxes], [classes], [classes])
        assert result == pytest.approx(1.0, abs=0.01)

    def test_iou_threshold_boundary(self):
        """IoU slightly above 0.5 → TP; slightly below → FP."""
        t_box = [[0, 0, 100, 100]]
        # This box has IoU ≈ 0.25 with the teacher → FP
        s_box_fp = [[50, 0, 150, 100]]
        result_fp = compute_map_at_50([t_box], [s_box_fp])
        # IoU = 0.33 < 0.5 → no TP → AP = 0
        assert result_fp == pytest.approx(0.0)

        # IoU = 1.0 → TP
        result_tp = compute_map_at_50([t_box], [t_box])
        assert result_tp > 0.9

    def test_class_agnostic_when_no_classes_provided(self):
        """When no class lists are provided all boxes use class 0."""
        t_boxes = [[0, 0, 10, 10]]
        s_boxes = [[0, 0, 10, 10]]
        # No class args → class-agnostic
        result = compute_map_at_50([t_boxes], [s_boxes])
        assert result == pytest.approx(1.0, abs=0.01)


# ─── New: ReservoirBuffer ─────────────────────────────────────────────────────


class TestReservoirBuffer:
    def test_basic_add_and_len(self):
        buf = ReservoirBuffer(maxsize=5)
        for i in range(3):
            buf.add(f"item_{i}")
        assert len(buf) == 3

    def test_does_not_exceed_maxsize(self):
        buf = ReservoirBuffer(maxsize=4)
        for i in range(20):
            buf.add(i, loss=float(i))
        assert len(buf) <= 4

    def test_clear_resets_buffer(self):
        buf = ReservoirBuffer(maxsize=10)
        for i in range(5):
            buf.add(i)
        buf.clear()
        assert len(buf) == 0

    def test_sample_returns_at_most_n(self):
        buf = ReservoirBuffer(maxsize=10)
        for i in range(8):
            buf.add(i, loss=float(i))
        samples = buf.sample(3)
        assert len(samples) <= 3

    def test_sample_empty_buffer(self):
        buf = ReservoirBuffer(maxsize=5)
        assert buf.sample(3) == []

    def test_hard_mining_bias_selects_harder_items(self):
        """With hard_mining_ratio=1.0 only the top half (highest loss) is sampled."""
        buf = ReservoirBuffer(maxsize=10)
        for i in range(10):
            buf.add(f"item_{i}", loss=float(i))
        # 10 items, half are loss ≥ 5 ("hard")
        samples = buf.sample(5, hard_mining_ratio=1.0)
        # All 5 samples should come from items with loss ≥ 5
        # (top-5 in a 10-item buffer with losses 0..9)
        item_numbers = [int(s.split('_')[1]) for s in samples]
        assert all(n >= 5 for n in item_numbers)

    def test_iter_yields_payloads(self):
        buf = ReservoirBuffer(maxsize=5)
        payloads = [f"x{i}" for i in range(5)]
        for p in payloads:
            buf.add(p, loss=1.0)
        result = list(buf)
        assert sorted(result) == sorted(payloads)

    def test_reservoir_samples_all_items_uniformly(self):
        """Items added early should have non-zero probability of survival.

        Fill a buffer of size 5 with 50 items and verify that the final buffer
        does not consist exclusively of the last 5 (which would indicate FIFO
        rather than reservoir sampling).
        """
        import random as _rand
        _rand.seed(0)
        buf = ReservoirBuffer(maxsize=5)
        for i in range(50):
            buf.add(i, loss=0.0)
        final = list(buf)
        # At least one of the first 20 items should have survived.
        assert any(v < 20 for v in final), (
            "Reservoir sampling should retain items from early in the stream."
        )


# ─── Agreement with the teacher, shipped students ────────────────────────────

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VIDEO = os.path.join(
    _ROOT, "node", "DLNode", "pose_estimation", "movenet", "D0002080169_00000_V_000.mp4"
)
_BUILTIN = {entry["name"]: entry for entry in student_models_registry.builtin_models()}


def _trainer(name, **settings):
    entry = _BUILTIN[name]
    return StudentTrainer(
        entry["path"], input_width=entry["input_width"], input_height=entry["input_height"],
        output_format=entry["output_format"], num_classes=entry["num_classes"], **settings,
    )


@pytest.fixture(scope="module")
def street_scene():
    """Frames of a street video with what the YOLOX-s teacher detects on them."""
    from node.DLNode.online_training.distill.yolox_onnx import YoloxOnnx

    teacher = YoloxOnnx(
        os.path.join(_ROOT, "node", "DLNode", "object_detection", "YOLOX", "model", "yolox_s.onnx")
    )
    capture = cv2.VideoCapture(_VIDEO)
    scene = []
    for index in range(0, 1500, 150):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        frame = capture.read()[1]
        scene.append((frame, *(values.tolist() for values in teacher.detect(frame, 0.3))))
    return scene


def test_a_detection_agrees_with_one_reference_of_its_class():
    references = [[0, 0, 100, 100], [200, 200, 300, 300]]
    boxes = [[0, 0, 100, 90], [2, 2, 100, 100], [200, 200, 300, 300], [500, 500, 600, 600]]

    counts = count_matches(references, [0, 2], boxes, [0.6, 0.9, 0.8, 0.7], [0, 0, 1, 0])

    # The first reference goes to the most confident of its two detections;
    # the second one is only overlapped by a detection of another class
    assert counts[:3] == (1, 3, 1)
    assert counts[3] == pytest.approx(0.9604)
    assert count_matches(references, [0, 2], [], [], []) == (0, 0, 2, 0.0)
    assert count_matches([], [], boxes[:1], [0.9], [0]) == (0, 1, 0, 0.0)


def test_agreement_is_pooled_over_the_most_recent_frames():
    box = [[0, 0, 10, 10]]
    meter = RollingAgreement(window=2)
    assert meter.result()["f1"] is None

    meter.add(box, [0], [], [], [])  # missed, then pushed out of the window
    meter.add(box, [0], box, [0.9], [0])
    meter.add(box, [0], box + [[50, 50, 60, 60]], [0.9, 0.8], [0, 0])

    result = meter.result()
    assert result["precision"] == pytest.approx(2 / 3) and result["recall"] == 1.0
    assert result["f1"] == pytest.approx(0.8) and result["mean_iou"] == pytest.approx(1.0)

    quiet = RollingAgreement()
    quiet.add([], [], [], [], [])  # nothing to find, nothing found
    assert quiet.result()["f1"] == 1.0 and quiet.result()["mean_iou"] is None


def test_shipped_students_are_described_as_the_inspector_reads_them():
    from node.DLNode.object_detection.onnx_inspector import inspect_onnx_model

    assert {"yolox_nano", "yolox_tiny", "yolo11_n"} <= set(_BUILTIN)
    for entry in _BUILTIN.values():
        meta = inspect_onnx_model(entry["path"])
        # A COCO YOLOX file (85 channels) used to be taken for a 21-class NanoDet
        for key in ("output_format", "num_classes", "input_width", "input_height"):
            assert meta[key] == entry[key], (entry["name"], key)


@pytest.mark.parametrize("backprop", [True, False], ids=["pytorch", "onnxruntime-only"])
@pytest.mark.parametrize("name", list(_BUILTIN))
def test_a_shipped_student_detects_what_the_teacher_detects(
    name, backprop, street_scene, monkeypatch
):
    from node.DLNode.online_training import student_trainer

    if not backprop:  # as when PyTorch is not installed
        monkeypatch.setattr(student_trainer, "is_torch_backprop_available", lambda: False)
    trainer = _trainer(name)
    assert backprop or trainer.backprop_mode == "affine-head"

    trainer.train_step(street_scene[0][0], [], [], [], teacher_valid=False)
    assert trainer.metrics()["f1"] is None  # no teacher result: nothing measured

    for frame, boxes, scores, class_ids in street_scene:
        trainer.train_step(frame, boxes, scores, class_ids)
    metrics = trainer.metrics()
    # A student decoded with the wrong format or input size agrees with nothing
    assert metrics["f1"] > 0.4 and metrics["mean_iou"] > 0.6
    assert metrics["f1_gain"] == 0.0  # training is off: it is the original student
    assert metrics["inference_ms"] > 0.0 and metrics["train_ms"] == 0.0


@pytest.mark.parametrize("name", [n for n in ("yolox_nano", "nanodet-plus-m_416") if n in _BUILTIN])
def test_the_baseline_is_the_original_student_on_the_same_frames(street_scene, name):
    pytest.importorskip("torch")
    pytest.importorskip("onnx2torch")
    trained = _trainer(name, learning_rate=1e-3, backprop_depth=-1)
    trained.training_active = True
    untouched = _trainer(name)

    for index, (frame, boxes, scores, class_ids) in enumerate(street_scene[:6]):
        trained.train_step(frame, boxes, scores, class_ids)
        if index % 5 == 0:  # the frames on which the original student is run again
            untouched.train_step(frame, boxes, scores, class_ids)

    assert trained.get_stats()["network_updates"] > 0
    metrics = trained.metrics()
    assert metrics["baseline_f1"] == pytest.approx(untouched.metrics()["f1"])
    assert metrics["f1_gain"] is not None


def test_shipped_students_come_first_and_cannot_be_replaced_or_deleted(monkeypatch):
    from node.DLNode.node_online_training import Node

    monkeypatch.setattr(student_models_registry, "load_registry", lambda: [
        {"name": "yolox_nano", "path": "elsewhere.onnx", "output_format": "nanodet"},
        {"name": "mine", "path": "mine.onnx"},
    ])
    monkeypatch.setattr(student_models_registry, "remove_entry", lambda name: None)
    monkeypatch.setattr(Node, "_student_models", {})

    names = Node._get_student_model_names()

    assert names == list(_BUILTIN) + ["mine"]
    assert Node._student_models["yolox_nano"]["output_format"] == "yolox"
    assert not Node._delete_student_model("yolox_nano") and "yolox_nano" in Node._student_models
    assert Node._delete_student_model("mine") and "mine" not in Node._student_models
