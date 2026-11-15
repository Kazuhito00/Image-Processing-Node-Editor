#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Optimised hot paths must produce exactly the same results as before."""
import cv2
import numpy as np
import pytest

from node.basenode import Node
from node.timestamped_queue import NodeDataQueueManager


def _reference_texture(image, width, height):
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    data = np.asarray(np.flip(resized, 2).ravel(), dtype=np.float32)
    return np.true_divide(data, 255.0)


def _reference_segmentation_overlay(node, image, score_th, class_num, seg_map):
    debug_image = image.copy()
    seg_map = np.where(seg_map > score_th, 0, 1)
    color_map = node.get_color_map_list(max(class_num, len(seg_map)))
    for index, mask in enumerate(seg_map):
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = color_map[index * 3 : index * 3 + 3]
        mask = np.stack((mask,) * 3, axis=-1).astype("uint8")
        mask_image = np.where(mask, debug_image, bg_image)
        debug_image = cv2.addWeighted(debug_image, 0.5, mask_image, 0.5, 0)
    return debug_image


@pytest.mark.parametrize(
    "image",
    [
        np.random.default_rng(0).integers(0, 256, (120, 160, 3), dtype=np.uint8),
        # Non-uint8 input (start-up black frames) takes the generic path
        np.zeros((48, 64, 3)),
    ],
)
def test_convert_cv_to_dpg_matches_reference(image):
    texture = Node().convert_cv_to_dpg(image, 64, 48)

    assert texture.dtype == np.float32
    assert texture.shape == (64 * 48 * 3,)
    assert np.array_equal(texture, _reference_texture(image, 64, 48))


def test_segmentation_overlay_matches_reference():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, (60, 80, 3), dtype=np.uint8)
    # Overlapping masks, plus one empty mask (skipped by the fast path)
    seg_map = rng.random((4, 60, 80)).astype(np.float32)
    seg_map[2] = 0.0
    node = Node()

    result = node.draw_semantic_segmentation_info(image, 0.5, 4, seg_map)

    expected = _reference_segmentation_overlay(node, image, 0.5, 4, seg_map)
    assert np.array_equal(result, expected)
    assert not np.shares_memory(result, image)


def test_queue_lookup_does_not_create_or_fail_on_missing_node():
    manager = NodeDataQueueManager()

    assert manager.get_latest_data("1:Missing", "image") is None
    assert manager.get_oldest_data("1:Missing", "image") is None
    assert manager._queues == {}

    manager.put_data("1:Cam", "image", "a", timestamp=1.0)
    manager.put_data("1:Cam", "image", "b", timestamp=2.0)
    assert manager.get_oldest_data("1:Cam", "image") == "a"
    assert manager.get_latest_data("1:Cam", "image") == "b"
    assert manager.get_latest_data("1:Cam", "audio") is None

    manager.remove_node("1:Cam")
    assert manager.get_latest_data("1:Cam", "image") is None


def test_yolo_postprocess_matches_per_row_decoding():
    pytest.importorskip("onnxruntime")
    from node.DLNode.object_detection.YOLO.yolo import YOLO

    rng = np.random.default_rng(2)
    outputs = rng.random((1, 1, 84, 500), dtype=np.float32)
    outputs[0, 0, :4] *= 600
    rows = outputs[0, 0].T

    model = object.__new__(YOLO)
    boxes, scores, class_ids = model._postprocess(outputs, 0.5, 0.99)

    kept = [row for row in rows if row[4:].max() >= 0.99]
    assert len(kept) > 0
    expected_boxes = [
        [
            int(x - w / 2),
            int(y - h / 2),
            int(x + w / 2),
            int(y + h / 2),
        ]
        for x, y, w, h in (row[:4] for row in kept)
    ]
    assert boxes.tolist() == expected_boxes
    assert scores.tolist() == [row[4:].max() for row in kept]
    assert class_ids.tolist() == [int(row[4:].argmax()) for row in kept]

    empty = model._postprocess(outputs, 0.5, 2.0)
    assert all(len(part) == 0 for part in empty)
