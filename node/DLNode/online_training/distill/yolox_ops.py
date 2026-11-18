#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""YOLOX geometry shared by the teacher, the student and the evaluation.

NumPy / OpenCV only, so that inference with a distilled student never needs
PyTorch.

A YOLOX network predicts, for every cell ("anchor") of three grids (strides
8, 16 and 32), a box, an objectness and one score per class.  Two networks fed
with the same pixels at the same place therefore produce predictions that can
be compared anchor by anchor: this is what dense distillation relies on.
"""
import cv2
import numpy as np

STRIDES = (8, 16, 32)
PAD_VALUE = 114


def check_input_size(height, width):
    if height % max(STRIDES) or width % max(STRIDES):
        raise ValueError(
            f"YOLOX input size must be a multiple of {max(STRIDES)}, got {width}x{height}"
        )


def make_grid(height, width):
    """Anchor grid of a ``height`` x ``width`` input.

    Returns ``(cells, strides)``: ``cells[A, 2]`` are the (x, y) cell indices
    and ``strides[A]`` the stride of each anchor, level by level in row-major
    order (the order of the network output).
    """
    cells, strides = [], []
    for stride in STRIDES:
        rows, cols = height // stride, width // stride
        ys, xs = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        cells.append(np.stack([xs.ravel(), ys.ravel()], axis=1))
        strides.append(np.full(rows * cols, stride))
    return (
        np.concatenate(cells).astype(np.float32),
        np.concatenate(strides).astype(np.float32),
    )


def cell_map(teacher_hw, student_hw, scale):
    """Student anchor each teacher anchor falls into.

    The teacher looks at the image at its own (higher) resolution; a point at
    pixel ``p`` of the teacher input is at pixel ``p * scale`` of the student
    input.  Level by level, the centre of every teacher cell is mapped to the
    student cell that contains it.  Returns ``index[A_teacher]``, ``-1`` for
    teacher cells that fall outside the student grid.
    """
    (t_h, t_w), (s_h, s_w) = teacher_hw, student_hw
    check_input_size(t_h, t_w)
    check_input_size(s_h, s_w)
    index, offset = [], 0
    for stride in STRIDES:
        t_rows, t_cols = t_h // stride, t_w // stride
        s_rows, s_cols = s_h // stride, s_w // stride
        ys, xs = np.meshgrid(np.arange(t_rows), np.arange(t_cols), indexing="ij")
        col = np.floor((xs.ravel() + 0.5) * scale).astype(np.int64)
        row = np.floor((ys.ravel() + 0.5) * scale).astype(np.int64)
        inside = (col < s_cols) & (row < s_rows)
        index.append(np.where(inside, offset + row * s_cols + col, -1))
        offset += s_rows * s_cols
    return np.concatenate(index)


def project_dense(boxes, objectness, class_scores, index, anchors, scale):
    """Teacher dense predictions expressed on the student grid.

    Several teacher cells can fall into one student cell (the student grid is
    coarser): the one with the highest objectness speaks for the cell.  Boxes
    are converted to student input pixels.
    """
    valid = index >= 0
    target = np.where(valid, index, 0)
    out_obj = np.zeros(anchors, dtype=np.float32)
    np.maximum.at(out_obj, target[valid], objectness[valid])
    winners = np.flatnonzero(valid & (objectness >= out_obj[target]))
    source = np.full(anchors, -1, dtype=np.int64)
    source[target[winners]] = winners
    covered = source >= 0
    out_cls = np.zeros((anchors, class_scores.shape[1]), dtype=np.float32)
    out_box = np.zeros((anchors, 4), dtype=np.float32)
    out_cls[covered] = class_scores[source[covered]]
    out_box[covered] = boxes[source[covered]] * scale
    return out_box, out_obj, out_cls


def flip_index(height, width):
    """Permutation of the anchors under a horizontal flip of the input."""
    index, offset = [], 0
    for stride in STRIDES:
        rows, cols = height // stride, width // stride
        ys, xs = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        index.append(offset + ys.ravel() * cols + (cols - 1 - xs.ravel()))
        offset += rows * cols
    return np.concatenate(index).astype(np.int64)


def letterbox(image, height, width):
    """YOLOX preprocessing: resize keeping the aspect ratio, pad bottom/right.

    Returns ``(canvas, ratio)``; ``canvas`` is a BGR uint8 image and boxes in
    canvas pixels are divided by ``ratio`` to get back to ``image`` pixels.
    """
    src_h, src_w = image.shape[:2]
    ratio = min(height / src_h, width / src_w)
    new_h, new_w = max(1, int(src_h * ratio)), max(1, int(src_w * ratio))
    canvas = np.full((height, width, 3), PAD_VALUE, dtype=np.uint8)
    canvas[:new_h, :new_w] = cv2.resize(
        image, (new_w, new_h), interpolation=cv2.INTER_LINEAR
    )
    return canvas, ratio


def to_blob(canvases):
    """BGR uint8 canvas (or list of canvases) -> float32 ``[N, 3, H, W]``.

    YOLOX takes raw BGR pixel values in [0, 255], without normalisation.
    """
    if isinstance(canvases, np.ndarray) and canvases.ndim == 3:
        canvases = [canvases]
    batch = np.stack(canvases).astype(np.float32)
    return np.ascontiguousarray(batch.transpose(0, 3, 1, 2))


def decode_boxes(raw, cells, strides):
    """Raw box outputs ``[..., 4]`` -> ``[x1, y1, x2, y2]`` in input pixels."""
    centre = (raw[..., :2] + cells) * strides[:, None]
    size = np.exp(np.minimum(raw[..., 2:4], 10.0)) * strides[:, None]
    return np.concatenate([centre - size / 2, centre + size / 2], axis=-1)


def detections(boxes, objectness, class_scores, score_th=0.3, nms_th=0.45, ratio=1.0):
    """Final detections from dense predictions (class-aware NMS).

    ``boxes[N, 4]`` are in input pixels, ``objectness[N]`` and
    ``class_scores[N, C]`` are probabilities.  Returns ``(boxes, scores,
    class_ids)`` with the boxes in original image pixels (``/ ratio``).
    """
    empty = (
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.int64),
    )
    if len(boxes) == 0:
        return empty
    class_ids = class_scores.argmax(axis=1)
    scores = objectness * class_scores[np.arange(len(class_ids)), class_ids]
    keep = scores >= score_th
    if not keep.any():
        return empty
    boxes = boxes[keep].astype(np.float32) / ratio
    scores = scores[keep].astype(np.float32)
    class_ids = class_ids[keep]

    # Shift the boxes of each class apart so that one NMS call is class-aware
    span = float(boxes.max() - min(0.0, boxes.min())) + 1.0
    shifted = boxes + (class_ids * span)[:, None]
    rects = np.concatenate([shifted[:, :2], shifted[:, 2:] - shifted[:, :2]], axis=1)
    kept = cv2.dnn.NMSBoxes(rects.tolist(), scores.tolist(), score_th, nms_th)
    kept = np.asarray(kept, dtype=np.int64).reshape(-1)
    return boxes[kept], scores[kept], class_ids[kept]


def box_iou(boxes_a, boxes_b):
    """IoU matrix ``[len(a), len(b)]`` of two ``[x1, y1, x2, y2]`` box sets."""
    boxes_a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    boxes_b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    inter = np.prod(np.clip(bottom_right - top_left, 0, None), axis=2)
    area_a = np.prod(np.clip(boxes_a[:, 2:] - boxes_a[:, :2], 0, None), axis=1)
    area_b = np.prod(np.clip(boxes_b[:, 2:] - boxes_b[:, :2], 0, None), axis=1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


class DetectionMeter:
    """Agreement of a detector with reference detections, over many images.

    Accumulates true/false positives (same class, IoU >= ``iou_th``) and
    reports ``f1`` at the detector's operating point and ``ap50``, the area
    under the precision/recall curve (classes pooled).
    """

    def __init__(self, iou_th=0.5):
        self.iou_th = iou_th
        self._scores = []
        self._hits = []
        self._references = 0

    def add(self, ref_boxes, ref_classes, boxes, scores, classes):
        self._references += len(ref_boxes)
        if len(boxes) == 0:
            return
        order = np.argsort(-np.asarray(scores))
        iou = box_iou(np.asarray(boxes)[order], ref_boxes) if len(ref_boxes) else None
        used = set()
        for rank, det in enumerate(order):
            hit = False
            if iou is not None:
                candidates = [
                    (iou[rank, ref], ref)
                    for ref in range(len(ref_boxes))
                    if ref not in used
                    and ref_classes[ref] == classes[det]
                    and iou[rank, ref] >= self.iou_th
                ]
                if candidates:
                    used.add(max(candidates)[1])
                    hit = True
            self._scores.append(float(scores[det]))
            self._hits.append(hit)

    def result(self):
        hits = np.asarray(self._hits, dtype=np.float64)
        true_pos = float(hits.sum())
        false_pos = float(len(hits) - true_pos)
        false_neg = float(self._references - true_pos)
        precision = true_pos / max(true_pos + false_pos, 1.0)
        recall = true_pos / max(true_pos + false_neg, 1.0)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        if self._references == 0:
            # Nothing to find: perfect unless something was detected
            f1 = ap50 = 1.0 if len(hits) == 0 else 0.0
        else:
            order = np.argsort(-np.asarray(self._scores)) if len(hits) else []
            cum_hits = np.cumsum(hits[order]) if len(hits) else np.zeros(0)
            prec = cum_hits / (np.arange(len(cum_hits)) + 1)
            rec = cum_hits / self._references
            # Area under the monotone precision envelope
            for i in range(len(prec) - 2, -1, -1):
                prec[i] = max(prec[i], prec[i + 1])
            ap50 = float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))
        return {
            "f1": float(f1),
            "ap50": float(ap50),
            "precision": float(precision),
            "recall": float(recall),
            "references": int(self._references),
            "detections": int(len(hits)),
        }
