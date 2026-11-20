#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Agreement of a student with its teacher over the most recent frames.

The OnlineTraining node scores every frame *before* training on it, so these
figures always describe frames the student has not been trained on yet
(test-then-train).  They are pooled over a window of frames: a single frame
holds too few objects for a precision or a recall to mean anything.
"""
import collections

import numpy as np

from node.DLNode.online_training.distill.yolox_ops import box_iou


def count_matches(ref_boxes, ref_classes, boxes, scores, classes, iou_th=0.5):
    """``(true_pos, false_pos, false_neg, iou_sum)`` of detections against references.

    A detection is a true positive when it overlaps a reference of the same
    class with an IoU of at least ``iou_th``.  Each reference is used once,
    the most confident detections being served first.  ``iou_sum`` is the sum
    of the IoU of the true positives.
    """
    references = len(ref_boxes)
    if len(boxes) == 0:
        return 0, 0, references, 0.0
    if references == 0:
        return 0, len(boxes), 0, 0.0

    iou = box_iou(boxes, ref_boxes)
    same_class = np.asarray(classes).reshape(-1, 1) == np.asarray(ref_classes).reshape(1, -1)
    iou = np.where(same_class & (iou >= iou_th), iou, 0.0)
    true_pos, iou_sum = 0, 0.0
    for det in np.argsort(-np.asarray(scores, dtype=np.float64)):
        ref = int(np.argmax(iou[det]))
        if iou[det, ref] > 0.0:
            true_pos += 1
            iou_sum += float(iou[det, ref])
            iou[:, ref] = 0.0
    return true_pos, len(boxes) - true_pos, references - true_pos, iou_sum


class RollingAgreement:
    """Precision, recall and F1 against the teacher, pooled over recent frames."""

    def __init__(self, window=100):
        self._frames = collections.deque(maxlen=int(window))

    def __len__(self):
        return len(self._frames)

    def add(self, ref_boxes, ref_classes, boxes, scores, classes):
        self._frames.append(count_matches(ref_boxes, ref_classes, boxes, scores, classes))

    def clear(self):
        self._frames.clear()

    def result(self):
        """``f1``, ``precision``, ``recall``, ``mean_iou``; ``None`` when unknown.

        ``mean_iou`` is the localisation quality of the true positives.  A
        window where the teacher saw nothing and the student detected nothing
        is a perfect agreement.
        """
        if not self._frames:
            return {"f1": None, "precision": None, "recall": None, "mean_iou": None}
        true_pos, false_pos, false_neg, iou_sum = np.sum(self._frames, axis=0)
        if true_pos + false_pos + false_neg == 0:
            return {"f1": 1.0, "precision": 1.0, "recall": 1.0, "mean_iou": None}
        precision = true_pos / max(true_pos + false_pos, 1.0)
        recall = true_pos / max(true_pos + false_neg, 1.0)
        return {
            "f1": float(2 * precision * recall / max(precision + recall, 1e-9)),
            "precision": float(precision),
            "recall": float(recall),
            "mean_iou": float(iou_sum / true_pos) if true_pos else None,
        }
