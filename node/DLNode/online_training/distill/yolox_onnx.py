#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""YOLOX ONNX model run with onnxruntime: dense outputs and detections.

Used for the teacher and for the deployed student.  No PyTorch here.
"""
import numpy as np
import onnxruntime

from node.DLNode.online_training.distill import yolox_ops


def default_providers(use_gpu=True):
    available = onnxruntime.get_available_providers()
    if use_gpu and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class YoloxOnnx:
    """A YOLOX detector exported without the box decoding.

    The expected output is ``[1, anchors, 5 + classes]``: raw box, objectness
    probability and class probabilities (the official YOLOX export with
    ``decode_in_inference=False``).

    Parameters
    ----------
    model : str or bytes
        Path of the ONNX file, or its serialized content.
    """

    def __init__(self, model, providers=None, threads=0):
        options = onnxruntime.SessionOptions()
        options.log_severity_level = 3
        if threads:
            options.intra_op_num_threads = int(threads)
        self.session = onnxruntime.InferenceSession(
            model, sess_options=options, providers=providers or default_providers()
        )
        graph_input = self.session.get_inputs()[0]
        self.input_name = graph_input.name
        shape = graph_input.shape
        if len(shape) != 4 or not all(isinstance(d, int) for d in shape[1:]):
            raise ValueError(f"YOLOX model with a fixed input size expected, got {shape}")
        self.height, self.width = int(shape[2]), int(shape[3])
        yolox_ops.check_input_size(self.height, self.width)
        self.cells, self.strides = yolox_ops.make_grid(self.height, self.width)

        out_shape = self.session.get_outputs()[0].shape
        anchors = len(self.strides)
        if len(out_shape) != 3 or out_shape[1] != anchors or not isinstance(out_shape[2], int):
            raise ValueError(
                f"not a raw YOLOX output: expected [1, {anchors}, 5 + classes], got {out_shape}"
            )
        self.num_classes = int(out_shape[2]) - 5
        if self.num_classes < 1:
            raise ValueError(f"not a raw YOLOX output: {out_shape}")

    def raw(self, canvas):
        """Network output ``[anchors, 5 + classes]`` for one input-sized canvas."""
        blob = yolox_ops.to_blob(canvas)
        return self.session.run(None, {self.input_name: blob})[0][0]

    def dense(self, canvas):
        """Dense predictions: ``(boxes[A, 4], objectness[A], class_scores[A, C])``."""
        raw = self.raw(canvas)
        boxes = yolox_ops.decode_boxes(raw[:, :4], self.cells, self.strides)
        return boxes.astype(np.float32), raw[:, 4], raw[:, 5:]

    def detect(self, image, score_th=0.3, nms_th=0.45):
        """Detections ``(boxes, scores, class_ids)`` in ``image`` pixels."""
        canvas, ratio = yolox_ops.letterbox(image, self.height, self.width)
        boxes, objectness, class_scores = self.dense(canvas)
        return yolox_ops.detections(
            boxes, objectness, class_scores, score_th, nms_th, ratio
        )
