#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import copy
import cv2
import numpy as np
import onnxruntime
import math


# Désactive cuDNN et utilise le fallback CUDA
os.environ["ORT_CUDA_USE_CUDNN"] = "0"

class YOLOTENNIS:
    def __init__(
        self,
        model_path='tennis.onnx',
        class_score_th=0.0,
        nms_th=0.45,
        nms_score_th=0.1,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    ):
        self.class_score_th = class_score_th
        self.nms_th = nms_th
        self.nms_score_th = nms_score_th
        
        self.onnx_session = onnxruntime.InferenceSession(model_path, providers=providers)
        self.input_detail = self.onnx_session.get_inputs()[0]
        self.input_name = self.input_detail.name
        self.output_name = self.onnx_session.get_outputs()[0].name
        
        self.input_shape = (608, 608)
        self.input_width, self.input_height = self.input_shape

    def __call__(self, image):
        # Store original dimensions
        original_height, original_width = image.shape[:2]
        
        temp_image = cv2.resize(image, (608, 608), interpolation=cv2.INTER_AREA)
        image = self._preprocess(temp_image)
        # Debug: print("preprocess", image.shape) 
        results = self.onnx_session.run(None, {self.input_name: image})
        
        bboxes, scores, class_ids = self._postprocess(
            results[0], self.nms_th, self.nms_score_th, original_width, original_height
        )
        return bboxes, scores, class_ids

    def _preprocess(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_data = np.array(img) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        return image_data
    

    def _postprocess(self, outputs, nms_th, nms_score_th, original_width, original_height):
        outputs = np.transpose(np.squeeze(outputs[0]))

        # Calculate scaling factors from 608x608 to original dimensions
        scale_x = original_width / 608.0
        scale_y = original_height / 608.0

        # Vectorised decoding: one pass over all candidate rows instead of a
        # Python loop over thousands of predictions per frame.
        classes_scores = outputs[:, 4:]
        max_scores = classes_scores.max(axis=1)
        keep = max_scores >= nms_score_th
        if not keep.any():
            return np.array([]), np.array([]), np.array([])

        kept = outputs[keep]
        x, y, w, h = kept[:, 0], kept[:, 1], kept[:, 2], kept[:, 3]

        # Convert (x, y, w, h) to (x1, y1, x2, y2), scale to the original
        # dimensions and truncate like int()
        boxes = np.stack(
            (
                (x - w / 2) * scale_x,
                (y - h / 2) * scale_y,
                (x + w / 2) * scale_x,
                (y + h / 2) * scale_y,
            ),
            axis=1,
        ).astype(np.int64)
        scores = max_scores[keep]
        class_ids = classes_scores[keep].argmax(axis=1)

        return boxes, scores, class_ids


    def draw(self, image, score_th, bboxes, scores, class_ids, coco_classes, thickness=1):
        debug_image = copy.deepcopy(image)
        for bbox, score, class_id in zip(bboxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, bbox)
            if score < score_th:
                continue
            color = self._get_color(class_id)
            debug_image = cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, thickness)
            text = f'{coco_classes[int(class_id)]}: {score:.2f}'
            debug_image = cv2.putText(debug_image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        return debug_image

    def _get_color(self, index):
        temp_index = abs(int(index + 5)) * 3
        color = (
            (29 * temp_index) % 255,
            (17 * temp_index) % 255,
            (37 * temp_index) % 255,
        )
        return color
