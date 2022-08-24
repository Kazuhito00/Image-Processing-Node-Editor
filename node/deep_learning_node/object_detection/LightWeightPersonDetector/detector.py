#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import copy

import cv2
import numpy as np


class LWPDetector(object):

    def __init__(
            self,
            model_path='model.onnx',
            input_shape=(192, 192),
            score_th=0.0,
            nms_th=0.5,
            providers=[
            # ('TensorrtExecutionProvider', {
            #     'trt_engine_cache_enable': True,
            #     'trt_engine_cache_path': '.',
            #     'trt_fp16_enable': True,
            # }),
                'CUDAExecutionProvider',
                'CPUExecutionProvider',
            ],
            num_threads=None,  # Valid only when using Tensorflow-Lite
    ):
        # 入力サイズ
        self.input_shape = input_shape

        # 閾値
        self.score_th = score_th
        self.nms_th = nms_th

        # モデル読み込み
        self.extension = os.path.splitext(model_path)[1][1:]
        if self.extension == 'onnx':
            import onnxruntime

            self.model = onnxruntime.InferenceSession(
                model_path,
                providers=providers,
            )

            self.input_name = self.model.get_inputs()[0].name
            self.output_name = self.model.get_outputs()[0].name
        elif self.extension == 'tflite':
            try:
                from tflite_runtime.interpreter import Interpreter
                self.model = Interpreter(
                    model_path=model_path,
                    num_threads=num_threads,
                )
            except ImportError:
                import tensorflow as tf
                self.model = tf.lite.Interpreter(
                    model_path=model_path,
                    num_threads=num_threads,
                )

            self.model.allocate_tensors()

            self.input_name = self.model.get_input_details()[0]['index']
            self.output_name = self.model.get_output_details()[0]['index']
        else:
            raise ValueError("Invalid extension %s." % (model_path))

    def __call__(self, image):
        temp_image = copy.deepcopy(image)

        # 前処理
        image, ratio = self._preprocess(temp_image, self.input_shape)

        # 推論実施
        results = None
        if self.extension == 'onnx':
            results = self.model.run(
                None,
                {self.input_name: image[None, :, :, :]},
            )[0]
        elif self.extension == 'tflite':
            image = image.reshape(
                -1,
                3,
                self.input_shape[0],
                self.input_shape[1],
            )
            self.model.set_tensor(self.input_name, image)
            self.model.invoke()
            results = self.model.get_tensor(self.output_name)

        # 後処理
        bboxes, scores, class_ids = self._postprocess(
            results,
            self.input_shape,
            ratio,
            self.score_th,
            self.nms_th,
        )

        return bboxes, scores, class_ids

    def _preprocess(self, image, input_size):
        # リサイズ
        ratio = min(input_size[0] / image.shape[0],
                    input_size[1] / image.shape[1])
        resized_image = cv2.resize(
            image,
            (int(image.shape[1] * ratio), int(image.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR,
        )
        resized_image = resized_image.astype(np.uint8)

        # パディング込み画像作成
        padded_image = np.ones(
            (input_size[0], input_size[1], 3),
            dtype=np.uint8,
        )
        padded_image *= 114
        padded_image[:int(image.shape[0] * ratio), :int(image.shape[1] *
                                                        ratio)] = resized_image

        padded_image = padded_image.transpose((2, 0, 1))
        padded_image = np.ascontiguousarray(padded_image, dtype=np.float32)

        return padded_image, ratio

    def _postprocess(
        self,
        outputs,
        img_size,
        ratio,
        score_th,
        nms_th,
    ):
        grids = []
        expanded_strides = []

        strides = [8, 16, 32]

        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        predictions = outputs[0]
        bboxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        scores = scores.T[0]

        bboxes_xyxy = np.ones_like(bboxes)
        bboxes_xyxy[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2.
        bboxes_xyxy[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2.
        bboxes_xyxy[:, 2] = bboxes[:, 0] + bboxes[:, 2] / 2.
        bboxes_xyxy[:, 3] = bboxes[:, 1] + bboxes[:, 3] / 2.
        bboxes_xyxy /= ratio

        return self._nms(bboxes_xyxy, scores, score_th, nms_th)

    def _nms(self, bboxes, scores, score_th, nms_th):
        indexes = cv2.dnn.NMSBoxes(
            bboxes.tolist(),
            scores.tolist(),
            score_th,
            nms_th,
        )

        result_bboxes, result_scores, result_class_ids = [], [], []
        if len(indexes) > 0:
            if indexes.ndim == 2:
                result_bboxes = bboxes[indexes[:, 0]]
                result_scores = scores[indexes[:, 0]]
                result_class_ids = np.zeros(result_scores.shape)
            elif indexes.ndim == 1:
                result_bboxes = bboxes[indexes[:]]
                result_scores = scores[indexes[:]]
                result_class_ids = np.zeros(result_scores.shape)

        return result_bboxes, result_scores, result_class_ids

    def draw(
        self,
        image,
        score_th,
        bboxes,
        scores,
        class_ids,
        coco_classes,
        thickness=3,
    ):
        debug_image = copy.deepcopy(image)

        for bbox, score, class_id in zip(bboxes, scores, class_ids):
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(
                bbox[3])

            if score_th > score:
                continue

            color = self._get_color(class_id)

            # バウンディングボックス
            debug_image = cv2.rectangle(
                debug_image,
                (x1, y1),
                (x2, y2),
                color,
                thickness=thickness,
            )

            # クラスID、スコア
            score = '%.2f' % score
            text = '%s:%s' % (str(coco_classes[int(class_id)]), score)
            debug_image = cv2.putText(
                debug_image,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                thickness=thickness,
            )

        return debug_image

    def _get_color(self, index):
        temp_index = abs(int(index + 5)) * 3
        color = (
            (29 * temp_index) % 255,
            (17 * temp_index) % 255,
            (37 * temp_index) % 255,
        )
        return color


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    # Load model
    model_path = 'model/model.onnx'
    model = LWPDetector(model_path)

    class_list = ['person']

    while True:
        # Capture read
        ret, frame = cap.read()
        if not ret:
            break

        # Inference execution
        bboxes, scores, class_ids = model(frame)

        # Draw
        frame = model.draw(
            frame,
            0.3,
            bboxes,
            scores,
            class_ids,
            class_list,
        )

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        cv2.imshow('LightWeight Person Detector', frame)

    cap.release()
    cv2.destroyAllWindows()
