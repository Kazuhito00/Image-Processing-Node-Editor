#!/usr/bin/env python
import copy

import cv2 as cv
import numpy as np
import onnxruntime


class MoveNet(object):
    def __init__(
        self,
        model_path,
        input_shape=(192, 192),
        providers=[
            # ('TensorrtExecutionProvider', {
            #     'trt_engine_cache_enable': True,
            #     'trt_engine_cache_path': '.',
            #     'trt_fp16_enable': True,
            # }),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    ):
        # モデル読み込み
        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            providers=providers,
        )

        self.input_detail = self.onnx_session.get_inputs()[0]
        self.input_name = self.input_detail.name
        self.output_detail = self.onnx_session.get_outputs()[0]

        # 各種設定
        self.input_shape = input_shape

    def __call__(self, image):
        image_width, image_height = image.shape[1], image.shape[0]

        # Pre process:Resize, BGR->RGB, Reshape, Int32 Cast
        input_image = cv.resize(
            image,
            dsize=(self.input_shape[1], self.input_shape[0]),
        )
        input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
        input_image = input_image.reshape(-1, self.input_shape[0],
                                          self.input_shape[1], 3)
        input_image = input_image.astype('int32')

        # Inference
        outputs = self.onnx_session.run(None, {self.input_name: input_image})

        # Post process
        keypoints_with_scores = outputs[0]
        keypoints_with_scores = np.squeeze(keypoints_with_scores)

        results_list = []
        landmark_dict = {}
        # SinglePose
        if keypoints_with_scores.shape == (17, 3):
            # キーポイント
            for id in range(17):
                keypoint_x = int(image_width * keypoints_with_scores[id][1])
                keypoint_y = int(image_height * keypoints_with_scores[id][0])
                score = keypoints_with_scores[id][2]

                landmark_dict[id] = [keypoint_x, keypoint_y, score]

            results_list.append(copy.deepcopy(landmark_dict))
        # MultiPose
        elif keypoints_with_scores.shape == (6, 56):
            for keypoints_with_score in keypoints_with_scores:
                # キーポイント
                for id in range(17):
                    keypoint_x = int(image_width *
                                     keypoints_with_score[(id * 3) + 1])
                    keypoint_y = int(image_height *
                                     keypoints_with_score[(id * 3) + 0])
                    score = keypoints_with_score[(id * 3) + 2]

                    landmark_dict[id] = [keypoint_x, keypoint_y, score]

                # バウンディングボックス
                bbox_ymin = int(image_height * keypoints_with_score[51])
                bbox_xmin = int(image_width * keypoints_with_score[52])
                bbox_ymax = int(image_height * keypoints_with_score[53])
                bbox_xmax = int(image_width * keypoints_with_score[54])
                bbox_score = keypoints_with_score[55]
                landmark_dict['bbox'] = [
                    bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, bbox_score
                ]

                results_list.append(copy.deepcopy(landmark_dict))

        return results_list


class MoveNetSinglePoseLightning(object):
    def __init__(
        self,
        model_path,
        providers=[
            # ('TensorrtExecutionProvider', {
            #     'trt_engine_cache_enable': True,
            #     'trt_engine_cache_path': '.',
            #     'trt_fp16_enable': True,
            # }),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    ):
        self.model = MoveNet(
            model_path=model_path,
            input_shape=(192, 192),
            providers=providers,
        )

    def __call__(self, image):
        return self.model(image)


class MoveNetSinglePoseThunder(object):
    def __init__(
        self,
        model_path,
        providers=[
            # ('TensorrtExecutionProvider', {
            #     'trt_engine_cache_enable': True,
            #     'trt_engine_cache_path': '.',
            #     'trt_fp16_enable': True,
            # }),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    ):
        self.model = MoveNet(
            model_path=model_path,
            input_shape=(256, 256),
            providers=providers,
        )

    def __call__(self, image):
        return self.model(image)


class MoveNetMultiPoseLightning(object):
    def __init__(
        self,
        model_path,
        providers=[
            # ('TensorrtExecutionProvider', {
            #     'trt_engine_cache_enable': True,
            #     'trt_engine_cache_path': '.',
            #     'trt_fp16_enable': True,
            # }),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    ):
        self.model = MoveNet(
            model_path=model_path,
            input_shape=(256, 256),
            providers=providers,
        )

    def __call__(self, image):
        return self.model(image)


def draw_singlepose_landmarks(image, results_list, score_th):
    for results in results_list:
        # キーポイント
        for id in range(17):
            landmark_x, landmark_y = results[id][0], results[id][1]
            visibility = results[id][2]

            if score_th > visibility:
                continue
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), -1)

        # Line：鼻 → 左目
        if results[0][2] > score_th and results[1][2] > score_th:
            cv.line(image, results[0][:2], results[1][:2], (0, 255, 0), 2)
        # Line：鼻 → 右目
        if results[0][2] > score_th and results[2][2] > score_th:
            cv.line(image, results[0][:2], results[2][:2], (0, 255, 0), 2)
        # Line：左目 → 左耳
        if results[1][2] > score_th and results[3][2] > score_th:
            cv.line(image, results[1][:2], results[3][:2], (0, 255, 0), 2)
        # Line：右目 → 右耳
        if results[2][2] > score_th and results[4][2] > score_th:
            cv.line(image, results[2][:2], results[4][:2], (0, 255, 0), 2)
        # Line：左肩 → 右肩
        if results[5][2] > score_th and results[6][2] > score_th:
            cv.line(image, results[5][:2], results[6][:2], (0, 255, 0), 2)
        # Line：左肩 → 左肘
        if results[5][2] > score_th and results[7][2] > score_th:
            cv.line(image, results[5][:2], results[7][:2], (0, 255, 0), 2)
        # Line：左肘 → 左手首
        if results[7][2] > score_th and results[9][2] > score_th:
            cv.line(image, results[7][:2], results[9][:2], (0, 255, 0), 2)
        # Line：右肩 → 右肘
        if results[6][2] > score_th and results[8][2] > score_th:
            cv.line(image, results[6][:2], results[8][:2], (0, 255, 0), 2)
        # Line：右肘 → 右手首
        if results[8][2] > score_th and results[10][2] > score_th:
            cv.line(image, results[8][:2], results[10][:2], (0, 255, 0), 2)
        # Line：左股関節 → 右股関節
        if results[11][2] > score_th and results[12][2] > score_th:
            cv.line(image, results[11][:2], results[12][:2], (0, 255, 0), 2)
        # Line：左肩 → 左股関節
        if results[5][2] > score_th and results[11][2] > score_th:
            cv.line(image, results[5][:2], results[11][:2], (0, 255, 0), 2)
        # Line：左股関節 → 左ひざ
        if results[11][2] > score_th and results[13][2] > score_th:
            cv.line(image, results[11][:2], results[13][:2], (0, 255, 0), 2)
        # Line：左ひざ → 左足首
        if results[13][2] > score_th and results[15][2] > score_th:
            cv.line(image, results[13][:2], results[15][:2], (0, 255, 0), 2)
        # Line：右肩 → 右股関節
        if results[6][2] > score_th and results[12][2] > score_th:
            cv.line(image, results[6][:2], results[12][:2], (0, 255, 0), 2)
        # Line：右股関節 → 右ひざ
        if results[12][2] > score_th and results[14][2] > score_th:
            cv.line(image, results[12][:2], results[14][:2], (0, 255, 0), 2)
        # Line：右ひざ → 右足首
        if results[14][2] > score_th and results[16][2] > score_th:
            cv.line(image, results[14][:2], results[16][:2], (0, 255, 0), 2)

        bbox = results.get('bbox', None)
        if bbox is not None:
            if bbox[4] > score_th:
                image = cv.rectangle(
                    image,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    (0, 255, 0),
                    thickness=2,
                )

    return image


if __name__ == '__main__':
    cap = cv.VideoCapture(0)

    # Load model
    model_path = 'model/movenet_singlepose_thunder_4.onnx'
    model = MoveNetSinglePoseThunder(model_path)
    # model_path = 'model/movenet_multipose_lightning_1.onnx'
    # model = MoveNetMultiPoseLightning(model_path)

    score_th = 0.4

    while True:
        # Capture read
        ret, frame = cap.read()
        if not ret:
            break

        # Inference execution
        results = model(frame)

        # Draw
        frame = draw_singlepose_landmarks(frame, results, score_th)

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        cv.imshow('MediaPipe Pose', frame)
    cap.release()
    cv.destroyAllWindows()
