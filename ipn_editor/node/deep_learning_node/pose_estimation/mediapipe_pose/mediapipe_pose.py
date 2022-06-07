#!/usr/bin/env python
import copy

import cv2 as cv


class MediaPipePose(object):

    def __init__(
        self,
        model_path,
        model_complexity,
        enable_segmentation,
        min_detection_confidence,
        min_tracking_confidence,
        providers=None,
    ):
        import mediapipe as mp

        mp_pose = mp.solutions.pose

        self.pose = mp_pose.Pose(
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def __call__(self, image):
        image_width, image_height = image.shape[1], image.shape[0]

        # Pre process:BGR->RGB
        input_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Inference
        results = self.pose.process(input_image)

        # Post process
        results_list = []
        if results.pose_landmarks is not None:
            landmark_dict = {}

            # 各キーポイント
            for id, landmark in enumerate(results.pose_landmarks.landmark):
                x = min(int(landmark.x * image_width), image_width - 1)
                y = min(int(landmark.y * image_height), image_height - 1)
                z = landmark.z
                visibility = landmark.visibility
                landmark_dict[id] = [x, y, z, visibility]

            results_list.append(copy.deepcopy(landmark_dict))

        return results_list


class MediaPipePoseComplexity0(object):

    def __init__(
        self,
        model_path,
        providers=None,
    ):
        self.model = MediaPipePose(
            None,
            model_complexity=0,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def __call__(self, image):
        return self.model(image)


class MediaPipePoseComplexity1(object):

    def __init__(
        self,
        model_path,
        providers=None,
    ):
        self.model = MediaPipePose(
            None,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def __call__(self, image):
        return self.model(image)


class MediaPipePoseComplexity2(object):

    def __init__(
        self,
        model_path,
        providers=None,
    ):
        self.model = MediaPipePose(
            None,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def __call__(self, image):
        return self.model(image)


def draw_landmarks(image, results_list, score_th):
    for results in results_list:
        # キーポイント
        for id in range(33):
            landmark_x, landmark_y = results[id][0], results[id][1]
            visibility = results[id][3]

            if score_th > visibility:
                continue
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), -1)

        # 接続線
        # 右目
        if results[1][3] > score_th and results[2][3] > score_th:
            cv.line(image, results[1][:2], results[2][:2], (0, 255, 0), 2)
        if results[2][3] > score_th and results[3][3] > score_th:
            cv.line(image, results[2][:2], results[3][:2], (0, 255, 0), 2)

        # 左目
        if results[4][3] > score_th and results[5][3] > score_th:
            cv.line(image, results[4][:2], results[5][:2], (0, 255, 0), 2)
        if results[5][3] > score_th and results[6][3] > score_th:
            cv.line(image, results[5][:2], results[6][:2], (0, 255, 0), 2)

        # 口
        if results[9][3] > score_th and results[10][3] > score_th:
            cv.line(image, results[9][:2], results[10][:2], (0, 255, 0), 2)

        # 肩
        if results[11][3] > score_th and results[12][3] > score_th:
            cv.line(image, results[11][:2], results[12][:2], (0, 255, 0), 2)

        # 右腕
        if results[11][3] > score_th and results[13][3] > score_th:
            cv.line(image, results[11][:2], results[13][:2], (0, 255, 0), 2)
        if results[13][3] > score_th and results[15][3] > score_th:
            cv.line(image, results[13][:2], results[15][:2], (0, 255, 0), 2)

        # 左腕
        if results[12][3] > score_th and results[14][3] > score_th:
            cv.line(image, results[12][:2], results[14][:2], (0, 255, 0), 2)
        if results[14][3] > score_th and results[16][3] > score_th:
            cv.line(image, results[14][:2], results[16][:2], (0, 255, 0), 2)

        # 右手
        if results[15][3] > score_th and results[17][3] > score_th:
            cv.line(image, results[15][:2], results[17][:2], (0, 255, 0), 2)
        if results[17][3] > score_th and results[19][3] > score_th:
            cv.line(image, results[17][:2], results[19][:2], (0, 255, 0), 2)
        if results[19][3] > score_th and results[21][3] > score_th:
            cv.line(image, results[19][:2], results[21][:2], (0, 255, 0), 2)
        if results[21][3] > score_th and results[15][3] > score_th:
            cv.line(image, results[21][:2], results[15][:2], (0, 255, 0), 2)

        # 左手
        if results[16][3] > score_th and results[18][3] > score_th:
            cv.line(image, results[16][:2], results[18][:2], (0, 255, 0), 2)
        if results[18][3] > score_th and results[20][3] > score_th:
            cv.line(image, results[18][:2], results[20][:2], (0, 255, 0), 2)
        if results[20][3] > score_th and results[22][3] > score_th:
            cv.line(image, results[20][:2], results[22][:2], (0, 255, 0), 2)
        if results[22][3] > score_th and results[16][3] > score_th:
            cv.line(image, results[22][:2], results[16][:2], (0, 255, 0), 2)

        # 胴体
        if results[11][3] > score_th and results[23][3] > score_th:
            cv.line(image, results[11][:2], results[23][:2], (0, 255, 0), 2)
        if results[12][3] > score_th and results[24][3] > score_th:
            cv.line(image, results[12][:2], results[24][:2], (0, 255, 0), 2)
        if results[23][3] > score_th and results[24][3] > score_th:
            cv.line(image, results[23][:2], results[24][:2], (0, 255, 0), 2)

        # 右足
        if results[23][3] > score_th and results[25][3] > score_th:
            cv.line(image, results[23][:2], results[25][:2], (0, 255, 0), 2)
        if results[25][3] > score_th and results[27][3] > score_th:
            cv.line(image, results[25][:2], results[27][:2], (0, 255, 0), 2)
        if results[27][3] > score_th and results[29][3] > score_th:
            cv.line(image, results[27][:2], results[29][:2], (0, 255, 0), 2)
        if results[29][3] > score_th and results[31][3] > score_th:
            cv.line(image, results[29][:2], results[31][:2], (0, 255, 0), 2)

        # 左足
        if results[24][3] > score_th and results[26][3] > score_th:
            cv.line(image, results[24][:2], results[26][:2], (0, 255, 0), 2)
        if results[26][3] > score_th and results[28][3] > score_th:
            cv.line(image, results[26][:2], results[28][:2], (0, 255, 0), 2)
        if results[28][3] > score_th and results[30][3] > score_th:
            cv.line(image, results[28][:2], results[30][:2], (0, 255, 0), 2)
        if results[30][3] > score_th and results[32][3] > score_th:
            cv.line(image, results[30][:2], results[32][:2], (0, 255, 0), 2)

    return image


if __name__ == '__main__':
    cap = cv.VideoCapture(0)

    # Load model
    model = MediaPipePoseComplexity0(None)

    score_th = 0.5

    while True:
        # Capture read
        ret, frame = cap.read()
        if not ret:
            break

        # Inference execution
        results = model(frame)

        # Draw
        frame = draw_landmarks(frame, results, score_th)

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        cv.imshow('MediaPipe Pose', frame)
    cap.release()
    cv.destroyAllWindows()
