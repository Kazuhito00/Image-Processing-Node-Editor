#!/usr/bin/env python
import copy

import cv2 as cv
import numpy as np


class MediaPipeHands(object):

    def __init__(
        self,
        model_path,
        model_complexity,
        max_num_hands,
        min_detection_confidence,
        min_tracking_confidence,
        providers=None,
    ):
        import mediapipe as mp

        mp_hands = mp.solutions.hands

        self.hands = mp_hands.Hands(
            model_complexity=model_complexity,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def __call__(self, image):
        image_width, image_height = image.shape[1], image.shape[0]

        # Pre process:BGR->RGB
        input_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Inference
        results = self.hands.process(input_image)

        # Post process
        results_list = []
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness,
            ):
                landmark_dict = {}

                # 手のひらの重心
                cx, cy = self._calc_palm_moment(image, hand_landmarks)
                landmark_dict['palm_moment'] = [cx, cy]

                # 各キーポイント
                for id, landmark in enumerate(hand_landmarks.landmark):
                    x = min(int(landmark.x * image_width), image_width - 1)
                    y = min(int(landmark.y * image_height), image_height - 1)
                    z = landmark.z
                    visibility = 1.0
                    landmark_dict[id] = [x, y, z, visibility]
                # ラベル
                landmark_dict['label'] = handedness.classification[0].label

                results_list.append(copy.deepcopy(landmark_dict))

        return results_list

    def _calc_palm_moment(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        palm_array = np.empty((0, 2), int)

        for index, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            if index == 0:  # 手首1
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 1:  # 手首2
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 5:  # 人差指：付け根
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 9:  # 中指：付け根
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 13:  # 薬指：付け根
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 17:  # 小指：付け根
                palm_array = np.append(palm_array, landmark_point, axis=0)
        M = cv.moments(palm_array)
        cx, cy = 0, 0
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

        return cx, cy


class MediaPipeHandsComplexity0(object):

    def __init__(
        self,
        model_path,
        providers=None,
    ):
        self.model = MediaPipeHands(
            None,
            model_complexity=0,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

    def __call__(self, image):
        return self.model(image)


class MediaPipeHandsComplexity1(object):

    def __init__(
        self,
        model_path,
        providers=None,
    ):
        self.model = MediaPipeHands(
            None,
            model_complexity=1,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

    def __call__(self, image):
        return self.model(image)


def draw_landmarks(image, results_list, score_th):
    for results in results_list:
        # キーポイント
        for id in range(21):
            if score_th > results[id][3]:
                continue
            landmark_x, landmark_y = results[id][0], results[id][1]
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), -1)

        # 接続線
        # 親指
        cv.line(image, results[2][:2], results[3][:2], (0, 255, 0), 2)
        cv.line(image, results[3][:2], results[4][:2], (0, 255, 0), 2)

        # 人差指
        cv.line(image, results[5][:2], results[6][:2], (0, 255, 0), 2)
        cv.line(image, results[6][:2], results[7][:2], (0, 255, 0), 2)
        cv.line(image, results[7][:2], results[8][:2], (0, 255, 0), 2)

        # 中指
        cv.line(image, results[9][:2], results[10][:2], (0, 255, 0), 2)
        cv.line(image, results[10][:2], results[11][:2], (0, 255, 0), 2)
        cv.line(image, results[11][:2], results[12][:2], (0, 255, 0), 2)

        # 薬指
        cv.line(image, results[13][:2], results[14][:2], (0, 255, 0), 2)
        cv.line(image, results[14][:2], results[15][:2], (0, 255, 0), 2)
        cv.line(image, results[15][:2], results[16][:2], (0, 255, 0), 2)

        # 小指
        cv.line(image, results[17][:2], results[18][:2], (0, 255, 0), 2)
        cv.line(image, results[18][:2], results[19][:2], (0, 255, 0), 2)
        cv.line(image, results[19][:2], results[20][:2], (0, 255, 0), 2)

        # 手の平
        cv.line(image, results[0][:2], results[1][:2], (0, 255, 0), 2)
        cv.line(image, results[1][:2], results[2][:2], (0, 255, 0), 2)
        cv.line(image, results[2][:2], results[5][:2], (0, 255, 0), 2)
        cv.line(image, results[5][:2], results[9][:2], (0, 255, 0), 2)
        cv.line(image, results[9][:2], results[13][:2], (0, 255, 0), 2)
        cv.line(image, results[13][:2], results[17][:2], (0, 255, 0), 2)
        cv.line(image, results[17][:2], results[0][:2], (0, 255, 0), 2)

        cx, cy = results['palm_moment']
        cv.putText(image, results['label'], (cx - 20, cy),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv.LINE_AA)

    return image


if __name__ == '__main__':
    cap = cv.VideoCapture(0)

    # Load model
    model = MediaPipeHandsComplexity0(None)

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
        cv.imshow('MediaPipe Hands', frame)
    cap.release()
    cv.destroyAllWindows()
