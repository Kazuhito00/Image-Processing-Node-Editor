#!/usr/bin/env python
import copy

import cv2 as cv
import numpy as np


class MediaPipeFaceDetection(object):
    def __init__(
        self,
        model_path,
        model_selection,
        min_detection_confidence,
        providers=None,
    ):
        import mediapipe as mp

        mp_face_detection = mp.solutions.face_detection

        self.face_detection = mp_face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence,
        )

    def __call__(self, image):
        image_width, image_height = image.shape[1], image.shape[0]

        # Pre process:BGR->RGB
        input_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Inference
        results = self.face_detection.process(input_image)

        # Post process
        results_list = []
        if results.detections is not None:
            for detection in results.detections:
                landmark_dict = {}

                # 各キーポイント
                for id, keypoint in enumerate(
                        detection.location_data.relative_keypoints):
                    x = min(int(keypoint.x * image_width), image_width - 1)
                    y = min(int(keypoint.y * image_height), image_height - 1)
                    visibility = detection.score[0]
                    landmark_dict[id] = [x, y, visibility]

                # バウンディングボックス
                bbox = detection.location_data.relative_bounding_box
                bbox_xmin = int(bbox.xmin * image_width)
                bbox_ymin = int(bbox.ymin * image_height)
                bbox_xmax = bbox_xmin + int(bbox.width * image_width)
                bbox_ymax = bbox_ymin + int(bbox.height * image_height)
                landmark_dict['bbox'] = [
                    bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax
                ]

                results_list.append(copy.deepcopy(landmark_dict))

        return results_list


class MediaPipeFaceDetectionModel0(object):
    def __init__(
        self,
        model_path,
        providers=None,
    ):
        self.model = MediaPipeFaceDetection(
            None,
            model_selection=0,
            min_detection_confidence=0.7,
        )

    def __call__(self, image):
        return self.model(image)


class MediaPipeFaceDetectionModel1(object):
    def __init__(
        self,
        model_path,
        providers=None,
    ):
        self.model = MediaPipeFaceDetection(
            None,
            model_selection=1,
            min_detection_confidence=0.7,
        )

    def __call__(self, image):
        return self.model(image)


def draw_landmarks(image, results_list, score_th):
    for results in results_list:
        # キーポイント
        for id in range(6):
            if score_th > results[id][2]:
                continue
            landmark_x, landmark_y = results[id][0], results[id][1]
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), -1)

        # バウンディングボックス
        bbox = results.get('bbox', None)
        if bbox is not None:
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
    model = MediaPipeFaceDetectionModel0(None)

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
