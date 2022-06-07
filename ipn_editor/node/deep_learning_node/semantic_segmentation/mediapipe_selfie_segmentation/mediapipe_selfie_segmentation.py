#!/usr/bin/env python
import copy

import cv2 as cv
import numpy as np


class MediaPipeSelfieSegmentation(object):

    def __init__(
        self,
        model_path,
        model_selection,
        providers=None,
    ):
        import mediapipe as mp

        mp_selfie_segmentation = mp.solutions.selfie_segmentation

        self.selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(
            model_selection=model_selection)

    def __call__(self, image):
        # Pre process:BGR->RGB
        input_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Inference
        result = self.selfie_segmentation.process(input_image)

        # Post process:squeeze
        segmentation_map = copy.deepcopy(result.segmentation_mask)
        segmentation_map = np.expand_dims(segmentation_map, 0)
        return segmentation_map

    def get_class_num(self):
        return 1


class MediaPipeSelfieSegmentationNormal(object):

    def __init__(
        self,
        model_path,
        providers=None,
    ):
        self.model = MediaPipeSelfieSegmentation(None, model_selection=0)

    def __call__(self, image):
        return self.model(image)

    def get_class_num(self):
        return self.model.get_class_num()


class MediaPipeSelfieSegmentationLandScape(object):

    def __init__(
        self,
        model_path,
        providers=None,
    ):
        self.model = MediaPipeSelfieSegmentation(None, model_selection=1)

    def __call__(self, image):
        return self.model(image)

    def get_class_num(self):
        return self.model.get_class_num()


def get_color_map_list(num_classes, custom_color=None):
    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3 + 2] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = color_map[3:]

    if custom_color:
        color_map[:len(custom_color)] = custom_color
    return color_map


if __name__ == '__main__':
    cap = cv.VideoCapture(0)

    # Load model
    model = MediaPipeSelfieSegmentationNormal(None)

    score_th = 0.5
    class_num = model.get_class_num()

    while True:
        # Capture read
        ret, frame = cap.read()
        if not ret:
            break

        # Inference execution
        segmentation_map = model(frame)
        segmentation_map = np.where(segmentation_map > score_th, 0, 1)

        # color map list
        color_map = get_color_map_list(class_num)

        for index, mask in enumerate(segmentation_map):
            bg_image = np.zeros(frame.shape, dtype=np.uint8)
            bg_image[:] = (color_map[index * 3 + 0], color_map[index * 3 + 1],
                           color_map[index * 3 + 2])

            mask = np.stack((mask, ) * 3, axis=-1).astype('uint8')

            mask_image = np.where(mask, frame, bg_image)
            frame = cv.addWeighted(frame, 0.5, mask_image, 0.5, 1.0)

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        cv.imshow('MediaPipe SelfieSegmentation', frame)
    cap.release()
    cv.destroyAllWindows()
