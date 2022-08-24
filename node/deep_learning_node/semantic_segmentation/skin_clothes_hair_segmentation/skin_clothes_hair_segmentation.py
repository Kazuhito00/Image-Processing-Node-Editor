#!/usr/bin/env python
import cv2 as cv
import numpy as np
import onnxruntime


class SkinClothesHairSegmentation(object):
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
        # モデル読み込み
        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            providers=providers,
        )

        self.input_detail = self.onnx_session.get_inputs()[0]
        self.input_name = self.input_detail.name
        self.output_detail = self.onnx_session.get_outputs()[0]

        # 各種設定
        self.input_shape = self.input_detail.shape[-2:]

    def __call__(self, image):
        image_width, image_height = image.shape[1], image.shape[0]

        # Pre process:Resize, BGR->RGB, Transpose, float32 cast
        input_image = cv.resize(
            image,
            dsize=(self.input_shape[1], self.input_shape[0]),
        )
        input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x = (input_image / 255 - mean) / std
        x = x.transpose(2, 0, 1).astype('float32')
        x = x.reshape(-1, 3, self.input_shape[0], self.input_shape[1])

        # Inference
        input_name = self.onnx_session.get_inputs()[0].name
        result = self.onnx_session.run(None, {input_name: x})

        # Post process:squeeze
        segmentation_map = np.array(result).squeeze()
        segmentation_map = segmentation_map.transpose(1, 2, 0)
        segmentation_map = cv.resize(
            segmentation_map,
            dsize=(image_width, image_height),
            interpolation=cv.INTER_LINEAR,
        )
        segmentation_map = segmentation_map.transpose(2, 0, 1)

        return segmentation_map

    def get_class_num(self):
        return self.output_detail.shape[1]


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
    model_path = 'model/DeepLabV3Plus(timm-mobilenetv3_small_100)_452_2.16M_0.8385/best_model_simplifier.onnx'
    model = SkinClothesHairSegmentation(model_path)

    score_th = 0.5
    class_num = model.get_class_num()

    while True:
        # Capture read
        ret, frame = cap.read()
        if not ret:
            break
        image_width, image_height = frame.shape[1], frame.shape[0]

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
        cv.imshow('Road Segmentation', frame)
    cap.release()
    cv.destroyAllWindows()
