#!/usr/bin/env python
import cv2 as cv
import numpy as np
import onnxruntime


class DeepLabV3(object):
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
        self.input_shape = self.input_detail.shape[1:3]

    def __call__(self, image):
        image_width, image_height = image.shape[1], image.shape[0]

        # Pre process:Resize, BGR->RGB, Transpose, float32 cast
        input_image = cv.resize(
            image,
            dsize=(self.input_shape[1], self.input_shape[0]),
        )
        input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
        input_image = np.expand_dims(input_image, axis=0)
        input_image = input_image.astype('float32')

        input_image = input_image / 127.5 - 1

        # Inference
        input_name = self.onnx_session.get_inputs()[0].name
        result = self.onnx_session.run(None, {input_name: input_image})

        # Post process:squeeze, resize, argmax
        segmentation_map = np.squeeze(result[0])
        segmentation_map = cv.resize(
            segmentation_map,
            dsize=(image_width, image_height),
            interpolation=cv.INTER_LINEAR,
        )
        segmentation_map = segmentation_map.transpose(2, 0, 1)

        # Interface adjustment
        segmentation_map = np.argmax(segmentation_map, axis=0)

        class_num = self.get_class_num()

        segmentation_map_list = []
        for index in range(0, class_num):
            mask = np.where(segmentation_map == index, 1.0, 0.0)
            segmentation_map_list.append(mask)
        segmentation_map = np.array(segmentation_map_list)

        return segmentation_map

    def get_class_num(self):
        return self.output_detail.shape[3]


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
    model_path = 'model/deeplab_v3-mobilenetv2_1_default_1.onnx'
    model = DeepLabV3(model_path)

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
        cv.imshow('DeepLabV3', frame)
    cap.release()
    cv.destroyAllWindows()
