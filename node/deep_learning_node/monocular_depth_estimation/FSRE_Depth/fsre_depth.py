#!/usr/bin/env python
import cv2 as cv
import numpy as np
import onnxruntime


class FSRE_Depth(object):
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
        self.output_name = self.onnx_session.get_outputs()[0].name

        # 各種設定
        self.input_shape = self.input_detail.shape[2:]

    def __call__(self, image):
        image_width, image_height = image.shape[1], image.shape[0]

        # Pre process:Resize, BGR->RGB, Transpose, float32 cast
        input_image = cv.resize(
            image,
            dsize=(self.input_shape[1], self.input_shape[0]),
        )
        input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, axis=0)
        input_image = input_image.astype('float32')
        input_image = input_image / 255.0

        # Inference
        input_name = self.onnx_session.get_inputs()[0].name
        result = self.onnx_session.run(None, {input_name: input_image})

        # Post process:squeeze, RGB->BGR, Transpose, uint8 cast
        depth_map = result[0]
        depth_map = depth_map * 255.0
        depth_map = np.asarray(depth_map, dtype="uint8")

        depth_map = depth_map.reshape(self.input_shape[0], self.input_shape[1])
        output_image = cv.resize(
            depth_map,
            dsize=(image_width, image_height),
        )

        return output_image


if __name__ == '__main__':
    cap = cv.VideoCapture(0)

    # Load model
    model_path = 'fsre_depth_192x320/fsre_depth_full_192x320.onnx'
    model = FSRE_Depth(model_path)

    while True:
        # Capture read
        ret, frame = cap.read()
        if not ret:
            break
        image_width, image_height = frame.shape[1], frame.shape[0]

        # Inference execution
        output_image = model(frame)

        depth_image = cv.applyColorMap(output_image, cv.COLORMAP_JET)
        depth_image = cv.resize(depth_image, dsize=(image_width, image_height))

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        cv.imshow('FSRE-Depth Input', frame)
        cv.imshow('FSRE-Depth Output', output_image)

    cap.release()
    cv.destroyAllWindows()
