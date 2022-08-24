#!/usr/bin/env python
import cv2 as cv
import numpy as np
import onnxruntime


class AGLLNet(object):
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
        result = self.onnx_session.run([self.output_name],
                                       {self.input_name: input_image})

        # Post process:squeeze, RGB->BGR, Transpose, uint8 cast
        result = np.array(result)[0]
        output_image = result[0, :, :, 4:7]
        output_image = np.clip(output_image * 255.0, 0, 255).astype(np.uint8)
        output_image = cv.cvtColor(output_image, cv.COLOR_RGB2BGR)

        output_image = cv.resize(
            output_image,
            dsize=(image_width, image_height),
        )

        return output_image


if __name__ == '__main__':
    cap = cv.VideoCapture(0)

    # Load model
    model_path = 'saved_model_256x256/model_float32.onnx'
    model = AGLLNet(model_path)

    while True:
        # Capture read
        ret, frame = cap.read()
        if not ret:
            break

        # Inference execution
        output_image = model(frame)

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        cv.imshow('AGLLNet Input', frame)
        cv.imshow('AGLLNet Output', output_image)

    cap.release()
    cv.destroyAllWindows()
