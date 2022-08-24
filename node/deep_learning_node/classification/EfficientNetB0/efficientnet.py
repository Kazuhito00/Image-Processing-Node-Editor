#!/usr/bin/env python
import cv2 as cv
import numpy as np
import onnxruntime


class EfficientNet(object):

    def __init__(
        self,
        model_path,
        type='B0',
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
        if type == 'B0':
            self.input_shape = (224, 224)
        elif type == 'B1':
            self.input_shape = (240, 240)
        elif type == 'B2':
            self.input_shape = (260, 260)
        elif type == 'B3':
            self.input_shape = (300, 300)
        elif type == 'B4':
            self.input_shape = (380, 380)
        elif type == 'B5':
            self.input_shape = (456, 456)
        elif type == 'B6':
            self.input_shape = (528, 528)
        elif type == 'B7':
            self.input_shape = (600, 600)

    def __call__(self, image, top_k=5):
        # Pre process:Resize, BGR->RGB, Transpose, float32 cast
        input_image = cv.resize(
            image,
            dsize=(self.input_shape[1], self.input_shape[0]),
        )
        input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
        input_image = np.expand_dims(input_image, axis=0).astype('float32')

        # Inference
        input_name = self.onnx_session.get_inputs()[0].name
        result = self.onnx_session.run(None, {input_name: input_image})

        # sort result
        result = np.array(result).squeeze()
        result_sorted_index = np.argsort(result)[::-1][:top_k]
        class_scores = result[result_sorted_index]
        class_ids = result_sorted_index

        return class_scores, class_ids


class EfficientNetB0(object):
    _model = None

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
        self._model = EfficientNet(model_path, 'B0', providers)

    def __call__(self, image, top_k=5):
        return self._model(image, top_k)


class EfficientNetB1(object):
    _model = None

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
        self._model = EfficientNet(model_path, 'B1', providers)

    def __call__(self, image, top_k=5):
        return self._model(image, top_k)


class EfficientNetB2(object):
    _model = None

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
        self._model = EfficientNet(model_path, 'B2', providers)

    def __call__(self, image, top_k=5):
        return self._model(image, top_k)


class EfficientNetB3(object):
    _model = None

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
        self._model = EfficientNet(model_path, 'B3', providers)

    def __call__(self, image, top_k=5):
        return self._model(image, top_k)


class EfficientNetB4(object):
    _model = None

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
        self._model = EfficientNet(model_path, 'B4', providers)

    def __call__(self, image, top_k=5):
        return self._model(image, top_k)


class EfficientNetB5(object):
    _model = None

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
        self._model = EfficientNet(model_path, 'B5', providers)

    def __call__(self, image, top_k=5):
        return self._model(image, top_k)


class EfficientNetB6(object):
    _model = None

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
        self._model = EfficientNet(model_path, 'B6', providers)

    def __call__(self, image, top_k=5):
        return self._model(image, top_k)


class EfficientNetB7(object):
    _model = None

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
        self._model = EfficientNet(model_path, 'B7', providers)

    def __call__(self, image, top_k=5):
        return self._model(image, top_k)


if __name__ == '__main__':
    cap = cv.VideoCapture(0)

    # Load model
    model_path = 'model/EfficientNetB0.onnx'
    model = EfficientNetB0(model_path)

    while True:
        # Capture read
        ret, frame = cap.read()
        if not ret:
            break

        # Inference execution
        class_scores, class_ids = model(frame)
        print(class_scores)
        print(class_ids)

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        cv.imshow('EfficientNet Input', frame)

    cap.release()
    cv.destroyAllWindows()
