#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse
from itertools import product

import cv2 as cv
import numpy as np
import onnxruntime


class YuNet(object):

    # Feature map用定義
    MIN_SIZES = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
    STEPS = [8, 16, 32, 64]
    VARIANCE = [0.1, 0.2]

    def __init__(
        self,
        model_path,
        input_shape=[160, 120],
        conf_th=0.6,
        nms_th=0.3,
        topk=5000,
        keep_topk=750,
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

        self.input_name = self.onnx_session.get_inputs()[0].name
        output_name_01 = self.onnx_session.get_outputs()[0].name
        output_name_02 = self.onnx_session.get_outputs()[1].name
        output_name_03 = self.onnx_session.get_outputs()[2].name
        self.output_names = [output_name_01, output_name_02, output_name_03]

        # 各種設定
        self.input_shape = input_shape  # [w, h]
        self.conf_th = conf_th
        self.nms_th = nms_th
        self.topk = topk
        self.keep_topk = keep_topk

        # priors生成
        self.priors = None
        self._generate_priors()

    def __call__(self, image):
        image_width, image_height = image.shape[1], image.shape[0]

        # 前処理
        temp_image = copy.deepcopy(image)
        temp_image = self._preprocess(temp_image)

        # 推論
        result = self.onnx_session.run(
            self.output_names,
            {self.input_name: temp_image},
        )

        # 後処理
        bboxes, landmarks, scores = self._postprocess(result)

        results_list = []
        for bbox, landmark, score in zip(bboxes, landmarks, scores):
            landmark_dict = {}

            # 各キーポイント
            for id, keypoint in enumerate(landmark):
                x = min(int((keypoint[0] / self.input_shape[0]) * image_width),
                        image_width - 1)
                y = min(
                    int((keypoint[1] / self.input_shape[1]) * image_height),
                    image_height - 1)
                visibility = score
                landmark_dict[id] = [x, y, visibility]

            # バウンディングボックス
            bbox_xmin = int((bbox[0] / self.input_shape[0]) * image_width)
            bbox_ymin = int((bbox[1] / self.input_shape[1]) * image_height)
            bbox_xmax = bbox_xmin + int(
                (bbox[2] / self.input_shape[0]) * image_width)
            bbox_ymax = bbox_ymin + int(
                (bbox[3] / self.input_shape[1]) * image_height)
            landmark_dict['bbox'] = [
                bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax
            ]

            results_list.append(copy.deepcopy(landmark_dict))

        return results_list

    def _generate_priors(self):
        w, h = self.input_shape

        feature_map_2th = [
            int(int((h + 1) / 2) / 2),
            int(int((w + 1) / 2) / 2)
        ]
        feature_map_3th = [
            int(feature_map_2th[0] / 2),
            int(feature_map_2th[1] / 2)
        ]
        feature_map_4th = [
            int(feature_map_3th[0] / 2),
            int(feature_map_3th[1] / 2)
        ]
        feature_map_5th = [
            int(feature_map_4th[0] / 2),
            int(feature_map_4th[1] / 2)
        ]
        feature_map_6th = [
            int(feature_map_5th[0] / 2),
            int(feature_map_5th[1] / 2)
        ]

        feature_maps = [
            feature_map_3th, feature_map_4th, feature_map_5th, feature_map_6th
        ]

        priors = []
        for k, f in enumerate(feature_maps):
            min_sizes = self.MIN_SIZES[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / w
                    s_ky = min_size / h

                    cx = (j + 0.5) * self.STEPS[k] / w
                    cy = (i + 0.5) * self.STEPS[k] / h

                    priors.append([cx, cy, s_kx, s_ky])

        self.priors = np.array(priors, dtype=np.float32)

    def _preprocess(self, image):
        # BGR -> RGB 変換
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # リサイズ
        image = cv.resize(
            image,
            (self.input_shape[0], self.input_shape[1]),
            interpolation=cv.INTER_LINEAR,
        )

        # リシェイプ
        image = image.astype(np.float32)
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, 3, self.input_shape[1], self.input_shape[0])

        return image

    def _postprocess(self, result):
        # 結果デコード
        dets = self._decode(result)

        # NMS
        keepIdx = cv.dnn.NMSBoxes(
            bboxes=dets[:, 0:4].tolist(),
            scores=dets[:, -1].tolist(),
            score_threshold=self.conf_th,
            nms_threshold=self.nms_th,
            top_k=self.topk,
        )

        # bboxes, landmarks, scores へ成形
        scores = []
        bboxes = []
        landmarks = []
        if len(keepIdx) > 0:
            dets = dets[keepIdx]
            if len(dets.shape) == 3:
                dets = np.squeeze(dets, axis=1)
            for det in dets[:self.keep_topk]:
                scores.append(det[-1])
                bboxes.append(det[0:4].astype(np.int32))
                landmarks.append(det[4:14].astype(np.int32).reshape((5, 2)))

        return bboxes, landmarks, scores

    def _decode(self, result):
        loc, conf, iou = result

        # スコア取得
        cls_scores = conf[:, 1]
        iou_scores = iou[:, 0]

        _idx = np.where(iou_scores < 0.)
        iou_scores[_idx] = 0.
        _idx = np.where(iou_scores > 1.)
        iou_scores[_idx] = 1.
        scores = np.sqrt(cls_scores * iou_scores)
        scores = scores[:, np.newaxis]

        scale = np.array(self.input_shape)

        # バウンディングボックス取得
        bboxes = np.hstack(
            ((self.priors[:, 0:2] +
              loc[:, 0:2] * self.VARIANCE[0] * self.priors[:, 2:4]) * scale,
             (self.priors[:, 2:4] * np.exp(loc[:, 2:4] * self.VARIANCE)) *
             scale))
        bboxes[:, 0:2] -= bboxes[:, 2:4] / 2

        # ランドマーク取得
        landmarks = np.hstack(
            ((self.priors[:, 0:2] +
              loc[:, 4:6] * self.VARIANCE[0] * self.priors[:, 2:4]) * scale,
             (self.priors[:, 0:2] +
              loc[:, 6:8] * self.VARIANCE[0] * self.priors[:, 2:4]) * scale,
             (self.priors[:, 0:2] +
              loc[:, 8:10] * self.VARIANCE[0] * self.priors[:, 2:4]) * scale,
             (self.priors[:, 0:2] +
              loc[:, 10:12] * self.VARIANCE[0] * self.priors[:, 2:4]) * scale,
             (self.priors[:, 0:2] +
              loc[:, 12:14] * self.VARIANCE[0] * self.priors[:, 2:4]) * scale))

        dets = np.hstack((bboxes, landmarks, scores))

        return dets


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument(
        "--model",
        type=str,
        default='model/face_detection_yunet_120x160.onnx',
    )
    parser.add_argument(
        '--input_shape',
        type=str,
        default="160,120",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        '--score_th',
        type=float,
        default=0.6,
        help='Conf confidence',
    )
    parser.add_argument(
        '--nms_th',
        type=float,
        default=0.3,
        help='NMS IoU threshold',
    )
    parser.add_argument(
        '--topk',
        type=int,
        default=5000,
    )
    parser.add_argument(
        '--keep_topk',
        type=int,
        default=750,
    )

    args = parser.parse_args()

    return args


def main():
    # 引数解析 #################################################################
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    if args.movie is not None:
        cap_device = args.movie

    model_path = args.model
    input_shape = tuple(map(int, args.input_shape.split(',')))
    score_th = args.score_th
    nms_th = args.nms_th
    topk = args.topk
    keep_topk = args.keep_topk

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード #############################################################
    yunet = YuNet(
        model_path=model_path,
        input_shape=input_shape,
        conf_th=score_th,
        nms_th=nms_th,
        topk=topk,
        keep_topk=keep_topk,
    )

    while True:
        # カメラキャプチャ ################################################
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)

        # 推論実施 ########################################################
        results_list = yunet(frame)

        # デバッグ描画
        debug_image = draw_debug(frame, results_list, score_th)

        # キー処理(ESC：終了) ##############################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # 画面反映 #########################################################
        cv.imshow('YuNet ONNX Sample', debug_image)

    cap.release()
    cv.destroyAllWindows()


def draw_debug(image, results_list, score_th):
    for results in results_list:
        # キーポイント
        for id in range(5):
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
    main()
