#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy

import numpy as np

from node.preview_release_node.mot.bytetrack.tracker.byte_tracker import BYTETracker


class dict_dot_notation(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class MultiClassByteTrack(object):
    def __init__(
        self,
        fps=30,
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        min_box_area=10,
        mot20=False,
    ):
        self.min_box_area = min_box_area

        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.mot20 = mot20
        self.fps = fps

        # ByteTracker保持用Dict生成
        self.tracker_dict = {}

    def __call__(
        self,
        image,
        bboxes,
        scores,
        class_ids,
    ):
        # 未トラッキングのクラスのトラッキングインスタンスを追加
        for class_id in np.unique(class_ids):
            if not int(class_id) in self.tracker_dict:
                self.tracker_dict[int(class_id)] = BYTETracker(
                    args=dict_dot_notation({
                        'track_thresh': self.track_thresh,
                        'track_buffer': self.track_buffer,
                        'match_thresh': self.match_thresh,
                        'mot20': self.mot20,
                    }),
                    frame_rate=self.fps,
                )

        t_ids = []
        t_bboxes = []
        t_scores = []
        t_class_ids = []
        for class_id in self.tracker_dict.keys():
            # 対象クラス抽出
            target_index = np.in1d(class_ids, np.array(int(class_id)))

            if len(target_index) == 0:
                continue

            target_bboxes = np.array(bboxes)[target_index]
            target_scores = np.array(scores)[target_index]
            target_class_ids = np.array(class_ids)[target_index]

            # トラッカー用変数に格納
            detections = [[*b, s, l] for b, s, l in zip(
                target_bboxes, target_scores, target_class_ids)]
            detections = np.array(detections)

            # トラッカー更新
            result = self._tracker_update(
                self.tracker_dict[class_id],
                image,
                detections,
            )

            # 結果格納
            for bbox, score, t_id in zip(result[0], result[1], result[2]):
                t_ids.append(str(int(class_id)) + '_' + str(t_id))
                t_bboxes.append(bbox)
                t_scores.append(score)
                t_class_ids.append(int(class_id))

        return t_ids, t_bboxes, t_scores, t_class_ids

    def _tracker_update(self, tracker, image, detections):
        image_info = {'id': 0}
        image_info['image'] = copy.deepcopy(image)
        image_info['width'] = image.shape[1]
        image_info['height'] = image.shape[0]

        online_targets = []
        if detections is not None and len(detections) != 0:
            online_targets = tracker.update(
                detections[:, :-1],
                [image_info['height'], image_info['width']],
                [image_info['height'], image_info['width']],
            )

        online_tlwhs = []
        online_ids = []
        online_scores = []
        for online_target in online_targets:
            tlwh = online_target.tlwh
            track_id = online_target.track_id
            if tlwh[2] * tlwh[3] > self.min_box_area:
                online_tlwhs.append(
                    np.array([
                        tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]
                    ]))
                online_ids.append(track_id)
                online_scores.append(online_target.score)

        return online_tlwhs, online_scores, online_ids
