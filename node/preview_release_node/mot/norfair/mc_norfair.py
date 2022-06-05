# -*- coding: utf-8 -*-
import numpy as np

from node.preview_release_node.mot.norfair.tracker import Detection
from node.preview_release_node.mot.norfair.tracker import Tracker as NorfairTracker


def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)


class MultiClassNorfair(object):

    def __init__(
        self,
        fps=30,
        max_distance_between_points=30,
    ):
        self.fps = fps
        self.max_distance_between_points = max_distance_between_points

        # Norfair保持用Dict生成
        self.tracker_dict = {}

    def __call__(self, _, bboxes, scores, class_ids):
        # 未トラッキングのクラスのトラッキングインスタンスを追加
        for class_id in np.unique(class_ids):
            if not int(class_id) in self.tracker_dict:
                self.tracker_dict[int(class_id)] = NorfairTracker(
                    distance_function=euclidean_distance,
                    distance_threshold=self.max_distance_between_points,
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

            # トラッカー用変数に格納
            detections = []
            for bbox, score in zip(target_bboxes, target_scores):
                points = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]])
                points_score = np.array([score, score])

                detection = Detection(points=points, scores=points_score)
                detections.append(detection)

            # トラッカー更新
            results = self.tracker_dict[class_id].update(detections=detections)

            # 結果格納
            for result in results:
                x1 = result.estimate[0][0]
                y1 = result.estimate[0][1]
                x2 = result.estimate[1][0]
                y2 = result.estimate[1][1]

                t_ids.append(str(int(class_id)) + '_' + str(result.id))
                t_bboxes.append([x1, y1, x2, y2])
                t_scores.append(score)
                t_class_ids.append(int(class_id))

        return t_ids, t_bboxes, t_scores, t_class_ids
