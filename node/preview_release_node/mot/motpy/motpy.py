# -*- coding: utf-8 -*-
from node.preview_release_node.mot.motpy.tracker import Detection, MultiObjectTracker


class Motpy(object):
    def __init__(
        self,
        fps=30,
        min_steps_alive=3,
        max_staleness=5,
        order_pos=1,
        dim_pos=2,
        order_size=0,
        dim_size=2,
        q_var_pos=5000.0,
        r_var_pos=0.1,
        min_iou=0.25,
        multi_match_min_iou=0.93,
    ):
        self.min_steps_alive = min_steps_alive

        self.tracker = MultiObjectTracker(
            dt=1 / fps,
            tracker_kwargs={'max_staleness': max_staleness},
            model_spec={
                'order_pos': order_pos,
                'dim_pos': dim_pos,
                'order_size': order_size,
                'dim_size': dim_size,
                'q_var_pos': q_var_pos,
                'r_var_pos': r_var_pos
            },
            matching_fn_kwargs={
                'min_iou': min_iou,
                'multi_match_min_iou': multi_match_min_iou
            },
        )

    def __call__(self, _, bboxes, scores, class_ids):
        detections = [
            Detection(box=b, score=s, class_id=l)
            for b, s, l in zip(bboxes, scores, class_ids)
        ]

        _ = self.tracker.step(detections=detections)
        results = self.tracker.active_tracks(
            max_staleness_to_positive_ratio=3.0,
            max_staleness=999,
            min_steps_alive=self.min_steps_alive,
        )

        tracker_ids = []
        tracker_bboxes = []
        tracker_class_ids = []
        tracker_scores = []
        for result in results:
            tracker_ids.append(result.id)
            tracker_bboxes.append(result.box.tolist())
            tracker_class_ids.append(result.class_id)
            tracker_scores.append(result.score)

        return tracker_ids, tracker_bboxes, tracker_scores, tracker_class_ids
