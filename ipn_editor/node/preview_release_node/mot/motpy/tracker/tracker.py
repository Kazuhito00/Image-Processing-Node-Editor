import time
import uuid
from collections.abc import Iterable
from typing import (Any, Callable, Dict, List, Optional, Sequence, Tuple, Type,
                    Union)

import numpy as np
import scipy
from filterpy.kalman import KalmanFilter

# from motpy.core import Box, Detection, Track, Vector, setup_logger
# from motpy.metrics import angular_similarity, calculate_iou
# from motpy.model import Model, ModelPreset
from node.preview_release_node.mot.motpy.tracker.core import Box, Detection, Track, Vector, setup_logger
from node.preview_release_node.mot.motpy.tracker.metrics import angular_similarity, calculate_iou
from node.preview_release_node.mot.motpy.tracker.model import Model, ModelPreset

logger = setup_logger(__name__)


def get_kalman_object_tracker(model: Model,
                              x0: Optional[Vector] = None) -> KalmanFilter:
    """ returns Kalman-based tracker based on a specified motion model spec.
        e.g. for spec = {'order_pos': 1, 'dim_pos': 2, 'order_size': 0, 'dim_size': 1}
        we expect the following setup:
        state x, x', y, y', w, h
        where x and y are centers of boxes
              w and h are width and height
    """

    tracker = KalmanFilter(dim_x=model.state_length,
                           dim_z=model.measurement_length)
    tracker.F = model.build_F()
    tracker.Q = model.build_Q()
    tracker.H = model.build_H()
    tracker.R = model.build_R()
    tracker.P = model.build_P()

    if x0 is not None:
        tracker.x = x0

    return tracker


DEFAULT_MODEL_SPEC = ModelPreset.constant_velocity_and_static_box_size_2d.value


def exponential_moving_average_fn(gamma: float) -> Callable:
    def fn(old, new):
        if new is None:
            return old

        if isinstance(new, Iterable):
            new = np.array(new)

        if old is None:
            return new  # first call

        if isinstance(old, Iterable):
            old = np.array(old)

        return gamma * old + (1 - gamma) * new

    return fn


class SingleObjectTracker:
    def __init__(self,
                 max_staleness: float = 12.0,
                 smooth_score_gamma: float = 0.8,
                 smooth_feature_gamma: float = 0.9,
                 score0: Optional[float] = None,
                 class_id0: Optional[int] = None):
        self.id: str = str(uuid.uuid4())
        self.steps_alive: int = 1
        self.steps_positive: int = 1
        self.staleness: float = 0.0
        self.max_staleness: float = max_staleness

        self.update_score_fn: Callable = exponential_moving_average_fn(
            smooth_score_gamma)
        self.update_feature_fn: Callable = exponential_moving_average_fn(
            smooth_feature_gamma)

        self.score: Optional[float] = score0
        self.feature: Optional[Vector] = None

        self.class_id_counts: Dict = dict()
        self.class_id: Optional[int] = self.update_class_id(class_id0)

        logger.debug(f'creating new tracker {self.id}')

    def box(self) -> Box:
        raise NotImplementedError()

    def is_invalid(self) -> bool:
        raise NotImplementedError()

    def _predict(self) -> None:
        raise NotImplementedError()

    def predict(self) -> None:
        self._predict()
        self.steps_alive += 1

    def update_class_id(self, class_id: Optional[int]) -> Optional[int]:
        """ find most frequent prediction of class_id in recent K class_ids """
        if class_id is None:
            return None

        if class_id in self.class_id_counts:
            self.class_id_counts[class_id] += 1
        else:
            self.class_id_counts[class_id] = 1

        return max(self.class_id_counts, key=self.class_id_counts.get)

    def _update_box(self, detection: Detection) -> None:
        raise NotImplementedError()

    def update(self, detection: Detection) -> None:
        self._update_box(detection)

        self.steps_positive += 1

        self.class_id = self.update_class_id(detection.class_id)
        self.score = self.update_score_fn(old=self.score, new=detection.score)
        self.feature = self.update_feature_fn(old=self.feature,
                                              new=detection.feature)

        # reduce the staleness of a tracker, faster than growth rate
        self.unstale(rate=3)

    def stale(self, rate: float = 1.0) -> float:
        self.staleness += rate
        return self.staleness

    def unstale(self, rate: float = 2.0) -> float:
        self.staleness = max(0, self.staleness - rate)
        return self.staleness

    def is_stale(self) -> bool:
        return self.staleness >= self.max_staleness

    def __repr__(self) -> str:
        return f'(box: {str(self.box())}, score: {self.score}, class_id: {self.class_id}, staleness: {self.staleness:.2f})'


class KalmanTracker(SingleObjectTracker):
    """ A single object tracker using Kalman filter with specified motion model specification """
    def __init__(self,
                 model_kwargs: dict = DEFAULT_MODEL_SPEC,
                 x0: Optional[Vector] = None,
                 box0: Optional[Box] = None,
                 **kwargs) -> None:

        super(KalmanTracker, self).__init__(**kwargs)

        self.model_kwargs: dict = model_kwargs
        self.model = Model(**self.model_kwargs)

        if x0 is None:
            x0 = self.model.box_to_x(box0)

        self._tracker: KalmanFilter = get_kalman_object_tracker(
            model=self.model, x0=x0)

    def _predict(self) -> None:
        self._tracker.predict()

    def _update_box(self, detection: Detection) -> None:
        z = self.model.box_to_z(detection.box)
        self._tracker.update(z)

    def box(self) -> Box:
        return self.model.x_to_box(self._tracker.x)

    def is_invalid(self) -> bool:
        try:
            has_nans = any(np.isnan(self._tracker.x))
            return has_nans
        except Exception as e:
            logger.warning(f'invalid tracker - exception: {e}')
            return True


class SimpleTracker(SingleObjectTracker):
    """ A simple single tracker with no motion modeling and box update using exponential moving averege """
    def __init__(self,
                 box0: Optional[Box] = None,
                 box_update_gamma: float = 0.5,
                 **kwargs):

        super(SimpleTracker, self).__init__(**kwargs)
        self._box: Box = box0

        self.update_box_fn: Callable = exponential_moving_average_fn(
            box_update_gamma)

    def _predict(self) -> None:
        pass

    def _update_box(self, detection: Detection) -> None:
        self._box = self.update_box_fn(old=self._box, new=detection.box)

    def box(self) -> Box:
        return self._box

    def is_invalid(self) -> bool:
        try:
            return any(np.isnan(self._box))
        except Exception as e:
            logger.warning(f'invalid tracker - exception: {e}')
            return True


""" assignment cost calculation & matching methods """


def _sequence_has_none(seq: Sequence[Any]) -> bool:
    return any([r is None for r in seq])


def cost_matrix_iou_feature(
        trackers: Sequence[SingleObjectTracker],
        detections: Sequence[Detection],
        feature_similarity_fn=angular_similarity,
        feature_similarity_beta: float = None
) -> Tuple[np.ndarray, np.ndarray]:

    # boxes
    b1 = np.array([t.box() for t in trackers])
    b2 = np.array([d.box for d in detections])

    # box iou
    inferred_dim = int(len(b1[0]) / 2)
    iou_mat = calculate_iou(b1, b2, dim=inferred_dim)

    # feature similarity
    if feature_similarity_beta is not None:
        # get features
        f1 = [t.feature for t in trackers]
        f2 = [d.feature for d in detections]

        if _sequence_has_none(f1) or _sequence_has_none(f2):
            # fallback to pure IOU due to missing features
            apt_mat = iou_mat
        else:
            sim_mat = feature_similarity_fn(f1, f2)
            sim_mat = feature_similarity_beta + (
                1 - feature_similarity_beta) * sim_mat

            # combined aptitude
            apt_mat = np.multiply(iou_mat, sim_mat)
    else:
        apt_mat = iou_mat

    cost_mat = -1.0 * apt_mat
    return cost_mat, iou_mat


EPS = 1e-7


def match_by_cost_matrix(trackers: Sequence[SingleObjectTracker],
                         detections: Sequence[Detection],
                         min_iou: float = 0.1,
                         multi_match_min_iou: float = 1. + EPS,
                         **kwargs) -> np.ndarray:
    if len(trackers) == 0 or len(detections) == 0:
        return []

    cost_mat, iou_mat = cost_matrix_iou_feature(trackers, detections, **kwargs)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_mat)

    matches = []
    for r, c in zip(row_ind, col_ind):
        # check linear assignment winner
        if iou_mat[r, c] >= min_iou:
            matches.append((r, c))

        # check other high IOU detections
        if multi_match_min_iou < 1.:
            for c2 in range(iou_mat.shape[1]):
                if c2 != c and iou_mat[r, c2] > multi_match_min_iou:
                    matches.append((r, c2))

    return np.array(matches)


class BaseMatchingFunction:
    def __call__(self, trackers: Sequence[SingleObjectTracker],
                 detections: Sequence[Detection]) -> np.ndarray:
        raise NotImplementedError()


class IOUAndFeatureMatchingFunction(BaseMatchingFunction):
    """ class implements the basic matching function, taking into account
    detection boxes overlap measured using IOU metric and optional 
    feature similarity measured with a specified metric """
    def __init__(self,
                 min_iou: float = 0.1,
                 multi_match_min_iou: float = 1. + EPS,
                 feature_similarity_fn: Callable = angular_similarity,
                 feature_similarity_beta: Optional[float] = None) -> None:
        self.min_iou = min_iou
        self.multi_match_min_iou = multi_match_min_iou
        self.feature_similarity_fn = feature_similarity_fn
        self.feature_similarity_beta = feature_similarity_beta

    def __call__(self, trackers: Sequence[SingleObjectTracker],
                 detections: Sequence[Detection]) -> np.ndarray:
        return match_by_cost_matrix(
            trackers,
            detections,
            min_iou=self.min_iou,
            multi_match_min_iou=self.multi_match_min_iou,
            feature_similarity_fn=self.feature_similarity_fn,
            feature_similarity_beta=self.feature_similarity_beta)


class MultiObjectTracker:
    def __init__(self,
                 dt: float,
                 model_spec: Union[str, Dict] = DEFAULT_MODEL_SPEC,
                 matching_fn: Optional[BaseMatchingFunction] = None,
                 tracker_kwargs: Dict = None,
                 matching_fn_kwargs: Dict = None,
                 active_tracks_kwargs: Dict = None) -> None:
        """
            model_spec specifies the dimension and order for position and size of the object
            matching_fn determines the strategy on which the trackers and detections are assigned.

            tracker_kwargs are passed to each single object tracker
            active_tracks_kwargs limits surfacing of fresh/fading out tracks
        """

        self.trackers: List[SingleObjectTracker] = []

        # kwargs to be passed to each single object tracker
        self.tracker_kwargs: Dict = tracker_kwargs if tracker_kwargs is not None else {}
        self.tracker_clss: Optional[Type[SingleObjectTracker]] = None

        # translate model specification into single object tracker to be used
        if model_spec is None:
            self.tracker_clss = SimpleTracker
            if dt is not None:
                logger.warning(
                    'specified dt is ignored in simple tracker mode')
        elif isinstance(model_spec, dict):
            self.tracker_clss = KalmanTracker
            self.tracker_kwargs['model_kwargs'] = model_spec
            self.tracker_kwargs['model_kwargs']['dt'] = dt
        elif isinstance(model_spec,
                        str) and model_spec in ModelPreset.__members__:
            self.tracker_clss = KalmanTracker
            self.tracker_kwargs['model_kwargs'] = ModelPreset[model_spec].value
            self.tracker_kwargs['model_kwargs']['dt'] = dt
        else:
            raise NotImplementedError(f'unsupported motion model {model_spec}')

        logger.debug(
            f'using single tracker of class: {self.tracker_clss} with kwargs: {self.tracker_kwargs}'
        )

        self.matching_fn: BaseMatchingFunction = matching_fn
        self.matching_fn_kwargs: Dict = matching_fn_kwargs if matching_fn_kwargs is not None else {}
        if self.matching_fn is None:
            self.matching_fn = IOUAndFeatureMatchingFunction(
                **self.matching_fn_kwargs)

        # kwargs to be used when self.step returns active tracks
        self.active_tracks_kwargs: Dict = active_tracks_kwargs if active_tracks_kwargs is not None else {}
        logger.debug('using active_tracks_kwargs: %s' %
                     str(self.active_tracks_kwargs))

        self.detections_matched_ids = []

    def active_tracks(self,
                      max_staleness_to_positive_ratio: float = 3.0,
                      max_staleness: float = 999,
                      min_steps_alive: int = -1) -> List[Track]:
        """ returns all active tracks after optional filtering by tracker steps count and staleness """

        tracks: List[Track] = []
        for tracker in self.trackers:
            cond1 = tracker.staleness / tracker.steps_positive < max_staleness_to_positive_ratio  # early stage
            cond2 = tracker.staleness < max_staleness
            cond3 = tracker.steps_alive >= min_steps_alive
            if cond1 and cond2 and cond3:
                tracks.append(
                    Track(id=tracker.id,
                          box=tracker.box(),
                          score=tracker.score,
                          class_id=tracker.class_id))

        logger.debug('active/all tracks: %d/%d' %
                     (len(self.trackers), len(tracks)))
        return tracks

    def cleanup_trackers(self) -> None:
        count_before = len(self.trackers)
        self.trackers = [
            t for t in self.trackers if not (t.is_stale() or t.is_invalid())
        ]
        count_after = len(self.trackers)
        logger.debug('deleted %s/%s trackers' %
                     (count_before - count_after, count_before))

    def step(self, detections: Sequence[Detection]) -> List[Track]:
        """ the method matches the new detections with existing trackers,
        creates new trackers if necessary and performs the cleanup.
        Returns the active tracks after active filtering applied """
        t0 = time.time()

        # filter out empty detections
        detections = [det for det in detections if det.box is not None]

        # predict state in all trackers
        for t in self.trackers:
            t.predict()

        # match trackers with detections
        logger.debug('step with %d detections' % len(detections))
        matches = self.matching_fn(self.trackers, detections)
        logger.debug('matched %d pairs' % len(matches))

        self.detections_matched_ids = [None] * len(detections)

        # assigned trackers: correct
        for match in matches:
            track_idx, det_idx = match[0], match[1]
            self.trackers[track_idx].update(detection=detections[det_idx])
            self.detections_matched_ids[det_idx] = self.trackers[track_idx].id

        # not assigned detections: create new trackers POF
        assigned_det_idxs = set(matches[:, 1]) if len(matches) > 0 else []
        for det_idx in set(range(
                len(detections))).difference(assigned_det_idxs):
            det = detections[det_idx]
            tracker = self.tracker_clss(box0=det.box,
                                        score0=det.score,
                                        class_id0=det.class_id,
                                        **self.tracker_kwargs)
            self.detections_matched_ids[det_idx] = tracker.id
            self.trackers.append(tracker)

        # unassigned trackers
        assigned_track_idxs = set(matches[:, 0]) if len(matches) > 0 else []
        for track_idx in set(range(len(
                self.trackers))).difference(assigned_track_idxs):
            self.trackers[track_idx].stale()

        # cleanup dead trackers
        self.cleanup_trackers()

        # log step timing
        elapsed = (time.time() - t0) * 1000.
        logger.debug(f'tracking step time: {elapsed:.3f} ms')

        return self.active_tracks(**self.active_tracks_kwargs)
