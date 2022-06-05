import math
import random

from motpy.core import Detection

CANVAS_SIZE = 1000


def _random_color():
    color_rgb = [random.randint(0, 255) for _ in range(3)]
    return color_rgb


class Actor():
    """ Actor is a box moving in 2d space """
    def __init__(self,
                 color=None,
                 max_omega: float = 0.05,
                 miss_prob: float = 0.1,
                 disappear_prob: float = 0.01,
                 det_err_sigma: float = 1.0,
                 canvas_size: int = 400):

        self.max_omega = max_omega
        self.miss_prob = miss_prob
        self.disappear_prob = disappear_prob
        self.det_err_sigma = det_err_sigma
        self.canvas_size = canvas_size
        self.class_id = random.randint(1, 10)

        # randomize size
        self.width = random.randint(50, 120)
        self.height = random.randint(50, 120)

        # randomize motion
        self.omega_x = random.uniform(-self.max_omega, self.max_omega)
        self.omega_y = random.uniform(-self.max_omega, self.max_omega)
        self.fi_x = random.randint(-180, 180)
        self.fi_y = random.randint(-90, 90)

        # let's treat color as a kind of feature
        if color is None:
            self.color = _random_color()

        self.disappear_steps = 0

    def position_at(self, step: int):
        half = self.canvas_size / 2 - 50
        x = half * math.cos(self.omega_x * step + self.fi_x) + half
        y = half * math.cos(self.omega_y * step + self.fi_y) + half
        return (x, y)

    def detections(self, step: int):
        """ returns ground truth and potentially missing detection for a given actor """
        xmin, ymin = self.position_at(step)
        box_gt = [xmin, ymin, xmin + self.width, ymin + self.height]

        # detection has some noise around the face coordinates
        box_pred = [random.gauss(0, self.det_err_sigma) + v for v in box_gt]

        # due to flakyness, some detections are missing
        if random.random() < self.miss_prob:
            box_pred = None

        # disappear_steps
        if random.random() < self.disappear_prob:
            self.disappear_steps = random.randint(1, 24)

        if self.disappear_steps > 0:
            box_pred = None
            self.disappear_steps -= 1

        # wrap boxes and features as detections
        det_gt = Detection(box=box_gt,
                           score=1.,
                           class_id=self.class_id,
                           feature=self.color)

        feature_pred = [random.gauss(0, 5) + v for v in self.color]
        det_pred = Detection(box=box_pred,
                             score=random.uniform(0.5, 1.),
                             class_id=max(
                                 0, self.class_id + random.randint(-1, 1)),
                             feature=feature_pred)

        return det_gt, det_pred


def data_generator(num_steps: int = 1000,
                   num_objects: int = 1,
                   max_omega: float = 0.01,
                   miss_prob: float = 0.1,
                   disappear_prob: float = 0.0,
                   det_err_sigma: float = 1.0):

    actors = [
        Actor(max_omega=max_omega,
              miss_prob=miss_prob,
              disappear_prob=disappear_prob,
              det_err_sigma=det_err_sigma,
              canvas_size=CANVAS_SIZE) for _ in range(num_objects)
    ]

    for step in range(num_steps):
        dets_gt, dets_pred = [], []

        for actor in actors:
            det_gt, det_pred = actor.detections(step)
            dets_gt.append(det_gt)
            dets_pred.append(det_pred)

        yield dets_gt, dets_pred
