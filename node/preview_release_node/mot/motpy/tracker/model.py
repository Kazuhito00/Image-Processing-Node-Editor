from enum import Enum

import numpy as np
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag

# from motpy.core import Box, Vector
from node.preview_release_node.mot.motpy.tracker.core import Box, Vector
""" The list of model presets below is not complete, more reasonable
options will be added in the future """


class ModelPreset(Enum):
    constant_velocity_and_static_box_size_2d = {
        'order_pos': 1,
        'dim_pos': 2,
        'order_size': 0,
        'dim_size': 2
    }

    constant_acceleration_and_static_box_size_2d = {
        'order_pos': 2,
        'dim_pos': 2,
        'order_size': 0,
        'dim_size': 2
    }


def _base_dim_block(dt: float, order: int = 1):
    block = np.array([[1, dt, (dt**2) / 2], [0, 1, dt], [0, 0, 1]])
    cutoff = order + 1
    return block[:cutoff, :cutoff]


def _zero_pad(arr, length: int):
    ret = np.zeros((length, ))
    ret[:arr.shape[0]] = arr
    return ret


class Model:
    def __init__(self,
                 dt: float,
                 order_pos: int = 1,
                 dim_pos: int = 2,
                 order_size: int = 0,
                 dim_size: int = 2,
                 q_var_pos: float = 70.0,
                 q_var_size: float = 10.0,
                 r_var_pos: float = 1,
                 r_var_size: float = 1,
                 p_cov_p0: float = 1000.):

        self.dt = dt
        self.order_pos = order_pos
        self.dim_pos = dim_pos
        self.order_size = order_size
        self.dim_size = dim_size

        self.q_var_pos = q_var_pos
        self.q_var_size = q_var_size
        self.r_var_pos = r_var_pos
        self.r_var_size = r_var_size
        self.p_cov_p0 = p_cov_p0

        if self.order_pos > 2 or self.order_size > 2:
            raise ValueError('Currently only system orders <= 2 are supported')

        # the expected input/output box length
        self.dim_box = 2 * max(self.dim_pos, self.dim_size)

        # precalculate utility indexes
        self.pos_idxs, self.size_idxs, self.z_in_x_idxs, self.offset_idx = self._calc_idxs(
        )

        # number of variables in model state
        self.state_length = self.dim_pos * (self.order_pos + 1) + \
            self.dim_size * (self.order_size + 1)

        # length of z (observation) vector
        self.measurement_length = self.dim_pos + self.dim_size

    def _calc_idxs(self):
        offset_idx = max(self.dim_pos, self.dim_size)

        pos_idxs = [
            pidx * (self.order_pos + 1) for pidx in range(self.dim_pos)
        ]

        size_idxs = [
            self.dim_pos * (self.order_pos + 1) + sidx * (self.order_size + 1)
            for sidx in range(self.dim_size)
        ]

        # indexes of measured quantities (z) in state (x) vector
        z_in_x_idxs = np.concatenate((pos_idxs, size_idxs))

        return np.array(pos_idxs), np.array(size_idxs), z_in_x_idxs, offset_idx

    def build_F(self):
        """ returns constructed F matrix with specified positional
            e.g. (x,y,z) and size e.g. (w,h) dimensions """
        block_pos = _base_dim_block(self.dt, self.order_pos)
        block_size = _base_dim_block(self.dt, self.order_size)
        diag_components = [block_pos] * self.dim_pos + [block_size
                                                        ] * self.dim_size
        return block_diag(*diag_components)

    def build_Q(self):
        """ process noise """
        var_pos = self.q_var_pos
        var_size = self.q_var_size

        q_pos = var_pos if self.order_pos == 0 else Q_discrete_white_noise(
            dim=self.order_pos + 1, dt=self.dt, var=var_pos)

        q_size = var_size if self.order_size == 0 else Q_discrete_white_noise(
            dim=self.order_size + 1, dt=self.dt, var=var_size)

        diag_components = [q_pos] * self.dim_pos + [q_size] * self.dim_size
        return block_diag(*diag_components)

    def build_R(self):
        """ measurement noise, expected order is positon first, then size """
        block_pos = np.eye(self.dim_pos) * self.r_var_pos
        block_size = np.eye(self.dim_size) * self.r_var_size
        return block_diag(block_pos, block_size)

    def build_H(self):
        """ measurement matrix """

        # we only measure the first variable in each dimension
        def _base_block(order):
            return np.array([1] + [0] * order)
        diag_components = \
            [_base_block(self.order_pos)] * self.dim_pos +\
            [_base_block(self.order_size)] * self.dim_size
        return block_diag(*diag_components)

    def build_P(self):
        return np.eye(self.state_length) * self.p_cov_p0

    def box_to_z(self, box: Box) -> Vector:
        assert self.dim_box == len(box)
        box = np.array(box).reshape(2, (int(self.dim_box / 2)))
        center = (np.sum(box, axis=0) / 2.0)[:self.dim_pos]
        length = (box[1, :] - box[0, :])[:self.dim_size]
        return np.concatenate((center, length))

    def box_to_x(self, box: Box) -> Vector:
        """ box is expected to be in [xmin, ymin, zmin, ..., xmax, ymax, zmax, ...] format
        for 2d-1ord+2d-0ord case returns np.array([cx, 0, 0, cy, 0, 0, w, h]) """
        x = np.zeros((self.state_length, ))
        x[self.z_in_x_idxs] = self.box_to_z(box)
        return x

    def x_to_box(self, x):
        size = max(self.dim_pos, self.dim_size)
        center = _zero_pad(x[self.pos_idxs], size)
        length = _zero_pad(x[self.size_idxs], size)
        return np.concatenate((center - length / 2, center + length / 2))
