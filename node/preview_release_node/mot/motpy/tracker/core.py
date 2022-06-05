import collections
import logging
import os
import sys
from typing import Optional

import numpy as np

""" types """

# Box is of shape (1,2xdim), e.g. for dim=2 [xmin, ymin, xmax, ymax] format is accepted
Box = np.ndarray

# Vector is of shape (1, N)
Vector = np.ndarray

# Track is meant as an output from the object tracker
Track = collections.namedtuple('Track', 'id box score class_id')


# numpy/opencv image alias
NpImage = np.ndarray


class Detection:
    def __init__(
            self,
            box: Box,
            score: Optional[float] = None,
            class_id: Optional[int] = None,
            feature: Optional[Vector] = None):
        self.box = box
        self.score = score
        self.class_id = class_id
        self.feature = feature

    def __repr__(self):
        return f'Detection(box={self.box}, score={self.score:.5f}, class_id={self.class_id}, feature={self.feature})'


""" utils """

LOG_FORMAT = "%(asctime)s\t%(threadName)s-%(name)s:%(levelname)s:%(message)s"


def setup_logger(name: str,
                 level: Optional[str] = None,
                 is_main: bool = False,
                 envvar_name: str = 'MOTPY_LOG_LEVEL'):
    if level is None:
        level = os.getenv(envvar_name)
        if level is None:
            level = 'INFO'
        else:
            print(f'[{name}] envvar {envvar_name} sets log level to {level}')

    level_val = logging.getLevelName(level)

    logger = logging.getLogger(name)
    logger.setLevel(level_val)
    logger.addHandler(logging.NullHandler())

    if is_main:
        logging.basicConfig(stream=sys.stdout, level=level_val, format=LOG_FORMAT)

    return logger
