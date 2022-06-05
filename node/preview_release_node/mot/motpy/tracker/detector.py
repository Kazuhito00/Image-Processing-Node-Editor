from typing import Sequence

from motpy.core import NpImage
from motpy.tracker import Detection


class BaseObjectDetector:
    def __init__(self) -> None:
        pass

    def process_image(self, image: NpImage) -> Sequence[Detection]:
        raise NotImplementedError('subclass the BaseObjectDetector with your custom detector')
