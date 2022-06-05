import numpy as np

from motpy import track_to_string
from motpy.core import Detection, Track, setup_logger
from motpy.testing import CANVAS_SIZE, data_generator

logger = setup_logger(__name__)

try:
    import cv2
except BaseException:
    logger.error(
        'Could not import opencv. Please install opencv-python package or some of the testing functionalities will not be available'
    )
""" methods below require opencv-python package installed """


def draw_rectangle(img, box, color, thickness: int = 3) -> None:
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                  color, thickness)


def draw_text(img, text, pos, color=(255, 255, 255)) -> None:
    tl_pt = (int(pos[0]), int(pos[1]) - 7)
    cv2.putText(img,
                text,
                tl_pt,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=color)


def draw_track(img,
               track: Track,
               random_color: bool = True,
               fallback_color=(200, 20, 20),
               thickness: int = 5,
               text_at_bottom: bool = False,
               text_verbose: int = 1):
    color = [ord(c) * ord(c) % 256
             for c in track.id[:3]] if random_color else fallback_color
    draw_rectangle(img, track.box, color=color, thickness=thickness)
    pos = (track.box[0], track.box[3]) if text_at_bottom else (track.box[0],
                                                               track.box[1])

    if text_verbose > 0:
        text = track_to_string(track) if text_verbose == 2 else track.id[:8]
        draw_text(img, text, pos=pos)

    return img


def draw_detection(img, detection: Detection) -> None:
    draw_rectangle(img, detection.box, color=(0, 220, 0), thickness=1)


def image_generator(*args, **kwargs):
    def _empty_canvas(canvas_size=(CANVAS_SIZE, CANVAS_SIZE, 3)):
        img = np.ones(canvas_size, dtype=np.uint8) * 30
        return img

    data_gen = data_generator(*args, **kwargs)
    for dets_gt, dets_pred in data_gen:
        img = _empty_canvas()

        # overlay actor shapes
        for det_gt in dets_gt:
            xmin, ymin, xmax, ymax = det_gt.box
            feature = det_gt.feature
            for channel in range(3):
                img[int(ymin):int(ymax),
                    int(xmin):int(xmax), channel] = feature[channel]

        yield img, dets_gt, dets_pred


if __name__ == "__main__":
    for img, dets_gt, dets_pred in image_generator(num_steps=1000,
                                                   num_objects=10):

        for det_gt, det_pred in zip(dets_gt, dets_pred):
            draw_rectangle(img, det_gt.box, color=det_gt.feature)

            if det_pred.box is not None:
                draw_rectangle(img,
                               det_pred.box,
                               color=det_pred.feature,
                               thickness=1)

        cv2.imshow('preview', img)
        c = cv2.waitKey(33)
        if c == ord('q'):
            break
