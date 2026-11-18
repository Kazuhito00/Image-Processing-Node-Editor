#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Distil a YOLOX teacher into a YOLOX student on a recorded stream.

    python -m node.DLNode.online_training.distill.cli \\
        --teacher yolox_s.onnx --student yolox_nano.onnx \\
        --source camera.mp4 --workdir _Distillation/run1

The video (or image directory) is replayed as if it were live: the first part
feeds the engine, the last part (``--holdout``) is never shown to it and is
used to measure the student before and after, against the detections of the
teacher at its full resolution.

Use it to train on a machine with a GPU from recorded footage, or to check on
your own data that distillation helps before enabling it in the application.
The distilled student is written to ``<workdir>/deployed.onnx``.
"""
import argparse
import glob
import logging
import os
import sys

import cv2

from node.DLNode.online_training.distill import yolox_ops
from node.DLNode.online_training.distill.engine import DistillationEngine, DistillConfig
from node.DLNode.online_training.distill.yolox_onnx import YoloxOnnx


def read_frames(source, every):
    if os.path.isdir(source):
        paths = sorted(
            p for p in glob.glob(os.path.join(source, "*"))
            if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        )
        for path in paths[::every]:
            image = cv2.imread(path)
            if image is not None:
                yield image
        return
    capture = cv2.VideoCapture(source)
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % every == 0:
            yield frame
        index += 1
    capture.release()


def measure(detect, frames, references):
    meter = yolox_ops.DetectionMeter()
    for frame, (ref_boxes, _, ref_classes) in zip(frames, references):
        boxes, scores, classes = detect(frame)
        meter.add(ref_boxes, ref_classes, boxes, scores, classes)
    return meter.result()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--source", required=True, help="video file or image directory")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--every", type=int, default=3, help="use one frame out of N")
    parser.add_argument("--holdout", type=float, default=0.3, help="final share kept for testing")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lr", type=float, default=DistillConfig.learning_rate)
    parser.add_argument("--steps", type=int, default=0, help="steps per round (0: automatic)")
    parser.add_argument("--batch", type=int, default=0, help="batch size (0: automatic)")
    parser.add_argument("--teacher-view", choices=("student", "full"), default="student",
                        help="what the teacher looks at (see DistillConfig.teacher_view)")
    parser.add_argument("--trainable", type=float, default=DistillConfig.trainable_fraction,
                        help="share of the layers trained, from the output")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    frames = list(read_frames(args.source, max(1, args.every)))
    split = int(len(frames) * (1.0 - args.holdout))
    stream, held_out = frames[:split], frames[split:]
    if not stream or not held_out:
        sys.exit("not enough frames in the source")

    config = DistillConfig(
        device=args.device, learning_rate=args.lr, steps_per_round=args.steps,
        batch_size=args.batch, trainable_fraction=args.trainable,
        teacher_view=args.teacher_view,
        min_new_samples=0,
    )
    engine = DistillationEngine(args.teacher, args.student, args.workdir, config)

    teacher = YoloxOnnx(args.teacher)
    references = [teacher.detect(f, config.score_th, config.nms_th) for f in held_out]
    before = measure(YoloxOnnx(args.student).detect, held_out, references)

    for frame in stream:
        engine.collect(frame)
    status = engine.status()
    print(f"stream: {len(stream)} frames -> {status['train_samples']} training samples, "
          f"{status['val_samples']} most recent ones for validation; "
          f"held out: {len(held_out)} frames")
    if not engine.ready_to_train():
        sys.exit("not enough samples collected to train")

    print("round  steps  train-loss  val-loss cand/deployed   val-AP50 cand/deployed  decision")
    for _ in range(args.rounds):
        record = engine.train_round()
        decision = (
            f"deployed v{record['version']}" if record["deployed_now"]
            else "rolled back" if record["rolled_back"] else "kept"
        )
        print(f"{record['round']:5d} {record['steps']:6d} {record['train_loss']:11.4f} "
              f"{record['candidate']['loss']:10.4f} / {record['deployed']['loss']:.4f} "
              f"{record['candidate']['ap50']:11.3f} / {record['deployed']['ap50']:.3f}"
              f"     {decision}  ({record['seconds']}s)")

    after = measure(engine.detect, held_out, references)
    print("\nheld-out frames (never used), reference = teacher at full resolution")
    for name, result in (("original student", before), ("distilled student", after)):
        print(f"  {name:18s} AP50={result['ap50']:.3f} F1={result['f1']:.3f} "
              f"precision={result['precision']:.3f} recall={result['recall']:.3f}")
    print(f"deployed student: {engine.deployed_path if engine.version else '(unchanged)'}")
    return before, after


if __name__ == "__main__":
    main()
