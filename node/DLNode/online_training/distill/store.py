#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""On-disk store of distillation samples.

A sample is one image seen by the student together with what the teacher
predicted on it, anchor by anchor.  Samples live on disk, one compressed file
each, so that they survive a restart and can be trained on elsewhere.

Samples are kept in the order they were collected and split in two:

* the most recent ones form the **validation window**.  They have never been
  used for training;
* all the older ones are the **training set**.

A student trained on the past is therefore always judged on what came next,
which is the question that matters in production: *will it be better on the
frames to come?*  Frames of a video look alike from one second to the next; a
validation set drawn among the training frames would answer yes far too
easily.  As new samples arrive, the oldest validation samples move to the
training set.
"""
import json
import os
import random
import re

import cv2
import numpy as np

# Teacher class scores and boxes are only kept where its objectness is
# significant; elsewhere they carry no information (and no weight in the loss).
POSITIVE_OBJECTNESS = 0.05
MAX_POSITIVES = 400

_NAME = re.compile(r"^(\d+)_p(\d{3})\.npz$")


def pack_sample(canvas_jpeg, boxes, objectness, class_scores, ratio):
    """Arrays stored for one sample (see :func:`unpack_sample`)."""
    positives = np.flatnonzero(objectness >= POSITIVE_OBJECTNESS)
    if len(positives) > MAX_POSITIVES:
        positives = positives[np.argsort(-objectness[positives])[:MAX_POSITIVES]]
        positives.sort()
    return {
        "jpeg": np.frombuffer(canvas_jpeg, dtype=np.uint8),
        "obj": objectness.astype(np.float16),
        "pos": positives.astype(np.int32),
        "cls": class_scores[positives].astype(np.float16),
        "box": boxes[positives].astype(np.float32),
        "ratio": np.float32(ratio),
    }


def unpack_sample(arrays):
    """``canvas`` (BGR uint8), ``obj[A]``, ``pos[K]``, ``cls[K, C]``, ``box[K, 4]``."""
    return {
        "canvas": cv2.imdecode(arrays["jpeg"], cv2.IMREAD_COLOR),
        "obj": arrays["obj"].astype(np.float32),
        "pos": arrays["pos"].astype(np.int64),
        "cls": arrays["cls"].astype(np.float32),
        "box": arrays["box"].astype(np.float32),
        "ratio": float(arrays["ratio"]),
    }


class SampleStore:
    """Bounded, chronological sample set in a directory.

    Parameters
    ----------
    root : str
        Directory of the store (created if needed).
    signature : dict
        What the samples depend on (models, input size).  A store created
        with another signature is set aside and a new one is started.
    max_train : int
        Capacity of the training set (the store holds ``max_train +
        val_window`` samples).  When it is full, a random training sample is
        replaced: old samples disappear progressively, so the set follows the
        scene as it changes while keeping some memory of the past.
    val_window : int
        Maximum size of the validation window; it is never more than a
        quarter of the samples.
    """

    def __init__(self, root, signature, max_train=4000, val_window=96, seed=0):
        self.root = root
        self.max_train = int(max_train)
        self.val_window = int(val_window)
        self._rng = random.Random(seed)
        self._prepare(signature)
        self._files = sorted(f for f in os.listdir(self.root) if _NAME.match(f))
        self._next_id = max((int(_NAME.match(f).group(1)) for f in self._files), default=-1) + 1
        self.added = 0

    def _prepare(self, signature):
        meta_path = os.path.join(self.root, "store.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path) as fp:
                    same = json.load(fp) == signature
            except (OSError, ValueError):
                same = False
            if not same:
                os.rename(self.root, f"{self.root}.old-{int(os.path.getmtime(meta_path))}")
        os.makedirs(self.root, exist_ok=True)
        with open(meta_path, "w") as fp:
            json.dump(signature, fp, indent=1)

    # ------------------------------------------------------------------ #

    def _val_count(self):
        return min(self.val_window, len(self._files) // 4)

    def train_names(self):
        return self._files[: len(self._files) - self._val_count()]

    def val_names(self):
        """The most recent samples, never trained on so far."""
        return self._files[len(self._files) - self._val_count():]

    def add(self, arrays, priority=0.0):
        """Store a sample as the most recent one."""
        name = f"{self._next_id:09d}_p{int(round(min(max(priority, 0.0), 0.999) * 1000)):03d}.npz"
        self._next_id += 1
        path = os.path.join(self.root, name)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fp:
            np.savez_compressed(fp, **arrays)
        os.replace(tmp, path)
        self._files.append(name)
        self.added += 1

        if len(self._files) > self.max_train + self.val_window:
            train = len(self._files) - self._val_count()
            victim = self._files.pop(self._rng.randrange(train))
            try:
                os.remove(os.path.join(self.root, victim))
            except OSError:
                pass

    def load(self, name):
        with np.load(os.path.join(self.root, name)) as arrays:
            return unpack_sample(arrays)

    def sample_train(self, count, hard_fraction=0.5):
        """Names of ``count`` training samples.

        ``hard_fraction`` of them are drawn in proportion to their priority
        (the teacher/student disagreement when they were collected), the rest
        uniformly: hard images are seen more often, easy ones are not
        forgotten.
        """
        files = self.train_names()
        if not files:
            return []
        weights = [int(_NAME.match(f).group(2)) + 1 for f in files]
        chosen = []
        for _ in range(count):
            if self._rng.random() < hard_fraction:
                chosen.append(self._rng.choices(files, weights=weights)[0])
            else:
                chosen.append(self._rng.choice(files))
        return chosen
