#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Catalogue of the small YOLOX students offered for distillation.

A student must be a YOLOX model exported without box decoding, with the same
classes as the teacher and an input no larger than the teacher's.  Two
pretrained architectures small enough to be trained in the background on a
CPU ship with the application: ``yolox_nano`` (0.9 M parameters) and
``yolox_tiny`` (5.1 M).  Each one is offered at its native 416 x 416 input and
at 320 x 320, which is about 40 % cheaper per image and well within the
multi-scale range (320 to 640) the official weights were trained on.

The YOLOX graph is fully convolutional and its exported ONNX file only fixes
the resolution in the input and output declarations: a 320 x 320 variant is
derived from the 416 x 416 file by :func:`resize_yolox_onnx`, checked with
onnxruntime and cached in ``<cache_dir>/<key>.onnx``.  The engine then treats
it like any other student file.
"""
import dataclasses
import glob
import os
import sys

import numpy as np
import onnx

from node.DLNode.online_training.distill import yolox_ops

_HERE = os.path.dirname(os.path.abspath(__file__))


@dataclasses.dataclass(frozen=True)
class StudentSpec:
    """A student of the catalogue."""

    key: str          # identifier, stable across versions (saved in the node settings)
    base: str         # stem of the ONNX file the student is built from
    size: int         # square input size in pixels
    params: str       # order of magnitude, for the interface and the docs
    description: str

    @property
    def native(self):
        """Is this the file as shipped (no resizing needed)?"""
        return self.size == _NATIVE_SIZE[self.base]

    @property
    def label(self):
        return f"{self.key} - {self.size}px, {self.params}"


_NATIVE_SIZE = {"yolox_nano": 416, "yolox_tiny": 416}

STUDENTS = (
    StudentSpec(
        "yolox_nano_320", "yolox_nano", 320, "0.9M",
        "fastest: weak CPUs, large or close objects",
    ),
    StudentSpec(
        "yolox_nano", "yolox_nano", 416, "0.9M",
        "default: the smallest YOLOX at its native resolution",
    ),
    StudentSpec(
        "yolox_tiny_320", "yolox_tiny", 320, "5.1M",
        "more capacity than nano, same cost per image as nano 416",
    ),
    StudentSpec(
        "yolox_tiny", "yolox_tiny", 416, "5.1M",
        "most accurate small student; needs a GPU or a strong CPU to train",
    ),
)
DEFAULT_STUDENT = "yolox_nano"

_BY_KEY = {spec.key: spec for spec in STUDENTS}


# ---------------------------------------------------------------------- #
# Model files


def model_dirs():
    """Directories searched for ``yolox*.onnx`` files."""
    dirs = [os.path.join(_HERE, "..", "..", "object_detection", "YOLOX", "model")]
    if getattr(sys, "frozen", False):
        from src.utils.paths import get_models_dir

        dirs.append(get_models_dir("online_training"))
    else:
        dirs.append(os.path.join(_HERE, "..", "models"))
    return [os.path.normpath(d) for d in dirs]


def find_yolox_models():
    """``{name: path}`` of the YOLOX ONNX files available, smallest first.

    Any ``yolox*.onnx`` file dropped in the model directories is listed, which
    is how a larger teacher (yolox_m, yolox_l, yolox_x) is added.
    """
    paths = []
    for directory in model_dirs():
        paths.extend(glob.glob(os.path.join(directory, "yolox*.onnx")))
    paths.sort(key=os.path.getsize)
    models = {}
    for path in paths:
        models.setdefault(os.path.splitext(os.path.basename(path))[0], path)
    return models


def student_catalogue(models=None):
    """``{key: StudentSpec}`` of the students whose base file is available.

    Catalogue order (cheapest first).  ``models`` is the result of
    :func:`find_yolox_models` (looked up when omitted).
    """
    models = find_yolox_models() if models is None else models
    return {spec.key: spec for spec in STUDENTS if spec.base in models}


def student_choices(models=None):
    """Keys offered in the interface: the catalogue, then any other YOLOX file.

    Extra ``yolox*.onnx`` files are kept selectable so that a custom student
    can be dropped in the model directory, as for teachers.
    """
    models = find_yolox_models() if models is None else models
    choices = list(student_catalogue(models))
    bases = {spec.base for spec in STUDENTS}
    choices.extend(name for name in models if name not in bases and name not in choices)
    return choices


def describe_student(key):
    """One line about a student key, for tooltips and the documentation."""
    spec = _BY_KEY.get(key)
    if spec is None:
        return key
    return f"{spec.label}: {spec.description}"


# ---------------------------------------------------------------------- #
# Resized variants


def resize_yolox_onnx(model, height, width):
    """Serialized copy of a YOLOX ONNX model with a ``height`` x ``width`` input.

    Only the declared input and output shapes are changed: the official YOLOX
    export reshapes its head with ``[1, 5 + classes, -1]`` and upsamples with
    scale factors, so the graph itself works at any multiple of 32.  The
    result is run once with onnxruntime to make sure of it.

    ``model`` is a path or the serialized model.
    """
    yolox_ops.check_input_size(height, width)
    proto = onnx.load(model) if isinstance(model, str) else onnx.load_from_string(model)
    graph = proto.graph
    if len(graph.input) != 1 or len(graph.output) != 1:
        raise ValueError("a YOLOX model with one input and one output is expected")

    dims = graph.input[0].type.tensor_type.shape.dim
    if len(dims) != 4:
        raise ValueError(f"4-D image input expected, got {len(dims)} dimensions")
    dims[2].dim_value, dims[3].dim_value = int(height), int(width)

    anchors = len(yolox_ops.make_grid(height, width)[1])
    out_dims = graph.output[0].type.tensor_type.shape.dim
    if len(out_dims) != 3:
        raise ValueError(f"raw YOLOX output [1, anchors, 5 + classes] expected, got {out_dims}")
    out_dims[1].dim_value = anchors
    # Intermediate shapes were inferred for the original size: drop them
    del graph.value_info[:]

    data = proto.SerializeToString()
    _check_runs(data, graph.input[0].name, height, width, anchors, int(out_dims[2].dim_value))
    return data


def _check_runs(data, input_name, height, width, anchors, channels):
    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.log_severity_level = 3
    session = onnxruntime.InferenceSession(
        data, sess_options=options, providers=["CPUExecutionProvider"]
    )
    blob = np.zeros((1, 3, height, width), dtype=np.float32)
    out = session.run(None, {input_name: blob})[0]
    if out.shape != (1, anchors, channels) or not np.isfinite(out).all():
        raise ValueError(
            f"the model does not work at {width}x{height}: output {out.shape}, "
            f"expected {(1, anchors, channels)}"
        )


def resolve_student(key, cache_dir, models=None):
    """Path of the ONNX file of student ``key``, building it if needed.

    ``key`` is a catalogue key or the stem of a ``yolox*.onnx`` file.  Resized
    variants are written to ``cache_dir`` once, and rebuilt when the base file
    changes.
    """
    models = find_yolox_models() if models is None else models
    spec = _BY_KEY.get(key)
    if spec is None:
        if key in models:
            return models[key]
        raise ValueError(f"unknown student {key!r}; available: {', '.join(student_choices(models))}")
    if spec.base not in models:
        raise ValueError(f"student {key!r} needs {spec.base}.onnx in {model_dirs()}")
    base_path = models[spec.base]
    if spec.native:
        return base_path

    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{spec.key}.onnx")
    if os.path.isfile(path) and os.path.getmtime(path) >= os.path.getmtime(base_path):
        return path
    data = resize_yolox_onnx(base_path, spec.size, spec.size)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fp:
        fp.write(data)
    os.replace(tmp, path)
    return path
