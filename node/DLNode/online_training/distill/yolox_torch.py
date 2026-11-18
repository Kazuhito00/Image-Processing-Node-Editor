#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Trainable YOLOX student built from its ONNX file, and the distillation loss.

Why train from the ONNX file
----------------------------
The deployed student is an ONNX file.  Training that very graph (converted to
PyTorch with ``onnx2torch``) and writing the trained weights back into the
same file guarantees that the model in production is exactly the one that was
trained and validated: same operators, same input, same output.

Two things make the exported graph unfit for training as is, and are fixed
here:

* the detection head ends with sigmoids.  Training on probabilities is
  numerically poor, so the head sigmoids are removed and the student is
  trained on logits (``strip_head_sigmoids``); they are still in the file that
  is deployed;
* the graph is exported for one image at a time; it is patched to accept
  batches (``_allow_batches``).

The batch normalisations of an exported model are folded into the
convolutions.  This is an advantage here: there are no batch statistics to
drift when training on a few correlated images of a live stream.
"""
import collections
import copy
import hashlib
import logging
import os
import sys
import types

import numpy as np

from node.DLNode.online_training.distill import yolox_ops

logger = logging.getLogger(__name__)



def _import_onnx2torch():
    """Import ``onnx2torch.convert``, with or without a working torchvision.

    onnx2torch imports torchvision for the NMS and RoiAlign operators, which a
    YOLOX graph does not contain.  The compiled operators of torchvision
    cannot always be loaded (notably in packaged builds); in that case a
    placeholder stands in for torchvision while onnx2torch is imported.
    """
    try:
        from onnx2torch import convert
        return convert
    except Exception:
        pass

    def purge(prefix):
        for name in [n for n in sys.modules if n == prefix or n.startswith(prefix + ".")]:
            del sys.modules[name]

    def unavailable(*args, **kwargs):
        raise NotImplementedError("torchvision operators are not available")

    purge("onnx2torch")
    purge("torchvision")
    placeholder = types.ModuleType("torchvision")
    placeholder.ops = types.ModuleType("torchvision.ops")
    for name in ("nms", "box_convert", "roi_align"):
        setattr(placeholder.ops, name, unavailable)
    sys.modules["torchvision"] = placeholder
    sys.modules["torchvision.ops"] = placeholder.ops
    try:
        from onnx2torch import convert
        return convert
    finally:
        # Leave the rest of the application free to import the real one
        purge("torchvision")


try:
    import onnx
    import torch
    import torch.nn.functional as F
    from onnx import numpy_helper

    _onnx_to_torch = _import_onnx2torch()
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on the installation
    torch = None
    _IMPORT_ERROR = exc


def backend_status():
    """``(available, reason)``: can students be trained on this machine?"""
    if _IMPORT_ERROR is not None:
        return False, (
            "training needs PyTorch and onnx2torch "
            f"(pip install torch onnx2torch): {_IMPORT_ERROR}"
        )
    return True, ""


def pick_device(requested="auto"):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, training on CPU")
        return "cpu"
    return requested


def strip_head_sigmoids(model):
    """Remove the sigmoids of the detection head, in place.

    YOLOX also uses sigmoids inside its SiLU activations (``x * sigmoid(x)``);
    the head ones are those whose result goes straight into a concatenation.
    Returns the number of sigmoids removed.
    """
    graph = model.graph
    consumers = collections.defaultdict(list)
    for node in graph.node:
        for name in node.input:
            consumers[name].append(node)
    outputs = {output.name for output in graph.output}

    removed = 0
    for node in list(graph.node):
        if node.op_type != "Sigmoid" or node.output[0] in outputs:
            continue
        users = consumers[node.output[0]]
        if not users or any(user.op_type != "Concat" for user in users):
            continue
        for user in users:
            for position, name in enumerate(user.input):
                if name == node.output[0]:
                    user.input[position] = node.input[0]
        graph.node.remove(node)
        removed += 1
    return removed


def _allow_batches(model):
    """Let the exported graph process several images at once, in place."""
    graph = model.graph
    initializers = {init.name: init for init in graph.initializer}
    for node in graph.node:
        if node.op_type != "Reshape" or node.input[1] not in initializers:
            continue
        init = initializers[node.input[1]]
        shape = numpy_helper.to_array(init).copy()
        if shape.size and shape[0] == 1:
            shape[0] = 0  # ONNX: keep the dimension of the input
            init.CopyFrom(numpy_helper.from_array(shape, init.name))


def _digest(array):
    array = np.ascontiguousarray(array)
    return (array.shape, array.dtype.str, hashlib.sha1(array.tobytes()).hexdigest())


class TrainableYolox:
    """YOLOX student whose weights can be trained and written back to ONNX."""

    def __init__(self, onnx_path, device="cpu"):
        available, reason = backend_status()
        if not available:
            raise RuntimeError(reason)
        self.device = torch.device(device)
        self._original = onnx.load(onnx_path)

        graph_input = self._original.graph.input[0]
        dims = [d.dim_value for d in graph_input.type.tensor_type.shape.dim]
        self.input_name = graph_input.name
        self.height, self.width = int(dims[2]), int(dims[3])
        yolox_ops.check_input_size(self.height, self.width)
        cells, strides = yolox_ops.make_grid(self.height, self.width)
        self.cells = torch.from_numpy(cells).to(self.device)
        self.strides = torch.from_numpy(strides).to(self.device)

        logits_model = copy.deepcopy(self._original)
        if strip_head_sigmoids(logits_model) < 2:
            raise RuntimeError(
                f"{os.path.basename(onnx_path)}: no YOLOX detection head found "
                "(expected sigmoids feeding the output concatenation)"
            )
        _allow_batches(logits_model)
        self.module = _onnx_to_torch(logits_model).to(self.device)
        self.module.eval()

        self._check_blob = self._make_check_blob()
        self._check_against_original()
        self.num_classes = int(self.logits(self._check_blob).shape[2]) - 5
        self._initializer_of = self._map_parameters()

    # ------------------------------------------------------------------ #
    # Construction checks

    def _make_check_blob(self):
        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, (2, self.height, self.width, 3), dtype=np.uint8)
        return yolox_ops.to_blob(list(image))

    def _run_onnx(self, model_bytes, blob):
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.log_severity_level = 3
        session = onnxruntime.InferenceSession(
            model_bytes, sess_options=options, providers=["CPUExecutionProvider"]
        )
        return np.concatenate(
            [session.run(None, {self.input_name: blob[i:i + 1]})[0] for i in range(len(blob))]
        )

    def _probabilities(self, blob):
        """Output of the trainable model in the format of the ONNX file."""
        with torch.no_grad():
            out = self.logits(blob)
            out = torch.cat([out[..., :4], torch.sigmoid(out[..., 4:])], dim=2)
        return out.cpu().numpy()

    def _check_against_original(self):
        """The trainable model must compute what the ONNX file computes."""
        expected = self._run_onnx(self._original.SerializeToString(), self._check_blob)
        actual = self._probabilities(self._check_blob)
        if expected.shape != actual.shape or not np.allclose(
            expected, actual, rtol=1e-3, atol=1e-3
        ):
            raise RuntimeError(
                "the trainable model does not reproduce the ONNX model "
                f"(shapes {actual.shape} vs {expected.shape}); "
                "is it a YOLOX model exported with decode_in_inference=False?"
            )

    def _map_parameters(self):
        """Find, for each trainable tensor, the ONNX initializer it came from."""
        by_digest = collections.defaultdict(list)
        for init in self._original.graph.initializer:
            by_digest[_digest(numpy_helper.to_array(init))].append(init.name)
        mapping = {}
        for name, parameter in self.module.named_parameters():
            names = by_digest.get(_digest(parameter.detach().cpu().numpy()))
            if names:
                mapping[name] = names.pop(0)
            else:
                # Cannot be written back to the ONNX file: leave it untouched
                parameter.requires_grad_(False)
                logger.warning("distillation: tensor %s is not trainable (no ONNX initializer)", name)
        if not mapping:
            raise RuntimeError("no trainable tensor could be mapped to the ONNX file")
        return mapping

    # ------------------------------------------------------------------ #
    # Forward

    def logits(self, blob):
        """``[N, anchors, 5 + classes]``: raw box, objectness and class logits."""
        if isinstance(blob, np.ndarray):
            blob = torch.from_numpy(blob)
        return self.module(blob.to(self.device))

    def predict(self, blob):
        """``(boxes[N, A, 4] in input pixels, objectness logits, class logits)``."""
        out = self.logits(blob)
        centre = (out[..., :2] + self.cells) * self.strides[:, None]
        size = torch.exp(out[..., 2:4].clamp(max=10.0)) * self.strides[:, None]
        boxes = torch.cat([centre - size / 2, centre + size / 2], dim=2)
        return boxes, out[..., 4], out[..., 5:]

    # ------------------------------------------------------------------ #
    # Weights

    def trainable_parameters(self, fraction=1.0):
        """Select the last ``fraction`` of the tensors for training.

        Tensors are ordered from the input to the output of the network:
        ``1.0`` trains everything, a smaller value freezes the first layers
        (cheaper: nothing is back-propagated through them).
        """
        named = [(n, p) for n, p in self.module.named_parameters() if n in self._initializer_of]
        first = int(round(len(named) * (1.0 - min(max(fraction, 0.0), 1.0))))
        first = min(first, len(named) - 1)
        for position, (_, parameter) in enumerate(named):
            parameter.requires_grad_(position >= first)
        return [parameter for _, parameter in named[first:]]

    def state(self):
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.module.named_parameters()
            if name in self._initializer_of
        }

    def load_state(self, state):
        with torch.no_grad():
            for name, parameter in self.module.named_parameters():
                if name in state:
                    parameter.copy_(state[name].to(parameter.device))

    def onnx_state(self, model_bytes):
        """Read the weights of an ONNX file produced by :meth:`export`."""
        initializers = {
            init.name: init for init in onnx.load_from_string(model_bytes).graph.initializer
        }
        state = {}
        for name, init_name in self._initializer_of.items():
            if init_name not in initializers:
                raise ValueError("ONNX file does not match the student model")
            state[name] = torch.from_numpy(numpy_helper.to_array(initializers[init_name]).copy())
        return state

    def export(self, state=None):
        """Serialized ONNX model with the given (default: current) weights.

        The graph is the original one, with its weights replaced: the result
        is a drop-in replacement of the original file.  It is run once and
        compared with the PyTorch model before being returned.
        """
        state = self.state() if state is None else state
        model = copy.deepcopy(self._original)
        initializers = {init.name: init for init in model.graph.initializer}
        for name, tensor in state.items():
            init = initializers[self._initializer_of[name]]
            dtype = numpy_helper.to_array(init).dtype
            init.CopyFrom(numpy_helper.from_array(tensor.numpy().astype(dtype), init.name))
        model_bytes = model.SerializeToString()

        previous = self.state()
        self.load_state(state)
        try:
            expected = self._probabilities(self._check_blob)
        finally:
            self.load_state(previous)
        actual = self._run_onnx(model_bytes, self._check_blob)
        if not np.isfinite(actual).all() or not np.allclose(expected, actual, rtol=1e-3, atol=1e-3):
            raise RuntimeError("exported ONNX model does not match the trained weights")
        return model_bytes


# ---------------------------------------------------------------------- #
# Distillation loss


def _giou(boxes_a, boxes_b):
    top_left = torch.max(boxes_a[:, :2], boxes_b[:, :2])
    bottom_right = torch.min(boxes_a[:, 2:], boxes_b[:, 2:])
    inter = (bottom_right - top_left).clamp(min=0).prod(dim=1)
    area_a = (boxes_a[:, 2:] - boxes_a[:, :2]).clamp(min=0).prod(dim=1)
    area_b = (boxes_b[:, 2:] - boxes_b[:, :2]).clamp(min=0).prod(dim=1)
    union = area_a + area_b - inter + 1e-7
    hull = (
        torch.max(boxes_a[:, 2:], boxes_b[:, 2:]) - torch.min(boxes_a[:, :2], boxes_b[:, :2])
    ).clamp(min=0).prod(dim=1) + 1e-7
    return inter / union - (hull - union) / hull


def _soft_focal_bce(logits, targets, gamma):
    """Cross-entropy with soft targets, focused on what is still wrong.

    Quality focal loss (Li et al., 2020): the cross-entropy of each prediction
    is scaled by ``|target - prediction| ** gamma``.  The thousands of
    background anchors the student already gets right stop dominating the
    few anchors that matter.  ``gamma = 0`` is the plain cross-entropy.
    """
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if gamma:
        loss = loss * (targets - torch.sigmoid(logits)).abs().pow(gamma)
    return loss


def distillation_loss(boxes, obj_logits, cls_logits, target, gamma=2.0, box_weight=5.0):
    """Dense distillation loss of a batch.

    The student imitates the teacher on every anchor:

    * objectness: soft cross-entropy with the teacher objectness, on all
      anchors.  This is where the student learns what is an object *and* what
      is background, including on images where the teacher sees nothing;
    * classes: soft cross-entropy with the teacher class probabilities,
      weighted by the teacher objectness (Mehta & Ozturk, 2018): the class
      output of the teacher means nothing where there is no object;
    * boxes: ``1 - GIoU`` with the teacher box of the same anchor, weighted by
      the teacher score.

    Everything is divided by the (soft) number of objects anchors of the
    batch, as in the YOLOX training loss.

    ``target`` holds ``obj[N, A]`` and, for the anchors where the teacher
    objectness is significant, ``pos_image[K]``, ``pos_anchor[K]``,
    ``cls[K, C]`` and ``box[K, 4]``.

    Returns ``(loss, parts)`` with ``parts`` the three terms as floats.
    """
    teacher_obj = target["obj"]
    normaliser = teacher_obj.sum().clamp(min=1.0)

    loss_obj = _soft_focal_bce(obj_logits, teacher_obj, gamma).sum() / normaliser

    image, anchor = target["pos_image"], target["pos_anchor"]
    if len(image):
        weight = teacher_obj[image, anchor]
        loss_cls = (
            _soft_focal_bce(cls_logits[image, anchor], target["cls"], gamma).sum(dim=1) * weight
        ).sum() / normaliser
        score = weight * target["cls"].max(dim=1).values
        loss_box = (
            (1.0 - _giou(boxes[image, anchor], target["box"])) * score
        ).sum() / normaliser
    else:
        loss_cls = loss_box = obj_logits.sum() * 0.0

    loss = loss_obj + loss_cls + box_weight * loss_box
    return loss, {
        "obj": float(loss_obj.detach()),
        "cls": float(loss_cls.detach()),
        "box": float(loss_box.detach()),
    }
