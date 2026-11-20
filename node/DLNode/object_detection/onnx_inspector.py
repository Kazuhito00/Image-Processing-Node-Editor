#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ONNX model inspector for CvStudio.

Extracts metadata from an ONNX file:
  - input/output names and shapes
  - auto-detection of output format (YOLO11-style vs YOLOX-style)
  - class names embedded in Ultralytics ONNX metadata
"""

import ast
import json
import logging
import os

import onnxruntime
from node.DLNode.object_detection.onnx_session_utils import make_session

logger = logging.getLogger(__name__)

# Class counts of the usual detection datasets, used to tell a YOLOX output
# ``[anchors, classes + 5]`` from a NanoDet one ``[anchors, classes + 4 * bins]``
_STANDARD_CLASS_COUNTS = (80, 1, 2, 3, 4, 5, 10, 20, 21, 91)


def has_dynamic_batch_dim(input_shape: list) -> bool:
    """True when the first input dimension is dynamic/symbolic."""
    if not input_shape:
        return False
    batch_dim = input_shape[0]
    return (
        batch_dim is None
        or isinstance(batch_dim, str)
        or (isinstance(batch_dim, int) and batch_dim < 0)
    )


def supports_batched_detection(input_shape: list, output_format: str) -> bool:
    """True when ObjectDetection can run a real batched ONNX call for this model."""
    if not has_dynamic_batch_dim(input_shape):
        return False
    return output_format in {"yolo11", "yolo11_obb", "yolox", "nanodet"}


def _dim_value(dim) -> object:
    """Extract an int value from an ONNX TensorShapeProto.Dimension, or a string param name."""
    if dim.HasField("dim_value"):
        return dim.dim_value
    if dim.HasField("dim_param"):
        return dim.dim_param
    return None


def _shape_from_type_proto(type_proto) -> list:
    """Return the shape list from an ONNX TypeProto (tensor_type)."""
    tensor_type = type_proto.tensor_type
    if not tensor_type.HasField("shape"):
        return []
    return [_dim_value(d) for d in tensor_type.shape.dim]


def _detect_classify_format(output_shape: list) -> tuple:
    """Return ``(output_format, num_classes)`` for classification model outputs.

    Classification models produce rank-1 ``[num_classes]`` or rank-2
    ``[batch, num_classes]`` output tensors.  Returns ``("classify", N)``
    when the shape matches, or ``("unknown", 0)`` otherwise.
    """
    if len(output_shape) == 2:
        last_dim = output_shape[-1]
        if isinstance(last_dim, int) and last_dim > 1:
            return "classify", last_dim
    if len(output_shape) == 1:
        last_dim = output_shape[0]
        if isinstance(last_dim, int) and last_dim > 1:
            return "classify", last_dim
    return "unknown", 0


def _inspect_onnx_static(model_path: str) -> dict:
    """Inspect ONNX model metadata using the ``onnx`` package only (no runtime session).

    This fallback is used when onnxruntime cannot load the model (e.g. quantized
    models with unsupported ops like ConvInteger).
    """
    try:
        import onnx  # noqa: PLC0415
    except ImportError:
        raise RuntimeError(
            "The 'onnx' package is required to inspect this model. "
            "Install it: pip install onnx"
        )

    model_proto = onnx.load(model_path)
    graph = model_proto.graph

    # ---- Input info --------------------------------------------------------
    # Skip initializer-only inputs (weights); find the real data input.
    initializer_names = {init.name for init in graph.initializer}
    data_inputs = [inp for inp in graph.input if inp.name not in initializer_names]
    if not data_inputs:
        data_inputs = list(graph.input)

    input_info = data_inputs[0]
    input_name = input_info.name
    input_shape = _shape_from_type_proto(input_info.type)
    logger.info(f"[ONNX Inspector/static] Input tensor: name='{input_name}', shape={input_shape}")

    input_height = 640
    input_width = 640
    if len(input_shape) == 4:
        if isinstance(input_shape[2], int) and input_shape[2] > 0:
            input_height = input_shape[2]
        if isinstance(input_shape[3], int) and input_shape[3] > 0:
            input_width = input_shape[3]

    # ---- Output info -------------------------------------------------------
    output_info = graph.output[0]
    output_name = output_info.name
    output_shape = _shape_from_type_proto(output_info.type)
    logger.info(f"[ONNX Inspector/static] Output tensor: name='{output_name}', shape={output_shape}")

    num_outputs = len(graph.output)

    # ---- Format detection (mirrors session-based logic) --------------------
    output_format = "unknown"
    num_classes = 0

    # NanoDet-multi: 6 outputs — 3 cls + 3 reg heads per stride
    if num_outputs == 6:
        out_shapes = [_shape_from_type_proto(out.type) for out in graph.output]
        if all(len(s) == 3 for s in out_shapes):
            # Require all last dimensions to be concrete integers; dynamic dims
            # prevent reliable format detection and must be treated as unknown.
            if all(isinstance(s[2], int) for s in out_shapes):
                last_dims = [s[2] for s in out_shapes]
                unique_last = set(last_dims)
                if len(unique_last) == 2:
                    d1, d2 = sorted(unique_last)  # d1 < d2
                    # d1 is the smaller dim: matches reg_channels = 4*(reg_max+1)
                    # d2 is the larger dim: num_classes
                    for reg_max in (7, 15):
                        reg_ch = 4 * (reg_max + 1)
                        if d1 == reg_ch and d2 > 0:
                            output_format = "nanodet_multi"
                            num_classes = d2
                            break

    if output_format == "unknown" and num_outputs >= 2:
        has_boxes_output = False
        for out in graph.output:
            out_shape = _shape_from_type_proto(out.type)
            if len(out_shape) >= 2:
                last_dim = out_shape[-1]
                if isinstance(last_dim, int) and last_dim == 4:
                    has_boxes_output = True
                    break
        if has_boxes_output:
            output_format = "ssd"
            for out in graph.output:
                out_shape = _shape_from_type_proto(out.type)
                if len(out_shape) == 3 and isinstance(out_shape[-1], int) and out_shape[-1] > 4:
                    num_classes = out_shape[-1]
                    break

    if output_format == "unknown" and len(output_shape) == 3:
        dim1 = output_shape[1]
        dim2 = output_shape[2]
        if isinstance(dim1, int) and isinstance(dim2, int):
            if dim1 < dim2 and dim1 > 4:
                output_format = "yolo11"
                num_classes = dim1 - 4
            elif dim2 < dim1 and dim2 > 5:
                output_format = "yolox"
                num_classes = dim2 - 5

    # ---- Classification model: rank-1/rank-2 output -------------------------
    if output_format == "unknown":
        output_format, num_classes = _detect_classify_format(output_shape)

    # ---- Embedded class names ----------------------------------------------
    class_names: dict = {}
    _model_task: str = ""
    for prop in model_proto.metadata_props:
        if prop.key == "task":
            _model_task = prop.value.lower()
        elif prop.key == "names":
            try:
                parsed = ast.literal_eval(prop.value)
                if isinstance(parsed, dict):
                    class_names = {int(k): str(v) for k, v in parsed.items()}
                    if num_classes == 0:
                        num_classes = len(class_names)
            except Exception:
                pass
        elif prop.key == "labels" and not class_names:
            try:
                parsed = json.loads(prop.value)
                if isinstance(parsed, dict):
                    class_names = {int(k): str(v) for k, v in parsed.items()}
                elif isinstance(parsed, list):
                    class_names = {i: str(v) for i, v in enumerate(parsed)}
                if num_classes == 0:
                    num_classes = len(class_names)
            except Exception:
                pass

    # ---- OBB override -------------------------------------------------------
    if _model_task == "obb" and output_format in ("yolo11", "unknown"):
        output_format = "yolo11_obb"
        if class_names:
            num_classes = len(class_names)
        elif len(output_shape) == 3 and isinstance(output_shape[1], int):
            num_classes = output_shape[1] - 5

    result = {
        "input_name": input_name,
        "input_shape": input_shape,
        "output_name": output_name,
        "output_shape": output_shape,
        "output_format": output_format,
        "num_classes": num_classes,
        "class_names": class_names,
        "input_width": input_width,
        "input_height": input_height,
        "has_dynamic_batch": has_dynamic_batch_dim(input_shape),
        "supports_batched_detection": supports_batched_detection(input_shape, output_format),
    }
    logger.info(
        f"[ONNX Inspector/static] Inspection complete — "
        f"format='{output_format}', classes={num_classes}, "
        f"input={input_width}x{input_height}"
    )
    return result


def inspect_onnx_model(model_path: str) -> dict:
    """Inspect an ONNX model and return a metadata dictionary.

    Parameters
    ----------
    model_path : str
        Absolute path to the .onnx file.

    Returns
    -------
    dict with keys:
        input_name       (str)   – name of the first input tensor
        input_shape      (list)  – shape of the first input tensor, e.g. [1, 3, 640, 640]
        output_name      (str)   – name of the first output tensor
        output_shape     (list)  – shape of the first output tensor
        output_format    (str)   – 'yolo11' | 'yolo11_obb' | 'yolox' | 'unknown'
        num_classes      (int)   – detected number of classes (0 if unknown)
        class_names      (dict)  – {int_id: str_name} extracted from ONNX metadata,
                                   empty dict if not available
        input_width      (int)   – inferred from shape[3] if available, else 640
        input_height     (int)   – inferred from shape[2] if available, else 640
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    logger.info(f"[ONNX Inspector] Loading model: {model_path}")

    try:
        session = make_session(model_path, providers=["CPUExecutionProvider"])
    except Exception as exc:
        # Quantized models (e.g. ConvInteger) may not be supported by the CPU
        # execution provider.  Fall back to reading metadata directly via the
        # ``onnx`` package without creating an inference session.
        logger.warning(
            f"[ONNX Inspector] Session creation failed ({exc}); "
            f"falling back to static graph inspection via the onnx package."
        )
        return _inspect_onnx_static(model_path)

    # ---- Input info --------------------------------------------------------
    input_detail = session.get_inputs()[0]
    input_name = input_detail.name
    input_shape = list(input_detail.shape)
    logger.info(f"[ONNX Inspector] Input tensor: name='{input_name}', shape={input_shape}")

    input_height = 640
    input_width = 640
    if len(input_shape) == 4:
        if isinstance(input_shape[2], int) and input_shape[2] > 0:
            input_height = input_shape[2]
        else:
            logger.warning(
                f"[ONNX Inspector] Height dimension is dynamic or invalid "
                f"('{input_shape[2]}'); defaulting to {input_height}."
            )
        if isinstance(input_shape[3], int) and input_shape[3] > 0:
            input_width = input_shape[3]
        else:
            logger.warning(
                f"[ONNX Inspector] Width dimension is dynamic or invalid "
                f"('{input_shape[3]}'); defaulting to {input_width}."
            )
    else:
        logger.warning(
            f"[ONNX Inspector] Unexpected input rank {len(input_shape)} "
            f"(expected 4 for BCHW); defaulting to {input_height}x{input_width}."
        )
    logger.info(f"[ONNX Inspector] Effective input size: {input_width}x{input_height} (WxH)")

    # ---- Output info -------------------------------------------------------
    output_detail = session.get_outputs()[0]
    output_name = output_detail.name
    output_shape = list(output_detail.shape)
    logger.info(f"[ONNX Inspector] Output tensor: name='{output_name}', shape={output_shape}")

    # Check for SSD-style multi-output model
    all_outputs = session.get_outputs()
    num_outputs = len(all_outputs)
    logger.info(f"[ONNX Inspector] Model has {num_outputs} output(s).")

    # ---- Format detection --------------------------------------------------
    # YOLO11 / YOLOv8 Ultralytics:  output shape [1, num_classes+4, num_anchors]
    # YOLOX:                         output shape [1, num_anchors, num_classes+5]
    # SSD:                           multiple outputs (boxes, scores, class_ids, ...)
    # NanoDet-multi:                 6 outputs — 3 cls heads + 3 reg heads per stride
    output_format = "unknown"
    num_classes = 0

    # NanoDet-multi detection: 6 outputs where half have the same last dim (cls)
    # and half have last dim == 4*(reg_max+1) (reg), grouped by equal anchor counts.
    if num_outputs == 6:
        out_shapes = [list(out.shape) for out in all_outputs]
        # All must be rank-3 (1, n_anchors, channels)
        if all(len(s) == 3 for s in out_shapes):
            # Require all last dimensions to be concrete integers; dynamic dims
            # prevent reliable format detection and must be treated as unknown.
            if all(isinstance(s[2], int) for s in out_shapes):
                last_dims = [s[2] for s in out_shapes]
                unique_last = set(last_dims)
                # Expect exactly 2 distinct last dims: cls channels and reg channels
                if len(unique_last) == 2:
                    d1, d2 = sorted(unique_last)  # d1 < d2
                    # d1 is the smaller dim: matches reg_channels = 4*(reg_max+1)
                    # d2 is the larger dim: num_classes
                    for reg_max in (7, 15):
                        reg_ch = 4 * (reg_max + 1)
                        if d1 == reg_ch and d2 > 0:
                            output_format = "nanodet_multi"
                            num_classes = d2
                            logger.info(
                                f"[ONNX Inspector] Detected NanoDet-multi format: "
                                f"num_classes={num_classes}, reg_max={reg_max}, "
                                f"6 separate heads"
                            )
                            break

    # SSD detection: multiple outputs where one has last dim == 4 (boxes)
    if output_format == "unknown" and num_outputs >= 2:
        has_boxes_output = False
        for out in all_outputs:
            out_shape = list(out.shape)
            if len(out_shape) >= 2:
                last_dim = out_shape[-1]
                if isinstance(last_dim, int) and last_dim == 4:
                    has_boxes_output = True
                    break
        if has_boxes_output:
            output_format = "ssd"
            # Try to determine num_classes from a multi-class scores output
            for out in all_outputs:
                out_shape = list(out.shape)
                if len(out_shape) == 3 and isinstance(out_shape[-1], int) and out_shape[-1] > 4:
                    num_classes = out_shape[-1]
                    break
            logger.info(
                f"[ONNX Inspector] Detected SSD-style multi-output model "
                f"({num_outputs} outputs). num_classes={num_classes}"
            )

    if output_format == "unknown" and len(output_shape) == 3:
        dim1 = output_shape[1]  # middle dimension
        dim2 = output_shape[2]  # last dimension
        if isinstance(dim1, int) and isinstance(dim2, int):
            # YOLO11: dim1 = num_classes+4, dim2 >> dim1 (many anchors)
            # YOLOX:  dim2 = num_classes+5, dim1 >> dim2 (many anchors)
            # NanoDet/GFL: dim2 = num_classes + 4*(reg_max+1), dim1 >> dim2
            #   e.g. 80 + 4*8 = 112 for reg_max=7

            # Check NanoDet/GFL pattern first: dim1 > dim2 and dim2 matches
            # num_classes + 4*(reg_max+1) for reg_max in [7, 15]
            detected_nanodet = False
            if dim1 > dim2 and dim2 > 5:
                for reg_max in (7, 15):
                    reg_channels = 4 * (reg_max + 1)
                    candidate_classes = dim2 - reg_channels
                    if candidate_classes > 0 and candidate_classes != (dim2 - 5):
                        # Only pick nanodet when the YOLOX interpretation (dim2-5)
                        # gives a non-standard class count but GFL gives a standard one
                        # (85 channels are a COCO YOLOX, not a 21-class NanoDet)
                        if (
                            candidate_classes in _STANDARD_CLASS_COUNTS
                            and (dim2 - 5) not in _STANDARD_CLASS_COUNTS
                        ):
                            output_format = "nanodet"
                            num_classes = candidate_classes
                            detected_nanodet = True
                            logger.info(
                                f"[ONNX Inspector] Detected NanoDet/GFL format: "
                                f"num_classes={num_classes}, reg_max={reg_max}"
                            )
                            break

            if not detected_nanodet:
                if dim1 < dim2 and dim1 > 4:
                    # Likely YOLO11 style: [1, C+4, anchors]
                    output_format = "yolo11"
                    num_classes = dim1 - 4
                elif dim2 < dim1 and dim2 > 5:
                    # Likely YOLOX style: [1, anchors, C+5]
                    output_format = "yolox"
                    num_classes = dim2 - 5
                elif dim1 < dim2 and dim1 == 4:
                    # Edge case: exactly 4 dims on axis1 → treat as yolo11 with 0 classes
                    output_format = "yolo11"
                    num_classes = 0
                else:
                    logger.warning(
                        f"[ONNX Inspector] Cannot determine output format from shape "
                        f"{output_shape} (dim1={dim1}, dim2={dim2}). "
                        f"Will default to 'yolo11' at inference time."
                    )
        else:
            logger.warning(
                f"[ONNX Inspector] Output shape has dynamic dimensions "
                f"{output_shape}; format detection skipped."
            )
    else:
        if output_format == "unknown":
            output_format, num_classes = _detect_classify_format(output_shape)
            if output_format == "classify":
                logger.info(
                    f"[ONNX Inspector] Detected classification model "
                    f"(rank-{len(output_shape)} output): num_classes={num_classes}"
                )
            else:
                logger.warning(
                    f"[ONNX Inspector] Unexpected output rank {len(output_shape)} "
                    f"(expected 3); format detection skipped."
                )

    if output_format != "unknown":
        logger.info(
            f"[ONNX Inspector] Detected output format: '{output_format}', "
            f"estimated num_classes={num_classes}"
        )

    # ---- Embedded class names (Ultralytics) --------------------------------
    class_names: dict = {}
    _model_task: str = ""
    try:
        model_meta = session.get_modelmeta()
        custom_meta = model_meta.custom_metadata_map  # dict[str, str]
        # Read the Ultralytics 'task' field (e.g. 'detect', 'obb', 'segment')
        _model_task = str(custom_meta.get("task", "")).lower()
        if "names" in custom_meta:
            raw = custom_meta["names"]
            # Ultralytics stores as Python dict literal: {0: 'cat', 1: 'dog', ...}
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, dict):
                    class_names = {int(k): str(v) for k, v in parsed.items()}
                    if num_classes == 0:
                        num_classes = len(class_names)
                    logger.info(
                        f"[ONNX Inspector] Found {len(class_names)} class names "
                        f"in ONNX 'names' metadata."
                    )
            except Exception as exc:
                logger.warning(
                    f"[ONNX Inspector] Failed to parse 'names' metadata: {exc}"
                )
        if not class_names and "labels" in custom_meta:
            raw = custom_meta["labels"]
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    class_names = {int(k): str(v) for k, v in parsed.items()}
                elif isinstance(parsed, list):
                    class_names = {i: str(v) for i, v in enumerate(parsed)}
                if num_classes == 0:
                    num_classes = len(class_names)
                logger.info(
                    f"[ONNX Inspector] Found {len(class_names)} class names "
                    f"in ONNX 'labels' metadata."
                )
            except Exception as exc:
                logger.warning(
                    f"[ONNX Inspector] Failed to parse 'labels' metadata: {exc}"
                )
        if not class_names:
            logger.info(
                "[ONNX Inspector] No embedded class names found in ONNX metadata."
            )
    except Exception as exc:
        logger.warning(f"[ONNX Inspector] Could not read model metadata: {exc}")

    # ---- OBB override -------------------------------------------------------
    # Ultralytics OBB (Oriented Bounding Box) models have task='obb' in their
    # metadata.  Their output shape is [1, nc+5, anchors] (4 box + nc classes +
    # 1 angle) instead of the standard [1, nc+4, anchors] used by yolo11.
    # The shape-based detector above would compute nc = dim1-4 (off by one);
    # correct it here using the confirmed class count from embedded names when
    # available, or by subtracting 5 instead of 4.
    if _model_task == "obb" and output_format in ("yolo11", "unknown"):
        output_format = "yolo11_obb"
        if class_names:
            num_classes = len(class_names)
        elif len(output_shape) == 3 and isinstance(output_shape[1], int):
            num_classes = output_shape[1] - 5
        logger.info(
            f"[ONNX Inspector] OBB task detected — overriding format to "
            f"'yolo11_obb', num_classes={num_classes}"
        )

    result = {
        "input_name": input_name,
        "input_shape": input_shape,
        "output_name": output_name,
        "output_shape": output_shape,
        "output_format": output_format,
        "num_classes": num_classes,
        "class_names": class_names,
        "input_width": input_width,
        "input_height": input_height,
        "has_dynamic_batch": has_dynamic_batch_dim(input_shape),
        "supports_batched_detection": supports_batched_detection(input_shape, output_format),
    }
    logger.info(
        f"[ONNX Inspector] Inspection complete — "
        f"format='{output_format}', classes={num_classes}, "
        f"input={input_width}x{input_height}"
    )
    return result


def load_class_names_from_file(filepath: str) -> dict:
    """Load class names from a .txt or .json file.

    .txt  – one class name per line, index is line number starting at 0
    .json – either {"0": "cat", "1": "dog"} or ["cat", "dog"]

    Returns a dict {int_id: str_name} or empty dict on failure.
    """
    if not filepath or not os.path.isfile(filepath):
        logger.warning(f"[ONNX Inspector] Class names file not found: '{filepath}'")
        return {}

    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == ".txt":
            with open(filepath, "r", encoding="utf-8") as f:
                lines = [l.rstrip("\n") for l in f.readlines()]
            result = {i: name for i, name in enumerate(lines) if name}
            logger.info(f"[ONNX Inspector] Loaded {len(result)} class names from TXT: {filepath}")
            return result
        elif ext == ".json":
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                result = {int(k): str(v) for k, v in data.items()}
            elif isinstance(data, list):
                result = {i: str(v) for i, v in enumerate(data)}
            else:
                logger.warning(f"[ONNX Inspector] Unexpected JSON format in {filepath}")
                return {}
            logger.info(f"[ONNX Inspector] Loaded {len(result)} class names from JSON: {filepath}")
            return result
        else:
            logger.warning(f"[ONNX Inspector] Unsupported class names file extension: '{ext}'")
    except Exception as exc:
        logger.warning(f"[ONNX Inspector] Failed to load class names from '{filepath}': {exc}")
    return {}
