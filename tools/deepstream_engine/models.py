"""
Model catalog for the DeepStream engine.

Resolves the model name stored in a CvStudio node (e.g. "YOLOX-Nano(416x416)")
to the real ONNX file and the properties nvinfer needs: tensor layout, input
size, batch dimension and class names.

Resolution order:
  1. CvStudio's object-detection registry (custom uploads + built-ins),
  2. the built-in tables below (mirrors of the CvStudio node tables),
  3. the ONNX file itself, which overrides sizes / class names when readable.
"""

from __future__ import annotations

import ast
import json
import re
import runpy
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_DL_ROOT = REPO_ROOT / "node" / "DLNode"
_OD_ROOT = _DL_ROOT / "object_detection"
_CLS_ROOT = _DL_ROOT / "classification"

# Detector tensor layouts the generated bbox parser library understands.
SUPPORTED_DETECTOR_FORMATS = {"yolo11", "yolox"}

# name -> (candidate paths relative to object_detection/, format, width, height, classes)
_BUILTIN_DETECTORS: dict[str, tuple[tuple[str, ...], str, int, int, str]] = {
    "YOLOX-Nano(416x416)": (("YOLOX/model/yolox_nano.onnx",), "yolox", 416, 416, "coco"),
    "YOLOX-Tiny(416x416)": (("YOLOX/model/yolox_tiny.onnx",), "yolox", 416, 416, "coco"),
    "YOLOX-S(640x640)": (("YOLOX/model/yolox_s.onnx",), "yolox", 640, 640, "coco"),
    "YOLO11Nano": (("YOLO/model/yolo11_n.onnx",), "yolo11", 608, 416, "coco"),
    "yolo8n_B": (
        ("YOLO/model/yolov8n.onnx", "YOLO/model/yolo11_n.onnx"),
        "yolo11", 640, 640, "coco",
    ),
    "YOLOv8m(640x640)": (("YOLO/model/yolov8m.onnx",), "yolo11", 640, 640, "coco"),
    "yolo8s_B": (
        ("YOLO/model/yolov8s.onnx", "YOLO/model/yolo11_n.onnx"),
        "yolo11", 640, 640, "coco",
    ),
    "FreeYOLO-Nano(640x640)": (
        ("FreeYOLO/model/yolo_free_nano_640x640.onnx",), "yolox", 640, 640, "coco",
    ),
    "FreeYOLO-CrowdHuman(640x640)": (
        ("FreeYOLO/model/yolo_free_nano_crowdhuman_640x640.onnx",),
        "yolox", 640, 640, "person",
    ),
    "Light-Weight Person Detector": (
        ("LightWeightPersonDetector/model/model.onnx",), "yolox", 192, 192, "person",
    ),
    "YOLOTENNIS": (("TennisYOLO/model/tennis.onnx",), "yolo11", 608, 608, "tennis"),
    "YOLO-DOTA-OBB(416x416)": (("YOLO/model/yolo-dota.onnx",), "yolo11_obb", 416, 416, ""),
    "BlazeFace(128x128)": (("blaze.onnx",), "blazeface", 128, 128, ""),
}

_BUILTIN_CLASSIFIERS: dict[str, Path] = {
    "MobileNetV3 Small": _CLS_ROOT / "MobileNetV3/model/MobileNetV3Small.onnx",
    "MobileNetV3 Large": _CLS_ROOT / "MobileNetV3/model/MobileNetV3Large.onnx",
    "EfficientNet B0": _CLS_ROOT / "EfficientNetB0/model/EfficientNetB0.onnx",
    "ResNet50": _CLS_ROOT / "ResNet50/model/ResNet50.onnx",
    "Yolo-cls": _CLS_ROOT / "Yolo-cls/model/son.onnx",
    "Gender Recognition": _CLS_ROOT / "GenderRecognition/model/GenderRecognition.onnx",
    "Pedestrian Gender": REPO_ROOT / "pedestrian_gender.onnx",
}


@dataclass
class ModelSpec:
    """Everything nvinfer needs to know about one ONNX model."""

    name: str
    task: str  # "detect" | "classify"
    output_format: str = "unknown"
    path: Path | None = None  # existing ONNX file, None when not found
    input_width: int = 0
    input_height: int = 0
    channels_last: bool = False
    dynamic_batch: bool = False
    dynamic_size: bool = False
    num_classes: int = 0
    class_names: list[str] = field(default_factory=list)
    inspected: bool = False

    @property
    def filename(self) -> str:
        """File name of the model inside the generated project's models/ dir."""
        return f"{sanitize_filename(self.name)}.onnx"


def sanitize_filename(name: str) -> str:
    """Convert a model display name to a safe file stem."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip().lower()).strip("_.")
    return stem or "model"


def resolve_detector(name: str) -> ModelSpec:
    """Resolve an ObjectDetection model name."""
    spec = ModelSpec(name=name, task="detect")

    entry = _registry_entry(name)
    if entry is not None:
        spec.path = Path(entry["path"])
        spec.output_format = str(entry.get("output_format") or "unknown")
        spec.input_width = int(entry.get("input_width") or 0)
        spec.input_height = int(entry.get("input_height") or 0)
        spec.class_names = _names_from_mapping(entry.get("class_names") or {})
    elif name in _BUILTIN_DETECTORS:
        candidates, fmt, width, height, classes = _BUILTIN_DETECTORS[name]
        paths = [_OD_ROOT / rel for rel in candidates]
        spec.path = next((p for p in paths if p.is_file()), None)
        spec.output_format = fmt
        spec.input_width, spec.input_height = width, height
        spec.class_names = _builtin_class_names(classes)
    else:
        # A custom upload referenced by a JSON coming from another machine.
        upload = _OD_ROOT / "CustomONNX" / "models" / f"{name}.onnx"
        spec.path = upload if upload.is_file() else None

    _apply_onnx_metadata(spec)
    if spec.output_format == "unknown" and not spec.inspected:
        # Nothing but the name to go on: follow CvStudio's naming convention.
        spec.output_format = "yolox" if "yolox" in name.lower() else "yolo11"
    spec.num_classes = len(spec.class_names) or spec.num_classes
    return spec


def resolve_classifier(name: str) -> ModelSpec:
    """Resolve a Classification model name."""
    spec = ModelSpec(name=name, task="classify", output_format="classify")
    path = _BUILTIN_CLASSIFIERS.get(name)
    if path is None:
        path = _CLS_ROOT / "CustomONNX" / "models" / f"{name}.onnx"
    spec.path = path if path.is_file() else None
    _apply_onnx_metadata(spec)
    spec.num_classes = len(spec.class_names) or spec.num_classes
    return spec


def _registry_entry(name: str) -> dict | None:
    registry = _OD_ROOT / "custom_models_registry.json"
    try:
        entries = json.loads(registry.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if (
            isinstance(entry, dict)
            and entry.get("name") == name
            and Path(str(entry.get("path", ""))).is_file()
        ):
            return entry
    return None


def _names_from_mapping(mapping: dict) -> list[str]:
    """{id: name} (int or str keys) -> list ordered by class id, gaps filled."""
    names: dict[int, str] = {}
    for key, value in mapping.items():
        try:
            names[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    if not names:
        return []
    return [names.get(i, f"class_{i}") for i in range(max(names) + 1)]


def _builtin_class_names(kind: str) -> list[str]:
    if kind == "person":
        return ["person"]
    if kind == "tennis":
        return ["player1", "player2", "ball"]
    if kind == "coco":
        try:
            module = runpy.run_path(str(_OD_ROOT / "coco_class_names.py"))
            return _names_from_mapping(module["coco_class_names"])
        except (OSError, KeyError, SyntaxError):
            return [f"class_{i}" for i in range(80)]
    return []


def _apply_onnx_metadata(spec: ModelSpec) -> None:
    """Refine *spec* with what the ONNX file itself declares (best effort)."""
    if spec.path is None:
        return
    try:
        import onnx
    except ImportError:
        return
    try:
        model = onnx.load(str(spec.path), load_external_data=False)
    except Exception:
        return

    initializers = {init.name for init in model.graph.initializer}
    inputs = [i for i in model.graph.input if i.name not in initializers]
    outputs = list(model.graph.output)
    if not inputs or not outputs:
        return
    in_shape = _shape(inputs[0])
    out_shape = _shape(outputs[0])
    if len(in_shape) != 4:
        return
    spec.inspected = True

    spec.dynamic_batch = not isinstance(in_shape[0], int)
    spec.channels_last = in_shape[3] == 3 and in_shape[1] != 3
    height, width = in_shape[1:3] if spec.channels_last else in_shape[2:4]
    if isinstance(height, int) and isinstance(width, int):
        spec.input_height, spec.input_width = height, width
    else:
        spec.dynamic_size = True

    meta = {prop.key: prop.value for prop in model.metadata_props}
    names = _parse_ultralytics_names(meta.get("names", ""))
    # Built-in tables carry CvStudio's own labels; only fill the gaps.
    if names and not spec.class_names:
        spec.class_names = names

    if spec.task == "classify":
        if len(out_shape) == 2 and isinstance(out_shape[1], int):
            spec.num_classes = out_shape[1]
        return

    if len(inputs) > 1 or len(outputs) > 1 or len(out_shape) != 3:
        if spec.output_format in SUPPORTED_DETECTOR_FORMATS:
            spec.output_format = "unknown"
        return
    dims = [d for d in out_shape[1:] if isinstance(d, int)]
    if len(dims) != 2:
        return
    attrs = min(dims)
    if spec.output_format == "unknown":
        if meta.get("task") == "detect":
            spec.output_format = "yolo11"
        elif meta.get("task"):
            spec.output_format = str(meta["task"])
    if spec.output_format == "yolo11":
        spec.num_classes = attrs - 4
    elif spec.output_format == "yolox":
        spec.num_classes = attrs - 5
    if spec.class_names and len(spec.class_names) != spec.num_classes > 0:
        spec.class_names = (
            spec.class_names[: spec.num_classes]
            + [f"class_{i}" for i in range(len(spec.class_names), spec.num_classes)]
        )


def _shape(value_info) -> list:
    return [
        dim.dim_value if dim.HasField("dim_value") else (dim.dim_param or "?")
        for dim in value_info.type.tensor_type.shape.dim
    ]


def _parse_ultralytics_names(raw: str) -> list[str]:
    """Parse the "{0: 'person', ...}" metadata written by Ultralytics exports."""
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []
    if isinstance(parsed, dict):
        return _names_from_mapping(parsed)
    if isinstance(parsed, (list, tuple)):
        return [str(n) for n in parsed]
    return []
