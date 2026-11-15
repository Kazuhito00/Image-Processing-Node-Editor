"""
Inference element mapper.

Maps CvStudio DL nodes to nvinfer configurations:
  - ObjectDetection (YOLO11 / YOLOv8 / YOLOX / FreeYOLO tensors) → primary GIE,
    decoded by the bbox parser library generated with the project.
  - Classification → secondary GIE on detected objects (or primary GIE on the
    full frame when the graph has no detector).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tools.deepstream_engine.hardware_profile import HardwareProfile, RTX_5070_PROFILE
from tools.deepstream_engine.models import (
    SUPPORTED_DETECTOR_FORMATS,
    ModelSpec,
    resolve_classifier,
    resolve_detector,
)
from tools.deepstream_engine.parser import NodeInfo

INFERENCE_NODE_TAGS = {"ObjectDetection", "Classification"}

# DL nodes whose post-processing has no DeepStream counterpart in this engine.
UNSUPPORTED_INFERENCE_TAGS = {
    "PoseEstimation",
    "FaceDetection",
    "SemanticSegmentation",
    "MonocularDepthEstimation",
    "LLIE",
    "OnlineTraining",
    "AudioClassification",
}

PARSER_LIB_DIR = "nvdsinfer_custom_impl_cvstudio"
PARSER_LIB = f"{PARSER_LIB_DIR}/libnvdsinfer_custom_impl_cvstudio.so"

# output_format -> nvinfer pre-processing + parser, mirroring CvStudio's own
# pre-processing so the DeepStream results match what the editor shows.
_DETECTOR_FORMATS = {
    # Ultralytics: RGB, [0, 1], stretch resize.
    "yolo11": {
        "net-scale-factor": "0.0039215697906911373",
        "model-color-format": "0",
        "maintain-aspect-ratio": "0",
        "parse-bbox-func-name": "NvDsInferParseCvStudioYolo",
    },
    # YOLOX / FreeYOLO: BGR, [0, 255], top-left letterbox.
    "yolox": {
        "net-scale-factor": "1.0",
        "model-color-format": "1",
        "maintain-aspect-ratio": "1",
        "symmetric-padding": "0",
        "parse-bbox-func-name": "NvDsInferParseCvStudioYoloX",
    },
}

_NMS_IOU_THRESHOLD = 0.45


@dataclass
class InferenceConfig:
    """nvinfer configuration for a single model."""

    gie_id: int
    gie_type: str  # "primary" or "secondary"
    node_tag: str
    model_name: str = ""
    source_model_path: str = ""  # ONNX found on this machine, "" when missing
    onnx_path: str = ""  # relative to the generated project
    trt_engine_path: str = ""
    labels_file: str = ""
    input_width: int = 0
    input_height: int = 0
    num_classes: int = 0
    output_format: str = ""
    score_threshold: float = 0.4
    nms_threshold: float = _NMS_IOU_THRESHOLD
    class_names: list[str] = field(default_factory=list)
    network_type: int = 0  # 0=Detector, 1=Classifier
    batch_size: int = 1
    interval: int = 0
    operate_on_gie_id: int = -1
    config_entries: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def uses_custom_parser(self) -> bool:
        return self.network_type == 0


def map_inference(
    node: NodeInfo,
    gie_index: int = 0,
    is_primary: bool = True,
    profile: HardwareProfile = RTX_5070_PROFILE,
    num_sources: int = 1,
    warnings: list[str] | None = None,
) -> InferenceConfig | None:
    """Convert a CvStudio DL node into an nvinfer configuration."""
    warnings = warnings if warnings is not None else []
    if node.node_tag == "ObjectDetection":
        return _map_object_detection(node, gie_index, profile, num_sources, warnings)
    if node.node_tag == "Classification":
        return _map_classification(node, gie_index, is_primary, profile, num_sources, warnings)
    return None


def _map_object_detection(
    node: NodeInfo,
    gie_index: int,
    profile: HardwareProfile,
    num_sources: int,
    warnings: list[str],
) -> InferenceConfig | None:
    label = f"{node.node_id}:{node.node_tag}"
    model_name = node.text_setting("Input02Value", "YOLOX-Nano(416x416)")
    spec = resolve_detector(model_name)

    if spec.output_format not in SUPPORTED_DETECTOR_FORMATS:
        warnings.append(
            f"{label}: model '{model_name}' has output format '{spec.output_format}', "
            f"which has no DeepStream bbox parser (supported: "
            f"{', '.join(sorted(SUPPORTED_DETECTOR_FORMATS))}); detector skipped."
        )
        return None
    if spec.dynamic_size:
        warnings.append(
            f"{label}: model '{model_name}' has a dynamic input size; "
            f"'infer-dims' was set to 3;640;640, adjust it to the training size."
        )
        spec.input_width = spec.input_height = 640

    config = _base_config(spec, node, gie_index, "primary", profile, warnings, label)
    config.network_type = 0
    config.score_threshold = node.float_setting("Input03Value", 0.4)
    config.batch_size = _batch_size(spec, num_sources, profile, warnings, label)
    config.interval = max(profile.infer_interval - 1, 0)

    prop = _common_properties(config, spec, profile)
    prop.update(_DETECTOR_FORMATS[spec.output_format])
    prop.update({
        "num-detected-classes": str(config.num_classes or 1),
        "interval": str(config.interval),
        "process-mode": "1",
        "network-type": "0",
        "cluster-mode": "2",
        "custom-lib-path": PARSER_LIB,
    })
    rejected = _rejected_class_ids(node.setting("RejectedClassesValue", ""))
    if rejected:
        prop["filter-out-class-ids"] = ";".join(str(i) for i in rejected)

    config.config_entries = {
        "[property]": prop,
        "[class-attrs-all]": {
            "pre-cluster-threshold": f"{config.score_threshold:g}",
            "nms-iou-threshold": f"{config.nms_threshold:g}",
            "topk": "300",
        },
    }
    return config


def _map_classification(
    node: NodeInfo,
    gie_index: int,
    is_primary: bool,
    profile: HardwareProfile,
    num_sources: int,
    warnings: list[str],
) -> InferenceConfig:
    label = f"{node.node_id}:{node.node_tag}"
    model_name = node.text_setting("Input02Value", "MobileNetV3 Small")
    spec = resolve_classifier(model_name)
    if spec.dynamic_size or not spec.input_width:
        spec.input_width = spec.input_height = 224

    gie_type = "primary" if is_primary else "secondary"
    config = _base_config(spec, node, gie_index, gie_type, profile, warnings, label)
    config.network_type = 1
    config.score_threshold = node.float_setting("ScoreThresholdValue", 0.5)
    config.operate_on_gie_id = -1 if is_primary else 1
    if is_primary:
        config.batch_size = _batch_size(spec, num_sources, profile, warnings, label)
    else:
        # Objects of all streams are batched together.
        config.batch_size = min(profile.max_batch_size, 16) if spec.dynamic_batch else 1

    prop = _common_properties(config, spec, profile)
    prop.update({
        "net-scale-factor": "0.0039215697906911373",
        "model-color-format": "0",
        "process-mode": "1" if is_primary else "2",
        "network-type": "1",
        "classifier-threshold": f"{config.score_threshold:g}",
    })
    if not is_primary:
        prop["operate-on-gie-id"] = str(config.operate_on_gie_id)
    if spec.channels_last:
        prop["network-input-order"] = "1"

    warnings.append(
        f"{label}: classifier '{model_name}' is configured with RGB input scaled to "
        f"[0, 1]; adjust net-scale-factor/offsets in the generated config if the model "
        f"was trained with another normalisation."
    )
    config.config_entries = {"[property]": prop}
    return config


def _base_config(
    spec: ModelSpec,
    node: NodeInfo,
    gie_index: int,
    gie_type: str,
    profile: HardwareProfile,
    warnings: list[str],
    label: str,
) -> InferenceConfig:
    onnx_path = f"models/{spec.filename}"
    if spec.path is None:
        warnings.append(
            f"{label}: ONNX file for model '{spec.name}' was not found on this machine; "
            f"copy it to {onnx_path} before running."
        )
    if profile.int8_enabled:
        warnings.append(
            f"{label}: INT8 needs a calibration table at models/{spec.filename}.calib; "
            f"without it TensorRT falls back to FP32."
        )
    return InferenceConfig(
        gie_id=gie_index + 1,
        gie_type=gie_type,
        node_tag=node.node_tag,
        model_name=spec.name,
        source_model_path=str(spec.path) if spec.path else "",
        onnx_path=onnx_path,
        input_width=spec.input_width,
        input_height=spec.input_height,
        num_classes=spec.num_classes,
        output_format=spec.output_format,
        class_names=list(spec.class_names),
    )


def _common_properties(
    config: InferenceConfig,
    spec: ModelSpec,
    profile: HardwareProfile,
) -> dict[str, str]:
    # nvinfer network-mode: 0=FP32, 1=INT8, 2=FP16
    if profile.int8_enabled:
        mode, precision = "1", "int8"
    elif profile.fp16_enabled:
        mode, precision = "2", "fp16"
    else:
        mode, precision = "0", "fp32"

    # Same name nvinfer gives the engine it builds, so the first-run build is
    # reused on every later start.
    config.trt_engine_path = (
        f"{config.onnx_path}_b{config.batch_size}_gpu{profile.gpu_id}_{precision}.engine"
    )
    prop = {
        "gpu-id": str(profile.gpu_id),
        "onnx-file": config.onnx_path,
        "model-engine-file": config.trt_engine_path,
        "batch-size": str(config.batch_size),
        "network-mode": mode,
        "gie-unique-id": str(config.gie_id),
        "workspace-size": str(profile.tensorrt_workspace_mb),
    }
    if profile.int8_enabled:
        prop["int8-calib-file"] = f"{config.onnx_path}.calib"
    if config.class_names:
        config.labels_file = f"labels_gie{config.gie_id}.txt"
        prop["labelfile-path"] = config.labels_file
    if (spec.dynamic_size or not spec.inspected) and config.input_width:
        if spec.channels_last:
            prop["infer-dims"] = f"{config.input_height};{config.input_width};3"
        else:
            prop["infer-dims"] = f"3;{config.input_height};{config.input_width}"
    return prop


def _batch_size(
    spec: ModelSpec,
    num_sources: int,
    profile: HardwareProfile,
    warnings: list[str],
    label: str,
) -> int:
    """Full-frame batch size: one frame per stream when the model allows it."""
    if spec.dynamic_batch:
        return max(1, min(num_sources, profile.max_batch_size))
    if num_sources > 1:
        warnings.append(
            f"{label}: model '{spec.name}' is not known to accept a dynamic batch, so "
            f"the {num_sources} streams are inferred one frame at a time; re-export the "
            f"ONNX with a dynamic batch axis for batched inference."
        )
    return 1


def _rejected_class_ids(value) -> list[int]:
    """Parse CvStudio's class rejection filter ("0: person, 2: car" or "0,2")."""
    ids = set()
    for part in str(value or "").split(","):
        head = part.split(":")[0].strip()
        if head.isdigit():
            ids.add(int(head))
    return sorted(ids)
