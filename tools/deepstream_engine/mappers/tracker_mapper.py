"""
Tracker element mapper.

Maps the CvStudio MultiObjectTracking node to nvtracker. CvStudio's Python
trackers do not exist in DeepStream, so each one is mapped to the closest
NvMultiObjectTracker algorithm, using the configuration files shipped with
the DeepStream SDK (none of the selected ones needs an extra ReID model).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tools.deepstream_engine.hardware_profile import HardwareProfile, RTX_5070_PROFILE
from tools.deepstream_engine.parser import NodeInfo

TRACKER_NODE_TAGS = {"MultiObjectTracking"}

DEEPSTREAM_ROOT = "/opt/nvidia/deepstream/deepstream"
_TRACKER_LIB = f"{DEEPSTREAM_ROOT}/lib/libnvds_nvmultiobjecttracker.so"
_TRACKER_CONFIG_DIR = f"{DEEPSTREAM_ROOT}/samples/configs/deepstream-app"

# CvStudio tracker name (lower case) -> NvMultiObjectTracker algorithm
_ALGORITHMS = {
    "motpy": "IOU",
    "iou tracker": "IOU",
    "bytetrack": "NvSORT",
    "sort": "NvSORT",
    "oc-sort": "NvSORT",
    "kalman filter": "NvSORT",
    "norfair": "NvDCF_perf",
    "centertrack": "NvDCF_perf",
    "bot-sort": "NvDCF_perf",
}
_DEFAULT_ALGORITHM = "NvSORT"


@dataclass
class TrackerConfig:
    """nvtracker configuration."""

    tracker_name: str = "ByteTrack"  # CvStudio tracker
    algorithm: str = _DEFAULT_ALGORITHM  # DeepStream tracker
    tracker_width: int = 960
    tracker_height: int = 544
    config_entries: dict[str, str] = field(default_factory=dict)


def map_tracker(
    node: NodeInfo,
    profile: HardwareProfile = RTX_5070_PROFILE,
    warnings: list[str] | None = None,
) -> TrackerConfig | None:
    """Convert a CvStudio tracker node into an nvtracker config."""
    warnings = warnings if warnings is not None else []
    label = f"{node.node_id}:{node.node_tag}"

    if node.setting("EnableCheckbox", True) is False:
        warnings.append(f"{label}: tracking is disabled in CvStudio; tracker skipped.")
        return None

    tracker_name = node.text_setting("Input02Value", "ByteTrack")
    algorithm = _ALGORITHMS.get(tracker_name.lower())
    if algorithm is None:
        algorithm = _DEFAULT_ALGORITHM
        warnings.append(
            f"{label}: unknown tracker '{tracker_name}', using {algorithm}."
        )

    config = TrackerConfig(tracker_name=tracker_name, algorithm=algorithm)
    config.config_entries = {
        "tracker-width": str(config.tracker_width),
        "tracker-height": str(config.tracker_height),
        "gpu-id": str(profile.gpu_id),
        "ll-lib-file": _TRACKER_LIB,
        "ll-config-file": f"{_TRACKER_CONFIG_DIR}/config_tracker_{algorithm}.yml",
        "display-tracking-id": "1",
    }
    return config
