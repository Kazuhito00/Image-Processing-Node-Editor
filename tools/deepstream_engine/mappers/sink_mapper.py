"""
Sink element mapper.

Maps CvStudio output/action nodes to DeepStream sink elements:
  - VideoWriter → File Sink (mp4/mkv with nvv4l2h264enc)
  - ImageConcat / Display → Fake Sink or EGL Sink
  - MQTT → Message Sink (nvmsgbroker)
  - MongoDB → Message Sink (custom)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tools.deepstream_engine.parser import NodeInfo
from tools.deepstream_engine.hardware_profile import HardwareProfile, RTX_5070_PROFILE


SINK_NODE_TAGS = {"VideoWriter", "ImageConcat", "MQTT", "MongoDB", "VideoRecorder"}

# Additional nodes that indicate display intent
DISPLAY_NODE_TAGS = {"Heatmap", "ObjHeatmap", "Chart", "ObjChart", "Map"}


@dataclass
class SinkConfig:
    """DeepStream sink configuration."""

    sink_id: int
    sink_type: str  # "file", "egl", "rtsp", "msg", "fake"
    output_path: str = ""
    properties: dict[str, Any] = field(default_factory=dict)
    config_entries: dict[str, str] = field(default_factory=dict)


def map_sink(
    node: NodeInfo,
    sink_index: int = 0,
    profile: HardwareProfile = RTX_5070_PROFILE,
) -> SinkConfig | None:
    """Convert a CvStudio output node into a DeepStream sink config."""
    tag = node.node_tag
    settings = node.settings

    if tag in ("VideoWriter", "VideoRecorder"):
        return _map_file_sink(node, sink_index, profile)
    elif tag == "ImageConcat":
        return map_display_sink(sink_index, profile)
    elif tag == "MQTT":
        return _map_mqtt_sink(node, sink_index, profile)
    elif tag == "MongoDB":
        return _map_msgbroker_sink(node, sink_index, profile)

    return None


def map_display_sink(sink_index: int = 0, profile: HardwareProfile = RTX_5070_PROFILE) -> SinkConfig:
    """Create an EGL display sink for visualization."""
    return SinkConfig(
        sink_id=sink_index,
        sink_type="egl",
        config_entries={
            "type": "2",  # EGL
            "sync": "0",
            "gpu-id": str(profile.gpu_id),
            "width": "1920",
            "height": "1080",
            "nvbuf-memory-type": str(profile.nvbuf_memory_type),
        },
    )


def map_fake_sink(sink_index: int = 0) -> SinkConfig:
    """Create a fake sink (headless / metadata-only processing)."""
    return SinkConfig(
        sink_id=sink_index,
        sink_type="fake",
        config_entries={
            "type": "1",  # Fake sink
            "enable": "1",
        },
    )


def _map_file_sink(
    node: NodeInfo,
    sink_index: int,
    profile: HardwareProfile,
) -> SinkConfig:
    """Map VideoWriter to file sink with hardware encoding."""
    settings = node.settings
    output_path = _extract_text_value(settings, "Input01Value", "output/output.mp4")

    return SinkConfig(
        sink_id=sink_index,
        sink_type="file",
        output_path=output_path,
        config_entries={
            "type": "3",  # File
            "container": "1",  # MP4
            "codec": "1",  # H.264 HW encoder
            "sync": "0",
            "bitrate": "8000000",
            "gpu-id": str(profile.gpu_id),
            "output-file": output_path,
        },
    )


def _map_mqtt_sink(
    node: NodeInfo,
    sink_index: int,
    profile: HardwareProfile,
) -> SinkConfig:
    """Map MQTT action node to DeepStream message sink."""
    settings = node.settings
    broker_url = _extract_text_value(settings, "Input01Value", "localhost")
    topic = _extract_text_value(settings, "Input02Value", "cvstudio/detections")

    return SinkConfig(
        sink_id=sink_index,
        sink_type="msg",
        config_entries={
            "type": "6",  # Message sink
            "msg-conv-config": "msg_conv_config.txt",
            "msg-broker-proto-lib": "/opt/nvidia/deepstream/deepstream/lib/libnvds_mqtt_proto.so",
            "msg-broker-conn-str": broker_url,
            "topic": topic,
            "msg-conv-msg2p-lib": "/opt/nvidia/deepstream/deepstream/lib/libnvds_utils.so",
        },
    )


def _map_msgbroker_sink(
    node: NodeInfo,
    sink_index: int,
    profile: HardwareProfile,
) -> SinkConfig:
    """Map MongoDB action node to message broker sink."""
    settings = node.settings
    conn_str = _extract_text_value(settings, "Input01Value", "mongodb://localhost:27017")

    return SinkConfig(
        sink_id=sink_index,
        sink_type="msg",
        config_entries={
            "type": "6",
            "msg-conv-config": "msg_conv_config.txt",
            "msg-broker-proto-lib": "/opt/nvidia/deepstream/deepstream/lib/libnvds_redis_proto.so",
            "msg-broker-conn-str": conn_str,
            "msg-conv-msg2p-lib": "/opt/nvidia/deepstream/deepstream/lib/libnvds_utils.so",
        },
    )


def _extract_text_value(settings: dict, key_suffix: str, default: str = "") -> str:
    """Extract a text value from node settings."""
    for key, value in settings.items():
        if key.endswith(key_suffix):
            if isinstance(value, str) and value:
                return value
    return default
