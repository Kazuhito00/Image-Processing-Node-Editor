"""
Source element mapper.

Maps CvStudio input nodes to deepstream-app [sourceN] groups:
  - RTSP    → type=4 (RTSP, hardware decode, auto-reconnect)
  - Video   → type=3 (file URI)
  - HLS     → type=2 (URI)
  - YouTube → type=2 (URI, needs a resolved stream URL)
  - Webcam  → type=1 (V4L2 camera)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tools.deepstream_engine.hardware_profile import HardwareProfile, RTX_5070_PROFILE
from tools.deepstream_engine.parser import NodeInfo

SOURCE_NODE_TAGS = {"RTSP", "Video", "Webcam", "HLS", "YouTube"}

# Image-producing nodes with no DeepStream source equivalent.
UNSUPPORTED_SOURCE_TAGS = {"Image", "WebRTC", "MJPEG", "ScreenCapture"}

PLACEHOLDER_RTSP = "rtsp://CHANGE_ME:554/stream"
PLACEHOLDER_FILE = "file:///CHANGE_ME/video.mp4"
PLACEHOLDER_URI = "https://CHANGE_ME/stream.m3u8"


@dataclass
class DeepStreamSource:
    """A mapped deepstream-app source group."""

    source_id: int
    source_type: str  # "rtsp", "file", "uri", "v4l2"
    node_tag: str = ""
    uri: str = ""
    device: str = ""
    loop: bool = False
    config_entries: dict[str, str] = field(default_factory=dict)

    @property
    def is_live(self) -> bool:
        return self.source_type in ("rtsp", "v4l2") or self.node_tag in ("HLS", "YouTube")

    @property
    def local_path(self) -> Path | None:
        """Host path of a file source, None for network / camera sources."""
        if self.source_type == "file" and self.uri.startswith("file:///"):
            path = Path(self.uri[len("file://"):])
            return None if "CHANGE_ME" in self.uri else path
        return None


def map_source(
    node: NodeInfo,
    source_index: int = 0,
    profile: HardwareProfile = RTX_5070_PROFILE,
    warnings: list[str] | None = None,
) -> DeepStreamSource | None:
    """Convert a CvStudio input node into a DeepStreamSource config."""
    warnings = warnings if warnings is not None else []
    tag = node.node_tag
    label = f"{node.node_id}:{tag}"
    gpu = str(profile.gpu_id)
    decode = {"num-sources": "1", "gpu-id": gpu, "cudadec-memtype": "0"}

    if tag == "RTSP":
        uri = _hint(node, "uri") or node.text_setting("Input01Value")
        if not uri:
            uri = PLACEHOLDER_RTSP
            warnings.append(
                f"{label}: no RTSP URL in the export; set 'uri' in [source{source_index}]."
            )
        return DeepStreamSource(
            source_id=source_index,
            source_type="rtsp",
            node_tag=tag,
            uri=uri,
            config_entries={
                "type": "4",
                "uri": uri,
                **decode,
                "latency": "200",
                "rtsp-reconnect-interval-sec": "10",
            },
        )

    if tag == "Video":
        # CvStudio does not persist the selected file in the export; the
        # Deploy node / --source option provide it as a hint.
        path = _hint(node, "uri")
        if path:
            uri = _to_file_uri(path)
        else:
            uri = PLACEHOLDER_FILE
            warnings.append(
                f"{label}: the CvStudio export does not store the video file path; "
                f"pass --source {node.node_id}=/path/to/video.mp4 or set 'uri' in "
                f"[source{source_index}]."
            )
        return DeepStreamSource(
            source_id=source_index,
            source_type="file",
            node_tag=tag,
            uri=uri,
            loop=bool(node.setting("Input02Value", False)),
            config_entries={"type": "3", "uri": uri, **decode},
        )

    if tag in ("HLS", "YouTube"):
        uri = _hint(node, "uri") or node.text_setting("Input01Value")
        if not uri:
            uri = PLACEHOLDER_URI
            warnings.append(
                f"{label}: no URL in the export; set 'uri' in [source{source_index}]."
            )
        elif tag == "YouTube":
            warnings.append(
                f"{label}: DeepStream cannot open a YouTube page URL; replace 'uri' in "
                f"[source{source_index}] with the stream URL given by "
                f"`yt-dlp -g \"{uri}\"`."
            )
        return DeepStreamSource(
            source_id=source_index,
            source_type="uri",
            node_tag=tag,
            uri=uri,
            config_entries={"type": "2", "uri": uri, **decode},
        )

    if tag == "Webcam":
        device_no = _device_number(_hint(node, "device"))
        if device_no is None:
            device_no = 0
            warnings.append(
                f"{label}: the CvStudio export does not store the camera device; "
                f"using /dev/video0 (pass --source {node.node_id}=/dev/videoN to change)."
            )
        return DeepStreamSource(
            source_id=source_index,
            source_type="v4l2",
            node_tag=tag,
            device=f"/dev/video{device_no}",
            config_entries={
                "type": "1",
                "camera-width": "640",
                "camera-height": "480",
                "camera-fps-n": "30",
                "camera-fps-d": "1",
                "camera-v4l2-dev-node": str(device_no),
                "gpu-id": gpu,
            },
        )

    return None


def _hint(node: NodeInfo, key: str) -> str:
    value = node.hints.get(key)
    if value is None or isinstance(value, bool):
        return ""
    return str(value).strip()


def _to_file_uri(path: str) -> str:
    if "://" in path:
        return path
    return Path(path).expanduser().absolute().as_uri()


def _device_number(value: str) -> int | None:
    """'2' or '/dev/video2' -> 2."""
    digits = value.rsplit("video", 1)[-1] if value else ""
    return int(digits) if digits.isdigit() else None
