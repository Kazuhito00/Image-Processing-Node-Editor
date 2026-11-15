"""
Sink element mapper.

Maps CvStudio output nodes to deepstream-app [sinkN] groups:
  - VideoWriter / VideoRecorder → type=3 (MP4 file, H.264)
  - RTMPOutput / Streaming      → type=4 (RTSP server; DeepStream has no RTMP sink)
  - no output node              → type=4 so the annotated video stays viewable
                                  on a headless machine
  - display requested           → type=2 (EGL window)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tools.deepstream_engine.hardware_profile import HardwareProfile, RTX_5070_PROFILE
from tools.deepstream_engine.parser import NodeInfo

FILE_SINK_TAGS = {"VideoWriter", "VideoRecorder"}
STREAM_SINK_TAGS = {"RTMPOutput", "Streaming"}
SINK_NODE_TAGS = FILE_SINK_TAGS | STREAM_SINK_TAGS

RTSP_PORT = 8554
RTSP_MOUNT = "/ds-test"
_UDP_PORT = 5400


@dataclass
class SinkConfig:
    """deepstream-app sink configuration."""

    sink_id: int
    sink_type: str  # "file", "rtsp", "egl", "fake"
    output_path: str = ""
    rtsp_port: int = 0
    config_entries: dict[str, str] = field(default_factory=dict)


def map_sink(
    node: NodeInfo,
    sink_index: int = 0,
    profile: HardwareProfile = RTX_5070_PROFILE,
    live: bool = True,
    warnings: list[str] | None = None,
) -> SinkConfig | None:
    """Convert a CvStudio output node into a DeepStream sink config."""
    warnings = warnings if warnings is not None else []
    if node.node_tag in FILE_SINK_TAGS:
        return map_file_sink(f"output/output_{node.node_id}.mp4", sink_index, profile)
    if node.node_tag in STREAM_SINK_TAGS:
        sink = map_rtsp_sink(sink_index, profile, live)
        warnings.append(
            f"{node.node_id}:{node.node_tag}: DeepStream has no RTMP sink; the stream "
            f"is served over RTSP at rtsp://<host>:{sink.rtsp_port}{RTSP_MOUNT} instead."
        )
        return sink
    return None


def map_file_sink(
    output_path: str,
    sink_index: int = 0,
    profile: HardwareProfile = RTX_5070_PROFILE,
) -> SinkConfig:
    """MP4 file sink with H.264 hardware encoding."""
    return SinkConfig(
        sink_id=sink_index,
        sink_type="file",
        output_path=output_path,
        config_entries={
            "type": "3",
            "container": "1",  # MP4
            "codec": "1",  # H.264
            "enc-type": "0",  # hardware encoder
            "sync": "0",
            "bitrate": "8000000",
            "output-file": output_path,
            "source-id": "0",
            "gpu-id": str(profile.gpu_id),
            "nvbuf-memory-type": str(profile.nvbuf_memory_type),
        },
    )


def map_rtsp_sink(
    sink_index: int = 0,
    profile: HardwareProfile = RTX_5070_PROFILE,
    live: bool = True,
) -> SinkConfig:
    """RTSP server sink; each sink gets its own RTSP and UDP port."""
    rtsp_port = RTSP_PORT + sink_index
    return SinkConfig(
        sink_id=sink_index,
        sink_type="rtsp",
        rtsp_port=rtsp_port,
        config_entries={
            "type": "4",
            "codec": "1",
            "enc-type": "0",
            "sync": "0" if live else "1",
            "bitrate": "4000000",
            "rtsp-port": str(rtsp_port),
            "udp-port": str(_UDP_PORT + sink_index),
            "source-id": "0",
            "gpu-id": str(profile.gpu_id),
            "nvbuf-memory-type": str(profile.nvbuf_memory_type),
        },
    )


def map_display_sink(
    sink_index: int = 0,
    profile: HardwareProfile = RTX_5070_PROFILE,
    live: bool = True,
) -> SinkConfig:
    """On-screen EGL window (needs an X display)."""
    return SinkConfig(
        sink_id=sink_index,
        sink_type="egl",
        config_entries={
            "type": "2",
            "sync": "0" if live else "1",
            "source-id": "0",
            "gpu-id": str(profile.gpu_id),
            "nvbuf-memory-type": str(profile.nvbuf_memory_type),
        },
    )


def map_fake_sink(sink_index: int = 0) -> SinkConfig:
    """Fake sink (metadata-only processing, nothing rendered)."""
    return SinkConfig(
        sink_id=sink_index,
        sink_type="fake",
        config_entries={"type": "1", "sync": "0", "source-id": "0"},
    )
