"""
DeepStream pipeline builder.

Assembles a complete deepstream-app pipeline from the mapped elements:
  sources → streammux → PGIE → tracker → SGIE(s) → tiler → OSD → sinks
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tools.deepstream_engine.hardware_profile import HardwareProfile, RTX_5070_PROFILE
from tools.deepstream_engine.mappers.inference_mapper import (
    INFERENCE_NODE_TAGS,
    UNSUPPORTED_INFERENCE_TAGS,
    InferenceConfig,
    map_inference,
)
from tools.deepstream_engine.mappers.sink_mapper import (
    SINK_NODE_TAGS,
    SinkConfig,
    map_display_sink,
    map_rtsp_sink,
    map_sink,
)
from tools.deepstream_engine.mappers.source_mapper import (
    SOURCE_NODE_TAGS,
    UNSUPPORTED_SOURCE_TAGS,
    DeepStreamSource,
    map_source,
)
from tools.deepstream_engine.mappers.tracker_mapper import (
    TRACKER_NODE_TAGS,
    TrackerConfig,
    map_tracker,
)
from tools.deepstream_engine.parser import CvStudioProject

# Editor-only nodes: nothing to convert, nothing to report.
_SILENT_NODE_TAGS = {"Deploy", "Settings", "SystemSizing", "SystemResource", "OnvifScan"}


@dataclass
class DeepStreamPipeline:
    """Complete DeepStream pipeline configuration."""

    name: str = "cvstudio_deepstream"
    sources: list[DeepStreamSource] = field(default_factory=list)
    primary_gie: InferenceConfig | None = None
    secondary_gies: list[InferenceConfig] = field(default_factory=list)
    tracker: TrackerConfig | None = None
    sinks: list[SinkConfig] = field(default_factory=list)
    osd_enabled: bool = True
    file_loop: bool = False
    streammux_config: dict[str, str] = field(default_factory=dict)
    tiler_config: dict[str, str] = field(default_factory=dict)
    profile: HardwareProfile = field(default_factory=lambda: RTX_5070_PROFILE)
    # Conversion report
    warnings: list[str] = field(default_factory=list)
    ignored_nodes: list[str] = field(default_factory=list)

    @property
    def gies(self) -> list[InferenceConfig]:
        primary = [self.primary_gie] if self.primary_gie else []
        return primary + self.secondary_gies

    @property
    def uses_custom_parser(self) -> bool:
        return any(gie.uses_custom_parser for gie in self.gies)

    @property
    def is_live(self) -> bool:
        return any(source.is_live for source in self.sources)


def build_pipeline(
    project: CvStudioProject,
    profile: HardwareProfile = RTX_5070_PROFILE,
    project_name: str = "cvstudio_deepstream",
    display: bool = False,
) -> DeepStreamPipeline:
    """Analyze a CvStudio project and build the corresponding DeepStream pipeline."""
    pipeline = DeepStreamPipeline(name=project_name, profile=profile)
    warnings = pipeline.warnings
    nodes = [project.nodes[node_id] for node_id in sorted(project.nodes)]

    # --- 1. Sources ---
    for node in nodes:
        if node.node_tag in SOURCE_NODE_TAGS:
            source = map_source(node, len(pipeline.sources), profile, warnings)
            if source:
                pipeline.sources.append(source)
        elif node.node_tag in UNSUPPORTED_SOURCE_TAGS:
            warnings.append(
                f"{node.node_id}:{node.node_tag}: this input has no DeepStream source "
                f"equivalent; node ignored."
            )
    if not pipeline.sources:
        raise ValueError(
            "The project has no video source DeepStream can read "
            f"(supported input nodes: {', '.join(sorted(SOURCE_NODE_TAGS))})."
        )
    num_sources = len(pipeline.sources)
    if num_sources > profile.max_simultaneous_streams:
        warnings.append(
            f"{num_sources} sources exceed the {profile.max_simultaneous_streams} "
            f"simultaneous streams the {profile.name} profile is sized for."
        )
    pipeline.file_loop = any(source.loop for source in pipeline.sources)

    # --- 2. Inference: one detector as PGIE, classifiers as SGIEs ---
    for node in nodes:
        if node.node_tag in UNSUPPORTED_INFERENCE_TAGS:
            warnings.append(
                f"{node.node_id}:{node.node_tag}: the post-processing of this model "
                f"type is not available in DeepStream; node ignored."
            )
    for node in nodes:
        if node.node_tag != "ObjectDetection":
            continue
        if pipeline.primary_gie:
            warnings.append(
                f"{node.node_id}:{node.node_tag}: deepstream-app runs a single primary "
                f"detector; additional detector ignored."
            )
            continue
        pipeline.primary_gie = map_inference(
            node, 0, True, profile, num_sources, warnings
        )
    for node in nodes:
        if node.node_tag != "Classification":
            continue
        gie_index = len(pipeline.gies)
        if pipeline.primary_gie is None:
            pipeline.primary_gie = map_inference(
                node, gie_index, True, profile, num_sources, warnings
            )
        elif pipeline.primary_gie.network_type == 0:
            pipeline.secondary_gies.append(
                map_inference(node, gie_index, False, profile, num_sources, warnings)
            )
        else:
            warnings.append(
                f"{node.node_id}:{node.node_tag}: only one full-frame classifier can "
                f"run without a detector; additional classifier ignored."
            )

    # --- 3. Tracker (needs detections to track) ---
    for node in nodes:
        if node.node_tag not in TRACKER_NODE_TAGS:
            continue
        if pipeline.tracker:
            warnings.append(
                f"{node.node_id}:{node.node_tag}: a DeepStream pipeline has a single "
                f"tracker; additional tracker ignored."
            )
        elif pipeline.primary_gie is None or pipeline.primary_gie.network_type != 0:
            warnings.append(
                f"{node.node_id}:{node.node_tag}: no object detector in the pipeline; "
                f"tracker skipped."
            )
        else:
            pipeline.tracker = map_tracker(node, profile, warnings)

    # --- 4. Sinks ---
    live = pipeline.is_live
    for node in nodes:
        if node.node_tag in SINK_NODE_TAGS:
            sink = map_sink(node, len(pipeline.sinks), profile, live, warnings)
            if sink:
                pipeline.sinks.append(sink)
    if not pipeline.sinks:
        # Keep the annotated video reachable on a headless machine.
        pipeline.sinks.append(map_rtsp_sink(0, profile, live))
    if display:
        pipeline.sinks.append(map_display_sink(len(pipeline.sinks), profile, live))

    # --- 5. Streammux ---
    pipeline.streammux_config = {
        "gpu-id": str(profile.gpu_id),
        "live-source": "1" if live else "0",
        "batch-size": str(num_sources),
        "batched-push-timeout": "40000",
        "width": "1920",
        "height": "1080",
        "enable-padding": "0",
        "nvbuf-memory-type": str(profile.nvbuf_memory_type),
    }

    # --- 6. Tiler (composites all streams into one frame for the sinks) ---
    if num_sources > 1:
        cols = _compute_tiler_cols(num_sources)
        rows = (num_sources + cols - 1) // cols
        pipeline.tiler_config = {
            "rows": str(rows),
            "columns": str(cols),
            "width": "1920",
            "height": "1080",
            "gpu-id": str(profile.gpu_id),
            "nvbuf-memory-type": str(profile.nvbuf_memory_type),
        }

    # --- 7. OSD ---
    pipeline.osd_enabled = bool(pipeline.gies)

    # --- 8. Everything else has no DeepStream counterpart ---
    handled = (
        SOURCE_NODE_TAGS
        | UNSUPPORTED_SOURCE_TAGS
        | INFERENCE_NODE_TAGS
        | UNSUPPORTED_INFERENCE_TAGS
        | TRACKER_NODE_TAGS
        | SINK_NODE_TAGS
        | _SILENT_NODE_TAGS
    )
    pipeline.ignored_nodes = [
        f"{node.node_id}:{node.node_tag}" for node in nodes if node.node_tag not in handled
    ]

    return pipeline


def _compute_tiler_cols(num_sources: int) -> int:
    """Number of tiler columns for a near-square grid."""
    if num_sources <= 2:
        return num_sources
    if num_sources <= 4:
        return 2
    if num_sources <= 9:
        return 3
    return 4
