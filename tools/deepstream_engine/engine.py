"""
DeepStream Engine – Main orchestrator.

Converts a CvStudio JSON save file into a complete NVIDIA DeepStream project
(deepstream-app configuration, nvinfer configurations, bbox parser library,
Docker deployment files) for the target hardware.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from tools.deepstream_engine.config_writer import MAIN_CONFIG, write_project
from tools.deepstream_engine.hardware_profile import HardwareProfile, get_profile
from tools.deepstream_engine.parser import CvStudioProject, parse_dict
from tools.deepstream_engine.pipeline_builder import DeepStreamPipeline, build_pipeline
from tools.deepstream_engine.scaffold import generate_scaffold


class DeepStreamEngine:
    """
    Main engine for converting CvStudio projects to DeepStream.

    Usage:
        engine = DeepStreamEngine()
        result = engine.convert("my_project.json", "output/my_deepstream_project")
    """

    def __init__(self, profile_name: str = "RTX_5070"):
        self.profile: HardwareProfile = get_profile(profile_name)

    def convert(
        self,
        input_json: str | Path,
        output_dir: str | Path,
        project_name: str | None = None,
        overwrite: bool = False,
        sources: dict[int, str] | None = None,
        display: bool = False,
    ) -> ConversionResult:
        """
        Convert a CvStudio JSON save file to a DeepStream project.

        Args:
            input_json: Path to the CvStudio JSON save file.
            output_dir: Directory where the DeepStream project will be created.
            project_name: Optional project name (defaults to JSON filename).
            overwrite: If True, overwrite existing output directory.
            sources: Optional {node id: URI / file path / camera device} for
                input nodes; CvStudio does not save the file of a Video node
                nor the device of a Webcam node in its export.
            display: Also open an on-screen window (needs an X display).

        Returns:
            ConversionResult with details about the generated project.
        """
        input_json = Path(input_json)
        if not input_json.exists():
            raise FileNotFoundError(f"Input file not found: {input_json}")
        try:
            with open(input_json, "r", encoding="utf-8") as fp:
                data = json.load(fp)
        except ValueError as exc:
            raise ValueError(f"{input_json} is not valid JSON: {exc}") from exc

        if project_name is None:
            project_name = input_json.stem.replace(" ", "_")
        return self.convert_from_dict(
            data, output_dir, project_name, overwrite, sources, display
        )

    def convert_from_dict(
        self,
        data: dict,
        output_dir: str | Path,
        project_name: str = "cvstudio_deepstream",
        overwrite: bool = False,
        sources: dict[int, str] | None = None,
        display: bool = False,
    ) -> ConversionResult:
        """Convert a CvStudio project from an already-loaded export dict."""
        output_dir = Path(output_dir)

        # Validate everything before touching the output directory.
        project = parse_dict(data)
        self._apply_source_overrides(project, sources or {})
        pipeline = build_pipeline(project, self.profile, project_name, display)

        if output_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Output directory already exists: {output_dir}. "
                    "Use overwrite=True to replace it."
                )
            # Only ever delete a directory this engine generated.
            if not output_dir.is_dir() or (
                any(output_dir.iterdir()) and not (output_dir / MAIN_CONFIG).is_file()
            ):
                raise FileExistsError(
                    f"Refusing to overwrite {output_dir}: it is not a generated "
                    "DeepStream project."
                )
            shutil.rmtree(output_dir)

        config_files = write_project(pipeline, output_dir)
        scaffold_files = generate_scaffold(pipeline, output_dir)
        models_copied = self._copy_models(pipeline, output_dir)

        return ConversionResult(
            project_name=project_name,
            output_dir=str(output_dir),
            pipeline=pipeline,
            config_files=config_files,
            scaffold_files=scaffold_files,
            models_copied=models_copied,
            nodes_mapped=len(project.nodes) - len(pipeline.ignored_nodes),
            links_mapped=len(project.links),
        )

    @staticmethod
    def _apply_source_overrides(project: CvStudioProject, sources: dict[int, str]) -> None:
        for node_id, value in sources.items():
            node = project.nodes.get(int(node_id))
            if node is None:
                raise ValueError(f"Source override: the project has no node with id {node_id}")
            node.hints["uri"] = value
            node.hints["device"] = value

    @staticmethod
    def _copy_models(pipeline: DeepStreamPipeline, output_dir: Path) -> list[str]:
        """Copy the ONNX models found on this machine into the project."""
        copied = []
        for gie in pipeline.gies:
            if not gie.source_model_path:
                continue
            dest = output_dir / gie.onnx_path
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(gie.source_model_path, dest)
            copied.append(gie.onnx_path)
        return copied


class ConversionResult:
    """Result of a CvStudio → DeepStream conversion."""

    def __init__(
        self,
        project_name: str,
        output_dir: str,
        pipeline: DeepStreamPipeline,
        config_files: list[str],
        scaffold_files: list[str],
        models_copied: list[str],
        nodes_mapped: int,
        links_mapped: int,
    ):
        self.project_name = project_name
        self.output_dir = output_dir
        self.pipeline = pipeline
        self.config_files = config_files
        self.scaffold_files = scaffold_files
        self.models_copied = models_copied
        self.nodes_mapped = nodes_mapped
        self.links_mapped = links_mapped

    @property
    def all_files(self) -> list[str]:
        """All generated files."""
        return self.config_files + self.scaffold_files

    @property
    def warnings(self) -> list[str]:
        """Points to review before running the generated project."""
        return self.pipeline.warnings

    @property
    def missing_models(self) -> list[str]:
        """Project-relative ONNX paths the user still has to provide."""
        return [g.onnx_path for g in self.pipeline.gies if not g.source_model_path]

    def summary(self) -> str:
        """Human-readable summary of the conversion."""
        pipeline = self.pipeline
        pgie = pipeline.primary_gie
        lines = [
            f"═══ DeepStream Project: {self.project_name} ═══",
            f"Output: {self.output_dir}",
            "",
            "Pipeline Summary:",
            f"  Sources: {len(pipeline.sources)}",
            f"  Primary GIE: {pgie.model_name if pgie else 'None'}",
            f"  Secondary GIEs: {len(pipeline.secondary_gies)}",
            f"  Tracker: {pipeline.tracker.algorithm if pipeline.tracker else 'None'}",
            f"  Sinks: {', '.join(s.sink_type for s in pipeline.sinks)}",
            "",
            f"Nodes converted: {self.nodes_mapped}",
            f"Generated files ({len(self.all_files)}):",
        ]
        lines.extend(f"  • {f}" for f in sorted(self.all_files))

        if self.models_copied:
            lines.append("")
            lines.append(f"Models copied ({len(self.models_copied)}):")
            lines.extend(f"  • {m}" for m in self.models_copied)

        if pipeline.ignored_nodes:
            lines.append("")
            lines.append(
                f"Nodes without a DeepStream equivalent ({len(pipeline.ignored_nodes)}), "
                f"left out: {', '.join(pipeline.ignored_nodes)}"
            )

        if self.warnings:
            lines.append("")
            lines.append(f"Warnings ({len(self.warnings)}):")
            lines.extend(f"  ! {w}" for w in self.warnings)

        lines.append("")
        lines.append(
            f"Target: {pipeline.profile.name} "
            f"({pipeline.profile.vram_gb}GB VRAM, {pipeline.profile.system_ram_gb}GB RAM)"
        )
        lines.append("")
        lines.append("Next steps:")
        step = 1
        for model in self.missing_models:
            lines.append(f"  {step}. Copy the missing model to {model}")
            step += 1
        lines.append(f"  {step}. cd {self.output_dir} && make run")

        return "\n".join(lines)
