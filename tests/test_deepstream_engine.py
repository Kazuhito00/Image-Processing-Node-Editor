"""
Tests for the DeepStream Engine.

Tests the full conversion pipeline: JSON parse → pipeline build → config write.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from tools.deepstream_engine.parser import parse, CvStudioProject, NodeInfo, LinkInfo
from tools.deepstream_engine.hardware_profile import get_profile, RTX_5070_PROFILE
from tools.deepstream_engine.pipeline_builder import build_pipeline
from tools.deepstream_engine.config_writer import write_project
from tools.deepstream_engine.scaffold import generate_scaffold
from tools.deepstream_engine.engine import DeepStreamEngine, ConversionResult
from tools.deepstream_engine.mappers.source_mapper import map_source
from tools.deepstream_engine.mappers.inference_mapper import map_inference
from tools.deepstream_engine.mappers.tracker_mapper import map_tracker
from tools.deepstream_engine.mappers.sink_mapper import map_sink


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def sample_project_json():
    """A minimal CvStudio JSON that exercises all major node types."""
    return {
        "node_list": [
            "1:RTSP",
            "2:ObjectDetection",
            "3:MultiObjectTracking",
            "4:VideoWriter",
        ],
        "link_list": [
            ["1:RTSP:IMAGE:Output01", "2:ObjectDetection:IMAGE:Input01"],
            ["2:ObjectDetection:IMAGE:Output01", "3:MultiObjectTracking:IMAGE:Input01"],
            ["3:MultiObjectTracking:IMAGE:Output01", "4:VideoWriter:IMAGE:Input01"],
        ],
        "1:RTSP": {
            "id": "1",
            "name": "RTSP",
            "setting": {
                "1:RTSP:TEXT:Input01Value": "rtsp://192.168.1.100:554/stream1",
                "ver": "0.0.1",
                "pos": [100, 100],
            },
        },
        "2:ObjectDetection": {
            "id": "2",
            "name": "ObjectDetection",
            "setting": {
                "2:ObjectDetection:TEXT:Input02Value": "YOLOX-Nano(416x416)",
                "2:ObjectDetection:FLOAT:Input03Value": 0.5,
                "ver": "0.0.1",
                "pos": [300, 100],
            },
        },
        "3:MultiObjectTracking": {
            "id": "3",
            "name": "MultiObjectTracking",
            "setting": {
                "3:MultiObjectTracking:TEXT:Input02Value": "ByteTrack",
                "ver": "0.0.1",
                "pos": [500, 100],
            },
        },
        "4:VideoWriter": {
            "id": "4",
            "name": "VideoWriter",
            "setting": {
                "4:VideoWriter:TEXT:Input01Value": "output/result.mp4",
                "ver": "0.0.1",
                "pos": [700, 100],
            },
        },
    }


@pytest.fixture
def multi_source_project_json():
    """CvStudio JSON with multiple sources and secondary inference."""
    return {
        "node_list": [
            "1:RTSP",
            "2:Webcam",
            "3:ObjectDetection",
            "4:Classification",
            "5:MultiObjectTracking",
            "6:MQTT",
        ],
        "link_list": [
            ["1:RTSP:IMAGE:Output01", "3:ObjectDetection:IMAGE:Input01"],
            ["2:Webcam:IMAGE:Output01", "3:ObjectDetection:IMAGE:Input01"],
            ["3:ObjectDetection:IMAGE:Output01", "4:Classification:IMAGE:Input01"],
            ["3:ObjectDetection:IMAGE:Output01", "5:MultiObjectTracking:IMAGE:Input01"],
        ],
        "1:RTSP": {
            "id": "1",
            "name": "RTSP",
            "setting": {
                "1:RTSP:TEXT:Input01Value": "rtsp://camera1/stream",
                "ver": "0.0.1",
                "pos": [0, 0],
            },
        },
        "2:Webcam": {
            "id": "2",
            "name": "Webcam",
            "setting": {
                "2:Webcam:TEXT:Input01Value": "/dev/video0",
                "ver": "0.0.1",
                "pos": [0, 200],
            },
        },
        "3:ObjectDetection": {
            "id": "3",
            "name": "ObjectDetection",
            "setting": {
                "3:ObjectDetection:TEXT:Input02Value": "YOLO11Nano",
                "3:ObjectDetection:FLOAT:Input03Value": 0.4,
                "ver": "0.0.1",
                "pos": [300, 100],
            },
        },
        "4:Classification": {
            "id": "4",
            "name": "Classification",
            "setting": {
                "4:Classification:TEXT:Input02Value": "resnet50",
                "ver": "0.0.1",
                "pos": [500, 0],
            },
        },
        "5:MultiObjectTracking": {
            "id": "5",
            "name": "MultiObjectTracking",
            "setting": {
                "5:MultiObjectTracking:TEXT:Input02Value": "OC-SORT",
                "ver": "0.0.1",
                "pos": [500, 200],
            },
        },
        "6:MQTT": {
            "id": "6",
            "name": "MQTT",
            "setting": {
                "6:MQTT:TEXT:Input01Value": "mqtt://broker.local:1883",
                "6:MQTT:TEXT:Input02Value": "detections/stream1",
                "ver": "0.0.1",
                "pos": [700, 100],
            },
        },
    }


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ===========================================================================
# Parser Tests
# ===========================================================================


class TestParser:
    """Tests for the CvStudio JSON parser."""

    def test_parse_basic_project(self, sample_project_json, tmp_dir):
        json_path = tmp_dir / "test.json"
        json_path.write_text(json.dumps(sample_project_json))

        project = parse(json_path)

        assert len(project.nodes) == 4
        assert len(project.links) == 3
        assert project.nodes[1].node_tag == "RTSP"
        assert project.nodes[2].node_tag == "ObjectDetection"

    def test_parse_links(self, sample_project_json, tmp_dir):
        json_path = tmp_dir / "test.json"
        json_path.write_text(json.dumps(sample_project_json))

        project = parse(json_path)

        assert project.links[0].source_node_id == 1
        assert project.links[0].target_node_id == 2
        assert project.links[0].source_node_tag == "RTSP"
        assert project.links[0].target_node_tag == "ObjectDetection"

    def test_nodes_by_tag(self, sample_project_json, tmp_dir):
        json_path = tmp_dir / "test.json"
        json_path.write_text(json.dumps(sample_project_json))

        project = parse(json_path)

        rtsp_nodes = project.nodes_by_tag("RTSP")
        assert len(rtsp_nodes) == 1
        assert rtsp_nodes[0].node_id == 1

    def test_get_downstream(self, sample_project_json, tmp_dir):
        json_path = tmp_dir / "test.json"
        json_path.write_text(json.dumps(sample_project_json))

        project = parse(json_path)

        downstream = project.get_downstream(1)
        assert len(downstream) == 1
        assert downstream[0].node_tag == "ObjectDetection"

    def test_get_upstream(self, sample_project_json, tmp_dir):
        json_path = tmp_dir / "test.json"
        json_path.write_text(json.dumps(sample_project_json))

        project = parse(json_path)

        upstream = project.get_upstream(2)
        assert len(upstream) == 1
        assert upstream[0].node_tag == "RTSP"


# ===========================================================================
# Mapper Tests
# ===========================================================================


class TestSourceMapper:
    """Tests for source node mapping."""

    def test_map_rtsp_source(self):
        node = NodeInfo(
            node_id=1,
            node_tag="RTSP",
            settings={"1:RTSP:TEXT:Input01Value": "rtsp://host/stream"},
        )
        source = map_source(node, 0)
        assert source is not None
        assert source.source_type == "rtsp"
        assert source.uri == "rtsp://host/stream"
        assert source.config_entries["type"] == "4"

    def test_map_webcam_source(self):
        node = NodeInfo(
            node_id=1,
            node_tag="Webcam",
            settings={"1:Webcam:TEXT:Input01Value": "/dev/video1"},
        )
        source = map_source(node, 0)
        assert source is not None
        assert source.source_type == "v4l2"
        assert source.config_entries["type"] == "1"

    def test_map_video_source(self):
        node = NodeInfo(
            node_id=1,
            node_tag="Video",
            settings={"1:Video:TEXT:Input01Value": "/path/to/video.mp4"},
        )
        source = map_source(node, 0)
        assert source is not None
        assert source.source_type == "file"
        assert "file://" in source.config_entries["uri"]

    def test_unsupported_source(self):
        node = NodeInfo(node_id=1, node_tag="Unknown", settings={})
        source = map_source(node, 0)
        assert source is None


class TestInferenceMapper:
    """Tests for inference node mapping."""

    def test_map_object_detection_yolox(self):
        node = NodeInfo(
            node_id=2,
            node_tag="ObjectDetection",
            settings={
                "2:ObjectDetection:TEXT:Input02Value": "YOLOX-Nano(416x416)",
                "2:ObjectDetection:FLOAT:Input03Value": 0.5,
            },
        )
        config = map_inference(node, 0, True)
        assert config is not None
        assert config.gie_type == "primary"
        assert config.input_width == 416
        assert config.input_height == 416
        assert config.output_format == "yolox"
        assert config.score_threshold == 0.5

    def test_map_object_detection_yolo11(self):
        node = NodeInfo(
            node_id=2,
            node_tag="ObjectDetection",
            settings={
                "2:ObjectDetection:TEXT:Input02Value": "YOLO11Nano",
                "2:ObjectDetection:FLOAT:Input03Value": 0.4,
            },
        )
        config = map_inference(node, 0, True)
        assert config is not None
        assert config.output_format == "yolo11"
        assert config.input_width == 608  # YOLO11Nano is exported at 608x416

    def test_map_classification_sgie(self):
        node = NodeInfo(
            node_id=3,
            node_tag="Classification",
            settings={"3:Classification:TEXT:Input02Value": "resnet50"},
        )
        config = map_inference(node, 1, False)
        assert config is not None
        assert config.gie_type == "secondary"
        assert config.network_type == 1

    def test_map_segmentation(self):
        node = NodeInfo(
            node_id=3,
            node_tag="SemanticSegmentation",
            settings={},
        )
        # Segmentation has no DeepStream post-processing in this engine.
        config = map_inference(node, 0, True)
        assert config is None


class TestTrackerMapper:
    """Tests for tracker node mapping."""

    def test_map_bytetrack(self):
        node = NodeInfo(
            node_id=3,
            node_tag="MultiObjectTracking",
            settings={"3:MOT:TEXT:Input02Value": "ByteTrack"},
        )
        config = map_tracker(node)
        assert config is not None
        assert config.tracker_name == "ByteTrack"

    def test_map_ocsort(self):
        node = NodeInfo(
            node_id=3,
            node_tag="MultiObjectTracking",
            settings={"3:MOT:TEXT:Input02Value": "OC-SORT"},
        )
        config = map_tracker(node)
        assert config is not None
        assert config.tracker_name == "OC-SORT"


class TestSinkMapper:
    """Tests for sink node mapping."""

    def test_map_video_writer(self):
        node = NodeInfo(
            node_id=4,
            node_tag="VideoWriter",
            settings={"4:VW:TEXT:Input01Value": "output/out.mp4"},
        )
        sink = map_sink(node, 0)
        assert sink is not None
        assert sink.sink_type == "file"
        assert sink.config_entries["codec"] == "1"

    def test_map_mqtt_sink(self):
        node = NodeInfo(
            node_id=5,
            node_tag="MQTT",
            settings={
                "5:MQTT:TEXT:Input01Value": "mqtt://broker:1883",
                "5:MQTT:TEXT:Input02Value": "test/topic",
            },
        )
        # The MQTT node stores no broker settings, so it cannot be converted.
        sink = map_sink(node, 0)
        assert sink is None


# ===========================================================================
# Pipeline Builder Tests
# ===========================================================================


class TestPipelineBuilder:
    """Tests for the pipeline builder."""

    def test_build_basic_pipeline(self, sample_project_json, tmp_dir):
        json_path = tmp_dir / "test.json"
        json_path.write_text(json.dumps(sample_project_json))

        project = parse(json_path)
        pipeline = build_pipeline(project)

        assert len(pipeline.sources) == 1
        assert pipeline.primary_gie is not None
        assert pipeline.primary_gie.model_name == "YOLOX-Nano(416x416)"
        assert pipeline.tracker is not None
        assert len(pipeline.sinks) == 1
        assert pipeline.osd_enabled is True

    def test_build_multi_source_pipeline(self, multi_source_project_json, tmp_dir):
        json_path = tmp_dir / "test.json"
        json_path.write_text(json.dumps(multi_source_project_json))

        project = parse(json_path)
        pipeline = build_pipeline(project)

        assert len(pipeline.sources) == 2
        assert pipeline.primary_gie is not None
        assert len(pipeline.secondary_gies) == 1
        assert pipeline.tracker is not None
        assert pipeline.streammux_config["batch-size"] == "2"

    def test_pipeline_hardware_profile(self, sample_project_json, tmp_dir):
        json_path = tmp_dir / "test.json"
        json_path.write_text(json.dumps(sample_project_json))

        project = parse(json_path)
        profile = get_profile("RTX_5070_HIGH_BATCH")
        pipeline = build_pipeline(project, profile)

        assert pipeline.profile.max_batch_size == 16
        assert pipeline.profile.tensorrt_workspace_mb == 6144


# ===========================================================================
# Config Writer Tests
# ===========================================================================


class TestConfigWriter:
    """Tests for configuration file generation."""

    def test_write_project_files(self, sample_project_json, tmp_dir):
        json_path = tmp_dir / "input.json"
        json_path.write_text(json.dumps(sample_project_json))

        project = parse(json_path)
        pipeline = build_pipeline(project)

        output = tmp_dir / "output"
        files = write_project(pipeline, output)

        assert "deepstream_app_config.txt" in files
        assert "config_pgie.txt" in files
        assert (output / "deepstream_app_config.txt").exists()
        assert (output / "config_pgie.txt").exists()

    def test_main_config_content(self, sample_project_json, tmp_dir):
        json_path = tmp_dir / "input.json"
        json_path.write_text(json.dumps(sample_project_json))

        project = parse(json_path)
        pipeline = build_pipeline(project)

        output = tmp_dir / "output"
        write_project(pipeline, output)

        content = (output / "deepstream_app_config.txt").read_text()
        assert "[application]" in content
        assert "[source0]" in content
        assert "[primary-gie]" in content
        assert "[tracker]" in content
        assert "[osd]" in content
        assert "[sink0]" in content


# ===========================================================================
# Scaffold Tests
# ===========================================================================


class TestScaffold:
    """Tests for project scaffolding generation."""

    def test_generate_all_scaffold_files(self, sample_project_json, tmp_dir):
        json_path = tmp_dir / "input.json"
        json_path.write_text(json.dumps(sample_project_json))

        project = parse(json_path)
        pipeline = build_pipeline(project)

        output = tmp_dir / "output"
        output.mkdir()
        files = generate_scaffold(pipeline, output)

        assert "Dockerfile" in files
        assert "docker-compose.yml" in files
        assert "Makefile" in files
        assert "run.sh" in files
        assert "README.md" in files
        assert ".gitignore" in files
        assert "nvdsinfer_custom_impl_cvstudio/nvdsparsebbox_cvstudio.cpp" in files

    def test_dockerfile_mentions_target(self, sample_project_json, tmp_dir):
        json_path = tmp_dir / "input.json"
        json_path.write_text(json.dumps(sample_project_json))

        project = parse(json_path)
        pipeline = build_pipeline(project)

        output = tmp_dir / "output"
        output.mkdir()
        generate_scaffold(pipeline, output)

        content = (output / "Dockerfile").read_text()
        assert "RTX_5070" in content
        assert "16GB VRAM" in content
        assert "deepstream:7.1" in content

    def test_run_script_executable(self, sample_project_json, tmp_dir):
        json_path = tmp_dir / "input.json"
        json_path.write_text(json.dumps(sample_project_json))

        project = parse(json_path)
        pipeline = build_pipeline(project)

        output = tmp_dir / "output"
        output.mkdir()
        generate_scaffold(pipeline, output)

        run_sh = output / "run.sh"
        assert run_sh.exists()
        assert os.access(run_sh, os.X_OK)


# ===========================================================================
# Full Engine Tests
# ===========================================================================


class TestDeepStreamEngine:
    """End-to-end tests for the DeepStream Engine."""

    def test_full_conversion(self, sample_project_json, tmp_dir):
        json_path = tmp_dir / "input.json"
        json_path.write_text(json.dumps(sample_project_json))

        engine = DeepStreamEngine(profile_name="RTX_5070")
        result = engine.convert(json_path, tmp_dir / "project")

        assert isinstance(result, ConversionResult)
        assert result.nodes_mapped == 4
        assert result.links_mapped == 3
        assert len(result.all_files) > 0
        assert (tmp_dir / "project" / "deepstream_app_config.txt").exists()
        assert (tmp_dir / "project" / "Dockerfile").exists()
        assert (tmp_dir / "project" / "models").is_dir()

    def test_conversion_overwrite(self, sample_project_json, tmp_dir):
        json_path = tmp_dir / "input.json"
        json_path.write_text(json.dumps(sample_project_json))

        engine = DeepStreamEngine()
        engine.convert(json_path, tmp_dir / "project")

        # Second conversion should fail without overwrite
        with pytest.raises(FileExistsError):
            engine.convert(json_path, tmp_dir / "project")

        # Should succeed with overwrite
        result = engine.convert(json_path, tmp_dir / "project", overwrite=True)
        assert result.nodes_mapped == 4

    def test_conversion_file_not_found(self, tmp_dir):
        engine = DeepStreamEngine()
        with pytest.raises(FileNotFoundError):
            engine.convert(tmp_dir / "nonexistent.json", tmp_dir / "output")

    def test_conversion_summary(self, sample_project_json, tmp_dir):
        json_path = tmp_dir / "input.json"
        json_path.write_text(json.dumps(sample_project_json))

        engine = DeepStreamEngine()
        result = engine.convert(json_path, tmp_dir / "project")

        summary = result.summary()
        assert "DeepStream Project" in summary
        assert "RTX_5070" in summary
        assert "YOLOX-Nano" in summary
        assert "Sources: 1" in summary

    def test_multi_source_conversion(self, multi_source_project_json, tmp_dir):
        json_path = tmp_dir / "input.json"
        json_path.write_text(json.dumps(multi_source_project_json))

        engine = DeepStreamEngine()
        result = engine.convert(json_path, tmp_dir / "project")

        assert result.pipeline.streammux_config["batch-size"] == "2"
        assert len(result.pipeline.sources) == 2
        assert result.pipeline.tracker is not None

    def test_convert_from_dict(self, sample_project_json, tmp_dir):
        engine = DeepStreamEngine()
        result = engine.convert_from_dict(
            sample_project_json,
            tmp_dir / "project",
            project_name="test_app",
        )
        assert result.project_name == "test_app"
        assert result.nodes_mapped == 4


# ===========================================================================
# Hardware Profile Tests
# ===========================================================================


class TestHardwareProfile:
    """Tests for hardware profiles."""

    def test_default_profile(self):
        profile = get_profile("RTX_5070")
        assert profile.vram_gb == 16
        assert profile.system_ram_gb == 32
        assert profile.fp16_enabled is True
        assert profile.int8_enabled is False
        assert profile.max_batch_size == 8

    def test_high_batch_profile(self):
        profile = get_profile("RTX_5070_HIGH_BATCH")
        assert profile.max_batch_size == 16
        assert profile.tensorrt_workspace_mb == 6144

    def test_int8_profile(self):
        profile = get_profile("RTX_5070_INT8")
        assert profile.int8_enabled is True
