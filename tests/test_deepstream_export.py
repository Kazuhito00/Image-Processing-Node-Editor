"""
Tests for the CvStudio → DeepStream export on real CvStudio exports.

Uses the key format CvStudio actually writes ("1:RTSP:Text:Input01Value") and
checks that the generated project is one deepstream-app can run, including
the bbox parser library, which is compiled and executed.
"""

import configparser
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tools.deepstream_engine.cli import main as cli_main
from tools.deepstream_engine.engine import DeepStreamEngine
from tools.deepstream_engine.templates.bbox_parser import PARSER_SOURCE

HARNESS_DIR = Path(__file__).parent / "deepstream_parser_harness"


def _node(node_id, tag, **settings):
    return {
        "id": str(node_id),
        "name": tag,
        "setting": {
            "ver": "0.0.1",
            "pos": [0, 0],
            **{f"{node_id}:{tag}:{key}": value for key, value in settings.items()},
        },
    }


def _export(*nodes, hints=None):
    """Build a dict shaped like CvStudio's File > Export output."""
    data = {"node_list": [], "link_list": []}
    for node in nodes:
        key = f"{node['id']}:{node['name']}"
        data["node_list"].append(key)
        data[key] = node
    if hints:
        data["deepstream_hints"] = hints
    return data


def _rtsp(node_id=1, url="rtsp://10.0.0.5:554/cam"):
    return _node(node_id, "RTSP", **{"Text:Input01Value": url})


def _detector(node_id=2, model="YOLOX-Nano(416x416)", score=0.35, rejected=""):
    return _node(
        node_id,
        "ObjectDetection",
        **{
            "Text:Input02Value": model,
            "Float:Input03Value": score,
            "RejectedClassesValue": rejected,
        },
    )


def _tracker(node_id=3, name="ByteTrack", enabled=True):
    return _node(
        node_id,
        "MultiObjectTracking",
        **{"Text:Input02Value": name, "EnableCheckbox": enabled},
    )


def _read_config(path):
    config = configparser.ConfigParser(interpolation=None)
    config.read(path, encoding="utf-8")
    return config


def _convert(tmp_path, data, **kwargs):
    out = tmp_path / "project"
    result = DeepStreamEngine().convert_from_dict(data, out, "demo", **kwargs)
    return result, out


class TestGeneratedProject:
    def test_detector_tracker_pipeline(self, tmp_path):
        data = _export(
            _rtsp(),
            _detector(rejected="2: car, 7: truck"),
            _tracker(name="ByteTrack"),
            _node(4, "VideoWriter", WriteMode="Standard"),
        )
        result, out = _convert(tmp_path, data)

        app = _read_config(out / "deepstream_app_config.txt")
        assert app["source0"]["type"] == "4"
        assert app["source0"]["uri"] == "rtsp://10.0.0.5:554/cam"
        assert app["streammux"]["live-source"] == "1"
        assert app["primary-gie"]["config-file"] == "config_pgie.txt"
        assert app["tracker"]["ll-config-file"].endswith("config_tracker_NvSORT.yml")
        assert app["sink0"]["type"] == "3"
        assert app["tiled-display"]["enable"] == "0"

        pgie = _read_config(out / "config_pgie.txt")["property"]
        # YOLOX: BGR, raw pixels, top-left letterbox, grid decoding parser.
        assert pgie["parse-bbox-func-name"] == "NvDsInferParseCvStudioYoloX"
        assert pgie["model-color-format"] == "1"
        assert pgie["net-scale-factor"] == "1.0"
        assert pgie["maintain-aspect-ratio"] == "1"
        assert pgie["symmetric-padding"] == "0"
        assert pgie["network-mode"] == "2"  # FP16
        assert pgie["num-detected-classes"] == "80"
        assert pgie["filter-out-class-ids"] == "2;7"
        thresholds = _read_config(out / "config_pgie.txt")["class-attrs-all"]
        assert thresholds["pre-cluster-threshold"] == "0.35"

        # Everything the configuration points to exists in the project.
        assert (out / pgie["onnx-file"]).is_file()
        assert (out / Path(pgie["custom-lib-path"]).parent / "Makefile").is_file()
        labels = (out / pgie["labelfile-path"]).read_text().splitlines()
        assert len(labels) == 80 and labels[0] == "person"
        assert result.missing_models == []

    def test_ultralytics_model_uses_rgb_normalised_input(self, tmp_path):
        data = _export(_rtsp(), _detector(model="YOLO11Nano"))
        _, out = _convert(tmp_path, data)

        pgie = _read_config(out / "config_pgie.txt")["property"]
        assert pgie["parse-bbox-func-name"] == "NvDsInferParseCvStudioYolo"
        assert pgie["model-color-format"] == "0"
        assert pgie["net-scale-factor"].startswith("0.00392")
        assert pgie["maintain-aspect-ratio"] == "0"

    def test_int8_profile_uses_int8_network_mode(self, tmp_path):
        data = _export(_rtsp(), _detector())
        out = tmp_path / "project"
        DeepStreamEngine("RTX_5070_INT8").convert_from_dict(data, out, "demo")

        pgie = _read_config(out / "config_pgie.txt")["property"]
        assert pgie["network-mode"] == "1"
        assert pgie["int8-calib-file"].endswith(".calib")

    def test_without_output_node_the_video_is_served_over_rtsp(self, tmp_path):
        result, out = _convert(tmp_path, _export(_rtsp(), _detector()))

        sink = _read_config(out / "deepstream_app_config.txt")["sink0"]
        assert sink["type"] == "4"
        assert sink["rtsp-port"] == "8554"
        assert "rtsp://<host>:8554/ds-test" in (out / "README.md").read_text()

    def test_multiple_sources_are_tiled_and_batched_in_streammux(self, tmp_path):
        data = _export(_rtsp(1), _rtsp(2, "rtsp://10.0.0.6/cam"), _detector(3))
        _, out = _convert(tmp_path, data)

        app = _read_config(out / "deepstream_app_config.txt")
        assert app["streammux"]["batch-size"] == "2"
        assert app["tiled-display"]["enable"] == "1"
        assert app["tiled-display"]["columns"] == "2"
        assert app["source1"]["uri"] == "rtsp://10.0.0.6/cam"

    def test_classifier_runs_on_detected_objects(self, tmp_path):
        data = _export(
            _rtsp(),
            _detector(),
            _node(3, "Classification", **{"Text:Input02Value": "Gender Recognition"}),
        )
        _, out = _convert(tmp_path, data)

        app = _read_config(out / "deepstream_app_config.txt")
        assert app["secondary-gie0"]["operate-on-gie-id"] == "1"
        sgie = _read_config(out / "config_sgie_0.txt")["property"]
        assert sgie["process-mode"] == "2"
        assert sgie["network-type"] == "1"
        assert sgie["gie-unique-id"] == app["secondary-gie0"]["gie-unique-id"] == "2"

    def test_classifier_alone_runs_on_the_full_frame(self, tmp_path):
        data = _export(
            _rtsp(),
            _node(2, "Classification", **{"Text:Input02Value": "Gender Recognition"}),
        )
        result, out = _convert(tmp_path, data)

        pgie = _read_config(out / "config_pgie.txt")["property"]
        assert pgie["process-mode"] == "1"
        assert pgie["network-type"] == "1"
        assert "custom-lib-path" not in pgie
        assert not (out / "nvdsinfer_custom_impl_cvstudio").exists()


class TestSourcesNotStoredInTheExport:
    def test_video_node_without_path_is_reported(self, tmp_path):
        data = _export(_node(1, "Video", **{"Text:Input02Value": True}), _detector())
        result, out = _convert(tmp_path, data)

        app = _read_config(out / "deepstream_app_config.txt")
        assert "CHANGE_ME" in app["source0"]["uri"]
        assert app["tests"]["file-loop"] == "1"
        assert any("--source 1=" in warning for warning in result.warnings)

    def test_video_path_given_as_override(self, tmp_path):
        data = _export(_node(1, "Video"), _detector())
        _, out = _convert(tmp_path, data, sources={1: "/data/videos/shop.mp4"})

        app = _read_config(out / "deepstream_app_config.txt")
        assert app["source0"]["type"] == "3"
        assert app["source0"]["uri"] == "file:///data/videos/shop.mp4"
        assert app["streammux"]["live-source"] == "0"
        # The file must be reachable from inside the container.
        compose = (out / "docker-compose.yml").read_text()
        assert '"/data/videos:/data/videos:ro"' in compose
        assert 'restart: "no"' in compose

    def test_hints_written_by_the_deploy_node(self, tmp_path):
        data = _export(
            _node(1, "Video"),
            _node(2, "Webcam"),
            _detector(3),
            hints={"1:Video": {"uri": "/data/a.mp4"}, "2:Webcam": {"device": "2"}},
        )
        _, out = _convert(tmp_path, data)

        app = _read_config(out / "deepstream_app_config.txt")
        assert app["source0"]["uri"] == "file:///data/a.mp4"
        assert app["source1"]["type"] == "1"
        assert app["source1"]["camera-v4l2-dev-node"] == "2"
        assert "/dev/video2:/dev/video2" in (out / "docker-compose.yml").read_text()


class TestUnsupportedContent:
    def test_detector_format_without_parser_is_skipped(self, tmp_path):
        data = _export(_rtsp(), _detector(model="BlazeFace(128x128)"), _tracker())
        result, out = _convert(tmp_path, data)

        assert result.pipeline.primary_gie is None
        assert result.pipeline.tracker is None
        assert not (out / "config_pgie.txt").exists()
        assert any("blazeface" in warning for warning in result.warnings)

    def test_unknown_model_is_listed_as_missing(self, tmp_path):
        data = _export(_rtsp(), _detector(model="my_custom_yolo"))
        result, out = _convert(tmp_path, data)

        assert result.missing_models == ["models/my_custom_yolo.onnx"]
        assert "Copy the missing model to models/my_custom_yolo.onnx" in result.summary()

    def test_nodes_without_equivalent_are_reported(self, tmp_path):
        data = _export(_rtsp(), _detector(), _node(3, "Resize"), _node(4, "PoseEstimation"))
        result, _ = _convert(tmp_path, data)

        assert result.pipeline.ignored_nodes == ["3:Resize"]
        assert any(w.startswith("4:PoseEstimation") for w in result.warnings)

    def test_disabled_tracker_is_not_generated(self, tmp_path):
        data = _export(_rtsp(), _detector(), _tracker(enabled=False))
        result, out = _convert(tmp_path, data)

        assert "tracker" not in _read_config(out / "deepstream_app_config.txt")

    def test_project_without_source_is_rejected_before_writing(self, tmp_path):
        out = tmp_path / "project"
        with pytest.raises(ValueError, match="no video source"):
            DeepStreamEngine().convert_from_dict(_export(_detector()), out)
        assert not out.exists()

    def test_not_a_cvstudio_export(self, tmp_path):
        with pytest.raises(ValueError, match="node_list"):
            DeepStreamEngine().convert_from_dict({"foo": 1}, tmp_path / "project")


class TestOverwrite:
    def test_overwrite_refuses_a_directory_it_did_not_generate(self, tmp_path):
        out = tmp_path / "documents"
        out.mkdir()
        (out / "important.txt").write_text("keep me")

        with pytest.raises(FileExistsError, match="Refusing"):
            DeepStreamEngine().convert_from_dict(
                _export(_rtsp(), _detector()), out, overwrite=True
            )
        assert (out / "important.txt").read_text() == "keep me"


class TestGeneratedBuildFiles:
    def test_generated_makefile_builds_the_parser_library(self, tmp_path):
        if shutil.which("g++") is None or shutil.which("make") is None:
            pytest.skip("g++ / make not available")
        _, out = _convert(tmp_path, _export(_rtsp(), _detector()))
        # Same layout as a DeepStream SDK install, with the stub header.
        includes = tmp_path / "nvds" / "sources" / "includes"
        includes.mkdir(parents=True)
        shutil.copy(HARNESS_DIR / "nvdsinfer_custom_impl.h", includes)

        subprocess.run(
            ["make", "-C", str(out / "nvdsinfer_custom_impl_cvstudio"),
             f"NVDS_ROOT={tmp_path / 'nvds'}"],
            check=True, capture_output=True,
        )

        pgie = _read_config(out / "config_pgie.txt")["property"]
        library = out / pgie["custom-lib-path"]
        assert library.is_file()
        symbols = subprocess.run(
            ["nm", "-D", "--defined-only", str(library)],
            check=True, capture_output=True, text=True,
        ).stdout
        assert f" T {pgie['parse-bbox-func-name']}\n" in symbols

    def test_docker_compose_is_valid_yaml(self, tmp_path):
        yaml = pytest.importorskip("yaml")
        data = _export(_node(1, "Video"), _node(2, "Webcam"), _detector(3))
        sources = {1: "/data/v.mp4", 2: "/dev/video2"}
        _, out = _convert(tmp_path, data, sources=sources, display=True)

        service = yaml.safe_load((out / "docker-compose.yml").read_text())["services"]["deepstream"]
        assert service["runtime"] == "nvidia"
        assert service["restart"] == "no"
        assert "/data:/data:ro" in service["volumes"]
        assert service["devices"] == ["/dev/video2:/dev/video2"]


class TestCli:
    def test_source_option(self, tmp_path, capsys):
        import json

        json_path = tmp_path / "wf.json"
        json_path.write_text(json.dumps(_export(_node(1, "Video"), _detector())))
        out = tmp_path / "out"

        assert cli_main([str(json_path), "-o", str(out), "--source", "1=/data/v.mp4"]) == 0
        app = _read_config(out / "deepstream_app_config.txt")
        assert app["source0"]["uri"] == "file:///data/v.mp4"

    def test_invalid_source_option(self, tmp_path, capsys):
        json_path = tmp_path / "wf.json"
        json_path.write_text("{}")

        assert cli_main([str(json_path), "--source", "/data/v.mp4"]) == 1
        assert "NODE_ID=URI" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# bbox parser library: compiled and run against a numpy reference
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def parser_harness(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ not available")
    build_dir = tmp_path_factory.mktemp("parser")
    source = build_dir / "nvdsparsebbox_cvstudio.cpp"
    source.write_text(PARSER_SOURCE)
    exe = build_dir / "harness"
    subprocess.run(
        [compiler, "-std=c++14", "-O1", f"-I{HARNESS_DIR}", str(source),
         str(HARNESS_DIR / "harness.cpp"), "-o", str(exe)],
        check=True,
    )

    def run(kind, net_w, net_h, num_classes, threshold, tensor):
        tensor_file = build_dir / "tensor.bin"
        tensor.astype(np.float32).tofile(tensor_file)
        result = subprocess.run(
            [str(exe), kind, str(net_w), str(net_h), str(num_classes), str(threshold),
             str(tensor_file), *map(str, tensor.shape)],
            capture_output=True, text=True, check=True,
        )
        return [[float(v) for v in line.split()] for line in result.stdout.splitlines()]

    return run


def _yolox_grid(net_w, net_h):
    cells = []
    for stride in (8, 16, 32):
        for y in range(net_h // stride):
            for x in range(net_w // stride):
                cells.append((x, y, stride))
    return np.array(cells, dtype=np.float32)


class TestBboxParser:
    def test_yolo_attributes_first_layout(self, parser_harness):
        # Ultralytics layout [1, 4 + C, N]: 3 classes, 5 candidates.
        boxes = np.zeros((5, 7), dtype=np.float32)
        boxes[1] = [320, 240, 100, 50, 0.1, 0.9, 0.2]   # class 1, kept
        boxes[3] = [20, 20, 80, 80, 0.6, 0.1, 0.0]      # class 0, clipped at 0
        boxes[4] = [100, 100, 40, 40, 0.2, 0.1, 0.24]   # below threshold
        tensor = boxes.T[np.newaxis]

        objects = parser_harness("yolo", 640, 480, 3, 0.25, tensor)

        assert objects == [
            pytest.approx([1, 270, 215, 100, 50, 0.9], abs=1e-4),
            pytest.approx([0, 0, 0, 60, 60, 0.6], abs=1e-4),
        ]

    def test_yolo_boxes_first_layout_gives_the_same_objects(self, parser_harness):
        rng = np.random.default_rng(0)
        boxes = np.concatenate(
            [rng.uniform(50, 300, (40, 4)), rng.uniform(0, 1, (40, 6))], axis=1
        ).astype(np.float32)

        attrs_first = parser_harness("yolo", 416, 416, 6, 0.5, boxes.T)
        boxes_first = parser_harness("yolo", 416, 416, 6, 0.5, boxes)

        assert attrs_first == boxes_first
        assert len(attrs_first) == int((boxes[:, 4:].max(axis=1) >= 0.5).sum())

    def test_yolox_grid_decoding_matches_reference(self, parser_harness):
        net_w, net_h, num_classes, threshold = 416, 320, 4, 0.3
        grid = _yolox_grid(net_w, net_h)
        rng = np.random.default_rng(1)
        raw = np.concatenate(
            [
                rng.uniform(-0.5, 1.5, (len(grid), 2)),   # cell offsets
                rng.uniform(0.0, 2.5, (len(grid), 2)),    # log sizes
                rng.uniform(0, 1, (len(grid), 1)) ** 4,   # objectness
                rng.uniform(0, 1, (len(grid), num_classes)),
            ],
            axis=1,
        ).astype(np.float32)

        objects = parser_harness("yolox", net_w, net_h, num_classes, threshold, raw[np.newaxis])

        # Reference: the decoding CvStudio applies to YOLOX outputs.
        stride = grid[:, 2]
        cx = (raw[:, 0] + grid[:, 0]) * stride
        cy = (raw[:, 1] + grid[:, 1]) * stride
        w = np.exp(raw[:, 2]) * stride
        h = np.exp(raw[:, 3]) * stride
        class_ids = raw[:, 5:].argmax(axis=1)
        scores = raw[:, 4] * raw[:, 5:].max(axis=1)
        x1, x2 = np.clip(cx - w / 2, 0, net_w), np.clip(cx + w / 2, 0, net_w)
        y1, y2 = np.clip(cy - h / 2, 0, net_h), np.clip(cy + h / 2, 0, net_h)
        keep = (scores >= threshold) & (x2 - x1 >= 1) & (y2 - y1 >= 1)
        expected = np.stack(
            [class_ids, x1, y1, x2 - x1, y2 - y1, scores], axis=1
        )[keep]

        assert len(objects) == len(expected) > 0
        np.testing.assert_allclose(np.array(objects), expected, rtol=1e-4, atol=2e-3)

    def test_unexpected_tensor_rank_is_an_error(self, parser_harness):
        with pytest.raises(subprocess.CalledProcessError):
            parser_harness("yolo", 64, 64, 3, 0.5, np.zeros((2, 2, 7, 5), np.float32))
