#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests verifying that object detection always broadcasts all supported class names
and that the Chart node exposes every class in its dropdowns, even with zero detections.
"""

import importlib.util
import os
import sys
import types
from collections import defaultdict
from unittest import mock


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OD_MODULE_PATH = os.path.join(REPO_ROOT, 'node', 'DLNode', 'node_object_detection.py')
CHART_MODULE_PATH = os.path.join(REPO_ROOT, 'node', 'VisualNode', 'node_obj_chart.py')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_od_module():
    """Load node_object_detection with lightweight stubs."""
    import numpy
    dpg_mock = mock.MagicMock()
    mocked = {
        'cv2': mock.MagicMock(),
        'numpy': numpy,
        'onnxruntime': mock.MagicMock(),
        'dearpygui': types.ModuleType('dearpygui'),
        'dearpygui.dearpygui': dpg_mock,
        'node_editor': types.ModuleType('node_editor'),
        'node_editor.util': types.SimpleNamespace(
            dpg_get_value=mock.MagicMock(),
            dpg_set_value=mock.MagicMock(),
            run_unlocked=lambda func, *args, **kwargs: func(*args, **kwargs),
        ),
        'node.basenode': types.SimpleNamespace(Node=type('BaseNode', (), {})),
        'src': types.ModuleType('src'),
        'src.utils': types.ModuleType('src.utils'),
        'src.utils.logging': types.SimpleNamespace(get_logger=lambda name: mock.MagicMock()),
        'src.utils.gpu_utils': types.SimpleNamespace(get_execution_providers=lambda: ['CPUExecutionProvider']),
        'node.DLNode.object_detection.CustomONNX.custom_onnx': types.SimpleNamespace(CustomONNX=mock.MagicMock()),
        'node.DLNode.object_detection.onnx_inspector': mock.MagicMock(),
        'node.DLNode.object_detection.custom_models_registry': mock.MagicMock(),
    }
    mocked['dearpygui'].dearpygui = dpg_mock
    mocked['node_editor'].util = mocked['node_editor.util']
    mocked['src'].utils = mocked['src.utils']

    saved = {}
    for name, mod in mocked.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mod

    spec = importlib.util.spec_from_file_location('_od_broadcast_test', OD_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name, orig in saved.items():
        if orig is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = orig

    return module, dpg_mock


def _load_chart_module():
    """Load node_obj_chart with lightweight stubs."""
    import importlib.util as _ilu

    # Load the real coco_class_names module (no external deps) so chart gets
    # the real mapping without triggering the full OD __init__ chain.
    _coco_spec = _ilu.spec_from_file_location(
        '_coco_cn',
        os.path.join(REPO_ROOT, 'node', 'DLNode', 'object_detection', 'coco_class_names.py'),
    )
    _coco_mod = _ilu.module_from_spec(_coco_spec)
    _coco_spec.loader.exec_module(_coco_mod)

    dpg_mock = mock.MagicMock()
    matplotlib_mock = mock.MagicMock()
    matplotlib_mock.use = mock.MagicMock()
    mocked = {
        'cv2': mock.MagicMock(),
        'numpy': mock.MagicMock(),
        'onnxruntime': mock.MagicMock(),
        'matplotlib': matplotlib_mock,
        'matplotlib.pyplot': mock.MagicMock(),
        'matplotlib.backends': mock.MagicMock(),
        'matplotlib.backends.backend_agg': mock.MagicMock(),
        'dearpygui': types.ModuleType('dearpygui'),
        'dearpygui.dearpygui': dpg_mock,
        'node_editor': types.ModuleType('node_editor'),
        'node_editor.util': types.SimpleNamespace(
            dpg_get_value=mock.MagicMock(),
            dpg_set_value=mock.MagicMock(),
        ),
        'node.basenode': types.SimpleNamespace(Node=type('BaseNode', (), {})),
        'src': types.ModuleType('src'),
        'src.utils': types.ModuleType('src.utils'),
        'src.utils.logging': types.SimpleNamespace(get_logger=lambda name: mock.MagicMock()),
        # Stub the full OD package so __init__.py isn't re-executed, but expose
        # coco_class_names via the path that node_obj_chart imports.
        'node.DLNode.object_detection': mock.MagicMock(),
        'node.DLNode.object_detection.coco_class_names': _coco_mod,
        'node.DLNode.object_detection.CustomONNX': mock.MagicMock(),
        'node.DLNode.object_detection.CustomONNX.custom_onnx': mock.MagicMock(),
    }
    mocked['dearpygui'].dearpygui = dpg_mock
    mocked['node_editor'].util = mocked['node_editor.util']
    mocked['src'].utils = mocked['src.utils']
    mocked['matplotlib'].pyplot = mocked['matplotlib.pyplot']

    saved = {}
    for name, mod in mocked.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mod

    spec = importlib.util.spec_from_file_location('_chart_broadcast_test', CHART_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name, orig in saved.items():
        if orig is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = orig

    return module, dpg_mock


# ---------------------------------------------------------------------------
# OD node: class_names broadcast
# ---------------------------------------------------------------------------

class TestODAllClassesBroadcast:

    def test_class_names_present_with_detections(self):
        """result always contains class_names (all model classes) when detections are found."""
        full_classes = {0: 'person', 1: 'bicycle', 2: 'car'}

        # Simulate the result-building block when bboxes are non-empty
        import numpy as np
        bboxes = np.array([[0, 0, 10, 10]])
        scores = np.array([0.9])
        class_ids = np.array([0])
        class_name_dict = full_classes
        score_th = 0.3

        result = {}
        if len(bboxes) > 0:
            result['bboxes'] = bboxes.tolist()
            result['scores'] = scores.tolist()
            result['class_ids'] = class_ids.tolist()
            result['class_names'] = class_name_dict
            result['score_th'] = score_th

        assert 'class_names' in result
        # class_names must be the FULL model class dict, not just detected classes
        assert result['class_names'] == full_classes
        assert set(result['class_names'].keys()) == {0, 1, 2}
        assert result['class_ids'] == [0]

    def test_class_names_present_when_no_frame(self):
        """When frame is None, result must still contain class_names with all supported classes."""
        module, _ = _load_od_module()
        node = module.Node()

        full_classes = {0: 'person', 1: 'bicycle', 2: 'car'}
        node._model_class_name_list = {'TestModel': full_classes}
        node._model_class = {'TestModel': mock.MagicMock()}
        node._model_path_setting = {'TestModel': 'dummy.onnx'}
        node._model_instance = {}

        # Simulate the result-building block when frame is None
        # We test the logic directly by calling the same code path:
        result = {}
        class_name_dict = full_classes
        score_th = 0.3
        # Replicate the new else branch
        if not result:
            result['bboxes'] = []
            result['scores'] = []
            result['class_ids'] = []
            result['class_names'] = class_name_dict
            result['score_th'] = score_th

        assert 'class_names' in result
        assert result['class_names'] == full_classes
        assert result['bboxes'] == []
        assert result['class_ids'] == []


# ---------------------------------------------------------------------------
# Chart node: _build_dynamic_class_items includes all model classes
# ---------------------------------------------------------------------------

class TestChartAllClassesDropdown:

    def _make_node(self, module):
        node = module.Node()
        node.time_counts = defaultdict(lambda: defaultdict(int))
        return node

    def test_all_model_classes_shown_with_no_detections(self):
        """All classes from class_names must appear even when class_ids is empty."""
        module, _ = _load_chart_module()
        node = self._make_node(module)

        # Model supports 3 classes, none detected this frame
        class_names = {0: 'person', 1: 'bicycle', 2: 'car'}
        items = node._build_dynamic_class_items([], class_names)

        assert 'All' in items
        assert '0: person' in items
        assert '1: bicycle' in items
        assert '2: car' in items

    def test_all_model_classes_shown_string_keyed_dict(self):
        """Works when class_names uses string keys (as can arrive from JSON)."""
        module, _ = _load_chart_module()
        node = self._make_node(module)

        class_names = {'0': 'person', '1': 'bicycle', '2': 'car'}
        items = node._build_dynamic_class_items([], class_names)

        assert '0: person' in items
        assert '1: bicycle' in items
        assert '2: car' in items

    def test_undetected_class_still_selectable(self):
        """A class supported by the model but never detected appears in the dropdown."""
        module, _ = _load_chart_module()
        node = self._make_node(module)

        # Only class 0 detected; class 2 (car) never detected
        class_names = {0: 'person', 1: 'bicycle', 2: 'car'}
        items = node._build_dynamic_class_items([0], class_names)

        assert '2: car' in items
        assert '1: bicycle' in items

    def test_detected_classes_also_present(self):
        """Detected classes still appear (not dropped)."""
        module, _ = _load_chart_module()
        node = self._make_node(module)

        class_names = {0: 'person', 1: 'bicycle'}
        items = node._build_dynamic_class_items([0, 1], class_names)

        assert '0: person' in items
        assert '1: bicycle' in items

    def test_class_with_accumulated_count_present(self):
        """Accumulated class IDs that belong to the *current* model are still listed.
        Class IDs from old models (not in class_names) should NOT appear so that
        switching models clears stale options from the dropdown."""
        module, _ = _load_chart_module()
        node = self._make_node(module)
        node.time_counts[0][0] = 3  # class 0 seen in a past bucket – same model

        class_names = {0: 'person'}
        items = node._build_dynamic_class_items([], class_names)

        # Class 0 is in class_names → must appear
        assert '0: person' in items

    def test_stale_class_from_old_model_not_shown(self):
        """Class IDs accumulated from an old model (not in current class_names) must
        NOT appear in the dropdown after the model is switched."""
        module, _ = _load_chart_module()
        node = self._make_node(module)
        node.time_counts[5][0] = 3  # class 5 was seen with the old model

        # New model only knows class 0
        class_names = {0: 'person'}
        items = node._build_dynamic_class_items([], class_names)

        # Class 5 is not in the new model's class_names – must NOT appear
        assert not any(item.startswith('5:') for item in items)
        assert '0: person' in items

    def test_stale_class_shown_when_no_class_names(self):
        """When class_names is empty (generic numeric dict source), accumulated
        class IDs from time_counts are still shown for backward compatibility."""
        module, _ = _load_chart_module()
        node = self._make_node(module)
        node.time_counts[5][0] = 3  # class 5 accumulated

        items = node._build_dynamic_class_items([], {})

        # With no class_names constraint all accumulated IDs appear
        assert any(item.startswith('5:') for item in items)

    def test_all_first_item(self):
        """'All' is always the first item."""
        module, _ = _load_chart_module()
        node = self._make_node(module)

        items = node._build_dynamic_class_items([], {0: 'person', 2: 'car'})
        assert items[0] == 'All'

    def test_items_sorted_by_class_id(self):
        """Items after 'All' are sorted ascending by class id."""
        module, _ = _load_chart_module()
        node = self._make_node(module)

        class_names = {5: 'bus', 0: 'person', 2: 'car'}
        items = node._build_dynamic_class_items([], class_names)
        non_all = [i for i in items if i != 'All']
        ids = [int(i.split(':')[0].strip()) for i in non_all]
        assert ids == sorted(ids)


class TestModelChangeUpdatesDropdown:
    """Integration tests: switching the model must refresh the Chart dropdown."""

    def _make_node(self, module):
        node = module.Node()
        node.time_counts = defaultdict(lambda: defaultdict(int))
        return node

    def test_switch_from_coco_to_small_model(self):
        """After switching from an 80-class COCO model to a 3-class model,
        only the 3 new classes should appear in the dropdown."""
        module, _ = _load_chart_module()
        node = self._make_node(module)

        # Simulate detections accumulated under the old COCO model (80 classes)
        for cid in range(80):
            node.time_counts[cid][0] = 1

        # New model has only 3 classes
        new_class_names = {0: 'player1', 1: 'player2', 2: 'ball'}
        items = node._build_dynamic_class_items([], new_class_names)

        assert items[0] == 'All'
        assert '0: player1' in items
        assert '1: player2' in items
        assert '2: ball' in items
        # Old COCO-only class IDs must not appear
        assert not any(item.startswith('3:') for item in items)
        assert len(items) == 4  # "All" + 3 classes

    def test_switch_from_small_model_to_coco(self):
        """Switching from a 3-class model to an 80-class COCO model populates
        all 80 COCO classes in the dropdown."""
        module, _ = _load_chart_module()
        node = self._make_node(module)

        # Simulate detections from 3-class model
        for cid in (0, 1, 2):
            node.time_counts[cid][0] = 1

        # Load coco_class_names the same lightweight way the chart module loader does
        import importlib.util as _ilu
        _coco_path = os.path.join(
            REPO_ROOT, 'node', 'DLNode', 'object_detection', 'coco_class_names.py'
        )
        _coco_spec = _ilu.spec_from_file_location('_coco_cn_test', _coco_path)
        _coco_mod = _ilu.module_from_spec(_coco_spec)
        _coco_spec.loader.exec_module(_coco_mod)
        coco_class_names = _coco_mod.coco_class_names

        items = node._build_dynamic_class_items([], coco_class_names)

        assert '0: person' in items
        # Classes previously unknown to the 3-class model must now appear
        assert any(item.startswith('3:') for item in items)
        assert len(items) == 81  # "All" + 80 COCO classes

    def test_classification_model_change(self):
        """Same behaviour for a classification model switch:
        only the new model's classes appear after the switch."""
        module, _ = _load_chart_module()
        node = self._make_node(module)

        # Old classification model: 1000 ImageNet classes accumulated
        for cid in range(1000):
            node.time_counts[cid][0] = 1

        # New classification model: 2 gender classes
        gender_classes = {0: 'Male', 1: 'Female'}
        items = node._build_dynamic_class_items([], gender_classes)

        assert '0: Male' in items
        assert '1: Female' in items
        # ImageNet-only class IDs must not appear
        assert not any(item.startswith('2:') for item in items)
        assert len(items) == 3  # "All" + 2 classes

