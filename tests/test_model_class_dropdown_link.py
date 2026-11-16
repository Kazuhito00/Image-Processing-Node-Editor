#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for linking model selection to class rejection dropdown"""

import pytest
import sys
import os
import importlib.util

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


_COCO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'node', 'DLNode', 'object_detection', 'coco_class_names.py'
)
_COCO_SPEC = importlib.util.spec_from_file_location('_test_coco_class_names', _COCO_PATH)
_COCO_MODULE = importlib.util.module_from_spec(_COCO_SPEC)
_COCO_SPEC.loader.exec_module(_COCO_MODULE)
coco_class_names = _COCO_MODULE.coco_class_names
_OD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'node', 'DLNode', 'node_object_detection.py'
)


def test_coco_class_names():
    """Test that COCO class name dictionary is properly defined."""
    assert len(coco_class_names) == 80, "COCO should have 80 classes"
    assert 0 in coco_class_names
    assert coco_class_names[0] == 'person'


def test_get_class_rejection_dropdown_items():
    """Test the function that generates dropdown items from class names"""
    module = _load_od_module()
    get_class_rejection_dropdown_items = module.get_class_rejection_dropdown_items

    # Test with COCO classes
    coco_items = get_class_rejection_dropdown_items(coco_class_names)
    assert len(coco_items) == 80, "Should have 80 items for COCO"
    assert coco_items[0] == "0: person", "First item should be formatted as 'ID: name'"
    assert coco_items[1] == "1: bicycle", "Second item should be '1: bicycle'"

    # Test with a small custom dict
    custom = {0: 'player1', 1: 'player2', 2: 'ball'}
    items = get_class_rejection_dropdown_items(custom)
    assert len(items) == 3
    assert items[0] == "0: player1"
    assert items[1] == "1: player2"
    assert items[2] == "2: ball"

    # Test with person-only
    person_only = {0: 'person'}
    person_items = get_class_rejection_dropdown_items(person_only)
    assert len(person_items) == 1
    assert person_items[0] == "0: person"


def test_get_batch_badge_label():
    """Only true batch-capable models should show the B marker."""
    module = _load_od_module()
    get_batch_badge_label = module.get_batch_badge_label

    assert get_batch_badge_label(True) == "B"
    assert get_batch_badge_label(False) == ""


def _load_od_module():
    """Load node_object_detection with lightweight stubs."""
    import unittest.mock as mock
    import types

    dpg_mock = mock.MagicMock()
    mocked = {
        'cv2': mock.MagicMock(),
        'numpy': mock.MagicMock(),
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
        'node.DLNode.object_detection': types.ModuleType('node.DLNode.object_detection'),
        'node.DLNode.object_detection.coco_class_names': _COCO_MODULE,
        'node.DLNode.object_detection.BlazeFace': types.ModuleType('node.DLNode.object_detection.BlazeFace'),
        'node.DLNode.object_detection.BlazeFace.blazeface': types.SimpleNamespace(BlazeFace=mock.MagicMock()),
        'node.DLNode.object_detection.CustomONNX': mock.MagicMock(),
        'node.DLNode.object_detection.CustomONNX.custom_onnx': types.SimpleNamespace(CustomONNX=mock.MagicMock()),
        'node.DLNode.object_detection.custom_models_registry': types.SimpleNamespace(
            load_registry=lambda: [],
            save_entry=mock.MagicMock(),
            remove_entry=mock.MagicMock(),
            get_entry=lambda name: None,
        ),
        'node.DLNode.object_detection.onnx_inspector': types.SimpleNamespace(
            inspect_onnx_model=lambda path: {
                'output_format': 'yolo11',
                'input_width': 640,
                'input_height': 640,
                'num_classes': 80,
                'class_names': {},
                'supports_batched_detection': False,
                'has_dynamic_batch': False,
            }
        ),
    }
    mocked['dearpygui'].dearpygui = dpg_mock
    mocked['node_editor'].util = mocked['node_editor.util']
    mocked['src'].utils = mocked['src.utils']
    mocked['node.DLNode.object_detection'].coco_class_names = _COCO_MODULE
    mocked['node.DLNode.object_detection'].onnx_inspector = mocked['node.DLNode.object_detection.onnx_inspector']
    mocked['node.DLNode.object_detection'].custom_models_registry = mocked['node.DLNode.object_detection.custom_models_registry']

    saved = {}
    for name, mod in mocked.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mod

    spec = importlib.util.spec_from_file_location('_od_dropdown_test', _OD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name, orig in saved.items():
        if orig is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = orig

    return module


def test_builtin_models_defined():
    """Test that _BUILTIN_MODELS contains expected entries."""
    module = _load_od_module()
    _BUILTIN_MODELS = module._BUILTIN_MODELS

    names = {m['name'] for m in _BUILTIN_MODELS}
    assert 'YOLOX-Nano(416x416)' in names
    assert 'YOLO11Nano' in names
    assert 'yolo8n_B' in names
    assert 'yolo8s_B' in names
    assert 'YOLOTENNIS' in names
    assert 'Light-Weight Person Detector' in names

    # Each entry must have required keys
    for m in _BUILTIN_MODELS:
        for key in ('name', 'path', 'output_format', 'input_width', 'input_height',
                    'num_classes', 'class_names'):
            assert key in m, f"Missing key '{key}' in built-in model '{m.get('name')}'"

    # Tennis model should have 3 classes
    tennis = next((m for m in _BUILTIN_MODELS if m['name'] == 'YOLOTENNIS'), None)
    assert tennis is not None, "YOLOTENNIS should be in _BUILTIN_MODELS"
    assert tennis['num_classes'] == 3
    assert tennis['class_names'].get(0) == 'player1'
    assert tennis['class_names'].get(1) == 'player2'
    assert tennis['class_names'].get(2) == 'ball'

    for name in ('yolo8n_B', 'yolo8s_B'):
        model = next((m for m in _BUILTIN_MODELS if m['name'] == name), None)
        assert model is not None, f"{name} should be in _BUILTIN_MODELS"
        assert model.get('supports_batched_detection') is True


def test_callback_function_exists():
    """Test that on_model_change callback and related helpers are present."""
    with open(_OD_PATH, 'r') as f:
        content = f.read()

    assert 'def on_model_change' in content, "Should have on_model_change callback"
    assert 'dpg.configure_item' in content, "Should use dpg.configure_item"
    assert 'get_class_rejection_dropdown_items' in content
    assert 'callback=on_model_change' in content


def test_rejected_classes_cleared_on_model_change():
    """Test that rejected classes are cleared when model changes."""
    with open(_OD_PATH, 'r') as f:
        content = f.read()

    assert 'dpg_set_value(node.tag_node_rejected_classes_value_name, "")' in content, \
        "Should clear rejected classes when model changes"


def test_update_batch_badge_uses_batch_capability_map():
    """The runtime badge should reflect the selected model batch capability."""
    module = _load_od_module()
    node = module.Node()
    node.tag_node_batch_badge_value_name = 'badge'
    node._model_batch_capable = {'dynamic': True, 'static': False}

    calls = []

    def fake_set_value(tag, value):
        calls.append((tag, value))

    original = module.dpg_set_value
    module.dpg_set_value = fake_set_value
    try:
        node._update_batch_badge('dynamic')
        node._update_batch_badge('static')
    finally:
        module.dpg_set_value = original

    assert calls == [('badge', 'B'), ('badge', '')]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
