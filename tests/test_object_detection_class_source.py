#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the object detection ONNX upload class-source selection.

When an uploaded ONNX model does not embed class names, the upload preview
dialog lets the user choose between COCO dataset labels (like the built-in
YOLO models) and generic ``class_<i>`` placeholders.
"""

import importlib.util
import os
import sys
import types
from unittest import mock


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE_PATH = os.path.join(REPO_ROOT, 'node', 'DLNode', 'node_object_detection.py')


def _load_object_detection_module():
    """Load the object detection module with lightweight dependency stubs."""
    dpg_mock = mock.MagicMock()

    mocked_modules = {
        'cv2': mock.MagicMock(),
        'numpy': mock.MagicMock(),
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
    mocked_modules['dearpygui'].dearpygui = dpg_mock
    mocked_modules['node_editor'].util = mocked_modules['node_editor.util']
    mocked_modules['src'].utils = mocked_modules['src.utils']

    original_modules = {}
    for name, module in mocked_modules.items():
        original_modules[name] = sys.modules.get(name)
        sys.modules[name] = module

    spec = importlib.util.spec_from_file_location('node_object_detection_class_source', MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name, original in original_modules.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original

    return module, dpg_mock


def test_build_class_names_coco():
    """COCO source should produce the COCO labels for the model's classes."""
    module, _ = _load_object_detection_module()
    names = module.build_class_names_from_source(module._CLASS_SOURCE_COCO, 80)
    assert len(names) == 80
    assert names[0] == 'person'
    assert names[2] == 'car'


def test_build_class_names_coco_zero_defaults_to_full_coco():
    """COCO source with unknown class count falls back to the full COCO set."""
    module, _ = _load_object_detection_module()
    names = module.build_class_names_from_source(module._CLASS_SOURCE_COCO, 0)
    assert len(names) == len(module.coco_class_names)


def test_build_class_names_generic():
    """Generic source should produce class_<i> placeholders."""
    module, _ = _load_object_detection_module()
    names = module.build_class_names_from_source(module._CLASS_SOURCE_GENERIC, 3)
    assert names == {0: 'class_0', 1: 'class_1', 2: 'class_2'}


def test_class_source_change_rebuilds_pending_class_names():
    """Switching the class source updates the pending class names."""
    module, _ = _load_object_detection_module()
    node = module.Node()
    node.tag_preview_details = 'details'
    node._pending_meta = {'num_classes': 80}

    node._on_class_source_change(module._CLASS_SOURCE_GENERIC)
    assert node._pending_class_names[0] == 'class_0'

    node._on_class_source_change(module._CLASS_SOURCE_COCO)
    assert node._pending_class_names[0] == 'person'
