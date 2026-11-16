#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the object detection upload preview quit button."""

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

    spec = importlib.util.spec_from_file_location('node_object_detection_module', MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name, original in original_modules.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original

    return module, dpg_mock


def test_upload_preview_toggles_buttons_after_success():
    """Successful uploads should switch the dialog to Quit-only mode."""
    module, dpg_mock = _load_object_detection_module()
    node = module.Node()
    node.tag_preview_confirm = 'confirm'
    node.tag_preview_cancel = 'cancel'
    node.tag_preview_quit = 'quit'

    node._set_upload_preview_actions(upload_succeeded=True)

    dpg_mock.configure_item.assert_any_call('confirm', show=False)
    dpg_mock.configure_item.assert_any_call('cancel', show=False)
    dpg_mock.configure_item.assert_any_call('quit', show=True)


def test_upload_preview_toggles_buttons_before_success():
    """Pending uploads should keep Confirm/Cancel visible and hide Quit."""
    module, dpg_mock = _load_object_detection_module()
    node = module.Node()
    node.tag_preview_confirm = 'confirm'
    node.tag_preview_cancel = 'cancel'
    node.tag_preview_quit = 'quit'

    node._set_upload_preview_actions(upload_succeeded=False)

    dpg_mock.configure_item.assert_any_call('confirm', show=True)
    dpg_mock.configure_item.assert_any_call('cancel', show=True)
    dpg_mock.configure_item.assert_any_call('quit', show=False)


def test_quit_button_closes_preview_dialog():
    """Quit should close the preview dialog."""
    module, dpg_mock = _load_object_detection_module()
    node = module.Node()
    node.tag_preview_window = 'preview'

    node._close_upload_preview()

    dpg_mock.hide_item.assert_called_once_with('preview')
