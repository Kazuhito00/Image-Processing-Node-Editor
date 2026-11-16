#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import shutil
import time
import os
import sys

import numpy as np
import cv2
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value, run_unlocked

from node.node_abc import DpgNodeABC

from node.DLNode.classification.MobileNetV3.mobilenet_v3 import MobileNetV3
from node.DLNode.classification.EfficientNetB0.efficientnet import EfficientNetB0
from node.DLNode.classification.ResNet50.resnet50 import ResNet50
from node.DLNode.classification.GenderRecognition.gender_recognition import GenderRecognition
from node.DLNode.classification.PedestrianGender.pedestrian_gender import PedestrianGender

# Import YoloCls using importlib.util due to hyphenated directory name
import importlib.util
_yolo_cls_init_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'classification', 'Yolo-cls', '__init__.py'
)
_yolo_cls_spec = importlib.util.spec_from_file_location(
    'yolo_cls_init_module', _yolo_cls_init_path
)
_yolo_cls_module = importlib.util.module_from_spec(_yolo_cls_spec)
_yolo_cls_spec.loader.exec_module(_yolo_cls_module)
YoloCls = _yolo_cls_module.YoloCls

from node.DLNode.classification.imagenet_class_names import imagenet_class_names
from node.DLNode.classification.esc50_class_names import esc50_class_names

gender_class_names = GenderRecognition.CLASS_NAMES
pedestrian_gender_class_names = PedestrianGender.CLASS_NAMES
from node.DLNode.classification.CustomONNX.custom_onnx import CustomONNX as CustomONNXClassification
from node.DLNode.classification import custom_models_registry as _cls_registry
from node.DLNode.object_detection import onnx_inspector

from node.basenode import Node

from src.utils.logging import get_logger
logger = get_logger(__name__)

if getattr(sys, 'frozen', False):
    from src.utils.paths import get_models_dir
    _CLS_UPLOADS_DIR = get_models_dir('classification')
else:
    _CLS_UPLOADS_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'classification', 'CustomONNX', 'models'
    )

_BUILTIN_CLS_MODEL_NAMES = {
    'MobileNetV3 Small',
    'MobileNetV3 Large',
    'EfficientNet B0',
    'ResNet50',
    'Yolo-cls',
    'Gender Recognition',
    'Pedestrian Gender',
}


class FactoryNode:
    node_label = 'Classification'
    node_tag = 'Classification'
    

    def __init__(self):
        pass

    
    def add_node(self, parent, node_id, pos=[0, 0], callback=None, opencv_setting_dict=None):
        """Adds a node to the processing graph."""
        node = Node()
        
        # タグ名
        tag_node_name = str(node_id) + ':' + node.node_tag
        tag_node_input01_name = tag_node_name + ':' + node.TYPE_IMAGE + ':Input01'
        tag_node_input01_value_name = tag_node_name + ':' + node.TYPE_IMAGE + ':Input01Value'
        tag_node_input02_name = tag_node_name + ':' + node.TYPE_TEXT + ':Input02'
        tag_node_input02_value_name = tag_node_name + ':' + node.TYPE_TEXT + ':Input02Value'
        tag_node_output01_name = tag_node_name + ':' + node.TYPE_IMAGE + ':Output01'
        tag_node_output01_value_name = tag_node_name + ':' + node.TYPE_IMAGE + ':Output01Value'
        tag_node_output02_name = tag_node_name + ':' + node.TYPE_TIME_MS + ':Output02'
        tag_node_output02_value_name = tag_node_name + ':' + node.TYPE_TIME_MS + ':Output02Value'
        tag_node_output_json_name = tag_node_name + ':' + node.TYPE_JSON + ':OutputJson'
        tag_node_output_json_value_name = tag_node_name + ':' + node.TYPE_JSON + ':OutputJsonValue'

        tag_score_threshold_name = tag_node_name + ':' + node.TYPE_FLOAT + ':ScoreThreshold'
        tag_score_threshold_value_name = tag_node_name + ':' + node.TYPE_FLOAT + ':ScoreThresholdValue'

        tag_bbox_thickness_name = tag_node_name + ':' + node.TYPE_INT + ':BboxThickness'
        tag_bbox_thickness_value_name = tag_node_name + ':' + node.TYPE_INT + ':BboxThicknessValue'

        tag_provider_select_name = tag_node_name + ':' + node.TYPE_TEXT + ':Provider'
        tag_provider_select_value_name = tag_node_name + ':' + node.TYPE_IMAGE + ':ProviderValue'

        tag_class_filter_name = tag_node_name + ':' + node.TYPE_TEXT + ':ClassFilter'
        tag_class_filter_value_name = tag_node_name + ':' + node.TYPE_TEXT + ':ClassFilterValue'

        tag_batch_size_name = tag_node_name + ':' + node.TYPE_INT + ':BatchSize'
        tag_batch_size_value_name = tag_node_name + ':' + node.TYPE_INT + ':BatchSizeValue'

        # OpenCV向け設定
        node._opencv_setting_dict = opencv_setting_dict
        small_window_w = node._opencv_setting_dict['process_width']
        small_window_h = node._opencv_setting_dict['process_height']
        use_pref_counter = node._opencv_setting_dict['use_pref_counter']
        use_gpu = node._opencv_setting_dict['use_gpu']

        # 初期化用黒画像
        black_image = np.zeros((small_window_w, small_window_h, 3))
        black_texture = node.convert_cv_to_dpg(
            black_image,
            small_window_w,
            small_window_h,
        )

        # テクスチャ登録
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                small_window_w,
                small_window_h,
                black_texture,
                tag=tag_node_output01_value_name,
                format=dpg.mvFormat_Float_rgb,
            )

        # Create yellow theme for JSON button
        with dpg.theme() as yellow_button_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (255, 255, 153, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (255, 255, 153, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (255, 255, 153, 255))

        # ---- ONNX upload file dialog ----------------------------------------
        onnx_file_dialog_tag = "cls_onnx_select:" + str(node_id)
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            modal=True,
            height=400,
            callback=node._callback_onnx_select,
            tag=onnx_file_dialog_tag,
        ):
            dpg.add_file_extension("ONNX (*.onnx){.onnx}")
            dpg.add_file_extension("", color=(150, 255, 150, 255))
        node.tag_upload_file_dialog = onnx_file_dialog_tag

        # ---- ONNX preview / confirmation dialog -----------------------------
        preview_window_tag  = "cls_onnx_preview_window:"  + str(node_id)
        preview_name_tag    = "cls_onnx_preview_name:"    + str(node_id)
        preview_details_tag = "cls_onnx_preview_details:" + str(node_id)
        preview_status_tag  = "cls_onnx_preview_status:"  + str(node_id)
        preview_confirm_tag = "cls_onnx_preview_confirm:" + str(node_id)
        preview_cancel_tag  = "cls_onnx_preview_cancel:"  + str(node_id)
        preview_quit_tag    = "cls_onnx_preview_quit:"    + str(node_id)

        node.tag_preview_window  = preview_window_tag
        node.tag_preview_name    = preview_name_tag
        node.tag_preview_details = preview_details_tag
        node.tag_preview_status  = preview_status_tag
        node.tag_preview_confirm = preview_confirm_tag
        node.tag_preview_cancel  = preview_cancel_tag
        node.tag_preview_quit    = preview_quit_tag

        def _on_upload_confirm(sender, app_data, user_data):
            node._do_confirm_upload()

        def _on_close_preview(sender, app_data, user_data):
            node._close_upload_preview()

        with dpg.window(
            label="ONNX Model Preview",
            tag=preview_window_tag,
            modal=True,
            show=False,
            width=430,
            no_close=True,
        ):
            dpg.add_text("Model name (editable):")
            dpg.add_input_text(tag=preview_name_tag, width=410)
            dpg.add_separator()
            dpg.add_group(tag=preview_details_tag)
            dpg.add_separator()
            dpg.add_text("", tag=preview_status_tag)
            dpg.add_spacer(height=4)
            with dpg.group(horizontal=True):
                dpg.add_button(label="  Confirm Upload  ", tag=preview_confirm_tag, callback=_on_upload_confirm)
                dpg.add_spacer(width=10)
                dpg.add_button(label="  Cancel  ", tag=preview_cancel_tag, callback=_on_close_preview)
                dpg.add_spacer(width=10)
                dpg.add_button(label="  Quit  ", tag=preview_quit_tag, callback=_on_close_preview, show=False)

        # ノード
        with dpg.node(
                tag=tag_node_name,
                parent=parent,
                label=node.node_label,
                pos=pos,
        ):
            # 入力端子
            with dpg.node_attribute(
                    tag=tag_node_input01_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_text(
                    tag=tag_node_input01_value_name,
                    default_value='Input BGR image',
                )
            # 画像
            with dpg.node_attribute(
                    tag=tag_node_output01_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_image(tag_node_output01_value_name)
            # 使用アルゴリズム
            with dpg.node_attribute(
                    tag=tag_node_input02_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                def _on_model_changed(sender, app_data, user_data):
                    items = node._build_class_filter_items(
                        node._model_class_name_dict.get(app_data, {})
                    )
                    try:
                        dpg.configure_item(tag_class_filter_value_name, items=items, default_value='All')
                        dpg_set_value(tag_class_filter_value_name, 'All')
                    except Exception:
                        pass
                    try:
                        is_builtin = app_data in _BUILTIN_CLS_MODEL_NAMES
                        dpg.configure_item(node.tag_delete_btn, enabled=not is_builtin)
                    except Exception:
                        pass

                initial_model = list(node._model_class.keys())[0]
                dpg.add_combo(
                    list(node._model_class.keys()),
                    default_value=initial_model,
                    width=small_window_w,
                    tag=tag_node_input02_value_name,
                    callback=_on_model_changed,
                )
            # Class filter dropdown
            with dpg.node_attribute(
                    tag=tag_class_filter_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                initial_class_items = node._build_class_filter_items(
                    node._model_class_name_dict.get(initial_model, {})
                )
                dpg.add_combo(
                    initial_class_items,
                    default_value='All',
                    width=small_window_w,
                    tag=tag_class_filter_value_name,
                )
            # Confidence threshold slider
            with dpg.node_attribute(
                    tag=tag_score_threshold_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_slider_float(
                    tag=tag_score_threshold_value_name,
                    default_value=0.5,
                    min_value=0.0,
                    max_value=1.0,
                    width=small_window_w,
                )
            # Bounding box thickness slider
            with dpg.node_attribute(
                    tag=tag_bbox_thickness_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_slider_int(
                    tag=tag_bbox_thickness_value_name,
                    default_value=2,
                    min_value=1,
                    max_value=10,
                    width=small_window_w,
                )
            if use_gpu:
                # CPU/GPU切り替え
                with dpg.node_attribute(
                        tag=tag_provider_select_name,
                        attribute_type=dpg.mvNode_Attr_Static,
                ):
                    dpg.add_radio_button(
                        ("cpu", "cuda", "TRTcuda"),
                        tag=tag_provider_select_value_name,
                        default_value='cpu',
                        horizontal=True,
                    )
                # Batch size slider (GPU only)
                with dpg.node_attribute(
                        tag=tag_batch_size_name,
                        attribute_type=dpg.mvNode_Attr_Static,
                ):
                    dpg.add_slider_int(
                        label='Batch',
                        tag=tag_batch_size_value_name,
                        default_value=node.DEFAULT_BATCH_SIZE,
                        min_value=1,
                        max_value=node.MAX_BATCH_SIZE,
                        width=small_window_w - 80,
                    )
            # 処理時間
            if use_pref_counter:
                with dpg.node_attribute(
                        tag=tag_node_output02_name,
                        attribute_type=dpg.mvNode_Attr_Output,
                ):
                    dpg.add_text(
                        tag=tag_node_output02_value_name,
                        default_value='elapsed time(ms)',
                    )

            # JSON output button
            with dpg.node_attribute(
                    tag=tag_node_output_json_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                btn = dpg.add_button(
                    label="JSON",
                    tag=tag_node_output_json_value_name,
                    width=small_window_w,
                    enabled=False,
                )
                dpg.bind_item_theme(btn, yellow_button_theme)

            # ---- Add Model button (yellow, opens ONNX upload dialog) --------
            node.tag_upload_btn = tag_node_name + ':UploadONNX'

            def _on_upload_clicked(sender, app_data, user_data):
                dpg.show_item(onnx_file_dialog_tag)

            with dpg.node_attribute(
                    tag=tag_node_name + ':UploadAttr',
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                with dpg.theme() as add_model_btn_theme:
                    with dpg.theme_component(dpg.mvButton):
                        dpg.add_theme_color(dpg.mvThemeCol_Button, (255, 220, 0, 255))
                        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (255, 235, 50, 255))
                        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (220, 190, 0, 255))
                        dpg.add_theme_color(dpg.mvThemeCol_Text, (0, 0, 0, 255))

                add_model_btn = dpg.add_button(
                    label=u"Add Model",
                    tag=node.tag_upload_btn,
                    width=small_window_w,
                    callback=_on_upload_clicked,
                )
                dpg.bind_item_theme(add_model_btn, add_model_btn_theme)

            # ---- Delete model button (removes the selected model) -----------
            node.tag_delete_btn = tag_node_name + ':DeleteONNX'

            def _on_delete_clicked(sender, app_data, user_data):
                model_combo_tag = node.tag_node_name + ':' + node.TYPE_TEXT + ':Input02Value'
                selected = dpg_get_value(model_combo_tag)
                node._delete_selected_model(selected)

            with dpg.node_attribute(
                    tag=tag_node_name + ':DeleteAttr',
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                with dpg.theme() as delete_model_btn_theme:
                    with dpg.theme_component(dpg.mvButton):
                        dpg.add_theme_color(dpg.mvThemeCol_Button, (200, 60, 60, 255))
                        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (230, 90, 90, 255))
                        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (170, 40, 40, 255))
                        dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255, 255))

                delete_btn = dpg.add_button(
                    label=u"Delete Model",
                    tag=node.tag_delete_btn,
                    width=small_window_w,
                    callback=_on_delete_clicked,
                )
                dpg.bind_item_theme(delete_btn, delete_model_btn_theme)
                # Disable for the initially-selected model if it is built-in
                if initial_model in _BUILTIN_CLS_MODEL_NAMES:
                    dpg.configure_item(delete_btn, enabled=False)

        node.tag_node_name = tag_node_name
        return node



class Node(Node):
    _ver = '0.0.1'

    node_label = 'Classification'
    node_tag = 'Classification'

    DEFAULT_BATCH_SIZE = 1
    MAX_BATCH_SIZE = 32

    _opencv_setting_dict = None

    # モデル設定
    _model_class = {
        'MobileNetV3 Small': MobileNetV3,
        'MobileNetV3 Large': MobileNetV3,
        'EfficientNet B0': EfficientNetB0,
        'ResNet50': ResNet50,
        'Yolo-cls': YoloCls,
        'Gender Recognition': GenderRecognition,
        'Pedestrian Gender': PedestrianGender,
    }
    _model_base_path = os.path.dirname(os.path.abspath(__file__)) + '/classification/'
    # pedestrian_gender.onnx lives at the repository root
    _repo_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    _model_path_setting = {
        'MobileNetV3 Small':
        _model_base_path + 'MobileNetV3/model/MobileNetV3Small.onnx',
        'MobileNetV3 Large':
        _model_base_path + 'MobileNetV3/model/MobileNetV3Large.onnx',
        'EfficientNet B0':
        _model_base_path + 'EfficientNetB0/model/EfficientNetB0.onnx',
        'ResNet50':
        _model_base_path + 'ResNet50/model/ResNet50.onnx',
        'Yolo-cls':
        _model_base_path + 'Yolo-cls/model/son.onnx',
        'Gender Recognition':
        _model_base_path + 'GenderRecognition/model/GenderRecognition.onnx',
        'Pedestrian Gender':
        os.path.join(_repo_root, 'pedestrian_gender.onnx'),
    }
    _model_class_name_dict = {
        'MobileNetV3 Small': imagenet_class_names,
        'MobileNetV3 Large': imagenet_class_names,
        'EfficientNet B0': imagenet_class_names,
        'ResNet50': imagenet_class_names,
        'Yolo-cls': esc50_class_names,
        'Gender Recognition': gender_class_names,
        'Pedestrian Gender': pedestrian_gender_class_names,
    }

    _model_instance = {}
    _class_name_dict = None

    def __init__(self):
        pass

    @staticmethod
    def _build_class_filter_items(class_name_dict):
        """Build the list of items for the class filter combo."""
        return ['All'] + [
            f"{idx}: {label}"
            for idx, label in sorted(class_name_dict.items(), key=lambda x: x[0])
        ]

    @classmethod
    def _load_custom_models_from_registry(cls):
        """Load user-uploaded models from the registry into the class dicts."""
        try:
            entries = _cls_registry.load_registry()
        except Exception as exc:
            logger.warning(f"[Classification] Could not load custom models registry: {exc}")
            return
        for entry in entries:
            name = entry.get('name', '')
            path = entry.get('path', '')
            if not name or not path:
                continue
            if name in cls._model_class:
                continue
            in_w = int(entry.get('input_width', 224))
            in_h = int(entry.get('input_height', 224))
            num_classes = int(entry.get('num_classes', 0))
            raw_class_names = entry.get('class_names', {})
            class_names = {int(k): str(v) for k, v in raw_class_names.items()} if raw_class_names else {}
            if not class_names and num_classes == 0 and os.path.isfile(path):
                # Legacy registry entry with no class info: re-inspect the ONNX to deduce
                try:
                    meta = onnx_inspector.inspect_onnx_model(path)
                    num_classes = meta.get('num_classes', 0)
                    class_names = meta.get('class_names', {})
                    if not class_names and num_classes > 0:
                        class_names = {i: f"class_{i}" for i in range(num_classes)}
                    if num_classes > 0:
                        updated_entry = dict(entry)
                        updated_entry['num_classes'] = num_classes
                        updated_entry['class_names'] = {str(k): v for k, v in class_names.items()}
                        try:
                            _cls_registry.save_entry(updated_entry)
                        except Exception:
                            pass
                except Exception as exc:
                    logger.warning(f"[Classification] Could not re-inspect {path} for class names: {exc}")
            if not class_names and num_classes > 0:
                class_names = {i: f"class_{i}" for i in range(num_classes)}
            cls._register_custom_model(name, path, in_w, in_h, class_names)
            logger.info(f"[Classification] Loaded custom model from registry: {name}")

    @classmethod
    def _register_custom_model(cls, name, path, in_w, in_h, class_names=None):
        """Add a custom ONNX model to the class-level runtime dictionaries."""
        if class_names is None:
            class_names = {}

        def _make_factory(p, w, h):
            def factory(model_path, providers=None):
                if providers is None:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                return CustomONNXClassification(
                    model_path=p, input_width=w, input_height=h, providers=providers
                )
            return factory

        cls._model_class[name] = _make_factory(path, in_w, in_h)
        cls._model_path_setting[name] = path
        cls._model_class_name_dict[name] = class_names if class_names else {0: 'class_0'}

    # ------------------------------------------------------------------
    # Delete model
    # ------------------------------------------------------------------
    @classmethod
    def _delete_custom_model(cls, name: str) -> bool:
        """Remove a custom model from the persistent registry and runtime dicts.

        Built-in models cannot be deleted.
        Returns True when the model was found and removed.
        """
        if not name:
            return False
        if name in _BUILTIN_CLS_MODEL_NAMES:
            logger.warning(f"[Delete] '{name}' is a built-in model and cannot be deleted.")
            return False
        found = name in cls._model_class or name in cls._model_path_setting
        # Delete the ONNX file from disk if it lives in the uploads directory
        onnx_path = cls._model_path_setting.get(name)
        if onnx_path:
            try:
                onnx_abs = os.path.abspath(onnx_path)
                uploads_abs = os.path.abspath(_CLS_UPLOADS_DIR)
                if (os.path.commonpath([onnx_abs, uploads_abs]) == uploads_abs
                        and onnx_abs != uploads_abs
                        and os.path.isfile(onnx_abs)):
                    os.remove(onnx_abs)
                    logger.info(f"[Delete] ONNX file deleted: {onnx_abs}")
            except Exception as exc:
                logger.warning(f"[Delete] Could not delete ONNX file for '{name}': {exc}")
        try:
            _cls_registry.remove_entry(name)
            logger.info(f"[Delete] Registry entry removed for '{name}'.")
        except Exception as exc:
            logger.warning(f"[Delete] Could not remove registry entry for '{name}': {exc}")
        cls._model_class.pop(name, None)
        cls._model_path_setting.pop(name, None)
        cls._model_class_name_dict.pop(name, None)
        # Drop any cached inference instances for this model
        for key in [k for k in cls._model_instance if k == name or k.startswith(name + '_')]:
            cls._model_instance.pop(key, None)
        return found

    def _delete_selected_model(self, name: str):
        """Delete the currently-selected model and refresh the node dropdowns."""
        if not name:
            logger.warning("[Delete] No model selected — nothing to delete.")
            return
        if name in _BUILTIN_CLS_MODEL_NAMES:
            logger.warning(f"[Delete] '{name}' is a built-in model and cannot be deleted.")
            return
        Node._delete_custom_model(name)

        # Update the model combobox with the remaining models
        model_combo_tag = self.tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'
        remaining = list(Node._model_class.keys())
        new_default = remaining[0] if remaining else ""
        try:
            dpg.configure_item(model_combo_tag, items=remaining, default_value=new_default)
            dpg_set_value(model_combo_tag, new_default)
            logger.info(f"[Delete] Model '{name}' deleted — '{new_default}' now selected.")
        except Exception as exc:
            logger.warning(f"[Delete] Could not update model dropdown: {exc}")

        # Refresh the class filter dropdown for the new selection
        try:
            if new_default and new_default in Node._model_class_name_dict:
                class_items = Node._build_class_filter_items(
                    Node._model_class_name_dict[new_default]
                )
            else:
                class_items = ['All']
            class_filter_tag = self.tag_node_name + ':' + self.TYPE_TEXT + ':ClassFilterValue'
            dpg.configure_item(class_filter_tag, items=class_items, default_value='All')
            dpg_set_value(class_filter_tag, 'All')
        except Exception as exc:
            logger.warning(f"[Delete] Could not update class filter dropdown: {exc}")

        # Update delete button state for the new selection
        try:
            is_builtin = new_default in _BUILTIN_CLS_MODEL_NAMES
            dpg.configure_item(self.tag_delete_btn, enabled=not is_builtin)
        except Exception as exc:
            logger.warning(f"[Delete] Could not update delete button state: {exc}")

    # ------------------------------------------------------------------
    # Upload callbacks
    # ------------------------------------------------------------------

    def _callback_onnx_select(self, sender, data, user_data=None):
        """Handle ONNX file selection from the file dialog."""
        if data.get("file_name") == ".":
            return
        onnx_path = data.get("file_path_name", "")
        if not onnx_path or not os.path.isfile(onnx_path):
            return

        try:
            meta = onnx_inspector.inspect_onnx_model(onnx_path)
        except Exception as exc:
            logger.error(f"[Classification Upload] ONNX inspection failed: {exc}")
            self._pending_onnx_path = None
            self._pending_meta = None
            try:
                dpg.delete_item(self.tag_preview_details, children_only=True)
                dpg.add_text(
                    f"Inspection error: {exc}",
                    parent=self.tag_preview_details,
                    color=(255, 100, 100, 255),
                )
                self._set_upload_preview_actions(upload_succeeded=False)
                dpg.set_value(self.tag_preview_status, "")
                dpg.show_item(self.tag_preview_window)
            except Exception:
                pass
            return

        self._pending_onnx_path = onnx_path
        self._pending_meta = meta

        base_name = os.path.splitext(os.path.basename(onnx_path))[0]
        dpg.set_value(self.tag_preview_name, base_name)

        dpg.delete_item(self.tag_preview_details, children_only=True)
        in_w = meta.get("input_width", 224)
        in_h = meta.get("input_height", 224)
        num_cls = meta.get("num_classes", 0)
        class_names = meta.get("class_names", {})
        dpg.add_text(f"Input dimensions : {in_w} x {in_h} px (W x H)", parent=self.tag_preview_details)
        dpg.add_text(f"Number of classes: {num_cls}", parent=self.tag_preview_details)
        if class_names:
            max_show = 10
            dpg.add_text("Class list (first 10):", parent=self.tag_preview_details)
            for cid, cname in sorted(class_names.items(), key=lambda x: x[0])[:max_show]:
                dpg.add_text(f"  {cid}: {cname}", parent=self.tag_preview_details)

        self._set_upload_preview_actions(upload_succeeded=False)
        dpg.set_value(self.tag_preview_status, "")
        dpg.show_item(self.tag_preview_window)

    def _set_upload_preview_actions(self, upload_succeeded: bool):
        dpg.configure_item(self.tag_preview_confirm, show=not upload_succeeded)
        dpg.configure_item(self.tag_preview_cancel, show=not upload_succeeded)
        dpg.configure_item(self.tag_preview_quit, show=upload_succeeded)

    def _close_upload_preview(self):
        dpg.hide_item(self.tag_preview_window)

    def _do_confirm_upload(self):
        onnx_path = getattr(self, '_pending_onnx_path', None)
        meta = getattr(self, '_pending_meta', None)
        if not onnx_path or meta is None:
            dpg.set_value(self.tag_preview_status, "No pending upload — please select a file first.")
            return

        custom_name = dpg.get_value(self.tag_preview_name).strip()
        if not custom_name:
            custom_name = os.path.splitext(os.path.basename(onnx_path))[0]

        os.makedirs(_CLS_UPLOADS_DIR, exist_ok=True)
        dest_path = onnx_path
        try:
            basename = os.path.basename(onnx_path)
            candidate = os.path.join(_CLS_UPLOADS_DIR, basename)
            if os.path.abspath(onnx_path) != os.path.abspath(candidate):
                shutil.copy2(onnx_path, candidate)
                dest_path = candidate
        except Exception as exc:
            logger.warning(f"[Classification Upload] Could not copy ONNX: {exc}")

        try:
            Node._finalise_upload(self, dest_path, meta, custom_name=custom_name)
            dpg.set_value(self.tag_preview_status, f"\u2713 Model '{custom_name}' uploaded successfully!")
            self._set_upload_preview_actions(upload_succeeded=True)
        except Exception as exc:
            logger.error(f"[Classification Upload] Finalise failed: {exc}", exc_info=True)
            dpg.set_value(self.tag_preview_status, f"\u2717 Upload failed: {exc}")
            self._set_upload_preview_actions(upload_succeeded=False)

        self._pending_onnx_path = None
        self._pending_meta = None

    @staticmethod
    def _finalise_upload(node, onnx_path: str, meta: dict, custom_name: str = None):
        base = custom_name if custom_name else os.path.splitext(os.path.basename(onnx_path))[0]
        name = base
        counter = 1
        while name in Node._model_class:
            name = f"{base}_{counter}"
            counter += 1

        in_w = meta.get("input_width", 224)
        in_h = meta.get("input_height", 224)
        num_classes = meta.get("num_classes", 0)
        class_names = meta.get("class_names", {})
        if not class_names and num_classes > 0:
            class_names = {i: f"class_{i}" for i in range(num_classes)}

        Node._register_custom_model(name, onnx_path, in_w, in_h, class_names)

        registry_entry = {
            "name": name,
            "path": onnx_path,
            "input_width": in_w,
            "input_height": in_h,
            "output_format": meta.get("output_format", "unknown"),
            "num_classes": num_classes,
            "class_names": {str(k): v for k, v in class_names.items()},
        }
        try:
            _cls_registry.save_entry(registry_entry)
        except Exception as exc:
            logger.warning(f"[Classification Upload] Could not save registry entry: {exc}")

        model_combo_tag = node.tag_node_name + ':' + node.TYPE_TEXT + ':Input02Value'
        try:
            current_items = dpg.get_item_configuration(model_combo_tag).get("items", [])
            if name not in current_items:
                current_items = list(current_items) + [name]
            dpg.configure_item(model_combo_tag, items=current_items, default_value=name)
        except Exception as exc:
            logger.warning(f"[Classification Upload] Could not update model dropdown: {exc}")

        class_filter_tag = node.tag_node_name + ':' + node.TYPE_TEXT + ':ClassFilterValue'
        try:
            new_class_items = Node._build_class_filter_items(class_names)
            dpg.configure_item(class_filter_tag, items=new_class_items, default_value='All')
            dpg_set_value(class_filter_tag, 'All')
        except Exception as exc:
            logger.warning(f"[Classification Upload] Could not update class filter dropdown: {exc}")

        # Enable delete button — newly uploaded model is always custom (not built-in)
        try:
            dpg.configure_item(node.tag_delete_btn, enabled=True)
        except Exception as exc:
            logger.warning(f"[Classification Upload] Could not enable delete button: {exc}")

    def update(
        self,
        node_id,
        connection_list,
        node_image_dict,
        node_result_dict,
        node_audio_dict,
    ):
        tag_node_name = str(node_id) + ':' + self.node_tag
        input_value02_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'
        output_value01_tag = tag_node_name + ':' + self.TYPE_IMAGE + ':Output01Value'
        output_value02_tag = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02Value'
        tag_score_threshold_value = tag_node_name + ':' + self.TYPE_FLOAT + ':ScoreThresholdValue'
        tag_bbox_thickness_value = tag_node_name + ':' + self.TYPE_INT + ':BboxThicknessValue'
        tag_class_filter_value = tag_node_name + ':' + self.TYPE_TEXT + ':ClassFilterValue'
        tag_batch_size_value = tag_node_name + ':' + self.TYPE_INT + ':BatchSizeValue'

        tag_provider_select_value_name = tag_node_name + ':' + self.TYPE_IMAGE + ':ProviderValue'

        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']
        use_gpu = self._opencv_setting_dict['use_gpu']

        # 接続情報確認
        src_node_name = ''
        connection_info_src = ''
        for connection_info in connection_list:
            connection_type = connection_info[0].split(':')[2]
            if connection_type == self.TYPE_INT:
                # 接続タグ取得
                source_tag = connection_info[0] + 'Value'
                destination_tag = connection_info[1] + 'Value'
                # 値更新
                input_value = int(dpg_get_value(source_tag))
                input_value = max([self._min_val, input_value])
                input_value = min([self._max_val, input_value])
                dpg_set_value(destination_tag, input_value)
            if connection_type == self.TYPE_IMAGE or connection_type == self.TYPE_AUDIO:
                # 画像取得元のノード名(ID付き)を取得
                connection_info_src = connection_info[0]
                connection_info_src = connection_info_src.split(':')[:2]
                src_node_name = connection_info_src[1]
                connection_info_src = ':'.join(connection_info_src)

        # 画像取得
        frame = self.get_input_frame(connection_list, node_image_dict, node_audio_dict)

        # CPU/GPU選択状態取得
        provider = 'cpu'
        if use_gpu:
            provider = dpg_get_value(tag_provider_select_value_name)

        # Batch size (cuda and TRTcuda)
        batch_size = self.DEFAULT_BATCH_SIZE
        if use_gpu and provider in ('cuda', 'TRTcuda'):
            try:
                batch_size = int(dpg_get_value(tag_batch_size_value))
            except Exception:
                batch_size = self.DEFAULT_BATCH_SIZE
        batch_size = max(1, min(self.MAX_BATCH_SIZE, batch_size))
        batch_mode = batch_size > 1

        # モデル情報取得
        model_name = dpg_get_value(input_value02_tag)
        model_path = self._model_path_setting[model_name]
        model_class = self._model_class[model_name]

        class_name_dict = self._model_class_name_dict[model_name]

        model_name_with_provider = model_name + '_' + provider

        # モデル取得
        if frame is not None:
            if model_name_with_provider not in self._model_instance:
                if provider == 'cpu':
                    ort_providers = ['CPUExecutionProvider']
                    self._model_instance[
                        model_name_with_provider] = model_class(
                            model_path,
                            providers=ort_providers,
                        )
                elif provider == 'TRTcuda':
                    self._model_instance[
                        model_name_with_provider] = model_class(
                            model_path,
                            providers=['TensorrtExecutionProvider',
                                       'CUDAExecutionProvider',
                                       'CPUExecutionProvider'],
                        )
                else:
                    self._model_instance[
                        model_name_with_provider] = model_class(model_path)

        # 計測開始
        if frame is not None and use_pref_counter:
            start_time = time.monotonic()

        # OD class filter: read before the crop loop so only matching bboxes are inferred
        od_class_filter_value = 'All'
        try:
            od_class_filter_value = dpg_get_value(tag_class_filter_value) or 'All'
        except Exception:
            pass
        selected_od_class_id = None
        if od_class_filter_value != 'All' and ':' in od_class_filter_value:
            try:
                selected_od_class_id = int(od_class_filter_value.split(':')[0].strip())
            except Exception:
                selected_od_class_id = None

        # 接続元がObjectDetectionノードの場合、各バウンディングボックスに対して推論
        result = {}
        frame_list, class_id_list, score_list = [], [], []
        od_target_bboxes = []
        od_target_scores = []
        od_target_class_ids = []
        if frame is not None:
            if src_node_name == 'ObjectDetection':
                # 物体検出情報取得
                node_result = node_result_dict.get(connection_info_src, [])
                od_bboxes = node_result.get('bboxes', [])
                od_scores = node_result.get('scores', [])
                od_class_ids = node_result.get('class_ids', [])
                od_class_names = node_result.get('class_names', [])
                od_score_th = node_result.get('score_th', [])
                # Use the clean (pre-annotation) frame from the OD node for cropping
                # and as the drawing base, so OD boxes/labels don't leak into the output.
                od_raw_frame = node_result.get('raw_frame', None)

                # Update dropdown with OD class names so user can filter by OD class
                try:
                    if isinstance(od_class_names, dict) and od_class_names:
                        od_filter_items = self._build_class_filter_items(od_class_names)
                        current_items = dpg.get_item_configuration(tag_class_filter_value).get('items', [])
                        if current_items != od_filter_items:
                            dpg.configure_item(tag_class_filter_value, items=od_filter_items)
                except Exception:
                    pass

                # バウンディングボックスで切り抜き (selected OD class only)
                # Crop from the clean raw frame when available to avoid including OD annotations.
                crop_source = od_raw_frame if od_raw_frame is not None else frame
                for od_bbox, od_score, od_class_id in zip(
                        od_bboxes, od_scores, od_class_ids):
                    x1, y1 = int(od_bbox[0]), int(od_bbox[1])
                    x2, y2 = int(od_bbox[2]), int(od_bbox[3])

                    if od_score_th > od_score:
                        continue

                    # Skip bboxes that don't match the selected OD class
                    if selected_od_class_id is not None and int(od_class_id) != selected_od_class_id:
                        continue

                    frame_list.append(copy.deepcopy(crop_source[y1:y2, x1:x2]))
                    od_target_bboxes.append([x1, y1, x2, y2])
                    od_target_scores.append(od_score)
                    od_target_class_ids.append(od_class_id)

                # 各バウンディングボックスに対しClassification推論
                if batch_mode and len(frame_list) > 0:
                    batch_scores, batch_ids = run_unlocked(
                        self._run_batch_inference,
                        self._model_instance[model_name_with_provider],
                        frame_list,
                    )
                    score_list = list(batch_scores)
                    class_id_list = list(batch_ids)
                else:
                    for temp_frame in frame_list:
                        class_scores, class_ids = run_unlocked(
                            self._model_instance[model_name_with_provider],
                            temp_frame,
                        )
                        score_list.append(class_scores[0])
                        class_id_list.append(class_ids[0])
                result['use_object_detection'] = True
                result['class_ids'] = class_id_list
                result['class_scores'] = score_list
                result['class_names'] = class_name_dict
                result['od_bboxes'] = od_target_bboxes
                result['od_scores'] = od_target_scores
                result['od_class_ids'] = od_target_class_ids
                result['od_class_names'] = od_class_names
                result['od_score_th'] = od_score_th
                result['od_raw_frame'] = od_raw_frame
            else:
                class_scores, class_ids = run_unlocked(
                    self._model_instance[model_name_with_provider], frame
                )
                result['use_object_detection'] = False
                result['class_ids'] = class_ids.tolist()
                result['class_scores'] = class_scores.tolist()
                result['class_names'] = class_name_dict

        # 信頼度しきい値取得 (classification confidence threshold)
        score_threshold = 0.0
        try:
            score_threshold = float(dpg_get_value(tag_score_threshold_value))
        except Exception:
            pass
        if frame is not None:
            result['score_th'] = score_threshold

        # バウンディングボックス線幅取得
        bbox_thickness = 2
        try:
            bbox_thickness = int(dpg_get_value(tag_bbox_thickness_value))
        except Exception:
            pass

        # 計測終了
        if frame is not None and use_pref_counter:
            elapsed_time = time.monotonic() - start_time
            elapsed_time = int(elapsed_time * 1000)
            dpg_set_value(output_value02_tag,
                          str(elapsed_time).zfill(4) + 'ms')

        # 描画
        output_frame = frame
        if frame is not None:
            # When connected to OD, use the clean (pre-annotation) frame as the drawing
            # base so OD boxes/labels do not appear in the classification output texture.
            base_frame = frame
            if result.get('use_object_detection') and result.get('od_raw_frame') is not None:
                base_frame = result['od_raw_frame']

            debug_frame = copy.deepcopy(base_frame)
            
            # Draw labels on the original frame
            if result['use_object_detection']:
                debug_frame = self.draw_classification_with_od_info(
                    debug_frame,
                    result['class_ids'],
                    result['class_scores'],
                    result['class_names'],
                    result['od_bboxes'],
                    result['od_scores'],
                    result['od_class_ids'],
                    result['od_class_names'],
                    result['od_score_th'],
                    thickness=bbox_thickness,
                    score_threshold=score_threshold,
                )
            else:
                debug_frame = self.draw_classification_info(
                    debug_frame,
                    result['class_ids'],
                    result['class_scores'],
                    result['class_names'],
                    score_threshold=score_threshold,
                )
            
            output_frame = debug_frame

            texture = self.convert_cv_to_dpg(
                debug_frame,
                small_window_w,
                small_window_h,
            )
            dpg_set_value(output_value01_tag, texture)

        return {"image": output_frame, "json": result, "audio": None}

    def close(self, node_id):
        super().close(node_id)
    
    def draw_classification_info(
        self,
        image,
        class_ids,
        class_scores,
        class_names,
        score_threshold=0.0,
    ):
        """
        Override base class method to add color differentiation based on ranking.
        Position 1 (index 0, highest score): Red
        Position 2 (index 1): Yellow
        Position 3 (index 2): Blue
        Position 4 (index 3): Violet
        Position 5 (index 4): Magenta

        Labels whose score is below ``score_threshold`` are not drawn on the
        texture, but all predictions are still forwarded in the JSON output so
        that downstream chart nodes receive complete data.
        """
        debug_image = copy.deepcopy(image)
        
        # Define colors for top 5 positions (BGR format)
        rank_colors = [
            (0, 0, 255),      # Position 1 (index 0): Red (highest score)
            (0, 255, 255),    # Position 2 (index 1): Yellow
            (255, 0, 0),      # Position 3 (index 2): Blue
            (255, 0, 128),    # Position 4 (index 3): Violet (purple-ish)
            (255, 0, 255),    # Position 5 (index 4): Magenta
        ]
        
        draw_index = 0
        for class_score, class_id in zip(class_scores, class_ids):
            if float(class_score) < score_threshold:
                continue

            score = "%.2f" % class_score
            text = "%s:%s(%s)" % (str(class_id), str(class_names[int(class_id)]), score)
            
            if draw_index < len(rank_colors):
                color = rank_colors[draw_index]
            else:
                color = (0, 255, 0)  # Default green for lower rankings
            
            debug_image = cv2.putText(
                debug_image,
                text,
                (15, 25 + (draw_index * 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                thickness=2,
            )
            draw_index += 1

        return debug_image

    def draw_classification_with_od_info(
        self,
        image,
        class_id_list,
        score_list,
        class_name_dict,
        od_bboxes,
        od_scores,
        od_class_ids,
        od_class_names,
        od_score_th,
        thickness=2,
        score_threshold=0.0,
    ):
        """Draw bounding boxes colored by classification class; no OD label.

        Only bboxes whose classification score meets ``score_threshold`` are
        drawn on the image.
        """
        debug_image = copy.deepcopy(image)

        for class_id, score, od_bbox, od_score, od_class_id in zip(
            class_id_list,
            score_list,
            od_bboxes,
            od_scores,
            od_class_ids,
        ):
            x1, y1 = int(od_bbox[0]), int(od_bbox[1])
            x2, y2 = int(od_bbox[2]), int(od_bbox[3])

            if od_score_th > od_score:
                continue

            # Skip boxes whose classification score is below threshold
            if float(score) < score_threshold:
                continue

            # Color box by classification class, not OD class
            color = self.get_color(class_id)

            debug_image = cv2.rectangle(
                debug_image,
                (x1, y1),
                (x2, y2),
                color,
                thickness=thickness,
            )

            # Classification label only (no OD Detection label)
            score_text = '%.2f' % score
            class_name = self.get_class_name(class_id, class_name_dict)
            text = f'{int(class_id)}:{class_name}({score_text})'
            debug_image = cv2.putText(
                debug_image,
                text,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                thickness=2,
            )

        return debug_image

    @staticmethod
    def _run_batch_inference(model_instance, frame_list, top_k=5):
        """Run classification inference on a batch of frames in a single ONNX call.

        Falls back to sequential inference if the model does not expose
        ``onnx_session`` or ``input_shape``.

        Returns:
            top_scores: list of top-1 score per frame
            top_ids:    list of top-1 class-id per frame
        """
        session = getattr(model_instance, 'onnx_session', None)
        input_shape = getattr(model_instance, 'input_shape', None)

        if session is None or input_shape is None or len(frame_list) == 0:
            # Fallback: sequential
            top_scores, top_ids = [], []
            for f in frame_list:
                scores, ids = model_instance(f)
                top_scores.append(scores[0])
                top_ids.append(ids[0])
            return top_scores, top_ids

        h, w = int(input_shape[0]), int(input_shape[1])
        preprocessed = []
        for f in frame_list:
            img = cv2.resize(f, (w, h))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            preprocessed.append(img.astype('float32'))

        batch = np.stack(preprocessed, axis=0)  # [N, H, W, C]

        input_name = session.get_inputs()[0].name
        try:
            outputs = session.run(None, {input_name: batch})
            results = np.array(outputs[0])  # [N, num_classes]
        except Exception as exc:
            # Fallback on batch-unsupported models
            logger.warning(f"[Classification] Batch inference failed, falling back to sequential: {exc}")
            top_scores, top_ids = [], []
            for f in frame_list:
                scores, ids = model_instance(f)
                top_scores.append(scores[0])
                top_ids.append(ids[0])
            return top_scores, top_ids

        top_scores, top_ids = [], []
        for row in results:
            sorted_idx = np.argsort(row)[::-1][:top_k]
            top_scores.append(float(row[sorted_idx[0]]))
            top_ids.append(int(sorted_idx[0]))
        return top_scores, top_ids

    def get_setting_dict(self, node_id):
        tag_node_name = str(node_id) + ':' + self.node_tag
        input_value02_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'

        # 選択モデル
        model_name = dpg_get_value(input_value02_tag)
        tag_score_threshold_value = tag_node_name + ':' + self.TYPE_FLOAT + ':ScoreThresholdValue'
        score_threshold = dpg_get_value(tag_score_threshold_value)
        tag_bbox_thickness_value = tag_node_name + ':' + self.TYPE_INT + ':BboxThicknessValue'
        bbox_thickness = dpg_get_value(tag_bbox_thickness_value)
        tag_batch_size_value = tag_node_name + ':' + self.TYPE_INT + ':BatchSizeValue'
        batch_size = self.DEFAULT_BATCH_SIZE
        try:
            batch_size = int(dpg_get_value(tag_batch_size_value))
        except Exception:
            pass
        batch_size = max(1, min(self.MAX_BATCH_SIZE, batch_size))

        tag_class_filter_value = tag_node_name + ':' + self.TYPE_TEXT + ':ClassFilterValue'
        class_filter = 'All'
        try:
            class_filter = dpg_get_value(tag_class_filter_value) or 'All'
        except Exception:
            pass

        pos = dpg.get_item_pos(tag_node_name)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos
        setting_dict[input_value02_tag] = model_name
        setting_dict[tag_score_threshold_value] = score_threshold
        setting_dict[tag_bbox_thickness_value] = bbox_thickness
        setting_dict[tag_batch_size_value] = int(batch_size)
        setting_dict[tag_class_filter_value] = class_filter

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        tag_node_name = str(node_id) + ':' + self.node_tag
        input_value02_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'
        tag_score_threshold_value = tag_node_name + ':' + self.TYPE_FLOAT + ':ScoreThresholdValue'
        tag_bbox_thickness_value = tag_node_name + ':' + self.TYPE_INT + ':BboxThicknessValue'
        tag_batch_size_value = tag_node_name + ':' + self.TYPE_INT + ':BatchSizeValue'
        tag_class_filter_value = tag_node_name + ':' + self.TYPE_TEXT + ':ClassFilterValue'

        model_name = setting_dict[input_value02_tag]
        dpg_set_value(input_value02_tag, model_name)

        # Rebuild the class filter dropdown items to match the restored model,
        # since the DPG callback only fires on user interaction, not on
        # programmatic dpg_set_value calls.
        try:
            class_name_dict = Node._model_class_name_dict.get(model_name, {})
            class_items = Node._build_class_filter_items(class_name_dict)
            dpg.configure_item(tag_class_filter_value, items=class_items)
        except Exception:
            pass

        # Restore the previously-selected class filter value (saved in newer versions).
        # Fall back to 'All' for sessions saved before this field existed.
        if tag_class_filter_value in setting_dict:
            saved_filter = setting_dict[tag_class_filter_value]
            try:
                dpg_set_value(tag_class_filter_value, saved_filter)
            except Exception:
                pass
        else:
            try:
                dpg_set_value(tag_class_filter_value, 'All')
            except Exception:
                pass

        # Update the delete button state for the restored model
        try:
            is_builtin = model_name in _BUILTIN_CLS_MODEL_NAMES
            dpg.configure_item(self.tag_delete_btn, enabled=not is_builtin)
        except Exception:
            pass

        if tag_score_threshold_value in setting_dict:
            dpg_set_value(tag_score_threshold_value, float(setting_dict[tag_score_threshold_value]))

        if tag_bbox_thickness_value in setting_dict:
            dpg_set_value(tag_bbox_thickness_value, int(setting_dict[tag_bbox_thickness_value]))

        if tag_batch_size_value in setting_dict:
            try:
                dpg_set_value(tag_batch_size_value, max(1, min(self.MAX_BATCH_SIZE, int(setting_dict[tag_batch_size_value]))))
            except Exception:
                pass


# Load user-uploaded custom models from registry at import time
Node._load_custom_models_from_registry()
