#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import os
import sys
import shutil

import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value, run_unlocked

from node.node_abc import DpgNodeABC
#from node_editor.util import convert_cv_to_dpg

from node.DLNode.semantic_segmentation.deeplab_v3.deeplab_v3 import DeepLabV3
from node.DLNode.semantic_segmentation.road_segmentation_adas_0001.road_segmentation import RoadSegmentation
from node.DLNode.semantic_segmentation.skin_clothes_hair_segmentation.skin_clothes_hair_segmentation import SkinClothesHairSegmentation
from node.DLNode.semantic_segmentation.mediapipe_selfie_segmentation.mediapipe_selfie_segmentation import (
    MediaPipeSelfieSegmentationNormal,
    MediaPipeSelfieSegmentationLandScape,
)
from node.DLNode.semantic_segmentation.yolov8_seg.yolov8_seg import YOLOv8Seg
from node.DLNode.semantic_segmentation.CustomONNX.custom_onnx import CustomONNX as CustomONNXSeg
from node.DLNode.semantic_segmentation.aerial_segmentation_flair.aerial_segmentation_flair import (
    FlairAerialSegmentation,
    FlairAerialSegmentationONNX,
    colorize_flair_mask,
    overlay_flair,
    overlay_flair2,
)
from node.DLNode.semantic_segmentation.pothole.pothole_seg import PotholeYOLOSeg, PotholeYOLOSegV12
from node.DLNode.semantic_segmentation import custom_models_registry as _seg_registry
from node.DLNode.object_detection import onnx_inspector

from node.basenode import Node

from src.utils.logging import get_logger
logger = get_logger(__name__)

_SEG_BASE = os.path.dirname(os.path.abspath(__file__))

if getattr(sys, 'frozen', False):
    from src.utils.paths import get_models_dir
    _SEG_UPLOADS_DIR = get_models_dir('semantic_segmentation')
else:
    _SEG_UPLOADS_DIR = os.path.join(
        _SEG_BASE, 'semantic_segmentation', 'CustomONNX', 'models'
    )

# Built-in model names — these cannot be deleted by the user.
_BUILTIN_SEG_MODEL_NAMES: set = {
    'DeepLabV3',
    'Road Segmentation ADAS 0001',
    'Skin Clothes Hair Segmentation',
    'MediaPipe SelfieSegmentation(Normal)',
    'MediaPipe SelfieSegmentation(LandScape)',
    'YOLOv8-nano-seg',
    'FLAIR Aerial (IGN)',
    'FLAIR Aerial INT8 (ONNX)',
    'Pothole YOLO-seg',
    'Pothole YOLO-seg (v12)',
}

class FactoryNode:
    node_label = 'SemanticSegmentation'
    node_tag = 'SemanticSegmentation'
    

    def __init__(self):
        pass

    
    def add_node(self, parent, node_id, pos=[0, 0], opencv_setting_dict=None, callback=None):
        """Adds a node to the processing graph."""
        node = Node()
        return node.add_node(parent, node_id, pos, opencv_setting_dict, callback)

class Node(Node):
    _ver = '0.0.1'

    node_label = 'Semantic Segmentation'
    node_tag = 'SemanticSegmentation'

    _min_val = 0.0
    _max_val = 1.0

    _opencv_setting_dict = None

    # モデル設定
    _model_class = {
        'DeepLabV3':
        DeepLabV3,
        'Road Segmentation ADAS 0001':
        RoadSegmentation,
        'Skin Clothes Hair Segmentation':
        SkinClothesHairSegmentation,
        'MediaPipe SelfieSegmentation(Normal)':
        MediaPipeSelfieSegmentationNormal,
        'MediaPipe SelfieSegmentation(LandScape)':
        MediaPipeSelfieSegmentationLandScape,
        'YOLOv8-nano-seg':
        YOLOv8Seg,
        'FLAIR Aerial (IGN)':
        FlairAerialSegmentation,
        'FLAIR Aerial INT8 (ONNX)':
        FlairAerialSegmentationONNX,
        'Pothole YOLO-seg':
        PotholeYOLOSeg,
        'Pothole YOLO-seg (v12)':
        PotholeYOLOSegV12,
    }
    _model_base_path = os.path.dirname(os.path.abspath(__file__)) + '/semantic_segmentation/'
    _model_path_setting = {
        'DeepLabV3':
        _model_base_path + 'deeplab_v3/model/deeplab_v3_1_default_1.onnx',
        'Road Segmentation ADAS 0001': _model_base_path +
        'road_segmentation_adas_0001/saved_model/model_float32.onnx',
        'Skin Clothes Hair Segmentation': _model_base_path +
        'skin_clothes_hair_segmentation/model/DeepLabV3Plus(timm-mobilenetv3_small_100)_452_2.16M_0.8385/best_model_simplifier.onnx',
        'MediaPipe SelfieSegmentation(Normal)': None,
        'MediaPipe SelfieSegmentation(LandScape)': None,
        'YOLOv8-nano-seg': _model_base_path +
        'yolov8_seg/model/yolov8n-seg.onnx',
        'FLAIR Aerial (IGN)': None,
        'FLAIR Aerial INT8 (ONNX)': _model_base_path +
        'aerial_segmentation_flair/model/flair_aerial_seg_int8_N.onnx',
        'Pothole YOLO-seg': _model_base_path +
        'pothole/model/pothole.onnx',
        'Pothole YOLO-seg (v12)': _model_base_path +
        'pothole/model/potehole_12.onnx',
    }
    _model_instance = {}

    def __init__(self):
        pass

    @classmethod
    def _load_custom_models_from_registry(cls):
        """Load user-uploaded models from the registry into the class dicts."""
        try:
            entries = _seg_registry.load_registry()
        except Exception as exc:
            logger.warning(f"[SemanticSegmentation] Could not load custom models registry: {exc}")
            return
        for entry in entries:
            name = entry.get('name', '')
            path = entry.get('path', '')
            if not name or not path:
                continue
            if name in cls._model_class:
                continue
            in_w = int(entry.get('input_width', 512))
            in_h = int(entry.get('input_height', 512))
            num_classes = int(entry.get('num_classes', 2))
            cls._register_custom_model(name, path, in_w, in_h, num_classes)
            logger.info(f"[SemanticSegmentation] Loaded custom model from registry: {name}")

    @classmethod
    def _register_custom_model(cls, name, path, in_w, in_h, num_classes=2):
        """Add a custom ONNX model to the class-level runtime dictionaries."""
        def _make_factory(p, w, h, nc):
            def factory(model_path, providers=None):
                if providers is None:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                return CustomONNXSeg(
                    model_path=p, input_width=w, input_height=h,
                    num_classes=nc, providers=providers
                )
            return factory

        cls._model_class[name] = _make_factory(path, in_w, in_h, num_classes)
        cls._model_path_setting[name] = path

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
            logger.error(f"[SemanticSegmentation Upload] ONNX inspection failed: {exc}")
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
        in_w = meta.get("input_width", 512)
        in_h = meta.get("input_height", 512)
        num_cls = meta.get("num_classes", 0)
        dpg.add_text(f"Input dimensions : {in_w} x {in_h} px (W x H)", parent=self.tag_preview_details)
        dpg.add_text(f"Number of classes: {num_cls}", parent=self.tag_preview_details)

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

        os.makedirs(_SEG_UPLOADS_DIR, exist_ok=True)
        dest_path = onnx_path
        try:
            basename = os.path.basename(onnx_path)
            candidate = os.path.join(_SEG_UPLOADS_DIR, basename)
            if os.path.abspath(onnx_path) != os.path.abspath(candidate):
                shutil.copy2(onnx_path, candidate)
                dest_path = candidate
        except Exception as exc:
            logger.warning(f"[SemanticSegmentation Upload] Could not copy ONNX: {exc}")

        try:
            Node._finalise_upload(self, dest_path, meta, custom_name=custom_name)
            dpg.set_value(self.tag_preview_status, f"\u2713 Model '{custom_name}' uploaded successfully!")
            self._set_upload_preview_actions(upload_succeeded=True)
        except Exception as exc:
            logger.error(f"[SemanticSegmentation Upload] Finalise failed: {exc}", exc_info=True)
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

        in_w = meta.get("input_width", 512)
        in_h = meta.get("input_height", 512)
        num_classes = max(2, meta.get("num_classes", 2))
        Node._register_custom_model(name, onnx_path, in_w, in_h, num_classes)

        registry_entry = {
            "name": name,
            "path": onnx_path,
            "input_width": in_w,
            "input_height": in_h,
            "output_format": meta.get("output_format", "unknown"),
            "num_classes": num_classes,
        }
        try:
            _seg_registry.save_entry(registry_entry)
        except Exception as exc:
            logger.warning(f"[SemanticSegmentation Upload] Could not save registry entry: {exc}")

        model_combo_tag = node.tag_node_name + ':' + node.TYPE_TEXT + ':Input02Value'
        try:
            current_items = dpg.get_item_configuration(model_combo_tag).get("items", [])
            if name not in current_items:
                current_items = list(current_items) + [name]
            dpg.configure_item(model_combo_tag, items=current_items, default_value=name)
        except Exception as exc:
            logger.warning(f"[SemanticSegmentation Upload] Could not update model dropdown: {exc}")

        # Disable the delete button when the newly-selected model is built-in
        try:
            is_builtin = name in _BUILTIN_SEG_MODEL_NAMES
            dpg.configure_item(node.tag_delete_btn, enabled=not is_builtin)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Delete model
    # ------------------------------------------------------------------

    @classmethod
    def _delete_custom_model(cls, name: str) -> bool:
        """Remove a custom model from the registry and runtime dicts.

        Built-in models cannot be deleted.  Returns True when the model
        was found and removed.
        """
        if not name:
            return False
        if name in _BUILTIN_SEG_MODEL_NAMES:
            logger.warning(f"[Delete] '{name}' is a built-in model and cannot be deleted.")
            return False
        found = name in cls._model_class or name in cls._model_path_setting
        onnx_path = cls._model_path_setting.get(name)
        if onnx_path:
            try:
                onnx_real = os.path.realpath(onnx_path)
                uploads_real = os.path.realpath(_SEG_UPLOADS_DIR)
                if (
                    os.path.commonpath([onnx_real, uploads_real]) == uploads_real
                    and onnx_real != uploads_real
                    and os.path.isfile(onnx_real)
                ):
                    os.remove(onnx_real)
                    logger.info(f"[Delete] ONNX file deleted: {onnx_real}")
            except Exception as exc:
                logger.warning(f"[Delete] Could not delete ONNX file for '{name}': {exc}")
        try:
            _seg_registry.remove_entry(name)
        except Exception as exc:
            logger.warning(f"[Delete] Could not remove registry entry for '{name}': {exc}")
        cls._model_class.pop(name, None)
        cls._model_path_setting.pop(name, None)
        for key in [k for k in cls._model_instance if k == name or k.startswith(name + '_')]:
            cls._model_instance.pop(key, None)
        return found

    def _delete_selected_model(self, name: str):
        """Delete the currently-selected model and refresh the combobox."""
        if not name:
            return
        if name in _BUILTIN_SEG_MODEL_NAMES:
            logger.warning(f"[Delete] '{name}' is a built-in model and cannot be deleted.")
            return
        Node._delete_custom_model(name)
        model_combo_tag = self.tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'
        remaining = list(Node._model_class.keys())
        new_default = remaining[0] if remaining else ""
        try:
            dpg.configure_item(model_combo_tag, items=remaining, default_value=new_default)
        except Exception as exc:
            logger.warning(f"[Delete] Could not update model dropdown: {exc}")
        # Re-enable / disable the delete button for the new default
        try:
            is_builtin = new_default in _BUILTIN_SEG_MODEL_NAMES
            dpg.configure_item(self.tag_delete_btn, enabled=not is_builtin)
        except Exception:
            pass

    def add_node(
        self,
        parent,
        node_id,
        pos=[0, 0],
        opencv_setting_dict=None,
        callback=None,
    ):
        # タグ名
        tag_node_name = str(node_id) + ':' + self.node_tag
        tag_node_input01_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Input01'
        tag_node_input01_value_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Input01Value'
        tag_node_input02_name = tag_node_name + ':' + self.TYPE_TEXT + ':Input02'
        tag_node_input02_value_name = tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'
        tag_node_input03_name = tag_node_name + ':' + self.TYPE_FLOAT + ':Input03'
        tag_node_input03_value_name = tag_node_name + ':' + self.TYPE_FLOAT + ':Input03Value'
        tag_node_output01_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Output01'
        tag_node_output01_value_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Output01Value'
        tag_node_output02_name = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02'
        tag_node_output02_value_name = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02Value'
        tag_node_output_json_name = tag_node_name + ':' + self.TYPE_JSON + ':OutputJson'
        tag_node_output_json_value_name = tag_node_name + ':' + self.TYPE_JSON + ':OutputJsonValue'

        tag_provider_select_name = tag_node_name + ':' + self.TYPE_TEXT + ':Provider'
        tag_provider_select_value_name = tag_node_name + ':' + self.TYPE_IMAGE + ':ProviderValue'

        # OpenCV向け設定
        self._opencv_setting_dict = opencv_setting_dict
        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']
        use_gpu = self._opencv_setting_dict['use_gpu']

        # 初期化用黒画像
        black_image = np.zeros((small_window_w, small_window_h, 3))
        black_texture = self.convert_cv_to_dpg(
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
        onnx_file_dialog_tag = "seg_onnx_select:" + str(node_id)
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            modal=True,
            height=400,
            callback=self._callback_onnx_select,
            tag=onnx_file_dialog_tag,
        ):
            dpg.add_file_extension("ONNX (*.onnx){.onnx}")
            dpg.add_file_extension("", color=(150, 255, 150, 255))
        self.tag_upload_file_dialog = onnx_file_dialog_tag

        # ---- ONNX preview / confirmation dialog -----------------------------
        preview_window_tag  = "seg_onnx_preview_window:"  + str(node_id)
        preview_name_tag    = "seg_onnx_preview_name:"    + str(node_id)
        preview_details_tag = "seg_onnx_preview_details:" + str(node_id)
        preview_status_tag  = "seg_onnx_preview_status:"  + str(node_id)
        preview_confirm_tag = "seg_onnx_preview_confirm:" + str(node_id)
        preview_cancel_tag  = "seg_onnx_preview_cancel:"  + str(node_id)
        preview_quit_tag    = "seg_onnx_preview_quit:"    + str(node_id)

        self.tag_preview_window  = preview_window_tag
        self.tag_preview_name    = preview_name_tag
        self.tag_preview_details = preview_details_tag
        self.tag_preview_status  = preview_status_tag
        self.tag_preview_confirm = preview_confirm_tag
        self.tag_preview_cancel  = preview_cancel_tag
        self.tag_preview_quit    = preview_quit_tag

        def _on_upload_confirm(sender, app_data, user_data):
            self._do_confirm_upload()

        def _on_close_preview(sender, app_data, user_data):
            self._close_upload_preview()

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
                label=self.node_label,
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
                def _on_model_change(sender, app_data, user_data):
                    selected = app_data
                    try:
                        is_builtin = selected in _BUILTIN_SEG_MODEL_NAMES
                        dpg.configure_item(self.tag_delete_btn, enabled=not is_builtin)
                    except Exception:
                        pass

                dpg.add_combo(
                    list(self._model_class.keys()),
                    default_value=list(self._model_class.keys())[0],
                    width=small_window_w,
                    tag=tag_node_input02_value_name,
                    callback=_on_model_change,
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
            # スコア閾値
            with dpg.node_attribute(
                    tag=tag_node_input03_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_slider_float(
                    tag=tag_node_input03_value_name,
                    label="score",
                    width=small_window_w - 80,
                    default_value=0.5,
                    min_value=self._min_val,
                    max_value=self._max_val,
                    callback=None,
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
            self.tag_upload_btn = tag_node_name + ':UploadONNX'

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
                    tag=self.tag_upload_btn,
                    width=small_window_w,
                    callback=_on_upload_clicked,
                )
                dpg.bind_item_theme(add_model_btn, add_model_btn_theme)

            # ---- Delete model button (red, only enabled for custom models) --
            self.tag_delete_btn = tag_node_name + ':DeleteONNX'

            def _on_delete_clicked(sender, app_data, user_data):
                selected = dpg_get_value(tag_node_input02_value_name)
                self._delete_selected_model(selected)

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
                    tag=self.tag_delete_btn,
                    width=small_window_w,
                    callback=_on_delete_clicked,
                )
                dpg.bind_item_theme(delete_btn, delete_model_btn_theme)
                # Disable for the initially-selected model if it is built-in
                default_selected = list(self._model_class.keys())[0] if self._model_class else ""
                if default_selected in _BUILTIN_SEG_MODEL_NAMES:
                    dpg.configure_item(delete_btn, enabled=False)

        self.tag_node_name = tag_node_name
        return self

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
        input_value03_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Input03Value'
        output_value01_tag = tag_node_name + ':' + self.TYPE_IMAGE + ':Output01Value'
        output_value02_tag = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02Value'

        tag_provider_select_value_name = tag_node_name + ':' + self.TYPE_IMAGE + ':ProviderValue'

        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']
        use_gpu = self._opencv_setting_dict['use_gpu']

        # 接続情報確認
        for connection_info in connection_list:
            connection_type = connection_info[0].split(':')[2]
            if connection_type == self.TYPE_FLOAT:
                # 接続タグ取得
                source_tag = connection_info[0] + 'Value'
                destination_tag = connection_info[1] + 'Value'
                # 値更新
                input_value = round(float(dpg_get_value(source_tag)), 3)
                input_value = max([self._min_val, input_value])
                input_value = min([self._max_val, input_value])
                dpg_set_value(destination_tag, input_value)

        # 画像取得
        frame = self.get_input_frame(connection_list, node_image_dict, node_audio_dict)

        # スコア閾値
        score_th = round(float(dpg_get_value(input_value03_tag)), 3)

        # CPU/GPU選択状態取得
        provider = 'cpu'
        if use_gpu:
        	provider = dpg_get_value(tag_provider_select_value_name)

        # モデル情報取得
        model_name = dpg_get_value(input_value02_tag)
        model_path = self._model_path_setting[model_name]
        model_class = self._model_class[model_name]

        model_name_with_provider = model_name + '_' + provider

        # モデル取得
        if frame is not None:
            if model_name_with_provider not in self._model_instance:
                if provider == 'cpu':
                    providers = ['CPUExecutionProvider']
                    self._model_instance[
                        model_name_with_provider] = model_class(
                            model_path,
                            providers=providers,
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

        result = {}
        debug_frame = None
        if frame is not None:
            class_num = self._model_instance[
                model_name_with_provider].get_class_num()

            # PotholeYOLOSeg returns (masks, class_ids) tuple and produces
            # a flat {class_name: pixel_count} JSON for the Chart node.
            if isinstance(self._model_instance[model_name_with_provider], PotholeYOLOSeg):
                segmentation_map, class_ids = run_unlocked(
                    self._model_instance[model_name_with_provider], frame
                )
                pixel_counts = self._model_instance[
                    model_name_with_provider].compute_pixel_counts(
                        segmentation_map, class_ids)
                result = pixel_counts  # flat numeric dict → Chart node compatible
            else:
                segmentation_map = run_unlocked(
                    self._model_instance[model_name_with_provider], frame
                )
                result['score_th'] = score_th
                result['class_num'] = class_num
                result['segmentation_map'] = segmentation_map

        # 計測終了
        if frame is not None and use_pref_counter:
            elapsed_time = time.monotonic() - start_time
            elapsed_time = int(elapsed_time * 1000)
            dpg_set_value(output_value02_tag,
                          str(elapsed_time).zfill(4) + 'ms')

        # 描画
        if frame is not None:
            # PotholeYOLOSegV12: coloured overlay + contours + bounding boxes
            if isinstance(self._model_instance[model_name_with_provider], PotholeYOLOSegV12):
                debug_frame = self._model_instance[
                    model_name_with_provider].draw_result(frame, segmentation_map)
            # PotholeYOLOSeg (v1): draw contours only
            elif isinstance(self._model_instance[model_name_with_provider], PotholeYOLOSeg):
                debug_frame = self.draw_yolov8_seg_contours(
                    frame,
                    segmentation_map,
                )
            # Special handling for YOLOv8-seg to draw only contours
            elif model_name == 'YOLOv8-nano-seg':
                debug_frame = self.draw_yolov8_seg_contours(
                    frame,
                    segmentation_map,
                )
            elif model_name == 'FLAIR Aerial (IGN)':
                mask = FlairAerialSegmentation.get_argmax_mask(segmentation_map)
                debug_frame = overlay_flair(frame, mask, alpha=0.5)
            elif model_name == 'FLAIR Aerial INT8 (ONNX)':
                mask = FlairAerialSegmentationONNX.get_argmax_mask(segmentation_map)
                debug_frame = overlay_flair2(frame, mask, alpha=0.5)
            else:
                debug_frame = self.draw_semantic_segmentation_info(
                    frame,
                    score_th,
                    class_num,
                    segmentation_map,
                )
            texture = self.convert_cv_to_dpg(
                debug_frame,
                small_window_w,
                small_window_h,
            )
            dpg_set_value(output_value01_tag, texture)

        return {"image": debug_frame if debug_frame is not None else frame, "json": result, "audio": None}

    def close(self, node_id):
        super().close(node_id)

    def get_setting_dict(self, node_id):
        tag_node_name = str(node_id) + ':' + self.node_tag
        input_value02_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'
        input_value03_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Input03Value'

        # 選択モデル
        model_name = dpg_get_value(input_value02_tag)
        # スコア閾値
        score_th = round(float(dpg_get_value(input_value03_tag)), 3)

        pos = dpg.get_item_pos(tag_node_name)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos
        setting_dict[input_value02_tag] = model_name
        setting_dict[input_value03_tag] = score_th

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        tag_node_name = str(node_id) + ':' + self.node_tag
        input_value02_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'
        input_value03_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Input03Value'

        model_name = setting_dict[input_value02_tag]
        score_th = setting_dict[input_value03_tag]

        dpg_set_value(input_value02_tag, model_name)
        dpg_set_value(input_value03_tag, score_th)


# Load user-uploaded custom models from registry at import time
Node._load_custom_models_from_registry()
