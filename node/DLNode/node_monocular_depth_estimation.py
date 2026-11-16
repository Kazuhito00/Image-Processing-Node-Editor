#!/usr/bin/env python
# -*- coding: utf-8 -*-
import shutil
import time
import os
import sys

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value, run_unlocked

from node.node_abc import DpgNodeABC
#from node_editor.util import convert_cv_to_dpg
from node.basenode import Node
from node.DLNode.monocular_depth_estimation.FSRE_Depth.fsre_depth import FSRE_Depth
from node.DLNode.monocular_depth_estimation.HR_Depth.hr_depth import HR_Depth
from node.DLNode.monocular_depth_estimation.CustomONNX.custom_onnx import CustomONNX as CustomONNXDepth
from node.DLNode.monocular_depth_estimation import custom_models_registry as _depth_registry
from node.DLNode.object_detection import onnx_inspector
from src.utils.logging import get_logger

logger = get_logger(__name__)

if getattr(sys, 'frozen', False):
    from src.utils.paths import get_models_dir
    _DEPTH_UPLOADS_DIR = get_models_dir('monocular_depth_estimation')
else:
    _DEPTH_UPLOADS_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'monocular_depth_estimation', 'CustomONNX', 'models'
    )

class FactoryNode:
    node_label = 'MonocularDepthEstimation'
    node_tag = 'MonocularDepthEstimation'
    

    def __init__(self):
        pass

    
    def add_node(
        self,
        parent,
        node_id,
        pos=[0, 0],
        opencv_setting_dict=None,
        callback=None,
    ):

        

        node = Node()
        node.tag_node_name = str(node_id) + ':' + node.node_tag
        node.tag_node_input01_name = node.tag_node_name + ':' + node.TYPE_IMAGE + ':Input01'
        node.tag_node_input01_value_name = node.tag_node_name + ':' + node.TYPE_IMAGE + ':Input01Value'
        node.tag_node_input02_name = node.tag_node_name + ':' + node.TYPE_TEXT + ':Input02'
        node.tag_node_input02_value_name = node.tag_node_name + ':' + node.TYPE_TEXT + ':Input02Value'
        node.tag_node_output01_name = node.tag_node_name + ':' + node.TYPE_IMAGE + ':Output01'
        node.tag_node_output01_value_name = node.tag_node_name + ':' + node.TYPE_IMAGE + ':Output01Value'
        node.tag_node_output02_name = node.tag_node_name + ':' + node.TYPE_TIME_MS + ':Output02'
        node.tag_node_output02_value_name = node.tag_node_name + ':' +node.TYPE_TIME_MS + ':Output02Value'
        node.tag_node_output_json_name = node.tag_node_name + ':' + node.TYPE_JSON + ':OutputJson'
        node.tag_node_output_json_value_name = node.tag_node_name + ':' + node.TYPE_JSON + ':OutputJsonValue'

        node.tag_provider_select_name = node.tag_node_name + ':' + node.TYPE_TEXT + ':Provider'
        node.tag_provider_select_value_name = node.tag_node_name + ':' + node.TYPE_IMAGE + ':ProviderValue'


        node._opencv_setting_dict = opencv_setting_dict
        node.small_window_w = node._opencv_setting_dict['process_width']
        node.small_window_h = node._opencv_setting_dict['process_height']
        use_pref_counter = node._opencv_setting_dict['use_pref_counter']
        use_gpu = node._opencv_setting_dict['use_gpu']


        black_image = np.zeros((node.small_window_w, node.small_window_h, 3))
        black_texture = node.convert_cv_to_dpg(
            black_image,
            node.small_window_w,
            node.small_window_h,
        )


        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                node.small_window_w,
                node.small_window_h,
                black_texture,
                tag=node.tag_node_output01_value_name,
                format=dpg.mvFormat_Float_rgb,
            )

        # Create yellow theme for JSON button
        with dpg.theme() as yellow_button_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (255, 255, 153, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (255, 255, 153, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (255, 255, 153, 255))

        # ---- ONNX upload file dialog ----------------------------------------
        onnx_file_dialog_tag = "depth_onnx_select:" + str(node_id)
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
        preview_window_tag  = "depth_onnx_preview_window:"  + str(node_id)
        preview_name_tag    = "depth_onnx_preview_name:"    + str(node_id)
        preview_details_tag = "depth_onnx_preview_details:" + str(node_id)
        preview_status_tag  = "depth_onnx_preview_status:"  + str(node_id)
        preview_confirm_tag = "depth_onnx_preview_confirm:" + str(node_id)
        preview_cancel_tag  = "depth_onnx_preview_cancel:"  + str(node_id)
        preview_quit_tag    = "depth_onnx_preview_quit:"    + str(node_id)

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

        with dpg.node(
                tag=node.tag_node_name,
                parent=parent,
                label=node.node_label,
                pos=pos,
        ):

            with dpg.node_attribute(
                    tag=node.tag_node_input01_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_text(
                    tag=node.tag_node_input01_value_name,
                    default_value='Input BGR image',
                )

            with dpg.node_attribute(
                    tag=node.tag_node_output01_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_image(node.tag_node_output01_value_name)

            with dpg.node_attribute(
                    tag=node.tag_node_input02_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_combo(
                    list(node._model_class.keys()),
                    default_value=list(node._model_class.keys())[0],
                    width=node.small_window_w,
                    tag=node.tag_node_input02_value_name,
                )
            if use_gpu:

                with dpg.node_attribute(
                        tag=node.tag_provider_select_name,
                        attribute_type=dpg.mvNode_Attr_Static,
                ):
                    dpg.add_radio_button(
                        ("cpu", "cuda", "TRTcuda"),
                        tag=node.tag_provider_select_value_name,
                        default_value='cpu',
                        horizontal=True,
                    )

            if use_pref_counter:
                with dpg.node_attribute(
                        tag=node.tag_node_output02_name,
                        attribute_type=dpg.mvNode_Attr_Output,
                ):
                    dpg.add_text(
                        tag=node.tag_node_output02_value_name,
                        default_value='elapsed time(ms)',
                    )

            # JSON output button
            with dpg.node_attribute(
                    tag=node.tag_node_output_json_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                btn = dpg.add_button(
                    label="JSON",
                    tag=node.tag_node_output_json_value_name,
                    width=node.small_window_w,
                    enabled=False,
                )
                dpg.bind_item_theme(btn, yellow_button_theme)

            # ---- Add Model button (yellow, opens ONNX upload dialog) --------
            node.tag_upload_btn = node.tag_node_name + ':UploadONNX'

            def _on_upload_clicked(sender, app_data, user_data):
                dpg.show_item(onnx_file_dialog_tag)

            with dpg.node_attribute(
                    tag=node.tag_node_name + ':UploadAttr',
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
                    width=node.small_window_w,
                    callback=_on_upload_clicked,
                )
                dpg.bind_item_theme(add_model_btn, add_model_btn_theme)

        return node


class Node(Node):
    _ver = '0.0.1'

    node_label = 'Monocular Depth Estimation'
    node_tag = 'MonocularDepthEstimation'

    _opencv_setting_dict = None


    _model_class = {
        'FSRE-Depth(320x192)': FSRE_Depth,
        'FSRE-Depth(640x384)': FSRE_Depth,
        'Lite-HR-Depth(1280x384)': HR_Depth,
        'HR-Depth(1280x384)': HR_Depth,
    }
    _model_base_path = os.path.dirname(os.path.abspath(__file__)) + '/monocular_depth_estimation/'
    _model_path_setting = {
        'FSRE-Depth(320x192)':
        _model_base_path +
        'FSRE_Depth/fsre_depth_192x320/fsre_depth_full_192x320.onnx',
        'FSRE-Depth(640x384)':
        _model_base_path +
        'FSRE_Depth/fsre_depth_384x640/fsre_depth_full_384x640.onnx',
        'Lite-HR-Depth(1280x384)':
        _model_base_path +
        'HR_Depth/saved_model_lite_hr_depth_384x1280/lite_hr_depth_k_t_encoder_depth_384x1280.onnx',
        'HR-Depth(1280x384)':
        _model_base_path +
        'HR_Depth/saved_model_hr_depth_384x1280/hr_depth_k_m_depth_encoder_depth_384x1280.onnx',
    }

    _model_instance = {}

    def __init__(self):
        pass

    @classmethod
    def _load_custom_models_from_registry(cls):
        """Load user-uploaded models from the registry into the class dicts."""
        try:
            entries = _depth_registry.load_registry()
        except Exception as exc:
            logger.warning(f"[DepthEstimation] Could not load custom models registry: {exc}")
            return
        for entry in entries:
            name = entry.get('name', '')
            path = entry.get('path', '')
            if not name or not path:
                continue
            if name in cls._model_class:
                continue
            in_w = int(entry.get('input_width', 320))
            in_h = int(entry.get('input_height', 192))
            cls._register_custom_model(name, path, in_w, in_h)
            logger.info(f"[DepthEstimation] Loaded custom model from registry: {name}")

    @classmethod
    def _register_custom_model(cls, name, path, in_w, in_h):
        """Add a custom ONNX model to the class-level runtime dictionaries."""
        def _make_factory(p, w, h):
            def factory(model_path, providers=None):
                if providers is None:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                return CustomONNXDepth(
                    model_path=p, input_width=w, input_height=h, providers=providers
                )
            return factory

        cls._model_class[name] = _make_factory(path, in_w, in_h)
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
            logger.error(f"[DepthEstimation Upload] ONNX inspection failed: {exc}")
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
        in_w = meta.get("input_width", 320)
        in_h = meta.get("input_height", 192)
        dpg.add_text(f"Input dimensions : {in_w} x {in_h} px (W x H)", parent=self.tag_preview_details)
        dpg.add_text(f"Output format    : {meta.get('output_format', 'unknown')}", parent=self.tag_preview_details)

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

        os.makedirs(_DEPTH_UPLOADS_DIR, exist_ok=True)
        dest_path = onnx_path
        try:
            basename = os.path.basename(onnx_path)
            candidate = os.path.join(_DEPTH_UPLOADS_DIR, basename)
            if os.path.abspath(onnx_path) != os.path.abspath(candidate):
                shutil.copy2(onnx_path, candidate)
                dest_path = candidate
        except Exception as exc:
            logger.warning(f"[DepthEstimation Upload] Could not copy ONNX: {exc}")

        try:
            Node._finalise_upload(self, dest_path, meta, custom_name=custom_name)
            dpg.set_value(self.tag_preview_status, f"\u2713 Model '{custom_name}' uploaded successfully!")
            self._set_upload_preview_actions(upload_succeeded=True)
        except Exception as exc:
            logger.error(f"[DepthEstimation Upload] Finalise failed: {exc}", exc_info=True)
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

        in_w = meta.get("input_width", 320)
        in_h = meta.get("input_height", 192)
        Node._register_custom_model(name, onnx_path, in_w, in_h)

        registry_entry = {
            "name": name,
            "path": onnx_path,
            "input_width": in_w,
            "input_height": in_h,
            "output_format": meta.get("output_format", "unknown"),
            "num_classes": meta.get("num_classes", 0),
        }
        try:
            _depth_registry.save_entry(registry_entry)
        except Exception as exc:
            logger.warning(f"[DepthEstimation Upload] Could not save registry entry: {exc}")

        model_combo_tag = node.tag_node_name + ':' + node.TYPE_TEXT + ':Input02Value'
        try:
            current_items = dpg.get_item_configuration(model_combo_tag).get("items", [])
            if name not in current_items:
                current_items = list(current_items) + [name]
            dpg.configure_item(model_combo_tag, items=current_items, default_value=name)
        except Exception as exc:
            logger.warning(f"[DepthEstimation Upload] Could not update model dropdown: {exc}")

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

        tag_provider_select_value_name = tag_node_name + ':' + self.TYPE_IMAGE + ':ProviderValue'

        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']
        use_gpu = self._opencv_setting_dict['use_gpu']


        for connection_info in connection_list:
            connection_type = connection_info[0].split(':')[2]
            if connection_type == self.TYPE_INT:

                source_tag = connection_info[0] + 'Value'
                destination_tag = connection_info[1] + 'Value'

                input_value = int(dpg_get_value(source_tag))
                input_value = max([self._min_val, input_value])
                input_value = min([self._max_val, input_value])
                dpg_set_value(destination_tag, input_value)

        frame = self.get_input_frame(connection_list, node_image_dict, node_audio_dict)


        provider = 'cpu'
        if use_gpu:
        	provider = dpg_get_value(tag_provider_select_value_name)


        model_name = dpg_get_value(input_value02_tag)
        model_path = self._model_path_setting[model_name]
        model_class = self._model_class[model_name]

        model_name_with_provider = model_name + '_' + provider


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


        if frame is not None and use_pref_counter:
            start_time = time.monotonic()

        result = {}
        if frame is not None:
            depth_map = run_unlocked(
                self._model_instance[model_name_with_provider], frame
            )
            frame = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)
            result['depth_map'] = depth_map


        if frame is not None and use_pref_counter:
            elapsed_time = time.monotonic() - start_time
            elapsed_time = int(elapsed_time * 1000)
            dpg_set_value(output_value02_tag,
                          str(elapsed_time).zfill(4) + 'ms')


        if frame is not None:
            texture = self.convert_cv_to_dpg(
                frame,
                small_window_w,
                small_window_h,
            )
            dpg_set_value(output_value01_tag, texture)

        return {"image": frame, "json": result, "audio": None}

    def close(self, node_id):
        super().close(node_id)

    def get_setting_dict(self, node_id):
        tag_node_name = str(node_id) + ':' + self.node_tag
        input_value02_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'


        model_name = dpg_get_value(input_value02_tag)

        pos = dpg.get_item_pos(tag_node_name)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos
        setting_dict[input_value02_tag] = model_name

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        tag_node_name = str(node_id) + ':' + self.node_tag
        input_value02_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'

        model_name = setting_dict[input_value02_tag]

        dpg_set_value(input_value02_tag, model_name)


# Load user-uploaded custom models from registry at import time
Node._load_custom_models_from_registry()
