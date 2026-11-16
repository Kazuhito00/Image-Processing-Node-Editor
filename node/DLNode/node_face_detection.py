#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import shutil
import time
import os
import sys
import requests
import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value, run_unlocked

from node.node_abc import DpgNodeABC
#from node_editor.util import convert_cv_to_dpg

from node.DLNode.face_detection.YuNet.yunet import YuNet
from node.DLNode.face_detection.mediapipe_facedetection.mediapipe_facedetection import (
    MediaPipeFaceDetectionModel0,
    MediaPipeFaceDetectionModel1,
)
from node.DLNode.face_detection.mediapipe_facemesh.mediapipe_facemesh import (
    MediaPipeFaceMeshNonRefine,
    MediaPipeFaceMeshRefine,
)
from node.DLNode.face_detection.CustomONNX.custom_onnx import CustomONNX as CustomONNXFaceDetection
from node.DLNode.face_detection import custom_models_registry as _fd_registry
from node.DLNode.object_detection import onnx_inspector
from src.utils.logging import get_logger
from src.utils.gpu_utils import get_execution_providers

logger = get_logger(__name__)

#from node.draw_node.draw_util.draw_util import draw_face_detection_info

from node.basenode import Node

if getattr(sys, 'frozen', False):
    from src.utils.paths import get_models_dir
    _FD_UPLOADS_DIR = get_models_dir('face_detection')
else:
    _FD_UPLOADS_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'face_detection', 'CustomONNX', 'models'
    )


class FactoryNode:
    node_label = 'FaceDetection'
    node_tag = 'FaceDetection'
    

    def __init__(self):
        pass

    
    def add_node(self, parent, node_id, pos=[0, 0], opencv_setting_dict=None, callback=None):
        """Adds a node to the processing graph."""
        node = Node()
        return node.add_node(parent, node_id, pos, opencv_setting_dict, callback)


class Node(Node):  # noqa: F811
    _ver = '0.0.1'

    node_label = 'Face Detection'
    node_tag = 'FaceDetection'

    _min_val = 0.0
    _max_val = 1.0

    _opencv_setting_dict = None


    _model_class = {
        'YuNet': YuNet,
        'MediaPipe FaceDetection(~2m)': MediaPipeFaceDetectionModel0,
        'MediaPipe FaceDetection(~5m)': MediaPipeFaceDetectionModel1,
        'MediaPipe FaceMesh': MediaPipeFaceMeshNonRefine,
        'MediaPipe FaceMesh(Refine Landmark)': MediaPipeFaceMeshRefine,
    }
    _model_base_path = os.path.dirname(os.path.abspath(__file__)) + '/face_detection/'
    _model_path_setting = {
        'YuNet':
        _model_base_path + 'YuNet/model/face_detection_yunet_120x160.onnx',
        'MediaPipe FaceDetection(~2m)': None,
        'MediaPipe FaceDetection(~5m)': None,
        'MediaPipe FaceMesh': None,
        'MediaPipe FaceMesh(Refine Landmark)': None,
    }

    _model_instance = {}

    def __init__(self):
        pass

    @classmethod
    def _load_custom_models_from_registry(cls):
        """Load user-uploaded models from the registry into the class dicts."""
        try:
            entries = _fd_registry.load_registry()
        except Exception as exc:
            logger.warning(f"[FaceDetection] Could not load custom models registry: {exc}")
            return
        for entry in entries:
            name = entry.get('name', '')
            path = entry.get('path', '')
            if not name or not path:
                continue
            if name in cls._model_class:
                continue
            in_w = int(entry.get('input_width', 320))
            in_h = int(entry.get('input_height', 240))
            cls._register_custom_model(name, path, in_w, in_h)
            logger.info(f"[FaceDetection] Loaded custom model from registry: {name}")

    @classmethod
    def _register_custom_model(cls, name, path, in_w, in_h):
        """Add a custom ONNX model to the class-level runtime dictionaries."""
        def _make_factory(p, w, h):
            def factory(model_path, providers=None):
                if providers is None:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                return CustomONNXFaceDetection(
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
            logger.error(f"[FaceDetection Upload] ONNX inspection failed: {exc}")
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
        in_h = meta.get("input_height", 240)
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

        os.makedirs(_FD_UPLOADS_DIR, exist_ok=True)
        dest_path = onnx_path
        try:
            basename = os.path.basename(onnx_path)
            candidate = os.path.join(_FD_UPLOADS_DIR, basename)
            if os.path.abspath(onnx_path) != os.path.abspath(candidate):
                shutil.copy2(onnx_path, candidate)
                dest_path = candidate
        except Exception as exc:
            logger.warning(f"[FaceDetection Upload] Could not copy ONNX: {exc}")

        try:
            Node._finalise_upload(self, dest_path, meta, custom_name=custom_name)
            dpg.set_value(self.tag_preview_status, f"\u2713 Model '{custom_name}' uploaded successfully!")
            self._set_upload_preview_actions(upload_succeeded=True)
        except Exception as exc:
            logger.error(f"[FaceDetection Upload] Finalise failed: {exc}", exc_info=True)
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
        in_h = meta.get("input_height", 240)
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
            _fd_registry.save_entry(registry_entry)
        except Exception as exc:
            logger.warning(f"[FaceDetection Upload] Could not save registry entry: {exc}")

        model_combo_tag = node.tag_node_name + ':' + node.TYPE_TEXT + ':Input02Value'
        try:
            current_items = dpg.get_item_configuration(model_combo_tag).get("items", [])
            if name not in current_items:
                current_items = list(current_items) + [name]
            dpg.configure_item(model_combo_tag, items=current_items, default_value=name)
        except Exception as exc:
            logger.warning(f"[FaceDetection Upload] Could not update model dropdown: {exc}")

    def add_node(
        self,
        parent,
        node_id,
        pos=[0, 0],
        opencv_setting_dict=None,
        callback=None,
    ):

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


        self._opencv_setting_dict = opencv_setting_dict
        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']
        use_gpu = self._opencv_setting_dict['use_gpu']


        black_image = np.zeros((small_window_w, small_window_h, 3))
        black_texture = self.convert_cv_to_dpg(
            black_image,
            small_window_w,
            small_window_h,
        )

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
        onnx_file_dialog_tag = "fd_onnx_select:" + str(node_id)
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
        preview_window_tag  = "fd_onnx_preview_window:"  + str(node_id)
        preview_name_tag    = "fd_onnx_preview_name:"    + str(node_id)
        preview_details_tag = "fd_onnx_preview_details:" + str(node_id)
        preview_status_tag  = "fd_onnx_preview_status:"  + str(node_id)
        preview_confirm_tag = "fd_onnx_preview_confirm:" + str(node_id)
        preview_cancel_tag  = "fd_onnx_preview_cancel:"  + str(node_id)
        preview_quit_tag    = "fd_onnx_preview_quit:"    + str(node_id)

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

        with dpg.node(
                tag=tag_node_name,
                parent=parent,
                label=self.node_label,
                pos=pos,
        ):

            with dpg.node_attribute(
                    tag=tag_node_input01_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_text(
                    tag=tag_node_input01_value_name,
                    default_value='Input BGR image',
                )

            with dpg.node_attribute(
                    tag=tag_node_output01_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_image(tag_node_output01_value_name)

            with dpg.node_attribute(
                    tag=tag_node_input02_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_combo(
                    list(self._model_class.keys()),
                    default_value=list(self._model_class.keys())[0],
                    width=small_window_w,
                    tag=tag_node_input02_value_name,
                )
            if use_gpu:

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

            with dpg.node_attribute(
                    tag=tag_node_input03_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_slider_float(
                    tag=tag_node_input03_value_name,
                    label="score",
                    width=small_window_w - 80,
                    default_value=0.3,
                    min_value=self._min_val,
                    max_value=self._max_val,
                    callback=None,
                )

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


        for connection_info in connection_list:
            connection_type = connection_info[0].split(':')[2]
            if connection_type == self.TYPE_FLOAT:

                source_tag = connection_info[0] + 'Value'
                destination_tag = connection_info[1] + 'Value'

                input_value = round(float(dpg_get_value(source_tag)), 3)
                input_value = max([self._min_val, input_value])
                input_value = min([self._max_val, input_value])
                dpg_set_value(destination_tag, input_value)

        frame = self.get_input_frame(connection_list, node_image_dict, node_audio_dict)


        score_th = round(float(dpg_get_value(input_value03_tag)), 3)


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
        debug_frame = None
        if frame is not None:
            results_list = run_unlocked(
                self._model_instance[model_name_with_provider], frame
            )
            result['model_name'] = model_name
            result['score_th'] = score_th
            result['results_list'] = results_list


        if frame is not None and use_pref_counter:
            elapsed_time = time.monotonic() - start_time
            elapsed_time = int(elapsed_time * 1000)
            dpg_set_value(output_value02_tag,
                          str(elapsed_time).zfill(4) + 'ms')

        if frame is not None:
            debug_frame = copy.deepcopy(frame)
            debug_frame = self.draw_face_detection_info(
                model_name,
                debug_frame,
                results_list,
                score_th,
            )
            texture = self.convert_cv_to_dpg(
                debug_frame,
                small_window_w,
                small_window_h,
            )
            dpg_set_value(output_value01_tag, texture)
        logger.debug(f"Face detection result: {result}")


        count = 1
        try:
            if result.get('results_list'):
                if count != 0:
                    logger.debug("Face detected")
                    count = 0
                    #requests.get("http://192.168.0.47/")
            else:
                    count = 1

        except Exception as e:
            logger.error(f"Error processing face detection result: {e}")

        return {"image": debug_frame if debug_frame is not None else frame, "json": result, "audio": None}

    def close(self, node_id):
        super().close(node_id)

    def get_setting_dict(self, node_id):
        tag_node_name = str(node_id) + ':' + self.node_tag
        input_value02_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'
        input_value03_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Input03Value'

        model_name = dpg_get_value(input_value02_tag)

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
