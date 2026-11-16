#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import shutil
import time
import os
import sys

import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value, run_unlocked

from node.node_abc import DpgNodeABC
#from node_editor.util import convert_cv_to_dpg


from node.basenode import Node

from node.DLNode.pose_estimation.movenet.movenet import (
    MoveNetSinglePoseLightning,
    MoveNetSinglePoseThunder,
    MoveNetMultiPoseLightning,
)
from node.DLNode.pose_estimation.mediapipe_hands.mediapipe_hands import (
    MediaPipeHandsComplexity0,
    MediaPipeHandsComplexity1,
)
from node.DLNode.pose_estimation.mediapipe_pose.mediapipe_pose import (
    MediaPipePoseComplexity0,
    MediaPipePoseComplexity1,
    MediaPipePoseComplexity2,
)

from node.DLNode.pose_estimation.tennis_keypoints.tennis_keypoints import tennis_keypoints 
from node.DLNode.pose_estimation.tennis_keypoints_2.tennis_keypoints_2 import tennis_keypoints_2
from node.DLNode.pose_estimation.CustomONNX.custom_onnx import CustomONNX as CustomONNXPose
from node.DLNode.pose_estimation import custom_models_registry as _pose_registry
from node.DLNode.object_detection import onnx_inspector
from src.utils.logging import get_logger
from src.utils.gpu_utils import get_execution_providers

logger = get_logger(__name__)

if getattr(sys, 'frozen', False):
    from src.utils.paths import get_models_dir
    _POSE_UPLOADS_DIR = get_models_dir('pose_estimation')
else:
    _POSE_UPLOADS_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'pose_estimation', 'CustomONNX', 'models'
    )

import random

class FactoryNode:
    node_label = 'PoseEstimation'
    node_tag = 'PoseEstimation'
    

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
        node.tag_node_name = str(node_id) + ':' + self.node_tag
        node.tag_node_input01_name = node.tag_node_name + ':' + node.TYPE_IMAGE + ':Input01'
        node.tag_node_input01_value_name = node.tag_node_name + ':' + node.TYPE_IMAGE + ':Input01Value'
        node.tag_node_input02_name = node.tag_node_name + ':' + node.TYPE_TEXT + ':Input02'
        node.tag_node_input02_value_name = node.tag_node_name + ':' + node.TYPE_TEXT + ':Input02Value'
        node.tag_node_input03_name = node.tag_node_name + ':' + node.TYPE_FLOAT + ':Input03'
        node.tag_node_input03_value_name = node.tag_node_name + ':' + node.TYPE_FLOAT + ':Input03Value'
        node.tag_node_output01_name = node.tag_node_name + ':' + node.TYPE_IMAGE + ':Output01'
        node.tag_node_output01_value_name = node.tag_node_name + ':' + node.TYPE_IMAGE + ':Output01Value'
        node.tag_node_output02_name = node.tag_node_name + ':' + node.TYPE_TIME_MS + ':Output02'
        node.tag_node_output02_value_name = node.tag_node_name + ':' + node.TYPE_TIME_MS + ':Output02Value'
        node.tag_node_output_json_name = node.tag_node_name + ':' + node.TYPE_JSON + ':Output03'
        node.tag_node_output_json_value_name = node.tag_node_name + ':' + node.TYPE_JSON + ':Output03Value'

        node.tag_provider_select_name = node.tag_node_name + ':' + node.TYPE_TEXT + ':Provider'
        node.tag_provider_select_value_name = node.tag_node_name + ':' + node.TYPE_IMAGE + ':ProviderValue'

        # OpenCV
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


        logger.debug(f"Creating pose estimation node: {node.node_label}")

        # ---- ONNX upload file dialog ----------------------------------------
        onnx_file_dialog_tag = "pose_onnx_select:" + str(node_id)
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
        preview_window_tag  = "pose_onnx_preview_window:"  + str(node_id)
        preview_name_tag    = "pose_onnx_preview_name:"    + str(node_id)
        preview_details_tag = "pose_onnx_preview_details:" + str(node_id)
        preview_status_tag  = "pose_onnx_preview_status:"  + str(node_id)
        preview_confirm_tag = "pose_onnx_preview_confirm:" + str(node_id)
        preview_cancel_tag  = "pose_onnx_preview_cancel:"  + str(node_id)
        preview_quit_tag    = "pose_onnx_preview_quit:"    + str(node_id)

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
                # CPU/GPU
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

            with dpg.node_attribute(
                    tag=node.tag_node_input03_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_slider_float(
                    tag=node.tag_node_input03_value_name,
                    label="score",
                    width=node.small_window_w - 80,
                    default_value=0.3,
                    min_value=node._min_val,
                    max_value=node._max_val,
                    callback=None,
                )

            if use_pref_counter:
                with dpg.node_attribute(
                        tag=node.tag_node_output02_name,
                        attribute_type=dpg.mvNode_Attr_Output,
                ):
                    dpg.add_text(
                        tag=node.tag_node_output02_value_name,
                        default_value='Elapsed time(ms)',
                    )

            with dpg.node_attribute(
                        tag=node.tag_node_output_json_name,
                        attribute_type=dpg.mvNode_Attr_Output,
                ):
                    dpg.add_text(
                        tag=node.tag_node_output_json_value_name,
                        default_value='Pose Results',
                    )

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

    node_label = 'PoseEstimation'
    node_tag = 'PoseEstimation'

    _min_val = 0.0
    _max_val = 1.0

    _opencv_setting_dict = None

    _model_class = {
        'MoveNet(SinglePose Lightning)': MoveNetSinglePoseLightning,
        'MoveNet(SinglePose Thunder)': MoveNetSinglePoseThunder,
        'MoveNet(MulitPose Lightning)': MoveNetMultiPoseLightning,
        'MediaPipe Hands(Complexity0)': MediaPipeHandsComplexity0,
        'MediaPipe Hands(Complexity1)': MediaPipeHandsComplexity1,
        'MediaPipe Pose(Complexity0)': MediaPipePoseComplexity0,
        'MediaPipe Pose(Complexity1)': MediaPipePoseComplexity1,
        'MediaPipe Pose(Complexity2)': MediaPipePoseComplexity2,
        'TennisKeyPoints': tennis_keypoints,
        'TennisKeyPoints_2': tennis_keypoints_2,
    }

    _model_base_path = os.path.dirname(os.path.abspath(__file__)) + '/pose_estimation/'
    _model_path_setting = {
        'MoveNet(SinglePose Lightning)':
        _model_base_path + 'movenet/model/movenet_singlepose_lightning_4.onnx',
        'MoveNet(SinglePose Thunder)':
        _model_base_path + 'movenet/model/movenet_singlepose_thunder_4.onnx',
        'MoveNet(MulitPose Lightning)':
        _model_base_path + 'movenet/model/movenet_multipose_lightning_1.onnx',
        'TennisKeyPoints': _model_base_path + 'tennis_keypoints/model/tennis.onnx',
        'TennisKeyPoints_2': _model_base_path + 'tennis_keypoints_2/model/tennis_old.onnx',
        'MediaPipe Hands(Complexity0)': None,
        'MediaPipe Hands(Complexity1)': None,
        'MediaPipe Pose(Complexity0)': None,
        'MediaPipe Pose(Complexity1)': None,
        'MediaPipe Pose(Complexity2)': None,
    }

    _model_instance = {}

    def __init__(self):
        pass

    @classmethod
    def _load_custom_models_from_registry(cls):
        """Load user-uploaded models from the registry into the class dicts."""
        try:
            entries = _pose_registry.load_registry()
        except Exception as exc:
            logger.warning(f"[PoseEstimation] Could not load custom models registry: {exc}")
            return
        for entry in entries:
            name = entry.get('name', '')
            path = entry.get('path', '')
            if not name or not path:
                continue
            if name in cls._model_class:
                continue
            in_w = int(entry.get('input_width', 192))
            in_h = int(entry.get('input_height', 192))
            cls._register_custom_model(name, path, in_w, in_h)
            logger.info(f"[PoseEstimation] Loaded custom model from registry: {name}")

    @classmethod
    def _register_custom_model(cls, name, path, in_w, in_h):
        """Add a custom ONNX model to the class-level runtime dictionaries."""
        def _make_factory(p, w, h):
            def factory(model_path, providers=None):
                if providers is None:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                return CustomONNXPose(
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
            logger.error(f"[PoseEstimation Upload] ONNX inspection failed: {exc}")
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
        in_w = meta.get("input_width", 192)
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

        os.makedirs(_POSE_UPLOADS_DIR, exist_ok=True)
        dest_path = onnx_path
        try:
            basename = os.path.basename(onnx_path)
            candidate = os.path.join(_POSE_UPLOADS_DIR, basename)
            if os.path.abspath(onnx_path) != os.path.abspath(candidate):
                shutil.copy2(onnx_path, candidate)
                dest_path = candidate
        except Exception as exc:
            logger.warning(f"[PoseEstimation Upload] Could not copy ONNX: {exc}")

        try:
            Node._finalise_upload(self, dest_path, meta, custom_name=custom_name)
            dpg.set_value(self.tag_preview_status, f"\u2713 Model '{custom_name}' uploaded successfully!")
            self._set_upload_preview_actions(upload_succeeded=True)
        except Exception as exc:
            logger.error(f"[PoseEstimation Upload] Finalise failed: {exc}", exc_info=True)
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

        in_w = meta.get("input_width", 192)
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
            _pose_registry.save_entry(registry_entry)
        except Exception as exc:
            logger.warning(f"[PoseEstimation Upload] Could not save registry entry: {exc}")

        model_combo_tag = node.tag_node_name + ':' + node.TYPE_TEXT + ':Input02Value'
        try:
            current_items = dpg.get_item_configuration(model_combo_tag).get("items", [])
            if name not in current_items:
                current_items = list(current_items) + [name]
            dpg.configure_item(model_combo_tag, items=current_items, default_value=name)
        except Exception as exc:
            logger.warning(f"[PoseEstimation Upload] Could not update model dropdown: {exc}")

    def update(
        self,
        node_id,
        connection_list,
        node_image_dict,
        node_result_dict,
        node_audio_dict,
    ):
        self.tag_node_name = str(node_id) + ':' + self.node_tag
        self.input_value02_tag = self.tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'
        self.input_value03_tag = self.tag_node_name + ':' + self.TYPE_FLOAT + ':Input03Value'
        self.output_value01_tag = self.tag_node_name + ':' + self.TYPE_IMAGE + ':Output01Value'
        self.output_value02_tag = self.tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02Value'

        tag_provider_select_value_name = self.tag_node_name + ':' + self.TYPE_IMAGE + ':ProviderValue'

        self.small_window_w = self._opencv_setting_dict['process_width']
        self.small_window_h = self._opencv_setting_dict['process_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']
        use_gpu = self._opencv_setting_dict['use_gpu']


        connection_info_src = ''
        for connection_info in connection_list:
            connection_type = connection_info[0].split(':')[2]
            if connection_type == self.TYPE_FLOAT:

                source_tag = connection_info[0] + 'Value'
                destination_tag = connection_info[1] + 'Value'

                input_value = round(float(dpg_get_value(source_tag)), 3)
                input_value = max([self._min_val, input_value])
                input_value = min([self._max_val, input_value])
                dpg_set_value(destination_tag, input_value)
            if connection_type == self.TYPE_IMAGE:

                connection_info_src = connection_info[0]
                connection_info_src = connection_info_src.split(':')[:2]
                connection_info_src = ':'.join(connection_info_src)


        frame = node_image_dict.get(connection_info_src, None)


        score_th = round(float(dpg_get_value(self.input_value03_tag)), 3)


        provider = 'cpu'
        if use_gpu:
        	provider = dpg_get_value(tag_provider_select_value_name)


        model_name = dpg_get_value(self.input_value02_tag)
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
            dpg_set_value(self.output_value02_tag,
                          str(elapsed_time).zfill(4) + 'ms')


        if frame is not None:
            debug_frame = copy.deepcopy(frame)
            debug_frame = self.draw_pose_estimation_info(
                model_name,
                debug_frame,
                results_list,
                score_th,
            )

            texture = self.convert_cv_to_dpg(
                debug_frame,
                self.small_window_w,
                self.small_window_h,
            )
            dpg_set_value(self.output_value01_tag, texture)

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
