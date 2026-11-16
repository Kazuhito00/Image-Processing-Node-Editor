#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Object Detection node for CvStudio.

A single inference class (CustomONNX) handles all ONNX-based detection.
All models — built-in and user-uploaded — are managed through a persistent
registry (custom_models_registry.json).  Users add their own models via the
"Add Model" dialog which combines a file picker and an optional class-name
editor.
"""

import collections
import copy
import os
import sys
import shutil
import time

import cv2
import dearpygui.dearpygui as dpg
import numpy as np

from node_editor.util import dpg_get_value, dpg_set_value, run_unlocked
from node.basenode import Node
from node.DLNode.object_detection.CustomONNX.custom_onnx import CustomONNX
from node.DLNode.object_detection.BlazeFace.blazeface import BlazeFace
from node.DLNode.object_detection.coco_class_names import coco_class_names
from node.DLNode.object_detection import onnx_inspector
from node.DLNode.object_detection import custom_models_registry
from src.utils.logging import get_logger
from src.utils.gpu_utils import get_execution_providers

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Built-in models
# All inference goes through CustomONNX; these entries are written to the
# registry on first run so they appear in the model combo alongside any
# user-uploaded models.
# ---------------------------------------------------------------------------

_OBJECT_DETECTION_BASE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'object_detection'
)


def _resolve_builtin_model_path(*relative_candidates):
    """Return the first existing built-in model path, else the first candidate."""
    if not relative_candidates:
        return ""
    for rel in relative_candidates:
        candidate = os.path.join(_OBJECT_DETECTION_BASE, rel)
        if os.path.isfile(candidate):
            return candidate
    return os.path.join(_OBJECT_DETECTION_BASE, relative_candidates[0])


# Directory where user-uploaded ONNX models are stored permanently
if getattr(sys, 'frozen', False):
    from src.utils.paths import get_models_dir
    _UPLOADS_DIR = get_models_dir('object_detection')
else:
    _UPLOADS_DIR = os.path.join(_OBJECT_DETECTION_BASE, 'CustomONNX', 'models')

_COCO_CLASSES = {k: v for k, v in coco_class_names.items()}

# ---------------------------------------------------------------------------
# Class-name sources offered in the upload preview when an ONNX model does not
# embed its own class names.  This mirrors the behaviour of the built-in YOLO
# models which default to the COCO dataset labels.
# ---------------------------------------------------------------------------
_CLASS_SOURCE_COCO = "COCO dataset (80 classes)"
_CLASS_SOURCE_GENERIC = "Generic (class_0, class_1, …)"
_CLASS_SOURCE_OPTIONS = [_CLASS_SOURCE_COCO, _CLASS_SOURCE_GENERIC]


def build_class_names_from_source(source_label, num_classes):
    """Build a ``{int_id: str_name}`` class dict from a chosen source label.

    Used when the uploaded ONNX model does not embed class names, letting the
    user pick the COCO dataset labels (like the built-in YOLO models) or fall
    back to generic ``class_<i>`` placeholders.
    """
    try:
        num_classes = int(num_classes)
    except (TypeError, ValueError):
        num_classes = 0

    if source_label == _CLASS_SOURCE_COCO:
        count = num_classes if num_classes > 0 else len(coco_class_names)
        return {i: coco_class_names.get(i, f"class_{i}") for i in range(count)}

    # Generic fallback
    return {i: f"class_{i}" for i in range(num_classes)}

_BUILTIN_MODELS = [
    {
        'name': 'YOLOX-Nano(416x416)',
        'path': os.path.join(_OBJECT_DETECTION_BASE, 'YOLOX', 'model', 'yolox_nano.onnx'),
        'output_format': 'yolox',
        'input_width': 416,
        'input_height': 416,
        'num_classes': 80,
        'class_names': _COCO_CLASSES,
    },
    {
        'name': 'YOLOX-Tiny(416x416)',
        'path': os.path.join(_OBJECT_DETECTION_BASE, 'YOLOX', 'model', 'yolox_tiny.onnx'),
        'output_format': 'yolox',
        'input_width': 416,
        'input_height': 416,
        'num_classes': 80,
        'class_names': _COCO_CLASSES,
    },
    {
        'name': 'YOLOX-S(640x640)',
        'path': os.path.join(_OBJECT_DETECTION_BASE, 'YOLOX', 'model', 'yolox_s.onnx'),
        'output_format': 'yolox',
        'input_width': 640,
        'input_height': 640,
        'num_classes': 80,
        'class_names': _COCO_CLASSES,
    },
    {
        'name': 'YOLO11Nano',
        'path': os.path.join(_OBJECT_DETECTION_BASE, 'YOLO', 'model', 'yolo11_n.onnx'),
        'output_format': 'yolo11',
        'input_width': 608,
        'input_height': 416,
        'num_classes': 80,
        'class_names': _COCO_CLASSES,
    },
    {
        'name': 'yolo8n_B',
        'path': _resolve_builtin_model_path(
            os.path.join('YOLO', 'model', 'yolov8n.onnx'),
            os.path.join('YOLO', 'model', 'yolo11_n.onnx'),
        ),
        'output_format': 'yolo11',
        'input_width': 640,
        'input_height': 640,
        'num_classes': 80,
        'class_names': _COCO_CLASSES,
        'supports_batched_detection': True,
    },
    {
        'name': 'YOLOv8m(640x640)',
        'path': os.path.join(_OBJECT_DETECTION_BASE, 'YOLO', 'model', 'yolov8m.onnx'),
        'output_format': 'yolo11',
        'input_width': 640,
        'input_height': 640,
        'num_classes': 80,
        'class_names': _COCO_CLASSES,
    },
    {
        'name': 'yolo8s_B',
        'path': _resolve_builtin_model_path(
            os.path.join('YOLO', 'model', 'yolov8s.onnx'),
            os.path.join('YOLO', 'model', 'yolo11_n.onnx'),
        ),
        'output_format': 'yolo11',
        'input_width': 640,
        'input_height': 640,
        'num_classes': 80,
        'class_names': _COCO_CLASSES,
        'supports_batched_detection': True,
    },
    {
        'name': 'FreeYOLO-Nano(640x640)',
        'path': os.path.join(_OBJECT_DETECTION_BASE, 'FreeYOLO', 'model', 'yolo_free_nano_640x640.onnx'),
        'output_format': 'yolox',
        'input_width': 640,
        'input_height': 640,
        'num_classes': 80,
        'class_names': _COCO_CLASSES,
    },
    {
        'name': 'FreeYOLO-CrowdHuman(640x640)',
        'path': os.path.join(_OBJECT_DETECTION_BASE, 'FreeYOLO', 'model', 'yolo_free_nano_crowdhuman_640x640.onnx'),
        'output_format': 'yolox',
        'input_width': 640,
        'input_height': 640,
        'num_classes': 1,
        'class_names': {0: 'person'},
    },
    {
        'name': 'Light-Weight Person Detector',
        'path': os.path.join(_OBJECT_DETECTION_BASE, 'LightWeightPersonDetector', 'model', 'model.onnx'),
        'output_format': 'yolox',
        'input_width': 192,
        'input_height': 192,
        'num_classes': 1,
        'class_names': {0: 'person'},
    },
    {
        'name': 'YOLOTENNIS',
        'path': os.path.join(_OBJECT_DETECTION_BASE, 'TennisYOLO', 'model', 'tennis.onnx'),
        'output_format': 'yolo11',
        'input_width': 608,
        'input_height': 608,
        'num_classes': 3,
        'class_names': {0: 'player1', 1: 'player2', 2: 'ball'},
    },
    {
        'name': 'YOLO-DOTA-OBB(416x416)',
        'path': os.path.join(_OBJECT_DETECTION_BASE, 'YOLO', 'model', 'yolo-dota.onnx'),
        'output_format': 'yolo11_obb',
        'input_width': 416,
        'input_height': 416,
        'num_classes': 16,
        'class_names': {
            0: 'plane', 1: 'ship', 2: 'storage tank', 3: 'baseball diamond',
            4: 'tennis court', 5: 'basketball court', 6: 'ground track field',
            7: 'harbor', 8: 'bridge', 9: 'large vehicle', 10: 'small vehicle',
            11: 'helicopter', 12: 'roundabout', 13: 'soccer ball field',
            14: 'swimming pool', 15: 'container crane',
        },
    },
    {
        'name': 'BlazeFace(128x128)',
        'path': os.path.join(_OBJECT_DETECTION_BASE, 'blaze.onnx'),
        'output_format': 'blazeface',
        'input_width': 128,
        'input_height': 128,
        'num_classes': 1,
        'class_names': {0: 'face'},
    },
]

# Set of built-in model names — used to block deletion of default models.
_BUILTIN_MODEL_NAMES: set = {m['name'] for m in _BUILTIN_MODELS}

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------



def get_class_rejection_dropdown_items(class_name_dict):
    """Return sorted ``["id: name", ...]`` list for the rejection combo."""
    return [f"{class_id}: {class_name_dict[class_id]}"
            for class_id in sorted(class_name_dict.keys())]


def get_batch_badge_label(is_batch_capable):
    """Return the label shown beside the model combo for true batch-capable models."""
    return "B" if is_batch_capable else ""



class FactoryNode:
    node_label = 'ObjectDetection'
    node_tag = 'ObjectDetection'
    

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
        
        node.tag_node_input_image_name = node.tag_node_name + ':' + node.TYPE_IMAGE + ':Input01'
        node.tag_node_input_image_value_name = node.tag_node_name + ':' + node.TYPE_IMAGE + ':Input01Value'
        
        node.tag_node_input_text_name = node.tag_node_name + ':' + node.TYPE_TEXT + ':Input02'
        node.tag_node_input_text_value_name = node.tag_node_name + ':' + node.TYPE_TEXT + ':Input02Value'
        node.tag_node_batch_badge_value_name = node.tag_node_name + ':BatchBadgeValue'
        

        node.tag_node_input_float_name = node.tag_node_name + ':' + node.TYPE_FLOAT + ':Input03'
        node.tag_node_input_float_value_name = node.tag_node_name + ':' + node.TYPE_FLOAT + ':Input03Value'
        
        # Tag for rejected classes input field
        node.tag_node_rejected_classes_name = node.tag_node_name + ':RejectedClasses'
        node.tag_node_rejected_classes_value_name = node.tag_node_name + ':RejectedClassesValue'
        
        # Tag for draw bounding boxes checkbox
        node.tag_node_draw_bbox_name = node.tag_node_name + ':DrawBBox'
        node.tag_node_draw_bbox_value_name = node.tag_node_name + ':DrawBBoxValue'

        # Tag for bounding box thickness slider
        node.tag_node_bbox_thickness_name = node.tag_node_name + ':BBoxThickness'
        node.tag_node_bbox_thickness_value_name = node.tag_node_name + ':BBoxThicknessValue'

        # Tag for TensorRT batch-size slider
        node.tag_node_batch_size_name = node.tag_node_name + ':BatchSize'
        node.tag_node_batch_size_value_name = node.tag_node_name + ':BatchSizeValue'

        # Callback to update rejected classes dropdown when model changes
        def on_model_change(sender, app_data, user_data):
            """Update the rejected classes dropdown when model selection changes"""
            selected_model = app_data
            if selected_model in node._model_class_name_list:
                class_names = node._model_class_name_list[selected_model]
                class_items = get_class_rejection_dropdown_items(class_names)
                # Update the dropdown items
                dpg.configure_item(node.tag_node_rejected_classes_value_name, items=class_items)
                # Clear the rejected classes selection to avoid invalid class IDs
                dpg_set_value(node.tag_node_rejected_classes_value_name, "")
            # Disable the delete button for built-in (non-deletable) models
            try:
                is_builtin = selected_model in _BUILTIN_MODEL_NAMES
                dpg.configure_item(node.tag_delete_btn, enabled=not is_builtin)
            except Exception:
                pass
            node._update_batch_badge(selected_model)

        node.tag_node_output_image_name = node.tag_node_name + ':' + node.TYPE_IMAGE + ':Output01'
        node.tag_node_output_image = node.tag_node_name + ':' + node.TYPE_IMAGE + ':Output01Value'
        
        node.tag_node_output_result_name = node.tag_node_name + ':' + node.TYPE_TIME_MS + ':Output02'
        node.tag_node_output_result = node.tag_node_name + ':' + node.TYPE_TIME_MS + ':Output02Value'
        
        node.tag_node_output_json_name = node.tag_node_name + ':' + node.TYPE_JSON + ':Output03'
        node.tag_node_output_json = node.tag_node_name + ':' + node.TYPE_JSON + ':Output03Value'
        
        
        

        node.tag_provider_select_name = node.tag_node_name + ':' + node.TYPE_TEXT + ':Provider'
        node.tag_provider_select_value_name = node.tag_node_name + ':' + node.TYPE_IMAGE + ':ProviderValue'
        node._opencv_setting_dict = opencv_setting_dict
        

        small_window_w = node._opencv_setting_dict['process_width']
        small_window_h = node._opencv_setting_dict['process_height']
        use_pref_counter = node._opencv_setting_dict['use_pref_counter']
        use_gpu = node._opencv_setting_dict['use_gpu']


        black_image = np.zeros((small_window_w, small_window_h, 3))
        black_texture = node.convert_cv_to_dpg(
            black_image,
            small_window_w,
            small_window_h,
        )


        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                small_window_w,
                small_window_h,
                black_texture,
                tag=node.tag_node_output_image,
                format=dpg.mvFormat_Float_rgb,
            )

        # Create yellow theme for JSON button
        with dpg.theme() as yellow_button_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (255, 255, 153, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (255, 255, 153, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (255, 255, 153, 255))

        onnx_file_dialog_tag = "onnx_select:" + str(node_id)
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

        # ---- ONNX preview / confirmation dialog ----------------------------
        preview_window_tag  = "onnx_preview_window:"  + str(node_id)
        preview_name_tag    = "onnx_preview_name:"    + str(node_id)
        preview_details_tag = "onnx_preview_details:" + str(node_id)
        preview_status_tag  = "onnx_preview_status:"  + str(node_id)
        preview_confirm_tag = "onnx_preview_confirm:" + str(node_id)
        preview_cancel_tag  = "onnx_preview_cancel:"  + str(node_id)
        preview_quit_tag    = "onnx_preview_quit:"    + str(node_id)
        preview_class_source_group_tag = "onnx_preview_class_source_group:" + str(node_id)
        preview_class_source_tag       = "onnx_preview_class_source:"       + str(node_id)

        node.tag_preview_window  = preview_window_tag
        node.tag_preview_name    = preview_name_tag
        node.tag_preview_details = preview_details_tag
        node.tag_preview_status  = preview_status_tag
        node.tag_preview_confirm = preview_confirm_tag
        node.tag_preview_cancel  = preview_cancel_tag
        node.tag_preview_quit    = preview_quit_tag
        node.tag_preview_class_source_group = preview_class_source_group_tag
        node.tag_preview_class_source       = preview_class_source_tag

        def _on_upload_confirm(sender, app_data, user_data):
            node._do_confirm_upload()

        def _on_close_preview(sender, app_data, user_data):
            node._close_upload_preview()

        def _on_class_source_change(sender, app_data, user_data):
            node._on_class_source_change(app_data)

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
            # Dynamic details area — cleared and repopulated on each upload
            dpg.add_group(tag=preview_details_tag)
            # Class-source picker — only shown when the ONNX has no embedded
            # class names, letting the user choose COCO labels like the
            # built-in YOLO models (or generic placeholders).
            with dpg.group(tag=preview_class_source_group_tag, show=False):
                dpg.add_separator()
                dpg.add_text(
                    "No class names embedded in this model.\n"
                    "Choose which class labels to use:",
                    color=(255, 200, 100, 255),
                )
                dpg.add_combo(
                    _CLASS_SOURCE_OPTIONS,
                    default_value=_CLASS_SOURCE_COCO,
                    width=410,
                    tag=preview_class_source_tag,
                    callback=_on_class_source_change,
                )
            dpg.add_separator()
            dpg.add_text("", tag=preview_status_tag)
            dpg.add_spacer(height=4)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="  Confirm Upload  ",
                    tag=preview_confirm_tag,
                    callback=_on_upload_confirm,
                )
                dpg.add_spacer(width=10)
                dpg.add_button(
                    label="  Cancel  ",
                    tag=preview_cancel_tag,
                    callback=_on_close_preview,
                )
                dpg.add_spacer(width=10)
                dpg.add_button(
                    label="  Quit  ",
                    tag=preview_quit_tag,
                    callback=_on_close_preview,
                    show=False,
                )

        with dpg.node(
                tag=node.tag_node_name,
                parent=parent,
                label=node.node_label,
                pos=pos,
        ):

            with dpg.node_attribute(
                    tag=node.tag_node_input_image_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_text(
                    tag=node.tag_node_input_image_value_name,
                    default_value='Image',
                )

            with dpg.node_attribute(
                    tag=node.tag_node_output_image_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_image(node.tag_node_output_image)

            with dpg.node_attribute(
                    tag=node.tag_node_input_text_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                default_model = list(node._model_class.keys())[0]
                with dpg.group(horizontal=True):
                    dpg.add_combo(
                        list(node._model_class.keys()),
                        default_value=default_model,
                        width=small_window_w - 32,
                        tag=node.tag_node_input_text_value_name,
                        callback=on_model_change,
                    )
                    dpg.add_text(
                        get_batch_badge_label(node._model_batch_capable.get(default_model, False)),
                        tag=node.tag_node_batch_badge_value_name,
                        color=(120, 255, 120, 255),
                    )

            # ---- Collapse / expand toggle for the settings section ----------
            # The configuration widgets below (provider, score, reject, draw,
            # thickness) make the node tall.  This arrow folds them away.
            node.tag_collapse_btn = node.tag_node_name + ':CollapseToggle'
            node._collapsible_attr_tags = []
            node._settings_collapsed = False

            def _on_collapse_toggle(sender, app_data, user_data):
                node._settings_collapsed = not node._settings_collapsed
                for attr_tag in node._collapsible_attr_tags:
                    try:
                        dpg.configure_item(attr_tag, show=not node._settings_collapsed)
                    except Exception:
                        pass
                try:
                    dpg.configure_item(
                        node.tag_collapse_btn,
                        label=(u"\u25B6 Settings" if node._settings_collapsed
                               else u"\u25BC Settings"),
                    )
                except Exception:
                    pass

            with dpg.node_attribute(
                    tag=node.tag_node_name + ':CollapseAttr',
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_button(
                    label=u"\u25BC Settings",
                    tag=node.tag_collapse_btn,
                    width=small_window_w,
                    callback=_on_collapse_toggle,
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
                node._collapsible_attr_tags.append(node.tag_provider_select_name)

                with dpg.node_attribute(
                        tag=node.tag_node_batch_size_name,
                        attribute_type=dpg.mvNode_Attr_Static,
                ):
                    dpg.add_slider_int(
                        tag=node.tag_node_batch_size_value_name,
                        label="Batch",
                        width=small_window_w - 80,
                        default_value=node.DEFAULT_BATCH_SIZE,
                        min_value=1,
                        max_value=node.MAX_BATCH_SIZE,
                    )
                node._collapsible_attr_tags.append(node.tag_node_batch_size_name)

            with dpg.node_attribute(
                    tag=node.tag_node_input_float_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_slider_float(
                    tag=node.tag_node_input_float_value_name,
                    label="score",
                    width=small_window_w - 80,
                    default_value=0.3,
                    min_value=node._min_val,
                    max_value=node._max_val,
                    callback=None,
                )
            node._collapsible_attr_tags.append(node.tag_node_input_float_name)

            # Rejected classes dropdown
            with dpg.node_attribute(
                    tag=node.tag_node_rejected_classes_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                # Get class names for the default model to populate dropdown
                default_model = list(node._model_class.keys())[0]
                default_class_names = node._model_class_name_list[default_model]
                class_items = get_class_rejection_dropdown_items(default_class_names)
                
                dpg.add_combo(
                    tag=node.tag_node_rejected_classes_value_name,
                    label="Reject",
                    items=class_items,
                    width=small_window_w - 80,
                    default_value="",
                )
            node._collapsible_attr_tags.append(node.tag_node_rejected_classes_name)
            
            # Draw bounding boxes checkbox
            with dpg.node_attribute(
                    tag=node.tag_node_draw_bbox_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_checkbox(
                    tag=node.tag_node_draw_bbox_value_name,
                    label="Draw Bounding Boxes",
                    default_value=True,
                )
            node._collapsible_attr_tags.append(node.tag_node_draw_bbox_name)

            # Bounding box thickness slider
            with dpg.node_attribute(
                    tag=node.tag_node_bbox_thickness_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_slider_int(
                    tag=node.tag_node_bbox_thickness_value_name,
                    label="thickness",
                    width=small_window_w - 80,
                    default_value=3,
                    min_value=1,
                    max_value=10,
                )
            node._collapsible_attr_tags.append(node.tag_node_bbox_thickness_name)

            if use_pref_counter:
                with dpg.node_attribute(
                        tag=node.tag_node_output_result_name,
                        attribute_type=dpg.mvNode_Attr_Output,
                ):
                    dpg.add_text(
                        tag=node.tag_node_output_result,
                        default_value='elapsed time(ms)',
                    )
            
            # JSON output button
            with dpg.node_attribute(
                    tag=node.tag_node_output_json_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                btn = dpg.add_button(
                    label="JSON",
                    tag=node.tag_node_output_json,
                    width=small_window_w,
                    enabled=False,
                )
                dpg.bind_item_theme(btn, yellow_button_theme)

            # ---- Add model button (opens upload dialog on click) ------------
            node.tag_upload_btn = node.tag_node_name + ':UploadONNX'

            def _on_upload_clicked(sender, app_data, user_data):
                logger.info(
                    f"[Upload] 'Add Model' button clicked — "
                    f"showing file dialog '{onnx_file_dialog_tag}'"
                )
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

                upload_btn = dpg.add_button(
                    label=u"Add Model",
                    tag=node.tag_upload_btn,
                    width=small_window_w,
                    callback=_on_upload_clicked,
                )
                dpg.bind_item_theme(upload_btn, add_model_btn_theme)

            # ---- Delete model button (removes the selected model) -----------
            node.tag_delete_btn = node.tag_node_name + ':DeleteONNX'

            def _on_delete_clicked(sender, app_data, user_data):
                model_combo_tag = node.tag_node_input_text_value_name
                selected = dpg_get_value(model_combo_tag)
                node._delete_selected_model(selected)

            with dpg.node_attribute(
                    tag=node.tag_node_name + ':DeleteAttr',
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
                default_selected = list(node._model_class.keys())[0] if node._model_class else ""
                if default_selected in _BUILTIN_MODEL_NAMES:
                    dpg.configure_item(delete_btn, enabled=False)



        return node




class Node(Node):
    _ver = '0.0.1'
    node_label = 'ObjectDetection'
    node_tag = 'ObjectDetection'

    _min_val = 0.0
    _max_val = 1.0

    _opencv_setting_dict = None

    DEFAULT_DRAW_BBOX = True
    DEFAULT_BBOX_THICKNESS = 3
    DEFAULT_BATCH_SIZE = 1
    MAX_BATCH_SIZE = 32
    CUDA_PROVIDER_OPTIONS = {
        # Keep CUDA host-side allocation conservative to reduce GPU memory
        # spikes in both standard CUDA execution and TensorRT CUDA fallback.
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_use_max_workspace": "0",
    }

    # All models (built-in + user-uploaded) populated from the registry at load time.
    _model_class: dict = {}           # name → CustomONNX factory callable
    _model_path_setting: dict = {}    # name → onnx file path
    _model_class_name_list: dict = {} # name → {int_id: str_name}
    _model_batch_capable: dict = {}   # name → supports true batched OD inference

    _model_instance: dict = {}

    # Batch-inference state (keyed by model_name_with_provider).
    # _frame_buffer  : deques of raw frames pending batch inference.
    # _result_buffer : deques of (bboxes, scores, class_ids) ready to emit.
    # _last_batch_size: last seen batch_size per key (detect slider changes).
    _frame_buffer: dict = {}
    _result_buffer: dict = {}
    _last_batch_size: dict = {}

    def __init__(self):
        pass

    @classmethod
    def _ensure_builtin_models(cls):
        """Write built-in model entries to the registry if they are not already there.

        This runs once at module load time so that built-in ONNX files appear in the
        model combo without requiring the user to upload them manually.  Entries are
        skipped when the ONNX file does not exist on disk (e.g., stripped builds).
        """
        try:
            existing = {e.get('name') for e in custom_models_registry.load_registry()}
        except Exception as exc:
            logger.warning(f"[Builtin] Could not read registry: {exc}")
            return
        for meta in _BUILTIN_MODELS:
            name = meta['name']
            path = meta['path']
            if name in existing:
                continue
            if not os.path.isfile(path):
                logger.debug(f"[Builtin] Skipping '{name}' — ONNX file not found: {path}")
                continue
            supports_batch = meta.get('supports_batched_detection', None)
            if supports_batch is None:
                supports_batch = False
                try:
                    inspected = onnx_inspector.inspect_onnx_model(path)
                    supports_batch = bool(inspected.get('supports_batched_detection', False))
                except Exception as exc:
                    logger.warning(f"[Builtin] Could not inspect batch support for '{name}': {exc}")
            else:
                supports_batch = bool(supports_batch)
            # Registry always stores class_names with string keys
            entry = {
                'name': name,
                'path': path,
                'output_format': meta['output_format'],
                'input_width': meta['input_width'],
                'input_height': meta['input_height'],
                'num_classes': meta['num_classes'],
                'class_names': {str(k): v for k, v in meta['class_names'].items()},
                'supports_batched_detection': supports_batch,
            }
            if meta.get('disable_optimizations'):
                entry['disable_optimizations'] = True
            if meta.get('nanodet_reg_first') is not None:
                entry['nanodet_reg_first'] = meta['nanodet_reg_first']
            try:
                custom_models_registry.save_entry(entry)
                logger.info(f"[Builtin] Registered built-in model: {name}")
            except Exception as exc:
                logger.warning(f"[Builtin] Could not register '{name}': {exc}")

    @classmethod
    def _load_custom_models_from_registry(cls):
        """Load all models (built-in and user-uploaded) from the registry.

        Populates ``_model_class``, ``_model_path_setting`` and
        ``_model_class_name_list``.  Called once at module load time and again
        after each successful upload.
        """
        try:
            entries = custom_models_registry.load_registry()
        except Exception as exc:
            logger.warning(f"Failed to load models registry: {exc}")
            return
        for entry in entries:
            name = entry.get('name', '')
            path = entry.get('path', '')
            if not name or not path:
                continue
            if name in cls._model_class:
                continue
            raw_classes = entry.get('class_names', {})
            class_names = {int(k): str(v) for k, v in raw_classes.items()}
            if not class_names:
                num_classes = int(entry.get('num_classes', 0))
                class_names = (
                    {i: f"class_{i}" for i in range(num_classes)}
                    if num_classes > 0
                    else dict(coco_class_names)
                )
            output_fmt = entry.get('output_format', 'yolo11')
            in_w = int(entry.get('input_width', 640))
            in_h = int(entry.get('input_height', 640))
            disable_opt = bool(entry.get('disable_optimizations', False))
            nanodet_reg_first = entry.get('nanodet_reg_first', None)
            if nanodet_reg_first is not None:
                nanodet_reg_first = bool(nanodet_reg_first)
            supports_batch = entry.get('supports_batched_detection', None)
            if supports_batch is None:
                try:
                    meta = onnx_inspector.inspect_onnx_model(path)
                    supports_batch = bool(meta.get('supports_batched_detection', False))
                    entry['supports_batched_detection'] = supports_batch
                except Exception as exc:
                    logger.warning(f"Could not inspect batch support for '{name}': {exc}")
                    supports_batch = False
                else:
                    try:
                        custom_models_registry.save_entry(entry)
                    except Exception as exc:
                        logger.warning(f"Could not persist batch support for '{name}': {exc}")
            cls._register_custom_model(name, path, class_names, output_fmt, in_w, in_h,
                                       disable_optimizations=disable_opt,
                                       nanodet_reg_first=nanodet_reg_first,
                                       supports_batch=bool(supports_batch))
            logger.info(f"Loaded model from registry: {name}")

    @classmethod
    def _register_custom_model(cls, name, path, class_names, output_fmt, in_w, in_h,
                               disable_optimizations=False, nanodet_reg_first=None,
                               supports_batch=False):
        """Add a model to the class-level runtime dictionaries."""
        def _make_factory(p, fmt, w, h, disable_opt, nd_reg_first):
            def factory(model_path, providers=None, provider_options=None):
                if providers is None:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                if fmt == 'blazeface':
                    return BlazeFace(
                        model_path=model_path,
                        providers=providers,
                    )
                return CustomONNX(
                    model_path=model_path,
                    input_width=w,
                    input_height=h,
                    output_format=fmt,
                    providers=providers,
                    provider_options=provider_options,
                    disable_optimizations=disable_opt,
                    nanodet_reg_first=nd_reg_first,
                )
            return factory

        cls._model_class[name] = _make_factory(path, output_fmt, in_w, in_h, disable_optimizations, nanodet_reg_first)
        cls._model_path_setting[name] = path
        cls._model_class_name_list[name] = class_names
        cls._model_batch_capable[name] = bool(supports_batch)

    # ------------------------------------------------------------------
    # Upload callback
    # ------------------------------------------------------------------

    def _callback_onnx_select(self, sender, data, user_data=None):
        """Handle ONNX file selection from the file dialog.

        Inspects the selected ONNX file to extract metadata and class names,
        then shows the preview/confirmation dialog instead of registering
        the model immediately.
        """
        logger.info(f"[Upload] File dialog callback triggered — sender={sender}, data={data}")
        if data.get("file_name") == ".":
            logger.info("[Upload] User cancelled the file dialog (file_name='.').")
            return
        onnx_path = data.get("file_path_name", "")
        logger.info(f"[Upload] Selected file path: '{onnx_path}'")
        if not onnx_path or not os.path.isfile(onnx_path):
            logger.warning(f"[Upload] No valid ONNX file selected (path='{onnx_path}').")
            return

        try:
            logger.info(f"[Upload] Inspecting ONNX model: {onnx_path}")
            meta = onnx_inspector.inspect_onnx_model(onnx_path)
            logger.info(
                f"[Upload] Inspection result: format='{meta.get('output_format')}', "
                f"input={meta.get('input_width')}x{meta.get('input_height')}, "
                f"num_classes={meta.get('num_classes')}, "
                f"class_names_count={len(meta.get('class_names', {}))}, "
                f"supports_batch={meta.get('supports_batched_detection', False)}"
            )
        except Exception as exc:
            logger.error(f"[Upload] ONNX inspection failed: {exc}", exc_info=True)
            # Show error in the preview dialog
            self._pending_onnx_path = None
            self._pending_meta = None
            self._pending_class_names = None
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

        # Class names come from the ONNX file metadata when available.  If the
        # model embeds no class names, fall back to a user-selectable source
        # (COCO labels like the built-in YOLO models, or generic placeholders).
        embedded_class_names = meta.get("class_names", {})
        num_classes = meta.get("num_classes", 0)
        self._pending_classes_embedded = bool(embedded_class_names)

        if embedded_class_names:
            class_names = embedded_class_names
            logger.info("[Upload] Using class names embedded in ONNX metadata.")
        else:
            logger.info(
                f"[Upload] No class names in metadata — offering class-source "
                f"selection (num_classes={num_classes})."
            )
            class_names = build_class_names_from_source(_CLASS_SOURCE_COCO, num_classes)

        # Store pending upload data on the instance
        self._pending_onnx_path = onnx_path
        self._pending_meta = meta
        self._pending_class_names = class_names

        # Populate preview dialog
        base_name = os.path.splitext(os.path.basename(onnx_path))[0]
        dpg.set_value(self.tag_preview_name, base_name)

        # Show/hide the class-source picker depending on whether the ONNX
        # already provides class names.
        try:
            if self._pending_classes_embedded:
                dpg.configure_item(self.tag_preview_class_source_group, show=False)
            else:
                dpg.set_value(self.tag_preview_class_source, _CLASS_SOURCE_COCO)
                dpg.configure_item(self.tag_preview_class_source_group, show=True)
        except Exception:
            pass

        self._render_preview_details(meta, class_names)

        # Clear any previous status message and show the dialog
        self._set_upload_preview_actions(upload_succeeded=False)
        dpg.set_value(self.tag_preview_status, "")
        dpg.show_item(self.tag_preview_window)

    def _render_preview_details(self, meta, class_names):
        """Render the model metadata + class list into the preview dialog."""
        dpg.delete_item(self.tag_preview_details, children_only=True)

        in_w = meta.get("input_width", 640)
        in_h = meta.get("input_height", 640)
        out_fmt = meta.get("output_format", "unknown")
        num_cls = meta.get("num_classes", len(class_names))

        dpg.add_text(
            f"Input dimensions : {in_w} x {in_h} px (W x H)",
            parent=self.tag_preview_details,
        )
        dpg.add_text(
            f"Output format    : {out_fmt}",
            parent=self.tag_preview_details,
        )
        dpg.add_text(
            f"Number of classes: {num_cls}",
            parent=self.tag_preview_details,
        )
        dpg.add_text(
            "Dynamic batch     : "
            + ("yes" if meta.get("has_dynamic_batch", False) else "no"),
            parent=self.tag_preview_details,
        )
        dpg.add_text(
            "ObjectDetection B : "
            + ("yes" if meta.get("supports_batched_detection", False) else "no"),
            parent=self.tag_preview_details,
        )

        if class_names:
            dpg.add_text("Class list:", parent=self.tag_preview_details)
            max_show = 40
            for cid, cname in sorted(class_names.items(), key=lambda x: x[0])[:max_show]:
                dpg.add_text(f"  {cid}: {cname}", parent=self.tag_preview_details)
            if len(class_names) > max_show:
                dpg.add_text(
                    f"  … and {len(class_names) - max_show} more classes",
                    parent=self.tag_preview_details,
                )
        else:
            dpg.add_text(
                "(No class names found in model metadata)",
                parent=self.tag_preview_details,
                color=(255, 200, 100, 255),
            )

    def _on_class_source_change(self, source_label):
        """Rebuild pending class names when the user changes the class source."""
        meta = getattr(self, '_pending_meta', None)
        if meta is None:
            return
        num_classes = meta.get("num_classes", 0)
        class_names = build_class_names_from_source(source_label, num_classes)
        self._pending_class_names = class_names
        self._render_preview_details(meta, class_names)


    def _set_upload_preview_actions(self, upload_succeeded: bool):
        """Show the appropriate action buttons for the current preview state."""
        dpg.configure_item(self.tag_preview_confirm, show=not upload_succeeded)
        dpg.configure_item(self.tag_preview_cancel, show=not upload_succeeded)
        dpg.configure_item(self.tag_preview_quit, show=upload_succeeded)

    def _close_upload_preview(self):
        """Close the upload preview dialog."""
        dpg.hide_item(self.tag_preview_window)

    def _do_confirm_upload(self):
        """Called when the user clicks 'Confirm Upload' in the preview dialog.

        Copies the ONNX file to the dedicated uploads directory, registers
        the model, updates the combobox, and reports success or failure.
        """
        onnx_path = getattr(self, '_pending_onnx_path', None)
        meta = getattr(self, '_pending_meta', None)
        class_names = getattr(self, '_pending_class_names', None)

        if not onnx_path or meta is None:
            dpg.set_value(self.tag_preview_status, "No pending upload — please select a file first.")
            return

        # Read the (possibly edited) model name from the dialog
        custom_name = dpg.get_value(self.tag_preview_name).strip()
        if not custom_name:
            custom_name = os.path.splitext(os.path.basename(onnx_path))[0]

        # Copy ONNX to the dedicated uploads directory
        os.makedirs(_UPLOADS_DIR, exist_ok=True)
        dest_path = onnx_path
        try:
            basename = os.path.basename(onnx_path)
            candidate = os.path.join(_UPLOADS_DIR, basename)
            if os.path.abspath(onnx_path) != os.path.abspath(candidate):
                shutil.copy2(onnx_path, candidate)
                dest_path = candidate
                logger.info(f"[Upload] Copied ONNX to: {dest_path}")
            else:
                logger.info("[Upload] Source and destination are the same — skipping copy.")
        except Exception as exc:
            logger.warning(f"[Upload] Could not copy ONNX to uploads dir: {exc}")
            # Fall back to using the original path
            dest_path = onnx_path

        # Finalise registration
        try:
            Node._finalise_upload(self, dest_path, meta, class_names, custom_name=custom_name)
            dpg.set_value(
                self.tag_preview_status,
                f"\u2713 Model '{custom_name}' uploaded successfully!",
            )
            self._set_upload_preview_actions(upload_succeeded=True)
            logger.info(f"[Upload] Upload confirmed for '{custom_name}'.")
        except Exception as exc:
            logger.error(f"[Upload] Finalise failed: {exc}", exc_info=True)
            dpg.set_value(
                self.tag_preview_status,
                f"\u2717 Upload failed: {exc}",
            )
            self._set_upload_preview_actions(upload_succeeded=False)

        # Clear pending state
        self._pending_onnx_path = None
        self._pending_meta = None
        self._pending_class_names = None

    @staticmethod
    def _finalise_upload(node, onnx_path: str, meta: dict, class_names: dict, custom_name: str = None):
        """Register the custom model and refresh the node UI dropdowns.

        Parameters
        ----------
        node : Node
            The node instance whose UI should be updated.
        onnx_path : str
            Path to the (possibly already-copied) ONNX file.
        meta : dict
            Metadata from ``onnx_inspector.inspect_onnx_model()``.
        class_names : dict
            ``{int_id: str_name}`` mapping for all classes.
        custom_name : str, optional
            Display name chosen by the user; defaults to the filename stem.
        """
        # Determine unique display name
        base = custom_name if custom_name else os.path.splitext(os.path.basename(onnx_path))[0]
        name = base
        counter = 1
        while name in Node._model_class:
            name = f"{base}_{counter}"
            counter += 1

        output_fmt = meta.get("output_format", "yolo11")
        in_w = meta.get("input_width", 640)
        in_h = meta.get("input_height", 640)
        num_classes = meta.get("num_classes", len(class_names))

        logger.info(
            f"[Upload] Registering '{name}' — format='{output_fmt}', "
            f"input={in_w}x{in_h}, classes={num_classes}"
        )

        supports_batch = bool(meta.get("supports_batched_detection", False))
        Node._register_custom_model(
            name,
            onnx_path,
            class_names,
            output_fmt,
            in_w,
            in_h,
            supports_batch=supports_batch,
        )

        registry_entry = {
            "name": name,
            "path": onnx_path,
            "class_names": {str(k): v for k, v in class_names.items()},
            "output_format": output_fmt,
            "input_width": in_w,
            "input_height": in_h,
            "num_classes": num_classes,
            "supports_batched_detection": supports_batch,
        }
        try:
            custom_models_registry.save_entry(registry_entry)
            logger.info(f"[Upload] Registry entry saved for '{name}'.")
        except Exception as exc:
            logger.warning(f"[Upload] Could not save registry entry for '{name}': {exc}")

        # Update model combobox
        model_combo_tag = node.tag_node_name + ':' + node.TYPE_TEXT + ':Input02Value'
        try:
            current_items = dpg.get_item_configuration(model_combo_tag).get("items", [])
            if name not in current_items:
                current_items = list(current_items) + [name]
            dpg.configure_item(model_combo_tag, items=current_items, default_value=name)
            node._update_batch_badge(name)
            logger.info(f"[Upload] Model dropdown updated — '{name}' selected.")
        except Exception as exc:
            logger.warning(f"[Upload] Could not update model dropdown: {exc}")

        # Update rejected-classes dropdown
        try:
            class_items = get_class_rejection_dropdown_items(class_names)
            dpg.configure_item(node.tag_node_rejected_classes_value_name, items=class_items)
            dpg_set_value(node.tag_node_rejected_classes_value_name, "")
        except Exception as exc:
            logger.warning(f"[Upload] Could not update classes dropdown: {exc}")

    # ------------------------------------------------------------------
    # Delete model
    # ------------------------------------------------------------------
    @classmethod
    def _delete_custom_model(cls, name: str) -> bool:
        """Remove a model from the persistent registry and runtime dictionaries.

        Built-in (default) models cannot be deleted.
        Returns True when the model was found and removed.
        """
        if not name:
            return False
        if name in _BUILTIN_MODEL_NAMES:
            logger.warning(f"[Delete] '{name}' is a built-in model and cannot be deleted.")
            return False
        found = name in cls._model_class or name in cls._model_path_setting
        # Delete the ONNX file from disk if it lives in the uploads directory
        onnx_path = cls._model_path_setting.get(name)
        if onnx_path:
            try:
                onnx_abs = os.path.abspath(onnx_path)
                uploads_abs = os.path.abspath(_UPLOADS_DIR)
                if (os.path.commonpath([onnx_abs, uploads_abs]) == uploads_abs
                        and onnx_abs != uploads_abs
                        and os.path.isfile(onnx_abs)):
                    os.remove(onnx_abs)
                    logger.info(f"[Delete] ONNX file deleted: {onnx_abs}")
            except Exception as exc:
                logger.warning(f"[Delete] Could not delete ONNX file for '{name}': {exc}")
        try:
            custom_models_registry.remove_entry(name)
            logger.info(f"[Delete] Registry entry removed for '{name}'.")
        except Exception as exc:
            logger.warning(f"[Delete] Could not remove registry entry for '{name}': {exc}")
        cls._model_class.pop(name, None)
        cls._model_path_setting.pop(name, None)
        cls._model_class_name_list.pop(name, None)
        cls._model_batch_capable.pop(name, None)
        # Drop any cached inference instances for this model (name_provider keys).
        for key in [k for k in cls._model_instance if k == name or k.startswith(name + '_')]:
            cls._model_instance.pop(key, None)
        return found

    def _delete_selected_model(self, name: str):
        """Delete the currently-selected model and refresh the node dropdowns."""
        if not name:
            logger.warning("[Delete] No model selected — nothing to delete.")
            return
        if name in _BUILTIN_MODEL_NAMES:
            logger.warning(f"[Delete] '{name}' is a built-in model and cannot be deleted.")
            return
        removed = Node._delete_custom_model(name)
        if not removed:
            logger.warning(f"[Delete] Model '{name}' not found in registry.")

        # Update the model combobox with the remaining models.
        model_combo_tag = self.tag_node_input_text_value_name
        remaining = list(Node._model_class.keys())
        new_default = remaining[0] if remaining else ""
        try:
            dpg.configure_item(model_combo_tag, items=remaining, default_value=new_default)
            self._update_batch_badge(new_default)
            logger.info(f"[Delete] Model '{name}' deleted — '{new_default}' now selected.")
        except Exception as exc:
            logger.warning(f"[Delete] Could not update model dropdown: {exc}")

        # Refresh the rejected-classes dropdown for the new selection.
        try:
            if new_default and new_default in Node._model_class_name_list:
                class_items = get_class_rejection_dropdown_items(
                    Node._model_class_name_list[new_default]
                )
            else:
                class_items = []
            dpg.configure_item(self.tag_node_rejected_classes_value_name, items=class_items)
            dpg_set_value(self.tag_node_rejected_classes_value_name, "")
        except Exception as exc:
            logger.warning(f"[Delete] Could not update classes dropdown: {exc}")

        # Update delete button state for the new selection
        try:
            is_builtin = new_default in _BUILTIN_MODEL_NAMES
            dpg.configure_item(self.tag_delete_btn, enabled=not is_builtin)
        except Exception as exc:
            logger.warning(f"[Delete] Could not update delete button state: {exc}")

    def _per_class_nms(self, bboxes, scores, class_ids):
        """Apply NMS per class to keep only the best detection per class.
        
        DEPRECATED: This method is no longer used as it was limiting detections to
        only 1 object per class. The method is kept for backward compatibility but
        should not be called. All detections from the model are now passed through
        without per-class filtering.
        
        Args:
            bboxes: List or numpy array of bounding boxes [x1, y1, x2, y2]
            scores: List or numpy array of confidence scores
            class_ids: List or numpy array of class IDs
            
        Returns:
            Filtered bboxes, scores, class_ids (as numpy arrays)
        """
        if len(bboxes) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Convert to numpy arrays
        bboxes = np.array(bboxes)
        scores = np.array(scores)
        class_ids = np.array(class_ids)
        
        # Get unique classes
        unique_classes = np.unique(class_ids)
        
        keep_indices = []
        
        for class_id in unique_classes:
            # Get indices for this class
            class_mask = class_ids == class_id
            class_bboxes = bboxes[class_mask]
            class_scores = scores[class_mask]
            class_indices = np.where(class_mask)[0]
            
            # If only one detection for this class, keep it
            if len(class_bboxes) == 1:
                keep_indices.append(class_indices[0])
                continue
            
            # Sort by score (descending)
            sorted_indices = np.argsort(-class_scores)
            
            # Keep the highest scoring detection for this class
            # This ensures only 1 bounding box per class
            keep_indices.append(class_indices[sorted_indices[0]])
        
        keep_indices = np.array(keep_indices)
        
        return bboxes[keep_indices], scores[keep_indices], class_ids[keep_indices]

    @staticmethod
    def _build_trt_provider_options(batch_size=1):
        """Build TensorRT provider options for a chosen max batch size.

        ``trt_max_workspace_size`` is scaled linearly with *batch_size*
        (1 GiB per batch slot) so the TensorRT engine builder has enough
        GPU scratch memory to optimise multi-frame execution plans.
        """
        batch_size = max(1, int(batch_size))
        cache_dir = os.path.join(_UPLOADS_DIR, "trt_engine_cache")
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except Exception as exc:
            logger.warning(f"Could not create TensorRT cache dir '{cache_dir}': {exc}")
            cache_dir = _UPLOADS_DIR
        workspace_bytes = batch_size * 1073741824  # 1 GiB × batch_size
        return {
            "trt_engine_cache_enable": "True",
            "trt_engine_cache_path": cache_dir,
            "trt_max_workspace_size": str(workspace_bytes),
        }

    def _update_batch_badge(self, model_name):
        """Refresh the small 'B' marker beside the model combo."""
        try:
            dpg_set_value(
                self.tag_node_batch_badge_value_name,
                get_batch_badge_label(self._model_batch_capable.get(model_name, False)),
            )
        except Exception as exc:
            logger.debug(f"Could not update batch badge for '{model_name}': {exc}")

    @staticmethod
    def _run_batch_inference(model_instance, frames):
        """Run inference on a list of frames, returning one result per frame.

        Uses ``model_instance.call_batch`` when available (``CustomONNX``),
        which submits a single batched ``session.run`` call for models that
        have a dynamic batch axis.  Falls back to sequential ``__call__`` for
        any model that does not expose ``call_batch``.

        Returns
        -------
        list[tuple[np.ndarray, np.ndarray, np.ndarray]]
            ``(bboxes, scores, class_ids)`` per frame.
        """
        if hasattr(model_instance, 'call_batch'):
            return model_instance.call_batch(frames)
        return [model_instance(f) for f in frames]



    def update(self, node_id, connection_list, node_image_dict, node_result_dict, node_audio_dict,):
            data = {}
            frame = None
            try:
                
                self.tag_node_name = str(node_id) + ':' + self.node_tag
                tag_node_output_image = self.tag_node_name + ':' + self.TYPE_IMAGE + ':Output01Value'
                self.tag_provider_select_value_name = self.tag_node_name + ':' + self.TYPE_IMAGE + ':ProviderValue'
                self.tag_node_rejected_classes_value_name = self.tag_node_name + ':RejectedClassesValue'
                self.tag_node_draw_bbox_value_name = self.tag_node_name + ':DrawBBoxValue'
                self.tag_node_batch_size_value_name = self.tag_node_name + ':BatchSizeValue'

                small_window_w = self._opencv_setting_dict['process_width']
                small_window_h = self._opencv_setting_dict['process_height']
                use_pref_counter = self._opencv_setting_dict['use_pref_counter']
                use_gpu = self._opencv_setting_dict['use_gpu']


                for connection_info in connection_list:
                    connection_type = connection_info[0].split(':')[2]
                    if connection_type == self.TYPE_FLOAT:
                        source_tag = connection_info[0] + 'Value'
                        destination_tag = connection_info[1] + 'Value'
                        logger.debug(f"Linking float: {source_tag} -> {destination_tag}")
                        input_value = round(float(dpg_get_value(source_tag)), 3)
                        input_value = max([self._min_val, input_value])
                        input_value = min([self._max_val, input_value])
                        dpg_set_value(destination_tag, input_value)

                frame = self.get_input_frame(connection_list, node_image_dict, node_audio_dict)

                # Get the timestamp of the source image for result alignment
                _image_source_timestamp = None
                for connection_info in connection_list:
                    connection_type = connection_info[0].split(':')[2]
                    if connection_type in [self.TYPE_IMAGE, self.TYPE_AUDIO]:
                        _src_node_id = ':'.join(connection_info[0].split(':')[:2])
                        if connection_type == self.TYPE_IMAGE:
                            _image_source_timestamp = node_image_dict.get_timestamp(_src_node_id)
                        elif node_audio_dict is not None:
                            _image_source_timestamp = node_audio_dict.get_timestamp(_src_node_id)
                        break

                if frame is not None:
                    logger.debug(f"Frame shape: {frame.shape}")


                try:
                    score_th = round(float(dpg_get_value(self.tag_node_input_float_value_name)), 3)
                except:
                    score_th = 0.3

                provider = 'cpu'
                if use_gpu:
                    provider = dpg_get_value(self.tag_provider_select_value_name)
                    if provider not in ('cpu', 'cuda', 'TRTcuda'):
                        provider = 'cpu'

                batch_size = self.DEFAULT_BATCH_SIZE
                if use_gpu and provider in ('cuda', 'TRTcuda'):
                    try:
                        batch_size = int(dpg_get_value(self.tag_node_batch_size_value_name))
                    except Exception:
                        batch_size = self.DEFAULT_BATCH_SIZE
                batch_size = max(1, min(self.MAX_BATCH_SIZE, batch_size))



                model_name = dpg_get_value(self.tag_node_input_text_value_name)
                # Fallback to first available model if the saved name is not registered
                if model_name not in self._model_class:
                    fallback = list(self._model_class.keys())[0]
                    logger.warning(
                        f"Model '{model_name}' not found in registry; falling back to '{fallback}'."
                    )
                    model_name = fallback
                model_path = self._model_path_setting[model_name]
                model_class = self._model_class[model_name]
                class_name_dict = self._model_class_name_list[model_name]

                model_name_with_provider = model_name + '_' + provider
                if provider == 'TRTcuda':
                    model_name_with_provider += f"_b{batch_size}"

                if frame is not None:
                    if model_name_with_provider not in self._model_instance:
                        provider_options = None
                        if provider == 'cpu':
                            providers = ['CPUExecutionProvider']
                        elif provider == 'TRTcuda':
                            trt_prefix = model_name + '_TRTcuda_'
                            for key in [k for k in self._model_instance if k.startswith(trt_prefix)]:
                                self._model_instance.pop(key, None)
                            providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
                            provider_options = [
                                self._build_trt_provider_options(batch_size),
                                dict(self.CUDA_PROVIDER_OPTIONS),
                                {},
                            ]
                        else:
                            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                            provider_options = [
                                dict(self.CUDA_PROVIDER_OPTIONS),
                                {},
                            ]

                        self._model_instance[model_name_with_provider] = model_class(
                            model_path,
                            providers=providers,
                            provider_options=provider_options,
                        )


                if frame is not None and use_pref_counter:
                    start_time = time.monotonic()

                result = {}
                debug_frame = None
                if frame is not None:
                    # For BlazeFace, propagate the score threshold to the model's
                    # conf_threshold so that the built-in NMS uses the same value.
                    model_instance = self._model_instance[model_name_with_provider]
                    if hasattr(model_instance, 'conf_threshold'):
                        model_instance.conf_threshold = score_th

                    # ------------------------------------------------------------------
                    # Batch inference logic
                    # When batch_size > 1 frames are accumulated into a rolling buffer.
                    # Every time the buffer reaches batch_size frames, a single batched
                    # call processes all frames at once (one session.run for CustomONNX
                    # models that have a dynamic batch axis).  Results are queued and
                    # drained one per update() call so every frame that arrives also
                    # produces an output — after the initial (batch_size - 1) frame
                    # warm-up the output is continuous.
                    # ------------------------------------------------------------------
                    key = model_name_with_provider
                    if batch_size <= 1:
                        # Fast path: single-frame inference (unchanged behaviour).
                        bboxes, scores, class_ids = run_unlocked(model_instance, frame)
                    else:
                        # Reset buffers whenever the user changes the batch slider.
                        if self._last_batch_size.get(key) != batch_size:
                            self._frame_buffer[key] = collections.deque()
                            self._result_buffer[key] = collections.deque()
                            self._last_batch_size[key] = batch_size

                        fb = self._frame_buffer[key]
                        rb = self._result_buffer[key]

                        # Always accumulate the current frame.
                        fb.append(copy.deepcopy(frame))

                        if rb:
                            # A previous batch already produced queued results.
                            bboxes, scores, class_ids = rb.popleft()
                        elif len(fb) >= batch_size:
                            # Buffer is full — run a batched inference call.
                            batch_frames = [fb.popleft() for _ in range(batch_size)]
                            batch_results = run_unlocked(
                                self._run_batch_inference, model_instance, batch_frames
                            )
                            rb.extend(batch_results)
                            bboxes, scores, class_ids = rb.popleft()
                            logger.debug(
                                f"[Batch] Processed {batch_size} frames in one call; "
                                f"{len(rb)} results queued."
                            )
                        else:
                            # Not enough frames yet — emit empty detections.
                            bboxes = np.empty((0, 4), dtype=np.float32)
                            scores = np.empty((0,), dtype=np.float32)
                            class_ids = np.empty((0,), dtype=np.int64)
                    
                    # Apply class rejection filter
                    if len(bboxes) > 0:
                        try:
                            rejected_classes_str = dpg_get_value(self.tag_node_rejected_classes_value_name)
                            if rejected_classes_str and rejected_classes_str.strip():
                                # Log the raw rejection string
                                logger.debug(f"Class rejection filter input: '{rejected_classes_str}'")
                                
                                # Parse the rejected class IDs
                                rejected_classes = set()
                                
                                # Handle dropdown format "0: person" or legacy format "0,1,2"
                                # Split by comma first
                                for class_str in rejected_classes_str.split(','):
                                    class_str = class_str.strip()
                                    if class_str:
                                        try:
                                            # If format is "ID: name", extract just the ID part
                                            if ':' in class_str:
                                                class_id_str = class_str.split(':')[0].strip()
                                                rejected_classes.add(int(class_id_str))
                                            else:
                                                # Legacy format: just the number
                                                rejected_classes.add(int(class_str))
                                        except ValueError:
                                            # Skip invalid class IDs
                                            pass
                                
                                # Validate rejected classes against model's class dictionary
                                valid_class_ids = set(class_name_dict.keys())
                                invalid_classes = rejected_classes - valid_class_ids
                                
                                if invalid_classes:
                                    logger.warning(f"Invalid class IDs for model '{model_name}': {invalid_classes}. "
                                                 f"Valid class IDs for this model: {sorted(valid_class_ids)}")
                                    # Filter out invalid class IDs
                                    rejected_classes = rejected_classes & valid_class_ids
                                
                                # Log before filtering
                                logger.debug(f"Before class rejection: {len(bboxes)} detections, class_ids={class_ids.tolist()}")
                                logger.debug(f"Rejected classes (validated): {rejected_classes}")
                                
                                # Filter out rejected classes
                                if rejected_classes:
                                    keep_mask = np.array([class_id not in rejected_classes for class_id in class_ids])
                                    bboxes = bboxes[keep_mask]
                                    scores = scores[keep_mask]
                                    class_ids = class_ids[keep_mask]
                                    
                                    # Log after filtering
                                    logger.debug(f"After class rejection: {len(bboxes)} detections, class_ids={class_ids.tolist()}")
                                    logger.info(f"Class rejection filter: Excluded {rejected_classes}, kept {len(bboxes)} detections")
                        except Exception as e:
                            logger.warning(f"Error applying class rejection filter: {e}")

                    # Use the source image timestamp so downstream nodes can
                    # correlate this result with the exact frame that was processed.
                    _result_ts = _image_source_timestamp if _image_source_timestamp is not None else time.time()

                    if len(bboxes) > 0:
                        result['bboxes'] = bboxes.tolist()
                        result['scores'] = scores.tolist()
                        result['class_ids'] = class_ids.tolist()
                        result['class_names'] = class_name_dict
                        result['score_th'] = score_th
                        result['timestamp'] = _result_ts
                        result['raw_frame'] = frame
                        logger.debug(f"JSON output: {len(bboxes)} detections, class_ids={class_ids.tolist()}")
                    else:
                        result['bboxes'] = []
                        result['scores'] = []
                        result['class_ids'] = []
                        result['class_names'] = class_name_dict
                        result['score_th'] = score_th
                        result['timestamp'] = _result_ts
                        result['raw_frame'] = frame
                        logger.debug(f"JSON output: 0 detections (all filtered out or no detections)")

                else:
                    # No frame available yet: still broadcast all supported class names
                    # so downstream nodes (e.g. Chart) can populate their dropdowns.
                    result['bboxes'] = []
                    result['scores'] = []
                    result['class_ids'] = []
                    result['class_names'] = class_name_dict
                    result['score_th'] = score_th

                if frame is not None and use_pref_counter:
                    elapsed_time = time.monotonic() - start_time
                    elapsed_time = int(elapsed_time * 1000)
                    dpg_set_value(self.tag_node_output_result,
                                  str(elapsed_time).zfill(4) + 'ms')

                # Get the draw bounding boxes checkbox state
                draw_bbox = dpg_get_value(self.tag_node_draw_bbox_value_name)
                if draw_bbox is None:
                    draw_bbox = self.DEFAULT_DRAW_BBOX

                # Get bbox thickness from slider
                bbox_thickness = dpg_get_value(self.tag_node_bbox_thickness_value_name)
                if bbox_thickness is None:
                    bbox_thickness = self.DEFAULT_BBOX_THICKNESS

                # Separate displayed image from output image
                # Display image: ALWAYS show bounding boxes for visualization
                # Output image: Respect checkbox setting (for video saving vs tracking)
                display_frame = None
                output_frame = None
                
                if frame is not None:
                    # Display image: ALWAYS draw bounding boxes (for user feedback)
                    display_frame = copy.deepcopy(frame)
                    display_frame = self.draw_object_detection_info(
                        display_frame,
                        score_th,
                        bboxes,
                        scores,
                        class_ids,
                        class_name_dict,
                        provider_label=provider,
                        thickness=bbox_thickness,
                    )
                    
                    # Output image: Respect checkbox setting
                    if draw_bbox:
                        # When checked: send frame WITH bounding boxes (for video recording)
                        output_frame = copy.deepcopy(frame)
                        output_frame = self.draw_object_detection_info(
                            output_frame,
                            score_th,
                            bboxes,
                            scores,
                            class_ids,
                            class_name_dict,
                            provider_label=provider,
                            thickness=bbox_thickness,
                        )
                    else:
                        # When unchecked: send clean frame (for tracking)
                        output_frame = frame
                    
                    # Update UI texture with display frame (always has bboxes)
                    texture = self.convert_cv_to_dpg(
                        display_frame,
                        small_window_w,
                        small_window_h,
                    )
                    dpg_set_value(tag_node_output_image, texture)

                data["image"] = output_frame if output_frame is not None else frame
                data["json"] = result
                data["audio"] = None
                return data
            except Exception as e:
                    logger.error(f"Error in object detection: {e}", exc_info=True)
                    return {"image": frame, "json": {}, "audio": None}


    def close(self, node_id):
        super().close(node_id)

    def get_setting_dict(self, node_id):
        self.tag_node_name = str(node_id) + ':' + self.node_tag
        input_value02_tag = self.tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'
        input_value03_tag = self.tag_node_name + ':' + self.TYPE_FLOAT + ':Input03Value'
        rejected_classes_tag = self.tag_node_name + ':RejectedClassesValue'
        draw_bbox_tag = self.tag_node_name + ':DrawBBoxValue'
        bbox_thickness_tag = self.tag_node_name + ':BBoxThicknessValue'
        batch_size_tag = self.tag_node_name + ':BatchSizeValue'


        model_name = dpg_get_value(input_value02_tag)

        score_th = round(float(dpg_get_value(input_value03_tag)), 3)
        
        rejected_classes = dpg_get_value(rejected_classes_tag) if dpg_get_value(rejected_classes_tag) else ""
        
        draw_bbox = dpg_get_value(draw_bbox_tag)
        if draw_bbox is None:
            draw_bbox = self.DEFAULT_DRAW_BBOX

        bbox_thickness = dpg_get_value(bbox_thickness_tag)
        if bbox_thickness is None:
            bbox_thickness = self.DEFAULT_BBOX_THICKNESS

        batch_size = dpg_get_value(batch_size_tag)
        if batch_size is None:
            batch_size = self.DEFAULT_BATCH_SIZE

        pos = dpg.get_item_pos(self.tag_node_name)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos
        setting_dict[input_value02_tag] = model_name
        setting_dict[input_value03_tag] = score_th
        setting_dict[rejected_classes_tag] = rejected_classes
        setting_dict[draw_bbox_tag] = draw_bbox
        setting_dict[bbox_thickness_tag] = bbox_thickness
        setting_dict[batch_size_tag] = int(batch_size)

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        self.tag_node_name = str(node_id) + ':' + self.node_tag
        input_value02_tag = self.tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'
        input_value03_tag = self.tag_node_name + ':' + self.TYPE_FLOAT + ':Input03Value'
        rejected_classes_tag = self.tag_node_name + ':RejectedClassesValue'
        draw_bbox_tag = self.tag_node_name + ':DrawBBoxValue'
        bbox_thickness_tag = self.tag_node_name + ':BBoxThicknessValue'
        batch_size_tag = self.tag_node_name + ':BatchSizeValue'

        model_name = setting_dict[input_value02_tag]
        score_th = setting_dict[input_value03_tag]
        rejected_classes = setting_dict.get(rejected_classes_tag, "")
        draw_bbox = setting_dict.get(draw_bbox_tag, self.DEFAULT_DRAW_BBOX)
        bbox_thickness = setting_dict.get(bbox_thickness_tag, self.DEFAULT_BBOX_THICKNESS)
        batch_size = int(setting_dict.get(batch_size_tag, self.DEFAULT_BATCH_SIZE))

        # If model_name is a custom model saved in registry but not yet in memory, reload it
        if model_name and model_name not in self._model_class:
            entry = custom_models_registry.get_entry(model_name)
            if entry:
                raw_classes = entry.get('class_names', {})
                class_names_restored = {int(k): str(v) for k, v in raw_classes.items()}
                if not class_names_restored:
                    num_classes = int(entry.get('num_classes', 0))
                    if num_classes > 0:
                        class_names_restored = {i: f"class_{i}" for i in range(num_classes)}
                    else:
                        class_names_restored = dict(coco_class_names)
                Node._register_custom_model(
                    model_name,
                    entry['path'],
                    class_names_restored,
                    entry.get('output_format', 'yolo11'),
                    int(entry.get('input_width', 640)),
                    int(entry.get('input_height', 640)),
                    supports_batch=bool(entry.get('supports_batched_detection', False)),
                )
                logger.info(f"Restored custom model from registry on set_setting_dict: {model_name}")
            else:
                logger.warning(f"Saved model '{model_name}' not found in registry; using default.")
                model_name = list(self._model_class.keys())[0]

        # Update model dropdown to include the model name if missing
        try:
            current_items = dpg.get_item_configuration(input_value02_tag).get("items", [])
            if model_name and model_name not in current_items:
                dpg.configure_item(input_value02_tag, items=list(current_items) + [model_name])
        except Exception:
            pass

        dpg_set_value(self.tag_node_input_text_value_name, model_name)
        self._update_batch_badge(model_name)
        dpg_set_value(self.tag_node_input_float_value_name, score_th)
        
        # Update the dropdown items to match the loaded model's classes
        if model_name in self._model_class_name_list:
            class_names = self._model_class_name_list[model_name]
            class_items = get_class_rejection_dropdown_items(class_names)
            try:
                dpg.configure_item(rejected_classes_tag, items=class_items)
            except:
                pass  # Ignore if the UI element doesn't exist yet
        
        # Set rejected classes if the tag exists in settings
        if rejected_classes_tag in setting_dict:
            try:
                dpg_set_value(rejected_classes_tag, rejected_classes)
            except:
                pass  # Ignore if the UI element doesn't exist yet
        
        # Set draw bounding boxes checkbox
        try:
            dpg_set_value(draw_bbox_tag, draw_bbox)
        except:
            pass  # Ignore if the UI element doesn't exist yet

        # Set bbox thickness slider
        try:
            dpg_set_value(bbox_thickness_tag, bbox_thickness)
        except Exception:
            pass  # Ignore if the UI element doesn't exist yet

        # Set TensorRT batch-size slider
        try:
            dpg_set_value(batch_size_tag, max(1, min(self.MAX_BATCH_SIZE, batch_size)))
        except Exception:
            pass  # Ignore if the UI element doesn't exist yet




    def draw_object_detection_info(
            self,
            image,
            score_th,
            bboxes,
            scores,
            class_ids,
            class_names,
            provider_label=None,
            thickness=3,
        ):
            debug_image = copy.deepcopy(image)
            logger.debug(f"Drawing object detection info on image with shape: {debug_image.shape}")
            
            # Calculate adaptive font scale and thickness based on image size
            # Use the smaller dimension to ensure text is always readable
            image_height, image_width = debug_image.shape[:2]
            min_dimension = min(image_height, image_width)
            
            # Scale font size: base size of 0.9 for ~640px, scale proportionally
            font_scale = max(0.3, min(2.0, (min_dimension / 640.0) * 0.9))
            
            # Scale thickness: base thickness of 3 for ~640px, scale proportionally
            adaptive_thickness = max(1, int((min_dimension / 640.0) * thickness))

            if provider_label:
                provider_raw = str(provider_label).upper()
                if "PTCUDA" in provider_raw or provider_raw == "PTCUDA":
                    provider_text = "cuda"
                elif "TRTCUDA" in provider_raw or "TENSORRT" in provider_raw:
                    provider_text = "TRTcuda"
                elif "CPU" in provider_raw:
                    provider_text = "cpu"
                else:
                    provider_text = provider_raw
                (_, provider_text_height), provider_baseline = cv2.getTextSize(
                    provider_text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    adaptive_thickness,
                )
                debug_image = cv2.putText(
                    debug_image,
                    provider_text,
                    (10, provider_text_height + provider_baseline + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 255, 255),
                    thickness=adaptive_thickness,
                )
            
            for bbox, score, class_id in zip(bboxes, scores, class_ids):
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

                if score_th > score:
                    continue

                color = self.get_color(class_id)

                debug_image = cv2.rectangle(
                    debug_image,
                    (x1, y1),
                    (x2, y2),
                    color,
                    thickness=adaptive_thickness,
                )


                score = '%.2f' % score
                text = '%s:%s(%s)' % (int(class_id), str(
                    class_names.get(int(class_id), str(int(class_id)))), score)
                
                # Calculate text size to position it better
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, adaptive_thickness
                )
                
                # Position text above the bounding box with some padding
                text_y = max(y1 - 5, text_height + 5)
                
                debug_image = cv2.putText(
                    debug_image,
                    text,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    thickness=adaptive_thickness,
                )

            return debug_image



    def get_color(self, index):
        temp_index = abs(int(index + 35)) * 3
        color = (
            (29 * temp_index) % 255,
            (17 * temp_index) % 255,
            (37 * temp_index) % 255,
        )
        return color

# Seed the registry with built-in models (skips entries already present)
Node._ensure_builtin_models()
# Load all models (built-in + user-uploaded) into runtime dictionaries
Node._load_custom_models_from_registry()
