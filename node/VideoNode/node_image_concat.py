#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import copy
import logging

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeABC
#from node_editor.util import self.convert_cv_to_dpg
#from node.draw_node.draw_util.draw_util import draw_info
from node.basenode import Node

logger = logging.getLogger(__name__)

def create_concat_image(frame_dict, slot_num):
    if slot_num == 1:
        frame = frame_dict[0]
        display_frame = copy.deepcopy(frame)
    

    elif slot_num == 2:
        frame = cv2.hconcat([frame_dict[0], frame_dict[1]])

        bg_image = np.zeros(
            (frame.shape[0] * 2, frame.shape[1], 3)).astype(np.uint8)
        bg_image[int(frame.shape[0] / 2):int(frame.shape[0] / 2) +
                 frame.shape[0], 0:frame.shape[1]] = frame

        display_frame = copy.deepcopy(bg_image)
   

    elif slot_num == 3 or slot_num == 4:
        hconcat_image01 = cv2.hconcat([frame_dict[0], frame_dict[1]])
        hconcat_image02 = cv2.hconcat([frame_dict[2], frame_dict[3]])
        frame = cv2.vconcat([hconcat_image01, hconcat_image02])
        display_frame = copy.deepcopy(frame)
    

    elif slot_num == 5 or slot_num == 6:
        hconcat_image01 = cv2.hconcat([frame_dict[0], frame_dict[1]])
        hconcat_image01 = cv2.hconcat([hconcat_image01, frame_dict[2]])
        hconcat_image02 = cv2.hconcat([frame_dict[3], frame_dict[4]])
        hconcat_image02 = cv2.hconcat([hconcat_image02, frame_dict[5]])
        frame = cv2.vconcat([hconcat_image01, hconcat_image02])
        display_frame = copy.deepcopy(frame)
    

    elif slot_num == 7 or slot_num == 8 or slot_num == 9:
        hconcat_image01 = cv2.hconcat([frame_dict[0], frame_dict[1]])
        hconcat_image01 = cv2.hconcat([hconcat_image01, frame_dict[2]])
        hconcat_image02 = cv2.hconcat([frame_dict[3], frame_dict[4]])
        hconcat_image02 = cv2.hconcat([hconcat_image02, frame_dict[5]])
        hconcat_image03 = cv2.hconcat([frame_dict[6], frame_dict[7]])
        hconcat_image03 = cv2.hconcat([hconcat_image03, frame_dict[8]])
        vconcat_image = cv2.vconcat([hconcat_image01, hconcat_image02])
        frame = cv2.vconcat([vconcat_image, hconcat_image03])
        display_frame = copy.deepcopy(frame)

    return frame, display_frame

class FactoryNode:
    node_label = 'VideoConcat'
    node_tag = 'VideoConcat'
    

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
        node._value_history = {}

        node.tag_node_name = str(node_id) + ':' + node.node_tag
        node.tag_node_input00_name = node.tag_node_name + ':' + node.TYPE_IMAGE + ':Input00'
        node.tag_node_input01_name = node.tag_node_name + ':' + node.TYPE_IMAGE + ':Input01'
        node.tag_node_input01_value_name = node.tag_node_name + ':' + node.TYPE_IMAGE + ':Input01Value'
        node.tag_node_output01_name = node.tag_node_name + ':' + node.TYPE_IMAGE + ':Output01'
        node.tag_node_output01_value_name = node.tag_node_name + ':' + node.TYPE_IMAGE + ':Output01Value'


        node._opencv_setting_dict = opencv_setting_dict
        small_window_w = node._opencv_setting_dict['process_width']
        small_window_h = node._opencv_setting_dict['process_height']


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
                tag=node.tag_node_output01_value_name,
                format=dpg.mvFormat_Float_rgb,
            )


        if node.tag_node_name not in node._slot_id:
            node._slot_id[node.tag_node_name] = 1
            node._slot_types[node.tag_node_name] = {1: node.TYPE_IMAGE}  # First slot is IMAGE by default


        with dpg.node(
                tag=node.tag_node_name,
                parent=parent,
                label=node.node_label,
                pos=pos,
        ):

            with dpg.node_attribute(
                    tag=node.tag_node_output01_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_image(node.tag_node_output01_value_name)

            with dpg.node_attribute(
                    tag=node.tag_node_input00_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_combo(
                    tag=node.tag_node_name + ':SlotType',
                    items=['IMAGE', 'AUDIO', 'JSON'],
                    default_value='IMAGE',
                    width=int(small_window_w / 3),
                    label='Slot Type',
                )
                dpg.add_button(
                    label='Add Slot',
                    width=int(small_window_w / 3),
                    callback=node._add_slot,
                    user_data=node.tag_node_name,
                )
            # スロット
            with dpg.node_attribute(
                    tag=node.tag_node_input01_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_text(
                    tag=node.tag_node_input01_value_name,
                    default_value='Input BGR image',
                )

        return node



class Node(Node):
    _ver = '0.0.2'

    node_label = 'VideoConcat'
    node_tag = 'VideoConcat'

    _opencv_setting_dict = None

    _max_slot_number = 9
    _slot_id = {}
    _slot_types = {}  # Track the type of each slot (IMAGE, AUDIO, JSON)
    
    # Reference height for classification text scaling (in pixels)
    _REFERENCE_HEIGHT = 480.0
    
    # Reference dimension for object detection text scaling (in pixels)
    # Object detection uses min dimension for aspect-ratio independence
    _OD_REFERENCE_DIMENSION = 640.0
    _OD_BASE_FONT_SCALE = 0.9
    _OD_MIN_FONT_SCALE = 0.3
    _OD_MAX_FONT_SCALE = 2.0
    _OD_BASE_THICKNESS = 3

    def __init__(self):
        pass

    def draw_classification_info(
        self,
        image,
        class_ids,
        class_scores,
        class_names,
        target_height=None,
    ):
        """
        Override base class method to display classification results
        bigger and at the bottom left of the image.
        
        Args:
            image: Input image to draw on
            class_ids: List of class IDs
            class_scores: List of class scores
            class_names: List of class names
            target_height: Target height for text scaling (used when image will be resized).
                          If None, uses the current image height.
        """
        debug_image = copy.deepcopy(image)
        height, width = debug_image.shape[:2]
        
        # Define colors for top 5 positions (BGR format) - matching node_classification.py
        rank_colors = [
            (0, 0, 255),      # Position 1 (index 0): Red (highest score)
            (0, 255, 255),    # Position 2 (index 1): Yellow
            (255, 0, 0),      # Position 3 (index 2): Blue
            (255, 0, 128),    # Position 4 (index 3): Violet
            (255, 0, 255),    # Position 5 (index 4): Magenta
        ]
        
        # Scale text parameters based on target height (or current height if not specified)
        # Reference height is 480px - text parameters are optimized for this size
        # Use target_height for scaling when provided (for concat node resizing)
        scaling_height = target_height if target_height is not None else height
        scale_factor = scaling_height / self._REFERENCE_HEIGHT
        font_scale = 1.0 * scale_factor  # Base 1.0, scaled by frame height
        thickness = max(1, int(3 * scale_factor))  # Base 3, scaled and min 1
        line_spacing = max(1, int(35 * scale_factor))  # Base 35, scaled and min 1
        
        # Calculate starting position from bottom
        num_lines = len(class_ids)
        start_y = height - 15 - (num_lines - 1) * line_spacing
        
        for index, (class_score, class_id) in enumerate(zip(class_scores, class_ids)):
            score = "%.2f" % class_score
            text = "%s:%s(%s)" % (str(class_id), str(class_names[int(class_id)]), score)
            
            # Select color based on position
            if index < len(rank_colors):
                color = rank_colors[index]
            else:
                color = (0, 255, 0)  # Default green for lower rankings
            
            # Position at bottom left
            y_position = start_y + (index * line_spacing)
            
            debug_image = cv2.putText(
                debug_image,
                text,
                (15, y_position),  # Bottom left position (x=15 for margin)
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness=thickness,
            )

        return debug_image

    def draw_object_detection_info(
        self,
        image,
        score_th,
        bboxes,
        scores,
        class_ids,
        class_names,
        thickness=_OD_BASE_THICKNESS,
        target_height=None,
        target_width=None,
    ):
        """
        Override base class method to support target_height and target_width parameters
        for proper text scaling when images are resized in concat node.
        
        Args:
            image: Input image to draw on
            score_th: Score threshold for filtering detections
            bboxes: Bounding boxes
            scores: Detection scores
            class_ids: Class IDs
            class_names: Class names dictionary
            thickness: Base thickness for drawing (default: 3, from _OD_BASE_THICKNESS constant)
            target_height: Target height for text scaling (used when image will be resized).
                          If None, uses the current image height.
            target_width: Target width for text scaling (used when image will be resized).
                         If None, uses the current image width.
        """
        debug_image = copy.deepcopy(image)
        image_height, image_width = debug_image.shape[:2]
        
        # Calculate scaling dimensions based on target size (if provided)
        # When both target dimensions are provided, use them directly for accurate scaling
        if target_height is not None and target_width is not None:
            scaling_height = target_height
            scaling_width = target_width
        elif target_height is not None:
            # Only target_height provided, estimate width based on aspect ratio
            aspect_ratio = image_width / image_height
            scaling_height = target_height
            scaling_width = int(target_height * aspect_ratio)
        elif target_width is not None:
            # Only target_width provided, estimate height based on aspect ratio
            aspect_ratio = image_height / image_width
            scaling_height = int(target_width * aspect_ratio)
            scaling_width = target_width
        else:
            # No target dimensions, use current image size
            scaling_height = image_height
            scaling_width = image_width
        
        min_dimension = min(scaling_height, scaling_width)
        
        # Scale font size: base size for reference dimension, scale proportionally
        font_scale = max(
            self._OD_MIN_FONT_SCALE,
            min(self._OD_MAX_FONT_SCALE, (min_dimension / self._OD_REFERENCE_DIMENSION) * self._OD_BASE_FONT_SCALE)
        )
        
        # Scale thickness: base thickness for reference dimension, scale proportionally
        adaptive_thickness = max(1, int((min_dimension / self._OD_REFERENCE_DIMENSION) * thickness))
        
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

            score_str = '%.2f' % score
            text = '%s:%s(%s)' % (int(class_id), str(class_names[int(class_id)]), score_str)
            
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

    def create_image_dict(
            self,
            slot_num,
            connection_info_src_dict,
            node_image_dict,
            node_result_dict,
            image_node_name,
            resize_width,
            resize_height,
            draw_info_on_result,
            slot_types_dict=None,
        ):
            frame_exist_flag = False

            black_image = np.zeros((resize_height, resize_width, 3)).astype(np.uint8)

            # Build a list of IMAGE slot indices (0-indexed)
            image_slot_indices = []
            for index in range(slot_num):
                # Check if this slot is an IMAGE type slot
                is_image_slot = True
                if slot_types_dict is not None:
                    # slot_number is 1-indexed in slot_types_dict
                    slot_type = slot_types_dict.get(index + 1, self.TYPE_IMAGE)
                    is_image_slot = (slot_type == self.TYPE_IMAGE)
                
                if is_image_slot:
                    image_slot_indices.append(index)
            
            # Count IMAGE type slots for display grid
            image_slot_count = len(image_slot_indices)
            
            # Determine grid size based on IMAGE slot count
            # Grid layout mapping: 1→1x1, 2→1x2(centered), 3-4→2x2, 5-6→2x3, 7-9→3x3
            # This list maps slot count to display grid size
            display_num_list = [1, 2, 4, 4, 6, 6, 9, 9, 9]
            if 0 < image_slot_count <= len(display_num_list):
                grid_size = display_num_list[image_slot_count - 1]
            elif image_slot_count > len(display_num_list):
                # For more than 9 slots, use the maximum 9-slot grid
                grid_size = display_num_list[-1]
            else:
                # image_slot_count is 0 or negative
                grid_size = 0
            
            # Only process the first grid_size IMAGE slots
            slots_to_process = image_slot_indices[:grid_size] if grid_size > 0 else []
            
            # Build frame_dict based on IMAGE slots to display
            # Process in reverse order to maintain compatibility with original grid layout
            # where slots are positioned from the newest (last added) to oldest (first added)
            frame_dict = {}
            for output_index, input_index in enumerate(reversed(slots_to_process)):
                node_id_name = connection_info_src_dict.get(input_index, None)
                # No defensive copy: draw_info() and cv2.resize() both return
                # new arrays and never modify the source frame.
                frame = node_image_dict.get(node_id_name, None)
                if frame is not None:
                    if draw_info_on_result:
                        node_result = node_result_dict[node_id_name]
                        image_node_name = node_id_name.split(':')[1]
                        frame = self.draw_info(
                            image_node_name, node_result, frame,
                            target_height=resize_height, target_width=resize_width
                        )
                    resize_frame = cv2.resize(frame, (resize_width, resize_height))
                    # cv2.resize already returns a new array: no extra copy needed
                    frame_dict[output_index] = resize_frame

                    frame_exist_flag = True
                else:
                    # Add black frame for IMAGE slots with no data
                    frame_dict[output_index] = copy.deepcopy(black_image)
            
            # Fill remaining grid positions with black frames
            for index in range(grid_size):
                if frame_dict.get(index, None) is None:
                    frame_dict[index] = copy.deepcopy(black_image)

            if not frame_exist_flag:
                frame_dict = None

            return frame_dict



    def update(
        self,
        node_id,
        connection_list,
        node_image_dict,
        node_result_dict,
        node_audio_dict,
    ):
        self.tag_node_name = str(node_id) + ':' + self.node_tag
        self.output_value01_tag = self.tag_node_name + ':' + self.TYPE_IMAGE + ':Output01Value'

        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        resize_width = self._opencv_setting_dict['result_width']
        resize_height = self._opencv_setting_dict['result_height']
        draw_info_on_result = self._opencv_setting_dict['draw_info_on_result']


        node_name_dict = {}
        connection_info_src = ''
        connection_info_src_dict = {}
        slot_data_dict = {}  # Store all slot data (images, audio, json)
        
        for connection_info in connection_list:

            slot_number = re.sub(r'\D', '', connection_info[1].split(':')[-1])
            if slot_number == '':
                continue
            slot_number = int(slot_number) - 1

            connection_type = connection_info[0].split(':')[2]
            # Support IMAGE, AUDIO, and JSON types
            if connection_type in [self.TYPE_IMAGE, self.TYPE_AUDIO, self.TYPE_JSON]:

                connection_info_src = connection_info[0]
                connection_info_src = connection_info_src.split(':')[:2]
                node_name = connection_info_src[1]
                connection_info_src = ':'.join(connection_info_src)

                node_name_dict[slot_number] = node_name
                
                # Store connection info with type
                if slot_number not in slot_data_dict:
                    slot_data_dict[slot_number] = {}
                slot_data_dict[slot_number]['type'] = connection_type
                slot_data_dict[slot_number]['source'] = connection_info_src
                
                # Only add to connection_info_src_dict if it's an IMAGE (for backward compatibility)
                if connection_type == self.TYPE_IMAGE:
                    connection_info_src_dict[slot_number] = connection_info_src

        slot_num = self._slot_id[self.tag_node_name]


        frame_dict = {}
        if len(connection_info_src_dict) > 0:
            # Get slot types for this node
            slot_types_dict = self._slot_types.get(self.tag_node_name, {})
            frame_dict = self.create_image_dict(
                slot_num,
                connection_info_src_dict,
                node_image_dict,
                node_result_dict,
                node_name,
                resize_width,
                resize_height,
                draw_info_on_result,
                slot_types_dict,
            )


        frame = None
        display_frame = None
        audio_data = None
        json_data = None
        
        if len(connection_info_src_dict) > 0 and frame_dict is not None:
            # Calculate number of IMAGE slots for concat
            slot_types_dict = self._slot_types.get(self.tag_node_name, {})
            # Check if slot_types_dict has content (empty dict is falsy in Python)
            if slot_types_dict:
                image_slot_count = sum(1 for slot_type in slot_types_dict.values() if slot_type == self.TYPE_IMAGE)
            else:
                # Fallback to total slot count if no type info available
                image_slot_count = slot_num
            frame, display_frame = create_concat_image(frame_dict, image_slot_count)

        # Collect audio and JSON data from slots
        audio_chunks = {}
        json_chunks = {}
        
        for slot_idx, slot_info in slot_data_dict.items():
            if slot_info['type'] == self.TYPE_AUDIO:
                audio_chunk = node_audio_dict.get(slot_info['source'], None)
                if audio_chunk is not None:
                    audio_chunks[slot_idx] = audio_chunk
            elif slot_info['type'] == self.TYPE_JSON:
                # Get JSON from node_result_dict
                json_chunk = node_result_dict.get(slot_info['source'], None)
                if json_chunk is not None:
                    json_chunks[slot_idx] = json_chunk
        
        # Prepare output data
        if len(audio_chunks) > 0:
            audio_data = audio_chunks
        if len(json_chunks) > 0:
            json_data = json_chunks

        # Forward _frame_packet from the primary IMAGE input so that
        # downstream nodes (e.g. VideoWriter) can reconstruct the correct
        # FramePacket pts_ms.  Without this, VideoWriter falls back to a
        # frame-counter-based pts_ms = 0, which causes the leading-audio trim
        # to be computed incorrectly and produces A/V desync proportional to
        # the recording-start offset within a step_duration window (up to
        # ~step_duration seconds of early audio).
        primary_frame_packet = None
        for slot_idx in sorted(slot_data_dict.keys()):
            slot_info = slot_data_dict[slot_idx]
            if slot_info['type'] == self.TYPE_IMAGE:
                img_json = node_result_dict.get(slot_info['source'], None)
                if isinstance(img_json, dict) and '_frame_packet' in img_json:
                    primary_frame_packet = img_json['_frame_packet']
                    break

        if primary_frame_packet is not None:
            if json_data is None:
                json_data = {}
            # json_data may be a dict keyed by slot index; adding '_frame_packet'
            # under a string key keeps both the per-slot JSON data and the packet
            # metadata accessible without collision.
            json_data['_frame_packet'] = primary_frame_packet

        if audio_chunks or json_chunks:
            logger.debug(
                "ImageConcat[%s] prepared frame=%s audio_chunks=%d json_chunks=%d",
                self.tag_node_name,
                display_frame is not None,
                len(audio_chunks),
                len(json_chunks),
            )

        if display_frame is not None:
            texture = self.convert_cv_to_dpg(
                display_frame,
                small_window_w,
                small_window_h,
            )
            dpg_set_value(self.output_value01_tag, texture)


        return {"image": frame, "json": json_data, "audio": audio_data}

    def close(self, node_id):
        pass


    def get_setting_dict(self, node_id):
        tag_node_name = str(node_id) + ':' + self.node_tag

        pos = dpg.get_item_pos(tag_node_name)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos
        setting_dict['slot_id'] = self._slot_id[tag_node_name]
        setting_dict['slot_types'] = self._slot_types.get(tag_node_name, {})

        return setting_dict

    

    def set_setting_dict(self, node_id, setting_dict):
        tag_node_name = str(node_id) + ':' + self.node_tag

        slot_number = int(setting_dict['slot_id'])
        slot_types = setting_dict.get('slot_types', {})
        
        # Initialize slot types if not present
        if tag_node_name not in self._slot_types:
            self._slot_types[tag_node_name] = {}
        
        # Restore slot types from settings
        for slot_idx_str, slot_type in slot_types.items():
            slot_idx = int(slot_idx_str) if isinstance(slot_idx_str, str) else slot_idx_str
            self._slot_types[tag_node_name][slot_idx] = slot_type
        
        # Add slots with their types
        for slot_idx in range(2, slot_number + 1):
            # Set the combo to the correct type before adding
            slot_type = slot_types.get(slot_idx, self.TYPE_IMAGE)
            slot_type_tag = tag_node_name + ':SlotType'
            if dpg.does_item_exist(slot_type_tag):
                dpg_set_value(slot_type_tag, slot_type)
            self._add_slot(None, None, tag_node_name)

    

    def _add_slot(self, sender, data, user_data):
        tag_node_name = user_data

        if self._max_slot_number > self._slot_id[tag_node_name]:
            self._slot_id[tag_node_name] += 1
            slot_number = self._slot_id[tag_node_name]

            # Get selected slot type from combo
            slot_type_tag = tag_node_name + ':SlotType'
            slot_type = dpg_get_value(slot_type_tag)
            
            # Store slot type
            if tag_node_name not in self._slot_types:
                self._slot_types[tag_node_name] = {}
            self._slot_types[tag_node_name][slot_number] = slot_type

            # Find the correct position to insert (before previous slot)
            # Get the type of the previous slot to construct the correct tag
            prev_slot_number = slot_number - 1
            prev_slot_type = self._slot_types[tag_node_name].get(prev_slot_number, self.TYPE_IMAGE)
            before_tag = tag_node_name + ':' + prev_slot_type + ':Input'
            before_tag += str(prev_slot_number).zfill(2)

            # Create tag names for the new slot
            tag_node_inputXX_name = tag_node_name + ':' + slot_type + ':Input'
            tag_node_inputXX_name += str(slot_number).zfill(2)

            tag_node_inputXX_value_name = tag_node_name + ':' + slot_type + ':Input'
            tag_node_inputXX_value_name += str(slot_number).zfill(2) + 'Value'

            # Set appropriate label based on slot type
            if slot_type == self.TYPE_IMAGE:
                label_text = 'Input BGR image'
            elif slot_type == self.TYPE_AUDIO:
                label_text = 'Input Audio chunk'
            elif slot_type == self.TYPE_JSON:
                label_text = 'Input JSON data'
            else:
                label_text = 'Input data'

            with dpg.node_attribute(
                    tag=tag_node_inputXX_name,
                    attribute_type=dpg.mvNode_Attr_Input,
                    parent=tag_node_name,
                    before=before_tag,
            ):
                dpg.add_text(
                    tag=tag_node_inputXX_value_name,
                    default_value=label_text,
                )




    def draw_info(self, node_name, node_result, image, target_height=None, target_width=None):
        # need some abstraction here
        classification_nodes = ['Classification']
        object_detection_nodes = ['ObjectDetection']
        semantic_segmentation_nodes = ['SemanticSegmentation']
        pose_estimation_nodes = ['PoseEstimation']
        face_detection_nodes = ['FaceDetection']
        multi_object_tracking_nodes = ['MultiObjectTracking']
        qr_code_detection_nodes = ['QRCodeDetection']

        debug_image = copy.deepcopy(image)
        if node_name in classification_nodes:
            use_object_detection = node_result.get('use_object_detection', [])
            class_ids = node_result.get('class_ids', [])
            class_scores = node_result.get('class_scores', [])
            class_names = node_result.get('class_names', [])

            if use_object_detection:
                od_bboxes = node_result.get('od_bboxes', [])
                od_scores = node_result.get('od_scores', [])
                od_class_ids = node_result.get('od_class_ids', [])
                od_class_names = node_result.get('od_class_names', [])
                od_score_th = node_result.get('od_score_th', [])
                # Note: draw_classification_with_od_info uses fixed font size (0.9) from base class
                # and doesn't have the same scaling issue as draw_classification_info.
                # If dynamic scaling is needed in the future, override this method.
                debug_image = self.draw_classification_with_od_info(
                    debug_image,
                    class_ids,
                    class_scores,
                    class_names,
                    od_bboxes,
                    od_scores,
                    od_class_ids,
                    od_class_names,
                    od_score_th,
                    thickness=3,
                )
            else:
                debug_image = self.draw_classification_info(
                    debug_image,
                    class_ids,
                    class_scores,
                    class_names,
                    target_height=target_height,
                )
        elif node_name in object_detection_nodes:
            bboxes = node_result.get('bboxes', [])
            scores = node_result.get('scores', [])
            class_ids = node_result.get('class_ids', [])
            class_names = node_result.get('class_names', [])
            score_th = node_result.get('score_th', [])
            debug_image = self.draw_object_detection_info(
                debug_image,
                score_th,
                bboxes,
                scores,
                class_ids,
                class_names,
                target_height=target_height,
                target_width=target_width,
            )
        elif node_name in semantic_segmentation_nodes:
            class_num = node_result.get('class_num', [])
            segmentation_map = node_result.get('segmentation_map', [])
            score_th = node_result.get('score_th', [])
            debug_image = self.draw_semantic_segmentation_info(
                debug_image,
                score_th,
                class_num,
                segmentation_map,
            )
        elif node_name in pose_estimation_nodes:
            model_name = node_result.get('model_name', [])
            results_list = node_result.get('results_list', [])
            score_th = node_result.get('score_th', [])
            debug_image = self.draw_pose_estimation_info(
                model_name,
                debug_image,
                results_list,
                score_th,
            )
        elif node_name in face_detection_nodes:
            model_name = node_result.get('model_name', [])
            results_list = node_result.get('results_list', [])
            score_th = node_result.get('score_th', [])
            debug_image = self.draw_face_detection_info(
                model_name,
                debug_image,
                results_list,
                score_th,
            )
        elif node_name in multi_object_tracking_nodes:
            track_ids = node_result.get('track_ids', [])
            bboxes = node_result.get('bboxes', [])
            scores = node_result.get('scores', [])
            class_ids = node_result.get('class_ids', [])
            class_names = node_result.get('class_names', [])
            track_id_dict = node_result.get('track_id_dict', [])
            debug_image = self.draw_multi_object_tracking_info(
                debug_image,
                track_ids,
                bboxes,
                scores,
                class_ids,
                class_names,
                track_id_dict,
            )
        return debug_image
