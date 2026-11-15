#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import os

import numpy as np
import dearpygui.dearpygui as dpg

import traceback
import uuid
import cv2

from src.utils.logging import get_logger
from node_editor.util import _dpg_lock

logger = get_logger(__name__)


class DataType:
    TYPE_BOOLEAN = "BOOLEAN"
    TYPE_TEXT = "TEXT"
    TYPE_IMAGE = "IMAGE"
    TYPE_FLOAT = "FLOAT"
    TYPE_TIME_MS = "TIME_MS"
    TYPE_SOUND = "SOUND"


class PortType:
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"


class Node:
    _ver = "0.0.1"
    node_label = "BaseNode"
    node_tag = "BaseNode"
    node_data = None

    TYPE_BOOLEAN = "BOOLEAN"
    TYPE_TEXT = "TEXT"
    TYPE_IMAGE = "IMAGE"
    TYPE_FLOAT = "FLOAT"
    TYPE_INT = "INT"
    TYPE_TIME_MS = "TIME_MS"
    TYPE_AUDIO = "AUDIO"
    TYPE_JSON = "JSON"
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"

    def __init__(self, node_id=1, connection_dict=None, opencv_setting_dict=None):
        self.id = self.generate_id()
        self.node_label = self.__class__.node_label
        self.node_tag = self.__class__.node_tag
        self.tag_node_name = f"{node_id}:{self.node_tag}"
        # Générer les tags dynamiquement en fonction du dictionnaire
        # self.tags = self.generate_tags(connection_dict)

        # Paramètres OpenCV
        self._opencv_setting_dict = opencv_setting_dict if opencv_setting_dict else {}
        self.small_window_w = self._opencv_setting_dict.get("process_width", 640)
        self.small_window_h = self._opencv_setting_dict.get("process_height", 480)
        self._small_window_w = self._opencv_setting_dict.get("process_width", 640)
        self._small_window_h = self._opencv_setting_dict.get("process_height", 480)
        self.use_pref_counter = self._opencv_setting_dict.get("use_pref_counter", False)
        self.use_gpu = self._opencv_setting_dict.get("use_gpu", False)

    def generate_id(self):
        return str(uuid.uuid4())

    def generate_tags(self, connection_dict):
        tags = {}

        # Parcours du dictionnaire pour générer les tags
        for index, connection_info in connection_dict.items():
            connection_type = connection_info.get("CONNECTION")
            data_type = connection_info.get("TYPE")

            if connection_type and data_type:
                if connection_type == "INPUT":
                    tags[f"{self.tag_node_name}:{data_type}:Input{index}"] = None
                    tags[f"{self.tag_node_name}:{data_type}:Input{index}Value"] = None
                elif connection_type == "OUTPUT":
                    tags[f"{self.tag_node_name}:{data_type}:Output{index}"] = None
                    tags[f"{self.tag_node_name}:{data_type}:Output{index}Value"] = None
        return tags

    def update(self, node_id, connection_list, node_image_dict, node_result_dict):
        pass

    def close(self, node_id):
        pass

    def convert_cv_to_dpg(self, image, width, height):
        resize_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

        if (
            resize_image.dtype == np.uint8
            and resize_image.ndim == 3
            and resize_image.shape[2] == 3
        ):
            # Fast path (called for every displayed frame): contiguous
            # channel swap, then a single float32 buffer normalised in place.
            texture_data = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB).astype(
                np.float32
            )
            texture_data /= 255.0
            return texture_data.ravel()

        data = np.flip(resize_image, 2)
        data = data.ravel()
        data = np.asarray(data, dtype=np.float32)

        texture_data = np.true_divide(data, 255.0)

        return texture_data

    def get_input_frame(self, connection_list, node_image_dict, node_audio_dict=None):
        """
        Récupère une frame depuis une connexion IMAGE ou AUDIO.
        """
        connection_info_src = ""
        connection_type_found = None

        for connection_info in connection_list:
            connection_type = connection_info[0].split(":")[2]
            logger.debug("Connection type detected: %s", connection_type)

            # ✅ Accepter IMAGE et AUDIO
            if connection_type in [self.TYPE_IMAGE, self.TYPE_AUDIO]:
                connection_info_src = ":".join(connection_info[0].split(":")[:2])
                connection_type_found = connection_type
                logger.debug(
                    "Connection source: %s, type: %s",
                    connection_info_src, connection_type_found,
                )
                break

        if not connection_info_src:
            return None

        # ✅ Chercher dans le bon dictionnaire selon le type
        frame = None
        if connection_type_found == self.TYPE_IMAGE:
            frame = node_image_dict.get(connection_info_src, None)
        elif connection_type_found == self.TYPE_AUDIO:
            # ✅ Le spectrogramme est dans node_audio_dict !
            if node_audio_dict is not None:
                frame = node_audio_dict.get(connection_info_src, None)

        return frame

    def get_setting_dict(self, node_id):
        self.tag_node_name = f"{node_id}:{self.node_tag}"
        setting_dict = {}

        for tag, value in self.tags.items():
            with _dpg_lock:
                if dpg.does_item_exist(tag):
                    setting_dict[tag] = dpg.get_value(tag)
                else:
                    logger.warning("get_setting_dict: tag %s does not exist, skipping", tag)

        pos = [0, 0]
        with _dpg_lock:
            try:
                pos = dpg.get_item_pos(self.tag_node_name)
            except Exception as exc:
                logger.warning(
                    "get_setting_dict: could not get position for %s: %s",
                    self.tag_node_name, exc,
                )

        setting_dict["ver"] = self._ver
        setting_dict["pos"] = pos
        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        self.tag_node_name = f"{node_id}:{self.node_tag}"

        # "ver" and "pos" are metadata entries, not DPG item tags.
        _skip_keys = {"ver", "pos"}
        for tag, value in setting_dict.items():
            if tag in _skip_keys:
                continue
            with _dpg_lock:
                if dpg.does_item_exist(tag):
                    try:
                        dpg.set_value(tag, value)
                    except Exception as exc:
                        logger.warning(
                            "set_setting_dict: set_value failed for tag %s: %s",
                            tag, exc,
                        )
                else:
                    logger.debug("set_setting_dict: tag %s does not exist, skipping", tag)

    def draw_info(self, node_name, node_result, image):
        classification_nodes = ["Classification"]
        object_detection_nodes = ["ObjectDetection"]
        semantic_segmentation_nodes = ["SemanticSegmentation"]
        pose_estimation_nodes = ["PoseEstimation"]
        face_detection_nodes = ["FaceDetection"]
        multi_object_tracking_nodes = ["MultiObjectTracking"]
        qr_code_detection_nodes = ["QRCodeDetection"]

        debug_image = copy.deepcopy(image)
        if node_name in classification_nodes:
            use_object_detection = node_result.get("use_object_detection", [])
            class_ids = node_result.get("class_ids", [])
            class_scores = node_result.get("class_scores", [])
            class_names = node_result.get("class_names", [])

            if use_object_detection:
                od_bboxes = node_result.get("od_bboxes", [])
                od_scores = node_result.get("od_scores", [])
                od_class_ids = node_result.get("od_class_ids", [])
                od_class_names = node_result.get("od_class_names", [])
                od_score_th = node_result.get("od_score_th", [])
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
                )
        elif node_name in object_detection_nodes:
            bboxes = node_result.get("bboxes", [])
            scores = node_result.get("scores", [])
            class_ids = node_result.get("class_ids", [])
            class_names = node_result.get("class_names", [])
            score_th = node_result.get("score_th", [])
            debug_image = self.draw_object_detection_info(
                debug_image,
                score_th,
                bboxes,
                scores,
                class_ids,
                class_names,
            )
        elif node_name in semantic_segmentation_nodes:
            class_num = node_result.get("class_num", [])
            segmentation_map = node_result.get("segmentation_map", [])
            score_th = node_result.get("score_th", [])
            debug_image = self.draw_semantic_segmentation_info(
                debug_image,
                score_th,
                class_num,
                segmentation_map,
            )
        elif node_name in pose_estimation_nodes:
            model_name = node_result.get("model_name", [])
            results_list = node_result.get("results_list", [])
            score_th = node_result.get("score_th", [])
            debug_image = self.draw_pose_estimation_info(
                model_name,
                debug_image,
                results_list,
                score_th,
            )
        elif node_name in face_detection_nodes:
            model_name = node_result.get("model_name", [])
            results_list = node_result.get("results_list", [])
            score_th = node_result.get("score_th", [])
            debug_image = self.draw_face_detection_info(
                model_name,
                debug_image,
                results_list,
                score_th,
            )
        elif node_name in multi_object_tracking_nodes:
            track_ids = node_result.get("track_ids", [])
            bboxes = node_result.get("bboxes", [])
            scores = node_result.get("scores", [])
            class_ids = node_result.get("class_ids", [])
            class_names = node_result.get("class_names", [])
            track_id_dict = node_result.get("track_id_dict", [])
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

    def get_class_name(self, class_id, class_names):
        """
        Safely get class name from class_names, handling both dict and list formats.
        
        Args:
            class_id: The class ID (will be converted to int)
            class_names: Dictionary or list of class names
            
        Returns:
            Class name string, or fallback 'class_N' if not found
        """
        class_id_int = int(class_id)
        if isinstance(class_names, dict):
            return class_names.get(class_id_int, f"class_{class_id_int}")
        elif isinstance(class_names, list) and 0 <= class_id_int < len(class_names):
            return class_names[class_id_int]
        else:
            return f"class_{class_id_int}"

    def draw_classification_info(
        self,
        image,
        class_ids,
        class_scores,
        class_names,
    ):
        debug_image = copy.deepcopy(image)
        for index, (class_score, class_id) in enumerate(zip(class_scores, class_ids)):
            score = "%.2f" % class_score
            class_name = self.get_class_name(class_id, class_names)
            text = "%s:%s(%s)" % (str(class_id), class_name, score)
            debug_image = cv2.putText(
                debug_image,
                text,
                (15, 25 + (index * 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                thickness=2,
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

    def draw_object_detection_info(
        self,
        image,
        score_th,
        bboxes,
        scores,
        class_ids,
        class_names,
        thickness=3,
    ):
        debug_image = copy.deepcopy(image)
        print("external :", debug_image.shape)
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
                thickness=thickness,
            )

            score = "%.2f" % score
            class_name = self.get_class_name(class_id, class_names)
            text = "%s:%s(%s)" % (int(class_id), class_name, score)
            debug_image = cv2.putText(
                debug_image,
                text,
                (x1, y1 - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                thickness=thickness,
            )

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
        thickness=3,
    ):
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

            color = self.get_color(od_class_id)

            debug_image = cv2.rectangle(
                debug_image,
                (x1, y1),
                (x2, y2),
                color,
                thickness=thickness,
            )

            # Object Detection
            score_text = "%.2f" % od_score
            od_class_name = self.get_class_name(od_class_id, od_class_names)
            text = "%s:%s(%s)" % (
                int(od_class_id),
                od_class_name,
                score_text,
            )
            debug_image = cv2.putText(
                debug_image,
                "Detection(" + text + ")",
                (x1, y1 - 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                thickness=thickness,
            )

            # Classification
            score_text = "%.2f" % score
            text = "%s:%s(%s)" % (
                int(class_id),
                str(class_name_dict[int(class_id)]),
                score_text,
            )
            debug_image = cv2.putText(
                debug_image,
                "Classification(" + text + ")",
                (x1, y1 - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                thickness=thickness,
            )

        return debug_image

    def draw_semantic_segmentation_info(
        self,
        image,
        score_th,
        class_num,
        segmentation_map,
    ):
        debug_image = copy.deepcopy(image)

        segmentation_map = np.asarray(segmentation_map)

        # color map list — use at least as many entries as there are masks
        color_map = self.get_color_map_list(max(class_num, len(segmentation_map)))

        # Blend only the pixels selected by each mask instead of building
        # full-frame background/mask images for every class.
        for index, mask in enumerate(segmentation_map):
            selected = mask > score_th
            if not selected.any():
                continue

            pixels = debug_image[selected]
            color = np.empty_like(pixels)
            color[:] = color_map[index * 3 : index * 3 + 3]
            debug_image[selected] = cv2.addWeighted(pixels, 0.5, color, 0.5, 0)

        return debug_image

    def draw_yolov8_seg_contours(self, image, segmentation_map):
        """
        Draw only contours for YOLOv8-seg segmentation masks
        Args:
            image: Input image
            segmentation_map: Array of binary masks from YOLOv8-seg
        Returns:
            debug_image: Image with contours drawn
        """
        debug_image = copy.deepcopy(image)
        
        # Extract contours from each mask and draw them
        for idx, mask in enumerate(segmentation_map):
            # Convert mask to uint8 for contour detection
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(
                mask_uint8,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Generate a color for this object (cycling through colors)
            color = self.get_color(idx)
            
            # Draw all contours for this mask
            cv2.drawContours(debug_image, contours, -1, color, 2)
        
        return debug_image

    def draw_pose_estimation_info(self, model_name, image, results_list, score_th):
        debug_image = copy.deepcopy(image)

        move_net_nodes = [
            "MoveNet(SinglePose Lightning)",
            "MoveNet(SinglePose Thunder)",
            "MoveNet(MulitPose Lightning)",
        ]
        mediapipe_hands_nodes = [
            "MediaPipe Hands(Complexity0)",
            "MediaPipe Hands(Complexity1)",
        ]
        mediapipe_pose_nodes = [
            "MediaPipe Pose(Complexity0)",
            "MediaPipe Pose(Complexity1)",
            "MediaPipe Pose(Complexity2)",
        ]

        if model_name in move_net_nodes:
            debug_image = self.draw_movenet_info(debug_image, results_list, score_th)
        elif model_name in mediapipe_hands_nodes:
            debug_image = self.draw_mediapipe_hands_info(debug_image, results_list)
        elif model_name in mediapipe_pose_nodes:
            debug_image = self.draw_mediapipe_pose_info(
                debug_image,
                results_list,
                score_th,
            )

        else:
            debug_image = self.draw_keypoints_info(debug_image, results_list, score_th)

        return debug_image

    def draw_keypoints_info(self, image, rescaled_kps, score_th):
        points = rescaled_kps
        # for x, y in rescaled_kps:
        #        cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1)

        points = [tuple(map(int, pt)) for pt in points]
        # Contour du terrain (4 coins)
        cv2.line(image, points[0], points[1], (0, 255, 0), 2)
        cv2.line(image, points[1], points[3], (0, 255, 0), 2)
        cv2.line(image, points[3], points[2], (0, 255, 0), 2)
        cv2.line(image, points[2], points[0], (0, 255, 0), 2)

        # Zones de service (interne)
        cv2.line(image, points[4], points[6], (0, 255, 0), 2)  # ligne interne haut
        cv2.line(image, points[5], points[7], (0, 255, 0), 2)  # ligne interne bas
        cv2.line(image, points[4], points[5], (0, 255, 0), 2)  # côté gauche
        cv2.line(image, points[6], points[7], (0, 255, 0), 2)  # côté droit

        # Lignes de service horizontales
        cv2.line(image, points[8], points[9], (0, 255, 0), 1)
        cv2.line(image, points[10], points[11], (0, 255, 0), 1)

        # Filet au centre
        cv2.line(image, points[12], points[13], (0, 255, 0), 2)

        # Draw keypoint numbers on the image
        for i, point in enumerate(points):
            x, y = point
            # Draw a small circle at each keypoint
            cv2.circle(image, (x, y), 4, (0, 0, 255), -1)
            # Draw the keypoint number with a background for better visibility
            text = str(i)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            text_padding = 5
            text_offset = 2  # Additional offset for text within the rectangle
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            # Calculate rectangle position with boundary checks
            rect_x1 = max(0, x + text_padding)
            rect_y1 = max(0, y - text_size[1] - text_padding)
            rect_x2 = min(image.shape[1], x + text_padding + text_size[0] + text_offset * 2)
            rect_y2 = min(image.shape[0], y)
            # Only draw if rectangle has valid dimensions
            if rect_x2 > rect_x1 and rect_y2 > rect_y1:
                # Draw white background rectangle for text
                cv2.rectangle(image, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)
                # Calculate text position within the rectangle bounds
                text_x = min(rect_x1 + text_offset, rect_x2 - text_size[0])
                text_x = max(rect_x1, text_x)  # Ensure text stays within rectangle
                text_y = min(y - text_padding, rect_y2 - 2)
                text_y = max(rect_y1 + text_size[1], text_y)  # Ensure text baseline is within rectangle
                # Draw text in red color
                cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

        return image

    def draw_mediapipe_hands_info(self, image, results_list):
        for results in results_list:
            for id in range(21):
                landmark_x, landmark_y = results[id][0], results[id][1]
                cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), -1)

            cv2.line(image, results[2][:2], results[3][:2], (0, 255, 0), 2)
            cv2.line(image, results[3][:2], results[4][:2], (0, 255, 0), 2)

            cv2.line(image, results[5][:2], results[6][:2], (0, 255, 0), 2)
            cv2.line(image, results[6][:2], results[7][:2], (0, 255, 0), 2)
            cv2.line(image, results[7][:2], results[8][:2], (0, 255, 0), 2)

            cv2.line(image, results[9][:2], results[10][:2], (0, 255, 0), 2)
            cv2.line(image, results[10][:2], results[11][:2], (0, 255, 0), 2)
            cv2.line(image, results[11][:2], results[12][:2], (0, 255, 0), 2)

            cv2.line(image, results[13][:2], results[14][:2], (0, 255, 0), 2)
            cv2.line(image, results[14][:2], results[15][:2], (0, 255, 0), 2)
            cv2.line(image, results[15][:2], results[16][:2], (0, 255, 0), 2)

            cv2.line(image, results[17][:2], results[18][:2], (0, 255, 0), 2)
            cv2.line(image, results[18][:2], results[19][:2], (0, 255, 0), 2)
            cv2.line(image, results[19][:2], results[20][:2], (0, 255, 0), 2)

            cv2.line(image, results[0][:2], results[1][:2], (0, 255, 0), 2)
            cv2.line(image, results[1][:2], results[2][:2], (0, 255, 0), 2)
            cv2.line(image, results[2][:2], results[5][:2], (0, 255, 0), 2)
            cv2.line(image, results[5][:2], results[9][:2], (0, 255, 0), 2)
            cv2.line(image, results[9][:2], results[13][:2], (0, 255, 0), 2)
            cv2.line(image, results[13][:2], results[17][:2], (0, 255, 0), 2)
            cv2.line(image, results[17][:2], results[0][:2], (0, 255, 0), 2)

            cx, cy = results["palm_moment"]
            cv2.putText(
                image,
                results["label"],
                (cx - 20, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return image

    def draw_mediapipe_pose_info(self, image, results_list, score_th):
        for results in results_list:
            for id in range(33):
                landmark_x, landmark_y = results[id][0], results[id][1]
                visibility = results[id][3]

                if score_th > visibility:
                    continue
                cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), -1)

            if results[1][3] > score_th and results[2][3] > score_th:
                cv2.line(image, results[1][:2], results[2][:2], (0, 255, 0), 2)
            if results[2][3] > score_th and results[3][3] > score_th:
                cv2.line(image, results[2][:2], results[3][:2], (0, 255, 0), 2)

            if results[4][3] > score_th and results[5][3] > score_th:
                cv2.line(image, results[4][:2], results[5][:2], (0, 255, 0), 2)
            if results[5][3] > score_th and results[6][3] > score_th:
                cv2.line(image, results[5][:2], results[6][:2], (0, 255, 0), 2)

            if results[9][3] > score_th and results[10][3] > score_th:
                cv2.line(image, results[9][:2], results[10][:2], (0, 255, 0), 2)

            if results[11][3] > score_th and results[12][3] > score_th:
                cv2.line(image, results[11][:2], results[12][:2], (0, 255, 0), 2)

            if results[11][3] > score_th and results[13][3] > score_th:
                cv2.line(image, results[11][:2], results[13][:2], (0, 255, 0), 2)
            if results[13][3] > score_th and results[15][3] > score_th:
                cv2.line(image, results[13][:2], results[15][:2], (0, 255, 0), 2)

            if results[12][3] > score_th and results[14][3] > score_th:
                cv2.line(image, results[12][:2], results[14][:2], (0, 255, 0), 2)
            if results[14][3] > score_th and results[16][3] > score_th:
                cv2.line(image, results[14][:2], results[16][:2], (0, 255, 0), 2)

            if results[15][3] > score_th and results[17][3] > score_th:
                cv2.line(image, results[15][:2], results[17][:2], (0, 255, 0), 2)
            if results[17][3] > score_th and results[19][3] > score_th:
                cv2.line(image, results[17][:2], results[19][:2], (0, 255, 0), 2)
            if results[19][3] > score_th and results[21][3] > score_th:
                cv2.line(image, results[19][:2], results[21][:2], (0, 255, 0), 2)
            if results[21][3] > score_th and results[15][3] > score_th:
                cv2.line(image, results[21][:2], results[15][:2], (0, 255, 0), 2)

            if results[16][3] > score_th and results[18][3] > score_th:
                cv2.line(image, results[16][:2], results[18][:2], (0, 255, 0), 2)
            if results[18][3] > score_th and results[20][3] > score_th:
                cv2.line(image, results[18][:2], results[20][:2], (0, 255, 0), 2)
            if results[20][3] > score_th and results[22][3] > score_th:
                cv2.line(image, results[20][:2], results[22][:2], (0, 255, 0), 2)
            if results[22][3] > score_th and results[16][3] > score_th:
                cv2.line(image, results[22][:2], results[16][:2], (0, 255, 0), 2)

            if results[11][3] > score_th and results[23][3] > score_th:
                cv2.line(image, results[11][:2], results[23][:2], (0, 255, 0), 2)
            if results[12][3] > score_th and results[24][3] > score_th:
                cv2.line(image, results[12][:2], results[24][:2], (0, 255, 0), 2)
            if results[23][3] > score_th and results[24][3] > score_th:
                cv2.line(image, results[23][:2], results[24][:2], (0, 255, 0), 2)

            if results[23][3] > score_th and results[25][3] > score_th:
                cv2.line(image, results[23][:2], results[25][:2], (0, 255, 0), 2)
            if results[25][3] > score_th and results[27][3] > score_th:
                cv2.line(image, results[25][:2], results[27][:2], (0, 255, 0), 2)
            if results[27][3] > score_th and results[29][3] > score_th:
                cv2.line(image, results[27][:2], results[29][:2], (0, 255, 0), 2)
            if results[29][3] > score_th and results[31][3] > score_th:
                cv2.line(image, results[29][:2], results[31][:2], (0, 255, 0), 2)

            if results[24][3] > score_th and results[26][3] > score_th:
                cv2.line(image, results[24][:2], results[26][:2], (0, 255, 0), 2)
            if results[26][3] > score_th and results[28][3] > score_th:
                cv2.line(image, results[26][:2], results[28][:2], (0, 255, 0), 2)
            if results[28][3] > score_th and results[30][3] > score_th:
                cv2.line(image, results[28][:2], results[30][:2], (0, 255, 0), 2)
            if results[30][3] > score_th and results[32][3] > score_th:
                cv2.line(image, results[30][:2], results[32][:2], (0, 255, 0), 2)
        return image

    def draw_movenet_info(self, image, results_list, score_th):
        for results in results_list:
            for id in range(17):
                landmark_x, landmark_y = results[id][0], results[id][1]
                visibility = results[id][2]

                if score_th > visibility:
                    continue
                cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), -1)

            if results[0][2] > score_th and results[1][2] > score_th:
                cv2.line(image, results[0][:2], results[1][:2], (0, 255, 0), 2)

            if results[0][2] > score_th and results[2][2] > score_th:
                cv2.line(image, results[0][:2], results[2][:2], (0, 255, 0), 2)

            if results[1][2] > score_th and results[3][2] > score_th:
                cv2.line(image, results[1][:2], results[3][:2], (0, 255, 0), 2)

            if results[2][2] > score_th and results[4][2] > score_th:
                cv2.line(image, results[2][:2], results[4][:2], (0, 255, 0), 2)

            if results[5][2] > score_th and results[6][2] > score_th:
                cv2.line(image, results[5][:2], results[6][:2], (0, 255, 0), 2)

            if results[5][2] > score_th and results[7][2] > score_th:
                cv2.line(image, results[5][:2], results[7][:2], (0, 255, 0), 2)

            if results[7][2] > score_th and results[9][2] > score_th:
                cv2.line(image, results[7][:2], results[9][:2], (0, 255, 0), 2)

            if results[6][2] > score_th and results[8][2] > score_th:
                cv2.line(image, results[6][:2], results[8][:2], (0, 255, 0), 2)

            if results[8][2] > score_th and results[10][2] > score_th:
                cv2.line(image, results[8][:2], results[10][:2], (0, 255, 0), 2)

            if results[11][2] > score_th and results[12][2] > score_th:
                cv2.line(image, results[11][:2], results[12][:2], (0, 255, 0), 2)

            if results[5][2] > score_th and results[11][2] > score_th:
                cv2.line(image, results[5][:2], results[11][:2], (0, 255, 0), 2)

            if results[11][2] > score_th and results[13][2] > score_th:
                cv2.line(image, results[11][:2], results[13][:2], (0, 255, 0), 2)

            if results[13][2] > score_th and results[15][2] > score_th:
                cv2.line(image, results[13][:2], results[15][:2], (0, 255, 0), 2)

            if results[6][2] > score_th and results[12][2] > score_th:
                cv2.line(image, results[6][:2], results[12][:2], (0, 255, 0), 2)

            if results[12][2] > score_th and results[14][2] > score_th:
                cv2.line(image, results[12][:2], results[14][:2], (0, 255, 0), 2)

            if results[14][2] > score_th and results[16][2] > score_th:
                cv2.line(image, results[14][:2], results[16][:2], (0, 255, 0), 2)

            bbox = results.get("bbox", None)
            if bbox is not None:
                if bbox[4] > score_th:
                    image = cv2.rectangle(
                        image,
                        (bbox[0], bbox[1]),
                        (bbox[2], bbox[3]),
                        (0, 255, 0),
                        thickness=2,
                    )

        return image

    def draw_face_detection_info(self, model_name, image, results_list, score_th):
        debug_image = copy.deepcopy(image)

        if (
            model_name == "MediaPipe FaceDetection(~2m)"
            or model_name == "MediaPipe FaceDetection(~5m)"
        ):
            debug_image = self.draw_mediapipe_face_detection_info(
                debug_image,
                results_list,
                score_th,
            )
        elif (
            model_name == "MediaPipe FaceMesh"
            or model_name == "MediaPipe FaceMesh(Refine Landmark)"
        ):
            debug_image = self.draw_mediapipe_facemesh_info(
                debug_image,
                results_list,
                score_th,
            )
        elif model_name == "YuNet":
            debug_image = self.draw_yunet_info(
                debug_image,
                results_list,
                score_th,
            )

        return debug_image

    def draw_mediapipe_face_detection_info(self, image, results_list, score_th):
        for results in results_list:
            for id in range(6):
                if score_th > results[id][2]:
                    continue
                landmark_x, landmark_y = results[id][0], results[id][1]
                cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), -1)

            bbox = results.get("bbox", None)
            if bbox is not None:
                image = cv2.rectangle(
                    image,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    (0, 255, 0),
                    thickness=2,
                )

        return image

    def draw_mediapipe_facemesh_info(self, image, results_list, score_th):
        for results in results_list:
            for id in range(len(results)):
                if score_th > results[id][3]:
                    continue
                landmark_x, landmark_y = results[id][0], results[id][1]
                cv2.circle(image, (landmark_x, landmark_y), 2, (0, 255, 0), -1)

            cv2.line(image, results[55][:2], results[65][:2], (0, 255, 0), 2)
            cv2.line(image, results[65][:2], results[52][:2], (0, 255, 0), 2)
            cv2.line(image, results[52][:2], results[53][:2], (0, 255, 0), 2)
            cv2.line(image, results[53][:2], results[46][:2], (0, 255, 0), 2)

            cv2.line(image, results[285][:2], results[295][:2], (0, 255, 0), 2)
            cv2.line(image, results[295][:2], results[282][:2], (0, 255, 0), 2)
            cv2.line(image, results[282][:2], results[283][:2], (0, 255, 0), 2)
            cv2.line(image, results[283][:2], results[276][:2], (0, 255, 0), 2)

            cv2.line(image, results[133][:2], results[173][:2], (0, 255, 0), 2)
            cv2.line(image, results[173][:2], results[157][:2], (0, 255, 0), 2)
            cv2.line(image, results[157][:2], results[158][:2], (0, 255, 0), 2)
            cv2.line(image, results[158][:2], results[159][:2], (0, 255, 0), 2)
            cv2.line(image, results[159][:2], results[160][:2], (0, 255, 0), 2)
            cv2.line(image, results[160][:2], results[161][:2], (0, 255, 0), 2)
            cv2.line(image, results[161][:2], results[246][:2], (0, 255, 0), 2)

            cv2.line(image, results[246][:2], results[163][:2], (0, 255, 0), 2)
            cv2.line(image, results[163][:2], results[144][:2], (0, 255, 0), 2)
            cv2.line(image, results[144][:2], results[145][:2], (0, 255, 0), 2)
            cv2.line(image, results[145][:2], results[153][:2], (0, 255, 0), 2)
            cv2.line(image, results[153][:2], results[154][:2], (0, 255, 0), 2)
            cv2.line(image, results[154][:2], results[155][:2], (0, 255, 0), 2)
            cv2.line(image, results[155][:2], results[133][:2], (0, 255, 0), 2)

            cv2.line(image, results[362][:2], results[398][:2], (0, 255, 0), 2)
            cv2.line(image, results[398][:2], results[384][:2], (0, 255, 0), 2)
            cv2.line(image, results[384][:2], results[385][:2], (0, 255, 0), 2)
            cv2.line(image, results[385][:2], results[386][:2], (0, 255, 0), 2)
            cv2.line(image, results[386][:2], results[387][:2], (0, 255, 0), 2)
            cv2.line(image, results[387][:2], results[388][:2], (0, 255, 0), 2)
            cv2.line(image, results[388][:2], results[466][:2], (0, 255, 0), 2)

            cv2.line(image, results[466][:2], results[390][:2], (0, 255, 0), 2)
            cv2.line(image, results[390][:2], results[373][:2], (0, 255, 0), 2)
            cv2.line(image, results[373][:2], results[374][:2], (0, 255, 0), 2)
            cv2.line(image, results[374][:2], results[380][:2], (0, 255, 0), 2)
            cv2.line(image, results[380][:2], results[381][:2], (0, 255, 0), 2)
            cv2.line(image, results[381][:2], results[382][:2], (0, 255, 0), 2)
            cv2.line(image, results[382][:2], results[362][:2], (0, 255, 0), 2)

            cv2.line(image, results[308][:2], results[415][:2], (0, 255, 0), 2)
            cv2.line(image, results[415][:2], results[310][:2], (0, 255, 0), 2)
            cv2.line(image, results[310][:2], results[311][:2], (0, 255, 0), 2)
            cv2.line(image, results[311][:2], results[312][:2], (0, 255, 0), 2)
            cv2.line(image, results[312][:2], results[13][:2], (0, 255, 0), 2)
            cv2.line(image, results[13][:2], results[82][:2], (0, 255, 0), 2)
            cv2.line(image, results[82][:2], results[81][:2], (0, 255, 0), 2)
            cv2.line(image, results[81][:2], results[80][:2], (0, 255, 0), 2)
            cv2.line(image, results[80][:2], results[191][:2], (0, 255, 0), 2)
            cv2.line(image, results[191][:2], results[78][:2], (0, 255, 0), 2)

            cv2.line(image, results[78][:2], results[95][:2], (0, 255, 0), 2)
            cv2.line(image, results[95][:2], results[88][:2], (0, 255, 0), 2)
            cv2.line(image, results[88][:2], results[178][:2], (0, 255, 0), 2)
            cv2.line(image, results[178][:2], results[87][:2], (0, 255, 0), 2)
            cv2.line(image, results[87][:2], results[14][:2], (0, 255, 0), 2)
            cv2.line(image, results[14][:2], results[317][:2], (0, 255, 0), 2)
            cv2.line(image, results[317][:2], results[402][:2], (0, 255, 0), 2)
            cv2.line(image, results[402][:2], results[318][:2], (0, 255, 0), 2)
            cv2.line(image, results[318][:2], results[324][:2], (0, 255, 0), 2)
            cv2.line(image, results[324][:2], results[308][:2], (0, 255, 0), 2)

        return image

    def draw_yunet_info(self, image, results_list, score_th):
        for results in results_list:
            for id in range(5):
                if score_th > results[id][2]:
                    continue
                landmark_x, landmark_y = results[id][0], results[id][1]
                cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), -1)

            bbox = results.get("bbox", None)
            if bbox is not None:
                image = cv2.rectangle(
                    image,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    (0, 255, 0),
                    thickness=2,
                )

        return image

    def draw_multi_object_tracking_info(
        self,
        image,
        track_ids,
        bboxes,
        scores,
        class_ids,
        class_names,
        track_id_dict,
        bbox_thickness=None,
    ):
        # Calculate font scale based on image height
        # Reference height: 720px, base font scale: 0.5
        # This makes font size proportional to image size
        image_height = image.shape[0]
        font_scale = max(0.3, min(1.0, (image_height / 720.0) * 0.5))
        
        # Calculate vertical offset and text thickness based on font scale
        vertical_offset_1 = int(36 * (font_scale / 0.5))
        vertical_offset_2 = int(12 * (font_scale / 0.5))
        text_thickness = max(1, int(2 * (font_scale / 0.5)))

        # Use provided bbox_thickness or fall back to adaptive default
        if bbox_thickness is None:
            bbox_thickness = 2
        
        for id, bbox, score, class_id in zip(track_ids, bboxes, scores, class_ids):
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            color = self.get_color(track_id_dict[id])

            image = cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                thickness=bbox_thickness,
            )

            score = "%.2f" % score
            text = "TID:%s(%s)" % (str(int(track_id_dict[id])), str(score))
            image = cv2.putText(
                image,
                text,
                (x1, y1 - vertical_offset_1),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness=text_thickness,
            )

            class_name = self.get_class_name(class_id, class_names)
            text = "CID:%s(%s)" % (str(int(class_id)), class_name)
            image = cv2.putText(
                image,
                text,
                (x1, y1 - vertical_offset_2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness=text_thickness,
            )

        return image

    def get_color_map_list(self, num_classes, custom_color=None):
        num_classes += 1
        color_map = num_classes * [0, 0, 0]
        for i in range(0, num_classes):
            j = 0
            lab = i
            while lab:
                color_map[i * 3 + 2] |= ((lab >> 0) & 1) << (7 - j)
                color_map[i * 3 + 1] |= ((lab >> 1) & 1) << (7 - j)
                color_map[i * 3] |= ((lab >> 2) & 1) << (7 - j)
                j += 1
                lab >>= 3
        color_map = color_map[3:]

        if custom_color:
            color_map[: len(custom_color)] = custom_color
        return color_map


if __name__ == "__main__":
    node = Node()
