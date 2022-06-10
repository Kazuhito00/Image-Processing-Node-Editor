#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import copy

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeABC
from node_editor.util import convert_cv_to_dpg
from node.draw_node.draw_util.draw_util import draw_info


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


def create_image_dict(
    slot_num,
    connection_info_src_dict,
    node_image_dict,
    node_result_dict,
    image_node_name,
    resize_width,
    resize_height,
    draw_info_on_result,
):
    frame_exist_flag = False

    # 初期化用黒画像
    black_image = np.zeros((resize_height, resize_width, 3)).astype(np.uint8)

    frame_dict = {}
    for index in range(slot_num - 1, -1, -1):
        node_id_name = connection_info_src_dict.get(index, None)
        frame = copy.deepcopy(node_image_dict.get(node_id_name, None))
        if frame is not None:
            if draw_info_on_result:
                node_result = node_result_dict[node_id_name]
                image_node_name = node_id_name.split(':')[1]
                frame = draw_info(image_node_name, node_result, frame)
            resize_frame = cv2.resize(frame, (resize_width, resize_height))
            frame_dict[slot_num - index - 1] = copy.deepcopy(resize_frame)

            frame_exist_flag = True
        else:
            frame_dict[slot_num - index - 1] = copy.deepcopy(black_image)

    display_num_list = [1, 2, 4, 4, 6, 6, 9, 9, 9]
    for index in range(display_num_list[slot_num - 1]):
        if frame_dict.get(index, None) is None:
            frame_dict[index] = copy.deepcopy(black_image)

    if not frame_exist_flag:
        frame_dict = None

    return frame_dict


class Node(DpgNodeABC):
    _ver = '0.0.1'

    node_label = 'Image Concat'
    node_tag = 'ImageConcat'

    _opencv_setting_dict = None

    _max_slot_number = 9
    _slot_id = {}

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
        self._value_history = {}

        # タグ名
        tag_node_name = str(node_id) + ':' + self.node_tag
        tag_node_input00_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Input00'
        tag_node_input01_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Input01'
        tag_node_input01_value_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Input01Value'
        tag_node_output01_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Output01'
        tag_node_output01_value_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Output01Value'

        # OpenCV向け設定
        self._opencv_setting_dict = opencv_setting_dict
        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']

        # 初期化用黒画像
        black_image = np.zeros((small_window_w, small_window_h, 3))
        black_texture = convert_cv_to_dpg(
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

        # スロットナンバー保持用Dict
        if tag_node_name not in self._slot_id:
            self._slot_id[tag_node_name] = 1

        # ノード
        with dpg.node(
                tag=tag_node_name,
                parent=parent,
                label=self.node_label,
                pos=pos,
        ):
            # 画像
            with dpg.node_attribute(
                    tag=tag_node_output01_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_image(tag_node_output01_value_name)
            # スロット追加ボタン
            with dpg.node_attribute(
                    tag=tag_node_input00_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_button(
                    label='Add Slot',
                    width=int(small_window_w / 3),
                    callback=self._add_slot,
                    user_data=tag_node_name,
                )
            # スロット
            with dpg.node_attribute(
                    tag=tag_node_input01_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_text(
                    tag=tag_node_input01_value_name,
                    default_value='Input BGR image',
                )

        return tag_node_name

    def update(
        self,
        node_id,
        connection_list,
        node_image_dict,
        node_result_dict,
    ):
        tag_node_name = str(node_id) + ':' + self.node_tag
        output_value01_tag = tag_node_name + ':' + self.TYPE_IMAGE + ':Output01Value'

        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        resize_width = self._opencv_setting_dict['result_width']
        resize_height = self._opencv_setting_dict['result_height']
        draw_info_on_result = self._opencv_setting_dict['draw_info_on_result']

        # 画像取得元のノード名(ID付き)を取得する
        node_name_dict = {}
        connection_info_src = ''
        connection_info_src_dict = {}
        for connection_info in connection_list:
            # タグ名からスロットナンバー取得
            slot_number = re.sub(r'\D', '', connection_info[1].split(':')[-1])
            if slot_number == '':
                continue
            slot_number = int(slot_number) - 1

            connection_type = connection_info[0].split(':')[2]
            if connection_type == self.TYPE_IMAGE:
                # 画像取得元のノード名(ID付き)を取得
                connection_info_src = connection_info[0]
                connection_info_src = connection_info_src.split(':')[:2]
                node_name = connection_info_src[1]
                connection_info_src = ':'.join(connection_info_src)

                node_name_dict[slot_number] = node_name
                connection_info_src_dict[slot_number] = connection_info_src

        slot_num = self._slot_id[tag_node_name]

        # 画像取得
        frame_dict = {}
        if len(connection_info_src_dict) > 0:
            frame_dict = create_image_dict(
                slot_num,
                connection_info_src_dict,
                node_image_dict,
                node_result_dict,
                node_name,
                resize_width,
                resize_height,
                draw_info_on_result,
            )

        # 結合画像生成
        frame = None
        display_frame = None
        if len(connection_info_src_dict) > 0 and frame_dict is not None:
            frame, display_frame = create_concat_image(frame_dict, slot_num)

        # 描画
        if display_frame is not None:
            texture = convert_cv_to_dpg(
                display_frame,
                small_window_w,
                small_window_h,
            )
            dpg_set_value(output_value01_tag, texture)

        return frame, None

    def close(self, node_id):
        pass

    def get_setting_dict(self, node_id):
        tag_node_name = str(node_id) + ':' + self.node_tag

        pos = dpg.get_item_pos(tag_node_name)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos
        setting_dict['slot_id'] = self._slot_id[tag_node_name]

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        tag_node_name = str(node_id) + ':' + self.node_tag

        slot_number = int(setting_dict['slot_id'])
        for _ in range(slot_number - 1):
            self._add_slot(None, None, tag_node_name)

    def _add_slot(self, sender, data, user_data):
        tag_node_name = user_data

        if self._max_slot_number > self._slot_id[tag_node_name]:
            self._slot_id[tag_node_name] += 1

            # 挿入先タグ名生成
            before_tag = tag_node_name + ':' + self.TYPE_IMAGE + ':Input'
            before_tag += str(self._slot_id[tag_node_name] - 1).zfill(2)

            # 追加スロットのタグを生成
            tag_node_inputXX_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Input'
            tag_node_inputXX_name += str(self._slot_id[tag_node_name]).zfill(2)

            tag_node_inputXX_value_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Input'
            tag_node_inputXX_value_name += str(
                self._slot_id[tag_node_name]).zfill(2) + 'Value'

            # スロット追加
            with dpg.node_attribute(
                    tag=tag_node_inputXX_name,
                    attribute_type=dpg.mvNode_Attr_Input,
                    parent=tag_node_name,
                    before=before_tag,
            ):
                dpg.add_text(
                    tag=tag_node_inputXX_value_name,
                    default_value='Input BGR image',
                )
