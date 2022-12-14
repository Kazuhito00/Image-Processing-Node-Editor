#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import re
import copy

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeABC
from node_editor.util import convert_cv_to_dpg
from node.draw_node.draw_util.draw_util import draw_info


def image_process(image, min_val, max_val):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.Canny(image, min_val, max_val)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image

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

    node_label = 'Image Alpha Blend'
    node_tag = 'ImageAlphaBlend'
    _max_slot_number = 2
    _slot_id = {}

    _minval_min = 1
    _minval_max = 254
    _maxval_min = 2
    _maxval_max = 255

    _opencv_setting_dict = None

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
        # タグ名
        tag_node_name = str(node_id) + ':' + self.node_tag
        tag_node_input01_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Input01'
        tag_node_input01_value_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Input01Value'
        tag_node_input02_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Input02'
        tag_node_input02_value_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Input02Value'
        tag_node_input03_name = tag_node_name + ':' + self.TYPE_INT + ':Input03'
        tag_node_input03_value_name = tag_node_name + ':' + self.TYPE_INT + ':Input03Value'
        tag_node_input04_name = tag_node_name + ':' + self.TYPE_INT + ':Input04'
        tag_node_input04_value_name = tag_node_name + ':' + self.TYPE_INT + ':Input04Value'
        tag_node_output01_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Output01'
        tag_node_output01_value_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Output01Value'
        tag_node_output02_name = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02'
        tag_node_output02_value_name = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02Value'

        # OpenCV向け設定
        self._opencv_setting_dict = opencv_setting_dict
        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']

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
            # 入力端子
            with dpg.node_attribute(
                    tag=tag_node_input01_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_text(
                    tag=tag_node_input01_value_name,
                    default_value='Input BGR image',
                )
            # 入力端子
            with dpg.node_attribute(
                    tag=tag_node_input02_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_text(
                    tag=tag_node_input02_value_name,
                    default_value='Input BGR image',
                )
            # 画像
            with dpg.node_attribute(
                    tag=tag_node_output01_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_image(tag_node_output01_value_name)
            # ヒステリシス
            with dpg.node_attribute(
                    tag=tag_node_input03_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_slider_int(
                    tag=tag_node_input03_value_name,
                    label="min val",
                    width=small_window_w - 80,
                    default_value=100,
                    min_value=self._minval_min,
                    max_value=self._minval_max,
                    callback=None,
                )
            with dpg.node_attribute(
                    tag=tag_node_input04_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_slider_int(
                    tag=tag_node_input04_value_name,
                    label="max val",
                    width=small_window_w - 80,
                    default_value=200,
                    min_value=self._maxval_min,
                    max_value=self._maxval_max,
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

        return tag_node_name

    def update(
        self,
        node_id,
        connection_list,
        node_image_dict,
        node_result_dict,
    ):
        tag_node_name = str(node_id) + ':' + self.node_tag
        input_value03_tag = tag_node_name + ':' + self.TYPE_INT + ':Input03Value'
        input_value04_tag = tag_node_name + ':' + self.TYPE_INT + ':Input04Value'
        output_value01_tag = tag_node_name + ':' + self.TYPE_IMAGE + ':Output01Value'
        output_value02_tag = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02Value'

        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']
        draw_info_on_result = self._opencv_setting_dict['draw_info_on_result']

        # 接続情報確認
        node_name_dict = {}
        connection_info_src = ''
        connection_info_src_dict = {}
        for connection_info in connection_list:
            print("list",connection_list)
            # タグ名からスロットナンバー取得
            slot_number = re.sub(r'\D', '', connection_info[1].split(':')[-1])
#            print(slot_number)
            if slot_number == '':
                continue
            slot_number = int(slot_number) - 1
            connection_type = connection_info[0].split(':')[2]
            connection_tag = connection_info[1].split(':')[3]
            if connection_type == self.TYPE_INT:
                # 接続タグ取得
                source_tag = connection_info[0] + 'Value'
                destination_tag = connection_info[1] + 'Value'
                # 値更新
                input_value = int(dpg_get_value(source_tag))
                if connection_tag == 'Input02':
                    input_value = max([self._minval_min, input_value])
                    input_value = min([self._minval_max, input_value])
                if connection_tag == 'Input03':
                    input_value = max([self._maxval_min, input_value])
                    input_value = min([self._maxval_max, input_value])
                dpg_set_value(destination_tag, input_value)
            if connection_type == self.TYPE_IMAGE:
                # 画像取得元のノード名(ID付き)を取得
                connection_info_src = connection_info[0]
                connection_info_src = connection_info_src.split(':')[:2]
                node_name = connection_info_src[1]
                connection_info_src = ':'.join(connection_info_src)
#                print(connection_info_src)
                node_name_dict[slot_number] = node_name
                connection_info_src_dict[slot_number] = connection_info_src
                print("src",connection_info_src)

        slot_num = self._slot_id[tag_node_name]
        # 画像取得
        frame_dict = {}
        print("src_dict",connection_info_src_dict)
        if len(connection_info_src_dict) > 0:
            frame_dict = create_image_dict(
                slot_num,
                connection_info_src_dict,
                node_image_dict,
                node_result_dict,
                node_name,
                small_window_w,
                small_window_h,
                draw_info_on_result,
            )
        print("ss")
        frame = node_image_dict.get(connection_info_src, None)
        print("sss")

        print(frame_dict)
        #frame = frame_dict[0]

        # ヒステリシス
        min_val = int(dpg_get_value(input_value03_tag))
        max_val = int(dpg_get_value(input_value04_tag))
        if min_val > max_val:
            min_val, max_val = max_val - 1, min_val + 1
            dpg_set_value(input_value03_tag, min_val)
            dpg_set_value(input_value04_tag, max_val)

        # 計測開始
        if frame is not None and use_pref_counter:
            start_time = time.perf_counter()

        if frame is not None:
            frame = image_process(frame, min_val, max_val)

        # 計測終了
        if frame is not None and use_pref_counter:
            elapsed_time = time.perf_counter() - start_time
            elapsed_time = int(elapsed_time * 1000)
            dpg_set_value(output_value02_tag,
                          str(elapsed_time).zfill(4) + 'ms')

        # 描画
        if frame is not None:
            texture = convert_cv_to_dpg(
                frame,
                small_window_w,
                small_window_h,
            )
            dpg_set_value(output_value01_tag, texture)

        return frame, None

    def close(self, node_id):
        pass

    def get_setting_dict(self, node_id):
        tag_node_name = str(node_id) + ':' + self.node_tag
        input_value03_tag = tag_node_name + ':' + self.TYPE_INT + ':Input03Value'

        kernel_size = dpg_get_value(input_value03_tag)

        pos = dpg.get_item_pos(tag_node_name)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos
        setting_dict[input_value03_tag] = kernel_size

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        tag_node_name = str(node_id) + ':' + self.node_tag
        input_value03_tag = tag_node_name + ':' + self.TYPE_INT + ':Input02Value'

        kernel_size = int(setting_dict[input_value03_tag])

        dpg_set_value(input_value03_tag, kernel_size)
