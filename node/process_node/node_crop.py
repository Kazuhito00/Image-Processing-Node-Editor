#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeABC
from node_editor.util import convert_cv_to_dpg


def image_process(image, min_x, max_x, min_y, max_y):
    if max_x < min_x:
        max_x = min_x + 0.01
    if max_y < min_y:
        max_y = min_y + 0.01

    image_height, image_width = image.shape[0], image.shape[1]
    min_x_ = int(min_x * image_width)
    max_x_ = int(max_x * image_width)
    min_y_ = int(min_y * image_height)
    max_y_ = int(max_y * image_height)
    image = image[min_y_:max_y_, min_x_:max_x_]
    return image


class Node(DpgNodeABC):
    _ver = '0.0.1'

    node_label = 'Crop'
    node_tag = 'Crop'

    _min_min_val = 0.0
    _min_max_val = 0.99
    _max_min_val = 0.01
    _max_max_val = 1.00

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
        tag_node_input02_name = tag_node_name + ':' + self.TYPE_FLOAT + ':Input02'
        tag_node_input02_value_name = tag_node_name + ':' + self.TYPE_FLOAT + ':Input02Value'
        tag_node_input03_name = tag_node_name + ':' + self.TYPE_FLOAT + ':Input03'
        tag_node_input03_value_name = tag_node_name + ':' + self.TYPE_FLOAT + ':Input03Value'
        tag_node_input04_name = tag_node_name + ':' + self.TYPE_FLOAT + ':Input04'
        tag_node_input04_value_name = tag_node_name + ':' + self.TYPE_FLOAT + ':Input04Value'
        tag_node_input05_name = tag_node_name + ':' + self.TYPE_FLOAT + ':Input05'
        tag_node_input05_value_name = tag_node_name + ':' + self.TYPE_FLOAT + ':Input05Value'
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
            # 領域指定
            with dpg.node_attribute(
                    tag=tag_node_input02_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_slider_float(
                    tag=tag_node_input02_value_name,
                    label="min x",
                    width=small_window_w - 80,
                    default_value=0,
                    min_value=self._min_min_val,
                    max_value=self._min_max_val,
                    callback=None,
                )
            with dpg.node_attribute(
                    tag=tag_node_input03_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_slider_float(
                    tag=tag_node_input03_value_name,
                    label="max x",
                    width=small_window_w - 80,
                    default_value=1.0,
                    min_value=self._max_min_val,
                    max_value=self._max_max_val,
                    callback=None,
                )
            with dpg.node_attribute(
                    tag=tag_node_input04_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_slider_float(
                    tag=tag_node_input04_value_name,
                    label="min y",
                    width=small_window_w - 80,
                    default_value=0,
                    min_value=self._min_min_val,
                    max_value=self._min_max_val,
                    callback=None,
                )
            with dpg.node_attribute(
                    tag=tag_node_input05_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_slider_float(
                    tag=tag_node_input05_value_name,
                    label="max y",
                    width=small_window_w - 80,
                    default_value=1.0,
                    min_value=self._max_min_val,
                    max_value=self._max_max_val,
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
        input_value02_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Input02Value'
        input_value03_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Input03Value'
        input_value04_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Input04Value'
        input_value05_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Input05Value'
        output_value01_tag = tag_node_name + ':' + self.TYPE_IMAGE + ':Output01Value'
        output_value02_tag = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02Value'

        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']

        # 接続情報確認
        connection_info_src = ''
        for connection_info in connection_list:
            connection_type = connection_info[0].split(':')[2]
            connection_tag = connection_info[1].split(':')[3]
            if connection_type == self.TYPE_FLOAT:
                # 接続タグ取得
                source_tag = connection_info[0] + 'Value'
                destination_tag = connection_info[1] + 'Value'
                # 値更新
                input_value = round(float(dpg_get_value(source_tag)), 3)
                if connection_tag == 'Input02' or connection_tag == 'Input04':
                    input_value = max([self._min_min_val, input_value])
                    input_value = min([self._min_max_val, input_value])
                if connection_tag == 'Input03' or connection_tag == 'Input05':
                    input_value = max([self._max_min_val, input_value])
                    input_value = min([self._max_max_val, input_value])
                dpg_set_value(destination_tag, input_value)
            if connection_type == self.TYPE_IMAGE:
                # 画像取得元のノード名(ID付き)を取得
                connection_info_src = connection_info[0]
                connection_info_src = connection_info_src.split(':')[:2]
                connection_info_src = ':'.join(connection_info_src)

        # 画像取得
        frame = node_image_dict.get(connection_info_src, None)

        # 領域指定
        min_x = float(dpg_get_value(input_value02_tag))
        max_x = float(dpg_get_value(input_value03_tag))
        min_y = float(dpg_get_value(input_value04_tag))
        max_y = float(dpg_get_value(input_value05_tag))
        if min_x > max_x:
            min_x, max_x = max_x - 0.01, min_x + 0.01
            dpg_set_value(input_value02_tag, min_x)
            dpg_set_value(input_value03_tag, max_x)
        if min_y > max_y:
            min_y, max_y = max_y - 0.01, min_y + 0.01
            dpg_set_value(input_value04_tag, min_y)
            dpg_set_value(input_value05_tag, max_y)

        # 計測開始
        if frame is not None and use_pref_counter:
            start_time = time.perf_counter()

        if frame is not None:
            frame = image_process(frame, min_x, max_x, min_y, max_y)

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
        input_value02_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Input02Value'
        input_value03_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Input03Value'
        input_value04_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Input04Value'
        input_value05_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Input05Value'

        # 領域指定
        min_x = float(dpg_get_value(input_value02_tag))
        max_x = float(dpg_get_value(input_value03_tag))
        min_y = float(dpg_get_value(input_value04_tag))
        max_y = float(dpg_get_value(input_value05_tag))

        pos = dpg.get_item_pos(tag_node_name)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos
        setting_dict[input_value02_tag] = min_x
        setting_dict[input_value03_tag] = max_x
        setting_dict[input_value04_tag] = min_y
        setting_dict[input_value05_tag] = max_y

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        tag_node_name = str(node_id) + ':' + self.node_tag
        input_value02_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Input02Value'
        input_value03_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Input03Value'
        input_value04_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Input04Value'
        input_value05_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Input05Value'

        min_x = float(setting_dict[input_value02_tag])
        max_x = float(setting_dict[input_value03_tag])
        min_y = float(setting_dict[input_value04_tag])
        max_y = float(setting_dict[input_value05_tag])

        dpg_set_value(input_value02_tag, min_x)
        dpg_set_value(input_value03_tag, max_x)
        dpg_set_value(input_value04_tag, min_y)
        dpg_set_value(input_value05_tag, max_y)
