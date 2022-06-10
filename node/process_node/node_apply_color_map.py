#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeABC
from node_editor.util import convert_cv_to_dpg


def image_process(image, colormap):
    image = cv2.applyColorMap(image, colormap)
    return image


class Node(DpgNodeABC):
    _ver = '0.0.1'

    node_label = 'ApplyColorMap'
    node_tag = 'ApplyColorMap'

    _opencv_setting_dict = None
    _colormap_types = {
        'COLORMAP_AUTUMN': cv2.COLORMAP_AUTUMN,
        'COLORMAP_BONE': cv2.COLORMAP_BONE,
        'COLORMAP_JET': cv2.COLORMAP_JET,
        'COLORMAP_WINTER': cv2.COLORMAP_WINTER,
        'COLORMAP_RAINBOW': cv2.COLORMAP_RAINBOW,
        'COLORMAP_OCEAN': cv2.COLORMAP_OCEAN,
        'COLORMAP_SUMMER': cv2.COLORMAP_SUMMER,
        'COLORMAP_SPRING': cv2.COLORMAP_SPRING,
        'COLORMAP_COOL': cv2.COLORMAP_COOL,
        'COLORMAP_HSV': cv2.COLORMAP_HSV,
        'COLORMAP_PINK': cv2.COLORMAP_PINK,
        'COLORMAP_HOT': cv2.COLORMAP_HOT,
        'COLORMAP_PARULA': cv2.COLORMAP_PARULA,
        'COLORMAP_MAGMA': cv2.COLORMAP_MAGMA,
        'COLORMAP_INFERNO': cv2.COLORMAP_INFERNO,
        'COLORMAP_PLASMA': cv2.COLORMAP_PLASMA,
        'COLORMAP_VIRIDIS': cv2.COLORMAP_VIRIDIS,
        'COLORMAP_CIVIDIS': cv2.COLORMAP_CIVIDIS,
        'COLORMAP_TWILIGHT': cv2.COLORMAP_TWILIGHT,
        'COLORMAP_TWILIGHT_SHIFTED': cv2.COLORMAP_TWILIGHT_SHIFTED,
        'COLORMAP_TURBO': cv2.COLORMAP_TURBO,
        'COLORMAP_DEEPGREEN': cv2.COLORMAP_DEEPGREEN,
    }

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
        tag_node_input02_name = tag_node_name + ':' + self.TYPE_TEXT + ':Input02'
        tag_node_input02_value_name = tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'
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
            # カラーマップタイプ
            with dpg.node_attribute(
                    tag=tag_node_input02_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_combo(
                    list(self._colormap_types.keys()),
                    default_value=list(self._colormap_types.keys())[2],
                    width=small_window_w - 40,
                    label="type",
                    tag=tag_node_input02_value_name,
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
        input_value02_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'
        output_value01_tag = tag_node_name + ':' + self.TYPE_IMAGE + ':Output01Value'
        output_value02_tag = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02Value'

        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']

        # 接続情報確認
        connection_info_src = ''
        for connection_info in connection_list:
            connection_type = connection_info[0].split(':')[2]
            if connection_type == self.TYPE_INT:
                # 接続タグ取得
                source_tag = connection_info[0] + 'Value'
                destination_tag = connection_info[1] + 'Value'
                # 値更新
                input_value = int(dpg_get_value(source_tag))
                input_value = max([self._min_val, input_value])
                input_value = min([self._max_val, input_value])
                dpg_set_value(destination_tag, input_value)
            if connection_type == self.TYPE_IMAGE:
                # 画像取得元のノード名(ID付き)を取得
                connection_info_src = connection_info[0]
                connection_info_src = connection_info_src.split(':')[:2]
                connection_info_src = ':'.join(connection_info_src)

        # 画像取得
        frame = node_image_dict.get(connection_info_src, None)

        # カラーマップタイプ
        colormap_type = dpg_get_value(input_value02_tag)
        colormap_type = self._colormap_types[colormap_type]

        # 計測開始
        if frame is not None and use_pref_counter:
            start_time = time.perf_counter()

        if frame is not None:
            frame = image_process(frame, colormap_type)

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
        input_value02_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'

        # カラーマップタイプ
        colormap_type = dpg_get_value(input_value02_tag)

        pos = dpg.get_item_pos(tag_node_name)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos
        setting_dict[input_value02_tag] = colormap_type

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        tag_node_name = str(node_id) + ':' + self.node_tag
        input_value02_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'

        colormap_type = setting_dict[input_value02_tag]

        dpg_set_value(input_value02_tag, colormap_type)
