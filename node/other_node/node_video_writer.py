#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import copy
import datetime

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeABC
from node_editor.util import convert_cv_to_dpg


class Node(DpgNodeABC):
    _ver = '0.0.1'

    node_label = 'Video Writer'
    node_tag = 'VideoWriter'

    _opencv_setting_dict = None

    _video_writer_dict = {}
    _start_label = 'Start'
    _stop_label = 'Stop'

    _prev_frame_flag = False

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

        tag_node_button_name = tag_node_name + ':' + self.TYPE_TEXT + ':Button'
        tag_node_button_value_name = tag_node_name + ':' + self.TYPE_TEXT + ':ButtonValue'

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
                tag=tag_node_input01_value_name,
                format=dpg.mvFormat_Float_rgb,
            )

        # ノード
        with dpg.node(
                tag=tag_node_name,
                parent=parent,
                label=self.node_label,
                pos=pos,
        ):
            # 画像
            with dpg.node_attribute(
                    tag=tag_node_input01_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_image(tag_node_input01_value_name)
            # 録画/再生追加ボタン
            with dpg.node_attribute(
                    tag=tag_node_button_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_button(
                    label=self._start_label,
                    tag=tag_node_button_value_name,
                    width=small_window_w,
                    callback=self._recording_button,
                    user_data=tag_node_name,
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
        input_value01_tag = tag_node_name + ':' + self.TYPE_IMAGE + ':Input01Value'
        tag_node_button_value_name = tag_node_name + ':' + self.TYPE_TEXT + ':ButtonValue'

        # 画像取得元のノード名(ID付き)を取得する
        connection_info_src = ''
        for connection_info in connection_list:
            connection_info_src = connection_info[0]
            connection_info_src = connection_info_src.split(':')[:2]
            connection_info_src = ':'.join(connection_info_src)

        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        writer_width = self._opencv_setting_dict['video_writer_width']
        writer_height = self._opencv_setting_dict['video_writer_height']

        # 画像取得
        frame = node_image_dict.get(connection_info_src, None)

        # 描画
        if frame is not None:
            rec_frame = copy.deepcopy(frame)
            # 録画中
            if tag_node_name in self._video_writer_dict:
                # 動画書き出し
                writer_frame = cv2.resize(rec_frame,
                                          (writer_width, writer_height))
                self._video_writer_dict[tag_node_name].write(writer_frame)

                # 録画表示
                rec_frame = cv2.circle(rec_frame, (80, 80),
                                       50, (0, 0, 255),
                                       thickness=-1)
                rec_frame = cv2.putText(rec_frame,
                                        'Rec', (130, 120),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        4.0, (0, 0, 255),
                                        thickness=10)

            # 画面反映
            texture = convert_cv_to_dpg(
                rec_frame,
                small_window_w,
                small_window_h,
            )
            dpg_set_value(input_value01_tag, texture)
        else:
            label = dpg.get_item_label(tag_node_button_value_name)
            if label == self._stop_label and self._prev_frame_flag:
                # 録画停止
                self._recording_button(None, None, tag_node_name)
                # 初期化用黒画像
                black_image = np.zeros((small_window_w, small_window_h, 3))
                # 画面反映
                texture = convert_cv_to_dpg(
                    black_image,
                    small_window_w,
                    small_window_h,
                )
                dpg_set_value(input_value01_tag, texture)

        if frame is not None:
            self._prev_frame_flag = True
        else:
            self._prev_frame_flag = False

        return frame, None

    def close(self, node_id):
        tag_node_name = str(node_id) + ':' + self.node_tag
        if tag_node_name in self._video_writer_dict:
            self._video_writer_dict[tag_node_name].release()
            self._video_writer_dict.pop(tag_node_name)

    def get_setting_dict(self, node_id):
        tag_node_name = str(node_id) + ':' + self.node_tag

        pos = dpg.get_item_pos(tag_node_name)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        pass

    def _recording_button(self, sender, data, user_data):
        tag_node_name = user_data
        tag_node_button_value_name = tag_node_name + ':' + self.TYPE_TEXT + ':ButtonValue'

        label = dpg.get_item_label(tag_node_button_value_name)

        if label == self._start_label:
            # 開始時刻
            datetime_now = datetime.datetime.now()
            startup_time_text = datetime_now.strftime('%Y%m%d_%H%M%S')

            # 録画設定
            writer_width = self._opencv_setting_dict['video_writer_width']
            writer_height = self._opencv_setting_dict['video_writer_height']
            writer_fps = self._opencv_setting_dict['video_writer_fps']
            video_writer_directory = self._opencv_setting_dict[
                'video_writer_directory']

            os.makedirs(video_writer_directory, exist_ok=True)

            # 録画開始
            if tag_node_name not in self._video_writer_dict:
                self._video_writer_dict[tag_node_name] = cv2.VideoWriter(
                    video_writer_directory + '/' + startup_time_text + '.mp4',
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    writer_fps,
                    (writer_width, writer_height),
                )

            dpg.set_item_label(tag_node_button_value_name, self._stop_label)
        elif label == self._stop_label:
            # 録画終了
            self._video_writer_dict[tag_node_name].release()
            self._video_writer_dict.pop(tag_node_name)

            dpg.set_item_label(tag_node_button_value_name, self._start_label)
