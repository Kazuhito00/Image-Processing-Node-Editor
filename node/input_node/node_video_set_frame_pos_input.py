#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import copy

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeABC
from node_editor.util import convert_cv_to_dpg


class Node(DpgNodeABC):
    _ver = '0.0.1'

    node_label = 'Video(Set Frame Position)'
    node_tag = 'VideoSetFramePos'

    _opencv_setting_dict = None

    _video_capture = {}
    _movie_filepath = {}
    _prev_movie_filepath = {}
    _prev_frame_pos = {}
    _prev_frame = {}

    _min_val = 0
    _max_val = 10000000

    _window_resize_rate = 1.5

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
        tag_node_input01_name = tag_node_name + ':' + self.TYPE_INT + ':Input01'
        tag_node_input02_name = tag_node_name + ':' + self.TYPE_INT + ':Input02'
        tag_node_input02_value_name = tag_node_name + ':' + self.TYPE_INT + ':Input02Value'
        tag_node_output01_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Output01'
        tag_node_output01_value_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Output01Value'
        tag_node_output02_name = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02'
        tag_node_output02_value_name = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02Value'
        tag_node_output03_name = tag_node_name + ':' + self.TYPE_INT + ':Output03'
        tag_node_output03_value_name = tag_node_name + ':' + self.TYPE_INT + ':Output03Value'

        # OpenCV向け設定
        self._opencv_setting_dict = opencv_setting_dict
        small_window_w = self._opencv_setting_dict['input_window_width']
        small_window_w = int(small_window_w * self._window_resize_rate)
        small_window_h = self._opencv_setting_dict['input_window_height']
        small_window_h = int(small_window_h * self._window_resize_rate)
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

        with dpg.file_dialog(
                directory_selector=False,
                show=False,
                modal=True,
                width=int(small_window_w * 3 / self._window_resize_rate),
                height=int(small_window_h * 3 / self._window_resize_rate),
                callback=self._callback_file_select,
                id='movie_select:' + str(node_id),
        ):
            dpg.add_file_extension('Movie (*.mp4 *.avi){.mp4,.avi}')
            dpg.add_file_extension('', color=(150, 255, 150, 255))

        # ノード
        with dpg.node(
                tag=tag_node_name,
                parent=parent,
                label=self.node_label,
                pos=pos,
        ):
            # ファイル選択
            with dpg.node_attribute(
                    tag=tag_node_input01_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_button(
                    label='Select Movie',
                    width=small_window_w,
                    callback=lambda: dpg.show_item(
                        'movie_select:' + str(node_id), ),
                )
            # カメラ画像
            with dpg.node_attribute(
                    tag=tag_node_output01_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_image(tag_node_output01_value_name)
            # シーク
            with dpg.node_attribute(
                    tag=tag_node_input02_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_slider_int(
                    tag=tag_node_input02_value_name,
                    label='',
                    width=small_window_w,
                    default_value=1,
                    min_value=self._min_val,
                    max_value=self._max_val,
                    format='',
                    callback=None,
                )
            # フレーム位置
            with dpg.node_attribute(
                    tag=tag_node_output03_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_text(
                    '0',
                    tag=tag_node_output03_value_name,
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
        tag_node_input02_value_name = tag_node_name + ':' + self.TYPE_INT + ':Input02Value'
        output_value01_tag = tag_node_name + ':' + self.TYPE_IMAGE + ':Output01Value'
        output_value02_tag = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02Value'
        output_value03_tag = tag_node_name + ':' + self.TYPE_INT + ':Output03Value'

        small_window_w = self._opencv_setting_dict['input_window_width']
        small_window_w = int(small_window_w * self._window_resize_rate)
        small_window_h = self._opencv_setting_dict['input_window_height']
        small_window_h = int(small_window_h * self._window_resize_rate)
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']

        # 接続情報確認
        seek_input_value = None
        for connection_info in connection_list:
            connection_type = connection_info[0].split(':')[2]
            if connection_type == self.TYPE_INT:
                # 接続タグ取得
                source_tag = connection_info[0] + 'Value'
                # 値取得
                seek_input_value = int(dpg_get_value(source_tag))
                seek_input_value = max([self._min_val, seek_input_value])
                seek_input_value = min([self._max_val, seek_input_value])

        # VideoCapture()インスタンス生成
        update_flag = False
        movie_path = self._movie_filepath.get(str(node_id), None)
        prev_movie_path = self._prev_movie_filepath.get(str(node_id), None)
        if prev_movie_path != movie_path:
            video_capture = self._video_capture.get(str(node_id), None)
            if video_capture is not None:
                video_capture.release()
            self._video_capture[str(node_id)] = cv2.VideoCapture(movie_path)
            self._prev_movie_filepath[str(node_id)] = movie_path

            # シーク位置リセット
            self._video_capture[str(node_id)].set(cv2.CAP_PROP_POS_FRAMES, 0)
            # フレーム数リセット
            dpg_set_value(tag_node_input02_value_name, 0)
            dpg_set_value(output_value03_tag, str(0))
            update_flag = True

        video_capture = self._video_capture.get(str(node_id), None)

        # シーク位置
        seek_value = int(dpg_get_value(tag_node_input02_value_name))

        # 計測開始
        if video_capture is not None and use_pref_counter:
            start_time = time.perf_counter()

        # 画像取得
        frame = None
        if video_capture is not None:
            # シーク位置のフレーム数を算出
            total_frame = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if seek_input_value is None:
                frame_pos = int(total_frame * (seek_value / self._max_val))
            else:
                frame_pos = seek_input_value

            # 範囲チェック
            if frame_pos < 0:
                frame_pos = 0
            if total_frame <= frame_pos:
                frame_pos = total_frame - 1

            # シーク位置が他ノードから入力されていた場合はシークバーの位置を変更
            if seek_input_value is not None:
                seek_set_value = int(self._max_val * (frame_pos / total_frame))
                dpg_set_value(tag_node_input02_value_name, seek_set_value)

            if str(node_id) in self._prev_frame_pos:
                # フレーム位置が変更されていたら画像を取得
                if self._prev_frame_pos[str(node_id)] != frame_pos:
                    update_flag = True
                else:
                    frame = copy.deepcopy(self._prev_frame[str(node_id)])
            else:
                # 初回画像取得
                update_flag = True

            if update_flag:
                # シーク
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                # 画像取得
                _, frame = video_capture.read()
                # 取得時の位置と画像を保持
                self._prev_frame_pos[str(node_id)] = frame_pos
                self._prev_frame[str(node_id)] = copy.deepcopy(frame)
                # フレーム数表示
                dpg_set_value(output_value03_tag, str(frame_pos))

        # 計測終了
        if video_capture is not None and use_pref_counter:
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
        tag_node_input02_value_name = tag_node_name + ':' + self.TYPE_INT + ':Input02Value'

        pos = dpg.get_item_pos(tag_node_name)

        seek_value = int(dpg_get_value(tag_node_input02_value_name))

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos
        setting_dict[tag_node_input02_value_name] = seek_value

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        tag_node_name = str(node_id) + ':' + self.node_tag
        tag_node_input02_value_name = tag_node_name + ':' + self.TYPE_INT + ':Input02Value'

        seek_value = setting_dict[tag_node_input02_value_name]

        dpg_set_value(tag_node_input02_value_name, seek_value)

    def _callback_file_select(self, sender, data):
        if data['file_name'] != '.':
            node_id = sender.split(':')[1]
            self._movie_filepath[node_id] = data['file_path_name']
