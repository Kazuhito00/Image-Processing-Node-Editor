#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeABC
from node_editor.util import convert_cv_to_dpg


class Node(DpgNodeABC):
    _ver = '0.0.1'

    node_label = 'Video'
    node_tag = 'Video'

    _opencv_setting_dict = None

    _video_capture = {}
    _movie_filepath = {}
    _prev_movie_filepath = {}
    _frame_count = {}

    _min_val = 1
    _max_val = 10

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
        tag_node_input02_name = tag_node_name + ':' + self.TYPE_TEXT + ':Input02'
        tag_node_input02_value_name = tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'
        tag_node_input03_name = tag_node_name + ':' + self.TYPE_INT + ':Input03'
        tag_node_input03_value_name = tag_node_name + ':' + self.TYPE_INT + ':Input03Value'
        tag_node_output01_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Output01'
        tag_node_output01_value_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Output01Value'
        tag_node_output02_name = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02'
        tag_node_output02_value_name = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02Value'

        # OpenCV向け設定
        self._opencv_setting_dict = opencv_setting_dict
        small_window_w = self._opencv_setting_dict['input_window_width']
        small_window_h = self._opencv_setting_dict['input_window_height']
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
                height=int(small_window_h * 3),
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
            # ループ要否
            with dpg.node_attribute(
                    tag=tag_node_input02_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_checkbox(
                    label='Loop',
                    tag=tag_node_input02_value_name,
                    callback=None,
                    user_data=tag_node_name,
                    default_value=True,
                )
            # カーネルサイズ
            with dpg.node_attribute(
                    tag=tag_node_input03_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_slider_int(
                    tag=tag_node_input03_value_name,
                    label="Skip Rate",
                    width=small_window_w - 80,
                    default_value=1,
                    min_value=self._min_val,
                    max_value=self._max_val,
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
        tag_node_input02_value_name = tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'
        tag_node_input03_value_name = tag_node_name + ':' + self.TYPE_INT + ':Input03Value'
        output_value01_tag = tag_node_name + ':' + self.TYPE_IMAGE + ':Output01Value'
        output_value02_tag = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02Value'

        small_window_w = self._opencv_setting_dict['input_window_width']
        small_window_h = self._opencv_setting_dict['input_window_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']

        # 接続情報確認
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

        # VideoCapture()インスタンス生成
        movie_path = self._movie_filepath.get(str(node_id), None)
        prev_movie_path = self._prev_movie_filepath.get(str(node_id), None)
        if prev_movie_path != movie_path:
            video_capture = self._video_capture.get(str(node_id), None)
            if video_capture is not None:
                video_capture.release()
            self._video_capture[str(node_id)] = cv2.VideoCapture(movie_path)
            self._prev_movie_filepath[str(node_id)] = movie_path
            self._frame_count[str(node_id)] = 0

        video_capture = self._video_capture.get(str(node_id), None)

        # ループ要否
        loop_flag = dpg_get_value(tag_node_input02_value_name)
        # スキップ割合
        skip_rate = int(dpg_get_value(tag_node_input03_value_name))

        # 計測開始
        if video_capture is not None and use_pref_counter:
            start_time = time.perf_counter()

        # 画像取得
        frame = None
        if video_capture is not None:
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    if loop_flag:
                        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        _, frame = video_capture.read()
                    else:
                        video_capture.release()
                        video_capture = None
                        self._movie_filepath.pop(str(node_id))
                        self._prev_movie_filepath.pop(str(node_id))
                        self._video_capture.pop(str(node_id))

                        break

                self._frame_count[str(node_id)] += 1
                if (self._frame_count[str(node_id)] % skip_rate) == 0:
                    break

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
        tag_node_input02_value_name = tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'
        tag_node_input03_value_name = tag_node_name + ':' + self.TYPE_INT + ':Input03Value'

        pos = dpg.get_item_pos(tag_node_name)

        loop_flag = dpg_get_value(tag_node_input02_value_name)
        skip_rate = int(dpg_get_value(tag_node_input03_value_name))

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos
        setting_dict[tag_node_input02_value_name] = loop_flag
        setting_dict[tag_node_input03_value_name] = skip_rate

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        tag_node_name = str(node_id) + ':' + self.node_tag
        tag_node_input02_value_name = tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'
        tag_node_input03_value_name = tag_node_name + ':' + self.TYPE_INT + ':Input03Value'

        loop_flag = setting_dict[tag_node_input02_value_name]
        skip_rate = int(setting_dict[tag_node_input03_value_name])

        dpg_set_value(tag_node_input02_value_name, loop_flag)
        dpg_set_value(tag_node_input03_value_name, skip_rate)

    def _callback_file_select(self, sender, data):
        if data['file_name'] != '.':
            node_id = sender.split(':')[1]
            self._movie_filepath[node_id] = data['file_path_name']
