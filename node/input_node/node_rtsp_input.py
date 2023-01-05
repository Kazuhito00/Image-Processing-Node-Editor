#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import multiprocessing as mp

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeABC
from node_editor.util import convert_cv_to_dpg


def receive_image_process(rtsp_url, image_queue, request):
    rtsp_capture = cv2.VideoCapture(rtsp_url)

    while True:
        ret, frame = rtsp_capture.read()

        if ret:
            if image_queue.qsize() == 0:
                image_queue.put(frame)
            time.sleep(0.001)
        else:
            # 取得失敗時は1秒待ち再接続
            time.sleep(1)
            rtsp_capture.release()
            rtsp_capture = cv2.VideoCapture(rtsp_url)

        # 0指定時はプロセスを終了する
        if request.value == 0:
            rtsp_capture.release()
            break


class Node(DpgNodeABC):
    _ver = '0.0.1'

    node_label = 'RTSP'
    node_tag = 'RTSPInput'

    _opencv_setting_dict = None
    _start_label = 'Start'
    _stop_label = 'Stop'

    _rtsp_capture = {}

    _image_queue = {}
    _request = {}
    _process = {}

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
        tag_node_input01_name = tag_node_name + ':' + self.TYPE_TEXT + ':Input01'
        tag_node_input01_value_name = tag_node_name + ':' + self.TYPE_TEXT + ':Input01Value'
        tag_node_output01_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Output01'
        tag_node_output01_value_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Output01Value'
        tag_node_output02_name = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02'
        tag_node_output02_value_name = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02Value'

        tag_node_button_name = tag_node_name + ':' + self.TYPE_TEXT + ':Button'
        tag_node_button_value_name = tag_node_name + ':' + self.TYPE_TEXT + ':ButtonValue'

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

        # ノード
        with dpg.node(
                tag=tag_node_name,
                parent=parent,
                label=self.node_label,
                pos=pos,
        ):
            # RTSP URL入力欄
            with dpg.node_attribute(
                    tag=tag_node_input01_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_input_text(
                    tag=tag_node_input01_value_name,
                    label='URL',
                    width=small_window_w - 30,
                )
            # カメラ画像
            with dpg.node_attribute(
                    tag=tag_node_output01_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_image(tag_node_output01_value_name)
            # 録画/再生追加ボタン
            with dpg.node_attribute(
                    tag=tag_node_button_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_button(
                    label=self._start_label,
                    tag=tag_node_button_value_name,
                    width=small_window_w,
                    callback=self._button,
                    user_data=tag_node_name,
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
        input_value01_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input01Value'
        output_value01_tag = tag_node_name + ':' + self.TYPE_IMAGE + ':Output01Value'
        output_value02_tag = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02Value'

        small_window_w = self._opencv_setting_dict['input_window_width']
        small_window_h = self._opencv_setting_dict['input_window_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']

        # multiprocessing使用有無
        use_mp = self._opencv_setting_dict['use_multiprocessing_rtsp']

        # RTSP URL取得
        rtsp_url = dpg_get_value(input_value01_tag)

        # VideoCapture()インスタンス取得
        rtsp_capture = None
        image_queue = None
        if rtsp_url != '':
            if use_mp:
                # multiprocessing使用
                if rtsp_url in self._image_queue:
                    image_queue = self._image_queue[rtsp_url]
            else:
                # multiprocessing未使用
                if rtsp_url in self._rtsp_capture:
                    rtsp_capture = self._rtsp_capture[rtsp_url]

        # 計測開始
        if rtsp_url != '' and use_pref_counter:
            start_time = time.perf_counter()

        # 画像取得
        frame = None
        if use_mp:
            # multiprocessing使用
            if image_queue is not None:
                num = image_queue.qsize()
                if num > 0:
                    frame = image_queue.get()
        else:
            # multiprocessing未使用
            if rtsp_capture is not None:
                ret, frame = rtsp_capture.read()
                if not ret:
                    return None, None

        # 計測終了
        if rtsp_url != '' and use_pref_counter:
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
        # multiprocessing使用有無
        use_mp = self._opencv_setting_dict['use_multiprocessing_rtsp']
        if use_mp:
            # multiprocessing使用
            for rtsp_url in self._process.keys():
                self._request[rtsp_url].value = 0
                if self._process[rtsp_url].is_alive():
                    self._process[rtsp_url].terminate()

    def get_setting_dict(self, node_id):
        tag_node_name = str(node_id) + ':' + self.node_tag
        tag_node_input01_value_name = tag_node_name + ':' + self.TYPE_TEXT + ':Input01Value'

        pos = dpg.get_item_pos(tag_node_name)
        rtsp_url = dpg_get_value(tag_node_input01_value_name)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos
        setting_dict[tag_node_input01_value_name] = rtsp_url

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        tag_node_name = str(node_id) + ':' + self.node_tag
        tag_node_input01_value_name = tag_node_name + ':' + self.TYPE_TEXT + ':Input01Value'

        rtsp_url = setting_dict[tag_node_input01_value_name]

        dpg_set_value(tag_node_input01_value_name, rtsp_url)

    def _button(self, sender, data, user_data):
        tag_node_name = user_data
        input_value01_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input01Value'
        tag_node_button_value_name = tag_node_name + ':' + self.TYPE_TEXT + ':ButtonValue'

        label = dpg.get_item_label(tag_node_button_value_name)

        # RTSP URL取得
        rtsp_url = dpg_get_value(input_value01_tag)

        # multiprocessing使用有無
        use_mp = self._opencv_setting_dict['use_multiprocessing_rtsp']

        if label == self._start_label:
            if rtsp_url != '':
                if use_mp:
                    # multiprocessing使用
                    if not (rtsp_url in self._process):
                        self._image_queue[rtsp_url] = mp.Queue(maxsize=1)
                        self._request[rtsp_url] = mp.Value('i', 1)
                        self._process[rtsp_url] = mp.Process(
                            target=receive_image_process,
                            args=(rtsp_url, self._image_queue[rtsp_url],
                                  self._request[rtsp_url]),
                        )
                        self._process[rtsp_url].start()
                else:
                    # multiprocessing未使用
                    if not (rtsp_url in self._rtsp_capture):
                        rtsp_capture = cv2.VideoCapture(rtsp_url)
                        self._rtsp_capture[rtsp_url] = rtsp_capture

            dpg.set_item_label(tag_node_button_value_name, self._stop_label)
        elif label == self._stop_label:
            if rtsp_url != '':
                if use_mp:
                    # multiprocessing使用
                    if rtsp_url in self._request:
                        self._request[rtsp_url].value = 0
                        if self._process[rtsp_url].is_alive():
                            self._process[rtsp_url].terminate()
                        self._image_queue.pop(rtsp_url)
                        self._request.pop(rtsp_url)
                        self._process.pop(rtsp_url)
                else:
                    # multiprocessing未使用
                    if rtsp_url in self._rtsp_capture:
                        self._rtsp_capture[rtsp_url].release()
                        self._rtsp_capture.pop(rtsp_url)

            dpg.set_item_label(tag_node_button_value_name, self._start_label)
