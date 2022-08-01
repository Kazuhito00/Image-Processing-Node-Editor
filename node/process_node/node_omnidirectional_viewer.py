#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeABC
from node_editor.util import convert_cv_to_dpg


def image_process(image, phi, theta):
    image = remap_image(image, phi, theta)
    return image


def create_rotation_matrix(roll, pitch, yaw):
    roll = roll * np.pi / 180
    pitch = pitch * np.pi / 180
    yaw = yaw * np.pi / 180

    matrix01 = np.array([
        [1, 0, 0],
        [0, np.cos(roll), np.sin(roll)],
        [0, -np.sin(roll), np.cos(roll)],
    ])

    matrix02 = np.array([
        [np.cos(pitch), 0, -np.sin(pitch)],
        [0, 1, 0],
        [np.sin(pitch), 0, np.cos(pitch)],
    ])

    matrix03 = np.array([
        [np.cos(yaw), np.sin(yaw), 0],
        [-np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ])

    matrix = np.dot(matrix03, np.dot(matrix02, matrix01))

    return matrix


def calculate_phi_and_theta(
    viewpoint,
    imagepoint,
    sensor_width,
    sensor_height,
    output_width,
    output_height,
    rotation_matrix,
):
    width = np.arange(
        (-1) * sensor_width,
        sensor_width,
        sensor_width * 2 / output_width,
    )
    height = np.arange(
        (-1) * sensor_height,
        sensor_height,
        sensor_height * 2 / output_height,
    )

    ww, hh = np.meshgrid(width, height)

    point_distance = (imagepoint - viewpoint)
    if point_distance == 0:
        point_distance = 0.1

    a1 = ww / point_distance
    a2 = hh / point_distance
    b1 = -a1 * viewpoint
    b2 = -a2 * viewpoint

    a = 1 + (a1**2) + (a2**2)
    b = 2 * ((a1 * b1) + (a2 * b2))
    c = (b1**2) + (b2**2) - 1

    d = ((b**2) - (4 * a * c))**(1 / 2)

    x = (-b + d) / (2 * a)
    y = (a1 * x) + b1
    z = (a2 * x) + b2

    xd = rotation_matrix[0][0] * x + rotation_matrix[0][
        1] * y + rotation_matrix[0][2] * z
    yd = rotation_matrix[1][0] * x + rotation_matrix[1][
        1] * y + rotation_matrix[1][2] * z
    zd = rotation_matrix[2][0] * x + rotation_matrix[2][
        1] * y + rotation_matrix[2][2] * z

    phi = np.arcsin(zd)
    theta = np.arcsin(yd / np.cos(phi))

    xd[xd > 0] = 0
    xd[xd < 0] = 1
    yd[yd > 0] = np.pi
    yd[yd < 0] = -np.pi

    offset = yd * xd
    gain = -2 * xd + 1
    theta = gain * theta + offset

    return phi, theta


def remap_image(image, phi, theta):
    input_height, input_width = image.shape[:2]

    phi = (phi * input_height / np.pi + input_height / 2)
    phi = phi.astype(np.float32)
    theta = (theta * input_width / (2 * np.pi) + input_width / 2)
    theta = theta.astype(np.float32)

    output_image = cv2.remap(image, theta, phi, cv2.INTER_CUBIC)

    return output_image


class Node(DpgNodeABC):
    _ver = '0.0.1'

    node_label = 'Omnidirectional Viewer'
    node_tag = 'OmnidirectionalViewer'

    _min_degree = 0
    _max_degree = 359
    _min_point = -1.0
    _max_point = 3.0

    _opencv_setting_dict = None

    _params = {}
    _sensor_size = 0.561
    _output_width = 960
    _output_height = 540

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
        tag_node_input02_name = tag_node_name + ':' + self.TYPE_INT + ':Input02'
        tag_node_input02_value_name = tag_node_name + ':' + self.TYPE_INT + ':Input02Value'
        tag_node_input03_name = tag_node_name + ':' + self.TYPE_INT + ':Input03'
        tag_node_input03_value_name = tag_node_name + ':' + self.TYPE_INT + ':Input03Value'
        tag_node_input04_name = tag_node_name + ':' + self.TYPE_INT + ':Input04'
        tag_node_input04_value_name = tag_node_name + ':' + self.TYPE_INT + ':Input04Value'
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
            # ピッチ
            with dpg.node_attribute(
                    tag=tag_node_input02_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_slider_int(
                    tag=tag_node_input02_value_name,
                    label="pitch",
                    width=small_window_w - 60,
                    default_value=0,
                    min_value=self._min_degree,
                    max_value=self._max_degree,
                    callback=None,
                )
            # ヨー
            with dpg.node_attribute(
                    tag=tag_node_input03_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_slider_int(
                    tag=tag_node_input03_value_name,
                    label="yaw",
                    width=small_window_w - 60,
                    default_value=0,
                    min_value=self._min_degree,
                    max_value=self._max_degree,
                    callback=None,
                )
            # ロール
            with dpg.node_attribute(
                    tag=tag_node_input04_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_slider_int(
                    tag=tag_node_input04_value_name,
                    label="roll",
                    width=small_window_w - 60,
                    default_value=0,
                    min_value=self._min_degree,
                    max_value=self._max_degree,
                    callback=None,
                )
            # 撮像位置
            with dpg.node_attribute(
                    tag=tag_node_input05_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_slider_float(
                    tag=tag_node_input05_value_name,
                    label="image point",
                    width=small_window_w - 100,
                    default_value=0,
                    min_value=self._min_point,
                    max_value=self._max_point,
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
        input_value02_tag = tag_node_name + ':' + self.TYPE_INT + ':Input02Value'
        input_value03_tag = tag_node_name + ':' + self.TYPE_INT + ':Input03Value'
        input_value04_tag = tag_node_name + ':' + self.TYPE_INT + ':Input04Value'
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
            if connection_type == self.TYPE_INT:
                # 接続タグ取得
                source_tag = connection_info[0] + 'Value'
                destination_tag = connection_info[1] + 'Value'
                # 値更新
                input_value = int(dpg_get_value(source_tag))
                input_value = max([self._min_degree, input_value])
                input_value = min([self._max_degree, input_value])
                dpg_set_value(destination_tag, input_value)
            if connection_type == self.TYPE_IMAGE:
                # 画像取得元のノード名(ID付き)を取得
                connection_info_src = connection_info[0]
                connection_info_src = connection_info_src.split(':')[:2]
                connection_info_src = ':'.join(connection_info_src)

        # 画像取得
        frame = node_image_dict.get(connection_info_src, None)

        # ピッチ角、ヨー確、ロール確
        pitch = int(dpg_get_value(input_value02_tag))
        yaw = int(dpg_get_value(input_value03_tag))
        roll = int(dpg_get_value(input_value04_tag))
        imagepoint = float(dpg_get_value(input_value05_tag))

        change_param_flag = False
        if node_id not in self._params:
            change_param_flag = True
        else:
            prev_pitch = self._params[node_id][0]
            prev_yaw = self._params[node_id][1]
            prev_roll = self._params[node_id][2]
            prev_imagepoint = self._params[node_id][3]

            if prev_pitch != pitch:
                change_param_flag = True
            if prev_yaw != yaw:
                change_param_flag = True
            if prev_roll != roll:
                change_param_flag = True
            if prev_imagepoint != imagepoint:
                change_param_flag = True

        if change_param_flag:
            sensor_width = self._sensor_size
            sensor_height = self._sensor_size
            sensor_height *= (self._output_height / self._output_width)

            # 回転行列生成
            rotation_matrix = create_rotation_matrix(
                roll,
                pitch,
                yaw,
            )

            # 角度座標φ, θ算出
            phi, theta = calculate_phi_and_theta(
                -1.0,
                imagepoint,
                sensor_width,
                sensor_height,
                self._output_width,
                self._output_height,
                rotation_matrix,
            )

            self._params[node_id] = [pitch, yaw, roll, imagepoint, phi, theta]

        # 計測開始
        if frame is not None and use_pref_counter:
            start_time = time.perf_counter()

        if frame is not None:
            phi, theta = self._params[node_id][4], self._params[node_id][5]
            frame = image_process(frame, phi, theta)

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
        input_value02_tag = tag_node_name + ':' + self.TYPE_INT + ':Input02Value'
        input_value03_tag = tag_node_name + ':' + self.TYPE_INT + ':Input03Value'
        input_value04_tag = tag_node_name + ':' + self.TYPE_INT + ':Input04Value'
        input_value05_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Input05Value'

        pitch = dpg_get_value(input_value02_tag)
        yaw = dpg_get_value(input_value03_tag)
        roll = dpg_get_value(input_value04_tag)
        imagepoint = dpg_get_value(input_value05_tag)

        pos = dpg.get_item_pos(tag_node_name)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos
        setting_dict[input_value02_tag] = pitch
        setting_dict[input_value03_tag] = yaw
        setting_dict[input_value04_tag] = roll
        setting_dict[input_value05_tag] = imagepoint

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        tag_node_name = str(node_id) + ':' + self.node_tag
        input_value02_tag = tag_node_name + ':' + self.TYPE_INT + ':Input02Value'
        input_value03_tag = tag_node_name + ':' + self.TYPE_INT + ':Input03Value'
        input_value04_tag = tag_node_name + ':' + self.TYPE_INT + ':Input04Value'
        input_value05_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Input05Value'

        pitch = int(setting_dict[input_value02_tag])
        yaw = int(setting_dict[input_value03_tag])
        roll = int(setting_dict[input_value04_tag])
        imagepoint = float(setting_dict[input_value05_tag])

        dpg_set_value(input_value02_tag, pitch)
        dpg_set_value(input_value03_tag, yaw)
        dpg_set_value(input_value04_tag, roll)
        dpg_set_value(input_value05_tag, imagepoint)
