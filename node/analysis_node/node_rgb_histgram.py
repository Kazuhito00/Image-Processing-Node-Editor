#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeABC


class Node(DpgNodeABC):
    _ver = '0.0.1'

    node_label = 'RGB Histgram'
    node_tag = 'RGBHistgram'

    _opencv_setting_dict = None
    _yaxis_divide_value = 32

    _default_xdata = None
    _default_ydata = None

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

        # OpenCV向け設定
        self._opencv_setting_dict = opencv_setting_dict
        small_window_w = self._opencv_setting_dict['result_width']
        small_window_h = self._opencv_setting_dict['result_height']

        self._default_xdata = np.linspace(0, 256 - 1, 256)
        self._default_ydata = np.linspace(0, 100, 256)

        # ノード
        with dpg.node(
                tag=tag_node_name,
                parent=parent,
                label=self.node_label,
                pos=pos,
        ):
            # ヒストグラム
            with dpg.node_attribute(
                    tag=tag_node_input01_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                with dpg.plot(
                        width=small_window_w,
                        height=small_window_h,
                        tag=tag_node_input01_value_name,
                        no_menus=True,
                ):
                    # 凡例
                    dpg.add_plot_legend(horizontal=True,
                                        location=dpg.mvPlot_Location_NorthEast)
                    # x軸
                    dpg.add_plot_axis(
                        dpg.mvXAxis,
                        tag=tag_node_input01_value_name + 'xaxis',
                    )
                    dpg.set_axis_limits(dpg.last_item(), 0, 256)

                    # y軸
                    dpg.add_plot_axis(
                        dpg.mvYAxis,
                        tag=tag_node_input01_value_name + 'yaxis',
                    )
                    dpg.add_line_series(
                        self._default_xdata,
                        self._default_ydata,
                        label='B',
                        parent=tag_node_input01_value_name + 'yaxis',
                        tag=tag_node_input01_value_name + 'line_b',
                    )
                    dpg.add_line_series(
                        self._default_xdata,
                        self._default_ydata,
                        label='R',
                        parent=tag_node_input01_value_name + 'yaxis',
                        tag=tag_node_input01_value_name + 'line_r',
                    )
                    dpg.add_line_series(
                        self._default_xdata,
                        self._default_ydata,
                        label='G',
                        parent=tag_node_input01_value_name + 'yaxis',
                        tag=tag_node_input01_value_name + 'line_g',
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
        tag_node_input01_value_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Input01Value'

        # 画像取得元のノード名(ID付き)を取得する
        connection_info_src = ''
        for connection_info in connection_list:
            connection_info_src = connection_info[0]
            connection_info_src = connection_info_src.split(':')[:2]
            connection_info_src = ':'.join(connection_info_src)

        # 画像取得
        frame = node_image_dict.get(connection_info_src, None)

        # 各チャンネルのヒストグラム算出
        result = None
        if frame is not None:
            b_histgram = cv2.calcHist([frame], [0], None, [256], [0, 256])
            g_histgram = cv2.calcHist([frame], [1], None, [256], [0, 256])
            r_histgram = cv2.calcHist([frame], [2], None, [256], [0, 256])

            # ヒストグラム反映
            dpg_set_value(tag_node_input01_value_name + 'line_b',
                          [self._default_xdata, b_histgram.T[0]])
            dpg_set_value(tag_node_input01_value_name + 'line_g',
                          [self._default_xdata, g_histgram.T[0]])
            dpg_set_value(tag_node_input01_value_name + 'line_r',
                          [self._default_xdata, r_histgram.T[0]])
            if dpg.does_item_exist(tag_node_input01_value_name + 'yaxis'):
                dpg.set_axis_limits(
                    tag_node_input01_value_name + 'yaxis', 0,
                    int(np.sum(b_histgram.T[0]) / self._yaxis_divide_value))

            result = {}
            result['r_histgram'] = list(r_histgram.T[0])
            result['g_histgram'] = list(g_histgram.T[0])
            result['b_histgram'] = list(b_histgram.T[0])

        return frame, result

    def close(self, node_id):
        pass

    def get_setting_dict(self, node_id):
        tag_node_name = str(node_id) + ':' + self.node_tag

        pos = dpg.get_item_pos(tag_node_name)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        pass
