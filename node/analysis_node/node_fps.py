#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
from collections import deque

import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeABC


class Node(DpgNodeABC):
    _ver = '0.0.1'

    node_label = 'FPS'
    node_tag = 'FPS'

    _opencv_setting_dict = None

    _buffer_len = 10
    _value_history = None

    _max_slot_number = 10
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
        tag_node_input00_name = tag_node_name + ':' + self.TYPE_TIME_MS + ':Input00'
        tag_node_input01_name = tag_node_name + ':' + self.TYPE_TIME_MS + ':Input01'
        tag_node_input01_value_name = tag_node_name + ':' + self.TYPE_TIME_MS + ':Input01Value'
        tag_node_output01_name = tag_node_name + ':' + self.TYPE_TEXT + ':Output01'
        tag_node_output01_value_name = tag_node_name + ':' + self.TYPE_TEXT + ':Output01Value'
        tag_node_output02_name = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02'
        tag_node_output02_value_name = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02Value'

        # OpenCV向け設定
        self._opencv_setting_dict = opencv_setting_dict
        small_window_w = self._opencv_setting_dict['result_width']

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
            # FPS表示
            with dpg.node_attribute(
                    tag=tag_node_output01_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_text(
                    tag=tag_node_output01_value_name,
                    default_value='FPS:',
                )
            # 合計時間表示
            with dpg.node_attribute(
                    tag=tag_node_output02_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_text(
                    tag=tag_node_output02_value_name,
                    default_value='Total time(ms)',
                )
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
        output_value01_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Output01Value'
        output_value02_tag = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02Value'

        total_elapsed_time = 0

        # 画像取得元のノード名(ID付き)を取得する
        for connection_info in connection_list:
            connection_type = connection_info[0].split(':')[2]
            if connection_type == self.TYPE_TIME_MS:
                # 接続タグ取得
                source_tag = connection_info[0] + 'Value'
                destination_tag = connection_info[1] + 'Value'

                # 値更新
                input_value = dpg_get_value(source_tag)

                # 数値のみを抽出
                input_value = re.sub(r'\D', '', input_value)
                if input_value != '':
                    input_value = int(input_value)

                    # 取得した経過時間をキューに追加
                    if source_tag not in self._value_history:
                        self._value_history[source_tag] = deque(
                            maxlen=self._buffer_len)
                        self._value_history[source_tag].append(input_value)
                    else:
                        self._value_history[source_tag].append(input_value)

                    # 平均処理時間
                    average_elapsed_time = sum(
                        self._value_history[source_tag]) / len(
                            self._value_history[source_tag])

                    # FPS算出
                    fps = 0
                    if average_elapsed_time > 0:
                        fps = 1000.0 / average_elapsed_time

                    # 表示テキスト生成
                    text = 'FPS:'
                    if fps > 1:
                        fps = int(fps)
                        text += '{:.0f}'.format(fps).zfill(3)
                    else:
                        text += '{:.2f}'.format(fps).zfill(3)
                    text += ' (' + '{:.0f}'.format(input_value).zfill(
                        4) + 'ms)'

                    # テキスト更新
                    dpg_set_value(destination_tag, text)

                    # 全スロットの合計時間
                    total_elapsed_time += average_elapsed_time

        # 全スロットの合計時間のFPS算出
        if total_elapsed_time > 0:
            fps = 1000.0 / total_elapsed_time
            text = 'FPS:'
            if fps > 1:
                fps = int(fps)
                text += '{:.0f}'.format(fps).zfill(3)
            else:
                text += '{:.2f}'.format(fps).zfill(3)
            # text += ' (' + '{:.0f}'.format(total_elapsed_time).zfill(4) + 'ms)'

            dpg_set_value(output_value01_tag, text)
            dpg_set_value(
                output_value02_tag,
                '{:.0f}'.format(total_elapsed_time).zfill(4) + 'ms',
            )

        return None, None

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
            before_tag = tag_node_name + ':' + self.TYPE_TIME_MS + ':Input'
            before_tag += str(self._slot_id[tag_node_name] - 1).zfill(2)

            # 追加スロットのタグを生成
            tag_node_inputXX_name = tag_node_name + ':' + self.TYPE_TIME_MS + ':Input'
            tag_node_inputXX_name += str(self._slot_id[tag_node_name]).zfill(2)

            tag_node_inputXX_value_name = tag_node_name + ':' + self.TYPE_TIME_MS + ':Input'
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
                    default_value='elapsed time(ms)',
                )
