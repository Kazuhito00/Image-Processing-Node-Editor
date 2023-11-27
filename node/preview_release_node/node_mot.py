#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time

import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeABC
from node_editor.util import convert_cv_to_dpg

from node.preview_release_node.mot.motpy.motpy import Motpy
# from node.preview_release_node.mot.bytetrack.mc_bytetrack import MultiClassByteTrack
# from node.preview_release_node.mot.norfair.mc_norfair import MultiClassNorfair

from node.draw_node.draw_util.draw_util import draw_multi_object_tracking_info


class Node(DpgNodeABC):
    _ver = '0.0.1'

    node_label = 'MOT(Preview Release Version)'
    node_tag = 'MultiObjectTracking'

    _opencv_setting_dict = None

    # モデル設定
    _model_class = {
        'motpy': Motpy,
        # 'ByteTrack': MultiClassByteTrack,
        # 'Norfair': MultiClassNorfair,
    }

    _model_instance = {}
    _class_name_dict = None
    _track_id_dict = {}

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
                    default_value='Input Detection Node',
                )
            # 画像
            with dpg.node_attribute(
                    tag=tag_node_output01_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_image(tag_node_output01_value_name)
            # 使用アルゴリズム
            with dpg.node_attribute(
                    tag=tag_node_input02_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_combo(
                    list(self._model_class.keys()),
                    default_value=list(self._model_class.keys())[0],
                    width=small_window_w,
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
        src_node_name = ''
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
                src_node_name = connection_info_src[1]
                connection_info_src = ':'.join(connection_info_src)

        # 画像取得
        frame = node_image_dict.get(connection_info_src, None)

        # モデル情報取得
        model_name = dpg_get_value(input_value02_tag)
        model_class = self._model_class[model_name]

        model_name_with_provider = tag_node_name + ':' + model_name

        # モデル取得
        if frame is not None:
            if model_name_with_provider not in self._model_instance:
                # ToDo：FPS初期指定未実施(デフォルト30)
                self._model_instance[model_name_with_provider] = model_class()

        # 計測開始
        if frame is not None and use_pref_counter:
            start_time = time.perf_counter()

        # 接続元がObjectDetectionノードの場合、各バウンディングボックスに対して推論
        result = {}
        if frame is not None:
            if src_node_name == 'ObjectDetection':
                # 物体検出情報取得
                node_result = node_result_dict.get(connection_info_src, [])
                od_bboxes = node_result.get('bboxes', [])
                od_scores = node_result.get('scores', [])
                od_class_ids = node_result.get('class_ids', [])
                od_class_names = node_result.get('class_names', [])

                track_ids, t_bboxes, t_scores, t_class_ids = [], [], [], []
                track_ids, t_bboxes, t_scores, t_class_ids = self._model_instance[
                    model_name_with_provider](
                        frame,
                        od_bboxes,
                        od_scores,
                        od_class_ids,
                    )

                if node_id not in self._track_id_dict:
                    self._track_id_dict[node_id] = {}

                # トラッキングIDと連番の紐付け
                for track_id in track_ids:
                    if track_id not in self._track_id_dict[node_id]:
                        new_id = len(self._track_id_dict[node_id])
                        self._track_id_dict[node_id][track_id] = new_id

                result['track_ids'] = track_ids
                result['bboxes'] = t_bboxes
                result['scores'] = t_scores
                result['class_ids'] = t_class_ids
                result['class_names'] = od_class_names
                result['track_id_dict'] = self._track_id_dict[node_id]

            elif src_node_name == 'Classification':
                node_result = node_result_dict.get(connection_info_src, [])
                use_object_detection = node_result.get(
                    'use_object_detection',
                    False,
                )
                if use_object_detection:
                    # 物体検出情報取得
                    od_bboxes = node_result.get('od_bboxes', [])
                    od_scores = node_result.get('class_scores', [])
                    od_class_ids = node_result.get('class_ids', [])
                    od_class_names = node_result.get('class_names', [])

                    track_ids, t_bboxes, t_scores, t_class_ids = self._model_instance[
                        model_name_with_provider](
                            frame,
                            od_bboxes,
                            od_scores,
                            od_class_ids,
                        )

                    if node_id not in self._track_id_dict:
                        self._track_id_dict[node_id] = {}

                    # トラッキングIDと連番の紐付け
                    for track_id in track_ids:
                        if track_id not in self._track_id_dict[node_id]:
                            new_id = len(self._track_id_dict[node_id])
                            self._track_id_dict[node_id][track_id] = new_id

                    result['track_ids'] = track_ids
                    result['bboxes'] = t_bboxes
                    result['scores'] = t_scores
                    result['class_ids'] = t_class_ids
                    result['class_names'] = od_class_names
                    result['track_id_dict'] = self._track_id_dict[node_id]

        # 計測終了
        if frame is not None and use_pref_counter:
            elapsed_time = time.perf_counter() - start_time
            elapsed_time = int(elapsed_time * 1000)
            dpg_set_value(output_value02_tag,
                          str(elapsed_time).zfill(4) + 'ms')

        # 描画
        if frame is not None:
            if src_node_name == 'ObjectDetection' or src_node_name == 'Classification':
                # 描画
                debug_frame = copy.deepcopy(frame)
                debug_frame = draw_multi_object_tracking_info(
                    debug_frame,
                    track_ids,
                    t_bboxes,
                    t_scores,
                    t_class_ids,
                    od_class_names,
                    self._track_id_dict[node_id],
                )
            else:
                debug_frame = np.zeros((small_window_w, small_window_h, 3))
            texture = convert_cv_to_dpg(
                debug_frame,
                small_window_w,
                small_window_h,
            )
            dpg_set_value(output_value01_tag, texture)

        return frame, result

    def close(self, node_id):
        pass

    def get_setting_dict(self, node_id):
        tag_node_name = str(node_id) + ':' + self.node_tag
        input_value02_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'

        # 選択モデル
        model_name = dpg_get_value(input_value02_tag)

        pos = dpg.get_item_pos(tag_node_name)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos
        setting_dict[input_value02_tag] = model_name

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        tag_node_name = str(node_id) + ':' + self.node_tag
        input_value02_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'

        model_name = setting_dict[input_value02_tag]

        dpg_set_value(input_value02_tag, model_name)
