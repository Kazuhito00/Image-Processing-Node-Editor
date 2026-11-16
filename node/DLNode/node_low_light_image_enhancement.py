#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import os

import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value, run_unlocked

from node.node_abc import DpgNodeABC
#from node_editor.util import convert_cv_to_dpg
from node.basenode import Node
from node.DLNode.low_light_image_enhancement.TBEFN.tbefn import TBEFN
from node.DLNode.low_light_image_enhancement.SCI.sci import SCI
from node.DLNode.low_light_image_enhancement.AGLLNet.agllnet import AGLLNet

class FactoryNode:
    node_label = 'LLIE'
    node_tag = 'LLIE'
    

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


        node = Node()
        node.tag_node_name = str(node_id) + ':' + node.node_tag
        node.tag_node_input01_name = node.tag_node_name + ':' + node.TYPE_IMAGE + ':Input01'
        node.tag_node_input01_value_name = node.tag_node_name + ':' + node.TYPE_IMAGE + ':Input01Value'
        node.tag_node_input02_name = node.tag_node_name + ':' + node.TYPE_TEXT + ':Input02'
        node.tag_node_input02_value_name = node.tag_node_name + ':' + node.TYPE_TEXT + ':Input02Value'
        node.tag_node_output01_name = node.tag_node_name + ':' + node.TYPE_IMAGE + ':Output01'
        node.tag_node_output01_value_name = node.tag_node_name + ':' + node.TYPE_IMAGE + ':Output01Value'
        node.tag_node_output02_name = node.tag_node_name + ':' + node.TYPE_TIME_MS + ':Output02'
        node.tag_node_output02_value_name = node.tag_node_name + ':' + node.TYPE_TIME_MS + ':Output02Value'

        node.tag_provider_select_name = node.tag_node_name + ':' + node.TYPE_TEXT + ':Provider'
        node.tag_provider_select_value_name = node.tag_node_name + ':' + node.TYPE_IMAGE + ':ProviderValue'


        node._opencv_setting_dict = opencv_setting_dict
        node.small_window_w = node._opencv_setting_dict['process_width']
        node.small_window_h = node._opencv_setting_dict['process_height']
        use_pref_counter = node._opencv_setting_dict['use_pref_counter']
        use_gpu = node._opencv_setting_dict['use_gpu']


        black_image = np.zeros((node.small_window_w, node.small_window_h, 3))
        black_texture = node.convert_cv_to_dpg(
            black_image,
            node.small_window_w,
            node.small_window_h,
        )


        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                node.small_window_w,
                node.small_window_h,
                black_texture,
                tag=node.tag_node_output01_value_name,
                format=dpg.mvFormat_Float_rgb,
            )


        with dpg.node(
                tag=node.tag_node_name,
                parent=parent,
                label=node.node_label,
                pos=pos,
        ):

            with dpg.node_attribute(
                    tag=node.tag_node_input01_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_text(
                    tag=node.tag_node_input01_value_name,
                    default_value='Input BGR image',
                )

            with dpg.node_attribute(
                    tag=node.tag_node_output01_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_image(node.tag_node_output01_value_name)

            with dpg.node_attribute(
                    tag=node.tag_node_input02_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_combo(
                    list(node._model_class.keys()),
                    default_value=list(node._model_class.keys())[0],
                    width=node.small_window_w,
                    tag=node.tag_node_input02_value_name,
                )
            if use_gpu:

                with dpg.node_attribute(
                        tag=node.tag_provider_select_name,
                        attribute_type=dpg.mvNode_Attr_Static,
                ):
                    dpg.add_radio_button(
                        ("cpu", "cuda", "TRTcuda"),
                        tag=node.tag_provider_select_value_name,
                        default_value='cpu',
                        horizontal=True,
                    )

            if use_pref_counter:
                with dpg.node_attribute(
                        tag=node.tag_node_output02_name,
                        attribute_type=dpg.mvNode_Attr_Output,
                ):
                    dpg.add_text(
                        tag=node.tag_node_output02_value_name,
                        default_value='elapsed time(ms)',
                    )

        return node


class Node(Node):
    _ver = '0.0.1'

    node_label = 'Low-Light Image Enhancement'
    node_tag = 'LLIE'

    _opencv_setting_dict = None

    _model_class = {
        'TBEFN(320x180)': TBEFN,
        'TBEFN(640x360)': TBEFN,
        'SCI(320x180)': SCI,
        'SCI(640x360)': SCI,
        'AGLLNet(256x256)': AGLLNet,
        'AGLLNet(512x384)': AGLLNet,
    }
    _model_base_path = os.path.dirname(os.path.abspath(__file__)) + '/low_light_image_enhancement/'
    _model_path_setting = {
        'TBEFN(320x180)':
        _model_base_path + 'TBEFN/saved_model_180x320/model_float32.onnx',
        'TBEFN(640x360)':
        _model_base_path + 'TBEFN/saved_model_360x640/model_float32.onnx',
        'SCI(320x180)':
        _model_base_path + 'SCI/sci_180x320/sci_180x320.onnx',
        'SCI(640x360)':
        _model_base_path + 'SCI/sci_360x640/sci_360x640.onnx',
        'AGLLNet(256x256)':
        _model_base_path + 'AGLLNet/saved_model_256x256/model_float32.onnx',
        'AGLLNet(512x384)':
        _model_base_path + 'AGLLNet/saved_model_384x512/model_float32.onnx',
    }

    _model_instance = {}

    def __init__(self):
        pass



    def update(
        self,
        node_id,
        connection_list,
        node_image_dict,
        node_result_dict,
        node_audio_dict,
    ):
        tag_node_name = str(node_id) + ':' + self.node_tag
        input_value02_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'
        output_value01_tag = tag_node_name + ':' + self.TYPE_IMAGE + ':Output01Value'
        output_value02_tag = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02Value'

        tag_provider_select_value_name = tag_node_name + ':' + self.TYPE_IMAGE + ':ProviderValue'

        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']
        use_gpu = self._opencv_setting_dict['use_gpu']


        connection_info_src = ''
        for connection_info in connection_list:
            connection_type = connection_info[0].split(':')[2]
            if connection_type == self.TYPE_INT:

                source_tag = connection_info[0] + 'Value'
                destination_tag = connection_info[1] + 'Value'

                input_value = int(dpg_get_value(source_tag))
                input_value = max([self._min_val, input_value])
                input_value = min([self._max_val, input_value])
                dpg_set_value(destination_tag, input_value)
            if connection_type == self.TYPE_IMAGE:

                connection_info_src = connection_info[0]
                connection_info_src = connection_info_src.split(':')[:2]
                connection_info_src = ':'.join(connection_info_src)


        frame = node_image_dict.get(connection_info_src, None)


        provider = 'cpu'
        if use_gpu:
        	provider = dpg_get_value(tag_provider_select_value_name)


        model_name = dpg_get_value(input_value02_tag)
        model_path = self._model_path_setting[model_name]
        model_class = self._model_class[model_name]

        model_name_with_provider = model_name + '_' + provider


        if frame is not None:
            if model_name_with_provider not in self._model_instance:
                if provider == 'cpu':
                    providers = ['CPUExecutionProvider']
                    self._model_instance[
                        model_name_with_provider] = model_class(
                            model_path,
                            providers=providers,
                        )
                elif provider == 'TRTcuda':
                    self._model_instance[
                        model_name_with_provider] = model_class(
                            model_path,
                            providers=['TensorrtExecutionProvider',
                                       'CUDAExecutionProvider',
                                       'CPUExecutionProvider'],
                        )
                else:
                    self._model_instance[
                        model_name_with_provider] = model_class(model_path)


        if frame is not None and use_pref_counter:
            start_time = time.monotonic()

        if frame is not None:
            frame = run_unlocked(
                self._model_instance[model_name_with_provider], frame
            )


        if frame is not None and use_pref_counter:
            elapsed_time = time.monotonic() - start_time
            elapsed_time = int(elapsed_time * 1000)
            dpg_set_value(output_value02_tag,
                          str(elapsed_time).zfill(4) + 'ms')


        if frame is not None:
            texture = self.convert_cv_to_dpg(
                frame,
                small_window_w,
                small_window_h,
            )
            dpg_set_value(output_value01_tag, texture)

        return {"image": frame, "json": None, "audio": None}

    def close(self, node_id):
        super().close(node_id)

    def get_setting_dict(self, node_id):
        tag_node_name = str(node_id) + ':' + self.node_tag
        input_value02_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'


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
