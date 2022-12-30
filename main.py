#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import copy
import json
import asyncio
import argparse
from collections import OrderedDict
import os

import cv2
import dearpygui.dearpygui as dpg

try:
    from .node_editor.util import check_camera_connection
    from .node_editor.node_editor import DpgNodeEditor
except ImportError:
    from node_editor.util import check_camera_connection
    from node_editor.node_editor import DpgNodeEditor


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--setting",
        type=str,
        # get abs
        default=os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                         'node_editor/setting/setting.json')),
    )
    parser.add_argument("--unuse_async_draw", action="store_true")
    parser.add_argument("--use_debug_print", action="store_true")

    args = parser.parse_args()

    return args


def async_main(node_editor):
    # 各ノードの処理結果保持用Dict
    node_image_dict = {}
    node_result_dict = {}

    # メインループ
    while not node_editor.get_terminate_flag():
        update_node_info(node_editor, node_image_dict, node_result_dict)


def update_node_info(
    node_editor,
    node_image_dict,
    node_result_dict,
    mode_async=True,
):
    # ノードリスト取得
    node_list = node_editor.get_node_list()

    # ノード接続情報取得
    sorted_node_connection_dict = node_editor.get_sorted_node_connection()

    # 各ノードの情報をアップデート
    for node_id_name in node_list:
        if node_id_name not in node_image_dict:
            node_image_dict[node_id_name] = None

        node_id, node_name = node_id_name.split(':')
        connection_list = sorted_node_connection_dict.get(node_id_name, [])

        # ノード名からインスタンスを取得
        node_instance = node_editor.get_node_instance(node_name)

        # 指定ノードの情報を更新
        if mode_async:
            try:
                image, result = node_instance.update(
                    node_id,
                    connection_list,
                    node_image_dict,
                    node_result_dict,
                )
            except Exception as e:
                print(e)
                sys.exit()
        else:
            image, result = node_instance.update(
                node_id,
                connection_list,
                node_image_dict,
                node_result_dict,
            )
        node_image_dict[node_id_name] = copy.deepcopy(image)
        node_result_dict[node_id_name] = copy.deepcopy(result)


def main():

    args = get_args()
    setting = args.setting
    unuse_async_draw = args.unuse_async_draw
    use_debug_print = args.use_debug_print

    # 動作設定
    print('**** Load Config ********')
    opencv_setting_dict = None
    with open(setting) as fp:
        opencv_setting_dict = json.load(fp)
    webcam_width = opencv_setting_dict['webcam_width']
    webcam_height = opencv_setting_dict['webcam_height']

    # 接続カメラチェック
    print('**** Check Camera Connection ********')
    device_no_list = check_camera_connection()
    camera_capture_list = []
    for device_no in device_no_list:
        video_capture = cv2.VideoCapture(device_no)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)
        camera_capture_list.append(video_capture)

    # カメラ設定保持
    opencv_setting_dict['device_no_list'] = device_no_list
    opencv_setting_dict['camera_capture_list'] = camera_capture_list

    # DearPyGui準備(コンテキスト生成、セットアップ、ビューポート生成)
    editor_width = opencv_setting_dict['editor_width']
    editor_height = opencv_setting_dict['editor_height']

    # Serial接続デバイスチェック
    serial_device_no_list = []
    serial_connection_list = []
    use_serial = opencv_setting_dict['use_serial']
    if use_serial == True:
        import serial
        try:
            from .node_editor.util import check_serial_connection
        except:
            from node_editor.util import check_serial_connection
        print('**** Check Serial Device Connection ********')
        serial_device_no_list = check_serial_connection()
        for serial_device_no in serial_device_no_list:
            ser = serial.Serial(serial_device_no,115200)
            serial_connection_list.append(ser)
        
    # Serial接続デバイス設定保持
    opencv_setting_dict['serial_device_no_list'] = serial_device_no_list
    opencv_setting_dict['serial_connection_list'] = serial_connection_list

    print('**** DearPyGui Setup ********')
    dpg.create_context()
    dpg.setup_dearpygui()
    dpg.create_viewport(
        title="Image Processing Node Editor",
        width=editor_width,
        height=editor_height,
    )

    # デフォルトフォント変更
    # このファイルのパスを取得
    current_path = os.path.dirname(os.path.abspath(__file__))
    with dpg.font_registry():
        with dpg.font(
                current_path +
                '/node_editor/font/YasashisaAntiqueFont/07YasashisaAntique.otf',
                16,
        ) as default_font:
            dpg.add_font_range_hint(dpg.mvFontRangeHint_Japanese)
    dpg.bind_font(default_font)

    # ノードエディター生成
    print('**** Create NodeEditor ********')
    menu_dict = OrderedDict({
        'InputNode': 'input_node',
        'ProcessNode': 'process_node',
        'DeepLearningNode': 'deep_learning_node',
        'AnalysisNode': 'analysis_node',
        'DrawNode': 'draw_node',
        'OtherNode': 'other_node',
        'PreviewReleaseNode': 'preview_release_node'
    })
    # print
    node_editor = DpgNodeEditor(
        width=editor_width - 15,
        height=editor_height - 40,
        opencv_setting_dict=opencv_setting_dict,
        menu_dict=menu_dict,
        use_debug_print=use_debug_print,
        node_dir=current_path + '/node',
    )

    # ビューポート表示
    dpg.show_viewport()

    # メインループ
    print('**** Start Main Event Loop ********')
    if not unuse_async_draw:
        event_loop = asyncio.get_event_loop()
        event_loop.run_in_executor(None, async_main, node_editor)
        dpg.start_dearpygui()
    else:
        # 各ノードの処理結果保持用Dict
        node_image_dict = {}
        node_result_dict = {}
        while dpg.is_dearpygui_running():
            update_node_info(
                node_editor,
                node_image_dict,
                node_result_dict,
                mode_async=False,
            )
            dpg.render_dearpygui_frame()

    # 終了処理
    print('**** Terminate process ********')
    # 各ノードの終了処理
    print('**** Close All Node ********')
    node_list = node_editor.get_node_list()
    for node_id_name in node_list:
        node_id, node_name = node_id_name.split(':')
        node_instance = node_editor.get_node_instance(node_name)
        node_instance.close(node_id)
    # OpenCV関連終了処理
    print('**** Release All VideoCapture ********')
    for camera_capture in camera_capture_list:
        camera_capture.release()
    # イベントループの停止
    print('**** Stop Event Loop ********')
    node_editor.set_terminate_flag()
    event_loop.stop()
    # DearPyGuiコンテキスト破棄
    print('**** Destroy DearPyGui Context ********')
    dpg.destroy_context()


if __name__ == '__main__':
    main()
