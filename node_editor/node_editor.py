#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import copy
import json
import platform
import datetime
from glob import glob
from collections import OrderedDict
from importlib import import_module

import dearpygui.dearpygui as dpg


class DpgNodeEditor(object):
    _ver = '0.0.1'

    _node_editor_tag = 'NodeEditor'
    _node_editor_label = 'Node editor'

    _node_id = 0
    _node_instance_list = {}
    _node_list = []
    _node_link_list = []

    _last_pos = None

    _terminate_flag = False

    _opencv_setting_dict = None

    _use_debug_print = False

    def __init__(
        self,
        width=1280,
        height=720,
        pos=[0, 0],
        opencv_setting_dict=None,
        node_dir='node',
        menu_dict=None,
        use_debug_print=False,
    ):
        # 各種初期化
        self._node_id = 0
        self._node_instance_list = {}
        self._node_list = []
        self._node_link_list = []
        self._node_connection_dict = OrderedDict([])
        self._use_debug_print = use_debug_print

        self._terminate_flag = False

        self._opencv_setting_dict = opencv_setting_dict

        # メニュー項目定義(key：メニュー名、value：ノードのコード格納ディレクトリ名)
        if menu_dict is None:
            menu_dict = OrderedDict({
                'Input Node': 'input_node',
                'Process Node': 'process_node',
                'Output Node': 'output_node'
            })

        # ファイルダイアログ設定
        datetime_now = datetime.datetime.now()
        with dpg.file_dialog(
                directory_selector=False,
                show=False,
                modal=True,
                height=int(height / 2),
                default_filename=datetime_now.strftime('%Y%m%d'),
                callback=self._callback_file_export,
                id='file_export',
        ):
            dpg.add_file_extension('.json')
            dpg.add_file_extension('', color=(150, 255, 150, 255))

        with dpg.file_dialog(
                directory_selector=False,
                show=False,
                modal=True,
                height=int(height / 2),
                callback=self._callback_file_import,
                id='file_import',
        ):
            dpg.add_file_extension('.json')
            dpg.add_file_extension('', color=(150, 255, 150, 255))

        # ノードエディター ウィンドウ生成
        with dpg.window(
                tag=self._node_editor_tag + 'Window',
                label=self._node_editor_label,
                width=width,
                height=height,
                pos=pos,
                menubar=True,
                on_close=self._callback_close_window,
        ):
            # メニューバー生成
            with dpg.menu_bar(label='MenuBar'):
                # Export/Importメニュー
                with dpg.menu(label='File'):
                    dpg.add_menu_item(
                        tag='Menu_File_Export',
                        label='Export',
                        callback=self._callback_file_export_menu,
                        user_data='Menu_File_Export',
                    )
                    dpg.add_menu_item(
                        tag='Menu_File_Import',
                        label='Import',
                        callback=self._callback_file_import_menu,
                        user_data='Menu_File_Import',
                    )

                # ノードメニュー生成
                for menu_info in menu_dict.items():
                    menu_label = menu_info[0]

                    with dpg.menu(label=menu_label):
                        # ノードのコード格納パス生成
                        node_sources_path = os.path.join(
                            node_dir,
                            menu_info[1],
                            '*.py',
                        )

                        # 指定ディレクトリ内のノードのコード一覧を取得
                        node_sources = glob(node_sources_path)
                        for node_source in node_sources:
                            # 動的インポート用のパスを生成
                            import_path = os.path.splitext(
                                os.path.normpath(node_source))[0]
                            if platform.system() == 'Windows':
                                import_path = import_path.replace('\\', '.')
                            else:
                                import_path = import_path.replace('/', '.')

                            import_path = import_path.split('.')
                            import_path = '.'.join(import_path[-3:])
                            # __init__.pyのみ除外
                            if import_path.endswith('__init__'):
                                continue
                            # モジュールを動的インポート
                            module = import_module(import_path)

                            # ノードインスタンス生成
                            node = module.Node()

                            # メニューアイテム追加
                            dpg.add_menu_item(
                                tag='Menu_' + node.node_tag,
                                label=node.node_label,
                                callback=self._callback_add_node,
                                user_data=node.node_tag,
                            )

                            # インスタンスリスト追加
                            self._node_instance_list[node.node_tag] = node

            # ノードエディター生成（初期状態はノード追加なし）
            with dpg.node_editor(
                    tag=self._node_editor_tag,
                    callback=self._callback_link,
                    minimap=True,
                    minimap_location=dpg.mvNodeMiniMap_Location_BottomRight,
            ):
                pass

            # インポート制限事項ポップアップ
            with dpg.window(
                    label='Delete Files',
                    modal=True,
                    show=False,
                    id='modal_file_import',
                    no_title_bar=True,
                    pos=[52, 52],
            ):
                dpg.add_text(
                    'Sorry. In the current implementation, \nfile import works only before adding a node.',
                )
                dpg.add_separator()
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label='OK',
                        width=375,
                        callback=lambda: dpg.configure_item(
                            'modal_file_import',
                            show=False,
                        ),
                    )

            # マウス・キーボードコールバック登録
            with dpg.handler_registry():
                dpg.add_mouse_click_handler(
                    callback=self._callback_save_last_pos)
                dpg.add_key_press_handler(
                    dpg.mvKey_Delete,
                    callback=self._callback_mv_key_del,
                )

    def get_node_list(self):
        return self._node_list

    def get_sorted_node_connection(self):
        return self._node_connection_dict

    def get_node_instance(self, node_name):
        return self._node_instance_list.get(node_name, None)

    def set_terminate_flag(self, flag=True):
        self._terminate_flag = flag

    def get_terminate_flag(self):
        return self._terminate_flag

    def _callback_add_node(self, sender, data, user_data):
        self._node_id += 1

        # ノードインスタンス取得
        node = self._node_instance_list[user_data]

        # ノードエディターにノードを追加
        last_pos = [0, 0]
        if self._last_pos is not None:
            last_pos = [self._last_pos[0] + 30, self._last_pos[1] + 30]
        tag_name = node.add_node(
            self._node_editor_tag,
            self._node_id,
            pos=last_pos,
            opencv_setting_dict=self._opencv_setting_dict,
        )

        self._node_list.append(tag_name)

        if self._use_debug_print:
            print('**** _callback_add_node ****')
            print('    Node ID         : ' + str(self._node_id))
            print('    sender          : ' + str(sender))
            print('    data            : ' + str(data))
            print('    user_data       : ' + str(user_data))
            print('    self._node_list : ' + ', '.join(self._node_list))
            print()

    def _callback_link(self, sender, data):
        # 各接続子の型を取得
        source = dpg.get_item_alias(data[0])
        destination = dpg.get_item_alias(data[1])
        source_type = source.split(':')[2]
        destination_type = destination.split(':')[2]

        # 型が一致するもののみ処理
        if source_type == destination_type:
            # 初回ノード生成時
            if len(self._node_link_list) == 0:
                dpg.add_node_link(source, destination, parent=sender)
                self._node_link_list.append([source, destination])
            # 2回目以降
            else:
                # 入力端子に複数接続しようとしていないかチェック
                duplicate_flag = False
                for node_link in self._node_link_list:
                    if destination == node_link[1]:
                        duplicate_flag = True
                if not duplicate_flag:
                    dpg.add_node_link(source, destination, parent=sender)
                    self._node_link_list.append([source, destination])

        # ノードグラフ再生成
        self._node_connection_dict = self._sort_node_graph(
            self._node_list,
            self._node_link_list,
        )

        if self._use_debug_print:
            print('**** _callback_link ****')
            print('    sender                     : ' + str(sender))
            print('    data                       : ', data)
            print('    self._node_list            :    ', self._node_list)
            print('    self._node_link_list       : ', self._node_link_list)
            print('    self._node_connection_dict : ',
                  self._node_connection_dict)
            print()


    def _callback_close_window(self, sender):
        dpg.delete_item(sender)

    def _sort_node_graph(self, node_list, node_link_list):
        node_id_dict = OrderedDict({})
        node_connection_dict = OrderedDict({})

        # ノードIDとノード接続を辞書形式で整理
        for node_link_info in node_link_list:
            source = dpg.get_item_alias(node_link_info[0])
            destination = dpg.get_item_alias(node_link_info[1])
            source_id = int(source.split(':')[0])
            destination_id = int(destination.split(':')[0])

            if destination_id not in node_id_dict:
                node_id_dict[destination_id] = [source_id]
            else:
                node_id_dict[destination_id].append(source_id)

            split_destination = destination.split(':')

            node_name = split_destination[0] + ':' + split_destination[1]
            if node_name not in node_connection_dict:
                node_connection_dict[node_name] = [[source, destination]]
            else:
                node_connection_dict[node_name].append([source, destination])

        node_id_list = list(node_id_dict.items())
        node_connection_list = list(node_connection_dict.items())

        # 入力から出力に向かって処理順序を入れ替える
        index = 0
        while index < len(node_id_list):
            swap_flag = False
            for check_id in node_id_list[index][1]:
                for check_index in range(index + 1, len(node_id_list)):
                    if node_id_list[check_index][0] == check_id:
                        node_id_list[check_index], node_id_list[
                            index] = node_id_list[index], node_id_list[
                                check_index]
                        node_connection_list[
                            check_index], node_connection_list[
                                index] = node_connection_list[
                                    index], node_connection_list[check_index]

                        swap_flag = True
                        break
            if not swap_flag:
                index += 1

        # 接続リストに登場しないノードを追加する(入力ノード等)
        index = 0
        unfinded_id_dict = {}
        while index < len(node_id_list):
            for check_id in node_id_list[index][1]:
                check_index = 0
                find_flag = False
                while check_index < len(node_id_list):
                    if check_id == node_id_list[check_index][0]:
                        find_flag = True
                        break
                    check_index += 1
                if not find_flag:
                    for index, node_id_name in enumerate(node_list):
                        node_id, node_name = node_id_name.split(':')
                        if node_id == check_id:
                            unfinded_id_dict[check_id] = node_id_name
                            break
            index += 1

        for unfinded_value in unfinded_id_dict.values():
            node_connection_list.insert(0, (unfinded_value, []))

        return OrderedDict(node_connection_list)

    def _callback_file_export(self, sender, data):
        setting_dict = {}

        # ノードリスト、接続リスト保存
        setting_dict['node_list'] = self._node_list
        setting_dict['link_list'] = self._node_link_list

        # 各ノードの設定値保存
        for node_id_name in self._node_list:
            node_id, node_name = node_id_name.split(':')
            node = self._node_instance_list[node_name]

            setting = node.get_setting_dict(node_id)

            setting_dict[node_id_name] = {
                'id': str(node_id),
                'name': str(node_name),
                'setting': setting
            }

        # JSONファイルへ書き出し
        with open(data['file_path_name'], 'w') as fp:
            json.dump(setting_dict, fp, indent=4)

        if self._use_debug_print:
            print('**** _callback_file_export ****')
            print('    sender          : ' + str(sender))
            print('    data            : ' + str(data))
            print('    setting_dict    : ', setting_dict)
            print()

    def _callback_file_export_menu(self):
        dpg.show_item('file_export')

    def _callback_file_import_menu(self):
        if self._node_id == 0:
            dpg.show_item('file_import')
        else:
            dpg.configure_item('modal_file_import', show=True)

    def _callback_file_import(self, sender, data):
        if data['file_name'] != '.':
            # JSONファイルから読み込み
            setting_dict = None
            with open(data['file_path_name']) as fp:
                setting_dict = json.load(fp)

            # 各ノードの設定値復元
            for node_id_name in setting_dict['node_list']:
                node_id, node_name = node_id_name.split(':')
                node = self._node_instance_list[node_name]

                node_id = int(node_id)

                if node_id > self._node_id:
                    self._node_id = node_id

                # ノードインスタンス取得
                node = self._node_instance_list[node_name]

                # バージョン警告
                ver = setting_dict[node_id_name]['setting']['ver']
                if ver != node._ver:
                    warning_node_name = setting_dict[node_id_name]['name']
                    print('WARNING : ' + warning_node_name, end='')
                    print(' is different version')
                    print('                     Load Version ->' + ver)
                    print('                     Code Version ->' + node._ver)
                    print()

                # ノードエディターにノードを追加
                pos = setting_dict[node_id_name]['setting']['pos']
                node.add_node(
                    self._node_editor_tag,
                    node_id,
                    pos=pos,
                    opencv_setting_dict=self._opencv_setting_dict,
                )

                # 設定値復元
                node.set_setting_dict(
                    node_id,
                    setting_dict[node_id_name]['setting'],
                )

            # ノードリスト、接続リスト復元
            self._node_list = setting_dict['node_list']
            self._node_link_list = setting_dict['link_list']

            # ノード接続復元
            for node_link in self._node_link_list:
                dpg.add_node_link(
                    node_link[0],
                    node_link[1],
                    parent=self._node_editor_tag,
                )

            # ノードグラフ再生成
            self._node_connection_dict = self._sort_node_graph(
                self._node_list,
                self._node_link_list,
            )

        if self._use_debug_print:
            print('**** _callback_file_import ****')
            print('    sender          : ' + str(sender))
            print('    data            : ' + str(data))
            print('    setting_dict    : ', setting_dict)
            print()

    def _callback_save_last_pos(self):
        if len(dpg.get_selected_nodes(self._node_editor_tag)) > 0:
            self._last_pos = dpg.get_item_pos(
                dpg.get_selected_nodes(self._node_editor_tag)[0])

    def _callback_mv_key_del(self):
        if len(dpg.get_selected_nodes(self._node_editor_tag)) > 0:
            # 選択中のノードのアイテムIDを取得
            item_id = dpg.get_selected_nodes(self._node_editor_tag)[0]
            # ノード名を特定
            node_id_name = dpg.get_item_alias(item_id)
            node_id, node_name = node_id_name.split(':')

            if node_name != 'ExecPythonCode':
                # ノード終了処理
                node_instance = self.get_node_instance(node_name)
                node_instance.close(node_id)
                # ノードリストから削除
                self._node_list.remove(node_id_name)
                # ノードリンクリストから削除
                copy_node_link_list = copy.deepcopy(self._node_link_list)
                for link_info in copy_node_link_list:
                    source_node = link_info[0].split(':')[:2]
                    source_node = ':'.join(source_node)
                    destination_node = link_info[1].split(':')[:2]
                    destination_node = ':'.join(destination_node)

                    if source_node == node_id_name or destination_node == node_id_name:
                        self._node_link_list.remove(link_info)

                # ノードグラフ再生成
                self._node_connection_dict = self._sort_node_graph(
                    self._node_list,
                    self._node_link_list,
                )

                # アイテム削除
                dpg.delete_item(item_id)

        if len(dpg.get_selected_links(self._node_editor_tag)) > 0:
            self._node_link_list.remove([
                dpg.get_item_alias(dpg.get_item_configuration(dpg.get_selected_links(self._node_editor_tag)[0])['attr_1']),
                dpg.get_item_alias(dpg.get_item_configuration(dpg.get_selected_links(self._node_editor_tag)[0])['attr_2'])
            ])

            self._node_connection_dict = self._sort_node_graph(
                self._node_list,
                self._node_link_list,
            )

            dpg.delete_item(dpg.get_selected_links(self._node_editor_tag)[0])

        if self._use_debug_print:
            print('**** _callback_mv_key_del ****')
            print('    self._node_list            :    ', self._node_list)
            print('    self._node_link_list       : ', self._node_link_list)
            print('    self._node_connection_dict : ',
                  self._node_connection_dict)
