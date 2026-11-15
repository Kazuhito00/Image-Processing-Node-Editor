#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import copy
import json
import platform
import datetime
import threading
from glob import glob
from collections import OrderedDict
from importlib import import_module

import dearpygui.dearpygui as dpg
from node.node_factory import NodeFactory
import time
from node_editor.style import STYLE
from node_editor.util import (  # Import shared DearPyGUI lock
    _dpg_lock,
    dpg_delete_item,
    schedule_deferred_delete,
)
from src.utils.logging import get_logger

# Uptime tracking
_start_time = time.time()
_last_uptime_state = None

dpg.create_context()

logger = get_logger(__name__)


def _is_ctrl_down():
    """Return True if either Left-Control or Right-Control is currently held.

    DearPyGui 2.x removed the generic ``mvKey_Control`` constant and replaced
    it with ``mvKey_LControl`` / ``mvKey_RControl``.  We try the new constants
    first and fall back to the legacy one so the code works across versions.
    """
    try:
        return dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)
    except AttributeError:
        return dpg.is_key_down(dpg.mvKey_Control)


def update_uptime_display():
    """Update the uptime text in the menu bar (far right)."""
    global _last_uptime_state
    elapsed = int(time.time() - _start_time)
    try:
        vp_width = dpg.get_viewport_client_width()
        # Called on every pipeline iteration: only touch the widgets when the
        # displayed second or the viewport width actually changed.
        if (elapsed, vp_width) == _last_uptime_state:
            return
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60
        dpg.set_value("uptime_display", f"Uptime: {hours:02d}:{minutes:02d}:{seconds:02d}")
        # Push the uptime text to the far right of the viewport
        text_width = 160  # approximate width of "Uptime: HH:MM:SS"
        dpg.set_item_indent("uptime_display", max(0, vp_width - text_width))
        _last_uptime_state = (elapsed, vp_width)
    except Exception:
        pass


# Darkening factor applied to the title bar when a node is selected (0-1, lower = darker)
_SELECTION_DARKNESS_FACTOR = 0.75

# Legacy node name migration for backward-compatible project file loading
_LEGACY_NODE_NAMES = {
    'Rtsp': 'RTSP',
}


def _darken_color_for_selection(color_tuple):
    """Return a slightly darker version of color_tuple for the selected title bar."""
    r, g, b, a = color_tuple
    r = max(0, int(r * _SELECTION_DARKNESS_FACTOR))
    g = max(0, int(g * _SELECTION_DARKNESS_FACTOR))
    b = max(0, int(b * _SELECTION_DARKNESS_FACTOR))
    return (r, g, b, a)


def node_style(module_name):
    tuple_style = STYLE[module_name]["style"][0]
    # Create enhanced color for selected state
    tuple_style_selected = _darken_color_for_selection(tuple_style)
    # Constant for text color to ensure consistency
    TEXT_COLOR_BLACK = (0, 0, 0, 255)
    
    with dpg.theme() as custom_theme:
        with dpg.theme_component(dpg.mvNode):
            # Jaune plein pour la barre de titre
            dpg.add_theme_color(
                dpg.mvNodeCol_TitleBar, tuple_style, category=dpg.mvThemeCat_Nodes
            )
            dpg.add_theme_color(
                dpg.mvNodeCol_TitleBarHovered,
                tuple_style,
                category=dpg.mvThemeCat_Nodes,
            )
            # Slightly darker header when the node is selected
            dpg.add_theme_color(
                dpg.mvNodeCol_TitleBarSelected,
                tuple_style_selected,
                category=dpg.mvThemeCat_Nodes,
            )
            # Texte en noir
            dpg.add_theme_color(
                dpg.mvThemeCol_Text, TEXT_COLOR_BLACK, category=dpg.mvThemeCat_Core
            )
        
        # Add combo box (drop list) styling with node color
        with dpg.theme_component(dpg.mvCombo):
            # Use the node's color for combo box background
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBg, tuple_style, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBgHovered, tuple_style, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBgActive, tuple_style, category=dpg.mvThemeCat_Core
            )
            # Keep text in black for readability
            dpg.add_theme_color(
                dpg.mvThemeCol_Text, TEXT_COLOR_BLACK, category=dpg.mvThemeCat_Core
            )
        
        # Add input fields styling with node color (mvInputInt, mvInputFloat, mvInputText, etc.)
        with dpg.theme_component(dpg.mvInputInt):
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBg, tuple_style, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBgHovered, tuple_style, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBgActive, tuple_style, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_Text, TEXT_COLOR_BLACK, category=dpg.mvThemeCat_Core
            )
        
        with dpg.theme_component(dpg.mvInputFloat):
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBg, tuple_style, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBgHovered, tuple_style, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBgActive, tuple_style, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_Text, TEXT_COLOR_BLACK, category=dpg.mvThemeCat_Core
            )
        
        with dpg.theme_component(dpg.mvInputText):
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBg, tuple_style, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBgHovered, tuple_style, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBgActive, tuple_style, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_Text, TEXT_COLOR_BLACK, category=dpg.mvThemeCat_Core
            )
        
        # Add slider styling with node color (mvSliderInt, mvSliderFloat)
        with dpg.theme_component(dpg.mvSliderInt):
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBg, tuple_style, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBgHovered, tuple_style, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBgActive, tuple_style, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_SliderGrab, TEXT_COLOR_BLACK, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_SliderGrabActive, TEXT_COLOR_BLACK, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_Text, TEXT_COLOR_BLACK, category=dpg.mvThemeCat_Core
            )
        
        with dpg.theme_component(dpg.mvSliderFloat):
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBg, tuple_style, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBgHovered, tuple_style, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBgActive, tuple_style, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_SliderGrab, TEXT_COLOR_BLACK, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_SliderGrabActive, TEXT_COLOR_BLACK, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_Text, TEXT_COLOR_BLACK, category=dpg.mvThemeCat_Core
            )
        
        # Add button styling with node color
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(
                dpg.mvThemeCol_Button, tuple_style, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ButtonHovered, tuple_style, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ButtonActive, tuple_style, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_Text, TEXT_COLOR_BLACK, category=dpg.mvThemeCat_Core
            )
    return custom_theme


class DpgNodeEditor(object):
    _ver = "0.0.1"

    _node_editor_tag = "CV_STUDIO"
    _node_editor_label = "CV_STUDIO"

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
        width=None,
        height=None,
        pos=[0, 0],
        opencv_setting_dict=None,
        node_dir="node",
        menu_dict=None,
        use_debug_print=False,
    ):
        self._node_id = 0

        self._node_factory_list = {}  # NodeFactorylist (objects), factory list
        self._node_instances_list = {}  # NodeInstanceList (objects), instances list
        self._node_list = []  # NodeList
        self._node_link_list = []
        self._node_connection_dict = OrderedDict([])
        self._use_debug_print = use_debug_print

        self._terminate_flag = False

        self._opencv_setting_dict = opencv_setting_dict
        self.window = None

        # Undo stack: stores up to 20 deleted-node snapshots for Ctrl+Z
        self._undo_stack = []
        # Clipboard: stores a list of node snapshots for Ctrl+C / Ctrl+V
        self._clipboard = None
        # Accumulated paste offset so repeated Ctrl+V staggers nodes (reset on Ctrl+C)
        self._clipboard_paste_offset = 0
        # Select-all flag: set by Ctrl+A, consumed by Delete
        self._select_all_flag = False
        # Background close() threads keyed by node_id_name.  Node close() can
        # block for a long time (thread joins, capture release, subprocess
        # wait); running it on the UI thread while _dpg_lock is held freezes
        # rendering and starves the async update thread.
        self._pending_close_threads = {}
        self._pending_close_lock = threading.Lock()

        if menu_dict is None:
            menu_dict = OrderedDict(
                {
                    "Input Node": "input_node",
                    "Process Node": "process_node",
                    "Output Node": "output_node",
                }
            )

        datetime_now = datetime.datetime.now()
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            modal=True,
            height=int(height / 2),
            default_filename=datetime_now.strftime("%Y%m%d"),
            callback=self._callback_file_export,
            id="file_export",
        ):
            dpg.add_file_extension(".json")
            dpg.add_file_extension("", color=(150, 255, 150, 255))

        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            modal=True,
            height=int(height / 2),
            callback=self._callback_file_import,
            id="file_import",
        ):
            dpg.add_file_extension(".json")
            dpg.add_file_extension("", color=(150, 255, 150, 255))

        with dpg.window(
            tag=self._node_editor_tag + "Window",
            label=self._node_editor_label,
            width=width,
            height=height,
            pos=pos,
            menubar=True,
            on_close=self._callback_close_window,
        ) as window:
            with dpg.menu_bar(label="MenuBar"):
                # Export/Import
                with dpg.menu(label="File"):
                    dpg.add_menu_item(
                        tag="Menu_File_Export",
                        label="Export",
                        callback=self._callback_file_export_menu,
                        user_data="Menu_File_Export",
                    )
                    dpg.add_menu_item(
                        tag="Menu_File_Import",
                        label="Import",
                        callback=self._callback_file_import_menu,
                        user_data="Menu_File_Import",
                    )
                
                # print(menu_dict.items())

                for menu_info in menu_dict.items():
                    menu_label = menu_info[0]
                    logger.debug(f"Creating menu: {menu_label}")
                    with dpg.menu(label=menu_label):
                        node_sources_path = os.path.join(
                            node_dir,
                            menu_info[1],
                            "node_*.py",
                        )

                        node_sources = sorted(glob(node_sources_path))
                        # print(node_sources)

                        for node_source in node_sources:
                            # Skip files starting with underscore (disabled nodes)
                            basename = os.path.basename(node_source)
                            if basename.startswith("_"):
                                continue
                                
                            import_path = os.path.splitext(
                                os.path.normpath(node_source)
                            )[0]
                            if platform.system() == "Windows":
                                import_path = import_path.replace("\\", ".")
                            else:
                                import_path = import_path.replace("/", ".")

                            import_path = import_path.split(".")
                            import_path = ".".join(import_path[-3:])

                            if import_path.endswith("__init__"):
                                continue

                            try:
                                module = import_module(import_path)
                                factorynode = module.FactoryNode()
                                if menu_label == "DataProcess" and factorynode.node_tag == "BAR":
                                    continue
                                # print("Factory Instance :", factorynode.node_tag)
                                dpg.add_menu_item(
                                    tag="Menu_" + factorynode.node_tag,
                                    label=factorynode.node_label,
                                    callback=self._callback_add_node,
                                    user_data=factorynode.node_tag,
                                )

                                factorynode.style = node_style(menu_label)
                                self._node_factory_list[factorynode.node_tag] = factorynode
                            except AttributeError:
                                # Skip files without FactoryNode class (utility modules)
                                logger.debug(f"Skipping {import_path}: no FactoryNode attribute")
                                continue

                # Uptime text (far right of menu bar)
                dpg.add_text(
                    tag="uptime_display",
                    default_value="Uptime: 00:00:00",
                )

            with dpg.node_editor(
                tag=self._node_editor_tag,
                callback=self._callback_link,
                minimap=True,
                minimap_location=dpg.mvNodeMiniMap_Location_BottomRight,
            ):
                pass

            with dpg.window(
                label="Delete Files",
                modal=True,
                show=False,
                id="modal_file_import",
                no_title_bar=True,
                pos=[52, 52],
            ):
                dpg.add_text(
                    "Sorry. In the current implementation, \nfile import works only before adding a node.",
                )
                dpg.add_separator()
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="OK",
                        width=375,
                        callback=lambda: dpg.configure_item(
                            "modal_file_import",
                            show=False,
                        ),
                    )
            
            with dpg.handler_registry():
                dpg.add_mouse_click_handler(callback=self._callback_save_last_pos)
                dpg.add_key_press_handler(
                    dpg.mvKey_Delete,
                    callback=self._callback_mv_key_del,
                )
                dpg.add_key_press_handler(
                    dpg.mvKey_Z,
                    callback=self._callback_undo,
                )
                dpg.add_key_press_handler(
                    dpg.mvKey_C,
                    callback=self._callback_copy_node,
                )
                dpg.add_key_press_handler(
                    dpg.mvKey_V,
                    callback=self._callback_paste_node,
                )
                dpg.add_key_press_handler(
                    dpg.mvKey_A,
                    callback=self._callback_select_all,
                )
            
            self.window = window

    def get_node_list(self):
        return self._node_list

    def get_sorted_node_connection(self):
        return self._node_connection_dict

    def get_node_instances(self, node_name):
        return self._node_instances_list.get(node_name, None)

    def get_node_factory(self, node_name):
        return self._node_factory_list.get(node_name, None)

    def set_terminate_flag(self, flag=True):
        self._terminate_flag = flag

    def get_terminate_flag(self):
        return self._terminate_flag

    def _callback_add_node(self, sender, data, user_data):
        with _dpg_lock:
            self._node_id += 1
            logger.debug(f"Adding node with ID: {self._node_id}")
            factorynode = self._node_factory_list[user_data]
            last_pos = [0, 0]

            if self._last_pos is not None:
                last_pos = [self._last_pos[0] + 30, self._last_pos[1] + 30]

            node = factorynode.add_node(
                self._node_editor_tag,
                self._node_id,
                pos=last_pos,
                opencv_setting_dict=self._opencv_setting_dict,
            )

            dpg.bind_item_theme(node.tag_node_name, factorynode.style)
            self._node_instances_list[node.tag_node_name] = node
            self._node_list.append(node.tag_node_name)

            if self._use_debug_print:
                logger.debug("_callback_add_node details:")
                logger.debug(f"    Node ID         : {self._node_id}")
                logger.debug(f"    sender          : {sender}")
                logger.debug(f"    data            : {data}")
                logger.debug(f"    user_data       : {user_data}")
                logger.debug(f"    self._node_list : {', '.join(self._node_list)}")

    def _callback_link(self, sender, data):
        with _dpg_lock:
            logger.debug("Link callback triggered")
            source = dpg.get_item_alias(data[0])
            destination = dpg.get_item_alias(data[1])

            source_parts = source.split(":") if source else []
            destination_parts = destination.split(":") if destination else []

            if len(source_parts) < 3 or len(destination_parts) < 3:
                logger.warning(
                    f"Cannot link: unregistered or malformed attribute aliases "
                    f"{source!r} -> {destination!r}"
                )
                return

            source_type = source_parts[2]
            destination_type = destination_parts[2]
            logger.debug(f"Linking {source_type} -> {destination_type}")

            # ✨ Permettre AUDIO → IMAGE et IMAGE → IMAGE
            connection_allowed = False

            if source_type == destination_type:
                # Connexion normale (même type)
                connection_allowed = True
            elif source_type == "AUDIO" and destination_type == "IMAGE":
                # Connexion spéciale : spectrogramme AUDIO → input IMAGE
                connection_allowed = True
                logger.info(f"Allowing AUDIO->IMAGE connection (spectrogram)")

            if connection_allowed:
                if len(self._node_link_list) == 0:
                    dpg.add_node_link(source, destination, parent=sender)
                    self._node_link_list.append([source, destination])
                else:
                    duplicate_flag = False
                    for node_link in self._node_link_list:
                        if destination == node_link[1]:
                            duplicate_flag = True
                    if not duplicate_flag:
                        dpg.add_node_link(source, destination, parent=sender)
                        self._node_link_list.append([source, destination])

            self._node_connection_dict = self._sort_node_graph(
                self._node_list,
                self._node_link_list,
            )

    def _callback_close_window(self, sender):
        dpg_delete_item(sender)

    def _sort_node_graph(self, node_list, node_link_list):
        node_id_dict = OrderedDict({})
        node_connection_dict = OrderedDict({})

        for node_link_info in node_link_list:
            source = dpg.get_item_alias(node_link_info[0])
            destination = dpg.get_item_alias(node_link_info[1])
            source_id = int(source.split(":")[0])
            destination_id = int(destination.split(":")[0])

            if destination_id not in node_id_dict:
                node_id_dict[destination_id] = [source_id]
            else:
                node_id_dict[destination_id].append(source_id)

            split_destination = destination.split(":")

            node_name = split_destination[0] + ":" + split_destination[1]
            if node_name not in node_connection_dict:
                node_connection_dict[node_name] = [[source, destination]]
            else:
                node_connection_dict[node_name].append([source, destination])

        node_id_list = list(node_id_dict.items())
        node_connection_list = list(node_connection_dict.items())

        # Topological sort via bubble-sort style swaps.
        # A cycle in the graph would cause infinite swapping: guard with a
        # maximum swap budget of n² (guaranteed to suffice for any DAG).
        n = len(node_id_list)
        max_swaps = n * n
        total_swaps = 0
        index = 0
        while index < n:
            swap_flag = False
            for check_id in node_id_list[index][1]:
                for check_index in range(index + 1, n):
                    if node_id_list[check_index][0] == check_id:
                        node_id_list[check_index], node_id_list[index] = (
                            node_id_list[index],
                            node_id_list[check_index],
                        )
                        (
                            node_connection_list[check_index],
                            node_connection_list[index],
                        ) = (
                            node_connection_list[index],
                            node_connection_list[check_index],
                        )

                        swap_flag = True
                        total_swaps += 1
                        break
            if not swap_flag:
                index += 1
            elif total_swaps > max_swaps:
                # Cycle detected — abort the sort to avoid an infinite loop.
                # The partial order is still usable; nodes in the cycle will
                # execute in insertion order which is safe enough at runtime.
                logger.warning(
                    "Cycle detected in node graph (total_swaps=%d > max %d). "
                    "Topological sort aborted — cyclic connections are allowed "
                    "but execution order in the cycle is not guaranteed.",
                    total_swaps, max_swaps,
                )
                break

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
                    for _idx, node_id_name in enumerate(node_list):
                        node_id, node_name = node_id_name.split(":")
                        if int(node_id) == check_id:
                            unfinded_id_dict[check_id] = node_id_name
                            break
            index += 1

        for unfinded_value in unfinded_id_dict.values():
            node_connection_list.insert(0, (unfinded_value, []))

        logger.debug(f"Node connection list: {node_connection_list}")
        return OrderedDict(node_connection_list)

    def _callback_file_export(self, sender, data):
        file_path_name = data.get("file_path_name", "")

        # Guard against a cancelled dialog or an empty/invalid selection so we
        # never try to write the graph to a directory or a missing path.
        if not file_path_name or data.get("file_name", "") == ".":
            logger.warning("Export cancelled: no valid file path was provided.")
            return

        # Ensure the exported file always has a .json extension so that it can
        # be located and re-imported later.
        if not file_path_name.lower().endswith(".json"):
            file_path_name += ".json"

        setting_dict = {}

        setting_dict["node_list"] = self._node_list
        setting_dict["link_list"] = self._node_link_list

        for node_id_name in self._node_list:
            node_id, node_name = node_id_name.split(":")
            node = self._node_instances_list[node_id_name]

            setting = node.get_setting_dict(node_id)

            setting_dict[node_id_name] = {
                "id": str(node_id),
                "name": str(node_name),
                "setting": setting,
            }

        try:
            with open(file_path_name, "w") as fp:
                json.dump(setting_dict, fp, indent=4)
        except (OSError, TypeError, ValueError) as error:
            logger.error(
                f"Failed to export node graph to '{file_path_name}': {error}"
            )
            return

        logger.info(f"Node graph exported to '{file_path_name}'.")

        if self._use_debug_print:
            logger.debug("_callback_file_export details:")
            logger.debug(f"    sender          : {sender}")
            logger.debug(f"    data            : {data}")
            logger.debug(f"    setting_dict    : {setting_dict}")

    def _callback_file_export_menu(self):
        dpg.show_item("file_export")

    def _callback_file_import_menu(self):
        dpg.show_item("file_import")

    def _callback_file_import(self, sender, data):
        if data["file_name"] != ".":
            # Do NOT clear existing nodes — JSON import is an additive operation.
            # Node IDs from the file are offset by the current _node_id to avoid
            # collisions with nodes already on the canvas.

            setting_dict = None
            with open(data["file_path_name"]) as fp:
                setting_dict = json.load(fp)

            # Build an ID remap: "old_id:OldName" -> "new_id:NewName"
            # The new name already incorporates legacy name migration so that
            # _remap_alias can rewrite both IDs and node names in one pass.
            id_offset = self._node_id
            id_remap = {}
            max_file_id = 0
            for node_id_name in setting_dict["node_list"]:
                node_id_str, node_name = node_id_name.split(":", 1)
                old_id = int(node_id_str)
                if old_id > max_file_id:
                    max_file_id = old_id
                new_id = old_id + id_offset
                migrated_name = _LEGACY_NODE_NAMES.get(node_name, node_name)
                new_id_name = f"{new_id}:{migrated_name}"
                id_remap[node_id_name] = new_id_name

            def _remap_alias(alias):
                """Remap 'old_id:OldName:attr' -> 'new_id:NewName:attr'."""
                parts = alias.split(":", 2)
                if len(parts) >= 2:
                    old_key = parts[0] + ":" + parts[1]
                    if old_key in id_remap:
                        new_key = id_remap[old_key]
                        new_parts = new_key.split(":", 1) + (parts[2:] or [])
                        return ":".join(new_parts)
                return alias

            for node_id_name in setting_dict["node_list"]:
                new_id_name = id_remap[node_id_name]
                new_id_str, node_name = new_id_name.split(":", 1)
                new_id = int(new_id_str)

                # Get the factory for this node type (name already migrated above)
                factorynode = self._node_factory_list.get(node_name)
                if factorynode is None:
                    logger.warning(f"Import: unknown node type '{node_name}', skipping.")
                    continue

                # Check version before creating node
                if "setting" in setting_dict[node_id_name] and "ver" in setting_dict[node_id_name]["setting"]:
                    saved_ver = setting_dict[node_id_name]["setting"]["ver"]
                    if hasattr(factorynode, '_ver'):
                        if saved_ver != factorynode._ver:
                            warning_node_name = setting_dict[node_id_name]["name"]
                            logger.warning(f"Node {warning_node_name} version mismatch:")
                            logger.warning(f"  Load Version: {saved_ver}")
                            logger.warning(f"  Code Version: {factorynode._ver}")

                # Create the node instance using the factory
                pos = setting_dict[node_id_name]["setting"].get("pos", [0, 0])
                try:
                    node = factorynode.add_node(
                        self._node_editor_tag,
                        new_id,
                        pos=pos,
                        opencv_setting_dict=self._opencv_setting_dict,
                    )
                except Exception as exc:
                    logger.error(f"Import: failed to create node '{node_name}' (id={new_id}): {exc}")
                    continue

                # Store the node instance
                dpg.bind_item_theme(node.tag_node_name, factorynode.style)
                self._node_instances_list[node.tag_node_name] = node

                # Remap settings keys: saved keys use old_id/old_name prefixes;
                # set_setting_dict expects keys with the current new_id/new_name.
                raw_settings = setting_dict[node_id_name]["setting"]
                remapped_settings = {
                    (_remap_alias(k) if isinstance(k, str) else k): v
                    for k, v in raw_settings.items()
                }

                try:
                    node.set_setting_dict(new_id, remapped_settings)
                except Exception as exc:
                    logger.warning(f"Import: set_setting_dict failed for '{node_name}' (id={new_id}): {exc}")
                    # Settings could not be restored; node appears with defaults.
                    # We still register it in _node_list so it remains manageable
                    # (selectable, deletable) rather than becoming an orphan.

                self._node_list.append(node.tag_node_name)

            # Advance _node_id past all newly imported nodes (exclusive, so +1)
            self._node_id += max_file_id + 1

            # Remap and add links from the imported file
            for link in setting_dict["link_list"]:
                new_src = _remap_alias(link[0])
                new_dst = _remap_alias(link[1])
                dpg.add_node_link(
                    new_src,
                    new_dst,
                    parent=self._node_editor_tag,
                )
                self._node_link_list.append([new_src, new_dst])

            self._node_connection_dict = self._sort_node_graph(
                self._node_list,
                self._node_link_list,
            )

        if self._use_debug_print:
            logger.debug("_callback_file_import details:")
            logger.debug(f"    sender          : {sender}")
            logger.debug(f"    data            : {data}")

    def _callback_save_last_pos(self):
        self._select_all_flag = False
        if len(dpg.get_selected_nodes(self._node_editor_tag)) > 0:
            self._last_pos = dpg.get_item_pos(
                dpg.get_selected_nodes(self._node_editor_tag)[0]
            )

    def _resolve_item_alias(self, item):
        """Return a stable DearPyGui alias whether selection yielded an ID or alias."""
        try:
            alias = dpg.get_item_alias(item)
            if alias:
                return alias
        except Exception as exc:
            logger.warning(
                "_resolve_item_alias: get_item_alias failed for item=%s: %s",
                item, exc, exc_info=True,
            )
        return item if isinstance(item, str) else None

    def _delete_item_by_selection(self, item, fallback_alias=None):
        """Delete a DearPyGui item selected by ID or alias."""
        try:
            if item is not None and dpg.does_item_exist(item):
                logger.debug("_delete_item_by_selection: deleting item %s", item)
                dpg_delete_item(item)
                return
        except Exception as exc:
            logger.error(
                "_delete_item_by_selection: failed to delete item %s: %s",
                item, exc, exc_info=True,
            )
        if fallback_alias is not None:
            try:
                logger.debug(
                    "_delete_item_by_selection: fallback delete alias %s", fallback_alias
                )
                dpg_delete_item(fallback_alias)
            except Exception as exc:
                logger.error(
                    "_delete_item_by_selection: fallback delete failed for alias %s: %s",
                    fallback_alias, exc, exc_info=True,
                )

    def _start_node_close(self, node_id_name, node_instance, node_id):
        """Run a node's close() in a background thread.

        close() implementations release cameras, join worker threads and wait on
        subprocesses, which can block for many seconds.  Deletion runs on the UI
        thread while _dpg_lock is held, so a blocking close() freezes rendering
        *and* blocks the async update thread in update_node_info until the close
        returns (reported by the fault watchdog as "Timeout (0:00:30)!" on
        Linux).  Running close() off the UI thread keeps the lock free.
        """
        if node_instance is None:
            return None

        def _run():
            try:
                node_instance.close(node_id)
            except Exception as exc:
                logger.error(
                    "_start_node_close: close() raised for %s: %s",
                    node_id_name, exc, exc_info=True,
                )

        thread = threading.Thread(
            target=_run, name=f"node-close-{node_id_name}", daemon=True
        )
        with self._pending_close_lock:
            self._pending_close_threads[node_id_name] = thread
        logger.debug("_start_node_close: closing %s in background", node_id_name)
        thread.start()
        return thread

    def _join_pending_closes(self, node_id_name=None, timeout=5.0):
        """Wait for background close() threads to finish (bounded by timeout).

        Called before a node is re-created (undo/paste/load) so that a pending
        close does not delete widgets of the freshly created node.
        """
        with self._pending_close_lock:
            if node_id_name is not None:
                items = [(node_id_name, self._pending_close_threads.get(node_id_name))]
            else:
                items = list(self._pending_close_threads.items())

        for name, thread in items:
            if thread is None:
                continue
            if thread.is_alive():
                thread.join(timeout)
                if thread.is_alive():
                    logger.warning(
                        "_join_pending_closes: close() for %s still running after %ss",
                        name, timeout,
                    )
                    continue
            with self._pending_close_lock:
                self._pending_close_threads.pop(name, None)

    def _purge_node_textures(self, node_id_name):
        """Delete any DPG texture items registered under this node's namespace.

        Textures are stored in a global texture_registry – they are NOT children
        of the node widget, so ``dpg.delete_item(node_widget)`` does not remove
        them.  If they are not removed explicitly before a node is re-created by
        undo, DearPyGui raises an error because the tag already exists, causing
        undo to fail silently for every node type that outputs an image.

        All DPG calls (get_aliases, get_alias_id, does_item_exist, get_item_type,
        remove_alias) are performed while holding _dpg_lock to prevent race
        conditions with the async worker thread on Windows where the thread
        scheduler exposes concurrent DPG alias-registry access.

        The alias is removed immediately (so undo can re-create the node with
        the same tag) but the item deletion itself is deferred through
        schedule_deferred_delete(): this callback runs *during* a frame, and
        destroying a texture the current draw list still references segfaults
        on Linux.  The render loop drains the queue between two frames.
        """
        import platform as _platform
        prefix = node_id_name + ':'
        purged = []
        logger.debug(
            "_purge_node_textures: starting purge for node=%s platform=%s",
            node_id_name, _platform.system(),
        )
        with _dpg_lock:
            try:
                all_aliases = list(dpg.get_aliases())
            except Exception as exc:
                logger.error(
                    "_purge_node_textures: could not retrieve alias list for %s: %s",
                    node_id_name, exc, exc_info=True,
                )
                return
            logger.debug(
                "_purge_node_textures: total aliases in registry=%d, scanning prefix='%s'",
                len(all_aliases), prefix,
            )
            for alias in all_aliases:
                if not alias.startswith(prefix):
                    continue
                try:
                    item_id = dpg.get_alias_id(alias)
                except Exception as exc:
                    logger.warning(
                        "_purge_node_textures: get_alias_id failed for alias %s: %s",
                        alias, exc,
                    )
                    continue
                try:
                    if not dpg.does_item_exist(item_id):
                        logger.debug(
                            "_purge_node_textures: alias %s (id=%s) no longer exists, skipping",
                            alias, item_id,
                        )
                        continue
                    item_type = str(dpg.get_item_type(item_id))
                except Exception as exc:
                    logger.warning(
                        "_purge_node_textures: could not inspect item %s (alias=%s): %s",
                        item_id, alias, exc,
                    )
                    continue
                logger.debug(
                    "_purge_node_textures: found alias=%s id=%s type=%s",
                    alias, item_id, item_type,
                )
                if 'texture' not in item_type.lower():
                    logger.debug(
                        "_purge_node_textures: alias=%s is not a texture (type=%s), skipping",
                        alias, item_type,
                    )
                    continue
                try:
                    logger.debug(
                        "_purge_node_textures: releasing alias=%s id=%s type=%s",
                        alias, item_id, item_type,
                    )
                    # Remove the alias first so the tag is immediately available
                    # again (undo re-creates the node with the same tag).
                    dpg.remove_alias(alias)
                    # Defer the actual deletion: this runs inside a DPG callback,
                    # i.e. during a frame, and deleting a texture still referenced
                    # by the current draw list segfaults on Linux.
                    schedule_deferred_delete(item_id, alias)
                    purged.append(alias)
                except Exception as exc:
                    logger.error(
                        "_purge_node_textures: could not release texture alias=%s (id=%s): %s",
                        alias, item_id, exc, exc_info=True,
                    )
                    continue
        if purged:
            logger.info(
                "_purge_node_textures: purged %d texture(s) for node %s: %s",
                len(purged), node_id_name, purged,
            )
        else:
            logger.debug(
                "_purge_node_textures: no textures found for node %s", node_id_name
            )

    def _delete_dpg_link(self, link_info):
        """Delete the visual dpg node_link item matching link_info [source, dest]."""
        try:
            children = dpg.get_item_children(self._node_editor_tag, slot=0)
            for child_id in children:
                config = dpg.get_item_configuration(child_id)
                attr_1_alias = dpg.get_item_alias(config.get("attr_1", 0))
                attr_2_alias = dpg.get_item_alias(config.get("attr_2", 0))
                if attr_1_alias == link_info[0] and attr_2_alias == link_info[1]:
                    logger.debug("_delete_dpg_link: removing link %s", link_info)
                    dpg_delete_item(child_id)
                    break
        except Exception as exc:
            logger.error(
                "_delete_dpg_link: failed to delete link %s: %s",
                link_info, exc, exc_info=True,
            )

    def _callback_mv_key_del(self):
        # Hold the shared DPG lock for the whole deletion so the async
        # update thread cannot touch items while they are being deleted
        # (concurrent DPG mutation causes segfaults).
        logger.debug("_callback_mv_key_del: Delete key pressed, acquiring lock")
        with _dpg_lock:
            try:
                self._delete_selection()
            except Exception:
                logger.error(
                    "_callback_mv_key_del: unexpected delete failure",
                    exc_info=True,
                )

    def _delete_selection(self):
        # If Ctrl+A was used, delete ALL nodes as a single batch (undoable)
        if self._select_all_flag:
            self._select_all_flag = False
            self._delete_all_nodes_with_undo()
            return

        selected_nodes = list(dpg.get_selected_nodes(self._node_editor_tag))
        selected_links = list(dpg.get_selected_links(self._node_editor_tag))
        logger.info(
            "_delete_selection: %d node(s) and %d link(s) selected",
            len(selected_nodes),
            len(selected_links),
        )
        for item_id in selected_nodes:
            node_id_name = self._resolve_item_alias(item_id)
            if not node_id_name:
                logger.warning(
                    "_delete_selection: could not resolve alias for item_id=%s, skipping",
                    item_id,
                )
                continue
            try:
                node_id, node_name = node_id_name.split(":", 1)
            except ValueError:
                logger.error(
                    "_delete_selection: malformed node alias %s for item_id=%s, skipping",
                    node_id_name, item_id, exc_info=True,
                )
                continue

            if node_name == "ExecPythonCode":
                logger.debug("_delete_selection: skipping ExecPythonCode node %s", node_id_name)
                continue
            if node_id_name not in self._node_list:
                logger.warning(
                    "_delete_selection: node %s not in _node_list, skipping", node_id_name
                )
                continue

            logger.info("_delete_selection: deleting node %s (item_id=%s)", node_id_name, item_id)
            node_instance = self.get_node_instances(node_id_name)
            try:
                exists_before_delete = dpg.does_item_exist(item_id)
            except Exception as exc:
                exists_before_delete = "unknown"
                logger.warning(
                    "_delete_selection: does_item_exist failed for node %s (item_id=%s): %s",
                    node_id_name, item_id, exc,
                )

            attached_links = [
                lnk for lnk in self._node_link_list
                if ":".join(lnk[0].split(":")[:2]) == node_id_name
                or ":".join(lnk[1].split(":")[:2]) == node_id_name
            ]
            logger.debug(
                "_delete_selection: pre-delete context node=%s exists=%s instance=%s attached_links=%d",
                node_id_name,
                exists_before_delete,
                type(node_instance).__name__ if node_instance is not None else None,
                len(attached_links),
            )
            for link_info in attached_links:
                logger.debug("_delete_selection: attached link for %s -> %s", node_id_name, link_info)

            # Snapshot for undo: save settings and involved links before deletion
            try:
                snapshot_settings = node_instance.get_setting_dict(node_id) if node_instance is not None else {}
                snapshot_links = attached_links
                self._undo_stack.append({
                    'node_id_name': node_id_name,
                    'node_id': int(node_id),
                    'node_name': node_name,
                    'settings': copy.deepcopy(snapshot_settings),
                    'links': copy.deepcopy(snapshot_links),
                })
                if len(self._undo_stack) > 20:
                    self._undo_stack.pop(0)
            except Exception as exc:
                logger.warning(
                    "_delete_selection: undo snapshot failed for %s: %s",
                    node_id_name, exc, exc_info=True,
                )

            if node_instance is not None:
                # close() may block (thread joins / capture release); run it in
                # the background so _dpg_lock is not held for its duration.
                logger.debug("_delete_selection: scheduling close() on %s", node_id_name)
                self._start_node_close(node_id_name, node_instance, node_id)

            self._node_list.remove(node_id_name)
            self._node_instances_list.pop(node_id_name, None)
            logger.debug(
                "_delete_selection: removed %s from node list and instances", node_id_name
            )

            # Remove links associated with the deleted node and
            # delete the corresponding visual dpg link items.
            copy_node_link_list = copy.deepcopy(self._node_link_list)
            for link_info in copy_node_link_list:
                source_node = ":".join(link_info[0].split(":")[:2])
                destination_node = ":".join(link_info[1].split(":")[:2])

                if source_node == node_id_name or destination_node == node_id_name:
                    self._node_link_list.remove(link_info)
                    # Delete the visual link from the node editor
                    self._delete_dpg_link(link_info)

            # Delete the node widget BEFORE purging textures: the node widget
            # contains add_image children that reference the texture.  Deleting
            # the texture first leaves those children with a dangling reference
            # which causes DearPyGui to crash (segfault) when the widget is
            # subsequently cleaned up.
            logger.debug(
                "_delete_selection: deleting widget for %s (item_id=%s) then purging textures",
                node_id_name, item_id,
            )
            self._delete_item_by_selection(item_id, fallback_alias=node_id_name)
            self._purge_node_textures(node_id_name)
            logger.info("_delete_selection: node %s fully removed", node_id_name)

        for selected_link in selected_links:
            try:
                if not dpg.does_item_exist(selected_link):
                    continue
            except Exception as exc:
                logger.warning(
                    "_delete_selection: does_item_exist failed for link %s: %s",
                    selected_link, exc,
                )
                continue
            try:
                config = dpg.get_item_configuration(selected_link)
            except Exception as exc:
                logger.warning(
                    "_delete_selection: get_item_configuration failed for link %s: %s",
                    selected_link, exc,
                )
                continue
            attr_1 = config.get("attr_1")
            attr_2 = config.get("attr_2")
            if attr_1 is None or attr_2 is None:
                continue
            link_info = [
                self._resolve_item_alias(attr_1),
                self._resolve_item_alias(attr_2),
            ]
            if not all(link_info) or link_info not in self._node_link_list:
                continue
            logger.debug("_delete_selection: removing selected link %s", link_info)
            self._node_link_list.remove(link_info)
            self._delete_item_by_selection(selected_link)

        if selected_nodes or selected_links:
            self._node_connection_dict = self._sort_node_graph(
                self._node_list,
                self._node_link_list,
            )

        if self._use_debug_print:
            logger.debug("_callback_mv_key_del details:")
            logger.debug(f"    self._node_list            : {self._node_list}")
            logger.debug(f"    self._node_link_list       : {self._node_link_list}")
            logger.debug(
                f"    self._node_connection_dict : {self._node_connection_dict}"
            )

    # ------------------------------------------------------------------
    # Select all (Ctrl+A): visually select all nodes and mark for Delete
    # ------------------------------------------------------------------
    def _callback_select_all(self):
        if not _is_ctrl_down():
            return
        self._select_all_flag = True
        for node_id_name in self._node_list:
            try:
                dpg.select_node(self._node_editor_tag, node_id_name)
            except Exception as exc:
                logger.debug("select_node failed for %s: %s", node_id_name, exc)
        logger.debug("Select all: %d node(s) selected.", len(self._node_list))

    # ------------------------------------------------------------------
    # Batch delete with undo (used when Ctrl+A + Delete is pressed)
    # ------------------------------------------------------------------
    def _delete_all_nodes_with_undo(self):
        """Delete all nodes, storing the entire graph in the undo stack."""
        if not self._node_list:
            return

        # Snapshot every node and the full link list
        batch_entries = []
        for node_id_name in list(self._node_list):
            node_id_str, node_name = node_id_name.split(":", 1)
            node_id = int(node_id_str)
            node_instance = self._node_instances_list.get(node_id_name)
            try:
                settings = node_instance.get_setting_dict(node_id_str) if node_instance is not None else {}
            except Exception as exc:
                logger.warning(
                    "_delete_all_nodes_with_undo: get_setting_dict failed for %s: %s",
                    node_id_name, exc,
                )
                settings = {}
            batch_entries.append({
                'node_id_name': node_id_name,
                'node_id': node_id,
                'node_name': node_name,
                'settings': copy.deepcopy(settings),
            })

        batch_entry = {
            'type': 'batch',
            'entries': batch_entries,
            'links': copy.deepcopy(self._node_link_list),
            'node_id': self._node_id,
        }

        # _clear_all_nodes also clears _undo_stack, so we restore the entry after.
        self._clear_all_nodes()
        self._undo_stack.append(batch_entry)
        logger.info("Batch delete: %d node(s) removed (undoable with Ctrl+Z).", len(batch_entries))

    # ------------------------------------------------------------------
    # Clear all nodes and links (used before loading a new JSON file)
    # ------------------------------------------------------------------
    def _clear_all_nodes(self):
        """Remove every node and link from the editor, resetting internal state."""
        with _dpg_lock:
            for node_id_name in list(self._node_list):
                node_id, _ = node_id_name.split(":")
                node_instance = self._node_instances_list.get(node_id_name)
                if node_instance is not None:
                    # Background close(): see _start_node_close – a blocking
                    # close() here would hold _dpg_lock and freeze the UI.
                    self._start_node_close(node_id_name, node_instance, node_id)
                # Delete the node widget BEFORE purging textures (same reason as
                # _delete_selection: add_image children hold texture references).
                try:
                    item_id = dpg.get_alias_id(node_id_name)
                    if dpg.does_item_exist(item_id):
                        logger.debug("_clear_all_nodes: deleting widget for %s", node_id_name)
                        dpg_delete_item(item_id)
                except Exception as exc:
                    logger.error(
                        "_clear_all_nodes: failed to delete widget for %s: %s",
                        node_id_name, exc, exc_info=True,
                    )
                self._purge_node_textures(node_id_name)

            self._node_list.clear()
            self._node_link_list.clear()
            self._node_instances_list.clear()
            self._node_connection_dict.clear()
            self._node_id = 0
            self._undo_stack.clear()
            self._clipboard = None
            self._clipboard_paste_offset = 0
            logger.info("All nodes cleared.")

        # Outside the lock: pending close() threads may need _dpg_lock themselves
        # (e.g. to delete orphan file dialogs), so joining them while holding the
        # lock would deadlock.
        self._join_pending_closes()

    # ------------------------------------------------------------------
    # Undo (Ctrl+Z): restore the last deleted node (up to 3 levels)
    # ------------------------------------------------------------------
    def _callback_undo(self):
        if not _is_ctrl_down():
            return
        if not self._undo_stack:
            logger.debug("Undo: nothing to undo.")
            return
        entry = self._undo_stack.pop()

        # Batch undo: restore multiple nodes at once (from Ctrl+A + Delete)
        if entry.get('type') == 'batch':
            restored_node_id = entry.get('node_id', self._node_id)
            for node_entry in entry['entries']:
                node_id_name = node_entry['node_id_name']
                node_id = node_entry['node_id']
                node_name = node_entry['node_name']
                settings = node_entry['settings']

                factorynode = self._node_factory_list.get(node_name)
                if factorynode is None:
                    logger.warning(f"Undo batch: factory not found for {node_name}.")
                    continue
                try:
                    pos = settings.get('pos', [0, 0])
                    # Wait for a pending background close() so it cannot delete
                    # widgets belonging to the node we are about to re-create.
                    self._join_pending_closes(node_id_name)
                    self._purge_node_textures(node_id_name)
                    node = factorynode.add_node(
                        self._node_editor_tag,
                        node_id,
                        pos=pos,
                        opencv_setting_dict=self._opencv_setting_dict,
                    )
                    dpg.bind_item_theme(node.tag_node_name, factorynode.style)
                    self._node_instances_list[node.tag_node_name] = node
                    node.set_setting_dict(node_id, settings)
                    self._node_list.append(node_id_name)
                    if node_id > self._node_id:
                        self._node_id = node_id
                except Exception as exc:
                    logger.error(
                        "Undo batch failed for %s: %s", node_id_name, exc, exc_info=True
                    )

            # Restore all links
            for link_info in entry.get('links', []):
                src_node = ":".join(link_info[0].split(":")[:2])
                dst_node = ":".join(link_info[1].split(":")[:2])
                if src_node in self._node_list and dst_node in self._node_list:
                    try:
                        dpg.add_node_link(
                            link_info[0], link_info[1],
                            parent=self._node_editor_tag,
                        )
                        self._node_link_list.append(link_info)
                    except Exception as exc:
                        logger.warning(f"Undo batch: could not restore link {link_info}: {exc}")

            self._node_id = max(self._node_id, restored_node_id)
            self._node_connection_dict = self._sort_node_graph(
                self._node_list, self._node_link_list
            )
            logger.info(f"Undo batch: restored {len(entry['entries'])} node(s).")
            return

        node_id_name = entry['node_id_name']
        node_id = entry['node_id']
        node_name = entry['node_name']
        settings = entry['settings']
        links = entry['links']

        factorynode = self._node_factory_list.get(node_name)
        if factorynode is None:
            logger.warning(f"Undo: factory not found for {node_name}.")
            return

        try:
            pos = settings.get('pos', [0, 0])
            # Wait for a pending background close() so it cannot delete widgets
            # belonging to the node we are about to re-create.
            self._join_pending_closes(node_id_name)
            self._purge_node_textures(node_id_name)
            node = factorynode.add_node(
                self._node_editor_tag,
                node_id,
                pos=pos,
                opencv_setting_dict=self._opencv_setting_dict,
            )
            dpg.bind_item_theme(node.tag_node_name, factorynode.style)
            self._node_instances_list[node.tag_node_name] = node
            node.set_setting_dict(node_id, settings)
            self._node_list.append(node_id_name)
            # Ensure _node_id stays ahead of restored node IDs to avoid future conflicts
            if node_id > self._node_id:
                self._node_id = node_id

            # Restore associated links (only if both endpoints still exist)
            for link_info in links:
                src_node = ":".join(link_info[0].split(":")[:2])
                dst_node = ":".join(link_info[1].split(":")[:2])
                if src_node in self._node_list and dst_node in self._node_list:
                    try:
                        dpg.add_node_link(
                            link_info[0], link_info[1],
                            parent=self._node_editor_tag,
                        )
                        self._node_link_list.append(link_info)
                    except Exception as exc:
                        logger.warning(f"Undo: could not restore link {link_info}: {exc}")

            self._node_connection_dict = self._sort_node_graph(
                self._node_list, self._node_link_list
            )
            logger.info(f"Undo: restored node {node_id_name}.")
        except Exception as exc:
            logger.error("Undo failed for %s: %s", node_id_name, exc, exc_info=True)

    # ------------------------------------------------------------------
    # Copy (Ctrl+C): snapshot ALL selected nodes into the clipboard
    # ------------------------------------------------------------------
    def _callback_copy_node(self):
        if not _is_ctrl_down():
            return
        selected = dpg.get_selected_nodes(self._node_editor_tag)
        if not selected:
            return

        # Clear select-all flag so a subsequent Delete only hits manually selected nodes
        self._select_all_flag = False

        entries = []
        for item_id in selected:
            node_id_name = dpg.get_item_alias(item_id)
            if not node_id_name or node_id_name not in self._node_list:
                continue
            node_id, node_name = node_id_name.split(":", 1)
            node_instance = self._node_instances_list.get(node_id_name)
            if node_instance is None:
                continue
            try:
                settings = node_instance.get_setting_dict(node_id)
                entries.append({
                    'node_name': node_name,
                    'node_id': node_id,
                    'settings': copy.deepcopy(settings),
                })
            except Exception as exc:
                logger.warning(f"Copy failed for {node_id_name}: {exc}")

        if entries:
            # Store as a list; also track the paste offset (reset to 0)
            self._clipboard = entries
            self._clipboard_paste_offset = 0
            logger.info(f"Copied {len(entries)} node(s) to clipboard.")

    # ------------------------------------------------------------------
    # Paste (Ctrl+V): create new nodes from the clipboard snapshot
    # ------------------------------------------------------------------
    def _callback_paste_node(self):
        if not _is_ctrl_down():
            return
        if not self._clipboard:
            return

        # Clear select-all flag so Delete after paste acts on the new nodes only
        self._select_all_flag = False

        # Accumulate paste offset so repeated pastes stagger instead of stacking
        self._clipboard_paste_offset += 40
        offset = self._clipboard_paste_offset

        for entry in self._clipboard:
            node_name = entry['node_name']
            original_node_id = entry.get('node_id')
            settings = copy.deepcopy(entry['settings'])

            factorynode = self._node_factory_list.get(node_name)
            if factorynode is None:
                logger.warning(f"Paste: factory not found for {node_name}.")
                continue

            original_pos = settings.get('pos', [0, 0])
            paste_pos = [original_pos[0] + offset, original_pos[1] + offset]
            settings['pos'] = paste_pos

            try:
                with _dpg_lock:
                    self._node_id += 1
                    new_node_id = self._node_id
                    node = factorynode.add_node(
                        self._node_editor_tag,
                        new_node_id,
                        pos=paste_pos,
                        opencv_setting_dict=self._opencv_setting_dict,
                    )
                    dpg.bind_item_theme(node.tag_node_name, factorynode.style)
                    self._node_instances_list[node.tag_node_name] = node
                    # Remap settings keys from the original node_id prefix to the
                    # new node_id prefix so that set_setting_dict can find them.
                    if original_node_id is not None:
                        old_prefix = str(original_node_id) + ':'
                        new_prefix = str(new_node_id) + ':'
                        remapped_settings = {}
                        for k, v in settings.items():
                            if isinstance(k, str) and k.startswith(old_prefix):
                                remapped_settings[new_prefix + k[len(old_prefix):]] = v
                            else:
                                remapped_settings[k] = v
                        settings = remapped_settings
                    node.set_setting_dict(new_node_id, settings)
                    self._node_list.append(node.tag_node_name)
                    logger.info(f"Pasted new node {node.tag_node_name} from clipboard.")
            except Exception as exc:
                logger.error(f"Paste failed for {node_name}: {exc}")

        self._node_connection_dict = self._sort_node_graph(
            self._node_list, self._node_link_list
        )
