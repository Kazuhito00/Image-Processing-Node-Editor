#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Deploy Node

Saves the current CvStudio schema (JSON), converts it into a production-ready
NVIDIA DeepStream project using the DeepStream Engine, and deploys it on the
host machine.

Workflow:
  1. Click "Save & Convert" → exports current graph JSON + generates DeepStream project
  2. Click "Deploy" → runs the generated project via `make run` or `./run.sh`
"""

import json
import os
import subprocess
import threading
import time
import traceback
from pathlib import Path

import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value
from node.node_abc import DpgNodeABC
from node.basenode import Node


# Default output directory for generated DeepStream projects
_DEFAULT_OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "cvstudio_deploy")


class FactoryNode:
    node_label = "Deploy"
    node_tag = "Deploy"

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
        node = _Node()
        node.tag_node_name = str(node_id) + ":" + node.node_tag
        node._opencv_setting_dict = opencv_setting_dict

        tag = node.tag_node_name

        with dpg.node(
            tag=tag,
            parent=parent,
            label=node.node_label,
            pos=pos,
        ):
            # --- Static attribute: output path ---
            with dpg.node_attribute(
                tag=tag + ":Static",
                attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_spacer(height=2)
                dpg.add_text(
                    tag=tag + ":Title",
                    default_value="DeepStream Deploy",
                )
                dpg.add_spacer(height=4)

                # Output directory
                dpg.add_input_text(
                    tag=tag + ":OutputDir",
                    label="Output Dir",
                    default_value=_DEFAULT_OUTPUT_DIR,
                    width=220,
                )

                # Project name
                dpg.add_input_text(
                    tag=tag + ":ProjectName",
                    label="Project",
                    default_value="cvstudio_ds",
                    width=220,
                )

                # Hardware profile combo
                dpg.add_combo(
                    tag=tag + ":Profile",
                    label="Profile",
                    items=["RTX_5070", "RTX_5070_HIGH_BATCH", "RTX_5070_INT8"],
                    default_value="RTX_5070",
                    width=220,
                )

                dpg.add_spacer(height=4)

                # Overwrite checkbox
                dpg.add_checkbox(
                    tag=tag + ":Overwrite",
                    label="Overwrite existing",
                    default_value=True,
                )

                dpg.add_spacer(height=6)

                # Save & Convert button
                dpg.add_button(
                    tag=tag + ":BtnConvert",
                    label="Save & Convert",
                    width=220,
                    callback=_callback_convert,
                    user_data=node,
                )

                dpg.add_spacer(height=4)

                # Deploy button
                dpg.add_button(
                    tag=tag + ":BtnDeploy",
                    label="Deploy",
                    width=220,
                    callback=_callback_deploy,
                    user_data=node,
                )

                dpg.add_spacer(height=6)

                # Status text
                dpg.add_text(
                    tag=tag + ":Status",
                    default_value="Ready",
                    color=(180, 180, 180),
                )

        return node


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def _get_node_editor():
    """Retrieve the DpgNodeEditor singleton from the main module."""
    import sys
    main_module = sys.modules.get("__main__") or sys.modules.get("main")
    if main_module and hasattr(main_module, "_node_editor_ref"):
        return main_module._node_editor_ref
    return None


def _build_schema_dict(node_editor):
    """
    Build the schema dict (same format as File > Export).
    Returns a dict ready to be serialized to JSON.
    """
    setting_dict = {}
    setting_dict["node_list"] = node_editor._node_list
    setting_dict["link_list"] = node_editor._node_link_list

    for node_id_name in node_editor._node_list:
        node_id, node_name = node_id_name.split(":")
        node_instance = node_editor._node_instances_list.get(node_id_name)
        if node_instance is None:
            continue

        setting = node_editor.get_node_setting_dict(node_id_name)
        setting_dict[node_id_name] = {
            "id": str(node_id),
            "name": str(node_name),
            "setting": setting,
        }

    return setting_dict


def _callback_convert(sender, data, user_data):
    """Save the current schema and convert to DeepStream project."""
    node = user_data
    tag = node.tag_node_name

    def _do_convert():
        try:
            dpg_set_value(tag + ":Status", "Saving schema...")

            # Get parameters from UI
            output_dir = dpg_get_value(tag + ":OutputDir")
            project_name = dpg_get_value(tag + ":ProjectName")
            profile = dpg_get_value(tag + ":Profile")
            overwrite = dpg_get_value(tag + ":Overwrite")

            # Build schema from current node editor state
            node_editor = _get_node_editor()
            if node_editor is None:
                dpg_set_value(tag + ":Status", "ERROR: No editor ref")
                return

            schema = _build_schema_dict(node_editor)

            # Save JSON to output dir
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            json_path = output_path / f"{project_name}.json"
            with open(json_path, "w", encoding="utf-8") as fp:
                json.dump(schema, fp, indent=4)

            dpg_set_value(tag + ":Status", "Converting to DeepStream...")

            # Convert using DeepStream Engine
            from tools.deepstream_engine.engine import DeepStreamEngine

            engine = DeepStreamEngine(profile_name=profile)
            project_output = output_path / project_name

            result = engine.convert(
                input_json=str(json_path),
                output_dir=str(project_output),
                project_name=project_name,
                overwrite=overwrite,
            )

            # Store result for deploy step
            node._last_result = result
            node._project_dir = str(project_output)

            status = (
                f"OK: {result.nodes_mapped} nodes, "
                f"{len(result.all_files)} files"
            )
            dpg_set_value(tag + ":Status", status)

        except Exception as e:
            dpg_set_value(tag + ":Status", f"ERR: {e}")
            traceback.print_exc()

    # Run in background thread to avoid blocking UI
    threading.Thread(target=_do_convert, daemon=True).start()


def _callback_deploy(sender, data, user_data):
    """Deploy the generated DeepStream project on the host machine."""
    node = user_data
    tag = node.tag_node_name

    project_dir = getattr(node, "_project_dir", None)
    if not project_dir or not Path(project_dir).exists():
        dpg_set_value(tag + ":Status", "Convert first!")
        return

    def _do_deploy():
        try:
            dpg_set_value(tag + ":Status", "Deploying...")

            project_path = Path(project_dir)

            # Try docker compose first, fallback to run.sh
            makefile = project_path / "Makefile"
            run_script = project_path / "run.sh"

            if makefile.exists():
                proc = subprocess.Popen(
                    ["make", "run"],
                    cwd=str(project_path),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                node._deploy_process = proc
                dpg_set_value(tag + ":Status", "Deployed (make run)")
            elif run_script.exists():
                proc = subprocess.Popen(
                    ["bash", str(run_script)],
                    cwd=str(project_path),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                node._deploy_process = proc
                dpg_set_value(tag + ":Status", "Deployed (run.sh)")
            else:
                dpg_set_value(tag + ":Status", "ERR: No Makefile/run.sh")

        except FileNotFoundError as e:
            dpg_set_value(tag + ":Status", f"ERR: {e}")
        except Exception as e:
            dpg_set_value(tag + ":Status", f"ERR: {e}")
            traceback.print_exc()

    threading.Thread(target=_do_deploy, daemon=True).start()


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class _Node(Node):
    _ver = "0.0.1"

    node_label = "Deploy"
    node_tag = "Deploy"

    _opencv_setting_dict = None
    _last_result = None
    _project_dir = None
    _deploy_process = None

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
        """No-op update – Deploy is triggered by button clicks only."""
        return None

    def get_setting_dict(self, node_id):
        tag = str(node_id) + ":" + self.node_tag
        setting_dict = {}
        setting_dict["ver"] = self._ver
        setting_dict["pos"] = dpg.get_item_pos(tag)
        setting_dict["output_dir"] = dpg_get_value(tag + ":OutputDir")
        setting_dict["project_name"] = dpg_get_value(tag + ":ProjectName")
        setting_dict["profile"] = dpg_get_value(tag + ":Profile")
        setting_dict["overwrite"] = dpg_get_value(tag + ":Overwrite")
        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        tag = str(node_id) + ":" + self.node_tag
        output_dir = setting_dict.get("output_dir", _DEFAULT_OUTPUT_DIR)
        project_name = setting_dict.get("project_name", "cvstudio_ds")
        profile = setting_dict.get("profile", "RTX_5070")
        overwrite = setting_dict.get("overwrite", True)

        dpg_set_value(tag + ":OutputDir", output_dir)
        dpg_set_value(tag + ":ProjectName", project_name)
        dpg_set_value(tag + ":Profile", profile)
        dpg_set_value(tag + ":Overwrite", overwrite)

    def close(self, node_id):
        # Kill deploy process if running
        if self._deploy_process and self._deploy_process.poll() is None:
            self._deploy_process.terminate()
