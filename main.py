#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os

# Add the current directory to Python path to allow imports from src package
# This is necessary when running main.py directly without installing the package
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

import copy
import json
import asyncio
import argparse
import atexit
import faulthandler
import logging
import platform
import signal
import threading
from collections import OrderedDict
import time
import multiprocessing
import cv2
import dearpygui.dearpygui as dpg

from src.utils.logging import setup_logging, get_logger
from src.utils.gpu_utils import log_gpu_info

from node_editor.util import (
    check_camera_connection,
    _dpg_lock,
    process_deferred_deletes,
)
from node_editor.scheduling import FramePacer
from node_editor.node_main import DpgNodeEditor
from node_editor.node_main import update_uptime_display
from node_editor.zoom import register_overview_fonts

# Import timestamped queue system
from node.timestamped_queue import NodeDataQueueManager
from node.queue_adapter import QueueBackedDict

# Setup logging
logger = get_logger(__name__)

from node_editor.splash import show_splash_screen as _show_splash_screen

# Reference to the node editor instance, used by nodes that need editor access
# (e.g., Deploy node for schema export)
_node_editor_ref = None
_fault_log_file_handle = None
_fault_log_atexit_registered = False

# Connection types that carry frame data (and therefore a source timestamp)
_DATA_CONNECTION_TYPES = frozenset(("IMAGE", "AUDIO", "JSON"))


def _log_thread_exception(args):
    thread_name = getattr(args.thread, "name", "unknown")
    logger.error(
        "Unhandled exception in thread '%s'",
        thread_name,
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )


def _unique_streams(*streams):
    """Return a deduplicated list of streams, always including sys.stderr."""
    seen_fds = set()
    result = []
    for s in streams:
        if s is None:
            continue
        try:
            fd = s.fileno()
        except Exception:
            fd = id(s)
        if fd not in seen_fds:
            seen_fds.add(fd)
            result.append(s)
    # Always include stderr as a fallback if not already present
    try:
        stderr_fd = sys.stderr.fileno()
    except Exception:
        stderr_fd = id(sys.stderr)
    if stderr_fd not in seen_fds:
        result.append(sys.stderr)
    return result


def configure_fault_diagnostics(fault_log_path=None):
    """Enable diagnostics for hard crashes (e.g., segfaults) and thread exceptions.

    Registers handlers for SIGSEGV, SIGABRT, SIGFPE, SIGBUS and SIGUSR1/SIGUSR2 so
    that a full stack trace of *all* threads is written on any hard crash.  The
    handlers use chain=True so that the default OS action (core dump) still fires
    after the Python trace is written.

    When a log file path is supplied the trace is written to that file **and** to
    stderr (fd 2), because Python I/O may be unreliable inside a signal handler
    after a segfault.  Using raw file descriptors (sys.stderr / the open fd of the
    log file) maximises the chance that output reaches disk before the process dies.

    A background watchdog via faulthandler.dump_traceback_later() writes a full
    thread dump every 30 seconds; this captures the last-known state even when the
    crash happens between two watchdog ticks.
    """
    global _fault_log_file_handle, _fault_log_atexit_registered

    target_stream = sys.stderr
    resolved_path = None
    path = fault_log_path or os.environ.get("CV_STUDIO_FAULT_LOG")
    if path:
        if _fault_log_file_handle is not None:
            _close_fault_log_file()
        resolved_path = os.path.abspath(path)
        resolved_dir = os.path.dirname(resolved_path)
        if resolved_dir:
            os.makedirs(resolved_dir, exist_ok=True)
        # Line-buffered text mode; every newline flushes to the OS.
        _fault_log_file_handle = open(resolved_path, "a", buffering=1, encoding="utf-8")
        target_stream = _fault_log_file_handle
        if not _fault_log_atexit_registered:
            atexit.register(_close_fault_log_file)
            _fault_log_atexit_registered = True

    # Primary handler: dumps all threads on SIGSEGV / hard faults.
    faulthandler.enable(file=target_stream, all_threads=True)

    # When writing to a file, also mirror the crash dump to stderr so that it is
    # visible in the terminal and survives even if the file flush is incomplete.
    if target_stream is not sys.stderr:
        try:
            faulthandler.enable(file=sys.stderr, all_threads=True)
        except Exception as exc:
            logger.warning("Failed to enable faulthandler on stderr: %s", exc)

    # Explicitly register crash signals with chain=True so the OS default
    # (e.g., core dump) still executes after the Python traceback is printed.
    # This gives the most exhaustive output: full Python stacks from every thread.
    # NOTE: SIGSEGV, SIGABRT, SIGFPE, SIGBUS are handled exclusively by
    # faulthandler.enable() on Linux — faulthandler.register() raises RuntimeError
    # for those signals.  We register them symbolically for the startup log message
    # but rely on faulthandler.enable() (called above) for the actual handler.
    _crash_signals = ("SIGSEGV", "SIGABRT", "SIGFPE", "SIGBUS")
    _registered_signals = [
        sig_name
        for sig_name in _crash_signals
        if getattr(signal, sig_name, None) is not None
    ]

    # SIGUSR1/SIGUSR2: on-demand dump triggered by the user.
    # faulthandler.register() only supports a single stream per signal (each
    # successive call replaces the previous handler).  To write to both the log
    # file and stderr we install a Python-level signal handler that calls
    # faulthandler.dump_traceback() for every output stream explicitly.
    _dump_streams = _unique_streams(target_stream)

    def _on_dump_signal(signum, frame, streams=_dump_streams):
        for s in streams:
            try:
                faulthandler.dump_traceback(file=s, all_threads=True)
                s.flush()
            except Exception:
                pass

    for sig_name in ("SIGUSR1", "SIGUSR2"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _on_dump_signal)
        except (OSError, ValueError) as exc:
            logger.warning("Failed to register fault dump signal %s: %s", sig_name, exc)

    # Windows does not have SIGUSR1/SIGUSR2.  Register SIGBREAK (Ctrl+Break)
    # as the on-demand thread-dump trigger instead so that Windows users can
    # get a full traceback from a running process by pressing Ctrl+Break.
    if sys.platform == "win32":
        sig_break = getattr(signal, "SIGBREAK", None)
        if sig_break is not None:
            try:
                signal.signal(sig_break, _on_dump_signal)
                logger.info("Windows: on-demand thread dump registered on SIGBREAK (Ctrl+Break)")
            except (OSError, ValueError) as exc:
                logger.warning("Failed to register SIGBREAK handler: %s", exc)

    # Watchdog: periodically write all-thread stacks so the last snapshot is
    # available even if the crash happens between two watchdog ticks.
    _watchdog_interval = int(os.environ.get("CV_STUDIO_WATCHDOG_INTERVAL", "30"))
    try:
        faulthandler.dump_traceback_later(
            _watchdog_interval, repeat=True, file=target_stream, exit=False
        )
        logger.info(
            "Fault watchdog enabled: full thread dump every %ds", _watchdog_interval
        )
    except Exception as exc:
        logger.warning("Failed to start faulthandler watchdog: %s", exc)

    threading.excepthook = _log_thread_exception
    logger.info(
        "Fault diagnostics enabled%s — crash signals: %s",
        f" (fault trace output: {resolved_path})" if resolved_path else " (fault trace output: stderr)",
        ", ".join(_registered_signals) if _registered_signals else "none",
    )


def _close_fault_log_file():
    global _fault_log_file_handle
    if _fault_log_file_handle is not None:
        file_handle = _fault_log_file_handle
        try:
            # Cancel the watchdog before closing the file so it doesn't write to
            # the closed file descriptor after this point.
            try:
                faulthandler.cancel_dump_traceback_later()
            except Exception:
                pass
            file_handle.flush()
            file_handle.close()
            try:
                faulthandler.enable(file=sys.stderr, all_threads=True)
            except Exception:
                pass
        finally:
            _fault_log_file_handle = None


def get_resource_path(relative_path):
    """
    Get the absolute path to a resource, works for both development and frozen mode.

    When running as a script, returns the path relative to the script directory.
    When running as a PyInstaller executable (.exe), returns the path relative to
    the temporary directory where PyInstaller extracts files (sys._MEIPASS).

    Args:
        relative_path (str): Relative path to the resource (e.g., 'assets/image.png')

    Returns:
        str: Absolute path to the resource
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
        frozen = True
    except AttributeError:
        # Running in normal Python environment (script mode)
        base_path = os.path.dirname(os.path.abspath(__file__))
        frozen = False

    # Normalize path separators for cross-platform compatibility
    # This handles cases where relative_path uses forward slashes on Windows
    resource_path = os.path.normpath(os.path.join(base_path, relative_path))
    
    # Debug logging to help troubleshoot path issues
    logger.debug(
        f"Resource path resolution:\n"
        f"  Frozen mode: {frozen}\n"
        f"  Base path: {base_path}\n"
        f"  Relative path: {relative_path}\n"
        f"  Resolved path: {resource_path}\n"
        f"  Path exists: {os.path.exists(resource_path)}"
    )
    
    return resource_path


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--setting",
        type=str,
        default=get_resource_path("node_editor/setting/setting.json"),
    )
    parser.add_argument("--unuse_async_draw", action="store_true")
    parser.add_argument("--use_debug_print", action="store_true")
    parser.add_argument(
        "--fault_log",
        type=str,
        default=None,
        help="Optional file path for faulthandler output (segfault traceback).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help=(
            "Optional file path for application log output. "
            "On Windows, defaults to %%APPDATA%%\\CV_Studio\\cv_studio.log "
            "when not provided."
        ),
    )
    parser.add_argument(
        "--smoke-test",
        nargs="?",
        const=30,
        default=None,
        type=int,
        metavar="FRAMES",
        help=(
            "Start the full application (config, DearPyGui, node discovery), "
            "add a node, zoom out and back, render FRAMES frames (default 30), "
            "then exit. Exit code 0 means the "
            "build starts without error. Used by CI to validate the frozen .exe."
        ),
    )
    args = parser.parse_args()
    return args


def async_main(node_editor, queue_manager, stopped=None):
    try:
        _async_main(node_editor, queue_manager)
    finally:
        # Lets the main thread know that no DearPyGui call can come from
        # this thread any more (see the shutdown sequence in main()).
        if stopped is not None:
            stopped.set()


def _async_main(node_editor, queue_manager):
    # Create queue-backed dictionaries for backward compatibility
    node_image_dict = QueueBackedDict(queue_manager, "image")
    node_result_dict = QueueBackedDict(queue_manager, "json")
    node_audio_dict = QueueBackedDict(queue_manager, "audio")

    # Inject live references so agent nodes can perform tool discovery
    # (these are dict references; mutations are visible without re-injection)
    node_result_dict._node_instances = node_editor._node_instances_list
    node_result_dict._node_link_list = node_editor._node_link_list
    
    logger.info("Async main loop started with timestamped queue system")
    
    while not node_editor.get_terminate_flag():
        update_node_info(
            node_editor, node_image_dict, node_result_dict, node_audio_dict
        )
        # Small sleep to prevent CPU hogging and keep UI responsive
        # Note: This function runs in a thread executor (not asyncio coroutine),
        # so time.sleep() is appropriate here to yield CPU to other threads
        # Increased to 10ms to ensure UI remains responsive during video playback
        time.sleep(0.01)  # 10ms sleep to yield CPU and keep UI responsive


def update_node_info(
    node_editor,
    node_image_dict,
    node_result_dict,
    node_audio_dict,
    mode_async=True,
):
    # All DearPyGui calls below must hold _dpg_lock: this function runs in the
    # async worker thread while the main thread can delete nodes concurrently
    # (unlocked concurrent DPG mutation segfaults, notably on Linux).
    with _dpg_lock:
        # Update uptime display
        update_uptime_display()

        try:
            dpg.set_item_pos(node_editor.window, [0, 0])
            dpg.set_item_width(node_editor.window, dpg.get_viewport_client_width())
            dpg.set_item_height(node_editor.window, dpg.get_viewport_client_height())
        except Exception as e:
            logger.error(f"Failed to set node editor window properties: {e}")

    # Snapshot the node list: the delete callback (main thread) can mutate it
    # while this loop runs in the async worker thread.
    node_list = list(node_editor.get_node_list())

    sorted_node_connection_dict = node_editor.get_sorted_node_connection()

    for node_id_name in node_list:
        if node_editor.get_terminate_flag():
            # Shutting down: the DearPyGui context is about to be destroyed.
            return
        if node_id_name not in node_image_dict:
            node_image_dict[node_id_name] = None

        node_id, _ = node_id_name.split(":")
        connection_list = sorted_node_connection_dict.get(node_id_name, [])
        # Hold the lock while updating the node: node updates draw to DPG
        # (textures, values) and must not run while the main thread deletes
        # items. Re-check the instance inside the lock in case the node was
        # deleted after the snapshot was taken.
        with _dpg_lock:
            node_instance = node_editor.get_node_instances(node_id_name)
            if node_instance is None:
                # Node was deleted after the snapshot was taken; skip it.
                continue
            logger.debug(
                "Processing node %s with connections: %s",
                node_id_name, connection_list,
            )
            if mode_async:
                try:
                    data = node_instance.update(
                        node_id,
                        connection_list,
                        node_image_dict,
                        node_result_dict,
                        node_audio_dict,
                    )
                except Exception as e:
                    logger.error(f"Error updating node {node_id_name}: {e}", exc_info=True)
                    # sys.exit()
                    continue
            else:
                data = node_instance.update(
                    node_id,
                    connection_list,
                    node_image_dict,
                    node_result_dict,
                    node_audio_dict,
                )
            # Deepcopy data while holding the lock so that any numpy arrays
            # that share memory with DPG texture buffers are fully copied
            # before the lock is released.  Without this, the main thread
            # can delete the texture (freeing its buffer) between here and
            # the copy is used below, which causes a segfault on Linux.
            # This is the only copy made: `data` is private to this function
            # from here on, so its members are stored without copying again
            # (copying every frame twice was the main per-node overhead).
            data = copy.deepcopy(data)

        try:
            # Determine if this is an input node or a processing node
            # Input nodes have no IMAGE/AUDIO/JSON input connections
            # Processing nodes have at least one IMAGE/AUDIO/JSON input connection
            has_data_input = False
            source_timestamp = None
            
            for connection_info in connection_list:
                # Validate connection_info structure before accessing
                if not connection_info or len(connection_info) < 2:
                    continue
                
                connection_parts = connection_info[0].split(":")
                if len(connection_parts) < 3:
                    continue
                    
                connection_type = connection_parts[2]
                if connection_type in _DATA_CONNECTION_TYPES:
                    has_data_input = True
                    # Get the timestamp from the source node
                    source_node_id = ":".join(connection_parts[:2])
                    
                    # Try to get timestamp based on connection type
                    if connection_type == "IMAGE":
                        source_timestamp = node_image_dict.get_timestamp(source_node_id)
                    elif connection_type == "AUDIO":
                        source_timestamp = node_audio_dict.get_timestamp(source_node_id)
                    elif connection_type == "JSON":
                        source_timestamp = node_result_dict.get_timestamp(source_node_id)
                    
                    # Use the first data connection's timestamp
                    if source_timestamp is not None:
                        break
            
            # Check if the node provided an explicit timestamp (e.g., FPS-based timestamp from Video node)
            # This allows input nodes to specify timestamps based on their internal timing (FPS, audio chunks, etc.)
            node_provided_timestamp = data.get("timestamp", None) if isinstance(data, dict) else None
            
            # Store data with appropriate timestamp
            # Priority:
            # 1. For processing nodes: preserve source timestamp from connected input
            # 2. For input nodes with explicit timestamp: use the node-provided timestamp (FPS-based, etc.)
            # 3. For input nodes without explicit timestamp: create new timestamp automatically
            if has_data_input and source_timestamp is not None:
                # Processing node - preserve source timestamp
                node_image_dict.set_with_timestamp(node_id_name, data["image"], source_timestamp)
                node_result_dict.set_with_timestamp(node_id_name, data["json"], source_timestamp)
                node_audio_dict.set_with_timestamp(node_id_name, data["audio"], source_timestamp)
                logger.debug("Node %s preserved timestamp %.6f from source", node_id_name, source_timestamp)
            elif node_provided_timestamp is not None:
                # Input node with explicit timestamp (e.g., Video node with FPS-based timing)
                node_image_dict.set_with_timestamp(node_id_name, data["image"], node_provided_timestamp)
                node_result_dict.set_with_timestamp(node_id_name, data["json"], node_provided_timestamp)
                node_audio_dict.set_with_timestamp(node_id_name, data["audio"], node_provided_timestamp)
                logger.debug("Node %s used explicit timestamp %.6f", node_id_name, node_provided_timestamp)
            else:
                # Input node without explicit timestamp - create new timestamp automatically
                node_image_dict[node_id_name] = data["image"]
                node_result_dict[node_id_name] = data["json"]
                node_audio_dict[node_id_name] = data["audio"]
                logger.debug("Node %s created new timestamp (input node)", node_id_name)
        except Exception as e:
            logger.error(f"Error processing node {node_id_name} results: {e}")


def show_splash_screen(duration_seconds=5.0, steps=150):
    """Delegate to the Apple-style splash screen module."""
    _show_splash_screen(duration_seconds=duration_seconds, steps=steps)


def _smoke_test_steps(node_editor):
    """Exercise the node editor during a smoke test, one step per frame.

    Adds a node, zooms out to the furthest overview level and back, so that
    the frozen build is checked on what the interface actually draws, not
    only on an empty editor.  Raises on any inconsistency.
    """
    yield "start"
    node_tag = next(
        (tag for tag in ("Blur", "Grayscale") if node_editor.get_node_factory(tag)),
        None,
    )
    if node_tag is None:
        raise RuntimeError("Smoke-test failed: no basic node available")
    node_editor._callback_add_node(None, None, node_tag)
    node_name = node_editor.get_node_list()[-1]
    with _dpg_lock:
        label = dpg.get_item_label(node_name)
    yield "node added"

    levels = 0
    while node_editor._zoom.scale > min(node_editor._zoom._levels):
        node_editor._callback_zoom_out()
        levels += 1
        yield "zoomed out"
    node_editor._callback_zoom_reset()
    yield "zoom reset"

    with _dpg_lock:
        restored = dpg.get_item_label(node_name)
    if node_editor._zoom.is_zoomed_out or restored != label:
        raise RuntimeError(
            f"Smoke-test failed: zoom not restored ({label!r} -> {restored!r})"
        )
    logger.info("Smoke-test: node %s zoomed through %d overview levels", node_name, levels)
    yield "done"


def _smoke_test_distillation():
    """Check that a YOLOX student can be converted for training and exported.

    A packaged build must be able to train: there, a missing or broken
    PyTorch / onnx2torch fails the smoke test.  From sources they are
    optional and their absence is only reported.
    """
    from node.DLNode.online_training.distill import yolox_torch
    from node.DLNode.node_yolox_distillation import find_yolox_models

    available, reason = yolox_torch.backend_status()
    if not available:
        if getattr(sys, "frozen", False):
            raise RuntimeError(f"Smoke-test failed: distillation unavailable ({reason})")
        logger.warning("Smoke-test: distillation training unavailable (%s)", reason)
        return
    models = find_yolox_models()
    if not models:
        raise RuntimeError("Smoke-test failed: no YOLOX model found for distillation")
    name, path = next(iter(models.items()))
    yolox_torch.TrainableYolox(path).export()
    logger.info("Smoke-test: distillation backend OK (%s converted and exported)", name)


def main():
    args = get_args()
    setting = args.setting
    unuse_async_draw = args.unuse_async_draw
    use_debug_print = args.use_debug_print
    fault_log = args.fault_log
    log_file = args.log_file
    smoke_test_frames = args.smoke_test
    smoke_test = smoke_test_frames is not None

    # Setup logging based on debug flag.
    # On Windows, setup_logging() automatically writes to
    # %APPDATA%\CV_Studio\cv_studio.log when log_file is None, ensuring a
    # persistent log even when the process has no attached console.
    log_level = logging.DEBUG if use_debug_print else logging.INFO
    root_logger = setup_logging(level=log_level, log_file=log_file)
    configure_fault_diagnostics(fault_log_path=fault_log)

    logger.info("=" * 60)
    logger.info("CV_STUDIO Starting")
    logger.info("=" * 60)
    if smoke_test:
        logger.info("Smoke-test mode: will render %d frame(s) then exit", smoke_test_frames)

    # Log platform and Python version for easier remote diagnosis.
    logger.info(
        "Platform: %s | Python: %s | Frozen: %s",
        platform.platform(),
        sys.version.split()[0],
        getattr(sys, "frozen", False),
    )

    # Log the active log file paths so users know where to find their logs.
    for handler in root_logger.handlers:
        if hasattr(handler, "baseFilename"):
            logger.info("Application log file: %s", handler.baseFilename)

    # Initialize timestamped buffer system
    logger.info("Initializing timestamped buffer system")
    queue_manager = NodeDataQueueManager(default_maxsize=10)
    logger.info("Buffer system initialized: keeps last 10 timestamped items per node for synchronization")

    logger.info("Loading configuration")
    logger.debug(f"Configuration file path: {setting}")
    
    # Verify the configuration file exists before attempting to load
    if not os.path.exists(setting):
        logger.error(f"Configuration file not found: {setting}")
        # Check if we're in a frozen (PyInstaller) environment
        if getattr(sys, 'frozen', False):
            logger.error(f"Running in frozen mode. Base path (_MEIPASS): {sys._MEIPASS}")
            logger.error("The setting.json file may not have been properly bundled with PyInstaller.")
            logger.error("Please ensure CV_Studio.spec includes: datas.append(('node_editor', 'node_editor'))")
        else:
            logger.error("Running in script mode. The setting.json file should be in node_editor/setting/")
        raise FileNotFoundError(f"Configuration file not found: {setting}")
    
    opencv_setting_dict = None
    with open(setting) as fp:
        opencv_setting_dict = json.load(fp)
    logger.info("Configuration loaded successfully")
    webcam_width = opencv_setting_dict["webcam_width"]
    webcam_height = opencv_setting_dict["webcam_height"]

    # Log GPU information
    if opencv_setting_dict.get("use_gpu", False):
        log_gpu_info()

    logger.info("Checking camera connections")
    device_no_list = check_camera_connection()
    camera_capture_list = []
    for device_no in device_no_list:
        video_capture = cv2.VideoCapture(device_no)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)
        camera_capture_list.append(video_capture)
        logger.info(f"Camera {device_no} connected")

    opencv_setting_dict["device_no_list"] = device_no_list
    opencv_setting_dict["camera_capture_list"] = camera_capture_list

    editor_width = opencv_setting_dict["editor_width"]
    editor_height = opencv_setting_dict["editor_height"]
    ui_fps = opencv_setting_dict.get("ui_fps", 60)

    serial_device_no_list = []
    serial_connection_list = []
    use_serial = opencv_setting_dict["use_serial"]
    if use_serial:
        import serial
        from node_editor.util import check_serial_connection
        logger.info("Checking serial device connections")
        serial_device_no_list = check_serial_connection()
        for serial_device_no in serial_device_no_list:
            ser = serial.Serial(serial_device_no, 115200)
            serial_connection_list.append(ser)
            logger.info(f"Serial device {serial_device_no} connected")

    opencv_setting_dict["serial_device_no_list"] = serial_device_no_list
    opencv_setting_dict["serial_connection_list"] = serial_connection_list

    logger.info("Setting up DearPyGui")

    dpg.create_context()
    dpg.setup_dearpygui()
    dpg.create_viewport(
        title="CV_STUDIO",
        width=editor_width,
        height=editor_height,
        x_pos=0,
        y_pos=0,
        # A vsync'd frame blocks until the monitor refresh with _dpg_lock
        # held, which starves the node update thread.  The render loop paces
        # itself instead (FramePacer), with the lock released.
        vsync=unuse_async_draw,
    )

    # Load Space Grotesk font (minimalist, futuristic, elegant, architectural)
    _font_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "node_editor", "font", "SpaceGrotesk",
    )
    _font_path = os.path.join(_font_dir, "SpaceGrotesk-Medium.otf")
    if not os.path.isfile(_font_path):
        _font_path = os.path.join(_font_dir, "SpaceGrotesk-Regular.otf")

    if os.path.isfile(_font_path):
        with dpg.font_registry():
            default_font = dpg.add_font(_font_path, 18)
            register_overview_fonts(_font_path)
        dpg.bind_font(default_font)
        logger.info(f"Loaded custom font: {os.path.basename(_font_path)}")
    else:
        logger.warning("Space Grotesk font not found, using default DearPyGui font")

    # Viewport must be visible before rendering splash frames.
    dpg.show_viewport(maximized=True)
    if not smoke_test:
        show_splash_screen()

    logger.info("Creating Node Editor")
    menu_dict = OrderedDict(
        {
            "Input": "InputNode",
            "VisionProcess": "ProcessNode",
            "VisionModel": "DLNode",
            "AudioProcess": "AudioProcessNode",
            "AudioModel": "AudioModelNode",
            "DataProcess": "StatsNode",
            "DataModel": "TimeseriesNode",
            "NLPModel": "NLPModelNode",
            "Trigger": "TriggerNode",
            "Agent": "AgentNode",
            "Action": "ActionNode",
            "Overlay": "OverlayNode",
            "Tracking": "TrackerNode",
            "Visual": "VisualNode",
            "Video": "VideoNode",
            "System": "SystemNode",
            "Map": "MapNode",
        }
    )

    current_path = os.path.dirname(os.path.abspath(__file__))

    node_editor = DpgNodeEditor(
        width=editor_width,
        height=editor_height,
        opencv_setting_dict=opencv_setting_dict,
        menu_dict=menu_dict,
        use_debug_print=use_debug_print,
        node_dir=current_path + "/node",
    )

    # Store reference for nodes that need access to the editor (e.g., Deploy node)
    sys.modules[__name__]._node_editor_ref = node_editor

    logger.info("Starting main event loop")
    event_loop = None
    worker_stopped = None
    if not unuse_async_draw:
        logger.info("Async draw is enabled")
        # asyncio.get_event_loop() is deprecated in Python 3.10+ when there is
        # no running loop (raises DeprecationWarning / RuntimeError on 3.12+).
        # Create and install a new event loop explicitly instead.
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        worker_stopped = threading.Event()
        event_loop.run_in_executor(
            None, async_main, node_editor, queue_manager, worker_stopped
        )
        # Use a manual render loop instead of dpg.start_dearpygui() so that
        # each frame render holds _dpg_lock.  dpg.start_dearpygui() runs the
        # DPG C-level loop without acquiring _dpg_lock, which races with the
        # worker thread's DPG calls and causes a segfault on Linux.
        # ui_frame() puts the render ahead of the worker thread, which would
        # otherwise keep the lock for a whole pass over the graph.  The frame
        # rate is capped by the pacer, which waits with the lock released:
        # that is when the worker thread updates the nodes.
        pacer = FramePacer(ui_fps)
        rendered_frames = 0
        smoke_steps = _smoke_test_steps(node_editor) if smoke_test else None
        while dpg.is_dearpygui_running():
            with _dpg_lock.ui_frame():
                dpg.render_dearpygui_frame()
            # Delete items queued during the frame (textures of deleted nodes):
            # destroying them while the frame is being drawn segfaults on Linux.
            process_deferred_deletes()
            rendered_frames += 1
            if smoke_steps is not None and next(smoke_steps, None) is None:
                smoke_steps = None
            if smoke_test and smoke_steps is None and rendered_frames >= smoke_test_frames:
                logger.info("Smoke-test: %d frames rendered, stopping", rendered_frames)
                dpg.stop_dearpygui()
            pacer.wait()

    else:
        logger.info("Async draw is disabled")
        # Create queue-backed dictionaries
        node_image_dict = QueueBackedDict(queue_manager, "image")
        node_result_dict = QueueBackedDict(queue_manager, "json")
        node_audio_dict = QueueBackedDict(queue_manager, "audio")
        # Inject live references for agent tool discovery
        node_result_dict._node_instances = node_editor._node_instances_list
        node_result_dict._node_link_list = node_editor._node_link_list
        
        rendered_frames = 0
        while dpg.is_dearpygui_running():
            update_node_info(
                node_editor,
                node_image_dict,
                node_result_dict,
                node_audio_dict,
                mode_async=False,
            )
            dpg.render_dearpygui_frame()
            process_deferred_deletes()
            rendered_frames += 1
            if smoke_test and rendered_frames >= smoke_test_frames:
                logger.info("Smoke-test: %d frames rendered, stopping", rendered_frames)
                dpg.stop_dearpygui()

    logger.info("Terminating process")

    # Drain any item deletion queued during the last frame
    process_deferred_deletes()

    # Signal async loop to stop before closing nodes so update calls cease
    node_editor.set_terminate_flag()
    if worker_stopped is not None and not worker_stopped.wait(timeout=10):
        logger.warning("Node update thread still running; shutting down anyway")

    logger.info("Closing all nodes")
    node_list = node_editor.get_node_list()
    for node_id_name in node_list:
        node_id, _ = node_id_name.split(":")
        node_instance = node_editor.get_node_instances(node_id_name)
        if node_instance is not None:
            node_instance.close(node_id)
        else:
            logger.warning(f"No instance found for node {node_id_name}, skipping close")

    logger.info("Releasing all video captures")
    for camera_capture in camera_capture_list:
        camera_capture.release()

    if event_loop is not None:
        logger.info("Stopping event loop")
        event_loop.stop()

    logger.info("Destroying DearPyGui context")
    dpg.destroy_context()

    logger.info("CV_STUDIO shutdown complete")

    if smoke_test:
        node_count = len(node_editor._node_factory_list)
        if node_count == 0:
            raise RuntimeError("Smoke-test failed: no node factory was discovered")
        _smoke_test_distillation()
        logger.info("SMOKE TEST PASSED: %d node types discovered", node_count)
        print(f"SMOKE TEST PASSED: {node_count} node types discovered", flush=True)


def _run():
    """Run main() and translate any failure into a non-zero exit code.

    Without this, an exception raised during startup in the frozen .exe may
    only show up in a console window that closes immediately, and CI would
    have no reliable signal that the build is broken.
    """
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        logger.exception("CV_STUDIO terminated with an unhandled exception")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Enable multiprocessing support for frozen executables (PyInstaller)
    # This must be called before any multiprocessing code runs
    # On Windows, when the executable spawns child processes for multiprocessing,
    # they will re-execute this script with special arguments that freeze_support() handles
    multiprocessing.freeze_support()
    _run()
