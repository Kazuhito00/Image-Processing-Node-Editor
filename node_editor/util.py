#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import logging
import numpy as np
import threading

import dearpygui.dearpygui as dpg

from node_editor.scheduling import UiPriorityLock


# Global lock for thread-safe DearPyGUI operations.
# Reentrant: nested DearPyGUI helpers are called within the same thread.
# This protects against race conditions between:
# - Main thread: rendering frames (see main.py)
# - Worker thread: updating nodes via async_main() in thread executor
# - DearPyGui callback thread: menus, links, node deletion
_dpg_lock = UiPriorityLock()


def run_unlocked(func, *args, **kwargs):
    """Call ``func`` with ``_dpg_lock`` released and return its result.

    Nodes are updated with the lock held, which freezes the interface for as
    long as the update lasts.  Long computations (model inference) go through
    this helper so that frames keep being drawn meanwhile.

    ``func`` must not call DearPyGui.  The node may be deleted while it runs:
    the caller must only touch DearPyGui items through the existence-checking
    helpers of this module afterwards.
    """
    with _dpg_lock.released():
        return func(*args, **kwargs)


logger = logging.getLogger(__name__)


def check_camera_connection(max_device_count=4, is_debug=False):
    device_no_list = []

    for device_no in range(0, max_device_count):
        if is_debug:
            print('Check Device No:' + str(device_no).zfill(2), end='')

        cap = cv2.VideoCapture(device_no)
        ret, _ = cap.read()
        if ret:
            device_no_list.append(device_no)
            if is_debug:
                print(' -> Find')
        else:
            if is_debug:
                print(' -> None')

    return device_no_list

def check_serial_connection(is_debug=False):
    import glob
    import serial
    import sys
    serial_device_no_list=[]
    serial_device_no_list=[]
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')

    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            serial_device_no_list.append(port)
        except (OSError, serial.SerialException):
            pass
    return serial_device_no_list

def dpg_set_value(tag, value):
    with _dpg_lock:
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, value)


def dpg_get_value(tag):
    value = None
    with _dpg_lock:
        if dpg.does_item_exist(tag):
            value = dpg.get_value(tag)
    return value


def dpg_configure_item(tag, **kwargs):
    """Thread-safe wrapper for dpg.configure_item.

    Silently skips when the item no longer exists (e.g. the node was deleted
    between the update snapshot and the configure call) and logs all errors so
    that crashes are never silent.
    """
    with _dpg_lock:
        try:
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, **kwargs)
            else:
                logger.debug("dpg_configure_item: tag %s does not exist, skipping", tag)
        except Exception as exc:
            logger.error(
                "dpg_configure_item: failed for tag %s: %s",
                tag, exc, exc_info=True,
            )


def dpg_delete_item(tag):
    """Thread-safe wrapper for dpg.delete_item.

    Silently skips when the item no longer exists and logs all errors so that
    crashes are never silent.
    """
    with _dpg_lock:
        try:
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
            else:
                logger.debug("dpg_delete_item: tag %s does not exist, skipping", tag)
        except Exception as exc:
            logger.error(
                "dpg_delete_item: failed for tag %s: %s",
                tag, exc, exc_info=True,
            )


# Items whose deletion must happen between two rendered frames.
# Deleting a texture from a DearPyGui callback deletes it *during* the frame,
# while the GPU draw list still references it; on Linux (OpenGL backend) this
# segfaults.  Such items are queued here instead and deleted by
# process_deferred_deletes(), which the render loop calls once the frame has
# been fully rendered.
_deferred_delete_queue = []
_deferred_delete_lock = threading.Lock()


def schedule_deferred_delete(item, description=None):
    """Queue a DPG item for deletion after the current frame has been rendered.

    Args:
        item: the item id (or tag) to delete.
        description: optional human readable label used for logging.
    """
    with _deferred_delete_lock:
        _deferred_delete_queue.append((item, description))
    logger.debug(
        "schedule_deferred_delete: queued item %s (%s) for post-frame deletion",
        item, description,
    )


def process_deferred_deletes():
    """Delete every item queued by schedule_deferred_delete().

    Must be called from the render loop, outside of dpg.render_dearpygui_frame(),
    so no draw command still references the deleted items.

    Returns:
        The number of items actually deleted.
    """
    with _deferred_delete_lock:
        if not _deferred_delete_queue:
            return 0
        pending = list(_deferred_delete_queue)
        _deferred_delete_queue.clear()

    deleted = 0
    with _dpg_lock:
        for item, description in pending:
            try:
                if not dpg.does_item_exist(item):
                    logger.debug(
                        "process_deferred_deletes: item %s (%s) no longer exists",
                        item, description,
                    )
                    continue
                dpg.delete_item(item)
                deleted += 1
                logger.debug(
                    "process_deferred_deletes: deleted item %s (%s)",
                    item, description,
                )
            except Exception as exc:
                logger.error(
                    "process_deferred_deletes: failed to delete item %s (%s): %s",
                    item, description, exc, exc_info=True,
                )
    return deleted
