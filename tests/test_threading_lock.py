#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test thread-safe locking for DearPyGUI operations"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_dpg_lock_exists():
    """Test that the DearPyGUI lock exists in util.py"""
    util_path = os.path.join(
        os.path.dirname(__file__), '..', 'node_editor', 'util.py'
    )
    assert os.path.exists(util_path), "util.py should exist"
    
    with open(util_path, 'r') as f:
        content = f.read()
    
    # Check that the lock is defined
    assert 'import threading' in content, "threading module should be imported"
    assert '_dpg_lock = UiPriorityLock()' in content, "_dpg_lock should be defined as a UiPriorityLock"
    scheduling_path = os.path.join(os.path.dirname(util_path), 'scheduling.py')
    with open(scheduling_path, 'r') as f:
        assert 'self._lock = threading.RLock()' in f.read(), "UiPriorityLock should be reentrant"
    print("✓ _dpg_lock exists and is a reentrant lock")


def test_dpg_set_value_uses_lock():
    """Test that dpg_set_value uses the lock"""
    util_path = os.path.join(
        os.path.dirname(__file__), '..', 'node_editor', 'util.py'
    )
    assert os.path.exists(util_path), "util.py should exist"
    
    with open(util_path, 'r') as f:
        content = f.read()
    
    # Find dpg_set_value function
    assert 'def dpg_set_value(tag, value):' in content, "dpg_set_value function should exist"
    
    # Check that the function uses the lock anywhere after its definition
    func_start = content.find('def dpg_set_value(tag, value):')
    func_section = content[func_start:func_start + 200]  # Check next 200 chars
    
    # Verify it uses the lock
    assert 'with _dpg_lock:' in func_section, "dpg_set_value should use _dpg_lock"
    print("✓ dpg_set_value uses _dpg_lock")


def test_dpg_get_value_uses_lock():
    """Test that dpg_get_value uses the lock"""
    util_path = os.path.join(
        os.path.dirname(__file__), '..', 'node_editor', 'util.py'
    )
    assert os.path.exists(util_path), "util.py should exist"
    
    with open(util_path, 'r') as f:
        content = f.read()
    
    # Find dpg_get_value function
    assert 'def dpg_get_value(tag):' in content, "dpg_get_value function should exist"
    
    # Check that the function uses the lock anywhere after its definition
    func_start = content.find('def dpg_get_value(tag):')
    func_section = content[func_start:func_start + 200]  # Check next 200 chars
    
    # Verify it uses the lock
    assert 'with _dpg_lock:' in func_section, "dpg_get_value should use _dpg_lock"
    print("✓ dpg_get_value uses _dpg_lock")


def test_callback_add_node_uses_lock():
    """Test that _callback_add_node uses the lock"""
    node_editor_path = os.path.join(
        os.path.dirname(__file__), '..', 'node_editor', 'node_editor.py'
    )
    assert os.path.exists(node_editor_path), "node_editor.py should exist"
    
    with open(node_editor_path, 'r') as f:
        content = f.read()
    
    # Verify _dpg_lock is imported
    assert 'from .util import _dpg_lock' in content, "_dpg_lock should be imported from util"
    
    # Find _callback_add_node method
    assert 'def _callback_add_node(self, sender, data, user_data):' in content
    
    # Check that the method uses the lock anywhere after its definition
    method_start = content.find('def _callback_add_node(self, sender, data, user_data):')
    method_section = content[method_start:method_start + 500]  # Check next 500 chars
    
    # Verify it uses the lock
    assert 'with _dpg_lock:' in method_section, "_callback_add_node should use _dpg_lock"
    print("✓ _callback_add_node uses _dpg_lock")


def test_callback_link_uses_lock():
    """Test that _callback_link uses the lock"""
    node_editor_path = os.path.join(
        os.path.dirname(__file__), '..', 'node_editor', 'node_editor.py'
    )
    assert os.path.exists(node_editor_path), "node_editor.py should exist"
    
    with open(node_editor_path, 'r') as f:
        content = f.read()
    
    # Find _callback_link method
    assert 'def _callback_link(self, sender, data):' in content
    
    # Check that the method uses the lock anywhere after its definition
    method_start = content.find('def _callback_link(self, sender, data):')
    method_section = content[method_start:method_start + 500]  # Check next 500 chars
    
    # Verify it uses the lock
    assert 'with _dpg_lock:' in method_section, "_callback_link should use _dpg_lock"
    print("✓ _callback_link uses _dpg_lock")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing Thread-Safe DearPyGUI Locking")
    print("=" * 60)
    
    try:
        test_dpg_lock_exists()
        test_dpg_set_value_uses_lock()
        test_dpg_get_value_uses_lock()
        test_callback_add_node_uses_lock()
        test_callback_link_uses_lock()
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
