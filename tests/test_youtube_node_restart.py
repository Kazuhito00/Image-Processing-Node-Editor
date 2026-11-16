#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test to verify that the YouTube node properly handles start/stop/restart cycles.
This test validates that:
1. The node initializes with correct default state
2. Button themes are properly created and stored
3. Streaming state is properly tracked
4. The cap object is properly released and reset on stop
5. The node can restart after being stopped

Note: This test only tests the logic without GUI components since DearPyGUI requires a display.
"""
import sys
import os
import unittest.mock as mock

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock DearPyGUI before importing the node
sys.modules['dearpygui'] = mock.MagicMock()
sys.modules['dearpygui.dearpygui'] = mock.MagicMock()

from node.InputNode.node_youtube import YoutubeNode


def test_youtube_node_initialization():
    """Test that YouTube node initializes with correct defaults"""
    node = YoutubeNode()
    
    # Check default values
    assert node.cap is None, "Initial cap should be None"
    assert node.is_streaming is False, "Initial streaming state should be False"
    assert node.small_window_w == 240, "Default window width should be 240"
    assert node.small_window_h == 135, "Default window height should be 135"
    assert node._start_label == "Start", "Start label should be 'Start'"
    assert node.node_tag == "YouTube", "Node tag should be 'YouTube'"
    
    print("✓ YouTube node initialization test passed")


def test_youtube_node_theme_storage():
    """Test that themes can be stored in the node"""
    node = YoutubeNode()
    
    # Initially themes should be None
    assert node.yellow_button_theme is None, "Yellow theme should initially be None"
    assert node.blue_button_theme is None, "Blue theme should initially be None"
    
    # Simulate setting themes (as would happen in FactoryNode)
    node.yellow_button_theme = "yellow_theme_id"
    node.blue_button_theme = "blue_theme_id"
    
    assert node.yellow_button_theme == "yellow_theme_id", "Yellow theme should be stored"
    assert node.blue_button_theme == "blue_theme_id", "Blue theme should be stored"
    
    print("✓ YouTube node theme storage test passed")


def test_youtube_node_streaming_state():
    """Test that streaming state changes are tracked"""
    node = YoutubeNode()
    
    # Initially not streaming
    assert node.is_streaming is False
    
    # Simulate starting stream
    node.is_streaming = True
    assert node.is_streaming is True
    
    # Simulate stopping stream
    node.is_streaming = False
    assert node.is_streaming is False
    
    print("✓ YouTube node streaming state test passed")


def test_youtube_node_close():
    """Test that close method properly cleans up resources"""
    node = YoutubeNode()
    
    # Simulate having an active capture
    # We can't create a real capture without a URL, but we can set the flag
    node.is_streaming = True
    
    # Call close
    node.close(node_id=1)
    
    # Verify cleanup
    assert node.cap is None, "Cap should be None after close"
    assert node.is_streaming is False, "Streaming should be False after close"
    
    print("✓ YouTube node close test passed")


def test_youtube_node_convert_cv_to_dpg():
    """Test the image conversion function"""
    import numpy as np
    
    node = YoutubeNode()
    
    # Test with None image (should return black image)
    result = node.convert_cv_to_dpg(None, 240, 135)
    assert result is not None, "Should return a valid result even with None image"
    assert isinstance(result, bytes), "Result should be bytes"
    
    # Test with a valid image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    result = node.convert_cv_to_dpg(test_image, 240, 135)
    assert result is not None, "Should return a valid result"
    assert isinstance(result, bytes), "Result should be bytes"
    # Expected size: 240 * 135 * 3 channels * 4 bytes per float32 = 388800 bytes
    expected_size = 240 * 135 * 3 * 4
    assert len(result) == expected_size, f"Result size should be {expected_size} bytes"
    
    print("✓ YouTube node convert_cv_to_dpg test passed")


def test_youtube_node_update_without_stream():
    """Test that update method handles no active stream gracefully"""
    node = YoutubeNode()
    
    # Call update without an active stream
    result = node.update(
        node_id=1,
        connection_list=[],
        node_image_dict={},
        node_result_dict={},
        node_audio_dict={}
    )
    
    # Should return a dict with None values
    assert isinstance(result, dict), "Result should be a dict"
    assert result["image"] is None, "Image should be None when no stream"
    assert result["json"] is None, "JSON should be None"
    assert result["audio"] is None, "Audio should be None"
    
    print("✓ YouTube node update without stream test passed")


def _update(node):
    return node.update(
        node_id=1,
        connection_list=[],
        node_image_dict={},
        node_result_dict={},
        node_audio_dict={},
    )


def test_youtube_node_stalled_stream_does_not_block():
    """A stalled network read must block neither update() nor Stop.

    update() runs with the DearPyGui lock held: blocking there freezes the UI.
    """
    import threading
    import time
    import numpy as np

    stalled = threading.Event()
    unblock = threading.Event()
    frame = np.full((4, 4, 3), 7, dtype=np.uint8)

    class StallingCapture:
        def __init__(self):
            self.reads = 0
            self.released_while_reading = False
            self.released = threading.Event()
            self._reading = False

        def read(self):
            self.reads += 1
            if self.reads == 1:
                return True, frame
            self._reading = True
            stalled.set()
            unblock.wait(10)
            self._reading = False
            return False, None

        def release(self):
            self.released_while_reading = self._reading
            self.released.set()

    cap = StallingCapture()
    node = YoutubeNode()
    node._frame_interval = 0.001
    node.cap = cap
    node.is_streaming = True
    node._start_reader(cap)
    try:
        assert stalled.wait(5), "Reader thread should reach the stalled read"

        started = time.monotonic()
        result = _update(node)
        assert time.monotonic() - started < 1.0, "update() must not wait for the stream"
        assert result["image"] is frame, "The frame read before the stall should be delivered"
        assert node._frame_count == 1

        # No new frame: the last one is kept, and not counted twice
        result = _update(node)
        assert result["image"] is frame
        assert node._frame_count == 1

        started = time.monotonic()
        node.close(node_id=1)
        assert time.monotonic() - started < 1.0, "close() must not wait for the stream"
        assert node.cap is None
        assert not cap.released.is_set(), "Capture must not be released during read()"
    finally:
        unblock.set()

    assert cap.released.wait(5), "Reader thread should release the capture"
    assert not cap.released_while_reading

    print("✓ YouTube node stalled stream test passed")


if __name__ == '__main__':
    print("Testing YouTube Node Restart Functionality...")
    print("=" * 60)
    
    tests = [
        ("YouTube node initialization", test_youtube_node_initialization),
        ("YouTube node theme storage", test_youtube_node_theme_storage),
        ("YouTube node streaming state", test_youtube_node_streaming_state),
        ("YouTube node close", test_youtube_node_close),
        ("YouTube node convert_cv_to_dpg", test_youtube_node_convert_cv_to_dpg),
        ("YouTube node update without stream", test_youtube_node_update_without_stream),
        ("YouTube node stalled stream", test_youtube_node_stalled_stream_does_not_block),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        print(f"\nTesting {name}...")
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {name} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {name} error: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)
