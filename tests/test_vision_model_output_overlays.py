#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests to verify that vision models return images with overlays"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_face_detection_returns_overlay():
    """Test that face detection node returns image with overlays"""
    
    file_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'node', 'DLNode', 'node_face_detection.py'
    )
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check that the return statement returns debug_frame, not raw frame
    assert 'return {"image": debug_frame if debug_frame is not None else frame' in content, \
        "Face detection should return debug_frame with overlays"
    
    # Check that debug_frame is created and used
    assert 'debug_frame = copy.deepcopy(frame)' in content, \
        "Face detection should create debug_frame"
    assert 'debug_frame = self.draw_face_detection_info' in content, \
        "Face detection should draw overlays on debug_frame"


def test_semantic_segmentation_returns_overlay():
    """Test that semantic segmentation node returns image with overlays"""
    
    file_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'node', 'DLNode', 'node_semantic_segmentation.py'
    )
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check that the return statement returns debug_frame, not raw frame
    assert 'return {"image": debug_frame if debug_frame is not None else frame' in content, \
        "Semantic segmentation should return debug_frame with overlays"
    
    # Check that debug_frame is created and used
    assert 'debug_frame = self.draw_semantic_segmentation_info' in content or \
           'debug_frame = self.draw_yolov8_seg_contours' in content, \
        "Semantic segmentation should draw overlays on debug_frame"


def test_pose_estimation_returns_overlay():
    """Test that pose estimation node returns image with overlays"""
    
    file_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'node', 'DLNode', 'node_pose_estimation.py'
    )
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check that the return statement returns debug_frame, not raw frame
    assert 'return {"image": debug_frame if debug_frame is not None else frame' in content, \
        "Pose estimation should return debug_frame with overlays"
    
    # Check that debug_frame is created and used
    assert 'debug_frame = copy.deepcopy(frame)' in content, \
        "Pose estimation should create debug_frame"
    assert 'debug_frame = self.draw_pose_estimation_info' in content, \
        "Pose estimation should draw overlays on debug_frame"


def test_object_detection_returns_overlay():
    """Test that object detection node returns image with overlays"""
    
    file_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'node', 'DLNode', 'node_object_detection.py'
    )
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check that the return statement returns debug_frame, not raw frame
    assert 'data["image"] = debug_frame if debug_frame is not None else frame' in content, \
        "Object detection should return debug_frame with overlays"
    
    # Check that debug_frame is created and used
    assert 'debug_frame = copy.deepcopy(frame)' in content, \
        "Object detection should create debug_frame"
    assert 'debug_frame = self.draw_object_detection_info' in content, \
        "Object detection should draw overlays on debug_frame"


def test_classification_returns_overlay():
    """Test that classification node returns image with overlays"""
    
    file_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'node', 'DLNode', 'node_classification.py'
    )
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check that the return statement returns output_frame with overlays
    assert 'return {"image": output_frame' in content, \
        "Classification should return output_frame with overlays"
    
    # Check that debug_frame is created and used
    assert 'debug_frame = copy.deepcopy(frame)' in content, \
        "Classification should create debug_frame"
    assert 'debug_frame = self.draw_classification_info' in content or \
           'debug_frame = self.draw_classification_with_od_info' in content, \
        "Classification should draw overlays on debug_frame"
    assert 'output_frame = debug_frame' in content, \
        "Classification should assign debug_frame to output_frame"


def test_monocular_depth_returns_processed():
    """Test that monocular depth estimation returns processed depth map"""
    
    file_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'node', 'DLNode', 'node_monocular_depth_estimation.py'
    )
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check that it returns the processed frame (depth map)
    assert 'frame = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)' in content, \
        "Monocular depth should convert depth_map to frame"
    assert 'return {"image": frame' in content, \
        "Monocular depth should return processed frame"


def test_llie_returns_processed():
    """Test that LLIE returns enhanced image"""
    
    file_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'node', 'DLNode', 'node_low_light_image_enhancement.py'
    )
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check that it returns the processed frame (enhanced image)
    assert 'frame = run_unlocked(' in content and \
        'self._model_instance[model_name_with_provider], frame' in content, \
        "LLIE should process frame through model"
    assert 'return {"image": frame' in content, \
        "LLIE should return processed frame"


def test_all_vision_models_have_update_method():
    """Test that all vision model nodes have update method that returns dict"""
    
    vision_models = [
        'node_face_detection.py',
        'node_semantic_segmentation.py',
        'node_pose_estimation.py',
        'node_monocular_depth_estimation.py',
        'node_low_light_image_enhancement.py',
        'node_object_detection.py',
        'node_classification.py',
    ]
    
    for model_file in vision_models:
        file_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'node', 'DLNode', model_file
        )
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check that update method exists
        assert 'def update(' in content, \
            f"{model_file} should have update method"
        
        # Check that it returns a dict with image key
        assert ('return {"image":' in content or 
                'return {' in content or
                'data["image"]' in content), \
            f"{model_file} should return a dict with image key"
        assert '"image"' in content, \
            f"{model_file} should have 'image' key in return dict"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
