#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for ObjChart node"""

import pytest
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_obj_chart_node_import():
    """Test that ObjChart node can be imported"""
    from node.VisualNode.node_obj_chart import FactoryNode, Node
    
    assert FactoryNode.node_label == 'Chart'
    assert FactoryNode.node_tag == 'ObjChart'
    assert Node.node_label == 'Chart'
    assert Node.node_tag == 'ObjChart'


def test_obj_chart_class_dropdown_items():
    """Test that class dropdown items contain class names"""
    from node.VisualNode.node_obj_chart import get_class_dropdown_items
    
    items = get_class_dropdown_items()
    
    # Check that "All" is the first item
    assert items[0] == "All"
    
    # Check that items contain class names with IDs
    assert "0: person" in items
    assert "2: car" in items
    assert "1: bicycle" in items
    
    # Check that all COCO classes are present (80 classes + "All")
    assert len(items) == 81


def test_obj_chart_dict_dropdown_items():
    """Test that dict dropdown items are built from dictionary keys."""
    from node.VisualNode.node_obj_chart import get_dict_dropdown_items

    data = {"cpu_percent": 10, "memory_percent": 20, 3: 30}
    items = get_dict_dropdown_items(data)

    assert items == ["cpu_percent", "memory_percent", "3"]

    assert get_dict_dropdown_items(None) == []
    assert get_dict_dropdown_items(["cpu_percent"]) == []


def test_obj_chart_time_bucket():
    """Test that time bucket calculation works correctly"""
    from node.VisualNode.node_obj_chart import Node
    
    node = Node(opencv_setting_dict={
        'process_height': 400,
        'process_width': 600
    })
    
    # Test minute bucket
    minute_bucket = node.get_time_bucket("minute")
    assert minute_bucket.second == 0
    assert minute_bucket.microsecond == 0
    
    # Test hour bucket
    hour_bucket = node.get_time_bucket("hour")
    assert hour_bucket.minute == 0
    assert hour_bucket.second == 0
    assert hour_bucket.microsecond == 0


def test_obj_chart_render_empty():
    """Test that chart renders even with no data"""
    from node.VisualNode.node_obj_chart import Node
    import numpy as np
    
    node = Node(opencv_setting_dict={
        'process_height': 400,
        'process_width': 600
    })
    
    # Render chart with no data (default bar chart)
    chart_image = node.render_chart("minute", ["All"], {}, "bar")
    
    assert chart_image is not None
    assert isinstance(chart_image, np.ndarray)
    assert len(chart_image.shape) == 3  # Should be a color image
    assert chart_image.shape[2] == 3  # Should have 3 channels (BGR)


def test_obj_chart_accumulation():
    """Test that detection counts accumulate correctly"""
    from node.VisualNode.node_obj_chart import Node
    
    node = Node(opencv_setting_dict={
        'process_height': 400,
        'process_width': 600
    })
    
    # Simulate some detections
    bucket = node.get_time_bucket("minute")
    
    # Add counts for class 0
    node.time_counts[0][bucket] += 5
    node.time_counts["All"][bucket] += 5
    
    # Add counts for class 1
    node.time_counts[1][bucket] += 3
    node.time_counts["All"][bucket] += 3
    
    # Verify counts
    assert node.time_counts[0][bucket] == 5
    assert node.time_counts[1][bucket] == 3
    assert node.time_counts["All"][bucket] == 8


def test_obj_chart_render_with_data():
    """Test that chart renders correctly with accumulated data"""
    from node.VisualNode.node_obj_chart import Node
    import numpy as np
    
    node = Node(opencv_setting_dict={
        'process_height': 400,
        'process_width': 600
    })
    
    # Add some test data
    bucket = node.get_time_bucket("minute")
    node.time_counts[0][bucket] = 10
    node.time_counts[1][bucket] = 5
    node.time_counts["All"][bucket] = 15
    
    # Render chart with data (bar chart)
    chart_image = node.render_chart("minute", [0, 1], {
        "0": "person",
        "1": "car"
    }, "bar")
    
    assert chart_image is not None
    assert isinstance(chart_image, np.ndarray)
    assert len(chart_image.shape) == 3
    assert chart_image.shape[2] == 3


def test_obj_chart_render_line_chart():
    """Test that line chart renders correctly"""
    from node.VisualNode.node_obj_chart import Node
    import numpy as np
    
    node = Node(opencv_setting_dict={
        'process_height': 400,
        'process_width': 600
    })
    
    # Add some test data
    bucket = node.get_time_bucket("minute")
    node.time_counts[0][bucket] = 10
    node.time_counts[1][bucket] = 5
    
    # Render line chart
    chart_image = node.render_chart("minute", [0, 1], {
        "0": "person",
        "1": "car"
    }, "line")
    
    assert chart_image is not None
    assert isinstance(chart_image, np.ndarray)
    assert len(chart_image.shape) == 3
    assert chart_image.shape[2] == 3


def test_obj_chart_24h_cleanup():
    """Test that data older than 24 hours is cleaned up"""
    from node.VisualNode.node_obj_chart import Node
    from datetime import datetime, timedelta
    
    node = Node(opencv_setting_dict={
        'process_height': 400,
        'process_width': 600
    })
    
    # Add some old data (25 hours ago)
    old_bucket = datetime.now() - timedelta(hours=25)
    old_bucket = old_bucket.replace(second=0, microsecond=0)
    node.time_counts[0][old_bucket] = 100
    node.time_counts["All"][old_bucket] = 100
    
    # Add some recent data
    recent_bucket = node.get_time_bucket("minute")
    node.time_counts[0][recent_bucket] = 10
    node.time_counts["All"][recent_bucket] = 10
    
    # Verify old data exists before cleanup
    assert old_bucket in node.time_counts[0]
    assert old_bucket in node.time_counts["All"]
    
    # Run cleanup
    node.cleanup_old_data()
    
    # Verify old data is removed
    assert old_bucket not in node.time_counts[0]
    
    # Verify recent data still exists
    assert recent_bucket in node.time_counts[0]
    assert node.time_counts[0][recent_bucket] == 10


def test_obj_chart_coco_class_names_fallback():
    """Test that COCO class names are used when class_names_dict is empty"""
    from node.VisualNode.node_obj_chart import Node
    import numpy as np
    
    node = Node(opencv_setting_dict={
        'process_height': 400,
        'process_width': 600
    })
    
    # Add test data for COCO classes
    bucket = node.get_time_bucket("minute")
    node.time_counts[0][bucket] = 10  # person
    node.time_counts[2][bucket] = 5   # car
    node.time_counts[56][bucket] = 3  # chair
    
    # Render chart with EMPTY class_names_dict (should use COCO names)
    chart_image = node.render_chart("minute", [0, 2, 56], {}, "bar")
    
    assert chart_image is not None
    assert isinstance(chart_image, np.ndarray)
    # The chart should be generated successfully with COCO class names
    # (person, car, chair) even without class_names_dict


def test_obj_chart_custom_class_names_override():
    """Test that custom class names from detection JSON override COCO names"""
    from node.VisualNode.node_obj_chart import Node
    import numpy as np
    
    node = Node(opencv_setting_dict={
        'process_height': 400,
        'process_width': 600
    })
    
    # Add test data
    bucket = node.get_time_bucket("minute")
    node.time_counts[0][bucket] = 10
    node.time_counts[1][bucket] = 5
    
    # Render with custom class names that override COCO names
    custom_names = {"0": "custom_person", "1": "custom_vehicle"}
    chart_image = node.render_chart("minute", [0, 1], custom_names, "line")
    
    assert chart_image is not None
    assert isinstance(chart_image, np.ndarray)
    # The chart should use custom names instead of COCO names


def test_obj_chart_microphone_db_support():
    """Test that objchart can handle microphone dB intensity data"""
    from node.VisualNode.node_obj_chart import Node
    import numpy as np
    
    node = Node(opencv_setting_dict={
        'process_height': 400,
        'process_width': 600
    })
    
    # Simulate microphone dB intensity JSON input
    microphone_json = {
        'timestamp': 1234567890.0,
        'sample_rate': 44100,
        'channels': 1,
        'chunk_duration': 1.0,
        'output_mode': 'dB Intensity',
        'samples': 1,
        'db_value': -25.5
    }
    
    # Verify objchart recognizes this as dB data
    assert 'db_value' in microphone_json
    assert 'output_mode' in microphone_json
    
    # Store dB value
    bucket = node.get_time_bucket("minute")
    node.time_counts["dB"][bucket] = microphone_json['db_value']
    
    # Verify storage
    assert "dB" in node.time_counts
    assert node.time_counts["dB"][bucket] == -25.5
    
    # Render chart with dB data
    chart_image = node.render_chart("minute", ["dB"], {"dB": "Decibel Intensity"}, "line")
    
    assert chart_image is not None
    assert isinstance(chart_image, np.ndarray)
    assert len(chart_image.shape) == 3
    assert chart_image.shape[2] == 3


def test_obj_chart_plots_distillation_metrics_and_leaves_unmeasured_ones_out(monkeypatch):
    """A metric that is None is unknown: it must not be drawn at zero"""
    import dearpygui.dearpygui as dpg
    import matplotlib.axes
    import numpy as np
    from datetime import timedelta
    from node.VisualNode.node_obj_chart import FactoryNode

    plotted = {}
    plot = matplotlib.axes.Axes.plot

    def record(axes, x, y, *args, **kwargs):
        plotted[kwargs.get('label')] = list(y)
        return plot(axes, x, y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, 'plot', record)
    dpg.create_context()
    try:
        settings = {'process_height': 400, 'process_width': 600, 'use_pref_counter': False}
        with dpg.node_editor(tag='chart_editor', parent=dpg.add_window()):
            node = FactoryNode().add_node('chart_editor', 1, opencv_setting_dict=settings)
        dpg.set_value('1:ObjChart:ChartTypeValue', 'line')
        connections = [['2:YoloxDistillation:JSON:Output02', '1:ObjChart:JSON:Input02']]
        source = {'class_ids': [0], 'distillation_losses': {'val_f1': None, 'train_samples': 10}}
        results = {'2:YoloxDistillation': source}

        node.update(1, connections, {}, results, None)
        # The first metric is the series shown by default, measured or not
        assert dpg.get_value('1:ObjChart:ClassSlot:0') == 'val_f1'

        # One minute later the metric has been measured
        for series in (node.time_counts, node._avg_accumulators):
            for buckets in series.values():
                for bucket in list(buckets):
                    buckets[bucket - timedelta(minutes=1)] = buckets.pop(bucket)
        source['distillation_losses'] = {'val_f1': 0.7, 'train_samples': 30}
        node.last_render_time = 0
        output = node.update(1, connections, {}, results, None)
    finally:
        dpg.destroy_context()

    assert np.isnan(plotted['val_f1'][0]) and plotted['val_f1'][1] == 0.7
    assert output['json'] == {'val_f1': 0.7, '_classes': ['val_f1']}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
