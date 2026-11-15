#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the node editor overview mode (node_editor/zoom.py)."""
import importlib.util
import os

import pytest

# Loaded from its file: several test modules replace the ``node_editor``
# package in sys.modules with a stub at collection time.
_spec = importlib.util.spec_from_file_location(
    "_node_editor_zoom_under_test",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "node_editor",
        "zoom.py",
    ),
)
zoom_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(zoom_module)

EditorZoom = zoom_module.EditorZoom
to_logical = zoom_module.to_logical
to_view = zoom_module.to_view


class FakeDpg:
    """Minimal in-memory stand-in for the DearPyGui calls used by EditorZoom."""

    def __init__(self):
        self.items = {}

    def add(self, tag, item_type="mvGroup", parent=None, **config):
        self.items[tag] = {
            "type": f"mvAppItemType::{item_type}",
            "children": [],
            "pos": [0, 0],
            "rect_min": [0, 0],
            "visible": True,
            "config": {"show": True, "width": 0, "height": 0, **config},
        }
        if parent is not None:
            self.items[parent]["children"].append(tag)
        return tag

    def does_item_exist(self, tag):
        return tag in self.items

    def get_item_pos(self, tag):
        return list(self.items[tag]["pos"])

    def set_item_pos(self, tag, pos):
        self.items[tag]["pos"] = list(pos)

    def get_item_children(self, tag, slot):
        return list(self.items[tag]["children"])

    def get_item_type(self, tag):
        return self.items[tag]["type"]

    def get_item_configuration(self, tag):
        return dict(self.items[tag]["config"])

    def configure_item(self, tag, **kwargs):
        self.items[tag]["config"].update(kwargs)

    def get_item_width(self, tag):
        return self.items[tag]["config"]["width"]

    def get_item_height(self, tag):
        return self.items[tag]["config"]["height"]

    def is_item_visible(self, tag):
        return self.items[tag]["visible"]

    def get_item_rect_min(self, tag):
        return list(self.items[tag]["rect_min"])


@pytest.fixture
def dpg(monkeypatch):
    fake = FakeDpg()
    monkeypatch.setattr(zoom_module, "dpg", fake)
    return fake


def _add_node(dpg, tag, pos, image_size=(0, 0)):
    """Node with a slider, a preview inside a group, and an already hidden text."""
    dpg.add(tag, "mvNode")
    dpg.set_item_pos(tag, pos)
    dpg.add(f"{tag}:tex", "mvRawTexture", width=240, height=135)
    attr = dpg.add(f"{tag}:attr", "mvNodeAttribute", parent=tag)
    dpg.add(f"{tag}:slider", "mvSliderInt", parent=attr)
    group = dpg.add(f"{tag}:group", parent=attr)
    dpg.add(
        f"{tag}:image", "mvImage", parent=group,
        width=image_size[0], height=image_size[1], texture_tag=f"{tag}:tex",
    )
    dpg.add(f"{tag}:label", "mvText", parent=group)
    dpg.add(f"{tag}:hidden", "mvText", parent=attr, show=False)
    return tag


def _show(dpg, tag):
    return dpg.items[tag]["config"]["show"]


def _size(dpg, tag):
    config = dpg.items[tag]["config"]
    return (config["width"], config["height"])


@pytest.mark.parametrize("scale", [0.7, 0.5, 0.35])
def test_view_mapping_keeps_the_pivot_fixed_and_is_reversible(scale):
    pivot = [300, 200]

    assert to_view(pivot, pivot, scale) == pivot
    assert to_view([500, 200], pivot, 0.5) == [400, 200]
    for logical in ([0, 0], [820, 430], [-150, 990]):
        back = to_logical(to_view(logical, pivot, scale), pivot, scale)
        # Rounding to whole pixels loses at most one logical pixel per axis
        assert abs(back[0] - logical[0]) <= 2 and abs(back[1] - logical[1]) <= 2


def test_zoom_out_compacts_nodes_and_keeps_scaled_previews(dpg):
    node = _add_node(dpg, "1:Blur", [100, 100], image_size=(240, 135))
    other = _add_node(dpg, "2:Canny", [500, 300])
    zoom = EditorZoom()

    assert zoom.zoom_out([node, other]) is True

    assert zoom.scale == 0.7 and zoom.is_zoomed_out
    assert not _show(dpg, "1:Blur:slider")
    assert not _show(dpg, "1:Blur:label")
    # The group holding the preview must stay visible, as must the pins
    assert _show(dpg, "1:Blur:group") and _show(dpg, "1:Blur:attr")
    assert _size(dpg, "1:Blur:image") == (168, 94)
    # An image without explicit size is scaled from its texture size
    assert _size(dpg, "2:Canny:image") == (168, 94)
    # Without a mouse position the pivot is the centre of the graph
    assert dpg.get_item_pos(node) == [160, 130]
    assert dpg.get_item_pos(other) == [440, 270]


def test_zoom_back_to_normal_restores_everything_exactly(dpg):
    tags = [
        _add_node(dpg, "1:Blur", [41, 67], image_size=(240, 135)),
        _add_node(dpg, "2:Canny", [433, 59]),
        _add_node(dpg, "3:Resize", [821, 437]),
    ]
    before = {tag: dict(item["config"]) for tag, item in dpg.items.items()}
    positions = {tag: dpg.get_item_pos(tag) for tag in tags}
    zoom = EditorZoom()

    for _ in range(5):
        zoom.zoom_out(tags)
    assert zoom.scale == 0.35
    assert zoom.zoom_out(tags) is False
    zoom.zoom_in(tags)
    zoom.reset(tags)

    assert not zoom.is_zoomed_out
    assert {tag: dpg.get_item_pos(tag) for tag in tags} == positions
    assert {tag: dict(item["config"]) for tag, item in dpg.items.items()} == before
    # A widget the node itself had hidden is not shown by the restoration
    assert not _show(dpg, "1:Blur:hidden")
    assert zoom.zoom_in(tags) is False


def test_pivot_follows_the_mouse(dpg):
    node = _add_node(dpg, "1:Blur", [100, 100])
    # The editor is panned/offset: grid (100, 100) is drawn at screen (180, 150)
    dpg.items[node]["rect_min"] = [180, 150]
    zoom = EditorZoom()

    zoom.zoom_out([node], mouse_pos=[380, 250])

    # Mouse is over grid point (300, 200), which must not move
    assert dpg.get_item_pos(node) == to_view([100, 100], [300, 200], 0.7)


def test_node_dragged_while_zoomed_out_keeps_its_new_place(dpg):
    node = _add_node(dpg, "1:Blur", [100, 100])
    fixed = _add_node(dpg, "2:Canny", [500, 300])
    zoom = EditorZoom()
    zoom.zoom_out([node, fixed])
    zoom.zoom_out([node, fixed])
    assert zoom.scale == 0.5

    view = dpg.get_item_pos(node)
    dpg.set_item_pos(node, [view[0] + 50, view[1] - 20])

    assert zoom.logical_pos(node) == [200, 60]
    assert zoom.logical_pos(fixed) == [500, 300]
    zoom.reset([node, fixed])
    assert dpg.get_item_pos(node) == [200, 60]
    assert dpg.get_item_pos(fixed) == [500, 300]


def test_node_added_while_zoomed_out_is_compacted_and_restored(dpg):
    first = _add_node(dpg, "1:Blur", [100, 100])
    zoom = EditorZoom()
    zoom.zoom_out([first])
    zoom.zoom_out([first])

    added = _add_node(dpg, "2:Flip", [300, 100])
    zoom.adopt(added)

    assert not _show(dpg, "2:Flip:slider")
    assert _size(dpg, "2:Flip:image") == (120, 68)
    assert dpg.get_item_pos(added) == [200, 100]
    assert zoom.logical_pos(added) == [300, 100]
    assert zoom.view_to_logical([200, 100]) == [300, 100]

    zoom.reset([first, added])
    assert dpg.get_item_pos(added) == [300, 100]
    assert _show(dpg, "2:Flip:slider")
    assert _size(dpg, "2:Flip:image") == (0, 0)


def test_deleted_nodes_are_ignored(dpg):
    kept = _add_node(dpg, "1:Blur", [100, 100])
    deleted = _add_node(dpg, "2:Canny", [500, 300])
    zoom = EditorZoom()
    zoom.zoom_out([kept, deleted])

    for tag in [t for t in dpg.items if t.startswith("2:Canny")]:
        del dpg.items[tag]

    assert zoom.reset([kept]) is True
    assert dpg.get_item_pos(kept) == [100, 100]
    assert _show(dpg, "1:Blur:slider")


def test_adopt_is_a_no_op_at_normal_zoom(dpg):
    node = _add_node(dpg, "1:Blur", [100, 100])
    zoom = EditorZoom()

    zoom.adopt(node)

    assert _show(dpg, "1:Blur:slider")
    assert zoom.logical_pos(node) == [100, 100]
    assert zoom.view_to_logical([10, 20]) == [10, 20]
