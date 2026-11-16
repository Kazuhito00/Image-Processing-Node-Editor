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

    mvNode = "mvNode"
    mvAll = "mvAll"
    mvNodeStyleVar_NodePadding = "NodePadding"
    mvNodeStyleVar_PinCircleRadius = "PinCircleRadius"
    mvNodeStyleVar_LinkThickness = "LinkThickness"
    mvStyleVar_ItemSpacing = "ItemSpacing"
    mvThemeCat_Nodes = "nodes"
    mvThemeCat_Core = "core"

    def __init__(self):
        self.items = {}
        self._next_id = 1000

    def add(self, tag, item_type="mvGroup", parent=None, **config):
        self.items[tag] = {
            "type": f"mvAppItemType::{item_type}",
            "children": [],
            "parent": parent,
            "theme": None,
            "font": None,
            "pos": [0, 0],
            "rect_min": [0, 0],
            "visible": True,
            "config": {"show": True, "width": 0, "height": 0, **config},
        }
        if parent is not None:
            self.items[parent]["children"].append(tag)
        return tag

    def _new(self, item_type, parent=None, **config):
        self._next_id += 1
        return self.add(self._next_id, item_type, parent=parent, **config)

    def add_theme(self):
        return self._new("mvTheme")

    def add_theme_component(self, item_type, parent):
        return self._new("mvThemeComponent", parent=parent, target=item_type)

    def add_theme_style(self, target, x, y=-1, category=None, parent=None):
        return self._new("mvThemeStyle", parent=parent, target=target, value=(x, y))

    def get_item_theme(self, tag):
        return self.items[tag]["theme"]

    def bind_item_theme(self, tag, theme):
        self.items[tag]["theme"] = theme or None

    def bind_item_font(self, tag, font):
        self.items[tag]["font"] = font or None

    def get_item_parent(self, tag):
        return self.items[tag]["parent"]

    def delete_item(self, tag):
        item = self.items.pop(tag)
        for child in item["children"]:
            self.delete_item(child)
        parent = item["parent"]
        if parent in self.items:
            self.items[parent]["children"].remove(tag)

    def styles(self, theme):
        """Overview style values currently held by ``theme``."""
        return {
            self.items[style]["config"]["target"]: self.items[style]["config"]["value"]
            for component in self.items[theme]["children"]
            for style in self.items[component]["children"]
        }

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
    if "editor" not in dpg.items:
        dpg.add("window", "mvWindowAppItem")
        dpg.add("editor", "mvNodeEditor", parent="window")
    dpg.add(tag, "mvNode", parent="editor", label=tag.split(":")[1])
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


@pytest.mark.parametrize("scale", [0.7, 0.5, 0.35, 0.25, 0.18])
def test_view_mapping_keeps_the_pivot_fixed_and_is_reversible(scale):
    pivot = [300, 200]

    assert to_view(pivot, pivot, scale) == pivot
    assert to_view([500, 200], pivot, 0.5) == [400, 200]
    for logical in ([0, 0], [820, 430], [-150, 990]):
        back = to_logical(to_view(logical, pivot, scale), pivot, scale)
        # Rounding to whole pixels loses at most a few logical pixels per axis
        tolerance = 0.5 / scale + 1
        assert abs(back[0] - logical[0]) <= tolerance
        assert abs(back[1] - logical[1]) <= tolerance


@pytest.mark.parametrize("label, expected", [
    ("Blur", "Blur"),
    ("Grayscale", "Grays"),
    ("ObjectDetection", "ObjDe"),
    ("MultiObjectTracking", "MuObT"),
    ("RTSP", "RTSP"),
    ("Text2Speech", "Te2Sp"),
    ("Draw Information", "DraIn"),
    ("A", "A"),
    ("", ""),
])
def test_short_label_keeps_every_word_within_five_characters(label, expected):
    assert zoom_module.short_label(label) == expected
    assert len(zoom_module.short_label(label)) <= 5


def test_titles_are_shortened_while_zoomed_out(dpg):
    node = _add_node(dpg, "1:ObjectDetection", [100, 100])
    short = _add_node(dpg, "2:Blur", [500, 300])
    zoom = EditorZoom()

    zoom.zoom_out([node, short])
    assert dpg.items[node]["config"]["label"] == "ObjDe"
    assert dpg.items[short]["config"]["label"] == "Blur"
    zoom.zoom_out([node, short])
    assert dpg.items[node]["config"]["label"] == "ObjDe"

    zoom.reset([node, short])
    assert dpg.items[node]["config"]["label"] == "ObjectDetection"


def test_title_changed_by_the_node_while_zoomed_out_is_kept(dpg):
    node = _add_node(dpg, "1:ObjectDetection", [100, 100])
    zoom = EditorZoom()
    zoom.zoom_out([node])

    dpg.configure_item(node, label="Renamed")
    zoom.reset([node])

    assert dpg.items[node]["config"]["label"] == "Renamed"


def test_nodes_get_smaller_fonts_and_paddings_as_the_view_moves_away(dpg, monkeypatch):
    monkeypatch.setattr(zoom_module, "_overview_fonts", {0.7: "font15", 0.18: "font8"})
    themed = _add_node(dpg, "1:Blur", [100, 100])
    bare = _add_node(dpg, "2:Canny", [500, 300])
    category_theme = dpg.add_theme()
    dpg.add_theme_component("mvNode", parent=category_theme)
    dpg.bind_item_theme(themed, category_theme)
    zoom = EditorZoom()

    zoom.zoom_out([themed, bare])
    # The category theme is kept (it carries the colours) and extended
    assert dpg.get_item_theme(themed) == category_theme
    assert dpg.styles(category_theme) == {"NodePadding": (6, 6), "ItemSpacing": (6, 3)}
    assert dpg.styles(dpg.get_item_theme(bare))["NodePadding"] == (6, 6)
    assert dpg.items[themed]["font"] == "font15"

    while zoom.zoom_out([themed, bare]):
        pass
    assert dpg.styles(category_theme) == {"NodePadding": (1, 1), "ItemSpacing": (1, 1)}
    # Pins and links are styled through the item holding the editor
    holder_styles = dpg.styles(dpg.get_item_theme("window"))
    assert holder_styles["PinCircleRadius"] == (1.5, -1)
    assert holder_styles["LinkThickness"] == (1.0, -1)
    assert dpg.items[themed]["font"] == "font8"

    zoom.reset([themed, bare])
    assert dpg.get_item_theme(themed) == category_theme
    assert len(dpg.items[category_theme]["children"]) == 1
    assert dpg.get_item_theme(bare) is None and dpg.get_item_theme("window") is None
    assert dpg.items[themed]["font"] is None


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

    for _ in range(7):
        zoom.zoom_out(tags)
    assert zoom.scale == 0.18
    assert zoom.zoom_out(tags) is False
    zoom.zoom_in(tags)
    zoom.reset(tags)

    assert not zoom.is_zoomed_out
    assert {tag: dpg.get_item_pos(tag) for tag in tags} == positions
    assert {tag: dict(item["config"]) for tag, item in dpg.items.items()} == before
    assert all(item["theme"] is None for item in dpg.items.values())
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

    for tag in [t for t in dpg.items if str(t).startswith("2:Canny")]:
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
