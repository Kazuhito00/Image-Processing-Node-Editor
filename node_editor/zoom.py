#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Overview ("zoom out") mode for the node editor.

DearPyGui's node editor has no zoom API, so zooming out is emulated:

* node positions are pulled towards a pivot (the mouse cursor),
* node contents are hidden, leaving the title bar, the pins and the links,
* node titles are shortened (see :func:`short_label`) and, together with the
  node paddings, drawn smaller and smaller as the view moves away,
* image previews are kept and displayed as thumbnails (resized by the GPU,
  the textures themselves are untouched).

Zooming back to 100 % restores every node exactly as it was.  Nothing here
runs per frame: all the work happens when the zoom level changes.

Positions used outside of this module (export, copy, undo) are always
*logical* positions, i.e. the positions the nodes have at 100 %.
"""
import logging
import re

import dearpygui.dearpygui as dpg

logger = logging.getLogger(__name__)

# Index 0 must be 1.0 (normal view); the other levels are overview scales.
ZOOM_LEVELS = (1.0, 0.7, 0.5, 0.35, 0.25, 0.18)

SHORT_LABEL_LENGTH = 5

_MIN_THUMBNAIL_SIZE = 16
_CONTENT_SLOT = 1

# Title font size per overview scale.  Text is shrunk less than the layout so
# that the short names stay readable from far away.
_FONT_SIZES = {0.7: 15, 0.5: 13, 0.35: 11, 0.25: 9, 0.18: 8}

# DearPyGui defaults, scaled down in overview mode
_NODE_PADDING = (8, 8)
_ITEM_SPACING = (8, 4)
_PIN_RADIUS = 4
_LINK_THICKNESS = 3

# scale -> font item, filled by register_overview_fonts()
_overview_fonts = {}

_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


def short_label(label, limit=SHORT_LABEL_LENGTH):
    """Abbreviate a node title to at most ``limit`` characters.

    The letters are shared between the words of the title so that every word
    stays recognisable: ``ObjectDetection`` -> ``ObjDe``, ``ImageConcat`` ->
    ``ImaCo``, ``MultiObjectTracking`` -> ``MuObT``, ``Blur`` -> ``Blur``.
    """
    label = label or ""
    words = _WORD.findall(label)[:limit]
    if not words:
        return label.strip()[:limit]

    taken = [0] * len(words)
    remaining = limit
    while remaining:
        progressed = False
        for index, word in enumerate(words):
            if remaining and taken[index] < len(word):
                taken[index] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return "".join(word[:count] for word, count in zip(words, taken))


def register_overview_fonts(font_path):
    """Load the title fonts of the overview levels.

    Must be called inside a ``dpg.font_registry()`` block.  Without it the
    overview keeps the regular font.
    """
    for scale, size in _FONT_SIZES.items():
        _overview_fonts[scale] = dpg.add_font(font_path, size)


def to_view(logical, pivot, scale):
    """Map a logical (100 %) position to its on-screen position at ``scale``."""
    return [
        int(round(pivot[0] + (logical[0] - pivot[0]) * scale)),
        int(round(pivot[1] + (logical[1] - pivot[1]) * scale)),
    ]


def to_logical(view, pivot, scale):
    """Inverse of :func:`to_view`."""
    return [
        int(round(pivot[0] + (view[0] - pivot[0]) / scale)),
        int(round(pivot[1] + (view[1] - pivot[1]) / scale)),
    ]


class _NodeState:
    __slots__ = ("logical", "view", "hidden", "images", "label", "own_theme")

    def __init__(self, logical):
        self.logical = list(logical)
        # Position this module last gave the node; if the node is found
        # elsewhere, the user dragged it while zoomed out.
        self.view = None
        # Items hidden by the overview mode (and only those)
        self.hidden = []
        # image item -> (configured size to restore, displayed size at 100 %)
        self.images = {}
        # (full title, short title) when the title was shortened
        self.label = None
        # True when the node had no theme and was given the overview one
        self.own_theme = False


class EditorZoom:
    """Zoom-out controller for one DearPyGui node editor.

    All methods must be called with the DearPyGui lock held.
    """

    def __init__(self, levels=ZOOM_LEVELS):
        if not levels or levels[0] != 1.0:
            raise ValueError("the first zoom level must be 1.0")
        self._levels = tuple(levels)
        self._index = 0
        self._pivot = [0, 0]
        self._nodes = {}
        # theme -> component holding the overview paddings of its nodes
        # (_EDITOR -> component thinning the pins and the links)
        self._density = {}
        self._own_theme = None
        # Item holding the editor, given the overview theme because it had none
        self._themed_holder = None

    _EDITOR = "editor"

    @property
    def scale(self):
        return self._levels[self._index]

    @property
    def is_zoomed_out(self):
        return self._index != 0

    def zoom_out(self, node_tags, mouse_pos=None):
        return self._set_level(self._index + 1, node_tags, mouse_pos)

    def zoom_in(self, node_tags):
        return self._set_level(self._index - 1, node_tags)

    def reset(self, node_tags):
        return self._set_level(0, node_tags)

    def adopt(self, node_tag, pos_is_logical=True):
        """Register a node created while zoomed out and show it compacted.

        ``pos_is_logical`` tells whether the node was created at a logical
        position (import, paste, undo) or at an on-screen position (next to
        an existing node).
        """
        if not self.is_zoomed_out or not dpg.does_item_exist(node_tag):
            return
        pos = dpg.get_item_pos(node_tag)
        logical = pos if pos_is_logical else to_logical(pos, self._pivot, self.scale)
        state = _NodeState(logical)
        self._nodes[node_tag] = state
        self._compact(node_tag, state)
        self._apply(node_tag, state)

    def logical_pos(self, node_tag):
        """Position of the node at 100 %, whatever the current zoom level."""
        state = self._nodes.get(node_tag) if self.is_zoomed_out else None
        if state is None:
            return list(dpg.get_item_pos(node_tag))
        self._sync_logical(node_tag, state)
        return list(state.logical)

    def view_to_logical(self, pos):
        """Convert an on-screen position to a logical one."""
        if not self.is_zoomed_out:
            return list(pos)
        return to_logical(pos, self._pivot, self.scale)

    # ------------------------------------------------------------------ #

    def _set_level(self, index, node_tags, mouse_pos=None):
        index = max(0, min(len(self._levels) - 1, index))
        if index == self._index:
            return False

        node_tags = [tag for tag in node_tags if dpg.does_item_exist(tag)]

        if not self.is_zoomed_out:
            # Leaving the normal view: the pivot stays the same until the
            # view is back to 100 %, so the restoration is exact.
            self._pivot = self._find_pivot(node_tags, mouse_pos)
            self._nodes = {}

        for tag in node_tags:
            state = self._nodes.get(tag)
            if state is None:
                pos = dpg.get_item_pos(tag)
                if self.is_zoomed_out:
                    pos = to_logical(pos, self._pivot, self.scale)
                state = self._nodes[tag] = _NodeState(pos)
                self._compact(tag, state)
            else:
                self._sync_logical(tag, state)

        # Forget nodes deleted since the last change
        for tag in [tag for tag in self._nodes if tag not in node_tags]:
            del self._nodes[tag]

        self._index = index
        # The paddings depend on the level: they are rebuilt by _apply()
        self._clear_density()
        if index == 0:
            for tag, state in self._nodes.items():
                self._restore(tag, state)
            self._nodes = {}
            if self._own_theme is not None:
                if dpg.does_item_exist(self._own_theme):
                    dpg.delete_item(self._own_theme)
                self._own_theme = None
        else:
            for tag, state in self._nodes.items():
                self._apply(tag, state)
        return True

    def _find_pivot(self, node_tags, mouse_pos):
        """Pivot in editor (grid) coordinates.

        The editor panning is not exposed by DearPyGui; it is deduced from
        the screen rectangle of a visible node.  Without a mouse position or
        a visible node, the centre of the graph is used.
        """
        if mouse_pos is not None:
            for tag in node_tags:
                try:
                    if not dpg.is_item_visible(tag):
                        continue
                    rect_min = dpg.get_item_rect_min(tag)
                    pos = dpg.get_item_pos(tag)
                except Exception:
                    continue
                return [
                    int(round(mouse_pos[0] - (rect_min[0] - pos[0]))),
                    int(round(mouse_pos[1] - (rect_min[1] - pos[1]))),
                ]

        positions = [dpg.get_item_pos(tag) for tag in node_tags]
        if not positions:
            return [0, 0]
        return [
            int(round(sum(p[0] for p in positions) / len(positions))),
            int(round(sum(p[1] for p in positions) / len(positions))),
        ]

    def _sync_logical(self, tag, state):
        """Update the logical position if the user moved the node."""
        if not dpg.does_item_exist(tag):
            return
        pos = list(dpg.get_item_pos(tag))
        if state.view is not None and pos != state.view:
            state.logical = to_logical(pos, self._pivot, self.scale)
            state.view = pos

    def _apply(self, tag, state):
        scale = self.scale
        state.view = to_view(state.logical, self._pivot, scale)
        dpg.set_item_pos(tag, state.view)
        for image, (_, (width, height)) in state.images.items():
            if dpg.does_item_exist(image):
                dpg.configure_item(
                    image,
                    width=max(_MIN_THUMBNAIL_SIZE, int(round(width * scale))),
                    height=max(_MIN_THUMBNAIL_SIZE, int(round(height * scale))),
                )
        font = _overview_fonts.get(scale)
        if font is not None:
            dpg.bind_item_font(tag, font)
        self._densify(tag, state)

    def _densify(self, tag, state):
        """Shrink the paddings of the node through its theme.

        Themes are shared by all the nodes of a category, and all the nodes
        are zoomed together: one extra component per theme is enough.
        """
        theme = dpg.get_item_theme(tag)
        if not theme:
            if self._own_theme is None:
                self._own_theme = dpg.add_theme()
            theme = self._own_theme
            dpg.bind_item_theme(tag, theme)
            state.own_theme = True
        if theme in self._density:
            return

        scale = self.scale

        def scaled(value):
            return max(1, int(round(value * scale)))

        component = dpg.add_theme_component(dpg.mvNode, parent=theme)
        dpg.add_theme_style(
            dpg.mvNodeStyleVar_NodePadding,
            scaled(_NODE_PADDING[0]), scaled(_NODE_PADDING[1]),
            category=dpg.mvThemeCat_Nodes, parent=component,
        )
        dpg.add_theme_style(
            dpg.mvStyleVar_ItemSpacing,
            scaled(_ITEM_SPACING[0]), scaled(_ITEM_SPACING[1]),
            category=dpg.mvThemeCat_Core, parent=component,
        )
        self._density[theme] = component

        if self._EDITOR not in self._density:
            self._densify_editor(dpg.get_item_parent(tag))

    def _densify_editor(self, editor):
        """Thin the pins and the links.

        They are drawn when the editor is closed, with the style of the item
        holding the editor: a theme on the editor itself comes too late.
        """
        self._density[self._EDITOR] = None
        holder = dpg.get_item_parent(editor) if editor else None
        if not holder:
            return
        theme = dpg.get_item_theme(holder)
        if not theme:
            if self._own_theme is None:
                self._own_theme = dpg.add_theme()
            theme = self._own_theme
            dpg.bind_item_theme(holder, theme)
            self._themed_holder = holder
        scale = self.scale
        component = dpg.add_theme_component(dpg.mvAll, parent=theme)
        dpg.add_theme_style(
            dpg.mvNodeStyleVar_PinCircleRadius,
            max(1.5, _PIN_RADIUS * scale),
            category=dpg.mvThemeCat_Nodes, parent=component,
        )
        dpg.add_theme_style(
            dpg.mvNodeStyleVar_LinkThickness,
            max(1.0, _LINK_THICKNESS * scale),
            category=dpg.mvThemeCat_Nodes, parent=component,
        )
        self._density[self._EDITOR] = component

    def _clear_density(self):
        for component in self._density.values():
            if component is not None and dpg.does_item_exist(component):
                dpg.delete_item(component)
        self._density = {}
        if self._themed_holder is not None:
            if dpg.does_item_exist(self._themed_holder):
                dpg.bind_item_theme(self._themed_holder, 0)
            self._themed_holder = None

    def _restore(self, tag, state):
        if state.label is not None:
            full, short = state.label
            # Leave the title alone if the node changed it in the meantime
            if dpg.get_item_configuration(tag).get("label") == short:
                dpg.configure_item(tag, label=full)
        if _overview_fonts:
            dpg.bind_item_font(tag, 0)
        if state.own_theme:
            dpg.bind_item_theme(tag, 0)
        for image, ((width, height), _) in state.images.items():
            if dpg.does_item_exist(image):
                dpg.configure_item(image, width=width, height=height)
        for item in state.hidden:
            if dpg.does_item_exist(item):
                dpg.configure_item(item, show=True)
        dpg.set_item_pos(tag, state.logical)

    def _compact(self, node_tag, state):
        """Hide the node contents, keeping the pins and the image previews."""
        try:
            full = dpg.get_item_configuration(node_tag).get("label")
            if full:
                short = short_label(full)
                if short and short != full:
                    dpg.configure_item(node_tag, label=short)
                    state.label = (full, short)
            for attribute in dpg.get_item_children(node_tag, _CONTENT_SLOT) or []:
                for item in dpg.get_item_children(attribute, _CONTENT_SLOT) or []:
                    self._compact_item(item, state)
        except Exception as exc:
            logger.warning("zoom: could not compact node %s: %s", node_tag, exc)

    def _compact_item(self, item, state):
        if self._is_image(item):
            config = dpg.get_item_configuration(item)
            if config.get("show", True):
                configured = (config.get("width", 0), config.get("height", 0))
                state.images[item] = (configured, self._displayed_size(config))
            return

        children = dpg.get_item_children(item, _CONTENT_SLOT) or []
        if children and self._contains_image(item):
            # Keep the container so the preview stays visible
            for child in children:
                self._compact_item(child, state)
            return

        if dpg.get_item_configuration(item).get("show", True):
            dpg.configure_item(item, show=False)
            state.hidden.append(item)

    @staticmethod
    def _displayed_size(image_config):
        """Size of an image at 100 %.

        An image created without an explicit size reports 0 until it has been
        rendered once; it is then displayed at the size of its texture.
        """
        width = image_config.get("width", 0)
        height = image_config.get("height", 0)
        if width > 0 and height > 0:
            return (width, height)
        texture = image_config.get("texture_tag")
        try:
            if texture is not None and dpg.does_item_exist(texture):
                width = width or dpg.get_item_width(texture) or 0
                height = height or dpg.get_item_height(texture) or 0
        except Exception:
            pass
        return (width, height)

    @staticmethod
    def _is_image(item):
        return dpg.get_item_type(item).endswith("::mvImage")

    def _contains_image(self, item):
        for child in dpg.get_item_children(item, _CONTENT_SLOT) or []:
            if self._is_image(child) or self._contains_image(child):
                return True
        return False
