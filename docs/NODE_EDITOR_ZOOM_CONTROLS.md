# Node editor – overview mode (zoom out)

The node editor can be zoomed out to get an overview of a large graph, then
zoomed back to the normal view.

## Controls

| Action | Shortcut | Menu |
|---|---|---|
| Zoom out | `Ctrl` + mouse wheel down | View > Zoom out |
| Zoom in | `Ctrl` + mouse wheel up | View > Zoom in |
| Back to the normal view | `Ctrl` + `0` | View > Normal view (100%) |

Levels: 100 % (normal), 70 %, 50 %, 35 %. The current level is shown in the
**View** menu title while zoomed out, e.g. `View (50%)`.

## What the overview shows

- Nodes are reduced to their title bar, their pins and their links.
- Image previews stay visible as live thumbnails.
- Sliders, combos, text fields and other controls are hidden: zoom back in to
  edit a node.
- Nodes are pulled together around the mouse cursor (or around the centre of
  the graph when the menu is used).

Nodes can still be moved, linked, added, copied, pasted and deleted while
zoomed out. Panning (middle mouse button) and the minimap work as usual.

## Guarantees

- Zooming back to 100 % restores every node exactly: position, widget
  visibility and preview size. A node dragged while zoomed out keeps its new
  place.
- Saved positions never depend on the zoom level: File > Export, copy/paste,
  undo and the Deploy node always use the positions at 100 %.
- No per-frame cost: work is done only when the zoom level changes.

## Limitations

- There is no zoom above 100 %. DearPyGui's node editor has no zoom API; the
  overview is emulated by `node_editor/zoom.py`.
- Text is not scaled, so node titles keep their size at every level.
- A node that changes the visibility of one of its own widgets while the
  editor is zoomed out (for example a progress bar that finishes) will see
  that widget shown again when zooming back in.
- Previews drawn with something other than an image widget (plots, draw
  lists) are hidden in the overview.

## Implementation

`node_editor/zoom.py` contains `EditorZoom`, which is independent of the rest
of the editor. `DpgNodeEditor` only:

- forwards `Ctrl` + wheel, `Ctrl` + `0` and the **View** menu to it,
- calls `EditorZoom.adopt()` after creating a node,
- converts positions with `EditorZoom.logical_pos()` when node settings are
  read (`DpgNodeEditor.get_node_setting_dict()`).

Tests: `tests/test_node_editor_overview_zoom.py`.
