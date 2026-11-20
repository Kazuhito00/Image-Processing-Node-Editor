#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Persistent registry for student ONNX models used in online training (distillation).

The registry is stored as a JSON file alongside this module.
Each entry contains everything needed to load and use the student model.

Entry schema
------------
{
    "name":          str   – display name shown in the model dropdown
    "path":          str   – absolute path to the .onnx file
    "class_names":   dict  – {str_id: str_name}  (e.g. {"0": "cat", "1": "dog"})
    "output_format": str   – "nanodet" | "yolo11" | "yolox" | "unknown"
    "input_width":   int
    "input_height":  int
    "num_classes":   int
}

Students shipped with the application (:func:`builtin_models`) follow the same
schema and are not stored in the file: their path depends on where the
application is installed.
"""

import json
import os
import sys

from node.DLNode.object_detection.coco_class_names import coco_class_names

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)

# (name, path from the repository root, output format, input width, input
# height, description).  Each of them was run in the OnlineTraining node
# against a YOLOX-s teacher before being listed here; all detect the 80 COCO
# classes.  ``nanodet-m_320_int8`` is deliberately absent: its quantised
# output is not decoded correctly (thousands of boxes per image).
_BUILTIN = (
    (
        "yolox_nano",
        "node/DLNode/object_detection/YOLOX/model/yolox_nano.onnx",
        "yolox", 416, 416,
        "0.9M parameters, the fastest; trained by real backpropagation",
    ),
    (
        "yolox_tiny",
        "node/DLNode/object_detection/YOLOX/model/yolox_tiny.onnx",
        "yolox", 416, 416,
        "5.1M parameters, more accurate; trained by real backpropagation",
    ),
    (
        "nanodet-plus-m_416",
        "nanodet-plus-m_416.onnx",
        "nanodet", 416, 416,
        "1.2M parameters; trained by real backpropagation",
    ),
    (
        "yolo11_n",
        "node/DLNode/object_detection/YOLO/model/yolo11_n.onnx",
        "yolo11", 608, 416,
        "2.6M parameters; its graph cannot be converted to PyTorch, so only "
        "the affine correction head is trained, not the network",
    ),
)


def builtin_models() -> list:
    """Entries of the students shipped with the application, cheapest first.

    Only those whose file is present are returned.
    """
    class_names = {str(index): name for index, name in coco_class_names.items()}
    entries = []
    for name, relative_path, output_format, width, height, description in _BUILTIN:
        path = os.path.join(_REPO_ROOT, *relative_path.split("/"))
        if os.path.isfile(path):
            entries.append({
                "name": name,
                "path": path,
                "class_names": dict(class_names),
                "output_format": output_format,
                "input_width": width,
                "input_height": height,
                "num_classes": len(class_names),
                "description": description,
                "builtin": True,
            })
    return entries


def _get_registry_path():
    """Return the registry path – writable location in frozen (onefile) mode."""
    if getattr(sys, "frozen", False):
        from src.utils.paths import get_registry_path
        return get_registry_path("student_models_registry.json")
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "student_models_registry.json"
    )


_REGISTRY_PATH = _get_registry_path()


def _load_raw() -> list:
    """Return the raw list from the JSON file (no validation)."""
    if not os.path.isfile(_REGISTRY_PATH):
        return []
    try:
        with open(_REGISTRY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def _save_raw(entries: list) -> None:
    with open(_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def load_registry() -> list:
    """Return a list of valid registry entries (path still exists)."""
    entries = _load_raw()
    valid = []
    for e in entries:
        if isinstance(e, dict) and "name" in e and "path" in e:
            if os.path.isfile(e["path"]):
                valid.append(e)
    return valid


def save_entry(entry: dict) -> None:
    """Add or update an entry (matched by name) and persist to disk."""
    entries = _load_raw()
    # Replace existing entry with same name
    updated = False
    for i, e in enumerate(entries):
        if isinstance(e, dict) and e.get("name") == entry.get("name"):
            entries[i] = entry
            updated = True
            break
    if not updated:
        entries.append(entry)
    _save_raw(entries)


def remove_entry(name: str) -> None:
    """Remove entry by name and persist."""
    entries = _load_raw()
    entries = [e for e in entries if not (isinstance(e, dict) and e.get("name") == name)]
    _save_raw(entries)


def get_entry(name: str) -> dict | None:
    """Return entry dict for given name, or None."""
    for e in _load_raw():
        if isinstance(e, dict) and e.get("name") == name:
            return e
    return None


# ---------------------------------------------------------------------------
# Catalog of pre-validated student models available for download.
#
# Populate ``download_url`` when the ONNX files are hosted (e.g. a GitHub
# Release asset).  Until then the download infrastructure is in place but
# :func:`download_model` will log a warning for models with an empty URL.
#
# Each entry follows the same schema as :func:`save_entry`; the ``path``
# key is omitted here (it is filled in after the file is downloaded).
# ---------------------------------------------------------------------------
DOWNLOADABLE_MODELS = [
    {
        "name": "yolov8n-coco",
        "description": (
            "YOLOv8 nano — 80 COCO classes, 640×640 input. "
            "Compatible with PyTorch backprop (onnx2torch ≥ 0.0.30). "
            "Export yourself: yolo export model=yolov8n.pt format=onnx opset=12"
        ),
        "output_format": "yolo11",
        "num_classes": 80,
        "input_width": 640,
        "input_height": 640,
        "download_url": "",   # set when hosted
    },
    {
        "name": "yolov8s-coco",
        "description": (
            "YOLOv8 small — 80 COCO classes, 640×640 input. "
            "Compatible with PyTorch backprop (onnx2torch ≥ 0.0.30). "
            "Export yourself: yolo export model=yolov8s.pt format=onnx opset=12"
        ),
        "output_format": "yolo11",
        "num_classes": 80,
        "input_width": 640,
        "input_height": 640,
        "download_url": "",
    },
    {
        "name": "nanodet-plus-m_416",
        "description": (
            "NanoDet-Plus-m — 80 COCO classes, 416×416 input. "
            "Compatible with PyTorch backprop (onnx2torch ≥ 0.0.30). "
            "Download from: github.com/RangiLyu/nanodet/releases"
        ),
        "output_format": "nanodet",
        "num_classes": 80,
        "input_width": 416,
        "input_height": 416,
        "download_url": "",
    },
]


def download_model(
    url: str,
    dest_path: str,
    progress_callback=None,
) -> bool:
    """Download a model ONNX file from *url* to *dest_path*.

    Parameters
    ----------
    url : str
        Direct download URL for the ``.onnx`` file.
    dest_path : str
        Local file path where the downloaded model will be saved.
    progress_callback : callable | None
        Optional ``callback(percent: float)`` invoked during download.

    Returns
    -------
    bool
        ``True`` on success, ``False`` on any error.
    """
    import urllib.request

    if not url:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "[StudentRegistry] download_model: empty URL — nothing to download."
        )
        return False

    try:
        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)

        def _reporthook(block_num, block_size, total_size):
            if progress_callback is not None and total_size > 0:
                pct = min(100.0, block_num * block_size / total_size * 100)
                try:
                    progress_callback(pct)
                except Exception:
                    pass

        urllib.request.urlretrieve(url, dest_path, reporthook=_reporthook)
        if progress_callback is not None:
            try:
                progress_callback(100.0)
            except Exception:
                pass
        return True
    except Exception as exc:
        import logging as _logging
        _logging.getLogger(__name__).error(
            "[StudentRegistry] download_model failed (%s → %s): %s",
            url, dest_path, exc,
        )
        return False
