"""
CvStudio JSON save-file parser.

Parses the export format produced by CvStudio's _callback_file_export:
{
    "node_list": ["<id>:<NodeTag>", ...],
    "link_list": [["<output_port>", "<input_port>"], ...],
    "<id>:<NodeTag>": {
        "id": "<id>",
        "name": "<NodeTag>",
        "setting": { ... }
    }
}

An optional top-level "deepstream_hints" object carries runtime information
that CvStudio does not persist in node settings (video file path, webcam
device):
{
    "deepstream_hints": {"<id>:<NodeTag>": {"uri": "/data/video.mp4"}}
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HINTS_KEY = "deepstream_hints"


@dataclass
class NodeInfo:
    """Parsed representation of a single CvStudio node."""

    node_id: int
    node_tag: str
    settings: dict[str, Any] = field(default_factory=dict)
    hints: dict[str, Any] = field(default_factory=dict)

    def setting(self, key_suffix: str, default: Any = None) -> Any:
        """Return the first setting whose key ends with *key_suffix*."""
        for key, value in self.settings.items():
            if key.endswith(key_suffix):
                return value
        return default

    def text_setting(self, key_suffix: str, default: str = "") -> str:
        value = self.setting(key_suffix)
        if isinstance(value, str) and value.strip():
            return value.strip()
        return default

    def float_setting(self, key_suffix: str, default: float) -> float:
        try:
            return float(self.setting(key_suffix))
        except (TypeError, ValueError):
            return default


@dataclass
class LinkInfo:
    """A connection between two node ports."""

    source_port: str
    target_port: str

    @property
    def source_node_id(self) -> int:
        return int(self.source_port.split(":")[0])

    @property
    def target_node_id(self) -> int:
        return int(self.target_port.split(":")[0])

    @property
    def source_node_tag(self) -> str:
        return self.source_port.split(":")[1]

    @property
    def target_node_tag(self) -> str:
        return self.target_port.split(":")[1]


@dataclass
class CvStudioProject:
    """Full parsed CvStudio project."""

    nodes: dict[int, NodeInfo] = field(default_factory=dict)
    links: list[LinkInfo] = field(default_factory=list)

    def nodes_by_tag(self, tag: str) -> list[NodeInfo]:
        return [n for n in self.nodes.values() if n.node_tag == tag]

    def get_downstream(self, node_id: int) -> list[NodeInfo]:
        """Return nodes that receive data from *node_id*."""
        downstream_ids = set()
        for link in self.links:
            if link.source_node_id == node_id:
                downstream_ids.add(link.target_node_id)
        return [self.nodes[nid] for nid in downstream_ids if nid in self.nodes]

    def get_upstream(self, node_id: int) -> list[NodeInfo]:
        """Return nodes that feed data into *node_id*."""
        upstream_ids = set()
        for link in self.links:
            if link.target_node_id == node_id:
                upstream_ids.add(link.source_node_id)
        return [self.nodes[nid] for nid in upstream_ids if nid in self.nodes]


def parse(path: str | Path) -> CvStudioProject:
    """Parse a CvStudio JSON save file and return a CvStudioProject."""
    with open(Path(path), "r", encoding="utf-8") as fp:
        return parse_dict(json.load(fp))


def parse_dict(data: dict) -> CvStudioProject:
    """Build a CvStudioProject from an already-loaded export dict."""
    if not isinstance(data, dict) or not isinstance(data.get("node_list"), list):
        raise ValueError(
            "Not a CvStudio export: missing 'node_list'. "
            "Use File > Export in CvStudio to produce the JSON."
        )

    project = CvStudioProject()
    hints = data.get(HINTS_KEY) or {}

    for node_id_name in data["node_list"]:
        node_id_str, _, node_tag = str(node_id_name).partition(":")
        if not node_id_str.isdigit() or not node_tag:
            raise ValueError(f"Invalid node identifier in node_list: {node_id_name!r}")

        node_data = data.get(node_id_name) or {}
        project.nodes[int(node_id_str)] = NodeInfo(
            node_id=int(node_id_str),
            node_tag=node_tag,
            settings=node_data.get("setting") or {},
            hints=dict(hints.get(node_id_name) or {}),
        )

    for link_pair in data.get("link_list") or []:
        if len(link_pair) >= 2:
            project.links.append(
                LinkInfo(source_port=link_pair[0], target_port=link_pair[1])
            )

    return project
