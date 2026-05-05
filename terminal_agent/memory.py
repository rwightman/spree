"""Small JSON-backed memory store for agent experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import MemoryPatch


@dataclass
class JsonMemoryStore:
    root: Path | str = field(default_factory=lambda: Path("runtime/memory"))
    max_list_items: int = 100

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    def load(self, agent_id: str) -> dict[str, Any]:
        path = self._campaign_path(agent_id)
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, agent_id: str, data: dict[str, Any]) -> None:
        path = self._campaign_path(agent_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def save_patch(self, agent_id: str, patch: MemoryPatch) -> dict[str, Any]:
        current = self.load(agent_id)
        merged = _merge_memory(current, patch.data, max_list_items=self.max_list_items)
        self.save(agent_id, merged)
        return merged

    def _campaign_path(self, agent_id: str) -> Path:
        return self.root / agent_id / "campaign.json"


def _merge_memory(current: dict[str, Any], patch: dict[str, Any], max_list_items: int = 100) -> dict[str, Any]:
    merged = dict(current)
    for key, value in patch.items():
        if isinstance(value, list):
            existing = merged.get(key, [])
            if not isinstance(existing, list):
                existing = [existing]
            merged[key] = _dedupe_list(existing + value)[-max_list_items:]
        elif isinstance(value, dict):
            existing = merged.get(key, {})
            if isinstance(existing, dict):
                merged[key] = _merge_memory(existing, value, max_list_items=max_list_items)
            else:
                merged[key] = value
        else:
            merged[key] = value
    return merged


def _dedupe_list(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for value in values:
        key = _stable_key(value)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _stable_key(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return repr(value)
