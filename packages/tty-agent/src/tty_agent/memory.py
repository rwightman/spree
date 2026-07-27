"""Small JSON-backed memory store for agent experiments."""

from __future__ import annotations

import fcntl
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .ids import validate_agent_id
from .models import MemoryPatch


@dataclass
class JsonMemoryStore:
    root: Path | str = field(default_factory=lambda: Path("runtime/memory"))
    max_list_items: int = 100

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    def load(self, agent_id: str) -> dict[str, Any]:
        path = self._campaign_path(agent_id)
        with _campaign_lock(path):
            return self._load_unlocked(path)

    def _load_unlocked(self, path: Path) -> dict[str, Any]:
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # A corrupt file (e.g. torn by a crash predating atomic saves) would
            # otherwise block this agent forever; quarantine it and restart with
            # empty memory instead of failing every subsequent run.
            _quarantine(path)
            return {}
        if not isinstance(data, dict):
            # Valid JSON can still violate the memory schema. Treat a wrong root
            # exactly like malformed JSON so later read-modify-write calls do not
            # fail while trying to merge it as an object.
            _quarantine(path)
            return {}
        return data

    def save(self, agent_id: str, data: dict[str, Any]) -> None:
        path = self._campaign_path(agent_id)
        with _campaign_lock(path):
            self._save_unlocked(path, data)

    def _save_unlocked(self, path: Path, data: dict[str, Any]) -> None:
        # The per-agent lock makes one reusable temp path safe, while replace
        # keeps the campaign file crash-safe and atomic for readers.
        tmp_path = path.with_name(".campaign.tmp")
        try:
            tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            tmp_path.replace(path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def save_patch(self, agent_id: str, patch: MemoryPatch) -> dict[str, Any]:
        path = self._campaign_path(agent_id)
        # The read-modify-write must be exclusive or concurrent finalizers
        # (parallel match modes share one store) can drop each other's patches.
        with _campaign_lock(path):
            current = self._load_unlocked(path)
            merged = _merge_memory(current, patch.data, max_list_items=self.max_list_items)
            self._save_unlocked(path, merged)
        return merged

    def _campaign_path(self, agent_id: str) -> Path:
        return self.root / validate_agent_id(agent_id) / "campaign.json"


@contextmanager
def _campaign_lock(campaign_path: Path) -> Iterator[None]:
    """Hold the exclusive per-agent lock shared by every campaign operation.

    Never delete the lock file: flock identity is the inode, so a process that
    opens a recreated file holds a "lock" concurrently with an existing holder
    of the old one. A stale lock file is harmless; the kernel releases the lock
    itself whenever its holder's fd closes, including on process death.
    """

    lock_path = campaign_path.parent / "campaign.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        yield


def _quarantine(path: Path) -> None:
    path.replace(path.with_suffix(".json.corrupt"))


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
