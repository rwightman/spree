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

    def save_patch(
            self,
            agent_id: str,
            patch: MemoryPatch,
            *,
            limits: "MemoryDocumentLimits | None" = None,
    ) -> dict[str, Any]:
        merged, _report = self.save_patch_with_report(agent_id, patch, limits=limits)
        return merged

    def save_patch_with_report(
            self,
            agent_id: str,
            patch: MemoryPatch,
            *,
            limits: "MemoryDocumentLimits | None" = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        path = self._campaign_path(agent_id)
        # The read-modify-write must be exclusive or concurrent finalizers
        # (parallel match modes share one store) can drop each other's patches.
        with _campaign_lock(path):
            current = self._load_unlocked(path)
            patch_data = patch.data
            patch_report = _unchanged_bound_report(patch_data)
            if limits is not None:
                effective_limits = MemoryDocumentLimits(
                    max_document_chars=limits.max_document_chars,
                    max_string_chars=limits.max_string_chars,
                    max_list_items=min(self.max_list_items, limits.max_list_items),
                )
                patch_data, patch_report = bound_memory_document(patch_data, effective_limits)
                # Build the full patch-priority candidate first. Bounding that
                # candidate makes every capacity exit visible in the returned
                # document report instead of hiding it inside merge slicing.
                merged = _merge_memory(
                    current,
                    patch_data,
                    max_list_items=None,
                    prefer_patch_order=True,
                )
                merged, document_report = bound_memory_document(merged, effective_limits)
            else:
                merged = _merge_memory(current, patch_data, max_list_items=self.max_list_items)
                document_report = _unchanged_bound_report(merged)
            self._save_unlocked(path, merged)
        return merged, {"patch": patch_report, "document": document_report}

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


@dataclass(frozen=True)
class MemoryDocumentLimits:
    """Code-enforced bounds for legacy durable campaign memory."""

    max_document_chars: int = 12_000
    max_string_chars: int = 500
    max_list_items: int = 40

    def __post_init__(self) -> None:
        for name in ("max_document_chars", "max_string_chars", "max_list_items"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.max_document_chars < 2:
            raise ValueError("max_document_chars must be >= 2 to fit an empty JSON object")


def bound_memory_document(
        data: dict[str, Any],
        limits: MemoryDocumentLimits,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize and bound one memory document without fuzzy matching."""

    stats = {
        "clipped_strings": 0,
        "deduped_list_items": 0,
        "trimmed_list_items": 0,
        "total_pruned_items": 0,
        "dropped_fields": 0,
    }
    before_chars = _memory_json_chars(data)
    bounded = {str(key): _bound_memory_value(value, limits, stats) for key, value in data.items()}
    while _memory_json_chars(bounded) > limits.max_document_chars:
        if _pop_largest_list_tail(bounded):
            stats["total_pruned_items"] += 1
            continue
        if _trim_longest_string(bounded, _memory_json_chars(bounded) - limits.max_document_chars):
            stats["clipped_strings"] += 1
            continue
        if not bounded:
            break
        bounded.pop(sorted(bounded)[-1])
        stats["dropped_fields"] += 1
    after_chars = _memory_json_chars(bounded)
    return bounded, {
        "changed": bounded != data,
        "before_chars": before_chars,
        "after_chars": after_chars,
        **stats,
    }


def _bound_memory_value(value: Any, limits: MemoryDocumentLimits, stats: dict[str, int]) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if len(stripped) > limits.max_string_chars:
            stats["clipped_strings"] += 1
            return _clip_memory_string(stripped, limits.max_string_chars)
        return stripped
    if isinstance(value, list):
        normalized = [_bound_memory_value(item, limits, stats) for item in value]
        deduped = _dedupe_list(normalized)
        stats["deduped_list_items"] += len(normalized) - len(deduped)
        if len(deduped) > limits.max_list_items:
            stats["trimmed_list_items"] += len(deduped) - limits.max_list_items
            deduped = deduped[:limits.max_list_items]
        return deduped
    if isinstance(value, dict):
        return {str(key): _bound_memory_value(item, limits, stats) for key, item in value.items()}
    return value


def _clip_memory_string(value: str, limit: int) -> str:
    marker = "…"
    if len(value) <= limit:
        return value
    if limit <= len(marker):
        return marker[:limit]
    return value[: limit - len(marker)] + marker


def _memory_json_chars(data: Any) -> int:
    return len(json.dumps(data, ensure_ascii=False, separators=(",", ":"), sort_keys=True))


def _pop_largest_list_tail(data: dict[str, Any]) -> bool:
    candidates: list[tuple[int, str, list[Any]]] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, list):
            if value:
                candidates.append((_memory_json_chars(value), path, value))
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
        elif isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], f"{path}.{key}")

    visit(data, "$")
    if not candidates:
        return False
    _size, _path, target = max(candidates, key=lambda candidate: (candidate[0], candidate[1]))
    target.pop()
    return True


def _trim_longest_string(data: dict[str, Any], excess: int) -> bool:
    candidates: list[tuple[int, str, Any, Any]] = []

    def visit(value: Any, path: str, parent: Any = None, key: Any = None) -> None:
        if isinstance(value, str) and value:
            candidates.append((len(value), path, parent, key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]", value, index)
        elif isinstance(value, dict):
            for item_key in sorted(value):
                visit(value[item_key], f"{path}.{item_key}", value, item_key)

    visit(data, "$")
    if not candidates:
        return False
    length, _path, parent, key = max(candidates, key=lambda candidate: (candidate[0], candidate[1]))
    keep = max(0, length - max(1, excess))
    parent[key] = _clip_memory_string(parent[key], keep) if keep else ""
    return True


def _unchanged_bound_report(data: dict[str, Any]) -> dict[str, Any]:
    chars = _memory_json_chars(data)
    return {
        "changed": False,
        "before_chars": chars,
        "after_chars": chars,
        "clipped_strings": 0,
        "deduped_list_items": 0,
        "trimmed_list_items": 0,
        "total_pruned_items": 0,
        "dropped_fields": 0,
    }


def _merge_memory(
        current: dict[str, Any],
        patch: dict[str, Any],
        max_list_items: int | None = 100,
        *,
        prefer_patch_order: bool = False,
) -> dict[str, Any]:
    merged = dict(current)
    for key, value in patch.items():
        if isinstance(value, list):
            existing = merged.get(key, [])
            if not isinstance(existing, list):
                existing = [existing]
            if prefer_patch_order:
                # Bounded legacy patches are prompted most-useful-first. Keep
                # that ordering, then fill capacity with unmentioned prior
                # entries so selective updates still default-persist.
                combined = _dedupe_list(value + existing)
                merged[key] = combined if max_list_items is None else combined[:max_list_items]
            else:
                assert max_list_items is not None
                merged[key] = _dedupe_list(existing + value)[-max_list_items:]
        elif isinstance(value, dict):
            existing = merged.get(key, {})
            if isinstance(existing, dict):
                merged[key] = _merge_memory(
                    existing,
                    value,
                    max_list_items=max_list_items,
                    prefer_patch_order=prefer_patch_order,
                )
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
