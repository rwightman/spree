"""Common seam for swappable memory subsystems.

Defined in ``docs/memory-simple.md``: the runner and campaign depend only on
this coarse service surface, while each implementation owns representation,
prompts, parsing, and cadence internally. The seam also carries the
measurement contract — every implementation journals mutations in the common
record shape below so metrics compare identically across arms.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .ids import validate_agent_id
from .models import ModelAdapter


@dataclass(frozen=True)
class MemoryContextKey:
    """One isolated memory context: an agent identity inside one activity."""

    root: Path
    agent_id: str
    context_id: str

    def __post_init__(self) -> None:
        validate_agent_id(self.agent_id)
        validate_agent_id(self.context_id)

    def directory(self) -> Path:
        return Path(self.root) / self.agent_id / self.context_id


@dataclass(frozen=True)
class MemoryEvent:
    """One unit of observed activity fed to a memory handle."""

    kind: str  # "terminal_step" today
    step: int
    observation: str = ""
    action: str = ""
    intent: str = ""
    warnings: tuple[str, ...] = ()

    def chars(self) -> int:
        return len(self.observation) + len(self.action) + len(self.intent)


@dataclass(frozen=True)
class ReconcileOutcome:
    """What one reconciliation attempt did, for traces and route events."""

    status: str  # "applied" | "no_memory_change" | "failed"
    accepted_ops: int = 0
    rejected_ops: int = 0
    covered_steps: int = 0
    error: str = ""
    usage: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "accepted_ops": self.accepted_ops,
            "rejected_ops": self.rejected_ops,
            "covered_steps": self.covered_steps,
            "error": self.error,
            "usage": self.usage,
        }


@dataclass(frozen=True)
class CommitOutcome:
    """What the durable end-of-session commit did."""

    status: str  # "applied" | "no_memory_change" | "failed"
    attempts: int = 1
    accepted_ops: int = 0
    rejected_ops: int = 0
    error: str = ""
    usage: dict[str, Any] | None = None
    audit_status: str = "not_run"  # "applied" | "no_memory_change" | "failed" | "not_run"
    audit_accepted_ops: int = 0
    audit_rejected_ops: int = 0
    audit_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "attempts": self.attempts,
            "accepted_ops": self.accepted_ops,
            "rejected_ops": self.rejected_ops,
            "error": self.error,
            "usage": self.usage,
            "audit_status": self.audit_status,
            "audit_accepted_ops": self.audit_accepted_ops,
            "audit_rejected_ops": self.audit_rejected_ops,
            "audit_error": self.audit_error,
        }


def merge_usage(total: dict[str, Any] | None, usage: dict[str, Any] | None) -> dict[str, Any] | None:
    """Accumulate provider usage dicts by summing numeric fields."""

    if not usage:
        return total
    merged = dict(total or {})
    for key, value in usage.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            existing = merged.get(key, 0)
            merged[key] = (existing if isinstance(existing, (int, float)) else 0) + value
        elif key not in merged:
            merged[key] = value
    return merged


def mutation_record(
        *,
        op: str,
        origin: str,
        accepted: bool,
        batch: str,
        section: str = "",
        item_id: str = "",
        reason: str = "",
        source_steps: tuple[int, ...] = (),
        fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one journal record in the cross-implementation schema.

    Every implementation writes these to a JSONL journal under its context
    root; the shared metric tooling reads only these keys.
    """

    return {
        "op": op,
        "origin": origin,  # "model" | "system"
        "accepted": accepted,
        "batch": batch,  # "reconcile" | "commit" | "audit" | "recovery"
        "section": section,
        "item_id": item_id,
        "reason": reason,
        "source_steps": list(source_steps),
        "fields": dict(fields or {}),
        "timestamp": time.time(),
    }


def write_journal_records(path: Path, records: list[dict[str, Any]]) -> None:
    """Append one batch of mutation records to a journal, fsynced.

    Shared by every implementation (and by the legacy inline instrumentation)
    so the on-disk journal format is identical across arms. Each call writes
    the whole batch as ONE JSONL line (``{"records": [...]}``) in application
    order — a torn trailing line is therefore a torn batch, and truncating it
    on recovery restores batch atomicity without any extra framing protocol.
    """

    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"records": records}, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def read_journal_records(path: Path) -> list[dict[str, Any]]:
    """Read all mutation records from a journal, flattening batch lines.

    Tolerates a torn trailing line (dropped, matching replay) and accepts
    bare-record lines for forward compatibility. This is the read surface the
    metric tooling and tests share.
    """

    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    lines = path.read_text(encoding="utf-8").split("\n")
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                # Only a non-newline-terminated trailing fragment is a torn
                # write; a corrupt complete line is corruption and raises.
                break
            raise
        if isinstance(parsed, dict) and isinstance(parsed.get("records"), list):
            records.extend(record for record in parsed["records"] if isinstance(record, dict))
        elif isinstance(parsed, dict):
            records.append(parsed)
    return records


@runtime_checkable
class MemoryHandle(Protocol):
    """One open memory context. Implementations own everything internal."""

    def observe(self, event: MemoryEvent) -> None: ...

    def render_context(self, budget_chars: int | None = None) -> str: ...

    def render_bootstrap(self, budget_chars: int | None = None) -> str: ...

    def maybe_reconcile(self, model: ModelAdapter, *, force: bool = False) -> ReconcileOutcome | None: ...

    def commit(self, model: ModelAdapter, extra_evidence: str = "") -> CommitOutcome: ...

    def close(self) -> None: ...


@runtime_checkable
class MemorySubsystem(Protocol):
    name: str

    def fingerprints(self) -> dict[str, str]: ...

    def open_context(self, agent_id: str, context_id: str) -> MemoryHandle: ...
