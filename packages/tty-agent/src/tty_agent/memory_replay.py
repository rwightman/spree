"""Offline, fixed-context replay helpers for memory reconciliation experiments.

Point replay reconstructs the original structured store immediately before one
journal batch, restores that batch's overlap context, and presents only the
same source events again. Alternative prompt/model settings therefore cannot
compound into a different memory trajectory before the point being measured.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .memory_subsystem import MemoryEvent


@dataclass(frozen=True)
class StructuredJournalBatch:
    """One atomic line from a structured-memory operations journal."""

    index: int
    batch: str
    source_steps: tuple[int, ...]
    records: tuple[dict[str, Any], ...]


def load_activity_memory_events(path: str | Path) -> dict[int, MemoryEvent]:
    """Reconstruct the exact terminal memory events written by ``ActivityRunner``.

    Non-step activity records are ignored. Step numbers must be unique because
    journal batches refer to them as stable evidence identifiers.
    """

    events: dict[int, MemoryEvent] = {}
    source = Path(path)
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid activity JSON at {source}:{line_number}: {exc}") from exc
        if not isinstance(record, dict):
            continue
        step = record.get("step")
        if not isinstance(step, int) or isinstance(step, bool):
            continue
        if step in events:
            raise ValueError(f"duplicate activity step {step} at {source}:{line_number}")

        observation = record.get("observation")
        validation = record.get("validation")
        action = record.get("action")
        observation_text = observation.get("model_text", "") if isinstance(observation, dict) else ""
        warnings: tuple[str, ...] = ()
        if isinstance(validation, dict) and validation.get("accepted") is False:
            notes = validation.get("notes", [])
            if isinstance(notes, list):
                warnings = tuple(str(note) for note in notes if note)
            if not warnings:
                warnings = ("action was rejected and not executed",)
        events[step] = MemoryEvent(
            kind="terminal_step",
            step=step,
            observation=str(observation_text),
            action=json.dumps(action, sort_keys=True) if action else "",
            warnings=warnings,
        )
    return events


def load_structured_journal_batches(path: str | Path) -> list[StructuredJournalBatch]:
    """Read complete atomic batches, ignoring a torn final fragment."""

    source = Path(path)
    batches: list[StructuredJournalBatch] = []
    raw = source.read_bytes()
    lines = raw.split(b"\n")
    for line_number, encoded_line in enumerate(lines, start=1):
        if not encoded_line.strip():
            continue
        try:
            line = encoded_line.decode("utf-8")
            payload = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if line_number == len(lines) and not raw.endswith(b"\n"):
                break
            raise ValueError(f"invalid journal JSON at {source}:{line_number}: {exc}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
            raise ValueError(f"journal line is not an atomic records batch at {source}:{line_number}")
        records = tuple(record for record in payload["records"] if isinstance(record, dict))
        if not records:
            raise ValueError(f"journal batch is empty at {source}:{line_number}")
        batch_names = {str(record.get("batch", "")) for record in records}
        if len(batch_names) != 1:
            raise ValueError(f"journal batch has mixed operation types at {source}:{line_number}")
        source_vectors = {
            tuple(int(step) for step in record.get("source_steps", []))
            for record in records
            if record.get("source_steps")
        }
        if len(source_vectors) > 1:
            raise ValueError(f"journal batch has inconsistent source steps at {source}:{line_number}")
        batches.append(
            StructuredJournalBatch(
                index=len(batches),
                batch=batch_names.pop(),
                source_steps=next(iter(source_vectors), ()),
                records=records,
            )
        )
    return batches


def select_batch_events(
        events: dict[int, MemoryEvent],
        batches: list[StructuredJournalBatch],
        batch_index: int,
        overlap_events: int,
) -> tuple[list[MemoryEvent], list[MemoryEvent]]:
    """Return the target batch's new events and its original overlap tail."""

    try:
        batch = batches[batch_index]
    except IndexError as exc:
        raise ValueError(f"journal batch index {batch_index} is out of range (0..{len(batches) - 1})") from exc
    if not batch.source_steps:
        operations = ",".join(sorted({str(record.get("op", "unknown")) for record in batch.records}))
        raise ValueError(
            f"journal batch {batch_index} ({batch.batch}; operations={operations}) has no source steps "
            "and cannot be replayed"
        )
    selected = [_required_event(events, step, batch_index) for step in batch.source_steps]
    if batch_index == 0 or overlap_events == 0:
        return selected, []
    prior_steps: list[int] = []
    for prior in reversed(batches[:batch_index]):
        if _is_overlap_boundary(prior):
            break
        if not prior.source_steps:
            continue
        prior_steps[:0] = prior.source_steps
        if len(prior_steps) >= overlap_events:
            break
    overlap = [_required_event(events, step, batch_index) for step in prior_steps[-overlap_events:]]
    return selected, overlap


def _is_overlap_boundary(batch: StructuredJournalBatch) -> bool:
    """Return whether the normal runner closes its memory handle here."""

    if batch.batch == "audit":
        return True
    return any(record.get("op") == "commit_failed" for record in batch.records)


def write_journal_prefix(source: str | Path, destination: str | Path, batch_count: int) -> None:
    """Initialize a fresh replay journal with its first ``batch_count`` batches."""

    source_path = Path(source)
    destination_path = Path(destination)
    if destination_path.exists():
        raise FileExistsError(f"point-replay journal already exists: {destination_path}")
    lines = source_path.read_bytes().splitlines(keepends=True)
    if batch_count < 0 or batch_count > len(lines):
        raise ValueError(f"journal batch count {batch_count} is out of range (0..{len(lines)})")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("xb") as handle:
        for line in lines[:batch_count]:
            handle.write(line)


def _required_event(events: dict[int, MemoryEvent], step: int, batch_index: int) -> MemoryEvent:
    try:
        return events[step]
    except KeyError as exc:
        raise ValueError(f"activity log is missing step {step} required by journal batch {batch_index}") from exc
