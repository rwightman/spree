"""The ``structured`` memory subsystem from ``docs/memory-structured.md``.

One store per (agent, context): a JSON document cache plus an append-only ops
journal, guarded by flock and atomic replacement. The journal is the source
of truth (``replay(ops_journal) == store_document``); recovery truncates a
torn trailing line and replays. The model's reconciliation response is a JSON
array of operations and nothing else — no envelope, no attestation. Capacity
is entirely code-owned; unmentioned items persist.

Crash safety is forward-only: everything journaled survives (each batch is one
fsynced JSONL line, applied atomically or not at all on replay), while pending
un-reconciled events and the covered/overlap boundary live only in process
memory and are lost with the process. That is the deliberate structured-arm trade;
the full design in ``docs/memory-design.md`` owns durable evidence logs.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, BinaryIO

from .memory_subsystem import (
    CommitOutcome,
    MemoryContextKey,
    MemoryEvent,
    ReconcileOutcome,
    merge_usage,
    mutation_record,
    write_journal_records,
)
from .models import CompactionPrompt, ModelAdapter, ModelError

_SCHEMA_VERSION = 1
_SECTIONS = ("goal", "fact", "hypothesis")
_GOAL_STATUSES = ("open", "blocked", "done", "abandoned")
_LEGAL_GOAL_MOVES = {
    "open": {"blocked", "done", "abandoned"},
    "blocked": {"open", "abandoned"},
    "done": {"open"},
    "abandoned": {"open"},
}
_WS_RE = re.compile(r"\s+")
_AUDIT_ALLOWED_OPERATIONS = frozenset({"revise", "contradict", "transition"})


@dataclass(frozen=True)
class SimpleGoal:
    id: str
    text: str
    status: str = "open"
    priority: int = 3
    context: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "status": self.status,
            "priority": self.priority,
            "context": self.context,
        }


@dataclass(frozen=True)
class SimpleItem:
    id: str
    text: str
    source_steps: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "source_steps": list(self.source_steps)}


@dataclass
class StructuredMemory:
    schema_version: int = _SCHEMA_VERSION
    next_id: int = 1
    state: dict[str, str] = field(default_factory=dict)
    goals: list[SimpleGoal] = field(default_factory=list)
    facts: list[SimpleItem] = field(default_factory=list)
    hypotheses: list[SimpleItem] = field(default_factory=list)
    archive: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "next_id": self.next_id,
            "state": dict(sorted(self.state.items())),
            "goals": [goal.to_dict() for goal in self.goals],
            "facts": [item.to_dict() for item in self.facts],
            "hypotheses": [item.to_dict() for item in self.hypotheses],
            "archive": list(self.archive),
        }

    def is_empty(self) -> bool:
        return not (self.state or self.goals or self.facts or self.hypotheses)

    def find(self, item_id: str) -> tuple[str, Any] | None:
        for goal in self.goals:
            if goal.id == item_id:
                return "goal", goal
        for section, items in (("fact", self.facts), ("hypothesis", self.hypotheses)):
            for item in items:
                if item.id == item_id:
                    return section, item
        return None


@dataclass(frozen=True)
class StructuredMemoryConfig:
    """Caps and cadence; all code-enforced. Part of the mutation fingerprint."""

    max_goals: int = 5
    max_facts: int = 30
    max_hypotheses: int = 10
    max_state_entries: int = 12
    max_text_chars: int = 300
    max_state_value_chars: int = 200
    max_archive_entries: int = 200
    reconcile_every_events: int = 20
    reconcile_pending_chars: int = 12_000
    max_events_per_reconcile: int = 40
    overlap_events: int = 3
    event_render_chars: int = 700
    context_budget_chars: int = 6_000
    bootstrap_budget_chars: int = 12_000
    commit_attempts: int = 2

    def __post_init__(self) -> None:
        at_least_one = (
            "max_goals",
            "max_facts",
            "max_hypotheses",
            "max_state_entries",
            "max_text_chars",
            "max_state_value_chars",
            "reconcile_every_events",
            "reconcile_pending_chars",
            "max_events_per_reconcile",
            "event_render_chars",
            "context_budget_chars",
            "bootstrap_budget_chars",
            "commit_attempts",
        )
        for name in at_least_one:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        for name in ("max_archive_entries", "overlap_events"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")


_RECONCILE_SYSTEM_PROMPT = (
    "You maintain structured working memory for a terminal-game agent. "
    'Return ONLY a JSON array of operation objects ("[]" if there is nothing new). Operations:\n'
    '{"op":"add","section":"goal|fact|hypothesis","text":"...","priority":1-5,"context":"..."}\n'
    '{"op":"revise","id":"m3","text":"...","priority":1-5,"context":"..."}\n'
    '{"op":"set","key":"location","value":"..."}\n'
    '{"op":"promote","id":"m5"}  (hypothesis confirmed -> fact)\n'
    '{"op":"contradict","id":"m2","replacement":"optional corrected text"}\n'
    '{"op":"transition","id":"m1","status":"open|blocked|done|abandoned","reason":"..."}\n'
    "Rules: anything you do not mention persists unchanged — never re-add existing items. "
    "Record confirmed observations as facts and unverified beliefs as hypotheses. "
    "Close goals the evidence completed or contradicted; keep goals prioritized. "
    "Keep mutable values (location, inventory, scores) in state keys, not facts. "
    "Keep reasoning concise and reserve enough output for the required JSON; emit the JSON as soon as "
    "the memory update is determined."
)

_AUDIT_EXTRA_PROMPT = (
    "This is a cleanup-only final audit over the resulting memory. There is no new activity evidence to extract. "
    "Inspect every active entry for contradictions, stale mutable snapshots recorded as timeless facts, duplicate "
    "concepts under different names, and goals whose existing memory already proves a status change. Return only "
    "revise, contradict without a replacement, or transition operations. Do not add or promote items and do not set "
    "state. Prefer keeping specific stable facts; contradict an older stale entry when a newer entry already captures "
    "the current truth. Preserve useful historical facts by revising them into clearly historical wording when "
    'appropriate. Return "[]" when no cleanup is warranted.'
)


def _with_prompt_appendix(base: str, appendix: str) -> str:
    appendix = appendix.strip()
    if not appendix:
        return base
    return f"{base}\n\nAdditional policy:\n{appendix}"


class StructuredMemorySubsystem:
    name = "structured"

    def __init__(
            self,
            root: str | Path,
            config: StructuredMemoryConfig | None = None,
            reconcile_prompt_appendix: str = "",
            audit_prompt_appendix: str = "",
    ) -> None:
        self.root = Path(root)
        self.config = config or StructuredMemoryConfig()
        self.reconcile_system_prompt = _with_prompt_appendix(
            _RECONCILE_SYSTEM_PROMPT,
            reconcile_prompt_appendix,
        )
        self.audit_extra_prompt = _with_prompt_appendix(
            _AUDIT_EXTRA_PROMPT,
            audit_prompt_appendix,
        )

    def fingerprints(self) -> dict[str, str]:
        schema = _digest(
            {
                "schema_version": _SCHEMA_VERSION,
                "sections": list(_SECTIONS),
                "goal_statuses": list(_GOAL_STATUSES),
            }
        )
        mutation = _digest(
            {
                "caps": {
                    "goals": self.config.max_goals,
                    "facts": self.config.max_facts,
                    "hypotheses": self.config.max_hypotheses,
                    "state": self.config.max_state_entries,
                    "text": self.config.max_text_chars,
                    "state_value": self.config.max_state_value_chars,
                    "archive": self.config.max_archive_entries,
                },
                "cadence": {
                    "every_events": self.config.reconcile_every_events,
                    "pending_chars": self.config.reconcile_pending_chars,
                    "max_events_per_reconcile": self.config.max_events_per_reconcile,
                    "overlap_events": self.config.overlap_events,
                    "commit_attempts": self.config.commit_attempts,
                },
                "render": {
                    "event_chars": self.config.event_render_chars,
                    "context_budget": self.config.context_budget_chars,
                    "bootstrap_budget": self.config.bootstrap_budget_chars,
                },
                "legal_goal_moves": {key: sorted(value) for key, value in _LEGAL_GOAL_MOVES.items()},
                "capacity": "code_owned_lowest_priority_then_oldest_numeric_id",
            }
        )
        prompts = _digest(
            {
                "reconcile": self.reconcile_system_prompt,
                "audit": self.audit_extra_prompt,
            }
        )
        return {"schema": schema, "mutation": mutation, "prompt": prompts}

    def open_context(self, agent_id: str, context_id: str) -> "StructuredMemoryHandle":
        key = MemoryContextKey(root=self.root, agent_id=agent_id, context_id=context_id)
        return StructuredMemoryHandle(
            key,
            self.config,
            self.fingerprints(),
            self.reconcile_system_prompt,
            self.audit_extra_prompt,
        )


class StructuredMemoryHandle:
    """One locked, open memory context."""

    def __init__(
            self,
            key: MemoryContextKey,
            config: StructuredMemoryConfig,
            fingerprints: dict[str, str],
            reconcile_system_prompt: str,
            audit_extra_prompt: str,
    ) -> None:
        self.key = key
        self.config = config
        self.fingerprints = fingerprints
        self.reconcile_system_prompt = reconcile_system_prompt
        self.audit_extra_prompt = audit_extra_prompt
        self._directory = key.directory()
        self._directory.mkdir(parents=True, exist_ok=True)
        self._journal_path = self._directory / "ops.jsonl"
        self._store_path = self._directory / "store.json"
        # Never delete the lock file: flock identity is the inode.
        self._lock_handle: BinaryIO | None = (self._directory / "memory.lock").open("a+b")
        try:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock_handle.close()
            self._lock_handle = None
            raise RuntimeError(f"memory context is already in use: {self._directory}") from exc
        try:
            self._check_manifest_compatibility()
            self.memory = self._replay_journal()
            self._write_store_cache()
            self._write_manifest()
        except BaseException:
            # A replay/cache failure must not strand the flock behind the
            # exception's live traceback reference to this handle.
            self.close()
            raise
        self._pending: list[MemoryEvent] = []
        self._covered_tail: list[MemoryEvent] = []
        self._batch_records: list[dict[str, Any]] = []
        self._events_seen = 0
        self._retry_after_seen = 0

    # ------------------------------------------------------------------ seam

    def observe(self, event: MemoryEvent) -> None:
        self._pending.append(event)
        self._events_seen += 1

    def render_context(self, budget_chars: int | None = None) -> str:
        return self._render(budget_chars or self.config.context_budget_chars)

    def render_bootstrap(self, budget_chars: int | None = None) -> str:
        budget = budget_chars or self.config.bootstrap_budget_chars
        if not self._pending:
            return self._render(budget)
        # A bootstrap after a failed catch-up reconciliation must still carry
        # the evidence a fresh chain would otherwise never see: reserve up to
        # half the budget for pending evidence so committed memory cannot
        # clip it away entirely.
        header = "\n\nRecent activity not yet reconciled into memory:\n"
        pending_text = header + self._events_text(self._pending)
        pending_budget = min(len(pending_text), budget // 2)
        memory_text = self._render(budget - pending_budget)
        return memory_text + _clip_with_marker(pending_text, budget - len(memory_text))

    def maybe_reconcile(self, model: ModelAdapter, *, force: bool = False) -> ReconcileOutcome | None:
        if not self._pending:
            return None
        if not force and not self._cadence_due():
            return None
        return self._reconcile(model, batch="reconcile", extra_evidence="")

    def commit(self, model: ModelAdapter, extra_evidence: str = "") -> CommitOutcome:
        tally = _CommitTally()
        # Phase 1: extract pending terminal evidence in ordinary bounded
        # reconciliation prefixes. This deliberately uses the same prompt and
        # provider settings as periodic reconciliation.
        # Every success consumes at least one event and failures are capped,
        # so this ends.
        while self._pending:
            outcome = self._reconcile(model, batch="commit", extra_evidence="")
            if not self._commit_step(tally, outcome) and tally.failures >= self.config.commit_attempts:
                return self._commit_failed(tally)
        # Additional social evidence is extraction input too, but is presented
        # exactly once rather than repeated for every backlog prefix.
        if extra_evidence:
            while True:
                outcome = self._reconcile(
                    model,
                    batch="commit",
                    extra_evidence=extra_evidence,
                    allow_empty_pending=True,
                )
                if self._commit_step(tally, outcome):
                    break
                if tally.failures >= self.config.commit_attempts:
                    return self._commit_failed(tally)

        # Phase 2: make one independent, eventless cleanup pass over the
        # resulting active memory. Its operation allowlist prevents an audit
        # from growing memory or triggering capacity eviction. Audit failure
        # is intentionally nonfatal: all successful extraction above remains
        # committed and observable in the journal.
        audit = self._reconcile(
            model,
            batch="audit",
            extra_evidence="",
            allow_empty_pending=True,
            allowed_operations=_AUDIT_ALLOWED_OPERATIONS,
        )
        tally.attempts += 1
        tally.usage = merge_usage(tally.usage, audit.usage)
        audit_error = ""
        if audit.status == "failed":
            audit_error = audit.error
        else:
            tally.accepted += audit.accepted_ops
            tally.rejected += audit.rejected_ops
        return CommitOutcome(
            status="applied" if tally.accepted else "no_memory_change",
            attempts=tally.attempts,
            accepted_ops=tally.accepted,
            rejected_ops=tally.rejected,
            usage=tally.usage,
            audit_status=audit.status,
            audit_accepted_ops=audit.accepted_ops,
            audit_rejected_ops=audit.rejected_ops,
            audit_error=audit_error,
        )

    def _commit_step(self, tally: "_CommitTally", outcome: ReconcileOutcome) -> bool:
        """Fold one reconcile outcome into the tally; False means it failed."""

        tally.attempts += 1
        tally.usage = merge_usage(tally.usage, outcome.usage)
        if outcome.status == "failed":
            tally.failures += 1
            tally.error = outcome.error
            return False
        tally.accepted += outcome.accepted_ops
        tally.rejected += outcome.rejected_ops
        return True

    def _commit_failed(self, tally: "_CommitTally") -> CommitOutcome:
        self._journal_marker("commit_failed", batch="commit", reason=tally.error)
        return CommitOutcome(
            status="failed",
            attempts=tally.attempts,
            accepted_ops=tally.accepted,
            rejected_ops=tally.rejected,
            error=tally.error,
            usage=tally.usage,
        )

    def close(self) -> None:
        if self._lock_handle is not None:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            self._lock_handle.close()
            self._lock_handle = None

    # ------------------------------------------------------------- reconcile

    def _cadence_due(self) -> bool:
        if self._events_seen < self._retry_after_seen:
            return False
        if len(self._pending) >= self.config.reconcile_every_events:
            return True
        return sum(event.chars() for event in self._pending) >= self.config.reconcile_pending_chars

    def _reconcile(
            self,
            model: ModelAdapter,
            *,
            batch: str,
            extra_evidence: str,
            allow_empty_pending: bool = False,
            allowed_operations: frozenset[str] | None = None,
    ) -> ReconcileOutcome:
        if not self._pending and not allow_empty_pending:
            return ReconcileOutcome(status="no_memory_change")
        # Bounded contiguous prefix: a large backlog must not grow the prompt
        # without limit (especially across failure retries); the remainder
        # stays pending for the next pass.
        events = self._pending[:self.config.max_events_per_reconcile]
        prompt = self._build_prompt(batch, extra_evidence, events)
        usage: dict[str, Any] | None = None
        try:
            ops = self._request_operations(model, prompt, batch)
            usage = merge_usage(usage, getattr(model, "last_usage", None))
            if not ops and events:
                # One mechanical retry for the lazy-empty case, then advance
                # with an explicit marker; semantic omission stays measurable.
                ops = self._request_operations(model, prompt, batch)
                usage = merge_usage(usage, getattr(model, "last_usage", None))
        except (ModelError, ValueError) as exc:
            usage = merge_usage(usage, getattr(model, "last_usage", None))
            if batch == "audit":
                self._journal_marker("audit_failed", batch=batch, reason=str(exc))
            else:
                self._retry_after_seen = self._events_seen + max(1, self.config.reconcile_every_events)
                self._journal_marker("reconcile_failed", batch=batch, reason=str(exc))
            return ReconcileOutcome(status="failed", error=str(exc), usage=usage)

        source_steps = tuple(event.step for event in events)
        records: list[dict[str, Any]] = []
        self._batch_records = records
        accepted = 0
        rejected = 0
        for op in ops:
            record = self._apply_operation(
                op,
                batch=batch,
                source_steps=source_steps,
                allowed_operations=allowed_operations,
            )
            if record["accepted"]:
                accepted += 1
            else:
                rejected += 1
        if accepted == 0:
            records.append(
                mutation_record(
                    op="no_memory_change",
                    origin="system",
                    accepted=True,
                    batch=batch,
                    source_steps=source_steps,
                )
            )
        # Journal before the covered boundary advances; the journal is the
        # source of truth and the store document is a cache. Handlers append
        # every record — including system capacity demotes — at its exact
        # application point, so any journal prefix replays to a live state.
        self._journal_records(records)
        self._write_store_cache()
        self._advance_covered(events)
        if batch != "audit":
            self._retry_after_seen = 0
        status = "applied" if accepted else "no_memory_change"
        return ReconcileOutcome(
            status=status,
            accepted_ops=accepted,
            rejected_ops=rejected,
            covered_steps=len(events),
            usage=usage,
        )

    def _request_operations(self, model: ModelAdapter, prompt: CompactionPrompt, batch: str) -> list[Any]:
        # Final extraction uses the normal reconciliation seam. Only the
        # independent cleanup pass uses audit settings.
        operation = "audit" if batch == "audit" else "compaction"
        return _operations_from_text(model.utility_text(prompt.messages(), operation))

    def _build_prompt(self, batch: str, extra_evidence: str, events: list[MemoryEvent]) -> CompactionPrompt:
        memory_text = self._render_all() if batch == "audit" else self._render(self.config.bootstrap_budget_chars)
        sections = [f"Current structured memory (ids are stable):\n{memory_text}"]
        if self._covered_tail and batch != "audit":
            sections.append(
                "Already-recorded recent activity (context only, do not re-record):\n"
                + self._events_text(self._covered_tail)
            )
        if events:
            sections.append(f"New activity to fold in:\n{self._events_text(events)}")
        if extra_evidence:
            sections.append(f"Additional evidence:\n{extra_evidence}")
        if batch == "audit":
            sections.append(self.audit_extra_prompt)
        return CompactionPrompt(system=self.reconcile_system_prompt, user="\n\n".join(sections))

    def _advance_covered(self, events: list[MemoryEvent]) -> None:
        overlap = self.config.overlap_events
        tail = self._covered_tail + events
        self._covered_tail = tail[-overlap:] if overlap > 0 else []
        self._pending = self._pending[len(events) :]

    # ------------------------------------------------------------ operations

    def _emit(self, record: dict[str, Any]) -> dict[str, Any]:
        """Journal a record and apply it through the canonical reducer.

        Live application and replay share ``_apply_record``, so the store is
        the fold of the journal by construction — there is no separate live
        mutation code that could diverge from replay.
        """

        self._batch_records.append(record)
        _apply_record(self.memory, record, self.config.max_archive_entries)
        return record

    def _apply_operation(
            self,
            op: Any,
            *,
            batch: str,
            source_steps: tuple[int, ...],
            allowed_operations: frozenset[str] | None = None,
    ) -> dict[str, Any]:
        """Validate one model operation and emit its record(s) in order."""

        def reject(name: Any, reason: str, fields: dict[str, Any]) -> dict[str, Any]:
            return self._emit(
                mutation_record(
                    op=str(name),
                    origin="model",
                    accepted=False,
                    batch=batch,
                    reason=reason,
                    source_steps=source_steps,
                    fields=fields,
                )
            )

        if not isinstance(op, dict):
            return reject("invalid", "operation must be a JSON object", {"value": _clipped(str(op), 200)})
        name = op.get("op")
        if not isinstance(name, str):
            return reject(
                "invalid",
                "op must be a string",
                {"value": _clipped(str(name), 200)},
            )
        op_fields = {key: value for key, value in op.items() if key != "op"}
        if allowed_operations is not None and name not in allowed_operations:
            return reject(name, f"{batch} batch does not allow {name!r}", op_fields)
        if batch == "audit" and name == "contradict" and str(op.get("replacement", "")).strip():
            return reject(name, "audit contradictions cannot add a replacement", op_fields)
        handlers = {
            "add": self._op_add,
            "revise": self._op_revise,
            "set": self._op_set,
            "promote": self._op_promote,
            "contradict": self._op_contradict,
            "transition": self._op_transition,
        }
        handler = handlers.get(name)
        if handler is None:
            return reject(name, f"unknown op {name!r}", op_fields)
        try:
            return handler(op, batch=batch, source_steps=source_steps)
        except _OpRejected as exc:
            return reject(name, str(exc), op_fields)

    def _op_add(self, op: dict[str, Any], *, batch: str, source_steps: tuple[int, ...]) -> dict[str, Any]:
        section = op.get("section")
        if section not in _SECTIONS:
            raise _OpRejected(f"unknown section {section!r}")
        text = self._required_text(op, "text")
        if self._duplicate_text(section, text):
            raise _OpRejected("duplicate of an existing item")
        fields: dict[str, Any] = {"text": text}
        if section == "goal":
            fields["priority"] = _bounded_int(op.get("priority", 3), 1, 5)
            fields["context"] = _clipped(str(op.get("context", "")), self.config.max_text_chars)
        record = self._emit(
            mutation_record(
                op="add",
                origin="model",
                accepted=True,
                batch=batch,
                section=section,
                item_id=self._issue_id(),
                source_steps=source_steps,
                fields=fields,
            )
        )
        self._enforce_capacity(section, batch=batch)
        return record

    def _op_revise(self, op: dict[str, Any], *, batch: str, source_steps: tuple[int, ...]) -> dict[str, Any]:
        section, item = self._require_item(op)
        if section == "goal":
            fields: dict[str, Any] = {
                "text": _clipped(str(op.get("text", item.text)).strip(), self.config.max_text_chars) or item.text,
                "priority": _bounded_int(op.get("priority", item.priority), 1, 5),
                "context": _clipped(str(op.get("context", item.context)), self.config.max_text_chars),
            }
        else:
            text = self._required_text(op, "text")
            if self._duplicate_text(section, text, ignore_id=item.id):
                raise _OpRejected("duplicate of an existing item")
            fields = {"text": text}
        return self._emit(
            mutation_record(
                op="revise",
                origin="model",
                accepted=True,
                batch=batch,
                section=section,
                item_id=item.id,
                source_steps=source_steps,
                fields=fields,
            )
        )

    def _op_set(self, op: dict[str, Any], *, batch: str, source_steps: tuple[int, ...]) -> dict[str, Any]:
        key = _WS_RE.sub("_", str(op.get("key", "")).strip().lower())
        if not key or len(key) > 64:
            raise _OpRejected("state key must be 1-64 characters")
        value = _clipped(str(op.get("value", "")).strip(), self.config.max_state_value_chars)
        if not value:
            raise _OpRejected("state value must be non-empty")
        if key not in self.memory.state and len(self.memory.state) >= self.config.max_state_entries:
            # Evicted before the set applies — the order it happens live — so
            # replay evicts the same key even when that key is re-set later.
            oldest = next(iter(self.memory.state))
            self._emit(
                mutation_record(
                    op="demote",
                    origin="system",
                    accepted=True,
                    batch=batch,
                    section="state",
                    item_id=oldest,
                    reason="capacity",
                    fields={"key": oldest, "value": self.memory.state[oldest]},
                )
            )
        return self._emit(
            mutation_record(
                op="set",
                origin="model",
                accepted=True,
                batch=batch,
                section="state",
                item_id=key,
                source_steps=source_steps,
                fields={"key": key, "value": value},
            )
        )

    def _op_promote(self, op: dict[str, Any], *, batch: str, source_steps: tuple[int, ...]) -> dict[str, Any]:
        section, item = self._require_item(op)
        if section != "hypothesis":
            raise _OpRejected("promote applies to hypotheses only")
        if self._duplicate_text("fact", item.text):
            raise _OpRejected("an equivalent fact already exists")
        record = self._emit(
            mutation_record(
                op="promote",
                origin="model",
                accepted=True,
                batch=batch,
                section="fact",
                item_id=item.id,
                source_steps=source_steps,
                fields={"text": item.text},
            )
        )
        self._enforce_capacity("fact", batch=batch)
        return record

    def _op_contradict(self, op: dict[str, Any], *, batch: str, source_steps: tuple[int, ...]) -> dict[str, Any]:
        section, item = self._require_item(op)
        if section == "goal":
            raise _OpRejected("contradict applies to facts and hypotheses; use transition for goals")
        record = self._emit(
            mutation_record(
                op="contradict",
                origin="model",
                accepted=True,
                batch=batch,
                section=section,
                item_id=item.id,
                source_steps=source_steps,
                fields={"text": item.text},
            )
        )
        replacement = _clipped(str(op.get("replacement", "")).strip(), self.config.max_text_chars)
        if replacement and not self._duplicate_text("fact", replacement):
            # The replacement is an ordinary fact addition — journaled as its
            # own model-origin `add` so analyzers counting additions see it —
            # and it counts against the same cap as any addition.
            self._emit(
                mutation_record(
                    op="add",
                    origin="model",
                    accepted=True,
                    batch=batch,
                    section="fact",
                    item_id=self._issue_id(),
                    reason="replacement",
                    source_steps=source_steps,
                    fields={"text": replacement, "replaces": item.id},
                )
            )
            self._enforce_capacity("fact", batch=batch)
        return record

    def _op_transition(self, op: dict[str, Any], *, batch: str, source_steps: tuple[int, ...]) -> dict[str, Any]:
        section, item = self._require_item(op)
        if section != "goal":
            raise _OpRejected("transition applies to goals only")
        status = op.get("status")
        if status not in _GOAL_STATUSES:
            raise _OpRejected(f"unknown goal status {status!r}")
        if status not in _LEGAL_GOAL_MOVES[item.status]:
            raise _OpRejected(f"illegal transition {item.status} -> {status}")
        reason = _clipped(str(op.get("reason", "")).strip(), self.config.max_text_chars)
        if status == "abandoned" and not reason:
            raise _OpRejected("abandoning a goal requires a reason")
        return self._emit(
            mutation_record(
                op="transition",
                origin="model",
                accepted=True,
                batch=batch,
                section="goal",
                item_id=item.id,
                source_steps=source_steps,
                fields={"status": status, "reason": reason},
            )
        )

    def _enforce_capacity(self, section: str, *, batch: str) -> None:
        cap, items = {
            "goal": (self.config.max_goals, self.memory.goals),
            "fact": (self.config.max_facts, self.memory.facts),
            "hypothesis": (self.config.max_hypotheses, self.memory.hypotheses),
        }[section]
        while len(items) > cap:
            victim = self._capacity_victim(section, items)
            self._emit(
                mutation_record(
                    op="demote",
                    origin="system",
                    accepted=True,
                    batch=batch,
                    section=section,
                    item_id=victim.id,
                    reason="capacity",
                    fields=victim.to_dict(),
                )
            )

    def _capacity_victim(self, section: str, items: list) -> Any:
        if section == "goal":
            # Lowest priority first (5 is lowest), oldest numeric id tiebreak.
            return max(items, key=lambda goal: (goal.priority, -_numeric_id(goal.id)))
        return min(items, key=lambda item: _numeric_id(item.id))

    # -------------------------------------------------------------- plumbing

    def _issue_id(self) -> str:
        # Peek only: the reducer advances next_id when the record applies.
        return f"m{self.memory.next_id}"

    def _require_item(self, op: dict[str, Any]) -> tuple[str, Any]:
        item_id = str(op.get("id", ""))
        found = self.memory.find(item_id)
        if found is None:
            raise _OpRejected(f"unknown item id {item_id!r}")
        return found

    def _required_text(self, op: dict[str, Any], key: str) -> str:
        text = _clipped(str(op.get(key, "")).strip(), self.config.max_text_chars)
        if not text:
            raise _OpRejected(f"{key} must be a non-empty string")
        return text

    def _duplicate_text(self, section: str, text: str, ignore_id: str = "") -> bool:
        normalized = _normalized(text)
        pools = {
            "goal": self.memory.goals,
            "fact": self.memory.facts,
            "hypothesis": self.memory.hypotheses,
        }
        return any(
            _normalized(item.text) == normalized and item.id != ignore_id
            for item in pools[section]
        )

    def _events_text(self, events: list[MemoryEvent]) -> str:
        # Chronological within a step: the screen was observed first, then
        # the action was taken in response to it — rendering the action first
        # would make the screen read as the action's result.
        blocks = []
        for event in events:
            lines = [f"Step {event.step}:"]
            if event.observation:
                lines.append(f"  observed: {_clipped(event.observation, self.config.event_render_chars)}")
            if event.intent:
                lines.append(f"  intent: {event.intent}")
            if event.action:
                lines.append(f"  action in response: {event.action}")
            for warning in event.warnings:
                lines.append(f"  warning: {warning}")
            blocks.append("\n".join(lines))
        return "\n".join(blocks)

    def _render(self, budget_chars: int) -> str:
        sections = self._render_sections()
        if not sections:
            return "(no structured memory yet)"
        return _fit_sections(sections, budget_chars)

    def _render_all(self) -> str:
        """Render every bounded active entry for the final audit."""

        sections = self._render_sections()
        return "\n\n".join(sections) if sections else "(no structured memory yet)"

    def _render_sections(self) -> list[str]:
        sections: list[str] = []
        if self.memory.state:
            entries = "\n".join(f"- {key}: {value}" for key, value in sorted(self.memory.state.items()))
            sections.append(f"State:\n{entries}")
        if self.memory.goals:
            ordered = sorted(self.memory.goals, key=lambda goal: (goal.priority, _numeric_id(goal.id)))
            lines = []
            for goal in ordered:
                suffix = f" [{goal.status}: {goal.context}]" if goal.status == "blocked" else f" [{goal.status}]"
                lines.append(f"- ({goal.id}, p{goal.priority}) {goal.text}{suffix}")
            sections.append("Goals:\n" + "\n".join(lines))
        if self.memory.facts:
            sections.append("Facts:\n" + "\n".join(f"- ({item.id}) {item.text}" for item in self.memory.facts))
        if self.memory.hypotheses:
            sections.append(
                "Hypotheses (unverified):\n"
                + "\n".join(f"- ({item.id}) {item.text}" for item in self.memory.hypotheses)
            )
        return sections

    # -------------------------------------------------------------- journal

    def _journal_records(self, records: list[dict[str, Any]]) -> None:
        write_journal_records(self._journal_path, records)

    def _journal_marker(self, op: str, *, batch: str, reason: str = "") -> None:
        self._journal_records(
            [mutation_record(op=op, origin="system", accepted=False, batch=batch, reason=reason)]
        )

    def _check_manifest_compatibility(self) -> None:
        """Refuse to replay a journal written under an incompatible config.

        Replay folds the journal under the *current* caps, so reopening with
        changed mutation rules (e.g. a different archive cap) would silently
        produce a different store than the one the journal's writer saw. The
        prompt fingerprint is deliberately exempt: prompt text changes what
        the model is asked, never how records fold.
        """

        path = self._directory / "manifest.json"
        if not path.exists():
            return
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(f"unreadable memory manifest {path}: {exc}") from exc
        stored_prints = stored.get("fingerprints", {})
        for key in ("schema", "mutation"):
            if stored_prints.get(key) != self.fingerprints.get(key):
                raise RuntimeError(
                    f"memory context {self._directory} was written under a different {key} "
                    f"fingerprint ({stored_prints.get(key)!r} != {self.fingerprints.get(key)!r}); "
                    "replaying its journal under the current configuration would change memory. "
                    "Use a fresh memory root or restore the original configuration."
                )

    def _write_manifest(self) -> None:
        # The arm identity for metric tooling: which subsystem produced this
        # context's journal, under which fingerprints and config.
        manifest = {
            "subsystem": "structured",
            "fingerprints": self.fingerprints,
            "config": asdict(self.config),
        }
        (self._directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _write_store_cache(self) -> None:
        tmp_path = self._store_path.with_name(f".{self._store_path.name}.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(self.memory.to_dict(), handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            tmp_path.replace(self._store_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def _replay_journal(self) -> StructuredMemory:
        memory = StructuredMemory()
        if not self._journal_path.exists():
            return memory
        raw = self._journal_path.read_bytes()
        offset = 0
        for line_number, line in enumerate(raw.split(b"\n"), start=1):
            if not line.strip():
                offset += len(line) + 1
                continue
            try:
                parsed = json.loads(line.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                if offset + len(line) >= len(raw):
                    # Torn trailing batch from a crash mid-append: truncate
                    # and continue — the batch never applied, atomically.
                    # Anything else is corruption and raises.
                    with self._journal_path.open("r+b") as handle:
                        handle.truncate(offset)
                    break
                raise ValueError(
                    f"corrupt memory journal {self._journal_path}:{line_number}: {exc}"
                ) from exc
            if isinstance(parsed, dict) and isinstance(parsed.get("records"), list):
                for record in parsed["records"]:
                    if isinstance(record, dict):
                        _apply_record(memory, record, self.config.max_archive_entries)
            elif isinstance(parsed, dict):
                _apply_record(memory, parsed, self.config.max_archive_entries)
            offset += len(line) + 1
        return memory


class _OpRejected(Exception):
    """Internal: one operation failed validation; others still apply."""


@dataclass
class _CommitTally:
    """Running totals across the reconcile passes of one commit."""

    attempts: int = 0
    failures: int = 0
    accepted: int = 0
    rejected: int = 0
    error: str = ""
    usage: dict[str, Any] | None = None


def _clip_with_marker(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = "\n[truncated to budget]"
    return text[: max(0, limit - len(marker))] + marker


def _fit_sections(sections: list[str], budget_chars: int) -> str:
    """Join sections within the budget, sharing space fairly.

    A plain prefix cut lets a long early section (facts) starve later ones
    (hypotheses) entirely. Shortest-first fair shares instead: each section
    gets at most an equal split of what remains, sections that fit keep
    everything, and their surplus rolls over to the longer ones.
    """

    remaining = max(0, budget_chars - 2 * (len(sections) - 1))
    budgets = [0] * len(sections)
    order = sorted(range(len(sections)), key=lambda index: len(sections[index]))
    for position, index in enumerate(order):
        share = remaining // (len(sections) - position)
        budgets[index] = min(len(sections[index]), share)
        remaining -= budgets[index]
    return "\n\n".join(
        _clip_with_marker(section, budgets[index])
        for index, section in enumerate(sections)
        if budgets[index] > 0
    )


def _apply_record(memory: StructuredMemory, record: dict[str, Any], max_archive: int) -> None:
    """The canonical mutation reducer: the store is the fold of the journal.

    Both live application (via ``_emit``) and replay-on-open run every record
    through this one function, so they cannot diverge. Records that fail
    validation, markers, and unknown ops are no-ops.
    """

    if not record.get("accepted"):
        return
    op = record.get("op")
    section = record.get("section", "")
    item_id = str(record.get("item_id", ""))
    fields = record.get("fields", {})
    source_steps = tuple(int(step) for step in record.get("source_steps", []))
    _track_next_id(memory, item_id)

    def archive(entry_section: str, exit_reason: str, item: dict[str, Any]) -> None:
        memory.archive.append({"section": entry_section, "exit": exit_reason, "item": item})
        overflow = len(memory.archive) - max_archive
        if overflow > 0:
            # Oldest-first eviction; evicted entries survive in the journal.
            del memory.archive[:overflow]

    if op == "add":
        if section == "goal":
            memory.goals.append(
                SimpleGoal(
                    id=item_id,
                    text=fields.get("text", ""),
                    priority=int(fields.get("priority", 3)),
                    context=fields.get("context", ""),
                )
            )
        elif section == "fact":
            memory.facts.append(SimpleItem(id=item_id, text=fields.get("text", ""), source_steps=source_steps))
        elif section == "hypothesis":
            memory.hypotheses.append(
                SimpleItem(id=item_id, text=fields.get("text", ""), source_steps=source_steps)
            )
    elif op == "revise":
        found = memory.find(item_id)
        if found is None:
            return
        found_section, item = found
        if found_section == "goal":
            memory.goals[memory.goals.index(item)] = replace(
                item,
                text=fields.get("text", item.text),
                priority=int(fields.get("priority", item.priority)),
                context=fields.get("context", item.context),
            )
        else:
            items = memory.facts if found_section == "fact" else memory.hypotheses
            items[items.index(item)] = replace(
                item,
                text=fields.get("text", item.text),
                source_steps=item.source_steps + source_steps,
            )
    elif op == "set":
        memory.state[fields.get("key", item_id)] = fields.get("value", "")
    elif op == "promote":
        found = memory.find(item_id)
        if found is not None and found[0] == "hypothesis":
            memory.hypotheses.remove(found[1])
            memory.facts.append(replace(found[1], source_steps=found[1].source_steps + source_steps))
    elif op == "contradict":
        found = memory.find(item_id)
        if found is not None and found[0] in {"fact", "hypothesis"}:
            items = memory.facts if found[0] == "fact" else memory.hypotheses
            items.remove(found[1])
            archive(found[0], "contradicted", found[1].to_dict())
    elif op == "transition":
        found = memory.find(item_id)
        if found is not None and found[0] == "goal":
            goal = found[1]
            status = fields.get("status", goal.status)
            updated = replace(goal, status=status, context=fields.get("reason") or goal.context)
            memory.goals[memory.goals.index(goal)] = updated
            if status in {"done", "abandoned"}:
                memory.goals.remove(updated)
                archive("goal", "goal_closed", updated.to_dict())
    elif op == "demote":
        if section == "state":
            key = fields.get("key", item_id)
            value = memory.state.pop(key, None)
            if value is not None:
                archive("state", "capacity", {"key": key, "value": value})
            return
        found = memory.find(item_id)
        if found is not None:
            found_section, item = found
            pools = {
                "goal": memory.goals,
                "fact": memory.facts,
                "hypothesis": memory.hypotheses,
            }
            pools[found_section].remove(item)
            archive(found_section, "capacity", item.to_dict())


def _track_next_id(memory: StructuredMemory, item_id: str) -> None:
    number = _numeric_id(item_id)
    if number >= memory.next_id:
        memory.next_id = number + 1


def _numeric_id(item_id: str) -> int:
    try:
        return int(str(item_id).lstrip("m"))
    except ValueError:
        return 0


def _normalized(text: str) -> str:
    return _WS_RE.sub(" ", text.strip().casefold())


def _clipped(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit]


def _bounded_int(value: Any, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return low + (high - low) // 2
    return max(low, min(high, number))


def _digest(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _operations_from_text(text: str) -> list[Any]:
    """Extract a JSON array of operation objects, tolerating code fences."""

    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL)
    if fence is not None:
        stripped = fence.group(1).strip()
    start = stripped.find("[")
    end = stripped.rfind("]")
    if start < 0 or end <= start:
        wrapper_start = stripped.find("{")
        if wrapper_start >= 0:
            try:
                wrapper = json.loads(stripped[wrapper_start : stripped.rfind("}") + 1])
            except json.JSONDecodeError as exc:
                raise ValueError(f"memory response contained no operation array: {text[:200]!r}") from exc
            operations = wrapper.get("operations")
            if isinstance(operations, list):
                return list(operations)
        raise ValueError(f"memory response contained no operation array: {text[:200]!r}")
    try:
        parsed = json.loads(stripped[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"memory response was not valid JSON: {exc}") from exc
    if not isinstance(parsed, list):
        raise ValueError("memory response must be a JSON array of operations")
    # Malformed elements are kept: validation rejects them with a journaled
    # record instead of silently dropping them here.
    return list(parsed)
