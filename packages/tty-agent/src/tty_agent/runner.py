"""Single-agent activity runner for terminal sessions."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from .agent import ActionExecution, TerminalAgent
from .actions import Action, ActionError, ActionPolicy, render_action_schema
from .evaluation import EvaluationProfile, EvaluationRecord, EvaluationResult, EvaluationSource
from .hints import InputModalityProfile, ObservationHints
from .ids import validate_agent_id
from .memory import JsonMemoryStore, MemoryDocumentLimits, bound_memory_document
from .memory_subsystem import (
    MemoryEvent,
    MemoryHandle,
    MemorySubsystem,
    mutation_record,
    write_journal_records,
)
from .models import (
    CompactionPrompt,
    DecisionPrompt,
    MemoryCommitPrompt,
    MemoryPatch,
    ModelAdapter,
    ModelError,
    ModelOutputTruncated,
    ModelStateError,
    SessionSummary,
)
from .prompt_modules import (
    DEFAULT_STABLE_LEVELS,
    DEFAULT_TACTICAL_LEVELS,
    GENERIC_TERMINAL_MODULES,
    PROMPT_MODULES_SCHEMA_VERSION,
    AssistanceLevel,
    PromptModule,
    PromptModuleResult,
    PromptRenderContext,
    collect_prompt_module_results,
    prompt_module_trace,
    render_prompt_modules,
)
from .transports.base import SessionDisconnected
from .terminal import Observation

PromptMode = Literal["stateless_full", "stateful_delta"]
PromptLayout = Literal["timeline_first", "cache_friendly"]
PromptStage = Literal["full", "bootstrap", "delta"]


def _recent_window(steps: list["StepRecord"], keep: int) -> list["StepRecord"]:
    # steps[-0:] would return the whole list, silently unbounding the prompt.
    return steps[-keep:] if keep > 0 else []


@dataclass
class ActivityBudget:
    max_decision_ticks: int = 50
    max_wall_seconds: float = 300.0
    max_validation_failures: int = 5
    started_at: float = field(default_factory=time.monotonic)
    decision_ticks: int = 0
    validation_failures: int = 0

    def remaining(self) -> bool:
        return self.decision_ticks < self.max_decision_ticks and self.wall_seconds_remaining() > 0

    def wall_seconds_remaining(self) -> float:
        return max(0.0, self.max_wall_seconds - (time.monotonic() - self.started_at))

    def tick_remaining(self) -> int:
        return max(0, self.max_decision_ticks - self.decision_ticks)

    def consume_tick(self) -> None:
        self.decision_ticks += 1

    def record_validation_failure(self) -> None:
        self.validation_failures += 1

    def too_many_validation_failures(self) -> bool:
        return self.validation_failures >= self.max_validation_failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_decision_ticks": self.max_decision_ticks,
            "decision_ticks": self.decision_ticks,
            "decision_ticks_remaining": self.tick_remaining(),
            "max_wall_seconds": self.max_wall_seconds,
            "wall_seconds_remaining": self.wall_seconds_remaining(),
            "validation_failures": self.validation_failures,
            "max_validation_failures": self.max_validation_failures,
        }


@dataclass(frozen=True)
class LegacyMemoryLimits:
    """Bounded working and durable memory for the legacy rewrite path."""

    summary_max_chars: int = 8_000
    current_state_max_chars: int = 1_000
    last_error_max_chars: int = 500
    item_max_chars: int = 400
    max_open_subgoals: int = 10
    max_discovered_facts: int = 40
    max_failed_actions: int = 12
    max_strategy_notes: int = 12
    repair_min_previous_items: int = 10
    repair_min_retained_fraction: float = 0.4
    campaign_max_chars: int = 12_000
    campaign_max_string_chars: int = 500
    campaign_max_list_items: int = 40

    def __post_init__(self) -> None:
        positive = (
            "summary_max_chars",
            "current_state_max_chars",
            "last_error_max_chars",
            "item_max_chars",
            "max_open_subgoals",
            "max_discovered_facts",
            "max_failed_actions",
            "max_strategy_notes",
            "campaign_max_chars",
            "campaign_max_string_chars",
            "campaign_max_list_items",
        )
        for name in positive:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        empty_summary_chars = _session_summary_chars(SessionSummary())
        if self.summary_max_chars < empty_summary_chars:
            raise ValueError(
                f"summary_max_chars must be >= {empty_summary_chars} to fit the empty summary schema"
            )
        if self.repair_min_previous_items < 0:
            raise ValueError("repair_min_previous_items must be >= 0")
        if not 0.0 <= self.repair_min_retained_fraction <= 1.0:
            raise ValueError("repair_min_retained_fraction must be between 0 and 1")

    def campaign_limits(self) -> MemoryDocumentLimits:
        return MemoryDocumentLimits(
            max_document_chars=self.campaign_max_chars,
            max_string_chars=self.campaign_max_string_chars,
            max_list_items=self.campaign_max_list_items,
        )


def _bound_session_summary(
        summary: SessionSummary,
        limits: LegacyMemoryLimits,
) -> tuple[SessionSummary, dict[str, Any]]:
    stats = {
        "clipped_strings": 0,
        "deduped_items": 0,
        "count_trimmed_items": 0,
        "total_trimmed_items": 0,
    }
    current_state = _bound_legacy_string(summary.current_state, limits.current_state_max_chars, stats)
    last_error = _bound_legacy_string(summary.last_error, limits.last_error_max_chars, stats)
    list_limits = {
        "open_subgoals": limits.max_open_subgoals,
        "discovered_facts": limits.max_discovered_facts,
        "failed_actions": limits.max_failed_actions,
        "strategy_notes": limits.max_strategy_notes,
    }
    values: dict[str, list[str]] = {}
    for key, max_items in list_limits.items():
        normalized: list[str] = []
        seen: set[str] = set()
        for raw in getattr(summary, key):
            item = _bound_legacy_string(raw, limits.item_max_chars, stats)
            if not item:
                continue
            # Only exact repeats are safe to collapse. Case and punctuation
            # can distinguish proper names, room labels, and command tokens.
            identity = item
            if identity in seen:
                stats["deduped_items"] += 1
                continue
            seen.add(identity)
            normalized.append(item)
        if len(normalized) > max_items:
            stats["count_trimmed_items"] += len(normalized) - max_items
            normalized = normalized[:max_items]
        values[key] = normalized

    def build() -> SessionSummary:
        return SessionSummary(
            current_state=current_state,
            last_error=last_error,
            open_subgoals=tuple(values["open_subgoals"]),
            discovered_facts=tuple(values["discovered_facts"]),
            failed_actions=tuple(values["failed_actions"]),
            strategy_notes=tuple(values["strategy_notes"]),
        )

    bounded = build()
    # Lists are model-ordered by future value. Remove their tails in a stable
    # low-value-first section order until the whole working set fits.
    prune_order = ("failed_actions", "strategy_notes", "discovered_facts", "open_subgoals")
    while _session_summary_chars(bounded) > limits.summary_max_chars:
        removed = False
        for key in prune_order:
            if values[key]:
                values[key].pop()
                stats["total_trimmed_items"] += 1
                removed = True
                bounded = build()
                break
        if removed:
            continue
        excess = _session_summary_chars(bounded) - limits.summary_max_chars
        if last_error:
            last_error = last_error[: max(0, len(last_error) - max(1, excess))]
        elif current_state:
            current_state = current_state[: max(0, len(current_state) - max(1, excess))]
        else:
            break
        stats["clipped_strings"] += 1
        bounded = build()

    before_counts = _session_summary_counts(summary)
    after_counts = _session_summary_counts(bounded)
    return bounded, {
        "changed": bounded != summary,
        "before_chars": _session_summary_chars(summary),
        "after_chars": _session_summary_chars(bounded),
        "before_counts": before_counts,
        "after_counts": after_counts,
        **stats,
    }


def _bound_legacy_string(value: str, limit: int, stats: dict[str, int]) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    stats["clipped_strings"] += 1
    marker = "…"
    return value[: max(0, limit - len(marker))] + marker


def _session_summary_chars(summary: SessionSummary) -> int:
    return len(json.dumps(summary.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True))


def _session_summary_counts(summary: SessionSummary) -> dict[str, int]:
    return {
        "open_subgoals": len(summary.open_subgoals),
        "discovered_facts": len(summary.discovered_facts),
        "failed_actions": len(summary.failed_actions),
        "strategy_notes": len(summary.strategy_notes),
    }


@dataclass(frozen=True)
class _LegacyRepairNeed:
    kind: Literal["bounds", "retention"]
    reason: str


def _legacy_compaction_repair_need(
        previous: SessionSummary,
        proposed: SessionSummary,
        limits: LegacyMemoryLimits,
        *,
        proposed_chars: int | None = None,
        bounds_violated: bool = False,
) -> _LegacyRepairNeed | None:
    proposed_chars = _session_summary_chars(proposed) if proposed_chars is None else proposed_chars
    previous_items = sum(_session_summary_counts(previous).values())
    proposed_items = sum(_session_summary_counts(proposed).values())
    if (
        previous_items >= limits.repair_min_previous_items
        and proposed_items < previous_items * limits.repair_min_retained_fraction
    ):
        return _LegacyRepairNeed(
            kind="retention",
            reason=(
                f"draft retained only {proposed_items} of {previous_items} prior list items; "
                f"minimum expected fraction is {limits.repair_min_retained_fraction:g}"
            ),
        )
    if proposed_chars > limits.summary_max_chars:
        return _LegacyRepairNeed(
            kind="bounds",
            reason=f"draft has {proposed_chars} characters; limit is {limits.summary_max_chars}",
        )
    if bounds_violated:
        return _LegacyRepairNeed(
            kind="bounds",
            reason="draft exceeded one or more configured field or item-count limits",
        )
    return None


def _summary_bounds_violated(bounds: dict[str, Any]) -> bool:
    return any(
        int(bounds.get(key, 0)) > 0
        for key in ("clipped_strings", "count_trimmed_items", "total_trimmed_items")
    )


@dataclass(frozen=True)
class ActivityProfile:
    name: str
    objective: str
    action_policy: ActionPolicy = field(default_factory=ActionPolicy)
    system_guidance: str = ""
    observe_timeout: float = 10.0
    stable_ms: int = 300
    byte_quiet_ms: int = 0
    prompt_fast_path: bool = False
    poll_interval: float = 0.05
    recent_steps_to_keep: int = 4
    screen_tail_chars: int = 800
    compact_every_steps: int = 20
    compact_recent_chars: int = 12_000
    invalid_json_retries: int = 1
    model_error_retries: int = 1
    legacy_memory_limits: LegacyMemoryLimits = field(default_factory=LegacyMemoryLimits)
    include_model_responses_in_context: bool = False
    prompt_mode: PromptMode = "stateless_full"
    prompt_layout: PromptLayout = "timeline_first"
    input_modality_profile: InputModalityProfile = field(default_factory=InputModalityProfile)
    prompt_modules: tuple[PromptModule, ...] = field(default=GENERIC_TERMINAL_MODULES, repr=False, compare=False)
    stable_prompt_module_levels: tuple[AssistanceLevel, ...] = DEFAULT_STABLE_LEVELS
    tactical_prompt_module_levels: tuple[AssistanceLevel, ...] = DEFAULT_TACTICAL_LEVELS
    completion_check: Callable[[Observation], bool] | None = field(default=None, repr=False, compare=False)

    def should_exit(self, observation: Observation, action: Action | None, budget: ActivityBudget) -> bool:
        del budget
        if action is not None and action.action == "hangup":
            return True
        return self.completion_check is not None and self.completion_check(observation)


@dataclass
class StepRecord:
    step: int
    observation: dict[str, Any]
    prompt: dict[str, str]
    action: dict[str, Any] | None
    validation: dict[str, Any]
    execution: dict[str, Any]
    budget: dict[str, Any]
    prompt_modules_schema_version: int = PROMPT_MODULES_SCHEMA_VERSION
    prompt_modules: list[dict[str, str]] = field(default_factory=list)
    active_profile: str = ""
    run_objective: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "observation": self.observation,
            "prompt": self.prompt,
            "action": self.action,
            "validation": self.validation,
            "execution": self.execution,
            "budget": self.budget,
            "prompt_modules_schema_version": self.prompt_modules_schema_version,
            "prompt_modules": self.prompt_modules,
            "active_profile": self.active_profile,
            "run_objective": self.run_objective,
            "events": self.events,
            "timestamp": self.timestamp,
        }


@dataclass
class ActivityResult:
    activity: str
    agent_id: str
    steps: list[StepRecord]
    session_summary: SessionSummary
    stop_reason: str
    run_objective: str = ""
    evaluation: EvaluationResult = field(default_factory=EvaluationResult)
    decision_ticks: int = 0


@dataclass(frozen=True)
class ActivityRoute:
    name: str
    profile: ActivityProfile
    matches: Callable[[Observation], bool] = field(repr=False, compare=False)
    priority: int = 0
    reason: str = ""


@dataclass
class ActivityRunState:
    agent: TerminalAgent
    model: ModelAdapter
    budget: ActivityBudget
    agent_id: str
    campaign_memory: dict[str, Any]
    session_summary: SessionSummary = field(default_factory=SessionSummary)
    recent_steps: list[StepRecord] = field(default_factory=list)
    # Leading entries of recent_steps already folded into session_summary; they
    # are kept only as prompt-continuity context and are not re-compacted.
    summarized_recent_steps: int = 0
    # A failed compaction keeps its input intact. Delay retries so a provider
    # failure or malformed utility response cannot trigger an expensive call
    # on every subsequent decision.
    compaction_retry_after_step: int = 0
    all_steps: list[StepRecord] = field(default_factory=list)
    stop_reason: str = "budget"
    last_observation: Observation | None = None
    previous_observation: Observation | None = None
    last_action_for_hints: Action | None = None
    active_profile: ActivityProfile | None = None
    active_route_name: str = "default"
    memory_handle: MemoryHandle | None = None
    decision_prompts_sent: dict[str, int] = field(default_factory=dict)
    evaluation_records: list[EvaluationRecord] = field(default_factory=list)
    response_pending: bool = False
    completed: bool = False


@dataclass
class PreparedActivityStep:
    observation: Observation | None = None
    active_profile: ActivityProfile | None = None
    route_events: list[dict[str, Any]] = field(default_factory=list)
    prompt: DecisionPrompt | None = None
    prompt_module_results: list[PromptModuleResult] = field(default_factory=list)
    action: Action | None = None
    validation: dict[str, Any] = field(default_factory=dict)
    terminal_step: StepRecord | None = None


@dataclass(frozen=True)
class _LegacyCompactionResult:
    summary: SessionSummary
    bounds: dict[str, Any]
    repair: dict[str, Any]
    model_response: dict[str, Any]


@dataclass(frozen=True)
class _LegacyCommitResult:
    patch: MemoryPatch
    accepted: bool


class ActivityRunner:
    def __init__(
            self,
            profile: ActivityProfile,
            memory_store: JsonMemoryStore | None = None,
            log_path: Path | str | None = None,
            run_objective: str = "",
            evaluation_profile: EvaluationProfile | None = None,
            evaluation_log_path: Path | str | None = None,
            memory_subsystem: MemorySubsystem | None = None,
            memory_context_id: str | None = None,
    ) -> None:
        self.profile = profile
        self.memory_store = memory_store or JsonMemoryStore()
        self.log_path = Path(log_path) if log_path else None
        self.run_objective = run_objective.strip()
        self.evaluation_profile = evaluation_profile
        self.evaluation_log_path = Path(evaluation_log_path) if evaluation_log_path else None
        # Optional swappable memory subsystem (docs/memory-structured.md). When
        # set, it replaces the legacy compaction + campaign-memory paths.
        self.memory_subsystem = memory_subsystem
        self.memory_context_id = memory_context_id

    def run(self, agent: TerminalAgent, model: ModelAdapter, budget: ActivityBudget | None = None) -> ActivityResult:
        return self._run(agent, model, budget, stop_on_completion=True)

    def start_state(
            self,
            agent: TerminalAgent,
            model: ModelAdapter,
            budget: ActivityBudget | None = None,
    ) -> ActivityRunState:
        agent_id = getattr(agent, "agent_id", "agent")
        memory_handle: MemoryHandle | None = None
        campaign_memory: dict[str, Any] = {}
        if self.memory_subsystem is not None:
            context_id = self.memory_context_id or self.profile.name
            memory_handle = self.memory_subsystem.open_context(agent_id, context_id)
            # Arm identity in the run log, so metric tooling can attribute the
            # run's journal to a subsystem + fingerprint set.
            self._append_log_record(
                {
                    "type": "memory_context",
                    "subsystem": getattr(self.memory_subsystem, "name", ""),
                    "fingerprints": self.memory_subsystem.fingerprints(),
                    "agent_id": agent_id,
                    "context_id": context_id,
                }
            )
        else:
            campaign_memory = self.memory_store.load(agent_id)
        return ActivityRunState(
            agent=agent,
            model=model,
            budget=budget or ActivityBudget(),
            agent_id=agent_id,
            campaign_memory=campaign_memory,
            active_profile=self.profile,
            memory_handle=memory_handle,
        )

    def run_step(
            self,
            state: ActivityRunState,
            profile_selector: Callable[[Observation, ActivityProfile], tuple[ActivityProfile, list[dict[str, Any]]]]
            | None = None,
            stop_on_completion: bool = True,
    ) -> StepRecord | None:
        prepared = self.prepare_step(
            state,
            profile_selector=profile_selector,
            stop_on_completion=stop_on_completion,
        )
        return self.commit_prepared_step(state, prepared, stop_on_completion=stop_on_completion)

    def prepare_step(
            self,
            state: ActivityRunState,
            profile_selector: Callable[[Observation, ActivityProfile], tuple[ActivityProfile, list[dict[str, Any]]]]
            | None = None,
            stop_on_completion: bool = True,
    ) -> PreparedActivityStep | None:
        if state.completed:
            return None
        if not state.budget.remaining():
            state.stop_reason = "budget"
            state.completed = True
            return None

        active_profile = state.active_profile or self.profile
        trigger_action = state.last_action_for_hints if state.response_pending else None
        try:
            observation = state.agent.observe_turn(
                timeout=active_profile.observe_timeout,
                stable_ms=active_profile.stable_ms,
                byte_quiet_ms=active_profile.byte_quiet_ms,
                poll_interval=active_profile.poll_interval,
                prompt_fast_path=active_profile.prompt_fast_path,
            )
        except SessionDisconnected:
            state.stop_reason = "disconnected"
            state.completed = True
            return None
        state.response_pending = False

        route_events: list[dict[str, Any]] = []
        if profile_selector is not None:
            selected_profile, route_events = profile_selector(observation, active_profile)
            active_profile = selected_profile
        state.active_profile = active_profile
        state.last_observation = observation
        self._evaluate_observation(
            state,
            observation,
            source="agent_action" if trigger_action is not None else "observation",
            visible_to_model=True,
            action=trigger_action,
        )

        if stop_on_completion and active_profile.should_exit(observation, None, state.budget):
            state.stop_reason = "profile_complete"
            state.completed = True
            return PreparedActivityStep(
                terminal_step=self._record_terminal_step(
                    state,
                    observation=observation,
                    stop_reason=state.stop_reason,
                    active_profile=active_profile,
                    events=route_events,
                )
            )

        if state.memory_handle is not None:
            reconcile_outcome = state.memory_handle.maybe_reconcile(state.model)
            if reconcile_outcome is not None:
                route_events.append({"type": "memory_reconcile", **reconcile_outcome.to_dict()})
        unsummarized_steps = (
            [] if state.memory_handle is not None else (state.recent_steps[state.summarized_recent_steps:])
        )
        if unsummarized_steps and self._should_compact(
            active_profile,
            state.all_steps,
            unsummarized_steps,
            retry_after_step=state.compaction_retry_after_step,
        ):
            try:
                compaction = self._compact(
                    active_profile,
                    state.model,
                    state.session_summary,
                    unsummarized_steps,
                    observation,
                )
            except ModelError as exc:
                retry_interval = active_profile.compact_every_steps or active_profile.recent_steps_to_keep
                state.compaction_retry_after_step = len(state.all_steps) + max(1, retry_interval)
                route_events.append(
                    self._compaction_event(
                        state.model,
                        unsummarized_steps,
                        error=exc,
                    )
                )
                self._write_legacy_memory_record(
                    state.agent_id,
                    mutation_record(
                        op="legacy_replace_summary",
                        origin="model",
                        accepted=False,
                        batch="reconcile",
                        section="summary",
                        reason=str(exc),
                        source_steps=tuple(step.step for step in unsummarized_steps),
                    ),
                )
            else:
                state.session_summary = compaction.summary
                state.recent_steps = _recent_window(state.recent_steps, active_profile.recent_steps_to_keep)
                state.summarized_recent_steps = len(state.recent_steps)
                state.compaction_retry_after_step = 0
                route_events.append(
                    self._compaction_event(
                        state.model,
                        unsummarized_steps,
                        summary=compaction.summary,
                        bounds=compaction.bounds,
                        repair=compaction.repair,
                        model_response=compaction.model_response,
                    )
                )
                self._write_legacy_memory_record(
                    state.agent_id,
                    mutation_record(
                        op="legacy_replace_summary",
                        origin="model",
                        accepted=True,
                        batch="reconcile",
                        section="summary",
                        source_steps=tuple(step.step for step in unsummarized_steps),
                        fields={
                            "summary": compaction.summary.to_dict(),
                            "bounds": compaction.bounds,
                            "repair": compaction.repair,
                        },
                    ),
                )

        hints = ObservationHints.from_observation(
            observation=observation,
            previous_observation=state.previous_observation,
            last_action=state.last_action_for_hints,
            modality_profile=active_profile.input_modality_profile,
        )
        # Decision prompts see a bounded window; the full recent_steps list is
        # retained for compaction. The final memory commit applies its own
        # character bound in case compaction remains unavailable.
        prompt_steps = _recent_window(state.recent_steps, active_profile.recent_steps_to_keep)
        prompt_module_results = self._prompt_module_results(
            active_profile,
            agent_id=state.agent_id,
            observation=observation,
            hints=hints,
            campaign_memory=state.campaign_memory,
            session_summary=state.session_summary,
            recent_steps=prompt_steps,
            budget=state.budget,
        )
        profile_prompt_count = state.decision_prompts_sent.get(active_profile.name, 0)
        prompt_stage = self._prompt_stage(active_profile, profile_prompt_count)
        if active_profile.prompt_mode == "stateful_delta" and prompt_stage == "bootstrap":
            # A provider-side conversation is one global chain, not one chain
            # per routed profile. Once any profile bootstraps, every other
            # profile must bootstrap again before it can safely send deltas.
            state.decision_prompts_sent.clear()
            profile_prompt_count = 0
        memory_text: str | None = None
        if state.memory_handle is not None:
            if prompt_stage == "bootstrap":
                # A bootstrap prompt seeds a fresh context (initial or after
                # state loss) and must not lose pending history that cadence
                # had not folded yet: drain the whole backlog in forced
                # catch-up passes (free when nothing is pending), then render
                # the larger bootstrap view. On failure, whatever stayed
                # pending is rendered as raw evidence by render_bootstrap.
                while True:
                    catchup_outcome = state.memory_handle.maybe_reconcile(state.model, force=True)
                    if catchup_outcome is None:
                        break
                    route_events.append({"type": "memory_reconcile", **catchup_outcome.to_dict()})
                    if catchup_outcome.status == "failed":
                        break
                memory_text = state.memory_handle.render_bootstrap()
            else:
                memory_text = state.memory_handle.render_context()
        prompt = self._build_decision_prompt(
            active_profile,
            agent_id=state.agent_id,
            campaign_memory=state.campaign_memory,
            session_summary=state.session_summary,
            recent_steps=prompt_steps,
            budget=state.budget,
            prompt_module_results=prompt_module_results,
            prompt_stage=prompt_stage,
            memory_text=memory_text,
        )

        action, validation = self._decide_with_retry(active_profile, state.model, prompt)
        if validation.get("requires_bootstrap") is True:
            # A remote Responses chain can expire or be deleted while the local
            # runner still expects delta prompts. Clear every profile counter so
            # the next decision reconstructs a complete, locally-owned bootstrap.
            state.decision_prompts_sent.clear()
        elif action is not None or validation.get("invalid_responses"):
            # Count only prompts the model actually received; otherwise a
            # stateful_delta bootstrap that never reached the provider would
            # permanently downgrade the run to delta prompts.
            state.decision_prompts_sent[active_profile.name] = profile_prompt_count + 1
        return PreparedActivityStep(
            observation=observation,
            active_profile=active_profile,
            route_events=route_events,
            prompt=prompt,
            prompt_module_results=prompt_module_results,
            action=action,
            validation=validation,
        )

    def commit_prepared_step(
            self,
            state: ActivityRunState,
            prepared: PreparedActivityStep | None,
            stop_on_completion: bool = True,
    ) -> StepRecord | None:
        if prepared is None:
            return None
        if prepared.terminal_step is not None:
            return prepared.terminal_step
        if prepared.observation is None or prepared.active_profile is None or prepared.prompt is None:
            return None

        observation = prepared.observation
        active_profile = prepared.active_profile
        prompt = prepared.prompt
        action = prepared.action
        validation = prepared.validation
        route_events = prepared.route_events
        prompt_module_results = prepared.prompt_module_results
        executed_action = action
        execution: dict[str, Any] = {}
        if action is None:
            state.budget.record_validation_failure()

        state.budget.consume_tick()

        disconnected = False
        if action is not None:
            try:
                execution = self._execution_record(state.agent.act_action(action))
            except (ActionError, UnicodeEncodeError) as exc:
                # A profile without require_encoding can validate text the
                # session encoding cannot represent; fail the step, not the run.
                executed_action = None
                state.budget.record_validation_failure()
                validation = self._execution_error_validation(validation, str(exc))
            except SessionDisconnected as exc:
                # A peer can drop between observing and acting. Record the step and
                # stop the same way a read-side disconnect does so a match
                # scheduler can still reconnect this agent.
                executed_action = None
                disconnected = True
                validation = self._execution_error_validation(validation, str(exc))

        step = StepRecord(
            step=state.budget.decision_ticks,
            observation=observation.as_dict(),
            prompt={
                "system": prompt.system,
                "user": prompt.user,
                "mode": prompt.mode,
                "stage": prompt.stage,
                "layout": active_profile.prompt_layout,
            },
            action=action.to_dict() if action else None,
            validation=validation,
            execution=execution,
            budget=state.budget.to_dict(),
            prompt_modules=prompt_module_trace(prompt_module_results),
            active_profile=active_profile.name,
            run_objective=self.run_objective,
            events=route_events,
        )
        state.all_steps.append(step)
        state.recent_steps.append(step)
        self._write_step(step)
        self._observe_memory_event(state, step)
        state.previous_observation = observation
        state.last_action_for_hints = executed_action
        state.response_pending = bool(executed_action is not None and executed_action.action != "hangup")

        if disconnected:
            state.stop_reason = "disconnected"
            state.completed = True
        elif state.budget.too_many_validation_failures():
            state.stop_reason = "validation_failures"
            state.completed = True
        elif executed_action and executed_action.action == "hangup":
            state.stop_reason = "hangup"
            state.completed = True
        elif stop_on_completion and active_profile.should_exit(observation, None, state.budget):
            state.stop_reason = "profile_complete"
            state.completed = True
        elif not state.budget.remaining():
            state.stop_reason = "budget"
            state.completed = True
        return step

    def finish_state(self, state: ActivityRunState) -> ActivityResult:
        self._drain_final_observation(state)
        self._run_final_evaluation_probe(state)

        if state.memory_handle is not None:
            try:
                if self._has_decision_steps(state.all_steps):
                    commit_outcome = state.memory_handle.commit(state.model)
                    self._append_log_record({"type": "memory_commit", **commit_outcome.to_dict()})
            finally:
                state.memory_handle.close()
                state.memory_handle = None
        elif self._has_decision_steps(state.all_steps) and state.last_observation is not None:
            active_profile = state.active_profile or self.profile
            commit = self._commit_memory(
                active_profile,
                state.model,
                state.campaign_memory,
                state.session_summary,
                state.recent_steps,
                state.last_observation,
                state.agent_id,
            )
            if commit.accepted:
                _saved, bound_report = self.memory_store.save_patch_with_report(
                    state.agent_id,
                    commit.patch,
                    limits=active_profile.legacy_memory_limits.campaign_limits(),
                )
                if bound_report["document"]["changed"]:
                    self._write_legacy_memory_record(
                        state.agent_id,
                        mutation_record(
                            op="legacy_bound_campaign",
                            origin="system",
                            accepted=True,
                            batch="commit",
                            section="campaign",
                            reason="configured durable-memory bounds",
                            fields={"bounds": bound_report["document"]},
                        ),
                    )
            # A failed commit leaves the old document byte-for-byte alone. A
            # successful explicit `{}` still passes through the bounded store
            # path, allowing old pre-limit memory to converge safely.

        active_profile = state.active_profile or self.profile
        return ActivityResult(
            activity=active_profile.name,
            agent_id=state.agent_id,
            steps=state.all_steps,
            session_summary=state.session_summary,
            stop_reason=state.stop_reason,
            run_objective=self.run_objective,
            evaluation=EvaluationResult(records=list(state.evaluation_records)),
            decision_ticks=state.budget.decision_ticks,
        )

    def _run(
            self,
            agent: TerminalAgent,
            model: ModelAdapter,
            budget: ActivityBudget | None = None,
            profile_selector: Callable[[Observation, ActivityProfile], tuple[ActivityProfile, list[dict[str, Any]]]]
            | None = None,
            stop_on_completion: bool = True,
    ) -> ActivityResult:
        state = self.start_state(agent, model, budget)
        try:
            while not state.completed and state.budget.remaining():
                self.run_step(
                    state,
                    profile_selector=profile_selector,
                    stop_on_completion=stop_on_completion,
                )
            if not state.completed and not state.budget.remaining():
                state.stop_reason = "budget"
                state.completed = True
            return self.finish_state(state)
        except BaseException:
            # An unexpected model/agent/prompt failure must not strand the
            # memory-context lock behind a live traceback reference.
            if state.memory_handle is not None:
                try:
                    state.memory_handle.close()
                finally:
                    state.memory_handle = None
            raise

    def _build_decision_prompt(
            self,
            profile: ActivityProfile,
            agent_id: str,
            campaign_memory: dict[str, Any],
            session_summary: SessionSummary,
            recent_steps: list[StepRecord],
            budget: ActivityBudget,
            prompt_module_results: list[PromptModuleResult],
            prompt_stage: PromptStage,
            memory_text: str | None = None,
    ) -> DecisionPrompt:
        if profile.prompt_mode == "stateful_delta" and prompt_stage == "delta":
            return self._build_stateful_delta_prompt(
                profile,
                agent_id=agent_id,
                session_summary=session_summary,
                recent_steps=recent_steps,
                budget=budget,
                prompt_module_results=prompt_module_results,
                memory_text=memory_text,
            )
        return self._build_stateless_full_prompt(
            profile,
            agent_id=agent_id,
            campaign_memory=campaign_memory,
            session_summary=session_summary,
            recent_steps=recent_steps,
            budget=budget,
            prompt_module_results=prompt_module_results,
            prompt_stage=prompt_stage,
            memory_text=memory_text,
        )

    def _memory_prompt_sections(
            self,
            campaign_memory: dict[str, Any],
            session_summary: SessionSummary,
            memory_text: str | None,
    ) -> list[str]:
        # A memory subsystem owns the whole memory section of the prompt;
        # otherwise the legacy campaign-memory + session-summary pair renders.
        if memory_text is not None:
            return [f"Memory:\n{memory_text}"]
        return [
            f"Campaign memory: {json.dumps(campaign_memory, indent=2, sort_keys=True)}",
            f"Session summary: {self._summary_text(session_summary)}",
        ]

    def _build_stateless_full_prompt(
            self,
            profile: ActivityProfile,
            agent_id: str,
            campaign_memory: dict[str, Any],
            session_summary: SessionSummary,
            recent_steps: list[StepRecord],
            budget: ActivityBudget,
            prompt_module_results: list[PromptModuleResult],
            prompt_stage: PromptStage,
            memory_text: str | None = None,
    ) -> DecisionPrompt:
        system = self._build_full_system_prompt(profile, prompt_stage)
        if profile.prompt_layout == "cache_friendly":
            user = self._build_cache_friendly_user_prompt(
                profile,
                agent_id=agent_id,
                campaign_memory=campaign_memory,
                session_summary=session_summary,
                recent_steps=recent_steps,
                budget=budget,
                prompt_module_results=prompt_module_results,
                memory_text=memory_text,
            )
        else:
            user = self._build_timeline_first_user_prompt(
                profile,
                agent_id=agent_id,
                campaign_memory=campaign_memory,
                session_summary=session_summary,
                recent_steps=recent_steps,
                budget=budget,
                prompt_module_results=prompt_module_results,
                memory_text=memory_text,
            )
        return DecisionPrompt(system=system, user=user, mode=profile.prompt_mode, stage=prompt_stage)

    def _build_full_system_prompt(self, profile: ActivityProfile, prompt_stage: PromptStage) -> str:
        system_parts = [
            "You are controlling an interactive terminal session.",
            "You may make mistakes and recover from them.",
            "Return only a JSON action object.",
            render_action_schema(profile.action_policy),
        ]
        if profile.prompt_mode == "stateful_delta" and prompt_stage == "bootstrap":
            system_parts.append(
                "This is the stateful session bootstrap. Future prompts may omit stable instructions, campaign "
                "memory, and full recent-step history; keep this context active across resumed calls."
            )
        if profile.system_guidance:
            system_parts.append(f"Activity-specific guidance:\n{profile.system_guidance}")
        return "\n".join(system_parts)

    def _build_timeline_first_user_prompt(
            self,
            profile: ActivityProfile,
            agent_id: str,
            campaign_memory: dict[str, Any],
            session_summary: SessionSummary,
            recent_steps: list[StepRecord],
            budget: ActivityBudget,
            prompt_module_results: list[PromptModuleResult],
            memory_text: str | None = None,
    ) -> str:
        module_text = render_prompt_modules(prompt_module_results)
        return "\n\n".join(
            self._objective_prompt_lines(profile)
            + [
                f"Agent: {agent_id}",
                f"Activity: {profile.name}",
                f"Budget: {json.dumps(budget.to_dict(), sort_keys=True)}",
            ]
            + self._memory_prompt_sections(campaign_memory, session_summary, memory_text)
            + [
                f"Recent steps:\n{self._recent_steps_text(profile, recent_steps)}",
                "---",
                f"Current step: {budget.decision_ticks + 1}",
                module_text,
                "---",
            ]
        )

    def _build_cache_friendly_user_prompt(
            self,
            profile: ActivityProfile,
            agent_id: str,
            campaign_memory: dict[str, Any],
            session_summary: SessionSummary,
            recent_steps: list[StepRecord],
            budget: ActivityBudget,
            prompt_module_results: list[PromptModuleResult],
            memory_text: str | None = None,
    ) -> str:
        stable_module_text = render_prompt_modules(prompt_module_results, levels=profile.stable_prompt_module_levels)
        tactical_module_text = render_prompt_modules(
            prompt_module_results,
            levels=profile.tactical_prompt_module_levels,
        )
        sections = (
            self._objective_prompt_lines(profile)
            + [
                f"Agent: {agent_id}",
                f"Activity: {profile.name}",
                stable_module_text,
            ]
            + self._memory_prompt_sections(campaign_memory, session_summary, memory_text)
            + [
                f"Recent steps:\n{self._recent_steps_text(profile, recent_steps)}",
                "---",
                f"Current step: {budget.decision_ticks + 1}",
                f"Budget: {json.dumps(budget.to_dict(), sort_keys=True)}",
                tactical_module_text,
                "---",
            ]
        )
        return "\n\n".join(section for section in sections if section)

    def _build_stateful_delta_prompt(
            self,
            profile: ActivityProfile,
            agent_id: str,
            session_summary: SessionSummary,
            recent_steps: list[StepRecord],
            budget: ActivityBudget,
            prompt_module_results: list[PromptModuleResult],
            memory_text: str | None = None,
    ) -> DecisionPrompt:
        system = "\n".join(
            [
                "Continue the existing terminal-control session from the bootstrap prompt.",
                "Use the action schema, run objective, active profile objective, stable guidance, and campaign "
                "memory already given.",
                "Return only one JSON action object.",
            ]
        )
        module_text = render_prompt_modules(prompt_module_results)
        user = "\n\n".join(
            self._objective_prompt_lines(profile, reminder=True)
            + [
                f"Agent: {agent_id}",
                f"Activity: {profile.name}",
                f"Budget: {json.dumps(budget.to_dict(), sort_keys=True)}",
                (
                    f"Memory update:\n{memory_text}"
                    if memory_text is not None
                    else f"Session summary update: {self._summary_text(session_summary)}"
                ),
                f"Previous step:\n{self._previous_step_delta_text(profile, recent_steps)}",
                "---",
                f"Current step: {budget.decision_ticks + 1}",
                module_text,
                "---",
                "Return exactly one JSON action.",
            ]
        )
        return DecisionPrompt(system=system, user=user, mode=profile.prompt_mode, stage="delta")

    def _objective_prompt_lines(self, profile: ActivityProfile | None = None, *, reminder: bool = False) -> list[str]:
        profile = profile or self.profile
        suffix = " reminder" if reminder else ""
        lines: list[str] = []
        if self.run_objective:
            lines.append(f"Run objective{suffix}: {self.run_objective}")
        lines.append(f"Profile objective{suffix}: {profile.objective}")
        return lines

    def _prompt_stage(self, profile: ActivityProfile, decision_prompts_sent: int) -> PromptStage:
        if profile.prompt_mode == "stateful_delta":
            return "bootstrap" if decision_prompts_sent == 0 else "delta"
        return "full"

    def _prompt_module_results(
            self,
            profile: ActivityProfile,
            agent_id: str,
            observation: Observation,
            hints: ObservationHints,
            campaign_memory: dict[str, Any],
            session_summary: SessionSummary,
            recent_steps: list[StepRecord],
            budget: ActivityBudget,
    ) -> list[PromptModuleResult]:
        context = PromptRenderContext(
            agent_id=agent_id,
            activity_name=profile.name,
            objective=profile.objective,
            observation=observation,
            hints=hints,
            recent_steps=tuple(recent_steps),
            campaign_memory=campaign_memory,
            session_summary=session_summary,
            budget=budget,
            run_objective=self.run_objective,
        )
        return collect_prompt_module_results(profile.prompt_modules, context)

    def _decide_with_retry(
            self,
            profile: ActivityProfile,
            model: ModelAdapter,
            prompt: DecisionPrompt,
    ) -> tuple[Action | None, dict[str, Any]]:
        invalid_responses: list[dict[str, str]] = []
        model_errors: list[dict[str, Any]] = []

        action, first_error, attempt_errors = self._try_model_decision(profile, model, prompt, "initial")
        model_errors.extend(attempt_errors)
        if action is not None:
            return action, self._accepted_validation(model, model_errors=model_errors)
        if first_error is None:
            return None, self._model_error_validation(model_errors)
        invalid_responses.append(self._invalid_response_record(model, "initial", first_error))

        retry_prompt = prompt
        for retry in range(1, profile.invalid_json_retries + 1):
            retry_prompt = self._build_retry_prompt(prompt, first_error, retry)
            action, retry_error, attempt_errors = self._try_model_decision(
                profile,
                model,
                retry_prompt,
                f"repair-{retry}",
            )
            model_errors.extend(attempt_errors)
            if action is not None:
                return action, self._accepted_validation(
                    model,
                    notes=[f"repaired_after_error: {first_error}"],
                    invalid_responses=invalid_responses,
                    model_errors=model_errors,
                )
            if retry_error is None:
                return None, self._model_error_validation(model_errors, invalid_responses=invalid_responses)
            first_error = retry_error
            invalid_responses.append(self._invalid_response_record(model, f"repair-{retry}", first_error))

        validation: dict[str, Any] = {
            "accepted": False,
            "notes": [first_error],
            "invalid_responses": invalid_responses,
        }
        if model_errors:
            validation["model_errors"] = model_errors
        return None, validation

    def _try_model_decision(
            self,
            profile: ActivityProfile,
            model: ModelAdapter,
            prompt: DecisionPrompt,
            stage: str,
    ) -> tuple[Action | None, str | None, list[dict[str, Any]]]:
        model_errors: list[dict[str, Any]] = []
        for provider_attempt in range(0, profile.model_error_retries + 1):
            try:
                return model.decide(prompt, profile.action_policy), None, model_errors
            except ModelStateError as exc:
                model_errors.append(self._model_error_record(exc, stage, provider_attempt))
                # Retrying the same delta cannot restore missing conversation
                # state. Let the outer runner issue a full bootstrap next tick.
                return None, None, model_errors
            except ModelError as exc:
                model_errors.append(self._model_error_record(exc, stage, provider_attempt))
            except ActionError as exc:
                return None, str(exc), model_errors
        return None, None, model_errors

    def _accepted_validation(
            self,
            model: ModelAdapter,
            notes: list[str] | None = None,
            invalid_responses: list[dict[str, str]] | None = None,
            model_errors: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        validation: dict[str, Any] = {
            "accepted": True,
            "notes": list(notes or []),
            "model_response": self._model_response_record(model),
        }
        request_metadata = getattr(model, "last_request_metadata", None)
        if isinstance(request_metadata, dict) and request_metadata.get("output_token_retries", 0):
            validation["notes"].append("recovered_after_output_truncation")
        if invalid_responses:
            validation["invalid_responses"] = invalid_responses
        if model_errors:
            validation["model_errors"] = model_errors
            validation["notes"].append("recovered_after_model_error")
        return validation

    def _model_error_validation(
            self,
            model_errors: list[dict[str, Any]],
            invalid_responses: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        last_error = model_errors[-1]["message"] if model_errors else "model provider failed"
        validation: dict[str, Any] = {
            "accepted": False,
            "notes": [f"model_error: {last_error}"],
            "model_errors": model_errors,
        }
        if any(error.get("requires_bootstrap") is True for error in model_errors):
            validation["requires_bootstrap"] = True
        if invalid_responses:
            validation["invalid_responses"] = invalid_responses
        return validation

    def _execution_error_validation(self, validation: dict[str, Any], error: str) -> dict[str, Any]:
        notes = list(validation.get("notes", []))
        notes.append(f"action_error: {error}")
        updated = dict(validation)
        updated["accepted"] = False
        updated["notes"] = notes
        return updated

    def _drain_final_observation(self, state: ActivityRunState) -> None:
        """Capture the result of the last admitted agent action before finalization."""

        if not state.response_pending or state.stop_reason in {"disconnected", "hangup"}:
            return

        active_profile = state.active_profile or self.profile
        trigger_action = state.last_action_for_hints
        try:
            observation = state.agent.observe_turn(
                timeout=active_profile.observe_timeout,
                stable_ms=active_profile.stable_ms,
                byte_quiet_ms=active_profile.byte_quiet_ms,
                poll_interval=active_profile.poll_interval,
                prompt_fast_path=active_profile.prompt_fast_path,
            )
        except SessionDisconnected:
            state.response_pending = False
            return

        state.response_pending = False
        state.last_observation = observation
        self._evaluate_observation(
            state,
            observation,
            source="agent_action" if trigger_action is not None else "observation",
            visible_to_model=False,
            final=True,
            action=trigger_action,
        )
        self._record_terminal_step(
            state,
            observation=observation,
            stop_reason=state.stop_reason,
            active_profile=active_profile,
        )

    def _run_final_evaluation_probe(self, state: ActivityRunState) -> None:
        """Run one evaluator-owned final query without changing agent state or budget."""

        evaluation_profile = self.evaluation_profile
        if evaluation_profile is None or evaluation_profile.final_probe is None:
            return

        probe = evaluation_profile.final_probe
        base_observation = state.last_observation
        if base_observation is None:
            return
        if state.stop_reason in {"disconnected", "hangup"}:
            self._record_final_probe(
                state,
                status="error",
                error=f"terminal unavailable after {state.stop_reason}",
            )
            return

        try:
            ready = probe.is_ready(base_observation)
        except Exception as exc:
            self._record_final_probe(state, status="error", error=f"probe readiness failed: {exc}")
            return
        if not ready:
            self._record_final_probe(
                state,
                status="no_match",
                error="final probe is not safe at the current terminal prompt",
            )
            return

        active_profile = state.active_profile or self.profile
        try:
            action = active_profile.action_policy.validate(probe.action)
            execution = self._execution_record(state.agent.act_action(action))
            observation = state.agent.observe_turn(
                timeout=probe.observe_timeout if probe.observe_timeout is not None else active_profile.observe_timeout,
                stable_ms=probe.stable_ms if probe.stable_ms is not None else active_profile.stable_ms,
                byte_quiet_ms=(
                    probe.byte_quiet_ms if probe.byte_quiet_ms is not None else active_profile.byte_quiet_ms
                ),
                poll_interval=(
                    probe.poll_interval if probe.poll_interval is not None else active_profile.poll_interval
                ),
                prompt_fast_path=(
                    probe.prompt_fast_path if probe.prompt_fast_path is not None else active_profile.prompt_fast_path
                ),
            )
        except (ActionError, UnicodeEncodeError, SessionDisconnected) as exc:
            self._record_final_probe(state, status="error", error=str(exc))
            return

        try:
            metrics = dict(evaluation_profile.extractor(observation) or {})
        except Exception as exc:
            self._record_final_probe(
                state,
                status="error",
                observation=observation,
                execution=execution,
                error=f"metric extraction failed: {exc}",
            )
            return

        self._record_final_probe(
            state,
            status="ok" if metrics else "no_match",
            metrics=metrics,
            observation=observation,
            execution=execution,
            error="" if metrics else "final probe response did not contain recognized metrics",
        )

    def _evaluate_observation(
            self,
            state: ActivityRunState,
            observation: Observation,
            source: EvaluationSource,
            visible_to_model: bool,
            final: bool = False,
            action: Action | None = None,
    ) -> None:
        evaluation_profile = self.evaluation_profile
        if evaluation_profile is None:
            return

        try:
            metrics = dict(evaluation_profile.extractor(observation) or {})
        except Exception as exc:
            self._record_evaluation(
                state,
                EvaluationRecord(
                    agent_id=state.agent_id,
                    evaluator=evaluation_profile.name,
                    source=source,
                    status="error",
                    decision_tick=state.budget.decision_ticks,
                    final=final,
                    visible_to_model=visible_to_model,
                    observation=self._evaluation_observation(observation),
                    action=action.to_dict() if action is not None else None,
                    error=f"metric extraction failed: {exc}",
                ),
            )
            return
        if not metrics:
            return

        self._record_evaluation(
            state,
            EvaluationRecord(
                agent_id=state.agent_id,
                evaluator=evaluation_profile.name,
                source=source,
                status="ok",
                decision_tick=state.budget.decision_ticks,
                metrics=metrics,
                final=final,
                visible_to_model=visible_to_model,
                observation=self._evaluation_observation(observation),
                action=action.to_dict() if action is not None else None,
            ),
        )

    def _record_final_probe(
            self,
            state: ActivityRunState,
            status: Literal["ok", "no_match", "error"],
            metrics: dict[str, Any] | None = None,
            observation: Observation | None = None,
            execution: dict[str, Any] | None = None,
            error: str = "",
    ) -> None:
        evaluation_profile = self.evaluation_profile
        if evaluation_profile is None or evaluation_profile.final_probe is None:
            return
        probe = evaluation_profile.final_probe
        self._record_evaluation(
            state,
            EvaluationRecord(
                agent_id=state.agent_id,
                evaluator=evaluation_profile.name,
                source="final_probe",
                status=status,
                decision_tick=state.budget.decision_ticks,
                metrics=dict(metrics or {}),
                final=True,
                visible_to_model=False,
                observation=self._evaluation_observation(observation) if observation is not None else {},
                action=probe.action.to_dict(),
                execution=dict(execution or {}),
                error=error,
                probe_turn_cost=probe.turn_cost,
            ),
        )

    def _record_evaluation(self, state: ActivityRunState, record: EvaluationRecord) -> None:
        state.evaluation_records.append(record)
        self._write_evaluation(record)

    def _evaluation_observation(self, observation: Observation) -> dict[str, Any]:
        return {
            "model_text": observation.model_text,
            "new_text": observation.new_text,
            "matched_prompt": observation.matched_prompt,
            "ready_reason": observation.ready_reason,
            "profile": observation.profile,
            "transcript_path": str(observation.transcript_path) if observation.transcript_path else None,
            "transcript_byte_start": observation.transcript_byte_start,
            "transcript_byte_end": observation.transcript_byte_end,
            "timestamp": observation.timestamp,
        }

    def _terminal_step_record(
            self,
            step_number: int,
            observation: Observation,
            budget: ActivityBudget,
            stop_reason: str,
            active_profile: ActivityProfile | None = None,
            events: list[dict[str, Any]] | None = None,
    ) -> StepRecord:
        return StepRecord(
            step=step_number,
            observation=observation.as_dict(),
            prompt={},
            action=None,
            validation={
                "accepted": True,
                "terminal": True,
                "stop_reason": stop_reason,
                "notes": ["terminal_observation", "does_not_consume_decision_tick"],
            },
            execution={},
            budget=budget.to_dict(),
            active_profile=(active_profile or self.profile).name,
            run_objective=self.run_objective,
            events=list(events or []),
        )

    def _record_terminal_step(
            self,
            state: ActivityRunState,
            observation: Observation,
            stop_reason: str,
            active_profile: ActivityProfile | None = None,
            events: list[dict[str, Any]] | None = None,
    ) -> StepRecord:
        step = self._terminal_step_record(
            step_number=len(state.all_steps) + 1,
            observation=observation,
            budget=state.budget,
            stop_reason=stop_reason,
            active_profile=active_profile,
            events=events,
        )
        state.all_steps.append(step)
        state.recent_steps.append(step)
        self._write_step(step)
        self._observe_memory_event(state, step)
        return step

    def _observe_memory_event(self, state: ActivityRunState, step: StepRecord) -> None:
        if state.memory_handle is None:
            return
        # A rejected or failed action must not read as an executed one:
        # surface the validation notes so memory does not learn the wrong
        # causal outcome. (Terminal observation records carry no accepted
        # flag and stay warning-free.)
        warnings: tuple[str, ...] = ()
        if step.validation.get("accepted") is False:
            notes = tuple(str(note) for note in step.validation.get("notes", []) if note)
            warnings = notes or ("action was rejected and not executed",)
        state.memory_handle.observe(
            MemoryEvent(
                kind="terminal_step",
                step=step.step,
                observation=str(step.observation.get("model_text", "")),
                action=json.dumps(step.action, sort_keys=True) if step.action else "",
                warnings=warnings,
            )
        )

    def _execution_record(self, result: ActionExecution) -> dict[str, Any]:
        return result.to_dict()

    def _has_decision_steps(self, steps: list[StepRecord]) -> bool:
        return any(not self._is_terminal_step(step) for step in steps)

    def _is_terminal_step(self, step: StepRecord) -> bool:
        return bool(step.validation.get("terminal"))

    def _build_retry_prompt(self, prompt: DecisionPrompt, error: str, attempt: int) -> DecisionPrompt:
        return DecisionPrompt(
            system=prompt.system,
            user="\n\n".join(
                [
                    prompt.user,
                    f"Your previous response could not be used: {error}",
                    f"Repair attempt: {attempt}",
                    "Return exactly one valid JSON action object and no surrounding prose.",
                ]
            ),
            mode=prompt.mode,
            stage=prompt.stage,
        )

    def _compact(
            self,
            profile: ActivityProfile,
            model: ModelAdapter,
            session_summary: SessionSummary,
            recent_steps: list[StepRecord],
            observation: Observation,
    ) -> _LegacyCompactionResult:
        limits = profile.legacy_memory_limits
        prompt = self._legacy_compaction_prompt(
            profile,
            session_summary,
            recent_steps,
            observation,
        )
        first_summary = model.compact(prompt)
        first_response = self._model_response_record(model)
        bounded_summary, bounds = _bound_session_summary(first_summary, limits)
        repair_need = _legacy_compaction_repair_need(
            session_summary,
            bounded_summary,
            limits,
            proposed_chars=_session_summary_chars(first_summary),
            bounds_violated=_summary_bounds_violated(bounds),
        )
        repair: dict[str, Any] = {"attempted": False}
        accepted_response = first_response
        if repair_need is not None:
            repair["attempted"] = True
            repair["reason"] = repair_need.reason
            repair_prompt = CompactionPrompt(
                system=prompt.system,
                user="\n\n".join(
                    [
                        prompt.user,
                        "The first draft needs one bounded repair pass.",
                        f"Repair reason: {repair_need.reason}",
                        f"First bounded draft:\n{self._summary_text(bounded_summary)}",
                        (
                            "Return a corrected replacement. Retain still-useful durable knowledge; "
                            "a current-state reset does not erase learned history. Stay within every listed limit."
                        ),
                    ]
                ),
            )
            try:
                repaired_summary = model.compact(repair_prompt)
            except ModelError as exc:
                if repair_need.kind == "retention":
                    raise ModelError(f"unsafe compaction draft could not be repaired: {exc}") from exc
                repair["status"] = "error"
                repair["error"] = self._model_error_record(exc, "compaction-repair", 0)
            else:
                repaired_bounded, repaired_bounds = _bound_session_summary(repaired_summary, limits)
                remaining_need = _legacy_compaction_repair_need(
                    session_summary,
                    repaired_bounded,
                    limits,
                    proposed_chars=_session_summary_chars(repaired_summary),
                    bounds_violated=_summary_bounds_violated(repaired_bounds),
                )
                if remaining_need is not None and remaining_need.kind == "retention":
                    raise ModelError(
                        f"unsafe compaction draft remained destructive after repair: {remaining_need.reason}"
                    )
                bounded_summary = repaired_bounded
                bounds = repaired_bounds
                accepted_response = self._model_response_record(model)
                repair["status"] = "accepted"
        return _LegacyCompactionResult(
            summary=bounded_summary,
            bounds=bounds,
            repair=repair,
            model_response=accepted_response,
        )

    def _legacy_compaction_prompt(
            self,
            profile: ActivityProfile,
            session_summary: SessionSummary,
            recent_steps: list[StepRecord],
            observation: Observation,
    ) -> CompactionPrompt:
        limits = profile.legacy_memory_limits
        return CompactionPrompt(
            system=(
                "Compact older terminal activity by rewriting it into a selective, bounded JSON working summary. "
                "Return only JSON with keys: current_state, last_error, open_subgoals, "
                "discovered_facts, failed_actions, strategy_notes. Each list field must "
                "be a list of strings. This is a replacement working set, not a transcript. "
                "Order every list from most to least useful for future decisions. Merge duplicates; "
                "remove contradicted, obsolete, or subsumed entries. Treat exact labels and proper names "
                "as distinct subjects. Put changing present conditions in current_state; retain stable "
                "knowledge and useful historical outcomes in discovered_facts. Keep failed actions only "
                "when they prevent a likely repeated mistake. Keep reasoning concise and reserve enough "
                "output for the required JSON; emit the JSON as soon as the memory update is determined."
            ),
            user="\n\n".join(
                self._objective_prompt_lines(profile)
                + [
                    f"Previous summary:\n{self._summary_text(session_summary)}",
                    f"Steps to summarize:\n{self._recent_steps_text(profile, recent_steps)}",
                    f"Current screen:\n{observation.model_text}",
                    (
                        "Hard limits: "
                        f"at most {limits.summary_max_chars} serialized characters; "
                        f"current_state {limits.current_state_max_chars} characters; "
                        f"last_error {limits.last_error_max_chars} characters; "
                        f"each list item {limits.item_max_chars} characters; "
                        f"{limits.max_open_subgoals} open_subgoals; "
                        f"{limits.max_discovered_facts} discovered_facts; "
                        f"{limits.max_failed_actions} failed_actions; "
                        f"{limits.max_strategy_notes} strategy_notes."
                    ),
                ]
            ),
        )

    def _compaction_event(
            self,
            model: ModelAdapter,
            steps: list[StepRecord],
            *,
            summary: SessionSummary | None = None,
            error: ModelError | None = None,
            bounds: dict[str, Any] | None = None,
            repair: dict[str, Any] | None = None,
            model_response: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "type": "model_utility",
            "operation": "compaction",
            "status": "error" if error is not None else "ok",
            "step_count": len(steps),
            "first_step": steps[0].step,
            "last_step": steps[-1].step,
            "model_response": model_response or self._model_response_record(model),
        }
        if summary is not None:
            event["summary"] = summary.to_dict()
        if bounds is not None:
            event["bounds"] = bounds
        if repair is not None:
            event["repair"] = repair
        if error is not None:
            event["error"] = self._model_error_record(error, "compaction", 0)
        return event

    def _commit_memory(
            self,
            profile: ActivityProfile,
            model: ModelAdapter,
            campaign_memory: dict[str, Any],
            session_summary: SessionSummary,
            recent_steps: list[StepRecord],
            observation: Observation,
            agent_id: str,
    ) -> _LegacyCommitResult:
        limits = profile.legacy_memory_limits
        bounded_campaign_memory, _existing_bounds = bound_memory_document(
            campaign_memory,
            limits.campaign_limits(),
        )
        bounded_session_summary, _summary_bounds = _bound_session_summary(session_summary, limits)
        prompt = MemoryCommitPrompt(
            system=(
                "Return only a selective JSON memory patch for durable campaign memory, using keys "
                "durable_facts, strategy_notes, open_tasks, and errors_to_avoid when applicable. "
                "This is not a transcript: include only stable, reusable information or unresolved work. "
                "Treat exact labels as distinct; merge duplicates; omit obsolete current-state details. "
                "Order lists from most to least useful. Each patch list becomes the priority prefix for that "
                "field: re-emit an existing entry when it must rank ahead of a new one; unmentioned entries "
                "persist only in remaining capacity. Keep reasoning concise and reserve enough output for "
                "the required JSON; emit the JSON as soon as the memory update is determined."
            ),
            user="\n\n".join(
                self._objective_prompt_lines(profile)
                + [
                    f"Existing campaign memory:\n{json.dumps(bounded_campaign_memory, indent=2, sort_keys=True)}",
                    f"Session summary:\n{self._summary_text(bounded_session_summary)}",
                    f"Recent steps:\n{self._bounded_recent_steps_text(profile, recent_steps)}",
                    f"Final screen:\n{observation.model_text}",
                    (
                        "Hard limits for the merged campaign document: "
                        f"{limits.campaign_max_chars} serialized characters, "
                        f"{limits.campaign_max_string_chars} characters per string, and "
                        f"{limits.campaign_max_list_items} items per list."
                    ),
                ]
            ),
        )
        last_error: ModelError | None = None
        for _ in range(1 + max(0, profile.model_error_retries)):
            try:
                patch = model.commit_memory(prompt)
            except ModelError as exc:
                last_error = exc
            else:
                bounded_data, bounds = bound_memory_document(patch.data, limits.campaign_limits())
                patch = MemoryPatch(bounded_data)
                self._write_legacy_memory_record(
                    agent_id,
                    mutation_record(
                        op="legacy_merge_patch",
                        origin="model",
                        accepted=True,
                        batch="commit",
                        section="campaign",
                        source_steps=tuple(step.step for step in recent_steps),
                        fields={"patch": patch.data, "bounds": bounds},
                    ),
                )
                return _LegacyCommitResult(patch=patch, accepted=True)
        # A failed commit must not vanish: record it, then keep the run's
        # result-building alive by committing nothing.
        self._write_memory_commit_failure(last_error, model)
        self._write_legacy_memory_record(
            agent_id,
            mutation_record(
                op="legacy_merge_patch",
                origin="model",
                accepted=False,
                batch="commit",
                section="campaign",
                reason=str(last_error) if last_error is not None else "",
                source_steps=tuple(step.step for step in recent_steps),
            ),
        )
        return _LegacyCommitResult(patch=MemoryPatch(), accepted=False)

    def _legacy_memory_journal_path(self, agent_id: str) -> Path:
        return Path(self.memory_store.root) / validate_agent_id(agent_id) / "ops.jsonl"

    def _write_legacy_memory_record(self, agent_id: str, record: dict[str, Any]) -> None:
        # Metric parity with memory subsystems (docs/memory-structured.md): the
        # legacy inline path journals its two pseudo-ops in the common
        # mutation-record schema, beside its campaign.json.
        write_journal_records(self._legacy_memory_journal_path(agent_id), [record])

    def _write_memory_commit_failure(self, error: ModelError | None, model: ModelAdapter) -> None:
        self._append_log_record(
            {
                "type": "memory_commit_failed",
                "error": self._truncate(str(error), 2_000) if error is not None else "",
                "model_response": self._model_response_record(model),
            }
        )

    def _append_log_record(self, record: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        record = {**record, "timestamp": time.time()}
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _summary_text(self, summary: SessionSummary) -> str:
        return "(empty)" if summary.is_empty() else json.dumps(summary.to_dict(), indent=2, sort_keys=True)

    def _invalid_response_record(self, model: ModelAdapter, attempt: str, error: str) -> dict[str, str]:
        record = self._model_response_record(model)
        record.update(
            {
                "attempt": attempt,
                "error": error,
            }
        )
        return record

    def _model_error_record(self, error: ModelError, stage: str, provider_attempt: int) -> dict[str, Any]:
        record = {
            "stage": stage,
            "provider_attempt": provider_attempt,
            "type": error.__class__.__name__,
            "message": self._truncate(str(error), 2_000),
            "command": list(error.command),
            "stdout": self._truncate(error.stdout, 2_000),
            "stderr": self._truncate(error.stderr, 2_000),
        }
        if error.status_code is not None:
            record["status_code"] = error.status_code
        if isinstance(error, ModelOutputTruncated):
            record["operation"] = error.operation
            record["requested_max_tokens"] = error.requested_max_tokens
            record["retry_ceiling"] = error.retry_ceiling
            if error.request_metadata:
                record["request_metadata"] = error.request_metadata
        if isinstance(error, ModelStateError):
            record["requires_bootstrap"] = True
        return record

    def _model_response_record(self, model: ModelAdapter) -> dict[str, Any]:
        raw = getattr(model, "last_response", "")
        parsed = getattr(model, "last_parsed_response", raw)
        record = {
            "response": self._truncate(raw, 2_000),
            "parsed_response": self._truncate(parsed, 2_000),
        }
        reasoning = getattr(model, "last_reasoning", "")
        if reasoning:
            record["reasoning"] = self._truncate(reasoning, 4_000)
        response_id = getattr(model, "last_response_id", None)
        if isinstance(response_id, str) and response_id:
            record["response_id"] = response_id
        usage = getattr(model, "last_usage", None)
        if isinstance(usage, dict):
            record["usage"] = usage
        provider_metadata = getattr(model, "last_provider_metadata", None)
        if isinstance(provider_metadata, dict):
            record["provider_metadata"] = provider_metadata
        request_metadata = getattr(model, "last_request_metadata", None)
        if isinstance(request_metadata, dict):
            record["request_metadata"] = request_metadata
        return record

    def _truncate(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + f"...[truncated {len(text) - limit} chars]"

    def _should_compact(
            self,
            profile: ActivityProfile,
            all_steps: list[StepRecord],
            recent_steps: list[StepRecord],
            retry_after_step: int = 0,
    ) -> bool:
        if len(all_steps) < retry_after_step:
            return False
        every = profile.compact_every_steps
        if every > 0 and len(all_steps) > 0 and len(all_steps) % every == 0:
            return True
        return self._steps_char_count(profile, recent_steps) >= profile.compact_recent_chars

    def _steps_char_count(self, profile: ActivityProfile, steps: list[StepRecord]) -> int:
        total = 0
        for step in steps:
            obs = step.observation
            # Compaction should react to the real volume of observed terminal
            # text even though decision prompts only include a bounded tail.
            for key in ("model_text", "new_text"):
                value = obs.get(key, "")
                if isinstance(value, str):
                    total += len(value)
            action = json.dumps(step.action or {}, sort_keys=True)
            validation = json.dumps(self._validation_for_context(profile, step.validation), sort_keys=True)
            total += len(action) + len(validation)
        return total

    def _recent_steps_text(self, profile: ActivityProfile, steps: list[StepRecord]) -> str:
        if not steps:
            return "(none)"
        return "\n\n".join(self._recent_step_contexts(profile, steps))

    def _bounded_recent_steps_text(self, profile: ActivityProfile, steps: list[StepRecord]) -> str:
        """Render the newest whole steps within the compaction character budget."""

        if not steps:
            return "(none)"
        limit = max(0, profile.compact_recent_chars)
        if limit == 0:
            return ""
        contexts = self._recent_step_contexts(profile, steps)
        full_text = "\n\n".join(contexts)
        if len(full_text) <= limit:
            return full_text

        marker = "[Older unsummarized step context omitted from final memory commit.]\n\n"
        if len(marker) >= limit:
            return marker[:limit]

        selected: list[str] = []
        used = len(marker)
        for context in reversed(contexts):
            separator_length = 2 if selected else 0
            if used + separator_length + len(context) > limit:
                break
            selected.append(context)
            used += separator_length + len(context)
        if not selected:
            return marker + contexts[-1][-(limit - len(marker)) :]
        return marker + "\n\n".join(reversed(selected))

    def _recent_step_contexts(self, profile: ActivityProfile, steps: list[StepRecord]) -> list[str]:
        lines = []
        for index, step in enumerate(steps):
            after_text = "Current terminal observation below."
            if index + 1 < len(steps):
                after_text = self._observation_effect_text(profile, steps[index + 1].observation)
            lines.append(self._step_context_text(profile, step, after_text))
        return lines

    def _previous_step_delta_text(self, profile: ActivityProfile, steps: list[StepRecord]) -> str:
        if not steps:
            return "(none)"
        step = steps[-1]
        return self._step_context_text(profile, step, "Current terminal observation below.")

    def _step_context_text(self, profile: ActivityProfile, step: StepRecord, after_text: str) -> str:
        action = step.action or {"action": "terminal_observation" if self._is_terminal_step(step) else "invalid"}
        validation = self._validation_for_context(profile, step.validation)
        return "\n".join(
            [
                f"Step {step.step}",
                f"Observed before action:\n{self._screen_tail(profile, step.observation.get('model_text', ''))}",
                f"Action chosen:\n{json.dumps(action, sort_keys=True)}",
                f"Validation: {json.dumps(validation, sort_keys=True)}",
                f"Observed after action:\n{after_text}",
            ]
        )

    def _screen_tail(self, profile: ActivityProfile, screen: Any) -> str:
        if not isinstance(screen, str) or profile.screen_tail_chars <= 0:
            return ""
        return screen[-profile.screen_tail_chars:]

    def _observation_effect_text(self, profile: ActivityProfile, observation: dict[str, Any]) -> str:
        new_text = self._screen_tail(profile, observation.get("new_text", ""))
        if new_text:
            return new_text
        return self._screen_tail(profile, observation.get("model_text", ""))

    def _validation_for_context(self, profile: ActivityProfile, validation: dict[str, Any]) -> dict[str, Any]:
        if profile.include_model_responses_in_context:
            return validation

        context: dict[str, Any] = {}
        for key in ("accepted", "notes"):
            if key in validation:
                context[key] = validation[key]

        invalid_responses = validation.get("invalid_responses")
        if isinstance(invalid_responses, list) and invalid_responses:
            context["invalid_responses"] = [
                {key: response[key] for key in ("attempt", "error") if isinstance(response, dict) and key in response}
                for response in invalid_responses
            ]
        return context

    def _write_step(self, step: StepRecord) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(step.to_dict(), sort_keys=True) + "\n")

    def _write_evaluation(self, record: EvaluationRecord) -> None:
        if self.evaluation_log_path is None:
            return
        self.evaluation_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.evaluation_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


class RoutedActivityRunner(ActivityRunner):
    def __init__(
            self,
            name: str,
            default_profile: ActivityProfile,
            routes: tuple[ActivityRoute, ...],
            memory_store: JsonMemoryStore | None = None,
            log_path: Path | str | None = None,
            run_objective: str = "",
            evaluation_profile: EvaluationProfile | None = None,
            evaluation_log_path: Path | str | None = None,
    ) -> None:
        super().__init__(
            default_profile,
            memory_store=memory_store,
            log_path=log_path,
            run_objective=run_objective,
            evaluation_profile=evaluation_profile,
            evaluation_log_path=evaluation_log_path,
        )
        self.name = name
        self.default_profile = default_profile
        self.routes = tuple(sorted(enumerate(routes), key=lambda item: (-item[1].priority, item[0])))

    def run(self, agent: TerminalAgent, model: ModelAdapter, budget: ActivityBudget | None = None) -> ActivityResult:
        result = self._run(agent, model, budget, stop_on_completion=False)
        result.activity = self.name
        return result

    def prepare_step(
            self,
            state: ActivityRunState,
            profile_selector: Callable[[Observation, ActivityProfile], tuple[ActivityProfile, list[dict[str, Any]]]]
            | None = None,
            stop_on_completion: bool = True,
    ) -> PreparedActivityStep | None:
        # Route on every step, including through the external
        # start_state/prepare_step/run_step API used by match schedulers; route
        # tracking lives on the state so one runner can serve concurrent runs.
        if profile_selector is None:

            def profile_selector(observation: Observation, current_profile: ActivityProfile):
                return self._select_profile(state, observation, current_profile)

        return super().prepare_step(state, profile_selector=profile_selector, stop_on_completion=stop_on_completion)

    def _select_profile(
            self,
            state: ActivityRunState,
            observation: Observation,
            current_profile: ActivityProfile,
    ) -> tuple[ActivityProfile, list[dict[str, Any]]]:
        route = self._matched_route(observation)
        route_name = route.name if route is not None else "default"
        selected_profile = route.profile if route is not None else self.default_profile
        if route_name == state.active_route_name and selected_profile.name == current_profile.name:
            return selected_profile, []

        event = {
            "type": "profile_switch",
            "from": current_profile.name,
            "to": selected_profile.name,
            "route": route_name,
        }
        if route is not None and route.reason:
            event["reason"] = route.reason
        elif route is None:
            event["reason"] = "no route matched; using default profile"
        state.active_route_name = route_name
        return selected_profile, [event]

    def _matched_route(self, observation: Observation) -> ActivityRoute | None:
        for _, route in self.routes:
            if route.matches(observation):
                return route
        return None
