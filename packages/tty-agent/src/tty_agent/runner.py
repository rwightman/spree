"""Single-agent activity runner for terminal sessions."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from .agent import ActionExecution, TerminalAgent
from .actions import Action, ActionError, ActionPolicy, render_action_schema
from .hints import InputModalityProfile, ObservationHints
from .memory import JsonMemoryStore
from .models import (
    CompactionPrompt,
    DecisionPrompt,
    MemoryCommitPrompt,
    MemoryPatch,
    ModelAdapter,
    ModelError,
    SessionSummary,
)
from .prompt_modules import (
    GENERIC_TERMINAL_MODULES,
    PROMPT_MODULES_SCHEMA_VERSION,
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
STATIC_PROMPT_MODULE_LEVELS = ("bbs_conventions", "game_interface", "strategic")
TACTICAL_PROMPT_MODULE_LEVELS = ("generic_terminal",)


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
    include_model_responses_in_context: bool = False
    prompt_mode: PromptMode = "stateless_full"
    prompt_layout: PromptLayout = "timeline_first"
    input_modality_profile: InputModalityProfile = field(default_factory=InputModalityProfile)
    prompt_modules: tuple[PromptModule, ...] = field(default=GENERIC_TERMINAL_MODULES, repr=False, compare=False)
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
    all_steps: list[StepRecord] = field(default_factory=list)
    stop_reason: str = "budget"
    last_observation: Observation | None = None
    previous_observation: Observation | None = None
    last_action_for_hints: Action | None = None
    active_profile: ActivityProfile | None = None
    decision_prompts_sent: dict[str, int] = field(default_factory=dict)
    completed: bool = False


class ActivityRunner:
    def __init__(
            self,
            profile: ActivityProfile,
            memory_store: JsonMemoryStore | None = None,
            log_path: Path | str | None = None,
            run_objective: str = "",
    ) -> None:
        self.profile = profile
        self.memory_store = memory_store or JsonMemoryStore()
        self.log_path = Path(log_path) if log_path else None
        self.run_objective = run_objective.strip()

    def run(self, agent: TerminalAgent, model: ModelAdapter, budget: ActivityBudget | None = None) -> ActivityResult:
        return self._run(agent, model, budget, stop_on_completion=True)

    def start_state(
            self,
            agent: TerminalAgent,
            model: ModelAdapter,
            budget: ActivityBudget | None = None,
    ) -> ActivityRunState:
        agent_id = getattr(agent, "agent_id", "agent")
        return ActivityRunState(
            agent=agent,
            model=model,
            budget=budget or ActivityBudget(),
            agent_id=agent_id,
            campaign_memory=self.memory_store.load(agent_id),
            active_profile=self.profile,
        )

    def run_step(
            self,
            state: ActivityRunState,
            profile_selector: Callable[[Observation, ActivityProfile], tuple[ActivityProfile, list[dict[str, Any]]]]
            | None = None,
            stop_on_completion: bool = True,
    ) -> StepRecord | None:
        if state.completed:
            return None
        if not state.budget.remaining():
            state.stop_reason = "budget"
            state.completed = True
            return None

        active_profile = state.active_profile or self.profile
        self.profile = active_profile
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

        route_events: list[dict[str, Any]] = []
        if profile_selector is not None:
            selected_profile, route_events = profile_selector(observation, active_profile)
            active_profile = selected_profile
            self.profile = active_profile
        state.active_profile = active_profile
        state.last_observation = observation

        if stop_on_completion and active_profile.should_exit(observation, None, state.budget):
            state.stop_reason = "profile_complete"
            step = self._terminal_step_record(
                step_number=len(state.all_steps) + 1,
                observation=observation,
                budget=state.budget,
                stop_reason=state.stop_reason,
                active_profile=active_profile,
                events=route_events,
            )
            state.all_steps.append(step)
            state.recent_steps.append(step)
            self._write_step(step)
            state.completed = True
            return step

        if not state.budget.remaining():
            state.stop_reason = "budget"
            step = self._terminal_step_record(
                step_number=len(state.all_steps) + 1,
                observation=observation,
                budget=state.budget,
                stop_reason=state.stop_reason,
                active_profile=active_profile,
                events=route_events,
            )
            state.all_steps.append(step)
            state.recent_steps.append(step)
            self._write_step(step)
            state.completed = True
            return step

        if self._should_compact(state.all_steps, state.recent_steps):
            state.session_summary = self._compact(
                state.model,
                state.session_summary,
                state.recent_steps,
                observation,
            )
            state.recent_steps = state.recent_steps[-self.profile.recent_steps_to_keep:]

        hints = ObservationHints.from_observation(
            observation=observation,
            previous_observation=state.previous_observation,
            last_action=state.last_action_for_hints,
            modality_profile=self.profile.input_modality_profile,
        )
        prompt_module_results = self._prompt_module_results(
            agent_id=state.agent_id,
            observation=observation,
            hints=hints,
            campaign_memory=state.campaign_memory,
            session_summary=state.session_summary,
            recent_steps=state.recent_steps,
            budget=state.budget,
        )
        profile_prompt_count = state.decision_prompts_sent.get(active_profile.name, 0)
        prompt_stage = self._prompt_stage(profile_prompt_count)
        prompt = self._build_decision_prompt(
            agent_id=state.agent_id,
            campaign_memory=state.campaign_memory,
            session_summary=state.session_summary,
            recent_steps=state.recent_steps,
            budget=state.budget,
            prompt_module_results=prompt_module_results,
            prompt_stage=prompt_stage,
        )

        action, validation = self._decide_with_retry(state.model, prompt)
        state.decision_prompts_sent[active_profile.name] = profile_prompt_count + 1
        executed_action = action
        execution: dict[str, Any] = {}
        if action is None:
            state.budget.record_validation_failure()

        state.budget.consume_tick()

        if action is not None:
            try:
                execution = self._execution_record(state.agent.act_action(action))
            except ActionError as exc:
                executed_action = None
                state.budget.record_validation_failure()
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
        state.recent_steps = state.recent_steps[-self.profile.recent_steps_to_keep:]
        self._write_step(step)
        state.previous_observation = observation
        state.last_action_for_hints = executed_action

        if state.budget.too_many_validation_failures():
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
        if self._has_decision_steps(state.all_steps) and state.last_observation is not None:
            patch = self._commit_memory(
                state.model,
                state.campaign_memory,
                state.session_summary,
                state.recent_steps,
                state.last_observation,
            )
            self.memory_store.save_patch(state.agent_id, patch)

        active_profile = state.active_profile or self.profile
        return ActivityResult(
            activity=active_profile.name,
            agent_id=state.agent_id,
            steps=state.all_steps,
            session_summary=state.session_summary,
            stop_reason=state.stop_reason,
            run_objective=self.run_objective,
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

    def _build_decision_prompt(
            self,
            agent_id: str,
            campaign_memory: dict[str, Any],
            session_summary: SessionSummary,
            recent_steps: list[StepRecord],
            budget: ActivityBudget,
            prompt_module_results: list[PromptModuleResult],
            prompt_stage: PromptStage,
    ) -> DecisionPrompt:
        if self.profile.prompt_mode == "stateful_delta" and prompt_stage == "delta":
            return self._build_stateful_delta_prompt(
                agent_id=agent_id,
                session_summary=session_summary,
                recent_steps=recent_steps,
                budget=budget,
                prompt_module_results=prompt_module_results,
            )
        return self._build_stateless_full_prompt(
            agent_id=agent_id,
            campaign_memory=campaign_memory,
            session_summary=session_summary,
            recent_steps=recent_steps,
            budget=budget,
            prompt_module_results=prompt_module_results,
            prompt_stage=prompt_stage,
        )

    def _build_stateless_full_prompt(
            self,
            agent_id: str,
            campaign_memory: dict[str, Any],
            session_summary: SessionSummary,
            recent_steps: list[StepRecord],
            budget: ActivityBudget,
            prompt_module_results: list[PromptModuleResult],
            prompt_stage: PromptStage,
    ) -> DecisionPrompt:
        system = self._build_full_system_prompt(prompt_stage)
        if self.profile.prompt_layout == "cache_friendly":
            user = self._build_cache_friendly_user_prompt(
                agent_id=agent_id,
                campaign_memory=campaign_memory,
                session_summary=session_summary,
                recent_steps=recent_steps,
                budget=budget,
                prompt_module_results=prompt_module_results,
            )
        else:
            user = self._build_timeline_first_user_prompt(
                agent_id=agent_id,
                campaign_memory=campaign_memory,
                session_summary=session_summary,
                recent_steps=recent_steps,
                budget=budget,
                prompt_module_results=prompt_module_results,
            )
        return DecisionPrompt(system=system, user=user, mode=self.profile.prompt_mode, stage=prompt_stage)

    def _build_full_system_prompt(self, prompt_stage: PromptStage) -> str:
        system_parts = [
            "You are controlling an interactive terminal session.",
            "You may make mistakes and recover from them.",
            "Return only a JSON action object.",
            render_action_schema(self.profile.action_policy),
        ]
        if self.profile.prompt_mode == "stateful_delta" and prompt_stage == "bootstrap":
            system_parts.append(
                "This is the stateful session bootstrap. Future prompts may omit stable instructions, campaign "
                "memory, and full recent-step history; keep this context active across resumed calls."
            )
        if self.profile.system_guidance:
            system_parts.append(f"Activity-specific guidance:\n{self.profile.system_guidance}")
        return "\n".join(system_parts)

    def _build_timeline_first_user_prompt(
            self,
            agent_id: str,
            campaign_memory: dict[str, Any],
            session_summary: SessionSummary,
            recent_steps: list[StepRecord],
            budget: ActivityBudget,
            prompt_module_results: list[PromptModuleResult],
    ) -> str:
        module_text = render_prompt_modules(prompt_module_results)
        return "\n\n".join(
            self._objective_prompt_lines()
            + [
                f"Agent: {agent_id}",
                f"Activity: {self.profile.name}",
                f"Budget: {json.dumps(budget.to_dict(), sort_keys=True)}",
                f"Campaign memory: {json.dumps(campaign_memory, indent=2, sort_keys=True)}",
                f"Session summary: {self._summary_text(session_summary)}",
                f"Recent steps:\n{self._recent_steps_text(recent_steps)}",
                "---",
                f"Current step: {budget.decision_ticks + 1}",
                module_text,
                "---",
            ]
        )

    def _build_cache_friendly_user_prompt(
            self,
            agent_id: str,
            campaign_memory: dict[str, Any],
            session_summary: SessionSummary,
            recent_steps: list[StepRecord],
            budget: ActivityBudget,
            prompt_module_results: list[PromptModuleResult],
    ) -> str:
        stable_module_text = render_prompt_modules(prompt_module_results, levels=STATIC_PROMPT_MODULE_LEVELS)
        tactical_module_text = render_prompt_modules(prompt_module_results, levels=TACTICAL_PROMPT_MODULE_LEVELS)
        sections = self._objective_prompt_lines() + [
            f"Agent: {agent_id}",
            f"Activity: {self.profile.name}",
            stable_module_text,
            f"Campaign memory: {json.dumps(campaign_memory, indent=2, sort_keys=True)}",
            f"Session summary: {self._summary_text(session_summary)}",
            f"Recent steps:\n{self._recent_steps_text(recent_steps)}",
            "---",
            f"Current step: {budget.decision_ticks + 1}",
            f"Budget: {json.dumps(budget.to_dict(), sort_keys=True)}",
            tactical_module_text,
            "---",
        ]
        return "\n\n".join(section for section in sections if section)

    def _build_stateful_delta_prompt(
            self,
            agent_id: str,
            session_summary: SessionSummary,
            recent_steps: list[StepRecord],
            budget: ActivityBudget,
            prompt_module_results: list[PromptModuleResult],
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
            self._objective_prompt_lines(reminder=True)
            + [
                f"Agent: {agent_id}",
                f"Activity: {self.profile.name}",
                f"Budget: {json.dumps(budget.to_dict(), sort_keys=True)}",
                f"Session summary update: {self._summary_text(session_summary)}",
                f"Previous step:\n{self._previous_step_delta_text(recent_steps)}",
                "---",
                f"Current step: {budget.decision_ticks + 1}",
                module_text,
                "---",
                "Return exactly one JSON action.",
            ]
        )
        return DecisionPrompt(system=system, user=user, mode=self.profile.prompt_mode, stage="delta")

    def _objective_prompt_lines(self, *, reminder: bool = False) -> list[str]:
        suffix = " reminder" if reminder else ""
        lines: list[str] = []
        if self.run_objective:
            lines.append(f"Run objective{suffix}: {self.run_objective}")
        lines.append(f"Profile objective{suffix}: {self.profile.objective}")
        return lines

    def _prompt_stage(self, decision_prompts_sent: int) -> PromptStage:
        if self.profile.prompt_mode == "stateful_delta":
            return "bootstrap" if decision_prompts_sent == 0 else "delta"
        return "full"

    def _prompt_module_results(
            self,
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
            activity_name=self.profile.name,
            objective=self.profile.objective,
            observation=observation,
            hints=hints,
            recent_steps=tuple(recent_steps),
            campaign_memory=campaign_memory,
            session_summary=session_summary,
            budget=budget,
            run_objective=self.run_objective,
        )
        return collect_prompt_module_results(self.profile.prompt_modules, context)

    def _decide_with_retry(self, model: ModelAdapter, prompt: DecisionPrompt) -> tuple[Action | None, dict[str, Any]]:
        invalid_responses: list[dict[str, str]] = []
        model_errors: list[dict[str, Any]] = []

        action, first_error, attempt_errors = self._try_model_decision(model, prompt, "initial")
        model_errors.extend(attempt_errors)
        if action is not None:
            return action, self._accepted_validation(model, model_errors=model_errors)
        if first_error is None:
            return None, self._model_error_validation(model_errors)
        invalid_responses.append(self._invalid_response_record(model, "initial", first_error))

        retry_prompt = prompt
        for retry in range(1, self.profile.invalid_json_retries + 1):
            retry_prompt = self._build_retry_prompt(prompt, first_error, retry)
            action, retry_error, attempt_errors = self._try_model_decision(model, retry_prompt, f"repair-{retry}")
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
            model: ModelAdapter,
            prompt: DecisionPrompt,
            stage: str,
    ) -> tuple[Action | None, str | None, list[dict[str, Any]]]:
        model_errors: list[dict[str, Any]] = []
        for provider_attempt in range(0, self.profile.model_error_retries + 1):
            try:
                return model.decide(prompt, self.profile.action_policy), None, model_errors
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
        )

    def _compact(
            self,
            model: ModelAdapter,
            session_summary: SessionSummary,
            recent_steps: list[StepRecord],
            observation: Observation,
    ) -> SessionSummary:
        prompt = CompactionPrompt(
            system=(
                "Compact older terminal activity into a conservative JSON session summary. "
                "Return only JSON with keys: current_state, last_error, open_subgoals, "
                "discovered_facts, failed_actions, strategy_notes. Each list field must "
                "be a list of strings."
            ),
            user="\n\n".join(
                self._objective_prompt_lines()
                + [
                    f"Previous summary:\n{self._summary_text(session_summary)}",
                    f"Steps to summarize:\n{self._recent_steps_text(recent_steps)}",
                    f"Current screen:\n{observation.model_text}",
                    "Preserve observed facts, failed commands, exact error messages, and unresolved goals.",
                ]
            ),
        )
        try:
            return model.compact(prompt)
        except ModelError as exc:
            return SessionSummary(
                current_state=session_summary.current_state,
                last_error=f"compaction_model_error: {exc}",
                open_subgoals=session_summary.open_subgoals,
                discovered_facts=session_summary.discovered_facts,
                failed_actions=session_summary.failed_actions,
                strategy_notes=session_summary.strategy_notes,
            )

    def _commit_memory(
            self,
            model: ModelAdapter,
            campaign_memory: dict[str, Any],
            session_summary: SessionSummary,
            recent_steps: list[StepRecord],
            observation: Observation,
    ):
        prompt = MemoryCommitPrompt(
            system="Return a JSON memory patch for durable campaign memory.",
            user="\n\n".join(
                self._objective_prompt_lines()
                + [
                    f"Existing campaign memory:\n{json.dumps(campaign_memory, indent=2, sort_keys=True)}",
                    f"Session summary:\n{self._summary_text(session_summary)}",
                    f"Recent steps:\n{self._recent_steps_text(recent_steps)}",
                    f"Final screen:\n{observation.model_text}",
                    "Return JSON with durable_facts, strategy_notes, open_tasks, and errors_to_avoid when applicable.",
                ]
            ),
        )
        try:
            return model.commit_memory(prompt)
        except ModelError:
            return MemoryPatch()

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
        return {
            "stage": stage,
            "provider_attempt": provider_attempt,
            "type": error.__class__.__name__,
            "message": self._truncate(str(error), 2_000),
            "command": list(error.command),
            "stdout": self._truncate(error.stdout, 2_000),
            "stderr": self._truncate(error.stderr, 2_000),
        }

    def _model_response_record(self, model: ModelAdapter) -> dict[str, str]:
        raw = getattr(model, "last_response", "")
        parsed = getattr(model, "last_parsed_response", raw)
        record = {
            "response": self._truncate(raw, 2_000),
            "parsed_response": self._truncate(parsed, 2_000),
        }
        reasoning = getattr(model, "last_reasoning", "")
        if reasoning:
            record["reasoning"] = self._truncate(reasoning, 4_000)
        return record

    def _truncate(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + f"...[truncated {len(text) - limit} chars]"

    def _should_compact(self, all_steps: list[StepRecord], recent_steps: list[StepRecord]) -> bool:
        every = self.profile.compact_every_steps
        if every > 0 and len(all_steps) > 0 and len(all_steps) % every == 0:
            return True
        return self._steps_char_count(recent_steps) >= self.profile.compact_recent_chars

    def _steps_char_count(self, steps: list[StepRecord]) -> int:
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
            validation = json.dumps(self._validation_for_context(step.validation), sort_keys=True)
            total += len(action) + len(validation)
        return total

    def _recent_steps_text(self, steps: list[StepRecord]) -> str:
        if not steps:
            return "(none)"
        lines = []
        for index, step in enumerate(steps):
            after_text = "Current terminal observation below."
            if index + 1 < len(steps):
                after_text = self._observation_effect_text(steps[index + 1].observation)
            lines.append(self._step_context_text(step, after_text))
        return "\n\n".join(lines)

    def _previous_step_delta_text(self, steps: list[StepRecord]) -> str:
        if not steps:
            return "(none)"
        step = steps[-1]
        return self._step_context_text(step, "Current terminal observation below.")

    def _step_context_text(self, step: StepRecord, after_text: str) -> str:
        action = step.action or {"action": "terminal_observation" if self._is_terminal_step(step) else "invalid"}
        validation = self._validation_for_context(step.validation)
        return "\n".join(
            [
                f"Step {step.step}",
                f"Observed before action:\n{self._screen_tail(step.observation.get('model_text', ''))}",
                f"Action chosen:\n{json.dumps(action, sort_keys=True)}",
                f"Validation: {json.dumps(validation, sort_keys=True)}",
                f"Observed after action:\n{after_text}",
            ]
        )

    def _screen_tail(self, screen: Any) -> str:
        if not isinstance(screen, str) or self.profile.screen_tail_chars <= 0:
            return ""
        return screen[-self.profile.screen_tail_chars:]

    def _observation_effect_text(self, observation: dict[str, Any]) -> str:
        new_text = self._screen_tail(observation.get("new_text", ""))
        if new_text:
            return new_text
        return self._screen_tail(observation.get("model_text", ""))

    def _validation_for_context(self, validation: dict[str, Any]) -> dict[str, Any]:
        if self.profile.include_model_responses_in_context:
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


class RoutedActivityRunner(ActivityRunner):
    def __init__(
            self,
            name: str,
            default_profile: ActivityProfile,
            routes: tuple[ActivityRoute, ...],
            memory_store: JsonMemoryStore | None = None,
            log_path: Path | str | None = None,
            run_objective: str = "",
    ) -> None:
        super().__init__(
            default_profile,
            memory_store=memory_store,
            log_path=log_path,
            run_objective=run_objective,
        )
        self.name = name
        self.default_profile = default_profile
        self.routes = tuple(sorted(enumerate(routes), key=lambda item: (-item[1].priority, item[0])))
        self._active_route_name = "default"

    def run(self, agent: TerminalAgent, model: ModelAdapter, budget: ActivityBudget | None = None) -> ActivityResult:
        self.profile = self.default_profile
        self._active_route_name = "default"
        result = self._run(
            agent,
            model,
            budget,
            profile_selector=self._select_profile,
            stop_on_completion=False,
        )
        result.activity = self.name
        return result

    def _select_profile(
            self,
            observation: Observation,
            current_profile: ActivityProfile,
    ) -> tuple[ActivityProfile, list[dict[str, Any]]]:
        route = self._matched_route(observation)
        route_name = route.name if route is not None else "default"
        selected_profile = route.profile if route is not None else self.default_profile
        if route_name == self._active_route_name and selected_profile.name == current_profile.name:
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
        self._active_route_name = route_name
        return selected_profile, [event]

    def _matched_route(self, observation: Observation) -> ActivityRoute | None:
        for _, route in self.routes:
            if route.matches(observation):
                return route
        return None
