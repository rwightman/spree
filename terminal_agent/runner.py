"""Single-agent activity runner for terminal sessions."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from .agent import TerminalAgent
from .actions import Action, ActionError, ActionPolicy, render_action_schema
from .hints import InputModalityProfile, ObservationHints
from .memory import JsonMemoryStore
from .models import CompactionPrompt, DecisionPrompt, MemoryCommitPrompt, ModelAdapter, SessionSummary
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
PromptStage = Literal["full", "bootstrap", "delta"]


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
    poll_interval: float = 0.05
    recent_steps_to_keep: int = 8
    screen_tail_chars: int = 800
    compact_every_steps: int = 20
    compact_recent_chars: int = 12_000
    invalid_json_retries: int = 1
    include_model_responses_in_context: bool = False
    prompt_mode: PromptMode = "stateless_full"
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
    budget: dict[str, Any]
    prompt_modules_schema_version: int = PROMPT_MODULES_SCHEMA_VERSION
    prompt_modules: list[dict[str, str]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "observation": self.observation,
            "prompt": self.prompt,
            "action": self.action,
            "validation": self.validation,
            "budget": self.budget,
            "prompt_modules_schema_version": self.prompt_modules_schema_version,
            "prompt_modules": self.prompt_modules,
            "timestamp": self.timestamp,
        }


@dataclass
class ActivityResult:
    activity: str
    agent_id: str
    steps: list[StepRecord]
    session_summary: SessionSummary
    stop_reason: str


class ActivityRunner:
    def __init__(
            self,
            profile: ActivityProfile,
            memory_store: JsonMemoryStore | None = None,
            log_path: Path | str | None = None,
    ) -> None:
        self.profile = profile
        self.memory_store = memory_store or JsonMemoryStore()
        self.log_path = Path(log_path) if log_path else None

    def run(self, agent: TerminalAgent, model: ModelAdapter, budget: ActivityBudget | None = None) -> ActivityResult:
        budget = budget or ActivityBudget()
        agent_id = getattr(agent, "agent_id", "agent")
        campaign_memory = self.memory_store.load(agent_id)
        session_summary = SessionSummary()
        recent_steps: list[StepRecord] = []
        all_steps: list[StepRecord] = []
        stop_reason = "budget"
        last_observation: Observation | None = None
        previous_observation: Observation | None = None
        last_action_for_hints: Action | None = None
        decision_prompts_sent = 0

        while budget.remaining():
            try:
                observation = agent.observe_turn(
                    timeout=self.profile.observe_timeout,
                    stable_ms=self.profile.stable_ms,
                    poll_interval=self.profile.poll_interval,
                )
            except SessionDisconnected:
                stop_reason = "disconnected"
                break
            last_observation = observation

            if self.profile.should_exit(observation, None, budget):
                stop_reason = "profile_complete"
                step = self._terminal_step_record(
                    step_number=len(all_steps) + 1,
                    observation=observation,
                    budget=budget,
                    stop_reason=stop_reason,
                )
                all_steps.append(step)
                recent_steps.append(step)
                self._write_step(step)
                break

            if not budget.remaining():
                stop_reason = "budget"
                step = self._terminal_step_record(
                    step_number=len(all_steps) + 1,
                    observation=observation,
                    budget=budget,
                    stop_reason=stop_reason,
                )
                all_steps.append(step)
                recent_steps.append(step)
                self._write_step(step)
                break

            if self._should_compact(all_steps, recent_steps):
                session_summary = self._compact(model, session_summary, recent_steps, observation)
                recent_steps = recent_steps[-self.profile.recent_steps_to_keep:]

            hints = ObservationHints.from_observation(
                observation=observation,
                previous_observation=previous_observation,
                last_action=last_action_for_hints,
                modality_profile=self.profile.input_modality_profile,
            )
            prompt_module_results = self._prompt_module_results(
                agent_id=agent_id,
                observation=observation,
                hints=hints,
                campaign_memory=campaign_memory,
                session_summary=session_summary,
                recent_steps=recent_steps,
                budget=budget,
            )
            prompt_stage = self._prompt_stage(decision_prompts_sent)
            prompt = self._build_decision_prompt(
                agent_id=agent_id,
                campaign_memory=campaign_memory,
                session_summary=session_summary,
                recent_steps=recent_steps,
                budget=budget,
                prompt_module_results=prompt_module_results,
                prompt_stage=prompt_stage,
            )

            action, validation = self._decide_with_retry(model, prompt)
            decision_prompts_sent += 1
            executed_action = action
            if action is None:
                budget.record_validation_failure()

            budget.consume_tick()

            if action is not None:
                try:
                    agent.act_action(action)
                except ActionError as exc:
                    executed_action = None
                    budget.record_validation_failure()
                    validation = self._execution_error_validation(validation, str(exc))

            step = StepRecord(
                step=budget.decision_ticks,
                observation=observation.as_dict(),
                prompt={
                    "system": prompt.system,
                    "user": prompt.user,
                    "mode": prompt.mode,
                    "stage": prompt.stage,
                },
                action=action.to_dict() if action else None,
                validation=validation,
                budget=budget.to_dict(),
                prompt_modules=prompt_module_trace(prompt_module_results),
            )
            all_steps.append(step)
            recent_steps.append(step)
            recent_steps = recent_steps[-self.profile.recent_steps_to_keep:]
            self._write_step(step)
            previous_observation = observation
            last_action_for_hints = executed_action

            if budget.too_many_validation_failures():
                stop_reason = "validation_failures"
                break
            if executed_action and executed_action.action == "hangup":
                stop_reason = "hangup"
                break
            if self.profile.should_exit(observation, None, budget):
                stop_reason = "profile_complete"
                break
            if not budget.remaining():
                stop_reason = "budget"
                break

        if self._has_decision_steps(all_steps) and last_observation is not None:
            patch = self._commit_memory(model, campaign_memory, session_summary, recent_steps, last_observation)
            self.memory_store.save_patch(agent_id, patch)

        return ActivityResult(
            activity=self.profile.name,
            agent_id=agent_id,
            steps=all_steps,
            session_summary=session_summary,
            stop_reason=stop_reason,
        )

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
        system = "\n".join(system_parts)
        module_text = render_prompt_modules(prompt_module_results)
        user = "\n\n".join(
            [
                f"Agent: {agent_id}",
                f"Activity: {self.profile.name}",
                f"Objective: {self.profile.objective}",
                f"Budget: {json.dumps(budget.to_dict(), sort_keys=True)}",
                f"Campaign memory: {json.dumps(campaign_memory, indent=2, sort_keys=True)}",
                f"Session summary: {self._summary_text(session_summary)}",
                f"Recent steps: {self._recent_steps_text(recent_steps)}",
                "---",
                module_text,
                "---",
            ]
        )
        return DecisionPrompt(system=system, user=user, mode=self.profile.prompt_mode, stage=prompt_stage)

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
                "Use the action schema, activity objective, stable guidance, and campaign memory already given.",
                "Return only one JSON action object.",
            ]
        )
        module_text = render_prompt_modules(prompt_module_results)
        user = "\n\n".join(
            [
                f"Agent: {agent_id}",
                f"Activity: {self.profile.name}",
                f"Objective reminder: {self.profile.objective}",
                f"Budget: {json.dumps(budget.to_dict(), sort_keys=True)}",
                f"Session summary update: {self._summary_text(session_summary)}",
                f"Previous step: {self._previous_step_delta_text(recent_steps)}",
                "---",
                module_text,
                "---",
                "Return exactly one JSON action.",
            ]
        )
        return DecisionPrompt(system=system, user=user, mode=self.profile.prompt_mode, stage="delta")

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
        )
        return collect_prompt_module_results(self.profile.prompt_modules, context)

    def _decide_with_retry(self, model: ModelAdapter, prompt: DecisionPrompt) -> tuple[Action | None, dict[str, Any]]:
        invalid_responses: list[dict[str, str]] = []
        try:
            action = model.decide(prompt, self.profile.action_policy)
            return action, {"accepted": True, "notes": [], "model_response": self._model_response_record(model)}
        except ActionError as first_exc:
            first_error = str(first_exc)
            invalid_responses.append(self._invalid_response_record(model, "initial", first_error))

        retry_prompt = prompt
        for retry in range(1, self.profile.invalid_json_retries + 1):
            retry_prompt = self._build_retry_prompt(prompt, first_error, retry)
            try:
                action = model.decide(retry_prompt, self.profile.action_policy)
                return action, {
                    "accepted": True,
                    "notes": [f"repaired_after_error: {first_error}"],
                    "model_response": self._model_response_record(model),
                    "invalid_responses": invalid_responses,
                }
            except ActionError as retry_exc:
                first_error = str(retry_exc)
                invalid_responses.append(self._invalid_response_record(model, f"repair-{retry}", first_error))

        return None, {"accepted": False, "notes": [first_error], "invalid_responses": invalid_responses}

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
            budget=budget.to_dict(),
        )

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
                [
                    f"Previous summary:\n{self._summary_text(session_summary)}",
                    f"Steps to summarize:\n{self._recent_steps_text(recent_steps)}",
                    f"Current screen:\n{observation.model_text}",
                    "Preserve observed facts, failed commands, exact error messages, and unresolved goals.",
                ]
            ),
        )
        return model.compact(prompt)

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
                [
                    f"Existing campaign memory:\n{json.dumps(campaign_memory, indent=2, sort_keys=True)}",
                    f"Session summary:\n{self._summary_text(session_summary)}",
                    f"Recent steps:\n{self._recent_steps_text(recent_steps)}",
                    f"Final screen:\n{observation.model_text}",
                    "Return JSON with durable_facts, strategy_notes, open_tasks, and errors_to_avoid when applicable.",
                ]
            ),
        )
        return model.commit_memory(prompt)

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
        for step in steps:
            action = step.action or {"action": "terminal_observation" if self._is_terminal_step(step) else "invalid"}
            validation = self._validation_for_context(step.validation)
            screen = self._screen_tail(step.observation.get("model_text", ""))
            lines.append(
                "\n".join(
                    [
                        f"Step {step.step}: action={json.dumps(action, sort_keys=True)}",
                        f"Validation: {json.dumps(validation, sort_keys=True)}",
                        f"Screen tail:\n{screen}",
                    ]
                )
            )
        return "\n\n".join(lines)

    def _previous_step_delta_text(self, steps: list[StepRecord]) -> str:
        if not steps:
            return "(none)"
        step = steps[-1]
        action = step.action or {"action": "terminal_observation" if self._is_terminal_step(step) else "invalid"}
        validation = self._validation_for_context(step.validation)
        return "\n".join(
            [
                f"Step {step.step}: action={json.dumps(action, sort_keys=True)}",
                f"Validation: {json.dumps(validation, sort_keys=True)}",
            ]
        )

    def _screen_tail(self, screen: Any) -> str:
        if not isinstance(screen, str) or self.profile.screen_tail_chars <= 0:
            return ""
        return screen[-self.profile.screen_tail_chars:]

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
