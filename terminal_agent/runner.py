"""Single-agent activity runner for terminal sessions."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agent import TerminalAgent
from .actions import Action, ActionError, ActionPolicy
from .memory import JsonMemoryStore
from .models import CompactionPrompt, DecisionPrompt, MemoryCommitPrompt, ModelAdapter, SessionSummary
from .transports.base import SessionDisconnected
from .terminal import Observation


ACTION_SCHEMA_TEXT = """Return exactly one JSON object using one of these forms:
{"action": "send", "text": "text to type"}
{"action": "send_raw", "text": "exact terminal text"}
{"action": "send_multiline", "lines": ["line one", "line two"]}
{"action": "wait"}
{"action": "hangup"}
"""


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
    observe_timeout: float = 10.0
    stable_ms: int = 300
    poll_interval: float = 0.05
    recent_steps_to_keep: int = 8
    compact_every_steps: int = 20
    compact_recent_chars: int = 12_000
    invalid_json_retries: int = 1

    def should_exit(self, observation: Observation, action: Action | None, budget: ActivityBudget) -> bool:
        del observation
        return action is not None and action.action == "hangup" or not budget.remaining()


@dataclass
class StepRecord:
    step: int
    observation: dict[str, Any]
    prompt: dict[str, str]
    action: dict[str, Any] | None
    validation: dict[str, Any]
    budget: dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "observation": self.observation,
            "prompt": self.prompt,
            "action": self.action,
            "validation": self.validation,
            "budget": self.budget,
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
                break

            if self._should_compact(all_steps, recent_steps):
                session_summary = self._compact(model, session_summary, recent_steps, observation)
                recent_steps = recent_steps[-self.profile.recent_steps_to_keep:]

            prompt = self._build_decision_prompt(
                agent_id=agent_id,
                observation=observation,
                campaign_memory=campaign_memory,
                session_summary=session_summary,
                recent_steps=recent_steps,
                budget=budget,
            )

            action: Action | None = None
            action, validation = self._decide_with_retry(model, prompt)
            if action is None:
                budget.record_validation_failure()

            budget.consume_tick()

            if action is not None:
                agent.act_action(action)

            step = StepRecord(
                step=budget.decision_ticks,
                observation=observation.as_dict(),
                prompt={"system": prompt.system, "user": prompt.user},
                action=action.to_dict() if action else None,
                validation=validation,
                budget=budget.to_dict(),
            )
            all_steps.append(step)
            recent_steps.append(step)
            recent_steps = recent_steps[-self.profile.recent_steps_to_keep:]
            self._write_step(step)

            if budget.too_many_validation_failures():
                stop_reason = "validation_failures"
                break
            if self.profile.should_exit(observation, action, budget):
                stop_reason = "hangup" if action and action.action == "hangup" else "budget"
                break

        if all_steps and last_observation is not None:
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
            observation: Observation,
            campaign_memory: dict[str, Any],
            session_summary: SessionSummary,
            recent_steps: list[StepRecord],
            budget: ActivityBudget,
    ) -> DecisionPrompt:
        system = "\n".join(
            [
                "You are controlling an interactive terminal session.",
                "You may make mistakes and recover from them.",
                "Return only a JSON action object.",
                ACTION_SCHEMA_TEXT,
            ]
        )
        user = "\n\n".join(
            [
                f"Agent: {agent_id}",
                f"Activity: {self.profile.name}",
                f"Objective: {self.profile.objective}",
                f"Budget: {json.dumps(budget.to_dict(), sort_keys=True)}",
                f"Campaign memory: {json.dumps(campaign_memory, indent=2, sort_keys=True)}",
                f"Session summary: {self._summary_text(session_summary)}",
                f"Recent steps: {self._recent_steps_text(recent_steps)}",
                f"Current screen:\n{observation.model_text}",
            ]
        )
        return DecisionPrompt(system=system, user=user)

    def _decide_with_retry(self, model: ModelAdapter, prompt: DecisionPrompt) -> tuple[Action | None, dict[str, Any]]:
        invalid_responses: list[dict[str, str]] = []
        try:
            action = model.decide(prompt, self.profile.action_policy)
            return action, {"accepted": True, "notes": []}
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
                    "invalid_responses": invalid_responses,
                }
            except ActionError as retry_exc:
                first_error = str(retry_exc)
                invalid_responses.append(self._invalid_response_record(model, f"repair-{retry}", first_error))

        return None, {"accepted": False, "notes": [first_error], "invalid_responses": invalid_responses}

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
        raw = getattr(model, "last_response", "")
        return {
            "attempt": attempt,
            "error": error,
            "response": self._truncate(raw, 2_000),
        }

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
            for key in ("model_text", "new_text"):
                value = obs.get(key, "")
                if isinstance(value, str):
                    total += len(value)
            action = json.dumps(step.action or {}, sort_keys=True)
            validation = json.dumps(step.validation, sort_keys=True)
            total += len(action) + len(validation)
        return total

    def _recent_steps_text(self, steps: list[StepRecord]) -> str:
        if not steps:
            return "(none)"
        lines = []
        for step in steps:
            action = step.action or {"action": "invalid"}
            validation = step.validation
            screen = step.observation.get("model_text", "")
            if isinstance(screen, str):
                screen = screen[-800:]
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

    def _write_step(self, step: StepRecord) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(step.to_dict(), sort_keys=True) + "\n")
