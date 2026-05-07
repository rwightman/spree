"""Composable prompt modules for terminal-agent activities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

from .hints import ObservationHints
from .terminal import Observation

if TYPE_CHECKING:
    from .models import SessionSummary
    from .runner import ActivityBudget, StepRecord

AssistanceLevel = Literal["generic_terminal", "bbs_conventions", "game_interface", "strategic"]
ASSISTANCE_LEVEL_ORDER: tuple[AssistanceLevel, ...] = (
    "generic_terminal",
    "bbs_conventions",
    "game_interface",
    "strategic",
)
PROMPT_MODULES_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PromptRenderContext:
    """Inputs available to prompt modules for one decision tick."""

    agent_id: str
    activity_name: str
    objective: str
    observation: Observation
    hints: ObservationHints
    recent_steps: tuple["StepRecord", ...]
    campaign_memory: dict[str, Any]
    session_summary: "SessionSummary"
    budget: "ActivityBudget"
    run_objective: str = ""


@dataclass(frozen=True)
class PromptModuleResult:
    """Rendered text from one prompt module."""

    name: str
    level: AssistanceLevel
    text: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "level": self.level, "text": self.text}


class PromptModule(Protocol):
    name: str
    level: AssistanceLevel

    def render(self, context: PromptRenderContext) -> str | None: ...


@dataclass(frozen=True)
class StaticPromptModule:
    """A prompt module with fixed text."""

    name: str
    level: AssistanceLevel
    text: str

    def render(self, context: PromptRenderContext) -> str:
        del context
        return self.text


@dataclass(frozen=True)
class RecentOutputModule:
    name: str = "terminal.recent_output"
    level: AssistanceLevel = "generic_terminal"
    max_chars: int = 1200

    def render(self, context: PromptRenderContext) -> str:
        recent_output = _bounded(context.hints.recent_output, self.max_chars) or "(none)"
        return f"Most recent terminal output:\n{recent_output}"


@dataclass(frozen=True)
class ActivePromptModule:
    name: str = "terminal.active_prompt"
    level: AssistanceLevel = "generic_terminal"
    max_chars: int = 400

    def render(self, context: PromptRenderContext) -> str:
        active_prompt = _bounded(context.hints.active_prompt, self.max_chars) or "(unknown - inspect the screen)"
        return f"Likely active prompt:\n{active_prompt}"


@dataclass(frozen=True)
class InputModalityModule:
    name: str = "terminal.input_modality"
    level: AssistanceLevel = "generic_terminal"

    def render(self, context: PromptRenderContext) -> str:
        return f"Input mode hint:\n{context.hints.input_mode} - {context.hints.input_mode_hint}"


@dataclass(frozen=True)
class PreviousActionEffectModule:
    name: str = "terminal.previous_action_effect"
    level: AssistanceLevel = "generic_terminal"

    def render(self, context: PromptRenderContext) -> str | None:
        if not context.hints.previous_action_effects:
            return None
        effects = "\n".join(f"- {effect}" for effect in context.hints.previous_action_effects)
        return f"Previous action effect:\n{effects}"


@dataclass(frozen=True)
class FullScreenModule:
    name: str = "terminal.full_screen"
    level: AssistanceLevel = "generic_terminal"

    def render(self, context: PromptRenderContext) -> str:
        return f"Full current screen:\n{context.observation.model_text or '(empty)'}"


GENERIC_TERMINAL_MODULES: tuple[PromptModule, ...] = (
    RecentOutputModule(),
    ActivePromptModule(),
    InputModalityModule(),
    PreviousActionEffectModule(),
    FullScreenModule(),
)


def collect_prompt_module_results(
        modules: tuple[PromptModule, ...],
        context: PromptRenderContext,
) -> list[PromptModuleResult]:
    results: list[PromptModuleResult] = []
    for module in modules:
        text = module.render(context)
        if text is None:
            continue
        results.append(PromptModuleResult(name=module.name, level=module.level, text=text))
    return results


def render_prompt_modules(results: list[PromptModuleResult]) -> str:
    """Render module results grouped by declared assistance level."""

    rendered_groups: list[str] = []
    for level in ASSISTANCE_LEVEL_ORDER:
        group = [result.text for result in results if result.level == level and result.text]
        if group:
            rendered_groups.append(f"[{level}]\n" + "\n\n".join(group))
    return "\n\n".join(rendered_groups)


def prompt_module_trace(results: list[PromptModuleResult]) -> list[dict[str, str]]:
    return [result.to_dict() for result in results]


def _bounded(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[-max_chars:]
