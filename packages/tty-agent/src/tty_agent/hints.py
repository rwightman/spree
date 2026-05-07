"""Generic observation hints derived from terminal state."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Pattern

from .actions import Action, is_printable_key
from .terminal import Observation

InputModeTarget = Literal["active_prompt", "recent_output", "screen_tail"]

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_SPACE_RE = re.compile(r"[ \t]+")
_ECHO_SPACE_RE = re.compile(r"\s+")
_INPUT_WAIT_RE = re.compile(r"\b(?:press|hit)\s+(?:any\s+)?(?:a\s+)?(?:key|enter|return)\b", re.IGNORECASE)
_YES_NO_CHOICE_RE = re.compile(
    r"\?.*(?:\b[Yy]es\b|\b[Nn]o\b|\([Yy]/[Nn]\)|\[[Yy]es\]|\[[Nn]o\])\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class InputModeRule:
    """A regex rule for classifying the currently visible input mode."""

    mode: str
    pattern: Pattern[str]
    hint: str
    priority: int = 0
    target: InputModeTarget = "active_prompt"

    @classmethod
    def from_pattern(
            cls,
            mode: str,
            pattern: str,
            hint: str,
            priority: int = 0,
            target: InputModeTarget = "active_prompt",
    ) -> "InputModeRule":
        return cls(
            mode=mode,
            pattern=re.compile(pattern, re.IGNORECASE | re.MULTILINE),
            hint=hint,
            priority=priority,
            target=target,
        )


@dataclass(frozen=True)
class InputModalityProfile:
    """Domain-supplied rules for turning prompt text into input-mode hints."""

    rules: tuple[InputModeRule, ...] = ()

    def classify(
            self,
            *,
            active_prompt: str,
            recent_output: str,
            screen_tail: str,
    ) -> tuple[str, str]:
        targets = {
            "active_prompt": active_prompt,
            "recent_output": recent_output,
            "screen_tail": screen_tail,
        }
        ranked_rules = sorted(enumerate(self.rules), key=lambda item: (-item[1].priority, item[0]))
        for _, rule in ranked_rules:
            if rule.pattern.search(targets[rule.target]):
                return rule.mode, rule.hint
        return "unknown", "inspect the screen"


@dataclass(frozen=True)
class ObservationHints:
    """Model-facing hints extracted mechanically from a terminal observation."""

    recent_output: str = ""
    active_prompt: str = "(unknown - inspect the screen)"
    input_mode: str = "unknown"
    input_mode_hint: str = "inspect the screen"
    previous_action_effects: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_observation(
            cls,
            observation: Observation,
            previous_observation: Observation | None,
            last_action: Action | None,
            modality_profile: InputModalityProfile,
    ) -> "ObservationHints":
        recent_output = observation.new_text.strip()
        active_prompt = _active_prompt(observation)
        input_mode, input_mode_hint = modality_profile.classify(
            active_prompt=active_prompt,
            recent_output=recent_output,
            screen_tail=_tail(observation.model_text),
        )
        return cls(
            recent_output=recent_output,
            active_prompt=active_prompt,
            input_mode=input_mode,
            input_mode_hint=input_mode_hint,
            previous_action_effects=_previous_action_effects(observation, previous_observation, last_action),
        )


def _previous_action_effects(
        observation: Observation,
        previous_observation: Observation | None,
        last_action: Action | None,
) -> tuple[str, ...]:
    if previous_observation is None or last_action is None or last_action.action in {"wait", "hangup"}:
        return ()

    effects = []
    visible_input = _visible_unsubmitted_input(last_action)
    if visible_input and _looks_like_echo_only(observation.new_text, visible_input):
        effects.append(
            "Previous input appears to have been echoed on screen without other new output. "
            "If it was meant as a line response, submit it with press_key enter."
        )

    if observation.model_text == previous_observation.model_text and not observation.new_text.strip():
        effects.append("Screen appears unchanged after the previous input.")

    return tuple(effects)


def _visible_unsubmitted_input(action: Action) -> str:
    if action.action == "type_text":
        return action.text
    if action.action == "press_key" and is_printable_key(action.key) and action.key.strip():
        return action.key
    return ""


def _looks_like_echo_only(new_text: str, visible_input: str) -> bool:
    recent = _compact_for_echo(new_text)
    visible = _compact_for_echo(visible_input)
    if not recent or not visible:
        return False
    slack = 24
    if recent == visible:
        return True
    if len(recent) <= len(visible) + slack and (recent.startswith(visible) or recent.endswith(visible)):
        return True
    prefix_window = recent[: max(32, len(visible) + slack)]
    return len(visible) > 1 and visible in prefix_window and len(recent) <= len(visible) + 64


def _active_prompt(observation: Observation) -> str:
    # The pyte screen is the settled visual state, while new_text is the byte
    # stream that just arrived. Some BBS/DOS doors write a live prompt, return
    # the cursor with CR, then blank the cells; recover those prompts from the
    # stream before falling back to older rendered screen text.
    nearest_pretty = _nearest_nonempty_pretty_line(observation)
    if nearest_pretty and _looks_like_prompt(nearest_pretty):
        return nearest_pretty

    new_text_prompt = _prompt_from_new_text(observation.new_text)
    if new_text_prompt:
        return new_text_prompt

    model_lines = observation.model_text.splitlines()
    for line in reversed(model_lines):
        clean = _clean_line(line)
        if clean and _looks_like_prompt(clean):
            return clean
    return "(unknown - inspect the screen)"


def _nearest_nonempty_pretty_line(observation: Observation) -> str:
    pretty_lines = observation.pretty_screen.splitlines()
    if not pretty_lines:
        return ""
    row = observation.cursor[0]
    for index in range(min(row, len(pretty_lines) - 1), -1, -1):
        line = _clean_line(pretty_lines[index])
        if line:
            return line
    return ""


def _prompt_from_new_text(new_text: str) -> str:
    for line in reversed(new_text.split("\n")):
        for segment in reversed(line.split("\r")):
            clean = _clean_line(segment)
            if clean and _looks_like_prompt(clean):
                return clean
    return ""


def _looks_like_prompt(line: str) -> bool:
    clean = _clean_line(line)
    return clean.endswith(("?", ":", ">")) or bool(_INPUT_WAIT_RE.search(clean) or _YES_NO_CHOICE_RE.search(clean))


def _clean_line(line: str) -> str:
    return _SPACE_RE.sub(" ", _CONTROL_RE.sub("", line).rstrip())


def _compact_for_echo(text: str) -> str:
    return _ECHO_SPACE_RE.sub("", _CONTROL_RE.sub("", text))


def _tail(text: str, chars: int = 1200) -> str:
    return text[-chars:]
