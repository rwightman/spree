"""Structured terminal actions and validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


class ActionError(ValueError):
    """Raised when a model action cannot be parsed or validated."""


CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
DEFAULT_SUPPORTED_KEYS = frozenset(
    {
        "enter",
        "escape",
        "tab",
        "backspace",
        "space",
        "up",
        "down",
        "left",
        "right",
    }
)
DEFAULT_ALLOWED_ACTIONS = frozenset({"press_key", "submit_line", "type_text", "submit_lines", "wait", "hangup"})
TOP_LEVEL_FIELDS = frozenset({"action", "arguments"})
ARGUMENT_FIELDS = frozenset({"text", "lines", "key"})


def is_printable_key(key: str) -> bool:
    return len(key) == 1 and not CONTROL_RE.search(key)


@dataclass(frozen=True)
class Action:
    """A validated terminal action.

    The action wrapper is structured, but the payload can still be open-ended
    text when the current activity policy allows it.
    """

    action: str
    text: str = ""
    lines: tuple[str, ...] = ()
    key: str = ""

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "Action":
        if "tool" in data:
            raise ActionError("tool is not supported; use action and arguments")
        unexpected_fields = sorted(set(data) - TOP_LEVEL_FIELDS)
        if unexpected_fields:
            raise ActionError(f"unexpected top-level fields: {', '.join(unexpected_fields)}")

        action = data.get("action")
        if not isinstance(action, str):
            raise ActionError("action must be a string")
        if not action:
            raise ActionError("action must not be empty")

        arguments = data.get("arguments", {})
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise ActionError("arguments must be an object when present")
        unexpected_arguments = sorted(set(arguments) - ARGUMENT_FIELDS)
        if unexpected_arguments:
            raise ActionError(f"unexpected argument fields: {', '.join(unexpected_arguments)}")

        text = arguments.get("text", "")
        if text is None:
            text = ""
        if not isinstance(text, str):
            raise ActionError("text must be a string when present")

        lines_value = arguments.get("lines", ())
        if lines_value is None:
            lines_value = ()
        if isinstance(lines_value, list):
            lines = tuple(lines_value)
        elif isinstance(lines_value, tuple):
            lines = lines_value
        else:
            raise ActionError("lines must be a list of strings when present")
        if not all(isinstance(line, str) for line in lines):
            raise ActionError("lines must be a list of strings")

        key = arguments.get("key", "")
        if key is None:
            key = ""
        if not isinstance(key, str):
            raise ActionError("key must be a string when present")

        if "newline" in data or "newline" in arguments:
            raise ActionError("newline is not supported; use submit_line or press_key enter")

        return cls(action=action, text=text, lines=lines, key=key)

    def to_dict(self) -> dict[str, Any]:
        arguments: dict[str, Any] = {}
        if self.action in {"submit_line", "type_text", "send_raw"}:
            arguments["text"] = self.text
        elif self.action == "submit_lines":
            arguments["lines"] = list(self.lines)
        elif self.action == "press_key":
            arguments["key"] = self.key
        return {"action": self.action, "arguments": arguments}


@dataclass(frozen=True)
class ActionPolicy:
    """Validation policy for an activity phase."""

    allowed_actions: frozenset[str] = field(default_factory=lambda: DEFAULT_ALLOWED_ACTIONS)
    supported_keys: frozenset[str] = field(default_factory=lambda: DEFAULT_SUPPORTED_KEYS)
    allow_printable_keys: bool = True
    max_text_chars: int = 1024
    max_line_chars: int = 240
    max_lines: int = 20
    allow_control_chars: bool = False
    require_encoding: str | None = None

    def validate(self, action: Action) -> Action:
        if action.action not in self.allowed_actions:
            raise ActionError(f"action {action.action!r} is not allowed")

        if action.action in {"wait", "hangup"}:
            if action.text or action.lines or action.key:
                raise ActionError(f"{action.action} must not include text, lines, or key")
            return action

        if action.action in {"submit_line", "type_text"}:
            self._validate_text(action.text, "text", self.max_text_chars)
            if action.lines or action.key:
                raise ActionError(f"{action.action} must not include lines or key")
            return action

        if action.action == "send_raw":
            self._validate_text(action.text, "text", self.max_text_chars, allow_control_chars=True)
            if action.lines or action.key:
                raise ActionError("send_raw must not include lines or key")
            return action

        if action.action == "submit_lines":
            if action.text:
                raise ActionError("submit_lines must use lines, not text")
            if not action.lines:
                raise ActionError("submit_lines requires at least one line")
            if len(action.lines) > self.max_lines:
                raise ActionError(f"too many lines: {len(action.lines)} > {self.max_lines}")
            for index, line in enumerate(action.lines, start=1):
                self._validate_text(line, f"line {index}", self.max_line_chars)
            return action

        if action.action == "press_key":
            if not action.key:
                raise ActionError("press_key requires key")
            if action.text or action.lines:
                raise ActionError("press_key must not include text or lines")
            if action.key in self.supported_keys:
                return action
            if self.allow_printable_keys and is_printable_key(action.key):
                self._validate_text(action.key, "key", 1)
                return action
            if self.allow_printable_keys:
                supported = ", ".join(sorted(self.supported_keys))
                raise ActionError(
                    f"unsupported key {action.key!r}; use one printable character or one of these named keys: "
                    f"{supported}"
                )
            else:
                supported = ", ".join(sorted(self.supported_keys))
                raise ActionError(
                    f"unsupported key {action.key!r}; supported keys: {supported}. "
                )

        raise ActionError(f"unsupported action {action.action!r}")

    def _validate_text(
            self,
            text: str,
            label: str,
            max_chars: int,
            allow_control_chars: bool | None = None,
    ) -> None:
        if len(text) > max_chars:
            raise ActionError(f"{label} too long: {len(text)} > {max_chars}")
        allow_controls = self.allow_control_chars if allow_control_chars is None else allow_control_chars
        if not allow_controls and CONTROL_RE.search(text):
            raise ActionError(f"{label} contains disallowed control characters")
        if self.require_encoding:
            try:
                text.encode(self.require_encoding)
            except UnicodeEncodeError as exc:
                raise ActionError(
                    f"{label} contains characters that cannot be encoded as {self.require_encoding}"
                ) from exc


def parse_action(text: str, policy: ActionPolicy | None = None) -> Action:
    """Parse a model response containing a JSON terminal action."""

    data = _extract_json_object(text)
    if not isinstance(data, dict):
        raise ActionError("model response must contain a JSON action object")
    action = Action.from_mapping(data)
    return (policy or ActionPolicy()).validate(action)


def render_action_schema(policy: ActionPolicy) -> str:
    """Render only the terminal action forms available under ``policy``."""

    lines = [
        "Use exactly one terminal action per response. Return JSON only.",
        "Allowed terminal actions:",
    ]
    if "press_key" in policy.allowed_actions:
        supported = ", ".join(sorted(policy.supported_keys))
        lines.extend(
            [
                "press_key",
                '{"action": "press_key", "arguments": {"key": "enter"}}',
                '{"action": "press_key", "arguments": {"key": "Q"}}',
                (
                    "Press exactly one key. Does not add Enter/Return unless the key is enter. "
                    f"Supported named keys: {supported}."
                ),
            ]
        )
    if "submit_line" in policy.allowed_actions:
        lines.extend(
            [
                "submit_line",
                '{"action": "submit_line", "arguments": {"text": "42"}}',
                "Type one line of text, then press Enter/Return. Empty text presses Enter/Return.",
            ]
        )
    if "type_text" in policy.allowed_actions:
        if "submit_line" in policy.allowed_actions:
            type_text_description = (
                "Type text without pressing Enter/Return. The cursor advances and the terminal may wait for "
                "more input. Most prompts want submit_line; use type_text only for partial input that you "
                "will finish later."
            )
        else:
            type_text_description = (
                "Type text without pressing Enter/Return. Some programs accept typed values immediately. If the "
                "same prompt remains with your text echoed, use press_key enter to submit it."
            )
        lines.extend(
            [
                "type_text",
                '{"action": "type_text", "arguments": {"text": "42"}}',
                type_text_description,
            ]
        )
    if "submit_lines" in policy.allowed_actions:
        lines.extend(
            [
                "submit_lines",
                '{"action": "submit_lines", "arguments": {"lines": ["line one", "line two"]}}',
                "Submit multiple lines. Each line is followed by Enter/Return.",
            ]
        )
    if "wait" in policy.allowed_actions:
        lines.extend(
            [
                "wait",
                '{"action": "wait", "arguments": {}}',
                "Send nothing and wait for more terminal output.",
            ]
        )
    if "hangup" in policy.allowed_actions:
        lines.extend(
            [
                "hangup",
                '{"action": "hangup", "arguments": {}}',
                "Close the terminal session.",
            ]
        )
    if "send_raw" in policy.allowed_actions:
        lines.extend(
            [
                "send_raw",
                '{"action": "send_raw", "arguments": {"text": "\\u001b[2J"}}',
                "Send exact low-level terminal text without adding Enter/Return.",
            ]
        )
    return "\n".join(lines)


def _extract_json_object(text: str) -> Any:
    decoder = json.JSONDecoder()
    stripped = _strip_fence(text.strip())

    try:
        value, end = decoder.raw_decode(stripped)
    except json.JSONDecodeError:
        value, end = _scan_for_json_object(decoder, stripped)

    trailing = stripped[end:].strip()
    if trailing:
        raise ActionError("unexpected text after JSON action")
    return value


def _scan_for_json_object(decoder: json.JSONDecoder, text: str) -> tuple[Any, int]:
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        prefix = text[:index].strip()
        suffix = text[index + end:].strip()
        if prefix and not prefix.lower().startswith(("json", "action", "terminal action")):
            continue
        if suffix:
            continue
        return value, len(text)
    raise ActionError("no JSON action object found")


def _strip_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text
