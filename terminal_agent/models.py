"""Model adapter interfaces and simple HTTP implementations."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Protocol

from .actions import Action, ActionPolicy, parse_action

OutputFilter = Callable[[str], str]


@dataclass(frozen=True)
class ModelMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class DecisionPrompt:
    system: str
    user: str

    def messages(self) -> list[ModelMessage]:
        return [ModelMessage("system", self.system), ModelMessage("user", self.user)]


@dataclass(frozen=True)
class CompactionPrompt:
    system: str
    user: str

    def messages(self) -> list[ModelMessage]:
        return [ModelMessage("system", self.system), ModelMessage("user", self.user)]


@dataclass(frozen=True)
class MemoryCommitPrompt:
    system: str
    user: str

    def messages(self) -> list[ModelMessage]:
        return [ModelMessage("system", self.system), ModelMessage("user", self.user)]


@dataclass(frozen=True)
class SessionSummary:
    current_state: str = ""
    last_error: str = ""
    open_subgoals: tuple[str, ...] = ()
    discovered_facts: tuple[str, ...] = ()
    failed_actions: tuple[str, ...] = ()
    strategy_notes: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, data: dict[str, object]) -> "SessionSummary":
        return cls(
            current_state=_string_field(data, "current_state"),
            last_error=_string_field(data, "last_error"),
            open_subgoals=_string_tuple(data, "open_subgoals"),
            discovered_facts=_string_tuple(data, "discovered_facts"),
            failed_actions=_string_tuple(data, "failed_actions"),
            strategy_notes=_string_tuple(data, "strategy_notes"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "current_state": self.current_state,
            "last_error": self.last_error,
            "open_subgoals": list(self.open_subgoals),
            "discovered_facts": list(self.discovered_facts),
            "failed_actions": list(self.failed_actions),
            "strategy_notes": list(self.strategy_notes),
        }

    def is_empty(self) -> bool:
        return not any(
            (
                self.current_state,
                self.last_error,
                self.open_subgoals,
                self.discovered_facts,
                self.failed_actions,
                self.strategy_notes,
            )
        )


@dataclass(frozen=True)
class MemoryPatch:
    data: dict[str, object] = field(default_factory=dict)


class ModelAdapter(Protocol):
    name: str

    def decide(self, prompt: DecisionPrompt, policy: ActionPolicy | None = None) -> Action: ...

    def compact(self, prompt: CompactionPrompt) -> SessionSummary: ...

    def commit_memory(self, prompt: MemoryCommitPrompt) -> MemoryPatch: ...


class TextChatAdapter:
    """Base class for chat APIs that return plain text."""

    name: str
    last_response: str = ""
    last_parsed_response: str = ""
    last_reasoning: str = ""
    output_filters: tuple[OutputFilter, ...] | None = None

    def decide(self, prompt: DecisionPrompt, policy: ActionPolicy | None = None) -> Action:
        self.last_reasoning = ""
        self.last_response = self.chat(prompt.messages())
        self.last_parsed_response = self._filter_output(self.last_response).strip()
        return parse_action(self.last_parsed_response, policy)

    def compact(self, prompt: CompactionPrompt) -> SessionSummary:
        self.last_reasoning = ""
        self.last_response = self.chat(prompt.messages()).strip()
        self.last_parsed_response = self._filter_output(self.last_response).strip()
        try:
            data = json.loads(self.last_parsed_response)
        except json.JSONDecodeError:
            data = {"current_state": self._fallback_output_text()}
        if not isinstance(data, dict):
            data = {"current_state": self._fallback_output_text()}
        return SessionSummary.from_mapping(data)

    def commit_memory(self, prompt: MemoryCommitPrompt) -> MemoryPatch:
        self.last_reasoning = ""
        self.last_response = self.chat(prompt.messages()).strip()
        self.last_parsed_response = self._filter_output(self.last_response).strip()
        try:
            data = json.loads(self.last_parsed_response)
        except json.JSONDecodeError:
            data = {"summary": self._fallback_output_text()}
        if not isinstance(data, dict):
            data = {"summary": self._fallback_output_text()}
        return MemoryPatch(data)

    def chat(self, messages: list[ModelMessage]) -> str:
        raise NotImplementedError

    def _filter_output(self, text: str) -> str:
        filtered = text
        output_filters = DEFAULT_OUTPUT_FILTERS if self.output_filters is None else self.output_filters
        for output_filter in output_filters:
            filtered = output_filter(filtered)
        return filtered

    def _fallback_output_text(self) -> str:
        if self.last_parsed_response or self.last_parsed_response != self.last_response:
            return self.last_parsed_response
        return self.last_response


class OpenAICompatibleAdapter(TextChatAdapter):
    """Adapter for OpenAI-compatible `/v1/chat/completions` servers."""

    def __init__(
            self,
            model: str,
            base_url: str = "http://localhost:11434/v1",
            api_key: str | None = None,
            name: str | None = None,
            timeout: float = 120.0,
            temperature: float = 0.2,
            max_tokens: int = 512,
            extra_body: dict[str, object] | None = None,
            output_filters: tuple[OutputFilter, ...] | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "local")
        self.name = name or f"openai-compatible:{model}"
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = dict(extra_body or {})
        self.output_filters = output_filters_for_model(model) if output_filters is None else output_filters

    def chat(self, messages: list[ModelMessage]) -> str:
        payload = {
            "model": self.model,
            "messages": [message.to_dict() for message in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        payload.update(self.extra_body)
        response = _post_json(
            f"{self.base_url}/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        try:
            message = response["choices"][0]["message"]
            self.last_reasoning = _optional_string(message.get("reasoning") or message.get("reasoning_content"))
            return _optional_string(message.get("content"))
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"unexpected OpenAI-compatible response: {response!r}") from exc


class AnthropicAdapter(TextChatAdapter):
    """Adapter for Anthropic Messages API."""

    def __init__(
            self,
            model: str,
            api_key: str | None = None,
            name: str | None = None,
            base_url: str = "https://api.anthropic.com/v1",
            anthropic_version: str = "2023-06-01",
            timeout: float = 120.0,
            temperature: float = 0.2,
            max_tokens: int = 512,
            cache_system_prompt: bool = True,
    ) -> None:
        self.model = model
        self.api_key = api_key if api_key is not None else os.getenv("ANTHROPIC_API_KEY", "")
        self.name = name or f"anthropic:{model}"
        self.base_url = base_url.rstrip("/")
        self.anthropic_version = anthropic_version
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_system_prompt = cache_system_prompt

    def chat(self, messages: list[ModelMessage]) -> str:
        system = "\n\n".join(message.content for message in messages if message.role == "system")
        chat_messages = [
            {"role": message.role, "content": message.content}
            for message in messages
            if message.role in {"user", "assistant"}
        ]
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": chat_messages,
        }
        if system:
            if self.cache_system_prompt:
                payload["system"] = [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                payload["system"] = system
        response = _post_json(
            f"{self.base_url}/messages",
            payload,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": self.anthropic_version,
            },
            timeout=self.timeout,
        )
        try:
            parts = response["content"]
            return "".join(part.get("text", "") for part in parts if part.get("type") == "text")
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"unexpected Anthropic response: {response!r}") from exc


class ScriptedModelAdapter(TextChatAdapter):
    """Deterministic adapter for tests and dry runs."""

    def __init__(self, responses: list[str], name: str = "scripted") -> None:
        self.responses = list(responses)
        self.name = name

    def chat(self, messages: list[ModelMessage]) -> str:
        del messages
        if not self.responses:
            return '{"action": "wait", "arguments": {}}'
        return self.responses.pop(0)


REASONING_BLOCK_RE = re.compile(
    r"<(think|thinking|reasoning|analysis)\b[^>]*>.*?</\1>\s*",
    re.DOTALL | re.IGNORECASE,
)
UNCLOSED_REASONING_BLOCK_RE = re.compile(
    r"<(think|thinking|reasoning|analysis)\b[^>]*>.*\Z",
    re.DOTALL | re.IGNORECASE,
)
GEMMA_CHANNEL_OPEN = r"<\|channel\|?>"
GEMMA_CHANNEL_CLOSE = r"(?:<channel\|>|<\|channel\|>)"
GEMMA_CLOSED_THOUGHT_CHANNEL_RE = re.compile(
    rf"{GEMMA_CHANNEL_OPEN}\s*thought\b.*?{GEMMA_CHANNEL_CLOSE}\s*",
    re.DOTALL | re.IGNORECASE,
)
GEMMA_THOUGHT_BEFORE_FINAL_RE = re.compile(
    rf"{GEMMA_CHANNEL_OPEN}\s*thought\b.*?(?={GEMMA_CHANNEL_OPEN}\s*(?:final|answer)\b)",
    re.DOTALL | re.IGNORECASE,
)
GEMMA_UNCLOSED_THOUGHT_CHANNEL_RE = re.compile(
    rf"{GEMMA_CHANNEL_OPEN}\s*thought\b.*\Z",
    re.DOTALL | re.IGNORECASE,
)
GEMMA_CHANNEL_MARKER_RE = re.compile(
    rf"(?:{GEMMA_CHANNEL_OPEN}\s*(?:final|answer|assistant)?\b\s*|{GEMMA_CHANNEL_CLOSE}\s*)",
    re.IGNORECASE,
)


def strip_reasoning_blocks(text: str) -> str:
    return UNCLOSED_REASONING_BLOCK_RE.sub("", REASONING_BLOCK_RE.sub("", text))


def strip_gemma4_channel_reasoning(text: str) -> str:
    filtered = GEMMA_CLOSED_THOUGHT_CHANNEL_RE.sub("", text)
    filtered = GEMMA_THOUGHT_BEFORE_FINAL_RE.sub("", filtered)
    filtered = GEMMA_UNCLOSED_THOUGHT_CHANNEL_RE.sub("", filtered)
    return GEMMA_CHANNEL_MARKER_RE.sub("", filtered)


DEFAULT_OUTPUT_FILTERS = (strip_reasoning_blocks,)
GEMMA4_OUTPUT_FILTERS = DEFAULT_OUTPUT_FILTERS + (strip_gemma4_channel_reasoning,)


def output_filters_for_model(model: str, filter_family: str | None = None) -> tuple[OutputFilter, ...]:
    family = _normalize_filter_family(filter_family) if filter_family else "auto"
    if family == "auto":
        family = _infer_filter_family(model)
    if family == "none":
        return ()
    if family == "gemma4":
        return GEMMA4_OUTPUT_FILTERS
    return DEFAULT_OUTPUT_FILTERS


def _infer_filter_family(model: str) -> str:
    normalized = model.lower().replace("_", "-").replace("/", "-")
    if "gemma-4" in normalized or "gemma4" in normalized:
        return "gemma4"
    return "default"


def _normalize_filter_family(filter_family: str) -> str:
    normalized = filter_family.lower().replace("_", "-")
    if normalized in {"auto", "model"}:
        return "auto"
    if normalized in {"off", "none", "raw"}:
        return "none"
    if normalized in {"gemma4", "gemma-4"}:
        return "gemma4"
    return "default"


def _post_json(
        url: str,
        payload: dict[str, object],
        headers: dict[str, str] | None = None,
        timeout: float = 120.0,
) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            **(headers or {}),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def _string_field(data: dict[str, object], key: str) -> str:
    value = data.get(key, "")
    return value if isinstance(value, str) else ""


def _optional_string(value: object) -> str:
    return value if isinstance(value, str) else ""


def _string_tuple(data: dict[str, object], key: str) -> tuple[str, ...]:
    value = data.get(key, ())
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(item for item in value if isinstance(item, str))
    if isinstance(value, tuple):
        return tuple(item for item in value if isinstance(item, str))
    return ()
