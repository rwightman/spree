"""Model adapter interfaces and simple HTTP implementations."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

from .actions import Action, ActionPolicy, parse_action


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

    def decide(self, prompt: DecisionPrompt, policy: ActionPolicy | None = None) -> Action:
        ...

    def compact(self, prompt: CompactionPrompt) -> SessionSummary:
        ...

    def commit_memory(self, prompt: MemoryCommitPrompt) -> MemoryPatch:
        ...


class TextChatAdapter:
    """Base class for chat APIs that return plain text."""

    name: str
    last_response: str = ""

    def decide(self, prompt: DecisionPrompt, policy: ActionPolicy | None = None) -> Action:
        self.last_response = self.chat(prompt.messages())
        return parse_action(self.last_response, policy)

    def compact(self, prompt: CompactionPrompt) -> SessionSummary:
        self.last_response = self.chat(prompt.messages()).strip()
        try:
            data = json.loads(self.last_response)
        except json.JSONDecodeError:
            data = {"current_state": self.last_response}
        if not isinstance(data, dict):
            data = {"current_state": self.last_response}
        return SessionSummary.from_mapping(data)

    def commit_memory(self, prompt: MemoryCommitPrompt) -> MemoryPatch:
        self.last_response = self.chat(prompt.messages()).strip()
        try:
            data = json.loads(self.last_response)
        except json.JSONDecodeError:
            data = {"summary": self.last_response}
        if not isinstance(data, dict):
            data = {"summary": self.last_response}
        return MemoryPatch(data)

    def chat(self, messages: list[ModelMessage]) -> str:
        raise NotImplementedError


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
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "local")
        self.name = name or f"openai-compatible:{model}"
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages: list[ModelMessage]) -> str:
        payload = {
            "model": self.model,
            "messages": [message.to_dict() for message in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        response = _post_json(
            f"{self.base_url}/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        try:
            return response["choices"][0]["message"]["content"]
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
            return '{"action": "wait"}'
        return self.responses.pop(0)


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


def _string_tuple(data: dict[str, object], key: str) -> tuple[str, ...]:
    value = data.get(key, ())
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(item for item in value if isinstance(item, str))
    if isinstance(value, tuple):
        return tuple(item for item in value if isinstance(item, str))
    return ()
