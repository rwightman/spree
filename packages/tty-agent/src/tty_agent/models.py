"""Model adapter interfaces and simple HTTP implementations."""

from __future__ import annotations

import http.client
import json
import os
import re
import socket
import subprocess
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
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


class ModelError(RuntimeError):
    """Raised when a model provider fails before producing a parseable response."""

    def __init__(
            self,
            message: str,
            *,
            command: list[str] | tuple[str, ...] = (),
            stdout: str | bytes | None = "",
            stderr: str | bytes | None = "",
            status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.command = tuple(command)
        self.stdout = _subprocess_text(stdout)
        self.stderr = _subprocess_text(stderr)
        self.status_code = status_code


class ModelTimeoutError(ModelError):
    """Raised when a model provider command times out."""


class ModelStateError(ModelError):
    """Raised when provider-side conversation state can no longer be resumed."""


@dataclass(frozen=True)
class DecisionPrompt:
    system: str
    user: str
    mode: str = "stateless_full"
    stage: str = "full"

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
    last_response_id: str | None = None
    last_usage: dict[str, object] | None = None
    last_provider_metadata: dict[str, object] | None = None
    output_filters: tuple[OutputFilter, ...] | None = None

    def decide(self, prompt: DecisionPrompt, policy: ActionPolicy | None = None) -> Action:
        self._reset_response_trace()
        self.last_response = self.chat(prompt.messages())
        self.last_parsed_response = self._filter_output(self.last_response).strip()
        return parse_action(self.last_parsed_response, policy)

    def compact(self, prompt: CompactionPrompt) -> SessionSummary:
        self._reset_response_trace()
        self.last_response = self.compaction_chat(prompt.messages()).strip()
        self.last_parsed_response = self._filter_output(self.last_response).strip()
        self._ensure_complete_utility_response("compaction")
        data = _json_mapping_from_text(self.last_parsed_response)
        if data is None:
            data = {"current_state": self._fallback_output_text()}
        summary = SessionSummary.from_mapping(data)
        if summary.is_empty():
            raise ModelError("compaction returned an empty session summary")
        return summary

    def commit_memory(self, prompt: MemoryCommitPrompt) -> MemoryPatch:
        self._reset_response_trace()
        self.last_response = self.memory_chat(prompt.messages()).strip()
        self.last_parsed_response = self._filter_output(self.last_response).strip()
        self._ensure_complete_utility_response("memory commit")
        data = _json_mapping_from_text(self.last_parsed_response)
        if data is None:
            data = {"summary": self._fallback_output_text()}
        if not data:
            raise ModelError("memory commit returned an empty patch")
        return MemoryPatch(data)

    def chat(self, messages: list[ModelMessage]) -> str:
        raise NotImplementedError

    def compaction_chat(self, messages: list[ModelMessage]) -> str:
        """Run a compaction request, allowing adapters to select utility settings."""

        return self.chat(messages)

    def memory_chat(self, messages: list[ModelMessage]) -> str:
        """Run a memory-commit request, allowing adapters to select utility settings."""

        return self.chat(messages)

    def _filter_output(self, text: str) -> str:
        filtered = text
        output_filters = DEFAULT_OUTPUT_FILTERS if self.output_filters is None else self.output_filters
        for output_filter in output_filters:
            filtered = output_filter(filtered)
        return filtered

    def _reset_response_trace(self) -> None:
        self.last_response = ""
        self.last_parsed_response = ""
        self.last_reasoning = ""
        self.last_response_id = None
        self.last_usage = None
        self.last_provider_metadata = None

    def _fallback_output_text(self) -> str:
        # Deliberately the filtered text, not the raw response: when filters strip
        # a response down to nothing the model only emitted reasoning, and raw
        # reasoning must not be stored as a summary or memory patch.
        return self.last_parsed_response

    def _ensure_complete_utility_response(self, operation: str) -> None:
        metadata = self.last_provider_metadata or {}
        finish_reason = metadata.get("finish_reason")
        if finish_reason in {"length", "max_tokens"}:
            raise ModelError(f"{operation} response was truncated ({finish_reason=})")
        if metadata.get("status") == "incomplete":
            raise ModelError(f"{operation} response was incomplete: {metadata.get('incomplete_details')!r}")
        if not self.last_parsed_response:
            raise ModelError(f"{operation} returned no usable content")


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
            extra_headers: dict[str, str] | None = None,
            compaction_reasoning: bool | None = None,
            compaction_extra_body: dict[str, object] | None = None,
            memory_reasoning: bool | None = None,
            memory_extra_body: dict[str, object] | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "local")
        self.name = name or f"openai-compatible:{model}"
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = dict(extra_body or {})
        self.extra_headers = dict(extra_headers or {})
        self.compaction_reasoning = compaction_reasoning
        self.compaction_extra_body = dict(compaction_extra_body or {})
        self.memory_reasoning = memory_reasoning
        self.memory_extra_body = dict(memory_extra_body or {})
        self.output_filters = output_filters_for_model(model) if output_filters is None else output_filters

    def chat(self, messages: list[ModelMessage]) -> str:
        return self._chat(messages, extra_body=self.extra_body)

    def compaction_chat(self, messages: list[ModelMessage]) -> str:
        return self._chat(
            messages,
            extra_body=_operation_extra_body(
                self.extra_body,
                self.compaction_extra_body,
                self.compaction_reasoning,
            ),
        )

    def memory_chat(self, messages: list[ModelMessage]) -> str:
        return self._chat(
            messages,
            extra_body=_operation_extra_body(
                self.extra_body,
                self.memory_extra_body,
                self.memory_reasoning,
            ),
        )

    def _chat(
            self,
            messages: list[ModelMessage],
            *,
            extra_body: dict[str, object],
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [message.to_dict() for message in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        payload.update(extra_body)
        response = _post_json(
            f"{self.base_url}/chat/completions",
            payload,
            headers={
                **self.extra_headers,
                # Provider credentials always win over user-supplied headers.
                "Authorization": f"Bearer {self.api_key}",
            },
            timeout=self.timeout,
        )
        try:
            message = response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelError(f"unexpected OpenAI-compatible response: {response!r}") from exc
        if not isinstance(message, dict):
            raise ModelError(f"unexpected OpenAI-compatible response: {response!r}")
        self.last_response_id = _optional_nonempty_string(response.get("id"))
        self.last_usage = _optional_mapping(response.get("usage"))
        provider_metadata: dict[str, object] = {}
        perf_metrics = _optional_mapping(response.get("perf_metrics"))
        if perf_metrics is not None:
            provider_metadata["perf_metrics"] = perf_metrics
        finish_reason = response.get("choices", [{}])[0].get("finish_reason")
        if isinstance(finish_reason, str):
            provider_metadata["finish_reason"] = finish_reason
        self.last_provider_metadata = provider_metadata or None
        self.last_reasoning = _optional_string(message.get("reasoning") or message.get("reasoning_content"))
        content = message.get("content")
        if not isinstance(content, str):
            # Silently coercing a malformed message to "" would surface later as
            # an inexplicable invalid-action retry instead of a provider error.
            raise ModelError(f"unexpected OpenAI-compatible response: {response!r}")
        return content


class ResponsesCompatibleAdapter(TextChatAdapter):
    """Adapter for OpenAI-compatible ``/v1/responses`` servers.

    Only terminal decisions participate in the optional provider-side chain.
    Compaction, memory commits, and campaign-social prompts remain stateless so
    their utility instructions do not pollute the gameplay conversation.
    """

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
            extra_headers: dict[str, str] | None = None,
            stateful: bool = False,
            response_id: str | None = None,
            state_file: str | Path | None = None,
            resume: bool = False,
            compaction_reasoning: bool | None = None,
            compaction_extra_body: dict[str, object] | None = None,
            memory_reasoning: bool | None = None,
            memory_extra_body: dict[str, object] | None = None,
    ) -> None:
        if response_id is not None and not stateful:
            raise ValueError("response_id requires stateful Responses mode")
        if state_file is not None and not stateful:
            raise ValueError("state_file requires stateful Responses mode")
        if resume and not stateful:
            raise ValueError("resume requires stateful Responses mode")
        if resume and response_id is None and state_file is None:
            raise ValueError("resume requires response_id or state_file")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "local")
        self.name = name or f"responses-compatible:{model}"
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = dict(extra_body or {})
        self.extra_headers = dict(extra_headers or {})
        self.compaction_reasoning = compaction_reasoning
        self.compaction_extra_body = dict(compaction_extra_body or {})
        self.memory_reasoning = memory_reasoning
        self.memory_extra_body = dict(memory_extra_body or {})
        self.output_filters = output_filters_for_model(model) if output_filters is None else output_filters
        self.stateful = stateful
        self.response_id = response_id
        self._decision_instructions: str | None = None
        self.state_file = Path(state_file) if state_file is not None else None
        # Supplying an ID is itself an explicit resume request. A state file is
        # only read when resume=True; otherwise it is an output checkpoint for
        # a possible later process and must not join an ordinary new run onto
        # an older provider-side conversation.
        self.resume = resume or response_id is not None
        if self.response_id is None and self.state_file is not None and self.resume:
            self.response_id = self._read_state_file()
        self._resume_pending = self.response_id is not None

    def decide(self, prompt: DecisionPrompt, policy: ActionPolicy | None = None) -> Action:
        self._reset_response_trace()
        use_state = self.stateful and prompt.mode == "stateful_delta" and prompt.stage in {"bootstrap", "delta"}
        if prompt.mode == "stateful_delta" and not self.stateful:
            raise ModelError("stateful_delta prompts require stateful Responses mode")
        if use_state and prompt.stage == "delta" and self.response_id is None:
            raise ModelStateError("Responses conversation state is unavailable; resend a bootstrap prompt")
        request_messages = prompt.messages()
        if use_state and prompt.stage == "bootstrap":
            if self._resume_pending:
                # Persisted provider state is consumed once. Any later
                # bootstrap on this adapter represents a new ActivityRunState
                # and therefore starts a fresh chain.
                self._resume_pending = False
            else:
                self.reset_state()
            self._decision_instructions = prompt.system
        elif use_state and prompt.stage == "delta":
            if self._decision_instructions is None:
                raise ModelStateError("Responses bootstrap instructions are unavailable; resend a bootstrap prompt")
            # OpenAI-style Responses APIs do not promise that a prior call's
            # instructions carry forward with previous_response_id. Reapply the
            # locally-owned bootstrap instructions while keeping the large user
            # context in provider-side state.
            request_messages = [
                ModelMessage("system", f"{self._decision_instructions}\n\n{prompt.system}"),
                ModelMessage("user", prompt.user),
            ]
        self.last_response = self._request(
            request_messages,
            use_state=use_state,
            allow_state_restart=prompt.stage == "bootstrap",
            extra_body=self.extra_body,
        )
        self.last_parsed_response = self._filter_output(self.last_response).strip()
        return parse_action(self.last_parsed_response, policy)

    def chat(self, messages: list[ModelMessage]) -> str:
        return self._request(
            messages,
            use_state=False,
            allow_state_restart=False,
            extra_body=self.extra_body,
        )

    def compaction_chat(self, messages: list[ModelMessage]) -> str:
        return self._request(
            messages,
            use_state=False,
            allow_state_restart=False,
            extra_body=_operation_extra_body(
                self.extra_body,
                self.compaction_extra_body,
                self.compaction_reasoning,
                reasoning_format="responses_effort",
            ),
        )

    def memory_chat(self, messages: list[ModelMessage]) -> str:
        return self._request(
            messages,
            use_state=False,
            allow_state_restart=False,
            extra_body=_operation_extra_body(
                self.extra_body,
                self.memory_extra_body,
                self.memory_reasoning,
                reasoning_format="responses_effort",
            ),
        )

    def _request(
            self,
            messages: list[ModelMessage],
            *,
            use_state: bool,
            allow_state_restart: bool,
            extra_body: dict[str, object],
    ) -> str:
        instructions, response_input = _responses_prompt(messages)
        payload: dict[str, object] = {
            "temperature": self.temperature,
            **extra_body,
            "model": self.model,
            "input": response_input,
            "max_output_tokens": self.max_tokens,
            "store": use_state,
        }
        if instructions:
            payload["instructions"] = instructions
        previous_response_id = self.response_id if use_state else None
        payload.pop("previous_response_id", None)
        if previous_response_id is not None:
            payload["previous_response_id"] = previous_response_id

        try:
            response = self._post_response(payload)
        except ModelError as exc:
            if previous_response_id is None or not _is_missing_response_state_error(exc):
                raise
            self._set_response_id(None)
            if not allow_state_restart:
                self._decision_instructions = None
                raise ModelStateError(
                    "Responses conversation state was rejected; resend a bootstrap prompt",
                    stderr=exc.stderr,
                    status_code=exc.status_code,
                ) from exc
            payload.pop("previous_response_id", None)
            response = self._post_response(payload)

        status = response.get("status")
        if status in {"failed", "cancelled"}:
            raise ModelError(f"Responses request ended with status {status!r}: {response.get('error')!r}")

        response_id = _optional_nonempty_string(response.get("id"))
        if use_state:
            if response_id is None:
                raise ModelError(f"stored Responses result had no response id: {response!r}")
            self._set_response_id(response_id)
        self.last_response_id = response_id
        self.last_usage = _optional_mapping(response.get("usage"))
        self.last_reasoning = _responses_reasoning_text(response)
        provider_metadata: dict[str, object] = {}
        if isinstance(status, str):
            provider_metadata["status"] = status
        incomplete_details = _optional_mapping(response.get("incomplete_details"))
        if incomplete_details is not None:
            provider_metadata["incomplete_details"] = incomplete_details
        self.last_provider_metadata = provider_metadata or None
        return _responses_output_text(response)

    def _post_response(self, payload: dict[str, object]) -> dict[str, object]:
        return _post_json(
            f"{self.base_url}/responses",
            payload,
            headers={
                **self.extra_headers,
                "Authorization": f"Bearer {self.api_key}",
            },
            timeout=self.timeout,
        )

    def _read_state_file(self) -> str | None:
        assert self.state_file is not None
        if not self.state_file.is_file():
            return None
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read Responses state file {self.state_file}: {exc}") from exc
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise ValueError(f"invalid Responses state file: {self.state_file}")
        if data.get("base_url") != self.base_url or data.get("model") != self.model:
            raise ValueError(
                f"Responses state file does not match the configured endpoint and model: {self.state_file}"
            )
        response_id = data.get("response_id")
        if response_id is not None and not isinstance(response_id, str):
            raise ValueError(f"invalid response_id in Responses state file: {self.state_file}")
        return response_id or None

    def reset_state(self) -> None:
        """Start the next bootstrap without prior provider conversation state."""

        self._resume_pending = False
        self._decision_instructions = None
        self._set_response_id(None)

    def _set_response_id(self, response_id: str | None) -> None:
        self.response_id = response_id
        if self.state_file is None:
            return
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "schema_version": 1,
            "base_url": self.base_url,
            "model": self.model,
            "response_id": response_id,
        }
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.state_file.parent,
                prefix=f".{self.state_file.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                json.dump(state, handle, indent=2, sort_keys=True)
                handle.write("\n")
                temporary_path = Path(handle.name)
            os.replace(temporary_path, self.state_file)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()


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
            output_filters: tuple[OutputFilter, ...] | None = None,
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
        self.output_filters = output_filters_for_model(model) if output_filters is None else output_filters

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
        except (KeyError, TypeError) as exc:
            raise ModelError(f"unexpected Anthropic response: {response!r}") from exc
        if not isinstance(parts, list):
            raise ModelError(f"unexpected Anthropic response: {response!r}")
        texts: list[str] = []
        for part in parts:
            if not isinstance(part, dict):
                raise ModelError(f"unexpected Anthropic response: {response!r}")
            if part.get("type") != "text":
                continue
            if not isinstance(part.get("text"), str):
                raise ModelError(f"unexpected Anthropic response: {response!r}")
            texts.append(part["text"])
        if not texts:
            # Thinking-only or empty content is a malformed completion for this
            # API shape; do not coerce it to an empty reply.
            raise ModelError(f"Anthropic response contained no text blocks: {response!r}")
        return "".join(texts)


class SubprocessCliAdapter(TextChatAdapter):
    """Base for adapters that shell out to a local coding-agent CLI.

    Subclasses supply the command to run; this base owns the session-id file used
    to resume stateful conversations across calls.
    """

    session_id: str | None = None
    session_file: Path | None = None

    def _read_session_file(self) -> str | None:
        if self.session_file is None or not self.session_file.exists():
            return None
        session_id = self.session_file.read_text(encoding="utf-8").strip()
        return session_id or None

    def _write_session_file(self) -> None:
        if self.session_file is None or self.session_id is None:
            return
        self.session_file.parent.mkdir(parents=True, exist_ok=True)
        self.session_file.write_text(self.session_id + "\n", encoding="utf-8")


class CodexCliAdapter(SubprocessCliAdapter):
    """Adapter that invokes the local ``codex exec`` CLI for each model call."""

    def __init__(
            self,
            model: str | None = None,
            profile: str | None = None,
            executable: str = "codex",
            timeout: float = 600.0,
            sandbox: str = "read-only",
            cwd: str | Path | None = None,
            extra_args: list[str] | None = None,
            stateful: bool = False,
            session_id: str | None = None,
            session_file: str | Path | None = None,
            name: str | None = None,
            output_filters: tuple[OutputFilter, ...] | None = None,
    ) -> None:
        self.model = model
        self.profile = profile
        self.executable = executable
        self.timeout = timeout
        self.sandbox = sandbox
        self.cwd = Path(cwd) if cwd is not None else None
        self.extra_args = list(extra_args or [])
        self.stateful = stateful
        self.session_id = session_id
        self.session_file = Path(session_file) if session_file is not None else None
        self.name = name or _codex_adapter_name(model, profile)
        self.output_filters = output_filters_for_model(model or "") if output_filters is None else output_filters
        if self.session_id is None:
            self.session_id = self._read_session_file()

    def chat(self, messages: list[ModelMessage]) -> str:
        prompt_text = _cli_prompt_text(messages)
        with tempfile.TemporaryDirectory(prefix="tty-agent-codex-") as temp_dir:
            output_path = Path(temp_dir) / "last-message.txt"
            command = self._command(output_path)
            if self.cwd is not None:
                self.cwd.mkdir(parents=True, exist_ok=True)
            try:
                result = subprocess.run(
                    command,
                    input=prompt_text,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout,
                    cwd=self.cwd,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                stdout = _subprocess_text(exc.stdout or exc.output)
                stderr = _subprocess_text(exc.stderr)
                detail = _command_failure_detail(stdout, stderr)
                raise ModelTimeoutError(
                    f"codex exec timed out after {self.timeout:g}s: {detail}",
                    command=command,
                    stdout=stdout,
                    stderr=stderr,
                ) from exc
            except OSError as exc:
                raise ModelError(f"failed to run codex executable {self.executable!r}: {exc}", command=command) from exc
            if result.returncode != 0:
                detail = _command_failure_detail(result.stdout, result.stderr)
                raise ModelError(
                    f"codex exec failed with exit code {result.returncode}: {detail}",
                    command=command,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
            if self.stateful:
                session_id = _extract_codex_session_id(result.stdout)
                if session_id is None and self.session_id is None:
                    raise ModelError(
                        "codex exec did not report a session id in --json output",
                        command=command,
                        stdout=result.stdout,
                        stderr=result.stderr,
                    )
                # A CLI that forks a new session on resume saves the turn under
                # the new id, so always resume the most recently reported one.
                if session_id is not None and session_id != self.session_id:
                    self.session_id = session_id
                    self._write_session_file()
            if output_path.exists():
                return output_path.read_text(encoding="utf-8").strip()
            if self.stateful:
                # --json stdout is an event stream, never a chat message.
                return ""
            return result.stdout.strip()

    def _command(self, output_path: Path) -> list[str]:
        if self.stateful and self.session_id:
            return self._resume_command(output_path)

        command = [
            self.executable,
            "exec",
            "--color",
            "never",
            "--skip-git-repo-check",
            "--sandbox",
            self.sandbox,
            "--output-last-message",
            str(output_path),
        ]
        if self.stateful:
            command.append("--json")
        else:
            command.append("--ephemeral")
        if self.model:
            command.extend(["--model", self.model])
        if self.profile:
            command.extend(["--profile", self.profile])
        command.extend(self.extra_args)
        command.append("-")
        return command

    def _resume_command(self, output_path: Path) -> list[str]:
        command = [
            self.executable,
            "exec",
            "resume",
            "--skip-git-repo-check",
            "--output-last-message",
            str(output_path),
            "--json",
        ]
        if self.model:
            command.extend(["--model", self.model])
        command.extend(self.extra_args)
        command.append(self.session_id or "")
        command.append("-")
        return command


class ClaudeCliAdapter(SubprocessCliAdapter):
    """Adapter that invokes the local ``claude -p`` CLI for each model call."""

    def __init__(
            self,
            model: str | None = None,
            executable: str = "claude",
            timeout: float = 600.0,
            cwd: str | Path | None = None,
            extra_args: list[str] | None = None,
            stateful: bool = False,
            session_id: str | None = None,
            session_file: str | Path | None = None,
            permission_mode: str | None = "dontAsk",
            tools: str | None = "",
            bare: bool = False,
            name: str | None = None,
            output_filters: tuple[OutputFilter, ...] | None = None,
    ) -> None:
        self.model = model
        self.executable = executable
        self.timeout = timeout
        self.cwd = Path(cwd) if cwd is not None else None
        self.extra_args = list(extra_args or [])
        self.stateful = stateful
        self.session_id = session_id
        self.session_file = Path(session_file) if session_file is not None else None
        self.permission_mode = permission_mode
        self.tools = tools
        self.bare = bare
        self.name = name or _claude_adapter_name(model)
        self.output_filters = output_filters_for_model(model or "") if output_filters is None else output_filters
        if self.session_id is None:
            self.session_id = self._read_session_file()

    def chat(self, messages: list[ModelMessage]) -> str:
        prompt_text = _cli_prompt_text(messages)
        command = self._command()
        if self.cwd is not None:
            self.cwd.mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run(
                command,
                input=prompt_text,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                cwd=self.cwd,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = _subprocess_text(exc.stdout or exc.output)
            stderr = _subprocess_text(exc.stderr)
            detail = _command_failure_detail(stdout, stderr)
            raise ModelTimeoutError(
                f"claude -p timed out after {self.timeout:g}s: {detail}",
                command=command,
                stdout=stdout,
                stderr=stderr,
            ) from exc
        except OSError as exc:
            raise ModelError(f"failed to run claude executable {self.executable!r}: {exc}", command=command) from exc
        if result.returncode != 0:
            detail = _command_failure_detail(result.stdout, result.stderr)
            raise ModelError(
                f"claude -p failed with exit code {result.returncode}: {detail}",
                command=command,
                stdout=result.stdout,
                stderr=result.stderr,
            )

        output, session_id = _parse_claude_json_result(result.stdout)
        if self.stateful:
            if session_id is None and self.session_id is None:
                raise ModelError(
                    "claude -p did not report a session id in JSON output",
                    command=command,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
            # A CLI that forks a new session on resume saves the turn under the
            # new id, so always resume the most recently reported one.
            if session_id is not None and session_id != self.session_id:
                self.session_id = session_id
                self._write_session_file()
        # A legitimately empty completion must stay empty; falling back to raw
        # stdout would hand the JSON envelope to the action parser and memory
        # commit paths.
        return output

    def _command(self) -> list[str]:
        command = [
            self.executable,
            "-p",
            "--output-format",
            "json",
            "--input-format",
            "text",
        ]
        if self.bare:
            command.append("--bare")
        if not self.stateful:
            command.append("--no-session-persistence")
        if self.stateful and self.session_id:
            command.extend(["--resume", self.session_id])
        if self.model:
            command.extend(["--model", self.model])
        if self.permission_mode:
            command.extend(["--permission-mode", self.permission_mode])
        if self.tools is not None:
            # The default empty string matters: omitting --tools would leave the
            # CLI's own tools enabled for a model that drives a BBS terminal.
            command.extend(["--tools", self.tools])
        command.extend(self.extra_args)
        return command


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


def _codex_adapter_name(model: str | None, profile: str | None) -> str:
    if model and profile:
        return f"codex:{model}:{profile}"
    if model:
        return f"codex:{model}"
    if profile:
        return f"codex:{profile}"
    return "codex"


def _claude_adapter_name(model: str | None) -> str:
    if model:
        return f"claude:{model}"
    return "claude"


def _cli_prompt_text(messages: list[ModelMessage]) -> str:
    """Flatten chat messages into one prompt for a coding-agent CLI."""

    lines = [
        "You are being invoked non-interactively as a decision model for a tty-agent harness.",
        "Do not run shell commands, inspect files, or modify the workspace.",
        "Return only the final text requested by the harness.",
        "",
    ]
    for message in messages:
        role = message.role.upper()
        lines.extend([f"{role} MESSAGE:", message.content.strip(), ""])
    return "\n".join(lines).rstrip() + "\n"


def _subprocess_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _command_failure_detail(stdout: str, stderr: str) -> str:
    detail = "\n".join(part for part in (stderr.strip(), stdout.strip()) if part)
    if not detail:
        return "(no output)"
    return detail[-2000:]


UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
SESSION_ID_KEYS = ("session_id", "conversation_id", "thread_id")


def _parse_claude_json_result(stdout: str) -> tuple[str, str | None]:
    """Extract final text and session id from ``claude -p --output-format json`` output."""

    data = _json_mapping_from_text(stdout)
    if data is None:
        return stdout.strip(), None
    session_id = _find_uuid_for_keys(data, SESSION_ID_KEYS) or _find_uuid_anywhere(data)
    for key in ("result", "response", "output", "text", "message"):
        value = data.get(key)
        if isinstance(value, str):
            return value.strip(), session_id
    content = data.get("content")
    if isinstance(content, str):
        return content.strip(), session_id
    if isinstance(content, list):
        text = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
        if text:
            return text.strip(), session_id
    return stdout.strip(), session_id


def _extract_codex_session_id(stdout: str) -> str | None:
    """Extract a Codex session id from ``codex exec --json`` event output."""

    fallback: str | None = None
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        preferred = _find_uuid_for_keys(event, SESSION_ID_KEYS)
        if preferred:
            return preferred
        if fallback is None:
            fallback = _find_uuid_anywhere(event)
    return fallback


def _find_uuid_for_keys(value: object, keys: tuple[str, ...]) -> str | None:
    if isinstance(value, dict):
        for key in keys:
            item = value.get(key)
            if isinstance(item, str) and UUID_RE.fullmatch(item):
                return item
        for item in value.values():
            found = _find_uuid_for_keys(item, keys)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_uuid_for_keys(item, keys)
            if found:
                return found
    return None


def _find_uuid_anywhere(value: object) -> str | None:
    if isinstance(value, str):
        match = UUID_RE.search(value)
        return match.group(0) if match else None
    if isinstance(value, dict):
        for item in value.values():
            found = _find_uuid_anywhere(item)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_uuid_anywhere(item)
            if found:
                return found
    return None


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


FENCED_JSON_RE = re.compile(r"```(?:\s*json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _json_mapping_from_text(text: str) -> dict[str, object] | None:
    """Parse a JSON object even when a text model wraps it in prose or fences."""

    decoder = json.JSONDecoder()
    candidates = [text.strip()]
    candidates.extend(match.group(1).strip() for match in FENCED_JSON_RE.finditer(text))

    for candidate in candidates:
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data

    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            data, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _post_json(
        url: str,
        payload: dict[str, object],
        headers: dict[str, str] | None = None,
        timeout: float = 120.0,
) -> dict[str, object]:
    """POST JSON and return the decoded response.

    Every provider-side failure is raised as ``ModelError`` so HTTP adapters get
    the same retry, logging, and graceful-stop handling as the CLI adapters.
    """

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
            # Decode defensively: an error page from a proxy in front of the
            # provider need not be valid UTF-8, and a decode failure here would
            # otherwise escape as a bare UnicodeDecodeError.
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = _read_error_body(exc)
        raise ModelError(
            f"HTTP {exc.code} from {url}: {detail}",
            stderr=detail,
            status_code=exc.code,
        ) from exc
    except socket.timeout as exc:
        raise ModelTimeoutError(f"request to {url} timed out after {timeout:g}s", stderr=str(exc)) from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, socket.timeout):
            raise ModelTimeoutError(f"request to {url} timed out after {timeout:g}s", stderr=str(exc)) from exc
        raise ModelError(f"could not reach {url}: {exc.reason}", stderr=str(exc)) from exc
    except http.client.HTTPException as exc:
        # Truncated or malformed HTTP framing, e.g. IncompleteRead when a
        # provider drops the connection partway through the body.
        raise ModelError(f"malformed HTTP response from {url}: {exc!r}", stderr=str(exc)) from exc
    except OSError as exc:
        raise ModelError(f"could not reach {url}: {exc}", stderr=str(exc)) from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ModelError(f"non-JSON response from {url}: {exc}", stdout=raw[:2_000]) from exc


def _read_error_body(error: urllib.error.HTTPError) -> str:
    """Read an HTTP error body without letting a second failure escape."""

    try:
        return error.read().decode("utf-8", errors="replace")
    except (OSError, http.client.HTTPException, ValueError):
        return "(error body unavailable)"


def _string_field(data: dict[str, object], key: str) -> str:
    value = data.get(key, "")
    return value if isinstance(value, str) else ""


def _optional_string(value: object) -> str:
    return value if isinstance(value, str) else ""


def _optional_nonempty_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_mapping(value: object) -> dict[str, object] | None:
    return dict(value) if isinstance(value, dict) else None


def _responses_prompt(messages: list[ModelMessage]) -> tuple[str, object]:
    instructions = "\n\n".join(message.content for message in messages if message.role == "system")
    inputs = [message.to_dict() for message in messages if message.role != "system"]
    if len(inputs) == 1 and inputs[0]["role"] == "user":
        response_input: object = inputs[0]["content"]
    else:
        response_input = inputs
    return instructions, response_input


def _responses_output_text(response: dict[str, object]) -> str:
    output = response.get("output")
    if not isinstance(output, list):
        raise ModelError(f"unexpected Responses result: {response!r}")
    texts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message" or item.get("role") != "assistant":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "output_text":
                continue
            text = part.get("text")
            if isinstance(text, str):
                texts.append(text)
    if not texts:
        raise ModelError(f"Responses result contained no assistant output text: {response!r}")
    return "".join(texts)


def _responses_reasoning_text(response: dict[str, object]) -> str:
    texts: list[str] = []
    reasoning = response.get("reasoning")
    if isinstance(reasoning, dict):
        texts.extend(_nested_text_values(reasoning.get("content")))
        texts.extend(_nested_text_values(reasoning.get("summary")))
    output = response.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "reasoning":
                continue
            texts.extend(_nested_text_values(item.get("content")))
            texts.extend(_nested_text_values(item.get("summary")))
    return "\n".join(text for text in texts if text)


def _nested_text_values(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    texts: list[str] = []
    for item in value:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            texts.append(item["text"])
    return texts


def _is_missing_response_state_error(error: ModelError) -> bool:
    if error.status_code not in {400, 404, 409, 422}:
        return False
    # ModelError's message includes the request URL (normally ending in
    # /responses), so matching it makes unrelated 404s look state-related.
    # Restrict classification to the provider's error body and require an
    # explicit reference to the prior-response field.
    detail = error.stderr.casefold().strip()
    if not detail:
        return False
    references_previous_response = any(
        marker in detail
        for marker in (
            "previous_response_id",
            "previous response",
            "previous_response_not_found",
            "prior response",
        )
    )
    rejects_reference = any(
        marker in detail
        for marker in (
            "not found",
            "not be found",
            "does not exist",
            "invalid",
            "not valid",
            "expired",
            "deleted",
            "missing",
            "unknown",
            "unavailable",
        )
    )
    return references_previous_response and rejects_reference


def _operation_extra_body(
        base: dict[str, object],
        override: dict[str, object],
        reasoning: bool | None,
        *,
        reasoning_format: str = "enabled",
) -> dict[str, object]:
    """Build one utility request body without mutating decision settings."""

    body = {**base, **override}
    if reasoning is None:
        return body
    reasoning_options = body.get("reasoning")
    if isinstance(reasoning_options, dict):
        reasoning_options = dict(reasoning_options)
    else:
        reasoning_options = {}
    if reasoning_format == "responses_effort":
        reasoning_options.pop("enabled", None)
        current_effort = reasoning_options.get("effort")
        if reasoning:
            if not isinstance(current_effort, str) or current_effort.casefold() in {"none", "off"}:
                reasoning_options["effort"] = "medium"
        else:
            reasoning_options["effort"] = "none"
    else:
        # Several Chat Completions-compatible servers use this extension. It
        # is deliberately not treated as a portable OpenAI API field.
        reasoning_options["enabled"] = reasoning
    body["reasoning"] = reasoning_options
    return body


def _string_tuple(data: dict[str, object], key: str) -> tuple[str, ...]:
    value = data.get(key, ())
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(item for item in value if isinstance(item, str))
    if isinstance(value, tuple):
        return tuple(item for item in value if isinstance(item, str))
    return ()
