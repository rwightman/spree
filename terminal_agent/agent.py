"""Agent contracts and concrete session-backed terminal agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .actions import Action, ActionError
from .ansi import strip_ansi
from .profiles import PromptProfile
from .terminal import Observation, TurnObserver
from .transports.base import TerminalSession, sent_bytes_trace


@dataclass(frozen=True)
class ActionExecution:
    sent_bytes: tuple[bytes, ...] = ()
    encoding: str = "utf-8"

    def to_dict(self) -> dict[str, Any]:
        return {"sent_bytes": sent_bytes_trace(self.sent_bytes, self.encoding)}


class TerminalAgent(Protocol):
    agent_id: str

    def observe_turn(
            self,
            timeout: float = 10.0,
            stable_ms: int = 300,
            poll_interval: float = 0.05,
            profile: PromptProfile | None = None,
            prompt_fast_path: bool = False,
    ) -> Observation: ...

    def act_action(self, action: Action) -> ActionExecution: ...


@dataclass
class TerminalSessionAgent:
    """Concrete terminal agent backed by a session and turn observer."""

    agent_id: str
    session: TerminalSession
    observer: TurnObserver
    metadata: dict[str, Any] = field(default_factory=dict)

    def observe(self, seconds: float = 1.0, plain: bool = True) -> str | bytes:
        data = self.session.read(seconds)
        self.observer.feed(data)
        return strip_ansi(data, encoding=self.session.encoding) if plain else data

    def observe_turn(
            self,
            timeout: float = 10.0,
            stable_ms: int = 300,
            poll_interval: float = 0.05,
            profile: PromptProfile | None = None,
            prompt_fast_path: bool = False,
    ) -> Observation:
        return self.observer.observe_turn(timeout, stable_ms, poll_interval, profile, prompt_fast_path)

    def act(self, text: str) -> None:
        self.session.send_line(text)

    def act_action(self, action: Action) -> ActionExecution:
        self.session.drain_sent_bytes()
        if action.action == "wait":
            return self._execution_result()
        if action.action == "hangup":
            self.close()
            return self._execution_result()
        if action.action == "submit_line":
            self.session.send_line(action.text)
            return self._execution_result()
        if action.action == "type_text":
            self.session.send_text(action.text)
            return self._execution_result()
        if action.action == "press_key":
            self.session.send_key(action.key)
            return self._execution_result()
        if action.action == "send_raw":
            self.session.send_bytes(action.text.encode(self.session.encoding))
            return self._execution_result()
        if action.action == "submit_lines":
            for line in action.lines:
                self.session.send_line(line)
            return self._execution_result()
        raise ActionError(f"unsupported action {action.action!r}")

    def close(self) -> None:
        self.session.close()

    def _execution_result(self) -> ActionExecution:
        return ActionExecution(sent_bytes=self.session.drain_sent_bytes(), encoding=self.session.encoding)
