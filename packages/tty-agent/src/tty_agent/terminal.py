"""Virtual terminal rendering and turn-boundary observation helpers."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyte

from .ansi import strip_ansi
from .profiles import DEFAULT_PROFILE, PromptProfile
from .transports.base import TerminalSession


_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACE_RE = re.compile(r"[ \t]+")


@dataclass(frozen=True)
class Observation:
    """Rendered state returned to a model or runner."""

    agent_id: str
    pretty_screen: str
    model_text: str
    new_text: str
    cursor: tuple[int, int]
    stable_ms: int
    byte_quiet_ms: int
    matched_prompt: str | None
    ready_reason: str
    profile: str
    transcript_path: Path | None
    transcript_byte_start: int
    transcript_byte_end: int
    bytes_read: int
    timed_out: bool
    timestamp: float
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "pretty_screen": self.pretty_screen,
            "model_text": self.model_text,
            "new_text": self.new_text,
            "cursor": list(self.cursor),
            "stable_ms": self.stable_ms,
            "byte_quiet_ms": self.byte_quiet_ms,
            "matched_prompt": self.matched_prompt,
            "ready_reason": self.ready_reason,
            "profile": self.profile,
            "transcript_path": str(self.transcript_path) if self.transcript_path else None,
            "transcript_byte_start": self.transcript_byte_start,
            "transcript_byte_end": self.transcript_byte_end,
            "bytes_read": self.bytes_read,
            "timed_out": self.timed_out,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class TerminalScreen:
    """Headless terminal screen backed by pyte."""

    def __init__(self, columns: int = 80, lines: int = 24, encoding: str = "utf-8") -> None:
        self.columns = columns
        self.lines = lines
        self.encoding = encoding
        self.screen = _ProcessInputScreen(columns, lines)
        self.stream = pyte.Stream(self.screen)

    @property
    def cursor(self) -> tuple[int, int]:
        return (self.screen.cursor.y, self.screen.cursor.x)

    def feed(self, data: bytes) -> bool:
        before = self.signature()
        text = data.decode(self.encoding, errors="replace").replace("\ufeff", "")
        self.stream.feed(text)
        return self.signature() != before

    def drain_process_input(self) -> tuple[bytes, ...]:
        return self.screen.drain_process_input(self.encoding)

    def pretty_screen(self) -> str:
        return "\n".join(self.screen.display)

    def model_text(self) -> str:
        lines = []
        for line in self.screen.display:
            clean = _CONTROL_RE.sub("", line).rstrip()
            clean = _SPACE_RE.sub(" ", clean)
            lines.append(clean)
        while lines and not lines[-1]:
            lines.pop()
        return "\n".join(lines)

    def signature(self) -> tuple[tuple[str, ...], tuple[int, int]]:
        return (tuple(self.screen.display), self.cursor)


class _ProcessInputScreen(pyte.Screen):
    """pyte screen that captures terminal replies for the remote process."""

    def __init__(self, columns: int, lines: int) -> None:
        super().__init__(columns, lines)
        self._process_input: list[str] = []

    def write_process_input(self, data: str) -> None:
        self._process_input.append(data)

    def drain_process_input(self, encoding: str) -> tuple[bytes, ...]:
        chunks = tuple(item.encode(encoding, errors="replace") for item in self._process_input)
        self._process_input.clear()
        return chunks


class TurnObserver:
    """Read from a session until the virtual terminal reaches a turn boundary."""

    def __init__(
            self,
            agent_id: str,
            session: TerminalSession,
            terminal: TerminalScreen | None = None,
            profile: PromptProfile = DEFAULT_PROFILE,
            metadata: dict[str, Any] | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.session = session
        self.terminal = terminal or TerminalScreen()
        self.profile = profile
        self.metadata = metadata or {}

    def feed(self, data: bytes) -> bool:
        changed = self.terminal.feed(data)
        for response in self.terminal.drain_process_input():
            self.session.send_bytes(response)
        return changed

    def observe_turn(
            self,
            timeout: float = 10.0,
            stable_ms: int = 300,
            byte_quiet_ms: int = 0,
            poll_interval: float = 0.05,
            profile: PromptProfile | None = None,
            prompt_fast_path: bool = False,
    ) -> Observation:
        active_profile = profile or self.profile
        start = time.monotonic()
        last_change = start
        last_byte = start
        transcript_byte_start = self.session.transcript_position()
        bytes_read = 0
        new_data = bytearray()
        matched_prompt: str | None = active_profile.match(self.terminal.model_text())

        while True:
            now = time.monotonic()
            elapsed = now - start
            if elapsed >= timeout:
                stable_elapsed_ms = int((now - last_change) * 1000)
                byte_quiet_elapsed_ms = int((now - last_byte) * 1000)
                model_text = self.terminal.model_text()
                matched_prompt = active_profile.match(model_text)
                return self._observation(
                    active_profile,
                    new_data,
                    stable_elapsed_ms,
                    byte_quiet_elapsed_ms,
                    transcript_byte_start,
                    matched_prompt,
                    "timeout",
                )

            data = self.session.read(min(poll_interval, max(0.0, timeout - elapsed)))
            now = time.monotonic()
            if not data:
                stable_elapsed_ms = int((now - last_change) * 1000)
                byte_quiet_elapsed_ms = int((now - last_byte) * 1000)
                model_text = self.terminal.model_text()
                matched_prompt = active_profile.match(model_text)
                screen_is_stable = stable_ms <= 0 or stable_elapsed_ms >= stable_ms
                bytes_are_quiet = byte_quiet_ms <= 0 or byte_quiet_elapsed_ms >= byte_quiet_ms
                if model_text and screen_is_stable and bytes_are_quiet:
                    return self._observation(
                        active_profile,
                        new_data,
                        stable_elapsed_ms,
                        byte_quiet_elapsed_ms,
                        transcript_byte_start,
                        matched_prompt,
                        "stable",
                    )
                continue

            bytes_read += len(data)
            new_data.extend(data)
            last_byte = now
            if self.feed(data):
                last_change = time.monotonic()
            model_text = self.terminal.model_text()
            matched_prompt = active_profile.match(model_text)
            stable_elapsed_ms = int((time.monotonic() - last_change) * 1000)
            byte_quiet_elapsed_ms = int((time.monotonic() - last_byte) * 1000)
            if prompt_fast_path and matched_prompt and bytes_read > 0:
                return self._observation(
                    active_profile,
                    new_data,
                    stable_elapsed_ms,
                    byte_quiet_elapsed_ms,
                    transcript_byte_start,
                    matched_prompt,
                    "prompt",
                )
            screen_is_stable = stable_ms <= 0 or stable_elapsed_ms >= stable_ms
            bytes_are_quiet = byte_quiet_ms <= 0 or byte_quiet_elapsed_ms >= byte_quiet_ms
            if model_text and screen_is_stable and bytes_are_quiet:
                return self._observation(
                    active_profile,
                    new_data,
                    stable_elapsed_ms,
                    byte_quiet_elapsed_ms,
                    transcript_byte_start,
                    matched_prompt,
                    "stable",
                )

    def _observation(
            self,
            profile: PromptProfile,
            new_data: bytes | bytearray,
            stable_ms: int,
            byte_quiet_ms: int,
            transcript_byte_start: int,
            matched_prompt: str | None,
            ready_reason: str,
    ) -> Observation:
        new_text = strip_ansi(bytes(new_data), encoding=self.terminal.encoding).replace("\ufeff", "")
        new_text = _CONTROL_RE.sub("", new_text)
        return Observation(
            agent_id=self.agent_id,
            pretty_screen=self.terminal.pretty_screen(),
            model_text=self.terminal.model_text(),
            new_text=new_text,
            cursor=self.terminal.cursor,
            stable_ms=stable_ms,
            byte_quiet_ms=byte_quiet_ms,
            matched_prompt=matched_prompt,
            ready_reason=ready_reason,
            profile=profile.name,
            transcript_path=self.session.transcript_path,
            transcript_byte_start=transcript_byte_start,
            transcript_byte_end=transcript_byte_start + len(new_data),
            bytes_read=len(new_data),
            timed_out=ready_reason == "timeout",
            timestamp=time.time(),
            metadata=dict(self.metadata),
        )
