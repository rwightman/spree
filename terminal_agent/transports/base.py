"""Shared transport contracts for terminal-agent sessions."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class SessionDisconnected(RuntimeError):
    """Raised when the remote terminal connection closes."""


class TerminalSession(Protocol):
    transcript_path: Path | None
    encoding: str

    def connect(self) -> None: ...

    def close(self) -> None: ...

    def send(self, text: str, newline: bool = True) -> None: ...

    def send_bytes(self, payload: bytes) -> None: ...

    def read(self, seconds: float = 1.0) -> bytes: ...
