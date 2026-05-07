"""Shared transport contracts for terminal-agent sessions."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Protocol


class SessionDisconnected(RuntimeError):
    """Raised when the remote terminal connection closes."""


def bytes_trace(payload: bytes, encoding: str) -> dict[str, Any]:
    return {
        "len": len(payload),
        "repr": repr(payload),
        "text": payload.decode(encoding, errors="replace").encode("unicode_escape").decode("ascii"),
        "hex": payload.hex(" "),
    }


def sent_bytes_trace(chunks: tuple[bytes, ...], encoding: str) -> dict[str, Any]:
    combined = b"".join(chunks)
    return {
        "encoding": encoding,
        "total_bytes": len(combined),
        "combined": bytes_trace(combined, encoding),
        "chunks": [bytes_trace(chunk, encoding) for chunk in chunks],
    }


class TerminalSession(Protocol):
    transcript_path: Path | None
    encoding: str

    def connect(self) -> None: ...

    def close(self) -> None: ...

    def send_text(self, text: str) -> None: ...

    def send_line(self, text: str = "") -> None: ...

    def send_key(self, key: str) -> None: ...

    def send_bytes(self, payload: bytes) -> None: ...

    def drain_sent_bytes(self) -> tuple[bytes, ...]: ...

    def read(self, seconds: float = 1.0) -> bytes: ...
