"""Shared transport contracts for tty-agent sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO
from typing import Protocol


class SessionDisconnected(RuntimeError):
    """Raised when the remote terminal connection closes."""


@dataclass
class TranscriptWriter:
    """Append-only writer for the authoritative raw session transcript.

    Bytes are flushed as they are read so a transcript survives a crash instead
    of only existing once a session closes cleanly. A session that reconnects to
    the same path appends to the existing transcript, and positions are absolute
    offsets into that file so observation byte ranges stay valid across
    reconnects.
    """

    path: Path | None = None
    _position: int = 0
    _handle: BinaryIO | None = field(default=None, init=False, repr=False)

    def open(self) -> None:
        if self.path is None or self._handle is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("ab")
        self._position = self._handle.tell()

    def record(self, payload: bytes) -> None:
        self._position += len(payload)
        if self._handle is None:
            return
        self._handle.write(payload)
        self._handle.flush()

    def position(self) -> int:
        return self._position

    def close(self) -> None:
        if self._handle is None:
            return
        self._handle.close()
        self._handle = None


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

    def transcript_position(self) -> int: ...

    def read(self, seconds: float = 1.0) -> bytes: ...
