"""RLogin transport for terminal automation.

The classic protocol sends four null-terminated fields on connect:
empty string, client username, server username, and terminal/speed. Many BBS
clients use the client username field for the BBS password, which is the
default here.
"""

from __future__ import annotations

import select
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..actions import ActionError, is_printable_key
from .base import SessionDisconnected


RLOGIN_KEY_BYTES = {
    "enter": b"\r\n",
    "escape": b"\x1b",
    "tab": b"\t",
    "backspace": b"\x7f",
    "space": b" ",
    "up": b"\x1b[A",
    "down": b"\x1b[B",
    "right": b"\x1b[C",
    "left": b"\x1b[D",
}


@dataclass
class RLoginSession:
    host: str = "127.0.0.1"
    port: int = 2513
    username: str = ""
    password: str | None = None
    timeout: float = 10.0
    transcript_path: Path | None = None
    encoding: str = "cp437"
    terminal: str = "ansi"
    speed: int = 38400
    reversed_login: bool = False
    _sock: socket.socket | None = field(default=None, init=False, repr=False)
    _transcript: bytearray = field(default_factory=bytearray, init=False, repr=False)
    _ack_pending: bool = field(default=True, init=False, repr=False)

    def connect(self) -> None:
        self._sock = socket.create_connection((self.host, self.port), self.timeout)
        self._sock.setblocking(False)
        self._sock.sendall(self._handshake())

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        if self.transcript_path is not None:
            self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
            self.transcript_path.write_bytes(bytes(self._transcript))

    def __enter__(self) -> "RLoginSession":
        self.connect()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def send_text(self, text: str) -> None:
        self.send_bytes(text.encode(self.encoding))

    def send_line(self, text: str = "") -> None:
        self.send_text(text)
        self.send_key("enter")

    def send_key(self, key: str) -> None:
        payload = RLOGIN_KEY_BYTES.get(key)
        if payload is None and is_printable_key(key):
            payload = key.encode(self.encoding)
        if payload is None:
            supported = ", ".join(sorted(RLOGIN_KEY_BYTES))
            raise ActionError(
                f"unsupported key {key!r}; use one printable character or one of these named keys: {supported}"
            )
        self.send_bytes(payload)

    def send_bytes(self, payload: bytes) -> None:
        if self._sock is None:
            raise RuntimeError("session is not connected")
        self._sock.sendall(payload)

    def read(self, seconds: float = 1.0) -> bytes:
        if self._sock is None:
            raise RuntimeError("session is not connected")

        deadline = time.monotonic() + seconds
        out = bytearray()

        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            ready, _, _ = select.select([self._sock], [], [], min(0.2, remaining))
            if not ready:
                continue
            chunk = self._sock.recv(4096)
            if not chunk:
                if not out:
                    raise SessionDisconnected("remote rlogin connection closed")
                break
            chunk = self._strip_initial_ack(chunk)
            out.extend(chunk)
            self._transcript.extend(chunk)

        return bytes(out)

    def _handshake(self) -> bytes:
        if self.password is None:
            client_user = self.username
            server_user = self.username
        elif self.reversed_login:
            client_user = self.username
            server_user = self.password
        else:
            client_user = self.password
            server_user = self.username
        terminal_speed = f"{self.terminal}/{self.speed}"
        fields = ("", client_user, server_user, terminal_speed)
        return b"".join(field.encode(self.encoding) + b"\x00" for field in fields)

    def _strip_initial_ack(self, data: bytes) -> bytes:
        if self._ack_pending and data.startswith(b"\x00"):
            self._ack_pending = False
            return data[1:]
        if data:
            self._ack_pending = False
        return data
