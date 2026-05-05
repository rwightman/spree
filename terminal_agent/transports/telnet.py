"""Small telnet client for terminal automation.

This intentionally avoids third-party dependencies. It handles enough telnet
option negotiation to keep classic terminal servers talking while preserving
the raw stream for transcripts.
"""

from __future__ import annotations

import select
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path

from .base import SessionDisconnected

IAC = 255
DONT = 254
DO = 253
WONT = 252
WILL = 251
SB = 250
SE = 240


@dataclass
class TelnetSession:
    host: str = "127.0.0.1"
    port: int = 2323
    timeout: float = 10.0
    transcript_path: Path | None = None
    encoding: str = "utf-8"
    _sock: socket.socket | None = field(default=None, init=False, repr=False)
    _transcript: bytearray = field(default_factory=bytearray, init=False, repr=False)
    _closed_by_peer: bool = field(default=False, init=False, repr=False)

    def connect(self) -> None:
        self._sock = socket.create_connection((self.host, self.port), self.timeout)
        self._sock.setblocking(False)

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        if self.transcript_path is not None:
            self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
            self.transcript_path.write_bytes(bytes(self._transcript))

    def __enter__(self) -> "TelnetSession":
        self.connect()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def send(self, text: str, newline: bool = True) -> None:
        payload = text.encode(self.encoding)
        if newline:
            payload += b"\r\n"
        self.send_bytes(payload)

    def send_bytes(self, payload: bytes) -> None:
        if self._sock is None:
            raise RuntimeError("session is not connected")
        self._sock.sendall(payload)

    def read(self, seconds: float = 1.0) -> bytes:
        """Read for up to ``seconds`` and return application bytes."""

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
                self._closed_by_peer = True
                if not out:
                    raise SessionDisconnected("remote terminal connection closed")
                break
            app_data = self._handle_telnet(chunk)
            out.extend(app_data)
            self._transcript.extend(app_data)

        return bytes(out)

    def read_until(self, needle: bytes, timeout: float | None = None) -> bytes:
        deadline = time.monotonic() + (timeout if timeout is not None else self.timeout)
        out = bytearray()
        while time.monotonic() < deadline:
            out.extend(self.read(0.25))
            if needle in out:
                break
        return bytes(out)

    def _handle_telnet(self, data: bytes) -> bytes:
        if self._sock is None:
            raise RuntimeError("session is not connected")

        out = bytearray()
        i = 0

        while i < len(data):
            byte = data[i]
            if byte != IAC:
                out.append(byte)
                i += 1
                continue

            i += 1
            if i >= len(data):
                break

            command = data[i]
            i += 1

            if command == IAC:
                out.append(IAC)
                continue

            if command in (DO, DONT, WILL, WONT):
                if i >= len(data):
                    break
                option = data[i]
                i += 1
                response = WONT if command in (DO, DONT) else DONT
                self._sock.sendall(bytes([IAC, response, option]))
                continue

            if command == SB:
                while i < len(data):
                    if data[i] == IAC and i + 1 < len(data) and data[i + 1] == SE:
                        i += 2
                        break
                    i += 1

        return bytes(out)
