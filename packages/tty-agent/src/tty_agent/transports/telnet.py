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

from ..actions import ActionError, is_printable_key
from .base import SessionDisconnected, TranscriptWriter

IAC = 255
DONT = 254
DO = 253
WONT = 252
WILL = 251
SB = 250
SE = 240

TELNET_KEY_BYTES = {
    "escape": b"\x1b",
    "tab": b"\t",
    "backspace": b"\x7f",
    "space": b" ",
    "up": b"\x1b[A",
    "down": b"\x1b[B",
    "right": b"\x1b[C",
    "left": b"\x1b[D",
}

TELNET_ENTER_SEQUENCES = {
    "cr": b"\r",
    "lf": b"\n",
    "crlf": b"\r\n",
}

# A negotiation held across reads is at most a few bytes, so anything larger
# means the stream lost its framing; drop it instead of buffering forever.
MAX_PENDING_NEGOTIATION_BYTES = 4096


@dataclass
class TelnetSession:
    host: str = "127.0.0.1"
    port: int = 2323
    timeout: float = 10.0
    transcript_path: Path | None = None
    encoding: str = "utf-8"
    enter_sequence: str = "cr"
    _sock: socket.socket | None = field(default=None, init=False, repr=False)
    _transcript: TranscriptWriter = field(init=False, repr=False)
    _sent_bytes: list[bytes] = field(default_factory=list, init=False, repr=False)
    _pending_negotiation: bytearray = field(default_factory=bytearray, init=False, repr=False)
    _discarding_subnegotiation: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._transcript = TranscriptWriter(self.transcript_path)

    def connect(self) -> None:
        # A reused session object must not carry protocol state from a previous
        # connection into a new one.
        self._pending_negotiation.clear()
        self._discarding_subnegotiation = False
        # create_connection leaves the socket in timeout mode, so sendall retries
        # partial sends itself instead of failing on a full send buffer.
        self._sock = socket.create_connection((self.host, self.port), self.timeout)
        try:
            self._transcript.open()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        self._transcript.close()

    def __enter__(self) -> "TelnetSession":
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
        payload = self._enter_bytes() if key == "enter" else TELNET_KEY_BYTES.get(key)
        if payload is None and is_printable_key(key):
            payload = key.encode(self.encoding)
        if payload is None:
            supported = ", ".join(["enter", *sorted(TELNET_KEY_BYTES)])
            raise ActionError(
                f"unsupported key {key!r}; use one printable character or one of these named keys: {supported}"
            )
        self.send_bytes(payload)

    def _enter_bytes(self) -> bytes:
        try:
            return TELNET_ENTER_SEQUENCES[self.enter_sequence]
        except KeyError as exc:
            supported = ", ".join(sorted(TELNET_ENTER_SEQUENCES))
            raise ActionError(
                f"unsupported telnet enter sequence {self.enter_sequence!r}; use one of: {supported}"
            ) from exc

    def send_bytes(self, payload: bytes) -> None:
        # RFC 854: a 0xFF data byte must be doubled so the server does not read
        # it as IAC (reachable via e.g. U+00A0 in cp437).
        self._write(payload.replace(bytes([IAC]), bytes([IAC, IAC])))
        self._sent_bytes.append(bytes(payload))

    def _write(self, payload: bytes) -> None:
        """Write to the socket without recording it in the agent action trace."""

        if self._sock is None:
            raise SessionDisconnected("session is not connected")
        try:
            self._sock.sendall(payload)
        except TimeoutError as exc:
            raise SessionDisconnected(
                f"remote terminal stopped accepting data for {self.timeout:g}s while sending"
            ) from exc
        except OSError as exc:
            raise SessionDisconnected(f"remote terminal connection closed while sending: {exc}") from exc

    def drain_sent_bytes(self) -> tuple[bytes, ...]:
        chunks = tuple(self._sent_bytes)
        self._sent_bytes.clear()
        return chunks

    def transcript_position(self) -> int:
        return self._transcript.position()

    def read(self, seconds: float = 1.0) -> bytes:
        """Read for up to ``seconds`` and return application bytes."""

        if self._sock is None:
            raise SessionDisconnected("session is not connected")

        deadline = time.monotonic() + seconds
        out = bytearray()

        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            ready, _, _ = select.select([self._sock], [], [], min(0.2, remaining))
            if not ready:
                continue
            try:
                chunk = self._sock.recv(4096)
            except (BlockingIOError, TimeoutError):
                continue
            except OSError as exc:
                # A peer that resets the connection surfaces here rather than as a
                # clean zero-length read.
                if out:
                    break
                raise SessionDisconnected(f"remote terminal connection closed: {exc}") from exc
            if not chunk:
                if not out:
                    raise SessionDisconnected("remote terminal connection closed")
                break
            app_data = self._handle_telnet(chunk)
            out.extend(app_data)
            self._transcript.record(app_data)

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
        """Strip telnet negotiation from ``data`` and return application bytes.

        Negotiations can straddle a TCP read, so an incomplete trailing sequence
        is held over and resumed with the next chunk. Dropping it instead would
        leak the continuation bytes into the application stream and leave the
        option unanswered.
        """

        if self._sock is None:
            raise SessionDisconnected("session is not connected")

        buffer = bytes(self._pending_negotiation) + data
        self._pending_negotiation.clear()
        out = bytearray()
        i = 0

        if self._discarding_subnegotiation:
            end, dangling_iac = _subnegotiation_end(buffer, 0)
            if end is None:
                if dangling_iac:
                    self._pending_negotiation.append(IAC)
                return b""
            self._discarding_subnegotiation = False
            i = end

        while i < len(buffer):
            byte = buffer[i]
            if byte != IAC:
                out.append(byte)
                i += 1
                continue

            if i + 1 >= len(buffer):
                self._hold_negotiation(buffer[i:])
                return bytes(out)

            command = buffer[i + 1]

            if command == IAC:
                out.append(IAC)
                i += 2
                continue

            if command in (DO, DONT, WILL, WONT):
                if i + 2 >= len(buffer):
                    self._hold_negotiation(buffer[i:])
                    return bytes(out)
                option = buffer[i + 2]
                response = WONT if command in (DO, DONT) else DONT
                self._write(bytes([IAC, response, option]))
                i += 3
                continue

            if command == SB:
                end, dangling_iac = _subnegotiation_end(buffer, i + 2)
                if end is None:
                    if not self._hold_negotiation(buffer[i:]):
                        # Too large to buffer. Keep skipping the payload as it
                        # streams in; forgetting the state here would emit the
                        # rest of the subnegotiation as terminal content.
                        self._discarding_subnegotiation = True
                        if dangling_iac:
                            self._pending_negotiation.append(IAC)
                    return bytes(out)
                i = end
                continue

            i += 2

        return bytes(out)

    def _hold_negotiation(self, tail: bytes) -> bool:
        """Buffer an incomplete negotiation for the next read, if it is small."""

        if len(tail) > MAX_PENDING_NEGOTIATION_BYTES:
            return False
        self._pending_negotiation.extend(tail)
        return True


def _subnegotiation_end(buffer: bytes, start: int) -> tuple[int | None, bool]:
    """Locate the ``IAC SE`` that ends a subnegotiation.

    Returns the index just past ``IAC SE`` (or ``None`` if the terminator has
    not arrived) plus whether the scan ended on an unpaired trailing ``IAC``.
    Only an unpaired ``IAC`` may be held for the next read: re-holding the
    second byte of a consumed ``IAC IAC`` escape would let a following data
    byte of 0xF0 falsely terminate the subnegotiation.
    """

    i = start
    while i < len(buffer):
        if buffer[i] != IAC:
            i += 1
            continue
        if i + 1 >= len(buffer):
            return None, True
        if buffer[i + 1] == SE:
            return i + 2, False
        i += 2
    return None, False
