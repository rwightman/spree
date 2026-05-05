"""PTY-backed local process transport."""

from __future__ import annotations

import errno
import fcntl
import os
import pty
import select
import struct
import subprocess
import termios
import time
from dataclasses import dataclass, field
from pathlib import Path

from .base import SessionDisconnected


@dataclass
class PtySession:
    argv: list[str]
    cwd: str | Path | None = None
    env: dict[str, str] | None = None
    encoding: str = "utf-8"
    transcript_path: Path | None = None
    timeout: float = 10.0
    columns: int = 80
    lines: int = 24
    _master_fd: int | None = field(default=None, init=False, repr=False)
    _proc: subprocess.Popen[bytes] | None = field(default=None, init=False, repr=False)
    _transcript: bytearray = field(default_factory=bytearray, init=False, repr=False)

    def connect(self) -> None:
        master_fd, slave_fd = pty.openpty()
        try:
            self._set_window_size(slave_fd)
            self._proc = subprocess.Popen(
                self.argv,
                cwd=self.cwd,
                env=self._child_env(),
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                close_fds=True,
                preexec_fn=lambda: self._prepare_child_terminal(slave_fd),
            )
        finally:
            os.close(slave_fd)
        self._master_fd = master_fd
        os.set_blocking(master_fd, False)

    def resize(self, columns: int, lines: int) -> None:
        if columns <= 0 or lines <= 0:
            raise ValueError("columns and lines must be positive")
        self.columns = columns
        self.lines = lines
        if self._master_fd is not None:
            self._set_window_size(self._master_fd)

    def close(self) -> None:
        if self._master_fd is not None:
            os.close(self._master_fd)
            self._master_fd = None
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=1)
        if self.transcript_path is not None:
            self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
            self.transcript_path.write_bytes(bytes(self._transcript))

    def __enter__(self) -> "PtySession":
        self.connect()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def send(self, text: str, newline: bool = True) -> None:
        payload = text.encode(self.encoding)
        if newline:
            payload += b"\n"
        self.send_bytes(payload)

    def send_bytes(self, payload: bytes) -> None:
        if self._master_fd is None:
            raise RuntimeError("session is not connected")
        os.write(self._master_fd, payload)

    def read(self, seconds: float = 1.0) -> bytes:
        if self._master_fd is None:
            raise RuntimeError("session is not connected")

        deadline = time.monotonic() + seconds
        out = bytearray()

        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            ready, _, _ = select.select([self._master_fd], [], [], min(0.2, remaining))
            if not ready:
                if self._proc is not None and self._proc.poll() is not None and not out:
                    raise SessionDisconnected("local PTY process exited")
                continue
            try:
                chunk = os.read(self._master_fd, 4096)
            except OSError as exc:
                if exc.errno == errno.EIO:
                    if not out:
                        raise SessionDisconnected("local PTY process exited") from exc
                    break
                raise
            if not chunk:
                if not out:
                    raise SessionDisconnected("local PTY process exited")
                break
            out.extend(chunk)
            self._transcript.extend(chunk)

        return bytes(out)

    def _child_env(self) -> dict[str, str]:
        env = dict(os.environ) if self.env is None else dict(self.env)
        env.setdefault("TERM", "xterm-256color")
        return env

    def _set_window_size(self, fd: int) -> None:
        if self.columns <= 0 or self.lines <= 0:
            raise ValueError("columns and lines must be positive")
        payload = struct.pack("HHHH", self.lines, self.columns, 0, 0)
        fcntl.ioctl(fd, termios.TIOCSWINSZ, payload)

    def _prepare_child_terminal(self, slave_fd: int) -> None:
        os.setsid()
        fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)
