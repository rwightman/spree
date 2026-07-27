import os
import sys

from tty_agent.terminal import TurnObserver
from tty_agent.transports.pty import PtySession


def test_pty_session_observes_local_process_turns():
    command = "import time; print('READY', flush=True); line=input(); print('ECHO:' + line, flush=True); time.sleep(1)"

    with PtySession([sys.executable, "-c", command]) as session:
        observer = TurnObserver("local", session)
        ready = observer.observe_turn(timeout=2.0, stable_ms=0, byte_quiet_ms=0)
        assert "READY" in ready.model_text

        session.send_line("hello")
        echoed = observer.observe_turn(timeout=2.0, stable_ms=50, byte_quiet_ms=50)
        assert "ECHO:hello" in echoed.model_text


def test_pty_session_sets_size_term_and_resizes():
    command = (
        "import os, time; "
        "first = os.get_terminal_size(0); "
        'print(f\'TERM:{os.environ.get("TERM", "")}\', flush=True); '
        "print(f'SIZE1:{first.columns}x{first.lines}', flush=True); "
        "input(); "
        "second = os.get_terminal_size(0); "
        "print(f'SIZE2:{second.columns}x{second.lines}', flush=True); "
        "time.sleep(1)"
    )

    with PtySession(
        [sys.executable, "-c", command],
        env={"PATH": os.environ["PATH"]},
        columns=100,
        lines=31,
    ) as session:
        observer = TurnObserver("local", session)
        first = observer.observe_turn(timeout=2.0, stable_ms=50, byte_quiet_ms=50)
        assert "TERM:xterm-256color" in first.model_text
        assert "SIZE1:100x31" in first.model_text

        session.resize(columns=120, lines=40)
        session.send_line("go")
        second = observer.observe_turn(timeout=2.0, stable_ms=50, byte_quiet_ms=50)
        assert "SIZE2:120x40" in second.model_text


def test_pty_enter_key_matches_empty_send_line():
    session = PtySession([sys.executable, "-c", ""])
    sent: list[bytes] = []
    session.send_bytes = sent.append  # type: ignore[method-assign]

    session.send_line("")
    line_bytes = sent[-1]
    session.send_key("enter")

    assert line_bytes == sent[-1] == b"\n"


def test_pty_key_accepts_printable_character():
    session = PtySession([sys.executable, "-c", ""])
    sent: list[bytes] = []
    session.send_bytes = sent.append  # type: ignore[method-assign]

    session.send_key("q")

    assert sent[-1] == b"q"


def test_pty_drains_sent_byte_chunks():
    session = PtySession([sys.executable, "-c", ""])
    sent: list[bytes] = []

    def send_bytes(payload: bytes) -> None:
        sent.append(payload)
        session._sent_bytes.append(payload)

    session.send_bytes = send_bytes  # type: ignore[method-assign]

    session.send_line("20")

    assert sent == [b"20", b"\n"]
    assert session.drain_sent_bytes() == (b"20", b"\n")
    assert session.drain_sent_bytes() == ()


def test_pty_reports_transcript_position():
    session = PtySession([sys.executable, "-c", ""])
    session._transcript.record(b"abc")

    assert session.transcript_position() == 3


def test_pty_send_retries_partial_writes(monkeypatch):
    import types

    chunks: list[bytes] = []
    behavior = iter([2, None, 3])  # short write, EAGAIN, remainder

    def fake_write(_fd, view):
        step = next(behavior)
        if step is None:
            raise BlockingIOError
        chunks.append(bytes(view[:step]))
        return step

    session = PtySession([sys.executable, "-c", ""], timeout=1.0)
    session._master_fd = 99
    monkeypatch.setattr("tty_agent.transports.pty.os", types.SimpleNamespace(write=fake_write))
    monkeypatch.setattr(
        "tty_agent.transports.pty.select",
        types.SimpleNamespace(select=lambda *_args: ([], [99], [])),
    )

    session.send_bytes(b"hello")

    assert chunks == [b"he", b"llo"]
    assert session.drain_sent_bytes() == (b"hello",)


def test_pty_send_reports_stalled_input_queue(monkeypatch):
    import types

    import pytest

    from tty_agent.transports.base import SessionDisconnected

    def always_blocked(_fd, _view):
        raise BlockingIOError

    session = PtySession([sys.executable, "-c", ""], timeout=0.05)
    session._master_fd = 99
    monkeypatch.setattr("tty_agent.transports.pty.os", types.SimpleNamespace(write=always_blocked))
    monkeypatch.setattr(
        "tty_agent.transports.pty.select",
        types.SimpleNamespace(select=lambda *_args: ([], [], [])),
    )

    with pytest.raises(SessionDisconnected) as excinfo:
        session.send_bytes(b"hello")

    assert "stayed full" in str(excinfo.value)
    assert session.drain_sent_bytes() == ()


def test_pty_connect_failure_does_not_leak_fds():
    import pytest

    session = PtySession(["/nonexistent-binary-for-fd-leak-test"])
    before = len(os.listdir("/proc/self/fd"))

    for _ in range(3):
        with pytest.raises(FileNotFoundError):
            session.connect()

    assert len(os.listdir("/proc/self/fd")) == before
    assert session._master_fd is None


def test_pty_connect_reaps_process_when_transcript_setup_fails(monkeypatch):
    import pytest

    session = PtySession([sys.executable, "-c", "import time; time.sleep(10)"])

    def fail_open():
        raise OSError("transcript setup failed")

    monkeypatch.setattr(session._transcript, "open", fail_open)

    with pytest.raises(OSError, match="transcript setup failed"):
        session.connect()

    assert session._master_fd is None
    assert session._proc is not None
    assert session._proc.poll() is not None
