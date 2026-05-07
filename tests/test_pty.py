import os
import sys

from terminal_agent.terminal import TurnObserver
from terminal_agent.transports.pty import PtySession


def test_pty_session_observes_local_process_turns():
    command = "import time; print('READY', flush=True); line=input(); print('ECHO:' + line, flush=True); time.sleep(1)"

    with PtySession([sys.executable, "-c", command]) as session:
        observer = TurnObserver("local", session)
        ready = observer.observe_turn(timeout=2.0, stable_ms=0)
        assert "READY" in ready.model_text

        session.send_line("hello")
        echoed = observer.observe_turn(timeout=2.0, stable_ms=50)
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
        first = observer.observe_turn(timeout=2.0, stable_ms=50)
        assert "TERM:xterm-256color" in first.model_text
        assert "SIZE1:100x31" in first.model_text

        session.resize(columns=120, lines=40)
        session.send_line("go")
        second = observer.observe_turn(timeout=2.0, stable_ms=50)
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
