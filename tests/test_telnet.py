import socket

import pytest

from tty_agent.ansi import strip_ansi
from tty_agent.transports.base import SessionDisconnected
from tty_agent.transports.telnet import DO, IAC, SB, SE, WILL, TelnetSession


class FakeSocket:
    def __init__(self, send_error=None):
        self.sent = bytearray()
        self.send_error = send_error
        self.closed = False

    def sendall(self, data):
        if self.send_error is not None:
            raise self.send_error
        self.sent.extend(data)

    def close(self):
        self.closed = True


def test_strip_ansi_decodes_cp437():
    assert strip_ansi(b"\x1b[31mHi \xb1\x1b[0m", encoding="cp437") == "Hi \u2592"


def test_telnet_negotiation_is_removed_from_application_stream():
    session = TelnetSession()
    session._sock = FakeSocket()

    data = session._handle_telnet(bytes([IAC, DO, 1]) + b"Hello" + bytes([IAC, WILL, 3]))

    assert data == b"Hello"
    assert bytes(session._sock.sent) == bytes([IAC, 252, 1, IAC, 254, 3])


def test_telnet_enter_key_matches_empty_send_line():
    session = TelnetSession()
    session._sock = FakeSocket()

    session.send_line("")
    line_bytes = bytes(session._sock.sent)
    session._sock.sent.clear()
    session.send_key("enter")

    assert line_bytes == bytes(session._sock.sent) == b"\r"


def test_telnet_enter_sequence_can_use_lf():
    session = TelnetSession(enter_sequence="lf")
    session._sock = FakeSocket()

    session.send_key("enter")

    assert bytes(session._sock.sent) == b"\n"


def test_telnet_enter_sequence_can_use_crlf():
    session = TelnetSession(enter_sequence="crlf")
    session._sock = FakeSocket()

    session.send_key("enter")

    assert bytes(session._sock.sent) == b"\r\n"


def test_telnet_key_accepts_printable_character():
    session = TelnetSession()
    session._sock = FakeSocket()

    session.send_key("q")

    assert bytes(session._sock.sent) == b"q"


def test_telnet_drains_sent_byte_chunks():
    session = TelnetSession()
    session._sock = FakeSocket()

    session.send_line("20")

    assert session.drain_sent_bytes() == (b"20", b"\r")
    assert session.drain_sent_bytes() == ()


def test_telnet_reports_transcript_position():
    session = TelnetSession()
    session._transcript.record(b"abc")

    assert session.transcript_position() == 3


def test_telnet_negotiation_split_across_reads_is_still_stripped():
    session = TelnetSession()
    session._sock = FakeSocket()

    first = session._handle_telnet(b"HELLO" + bytes([IAC]))
    second = session._handle_telnet(bytes([DO, 1]) + b"WORLD")

    assert first + second == b"HELLOWORLD"
    assert bytes(session._sock.sent) == bytes([IAC, 252, 1])


def test_telnet_command_split_before_option_is_still_stripped():
    session = TelnetSession()
    session._sock = FakeSocket()

    first = session._handle_telnet(b"A" + bytes([IAC, WILL]))
    second = session._handle_telnet(bytes([3]) + b"B")

    assert first + second == b"AB"
    assert bytes(session._sock.sent) == bytes([IAC, 254, 3])


def test_telnet_subnegotiation_split_across_reads_is_still_stripped():
    session = TelnetSession()
    session._sock = FakeSocket()

    first = session._handle_telnet(b"A" + bytes([IAC, SB, 24, 0]))
    second = session._handle_telnet(b"vt100" + bytes([IAC, SE]) + b"B")

    assert first + second == b"AB"


def test_telnet_drops_unterminated_negotiation_past_the_cap():
    session = TelnetSession()
    session._sock = FakeSocket()

    session._handle_telnet(bytes([IAC, SB]) + b"x" * 5000)

    assert session._pending_negotiation == bytearray()


def test_telnet_oversized_subnegotiation_does_not_leak_into_application_data():
    """Past the buffering cap the payload must still be skipped, not emitted."""

    session = TelnetSession()
    session._sock = FakeSocket()

    first = session._handle_telnet(bytes([IAC, SB]) + b"x" * 5000)
    second = session._handle_telnet(b"SECRET" + bytes([IAC, SE]) + b"APP")

    assert first == b""
    assert second == b"APP"
    assert session._discarding_subnegotiation is False


def test_telnet_oversized_subnegotiation_terminator_split_across_reads():
    session = TelnetSession()
    session._sock = FakeSocket()

    session._handle_telnet(bytes([IAC, SB]) + b"x" * 5000)
    split = session._handle_telnet(b"SECRET" + bytes([IAC]))
    resumed = session._handle_telnet(bytes([SE]) + b"APP")

    assert split == b""
    assert resumed == b"APP"


def test_telnet_negotiation_reply_failure_reports_disconnect():
    """Negotiation replies must honour the disconnect contract like agent sends."""

    session = TelnetSession()
    session._sock = FakeSocket(send_error=ConnectionResetError(104, "Connection reset by peer"))

    with pytest.raises(SessionDisconnected):
        session._handle_telnet(bytes([IAC, DO, 1]))


def test_telnet_send_after_peer_hangup_reports_disconnect():
    session = TelnetSession()
    session._sock = FakeSocket(send_error=ConnectionResetError(104, "Connection reset by peer"))

    with pytest.raises(SessionDisconnected):
        session.send_line("look")


def test_telnet_read_reports_disconnect_when_peer_resets():
    """An RST from the peer must not leak a raw ConnectionResetError."""

    readable, peer = socket.socketpair()
    peer.sendall(b"x")  # make select report the socket ready

    class ResettingSocket:
        def fileno(self):
            return readable.fileno()

        def recv(self, _size):
            raise ConnectionResetError(104, "Connection reset by peer")

    session = TelnetSession()
    session._sock = ResettingSocket()
    try:
        with pytest.raises(SessionDisconnected):
            session.read(0.05)
    finally:
        readable.close()
        peer.close()


def test_telnet_send_without_connection_reports_disconnect():
    session = TelnetSession()

    with pytest.raises(SessionDisconnected):
        session.send_key("enter")


def test_telnet_transcript_appends_across_reconnects(tmp_path):
    transcript = tmp_path / "agent-001.raw"

    first = TelnetSession(transcript_path=transcript)
    first._transcript.open()
    first._transcript.record(b"FIRST")
    assert transcript.read_bytes() == b"FIRST"
    first.close()

    second = TelnetSession(transcript_path=transcript)
    second._transcript.open()
    assert second.transcript_position() == 5
    second._transcript.record(b"SECOND")
    second.close()

    assert transcript.read_bytes() == b"FIRSTSECOND"


def test_telnet_escapes_iac_in_sent_application_bytes():
    session = TelnetSession(encoding="cp437")
    session._sock = FakeSocket()

    session.send_text("\xa0")  # U+00A0 encodes to 0xFF in cp437

    assert bytes(session._sock.sent) == bytes([IAC, IAC])
    # The action trace records what the agent sent, not the wire escaping.
    assert session.drain_sent_bytes() == (b"\xff",)


def test_telnet_send_timeout_reports_stall_not_close():
    session = TelnetSession(timeout=3.0)
    session._sock = FakeSocket(send_error=TimeoutError("timed out"))

    with pytest.raises(SessionDisconnected) as excinfo:
        session.send_bytes(b"x")

    assert "stopped accepting data" in str(excinfo.value)


def test_telnet_discard_mode_handles_escaped_iac_at_chunk_boundary():
    session = TelnetSession()
    session._sock = FakeSocket()

    # An oversized subnegotiation switches the parser into discard mode.
    oversized = bytes([IAC, SB, 1]) + b"x" * 5000
    assert session._handle_telnet(oversized) == b""
    assert session._discarding_subnegotiation is True

    # A chunk ending in an escaped IAC IAC pair must not re-hold the second
    # byte: the following 0xF0 payload byte is data, not SE.
    assert session._handle_telnet(b"y" * 10 + bytes([IAC, IAC])) == b""
    assert session._handle_telnet(bytes([SE]) + b"still-subnegotiation") == b""

    # Only the real IAC SE ends the discard.
    assert session._handle_telnet(bytes([IAC, SE]) + b"APP") == b"APP"
    assert session._discarding_subnegotiation is False


def test_telnet_discard_mode_still_matches_split_terminator():
    session = TelnetSession()
    session._sock = FakeSocket()

    oversized = bytes([IAC, SB, 1]) + b"x" * 5000
    assert session._handle_telnet(oversized) == b""

    # A lone trailing IAC is genuinely half of IAC SE and must be held.
    assert session._handle_telnet(b"tail" + bytes([IAC])) == b""
    assert session._handle_telnet(bytes([SE]) + b"APP") == b"APP"
    assert session._discarding_subnegotiation is False


def test_telnet_connect_resets_protocol_state(monkeypatch):
    class FakeConnectedSocket:
        def sendall(self, _data):
            return None

        def close(self):
            return None

    monkeypatch.setattr(
        "tty_agent.transports.telnet.socket.create_connection",
        lambda *_args, **_kwargs: FakeConnectedSocket(),
    )
    session = TelnetSession()
    session._pending_negotiation.extend(bytes([IAC]))
    session._discarding_subnegotiation = True

    session.connect()

    assert bytes(session._pending_negotiation) == b""
    assert session._discarding_subnegotiation is False


def test_telnet_connect_closes_socket_when_transcript_setup_fails(monkeypatch):
    connected_socket = FakeSocket()
    session = TelnetSession()

    def fail_open():
        raise OSError("transcript setup failed")

    monkeypatch.setattr(
        "tty_agent.transports.telnet.socket.create_connection",
        lambda *_args, **_kwargs: connected_socket,
    )
    monkeypatch.setattr(session._transcript, "open", fail_open)

    with pytest.raises(OSError, match="transcript setup failed"):
        session.connect()

    assert connected_socket.closed is True
    assert session._sock is None
