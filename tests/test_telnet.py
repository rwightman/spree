import socket

import pytest

from tty_agent.ansi import strip_ansi
from tty_agent.transports.base import SessionDisconnected
from tty_agent.transports.telnet import DO, IAC, SB, SE, WILL, TelnetSession


class FakeSocket:
    def __init__(self, send_error=None):
        self.sent = bytearray()
        self.send_error = send_error

    def sendall(self, data):
        if self.send_error is not None:
            raise self.send_error
        self.sent.extend(data)


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
