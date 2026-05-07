from tty_agent.ansi import strip_ansi
from tty_agent.transports.telnet import DO, IAC, WILL, TelnetSession


class FakeSocket:
    def __init__(self):
        self.sent = bytearray()

    def sendall(self, data):
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
    session._transcript.extend(b"abc")

    assert session.transcript_position() == 3
