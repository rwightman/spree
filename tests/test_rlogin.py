from tty_agent.transports.base import SessionDisconnected
from tty_agent.transports.rlogin import RLoginSession


class FakeSocket:
    def __init__(self):
        self.sent = bytearray()
        self.closed = False

    def sendall(self, data):
        self.sent.extend(data)

    def close(self):
        self.closed = True


def test_rlogin_handshake_uses_password_as_client_user_by_default():
    session = RLoginSession(username="QwenOne", password="secret", terminal="ansi-bbs", speed=38400)

    assert session._handshake() == b"\x00secret\x00QwenOne\x00ansi-bbs/38400\x00"


def test_rlogin_handshake_can_reverse_user_and_password_fields():
    session = RLoginSession(
        username="QwenOne",
        password="secret",
        terminal="ansi-bbs",
        speed=38400,
        reversed_login=True,
    )

    assert session._handshake() == b"\x00QwenOne\x00secret\x00ansi-bbs/38400\x00"


def test_rlogin_strips_initial_ack_once():
    session = RLoginSession()

    assert session._strip_initial_ack(b"\x00hello") == b"hello"
    assert session._strip_initial_ack(b"\x00there") == b"\x00there"


def test_rlogin_enter_key_matches_empty_send_line():
    session = RLoginSession()
    session._sock = FakeSocket()

    session.send_line("")
    line_bytes = bytes(session._sock.sent)
    session._sock.sent.clear()
    session.send_key("enter")

    assert line_bytes == bytes(session._sock.sent) == b"\r"


def test_rlogin_key_accepts_printable_character():
    session = RLoginSession()
    session._sock = FakeSocket()

    session.send_key("q")

    assert bytes(session._sock.sent) == b"q"


def test_rlogin_drains_sent_byte_chunks():
    session = RLoginSession()
    session._sock = FakeSocket()

    session.send_line("20")

    assert session.drain_sent_bytes() == (b"20", b"\r")
    assert session.drain_sent_bytes() == ()


def test_rlogin_reports_transcript_position():
    session = RLoginSession()
    session._transcript.record(b"abc")

    assert session.transcript_position() == 3


def test_rlogin_send_timeout_reports_stall_not_close():
    import pytest

    from tty_agent.transports.base import SessionDisconnected

    class TimingOutSocket:
        def sendall(self, _data):
            raise TimeoutError("timed out")

    session = RLoginSession(timeout=3.0)
    session._sock = TimingOutSocket()

    with pytest.raises(SessionDisconnected) as excinfo:
        session.send_bytes(b"x")

    assert "stopped accepting data" in str(excinfo.value)


def test_rlogin_connect_resets_ack_state(monkeypatch):
    class FakeConnectedSocket:
        def sendall(self, _data):
            return None

        def close(self):
            return None

    monkeypatch.setattr(
        "tty_agent.transports.rlogin.socket.create_connection",
        lambda *_args, **_kwargs: FakeConnectedSocket(),
    )
    session = RLoginSession()
    session._ack_pending = False

    session.connect()

    assert session._ack_pending is True
    assert session._strip_initial_ack(b"\x00hello") == b"hello"


def test_rlogin_connect_closes_socket_when_handshake_fails(monkeypatch):
    import pytest

    class FailingSocket(FakeSocket):
        def sendall(self, _data):
            raise OSError("handshake failed")

    connected_socket = FailingSocket()
    monkeypatch.setattr(
        "tty_agent.transports.rlogin.socket.create_connection",
        lambda *_args, **_kwargs: connected_socket,
    )
    session = RLoginSession()

    with pytest.raises(SessionDisconnected, match="handshake failed"):
        session.connect()

    assert connected_socket.closed is True
    assert session._sock is None
