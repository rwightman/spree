from terminal_agent.transports.rlogin import RLoginSession


class FakeSocket:
    def __init__(self):
        self.sent = bytearray()

    def sendall(self, data):
        self.sent.extend(data)


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
