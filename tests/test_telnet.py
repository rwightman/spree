from bbs_gym.ansi import strip_ansi
from bbs_gym.telnet import DO, IAC, WILL, TelnetSession


class FakeSocket:
    def __init__(self):
        self.sent = bytearray()

    def sendall(self, data):
        self.sent.extend(data)


def test_strip_ansi_decodes_cp437():
    assert strip_ansi(b"\x1b[31mHi \xb1\x1b[0m") == "Hi \u2592"


def test_telnet_negotiation_is_removed_from_application_stream():
    session = TelnetSession()
    session._sock = FakeSocket()

    data = session._handle_telnet(bytes([IAC, DO, 1]) + b"Hello" + bytes([IAC, WILL, 3]))

    assert data == b"Hello"
    assert bytes(session._sock.sent) == bytes([IAC, 252, 1, IAC, 254, 3])

