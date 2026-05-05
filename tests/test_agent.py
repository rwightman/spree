from terminal_agent.actions import Action
from terminal_agent.agent import TerminalSessionAgent
from terminal_agent.terminal import TurnObserver


class FakeSession:
    transcript_path = None
    encoding = "utf-8"

    def __init__(self):
        self.sent: list[tuple[str, str | bytes, bool | None]] = []
        self.closed = False

    def connect(self):
        pass

    def close(self):
        self.closed = True

    def send(self, text: str, newline: bool = True):
        self.sent.append(("send", text, newline))

    def send_bytes(self, payload: bytes):
        self.sent.append(("send_bytes", payload, None))

    def read(self, seconds: float = 1.0) -> bytes:
        del seconds
        return b""


def test_terminal_session_agent_dispatches_generic_actions():
    session = FakeSession()
    agent = TerminalSessionAgent("agent", session, TurnObserver("agent", session))

    agent.act_action(Action("wait"))
    agent.act_action(Action("send", text="look", newline=False))
    agent.act_action(Action("send_raw", text="x"))
    agent.act_action(Action("send_multiline", lines=("one", "two")))
    agent.act_action(Action("hangup"))

    assert session.sent == [
        ("send", "look", False),
        ("send_bytes", b"x", None),
        ("send", "one", True),
        ("send", "two", True),
    ]
    assert session.closed is True
