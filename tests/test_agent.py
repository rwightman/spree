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

    def send_text(self, text: str):
        self.sent.append(("send_text", text, None))

    def send_line(self, text: str = ""):
        self.sent.append(("send_line", text, None))

    def send_key(self, key: str):
        self.sent.append(("send_key", key, None))

    def send_bytes(self, payload: bytes):
        self.sent.append(("send_bytes", payload, None))

    def read(self, seconds: float = 1.0) -> bytes:
        del seconds
        return b""


def test_terminal_session_agent_dispatches_generic_actions():
    session = FakeSession()
    agent = TerminalSessionAgent("agent", session, TurnObserver("agent", session))

    agent.act_action(Action("wait"))
    agent.act_action(Action("send_line", text="look"))
    agent.act_action(Action("send_text", text="partial"))
    agent.act_action(Action("key", key="enter"))
    agent.act_action(Action("send_raw", text="x"))
    agent.act_action(Action("send_multiline", lines=("one", "two")))
    agent.act_action(Action("hangup"))

    assert session.sent == [
        ("send_line", "look", None),
        ("send_text", "partial", None),
        ("send_key", "enter", None),
        ("send_bytes", b"x", None),
        ("send_line", "one", None),
        ("send_line", "two", None),
    ]
    assert session.closed is True
