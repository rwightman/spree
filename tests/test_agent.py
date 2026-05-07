from tty_agent.actions import Action
from tty_agent.agent import ActionExecution, TerminalSessionAgent
from tty_agent.terminal import TurnObserver


class FakeSession:
    transcript_path = None
    encoding = "utf-8"

    def __init__(self):
        self.sent: list[tuple[str, str | bytes, bool | None]] = []
        self.sent_bytes: list[bytes] = []
        self.closed = False

    def connect(self):
        pass

    def close(self):
        self.closed = True

    def send_text(self, text: str):
        self.sent.append(("send_text", text, None))
        self.sent_bytes.append(text.encode(self.encoding))

    def send_line(self, text: str = ""):
        self.sent.append(("send_line", text, None))
        self.sent_bytes.extend([text.encode(self.encoding), b"\n"])

    def send_key(self, key: str):
        self.sent.append(("send_key", key, None))
        self.sent_bytes.append(b"\n" if key == "enter" else key.encode(self.encoding))

    def send_bytes(self, payload: bytes):
        self.sent.append(("send_bytes", payload, None))
        self.sent_bytes.append(payload)

    def drain_sent_bytes(self) -> tuple[bytes, ...]:
        chunks = tuple(self.sent_bytes)
        self.sent_bytes.clear()
        return chunks

    def read(self, seconds: float = 1.0) -> bytes:
        del seconds
        return b""


def test_terminal_session_agent_dispatches_generic_actions():
    session = FakeSession()
    agent = TerminalSessionAgent("agent", session, TurnObserver("agent", session))

    assert agent.act_action(Action("wait")) == ActionExecution(sent_bytes=(), encoding="utf-8")
    submit_execution = agent.act_action(Action("submit_line", text="look"))
    agent.act_action(Action("type_text", text="partial"))
    agent.act_action(Action("press_key", key="enter"))
    agent.act_action(Action("send_raw", text="x"))
    agent.act_action(Action("submit_lines", lines=("one", "two")))
    agent.act_action(Action("hangup"))

    assert session.sent == [
        ("send_line", "look", None),
        ("send_text", "partial", None),
        ("send_key", "enter", None),
        ("send_bytes", b"x", None),
        ("send_line", "one", None),
        ("send_line", "two", None),
    ]
    assert submit_execution.sent_bytes == (b"look", b"\n")
    assert session.closed is True
