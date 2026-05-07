from bbs_gym.accounts import AgentRecord, AgentRegistry
from bbs_gym.env import BbsGym


def test_bbs_gym_connect_uses_effective_model_metadata(monkeypatch, tmp_path):
    class FakeSession:
        encoding = "cp437"

        def __init__(self, host, port, transcript_path=None, encoding="cp437"):
            self.host = host
            self.port = port
            self.transcript_path = transcript_path
            self.encoding = encoding
            self.closed = False

        def connect(self):
            return None

        def close(self):
            self.closed = True

    monkeypatch.setattr("bbs_gym.env.TelnetSession", FakeSession)
    gym = BbsGym(transcript_dir=tmp_path, transport="telnet")

    agent = gym.connect(
        "codex-smoke",
        model_metadata={"provider": "codex", "model": "gpt-5.5", "sandbox": "read-only"},
    )

    assert agent.metadata["model"] == {"provider": "codex", "model": "gpt-5.5", "sandbox": "read-only"}
    assert agent.observer.metadata["model"] == {"provider": "codex", "model": "gpt-5.5", "sandbox": "read-only"}


def test_bbs_gym_telnet_uses_agent_registry_for_login(monkeypatch, tmp_path):
    class FakeSession:
        encoding = "cp437"

        def __init__(self, host, port, transcript_path=None, encoding="cp437"):
            self.host = host
            self.port = port
            self.transcript_path = transcript_path
            self.encoding = encoding
            self.closed = False
            self.chunks = [
                b"Synchronet BBS",
                b"Enter User Name or Number or 'New'\r\nLogin:",
                b"Password:",
                b"[Hit a key]",
            ]
            self.sent: list[bytes] = []
            self.position = 0

        def connect(self):
            return None

        def close(self):
            self.closed = True

        def read(self, _seconds):
            if self.chunks:
                data = self.chunks.pop(0)
                self.position += len(data)
                return data
            return b""

        def send_text(self, text):
            self.send_bytes(text.encode(self.encoding))

        def send_line(self, text=""):
            self.send_text(text)
            self.send_key("enter")

        def send_key(self, key):
            self.send_bytes(b"\r\n" if key == "enter" else key.encode(self.encoding))

        def send_bytes(self, payload):
            self.sent.append(bytes(payload))

        def drain_sent_bytes(self):
            chunks = tuple(self.sent)
            self.sent.clear()
            return chunks

        def transcript_position(self):
            return self.position

    monkeypatch.setattr("bbs_gym.env.TelnetSession", FakeSession)
    registry = AgentRegistry(
        {
            "agent-001": AgentRecord(
                agent_id="agent-001",
                bbs_alias="RLoginSmoke",
                bbs_password="rlogin-smoke-pass",
                model={"provider": "scripted"},
            )
        }
    )
    gym = BbsGym(transcript_dir=tmp_path, transport="telnet", agent_registry=registry)

    agent = gym.connect("agent-001")

    assert agent.metadata["authenticated"] is True
    assert agent.metadata["login_method"] == "telnet"
    assert agent.metadata["bbs_alias"] == "RLoginSmoke"
    assert "[Hit a key]" in agent.metadata["login_outcome"]["model_text_tail"]
    assert "[Hit a key]" in agent.observer.terminal.model_text()
    assert agent.session.drain_sent_bytes() == ()


def test_bbs_gym_telnet_login_handles_initial_password_prompt(monkeypatch, tmp_path):
    class FakeSession:
        encoding = "cp437"

        def __init__(self, host, port, transcript_path=None, encoding="cp437"):
            self.host = host
            self.port = port
            self.transcript_path = transcript_path
            self.encoding = encoding
            self.chunks = [b"Password:", b"[Hit a key]"]
            self.sent: list[bytes] = []
            self.position = 0

        def connect(self):
            return None

        def close(self):
            return None

        def read(self, _seconds):
            if self.chunks:
                data = self.chunks.pop(0)
                self.position += len(data)
                return data
            return b""

        def send_text(self, text):
            self.send_bytes(text.encode(self.encoding))

        def send_line(self, text=""):
            self.send_text(text)
            self.send_key("enter")

        def send_key(self, key):
            self.send_bytes(b"\r" if key == "enter" else key.encode(self.encoding))

        def send_bytes(self, payload):
            self.sent.append(bytes(payload))

        def drain_sent_bytes(self):
            chunks = tuple(self.sent)
            self.sent.clear()
            return chunks

        def transcript_position(self):
            return self.position

    monkeypatch.setattr("bbs_gym.env.TelnetSession", FakeSession)
    registry = AgentRegistry(
        {
            "agent-001": AgentRecord(
                agent_id="agent-001",
                bbs_alias="RLoginSmoke",
                bbs_password="rlogin-smoke-pass",
                model={"provider": "scripted"},
            )
        }
    )
    gym = BbsGym(transcript_dir=tmp_path, transport="telnet", agent_registry=registry)

    agent = gym.connect("agent-001")

    assert agent.metadata["authenticated"] is True
    assert "[Hit a key]" in agent.metadata["login_outcome"]["model_text_tail"]
    assert "[Hit a key]" in agent.observer.terminal.model_text()
