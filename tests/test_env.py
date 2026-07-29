from bbs_gym.accounts import AgentRecord, AgentRegistry
from bbs_gym.env import BbsGym


def test_bbs_gym_connect_uses_effective_model_metadata(monkeypatch, tmp_path):
    class FakeSession:
        encoding = "cp437"

        def __init__(self, host, port, transcript_path=None, encoding="cp437", enter_sequence="cr"):
            self.host = host
            self.port = port
            self.transcript_path = transcript_path
            self.encoding = encoding
            self.enter_sequence = enter_sequence
            self.closed = False

        def connect(self):
            return None

        def close(self):
            self.closed = True

    monkeypatch.setattr("bbs_gym.env.TelnetSession", FakeSession)
    gym = BbsGym(transcript_dir=tmp_path, transport="telnet")

    model_metadata = {
        "provider": "codex",
        "model": "gpt-5.5",
        "sandbox": "read-only",
        "fallbacks": [{"api_key": "do-not-log"}],
    }
    agent = gym.connect(
        "codex-smoke",
        model_metadata=model_metadata,
    )

    expected = {
        "provider": "codex",
        "model": "gpt-5.5",
        "sandbox": "read-only",
        "fallbacks": [{"api_key": "[redacted]"}],
    }
    assert agent.metadata["model"] == expected
    assert agent.observer.metadata["model"] == expected
    assert model_metadata["fallbacks"][0]["api_key"] == "do-not-log"


def test_bbs_gym_telnet_enter_sequence_is_forwarded(monkeypatch, tmp_path):
    class FakeSession:
        encoding = "cp437"

        def __init__(self, host, port, transcript_path=None, encoding="cp437", enter_sequence="cr"):
            self.host = host
            self.port = port
            self.transcript_path = transcript_path
            self.encoding = encoding
            self.enter_sequence = enter_sequence

        def connect(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr("bbs_gym.env.TelnetSession", FakeSession)
    gym = BbsGym(transcript_dir=tmp_path, transport="telnet", telnet_enter_sequence="lf")

    agent = gym.connect("tele-arena-codex")

    assert agent.session.enter_sequence == "lf"
    assert agent.metadata["telnet_enter_sequence"] == "lf"
    assert agent.observer.metadata["telnet_enter_sequence"] == "lf"


def test_bbs_gym_rlogin_finishes_terminal_setup_before_returning(monkeypatch, tmp_path):
    class FakeSession:
        encoding = "cp437"

        def __init__(
                self,
                host,
                port,
                username,
                password,
                transcript_path=None,
                encoding="cp437",
                terminal="ansi",
        ):
            self.host = host
            self.port = port
            self.username = username
            self.password = password
            self.transcript_path = transcript_path
            self.encoding = encoding
            self.terminal = terminal
            self.position = 0
            self.sent: list[bytes] = []
            self.chunks = [
                (
                    b"\x1b[s\x1b[0c\x1b[255B\x1b[255C\x1b[30;40m\b_\x1b[6n"
                    b"\x1b[u\x1b[!_\r\xef\xbb\xbf\x1b[6n\x1b[0m\x1b[2J\x1b[H\x0c\r"
                ),
                b"Logging on\r\n[Hit a key]",
            ]

        def connect(self):
            return None

        def close(self):
            return None

        def read(self, _seconds):
            if not self.chunks:
                return b""
            data = self.chunks.pop(0)
            self.position += len(data)
            return data

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

    monkeypatch.setattr("bbs_gym.env.RLoginSession", FakeSession)
    monkeypatch.setattr("bbs_gym.env._RLOGIN_LOGIN_STABLE_MS", 0)
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
    gym = BbsGym(transcript_dir=tmp_path, transport="rlogin", agent_registry=registry)

    agent = gym.connect("agent-001")

    assert agent.metadata["authenticated"] is True
    assert agent.metadata["login_method"] == "rlogin"
    assert "[Hit a key]" in agent.metadata["login_outcome"]["model_text_tail"]
    assert "[Hit a key]" in agent.observer.terminal.model_text()
    # ANSI device/cursor reports are protocol traffic, not agent actions.
    assert agent.session.drain_sent_bytes() == ()


def test_bbs_gym_telnet_uses_agent_registry_for_login(monkeypatch, tmp_path):
    class FakeSession:
        encoding = "cp437"

        def __init__(self, host, port, transcript_path=None, encoding="cp437", enter_sequence="cr"):
            self.host = host
            self.port = port
            self.transcript_path = transcript_path
            self.encoding = encoding
            self.enter_sequence = enter_sequence
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
                model={"provider": "scripted", "api_key": "sk-inline-secret"},
            )
        }
    )
    gym = BbsGym(transcript_dir=tmp_path, transport="telnet", agent_registry=registry)

    agent = gym.connect("agent-001")

    assert agent.metadata["authenticated"] is True
    assert agent.metadata["login_method"] == "telnet"
    assert agent.metadata["bbs_alias"] == "RLoginSmoke"
    # Registry model config flows into observation metadata; credentials must not.
    assert agent.metadata["model"] == {"provider": "scripted", "api_key": "[redacted]"}
    assert "[Hit a key]" in agent.metadata["login_outcome"]["model_text_tail"]
    assert "[Hit a key]" in agent.observer.terminal.model_text()
    assert agent.session.drain_sent_bytes() == ()


def test_bbs_gym_telnet_login_handles_initial_password_prompt(monkeypatch, tmp_path):
    class FakeSession:
        encoding = "cp437"

        def __init__(self, host, port, transcript_path=None, encoding="cp437", enter_sequence="cr"):
            self.host = host
            self.port = port
            self.transcript_path = transcript_path
            self.encoding = encoding
            self.enter_sequence = enter_sequence
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


def test_bbs_gym_closes_session_when_login_fails(monkeypatch, tmp_path):
    from tty_agent.transports.base import SessionDisconnected

    import pytest

    class FakeSession:
        encoding = "cp437"
        instances: list["FakeSession"] = []

        def __init__(self, host, port, transcript_path=None, encoding="cp437", enter_sequence="cr"):
            self.host = host
            self.port = port
            self.transcript_path = transcript_path
            self.encoding = encoding
            self.enter_sequence = enter_sequence
            self.closed = False
            FakeSession.instances.append(self)

        def connect(self):
            return None

        def close(self):
            self.closed = True

        def read(self, _seconds):
            raise SessionDisconnected("remote terminal connection closed")

        def send_text(self, text):
            return None

        def send_line(self, text=""):
            return None

        def send_key(self, key):
            return None

        def send_bytes(self, payload):
            return None

        def drain_sent_bytes(self):
            return ()

        def transcript_position(self):
            return 0

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

    with pytest.raises(SessionDisconnected):
        gym.connect("agent-001")

    assert len(FakeSession.instances) == 1
    assert FakeSession.instances[0].closed is True
    assert gym.agents == {}


def _leakcheck_session_class():
    class FakeSession:
        encoding = "cp437"
        instances: list = []

        def __init__(self, host, port, transcript_path=None, encoding="cp437", enter_sequence="cr"):
            self.host = host
            self.port = port
            self.transcript_path = transcript_path
            self.encoding = encoding
            self.enter_sequence = enter_sequence
            self.closed = False
            type(self).instances.append(self)

        def connect(self):
            return None

        def close(self):
            self.closed = True

    return FakeSession


def test_bbs_gym_rejects_path_traversal_agent_id(monkeypatch, tmp_path):
    import pytest

    monkeypatch.setattr("bbs_gym.env.TelnetSession", _leakcheck_session_class())
    gym = BbsGym(transcript_dir=tmp_path, transport="telnet")

    with pytest.raises(ValueError):
        gym.connect("../escape")

    assert gym.agents == {}


def test_bbs_gym_closes_session_when_connect_fails(monkeypatch, tmp_path):
    import pytest

    session_class = _leakcheck_session_class()

    def failing_connect(self):
        raise RuntimeError("transcript open failed")

    session_class.connect = failing_connect
    monkeypatch.setattr("bbs_gym.env.TelnetSession", session_class)
    gym = BbsGym(transcript_dir=tmp_path, transport="telnet")

    with pytest.raises(RuntimeError):
        gym.connect("agent-001")

    assert session_class.instances[0].closed is True
    assert gym.agents == {}


def test_bbs_gym_reconnect_closes_previous_registration(monkeypatch, tmp_path):
    session_class = _leakcheck_session_class()
    monkeypatch.setattr("bbs_gym.env.TelnetSession", session_class)
    gym = BbsGym(transcript_dir=tmp_path, transport="telnet")

    first = gym.connect("agent-001")
    second = gym.connect("agent-001")

    assert first.session.closed is True
    assert second.session.closed is False
    assert gym.agents["agent-001"] is second


def test_bbs_gym_run_id_separates_runtime_artifacts_from_account(monkeypatch, tmp_path):
    from types import SimpleNamespace

    session_class = _leakcheck_session_class()
    monkeypatch.setattr("bbs_gym.env.TelnetSession", session_class)
    monkeypatch.setattr(
        BbsGym,
        "_login_telnet",
        lambda *_args, **_kwargs: SimpleNamespace(
            ready_reason="stable",
            stable_ms=300,
            byte_quiet_ms=0,
            model_text="logged in",
        ),
    )
    registry = AgentRegistry(
        {
            "shared-claude-account": AgentRecord(
                agent_id="shared-claude-account",
                bbs_alias="Claude",
                bbs_password="bot-password",
                model={"provider": "scripted"},
            )
        }
    )
    gym = BbsGym(transcript_dir=tmp_path, transport="telnet", agent_registry=registry)

    agent = gym.connect("shared-claude-account", run_id="sre-claude-run1")

    assert agent.agent_id == "sre-claude-run1"
    assert agent.observer.agent_id == "sre-claude-run1"
    assert agent.session.transcript_path == tmp_path / "sre-claude-run1.raw"
    assert agent.metadata["account_agent_id"] == "shared-claude-account"
    assert agent.metadata["bbs_alias"] == "Claude"
    assert gym.agents["shared-claude-account"] is agent


def test_bbs_gym_rejects_path_traversal_run_id(monkeypatch, tmp_path):
    monkeypatch.setattr("bbs_gym.env.TelnetSession", _leakcheck_session_class())
    gym = BbsGym(transcript_dir=tmp_path, transport="telnet")

    import pytest

    with pytest.raises(ValueError):
        gym.connect("agent-001", run_id="../escape")

    assert gym.agents == {}
