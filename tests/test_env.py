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
