from pathlib import Path

from bbs_gym.profiles import BBS_PROFILE
from tty_agent.profiles import PromptProfile
from tty_agent.terminal import TerminalScreen, TurnObserver


class FakeSession:
    transcript_path = Path("runtime/transcripts/fake.raw")
    encoding = "utf-8"

    def __init__(self, chunks, position: int = 0):
        self.chunks = list(chunks)
        self.sent: list[bytes] = []
        self.position = position

    def read(self, _seconds):
        if self.chunks:
            data = self.chunks.pop(0)
            self.position += len(data)
            return data
        return b""

    def send_bytes(self, payload: bytes) -> None:
        self.sent.append(payload)

    def transcript_position(self) -> int:
        return self.position


def test_terminal_screen_renders_cp437_and_ansi():
    terminal = TerminalScreen(columns=20, lines=4, encoding="cp437")

    changed = terminal.feed(b"\x1b[31mHi \xb1\x1b[0m\r\nCommand:")

    assert changed is True
    assert "Hi \u2592" in terminal.model_text()
    assert "Command:" in terminal.model_text()


def test_terminal_screen_captures_ansi_process_replies():
    terminal = TerminalScreen(columns=80, lines=24, encoding="cp437")

    terminal.feed(b"\x1b[0c\x1b[6n")

    assert terminal.drain_process_input() == (b"\x1b[?6c", b"\x1b[1;1R")
    assert terminal.drain_process_input() == ()


def test_prompt_profile_matches_screen_text():
    assert BBS_PROFILE.match("Main Menu\nCommand:") == "menu-choice"
    assert BBS_PROFILE.match("Nothing to do here") is None


def test_observe_turn_records_prompt_but_returns_on_stability_by_default():
    profile = PromptProfile.from_patterns("test", {"command": r"Command:\s*$"})
    observer = TurnObserver("agent-001", FakeSession([b"Welcome\r\nCommand:"]), profile=profile)

    observation = observer.observe_turn(timeout=1.0, stable_ms=0, byte_quiet_ms=0)

    assert observation.agent_id == "agent-001"
    assert observation.matched_prompt == "command"
    assert observation.ready_reason == "stable"
    assert observation.timed_out is False
    assert observation.bytes_read == len(b"Welcome\r\nCommand:")
    assert "Welcome" in observation.model_text


def test_observe_turn_records_transcript_byte_offsets():
    observer = TurnObserver("agent-001", FakeSession([b"Command:"], position=42), profile=PromptProfile("empty"))

    observation = observer.observe_turn(timeout=1.0, stable_ms=0, byte_quiet_ms=0)

    assert observation.transcript_byte_start == 42
    assert observation.transcript_byte_end == 42 + len(b"Command:")


def test_observe_turn_sends_ansi_process_replies():
    session = FakeSession([b"\x1b[0c\x1b[6nStatic screen"])
    observer = TurnObserver("agent-001", session, profile=PromptProfile("empty"))

    observer.observe_turn(timeout=1.0, stable_ms=0, byte_quiet_ms=0)

    assert session.sent == [b"\x1b[?6c", b"\x1b[1;1R"]


def test_observe_turn_can_use_prompt_fast_path_when_enabled():
    profile = PromptProfile.from_patterns("test", {"command": r"Command:\s*$"})
    observer = TurnObserver("agent-001", FakeSession([b"Welcome\r\nCommand:"]), profile=profile)

    observation = observer.observe_turn(timeout=1.0, stable_ms=500, byte_quiet_ms=500, prompt_fast_path=True)

    assert observation.matched_prompt == "command"
    assert observation.ready_reason == "prompt"
    assert observation.timed_out is False


def test_observe_turn_can_return_on_stability_without_prompt():
    observer = TurnObserver("agent-001", FakeSession([b"Static screen"]), profile=PromptProfile("empty"))

    observation = observer.observe_turn(timeout=1.0, stable_ms=0, byte_quiet_ms=0)

    assert observation.matched_prompt is None
    assert observation.ready_reason == "stable"
    assert observation.timed_out is False
    assert observation.model_text == "Static screen"


def test_observe_turn_can_disable_byte_quiet():
    observer = TurnObserver("agent-001", FakeSession([b"Command:", b"\x1b[0m"]), profile=PromptProfile("empty"))

    observation = observer.observe_turn(timeout=1.0, stable_ms=0, byte_quiet_ms=0)

    assert observation.ready_reason == "stable"
    assert observation.model_text == "Command:"
    assert observation.bytes_read == len(b"Command:")


def test_observe_turn_can_require_byte_quiet_after_invisible_bytes():
    observer = TurnObserver("agent-001", FakeSession([b"Command:", b"\x1b[0m"]), profile=PromptProfile("empty"))

    observation = observer.observe_turn(timeout=1.0, stable_ms=0, byte_quiet_ms=50)

    assert observation.ready_reason == "stable"
    assert observation.model_text == "Command:"
    assert observation.byte_quiet_ms >= 50
    assert observation.bytes_read == len(b"Command:\x1b[0m")
