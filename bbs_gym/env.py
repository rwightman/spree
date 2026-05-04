"""Tiny multi-agent wrapper around terminal sessions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .actions import Action
from .ansi import strip_ansi
from .profiles import DEFAULT_PROFILE, PromptProfile
from .telnet import TelnetSession
from .terminal import Observation, TerminalScreen, TurnObserver


@dataclass
class AgentTerminal:
    agent_id: str
    node: int | None
    requested_node: int | None
    session: TelnetSession
    observer: TurnObserver

    def observe(self, seconds: float = 1.0, plain: bool = True) -> str | bytes:
        data = self.session.read(seconds)
        self.observer.feed(data)
        return strip_ansi(data) if plain else data

    def observe_turn(
        self,
        timeout: float = 10.0,
        stable_ms: int = 300,
        poll_interval: float = 0.05,
        profile: PromptProfile | None = None,
        prompt_fast_path: bool = False,
    ) -> Observation:
        return self.observer.observe_turn(timeout, stable_ms, poll_interval, profile, prompt_fast_path)

    def act(self, text: str, newline: bool = True) -> None:
        self.session.send(text, newline)

    def act_action(self, action: Action) -> None:
        if action.action == "wait":
            return
        if action.action == "hangup":
            self.close()
            return
        if action.action == "send":
            self.session.send(action.text, newline=action.newline)
            return
        if action.action == "send_raw":
            self.session.send_bytes(action.text.encode("cp437"))
            return
        if action.action == "send_multiline":
            for line in action.lines:
                self.session.send(line, newline=True)
            return
        raise ValueError(f"unsupported action {action.action!r}")

    def close(self) -> None:
        self.session.close()


class BbsGym:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 2323,
        transcript_dir: str | Path = "runtime/transcripts",
        profile: PromptProfile = DEFAULT_PROFILE,
        columns: int = 80,
        lines: int = 24,
    ) -> None:
        self.host = host
        self.port = port
        self.transcript_dir = Path(transcript_dir)
        self.profile = profile
        self.columns = columns
        self.lines = lines
        self.agents: dict[str, AgentTerminal] = {}

    def connect(self, agent_id: str, node: int | None = None) -> AgentTerminal:
        transcript = self.transcript_dir / f"{agent_id}.raw"
        session = TelnetSession(self.host, self.port, transcript_path=transcript)
        session.connect()
        terminal = TerminalScreen(columns=self.columns, lines=self.lines)
        observer = TurnObserver(agent_id, session, terminal=terminal, profile=self.profile, requested_node=node)
        agent = AgentTerminal(agent_id, None, node, session, observer)
        self.agents[agent_id] = agent
        return agent

    def close(self) -> None:
        for agent in list(self.agents.values()):
            agent.close()
        self.agents.clear()

    def __enter__(self) -> "BbsGym":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
