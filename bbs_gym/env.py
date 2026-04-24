"""Tiny multi-agent wrapper around terminal sessions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .ansi import strip_ansi
from .telnet import TelnetSession


@dataclass
class AgentTerminal:
    agent_id: str
    session: TelnetSession

    def observe(self, seconds: float = 1.0, plain: bool = True) -> str | bytes:
        data = self.session.read(seconds)
        return strip_ansi(data) if plain else data

    def act(self, text: str, newline: bool = True) -> None:
        self.session.send(text, newline)

    def close(self) -> None:
        self.session.close()


class BbsGym:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 2323,
        transcript_dir: str | Path = "runtime/transcripts",
    ) -> None:
        self.host = host
        self.port = port
        self.transcript_dir = Path(transcript_dir)
        self.agents: dict[str, AgentTerminal] = {}

    def connect(self, agent_id: str) -> AgentTerminal:
        transcript = self.transcript_dir / f"{agent_id}.raw"
        session = TelnetSession(self.host, self.port, transcript_path=transcript)
        session.connect()
        agent = AgentTerminal(agent_id, session)
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

