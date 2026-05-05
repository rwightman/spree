"""Tiny multi-agent wrapper around terminal sessions."""

from __future__ import annotations

from pathlib import Path

from terminal_agent.agent import TerminalSessionAgent
from terminal_agent.terminal import TerminalScreen, TurnObserver
from terminal_agent.transports.telnet import TelnetSession

from .profiles import DEFAULT_PROFILE, PromptProfile


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
        self.agents: dict[str, TerminalSessionAgent] = {}

    def connect(self, agent_id: str, node: int | None = None) -> TerminalSessionAgent:
        transcript = self.transcript_dir / f"{agent_id}.raw"
        session = TelnetSession(self.host, self.port, transcript_path=transcript, encoding="cp437")
        session.connect()
        terminal = TerminalScreen(columns=self.columns, lines=self.lines, encoding=session.encoding)
        metadata = {
            "requested_node": node,
            "transport": "telnet",
            "host": self.host,
            "port": self.port,
            "encoding": session.encoding,
        }
        observer = TurnObserver(agent_id, session, terminal=terminal, profile=self.profile, metadata=metadata)
        agent = TerminalSessionAgent(agent_id, session, observer, metadata)
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
