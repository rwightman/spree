"""Tiny multi-agent wrapper around terminal sessions."""

from __future__ import annotations

from pathlib import Path

from terminal_agent.agent import TerminalSessionAgent
from terminal_agent.terminal import TerminalScreen, TurnObserver
from terminal_agent.transports.rlogin import RLoginSession
from terminal_agent.transports.telnet import TelnetSession

from .accounts import AgentRegistry, AccountConfigError, load_agent_registry
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
            rlogin_port: int = 2513,
            transport: str = "telnet",
            agent_registry: AgentRegistry | None = None,
            agent_registry_path: str | Path | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.rlogin_port = rlogin_port
        self.transcript_dir = Path(transcript_dir)
        self.profile = profile
        self.columns = columns
        self.lines = lines
        self.transport = transport
        self.agent_registry = agent_registry or load_agent_registry(agent_registry_path, required=False)
        self.agents: dict[str, TerminalSessionAgent] = {}

    def connect(
            self,
            agent_id: str,
            node: int | None = None,
            transport: str | None = None,
    ) -> TerminalSessionAgent:
        active_transport = transport or self.transport
        transcript = self.transcript_dir / f"{agent_id}.raw"
        record = self.agent_registry.maybe_get(agent_id) if self.agent_registry is not None else None
        if active_transport == "telnet":
            session = TelnetSession(self.host, self.port, transcript_path=transcript, encoding="cp437")
        elif active_transport == "rlogin":
            if record is None:
                raise AccountConfigError(f"rlogin requires an agent registry entry for {agent_id!r}")
            password = record.resolve_password()
            if password is None:
                raise AccountConfigError(f"rlogin requires a resolved BBS password for {agent_id!r}")
            session = RLoginSession(
                self.host,
                self.rlogin_port,
                username=record.bbs_alias,
                password=password,
                transcript_path=transcript,
                encoding="cp437",
            )
        else:
            raise ValueError(f"unsupported BBS transport: {active_transport}")
        session.connect()
        terminal = TerminalScreen(columns=self.columns, lines=self.lines, encoding=session.encoding)
        metadata = {
            "requested_node": node,
            "transport": active_transport,
            "host": self.host,
            "port": self.port if active_transport == "telnet" else self.rlogin_port,
            "encoding": session.encoding,
        }
        if record is not None:
            metadata.update(
                {
                    "bbs_alias": record.bbs_alias,
                    "model": record.model,
                    "account_metadata": record.metadata,
                }
            )
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
