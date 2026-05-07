"""Tiny multi-agent wrapper around terminal sessions."""

from __future__ import annotations

from pathlib import Path

from terminal_agent.agent import TerminalSessionAgent
from terminal_agent.terminal import TerminalScreen, TurnObserver
from terminal_agent.transports.rlogin import RLoginSession
from terminal_agent.transports.telnet import TelnetSession

from .accounts import AgentRegistry, AccountConfigError, load_agent_registry
from .profiles import DEFAULT_PROFILE, PromptProfile

_TELNET_LOGIN_TIMEOUT = 8.0
_TELNET_LOGIN_STABLE_MS = 300
_TELNET_LOGIN_MAX_STEPS = 6


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
            rlogin_terminal: str = "ansi",
            transport: str = "telnet",
            agent_registry: AgentRegistry | None = None,
            agent_registry_path: str | Path | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.rlogin_port = rlogin_port
        self.rlogin_terminal = rlogin_terminal
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
            model_metadata: dict[str, object] | None = None,
    ) -> TerminalSessionAgent:
        active_transport = transport or self.transport
        transcript = self.transcript_dir / f"{agent_id}.raw"
        record = self.agent_registry.maybe_get(agent_id) if self.agent_registry is not None else None
        telnet_password: str | None = None
        if active_transport == "telnet":
            if record is not None:
                telnet_password = record.resolve_password()
                if telnet_password is None:
                    raise AccountConfigError(f"telnet login requires a resolved BBS password for {agent_id!r}")
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
                terminal=self.rlogin_terminal,
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
        if active_transport == "rlogin":
            metadata["terminal"] = self.rlogin_terminal
        if record is not None:
            metadata.update(
                {
                    "bbs_alias": record.bbs_alias,
                    "account_metadata": record.metadata,
                }
            )
        if model_metadata is not None:
            metadata["model"] = model_metadata
        elif record is not None:
            metadata["model"] = record.model
        observer = TurnObserver(agent_id, session, terminal=terminal, profile=self.profile, metadata=metadata)
        if active_transport == "telnet" and record is not None:
            if telnet_password is None:
                raise AccountConfigError(f"telnet login requires a resolved BBS password for {agent_id!r}")
            self._login_telnet(observer, session, record.bbs_alias, telnet_password)
            metadata["authenticated"] = True
            metadata["login_method"] = "telnet"
            observer.metadata = metadata
        elif active_transport == "rlogin":
            metadata["authenticated"] = True
            metadata["login_method"] = "rlogin"
        agent = TerminalSessionAgent(agent_id, session, observer, metadata)
        self.agents[agent_id] = agent
        return agent

    def _login_telnet(
            self,
            observer: TurnObserver,
            session: TelnetSession,
            alias: str,
            password: str,
    ) -> None:
        for _ in range(_TELNET_LOGIN_MAX_STEPS):
            observation = observer.observe_turn(
                timeout=_TELNET_LOGIN_TIMEOUT,
                stable_ms=_TELNET_LOGIN_STABLE_MS,
            )
            text = observation.model_text.casefold()
            if _is_password_prompt(text):
                session.send_line(password)
                break
            if _is_login_prompt(text):
                session.send_line(alias)
                break
            session.send_key("enter")
        else:
            raise AccountConfigError("telnet login did not reach the BBS login prompt")

        for _ in range(_TELNET_LOGIN_MAX_STEPS):
            observation = observer.observe_turn(
                timeout=_TELNET_LOGIN_TIMEOUT,
                stable_ms=_TELNET_LOGIN_STABLE_MS,
            )
            text = observation.model_text.casefold()
            if _is_password_prompt(text):
                session.send_line(password)
                break
        else:
            raise AccountConfigError("telnet login did not reach the BBS password prompt")

        observer.observe_turn(timeout=_TELNET_LOGIN_TIMEOUT, stable_ms=_TELNET_LOGIN_STABLE_MS)
        session.drain_sent_bytes()

    def close(self) -> None:
        for agent in list(self.agents.values()):
            agent.close()
        self.agents.clear()

    def __enter__(self) -> "BbsGym":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _is_login_prompt(text: str) -> bool:
    return "login:" in text or "enter user name" in text or "enter your user name" in text


def _is_password_prompt(text: str) -> bool:
    return "password:" in text
