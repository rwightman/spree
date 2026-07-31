"""Agent identity registry for BBS accounts and model bindings."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tty_agent.ids import validate_agent_id


class AccountConfigError(ValueError):
    """Raised when the agent registry is missing or invalid."""


@dataclass(frozen=True)
class AgentRecord:
    agent_id: str
    bbs_alias: str
    bbs_password: str | None = None
    bbs_password_env: str | None = None
    security_level: int | None = None
    model: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            validate_agent_id(self.agent_id)
        except ValueError as exc:
            raise AccountConfigError(str(exc)) from exc

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "AgentRecord":
        agent_id = _required_str(data, "agent_id")
        bbs_alias = _required_str(data, "bbs_alias")
        return cls(
            agent_id=agent_id,
            bbs_alias=bbs_alias,
            bbs_password=_optional_str(data, "bbs_password"),
            bbs_password_env=_optional_str(data, "bbs_password_env"),
            security_level=_optional_int(data, "security_level"),
            model=_optional_dict(data, "model"),
            metadata=_optional_dict(data, "metadata"),
        )

    def resolve_password(self, environ: dict[str, str] | None = None) -> str | None:
        if self.bbs_password is not None:
            return self.bbs_password
        if self.bbs_password_env is None:
            return None
        env = os.environ if environ is None else environ
        return env.get(self.bbs_password_env)

    def public_dict(self, include_password_state: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "agent_id": self.agent_id,
            "bbs_alias": self.bbs_alias,
            "security_level": self.security_level,
            "model": redacted_model_config(self.model),
            "metadata": self.metadata,
        }
        if self.bbs_password_env is not None:
            data["bbs_password_env"] = self.bbs_password_env
        if include_password_state:
            data["has_password"] = self.resolve_password() is not None
        return data

    def provision_dict(self) -> dict[str, Any]:
        password = self.resolve_password()
        if not password:
            raise AccountConfigError(f"agent {self.agent_id!r} has no resolved BBS password")
        return {
            "agent_id": self.agent_id,
            "bbs_alias": self.bbs_alias,
            "bbs_password": password,
            "security_level": self.security_level,
        }


@dataclass(frozen=True)
class AgentRegistry:
    agents: dict[str, AgentRecord]

    @classmethod
    def from_file(cls, path: str | Path) -> "AgentRegistry":
        registry_path = Path(path)
        try:
            data = json.loads(registry_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise AccountConfigError(f"agent registry not found: {registry_path}") from exc
        except json.JSONDecodeError as exc:
            raise AccountConfigError(f"invalid JSON in agent registry {registry_path}: {exc}") from exc
        if not isinstance(data, dict):
            raise AccountConfigError("agent registry root must be a JSON object")
        agents_value = data.get("agents", [])
        if not isinstance(agents_value, list):
            raise AccountConfigError("agent registry field 'agents' must be a list")

        records = [AgentRecord.from_mapping(item) for item in agents_value if isinstance(item, dict)]
        if len(records) != len(agents_value):
            raise AccountConfigError("each agent registry entry must be a JSON object")
        agents: dict[str, AgentRecord] = {}
        aliases: dict[str, str] = {}
        for record in records:
            if record.agent_id in agents:
                raise AccountConfigError(f"duplicate agent_id: {record.agent_id}")
            alias_key = record.bbs_alias.casefold()
            if alias_key in aliases:
                raise AccountConfigError(
                    f"duplicate bbs_alias {record.bbs_alias!r} for {record.agent_id!r} and {aliases[alias_key]!r}"
                )
            agents[record.agent_id] = record
            aliases[alias_key] = record.agent_id
        return cls(agents=agents)

    def get(self, agent_id: str) -> AgentRecord:
        try:
            return self.agents[agent_id]
        except KeyError as exc:
            raise AccountConfigError(f"unknown agent_id in registry: {agent_id}") from exc

    def maybe_get(self, agent_id: str) -> AgentRecord | None:
        return self.agents.get(agent_id)

    def public_list(self) -> list[dict[str, Any]]:
        return [record.public_dict() for record in self.agents.values()]

    def check(self) -> list[str]:
        errors = []
        for record in self.agents.values():
            if record.resolve_password() is None:
                errors.append(f"{record.agent_id}: missing password from bbs_password or {record.bbs_password_env}")
        return errors

    def provision_payload(self) -> dict[str, Any]:
        return {"agents": [record.provision_dict() for record in self.agents.values()]}


_SENSITIVE_KEY_MARKERS = ("api_key", "apikey", "authorization", "cookie", "secret", "password")


def redacted_model_config(config: dict[str, Any]) -> dict[str, Any]:
    """Copy a model config with provider credentials replaced by a placeholder.

    Model configs can hold inline API keys; anything shown to users, logged, or
    attached to observation metadata must go through this first.
    """

    redacted: dict[str, Any] = {}
    for key, value in config.items():
        key_fold = key.casefold().replace("-", "_")
        if any(marker in key_fold for marker in _SENSITIVE_KEY_MARKERS) or (
            key_fold == "token" or key_fold.endswith("_token")
        ):
            redacted[key] = "[redacted]"
        else:
            redacted[key] = _redacted_model_value(value)
    return redacted


def _redacted_model_value(value: Any) -> Any:
    if isinstance(value, dict):
        return redacted_model_config(value)
    if isinstance(value, list):
        return [_redacted_model_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redacted_model_value(item) for item in value)
    return value


def load_agent_registry(path: str | Path | None, required: bool = False) -> AgentRegistry | None:
    if path is None:
        return None
    registry_path = Path(path)
    if not registry_path.exists():
        if required:
            raise AccountConfigError(f"agent registry not found: {registry_path}")
        return None
    return AgentRegistry.from_file(registry_path)


def _required_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise AccountConfigError(f"agent registry field {key!r} must be a non-empty string")
    return value


def _optional_str(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise AccountConfigError(f"agent registry field {key!r} must be a string")
    return value


def _optional_int(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise AccountConfigError(f"agent registry field {key!r} must be an integer")
    return value


def _optional_dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise AccountConfigError(f"agent registry field {key!r} must be an object")
    return dict(value)
