import json

import pytest

from bbs_gym.accounts import AccountConfigError, AgentRegistry


def test_agent_registry_loads_and_resolves_password_env(tmp_path, monkeypatch):
    monkeypatch.setenv("BBS_AGENT_PASSWORD", "secret")
    path = tmp_path / "agents.json"
    path.write_text(
        json.dumps(
            {
                "agents": [
                    {
                        "agent_id": "qwen-local-001",
                        "bbs_alias": "QwenOne",
                        "bbs_password_env": "BBS_AGENT_PASSWORD",
                        "security_level": 50,
                        "model": {"provider": "openai-compatible", "model": "Qwen/Qwen3-32B"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    registry = AgentRegistry.from_file(path)
    record = registry.get("qwen-local-001")

    assert record.bbs_alias == "QwenOne"
    assert record.resolve_password() == "secret"
    assert registry.check() == []
    assert registry.provision_payload()["agents"][0]["bbs_password"] == "secret"


def test_agent_registry_rejects_duplicate_aliases(tmp_path):
    path = tmp_path / "agents.json"
    path.write_text(
        json.dumps(
            {
                "agents": [
                    {"agent_id": "one", "bbs_alias": "Same"},
                    {"agent_id": "two", "bbs_alias": "same"},
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(AccountConfigError):
        AgentRegistry.from_file(path)
