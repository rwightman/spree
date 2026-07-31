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


def test_agent_registry_rejects_unsafe_agent_ids(tmp_path):
    path = tmp_path / "agents.json"
    path.write_text(
        json.dumps({"agents": [{"agent_id": "../escape", "bbs_alias": "Escape"}]}),
        encoding="utf-8",
    )

    with pytest.raises(AccountConfigError, match="path separators"):
        AgentRegistry.from_file(path)


def test_public_dict_redacts_provider_credentials():
    from bbs_gym.accounts import AgentRecord, redacted_model_config

    record = AgentRecord(
        agent_id="agent-001",
        bbs_alias="AgentOne",
        model={
            "provider": "openai-compatible",
            "model": "some-model",
            "api_key": "sk-live-do-not-leak",
            "max_tokens": 512,
            "extra": {"auth_token": "also-secret"},
            "extra_headers": {
                "Authorization": "Bearer hidden",
                "X-API-Key": "header-secret",
                "x-session-affinity": "agent-001",
            },
            "providers": [{"name": "fallback", "api_key": "nested-list-secret"}],
        },
    )

    public = record.public_dict()

    assert public["model"]["api_key"] == "[redacted]"
    assert public["model"]["extra"]["auth_token"] == "[redacted]"
    assert public["model"]["extra_headers"] == {
        "Authorization": "[redacted]",
        "X-API-Key": "[redacted]",
        "x-session-affinity": "agent-001",
    }
    assert public["model"]["providers"][0]["api_key"] == "[redacted]"
    assert public["model"]["model"] == "some-model"
    assert public["model"]["max_tokens"] == 512
    # The record itself is untouched; only the public view is redacted.
    assert record.model["api_key"] == "sk-live-do-not-leak"
    assert redacted_model_config({"password": {"value": "x"}}) == {"password": "[redacted]"}
