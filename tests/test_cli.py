import argparse

from bbs_gym.accounts import AgentRecord, AgentRegistry
from bbs_gym.cli import build_activity_profile, build_model
from terminal_agent.models import OpenAICompatibleAdapter


def test_build_model_uses_agent_registry_model_config():
    registry = AgentRegistry(
        agents={
            "qwen-local-001": AgentRecord(
                agent_id="qwen-local-001",
                bbs_alias="QwenOne",
                model={
                    "provider": "openai-compatible",
                    "base_url": "http://localhost:8000/v1",
                    "model": "Qwen/Qwen3-32B",
                    "temperature": 0,
                    "max_tokens": 512,
                    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
                },
            )
        }
    )
    args = argparse.Namespace(
        agent_id="qwen-local-001",
        provider=None,
        scripted_response=[],
        model=None,
        base_url=None,
        api_key=None,
        temperature=None,
        max_tokens=None,
        response_filter=None,
        no_anthropic_cache=False,
    )

    model = build_model(args, registry)

    assert isinstance(model, OpenAICompatibleAdapter)
    assert model.model == "Qwen/Qwen3-32B"
    assert model.base_url == "http://localhost:8000/v1"
    assert model.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_build_activity_profile_applies_named_profile_overrides():
    args = argparse.Namespace(
        activity="tw2-game",
        objective="custom game objective",
        observe_timeout=12.5,
        stable_ms=750,
    )

    profile = build_activity_profile(args)

    assert profile.name == "tw2-game"
    assert profile.objective == "custom game objective"
    assert profile.observe_timeout == 12.5
    assert profile.stable_ms == 750
