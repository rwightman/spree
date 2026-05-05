import argparse

from bbs_gym.accounts import AgentRecord, AgentRegistry
from bbs_gym.cli import build_model
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
        no_anthropic_cache=False,
    )

    model = build_model(args, registry)

    assert isinstance(model, OpenAICompatibleAdapter)
    assert model.model == "Qwen/Qwen3-32B"
    assert model.base_url == "http://localhost:8000/v1"
    assert model.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
