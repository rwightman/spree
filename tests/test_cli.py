import argparse

from bbs_gym.accounts import AgentRecord, AgentRegistry
from bbs_gym.cli import build_activity_profile, build_activity_route_set, build_model, build_model_metadata
from tty_agent.models import CodexCliAdapter, OpenAICompatibleAdapter


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


def test_build_model_uses_codex_registry_model_config():
    registry = AgentRegistry(
        agents={
            "codex-001": AgentRecord(
                agent_id="codex-001",
                bbs_alias="CodexOne",
                model={
                    "provider": "codex",
                    "model": "gpt-5.5",
                    "profile": "bbs",
                    "timeout": 12,
                    "sandbox": "read-only",
                    "cwd": "runtime/codex-provider",
                    "extra_args": ["--ignore-rules"],
                    "stateful": True,
                    "session_file": "runtime/codex-provider/codex.session",
                },
            )
        }
    )
    args = argparse.Namespace(
        agent_id="codex-001",
        provider=None,
        scripted_response=[],
        model=None,
        base_url=None,
        api_key=None,
        temperature=None,
        max_tokens=None,
        response_filter=None,
        no_anthropic_cache=False,
        codex_profile=None,
        codex_executable=None,
        codex_timeout=None,
        codex_sandbox=None,
        codex_cwd=None,
        codex_arg=[],
        codex_stateful=False,
        codex_session_id=None,
        codex_session_file=None,
    )

    model = build_model(args, registry)

    assert isinstance(model, CodexCliAdapter)
    assert model.model == "gpt-5.5"
    assert model.profile == "bbs"
    assert model.timeout == 12
    assert model.sandbox == "read-only"
    assert str(model.cwd) == "runtime/codex-provider"
    assert model.extra_args == ["--ignore-rules"]
    assert model.stateful is True
    assert str(model.session_file) == "runtime/codex-provider/codex.session"


def test_build_model_metadata_reflects_cli_provider_override():
    registry = AgentRegistry(
        agents={
            "rlogin-smoke": AgentRecord(
                agent_id="rlogin-smoke",
                bbs_alias="RLoginSmoke",
                model={"provider": "scripted"},
            )
        }
    )
    args = argparse.Namespace(
        agent_id="rlogin-smoke",
        provider="codex",
        scripted_response=[],
        model="gpt-5.5",
        base_url=None,
        api_key=None,
        temperature=None,
        max_tokens=None,
        response_filter=None,
        no_anthropic_cache=False,
        codex_profile=None,
        codex_executable=None,
        codex_timeout=180,
        codex_sandbox="read-only",
        codex_cwd=None,
        codex_arg=[],
        codex_stateful=False,
        codex_session_id=None,
        codex_session_file=None,
    )

    metadata = build_model_metadata(args, registry)

    assert metadata["provider"] == "codex"
    assert metadata["model"] == "gpt-5.5"
    assert metadata["timeout"] == 180
    assert metadata["sandbox"] == "read-only"


def test_build_activity_profile_uses_stateful_delta_for_stateful_codex():
    registry = AgentRegistry(
        agents={
            "codex-001": AgentRecord(
                agent_id="codex-001",
                bbs_alias="CodexOne",
                model={
                    "provider": "codex",
                    "model": "gpt-5.5",
                    "stateful": True,
                },
            )
        }
    )
    args = argparse.Namespace(
        agent_id="codex-001",
        provider=None,
        activity="tw2-game",
        profile_objective=None,
        observe_timeout=None,
        stable_ms=None,
        byte_quiet_ms=None,
        recent_steps_to_keep=None,
        prompt_mode=None,
        prompt_layout=None,
        codex_stateful=False,
    )

    profile = build_activity_profile(args, registry)

    assert profile.prompt_mode == "stateful_delta"


def test_build_activity_profile_applies_named_profile_overrides():
    args = argparse.Namespace(
        activity="tw2-game",
        profile_objective="custom game objective",
        observe_timeout=12.5,
        stable_ms=750,
        byte_quiet_ms=900,
        recent_steps_to_keep=5,
        prompt_mode="stateful_delta",
        prompt_layout="cache_friendly",
    )

    profile = build_activity_profile(args)

    assert profile.name == "tw2-game"
    assert profile.objective == "custom game objective"
    assert profile.observe_timeout == 12.5
    assert profile.stable_ms == 750
    assert profile.byte_quiet_ms == 900
    assert profile.recent_steps_to_keep == 5
    assert profile.prompt_mode == "stateful_delta"
    assert profile.prompt_layout == "cache_friendly"


def test_build_activity_route_set_applies_profile_overrides():
    args = argparse.Namespace(
        agent_id="agent-001",
        provider=None,
        route_set="tw2-auto",
        profile_objective="custom routed objective",
        observe_timeout=12.5,
        stable_ms=750,
        byte_quiet_ms=900,
        recent_steps_to_keep=5,
        prompt_mode="stateful_delta",
        prompt_layout="cache_friendly",
        codex_stateful=False,
    )

    route_set = build_activity_route_set(args)

    assert route_set.name == "tw2-auto"
    assert route_set.default_profile.name == "tw2-entry"
    assert route_set.default_profile.objective == "custom routed objective"
    assert route_set.default_profile.observe_timeout == 12.5
    assert route_set.default_profile.stable_ms == 750
    assert route_set.default_profile.byte_quiet_ms == 900
    assert route_set.default_profile.recent_steps_to_keep == 5
    assert route_set.default_profile.prompt_mode == "stateful_delta"
    assert route_set.default_profile.prompt_layout == "cache_friendly"
    assert route_set.routes[0].profile.name == "tw2-game"
    assert route_set.routes[0].profile.recent_steps_to_keep == 5
    assert route_set.routes[0].profile.prompt_mode == "stateful_delta"
    assert route_set.routes[0].profile.prompt_layout == "cache_friendly"
