import argparse

from bbs_gym.accounts import AgentRecord, AgentRegistry
from bbs_gym.cli import (
    build_match_scheduler_config,
    build_activity_profile,
    build_activity_route_set,
    build_match_participants,
    build_model,
    build_model_metadata,
    match_participant_specs,
    smoke,
)
from bbs_gym.match import (
    MatchParticipantRuntime,
    MatchParticipantSpec,
    MatchSchedulerConfig,
    handle_match_disconnect,
    match_round_order,
)
from tty_agent.models import ClaudeCliAdapter, CodexCliAdapter, OpenAICompatibleAdapter
from tty_agent.transports.base import SessionDisconnected


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


def test_build_model_uses_claude_registry_model_config():
    registry = AgentRegistry(
        agents={
            "claude-001": AgentRecord(
                agent_id="claude-001",
                bbs_alias="ClaudeOne",
                model={
                    "provider": "claude",
                    "model": "claude-sonnet-4-6",
                    "executable": "claude",
                    "timeout": 12,
                    "cwd": "runtime/claude-provider",
                    "extra_args": ["--debug"],
                    "stateful": True,
                    "session_file": "runtime/claude-provider/claude.session",
                    "permission_mode": "dontAsk",
                    "tools": "",
                    "bare": True,
                },
            )
        }
    )
    args = argparse.Namespace(
        agent_id="claude-001",
        provider=None,
        scripted_response=[],
        model=None,
        base_url=None,
        api_key=None,
        temperature=None,
        max_tokens=None,
        response_filter=None,
        no_anthropic_cache=False,
        claude_executable=None,
        claude_timeout=None,
        claude_cwd=None,
        claude_arg=[],
        claude_stateful=False,
        claude_session_id=None,
        claude_session_file=None,
        claude_permission_mode=None,
        claude_tools=None,
        claude_bare=False,
    )

    model = build_model(args, registry)

    assert isinstance(model, ClaudeCliAdapter)
    assert model.model == "claude-sonnet-4-6"
    assert model.timeout == 12
    assert str(model.cwd) == "runtime/claude-provider"
    assert model.extra_args == ["--debug"]
    assert model.stateful is True
    assert str(model.session_file) == "runtime/claude-provider/claude.session"
    assert model.permission_mode == "dontAsk"
    assert model.tools == ""
    assert model.bare is True


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
        disabled_actions=[],
    )

    profile = build_activity_profile(args, registry)

    assert profile.prompt_mode == "stateful_delta"


def test_build_activity_profile_uses_stateful_delta_for_stateful_claude():
    registry = AgentRegistry(
        agents={
            "claude-001": AgentRecord(
                agent_id="claude-001",
                bbs_alias="ClaudeOne",
                model={
                    "provider": "claude",
                    "model": "claude-sonnet-4-6",
                    "stateful": True,
                },
            )
        }
    )
    args = argparse.Namespace(
        agent_id="claude-001",
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
        claude_stateful=False,
        disabled_actions=[],
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
        disabled_actions=[],
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


def test_build_activity_profile_can_disable_actions():
    args = argparse.Namespace(
        activity="bbs-door-line",
        profile_objective=None,
        agent_id="agent",
        provider=None,
        observe_timeout=None,
        stable_ms=None,
        byte_quiet_ms=None,
        recent_steps_to_keep=None,
        prompt_mode=None,
        prompt_layout=None,
        codex_stateful=False,
        disabled_actions=["hangup"],
    )

    profile = build_activity_profile(args)

    assert "hangup" not in profile.action_policy.allowed_actions
    assert "submit_line" in profile.action_policy.allowed_actions


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
        disabled_actions=[],
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


def test_match_participant_specs_parse_inline_provider_and_model():
    args = argparse.Namespace(
        match_config=None,
        participant=["codex-blue:codex:gpt-5.5", "claude-red:claude:sonnet"],
        agent_id=[],
    )

    specs = match_participant_specs(args)

    assert [spec.agent_id for spec in specs] == ["codex-blue", "claude-red"]
    assert specs[0].provider == "codex"
    assert specs[0].model == "gpt-5.5"
    assert specs[1].provider == "claude"
    assert specs[1].model == "sonnet"


def test_build_match_participants_formats_objectives_and_logs(tmp_path):
    args = argparse.Namespace(
        match_config=None,
        participant=["codex-blue:scripted:unused", "claude-red:scripted:unused"],
        agent_id=[],
        provider=None,
        scripted_response=['{"action": "wait", "arguments": {}}'],
        model=None,
        base_url=None,
        api_key=None,
        temperature=None,
        max_tokens=None,
        response_filter=None,
        no_anthropic_cache=False,
        activity="bbs-door-line",
        profile_objective=None,
        run_objective="{agent_id} should find {opponents}",
        observe_timeout=None,
        stable_ms=None,
        byte_quiet_ms=None,
        recent_steps_to_keep=None,
        prompt_mode=None,
        prompt_layout=None,
        codex_stateful=False,
        claude_stateful=False,
        disabled_actions=[],
        log_path=str(tmp_path / "match.jsonl"),
    )

    participants = build_match_participants(args, match_participant_specs(args), registry=None)

    assert [participant.spec.agent_id for participant in participants] == ["codex-blue", "claude-red"]
    assert participants[0].runner.run_objective == "codex-blue should find claude-red"
    assert participants[1].runner.run_objective == "claude-red should find codex-blue"
    assert participants[0].runner.profile.name == "bbs-door-line"
    assert participants[0].log_path == tmp_path / "match.codex-blue.jsonl"
    assert participants[1].log_path == tmp_path / "match.claude-red.jsonl"


def test_match_config_toml_supplies_scheduler_budget_and_participants(tmp_path):
    config_path = tmp_path / "melee.toml"
    config_path.write_text(
        """
activity = "bbs-door-line"
transport = "telnet"
telnet_enter = "lf"
run_objective = "Play as {agent_id}; opponents: {opponents}"
disabled_actions = ["hangup"]
log_path = "runtime/logs/melee.jsonl"

[scheduler]
mode = "sequential"
order = "shuffle"
seed = 17
disconnect_policy = "reconnect"
max_reconnects = 2
reconnect_delay = 0.0
max_workers = 4

[budget]
max_rounds = 250
max_decision_ticks = 125
max_wall_seconds = 3600

[[participants]]
agent_id = "codex-blue"
provider = "codex"
model = "gpt-5.5"
stateful = true
codex_session_file = "runtime/codex-blue.session"

[[participants]]
agent_id = "gemma-green"
provider = "openai-compatible"
model = "gemma4"
base_url = "http://localhost:8000/v1"
temperature = 0.6
""".strip()
        + "\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        match_config=str(config_path),
        participant=[],
        agent_id=[],
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
        claude_stateful=False,
        activity="tw2-game",
        profile_objective=None,
        run_objective=None,
        observe_timeout=None,
        stable_ms=None,
        byte_quiet_ms=None,
        recent_steps_to_keep=None,
        model_error_retries=None,
        prompt_mode=None,
        prompt_layout=None,
        disabled_actions=[],
        log_path=str(tmp_path / "default.jsonl"),
        max_rounds=50,
        max_decision_ticks=50,
        max_wall_seconds=600.0,
        scheduler_mode="sequential",
        match_order="fixed",
        match_seed=None,
        disconnect_policy="stop",
        max_reconnects=3,
        reconnect_delay=2.0,
        max_workers=None,
        host="127.0.0.1",
        port=2323,
        rlogin_port=2513,
        rlogin_terminal="ansi",
        transport="telnet",
        telnet_enter="cr",
        agents_config="config/agents.local.json",
        no_agents_config=False,
    )

    specs = match_participant_specs(args)
    participants = build_match_participants(args, specs, registry=None)

    assert args.match_order == "shuffle"
    assert args.scheduler_mode == "sequential"
    assert args.match_seed == 17
    assert args.disconnect_policy == "reconnect"
    assert args.max_reconnects == 2
    assert args.reconnect_delay == 0.0
    assert args.max_workers == 4
    assert args.max_rounds == 250
    assert args.max_decision_ticks == 125
    assert args.max_wall_seconds == 3600
    assert args.disabled_actions == ["hangup"]
    assert [spec.agent_id for spec in specs] == ["codex-blue", "gemma-green"]
    assert "hangup" not in participants[0].runner.profile.action_policy.allowed_actions
    assert isinstance(participants[0].model, CodexCliAdapter)
    assert participants[0].model.stateful is True
    assert str(participants[0].model.session_file) == "runtime/codex-blue.session"
    assert isinstance(participants[1].model, OpenAICompatibleAdapter)
    assert participants[1].model.base_url == "http://localhost:8000/v1"
    assert participants[1].model.temperature == 0.6
    scheduler = build_match_scheduler_config(args)
    assert scheduler == MatchSchedulerConfig(
        mode="sequential",
        order="shuffle",
        seed=17,
        disconnect_policy="reconnect",
        max_reconnects=2,
        reconnect_delay=0.0,
        max_rounds=250,
        max_decision_ticks=125,
        max_wall_seconds=3600,
        max_workers=4,
    )


def test_match_round_order_fixed_shuffle_and_rotate_are_deterministic():
    states = [
        (argparse.Namespace(spec=argparse.Namespace(agent_id="a")), argparse.Namespace(completed=False)),
        (argparse.Namespace(spec=argparse.Namespace(agent_id="b")), argparse.Namespace(completed=False)),
        (argparse.Namespace(spec=argparse.Namespace(agent_id="c")), argparse.Namespace(completed=False)),
    ]

    assert [participant.spec.agent_id for participant, _ in match_round_order(states, "fixed", random_rng(3), 1)] == [
        "a",
        "b",
        "c",
    ]
    assert [participant.spec.agent_id for participant, _ in match_round_order(states, "rotate", random_rng(3), 2)] == [
        "b",
        "c",
        "a",
    ]
    assert [participant.spec.agent_id for participant, _ in match_round_order(states, "shuffle", random_rng(3), 1)] == [
        "b",
        "c",
        "a",
    ]


def random_rng(seed: int):
    import random

    return random.Random(seed)


def test_handle_match_disconnect_reconnects_and_logs(tmp_path):
    class FakeAgent:
        def __init__(self, agent_id: str) -> None:
            self.agent_id = agent_id
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class FakeGym:
        def __init__(self) -> None:
            self.connected = []

        def connect(self, agent_id, model_metadata=None):
            self.connected.append((agent_id, model_metadata))
            return FakeAgent(agent_id)

    old_agent = FakeAgent("arena-codex")
    state = argparse.Namespace(agent=old_agent, completed=True, stop_reason="disconnected")
    participant = MatchParticipantRuntime(
        spec=MatchParticipantSpec("arena-codex", "codex", "gpt-5.5"),
        model=object(),
        model_metadata={"provider": "codex"},
        runner=object(),
        log_path=tmp_path / "agent.jsonl",
    )
    scheduler = MatchSchedulerConfig(disconnect_policy="reconnect", max_reconnects=2, reconnect_delay=0.0)
    match_log = tmp_path / "match.jsonl"
    gym = FakeGym()

    handle_match_disconnect(gym, participant, state, scheduler, match_log, round_number=7)

    assert old_agent.closed is True
    assert state.completed is False
    assert state.stop_reason == ""
    assert state.agent.agent_id == "arena-codex"
    assert participant.reconnects == 1
    assert gym.connected == [("arena-codex", {"provider": "codex"})]
    events = [line for line in match_log.read_text(encoding="utf-8").splitlines() if line]
    assert '"type": "participant_disconnected"' in events[0]
    assert '"type": "participant_reconnected"' in events[1]


def test_smoke_reports_disconnect_instead_of_tracebacking(monkeypatch, capsys, tmp_path):
    """SessionDisconnected is a RuntimeError, so the CLI must catch it explicitly."""

    class DisconnectingSession:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def read(self, _seconds):
            raise SessionDisconnected("remote terminal connection closed: [Errno 104] reset by peer")

    monkeypatch.setattr("bbs_gym.cli.TelnetSession", DisconnectingSession)
    args = argparse.Namespace(
        host="127.0.0.1",
        port=2323,
        timeout=1.0,
        seconds=0.1,
        tail=100,
        transcript=str(tmp_path / "smoke.raw"),
    )

    assert smoke(args) == 1
    assert "connection failed" in capsys.readouterr().err
