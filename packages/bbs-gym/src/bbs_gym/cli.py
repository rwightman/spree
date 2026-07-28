"""Command-line helpers for BBS gym sessions."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Any, get_args

from tty_agent.ansi import strip_ansi
from tty_agent.actions import DEFAULT_ALLOWED_ACTIONS
from tty_agent.evaluation import EvaluationProfile
from tty_agent.models import (
    AnthropicAdapter,
    ClaudeCliAdapter,
    CodexCliAdapter,
    OpenAICompatibleAdapter,
    ScriptedModelAdapter,
)
from tty_agent.models import output_filters_for_model
from tty_agent.runner import ActivityBudget, ActivityProfile, ActivityRunner, RoutedActivityRunner
from tty_agent.terminal import TerminalScreen, TurnObserver
from tty_agent.transports.base import SessionDisconnected
from tty_agent.transports.telnet import TelnetSession

from .accounts import AccountConfigError, AgentRegistry, load_agent_registry
from .activities import activity_profile
from .env import BbsGym
from .evaluation import TW2_EVALUATION_PROFILE
from .match import (
    DisconnectPolicy,
    MatchOrder,
    MatchParticipantRuntime,
    MatchParticipantSpec,
    MatchSchedulerConfig,
    MatchSchedulerMode,
    run_scheduled_match,
)
from .profiles import BBS_PROFILE, TW2_PROFILE
from .routing import ActivityRouteSet, activity_route_set, activity_route_set_names


DEFAULT_AGENTS_CONFIG = Path("config/agents.local.json")
DEFAULT_OPENAI_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MATCH_OBJECTIVE = (
    "Play this shared terminal activity as {agent_id}. Other active agents in the match: {opponents}. "
    "Explore, survive, improve your position, and interact with opponents when useful."
)


def smoke(args: argparse.Namespace) -> int:
    transcript = Path(args.transcript) if args.transcript else None
    try:
        with TelnetSession(args.host, args.port, args.timeout, transcript, encoding="cp437") as session:
            data = session.read(args.seconds)
    except (OSError, SessionDisconnected) as exc:
        print(f"connection failed: {exc}", file=sys.stderr)
        return 1

    text = strip_ansi(data, encoding="cp437")
    print(text[-args.tail:])
    return 0 if data else 2


def observe_turn(args: argparse.Namespace) -> int:
    transcript = Path(args.transcript) if args.transcript else None
    profile = TW2_PROFILE if args.profile == "tw2" else BBS_PROFILE
    try:
        with TelnetSession(args.host, args.port, args.timeout, transcript, encoding="cp437") as session:
            observer = TurnObserver(
                args.agent_id,
                session,
                terminal=TerminalScreen(encoding=session.encoding),
                profile=profile,
                metadata={
                    "transport": "telnet",
                    "host": args.host,
                    "port": args.port,
                    "encoding": session.encoding,
                },
            )
            observation = observer.observe_turn(
                timeout=args.timeout,
                stable_ms=args.stable_ms,
                byte_quiet_ms=args.byte_quiet_ms,
                poll_interval=args.poll_interval,
                prompt_fast_path=args.prompt_fast_path,
            )
    except (OSError, SessionDisconnected) as exc:
        print(f"connection failed: {exc}", file=sys.stderr)
        return 1

    print(observation.model_text[-args.tail:])
    if observation.matched_prompt:
        print(
            f"\n[matched_prompt={observation.matched_prompt} "
            f"stable_ms={observation.stable_ms} byte_quiet_ms={observation.byte_quiet_ms}]"
        )
    if observation.timed_out:
        print(
            f"\n[timed_out stable_ms={observation.stable_ms} byte_quiet_ms={observation.byte_quiet_ms}]",
            file=sys.stderr,
        )
        return 2
    return 0


def run_activity(args: argparse.Namespace) -> int:
    try:
        registry = load_agent_registry(args.agents_config, required=False)
        model = build_model(args, registry)
        model_metadata = build_model_metadata(args, registry)
        profile = build_activity_profile(args, registry)
    except (AccountConfigError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    evaluation_profile = _activity_evaluation_profile(profile.name)
    runner = ActivityRunner(
        profile,
        log_path=args.log_path,
        run_objective=args.run_objective or "",
        evaluation_profile=evaluation_profile,
        evaluation_log_path=args.metrics_path if evaluation_profile is not None else None,
    )

    try:
        with BbsGym(
            host=args.host,
            port=args.port,
            rlogin_port=args.rlogin_port,
            rlogin_terminal=args.rlogin_terminal,
            transport=args.transport,
            telnet_enter_sequence=args.telnet_enter,
            agent_registry=registry,
        ) as gym:
            agent = gym.connect(args.agent_id, node=args.node, model_metadata=model_metadata)
            result = runner.run(
                agent,
                model,
                ActivityBudget(
                    max_decision_ticks=args.max_decision_ticks,
                    max_wall_seconds=args.max_wall_seconds,
                ),
            )
    except (OSError, SessionDisconnected, AccountConfigError, ValueError) as exc:
        print(f"connection failed: {exc}", file=sys.stderr)
        return 1

    print(f"activity={result.activity} agent={result.agent_id} steps={len(result.steps)} stop={result.stop_reason}")
    if evaluation_profile is not None:
        _print_evaluation_summary(result, args.metrics_path)
    return 0


def run_routed(args: argparse.Namespace) -> int:
    try:
        registry = load_agent_registry(args.agents_config, required=False)
        model = build_model(args, registry)
        model_metadata = build_model_metadata(args, registry)
        route_set = build_activity_route_set(args, registry)
    except (AccountConfigError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    runner = RoutedActivityRunner(
        route_set.name,
        route_set.default_profile,
        route_set.routes,
        log_path=args.log_path,
        run_objective=args.run_objective or "",
        evaluation_profile=TW2_EVALUATION_PROFILE,
        evaluation_log_path=args.metrics_path,
    )

    try:
        with BbsGym(
            host=args.host,
            port=args.port,
            rlogin_port=args.rlogin_port,
            rlogin_terminal=args.rlogin_terminal,
            transport=args.transport,
            telnet_enter_sequence=args.telnet_enter,
            agent_registry=registry,
        ) as gym:
            agent = gym.connect(args.agent_id, node=args.node, model_metadata=model_metadata)
            result = runner.run(
                agent,
                model,
                ActivityBudget(
                    max_decision_ticks=args.max_decision_ticks,
                    max_wall_seconds=args.max_wall_seconds,
                ),
            )
    except (OSError, SessionDisconnected, AccountConfigError, ValueError) as exc:
        print(f"connection failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"route_set={route_set.name} agent={result.agent_id} steps={len(result.steps)} "
        f"stop={result.stop_reason} final_profile={runner.profile.name}"
    )
    _print_evaluation_summary(result, args.metrics_path)
    return 0


def run_match(args: argparse.Namespace) -> int:
    try:
        _apply_match_config(args)
        _validate_match_args(args)
        registry = None if args.no_agents_config else load_agent_registry(args.agents_config, required=False)
        specs = match_participant_specs(args)
        participants = build_match_participants(args, specs, registry)
        scheduler = build_match_scheduler_config(args)
    except (AccountConfigError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    match_log_path = Path(args.log_path)
    try:
        with BbsGym(
            host=args.host,
            port=args.port,
            rlogin_port=args.rlogin_port,
            rlogin_terminal=args.rlogin_terminal,
            transport=args.transport,
            telnet_enter_sequence=args.telnet_enter,
            agent_registry=registry,
        ) as gym:
            match_result = run_scheduled_match(gym, participants, scheduler, match_log_path)
    except (OSError, SessionDisconnected, AccountConfigError, ValueError) as exc:
        print(f"connection failed: {exc}", file=sys.stderr)
        return 1

    summary = ", ".join(_match_result_summary(result) for _, result in match_result.results)
    print(
        f"match participants={len(match_result.results)} commit_count={match_result.commit_count} "
        f"scheduler={scheduler.mode} {summary} log={match_log_path}"
    )
    return 0


def _apply_match_config(args: argparse.Namespace) -> None:
    if getattr(args, "_match_config_applied", False):
        return
    args._match_config_applied = True
    path = getattr(args, "match_config", None)
    if not path:
        return

    config = _load_match_config(Path(path))
    changed: list[str] = []
    changed += _set_config_values(
        args,
        config,
        {
            "host": "host",
            "port": "port",
            "rlogin_port": "rlogin_port",
            "rlogin_terminal": "rlogin_terminal",
            "transport": "transport",
            "telnet_enter": "telnet_enter",
            "agents_config": "agents_config",
            "no_agents_config": "no_agents_config",
            "activity": "activity",
            "profile_objective": "profile_objective",
            "run_objective": "run_objective",
            "log_path": "log_path",
            "metrics_path": "metrics_path",
            "observe_timeout": "observe_timeout",
            "stable_ms": "stable_ms",
            "byte_quiet_ms": "byte_quiet_ms",
            "recent_steps_to_keep": "recent_steps_to_keep",
            "model_error_retries": "model_error_retries",
            "prompt_mode": "prompt_mode",
            "prompt_layout": "prompt_layout",
            "disabled_actions": "disabled_actions",
        },
    )
    changed += _set_config_values(
        args,
        _config_mapping(config, "scheduler"),
        {
            "mode": "scheduler_mode",
            "order": "match_order",
            "seed": "match_seed",
            "disconnect_policy": "disconnect_policy",
            "max_reconnects": "max_reconnects",
            "reconnect_delay": "reconnect_delay",
            "max_workers": "max_workers",
        },
    )
    changed += _set_config_values(
        args,
        _config_mapping(config, "budget"),
        {
            "max_rounds": "max_rounds",
            "max_decision_ticks": "max_decision_ticks",
            "max_wall_seconds": "max_wall_seconds",
        },
    )
    if changed:
        # Config wins over command-line flags by design; say so instead of
        # silently ignoring what the user typed.
        print(f"match config {path} overrides: {', '.join(changed)}", file=sys.stderr)
    if "participants" in config:
        args._match_participants_config = _match_participant_specs_from_config(config["participants"])


def _load_match_config(path: Path) -> dict[str, Any]:
    try:
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read match config {path}: {exc}") from exc
    except (json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"invalid match config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("match config root must be an object")
    return data


def _set_config_values(args: argparse.Namespace, config: dict[str, Any], mapping: dict[str, str]) -> list[str]:
    changed: list[str] = []
    for config_key, arg_key in mapping.items():
        if config_key in config:
            if getattr(args, arg_key, None) != config[config_key]:
                changed.append(arg_key)
            setattr(args, arg_key, config[config_key])
    return changed


def _config_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"match config field {key!r} must be an object")
    return value


def _match_participant_specs_from_config(value: object) -> list[MatchParticipantSpec]:
    if not isinstance(value, list):
        raise ValueError("match config field 'participants' must be a list")
    specs: list[MatchParticipantSpec] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("each match participant must be an object")
        agent_id = _required_config_str(item, "agent_id", "match participant")
        provider = _config_str(item, "provider")
        model = _config_str(item, "model")
        specs.append(MatchParticipantSpec(agent_id=agent_id, provider=provider, model=model, config=dict(item)))
    return specs


def _validate_choice(value: object, name: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of: {', '.join(allowed)}")


def _validate_match_args(args: argparse.Namespace) -> None:
    _validate_choice(args.scheduler_mode, "scheduler_mode", get_args(MatchSchedulerMode))
    _validate_choice(args.match_order, "match_order", get_args(MatchOrder))
    _validate_choice(args.disconnect_policy, "disconnect_policy", get_args(DisconnectPolicy))
    _require_int(args.max_reconnects, "max_reconnects", minimum=0)
    _require_number(args.reconnect_delay, "reconnect_delay", minimum=0)
    if args.max_workers is not None:
        _require_int(args.max_workers, "max_workers", minimum=1)
    _require_int(args.max_rounds, "max_rounds", minimum=1)
    _require_int(args.max_decision_ticks, "max_decision_ticks", minimum=1)
    _require_number(args.max_wall_seconds, "max_wall_seconds", minimum=0, exclusive=True)


def _require_int(value: object, name: str, minimum: int) -> None:
    # A config file can supply any type; comparing it blind would raise
    # TypeError, which run-match does not translate into a config error.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def _require_number(value: object, name: str, minimum: float, exclusive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value <= minimum if exclusive else value < minimum:
        raise ValueError(f"{name} must be {'>' if exclusive else '>='} {minimum}")


def build_match_scheduler_config(args: argparse.Namespace) -> MatchSchedulerConfig:
    _apply_match_config(args)
    _validate_match_args(args)
    return MatchSchedulerConfig(
        mode=args.scheduler_mode,
        order=args.match_order,
        seed=args.match_seed,
        disconnect_policy=args.disconnect_policy,
        max_reconnects=args.max_reconnects,
        reconnect_delay=args.reconnect_delay,
        max_rounds=args.max_rounds,
        max_decision_ticks=args.max_decision_ticks,
        max_wall_seconds=args.max_wall_seconds,
        max_workers=args.max_workers,
    )


def _required_config_str(config: dict[str, Any], key: str, owner: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} field {key!r} must be a non-empty string")
    return value


def build_activity_profile(args: argparse.Namespace, registry: AgentRegistry | None = None) -> ActivityProfile:
    profile = activity_profile(args.activity, args.profile_objective)
    profile = _profile_with_action_overrides(profile, args)
    overrides = build_profile_overrides(args, registry)
    return replace(profile, **overrides) if overrides else profile


def build_activity_route_set(args: argparse.Namespace, registry: AgentRegistry | None = None) -> ActivityRouteSet:
    route_set = activity_route_set(args.route_set)
    overrides = build_profile_overrides(args, registry)
    default_overrides = dict(overrides)
    if getattr(args, "profile_objective", None):
        default_overrides["objective"] = args.profile_objective
    default_profile = _profile_with_action_overrides(route_set.default_profile, args)
    default_profile = replace(default_profile, **default_overrides) if default_overrides else default_profile
    routes = tuple(
        replace(
            route,
            profile=replace(_profile_with_action_overrides(route.profile, args), **overrides)
            if overrides
            else _profile_with_action_overrides(route.profile, args),
        )
        for route in route_set.routes
    )
    return replace(route_set, default_profile=default_profile, routes=routes)


def _profile_with_action_overrides(profile: ActivityProfile, args: argparse.Namespace) -> ActivityProfile:
    disabled_actions = _disabled_actions(args)
    if not disabled_actions:
        return profile
    allowed_actions = frozenset(
        action
        for action in profile.action_policy.allowed_actions
        if action not in disabled_actions
    )
    return replace(profile, action_policy=replace(profile.action_policy, allowed_actions=allowed_actions))


def _disabled_actions(args: argparse.Namespace) -> frozenset[str]:
    values = getattr(args, "disabled_actions", []) or []
    if isinstance(values, str):
        values = [values]
    if not all(isinstance(value, str) for value in values):
        raise ValueError("disabled_actions must be a list of action names")
    disabled = frozenset(values)
    unknown = disabled - (DEFAULT_ALLOWED_ACTIONS | {"send_raw"})
    if unknown:
        raise ValueError(f"unknown disabled action(s): {', '.join(sorted(unknown))}")
    return disabled


def match_participant_specs(args: argparse.Namespace) -> list[MatchParticipantSpec]:
    _apply_match_config(args)
    configured_specs = getattr(args, "_match_participants_config", None)
    if configured_specs is not None:
        specs = list(configured_specs)
    else:
        specs = [_parse_match_participant(value) for value in getattr(args, "participant", [])]
        specs.extend(MatchParticipantSpec(agent_id=agent_id) for agent_id in getattr(args, "agent_id", []))
    if len(specs) < 2:
        raise ValueError("run-match requires at least two --participant or --agent-id values")

    seen: set[str] = set()
    for spec in specs:
        if spec.agent_id in seen:
            raise ValueError(f"duplicate match agent_id: {spec.agent_id}")
        seen.add(spec.agent_id)
    return specs


def build_match_participants(
        args: argparse.Namespace,
        specs: list[MatchParticipantSpec],
        registry: AgentRegistry | None,
) -> list[MatchParticipantRuntime]:
    participants: list[MatchParticipantRuntime] = []
    log_paths: dict[Path, str] = {}
    for spec in specs:
        participant_args = _participant_args(args, spec, registry)
        opponents = [other.agent_id for other in specs if other.agent_id != spec.agent_id]
        profile = build_activity_profile(participant_args, registry)
        log_path = _agent_log_path(args.log_path, spec.agent_id)
        if log_path in log_paths:
            raise ValueError(
                f"agent ids {log_paths[log_path]!r} and {spec.agent_id!r} sanitize to the same "
                f"per-agent log path {log_path}; rename one of them"
            )
        log_paths[log_path] = spec.agent_id
        objective = _format_match_objective(args.run_objective or DEFAULT_MATCH_OBJECTIVE, spec.agent_id, opponents)
        evaluation_profile = _activity_evaluation_profile(profile.name)
        metrics_path = (
            _agent_log_path(getattr(args, "metrics_path", "runtime/metrics/match.jsonl"), spec.agent_id)
            if evaluation_profile is not None
            else None
        )
        runner = ActivityRunner(
            profile,
            log_path=log_path,
            run_objective=objective,
            evaluation_profile=evaluation_profile,
            evaluation_log_path=metrics_path,
        )
        participants.append(
            MatchParticipantRuntime(
                spec=spec,
                model=build_model(participant_args, registry),
                model_metadata=build_model_metadata(participant_args, registry),
                runner=runner,
                log_path=log_path,
                metrics_path=metrics_path,
            )
        )
    return participants


def build_profile_overrides(args: argparse.Namespace, registry: AgentRegistry | None = None) -> dict[str, object]:
    record = registry.maybe_get(args.agent_id) if registry is not None else None
    model_config = record.model if record is not None else {}
    provider = getattr(args, "provider", None) or _config_str(model_config, "provider") or "openai-compatible"
    overrides: dict[str, object] = {}
    if args.observe_timeout is not None:
        _require_number(args.observe_timeout, "observe_timeout", minimum=0, exclusive=True)
        overrides["observe_timeout"] = args.observe_timeout
    if args.stable_ms is not None:
        _require_int(args.stable_ms, "stable_ms", minimum=0)
        overrides["stable_ms"] = args.stable_ms
    if getattr(args, "byte_quiet_ms", None) is not None:
        _require_int(args.byte_quiet_ms, "byte_quiet_ms", minimum=0)
        overrides["byte_quiet_ms"] = args.byte_quiet_ms
    if getattr(args, "recent_steps_to_keep", None) is not None:
        _require_int(args.recent_steps_to_keep, "recent_steps_to_keep", minimum=0)
        overrides["recent_steps_to_keep"] = args.recent_steps_to_keep
    if getattr(args, "model_error_retries", None) is not None:
        _require_int(args.model_error_retries, "model_error_retries", minimum=0)
        overrides["model_error_retries"] = args.model_error_retries
    if getattr(args, "prompt_mode", None) is not None:
        overrides["prompt_mode"] = args.prompt_mode
    elif provider == "codex" and _codex_stateful(args, model_config):
        overrides["prompt_mode"] = "stateful_delta"
    elif provider == "claude" and _claude_stateful(args, model_config):
        overrides["prompt_mode"] = "stateful_delta"
    if getattr(args, "prompt_layout", None) is not None:
        overrides["prompt_layout"] = args.prompt_layout
    return overrides


def _parse_match_participant(value: str) -> MatchParticipantSpec:
    parts = value.split(":", 2)
    if len(parts) == 1:
        agent_id = parts[0].strip()
        if not agent_id:
            raise ValueError("match participant agent_id must not be empty")
        return MatchParticipantSpec(agent_id=agent_id)
    if len(parts) != 3:
        raise ValueError("match participant must be agent_id or agent_id:provider:model")
    agent_id, provider, model = (part.strip() for part in parts)
    if not agent_id or not provider or not model:
        raise ValueError("match participant must be agent_id or agent_id:provider:model")
    return MatchParticipantSpec(agent_id=agent_id, provider=provider, model=model)


def _participant_args(
        args: argparse.Namespace,
        spec: MatchParticipantSpec,
        registry: AgentRegistry | None = None,
) -> argparse.Namespace:
    data = vars(args).copy()
    data["agent_id"] = spec.agent_id
    if spec.config is not None:
        for key, value in spec.config.items():
            data[key.replace("-", "_")] = value
    if spec.provider is not None:
        data["provider"] = spec.provider
    if spec.model is not None:
        data["model"] = spec.model
    if "stateful" in data:
        provider = data.get("provider") or _registry_provider(registry, spec.agent_id)
        if provider == "codex":
            data["codex_stateful"] = bool(data["stateful"])
        elif provider == "claude":
            data["claude_stateful"] = bool(data["stateful"])
    return argparse.Namespace(**data)


def _registry_provider(registry: AgentRegistry | None, agent_id: str) -> str | None:
    record = registry.maybe_get(agent_id) if registry is not None else None
    if record is None:
        return None
    return _config_str(record.model, "provider")


def _format_match_objective(template: str, agent_id: str, opponents: list[str]) -> str:
    return template.replace("{agent_id}", agent_id).replace("{opponents}", ", ".join(opponents) or "none")


def _agent_log_path(match_log_path: str | Path, agent_id: str) -> Path:
    path = Path(match_log_path)
    safe_agent = "".join(char if char.isalnum() or char in "-_." else "_" for char in agent_id)
    suffix = path.suffix or ".jsonl"
    return path.with_name(f"{path.stem}.{safe_agent}{suffix}")


def _activity_evaluation_profile(activity_name: str) -> EvaluationProfile | None:
    return TW2_EVALUATION_PROFILE if activity_name == "tw2-game" else None


def _print_evaluation_summary(result: Any, metrics_path: str | Path) -> None:
    metrics = result.evaluation.final_metrics or result.evaluation.latest_metrics
    print(f"metrics={metrics_path} final={json.dumps(metrics, sort_keys=True, separators=(',', ':'))}")


def _match_result_summary(result: Any) -> str:
    summary = f"{result.agent_id}:steps={len(result.steps)} stop={result.stop_reason}"
    metrics = result.evaluation.final_metrics or result.evaluation.latest_metrics
    if "score" in metrics:
        summary += f" score={json.dumps(metrics['score'])}"
    return summary


def build_model(args: argparse.Namespace, registry: AgentRegistry | None):
    record = registry.maybe_get(args.agent_id) if registry is not None else None
    model_config = record.model if record is not None else {}
    provider = args.provider or _config_str(model_config, "provider") or "openai-compatible"

    if provider == "scripted":
        model = ScriptedModelAdapter(args.scripted_response or ['{"action": "wait", "arguments": {}}'])
    elif provider == "anthropic":
        model_name = args.model or _config_str(model_config, "model")
        if not model_name:
            raise ValueError("--model is required for anthropic provider")
        model = AnthropicAdapter(
            model=model_name,
            api_key=args.api_key or _config_secret(model_config, "api_key", "api_key_env"),
            base_url=args.base_url or _config_str(model_config, "base_url") or "https://api.anthropic.com/v1",
            temperature=_config_float(args.temperature, model_config, "temperature", 0.2),
            max_tokens=_config_int(args.max_tokens, model_config, "max_tokens", 512),
            cache_system_prompt=not args.no_anthropic_cache,
            output_filters=output_filters_for_model(
                model_name,
                args.response_filter or _config_str(model_config, "response_filter"),
            ),
        )
    elif provider == "codex":
        model = CodexCliAdapter(
            model=args.model or _config_str(model_config, "model"),
            profile=getattr(args, "codex_profile", None) or _config_str(model_config, "profile"),
            executable=getattr(args, "codex_executable", None) or _config_str(model_config, "executable") or "codex",
            timeout=_config_float(getattr(args, "codex_timeout", None), model_config, "timeout", 600.0),
            sandbox=getattr(args, "codex_sandbox", None) or _config_str(model_config, "sandbox") or "read-only",
            cwd=getattr(args, "codex_cwd", None) or _config_str(model_config, "cwd"),
            extra_args=_config_str_list(model_config, "extra_args") + (getattr(args, "codex_arg", []) or []),
            stateful=_codex_stateful(args, model_config),
            session_id=getattr(args, "codex_session_id", None) or _config_str(model_config, "session_id"),
            session_file=getattr(args, "codex_session_file", None) or _config_str(model_config, "session_file"),
            output_filters=output_filters_for_model(
                args.model or _config_str(model_config, "model") or "",
                args.response_filter or _config_str(model_config, "response_filter"),
            ),
        )
    elif provider == "claude":
        model = ClaudeCliAdapter(
            model=args.model or _config_str(model_config, "model"),
            executable=getattr(args, "claude_executable", None) or _config_str(model_config, "executable") or "claude",
            timeout=_config_float(getattr(args, "claude_timeout", None), model_config, "timeout", 600.0),
            cwd=getattr(args, "claude_cwd", None) or _config_str(model_config, "cwd"),
            extra_args=_config_str_list(model_config, "extra_args") + (getattr(args, "claude_arg", []) or []),
            stateful=_claude_stateful(args, model_config),
            session_id=getattr(args, "claude_session_id", None) or _config_str(model_config, "session_id"),
            session_file=getattr(args, "claude_session_file", None) or _config_str(model_config, "session_file"),
            permission_mode=getattr(args, "claude_permission_mode", None)
            or _config_str(model_config, "permission_mode")
            or "dontAsk",
            tools=_claude_tools(args, model_config),
            bare=_claude_bare(args, model_config),
            output_filters=output_filters_for_model(
                args.model or _config_str(model_config, "model") or "",
                args.response_filter or _config_str(model_config, "response_filter"),
            ),
        )
    elif provider == "openai-compatible":
        model_name = args.model or _config_str(model_config, "model")
        if not model_name:
            raise ValueError("--model is required for openai-compatible provider")
        model = OpenAICompatibleAdapter(
            model=model_name,
            base_url=args.base_url or _config_str(model_config, "base_url") or DEFAULT_OPENAI_BASE_URL,
            api_key=args.api_key or _config_secret(model_config, "api_key", "api_key_env"),
            temperature=_config_float(args.temperature, model_config, "temperature", 0.2),
            max_tokens=_config_int(args.max_tokens, model_config, "max_tokens", 512),
            extra_body=_config_dict(model_config, "extra_body"),
            output_filters=output_filters_for_model(
                model_name,
                args.response_filter or _config_str(model_config, "response_filter"),
            ),
        )
    else:
        raise ValueError(f"unknown model provider: {provider}")
    return model


def build_model_metadata(args: argparse.Namespace, registry: AgentRegistry | None) -> dict[str, object]:
    record = registry.maybe_get(args.agent_id) if registry is not None else None
    model_config = record.model if record is not None else {}
    provider = args.provider or _config_str(model_config, "provider") or "openai-compatible"

    if provider == "scripted":
        return {
            "provider": "scripted",
            "scripted_response_count": len(args.scripted_response or []),
        }
    if provider == "anthropic":
        model_name = args.model or _config_str(model_config, "model") or ""
        return {
            "provider": "anthropic",
            "model": model_name,
            "base_url": args.base_url or _config_str(model_config, "base_url") or "https://api.anthropic.com/v1",
            "temperature": _config_float(args.temperature, model_config, "temperature", 0.2),
            "max_tokens": _config_int(args.max_tokens, model_config, "max_tokens", 512),
            "cache_system_prompt": not args.no_anthropic_cache,
            "response_filter": args.response_filter or _config_str(model_config, "response_filter") or "auto",
        }
    if provider == "codex":
        model_name = args.model or _config_str(model_config, "model") or ""
        return _without_empty_values(
            {
                "provider": "codex",
                "model": model_name,
                "profile": getattr(args, "codex_profile", None) or _config_str(model_config, "profile"),
                "executable": getattr(args, "codex_executable", None)
                or _config_str(model_config, "executable")
                or "codex",
                "timeout": _config_float(getattr(args, "codex_timeout", None), model_config, "timeout", 600.0),
                "sandbox": getattr(args, "codex_sandbox", None) or _config_str(model_config, "sandbox") or "read-only",
                "cwd": getattr(args, "codex_cwd", None) or _config_str(model_config, "cwd"),
                "extra_args": _config_str_list(model_config, "extra_args") + (getattr(args, "codex_arg", []) or []),
                "stateful": _codex_stateful(args, model_config),
                "session_id": getattr(args, "codex_session_id", None) or _config_str(model_config, "session_id"),
                "session_file": getattr(args, "codex_session_file", None) or _config_str(model_config, "session_file"),
                "response_filter": args.response_filter or _config_str(model_config, "response_filter") or "auto",
            }
        )
    if provider == "claude":
        model_name = args.model or _config_str(model_config, "model") or ""
        return _without_empty_values(
            {
                "provider": "claude",
                "model": model_name,
                "executable": getattr(args, "claude_executable", None)
                or _config_str(model_config, "executable")
                or "claude",
                "timeout": _config_float(getattr(args, "claude_timeout", None), model_config, "timeout", 600.0),
                "cwd": getattr(args, "claude_cwd", None) or _config_str(model_config, "cwd"),
                "extra_args": _config_str_list(model_config, "extra_args") + (getattr(args, "claude_arg", []) or []),
                "stateful": _claude_stateful(args, model_config),
                "session_id": getattr(args, "claude_session_id", None) or _config_str(model_config, "session_id"),
                "session_file": getattr(args, "claude_session_file", None) or _config_str(model_config, "session_file"),
                "permission_mode": getattr(args, "claude_permission_mode", None)
                or _config_str(model_config, "permission_mode")
                or "dontAsk",
                "tools": _claude_tools(args, model_config),
                "bare": _claude_bare(args, model_config),
                "response_filter": args.response_filter or _config_str(model_config, "response_filter") or "auto",
            }
        )
    if provider == "openai-compatible":
        model_name = args.model or _config_str(model_config, "model") or ""
        return _without_empty_values(
            {
                "provider": "openai-compatible",
                "model": model_name,
                "base_url": args.base_url or _config_str(model_config, "base_url") or DEFAULT_OPENAI_BASE_URL,
                "temperature": _config_float(args.temperature, model_config, "temperature", 0.2),
                "max_tokens": _config_int(args.max_tokens, model_config, "max_tokens", 512),
                "extra_body": _config_dict(model_config, "extra_body"),
                "response_filter": args.response_filter or _config_str(model_config, "response_filter") or "auto",
            }
        )
    raise ValueError(f"unknown model provider: {provider}")


def accounts_list(args: argparse.Namespace) -> int:
    try:
        registry = load_agent_registry(args.agents_config, required=True)
    except AccountConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    assert registry is not None
    if args.json:
        print(json.dumps({"agents": registry.public_list()}, indent=2, sort_keys=True))
        return 0
    for record in registry.agents.values():
        password_state = "password=yes" if record.resolve_password() is not None else "password=missing"
        model_name = record.model.get("model", "")
        print(f"{record.agent_id}\talias={record.bbs_alias}\t{password_state}\tmodel={model_name}")
    return 0


def accounts_check(args: argparse.Namespace) -> int:
    try:
        registry = load_agent_registry(args.agents_config, required=True)
    except AccountConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    assert registry is not None
    errors = registry.check()
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"ok: {len(registry.agents)} account(s)")
    return 0


def accounts_provision(args: argparse.Namespace) -> int:
    try:
        registry = load_agent_registry(args.agents_config, required=True)
        assert registry is not None
        payload = registry.provision_payload()
    except AccountConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.dry_run:
        print(json.dumps({"agents": registry.public_list()}, indent=2, sort_keys=True))
        return 0

    runtime_tmp = Path("runtime/tmp")
    runtime_tmp.mkdir(parents=True, exist_ok=True)
    payload_path = runtime_tmp / "sbbs_agents.resolved.json"
    # The payload holds resolved plaintext BBS passwords, including ones the
    # config deliberately kept off disk via *_env keys; owner-only access.
    fd = os.open(payload_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    payload_path.chmod(0o600)
    script_path = Path("scripts/sbbs_provision_agents.js")
    commands = [
        ["docker", "compose", "cp", str(script_path), f"{args.compose_service}:/tmp/sbbs_provision_agents.js"],
        ["docker", "compose", "cp", str(payload_path), f"{args.compose_service}:/tmp/sbbs_agents.json"],
        [
            "docker",
            "compose",
            "exec",
            "-T",
            args.compose_service,
            "jsexec",
            "/tmp/sbbs_provision_agents.js",
            "/tmp/sbbs_agents.json",
        ],
    ]
    for command in commands:
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            print(f"command failed: {' '.join(command)}", file=sys.stderr)
            return result.returncode
    return 0


def _config_str(config: dict[str, Any], key: str) -> str | None:
    value = config.get(key)
    return value if isinstance(value, str) else None


def _config_secret(config: dict[str, Any], key: str, env_key: str) -> str | None:
    value = _config_str(config, key)
    if value is not None:
        return value
    env_name = _config_str(config, env_key)
    return None if env_name is None else os.environ.get(env_name)


def _config_dict(config: dict[str, Any], key: str) -> dict[str, object] | None:
    value = config.get(key)
    return dict(value) if isinstance(value, dict) else None


def _config_str_list(config: dict[str, Any], key: str) -> list[str]:
    value = config.get(key)
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    if isinstance(value, tuple):
        return [item for item in value if isinstance(item, str)]
    return []


def _config_bool(config: dict[str, Any], key: str, default: bool = False) -> bool:
    value = config.get(key)
    return value if isinstance(value, bool) else default


def _codex_stateful(args: argparse.Namespace, model_config: dict[str, Any]) -> bool:
    if getattr(args, "codex_stateful", False):
        return True
    if "stateful" in model_config:
        return _config_bool(model_config, "stateful")
    return False


def _claude_stateful(args: argparse.Namespace, model_config: dict[str, Any]) -> bool:
    if getattr(args, "claude_stateful", False):
        return True
    if "stateful" in model_config:
        return _config_bool(model_config, "stateful")
    return False


def _claude_bare(args: argparse.Namespace, model_config: dict[str, Any]) -> bool:
    if getattr(args, "claude_bare", False):
        return True
    if "bare" in model_config:
        return _config_bool(model_config, "bare")
    return False


def _claude_tools(args: argparse.Namespace, model_config: dict[str, Any]) -> str:
    value = getattr(args, "claude_tools", None)
    if value is not None:
        return value
    if "tools" in model_config:
        return _config_str(model_config, "tools") or ""
    # Claude's CLI enables its normal tool set when --tools is omitted. Keep
    # terminal-game sessions isolated by explicitly passing an empty tool list.
    return ""


def _without_empty_values(data: dict[str, object | None]) -> dict[str, object]:
    return {key: value for key, value in data.items() if value not in (None, "", [], {})}


def _config_float(value: float | None, config: dict[str, Any], key: str, default: float) -> float:
    if value is not None:
        return value
    config_value = config.get(key)
    return float(config_value) if isinstance(config_value, (int, float)) else default


def _config_int(value: int | None, config: dict[str, Any], key: str, default: int) -> int:
    if value is not None:
        return value
    config_value = config.get(key)
    return int(config_value) if isinstance(config_value, int) else default


def _add_connection_args(parser: argparse.ArgumentParser) -> None:
    """Transport and account-registry options shared by every session command."""

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2323)
    parser.add_argument("--rlogin-port", type=int, default=2513)
    parser.add_argument("--rlogin-terminal", default="ansi")
    parser.add_argument("--transport", choices=["telnet", "rlogin"], default="telnet")
    parser.add_argument("--telnet-enter", choices=["cr", "lf", "crlf"], default="cr")
    parser.add_argument("--agents-config", default=str(DEFAULT_AGENTS_CONFIG))


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    """Model provider options shared by every session command."""

    parser.add_argument("--provider", choices=["openai-compatible", "anthropic", "claude", "codex", "scripted"])
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--no-anthropic-cache", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--scripted-response", action="append", default=[])
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--response-filter", choices=["auto", "default", "gemma4", "none"])
    parser.add_argument("--codex-profile")
    parser.add_argument("--codex-executable")
    parser.add_argument("--codex-timeout", type=float)
    parser.add_argument("--codex-sandbox", choices=["read-only", "workspace-write", "danger-full-access"])
    parser.add_argument("--codex-cwd")
    parser.add_argument("--codex-arg", action="append", default=[])
    parser.add_argument("--codex-stateful", action="store_true")
    parser.add_argument("--codex-session-id")
    parser.add_argument("--codex-session-file")
    parser.add_argument("--claude-executable")
    parser.add_argument("--claude-timeout", type=float)
    parser.add_argument("--claude-cwd")
    parser.add_argument("--claude-arg", action="append", default=[])
    parser.add_argument("--claude-stateful", action="store_true")
    parser.add_argument("--claude-session-id")
    parser.add_argument("--claude-session-file")
    parser.add_argument(
        "--claude-permission-mode",
        choices=["acceptEdits", "auto", "bypassPermissions", "default", "dontAsk", "plan"],
    )
    parser.add_argument("--claude-tools")
    parser.add_argument("--claude-bare", action="store_true")


def _add_runner_args(
        parser: argparse.ArgumentParser,
        profile_objective_help: str,
        run_objective_help: str,
        max_decision_ticks: int,
        max_wall_seconds: float,
        log_path: str,
        metrics_path: str,
        disabled_actions_help: str,
        decision_ticks_help: str | None = None,
        wall_seconds_help: str | None = None,
) -> None:
    """Budget, prompt-shaping, and logging options shared by every session command."""

    if wall_seconds_help is None:
        wall_seconds_help = "soft wall-clock admission budget; a decision already in flight finishes before stopping"
    parser.add_argument("--profile-objective", help=profile_objective_help)
    parser.add_argument("--run-objective", help=run_objective_help)
    parser.add_argument("--max-decision-ticks", type=int, default=max_decision_ticks, help=decision_ticks_help)
    parser.add_argument("--max-wall-seconds", type=float, default=max_wall_seconds, help=wall_seconds_help)
    parser.add_argument("--observe-timeout", type=float)
    parser.add_argument("--stable-ms", type=int)
    parser.add_argument("--byte-quiet-ms", type=int)
    parser.add_argument("--recent-steps-to-keep", type=int)
    parser.add_argument("--model-error-retries", type=int)
    parser.add_argument("--prompt-mode", choices=["stateless_full", "stateful_delta"])
    parser.add_argument("--prompt-layout", choices=["timeline_first", "cache_friendly"])
    parser.add_argument(
        "--disable-action",
        dest="disabled_actions",
        action="append",
        default=[],
        help=disabled_actions_help,
    )
    parser.add_argument("--log-path", default=log_path)
    parser.add_argument(
        "--metrics-path",
        default=metrics_path,
        help="JSONL evaluator log path; run-match derives one per-participant path from this base",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bbs-gym")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke_parser = subparsers.add_parser("smoke", help="connect and print the initial BBS screen")
    smoke_parser.add_argument("--host", default="127.0.0.1")
    smoke_parser.add_argument("--port", type=int, default=2323)
    smoke_parser.add_argument("--timeout", type=float, default=10.0)
    smoke_parser.add_argument("--seconds", type=float, default=3.0)
    smoke_parser.add_argument("--tail", type=int, default=4000)
    smoke_parser.add_argument("--transcript", default="runtime/transcripts/smoke.raw")
    smoke_parser.set_defaults(func=smoke)

    turn_parser = subparsers.add_parser("observe-turn", help="connect and wait for a stable screen or prompt")
    turn_parser.add_argument("--host", default="127.0.0.1")
    turn_parser.add_argument("--port", type=int, default=2323)
    turn_parser.add_argument("--timeout", type=float, default=10.0)
    turn_parser.add_argument("--stable-ms", type=int, default=300)
    turn_parser.add_argument("--byte-quiet-ms", type=int, default=0)
    turn_parser.add_argument("--poll-interval", type=float, default=0.05)
    turn_parser.add_argument("--prompt-fast-path", action="store_true")
    turn_parser.add_argument("--tail", type=int, default=4000)
    turn_parser.add_argument("--agent-id", default="smoke")
    turn_parser.add_argument("--profile", choices=["bbs", "tw2"], default="bbs")
    turn_parser.add_argument("--transcript", default="runtime/transcripts/observe-turn.raw")
    turn_parser.set_defaults(func=observe_turn)

    run_parser = subparsers.add_parser("run-activity", help="run a bounded model-driven BBS activity")
    _add_connection_args(run_parser)
    run_parser.add_argument("--agent-id", default="agent-001")
    run_parser.add_argument("--node", type=int)
    _add_model_args(run_parser)
    run_parser.add_argument("--activity", default="bbs-main-menu")
    _add_runner_args(
        run_parser,
        profile_objective_help="override the selected profile's built-in objective",
        run_objective_help="stable session goal included in prompts without replacing profile-specific guidance",
        max_decision_ticks=20,
        max_wall_seconds=300.0,
        log_path="runtime/logs/activity.jsonl",
        metrics_path="runtime/metrics/activity.jsonl",
        disabled_actions_help=(
            "remove an action from the activity schema for this run; repeatable, e.g. --disable-action hangup"
        ),
    )
    run_parser.set_defaults(func=run_activity)

    routed_parser = subparsers.add_parser("run-routed", help="run a model-driven BBS activity with profile routing")
    _add_connection_args(routed_parser)
    routed_parser.add_argument("--agent-id", default="agent-001")
    routed_parser.add_argument("--node", type=int)
    _add_model_args(routed_parser)
    routed_parser.add_argument("--route-set", choices=activity_route_set_names(), default="tw2-auto")
    _add_runner_args(
        routed_parser,
        profile_objective_help="override the default profile's built-in objective",
        run_objective_help="stable session goal included across routed profile switches",
        max_decision_ticks=50,
        max_wall_seconds=600.0,
        log_path="runtime/logs/routed-activity.jsonl",
        metrics_path="runtime/metrics/routed-activity.jsonl",
        disabled_actions_help=(
            "remove an action from the activity schema for this run; repeatable, e.g. --disable-action hangup"
        ),
    )
    routed_parser.set_defaults(func=run_routed)

    match_parser = subparsers.add_parser("run-match", help="run a scheduled multi-agent BBS activity")
    match_parser.add_argument("--match-config", help="TOML or JSON file describing a multi-agent match")
    _add_connection_args(match_parser)
    match_parser.add_argument(
        "--no-agents-config",
        action="store_true",
        help="ignore the agent registry; useful for standalone telnet games with inline participants",
    )
    match_parser.add_argument(
        "--participant",
        action="append",
        default=[],
        help="match participant as agent_id or agent_id:provider:model; repeat for each player",
    )
    match_parser.add_argument(
        "--agent-id",
        action="append",
        default=[],
        help="agent id loaded from --agents-config; repeat for each player",
    )
    _add_model_args(match_parser)
    match_parser.add_argument("--activity", default="bbs-door-line")
    match_parser.add_argument(
        "--max-rounds",
        type=int,
        default=50,
        help="maximum scheduled rounds; in continuous mode this is the maximum queued action count",
    )
    match_parser.add_argument("--scheduler-mode", choices=sorted(get_args(MatchSchedulerMode)), default="sequential")
    match_parser.add_argument("--match-order", choices=sorted(get_args(MatchOrder)), default="fixed")
    match_parser.add_argument("--match-seed", type=int)
    match_parser.add_argument("--disconnect-policy", choices=sorted(get_args(DisconnectPolicy)), default="stop")
    match_parser.add_argument("--max-reconnects", type=int, default=3)
    match_parser.add_argument("--reconnect-delay", type=float, default=2.0)
    match_parser.add_argument("--max-workers", type=int)
    _add_runner_args(
        match_parser,
        profile_objective_help="override the selected profile's built-in objective",
        run_objective_help="match objective template; supports {agent_id} and {opponents}",
        max_decision_ticks=50,
        max_wall_seconds=600.0,
        log_path="runtime/logs/match.jsonl",
        metrics_path="runtime/metrics/match.jsonl",
        disabled_actions_help=(
            "remove an action from every participant's activity schema; repeatable, e.g. --disable-action hangup"
        ),
        decision_ticks_help="maximum committed decision ticks per participant",
        wall_seconds_help=(
            "soft match-level admission budget shared by all participants; in-flight decisions finish before stopping"
        ),
    )
    match_parser.set_defaults(func=run_match)

    accounts_parser = subparsers.add_parser("accounts", help="manage BBS agent account registry")
    accounts_subparsers = accounts_parser.add_subparsers(dest="accounts_command", required=True)

    accounts_list_parser = accounts_subparsers.add_parser("list", help="list configured agent accounts")
    accounts_list_parser.add_argument("--agents-config", default=str(DEFAULT_AGENTS_CONFIG))
    accounts_list_parser.add_argument("--json", action="store_true")
    accounts_list_parser.set_defaults(func=accounts_list)

    accounts_check_parser = accounts_subparsers.add_parser("check", help="validate configured agent accounts")
    accounts_check_parser.add_argument("--agents-config", default=str(DEFAULT_AGENTS_CONFIG))
    accounts_check_parser.set_defaults(func=accounts_check)

    accounts_provision_parser = accounts_subparsers.add_parser(
        "provision",
        help="create or update Synchronet users from the agent registry",
    )
    accounts_provision_parser.add_argument("--agents-config", default=str(DEFAULT_AGENTS_CONFIG))
    accounts_provision_parser.add_argument("--compose-service", default="bbs")
    accounts_provision_parser.add_argument("--dry-run", action="store_true")
    accounts_provision_parser.set_defaults(func=accounts_provision)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
