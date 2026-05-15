"""Command-line helpers for BBS gym sessions."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from tty_agent.ansi import strip_ansi
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
from tty_agent.transports.telnet import TelnetSession

from .accounts import AccountConfigError, AgentRegistry, load_agent_registry
from .activities import activity_profile
from .env import BbsGym
from .profiles import BBS_PROFILE, TW2_PROFILE
from .routing import ActivityRouteSet, activity_route_set, activity_route_set_names


DEFAULT_AGENTS_CONFIG = Path("config/agents.local.json")
DEFAULT_OPENAI_BASE_URL = "http://localhost:11434/v1"


def smoke(args: argparse.Namespace) -> int:
    transcript = Path(args.transcript) if args.transcript else None
    try:
        with TelnetSession(args.host, args.port, args.timeout, transcript, encoding="cp437") as session:
            data = session.read(args.seconds)
    except OSError as exc:
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
    except OSError as exc:
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
    except (AccountConfigError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    profile = build_activity_profile(args, registry)
    runner = ActivityRunner(profile, log_path=args.log_path, run_objective=args.run_objective or "")

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
    except (OSError, AccountConfigError, ValueError) as exc:
        print(f"connection failed: {exc}", file=sys.stderr)
        return 1

    print(f"activity={result.activity} agent={result.agent_id} steps={len(result.steps)} stop={result.stop_reason}")
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
    except (OSError, AccountConfigError, ValueError) as exc:
        print(f"connection failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"route_set={route_set.name} agent={result.agent_id} steps={len(result.steps)} "
        f"stop={result.stop_reason} final_profile={runner.profile.name}"
    )
    return 0


def build_activity_profile(args: argparse.Namespace, registry: AgentRegistry | None = None) -> ActivityProfile:
    profile = activity_profile(args.activity, args.profile_objective)
    overrides = build_profile_overrides(args, registry)
    return replace(profile, **overrides) if overrides else profile


def build_activity_route_set(args: argparse.Namespace, registry: AgentRegistry | None = None) -> ActivityRouteSet:
    route_set = activity_route_set(args.route_set)
    overrides = build_profile_overrides(args, registry)
    default_overrides = dict(overrides)
    if getattr(args, "profile_objective", None):
        default_overrides["objective"] = args.profile_objective
    default_profile = (
        replace(route_set.default_profile, **default_overrides) if default_overrides else route_set.default_profile
    )
    routes = tuple(
        replace(route, profile=replace(route.profile, **overrides) if overrides else route.profile)
        for route in route_set.routes
    )
    return replace(route_set, default_profile=default_profile, routes=routes)


def build_profile_overrides(args: argparse.Namespace, registry: AgentRegistry | None = None) -> dict[str, object]:
    record = registry.maybe_get(args.agent_id) if registry is not None else None
    model_config = record.model if record is not None else {}
    provider = getattr(args, "provider", None) or _config_str(model_config, "provider") or "openai-compatible"
    overrides: dict[str, object] = {}
    if args.observe_timeout is not None:
        overrides["observe_timeout"] = args.observe_timeout
    if args.stable_ms is not None:
        overrides["stable_ms"] = args.stable_ms
    if getattr(args, "byte_quiet_ms", None) is not None:
        overrides["byte_quiet_ms"] = args.byte_quiet_ms
    if getattr(args, "recent_steps_to_keep", None) is not None:
        overrides["recent_steps_to_keep"] = args.recent_steps_to_keep
    if getattr(args, "prompt_mode", None) is not None:
        overrides["prompt_mode"] = args.prompt_mode
    elif provider == "codex" and _codex_stateful(args, model_config):
        overrides["prompt_mode"] = "stateful_delta"
    elif provider == "claude" and _claude_stateful(args, model_config):
        overrides["prompt_mode"] = "stateful_delta"
    if getattr(args, "prompt_layout", None) is not None:
        overrides["prompt_layout"] = args.prompt_layout
    return overrides


def build_model(args: argparse.Namespace, registry: AgentRegistry | None):
    record = registry.maybe_get(args.agent_id) if registry is not None else None
    model_config = record.model if record is not None else {}
    provider = args.provider or _config_str(model_config, "provider") or "openai-compatible"

    if provider == "scripted":
        model = ScriptedModelAdapter(args.scripted_response or ['{"action": "wait", "arguments": {}}'])
    elif provider == "anthropic":
        model_name = args.model or _config_str(model_config, "model")
        if not model_name:
            print("--model is required for anthropic provider", file=sys.stderr)
            raise ValueError("--model is required for anthropic provider")
        model = AnthropicAdapter(
            model=model_name,
            api_key=args.api_key or _config_secret(model_config, "api_key", "api_key_env"),
            base_url=args.base_url or _config_str(model_config, "base_url") or "https://api.anthropic.com/v1",
            temperature=_config_float(args.temperature, model_config, "temperature", 0.2),
            max_tokens=_config_int(args.max_tokens, model_config, "max_tokens", 512),
            cache_system_prompt=not args.no_anthropic_cache,
        )
    elif provider == "codex":
        model = CodexCliAdapter(
            model=args.model or _config_str(model_config, "model"),
            profile=getattr(args, "codex_profile", None) or _config_str(model_config, "profile"),
            executable=getattr(args, "codex_executable", None) or _config_str(model_config, "executable") or "codex",
            timeout=_config_float(getattr(args, "codex_timeout", None), model_config, "timeout", 300.0),
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
            timeout=_config_float(getattr(args, "claude_timeout", None), model_config, "timeout", 300.0),
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
                "timeout": _config_float(getattr(args, "codex_timeout", None), model_config, "timeout", 300.0),
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
                "timeout": _config_float(getattr(args, "claude_timeout", None), model_config, "timeout", 300.0),
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
    payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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


def _claude_tools(args: argparse.Namespace, model_config: dict[str, Any]) -> str | None:
    value = getattr(args, "claude_tools", None)
    if value is not None:
        return value
    if "tools" in model_config:
        return _config_str(model_config, "tools")
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
    run_parser.add_argument("--host", default="127.0.0.1")
    run_parser.add_argument("--port", type=int, default=2323)
    run_parser.add_argument("--rlogin-port", type=int, default=2513)
    run_parser.add_argument("--rlogin-terminal", default="ansi")
    run_parser.add_argument("--transport", choices=["telnet", "rlogin"], default="telnet")
    run_parser.add_argument("--telnet-enter", choices=["cr", "lf", "crlf"], default="cr")
    run_parser.add_argument("--agents-config", default=str(DEFAULT_AGENTS_CONFIG))
    run_parser.add_argument("--agent-id", default="agent-001")
    run_parser.add_argument("--node", type=int)
    run_parser.add_argument("--provider", choices=["openai-compatible", "anthropic", "claude", "codex", "scripted"])
    run_parser.add_argument("--base-url")
    run_parser.add_argument("--api-key")
    run_parser.add_argument("--no-anthropic-cache", action="store_true")
    run_parser.add_argument("--model")
    run_parser.add_argument("--scripted-response", action="append", default=[])
    run_parser.add_argument("--temperature", type=float)
    run_parser.add_argument("--max-tokens", type=int)
    run_parser.add_argument("--response-filter", choices=["auto", "default", "gemma4", "none"])
    run_parser.add_argument("--codex-profile")
    run_parser.add_argument("--codex-executable")
    run_parser.add_argument("--codex-timeout", type=float)
    run_parser.add_argument("--codex-sandbox", choices=["read-only", "workspace-write", "danger-full-access"])
    run_parser.add_argument("--codex-cwd")
    run_parser.add_argument("--codex-arg", action="append", default=[])
    run_parser.add_argument("--codex-stateful", action="store_true")
    run_parser.add_argument("--codex-session-id")
    run_parser.add_argument("--codex-session-file")
    run_parser.add_argument("--claude-executable")
    run_parser.add_argument("--claude-timeout", type=float)
    run_parser.add_argument("--claude-cwd")
    run_parser.add_argument("--claude-arg", action="append", default=[])
    run_parser.add_argument("--claude-stateful", action="store_true")
    run_parser.add_argument("--claude-session-id")
    run_parser.add_argument("--claude-session-file")
    run_parser.add_argument(
        "--claude-permission-mode", choices=["acceptEdits", "auto", "bypassPermissions", "default", "dontAsk", "plan"]
    )
    run_parser.add_argument("--claude-tools")
    run_parser.add_argument("--claude-bare", action="store_true")
    run_parser.add_argument("--activity", default="bbs-main-menu")
    run_parser.add_argument(
        "--profile-objective",
        help="override the selected profile's built-in objective",
    )
    run_parser.add_argument(
        "--run-objective",
        help="stable session goal included in prompts without replacing profile-specific guidance",
    )
    run_parser.add_argument("--max-decision-ticks", type=int, default=20)
    run_parser.add_argument("--max-wall-seconds", type=float, default=300.0)
    run_parser.add_argument("--observe-timeout", type=float)
    run_parser.add_argument("--stable-ms", type=int)
    run_parser.add_argument("--byte-quiet-ms", type=int)
    run_parser.add_argument("--recent-steps-to-keep", type=int)
    run_parser.add_argument("--prompt-mode", choices=["stateless_full", "stateful_delta"])
    run_parser.add_argument("--prompt-layout", choices=["timeline_first", "cache_friendly"])
    run_parser.add_argument("--log-path", default="runtime/logs/activity.jsonl")
    run_parser.set_defaults(func=run_activity)

    routed_parser = subparsers.add_parser("run-routed", help="run a model-driven BBS activity with profile routing")
    routed_parser.add_argument("--host", default="127.0.0.1")
    routed_parser.add_argument("--port", type=int, default=2323)
    routed_parser.add_argument("--rlogin-port", type=int, default=2513)
    routed_parser.add_argument("--rlogin-terminal", default="ansi")
    routed_parser.add_argument("--transport", choices=["telnet", "rlogin"], default="telnet")
    routed_parser.add_argument("--telnet-enter", choices=["cr", "lf", "crlf"], default="cr")
    routed_parser.add_argument("--agents-config", default=str(DEFAULT_AGENTS_CONFIG))
    routed_parser.add_argument("--agent-id", default="agent-001")
    routed_parser.add_argument("--node", type=int)
    routed_parser.add_argument("--provider", choices=["openai-compatible", "anthropic", "claude", "codex", "scripted"])
    routed_parser.add_argument("--base-url")
    routed_parser.add_argument("--api-key")
    routed_parser.add_argument("--no-anthropic-cache", action="store_true")
    routed_parser.add_argument("--model")
    routed_parser.add_argument("--scripted-response", action="append", default=[])
    routed_parser.add_argument("--temperature", type=float)
    routed_parser.add_argument("--max-tokens", type=int)
    routed_parser.add_argument("--response-filter", choices=["auto", "default", "gemma4", "none"])
    routed_parser.add_argument("--codex-profile")
    routed_parser.add_argument("--codex-executable")
    routed_parser.add_argument("--codex-timeout", type=float)
    routed_parser.add_argument("--codex-sandbox", choices=["read-only", "workspace-write", "danger-full-access"])
    routed_parser.add_argument("--codex-cwd")
    routed_parser.add_argument("--codex-arg", action="append", default=[])
    routed_parser.add_argument("--codex-stateful", action="store_true")
    routed_parser.add_argument("--codex-session-id")
    routed_parser.add_argument("--codex-session-file")
    routed_parser.add_argument("--claude-executable")
    routed_parser.add_argument("--claude-timeout", type=float)
    routed_parser.add_argument("--claude-cwd")
    routed_parser.add_argument("--claude-arg", action="append", default=[])
    routed_parser.add_argument("--claude-stateful", action="store_true")
    routed_parser.add_argument("--claude-session-id")
    routed_parser.add_argument("--claude-session-file")
    routed_parser.add_argument(
        "--claude-permission-mode",
        choices=["acceptEdits", "auto", "bypassPermissions", "default", "dontAsk", "plan"],
    )
    routed_parser.add_argument("--claude-tools")
    routed_parser.add_argument("--claude-bare", action="store_true")
    routed_parser.add_argument("--route-set", choices=activity_route_set_names(), default="tw2-auto")
    routed_parser.add_argument(
        "--profile-objective",
        help="override the default profile's built-in objective",
    )
    routed_parser.add_argument(
        "--run-objective",
        help="stable session goal included across routed profile switches",
    )
    routed_parser.add_argument("--max-decision-ticks", type=int, default=50)
    routed_parser.add_argument("--max-wall-seconds", type=float, default=600.0)
    routed_parser.add_argument("--observe-timeout", type=float)
    routed_parser.add_argument("--stable-ms", type=int)
    routed_parser.add_argument("--byte-quiet-ms", type=int)
    routed_parser.add_argument("--recent-steps-to-keep", type=int)
    routed_parser.add_argument("--prompt-mode", choices=["stateless_full", "stateful_delta"])
    routed_parser.add_argument("--prompt-layout", choices=["timeline_first", "cache_friendly"])
    routed_parser.add_argument("--log-path", default="runtime/logs/routed-activity.jsonl")
    routed_parser.set_defaults(func=run_routed)

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
