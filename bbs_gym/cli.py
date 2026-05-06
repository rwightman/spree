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

from terminal_agent.ansi import strip_ansi
from terminal_agent.models import AnthropicAdapter, OpenAICompatibleAdapter, ScriptedModelAdapter
from terminal_agent.models import output_filters_for_model
from terminal_agent.runner import ActivityBudget, ActivityProfile, ActivityRunner
from terminal_agent.terminal import TerminalScreen, TurnObserver
from terminal_agent.transports.telnet import TelnetSession

from .accounts import AccountConfigError, AgentRegistry, load_agent_registry
from .activities import activity_profile
from .env import BbsGym
from .profiles import BBS_PROFILE, TW2_PROFILE


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
                poll_interval=args.poll_interval,
                prompt_fast_path=args.prompt_fast_path,
            )
    except OSError as exc:
        print(f"connection failed: {exc}", file=sys.stderr)
        return 1

    print(observation.model_text[-args.tail:])
    if observation.matched_prompt:
        print(f"\n[matched_prompt={observation.matched_prompt} stable_ms={observation.stable_ms}]")
    if observation.timed_out:
        print(f"\n[timed_out stable_ms={observation.stable_ms}]", file=sys.stderr)
        return 2
    return 0


def run_activity(args: argparse.Namespace) -> int:
    try:
        registry = load_agent_registry(args.agents_config, required=False)
        model = build_model(args, registry)
    except (AccountConfigError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    profile = build_activity_profile(args)
    runner = ActivityRunner(profile, log_path=args.log_path)

    try:
        with BbsGym(
            host=args.host,
            port=args.port,
            rlogin_port=args.rlogin_port,
            rlogin_terminal=args.rlogin_terminal,
            transport=args.transport,
            agent_registry=registry,
        ) as gym:
            agent = gym.connect(args.agent_id, node=args.node)
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


def build_activity_profile(args: argparse.Namespace) -> ActivityProfile:
    profile = activity_profile(args.activity, args.objective)
    overrides: dict[str, object] = {}
    if args.observe_timeout is not None:
        overrides["observe_timeout"] = args.observe_timeout
    if args.stable_ms is not None:
        overrides["stable_ms"] = args.stable_ms
    return replace(profile, **overrides) if overrides else profile


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
    else:
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
    return model


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
    run_parser.add_argument("--agents-config", default=str(DEFAULT_AGENTS_CONFIG))
    run_parser.add_argument("--agent-id", default="agent-001")
    run_parser.add_argument("--node", type=int)
    run_parser.add_argument("--provider", choices=["openai-compatible", "anthropic", "scripted"])
    run_parser.add_argument("--base-url")
    run_parser.add_argument("--api-key")
    run_parser.add_argument("--no-anthropic-cache", action="store_true")
    run_parser.add_argument("--model")
    run_parser.add_argument("--scripted-response", action="append", default=[])
    run_parser.add_argument("--temperature", type=float)
    run_parser.add_argument("--max-tokens", type=int)
    run_parser.add_argument("--response-filter", choices=["auto", "default", "gemma4", "none"])
    run_parser.add_argument("--activity", default="bbs-main-menu")
    run_parser.add_argument("--objective")
    run_parser.add_argument("--max-decision-ticks", type=int, default=20)
    run_parser.add_argument("--max-wall-seconds", type=float, default=300.0)
    run_parser.add_argument("--observe-timeout", type=float)
    run_parser.add_argument("--stable-ms", type=int)
    run_parser.add_argument("--log-path", default="runtime/logs/activity.jsonl")
    run_parser.set_defaults(func=run_activity)

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
