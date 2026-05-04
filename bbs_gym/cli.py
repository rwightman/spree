"""Command-line helpers for BBS gym sessions."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from .activities import activity_profile
from .env import BbsGym
from .models import AnthropicAdapter, OpenAICompatibleAdapter, ScriptedModelAdapter
from .ansi import strip_ansi
from .profiles import BBS_PROFILE, TW2_PROFILE
from .runner import ActivityBudget, ActivityProfile, ActivityRunner
from .telnet import TelnetSession
from .terminal import TurnObserver


def smoke(args: argparse.Namespace) -> int:
    transcript = Path(args.transcript) if args.transcript else None
    try:
        with TelnetSession(args.host, args.port, args.timeout, transcript) as session:
            data = session.read(args.seconds)
    except OSError as exc:
        print(f"connection failed: {exc}", file=sys.stderr)
        return 1

    text = strip_ansi(data)
    print(text[-args.tail :])
    return 0 if data else 2


def observe_turn(args: argparse.Namespace) -> int:
    transcript = Path(args.transcript) if args.transcript else None
    profile = TW2_PROFILE if args.profile == "tw2" else BBS_PROFILE
    try:
        with TelnetSession(args.host, args.port, args.timeout, transcript) as session:
            observer = TurnObserver(args.agent_id, session, profile=profile)
            observation = observer.observe_turn(
                timeout=args.timeout,
                stable_ms=args.stable_ms,
                poll_interval=args.poll_interval,
                prompt_fast_path=args.prompt_fast_path,
            )
    except OSError as exc:
        print(f"connection failed: {exc}", file=sys.stderr)
        return 1

    print(observation.model_text[-args.tail :])
    if observation.matched_prompt:
        print(f"\n[matched_prompt={observation.matched_prompt} stable_ms={observation.stable_ms}]")
    if observation.timed_out:
        print(f"\n[timed_out stable_ms={observation.stable_ms}]", file=sys.stderr)
        return 2
    return 0


def run_activity(args: argparse.Namespace) -> int:
    if args.provider == "scripted":
        model = ScriptedModelAdapter(args.scripted_response or ['{"action": "wait"}'])
    elif args.provider == "anthropic":
        if not args.model:
            print("--model is required for anthropic provider", file=sys.stderr)
            return 2
        model = AnthropicAdapter(
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            cache_system_prompt=not args.no_anthropic_cache,
        )
    else:
        if not args.model:
            print("--model is required for openai-compatible provider", file=sys.stderr)
            return 2
        model = OpenAICompatibleAdapter(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

    profile = activity_profile(args.activity, args.objective)
    if type(profile) is ActivityProfile:
        profile = replace(
            profile,
            observe_timeout=args.observe_timeout if args.observe_timeout is not None else profile.observe_timeout,
            stable_ms=args.stable_ms if args.stable_ms is not None else profile.stable_ms,
        )
    runner = ActivityRunner(profile, log_path=args.log_path)

    try:
        with BbsGym(host=args.host, port=args.port) as gym:
            agent = gym.connect(args.agent_id, node=args.node)
            result = runner.run(
                agent,
                model,
                ActivityBudget(
                    max_decision_ticks=args.max_decision_ticks,
                    max_wall_seconds=args.max_wall_seconds,
                ),
            )
    except OSError as exc:
        print(f"connection failed: {exc}", file=sys.stderr)
        return 1

    print(f"activity={result.activity} agent={result.agent_id} steps={len(result.steps)} stop={result.stop_reason}")
    return 0


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
    run_parser.add_argument("--agent-id", default="agent-001")
    run_parser.add_argument("--node", type=int)
    run_parser.add_argument("--provider", choices=["openai-compatible", "anthropic", "scripted"], default="openai-compatible")
    run_parser.add_argument("--base-url", default="http://localhost:11434/v1")
    run_parser.add_argument("--api-key")
    run_parser.add_argument("--no-anthropic-cache", action="store_true")
    run_parser.add_argument("--model")
    run_parser.add_argument("--scripted-response", action="append", default=[])
    run_parser.add_argument("--temperature", type=float, default=0.2)
    run_parser.add_argument("--max-tokens", type=int, default=512)
    run_parser.add_argument("--activity", default="bbs-main-menu")
    run_parser.add_argument("--objective")
    run_parser.add_argument("--max-decision-ticks", type=int, default=20)
    run_parser.add_argument("--max-wall-seconds", type=float, default=300.0)
    run_parser.add_argument("--observe-timeout", type=float)
    run_parser.add_argument("--stable-ms", type=int)
    run_parser.add_argument("--log-path", default="runtime/logs/activity.jsonl")
    run_parser.set_defaults(func=run_activity)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
