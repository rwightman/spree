"""Command-line helpers for BBS gym sessions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .ansi import strip_ansi
from .profiles import BBS_PROFILE, TW2_PROFILE
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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
