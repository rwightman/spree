"""Command-line helpers for BBS gym sessions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .ansi import strip_ansi
from .telnet import TelnetSession


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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

