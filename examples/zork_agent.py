"""Run a local Z-machine story file such as Zork through the tty-agent PTY path."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from tty_agent.actions import Action
from tty_agent.agent import TerminalSessionAgent
from tty_agent.profiles import TEXT_ADVENTURE_PROFILE
from tty_agent.terminal import TerminalScreen, TurnObserver
from tty_agent.transports.pty import PtySession


def main() -> None:
    args = parse_args()
    story_path = args.story.expanduser()
    if not story_path.exists():
        raise SystemExit(
            f"story file not found: {story_path}\n"
            "Put a local Z-code story file under runtime/zcode/; story/game data is intentionally not bundled."
        )
    if shutil.which(args.interpreter) is None:
        raise SystemExit(
            f"interpreter not found: {args.interpreter!r}\n"
            "Install a terminal Z-machine interpreter such as frotz, then rerun this example."
        )

    transcript_path = args.transcript.expanduser()
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    command = [args.interpreter, *args.interpreter_arg, str(story_path)]
    session = PtySession(
        command,
        columns=args.columns,
        lines=args.lines,
        transcript_path=transcript_path,
        env={
            "TERM": args.term,
            "PATH": os.environ.get("PATH", ""),
        },
    )

    with session:
        observer = TurnObserver(
            args.agent_id,
            session,
            terminal=TerminalScreen(columns=session.columns, lines=session.lines),
            profile=TEXT_ADVENTURE_PROFILE,
            metadata={
                "transport": "pty",
                "program": args.interpreter,
                "story": str(story_path),
                "encoding": session.encoding,
            },
        )
        agent = TerminalSessionAgent(args.agent_id, session, observer, observer.metadata)

        observation = agent.observe_turn(
            timeout=args.observe_timeout,
            stable_ms=args.stable_ms,
            byte_quiet_ms=args.byte_quiet_ms,
            prompt_fast_path=args.prompt_fast_path,
        )
        print_observation("initial", observation.model_text, args.tail_chars)

        for move in args.move:
            agent.act_action(Action("submit_line", text=move))
            observation = agent.observe_turn(
                timeout=args.observe_timeout,
                stable_ms=args.stable_ms,
                byte_quiet_ms=args.byte_quiet_ms,
                prompt_fast_path=args.prompt_fast_path,
            )
            print_observation(move, observation.model_text, args.tail_chars)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "story",
        type=Path,
        help="path to a local Z-code story file, e.g. runtime/zcode/zork1.z3",
    )
    parser.add_argument("--interpreter", default="frotz")
    parser.add_argument("--interpreter-arg", action="append", default=[])
    parser.add_argument("--agent-id", default="zork-pty")
    parser.add_argument("--transcript", type=Path, default=Path("runtime/transcripts/zork.raw"))
    parser.add_argument("--term", default="xterm-256color")
    parser.add_argument("--columns", type=int, default=100)
    parser.add_argument("--lines", type=int, default=30)
    parser.add_argument("--observe-timeout", type=float, default=5.0)
    parser.add_argument("--stable-ms", type=int, default=300)
    parser.add_argument("--byte-quiet-ms", type=int, default=0)
    parser.add_argument("--tail-chars", type=int, default=3000)
    parser.add_argument("--prompt-fast-path", action="store_true")
    parser.add_argument(
        "--move",
        action="append",
        default=[],
        help="scripted command to submit after startup; repeat for multiple moves",
    )
    return parser.parse_args()


def print_observation(label: str, text: str, tail_chars: int) -> None:
    print(f"\n== {label} ==")
    print(text[-tail_chars:])


if __name__ == "__main__":
    main()
