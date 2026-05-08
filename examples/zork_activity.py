"""Run a model-driven Zork activity through the tty-agent PTY path."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from tty_agent.actions import ActionPolicy
from tty_agent.agent import TerminalSessionAgent
from tty_agent.hints import InputModalityProfile, InputModeRule
from tty_agent.memory import JsonMemoryStore
from tty_agent.models import OpenAICompatibleAdapter, output_filters_for_model
from tty_agent.profiles import TEXT_ADVENTURE_PROFILE
from tty_agent.prompt_modules import GENERIC_TERMINAL_MODULES, StaticPromptModule
from tty_agent.runner import ActivityBudget, ActivityProfile, ActivityRunner
from tty_agent.terminal import TerminalScreen, TurnObserver
from tty_agent.transports.pty import PtySession


DEFAULT_INTERPRETER_ARGS = ("-p", "-w", "100", "-h", "30")

TEXT_ADVENTURE_GUIDANCE = StaticPromptModule(
    name="text_adventure.conventions",
    level="game_interface",
    text=(
        "This is parser-based interactive fiction. Use short imperative commands such as look, inventory, "
        "north, open mailbox, take lamp, read leaflet, or examine object. Movement commands are often "
        "directions like north, south, east, west, up, and down. Prefer submit_line with one complete command. "
        "If the game asks a yes/no question, submit_line y or submit_line n."
    ),
)

TEXT_ADVENTURE_INPUT_MODALITY = InputModalityProfile(
    rules=(
        InputModeRule.from_pattern(
            mode="line_input_expected",
            pattern=r">\s*$",
            hint="submit one complete parser command with submit_line",
            priority=10,
        ),
        InputModeRule.from_pattern(
            mode="line_input_expected",
            pattern=r"(?:yes/no|y/n|affirmative).*[:?]\s*$",
            hint="submit y or n with submit_line",
            priority=20,
        ),
    )
)


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

    args.transcript.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [args.interpreter, *args.interpreter_arg, str(story_path)]
    session = PtySession(
        command,
        columns=args.columns,
        lines=args.lines,
        transcript_path=args.transcript,
        env={
            "TERM": args.term,
            "PATH": os.environ.get("PATH", ""),
        },
    )
    model = OpenAICompatibleAdapter(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        output_filters=output_filters_for_model(args.model, args.response_filter),
    )
    profile = ActivityProfile(
        name="zork",
        objective=args.objective,
        action_policy=ActionPolicy(
            allowed_actions=frozenset({"submit_line", "press_key", "wait", "hangup"}),
            max_text_chars=160,
            max_line_chars=160,
            require_encoding="utf-8",
        ),
        observe_timeout=args.observe_timeout,
        stable_ms=args.stable_ms,
        byte_quiet_ms=args.byte_quiet_ms,
        recent_steps_to_keep=args.recent_steps_to_keep,
        screen_tail_chars=args.screen_tail_chars,
        compact_every_steps=args.compact_every_steps,
        prompt_layout=args.prompt_layout,
        input_modality_profile=TEXT_ADVENTURE_INPUT_MODALITY,
        prompt_modules=GENERIC_TERMINAL_MODULES + (TEXT_ADVENTURE_GUIDANCE,),
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
                "model": args.model,
            },
        )
        agent = TerminalSessionAgent(args.agent_id, session, observer, observer.metadata)
        runner = ActivityRunner(
            profile,
            memory_store=JsonMemoryStore(args.memory_root),
            log_path=args.log_path,
            run_objective=args.run_objective,
        )
        result = runner.run(
            agent,
            model,
            ActivityBudget(max_decision_ticks=args.max_decision_ticks, max_wall_seconds=args.max_wall_seconds),
        )

    print(f"activity={result.activity} agent={result.agent_id} stop_reason={result.stop_reason}")
    print(f"steps={len(result.steps)} log={args.log_path} transcript={args.transcript}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("story", type=Path, help="path to a local Z-code story file")
    parser.add_argument("--interpreter", default="dfrotz")
    parser.add_argument(
        "--interpreter-arg",
        action="append",
        default=None,
        help="argument passed before the story file; repeat for multiple arguments",
    )
    parser.add_argument("--agent-id", default="zork-gemma4")
    parser.add_argument("--model", default="gemma4")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--response-filter", choices=["auto", "default", "gemma4", "none"], default="gemma4")
    parser.add_argument("--transcript", type=Path, default=Path("runtime/transcripts/zork-activity.raw"))
    parser.add_argument("--log-path", type=Path, default=Path("runtime/logs/zork-activity.jsonl"))
    parser.add_argument("--memory-root", type=Path, default=Path("runtime/memory"))
    parser.add_argument("--term", default="xterm-256color")
    parser.add_argument("--columns", type=int, default=100)
    parser.add_argument("--lines", type=int, default=30)
    parser.add_argument("--observe-timeout", type=float, default=5.0)
    parser.add_argument("--stable-ms", type=int, default=300)
    parser.add_argument("--byte-quiet-ms", type=int, default=0)
    parser.add_argument("--recent-steps-to-keep", type=int, default=5)
    parser.add_argument("--screen-tail-chars", type=int, default=1600)
    parser.add_argument("--compact-every-steps", type=int, default=12)
    parser.add_argument("--prompt-layout", choices=["timeline_first", "cache_friendly"], default="cache_friendly")
    parser.add_argument("--max-decision-ticks", type=int, default=20)
    parser.add_argument("--max-wall-seconds", type=float, default=900.0)
    parser.add_argument(
        "--objective",
        default=(
            "Play Zork through normal text-adventure commands. Explore carefully, gather useful objects, "
            "avoid obvious danger, and make concrete progress."
        ),
    )
    parser.add_argument(
        "--run-objective",
        default=(
            "Play Zork I. Explore the starting area, collect useful items, learn exits, and avoid death. "
            "Use normal parser commands and recover from mistakes."
        ),
    )
    args = parser.parse_args()
    if args.interpreter_arg is None:
        args.interpreter_arg = list(DEFAULT_INTERPRETER_ARGS)
    return args


if __name__ == "__main__":
    main()
