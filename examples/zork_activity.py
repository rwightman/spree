"""Run a model-driven Zork activity through the tty-agent PTY path."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

from tty_agent.actions import Action, ActionPolicy
from tty_agent.agent import TerminalSessionAgent
from tty_agent.evaluation import EvaluationProbe, EvaluationProfile
from tty_agent.hints import InputModalityProfile, InputModeRule
from tty_agent.memory import JsonMemoryStore
from tty_agent.structured_memory import StructuredMemorySubsystem
from tty_agent.models import (
    ClaudeCliAdapter,
    CodexCliAdapter,
    OpenAICompatibleAdapter,
    ResponsesCompatibleAdapter,
    output_filters_for_model,
)
from tty_agent.profiles import TEXT_ADVENTURE_PROFILE
from tty_agent.prompt_modules import GENERIC_TERMINAL_MODULES, StaticPromptModule
from tty_agent.runner import ActivityBudget, ActivityProfile, ActivityRunner
from tty_agent.terminal import Observation, TerminalScreen, TurnObserver
from tty_agent.transports.pty import PtySession


DEFAULT_INTERPRETER_ARGS = ("-p", "-w", "100", "-h", "30")
ZORK_SCORE_RE = re.compile(
    r"Your score is\s+(?P<score>-?\d+)\s+\(total of\s+(?P<score_max>\d+)\s+points\),\s+"
    r"in\s+(?P<moves>\d+)\s+moves?\.\s+This gives you the rank of\s+(?P<rank>[^.\r\n]+)\.",
    re.IGNORECASE,
)

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


def extract_zork_metrics(observation: Observation) -> dict[str, object] | None:
    """Extract the newest standard Zork score response from terminal output."""

    matches = list(ZORK_SCORE_RE.finditer(observation.new_text))
    if not matches:
        return None
    match = matches[-1]
    return {
        "score": int(match.group("score")),
        "score_max": int(match.group("score_max")),
        "moves": int(match.group("moves")),
        "rank": match.group("rank").strip(),
    }


def zork_score_probe_ready(observation: Observation) -> bool:
    """Only issue the evaluator probe at Zork's normal command prompt."""

    return observation.matched_prompt == "command-prompt"


ZORK_EVALUATION_PROFILE = EvaluationProfile(
    name="zork-score",
    extractor=extract_zork_metrics,
    final_probe=EvaluationProbe(
        name="score",
        action=Action(action="submit_line", text="score"),
        ready=zork_score_probe_ready,
        turn_cost="none",
        observe_timeout=5.0,
        stable_ms=50,
        byte_quiet_ms=0,
        prompt_fast_path=True,
    ),
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
    output_filters = output_filters_for_model(args.model, args.response_filter)
    if args.provider == "codex":
        model = CodexCliAdapter(
            model=args.model,
            profile=args.codex_profile,
            executable=args.codex_executable,
            timeout=args.codex_timeout,
            sandbox=args.codex_sandbox,
            cwd=args.codex_cwd,
            extra_args=args.codex_arg,
            stateful=args.codex_stateful,
            session_file=args.codex_session_file,
            output_filters=output_filters,
        )
    elif args.provider == "claude":
        model = ClaudeCliAdapter(
            model=args.model,
            executable=args.claude_executable,
            timeout=args.claude_timeout,
            cwd=args.claude_cwd,
            extra_args=args.claude_arg,
            stateful=args.claude_stateful,
            session_file=args.claude_session_file,
            permission_mode=args.claude_permission_mode,
            tools=args.claude_tools,
            bare=args.claude_bare,
            output_filters=output_filters,
        )
    else:
        api_key = args.api_key
        if args.api_key_env:
            api_key = os.environ.get(args.api_key_env)
            if not api_key:
                raise SystemExit(f"{args.api_key_env} is not set or is empty")
        common_options = {
            "model": args.model,
            "base_url": args.base_url,
            "api_key": api_key,
            "timeout": args.model_timeout,
            "temperature": args.temperature,
            "audit_temperature": args.audit_temperature,
            "max_tokens": args.max_tokens,
            "compaction_max_tokens": args.compaction_max_tokens,
            "memory_max_tokens": args.memory_max_tokens,
            "max_tokens_retry_ceiling": args.max_tokens_retry_ceiling,
            "compaction_max_tokens_retry_ceiling": args.compaction_max_tokens_retry_ceiling,
            "memory_max_tokens_retry_ceiling": args.memory_max_tokens_retry_ceiling,
            "extra_body": args.extra_body_json,
            "extra_headers": dict(args.extra_header),
            "compaction_reasoning": args.compaction_reasoning,
            "compaction_extra_body": args.compaction_extra_body_json,
            "memory_reasoning": args.memory_reasoning,
            "memory_extra_body": args.memory_extra_body_json,
            "output_filters": output_filters,
        }
        if args.model_api == "responses":
            model = ResponsesCompatibleAdapter(
                **common_options,
                stateful=args.responses_stateful,
                response_id=args.responses_response_id,
                state_file=args.responses_state_file,
                resume=args.responses_resume,
            )
        else:
            model = OpenAICompatibleAdapter(**common_options)
    prompt_mode = args.prompt_mode or (
        "stateful_delta" if args.codex_stateful or args.claude_stateful or args.responses_stateful else "stateless_full"
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
        prompt_fast_path=args.prompt_fast_path,
        recent_steps_to_keep=args.recent_steps_to_keep,
        screen_tail_chars=args.screen_tail_chars,
        compact_every_steps=args.compact_every_steps,
        prompt_mode=prompt_mode,
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
                "model_api": args.model_api if args.provider == "openai-compatible" else None,
                "model_stateful": (
                    args.responses_stateful
                    if args.provider == "openai-compatible" and args.model_api == "responses"
                    else args.codex_stateful or args.claude_stateful
                ),
            },
        )
        agent = TerminalSessionAgent(args.agent_id, session, observer, observer.metadata)
        memory_subsystem = StructuredMemorySubsystem(args.memory_root) if args.memory_system == "structured" else None
        runner = ActivityRunner(
            profile,
            memory_store=JsonMemoryStore(args.memory_root),
            log_path=args.log_path,
            run_objective=args.run_objective,
            evaluation_profile=ZORK_EVALUATION_PROFILE,
            evaluation_log_path=args.metrics_path,
            memory_subsystem=memory_subsystem,
            memory_context_id="zork",
        )
        result = runner.run(
            agent,
            model,
            ActivityBudget(max_decision_ticks=args.max_decision_ticks, max_wall_seconds=args.max_wall_seconds),
        )

    print(f"activity={result.activity} agent={result.agent_id} stop_reason={result.stop_reason}")
    print(
        f"decision_ticks={result.decision_ticks} records={len(result.steps)} "
        f"log={args.log_path} transcript={args.transcript}"
    )
    final_metrics = result.evaluation.final_metrics or result.evaluation.latest_metrics
    print(f"metrics={args.metrics_path} final={json.dumps(final_metrics, sort_keys=True, separators=(',', ':'))}")


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
    parser.add_argument("--provider", choices=["openai-compatible", "claude", "codex"], default="openai-compatible")
    parser.add_argument("--model", default="gemma4")
    parser.add_argument(
        "--model-api",
        choices=["chat_completions", "responses"],
        default="chat_completions",
        help="HTTP API used by the openai-compatible provider",
    )
    parser.add_argument(
        "--responses-stateful",
        action="store_true",
        help="chain decision calls with previous_response_id; utility calls remain stateless",
    )
    parser.add_argument("--responses-response-id")
    parser.add_argument("--responses-state-file", type=Path)
    parser.add_argument(
        "--responses-resume",
        action="store_true",
        help="resume once from --responses-state-file; ordinary bootstraps start fresh chains",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="local")
    parser.add_argument(
        "--api-key-env",
        help="read the API key from this environment variable instead of placing it on the command line",
    )
    parser.add_argument(
        "--extra-header",
        action="append",
        type=parse_header,
        default=[],
        metavar="NAME=VALUE",
        help="additional HTTP request header; repeat for multiple headers",
    )
    parser.add_argument(
        "--extra-body-json",
        type=parse_json_object,
        default={},
        metavar="JSON",
        help="JSON object merged into each Chat Completions or Responses request body",
    )
    parser.add_argument(
        "--compaction-reasoning",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="request provider-specific reasoning control for compaction; omitted inherits decision settings",
    )
    parser.add_argument(
        "--compaction-extra-body-json",
        type=parse_json_object,
        default={},
        metavar="JSON",
        help="JSON object applied only to compaction requests",
    )
    parser.add_argument(
        "--memory-reasoning",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="request provider-specific reasoning control for final memory calls; omitted inherits decision settings",
    )
    parser.add_argument(
        "--memory-extra-body-json",
        type=parse_json_object,
        default={},
        metavar="JSON",
        help="JSON object applied only to final memory requests",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument(
        "--audit-temperature",
        type=float,
        help="sampling temperature for the structured final-memory audit; omitted inherits --temperature",
    )
    parser.add_argument(
        "--model-timeout",
        type=float,
        default=600.0,
        help="HTTP model-request timeout in seconds (including utility calls)",
    )
    parser.add_argument("--max-tokens", type=int, default=4096, help="initial decision output-token budget")
    parser.add_argument(
        "--max-tokens-retry-ceiling",
        type=int,
        default=16_384,
        help="maximum adaptive output-token budget for a truncated decision",
    )
    parser.add_argument(
        "--compaction-max-tokens",
        type=int,
        default=16_384,
        help="initial output-token budget for periodic compaction/reconciliation calls",
    )
    parser.add_argument(
        "--memory-max-tokens",
        type=int,
        default=32_768,
        help="initial output-token budget for final durable-memory calls",
    )
    parser.add_argument(
        "--compaction-max-tokens-retry-ceiling",
        type=int,
        default=32_768,
        help="maximum adaptive output-token budget for truncated compaction/reconciliation calls",
    )
    parser.add_argument(
        "--memory-max-tokens-retry-ceiling",
        type=int,
        default=32_768,
        help="maximum adaptive output-token budget for truncated final-memory calls",
    )
    parser.add_argument("--response-filter", choices=["auto", "default", "gemma4", "none"], default="auto")
    parser.add_argument("--codex-profile")
    parser.add_argument("--codex-executable", default="codex")
    parser.add_argument("--codex-timeout", type=float, default=300.0)
    parser.add_argument(
        "--codex-sandbox",
        choices=["read-only", "workspace-write", "danger-full-access"],
        default="read-only",
    )
    parser.add_argument("--codex-cwd")
    parser.add_argument("--codex-arg", action="append", default=[])
    parser.add_argument("--codex-stateful", action="store_true")
    parser.add_argument("--codex-session-file", type=Path)
    parser.add_argument("--claude-executable", default="claude")
    parser.add_argument("--claude-timeout", type=float, default=300.0)
    parser.add_argument("--claude-cwd")
    parser.add_argument("--claude-arg", action="append", default=[])
    parser.add_argument("--claude-stateful", action="store_true")
    parser.add_argument("--claude-session-file", type=Path)
    parser.add_argument(
        "--claude-permission-mode",
        choices=["acceptEdits", "auto", "bypassPermissions", "default", "dontAsk", "plan"],
        default="dontAsk",
    )
    parser.add_argument("--claude-tools", default="")
    parser.add_argument("--claude-bare", action="store_true")
    parser.add_argument("--transcript", type=Path, default=Path("runtime/transcripts/zork-activity.raw"))
    parser.add_argument("--log-path", type=Path, default=Path("runtime/logs/zork-activity.jsonl"))
    parser.add_argument("--metrics-path", type=Path, default=Path("runtime/metrics/zork-activity.jsonl"))
    parser.add_argument("--memory-root", type=Path, default=Path("runtime/memory"))
    parser.add_argument(
        "--memory-system",
        choices=["legacy", "structured"],
        default="legacy",
        help="memory subsystem for this run (docs/memory-simple.md)",
    )
    parser.add_argument("--term", default="xterm-256color")
    parser.add_argument("--columns", type=int, default=100)
    parser.add_argument("--lines", type=int, default=30)
    parser.add_argument("--observe-timeout", type=float, default=5.0)
    parser.add_argument("--stable-ms", type=int, default=50)
    parser.add_argument("--byte-quiet-ms", type=int, default=0)
    parser.add_argument(
        "--prompt-fast-path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="return as soon as the text-adventure prompt regex matches; use --no-prompt-fast-path to wait for stability",
    )
    parser.add_argument("--recent-steps-to-keep", type=int, default=5)
    parser.add_argument("--screen-tail-chars", type=int, default=1600)
    parser.add_argument("--compact-every-steps", type=int, default=12)
    parser.add_argument("--prompt-mode", choices=["stateless_full", "stateful_delta"])
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
    if args.responses_stateful and args.model_api != "responses":
        parser.error("--responses-stateful requires --model-api responses")
    if (
        args.responses_response_id is not None or args.responses_state_file is not None or args.responses_resume
    ) and not args.responses_stateful:
        parser.error("Responses state options require --responses-stateful")
    if args.responses_resume and args.responses_state_file is None and args.responses_response_id is None:
        parser.error("--responses-resume requires --responses-state-file or --responses-response-id")
    if args.prompt_mode == "stateful_delta" and not (
        args.codex_stateful or args.claude_stateful or args.responses_stateful
    ):
        parser.error("--prompt-mode stateful_delta requires a stateful model adapter")
    return args


def parse_header(value: str) -> tuple[str, str]:
    """Parse one ``NAME=VALUE`` command-line header."""

    name, separator, header_value = value.partition("=")
    if not separator or not name.strip():
        raise argparse.ArgumentTypeError("header must use NAME=VALUE syntax")
    return name.strip(), header_value


def parse_json_object(value: str) -> dict[str, object]:
    """Parse a JSON object used for provider-specific request options."""

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object")
    return parsed


if __name__ == "__main__":
    main()
