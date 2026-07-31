"""Run Tele-Arena through bbs-gym's BBS door-line profile.

This example expects an Ether/Tele-Arena telnet server already listening.
Ether's telnet input path expects LF for Enter, so this wrapper defaults to
``--telnet-enter lf`` while leaving the generic bbs-gym defaults unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bbs_gym.cli import main as bbs_gym_main


DEFAULT_RUN_OBJECTIVE = (
    "Play Tele-Arena through this telnet session. If asked for a character name, create or log in as "
    "ArenaCodex. Stay connected unless the run objective explicitly says to leave. Survive fights, gain "
    "experience and gold, buy and equip useful starter supplies, spend gold wisely, recover when hurt, and "
    "keep making progress instead of quitting early."
)


def main(argv: list[str] | None = None) -> int:
    args, passthrough = parse_args(argv)
    return bbs_gym_main([*build_bbs_gym_argv(args), *passthrough])


def build_bbs_gym_argv(args: argparse.Namespace) -> list[str]:
    cmd = [
        "run-activity",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--transport",
        "telnet",
        "--telnet-enter",
        args.telnet_enter,
        "--agent-id",
        args.agent_id,
        "--provider",
        args.provider,
        "--model",
        args.model,
        "--activity",
        args.activity,
        "--run-objective",
        args.run_objective,
        "--max-decision-ticks",
        str(args.max_decision_ticks),
        "--max-wall-seconds",
        str(args.max_wall_seconds),
        "--observe-timeout",
        str(args.observe_timeout),
        "--stable-ms",
        str(args.stable_ms),
        "--byte-quiet-ms",
        str(args.byte_quiet_ms),
        "--log-path",
        str(args.log_path),
    ]
    if args.base_url:
        cmd.extend(["--base-url", args.base_url])
    if args.api_key:
        cmd.extend(["--api-key", args.api_key])
    if args.temperature is not None:
        cmd.extend(["--temperature", str(args.temperature)])
    if args.max_tokens is not None:
        cmd.extend(["--max-tokens", str(args.max_tokens)])
    if args.response_filter:
        cmd.extend(["--response-filter", args.response_filter])
    if args.codex_profile:
        cmd.extend(["--codex-profile", args.codex_profile])
    if args.codex_executable:
        cmd.extend(["--codex-executable", args.codex_executable])
    if args.codex_timeout is not None:
        cmd.extend(["--codex-timeout", str(args.codex_timeout)])
    if args.codex_sandbox:
        cmd.extend(["--codex-sandbox", args.codex_sandbox])
    if args.codex_cwd:
        cmd.extend(["--codex-cwd", args.codex_cwd])
    for extra_arg in args.codex_arg:
        cmd.extend(["--codex-arg", extra_arg])
    if args.codex_stateful:
        cmd.append("--codex-stateful")
    if args.codex_session_file:
        cmd.extend(["--codex-session-file", str(args.codex_session_file)])
    if args.claude_executable:
        cmd.extend(["--claude-executable", args.claude_executable])
    if args.claude_timeout is not None:
        cmd.extend(["--claude-timeout", str(args.claude_timeout)])
    if args.claude_cwd:
        cmd.extend(["--claude-cwd", args.claude_cwd])
    for extra_arg in args.claude_arg:
        cmd.extend(["--claude-arg", extra_arg])
    if args.claude_stateful:
        cmd.append("--claude-stateful")
    if args.claude_session_file:
        cmd.extend(["--claude-session-file", str(args.claude_session_file)])
    if args.claude_permission_mode:
        cmd.extend(["--claude-permission-mode", args.claude_permission_mode])
    if args.claude_tools is not None:
        cmd.extend(["--claude-tools", args.claude_tools])
    if args.claude_bare:
        cmd.append("--claude-bare")
    if args.prompt_mode:
        cmd.extend(["--prompt-mode", args.prompt_mode])
    if args.prompt_layout:
        cmd.extend(["--prompt-layout", args.prompt_layout])
    if args.recent_steps_to_keep is not None:
        cmd.extend(["--recent-steps-to-keep", str(args.recent_steps_to_keep)])
    for disabled_action in args.disabled_actions:
        cmd.extend(["--disable-action", disabled_action])
    return cmd


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--telnet-enter", choices=["cr", "lf", "crlf"], default="lf")
    parser.add_argument("--agent-id", default="tele-arena-codex")
    parser.add_argument("--activity", choices=["bbs-door-safe", "bbs-door-line"], default="bbs-door-line")
    parser.add_argument(
        "--provider",
        choices=["openai-compatible", "fireworks", "xai", "claude", "codex"],
        default="codex",
    )
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--response-filter", choices=["auto", "default", "gemma4", "none"])
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
    parser.add_argument("--claude-executable")
    parser.add_argument("--claude-timeout", type=float)
    parser.add_argument("--claude-cwd")
    parser.add_argument("--claude-arg", action="append", default=[])
    parser.add_argument("--claude-stateful", action="store_true")
    parser.add_argument("--claude-session-file", type=Path)
    parser.add_argument(
        "--claude-permission-mode",
        choices=["acceptEdits", "auto", "bypassPermissions", "default", "dontAsk", "plan"],
    )
    parser.add_argument("--claude-tools")
    parser.add_argument("--claude-bare", action="store_true")
    parser.add_argument("--prompt-mode", choices=["stateless_full", "stateful_delta"])
    parser.add_argument("--prompt-layout", choices=["timeline_first", "cache_friendly"])
    parser.add_argument("--recent-steps-to-keep", type=int)
    parser.add_argument("--disable-action", dest="disabled_actions", action="append", default=["hangup"])
    parser.add_argument("--max-decision-ticks", type=int, default=100)
    parser.add_argument("--max-wall-seconds", type=float, default=2400.0)
    parser.add_argument("--observe-timeout", type=float, default=8.0)
    parser.add_argument("--stable-ms", type=int, default=300)
    parser.add_argument("--byte-quiet-ms", type=int, default=0)
    parser.add_argument("--log-path", type=Path, default=Path("runtime/logs/tele-arena-codex-bbs-door-line-lf.jsonl"))
    parser.add_argument("--run-objective", default=DEFAULT_RUN_OBJECTIVE)
    return parser.parse_known_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
