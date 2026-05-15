import importlib.util
from pathlib import Path


_MODULE_PATH = Path(__file__).parent.parent / "examples" / "tele_arena_activity.py"
_SPEC = importlib.util.spec_from_file_location("tele_arena_activity", _MODULE_PATH)
assert _SPEC is not None
tele_arena_activity = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(tele_arena_activity)

DEFAULT_RUN_OBJECTIVE = tele_arena_activity.DEFAULT_RUN_OBJECTIVE
build_bbs_gym_argv = tele_arena_activity.build_bbs_gym_argv
parse_args = tele_arena_activity.parse_args


def _option(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def test_tele_arena_example_defaults_to_bbs_door_line_with_lf_enter():
    args, passthrough = parse_args([])

    argv = build_bbs_gym_argv(args)

    assert passthrough == []
    assert argv[0] == "run-activity"
    assert _option(argv, "--activity") == "bbs-door-line"
    assert _option(argv, "--transport") == "telnet"
    assert _option(argv, "--telnet-enter") == "lf"
    assert _option(argv, "--provider") == "codex"
    assert _option(argv, "--model") == "gpt-5.5"
    assert _option(argv, "--run-objective") == DEFAULT_RUN_OBJECTIVE


def test_tele_arena_example_can_select_safe_activity():
    args, _passthrough = parse_args(["--activity", "bbs-door-safe"])

    argv = build_bbs_gym_argv(args)

    assert _option(argv, "--activity") == "bbs-door-safe"


def test_tele_arena_example_forwards_model_and_codex_options():
    args, passthrough = parse_args(
        [
            "--host",
            "localhost",
            "--port",
            "3001",
            "--provider",
            "openai-compatible",
            "--model",
            "gemma4",
            "--base-url",
            "http://127.0.0.1:8000/v1",
            "--api-key",
            "local",
            "--temperature",
            "0.6",
            "--max-tokens",
            "4096",
            "--response-filter",
            "gemma4",
            "--codex-stateful",
            "--codex-session-file",
            "runtime/codex-sessions/tele-arena.session",
            "--screen-tail-chars",
            "2000",
        ]
    )

    argv = build_bbs_gym_argv(args)

    assert passthrough == ["--screen-tail-chars", "2000"]
    assert _option(argv, "--host") == "localhost"
    assert _option(argv, "--port") == "3001"
    assert _option(argv, "--provider") == "openai-compatible"
    assert _option(argv, "--model") == "gemma4"
    assert _option(argv, "--base-url") == "http://127.0.0.1:8000/v1"
    assert _option(argv, "--api-key") == "local"
    assert _option(argv, "--temperature") == "0.6"
    assert _option(argv, "--max-tokens") == "4096"
    assert _option(argv, "--response-filter") == "gemma4"
    assert "--codex-stateful" in argv
    assert _option(argv, "--codex-session-file") == "runtime/codex-sessions/tele-arena.session"


def test_tele_arena_example_forwards_claude_options():
    args, passthrough = parse_args(
        [
            "--provider",
            "claude",
            "--model",
            "sonnet",
            "--claude-stateful",
            "--claude-session-file",
            "runtime/claude-sessions/tele-arena.session",
            "--claude-timeout",
            "120",
            "--claude-permission-mode",
            "dontAsk",
            "--claude-tools",
            "",
            "--claude-bare",
        ]
    )

    argv = build_bbs_gym_argv(args)

    assert passthrough == []
    assert _option(argv, "--provider") == "claude"
    assert _option(argv, "--model") == "sonnet"
    assert "--claude-stateful" in argv
    assert _option(argv, "--claude-session-file") == "runtime/claude-sessions/tele-arena.session"
    assert _option(argv, "--claude-timeout") == "120.0"
    assert _option(argv, "--claude-permission-mode") == "dontAsk"
    assert _option(argv, "--claude-tools") == ""
    assert "--claude-bare" in argv
