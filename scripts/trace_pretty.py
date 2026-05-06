"""Render a BBS gym JSONL trace as readable text."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def visible_controls(text: str) -> str:
    out: list[str] = []
    for char in text:
        code = ord(char)
        if char == "\r":
            out.append(r"\r")
        elif char == "\n":
            out.append(r"\n")
            out.append("\n")
        elif char == "\t":
            out.append(r"\t")
        elif char == "\x1b":
            out.append(r"\x1b")
        elif code < 32 or code == 127:
            out.append(f"\\x{code:02x}")
        else:
            out.append(char)
    return "".join(out)


def text_field(step: dict[str, Any], field: str) -> str:
    observation = step.get("observation")
    if not isinstance(observation, dict):
        return ""
    value = observation.get(field, "")
    return value if isinstance(value, str) else ""


def render_step(
        step: dict[str, Any],
        *,
        screen_field: str,
        show_new_text: bool,
        show_controls: bool,
        show_prompt: bool,
        show_model_response: bool,
) -> str:
    lines = [
        f"=== Step {step.get('step', '?')} ===",
        f"action: {json.dumps(step.get('action'), sort_keys=True)}",
    ]

    observation = step.get("observation")
    if isinstance(observation, dict):
        lines.append(
            "ready: "
            f"{observation.get('ready_reason')} "
            f"stable_ms={observation.get('stable_ms')} "
            f"matched_prompt={observation.get('matched_prompt')}"
        )
        lines.append(f"cursor: {observation.get('cursor')}")

    validation = step.get("validation")
    if isinstance(validation, dict):
        summary = {key: validation[key] for key in ("accepted", "notes") if key in validation}
        lines.append(f"validation: {json.dumps(summary, sort_keys=True)}")

    screen = text_field(step, screen_field)
    if show_controls:
        screen = visible_controls(screen)
    lines.extend(["", f"{screen_field}:", screen])

    if show_new_text and screen_field != "new_text":
        new_text = text_field(step, "new_text")
        if show_controls:
            new_text = visible_controls(new_text)
        lines.extend(["", "new_text:", new_text])

    if show_model_response and isinstance(validation, dict):
        model_response = validation.get("model_response")
        if isinstance(model_response, dict):
            reasoning = model_response.get("reasoning", "")
            raw = model_response.get("response", "")
            parsed = model_response.get("parsed_response", "")
            if isinstance(reasoning, str) and reasoning:
                lines.extend(["", "reasoning:", visible_controls(reasoning) if show_controls else reasoning])
            if isinstance(raw, str):
                lines.extend(["", "model_response:", visible_controls(raw) if show_controls else raw])
            if isinstance(parsed, str):
                lines.extend(["", "parsed_response:", visible_controls(parsed) if show_controls else parsed])

    if show_prompt:
        prompt = step.get("prompt")
        if isinstance(prompt, dict):
            lines.extend(["", "system_prompt:", str(prompt.get("system", ""))])
            lines.extend(["", "user_prompt:", str(prompt.get("user", ""))])

    return "\n".join(lines)


def iter_steps(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--screen-field", choices=["model_text", "pretty_screen", "new_text"], default="model_text")
    parser.add_argument("--show-new-text", action="store_true", help="also print observation.new_text")
    parser.add_argument("--show-controls", action="store_true", help=r"render CR/ESC/control bytes as \r, \x1b, etc.")
    parser.add_argument("--show-prompt", action="store_true")
    parser.add_argument("--show-model-response", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    rendered = "\n\n".join(
        render_step(
            step,
            screen_field=args.screen_field,
            show_new_text=args.show_new_text,
            show_controls=args.show_controls,
            show_prompt=args.show_prompt,
            show_model_response=args.show_model_response,
        )
        for step in iter_steps(args.trace)
    )
    rendered += "\n"

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
