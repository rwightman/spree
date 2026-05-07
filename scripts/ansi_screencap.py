#!/usr/bin/env python3
"""Render a colored terminal screen from an activity trace and raw transcript."""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyte


ANSI_COLORS = {
    "black": "#000000",
    "red": "#aa0000",
    "green": "#00aa00",
    "brown": "#aa5500",
    "blue": "#0000aa",
    "magenta": "#aa00aa",
    "cyan": "#00aaaa",
    "white": "#aaaaaa",
    "default": "#d8d8d8",
}
ANSI_BRIGHT_COLORS = {
    "black": "#555555",
    "red": "#ff5555",
    "green": "#55ff55",
    "brown": "#ffff55",
    "blue": "#5555ff",
    "magenta": "#ff55ff",
    "cyan": "#55ffff",
    "white": "#ffffff",
    "default": "#ffffff",
}
DEFAULT_BG = "#000000"


@dataclass(frozen=True)
class TraceSlice:
    trace_path: Path
    step: int
    transcript_path: Path
    byte_count: int
    encoding: str
    title: str


@dataclass(frozen=True)
class Cell:
    data: str = " "
    fg: str = "default"
    bg: str = "default"
    bold: bool = False
    italics: bool = False
    underscore: bool = False
    strikethrough: bool = False
    reverse: bool = False
    blink: bool = False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="activity JSONL trace")
    parser.add_argument("--step", type=int, help="render the screen after this step; defaults to the last step")
    parser.add_argument("--out", type=Path, help="output HTML path")
    parser.add_argument("--gif-out", type=Path, help="output animated GIF path")
    parser.add_argument("--start-step", type=int, help="first step to include in the animated GIF")
    parser.add_argument("--end-step", type=int, help="last step to include in the animated GIF")
    parser.add_argument("--duration-ms", type=int, default=650, help="animated GIF frame duration")
    parser.add_argument("--columns", type=int, default=80)
    parser.add_argument("--lines", type=int, default=24)
    parser.add_argument("--encoding", help="override transcript encoding")
    parser.add_argument(
        "--base-byte-offset",
        type=int,
        default=0,
        help="fallback byte offset for old traces without transcript_byte_end",
    )
    parser.add_argument("--font", type=Path, help="TrueType monospace font for GIF output")
    parser.add_argument("--font-size", type=int, default=16)
    args = parser.parse_args()

    trace_slice = load_trace_slice(args.trace, args.step, args.encoding, args.base_byte_offset)
    transcript = trace_slice.transcript_path.read_bytes()[:trace_slice.byte_count]
    screen = render_screen(transcript, args.columns, args.lines, trace_slice.encoding)

    wrote = False
    if args.out or not args.gif_out:
        out = args.out or trace_slice.trace_path.with_suffix(f".step-{trace_slice.step:04d}.ansi.html")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_html(screen, trace_slice, args.columns, args.lines), encoding="utf-8")
        print(out)
        wrote = True

    if args.gif_out:
        write_gif(
            trace_path=args.trace,
            out=args.gif_out,
            start_step=args.start_step,
            end_step=args.end_step,
            columns=args.columns,
            lines=args.lines,
            encoding_override=args.encoding,
            base_byte_offset=args.base_byte_offset,
            duration_ms=args.duration_ms,
            font_path=args.font,
            font_size=args.font_size,
        )
        print(args.gif_out)
        wrote = True

    if not wrote:
        raise SystemExit("no output requested")
    return 0


def load_trace_slice(
        trace_path: Path,
        step: int | None,
        encoding_override: str | None,
        base_byte_offset: int,
) -> TraceSlice:
    records = read_records(trace_path)
    if not records:
        raise SystemExit(f"trace is empty: {trace_path}")

    if step is None:
        selected = records
    else:
        selected = [record for record in records if int(record.get("step", 0)) <= step]
        if not selected or int(selected[-1].get("step", 0)) != step:
            raise SystemExit(f"step not found in trace: {step}")

    last = selected[-1]
    observation = last.get("observation") or {}
    transcript = observation.get("transcript_path")
    if not transcript:
        raise SystemExit("trace observation does not include a transcript_path")

    fallback_byte_count = base_byte_offset + sum(
        int((record.get("observation") or {}).get("bytes_read") or 0)
        for record in selected
    )
    byte_count = trace_byte_end(observation, fallback_byte_count)
    metadata = observation.get("metadata") if isinstance(observation.get("metadata"), dict) else {}
    encoding = encoding_override or metadata.get("encoding") or "utf-8"
    title = f"{trace_path.name} step {last.get('step')}"
    return TraceSlice(
        trace_path=trace_path,
        step=int(last.get("step", 0)),
        transcript_path=resolve_transcript_path(trace_path, Path(transcript)),
        byte_count=byte_count,
        encoding=str(encoding),
        title=title,
    )


def render_screen(data: bytes, columns: int, lines: int, encoding: str) -> pyte.Screen:
    screen = pyte.Screen(columns, lines)
    stream = pyte.Stream(screen)
    stream.feed(data.decode(encoding, errors="replace").replace("\ufeff", ""))
    return screen


def read_records(trace_path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def trace_byte_end(observation: dict[str, Any], fallback_byte_count: int) -> int:
    value = observation.get("transcript_byte_end")
    if isinstance(value, int):
        return value
    return fallback_byte_count


def resolve_transcript_path(trace_path: Path, transcript_path: Path) -> Path:
    if transcript_path.is_absolute():
        return transcript_path

    candidates = [
        transcript_path,
        trace_path.parent / transcript_path,
        trace_path.parent.parent / transcript_path,
        trace_path.parent.parent.parent / transcript_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return transcript_path


def render_html(screen: pyte.Screen, trace_slice: TraceSlice, columns: int, lines: int) -> str:
    body = "\n".join(render_row(screen, y, columns) for y in range(lines))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(trace_slice.title)}</title>
<style>
:root {{ color-scheme: dark; }}
body {{
    margin: 0;
    background: #101010;
    color: #d8d8d8;
    font-family: ui-monospace, "Cascadia Mono", "SFMono-Regular", Consolas, "Liberation Mono", monospace;
}}
.frame {{
    display: inline-block;
    margin: 24px;
    padding: 18px;
    background: #000;
    box-shadow: 0 12px 36px rgb(0 0 0 / 0.55);
}}
.meta {{
    margin: 0 0 12px;
    color: #9ca3af;
    font-size: 13px;
}}
pre {{
    margin: 0;
    font-size: 16px;
    line-height: 1;
    letter-spacing: 0;
    white-space: pre;
}}
</style>
</head>
<body>
<main class="frame">
<p class="meta">{html.escape(trace_slice.title)} · {html.escape(str(trace_slice.transcript_path))} · {trace_slice.byte_count} bytes</p>
<pre>{body}</pre>
</main>
</body>
</html>
"""


def write_gif(
        trace_path: Path,
        out: Path,
        start_step: int | None,
        end_step: int | None,
        columns: int,
        lines: int,
        encoding_override: str | None,
        base_byte_offset: int,
        duration_ms: int,
        font_path: Path | None,
        font_size: int,
) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("animated GIF output requires Pillow: python -m pip install pillow") from exc

    records = read_records(trace_path)
    frame_slices = list(
        iter_frame_slices(trace_path, records, start_step, end_step, encoding_override, base_byte_offset)
    )
    if not frame_slices:
        raise SystemExit("no trace steps matched the requested GIF range")

    transcript = frame_slices[-1].transcript_path.read_bytes()
    frames = [
        render_frame(
            render_screen(transcript[:frame.byte_count], columns, lines, frame.encoding),
            columns,
            lines,
            font_path,
            font_size,
        )
        for frame in frame_slices
    ]

    out.parent.mkdir(parents=True, exist_ok=True)
    first, rest = frames[0], frames[1:]
    first.save(out, save_all=True, append_images=rest, duration=duration_ms, loop=0, optimize=False)


def iter_frame_slices(
        trace_path: Path,
        records: Iterable[dict[str, Any]],
        start_step: int | None,
        end_step: int | None,
        encoding_override: str | None,
        base_byte_offset: int,
) -> Iterable[TraceSlice]:
    byte_count = 0
    transcript_path: Path | None = None
    encoding = encoding_override or "utf-8"

    for record in records:
        observation = record.get("observation") or {}
        byte_count += int(observation.get("bytes_read") or 0)
        step = int(record.get("step", 0))
        metadata = observation.get("metadata") if isinstance(observation.get("metadata"), dict) else {}
        encoding = str(encoding_override or metadata.get("encoding") or encoding)
        if observation.get("transcript_path"):
            transcript_path = resolve_transcript_path(trace_path, Path(observation["transcript_path"]))
        if start_step is not None and step < start_step:
            continue
        if end_step is not None and step > end_step:
            break
        if transcript_path is None:
            continue
        frame_byte_count = trace_byte_end(observation, base_byte_offset + byte_count)
        yield TraceSlice(
            trace_path=trace_path,
            step=step,
            transcript_path=transcript_path,
            byte_count=frame_byte_count,
            encoding=encoding,
            title=f"{trace_path.name} step {step}",
        )


def render_frame(
        screen: pyte.Screen,
        columns: int,
        lines: int,
        font_path: Path | None,
        font_size: int,
) -> Any:
    from PIL import Image, ImageDraw, ImageFont

    font = load_font(font_path, font_size)
    left, top, right, bottom = font.getbbox("M")
    char_width = right - left
    line_height = max(bottom - top + 2, font_size + 2)
    padding = 16
    image = Image.new("RGB", (columns * char_width + padding * 2, lines * line_height + padding * 2), DEFAULT_BG)
    draw = ImageDraw.Draw(image)

    for y in range(lines):
        row = screen.buffer.get(y, {})
        for x in range(columns):
            cell = row.get(x, Cell())
            fg, bg = cell_colors(cell)
            x0 = padding + x * char_width
            y0 = padding + y * line_height
            if bg != DEFAULT_BG:
                draw.rectangle((x0, y0, x0 + char_width, y0 + line_height), fill=bg)
            if cell.data and cell.data != " ":
                draw.text((x0, y0), cell.data, font=font, fill=fg)
            if cell.underscore:
                draw.line((x0, y0 + line_height - 2, x0 + char_width, y0 + line_height - 2), fill=fg)

    return image


def load_font(font_path: Path | None, font_size: int) -> Any:
    from PIL import ImageFont

    candidates = []
    if font_path is not None:
        candidates.append(font_path)
    candidates.extend(
        [
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
            Path("/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), font_size)
    return ImageFont.load_default()


def render_row(screen: pyte.Screen, y: int, columns: int) -> str:
    row = screen.buffer.get(y, {})
    rendered = []
    current_style: str | None = None
    current_text: list[str] = []

    for x in range(columns):
        cell = row.get(x, Cell())
        style = cell_style(cell)
        if current_style is not None and style != current_style:
            rendered.append(wrap_span(current_style, "".join(current_text)))
            current_text = []
        current_style = style
        current_text.append(cell.data or " ")

    if current_style is not None:
        rendered.append(wrap_span(current_style, "".join(current_text)))
    return "".join(rendered)


def cell_style(cell: Any) -> str:
    fg, bg = cell_colors(cell)

    styles = [f"color:{fg}", f"background-color:{bg}"]
    if cell.italics:
        styles.append("font-style:italic")
    if cell.underscore:
        styles.append("text-decoration:underline")
    if cell.strikethrough:
        styles.append("text-decoration:line-through")
    if cell.blink:
        styles.append("opacity:0.75")
    return ";".join(styles)


def cell_colors(cell: Any) -> tuple[str, str]:
    fg = (
        ANSI_BRIGHT_COLORS.get(cell.fg, ANSI_BRIGHT_COLORS["default"])
        if cell.bold
        else ANSI_COLORS.get(
            cell.fg,
            ANSI_COLORS["default"],
        )
    )
    bg = DEFAULT_BG if cell.bg == "default" else ANSI_COLORS.get(cell.bg, DEFAULT_BG)
    if cell.reverse:
        fg, bg = bg, fg
    return fg, bg


def wrap_span(style: str, text: str) -> str:
    return f'<span style="{style}">{html.escape(text)}</span>'


if __name__ == "__main__":
    raise SystemExit(main())
