"""ANSI helpers for transcript processing."""

from __future__ import annotations

import re


ANSI_RE = re.compile(
    rb"""
    \x1b
    (?:
        \[[0-?]*[ -/]*[@-~]
      | \][^\x07]*(?:\x07|\x1b\\)
      | [@-Z\\-_]
    )
    """,
    re.VERBOSE,
)


def strip_ansi(data: bytes) -> str:
    """Return CP437 text with terminal control sequences removed."""

    clean = ANSI_RE.sub(b"", data)
    return clean.decode("cp437", errors="replace")

