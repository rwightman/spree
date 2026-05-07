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


def strip_ansi(data: bytes, encoding: str = "utf-8") -> str:
    """Return decoded text with terminal control sequences removed."""

    clean = ANSI_RE.sub(b"", data)
    return clean.decode(encoding, errors="replace")
