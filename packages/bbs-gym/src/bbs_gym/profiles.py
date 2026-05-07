"""Prompt profiles for BBS and door-game observation guardrails."""

from __future__ import annotations

from tty_agent.profiles import PromptProfile


BBS_PROFILE = PromptProfile.from_patterns(
    "synchronet-bbs",
    {
        "login-name": r"(?:name|user(?:name)?|alias)\s*[:?]\s*$",
        "password": r"password\s*[:?]\s*$",
        "menu-choice": r"(?:choice|selection|command)\s*[:?>]\s*$",
        "press-key": r"(?:press|hit)\s+(?:any\s+)?key",
        "more": r"(?:more|\[more\])\s*(?:\?|:)?\s*$",
    },
)


TW2_PROFILE = PromptProfile.from_patterns(
    "tw2",
    {
        "tw2-command": r"(?:command|choice|selection)\s*[:?>]\s*$",
        "tw2-press-key": r"(?:press|hit)\s+(?:any\s+)?key",
        "tw2-more": r"(?:more|\[more\])\s*(?:\?|:)?\s*$",
    },
)


DEFAULT_PROFILE = BBS_PROFILE
