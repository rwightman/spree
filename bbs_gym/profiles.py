"""Prompt profiles for BBS and door-game observation guardrails."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Pattern


@dataclass(frozen=True)
class PromptProfile:
    """Named prompt patterns used to annotate stable terminal observations."""

    name: str
    prompts: dict[str, Pattern[str]] = field(default_factory=dict)

    @classmethod
    def from_patterns(cls, name: str, patterns: dict[str, str]) -> "PromptProfile":
        return cls(
            name=name,
            prompts={key: re.compile(pattern, re.IGNORECASE | re.MULTILINE) for key, pattern in patterns.items()},
        )

    def match(self, text: str) -> str | None:
        for prompt_name, pattern in self.prompts.items():
            if pattern.search(text):
                return prompt_name
        return None


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
