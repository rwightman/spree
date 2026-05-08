"""Prompt profiles for terminal observation guardrails."""

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


EMPTY_PROFILE = PromptProfile("empty")

SHELL_PROFILE = PromptProfile.from_patterns(
    "shell",
    {
        "shell-prompt": r"(?:^|\n).*(?:[$#>])\s*$",
    },
)

TEXT_ADVENTURE_PROFILE = PromptProfile.from_patterns(
    "text-adventure",
    {
        "command-prompt": r"(?:^|\n)>\s*$",
        "yes-no-prompt": r"(?:yes/no|y/n|affirmative).*[:?]\s*$",
        "more-prompt": r"(?:\[MORE\]|--More--)\s*$",
    },
)

DEFAULT_PROFILE = EMPTY_PROFILE
