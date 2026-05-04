"""Reusable activity profiles for common BBS gym tasks."""

from __future__ import annotations

import re

from .actions import Action, ActionPolicy
from .runner import ActivityBudget, ActivityProfile
from .terminal import Observation


TW2_SCREEN_RE = re.compile(r"(?:Trade\s+Wars|Trade\s+Wars\s+\(v\.ii\)|TW2)", re.IGNORECASE)


class Tw2EntryProfile(ActivityProfile):
    """Profile for getting from the BBS into Synchronet's bundled TW2 door."""

    def __init__(self) -> None:
        super().__init__(
            name="tw2-entry",
            objective=(
                "Navigate from the current BBS screen into the Trade Wars 2/TW2 door. "
                "Use normal terminal input. If you are at the main menu, try the BBS "
                "external programs/doors path such as X or D, then choose Games and "
                "Trade Wars 2. If you are unsure, ask the BBS for help with ?."
            ),
            action_policy=ActionPolicy(
                allowed_actions=frozenset({"send", "send_raw", "wait", "hangup"}),
                max_text_chars=80,
                max_line_chars=80,
                max_lines=1,
            ),
            observe_timeout=10.0,
            stable_ms=300,
            recent_steps_to_keep=4,
            compact_every_steps=10,
            compact_recent_chars=8_000,
        )

    def should_exit(self, observation: Observation, action: Action | None, budget: ActivityBudget) -> bool:
        if super().should_exit(observation, action, budget):
            return True
        return bool(TW2_SCREEN_RE.search(observation.model_text))


BBS_MAIN_MENU_PROFILE = ActivityProfile(
    name="bbs-main-menu",
    objective="Explore the BBS main menu, recover from mistakes, and do not enter sysop/admin areas.",
)

TW2_ENTRY_PROFILE = Tw2EntryProfile()

TW2_GAME_PROFILE = ActivityProfile(
    name="tw2-game",
    objective="Play the current Trade Wars 2 session through normal terminal commands and recover from mistakes.",
    action_policy=ActionPolicy(
        allowed_actions=frozenset({"send", "send_raw", "send_multiline", "wait", "hangup"}),
        max_text_chars=240,
        max_line_chars=240,
        max_lines=5,
    ),
    recent_steps_to_keep=8,
    compact_every_steps=20,
    compact_recent_chars=12_000,
)


def activity_profile(name: str, objective: str | None = None) -> ActivityProfile:
    if name == "tw2-entry":
        return TW2_ENTRY_PROFILE
    if name == "tw2-game":
        return TW2_GAME_PROFILE
    if name == "bbs-main-menu":
        return BBS_MAIN_MENU_PROFILE if objective is None else ActivityProfile(name=name, objective=objective)
    return ActivityProfile(name=name, objective=objective or "Explore the current BBS activity.")
