"""Reusable activity profiles for common BBS gym tasks."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from terminal_agent.actions import ActionPolicy
from terminal_agent.runner import ActivityProfile
from terminal_agent.terminal import Observation

from .prompt_modules import (
    BBS_INPUT_MODALITY_PROFILE,
    BBS_PROMPT_MODULES,
    TW2_INPUT_MODALITY_PROFILE,
    TW2_PROMPT_MODULES,
)

TW2_SCREEN_RE = re.compile(
    r"(?:Trade\s+Wars\s+\(v\.ii\)|TradeWars2/JavaScript|Command\s+\(\?=Help\)\?|Your ship is being initialized)",
    re.IGNORECASE,
)


def bbs_action_policy(**kwargs: Any) -> ActionPolicy:
    return ActionPolicy(require_encoding="cp437", **kwargs)


def tw2_entry_complete(observation: Observation) -> bool:
    return bool(TW2_SCREEN_RE.search(observation.model_text))


BBS_MAIN_MENU_PROFILE = ActivityProfile(
    name="bbs-main-menu",
    objective="Explore the BBS main menu, recover from mistakes, and do not enter sysop/admin areas.",
    action_policy=bbs_action_policy(),
    input_modality_profile=BBS_INPUT_MODALITY_PROFILE,
    prompt_modules=BBS_PROMPT_MODULES,
)

TW2_ENTRY_PROFILE = ActivityProfile(
    name="tw2-entry",
    objective=(
        "Navigate from the current BBS screen into the Trade Wars 2/TW2 door. "
        "Use normal terminal input. If you are at the main menu, try the BBS "
        "external programs/doors path such as X or D, then choose Games and "
        "Trade Wars 2. If you are unsure, ask the BBS for help with ?."
    ),
    action_policy=bbs_action_policy(
        allowed_actions=frozenset({"submit_line", "type_text", "press_key", "wait", "hangup"}),
        max_text_chars=80,
        max_line_chars=80,
        max_lines=1,
    ),
    input_modality_profile=BBS_INPUT_MODALITY_PROFILE,
    prompt_modules=BBS_PROMPT_MODULES,
    observe_timeout=10.0,
    stable_ms=300,
    recent_steps_to_keep=4,
    screen_tail_chars=1_200,
    compact_every_steps=10,
    compact_recent_chars=8_000,
    completion_check=tw2_entry_complete,
)

TW2_GAME_PROFILE = ActivityProfile(
    name="tw2-game",
    objective="Play the current Trade Wars 2 session through normal terminal commands and recover from mistakes.",
    action_policy=bbs_action_policy(
        allowed_actions=frozenset({"submit_line", "type_text", "submit_lines", "press_key", "wait", "hangup"}),
        max_text_chars=240,
        max_line_chars=240,
        max_lines=5,
    ),
    input_modality_profile=TW2_INPUT_MODALITY_PROFILE,
    prompt_modules=TW2_PROMPT_MODULES,
    recent_steps_to_keep=8,
    screen_tail_chars=1_600,
    compact_every_steps=20,
    compact_recent_chars=12_000,
)


def activity_profile(name: str, objective: str | None = None) -> ActivityProfile:
    if name == "tw2-entry":
        profile = TW2_ENTRY_PROFILE
    elif name == "tw2-game":
        profile = TW2_GAME_PROFILE
    elif name == "bbs-main-menu":
        profile = BBS_MAIN_MENU_PROFILE
    else:
        return ActivityProfile(
            name=name,
            objective=objective or "Explore the current BBS activity.",
            action_policy=bbs_action_policy(),
            input_modality_profile=BBS_INPUT_MODALITY_PROFILE,
            prompt_modules=BBS_PROMPT_MODULES,
        )
    if objective is not None:
        return replace(profile, objective=objective)
    return profile
