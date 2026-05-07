"""BBS-specific routed activity profiles."""

from __future__ import annotations

import re
from dataclasses import dataclass

from terminal_agent.runner import ActivityProfile, ActivityRoute
from terminal_agent.terminal import Observation

from .activities import BBS_DOOR_SAFE_PROFILE, TW2_ENTRY_PROFILE, TW2_GAME_PROFILE

TW2_GAME_RE = re.compile(
    r"(?:Trade\s+Wars\s+\(v\.ii\)|TradeWars2/JavaScript|Command\s+\(\?=Help\)\?|"
    r"Your ship is being initialized|Your offer\?|How many holds|(?:^|\n)Sector\s+\d+\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ActivityRouteSet:
    name: str
    default_profile: ActivityProfile
    routes: tuple[ActivityRoute, ...]


def looks_like_tw2(observation: Observation) -> bool:
    return bool(TW2_GAME_RE.search(observation.model_text))


def activity_route_set(name: str) -> ActivityRouteSet:
    if name == "tw2-auto":
        return ActivityRouteSet(
            name="tw2-auto",
            default_profile=TW2_ENTRY_PROFILE,
            routes=(
                ActivityRoute(
                    name="tw2-game",
                    profile=TW2_GAME_PROFILE,
                    matches=looks_like_tw2,
                    priority=100,
                    reason="Trade Wars 2 screen detected",
                ),
            ),
        )
    if name == "bbs-auto":
        return ActivityRouteSet(
            name="bbs-auto",
            default_profile=BBS_DOOR_SAFE_PROFILE,
            routes=(
                ActivityRoute(
                    name="tw2-game",
                    profile=TW2_GAME_PROFILE,
                    matches=looks_like_tw2,
                    priority=100,
                    reason="Trade Wars 2 screen detected",
                ),
            ),
        )
    raise ValueError(f"unknown activity route set: {name}")


def activity_route_set_names() -> tuple[str, ...]:
    return ("tw2-auto", "bbs-auto")
