"""Minimal tooling for driving BBS sessions from agents."""

__all__ = [
    "BbsGym",
    "AgentRecord",
    "AgentRegistry",
    "AccountConfigError",
    "BBS_MAIN_MENU_PROFILE",
    "BBS_DOOR_SAFE_PROFILE",
    "TW2_ENTRY_PROFILE",
    "TW2_GAME_PROFILE",
    "BBS_PROFILE",
    "TW2_PROFILE",
    "TW2_EVALUATION_PROFILE",
    "ActivityRouteSet",
    "activity_profile",
    "activity_route_set",
    "extract_tw2_metrics",
    "tw2_score_probe_ready",
]

from .accounts import AccountConfigError, AgentRecord, AgentRegistry
from .activities import (
    BBS_DOOR_SAFE_PROFILE,
    BBS_MAIN_MENU_PROFILE,
    TW2_ENTRY_PROFILE,
    TW2_GAME_PROFILE,
    activity_profile,
)
from .env import BbsGym
from .evaluation import TW2_EVALUATION_PROFILE, extract_tw2_metrics, tw2_score_probe_ready
from .profiles import BBS_PROFILE, TW2_PROFILE
from .routing import ActivityRouteSet, activity_route_set
