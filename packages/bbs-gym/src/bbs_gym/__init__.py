"""Minimal tooling for driving BBS sessions from agents."""

__all__ = [
    "BbsGym",
    "AgentRecord",
    "AgentRegistry",
    "AccountConfigError",
    "BBS_MAIN_MENU_PROFILE",
    "BBS_DOOR_SAFE_PROFILE",
    "SRE_GAME_PROFILE",
    "TW2_ENTRY_PROFILE",
    "TW2_GAME_PROFILE",
    "BBS_PROFILE",
    "TW2_PROFILE",
    "TW2_EVALUATION_PROFILE",
    "SRE_EVALUATION_PROFILE",
    "ActivityRouteSet",
    "activity_profile",
    "activity_route_set",
    "extract_tw2_metrics",
    "extract_sre_metrics",
    "extract_sre_scoreboard",
    "sre_score_probe_ready",
    "tw2_score_probe_ready",
    "CampaignForumPost",
    "CampaignParticipantRuntime",
    "CampaignParticipantSpec",
    "EpochCampaignConfig",
    "EpochCampaignResult",
    "SreCampaignAdapter",
    "SreCampaignAdapterConfig",
    "load_campaign_forum",
    "run_epoch_campaign",
]

from .accounts import AccountConfigError, AgentRecord, AgentRegistry
from .activities import (
    BBS_DOOR_SAFE_PROFILE,
    BBS_MAIN_MENU_PROFILE,
    SRE_GAME_PROFILE,
    TW2_ENTRY_PROFILE,
    TW2_GAME_PROFILE,
    activity_profile,
)
from .campaign import (
    CampaignForumPost,
    CampaignParticipantRuntime,
    CampaignParticipantSpec,
    EpochCampaignConfig,
    EpochCampaignResult,
    load_campaign_forum,
    run_epoch_campaign,
)
from .env import BbsGym
from .evaluation import (
    SRE_EVALUATION_PROFILE,
    TW2_EVALUATION_PROFILE,
    extract_sre_metrics,
    extract_sre_scoreboard,
    extract_tw2_metrics,
    sre_score_probe_ready,
    tw2_score_probe_ready,
)
from .profiles import BBS_PROFILE, TW2_PROFILE
from .routing import ActivityRouteSet, activity_route_set
from .sre_campaign import SreCampaignAdapter, SreCampaignAdapterConfig
