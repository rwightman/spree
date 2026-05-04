from pathlib import Path

from bbs_gym.actions import Action
from bbs_gym.activities import TW2_ENTRY_PROFILE, activity_profile
from bbs_gym.runner import ActivityBudget
from bbs_gym.terminal import Observation


def observation(text):
    return Observation(
        agent_id="agent",
        node=None,
        requested_node=1,
        pretty_screen=text,
        model_text=text,
        new_text=text,
        cursor=(0, 0),
        stable_ms=300,
        matched_prompt=None,
        ready_reason="stable",
        profile="test",
        transcript_path=Path("runtime/transcripts/test.raw"),
        bytes_read=len(text),
        timed_out=False,
        timestamp=0.0,
    )


def test_tw2_entry_profile_exits_when_trade_wars_visible():
    assert TW2_ENTRY_PROFILE.should_exit(
        observation("Welcome to Trade Wars (v.ii)"),
        Action("wait"),
        ActivityBudget(),
    )


def test_activity_profile_factory_returns_tw2_entry_profile():
    assert activity_profile("tw2-entry").name == "tw2-entry"

