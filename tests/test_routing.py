from pathlib import Path

import pytest

from bbs_gym.routing import activity_route_set, activity_route_set_names, looks_like_tw2
from tty_agent.terminal import Observation


def observation(text: str) -> Observation:
    return Observation(
        agent_id="agent",
        pretty_screen=text,
        model_text=text,
        new_text=text,
        cursor=(0, 0),
        stable_ms=300,
        byte_quiet_ms=0,
        matched_prompt=None,
        ready_reason="stable",
        profile="test",
        transcript_path=Path("runtime/transcripts/test.raw"),
        transcript_byte_start=0,
        transcript_byte_end=len(text),
        bytes_read=len(text),
        timed_out=False,
        timestamp=0.0,
        metadata={},
    )


def test_route_set_names_are_explicit():
    assert activity_route_set_names() == ("tw2-auto", "bbs-auto")


def test_tw2_auto_starts_with_entry_and_routes_to_tw2_game():
    route_set = activity_route_set("tw2-auto")

    assert route_set.default_profile.name == "tw2-entry"
    assert route_set.routes[0].profile.name == "tw2-game"
    assert route_set.routes[0].matches(observation("TradeWars2/JavaScript\nCommand (?=Help)?"))


def test_bbs_auto_starts_with_door_safe_and_routes_to_tw2_game():
    route_set = activity_route_set("bbs-auto")

    assert route_set.default_profile.name == "bbs-door-safe"
    assert route_set.routes[0].profile.name == "tw2-game"
    assert route_set.routes[0].matches(observation("Your ship is being initialized."))


def test_looks_like_tw2_rejects_plain_bbs_menu():
    assert not looks_like_tw2(observation("External Programs\nWhich or Quit:"))


def test_unknown_route_set_is_rejected():
    with pytest.raises(ValueError, match="unknown activity route set"):
        activity_route_set("missing")
