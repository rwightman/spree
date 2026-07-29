import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from bbs_gym.campaign import CampaignParticipantSpec
from bbs_gym.sre_campaign import SreCampaignAdapter, SreCampaignAdapterConfig


pytestmark = pytest.mark.skipif(
    os.environ.get("SPREE_RUN_SRE_INTEGRATION") != "1",
    reason="set SPREE_RUN_SRE_INTEGRATION=1 to run disposable DOSEMU integration tests",
)


def _join_empire(
        adapter: SreCampaignAdapter,
        participant: CampaignParticipantSpec,
        virtual_time: datetime,
        transcript_path: Path,
        empire_name: str,
        slot: str,
) -> None:
    agent = adapter.open_player_session(participant, virtual_time, transcript_path, {})
    session = agent.session
    observer = agent.observer
    try:
        adapter._wait_for_screen(observer, lambda text: "PAUSED" in text and "Solar Realms Elite" in text)
        session.send_key("enter")
        adapter._wait_for_screen(observer, lambda text: "Visit the Galaxy" in text and "[System]" in text)
        session.send_key("2")
        adapter._wait_for_screen(observer, lambda text: "[Galaxy Menu]" in text and "Join this Game" in text)
        session.send_key("4")
        adapter._wait_for_text(observer, "Are you sure you would like to join this game")
        session.send_key("y")
        player_or_slot = adapter._wait_for_screen(
            observer,
            lambda text: "Enter the name of your ruler" in text or "Which one would you like" in text,
        )
        if "Enter the name of your ruler" in player_or_slot.model_text:
            session.send_line(participant.player_name)
            adapter._wait_for_text(observer, "Are you [F]emale or [M]ale")
            session.send_key("m")
            adapter._wait_for_text(observer, "Is this correct")
            session.send_key("y")
            adapter._wait_for_screen(observer, lambda text: "Welcome to SRE:II" in text and "PAUSED" in text)
            session.send_key("enter")
            adapter._wait_for_text(observer, "Do you wish to view the instructions")
            session.send_key("n")
            adapter._wait_for_text(observer, "Which one would you like")
        session.send_key(slot)
        adapter._wait_for_text(observer, "Choose a name for your new empire")
        session.send_line(empire_name)
        adapter._wait_for_text(observer, f"Name your empire {empire_name}")
        session.send_key("y")
        adapter._wait_for_screen(
            observer,
            lambda text: "[Galaxy Menu]" in text and "[5] See Your Status" in text,
        )
    finally:
        try:
            agent.close()
        finally:
            adapter.cleanup_after_session()


def test_sre_native_two_player_identities_survive_relaunch_and_maintenance(tmp_path):
    start = datetime(2026, 7, 30, 12, tzinfo=timezone.utc)
    adapter = SreCampaignAdapter(
        SreCampaignAdapterConfig(
            campaign_dir=tmp_path / "campaign",
            source_world=Path("doors/sre"),
            dosemu_config=Path("docker/synchronet/dosemu.conf"),
            control_timeout=60.0,
            verify_clock=False,
        )
    )
    alpha = CampaignParticipantSpec("alpha-agent", "AlphaDoor")
    beta = CampaignParticipantSpec("beta-agent", "BetaDoor")

    adapter.initialize_world(start)
    _join_empire(adapter, alpha, start, tmp_path / "alpha-create.raw", "Alpha Realm", "a")
    _join_empire(adapter, beta, start, tmp_path / "beta-create.raw", "Beta Realm", "b")

    alpha_before = adapter.score_player(alpha, start, tmp_path / "alpha-before.raw")
    beta_before = adapter.score_player(beta, start, tmp_path / "beta-before.raw")
    assert alpha_before["empire"] == "Alpha Realm"
    assert beta_before["empire"] == "Beta Realm"
    assert {row["empire"] for row in beta_before["leaderboard"]} == {"Alpha Realm", "Beta Realm"}

    next_day = start + timedelta(days=1)
    adapter.run_maintenance(next_day)
    post_maintenance_scores = adapter.probe_scores(
        next_day,
        tmp_path / "post-maintenance-scores.raw",
    )
    assert {row["empire"] for row in post_maintenance_scores} == {"Alpha Realm", "Beta Realm"}

    alpha_after = adapter.score_player(alpha, next_day, tmp_path / "alpha-after.raw")
    beta_after = adapter.score_player(beta, next_day, tmp_path / "beta-after.raw")
    assert alpha_after["empire"] == "Alpha Realm"
    assert beta_after["empire"] == "Beta Realm"
    assert {row["empire"] for row in adapter.read_scores()} == {"Alpha Realm", "Beta Realm"}
