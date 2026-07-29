import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bbs_gym.campaign import (
    CampaignForumPost,
    CampaignParticipantRuntime,
    CampaignParticipantSpec,
    EpochCampaignConfig,
    campaign_epoch_order,
    load_campaign_forum,
    restore_world,
    run_epoch_campaign,
    snapshot_world,
)
from tty_agent.actions import Action
from tty_agent.evaluation import EvaluationRecord, EvaluationResult
from tty_agent.models import SessionSummary
from tty_agent.runner import ActivityResult


class FakeModel:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeRunner:
    def __init__(self, adapter) -> None:
        self.adapter = adapter
        self.run_objective = ""
        self.objectives = []

    def run(self, agent, _model, budget):
        self.objectives.append(self.run_objective)
        marker = self.adapter.world_path / "marker.txt"
        value = int(marker.read_text(encoding="ascii")) + 1
        marker.write_text(str(value), encoding="ascii")
        record = EvaluationRecord(
            agent_id=agent.agent_id,
            evaluator="fake",
            source="observation",
            status="ok",
            decision_tick=1,
            metrics={"score": value},
            final=True,
        )
        return ActivityResult(
            activity="fake-game",
            agent_id=agent.agent_id,
            steps=[],
            session_summary=SessionSummary(),
            stop_reason="budget",
            evaluation=EvaluationResult([record]),
            decision_ticks=min(1, budget.max_decision_ticks),
        )


class FailingRunner(FakeRunner):
    def run(self, agent, model, budget):
        super().run(agent, model, budget)
        raise RuntimeError("player process failed")


class FakeAdapter:
    name = "fake"

    def __init__(
            self,
            campaign_dir: Path,
            fail_maintenance: bool = False,
            fail_score_probe: bool = False,
    ) -> None:
        self.campaign_dir = campaign_dir
        self.world_path = campaign_dir / "world"
        self.fail_maintenance = fail_maintenance
        self.fail_score_probe = fail_score_probe
        self.opened = []
        self.maintenance_times = []
        self.score_probe_times = []
        self.score_snapshot = []
        self.cleaned = 0

    def initialize_world(self, virtual_time):
        self.world_path.mkdir()
        (self.world_path / "marker.txt").write_text("0", encoding="ascii")
        return {"virtual_time": virtual_time.isoformat()}

    def validate_world(self, virtual_time):
        return {
            "virtual_time": virtual_time.isoformat(),
            "marker": int((self.world_path / "marker.txt").read_text(encoding="ascii")),
        }

    def open_player_session(self, participant, virtual_time, transcript_path, model_metadata):
        del model_metadata
        self.opened.append((participant.agent_id, virtual_time.isoformat(), transcript_path))
        return FakeAgent(participant.agent_id)

    def close_player_session(self, agent):
        agent.close()

    def cleanup_after_session(self):
        self.cleaned += 1

    def score_player(self, participant, virtual_time, transcript_path):
        del participant, virtual_time, transcript_path
        score = int((self.world_path / "marker.txt").read_text(encoding="ascii"))
        self.score_snapshot = [{"score": score}]
        return {"score": score}

    def run_maintenance(self, virtual_time):
        self.maintenance_times.append(virtual_time)
        if self.fail_maintenance:
            raise RuntimeError("maintenance failed")
        marker = self.world_path / "marker.txt"
        marker.write_text(str(int(marker.read_text(encoding="ascii")) + 10), encoding="ascii")
        return {"status": "ok"}

    def probe_scores(self, virtual_time, transcript_path):
        del transcript_path
        self.score_probe_times.append(virtual_time)
        if self.fail_score_probe:
            raise RuntimeError("score probe failed")
        score = int((self.world_path / "marker.txt").read_text(encoding="ascii"))
        self.score_snapshot = [{"score": score}]
        return [dict(row) for row in self.score_snapshot]

    def read_scores(self):
        return [dict(row) for row in self.score_snapshot]


def participant(adapter, agent_id: str):
    model = FakeModel()
    return CampaignParticipantRuntime(
        spec=CampaignParticipantSpec(agent_id, f"Player-{agent_id}"),
        model=model,
        model_metadata={"provider": "fake"},
        runner=FakeRunner(adapter),
    )


def test_epoch_campaign_serializes_players_rotates_and_commits(tmp_path):
    adapter = FakeAdapter(tmp_path / "campaign")
    participants = [participant(adapter, "alpha"), participant(adapter, "bravo")]
    config = EpochCampaignConfig(
        start_time=datetime(2026, 7, 28, 12, tzinfo=timezone.utc),
        max_epochs=2,
        order="rotate",
        max_decision_ticks=3,
        max_session_wall_seconds=30,
    )

    result = run_epoch_campaign(adapter, participants, config)

    assert result.epochs_completed == 2
    assert result.stop_reason == "epochs"
    assert [item[0] for item in adapter.opened] == ["alpha", "bravo", "bravo", "alpha"]
    assert [value.isoformat() for value in adapter.maintenance_times] == [
        "2026-07-29T12:00:00+00:00",
        "2026-07-30T12:00:00+00:00",
    ]
    assert [value.isoformat() for value in adapter.score_probe_times] == [
        "2026-07-29T12:00:00+00:00",
        "2026-07-30T12:00:00+00:00",
    ]
    assert result.final_scores == ({"score": 24},)
    assert all(runtime.model.closed for runtime in participants)
    state = json.loads((adapter.campaign_dir / "campaign-state.json").read_text(encoding="utf-8"))
    assert state["next_epoch"] == 2
    assert Path(state["checkpoint"]).is_dir()
    events = [
        json.loads(line) for line in (adapter.campaign_dir / "campaign.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["type"] for event in events].count("participant_session_started") == 4
    assert [event["type"] for event in events].count("maintenance_completed") == 2
    assert events[-1]["type"] == "campaign_completed"


def test_post_maintenance_score_probe_failure_uses_last_extracted_scores(tmp_path):
    adapter = FakeAdapter(tmp_path / "campaign", fail_score_probe=True)
    config = EpochCampaignConfig(
        start_time=datetime(2026, 7, 28, 12, tzinfo=timezone.utc),
        max_session_wall_seconds=30,
    )

    result = run_epoch_campaign(adapter, [participant(adapter, "alpha")], config)

    assert (adapter.world_path / "marker.txt").read_text(encoding="ascii") == "11"
    assert result.final_scores == ({"score": 1},)
    events = [
        json.loads(line) for line in (adapter.campaign_dir / "campaign.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    failures = [event for event in events if event["type"] == "post_maintenance_score_probe_failed"]
    assert len(failures) == 1
    assert failures[0]["score_probe"]["status"] == "fallback"
    assert failures[0]["score_probe"]["source"] == "last_extracted"
    assert failures[0]["fallback_scores"] == [{"score": 1}]
    maintenance = next(event for event in events if event["type"] == "maintenance_completed")
    assert maintenance["score_probe"]["status"] == "fallback"
    assert maintenance["scores"] == [{"score": 1}]
    manifest = json.loads(
        (adapter.campaign_dir / "manifests" / "epoch-0001.json").read_text(encoding="utf-8")
    )
    assert manifest["score_probe"]["status"] == "fallback"
    assert manifest["scores"] == [{"score": 1}]
    assert events[-1]["score_probe"]["status"] == "fallback"


def test_campaign_resume_restores_last_committed_epoch(tmp_path):
    campaign_dir = tmp_path / "campaign"
    failing_adapter = FakeAdapter(campaign_dir, fail_maintenance=True)
    config = EpochCampaignConfig(
        start_time=datetime(2026, 7, 28, 12, tzinfo=timezone.utc),
        max_epochs=1,
        max_session_wall_seconds=30,
    )

    with pytest.raises(RuntimeError, match="maintenance failed"):
        run_epoch_campaign(failing_adapter, [participant(failing_adapter, "alpha")], config)

    assert (campaign_dir / "world" / "marker.txt").read_text(encoding="ascii") == "1"
    state = json.loads((campaign_dir / "campaign-state.json").read_text(encoding="utf-8"))
    assert state["next_epoch"] == 0

    resumed_adapter = FakeAdapter(campaign_dir)
    result = run_epoch_campaign(
        resumed_adapter,
        [participant(resumed_adapter, "alpha")],
        config,
        resume=True,
    )

    assert result.epochs_completed == 1
    assert (campaign_dir / "world" / "marker.txt").read_text(encoding="ascii") == "11"
    events = [json.loads(line) for line in (campaign_dir / "campaign.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["type"] == "campaign_resumed" for event in events)


def test_competing_campaign_process_does_not_write_without_lock(tmp_path):
    adapter = FakeAdapter(tmp_path / "campaign")
    runtime = participant(adapter, "alpha")
    config = EpochCampaignConfig(
        start_time=datetime(2026, 7, 28, 12, tzinfo=timezone.utc),
        max_session_wall_seconds=30,
    )
    lock_path = adapter.campaign_dir / ".campaign.lock"
    lock_path.parent.mkdir(parents=True)

    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already running"):
            run_epoch_campaign(adapter, [runtime], config)

    assert not (adapter.campaign_dir / "campaign.jsonl").exists()


def test_campaign_shuffle_order_is_stable_per_epoch(tmp_path):
    adapter = FakeAdapter(tmp_path / "campaign")
    participants = [participant(adapter, name) for name in ("alpha", "bravo", "charlie")]

    first = campaign_epoch_order(participants, "shuffle", 42, 7)
    second = campaign_epoch_order(participants, "shuffle", 42, 7)

    assert [item.spec.agent_id for item in first] == [item.spec.agent_id for item in second]
    assert sorted(item.spec.agent_id for item in first) == ["alpha", "bravo", "charlie"]


def test_forfeit_restores_last_safe_player_checkpoint(tmp_path):
    adapter = FakeAdapter(tmp_path / "campaign")
    alpha = participant(adapter, "alpha")
    bravo = participant(adapter, "bravo")
    bravo.runner = FailingRunner(adapter)
    config = EpochCampaignConfig(
        start_time=datetime(2026, 7, 28, 12, tzinfo=timezone.utc),
        max_session_wall_seconds=30,
        failure_policy="forfeit",
    )

    result = run_epoch_campaign(adapter, [alpha, bravo], config)

    assert result.epochs_completed == 1
    assert (adapter.world_path / "marker.txt").read_text(encoding="ascii") == "11"
    events = [
        json.loads(line) for line in (adapter.campaign_dir / "campaign.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    restored = [event for event in events if event["type"] == "participant_forfeit_restored"]
    assert [event["agent_id"] for event in restored] == ["bravo"]


def test_snapshot_restore_rejects_world_outside_campaign(tmp_path):
    campaign_dir = tmp_path / "campaign"
    world = campaign_dir / "world"
    world.mkdir(parents=True)
    (world / "state.bin").write_bytes(b"before")
    snapshot = snapshot_world(world, campaign_dir / "checkpoint" / "world")
    (world / "state.bin").write_bytes(b"after")

    restore_world(snapshot, world, campaign_dir)

    assert (world / "state.bin").read_bytes() == b"before"
    with pytest.raises(ValueError, match="outside campaign"):
        restore_world(snapshot, tmp_path / "outside", campaign_dir)
    outside_snapshot = tmp_path / "outside-snapshot"
    outside_snapshot.mkdir()
    with pytest.raises(ValueError, match="checkpoint outside campaign"):
        restore_world(outside_snapshot, world, campaign_dir)


class ForumModel(FakeModel):
    def __init__(self, messages: list[str | None]) -> None:
        super().__init__()
        self.messages = list(messages)
        self.prompts = []

    def decide(self, prompt, policy=None):
        self.prompts.append(prompt)
        message = self.messages.pop(0)
        action = Action("wait") if message is None else Action("submit_line", text=message)
        return policy.validate(action) if policy is not None else action


def forum_participant(adapter, agent_id: str, messages: list[str | None]):
    runtime = participant(adapter, agent_id)
    runtime.model = ForumModel(messages)
    return runtime


def test_campaign_social_rounds_commit_simultaneously_and_feed_next_epoch(tmp_path):
    adapter = FakeAdapter(tmp_path / "campaign")
    alpha = forum_participant(adapter, "alpha", ["Alliance?", "Deal.", None, None])
    bravo = forum_participant(adapter, "bravo", ["State your terms.", "Agreed.", None, None])
    config = EpochCampaignConfig(
        start_time=datetime(2026, 7, 28, 12, tzinfo=timezone.utc),
        max_epochs=2,
        max_session_wall_seconds=30,
        social_rounds=2,
        social_max_message_chars=80,
    )

    result = run_epoch_campaign(adapter, [alpha, bravo], config)

    assert result.epochs_completed == 2
    posts = load_campaign_forum(adapter.campaign_dir)
    assert posts[:4] == [
        CampaignForumPost(1, 1, "alpha", "Player-alpha", "Alliance?"),
        CampaignForumPost(1, 1, "bravo", "Player-bravo", "State your terms."),
        CampaignForumPost(1, 2, "alpha", "Player-alpha", "Deal."),
        CampaignForumPost(1, 2, "bravo", "Player-bravo", "Agreed."),
    ]
    alpha_model = alpha.model
    bravo_model = bravo.model
    assert isinstance(alpha_model, ForumModel)
    assert isinstance(bravo_model, ForumModel)
    assert "Alliance?" not in alpha_model.prompts[0].user
    assert "Alliance?" not in bravo_model.prompts[0].user
    assert "Alliance?" in alpha_model.prompts[1].user
    assert "State your terms." in bravo_model.prompts[1].user
    assert alpha.runner.objectives[0] == ""
    assert "Alliance?" in alpha.runner.objectives[1]
    assert "State your terms." in bravo.runner.objectives[1]
