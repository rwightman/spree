import json
import time
from pathlib import Path

from bbs_gym.match import (
    MatchParticipantRuntime,
    MatchParticipantSpec,
    MatchSchedulerConfig,
    run_scheduled_match,
)
from tty_agent.actions import Action
from tty_agent.agent import ActionExecution
from tty_agent.models import ScriptedModelAdapter
from tty_agent.runner import ActivityProfile, ActivityRunner
from tty_agent.terminal import Observation


class FakeAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.actions = []
        self.closed = False

    def observe_turn(self, **_kwargs):
        return Observation(
            agent_id=self.agent_id,
            pretty_screen="Command:",
            model_text="Command:",
            new_text="Command:",
            cursor=(0, 8),
            stable_ms=300,
            byte_quiet_ms=0,
            matched_prompt="menu-choice",
            ready_reason="stable",
            profile="test",
            transcript_path=Path(f"runtime/transcripts/{self.agent_id}.raw"),
            transcript_byte_start=0,
            transcript_byte_end=8,
            bytes_read=8,
            timed_out=False,
            timestamp=0.0,
            metadata={},
        )

    def act_action(self, action: Action) -> ActionExecution:
        self.actions.append(action)
        return ActionExecution()

    def close(self) -> None:
        self.closed = True


class FakeGym:
    def __init__(self) -> None:
        self.agents = {}

    def connect(self, agent_id: str, model_metadata=None):
        del model_metadata
        agent = FakeAgent(agent_id)
        self.agents[agent_id] = agent
        return agent


def test_run_scheduled_match_sequential_writes_order_and_steps(tmp_path):
    participants = [
        participant("alpha", tmp_path),
        participant("bravo", tmp_path),
    ]
    match_log = tmp_path / "match.jsonl"

    result = run_scheduled_match(
        FakeGym(),
        participants,
        MatchSchedulerConfig(
            mode="sequential",
            order="fixed",
            max_rounds=1,
            max_decision_ticks=5,
            max_wall_seconds=60,
        ),
        match_log,
    )

    assert result.rounds == 1
    assert [activity.agent_id for _, activity in result.results] == ["alpha", "bravo"]
    assert [activity.stop_reason for _, activity in result.results] == ["match_rounds", "match_rounds"]
    assert [len(activity.steps) for _, activity in result.results] == [1, 1]
    events = [json.loads(line) for line in match_log.read_text(encoding="utf-8").splitlines()]
    assert [event["type"] for event in events] == [
        "round_started",
        "commit_order",
        "agent_step_started",
        "agent_step_completed",
        "agent_step",
        "agent_step_started",
        "agent_step_completed",
        "agent_step",
        "round_completed",
    ]
    assert events[1]["order"] == ["alpha", "bravo"]


def test_run_scheduled_match_parallel_barrier_commits_configured_order(tmp_path):
    participants = [
        participant("alpha", tmp_path, delay=0.03),
        participant("bravo", tmp_path, delay=0.0),
    ]
    match_log = tmp_path / "barrier.jsonl"

    run_scheduled_match(
        FakeGym(),
        participants,
        MatchSchedulerConfig(
            mode="parallel_barrier",
            order="fixed",
            max_rounds=1,
            max_decision_ticks=5,
            max_wall_seconds=60,
        ),
        match_log,
    )

    events = [json.loads(line) for line in match_log.read_text(encoding="utf-8").splitlines()]
    assert _event(events, "commit_order")["order"] == ["alpha", "bravo"]
    assert _agent_step_order(events) == ["alpha", "bravo"]
    assert {event["agent_id"] for event in events if event["type"] == "agent_decision_completed"} == {
        "alpha",
        "bravo",
    }


def test_run_scheduled_match_parallel_race_commits_completion_order(tmp_path):
    participants = [
        participant("alpha", tmp_path, delay=0.05),
        participant("bravo", tmp_path, delay=0.0),
    ]
    match_log = tmp_path / "race.jsonl"

    run_scheduled_match(
        FakeGym(),
        participants,
        MatchSchedulerConfig(
            mode="parallel_race",
            order="fixed",
            max_rounds=1,
            max_decision_ticks=5,
            max_wall_seconds=60,
        ),
        match_log,
    )

    events = [json.loads(line) for line in match_log.read_text(encoding="utf-8").splitlines()]
    assert _event(events, "commit_order")["order"] == ["bravo", "alpha"]
    assert _agent_step_order(events) == ["bravo", "alpha"]


def _event(events: list[dict], event_type: str) -> dict:
    return next(event for event in events if event["type"] == event_type)


def _agent_step_order(events: list[dict]) -> list[str]:
    return [event["agent_id"] for event in events if event["type"] == "agent_step"]


class DelayedScriptedModelAdapter(ScriptedModelAdapter):
    def __init__(self, responses: list[str], delay: float) -> None:
        super().__init__(responses)
        self.delay = delay

    def decide(self, prompt, policy=None):
        if self.delay:
            time.sleep(self.delay)
        return super().decide(prompt, policy)


def participant(agent_id: str, tmp_path: Path, delay: float = 0.0) -> MatchParticipantRuntime:
    return MatchParticipantRuntime(
        spec=MatchParticipantSpec(agent_id, "scripted", "unused"),
        args=object(),
        model=DelayedScriptedModelAdapter(['{"action": "wait", "arguments": {}}'], delay),
        model_metadata={"provider": "scripted"},
        runner=ActivityRunner(ActivityProfile(name="test", objective="test"), log_path=tmp_path / f"{agent_id}.jsonl"),
        log_path=tmp_path / f"{agent_id}.jsonl",
    )
