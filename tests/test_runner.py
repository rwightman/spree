from pathlib import Path

from bbs_gym.actions import Action
from bbs_gym.activities import TW2_ENTRY_PROFILE
from bbs_gym.memory import JsonMemoryStore
from bbs_gym.models import ScriptedModelAdapter
from bbs_gym.runner import ActivityBudget, ActivityProfile, ActivityRunner
from bbs_gym.telnet import SessionDisconnected
from bbs_gym.terminal import Observation


class FakeAgent:
    agent_id = "agent-001"

    def __init__(self):
        self.actions = []

    def observe_turn(self, **_kwargs):
        return Observation(
            agent_id=self.agent_id,
            node=1,
            requested_node=1,
            pretty_screen="Command:",
            model_text="Command:",
            new_text="Command:",
            cursor=(0, 8),
            stable_ms=300,
            matched_prompt="menu-choice",
            ready_reason="stable",
            profile="test",
            transcript_path=Path("runtime/transcripts/test.raw"),
            bytes_read=8,
            timed_out=False,
            timestamp=0.0,
        )

    def act_action(self, action: Action):
        self.actions.append(action)


class DisconnectingAgent(FakeAgent):
    def observe_turn(self, **_kwargs):
        raise SessionDisconnected("closed")


class LongScreenAgent(FakeAgent):
    def observe_turn(self, **_kwargs):
        observation = super().observe_turn(**_kwargs)
        return Observation(
            **{
                **observation.as_dict(),
                "requested_node": observation.requested_node,
                "node": observation.node,
                "cursor": tuple(observation.cursor),
                "transcript_path": Path(observation.as_dict()["transcript_path"]),
                "model_text": "X" * 200,
                "pretty_screen": "X" * 200,
                "new_text": "X" * 200,
            }
        )


class Tw2ScreenAgent(FakeAgent):
    def observe_turn(self, **_kwargs):
        observation = super().observe_turn(**_kwargs)
        data = observation.as_dict()
        data.update(
            {
                "model_text": "Welcome to Trade Wars (v.ii)",
                "pretty_screen": "Welcome to Trade Wars (v.ii)",
                "new_text": "Welcome to Trade Wars (v.ii)",
                "cursor": tuple(observation.cursor),
                "transcript_path": Path(data["transcript_path"]),
            }
        )
        return Observation(**data)


def test_activity_runner_sends_actions_and_logs_memory(tmp_path):
    agent = FakeAgent()
    model = ScriptedModelAdapter(
        [
            '{"action": "send", "text": "?"}',
            '{"action": "hangup"}',
            '{"durable_facts": ["Asked for help."]}',
        ]
    )
    memory = JsonMemoryStore(tmp_path / "memory")
    runner = ActivityRunner(
        ActivityProfile(name="bbs-menu", objective="test the main menu"),
        memory_store=memory,
        log_path=tmp_path / "steps.jsonl",
    )

    result = runner.run(agent, model, ActivityBudget(max_decision_ticks=5))

    assert result.stop_reason == "hangup"
    assert [action.action for action in agent.actions] == ["send", "hangup"]
    assert memory.load("agent-001") == {"durable_facts": ["Asked for help."]}
    assert (tmp_path / "steps.jsonl").read_text(encoding="utf-8").count("\n") == 2


def test_activity_runner_counts_invalid_model_actions(tmp_path):
    agent = FakeAgent()
    model = ScriptedModelAdapter(["not json", "still not json"])
    runner = ActivityRunner(
        ActivityProfile(name="bbs-menu", objective="test validation", invalid_json_retries=0),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
    )

    result = runner.run(agent, model, ActivityBudget(max_decision_ticks=5, max_validation_failures=2))

    assert result.stop_reason == "validation_failures"
    assert agent.actions == []
    assert len(result.steps) == 2


def test_activity_runner_repairs_invalid_json_once(tmp_path):
    agent = FakeAgent()
    model = ScriptedModelAdapter(
        [
            "not json",
            '{"action": "send", "text": "?"}',
            '{"action": "hangup"}',
            '{"durable_facts": ["Recovered from malformed JSON."]}',
        ]
    )
    runner = ActivityRunner(
        ActivityProfile(name="bbs-menu", objective="test retry"),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
    )

    result = runner.run(agent, model, ActivityBudget(max_decision_ticks=5))

    assert result.stop_reason == "hangup"
    assert [action.to_dict() for action in agent.actions] == [
        {"action": "send", "text": "?", "newline": True},
        {"action": "hangup"},
    ]
    assert result.steps[0].validation["accepted"] is True
    assert "repaired_after_error" in result.steps[0].validation["notes"][0]
    assert result.steps[0].validation["invalid_responses"][0]["response"] == "not json"


def test_activity_runner_reports_disconnect(tmp_path):
    runner = ActivityRunner(
        ActivityProfile(name="bbs-menu", objective="test disconnect"),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
    )

    result = runner.run(DisconnectingAgent(), ScriptedModelAdapter([]), ActivityBudget(max_decision_ticks=5))

    assert result.stop_reason == "disconnected"
    assert result.steps == []


def test_activity_runner_compacts_on_recent_context_size(tmp_path):
    model = ScriptedModelAdapter(
        [
            '{"action": "send", "text": "?"}',
            '{"current_state": "Compacted summary", "last_error": "", "open_subgoals": [], "discovered_facts": [], "failed_actions": [], "strategy_notes": []}',
            '{"action": "hangup"}',
            '{"durable_facts": ["Compacted."]}',
        ]
    )
    runner = ActivityRunner(
        ActivityProfile(
            name="bbs-menu",
            objective="test compaction",
            compact_every_steps=0,
            compact_recent_chars=100,
        ),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
    )

    result = runner.run(LongScreenAgent(), model, ActivityBudget(max_decision_ticks=5))

    assert result.stop_reason == "hangup"
    assert result.session_summary.current_state == "Compacted summary"


def test_activity_runner_stops_when_profile_already_complete(tmp_path):
    agent = Tw2ScreenAgent()
    result = ActivityRunner(
        TW2_ENTRY_PROFILE,
        memory_store=JsonMemoryStore(tmp_path / "memory"),
    ).run(agent, ScriptedModelAdapter(['{"action": "send", "text": "should-not-run"}']))

    assert result.stop_reason == "profile_complete"
    assert result.steps == []
    assert agent.actions == []
