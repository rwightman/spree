import json
import time
from pathlib import Path

from bbs_gym.activities import TW2_ENTRY_PROFILE
from terminal_agent.actions import Action, ActionError, ActionPolicy
from terminal_agent.agent import ActionExecution
from terminal_agent.memory import JsonMemoryStore
from terminal_agent.models import ScriptedModelAdapter
from terminal_agent.prompt_modules import GENERIC_TERMINAL_MODULES
from terminal_agent.runner import ActivityBudget, ActivityProfile, ActivityRunner
from terminal_agent.terminal import Observation
from terminal_agent.transports.base import SessionDisconnected


class FakeAgent:
    agent_id = "agent-001"

    def __init__(self):
        self.actions = []

    def observe_turn(self, **_kwargs):
        return Observation(
            agent_id=self.agent_id,
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
            metadata={"node": 1, "requested_node": 1},
        )

    def act_action(self, action: Action) -> ActionExecution:
        self.actions.append(action)
        if action.action == "submit_line":
            return ActionExecution(sent_bytes=(action.text.encode("utf-8"), b"\n"))
        if action.action == "press_key":
            return ActionExecution(sent_bytes=(action.key.encode("utf-8"),))
        return ActionExecution()


class RejectingAgent(FakeAgent):
    def act_action(self, action: Action):
        self.actions.append(action)
        raise ActionError("unsupported key 'home'; supported keys: enter")


class DisconnectingAgent(FakeAgent):
    def observe_turn(self, **_kwargs):
        raise SessionDisconnected("closed")


class LongScreenAgent(FakeAgent):
    def observe_turn(self, **_kwargs):
        observation = super().observe_turn(**_kwargs)
        return Observation(
            **{
                **observation.as_dict(),
                "cursor": tuple(observation.cursor),
                "transcript_path": Path(observation.as_dict()["transcript_path"]),
                "model_text": "X" * 200,
                "pretty_screen": "X" * 200,
                "new_text": "X" * 200,
            }
        )


class SlowObserveAgent(FakeAgent):
    def observe_turn(self, **_kwargs):
        time.sleep(0.03)
        return super().observe_turn(**_kwargs)


class SequencedScreenAgent(FakeAgent):
    def __init__(self, screens: list[str]):
        super().__init__()
        self.screens = screens
        self.observations = 0

    def observe_turn(self, **_kwargs):
        observation = super().observe_turn(**_kwargs)
        screen = self.screens[min(self.observations, len(self.screens) - 1)]
        self.observations += 1
        data = observation.as_dict()
        data.update(
            {
                "model_text": screen,
                "pretty_screen": screen,
                "new_text": screen,
                "cursor": (0, len(screen)),
                "transcript_path": Path(data["transcript_path"]),
            }
        )
        return Observation(**data)


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


class Tw2AfterActionAgent(FakeAgent):
    def __init__(self):
        super().__init__()
        self.observations = 0

    def observe_turn(self, **_kwargs):
        self.observations += 1
        if self.observations == 1:
            return super().observe_turn(**_kwargs)
        observation = super().observe_turn(**_kwargs)
        data = observation.as_dict()
        data.update(
            {
                "model_text": "Trade Wars (v.ii)\n[Hit a key]",
                "pretty_screen": "Trade Wars (v.ii)\n[Hit a key]",
                "new_text": "Trade Wars (v.ii)\n[Hit a key]",
                "cursor": tuple(observation.cursor),
                "transcript_path": Path(data["transcript_path"]),
            }
        )
        return Observation(**data)


def test_activity_runner_sends_actions_and_logs_memory(tmp_path):
    agent = FakeAgent()
    model = ScriptedModelAdapter(
        [
            '{"action": "submit_line", "arguments": {"text": "?"}}',
            '{"action": "hangup", "arguments": {}}',
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
    assert [action.action for action in agent.actions] == ["submit_line", "hangup"]
    assert result.steps[0].execution["sent_bytes"]["combined"]["repr"] == "b'?\\n'"
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
            '{"action": "submit_line", "arguments": {"text": "?"}}',
            '{"action": "hangup", "arguments": {}}',
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
        {"action": "submit_line", "arguments": {"text": "?"}},
        {"action": "hangup", "arguments": {}},
    ]
    assert result.steps[0].validation["accepted"] is True
    assert "repaired_after_error" in result.steps[0].validation["notes"][0]
    assert result.steps[0].validation["invalid_responses"][0]["response"] == "not json"


def test_activity_runner_renders_schema_from_action_policy(tmp_path):
    agent = FakeAgent()
    model = ScriptedModelAdapter(['{"action": "wait", "arguments": {}}'])
    profile = ActivityProfile(
        name="limited",
        objective="test schema",
        action_policy=ActionPolicy(
            allowed_actions=frozenset({"press_key", "wait"}),
            supported_keys=frozenset({"enter"}),
        ),
    )

    result = ActivityRunner(profile, memory_store=JsonMemoryStore(tmp_path / "memory")).run(
        agent,
        model,
        ActivityBudget(max_decision_ticks=1),
    )
    system_prompt = result.steps[0].prompt["system"]

    assert '"press_key"' in system_prompt
    assert "Supported named keys: enter" in system_prompt
    assert '"submit_lines"' not in system_prompt
    assert '"send_raw"' not in system_prompt


def test_activity_runner_renders_and_traces_prompt_modules(tmp_path):
    agent = FakeAgent()
    model = ScriptedModelAdapter(['{"action": "wait", "arguments": {}}'])
    profile = ActivityProfile(
        name="modules",
        objective="test prompt modules",
        prompt_modules=GENERIC_TERMINAL_MODULES,
    )

    result = ActivityRunner(profile, memory_store=JsonMemoryStore(tmp_path / "memory")).run(
        agent,
        model,
        ActivityBudget(max_decision_ticks=1),
    )

    user_prompt = result.steps[0].prompt["user"]
    assert user_prompt.index("Recent steps:") < user_prompt.index("[generic_terminal]")
    assert user_prompt.index("Current step: 1") < user_prompt.index("[generic_terminal]")
    assert user_prompt.index("[generic_terminal]") < user_prompt.index("Full current screen:")
    assert "Most recent terminal output:\nCommand:" in user_prompt
    assert result.steps[0].prompt_modules_schema_version == 1
    assert [module["name"] for module in result.steps[0].prompt_modules] == [
        "terminal.recent_output",
        "terminal.active_prompt",
        "terminal.input_modality",
        "terminal.full_screen",
    ]


def test_activity_runner_renders_recent_steps_as_causal_timeline(tmp_path):
    agent = SequencedScreenAgent(["First prompt", "After first action", "Current prompt"])
    model = ScriptedModelAdapter(
        [
            '{"action": "wait", "arguments": {}}',
            '{"action": "wait", "arguments": {}}',
            '{"action": "hangup", "arguments": {}}',
            '{"durable_facts": ["Ended cleanly."]}',
        ]
    )

    result = ActivityRunner(
        ActivityProfile(name="bbs-menu", objective="test timeline prompt"),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
    ).run(agent, model, ActivityBudget(max_decision_ticks=5))
    third_prompt = result.steps[2].prompt["user"]

    assert "Step 1\nObserved before action:\nFirst prompt" in third_prompt
    assert 'Action chosen:\n{"action": "wait", "arguments": {}}' in third_prompt
    assert "Observed after action:\nAfter first action" in third_prompt
    assert "Step 2\nObserved before action:\nAfter first action" in third_prompt
    assert "Observed after action:\nCurrent terminal observation below." in third_prompt
    assert "Current step: 3" in third_prompt
    assert "Most recent terminal output:\nCurrent prompt" in third_prompt


def test_activity_runner_keeps_stateless_full_prompt_as_default(tmp_path):
    agent = FakeAgent()
    model = ScriptedModelAdapter(['{"action": "wait", "arguments": {}}'])

    result = ActivityRunner(
        ActivityProfile(name="bbs-menu", objective="test prompt mode"),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
    ).run(agent, model, ActivityBudget(max_decision_ticks=1))

    prompt = result.steps[0].prompt

    assert prompt["mode"] == "stateless_full"
    assert prompt["stage"] == "full"
    assert "Allowed terminal actions:" in prompt["system"]
    assert "Campaign memory:" in prompt["user"]
    assert "Recent steps:" in prompt["user"]


def test_activity_runner_stateful_delta_bootstraps_then_sends_delta_prompts(tmp_path):
    agent = FakeAgent()
    model = ScriptedModelAdapter(
        [
            '{"action": "wait", "arguments": {}}',
            '{"action": "hangup", "arguments": {}}',
            '{"durable_facts": ["Ended cleanly."]}',
        ]
    )
    profile = ActivityProfile(
        name="bbs-menu",
        objective="test stateful prompt mode",
        prompt_mode="stateful_delta",
    )

    result = ActivityRunner(profile, memory_store=JsonMemoryStore(tmp_path / "memory")).run(
        agent,
        model,
        ActivityBudget(max_decision_ticks=5),
    )
    first_prompt = result.steps[0].prompt
    second_prompt = result.steps[1].prompt

    assert first_prompt["mode"] == "stateful_delta"
    assert first_prompt["stage"] == "bootstrap"
    assert "Allowed terminal actions:" in first_prompt["system"]
    assert "stateful session bootstrap" in first_prompt["system"]
    assert "Campaign memory:" in first_prompt["user"]
    assert "Recent steps:" in first_prompt["user"]

    assert second_prompt["mode"] == "stateful_delta"
    assert second_prompt["stage"] == "delta"
    assert "Allowed terminal actions:" not in second_prompt["system"]
    assert "Campaign memory:" not in second_prompt["user"]
    assert "Recent steps:" not in second_prompt["user"]
    assert "Previous step:" in second_prompt["user"]
    assert "Current step: 2" in second_prompt["user"]
    assert 'Action chosen:\n{"action": "wait", "arguments": {}}' in second_prompt["user"]
    assert "[generic_terminal]" in second_prompt["user"]


def test_activity_runner_logs_action_execution_errors_without_crashing(tmp_path):
    agent = RejectingAgent()
    model = ScriptedModelAdapter(
        [
            '{"action": "press_key", "arguments": {"key": "enter"}}',
            '{"action": "hangup", "arguments": {}}',
            '{"durable_facts": ["Recovered after action dispatch error."]}',
        ]
    )
    runner = ActivityRunner(
        ActivityProfile(name="bbs-menu", objective="test dispatch error"),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
    )

    result = runner.run(agent, model, ActivityBudget(max_decision_ticks=5, max_validation_failures=2))

    assert result.stop_reason == "validation_failures"
    assert result.steps[0].validation["accepted"] is False
    assert "action_error" in result.steps[0].validation["notes"][0]


def test_activity_runner_logs_raw_and_filtered_model_responses(tmp_path):
    agent = FakeAgent()
    model = ScriptedModelAdapter(
        [
            '<think>hang up cleanly</think>\n{"action": "hangup", "arguments": {}}',
            '{"durable_facts": ["Ended cleanly."]}',
        ]
    )
    runner = ActivityRunner(
        ActivityProfile(name="bbs-menu", objective="test response traces"),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
    )

    result = runner.run(agent, model, ActivityBudget(max_decision_ticks=5))
    model_response = result.steps[0].validation["model_response"]

    assert result.stop_reason == "hangup"
    assert model_response["response"] == '<think>hang up cleanly</think>\n{"action": "hangup", "arguments": {}}'
    assert model_response["parsed_response"] == '{"action": "hangup", "arguments": {}}'


def test_activity_runner_logs_separate_model_reasoning(tmp_path):
    class ReasoningModel(ScriptedModelAdapter):
        def chat(self, messages):
            self.last_reasoning = "screen says quitting is appropriate"
            return super().chat(messages)

    agent = FakeAgent()
    model = ReasoningModel(
        [
            '{"action": "hangup", "arguments": {}}',
            '{"durable_facts": ["Ended cleanly."]}',
        ]
    )
    runner = ActivityRunner(
        ActivityProfile(name="bbs-menu", objective="test reasoning traces"),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
    )

    result = runner.run(agent, model, ActivityBudget(max_decision_ticks=5))
    model_response = result.steps[0].validation["model_response"]

    assert result.stop_reason == "hangup"
    assert model_response["reasoning"] == "screen says quitting is appropriate"


def test_activity_runner_excludes_model_responses_from_context_by_default(tmp_path):
    agent = FakeAgent()
    model = ScriptedModelAdapter(
        [
            '<think>private reasoning</think>\n{"action": "submit_line", "arguments": {"text": "?"}}',
            '{"action": "hangup", "arguments": {}}',
            '{"durable_facts": ["Ended cleanly."]}',
        ]
    )
    runner = ActivityRunner(
        ActivityProfile(name="bbs-menu", objective="test response context"),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
    )

    result = runner.run(agent, model, ActivityBudget(max_decision_ticks=5))
    second_prompt = result.steps[1].prompt["user"]

    assert result.stop_reason == "hangup"
    assert "private reasoning" not in second_prompt
    assert "model_response" not in second_prompt
    assert "parsed_response" not in second_prompt
    assert 'Action chosen:\n{"action": "submit_line", "arguments": {"text": "?"}}' in second_prompt
    assert 'Validation: {"accepted": true, "notes": []}' in second_prompt
    assert "Observed after action:\nCurrent terminal observation below." in second_prompt


def test_activity_runner_can_include_model_responses_in_context(tmp_path):
    agent = FakeAgent()
    model = ScriptedModelAdapter(
        [
            '<think>debug reasoning</think>\n{"action": "submit_line", "arguments": {"text": "?"}}',
            '{"action": "hangup", "arguments": {}}',
            '{"durable_facts": ["Ended cleanly."]}',
        ]
    )
    runner = ActivityRunner(
        ActivityProfile(
            name="bbs-menu",
            objective="test response context",
            include_model_responses_in_context=True,
        ),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
    )

    result = runner.run(agent, model, ActivityBudget(max_decision_ticks=5))
    second_prompt = result.steps[1].prompt["user"]

    assert result.stop_reason == "hangup"
    assert "debug reasoning" in second_prompt
    assert "model_response" in second_prompt


def test_activity_runner_uses_profile_screen_tail_chars(tmp_path):
    agent = FakeAgent()
    model = ScriptedModelAdapter(
        [
            '{"action": "submit_line", "arguments": {"text": "?"}}',
            '{"action": "hangup", "arguments": {}}',
            '{"durable_facts": ["Ended cleanly."]}',
        ]
    )
    profile = ActivityProfile(
        name="bbs-menu",
        objective="test screen tail sizing",
        screen_tail_chars=4,
    )

    result = ActivityRunner(profile, memory_store=JsonMemoryStore(tmp_path / "memory")).run(
        agent,
        model,
        ActivityBudget(max_decision_ticks=5),
    )
    second_prompt = result.steps[1].prompt["user"]

    assert "Observed before action:\nand:" in second_prompt
    assert "Observed before action:\nCommand:" not in second_prompt


def test_activity_runner_reports_disconnect(tmp_path):
    runner = ActivityRunner(
        ActivityProfile(name="bbs-menu", objective="test disconnect"),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
    )

    result = runner.run(DisconnectingAgent(), ScriptedModelAdapter([]), ActivityBudget(max_decision_ticks=5))

    assert result.stop_reason == "disconnected"
    assert result.steps == []


def test_activity_runner_reports_budget_when_wall_clock_expires_after_observe(tmp_path):
    agent = SlowObserveAgent()
    runner = ActivityRunner(
        ActivityProfile(name="bbs-menu", objective="test wall clock stop"),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
        log_path=tmp_path / "steps.jsonl",
    )

    result = runner.run(
        agent,
        ScriptedModelAdapter(['{"action": "hangup", "arguments": {}}']),
        ActivityBudget(max_decision_ticks=5, max_wall_seconds=0.01),
    )

    assert result.stop_reason == "budget"
    assert len(result.steps) == 1
    assert result.steps[0].action is None
    assert result.steps[0].validation["terminal"] is True
    assert result.steps[0].validation["stop_reason"] == "budget"
    assert agent.actions == []


def test_activity_runner_compacts_on_recent_context_size(tmp_path):
    model = ScriptedModelAdapter(
        [
            '{"action": "submit_line", "arguments": {"text": "?"}}',
            '{"current_state": "Compacted summary", "last_error": "", "open_subgoals": [], "discovered_facts": [], "failed_actions": [], "strategy_notes": []}',
            '{"action": "hangup", "arguments": {}}',
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
        log_path=tmp_path / "steps.jsonl",
    ).run(agent, ScriptedModelAdapter(['{"action": "submit_line", "arguments": {"text": "should-not-run"}}']))

    assert result.stop_reason == "profile_complete"
    assert len(result.steps) == 1
    assert result.steps[0].action is None
    assert result.steps[0].prompt == {}
    assert result.steps[0].validation["terminal"] is True
    assert result.steps[0].validation["stop_reason"] == "profile_complete"
    assert "Trade Wars" in result.steps[0].observation["model_text"]
    assert agent.actions == []
    assert JsonMemoryStore(tmp_path / "memory").load("agent-001") == {}

    log_lines = (tmp_path / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(log_lines) == 1
    assert json.loads(log_lines[0])["validation"]["terminal"] is True


def test_activity_runner_logs_terminal_observation_after_profile_completion(tmp_path):
    agent = Tw2AfterActionAgent()
    model = ScriptedModelAdapter(
        [
            '{"action": "press_key", "arguments": {"key": "2"}}',
            '{"durable_facts": ["Reached TW2."]}',
        ]
    )

    result = ActivityRunner(
        TW2_ENTRY_PROFILE,
        memory_store=JsonMemoryStore(tmp_path / "memory"),
        log_path=tmp_path / "steps.jsonl",
    ).run(agent, model, ActivityBudget(max_decision_ticks=5))

    assert result.stop_reason == "profile_complete"
    assert [action.to_dict() for action in agent.actions] == [
        {"action": "press_key", "arguments": {"key": "2"}}
    ]
    assert len(result.steps) == 2
    assert result.steps[0].action == {"action": "press_key", "arguments": {"key": "2"}}
    assert result.steps[1].action is None
    assert result.steps[1].validation["terminal"] is True
    assert result.steps[1].budget["decision_ticks"] == 1
    assert "Trade Wars" in result.steps[1].observation["model_text"]

    log_lines = (tmp_path / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(log_lines) == 2
    assert json.loads(log_lines[1])["validation"]["stop_reason"] == "profile_complete"
    assert JsonMemoryStore(tmp_path / "memory").load("agent-001") == {"durable_facts": ["Reached TW2."]}
