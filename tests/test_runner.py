import json
import time
from pathlib import Path

from bbs_gym.activities import TW2_ENTRY_PROFILE
from tty_agent.actions import Action, ActionError, ActionPolicy
from tty_agent.agent import ActionExecution
from tty_agent.memory import JsonMemoryStore
from tty_agent.models import (
    DecisionPrompt,
    ModelError,
    ModelStateError,
    ModelTimeoutError,
    OpenAICompatibleAdapter,
    ScriptedModelAdapter,
)
from tty_agent.prompt_modules import GENERIC_TERMINAL_MODULES, StaticPromptModule
from tty_agent.runner import (
    ActivityBudget,
    ActivityProfile,
    ActivityRoute,
    ActivityRunner,
    LegacyMemoryLimits,
    RoutedActivityRunner,
)
from tty_agent.terminal import Observation
from tty_agent.transports.base import SessionDisconnected


class FakeAgent:
    agent_id = "agent-001"

    def __init__(self):
        self.actions = []
        self.observe_kwargs = []

    def observe_turn(self, **_kwargs):
        self.observe_kwargs.append(_kwargs)
        return Observation(
            agent_id=self.agent_id,
            pretty_screen="Command:",
            model_text="Command:",
            new_text="Command:",
            cursor=(0, 8),
            stable_ms=300,
            byte_quiet_ms=300,
            matched_prompt="menu-choice",
            ready_reason="stable",
            profile="test",
            transcript_path=Path("runtime/transcripts/test.raw"),
            transcript_byte_start=0,
            transcript_byte_end=8,
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


class SendDisconnectingAgent(FakeAgent):
    def act_action(self, action):
        raise SessionDisconnected("remote terminal connection closed while sending")


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


def test_activity_runner_can_step_state_incrementally(tmp_path):
    agent = FakeAgent()
    model = ScriptedModelAdapter(
        [
            '{"action": "submit_line", "arguments": {"text": "look"}}',
            '{"action": "hangup", "arguments": {}}',
            '{"durable_facts": ["Stepped manually."]}',
        ]
    )
    memory = JsonMemoryStore(tmp_path / "memory")
    runner = ActivityRunner(
        ActivityProfile(name="bbs-menu", objective="test stepping"),
        memory_store=memory,
        log_path=tmp_path / "steps.jsonl",
    )
    state = runner.start_state(agent, model, ActivityBudget(max_decision_ticks=5))

    first = runner.run_step(state)
    second = runner.run_step(state)
    result = runner.finish_state(state)

    assert first is not None
    assert second is not None
    assert first.step == 1
    assert second.step == 2
    assert state.completed is True
    assert result.stop_reason == "hangup"
    assert [action.action for action in agent.actions] == ["submit_line", "hangup"]
    assert memory.load("agent-001") == {"durable_facts": ["Stepped manually."]}


def test_activity_runner_can_prepare_then_commit_step(tmp_path):
    agent = FakeAgent()
    model = ScriptedModelAdapter(['{"action": "submit_line", "arguments": {"text": "look"}}'])
    runner = ActivityRunner(
        ActivityProfile(name="bbs-menu", objective="test split stepping"),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
        log_path=tmp_path / "steps.jsonl",
    )
    state = runner.start_state(agent, model, ActivityBudget(max_decision_ticks=5))

    prepared = runner.prepare_step(state)

    assert prepared is not None
    assert prepared.action is not None
    assert prepared.action.action == "submit_line"
    assert agent.actions == []
    assert state.budget.decision_ticks == 0

    step = runner.commit_prepared_step(state, prepared)

    assert step is not None
    assert step.step == 1
    assert [action.action for action in agent.actions] == ["submit_line"]
    assert state.budget.decision_ticks == 1
    assert (tmp_path / "steps.jsonl").read_text(encoding="utf-8").count("\n") == 1


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


def test_activity_runner_forwards_prompt_fast_path_to_agent(tmp_path):
    agent = FakeAgent()
    model = ScriptedModelAdapter(['{"action": "wait", "arguments": {}}', "{}"])
    profile = ActivityProfile(
        name="text-adventure",
        objective="test fast prompt detection",
        stable_ms=50,
        byte_quiet_ms=0,
        prompt_fast_path=True,
    )

    ActivityRunner(profile, memory_store=JsonMemoryStore(tmp_path / "memory")).run(
        agent,
        model,
        ActivityBudget(max_decision_ticks=1),
    )

    assert agent.observe_kwargs[0]["stable_ms"] == 50
    assert agent.observe_kwargs[0]["byte_quiet_ms"] == 0
    assert agent.observe_kwargs[0]["prompt_fast_path"] is True


def test_routed_activity_runner_switches_profiles_from_observation(tmp_path):
    agent = SequencedScreenAgent(["BBS main menu", "TradeWars2/JavaScript\nCommand (?=Help)?"])
    model = ScriptedModelAdapter(
        [
            '{"action": "wait", "arguments": {}}',
            '{"action": "hangup", "arguments": {}}',
            "{}",
        ]
    )
    default_profile = ActivityProfile(
        name="bbs-safe",
        objective="default",
        action_policy=ActionPolicy(allowed_actions=frozenset({"wait", "hangup"})),
    )
    tw2_profile = ActivityProfile(
        name="tw2-game",
        objective="tw2",
        action_policy=ActionPolicy(allowed_actions=frozenset({"press_key", "wait", "hangup"})),
    )
    runner = RoutedActivityRunner(
        "auto",
        default_profile,
        (
            ActivityRoute(
                name="tw2",
                profile=tw2_profile,
                matches=lambda observation: "TradeWars2" in observation.model_text,
                priority=10,
                reason="test route matched",
            ),
        ),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
        log_path=tmp_path / "steps.jsonl",
        run_objective="win the routed run",
    )

    result = runner.run(agent, model, ActivityBudget(max_decision_ticks=3))

    assert result.activity == "auto"
    assert result.run_objective == "win the routed run"
    assert result.stop_reason == "hangup"
    assert [step.active_profile for step in result.steps] == ["bbs-safe", "tw2-game"]
    assert result.steps[1].events == [
        {
            "type": "profile_switch",
            "from": "bbs-safe",
            "to": "tw2-game",
            "route": "tw2",
            "reason": "test route matched",
        }
    ]
    assert '"press_key"' in result.steps[1].prompt["system"]
    assert '"submit_line"' not in result.steps[1].prompt["system"]
    assert "Run objective: win the routed run" in result.steps[1].prompt["user"]
    assert "Profile objective: tw2" in result.steps[1].prompt["user"]
    logged_steps = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    assert logged_steps[1]["active_profile"] == "tw2-game"
    assert logged_steps[1]["run_objective"] == "win the routed run"
    assert logged_steps[1]["events"][0]["type"] == "profile_switch"


def test_stateful_routed_profiles_bootstrap_after_each_switch(tmp_path):
    agent = SequencedScreenAgent(["BBS main menu", "TradeWars2 menu", "BBS main menu"])
    model = ScriptedModelAdapter(
        [
            '{"action": "wait", "arguments": {}}',
            '{"action": "wait", "arguments": {}}',
            '{"action": "hangup", "arguments": {}}',
            "{}",
        ]
    )
    default_profile = ActivityProfile(name="bbs-safe", objective="default", prompt_mode="stateful_delta")
    tw2_profile = ActivityProfile(name="tw2-game", objective="tw2", prompt_mode="stateful_delta")
    runner = RoutedActivityRunner(
        "auto",
        default_profile,
        (
            ActivityRoute(
                name="tw2",
                profile=tw2_profile,
                matches=lambda observation: "TradeWars2" in observation.model_text,
                priority=10,
            ),
        ),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
    )

    result = runner.run(agent, model, ActivityBudget(max_decision_ticks=3))

    assert [step.active_profile for step in result.steps] == ["bbs-safe", "tw2-game", "bbs-safe"]
    assert [step.prompt["stage"] for step in result.steps] == ["bootstrap", "bootstrap", "bootstrap"]


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
    assert "[generic_terminal]" not in user_prompt
    assert user_prompt.index("Recent steps:") < user_prompt.index("Current step: 1")
    assert user_prompt.index("Current step: 1") < user_prompt.index("Most recent terminal output:")
    assert user_prompt.index("Most recent terminal output:") < user_prompt.index("Full current screen:")
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
    assert prompt["layout"] == "timeline_first"
    assert "Allowed terminal actions:" in prompt["system"]
    assert "Campaign memory:" in prompt["user"]
    assert "Recent steps:" in prompt["user"]


def test_activity_runner_can_render_cache_friendly_prompt_layout(tmp_path):
    agent = FakeAgent()
    model = ScriptedModelAdapter(['{"action": "wait", "arguments": {}}'])
    profile = ActivityProfile(
        name="bbs-menu",
        objective="test cache-friendly prompt layout",
        prompt_layout="cache_friendly",
        prompt_modules=(
            *GENERIC_TERMINAL_MODULES,
            StaticPromptModule(
                name="bbs.static",
                level="bbs_conventions",
                text="Stable BBS convention guidance.",
            ),
            StaticPromptModule(
                name="tw2.static",
                level="game_interface",
                text="Stable game-interface guidance.",
            ),
        ),
    )

    result = ActivityRunner(profile, memory_store=JsonMemoryStore(tmp_path / "memory")).run(
        agent,
        model,
        ActivityBudget(max_decision_ticks=1),
    )

    prompt = result.steps[0].prompt
    user_prompt = prompt["user"]

    assert prompt["layout"] == "cache_friendly"
    assert "[generic_terminal]" not in user_prompt
    assert "[bbs_conventions]" not in user_prompt
    assert "[game_interface]" not in user_prompt
    assert user_prompt.index("Stable BBS convention guidance.") < user_prompt.index("Stable game-interface guidance.")
    assert user_prompt.index("Stable game-interface guidance.") < user_prompt.index("Campaign memory:")
    assert user_prompt.index("Campaign memory:") < user_prompt.index("Session summary:")
    assert user_prompt.index("Session summary:") < user_prompt.index("Recent steps:")
    assert user_prompt.index("Recent steps:") < user_prompt.index("Current step: 1")
    assert user_prompt.index("Current step: 1") < user_prompt.index("Budget:")
    assert user_prompt.index("Budget:") < user_prompt.index("Most recent terminal output:")
    assert "Full current screen:\nCommand:" in user_prompt
    assert "Return exactly one JSON action." not in user_prompt


def test_activity_runner_includes_run_objective_without_replacing_profile_objective(tmp_path):
    agent = FakeAgent()
    model = ScriptedModelAdapter(['{"action": "wait", "arguments": {}}'])

    result = ActivityRunner(
        ActivityProfile(name="bbs-menu", objective="handle the active profile"),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
        run_objective="play TW2 and maximize profit",
    ).run(agent, model, ActivityBudget(max_decision_ticks=1))

    user_prompt = result.steps[0].prompt["user"]
    assert result.run_objective == "play TW2 and maximize profit"
    assert result.steps[0].run_objective == "play TW2 and maximize profit"
    assert "Run objective: play TW2 and maximize profit" in user_prompt
    assert "Profile objective: handle the active profile" in user_prompt


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
    assert "[generic_terminal]" not in second_prompt["user"]
    assert "Most recent terminal output:" in second_prompt["user"]


def test_stateful_action_repair_preserves_prompt_mode_and_stage(tmp_path):
    runner = ActivityRunner(
        ActivityProfile(name="bbs-menu", objective="test repairs"),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
    )
    prompt = runner._build_retry_prompt(
        DecisionPrompt("system", "screen", mode="stateful_delta", stage="delta"),
        "invalid JSON",
        1,
    )

    assert prompt.mode == "stateful_delta"
    assert prompt.stage == "delta"


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


def test_activity_runner_retries_model_timeout_without_crashing(tmp_path):
    class TimeoutThenWaitModel(ScriptedModelAdapter):
        def __init__(self):
            super().__init__(
                [
                    '{"action": "wait", "arguments": {}}',
                    '{"action": "hangup", "arguments": {}}',
                    '{"durable_facts": ["Recovered from provider timeout."]}',
                ]
            )
            self.calls = 0

        def decide(self, prompt, policy=None):
            self.calls += 1
            if self.calls == 1:
                raise ModelTimeoutError(
                    "claude -p timed out after 600s: partial stderr",
                    command=["claude", "-p"],
                    stdout="partial stdout",
                    stderr="partial stderr",
                )
            return super().decide(prompt, policy)

    agent = FakeAgent()
    model = TimeoutThenWaitModel()
    runner = ActivityRunner(
        ActivityProfile(name="bbs-menu", objective="test provider retry", model_error_retries=1),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
    )

    result = runner.run(agent, model, ActivityBudget(max_decision_ticks=5))

    assert result.stop_reason == "hangup"
    assert result.steps[0].validation["accepted"] is True
    assert "recovered_after_model_error" in result.steps[0].validation["notes"]
    model_error = result.steps[0].validation["model_errors"][0]
    assert model_error["type"] == "ModelTimeoutError"
    assert model_error["stdout"] == "partial stdout"
    assert model_error["stderr"] == "partial stderr"
    assert model_error["command"] == ["claude", "-p"]


def test_activity_runner_records_model_timeout_failure_without_crashing(tmp_path):
    class AlwaysTimeoutModel(ScriptedModelAdapter):
        def __init__(self):
            super().__init__([])

        def decide(self, _prompt, _policy=None):
            raise ModelTimeoutError(
                "claude -p timed out after 600s: partial stderr",
                command=["claude", "-p"],
                stdout="partial stdout",
                stderr="partial stderr",
            )

    agent = FakeAgent()
    runner = ActivityRunner(
        ActivityProfile(name="bbs-menu", objective="test provider failure", model_error_retries=1),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
    )

    result = runner.run(agent, AlwaysTimeoutModel(), ActivityBudget(max_decision_ticks=5, max_validation_failures=1))

    assert result.stop_reason == "validation_failures"
    assert result.steps[0].action is None
    assert result.steps[0].validation["accepted"] is False
    assert result.steps[0].validation["model_errors"][0]["stdout"] == "partial stdout"
    assert result.steps[0].validation["model_errors"][1]["stderr"] == "partial stderr"
    assert "model_error:" in result.steps[0].validation["notes"][0]


def test_activity_runner_stops_cleanly_when_send_disconnects(tmp_path):
    """A peer that drops between observing and acting must stop like a read-side drop."""

    runner = ActivityRunner(
        ActivityProfile(name="bbs-menu", objective="test send disconnect"),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
    )
    model = ScriptedModelAdapter(['{"action": "submit_line", "arguments": {"text": "look"}}', "{}"])

    result = runner.run(SendDisconnectingAgent(), model, ActivityBudget(max_decision_ticks=5))

    assert result.stop_reason == "disconnected"
    assert len(result.steps) == 1
    assert "action_error:" in result.steps[0].validation["notes"][-1]


def test_activity_runner_retries_http_provider_failure(tmp_path):
    """An HTTP adapter failure must reach the same retry path as a CLI failure."""

    class HttpErrorThenWaitModel(ScriptedModelAdapter):
        def __init__(self):
            super().__init__(
                [
                    '{"action": "wait", "arguments": {}}',
                    '{"action": "hangup", "arguments": {}}',
                    "{}",
                ]
            )
            self.calls = 0

        def decide(self, prompt, policy=None):
            self.calls += 1
            if self.calls == 1:
                raise ModelError("HTTP 503 from http://localhost:11434/v1: server busy")
            return super().decide(prompt, policy)

    agent = FakeAgent()
    runner = ActivityRunner(
        ActivityProfile(name="bbs-menu", objective="test http retry", model_error_retries=1),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
    )

    result = runner.run(agent, HttpErrorThenWaitModel(), ActivityBudget(max_decision_ticks=5))

    assert result.stop_reason == "hangup"
    assert result.steps[0].validation["accepted"] is True
    assert "recovered_after_model_error" in result.steps[0].validation["notes"]
    assert result.steps[0].validation["model_errors"][0]["type"] == "ModelError"


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
            self.last_response_id = "resp-trace"
            self.last_usage = {"input_tokens": 10, "output_tokens": 5}
            self.last_provider_metadata = {"status": "completed"}
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
    assert model_response["response_id"] == "resp-trace"
    assert model_response["usage"] == {"input_tokens": 10, "output_tokens": 5}
    assert model_response["provider_metadata"] == {"status": "completed"}


def test_activity_runner_does_not_charge_validation_failure_for_adaptive_output_retry(monkeypatch, tmp_path):
    payloads = []
    responses = iter(
        [
            {
                "choices": [{"finish_reason": "length", "message": {"reasoning": "thinking", "content": ""}}],
                "usage": {
                    "completion_tokens": 4,
                    "completion_tokens_details": {"reasoning_tokens": 4},
                },
            },
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"action":"hangup","arguments":{}}'},
                    }
                ],
                "usage": {"completion_tokens": 2},
            },
            {
                "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
                "usage": {"completion_tokens": 1},
            },
        ]
    )

    def fake_post_json(url, payload, headers, timeout):
        del url, headers, timeout
        payloads.append(payload)
        return next(responses)

    monkeypatch.setattr("tty_agent.models._post_json", fake_post_json)
    model = OpenAICompatibleAdapter(model="test-model", max_tokens=4, max_tokens_retry_ceiling=8)

    result = ActivityRunner(
        ActivityProfile(name="bbs-menu", objective="test adaptive output retry"),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
    ).run(FakeAgent(), model, ActivityBudget(max_decision_ticks=1))

    assert result.stop_reason == "hangup"
    assert result.steps[0].budget["validation_failures"] == 0
    assert "recovered_after_output_truncation" in result.steps[0].validation["notes"]
    request_metadata = result.steps[0].validation["model_response"]["request_metadata"]
    assert request_metadata["output_token_retries"] == 1
    assert request_metadata["effective_max_tokens"] == 8
    assert [payload["max_tokens"] for payload in payloads] == [4, 8, 4]


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


def test_activity_runner_finishes_step_admitted_before_wall_clock_expires(tmp_path):
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

    assert result.stop_reason == "hangup"
    assert len(result.steps) == 1
    assert result.steps[0].action == {"action": "hangup", "arguments": {}}
    assert agent.actions == [Action(action="hangup")]


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


class PromptRecordingModel(ScriptedModelAdapter):
    def __init__(self, responses: list[str]) -> None:
        super().__init__(responses)
        self.chats: list[str] = []

    def chat(self, messages):
        self.chats.append("\n".join(message.content for message in messages))
        return super().chat(messages)


def test_compaction_covers_steps_beyond_prompt_window(tmp_path):
    agent = SequencedScreenAgent(["S1", "S2", "S3", "S4"])
    model = PromptRecordingModel(
        [
            '{"action": "submit_line", "arguments": {"text": "one"}}',
            '{"action": "submit_line", "arguments": {"text": "two"}}',
            '{"action": "submit_line", "arguments": {"text": "three"}}',
            '{"current_state": "Compacted", "last_error": "", "open_subgoals": [], "discovered_facts": [], "failed_actions": [], "strategy_notes": []}',
            '{"action": "hangup", "arguments": {}}',
            '{"durable_facts": ["Done."]}',
        ]
    )
    runner = ActivityRunner(
        ActivityProfile(
            name="bbs-menu",
            objective="test compaction coverage",
            recent_steps_to_keep=1,
            compact_every_steps=3,
        ),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
    )

    result = runner.run(agent, model, ActivityBudget(max_decision_ticks=6))

    assert result.stop_reason == "hangup"
    assert result.session_summary.current_state == "Compacted"

    compaction_chats = [chat for chat in model.chats if "Compact older terminal activity" in chat]
    assert len(compaction_chats) == 1
    assert "Keep reasoning concise and reserve enough output for the required JSON" in compaction_chats[0]
    # Every step since the last compaction is summarized, not just the prompt window.
    for marker in ("Step 1", "Step 2", "Step 3", '"text": "one"', '"text": "three"'):
        assert marker in compaction_chats[0]

    # Decision prompts still see only the bounded recent-step window.
    fourth_prompt = result.steps[3].prompt["user"]
    assert "Step 3" in fourth_prompt
    assert '"text": "one"' not in fourth_prompt
    assert '"text": "two"' not in fourth_prompt

    # The end-of-run memory commit sees the steps accumulated since compaction.
    memory_chats = [chat for chat in model.chats if "memory patch" in chat]
    assert len(memory_chats) == 1
    assert "Keep reasoning concise and reserve enough output for the required JSON" in memory_chats[0]
    assert "Step 4" in memory_chats[0]


def test_legacy_compaction_repairs_oversized_draft_and_enforces_limits(tmp_path):
    limits = LegacyMemoryLimits(
        summary_max_chars=350,
        current_state_max_chars=80,
        last_error_max_chars=40,
        item_max_chars=50,
        max_open_subgoals=2,
        max_discovered_facts=2,
        max_failed_actions=2,
        max_strategy_notes=2,
        campaign_max_chars=500,
        campaign_max_string_chars=50,
        campaign_max_list_items=3,
    )
    oversized = json.dumps(
        {
            "current_state": "state " * 100,
            "last_error": "error " * 100,
            "open_subgoals": [f"goal-{index} " * 20 for index in range(6)],
            "discovered_facts": [f"fact-{index} " * 20 for index in range(6)],
            "failed_actions": [f"failure-{index} " * 20 for index in range(6)],
            "strategy_notes": [f"strategy-{index} " * 20 for index in range(6)],
        }
    )
    repaired = json.dumps(
        {
            "current_state": "At the crossroads.",
            "last_error": "",
            "open_subgoals": ["Explore west."],
            "discovered_facts": ["Cyclops", "cyclops", "Cyclops"],
            "failed_actions": [],
            "strategy_notes": ["Map exits exactly."],
        }
    )
    model = PromptRecordingModel(
        [
            '{"action":"submit_line","arguments":{"text":"look"}}',
            oversized,
            repaired,
            '{"action":"hangup","arguments":{}}',
            "{}",
        ]
    )
    result = ActivityRunner(
        ActivityProfile(
            name="bbs-menu",
            objective="test bounded legacy compaction",
            compact_every_steps=1,
            legacy_memory_limits=limits,
        ),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
    ).run(FakeAgent(), model, ActivityBudget(max_decision_ticks=2))

    assert result.stop_reason == "hangup"
    assert result.session_summary.current_state == "At the crossroads."
    assert result.session_summary.discovered_facts == ("Cyclops", "cyclops")
    event = result.steps[1].events[0]
    assert event["repair"]["attempted"] is True
    assert event["repair"]["status"] == "accepted"
    assert "limit is 350" in event["repair"]["reason"]
    assert event["bounds"]["after_chars"] <= 350
    compaction_chats = [chat for chat in model.chats if "Compact older terminal activity" in chat]
    assert len(compaction_chats) == 2
    assert "Hard limits: at most 350 serialized characters" in compaction_chats[0]
    assert "The first draft needs one bounded repair pass" in compaction_chats[1]


def test_legacy_compaction_repairs_implausible_prior_memory_drop(tmp_path):
    initial_facts = [f"stable fact {index}" for index in range(10)]
    repaired_facts = initial_facts[:8] + ["new observation"]
    model = PromptRecordingModel(
        [
            '{"action":"submit_line","arguments":{"text":"look"}}',
            json.dumps({"current_state": "Room one", "discovered_facts": initial_facts}),
            '{"action":"submit_line","arguments":{"text":"north"}}',
            json.dumps({"current_state": "Room two", "discovered_facts": ["new observation"]}),
            json.dumps({"current_state": "Room two", "discovered_facts": repaired_facts}),
            '{"action":"hangup","arguments":{}}',
            "{}",
        ]
    )
    limits = LegacyMemoryLimits(
        repair_min_previous_items=10,
        repair_min_retained_fraction=0.4,
    )
    result = ActivityRunner(
        ActivityProfile(
            name="bbs-menu",
            objective="test destructive compaction guard",
            compact_every_steps=1,
            recent_steps_to_keep=1,
            legacy_memory_limits=limits,
        ),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
    ).run(FakeAgent(), model, ActivityBudget(max_decision_ticks=3))

    assert result.stop_reason == "hangup"
    assert result.session_summary.discovered_facts == tuple(repaired_facts)
    repair = result.steps[2].events[0]["repair"]
    assert repair["attempted"] is True
    assert repair["status"] == "accepted"
    assert "retained only 1 of 10 prior list items" in repair["reason"]


def test_legacy_compaction_keeps_prior_memory_when_destructive_repair_fails(tmp_path):
    initial_facts = [f"stable fact {index}" for index in range(10)]
    model = PromptRecordingModel(
        [
            '{"action":"submit_line","arguments":{"text":"look"}}',
            json.dumps({"current_state": "Room one", "discovered_facts": initial_facts}),
            '{"action":"submit_line","arguments":{"text":"north"}}',
            json.dumps({"current_state": "Room two", "discovered_facts": ["only new fact"]}),
            "",
            '{"action":"hangup","arguments":{}}',
            "{}",
        ]
    )
    result = ActivityRunner(
        ActivityProfile(
            name="bbs-menu",
            objective="test failed destructive compaction repair",
            compact_every_steps=1,
            recent_steps_to_keep=1,
            legacy_memory_limits=LegacyMemoryLimits(repair_min_previous_items=10),
        ),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
    ).run(FakeAgent(), model, ActivityBudget(max_decision_ticks=3))

    assert result.stop_reason == "hangup"
    assert result.session_summary.discovered_facts == tuple(initial_facts)
    event = result.steps[2].events[0]
    assert event["status"] == "error"
    assert "unsafe compaction draft could not be repaired" in event["error"]["message"]
    memory_chat = next(chat for chat in model.chats if "memory patch" in chat)
    assert "Step 2" in memory_chat


def test_failed_compaction_preserves_history_and_retries_on_next_boundary(tmp_path):
    agent = SequencedScreenAgent(["S1", "S2", "S3", "S4", "S5"])
    model = PromptRecordingModel(
        [
            '{"action": "submit_line", "arguments": {"text": "one"}}',
            '{"action": "submit_line", "arguments": {"text": "two"}}',
            "",
            '{"action": "submit_line", "arguments": {"text": "three"}}',
            '{"action": "submit_line", "arguments": {"text": "four"}}',
            '{"current_state": "Recovered summary", "discovered_facts": ["kept all steps"]}',
            '{"action": "hangup", "arguments": {}}',
            '{"durable_facts": ["Done."]}',
        ]
    )
    runner = ActivityRunner(
        ActivityProfile(
            name="bbs-menu",
            objective="test failed compaction recovery",
            recent_steps_to_keep=1,
            compact_every_steps=2,
        ),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
    )

    result = runner.run(agent, model, ActivityBudget(max_decision_ticks=6))

    assert result.stop_reason == "hangup"
    assert result.session_summary.current_state == "Recovered summary"
    assert result.steps[2].events[0]["type"] == "model_utility"
    assert result.steps[2].events[0]["operation"] == "compaction"
    assert result.steps[2].events[0]["status"] == "error"
    assert result.steps[2].events[0]["error"]["type"] == "ModelError"
    assert result.steps[4].events[0]["status"] == "ok"
    assert result.steps[4].events[0]["summary"]["current_state"] == "Recovered summary"

    compaction_chats = [chat for chat in model.chats if "Compact older terminal activity" in chat]
    assert len(compaction_chats) == 2
    for marker in ("Step 1", "Step 2", "Step 3", "Step 4"):
        assert marker in compaction_chats[1]


def test_failed_compaction_bounds_final_memory_commit_history(tmp_path):
    screens = [
        "OLDEST-1 " + "a" * 90,
        "MIDDLE-2 " + "b" * 90,
        "NEWEST-3 " + "c" * 90,
    ]
    model = PromptRecordingModel(
        [
            '{"action": "submit_line", "arguments": {"text": "one"}}',
            "",
            '{"action": "submit_line", "arguments": {"text": "two"}}',
            "",
            '{"action": "hangup", "arguments": {}}',
            '{"durable_facts": ["Done."]}',
        ]
    )
    runner = ActivityRunner(
        ActivityProfile(
            name="bbs-menu",
            objective="test bounded final memory",
            recent_steps_to_keep=1,
            screen_tail_chars=100,
            compact_every_steps=1,
            compact_recent_chars=450,
        ),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
    )

    result = runner.run(SequencedScreenAgent(screens), model, ActivityBudget(max_decision_ticks=3))

    assert result.stop_reason == "hangup"
    memory_chat = next(chat for chat in model.chats if "memory patch" in chat)
    recent_text = memory_chat.split("Recent steps:\n", 1)[1].split("\n\nFinal screen:", 1)[0]
    assert len(recent_text) <= 450
    assert "Older unsummarized step context omitted" in recent_text
    assert "NEWEST-3" in recent_text
    assert "OLDEST-1" not in recent_text


class FailFirstDecideModel(ScriptedModelAdapter):
    def __init__(self, responses: list[str], failures: int = 1) -> None:
        super().__init__(responses)
        self.failures = failures

    def decide(self, prompt, policy=None):
        if self.failures:
            self.failures -= 1
            raise ModelError("provider unreachable")
        return super().decide(prompt, policy)


def test_stateful_delta_resends_bootstrap_after_model_error(tmp_path):
    model = FailFirstDecideModel(
        [
            '{"action": "wait", "arguments": {}}',
            '{"action": "hangup", "arguments": {}}',
            '{"durable_facts": []}',
        ]
    )
    profile = ActivityProfile(
        name="bbs-menu",
        objective="test bootstrap retry",
        prompt_mode="stateful_delta",
        model_error_retries=0,
    )

    result = ActivityRunner(profile, memory_store=JsonMemoryStore(tmp_path / "memory")).run(
        FakeAgent(),
        model,
        ActivityBudget(max_decision_ticks=3),
    )

    # The bootstrap never reached the provider on tick 1, so tick 2 must send it again.
    assert result.steps[0].action is None
    assert result.steps[0].prompt["stage"] == "bootstrap"
    assert result.steps[1].prompt["stage"] == "bootstrap"
    assert result.steps[1].action is not None
    assert result.steps[2].prompt["stage"] == "delta"


def test_stateful_delta_rebootstraps_after_provider_state_is_lost(tmp_path):
    class ExpiredModelStateError(ModelStateError):
        pass

    class StateLosingModel(ScriptedModelAdapter):
        def __init__(self):
            super().__init__(
                [
                    '{"action": "wait", "arguments": {}}',
                    '{"action": "hangup", "arguments": {}}',
                    '{"durable_facts": []}',
                ]
            )
            self.decision_calls = 0

        def decide(self, prompt, policy=None):
            self.decision_calls += 1
            if self.decision_calls == 2:
                raise ExpiredModelStateError(
                    "previous response was not found",
                    status_code=404,
                )
            return super().decide(prompt, policy)

    profile = ActivityProfile(
        name="bbs-menu",
        objective="test remote state recovery",
        prompt_mode="stateful_delta",
        model_error_retries=2,
    )

    result = ActivityRunner(profile, memory_store=JsonMemoryStore(tmp_path / "memory")).run(
        FakeAgent(),
        StateLosingModel(),
        ActivityBudget(max_decision_ticks=4),
    )

    assert [step.prompt["stage"] for step in result.steps] == ["bootstrap", "delta", "bootstrap"]
    assert result.steps[1].action is None
    assert len(result.steps[1].validation["model_errors"]) == 1
    assert result.steps[1].validation["model_errors"][0]["type"] == "ExpiredModelStateError"
    assert result.steps[1].validation["model_errors"][0]["status_code"] == 404
    assert result.steps[1].validation["requires_bootstrap"] is True
    assert result.stop_reason == "hangup"


class EncodeFailingAgent(FakeAgent):
    def act_action(self, action):
        if action.action == "submit_line":
            raise UnicodeEncodeError("cp437", "→", 0, 1, "character maps to <undefined>")
        return super().act_action(action)


def test_runner_survives_unencodable_action_text(tmp_path):
    model = ScriptedModelAdapter(
        [
            '{"action": "submit_line", "arguments": {"text": "→"}}',
            '{"action": "hangup", "arguments": {}}',
            '{"durable_facts": []}',
        ]
    )

    result = ActivityRunner(
        ActivityProfile(name="bbs-menu", objective="test encode failure"),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
    ).run(EncodeFailingAgent(), model, ActivityBudget(max_decision_ticks=3))

    assert result.stop_reason == "hangup"
    assert "cp437" in json.dumps(result.steps[0].validation)
    assert result.steps[0].budget["validation_failures"] == 1


def test_routed_runner_routes_through_step_api(tmp_path):
    agent = SequencedScreenAgent(["Command:", "TradeWars2 menu", "TradeWars2 menu"])
    model = ScriptedModelAdapter(
        [
            '{"action": "wait", "arguments": {}}',
            '{"action": "wait", "arguments": {}}',
        ]
    )
    default_profile = ActivityProfile(name="bbs-safe", objective="default")
    tw2_profile = ActivityProfile(name="tw2-game", objective="tw2")
    runner = RoutedActivityRunner(
        "auto",
        default_profile,
        (
            ActivityRoute(
                name="tw2",
                profile=tw2_profile,
                matches=lambda observation: "TradeWars2" in observation.model_text,
                priority=10,
            ),
        ),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
    )
    state = runner.start_state(agent, model, ActivityBudget(max_decision_ticks=2))

    first = runner.run_step(state, stop_on_completion=False)
    second = runner.run_step(state, stop_on_completion=False)

    # Routing must work through the external stepping API, not only run().
    assert first.active_profile == "bbs-safe"
    assert second.active_profile == "tw2-game"
    assert second.events[0]["type"] == "profile_switch"
    assert state.active_route_name == "tw2"


def test_zero_recent_steps_window_means_empty_not_unbounded(tmp_path):
    agent = FakeAgent()
    model = ScriptedModelAdapter(
        [
            '{"action": "wait", "arguments": {}}',
            '{"action": "hangup", "arguments": {}}',
            '{"durable_facts": []}',
        ]
    )

    result = ActivityRunner(
        ActivityProfile(name="bbs-menu", objective="test zero window", recent_steps_to_keep=0),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
    ).run(agent, model, ActivityBudget(max_decision_ticks=3))

    # [-0:] would have leaked the entire history into the second prompt.
    assert "Recent steps:\n(none)" in result.steps[1].prompt["user"]


def test_failed_memory_commit_retries_and_is_recorded(tmp_path):
    class CommitFailingModel(ScriptedModelAdapter):
        def __init__(self, responses):
            super().__init__(responses)
            self.memory_calls = 0

        def memory_chat(self, messages):
            self.memory_calls += 1
            raise ModelError("memory provider down")

    model = CommitFailingModel(
        [
            '{"action": "submit_line", "arguments": {"text": "?"}}',
            '{"action": "hangup", "arguments": {}}',
        ]
    )
    memory = JsonMemoryStore(tmp_path / "memory")
    runner = ActivityRunner(
        ActivityProfile(name="bbs-menu", objective="test commit failure", model_error_retries=1),
        memory_store=memory,
        log_path=tmp_path / "steps.jsonl",
    )

    result = runner.run(FakeAgent(), model, ActivityBudget(max_decision_ticks=3))

    assert result.stop_reason == "hangup"
    # Bounded retry: 1 + model_error_retries attempts, no silent single try.
    assert model.memory_calls == 2
    # The failure is durable in the activity log, not swallowed.
    records = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    failures = [record for record in records if record.get("type") == "memory_commit_failed"]
    assert len(failures) == 1
    assert "memory provider down" in failures[0]["error"]
    # Campaign memory is unchanged rather than polluted or half-written.
    assert memory.load("agent-001") == {}


def test_successful_empty_memory_patch_bounds_existing_legacy_document(tmp_path):
    memory = JsonMemoryStore(tmp_path / "memory")
    memory.save(
        "agent-001",
        {
            "durable_facts": [f"important-{index}-" + "x" * 100 for index in range(8)],
            "strategy_notes": ["repeat", "repeat", "y" * 100],
        },
    )
    limits = LegacyMemoryLimits(
        campaign_max_chars=180,
        campaign_max_string_chars=30,
        campaign_max_list_items=3,
    )
    model = ScriptedModelAdapter(['{"action":"hangup","arguments":{}}', "{}"])

    result = ActivityRunner(
        ActivityProfile(name="bbs-menu", objective="test old memory convergence", legacy_memory_limits=limits),
        memory_store=memory,
    ).run(FakeAgent(), model, ActivityBudget(max_decision_ticks=1))

    assert result.stop_reason == "hangup"
    saved = memory.load("agent-001")
    assert len(json.dumps(saved, ensure_ascii=False, separators=(",", ":"), sort_keys=True)) <= 180
    assert all(len(item) <= 30 for items in saved.values() for item in items)
    records = _memory_journal(tmp_path)
    assert [record["op"] for record in records] == ["legacy_merge_patch", "legacy_bound_campaign"]
    assert records[0]["fields"]["patch"] == {}
    assert records[1]["fields"]["bounds"]["changed"] is True


def test_failed_memory_commit_leaves_existing_legacy_document_unchanged(tmp_path):
    class CommitFailingModel(ScriptedModelAdapter):
        def memory_chat(self, messages):
            del messages
            raise ModelError("memory provider down")

    memory = JsonMemoryStore(tmp_path / "memory")
    original = {"durable_facts": ["x" * 200, "keep this exact old document"]}
    memory.save("agent-001", original)
    limits = LegacyMemoryLimits(
        campaign_max_chars=100,
        campaign_max_string_chars=20,
        campaign_max_list_items=1,
    )

    ActivityRunner(
        ActivityProfile(
            name="bbs-menu",
            objective="test failed commit isolation",
            model_error_retries=0,
            legacy_memory_limits=limits,
        ),
        memory_store=memory,
    ).run(
        FakeAgent(),
        CommitFailingModel(['{"action":"hangup","arguments":{}}']),
        ActivityBudget(max_decision_ticks=1),
    )

    assert memory.load("agent-001") == original


def _memory_journal(tmp_path: Path) -> list[dict]:
    from tty_agent.memory_subsystem import read_journal_records

    return read_journal_records(tmp_path / "memory" / "agent-001" / "ops.jsonl")


def test_legacy_journal_records_compaction_and_commit(tmp_path):
    agent = SequencedScreenAgent(["S1", "S2", "S3", "S4"])
    model = ScriptedModelAdapter(
        [
            '{"action": "submit_line", "arguments": {"text": "one"}}',
            '{"action": "submit_line", "arguments": {"text": "two"}}',
            '{"action": "submit_line", "arguments": {"text": "three"}}',
            '{"current_state": "Compacted", "last_error": "", "open_subgoals": [], "discovered_facts": [], "failed_actions": [], "strategy_notes": []}',
            '{"action": "hangup", "arguments": {}}',
            '{"durable_facts": ["Done."]}',
        ]
    )
    runner = ActivityRunner(
        ActivityProfile(
            name="bbs-menu",
            objective="test legacy journal",
            recent_steps_to_keep=1,
            compact_every_steps=3,
        ),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
    )

    runner.run(agent, model, ActivityBudget(max_decision_ticks=6))

    records = _memory_journal(tmp_path)
    assert [record["op"] for record in records] == ["legacy_replace_summary", "legacy_merge_patch"]
    summary_record, commit_record = records
    assert summary_record["accepted"] is True
    assert summary_record["batch"] == "reconcile"
    assert summary_record["source_steps"] == [1, 2, 3]
    assert summary_record["fields"]["summary"]["current_state"] == "Compacted"
    assert commit_record["accepted"] is True
    assert commit_record["batch"] == "commit"
    assert commit_record["fields"]["patch"] == {"durable_facts": ["Done."]}
    # Same record schema as the structured subsystem's journal.
    from tty_agent.memory_subsystem import mutation_record

    expected_keys = set(mutation_record(op="x", origin="model", accepted=True, batch="reconcile"))
    assert all(set(record) == expected_keys for record in records)


def test_legacy_journal_records_failures(tmp_path):
    class FailingUtilityModel(ScriptedModelAdapter):
        def compaction_chat(self, messages):
            raise ModelError("compaction provider down")

        def memory_chat(self, messages):
            raise ModelError("memory provider down")

    model = FailingUtilityModel(
        [
            '{"action": "submit_line", "arguments": {"text": "one"}}',
            '{"action": "submit_line", "arguments": {"text": "two"}}',
            '{"action": "hangup", "arguments": {}}',
        ]
    )
    runner = ActivityRunner(
        ActivityProfile(
            name="bbs-menu",
            objective="test legacy journal failures",
            compact_every_steps=2,
            model_error_retries=0,
        ),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
    )

    result = runner.run(FakeAgent(), model, ActivityBudget(max_decision_ticks=4))

    assert result.stop_reason == "hangup"
    records = _memory_journal(tmp_path)
    assert [(record["op"], record["accepted"]) for record in records] == [
        ("legacy_replace_summary", False),
        ("legacy_merge_patch", False),
    ]
    assert "compaction provider down" in records[0]["reason"]
    assert "memory provider down" in records[1]["reason"]
