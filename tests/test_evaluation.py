import importlib.util
import json
from pathlib import Path

from tty_agent.actions import Action
from tty_agent.agent import ActionExecution
from tty_agent.memory import JsonMemoryStore
from tty_agent.models import ScriptedModelAdapter
from tty_agent.runner import ActivityBudget, ActivityProfile, ActivityRunner
from tty_agent.terminal import Observation


_MODULE_PATH = Path(__file__).parent.parent / "examples" / "zork_activity.py"
_SPEC = importlib.util.spec_from_file_location("zork_activity", _MODULE_PATH)
assert _SPEC is not None
zork_activity = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(zork_activity)

ZORK_EVALUATION_PROFILE = zork_activity.ZORK_EVALUATION_PROFILE
extract_zork_metrics = zork_activity.extract_zork_metrics


def observation(text: str, byte_start: int = 0) -> Observation:
    return Observation(
        agent_id="zork-agent",
        pretty_screen=text,
        model_text=text,
        new_text=text,
        cursor=(0, len(text)),
        stable_ms=50,
        byte_quiet_ms=0,
        matched_prompt="command-prompt",
        ready_reason="prompt",
        profile="text-adventure",
        transcript_path=Path("runtime/transcripts/zork-test.raw"),
        transcript_byte_start=byte_start,
        transcript_byte_end=byte_start + len(text),
        bytes_read=len(text),
        timed_out=False,
        timestamp=float(byte_start),
        metadata={"game": "zork"},
    )


class EvaluationAgent:
    agent_id = "zork-agent"

    def __init__(self, observations: list[Observation]) -> None:
        self.observations = list(observations)
        self.actions: list[Action] = []

    def observe_turn(self, **_kwargs) -> Observation:
        if not self.observations:
            raise AssertionError("unexpected terminal observation")
        return self.observations.pop(0)

    def act_action(self, action: Action) -> ActionExecution:
        self.actions.append(action)
        if action.action == "submit_line":
            return ActionExecution(sent_bytes=(action.text.encode("utf-8"), b"\n"))
        return ActionExecution()


class PromptRecordingModel(ScriptedModelAdapter):
    def __init__(self, responses: list[str]) -> None:
        super().__init__(responses)
        self.prompts: list[str] = []

    def chat(self, messages):
        self.prompts.append("\n".join(message.content for message in messages))
        return super().chat(messages)


def test_zork_score_extractor_parses_standard_wrapped_response():
    metrics = extract_zork_metrics(
        observation(
            "score\r\nYour score is -2 (total of 350 points), in 17 moves.\r\nThis gives you the rank of Beginner.\r\n>"
        )
    )

    assert metrics == {
        "score": -2,
        "score_max": 350,
        "moves": 17,
        "rank": "Beginner",
    }

    singular_metrics = extract_zork_metrics(
        observation("Your score is 0 (total of 350 points), in 1 move. This gives you the rank of Beginner.")
    )
    assert singular_metrics is not None
    assert singular_metrics["moves"] == 1


def test_zork_provider_utility_reasoning_options(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "zork_activity.py",
            "story.z3",
            "--no-compaction-reasoning",
            "--compaction-extra-body-json",
            '{"response_format":{"type":"json_object"}}',
            "--memory-reasoning",
            "--memory-extra-body-json",
            '{"response_format":{"type":"json_object"}}',
            "--audit-temperature",
            "0.25",
        ],
    )

    args = zork_activity.parse_args()

    assert args.compaction_reasoning is False
    assert args.model_timeout == 600.0
    assert args.max_tokens_retry_ceiling == 16_384
    assert args.compaction_max_tokens == 16_384
    assert args.compaction_max_tokens_retry_ceiling == 32_768
    assert args.compaction_extra_body_json == {"response_format": {"type": "json_object"}}
    assert args.memory_reasoning is True
    assert args.memory_max_tokens == 32_768
    assert args.memory_max_tokens_retry_ceiling == 32_768
    assert args.memory_extra_body_json == {"response_format": {"type": "json_object"}}
    assert args.audit_temperature == 0.25


def test_zork_responses_state_options(monkeypatch, tmp_path):
    state_file = tmp_path / "zork.responses.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "zork_activity.py",
            "story.z3",
            "--model-api",
            "responses",
            "--responses-stateful",
            "--responses-state-file",
            str(state_file),
            "--responses-resume",
        ],
    )

    args = zork_activity.parse_args()

    assert args.model_api == "responses"
    assert args.responses_stateful is True
    assert args.responses_state_file == state_file
    assert args.responses_resume is True


def test_final_zork_probe_is_outside_decision_steps_and_model_context(tmp_path):
    agent = EvaluationAgent(
        [
            observation("West of House\n>", 0),
            observation("look\nKitchen\n>", 20),
            observation(
                "score\nYour score is 42 (total of 350 points), in 12 moves.\n"
                "This gives you the rank of Amateur Adventurer.\n>",
                40,
            ),
        ]
    )
    model = PromptRecordingModel(
        [
            '{"action": "submit_line", "arguments": {"text": "look"}}',
            '{"durable_facts": ["Ended in the kitchen."]}',
        ]
    )
    metrics_path = tmp_path / "metrics.jsonl"
    runner = ActivityRunner(
        ActivityProfile(name="zork", objective="play zork", prompt_fast_path=True),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
        log_path=tmp_path / "steps.jsonl",
        evaluation_profile=ZORK_EVALUATION_PROFILE,
        evaluation_log_path=metrics_path,
    )

    result = runner.run(agent, model, ActivityBudget(max_decision_ticks=1))

    assert result.stop_reason == "budget"
    assert result.decision_ticks == 1
    assert [action.to_dict() for action in agent.actions] == [
        {"action": "submit_line", "arguments": {"text": "look"}},
        {"action": "submit_line", "arguments": {"text": "score"}},
    ]
    assert len(result.steps) == 2
    assert result.steps[0].action == {"action": "submit_line", "arguments": {"text": "look"}}
    assert result.steps[1].validation["terminal"] is True
    assert result.steps[1].budget["decision_ticks"] == 1
    assert "Kitchen" in result.steps[1].observation["new_text"]
    assert result.evaluation.final_metrics == {
        "score": 42,
        "score_max": 350,
        "moves": 12,
        "rank": "Amateur Adventurer",
    }

    probe_record = result.evaluation.records[-1]
    assert probe_record.agent_id == "zork-agent"
    assert probe_record.source == "final_probe"
    assert probe_record.visible_to_model is False
    assert probe_record.probe_turn_cost == "none"
    assert probe_record.decision_tick == 1
    assert "Your score is 42" in probe_record.observation["new_text"]
    assert "Your score is 42" not in model.prompts[-1]
    assert "Kitchen" in model.prompts[-1]
    assert JsonMemoryStore(tmp_path / "memory").load("zork-agent") == {
        "durable_facts": ["Ended in the kitchen."]
    }

    logged_steps = (tmp_path / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(logged_steps) == 2
    logged_metrics = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert len(logged_metrics) == 1
    assert logged_metrics[0]["metrics"]["score"] == 42
    assert logged_metrics[0]["agent_id"] == "zork-agent"
    assert logged_metrics[0]["visible_to_model"] is False


def test_agent_requested_zork_score_is_passively_recorded_and_visible(tmp_path):
    score_text = (
        "score\nYour score is 15 (total of 350 points), in 8 moves.\nThis gives you the rank of Novice Adventurer.\n>"
    )
    agent = EvaluationAgent(
        [
            observation("West of House\n>", 0),
            observation(score_text, 20),
        ]
    )
    model = PromptRecordingModel(
        [
            '{"action": "submit_line", "arguments": {"text": "score"}}',
            '{"action": "hangup", "arguments": {}}',
            '{"durable_facts": ["Asked for score."]}',
        ]
    )
    runner = ActivityRunner(
        ActivityProfile(name="zork", objective="play zork"),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
        evaluation_profile=ZORK_EVALUATION_PROFILE,
    )

    result = runner.run(agent, model, ActivityBudget(max_decision_ticks=5))

    assert result.stop_reason == "hangup"
    assert [action.to_dict() for action in agent.actions] == [
        {"action": "submit_line", "arguments": {"text": "score"}},
        {"action": "hangup", "arguments": {}},
    ]
    passive_record = result.evaluation.records[0]
    assert passive_record.source == "agent_action"
    assert passive_record.visible_to_model is True
    assert passive_record.metrics["score"] == 15
    assert score_text in model.prompts[1]
