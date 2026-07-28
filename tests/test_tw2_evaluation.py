import json
from pathlib import Path

from bbs_gym.activities import TW2_GAME_PROFILE
from bbs_gym.evaluation import TW2_EVALUATION_PROFILE, extract_tw2_metrics, tw2_score_probe_ready
from tty_agent.actions import Action
from tty_agent.agent import ActionExecution
from tty_agent.memory import JsonMemoryStore
from tty_agent.models import ScriptedModelAdapter
from tty_agent.runner import ActivityBudget, ActivityRunner
from tty_agent.terminal import Observation


TW2_FINAL_STATUS = """\
<Info>
   Pilot's Name: RLoginSmoke  Team [3]
       Fighters: 20
Sector Location: 1
     Turns left: 9
    Cargo Holds: 20
     # with Ore: 0
     # with Org: 1
     # with Equ: 0
        Credits: 219
    Door points: 100,000

Command (?=Help)? C
<Computer activated>
Computer command (?=help)? R
Ranking Players.

  T R A D E W A R S   I I - 500T   S C O R E B O A R D

Player Rankings
Rank     Value      Team   Player
==== ============= ====== ================
   1         12000        OtherPilot
   2         12739      3 RLoginSmoke

Computer command (?=help)? X
<Computer deactivated>
Command (?=Help)? """


def observation(
        text: str,
        byte_start: int = 0,
        *,
        model_text: str | None = None,
        metadata: dict[str, object] | None = None,
) -> Observation:
    return Observation(
        agent_id="tw2-agent",
        pretty_screen=model_text or text,
        model_text=model_text or text,
        new_text=text,
        cursor=(0, len(model_text or text)),
        stable_ms=300,
        byte_quiet_ms=50,
        matched_prompt="tw2-command",
        ready_reason="prompt",
        profile="tw2",
        transcript_path=Path("runtime/transcripts/tw2-test.raw"),
        transcript_byte_start=byte_start,
        transcript_byte_end=byte_start + len(text),
        bytes_read=len(text),
        timed_out=False,
        timestamp=float(byte_start),
        metadata=metadata or {},
    )


class EvaluationAgent:
    agent_id = "tw2-agent"

    def __init__(self, observations: list[Observation]) -> None:
        self.observations = list(observations)
        self.actions: list[Action] = []

    def observe_turn(self, **_kwargs) -> Observation:
        if not self.observations:
            raise AssertionError("unexpected terminal observation")
        return self.observations.pop(0)

    def act_action(self, action: Action) -> ActionExecution:
        self.actions.append(action)
        if action.action == "press_key":
            return ActionExecution(sent_bytes=(action.key.encode("cp437"),), encoding="cp437")
        if action.action == "type_text":
            return ActionExecution(sent_bytes=(action.text.encode("cp437"),), encoding="cp437")
        return ActionExecution(encoding="cp437")


class PromptRecordingModel(ScriptedModelAdapter):
    def __init__(self, responses: list[str]) -> None:
        super().__init__(responses)
        self.prompts: list[str] = []

    def chat(self, messages):
        self.prompts.append("\n".join(message.content for message in messages))
        return super().chat(messages)


def test_tw2_extractor_uses_leaderboard_value_as_score_and_keeps_door_points():
    metrics = extract_tw2_metrics(observation(TW2_FINAL_STATUS))

    assert metrics == {
        "pilot": "RLoginSmoke",
        "fighters": 20,
        "sector": 1,
        "turns_left": 9,
        "cargo_holds": 20,
        "cargo_ore": 0,
        "cargo_organics": 1,
        "cargo_equipment": 0,
        "credits": 219,
        "door_points": 100000,
        "onboard_value": 12239,
        "score": 12739,
        "rank": 2,
        "team": 3,
        "deployed_fighters": 5,
    }


def test_tw2_extractor_can_resolve_a_passive_ranking_from_bbs_metadata():
    metrics = extract_tw2_metrics(
        observation(
            """\
Player Rankings
Rank     Value      Team   Player
==== ============= ====== ================
   1         12239        RLoginSmoke

Computer command (?=help)? """,
            metadata={"bbs_alias": "RLoginSmoke"},
        )
    )

    assert metrics == {"score": 12239, "rank": 1}


def test_tw2_extractor_records_partial_credit_and_turn_updates():
    metrics = extract_tw2_metrics(
        observation("You now have 1,250 credits.\nYou have 7 turns left.\nCommand (?=Help)? ")
    )

    assert metrics == {"credits": 1250, "turns_left": 7}


def test_tw2_probe_only_runs_at_safe_menu_prompts():
    assert tw2_score_probe_ready(observation("Command (?=Help)? "))
    assert tw2_score_probe_ready(observation("Computer command (?=help)? "))
    assert not tw2_score_probe_ready(observation("What sector number is the port in? "))
    assert not tw2_score_probe_ready(observation("Command (?=Help)?\nHow many holds of Ore do you want? "))


def test_final_tw2_probe_is_hidden_and_does_not_consume_a_decision_tick(tmp_path):
    agent = EvaluationAgent(
        [
            observation("Sector 1\nCommand (?=Help)? ", 0),
            observation("D\nSector 1 redisplayed.\nCommand (?=Help)? ", 40),
            observation(TW2_FINAL_STATUS, 100),
        ]
    )
    model = PromptRecordingModel(
        [
            '{"action": "press_key", "arguments": {"key": "D"}}',
            '{"durable_facts": ["Ended in sector 1."]}',
        ]
    )
    metrics_path = tmp_path / "metrics.jsonl"
    runner = ActivityRunner(
        TW2_GAME_PROFILE,
        memory_store=JsonMemoryStore(tmp_path / "memory"),
        log_path=tmp_path / "steps.jsonl",
        evaluation_profile=TW2_EVALUATION_PROFILE,
        evaluation_log_path=metrics_path,
    )

    result = runner.run(agent, model, ActivityBudget(max_decision_ticks=1))

    assert result.stop_reason == "budget"
    assert result.decision_ticks == 1
    assert [action.to_dict() for action in agent.actions] == [
        {"action": "press_key", "arguments": {"key": "D"}},
        {"action": "type_text", "arguments": {"text": "XICRX"}},
    ]
    assert len(result.steps) == 2
    assert result.evaluation.final_metrics["score"] == 12739
    assert result.evaluation.final_metrics["door_points"] == 100000
    probe_record = result.evaluation.records[-1]
    assert probe_record.source == "final_probe"
    assert probe_record.visible_to_model is False
    assert probe_record.probe_turn_cost == "none"
    assert probe_record.decision_tick == 1
    assert "Player Rankings" not in model.prompts[-1]
    assert "Sector 1 redisplayed" in model.prompts[-1]
    assert JsonMemoryStore(tmp_path / "memory").load("tw2-agent") == {
        "durable_facts": ["Ended in sector 1."]
    }

    logged_metrics = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert len(logged_metrics) == 1
    assert logged_metrics[0]["metrics"]["score"] == 12739
    assert logged_metrics[0]["action"] == {"action": "type_text", "arguments": {"text": "XICRX"}}
