import json
from pathlib import Path

from bbs_gym.activities import SRE_GAME_PROFILE
from bbs_gym.evaluation import SRE_EVALUATION_PROFILE, extract_sre_metrics, sre_score_probe_ready
from tty_agent.actions import Action
from tty_agent.agent import ActionExecution
from tty_agent.memory import JsonMemoryStore
from tty_agent.models import ScriptedModelAdapter
from tty_agent.runner import ActivityBudget, ActivityRunner
from tty_agent.terminal import Observation


SRE_GALAXY_MENU = """\
───[Galaxy Menu]───
[1] Status
[2] New News
[3] Daily News
[4] Play SRE!
[5] See Your Status
[6] Read Messages
[7] Send Messages
[8] Scores
[9] Diplomacy
───────────────────
Which one? [Enter=4] """

SRE_STATUS = """\
Empire Status
───────────────────────────────────────────────────────────────────────────
-*SpreeProbe*-
Score: 123
Turns Left: 4
Money: 10,000
Pop: 40 Million (Tax Rate=23%)
Food: 0 Megatons
Covert: 0 Agents
Insurgency: Peaceful
Cmd. Ship: 0% completed
Military: [Effectiveness=100%] (Net Worth=12,345)
Planets: [Education=3] [Tourism=9] [Food=1] [Research=1] [Ore=9]
 [Government=1] [Urban=3] (Total=27)
You have 20 turns of protection left. You have 4 turns left today.
───────────────────────────────────────────────────────────────────────────
"""

SRE_SCOREBOARD = """\
List of Players/Scores:
ID  Empire                              Planets     Score   Net Worth Points
────────────────────────────────────────────────────────────────────────────
<A> Other Empire                            31       456      20,000      9
<B> SpreeProbe                              27       123      12,345      7
────────────────────────────────────────────────────────────────────────────
"""


def observation(
        text: str,
        byte_start: int = 0,
        *,
        model_text: str | None = None,
        metadata: dict[str, object] | None = None,
) -> Observation:
    return Observation(
        agent_id="sre-agent",
        pretty_screen=model_text or text,
        model_text=model_text or text,
        new_text=text,
        cursor=(0, len(model_text or text)),
        stable_ms=300,
        byte_quiet_ms=50,
        matched_prompt=None,
        ready_reason="stable",
        profile="synchronet-bbs",
        transcript_path=Path("runtime/transcripts/sre-test.raw"),
        transcript_byte_start=byte_start,
        transcript_byte_end=byte_start + len(text),
        bytes_read=len(text),
        timed_out=False,
        timestamp=float(byte_start),
        metadata=metadata or {},
    )


class EvaluationAgent:
    agent_id = "sre-agent"

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
        return ActionExecution(encoding="cp437")


class PromptRecordingModel(ScriptedModelAdapter):
    def __init__(self, responses: list[str]) -> None:
        super().__init__(responses)
        self.prompts: list[str] = []

    def chat(self, messages):
        self.prompts.append("\n".join(message.content for message in messages))
        return super().chat(messages)


def test_sre_extractor_reads_status_and_selects_the_players_scoreboard_row():
    metrics = extract_sre_metrics(observation(SRE_STATUS + SRE_SCOREBOARD))

    assert metrics == {
        "empire": "SpreeProbe",
        "score": 123,
        "turns_left": 4,
        "money": 10000,
        "net_worth": 12345,
        "planets": 27,
        "protection_turns_left": 20,
        "leaderboard": [
            {
                "player_id": "A",
                "empire": "Other Empire",
                "planets": 31,
                "score": 456,
                "net_worth": 20000,
                "points": 9,
            },
            {
                "player_id": "B",
                "empire": "SpreeProbe",
                "planets": 27,
                "score": 123,
                "net_worth": 12345,
                "points": 7,
            },
        ],
        "player_id": "B",
        "points": 7,
        "rank": 2,
    }


def test_sre_extractor_uses_a_single_scoreboard_row_without_status():
    metrics = extract_sre_metrics(
        observation(
            """\
List of Players/Scores:
ID Empire Planets Score Net Worth Points
<A> SpreeProbe 27 0 271 0
"""
        )
    )

    assert metrics == {
        "leaderboard": [
            {
                "player_id": "A",
                "empire": "SpreeProbe",
                "planets": 27,
                "score": 0,
                "net_worth": 271,
                "points": 0,
            }
        ],
        "player_id": "A",
        "empire": "SpreeProbe",
        "planets": 27,
        "score": 0,
        "net_worth": 271,
        "points": 0,
        "rank": 1,
    }


def test_sre_extractor_reads_money_and_bank_savings_from_later_turns():
    metrics = extract_sre_metrics(
        observation(
            SRE_STATUS.replace("Money: 10,000", "Money: 50,944 (Bank Savings=36,844)")
        )
    )

    assert metrics is not None
    assert metrics["money"] == 50944
    assert metrics["bank_savings"] == 36844


def test_sre_extractor_uses_rendered_screen_when_cursor_drawing_fragments_raw_delta():
    metrics = extract_sre_metrics(
        observation(
            "Empire Status\nScore:\n0\nTurns Left:\n4\nMoney:\n10,000\n",
            model_text=SRE_STATUS,
        )
    )

    assert metrics is not None
    assert metrics["empire"] == "SpreeProbe"
    assert metrics["score"] == 123
    assert metrics["money"] == 10000


def test_sre_probe_only_runs_at_the_complete_galaxy_menu_prompt():
    assert sre_score_probe_ready(observation(SRE_GALAXY_MENU))
    assert not sre_score_probe_ready(observation("[System]\n[8] Rankings\nWhich one? [Enter=2] "))
    assert not sre_score_probe_ready(observation(SRE_GALAXY_MENU + "\nEmpire Status\n░▒▓█PAUSED█▓▒░"))
    assert not sre_score_probe_ready(observation("Choose a name for your new empire:\n> "))


def test_final_sre_probe_is_hidden_and_does_not_consume_a_decision_tick(tmp_path):
    agent = EvaluationAgent(
        [
            observation(SRE_GALAXY_MENU, 0),
            observation(SRE_GALAXY_MENU, 100),
            observation(SRE_STATUS + "\n░▒▓█PAUSED█▓▒░", 200),
        ]
    )
    model = PromptRecordingModel(
        [
            '{"action": "press_key", "arguments": {"key": "1"}}',
            '{"durable_facts": ["Stayed at the Galaxy menu."]}',
        ]
    )
    metrics_path = tmp_path / "metrics.jsonl"
    runner = ActivityRunner(
        SRE_GAME_PROFILE,
        memory_store=JsonMemoryStore(tmp_path / "memory"),
        log_path=tmp_path / "steps.jsonl",
        evaluation_profile=SRE_EVALUATION_PROFILE,
        evaluation_log_path=metrics_path,
    )

    result = runner.run(agent, model, ActivityBudget(max_decision_ticks=1))

    assert result.stop_reason == "budget"
    assert result.decision_ticks == 1
    assert [action.to_dict() for action in agent.actions] == [
        {"action": "press_key", "arguments": {"key": "1"}},
        {"action": "press_key", "arguments": {"key": "5"}},
    ]
    assert len(result.steps) == 2
    assert result.evaluation.final_metrics["score"] == 123
    assert result.evaluation.final_metrics["money"] == 10000
    probe_record = result.evaluation.records[-1]
    assert probe_record.source == "final_probe"
    assert probe_record.visible_to_model is False
    assert probe_record.probe_turn_cost == "none"
    assert probe_record.decision_tick == 1
    assert "Empire Status" not in model.prompts[-1]
    assert "Galaxy Menu" in model.prompts[-1]
    assert JsonMemoryStore(tmp_path / "memory").load("sre-agent") == {
        "durable_facts": ["Stayed at the Galaxy menu."]
    }

    logged_metrics = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert len(logged_metrics) == 1
    assert logged_metrics[0]["metrics"]["score"] == 123
    assert logged_metrics[0]["action"] == {"action": "press_key", "arguments": {"key": "5"}}
