from pathlib import Path

from bbs_gym.activities import TW2_ENTRY_PROFILE, activity_profile
from bbs_gym.prompt_modules import BBS_PROMPT_MODULES, TW2_INPUT_MODALITY_PROFILE, TW2_PROMPT_MODULES
from terminal_agent.actions import Action
from terminal_agent.hints import ObservationHints
from terminal_agent.prompt_modules import (
    GENERIC_TERMINAL_MODULES,
    PromptRenderContext,
    collect_prompt_module_results,
    render_prompt_modules,
)
from terminal_agent.runner import ActivityBudget
from terminal_agent.terminal import Observation


def observation(text):
    return Observation(
        agent_id="agent",
        pretty_screen=text,
        model_text=text,
        new_text=text,
        cursor=(0, 0),
        stable_ms=300,
        byte_quiet_ms=300,
        matched_prompt=None,
        ready_reason="stable",
        profile="test",
        transcript_path=Path("runtime/transcripts/test.raw"),
        transcript_byte_start=0,
        transcript_byte_end=len(text),
        bytes_read=len(text),
        timed_out=False,
        timestamp=0.0,
        metadata={"requested_node": 1},
    )


def test_tw2_entry_profile_exits_when_trade_wars_visible():
    assert TW2_ENTRY_PROFILE.should_exit(
        observation("Welcome to Trade Wars (v.ii)"),
        Action("wait"),
        ActivityBudget(),
    )


def test_tw2_entry_profile_does_not_exit_on_bbs_menu_listing():
    assert not TW2_ENTRY_PROFILE.should_exit(
        observation("  2 | Trade Wars 2 - 500 Sectors\nWhich or (Q)uit:"),
        Action("wait"),
        ActivityBudget(),
    )


def test_activity_profile_factory_returns_tw2_entry_profile():
    assert activity_profile("tw2-entry").name == "tw2-entry"


def test_activity_profile_factory_overrides_named_profile_objectives():
    tw2_game = activity_profile("tw2-game", "custom game objective")
    tw2_entry = activity_profile("tw2-entry", "custom entry objective")

    assert tw2_game.objective == "custom game objective"
    assert tw2_entry.objective == "custom entry objective"
    assert tw2_entry.should_exit(
        observation("Welcome to Trade Wars (v.ii)"),
        None,
        ActivityBudget(),
    )


def test_tw2_input_modality_profile_classifies_value_prompts():
    obs = observation("How many fighters do you want to buy [0]-1?")
    hints = ObservationHints.from_observation(
        obs,
        previous_observation=None,
        last_action=None,
        modality_profile=TW2_INPUT_MODALITY_PROFILE,
    )

    assert hints.input_mode == "line_input_expected"
    assert "submit_line" in hints.input_mode_hint


def test_prompt_module_assistance_levels_are_ablatable():
    obs = observation("Command (?=Help)?")
    context = PromptRenderContext(
        agent_id="agent",
        activity_name="tw2-game",
        objective="test",
        observation=obs,
        hints=ObservationHints.from_observation(
            obs,
            previous_observation=None,
            last_action=None,
            modality_profile=TW2_INPUT_MODALITY_PROFILE,
        ),
        recent_steps=(),
        campaign_memory={},
        session_summary=None,
        budget=None,
    )

    generic_prompt = render_prompt_modules(collect_prompt_module_results(GENERIC_TERMINAL_MODULES, context))
    bbs_prompt = render_prompt_modules(collect_prompt_module_results(BBS_PROMPT_MODULES, context))
    tw2_prompt = render_prompt_modules(collect_prompt_module_results(TW2_PROMPT_MODULES, context))

    assert len(generic_prompt) < len(bbs_prompt) < len(tw2_prompt)
    assert "[generic_terminal]" in generic_prompt
    assert "[bbs_conventions]" in bbs_prompt
    assert "[game_interface]" in tw2_prompt
