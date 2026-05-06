from pathlib import Path

from bbs_gym.prompt_modules import (
    BBS_INPUT_MODALITY_PROFILE,
    BBS_PROMPT_MODULES,
    TW2_INPUT_MODALITY_PROFILE,
    TW2_PROMPT_MODULES,
    RLoginAuthenticatedModule,
)
from terminal_agent.hints import ObservationHints
from terminal_agent.models import SessionSummary
from terminal_agent.prompt_modules import (
    GENERIC_TERMINAL_MODULES,
    PromptRenderContext,
    collect_prompt_module_results,
    render_prompt_modules,
)
from terminal_agent.runner import ActivityBudget
from terminal_agent.terminal import Observation


def observation(
        text: str,
        metadata: dict[str, object] | None = None,
        *,
        pretty_screen: str | None = None,
        new_text: str | None = None,
        cursor: tuple[int, int] | None = None,
) -> Observation:
    return Observation(
        agent_id="agent",
        pretty_screen=pretty_screen if pretty_screen is not None else text,
        model_text=text,
        new_text=new_text if new_text is not None else text,
        cursor=cursor if cursor is not None else (0, len(text)),
        stable_ms=300,
        matched_prompt=None,
        ready_reason="stable",
        profile="test",
        transcript_path=Path("runtime/transcripts/test.raw"),
        bytes_read=len(text),
        timed_out=False,
        timestamp=0.0,
        metadata=metadata or {},
    )


def context(text: str, metadata: dict[str, object] | None = None) -> PromptRenderContext:
    obs = observation(text, metadata)
    return PromptRenderContext(
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
        session_summary=SessionSummary(),
        budget=ActivityBudget(),
    )


def test_bbs_modality_profile_classifies_bracketed_choice_as_hotkey():
    mode, hint = BBS_INPUT_MODALITY_PROFILE.classify(
        active_prompt="Search all groups for new messages? Yes [No]",
        recent_output="",
        screen_tail="Search all groups for new messages? Yes [No]",
    )

    assert mode == "hotkey_expected"
    assert "obvious printable key" in hint


def test_bbs_modality_profile_leaves_mixed_value_and_shortcut_prompt_unknown():
    mode, hint = BBS_INPUT_MODALITY_PROFILE.classify(
        active_prompt="Enter number of bulletin or [(Q)uit]:",
        recent_output="",
        screen_tail="Enter number of bulletin or [(Q)uit]:",
    )

    assert mode == "unknown"
    assert hint == "inspect the screen"


def test_tw2_modality_profile_prioritizes_tw2_line_input_hint():
    mode, hint = TW2_INPUT_MODALITY_PROFILE.classify(
        active_prompt="How many holds of organics do you want to buy [20]?",
        recent_output="",
        screen_tail="How many holds of organics do you want to buy [20]?",
    )

    assert mode == "line_input_expected"
    assert "already-typed text" in hint


def test_tw2_modality_profile_classifies_tw2_command_prompt():
    mode, hint = TW2_INPUT_MODALITY_PROFILE.classify(
        active_prompt="Command (?=Help)?",
        recent_output="",
        screen_tail="Command (?=Help)?",
    )

    assert mode == "hotkey_expected"
    assert "one-character commands" in hint


def test_tw2_modality_profile_classifies_cr_redraw_command_prompt():
    pretty_screen = "\n".join(
        [
            "Warps lead to   2, 3, 4, 5, 6, 7",
            "",
            "",
        ]
    )
    new_text = "Warps lead to   2, 3, 4, 5, 6, 7\r\n\r\nCommand (?=Help)? \rCommand (?=Help)? \r                  "
    obs = observation(
        "Warps lead to 2, 3, 4, 5, 6, 7",
        pretty_screen=pretty_screen,
        new_text=new_text,
        cursor=(2, 18),
    )
    hints = ObservationHints.from_observation(
        obs,
        previous_observation=None,
        last_action=None,
        modality_profile=TW2_INPUT_MODALITY_PROFILE,
    )

    assert hints.active_prompt == "Command (?=Help)?"
    assert hints.input_mode == "hotkey_expected"
    assert "press_key" in hints.input_mode_hint


def test_rlogin_authenticated_module_only_renders_for_rlogin_transport():
    module = RLoginAuthenticatedModule()

    assert module.render(context("Welcome", {"transport": "telnet"})) is None
    assert "already authenticated" in module.render(context("Welcome", {"transport": "rlogin"}))


def test_bbs_and_tw2_module_sets_have_expected_sizes():
    assert len(GENERIC_TERMINAL_MODULES) == 5
    assert len(BBS_PROMPT_MODULES) == 7
    assert len(TW2_PROMPT_MODULES) == 9


def test_prompt_module_assistance_levels_are_ablatable():
    render_context = context("Command (?=Help)?")

    generic_prompt = render_prompt_modules(collect_prompt_module_results(GENERIC_TERMINAL_MODULES, render_context))
    bbs_prompt = render_prompt_modules(collect_prompt_module_results(BBS_PROMPT_MODULES, render_context))
    tw2_prompt = render_prompt_modules(collect_prompt_module_results(TW2_PROMPT_MODULES, render_context))

    assert len(generic_prompt) < len(bbs_prompt) < len(tw2_prompt)
    assert "[generic_terminal]" in generic_prompt
    assert "[bbs_conventions]" not in generic_prompt
    assert "[game_interface]" not in generic_prompt
    assert "[bbs_conventions]" in bbs_prompt
    assert "[game_interface]" not in bbs_prompt
    assert "[game_interface]" in tw2_prompt
