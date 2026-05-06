from pathlib import Path

from terminal_agent.actions import Action
from terminal_agent.hints import InputModalityProfile, InputModeRule, ObservationHints
from terminal_agent.terminal import Observation


def observation(
        model_text: str,
        *,
        pretty_screen: str | None = None,
        new_text: str | None = None,
        cursor: tuple[int, int] = (0, 0),
) -> Observation:
    return Observation(
        agent_id="agent",
        pretty_screen=pretty_screen if pretty_screen is not None else model_text,
        model_text=model_text,
        new_text=model_text if new_text is None else new_text,
        cursor=cursor,
        stable_ms=300,
        matched_prompt=None,
        ready_reason="stable",
        profile="test",
        transcript_path=Path("runtime/transcripts/test.raw"),
        bytes_read=len(model_text),
        timed_out=False,
        timestamp=0.0,
        metadata={},
    )


def test_observation_hints_use_cursor_line_as_active_prompt():
    pretty_screen = "\n".join(
        [
            "Your offer?",
            "Some older output",
            "Command (?=Help)?",
            "",
        ]
    )
    hints = ObservationHints.from_observation(
        observation(
            "Your offer?\nSome older output\nCommand (?=Help)?",
            pretty_screen=pretty_screen,
            cursor=(2, 16),
        ),
        previous_observation=None,
        last_action=None,
        modality_profile=InputModalityProfile(),
    )

    assert hints.active_prompt == "Command (?=Help)?"
    assert hints.input_mode == "unknown"
    assert hints.input_mode_hint == "inspect the screen"


def test_observation_hints_classify_with_domain_supplied_rules():
    profile = InputModalityProfile(
        rules=(
            InputModeRule.from_pattern(
                mode="line_input_expected",
                pattern=r"quantity\?\s*$",
                hint="type a value and submit it",
            ),
        )
    )
    hints = ObservationHints.from_observation(
        observation("Quantity?", cursor=(0, 9)),
        previous_observation=None,
        last_action=None,
        modality_profile=profile,
    )

    assert hints.input_mode == "line_input_expected"
    assert hints.input_mode_hint == "type a value and submit it"


def test_observation_hints_detect_echoed_key_without_other_output():
    previous = observation("Quantity?", new_text="Quantity?", cursor=(0, 9))
    current = observation("Quantity? 0", new_text="0", cursor=(0, 11))
    hints = ObservationHints.from_observation(
        current,
        previous_observation=previous,
        last_action=Action("press_key", key="0"),
        modality_profile=InputModalityProfile(),
    )

    assert len(hints.previous_action_effects) == 1
    assert "echoed on screen" in hints.previous_action_effects[0]


def test_observation_hints_detect_echo_followed_by_redrawn_prompt_suffix():
    previous = observation("Quantity?", new_text="Quantity?", cursor=(0, 9))
    current = observation("Quantity? 0 Command (?=Help)?", new_text="0 Command (?=Help)?", cursor=(0, 27))
    hints = ObservationHints.from_observation(
        current,
        previous_observation=previous,
        last_action=Action("press_key", key="0"),
        modality_profile=InputModalityProfile(),
    )

    assert len(hints.previous_action_effects) == 1
    assert "echoed on screen" in hints.previous_action_effects[0]


def test_observation_hints_detect_unchanged_screen_after_input():
    previous = observation("Command:", new_text="Command:", cursor=(0, 8))
    current = observation("Command:", new_text="", cursor=(0, 8))
    hints = ObservationHints.from_observation(
        current,
        previous_observation=previous,
        last_action=Action("press_key", key="enter"),
        modality_profile=InputModalityProfile(),
    )

    assert hints.previous_action_effects == ("Screen appears unchanged after the previous input.",)
