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
        byte_quiet_ms=300,
        matched_prompt=None,
        ready_reason="stable",
        profile="test",
        transcript_path=Path("runtime/transcripts/test.raw"),
        transcript_byte_start=0,
        transcript_byte_end=len(model_text if new_text is None else new_text),
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


def test_observation_hints_recovers_active_prompt_from_cr_redraw():
    pretty_screen = "\n".join(
        [
            "Warps lead to   2, 3, 4, 5, 6, 7",
            "",
            "",
        ]
    )
    new_text = "Warps lead to   2, 3, 4, 5, 6, 7\r\n\r\nCommand (?=Help)? \rCommand (?=Help)? \r                  "
    hints = ObservationHints.from_observation(
        observation(
            "Warps lead to 2, 3, 4, 5, 6, 7",
            pretty_screen=pretty_screen,
            new_text=new_text,
            cursor=(2, 18),
        ),
        previous_observation=None,
        last_action=None,
        modality_profile=InputModalityProfile(),
    )

    assert hints.active_prompt == "Command (?=Help)?"


def test_observation_hints_prefers_new_text_prompt_over_stale_pretty_prompt():
    pretty_screen = "\n".join(
        [
            "Your offer?",
            "Warps lead to   2, 3, 4, 5, 6, 7",
            "",
            "",
        ]
    )
    new_text = "Warps lead to   2, 3, 4, 5, 6, 7\r\n\r\nCommand (?=Help)? \rCommand (?=Help)? \r                  "
    hints = ObservationHints.from_observation(
        observation(
            "Your offer?\nWarps lead to 2, 3, 4, 5, 6, 7",
            pretty_screen=pretty_screen,
            new_text=new_text,
            cursor=(3, 18),
        ),
        previous_observation=None,
        last_action=None,
        modality_profile=InputModalityProfile(),
    )

    assert hints.active_prompt == "Command (?=Help)?"


def test_observation_hints_does_not_treat_brackets_as_prompt_evidence():
    hints = ObservationHints.from_observation(
        observation("[ Scanning   0.0% ][ Done      100.0% ]", cursor=(0, 36)),
        previous_observation=None,
        last_action=None,
        modality_profile=InputModalityProfile(),
    )

    assert hints.active_prompt == "(unknown - inspect the screen)"


def test_observation_hints_recognizes_hit_key_prompt_with_trailing_noise():
    hints = ObservationHints.from_observation(
        observation("[Hit a key] #", cursor=(0, 13)),
        previous_observation=None,
        last_action=None,
        modality_profile=InputModalityProfile(),
    )

    assert hints.active_prompt == "[Hit a key] #"


def test_observation_hints_recognizes_yes_no_choice_without_trailing_punctuation():
    hints = ObservationHints.from_observation(
        observation("[+] Log off? [No] Yes", cursor=(0, 21)),
        previous_observation=None,
        last_action=None,
        modality_profile=InputModalityProfile(),
    )

    assert hints.active_prompt == "[+] Log off? [No] Yes"


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
