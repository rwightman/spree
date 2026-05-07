from pathlib import Path

from tty_agent.hints import ObservationHints
from tty_agent.prompt_modules import (
    GENERIC_TERMINAL_MODULES,
    PromptRenderContext,
    StaticPromptModule,
    collect_prompt_module_results,
    prompt_module_trace,
    render_prompt_modules,
)
from tty_agent.terminal import Observation


def observation(text: str) -> Observation:
    return Observation(
        agent_id="agent",
        pretty_screen=text,
        model_text=text,
        new_text=text,
        cursor=(0, len(text)),
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
        metadata={},
    )


def context() -> PromptRenderContext:
    obs = observation("Command:")
    return PromptRenderContext(
        agent_id="agent",
        activity_name="test",
        objective="test",
        observation=obs,
        hints=ObservationHints(recent_output="Command:", active_prompt="Command:"),
        recent_steps=(),
        campaign_memory={},
        session_summary=None,
        budget=None,
    )


def test_generic_prompt_modules_render_labeled_observation_sections():
    results = collect_prompt_module_results(GENERIC_TERMINAL_MODULES, context())
    rendered = render_prompt_modules(results)

    assert "[generic_terminal]" not in rendered
    assert "Most recent terminal output:\nCommand:" in rendered
    assert "Likely active prompt:\nCommand:" in rendered
    assert "Input mode hint:\nunknown - inspect the screen" in rendered
    assert rendered.rstrip().endswith("Full current screen:\nCommand:")


def test_prompt_modules_can_render_debug_level_headers():
    results = collect_prompt_module_results(GENERIC_TERMINAL_MODULES, context())
    rendered = render_prompt_modules(results, include_level_headers=True)

    assert "[generic_terminal]" in rendered
    assert "Most recent terminal output:\nCommand:" in rendered


def test_empty_module_results_are_traced_but_not_rendered():
    modules = (
        StaticPromptModule(name="silent", level="game_interface", text=""),
        StaticPromptModule(name="visible", level="game_interface", text="visible text"),
    )
    results = collect_prompt_module_results(modules, context())

    assert prompt_module_trace(results) == [
        {"name": "silent", "level": "game_interface", "text": ""},
        {"name": "visible", "level": "game_interface", "text": "visible text"},
    ]
    rendered = render_prompt_modules(results)
    assert "visible text" in rendered
    assert "silent" not in rendered
