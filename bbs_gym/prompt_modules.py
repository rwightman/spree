"""BBS and door-game prompt modules."""

from __future__ import annotations

from dataclasses import dataclass

from terminal_agent.hints import InputModalityProfile, InputModeRule
from terminal_agent.prompt_modules import (
    GENERIC_TERMINAL_MODULES,
    AssistanceLevel,
    PromptModule,
    PromptRenderContext,
    StaticPromptModule,
)

BBS_HOTKEY_CONVENTIONS = (
    "BBS convention: menu letters and numbers are often single-key hotkeys. Use press_key for those one-character "
    "choices so you do not send an extra Enter. Use submit_line when the BBS is asking for a typed line of text. "
    'In prompts like "Yes [No]" or "[Yes] No", brackets usually mark the currently selected/default choice. '
    "Bracketed shortcuts such as [(Q)uit] mark a single-key shortcut for that choice, while the same prompt may "
    "also accept typed values for other choices. "
    "If the prompt shows answer letters, use the obvious printable key such as Y or N; use arrow keys only when "
    "the UI appears to behave like a selector."
)

TW2_COMMAND_VOCABULARY = (
    "Trade Wars 2 command vocabulary is discoverable in-game with ?. Common command prompts use one-key commands "
    "such as P for port/dock, M for move, C for computer, I for information, and Q for quit/back out."
)

BBS_INPUT_MODALITY_PROFILE = InputModalityProfile(
    rules=(
        InputModeRule.from_pattern(
            mode="any_key_expected",
            pattern=r"(?:press|hit).{0,20}(?:any\s+)?key|press\s+enter",
            hint="press one key with press_key; press_key enter is a safe default when the screen says any key or Enter",
            priority=30,
            target="active_prompt",
        ),
        InputModeRule.from_pattern(
            mode="hotkey_expected",
            pattern=r"(?:\([Yy]/[Nn]\)|\[[Yy]es\]\s+No|Yes\s+\[[Nn]o\]|\[[Nn]o\]\s+Yes|No\s+\[[Yy]es\])",
            hint="choose with the obvious printable key when answer letters are visible; use Enter for the bracketed default",
            priority=20,
            target="active_prompt",
        ),
    )
)

TW2_INPUT_MODALITY_PROFILE = InputModalityProfile(
    rules=(
        InputModeRule.from_pattern(
            mode="line_input_expected",
            pattern=r"(?:how many|your offer|to which).{0,80}\?\s*$",
            hint="type the requested value and submit it with submit_line, or finish already-typed text with press_key enter",
            priority=50,
            target="active_prompt",
        ),
        InputModeRule.from_pattern(
            mode="hotkey_expected",
            pattern=r"command\s+\(\?=help\)\?\s*$",
            hint="one-character commands are usually single keypresses; use press_key unless you need to type a value",
            priority=40,
            target="active_prompt",
        ),
        *BBS_INPUT_MODALITY_PROFILE.rules,
    )
)


@dataclass(frozen=True)
class AuthenticatedSessionModule:
    name: str = "bbs.authenticated_session"
    level: AssistanceLevel = "bbs_conventions"

    def render(self, context: PromptRenderContext) -> str | None:
        if not context.observation.metadata.get("authenticated"):
            return None
        return (
            "This BBS session is already authenticated. The first screens may be welcome banners or bulletins, "
            "not a login prompt."
        )


@dataclass(frozen=True)
class TraceOnlyModule:
    name: str
    level: AssistanceLevel

    def render(self, context: PromptRenderContext) -> str:
        del context
        return ""


BBS_PROMPT_MODULES: tuple[PromptModule, ...] = (
    *GENERIC_TERMINAL_MODULES,
    AuthenticatedSessionModule(),
    StaticPromptModule(
        name="bbs.hotkey_conventions",
        level="bbs_conventions",
        text=BBS_HOTKEY_CONVENTIONS,
    ),
)

TW2_PROMPT_MODULES: tuple[PromptModule, ...] = (
    *BBS_PROMPT_MODULES,
    TraceOnlyModule(name="tw2.input_modes", level="game_interface"),
    StaticPromptModule(
        name="tw2.command_vocabulary",
        level="game_interface",
        text=TW2_COMMAND_VOCABULARY,
    ),
)
