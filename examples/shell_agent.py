"""Drive a deterministic local bash session through the tty-agent core."""

from __future__ import annotations

import os
from pathlib import Path

from tty_agent.actions import Action
from tty_agent.agent import TerminalSessionAgent
from tty_agent.profiles import SHELL_PROFILE
from tty_agent.terminal import TerminalScreen, TurnObserver
from tty_agent.transports.pty import PtySession


def main() -> None:
    session = PtySession(
        ["/bin/bash", "--norc", "--noprofile"],
        columns=100,
        lines=30,
        transcript_path=Path("runtime/transcripts/shell-agent.raw"),
        # Fixed startup files and PS1 keep prompt matching reproducible across machines.
        env={"PS1": r"\$ ", "TERM": "xterm-256color", "PATH": os.environ["PATH"]},
    )

    with session:
        observer = TurnObserver(
            "shell-agent",
            session,
            terminal=TerminalScreen(columns=session.columns, lines=session.lines),
            profile=SHELL_PROFILE,
            metadata={"transport": "pty", "program": "bash", "encoding": session.encoding},
        )
        agent = TerminalSessionAgent("shell-agent", session, observer, observer.metadata)

        first = agent.observe_turn(timeout=2.0, stable_ms=100, prompt_fast_path=True)
        print(first.model_text[-1000:])

        agent.act_action(Action("submit_line", text="printf 'hello from shell\\n'"))
        second = agent.observe_turn(timeout=2.0, stable_ms=100, prompt_fast_path=True)
        print(second.model_text[-1000:])

        agent.act_action(Action("submit_line", text="exit"))


if __name__ == "__main__":
    main()
