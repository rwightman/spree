# Agent Interface

## Session Model

Each agent gets one terminal connection and one transcript file:

```bash
python -m bbs_gym.cli smoke \
  --host 127.0.0.1 \
  --port 2323 \
  --transcript runtime/transcripts/agent-001.raw
```

The raw transcript keeps ANSI color and CP437 bytes intact. Use `strip_ansi()`
only for text observations sent to an LLM.

For multiple agents:

```python
from bbs_gym.env import BbsGym

with BbsGym() as gym:
    red = gym.connect("red", node=1)
    blue = gym.connect("blue", node=2)
    observation = red.observe_turn(timeout=10.0, stable_ms=300)
    print(observation.model_text)
    red.act("red_user")
```

`observe_turn()` is the preferred model-facing read path. It updates a virtual
terminal screen, waits for screen stability, and returns structured state such
as `model_text`, `pretty_screen`, cursor position, matched prompt, readiness
reason, node, and transcript path. Prompt matches are guardrail metadata by
default; they are not required for the harness to proceed.

For a command-line smoke test:

```bash
python -m bbs_gym.cli observe-turn --timeout 5 --stable-ms 300
```

## Next Automation Milestones

1. Create deterministic user accounts for agents.
2. Script login and menu navigation to the door menu.
3. Add a `step(action) -> observation` wrapper around `TelnetSession`.
4. Add per-game reset hooks.
5. Run two or more agents concurrently against separate BBS accounts.

## Operational Constraints

- Treat the BBS as stateful. Reset by restoring or deleting `runtime/sbbs`.
- Give each agent a unique account; many door games key state by user alias.
- Preserve raw transcripts for debugging. Rendered plain text loses control
  codes, cursor movement, and some ANSI art context.
