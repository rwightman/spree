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
    red = gym.connect("red")
    blue = gym.connect("blue")
    print(red.observe(seconds=2.0))
    red.act("red_user")
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
