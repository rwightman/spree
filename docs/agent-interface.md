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

## Activity Runner

The next layer is `ActivityRunner`, which wraps repeated observe/decide/act
steps in a bounded phase:

```python
from bbs_gym.env import BbsGym
from bbs_gym.models import OpenAICompatibleAdapter
from bbs_gym.runner import ActivityBudget, ActivityProfile, ActivityRunner

model = OpenAICompatibleAdapter(
    base_url="http://localhost:11434/v1",
    model="gemma3",
)

with BbsGym() as gym:
    agent = gym.connect("agent-001", node=1)
    result = ActivityRunner(
        ActivityProfile(
            name="bbs-main-menu",
            objective="Explore the BBS main menu and recover from mistakes.",
        ),
        log_path="runtime/logs/bbs-main-menu.jsonl",
    ).run(agent, model, ActivityBudget(max_decision_ticks=20))
```

The runner treats malformed JSON or disallowed actions as decision ticks, not
system crashes. The next observation lets the model recover when it typed a bad
BBS command, chose the wrong menu, or received an error from a door game.
Malformed JSON gets one repair attempt with the validation error before the
tick is counted as failed. The validation record includes a truncated copy of
the malformed model response for debugging local-model failures.

The same basic flow is available from the CLI for an OpenAI-compatible local
server such as Ollama, vLLM, or llama.cpp:

```bash
python -m bbs_gym.cli run-activity \
  --model gemma3 \
  --base-url http://localhost:11434/v1 \
  --max-decision-ticks 20
```

For Claude through Anthropic's API:

```bash
python -m bbs_gym.cli run-activity \
  --provider anthropic \
  --model "$ANTHROPIC_MODEL" \
  --api-key "$ANTHROPIC_API_KEY" \
  --max-decision-ticks 20
```

Anthropic prompt caching is enabled for the stable system prompt/action schema
by default. Use `--no-anthropic-cache` to disable it for compatibility testing.

For a dependency-free live smoke test of the runner itself:

```bash
python -m bbs_gym.cli run-activity \
  --provider scripted \
  --scripted-response '{"action":"wait"}' \
  --scripted-response '{"action":"hangup"}' \
  --scripted-response '{"durable_facts":["Scripted smoke reached the BBS."]}'
```

To aim a model at the TW2 entry task:

```bash
python -m bbs_gym.cli run-activity \
  --activity tw2-entry \
  --model gemma3 \
  --base-url http://localhost:11434/v1 \
  --max-decision-ticks 50
```

Session compaction now expects structured JSON rather than prose:

```json
{
  "current_state": "At the external programs menu.",
  "last_error": "Sending P produced Invalid command.",
  "open_subgoals": ["Enter TW2"],
  "discovered_facts": ["X opens external programs"],
  "failed_actions": ["P at the BBS main menu"],
  "strategy_notes": ["Use ? when menu commands are unclear"]
}
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
