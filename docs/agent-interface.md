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
reason, metadata, and transcript path. Prompt matches are guardrail metadata by
default; they are not required for the harness to proceed. Requested BBS nodes
are recorded in observation metadata until real Synchronet node
discovery/allocation lands.

For a command-line smoke test:

```bash
python -m bbs_gym.cli observe-turn --timeout 5 --stable-ms 300
```

## Activity Runner

The next layer is `ActivityRunner`, which wraps repeated observe/decide/act
steps in a bounded phase. It consumes any object implementing the
`terminal_agent.agent.TerminalAgent` protocol: `agent_id`, `observe_turn()`,
and `act_action()`:

```python
from bbs_gym.env import BbsGym
from terminal_agent.models import OpenAICompatibleAdapter
from terminal_agent.runner import ActivityBudget, ActivityProfile, ActivityRunner

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

For local reasoning models that emit tags such as `<think>...</think>`, JSONL
activity traces keep the raw model response and also record the filtered
response used for action parsing.

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

For a non-BBS PTY smoke test, `python -m examples.shell_agent` starts
deterministic `bash --norc --noprofile` with a fixed prompt and drives it
through the same core terminal-agent classes.

## Account Registry

Model identity, BBS account identity, and provider settings live in an agent
registry. Use `config/agents.example.json` as the template and keep real
passwords in environment variables or `config/agents.local.json`, which is
ignored by git.

```bash
python -m bbs_gym.cli accounts list
python -m bbs_gym.cli accounts check
python -m bbs_gym.cli accounts provision
```

The registry key is `agent_id`; the BBS login name is `bbs_alias`. Campaign
memory and logs key off `agent_id`, while observations record `bbs_alias`,
transport, and model config in metadata. For automated runs, prefer rlogin:

```bash
python -m bbs_gym.cli run-activity \
  --transport rlogin \
  --agent-id qwen-local-001
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

1. Prove and tune the TW2 entry activity against a live Synchronet container
   through rlogin with a real model.
2. Add real Synchronet node discovery/allocation.
3. Add per-game reset hooks.
4. Run two or more agents through alternating campaign turns.
5. Add score/task-completion extraction for game and social workflows.

## Operational Constraints

- Treat the BBS as stateful. Reset by restoring or deleting `runtime/sbbs`.
- Give each agent a unique account; many door games key state by user alias.
- Preserve raw transcripts for debugging. Rendered plain text loses control
  codes, cursor movement, and some ANSI art context.
- Keep generic terminal-agent code in `terminal_agent`; keep BBS policy,
  Synchronet profiles, and campaign orchestration in `bbs_gym`.
