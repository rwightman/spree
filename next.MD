# Next Design: Model Runner, Turns, Actions, And Memory

This document captures the current intended design for moving from a terminal
observation harness to real agent play: a local model, such as a Gemma-family
model served through an OpenAI-compatible endpoint, playing against a remote
Claude/GPT-style API model on a live BBS and door-game environment.

The central rule remains:

```text
Agents interact through terminal I/O.
They do not call BBS or game APIs directly during normal play.
```

The harness may inspect logs, scores, and filesystem snapshots for evaluation
or reset outside the agent's decision loop, but agents should experience the BBS
as users do.

## Current Starting Point

Already present:

- Dockerized Synchronet BBS.
- Generic `terminal_agent` core for actions, observations, runners, memory,
  model adapters, and transports.
- Telnet, rlogin, and local PTY session handling.
- `pyte`-backed terminal rendering with configurable encoding.
- `observe_turn()` with quiescence-first readiness.
- Raw transcripts.
- Structured observations with:
  - `model_text`,
  - `pretty_screen`,
  - cursor position,
  - `stable_ms`,
  - `matched_prompt`,
  - `ready_reason`,
  - transcript path.
- Basic prompt profiles for Synchronet BBS and TW2.
- BBS-specific `bbs_gym` shell for Synchronet CP437 defaults, profiles,
  activities, Docker config, and CLI commands.
- Structured actions and validation, including CP437 checks and one JSON repair
  retry with malformed-response logging.
- Model adapter interfaces.
- OpenAI-compatible and Anthropic HTTP adapters.
- Raw/filtered response tracking so reasoning tags can be logged while action
  parsing sees cleaned JSON.
- Anthropic prompt caching for stable system prompts.
- Single-agent activity runner.
- TW2 entry activity profile.
- TW2 in-game activity profile.
- JSON-backed memory store with dedupe/caps.
- JSONL step logging.
- Agent account registry with Synchronet provisioning through `jsexec`.
- Rlogin activity runs using pre-provisioned account identity.

Missing:

- Real Synchronet node discovery/allocation.
- Campaign scheduling.
- Scoring/extraction.
- Actual model-vs-model run against a configured local/API model pair.

The `terminal_agent` package is intentionally separate from `bbs_gym`.
Terminal observation, action validation, model adapters, memory, and the
activity runner are generic enough to drive shells, TUI programs, SSH sessions,
and terminal games. BBS-specific account policy, door profiles, campaign
scheduling, reset, and scoring stay in `bbs_gym`.

## Model Adapter Strategy

Treat all model calls as stateless:

```text
prompt in -> response out
```

Do not rely on provider memory features. The harness owns all memory,
compaction, and persistence.

Use a common internal model interface:

```python
class ModelAdapter:
    def decide(self, prompt: DecisionPrompt) -> Action:
        ...

    def compact(self, prompt: CompactionPrompt) -> SessionSummary:
        ...

    def commit_memory(self, prompt: MemoryCommitPrompt) -> MemoryPatch:
        ...
```

Adapters can implement this for:

- OpenAI-compatible local endpoints:
  - Ollama,
  - vLLM,
  - llama.cpp `llama-server`.
- OpenAI GPT API.
- Anthropic Claude API.

The local and OpenAI API paths can share most of an OpenAI-compatible adapter.
Claude should get a separate Anthropic adapter with the same internal contract.

Use the smallest common model API surface first:

```text
chat messages -> text response
```

Avoid depending on provider-specific tool calling, Responses API, JSON schema
enforcement, or multimodal input until each backend has been tested explicitly.

## Action Model

The model should always return a structured JSON action. The JSON wrapper is for
validation, logging, replay, and policy enforcement. It should not prevent
free-form BBS input where free-form input is appropriate.

Example short command:

```json
{"action": "send", "text": "P"}
```

Example open-ended chat/message input:

```json
{
  "action": "send",
  "text": "I think your trading plan is too defensive. Try moving cargo earlier."
}
```

Example multi-line post:

```json
{
  "action": "send_multiline",
  "lines": [
    "Subject: Trade route notes",
    "",
    "I found a decent early route near sector 42.",
    "Watch for fighters in adjacent sectors."
  ]
}
```

Initial action set:

```text
send            Send one text line, then Enter.
send_raw        Send exact bytes/text, no automatic Enter and no newline field.
send_multiline  Send multiple lines, Enter after each.
wait            Do nothing and observe again.
hangup          Close the session.
```

Later additions:

```text
key             Named key such as escape, tab, arrow_up.
macro           Harness-owned macro, only when allowed by the phase profile.
```

The action payload can be open-ended. The current activity policy decides how
open it should be:

```text
Door command prompt: short commands, numbers, limited text.
BBS menu: short commands.
Message editor: open-ended multi-line text.
Chat: open-ended text with rate/length policy.
Login/password: controlled credentials only.
Sysop/admin: blocked for ordinary agents.
```

## Validation Philosophy

Validation should constrain terminal mechanics and experiment safety, not
eliminate mistakes.

Allowed mistakes:

- invalid game commands,
- wrong quantities,
- choosing the wrong menu item,
- asking for help at the wrong prompt,
- posting a weak strategy,
- needing to retry after an error.

The next observation should show the mistake, and the model should get another
decision tick to recover.

Validation should reject or transform:

- malformed JSON,
- malformed JSON after one repair retry,
- action type not allowed in the current phase,
- input exceeding max length,
- too many lines in one action,
- text that cannot be encoded as CP437,
- binary/control sequences not on the allowlist,
- attempts to enter blocked sysop/admin areas through harness macros,
- runaway repeated no-op actions when the budget is exhausted.

## Three Clocks

Do not use one generic "tick" for everything. Use three layers.

### 1. I/O Tick

Low-level terminal progress:

```text
read bytes
update virtual terminal
wait for screen quiescence
produce Observation
```

This is what `observe_turn()` is becoming. It should not know about game rules,
daily turns, scoring, or model memory.

### 2. Decision Tick

One model action opportunity:

```text
Observation -> model prompt -> JSON Action -> validation -> terminal input
```

This is the core unit of model interaction. Invalid commands and recovery happen
across decision ticks.

### 3. Campaign Turn

High-level experiment schedule:

```text
one BBS day
one TW2 session
one BRE daily turn
one social/message-board window
one match round
```

Campaign turns enforce fairness and sane limits.

## Activity Sessions

A campaign turn contains one or more activity sessions.

Examples:

```text
pre-game social phase
game phase
post-game social phase
maintenance/checkpoint phase
```

An activity session has:

- an agent,
- a phase profile,
- a budget,
- an objective,
- a transport/session,
- a context builder,
- an action policy,
- exit conditions.

Minimal activity runner:

```python
def run_activity(agent, model, phase, memory, budget):
    session_summary = ""
    working_memory = {}
    recent_steps = []

    while budget.remaining():
        obs = agent.observe_turn()

        if should_compact(recent_steps, session_summary, obs):
            session_summary = model.compact(
                previous_summary=session_summary,
                old_steps=steps_to_compact(recent_steps),
                current_screen=obs.model_text,
            )
            recent_steps = keep_recent_steps(recent_steps)

        prompt = build_decision_prompt(
            phase=phase,
            observation=obs,
            campaign_memory=memory.load(agent.id),
            session_summary=session_summary,
            working_memory=working_memory,
            recent_steps=recent_steps,
            budget=budget,
        )

        action = model.decide(prompt)
        action = validate_action(action, phase.policy)
        agent.act_action(action)

        step = log_step(obs, prompt, action, budget)
        recent_steps.append(step)

        if phase.exit_condition(obs, action, budget):
            break

    patch = model.commit_memory(
        campaign_memory=memory.load(agent.id),
        session_summary=session_summary,
        recent_steps=recent_steps,
        final_observation=obs,
    )
    memory.save_patch(agent.id, patch)
```

## Campaign Scheduling

Start with a round-based scheduler.

Example campaign turn:

```text
campaign round N
  social phase
    agent A can read/post/reply for up to X decision ticks
    agent B can read/post/reply for up to X decision ticks

  game phase
    agent A gets one door session or up to Y decision ticks
    agent B gets one door session or up to Y decision ticks

  social phase
    agents can react to game results or messages

  maintenance/checkpoint
    save logs
    extract scores
    optionally snapshot or reset
```

For continuous games, define a "game turn" as a budget:

```text
max_decision_ticks = 100
max_wall_time = 10 minutes
max_inputs = 200
must_exit_to_bbs = true
```

For daily-turn games:

```text
max_door_entries_per_round = 1
must_stop_after_turn_complete = true
```

For social/message-board phases:

```text
max_social_decision_ticks = 30
max_posts = 3
max_chat_lines = 20
max_wall_time = 5 minutes
```

The model should see the budget in every prompt.

## Sequential Before Concurrent

Start with sequential execution:

```text
agent A social
agent B social
agent A game
agent B game
agent A post-game social
agent B post-game social
```

Only add true concurrency when a game/profile benefits from it and is verified
to support it.

Reasons:

- classic doors may have node/dropfile locking issues,
- sequential execution is much easier to debug,
- campaign fairness can be implemented without async complexity,
- many games are effectively turn/session based anyway.

TW2/TW2002 can later support concurrent sessions. BRE should default to
serialized sessions until locking behavior is verified.

## Node Model

An agent should be assigned:

```text
agent id
BBS user account
BBS node
transport
model adapter
memory namespace
```

The node must be explicit in logs because door-game bugs often reduce to:

- stale dropfiles,
- wrong node paths,
- single-node locks,
- per-node setup errors.

Even if Synchronet auto-assigns nodes initially, the harness should record what
node was used.

## Prompt Shape

Decision prompt should include:

```text
Agent identity
Current phase
Goal for this activity
Remaining budget
Current action policy
Durable campaign memory
Session summary so far
Working memory
Recent steps
Current screen
Required JSON action schema
```

Example:

```text
You are agent-001 on a BBS.

Current phase: TW2 game session
Remaining budget: 42 actions, 7 minutes
Goal: Improve your position without revealing strategy in public messages.

Action policy:
- Return exactly one JSON action.
- You may type terminal input.
- Invalid game commands are allowed, but recover from them.
- Do not use sysop/admin features.

Campaign memory:
- You previously learned that '?' shows help in TW2.

Session summary:
- You entered TW2.
- Sending 'P' at the current prompt produced "Invalid command."

Recent steps:
1. You typed "?".
2. The screen showed a command list.

Current screen:
...

Return one JSON action.
```

## Mistakes And Recovery

The harness should not treat model mistakes as runner failures.

If an agent enters an invalid command:

1. The BBS/game displays an error.
2. The next `observe_turn()` captures it.
3. The next decision prompt includes the recent action and result.
4. The model can retry.

Only the budget or phase policy should stop repeated mistakes.

If an agent returns malformed JSON, the harness should make one repair call
with the validator error and the action schema. If the repair succeeds, the
original decision tick is preserved. If the repair fails, the failed action
consumes one decision tick and is logged with validation notes.

Validation notes should include a truncated copy of the raw malformed model
response. Without that, local-model JSON failures are hard to diagnose after the
fact.

Useful anti-stall controls:

```text
max_repeated_actions
max_time_without_screen_change
max_empty_waits
max_parse_failures
max_validation_failures
```

## Memory Layers

Use layered memory. Do not commit every inner tick directly to long-term memory.

### 1. Raw Logs

Permanent audit trail:

- raw terminal bytes,
- rendered observations,
- prompts,
- model responses,
- parsed actions,
- validation results,
- timestamps.

Raw logs are not compacted or discarded during a run.

### 2. Immediate Context

Included in each decision prompt:

- current screen,
- last N steps,
- current phase,
- remaining budget,
- current objective,
- active constraints.

This supports short-horizon recovery from errors.

### 3. Working Memory

Lives for one activity session:

- current plan,
- current location/menu/game-state guess,
- facts discovered during this session,
- last error,
- pending subgoals.

This can be updated through compaction or explicit scratchpad extraction.

### 4. Session Summary

Temporary summary used when the current activity exceeds context budget.

It preserves older activity details after recent steps are removed from the
prompt.

Session summaries are structured JSON, not prose:

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

### 5. Campaign Memory

Durable memory across campaign turns:

- strategic notes,
- known commands,
- social facts,
- unresolved goals,
- successful routes/actions,
- repeated mistakes to avoid.

Campaign memory is committed only at activity/session boundaries.

Campaign memory is stored in harness-owned JSON, not provider memory. It is
portable across adapters, which is useful for debugging. For fair scoring,
model swaps should happen only at campaign-turn boundaries and must be recorded
in match metadata. Swapping Gemma to Claude halfway through an activity changes
the agent and should not be treated as the same run.

## Intra-Session Compaction

If context grows too large inside a campaign turn, compact inside the activity.

Trigger options:

```text
every N decision ticks
estimated prompt tokens > threshold
recent_steps length > threshold
large message-board transcript displayed
large help file displayed
```

Compaction call:

```python
session_summary = model.compact(
    previous_summary=session_summary,
    old_steps=steps_to_compact,
    current_screen=obs.model_text,
)
```

After compaction:

```text
keep session_summary
keep last N recent steps
discard older steps from prompt context
retain all raw logs on disk
```

Compaction should be conservative.

It should preserve:

- current location/state,
- successful actions,
- failed actions and exact error messages,
- visible quantities/assets,
- unresolved subgoals,
- command discoveries,
- message-board/social commitments.

It should avoid overclaiming:

```text
Observed: screen showed "Invalid command" after sending "P".
Likely: "P" is not valid at the current TW2 prompt.
```

The compactor should still return the fixed JSON shape above. If a model returns
prose, the adapter may salvage it into `current_state`, but that should be
treated as a degraded result.

## Memory Commit

At the end of an activity session, run an explicit memory commit call.

```python
memory_patch = model.commit_memory(
    campaign_memory=current_memory,
    session_summary=session_summary,
    recent_steps=recent_steps,
    final_observation=obs,
)
memory_store.save(agent_id, memory_patch)
```

Suggested memory patch:

```json
{
  "durable_facts": [
    {
      "text": "In TW2, '?' showed a command list.",
      "evidence": "turn-0003 step 14",
      "confidence": "observed"
    }
  ],
  "strategy_notes": [
    {
      "text": "Try trading before combat.",
      "evidence": "session summary",
      "confidence": "tentative"
    }
  ],
  "open_tasks": [
    "Find a profitable trade route.",
    "Check messages in the next social phase."
  ],
  "errors_to_avoid": [
    {
      "text": "Do not use P at the observed TW2 prompt unless the help screen confirms it.",
      "evidence": "Invalid command after P",
      "confidence": "likely"
    }
  ]
}
```

Store memory outside provider/model systems:

```text
runtime/memory/agent-001/campaign.json
runtime/memory/agent-001/sessions/turn-0003.json
runtime/memory/agent-001/summaries/turn-0003-game.json
```

JSON files are enough for the first version. SQLite can come later.

Campaign memory must not grow without bound. The first implementation should
dedupe exact list items and cap list fields. Longer campaigns should add a
between-turn consolidation pass that merges stale, duplicate, or contradictory
facts into a smaller memory state.

Anthropic prompt caching should be enabled for the stable system prompt/action
schema where supported. It is a backend optimization only; the harness still
sends complete stateless prompts each decision tick.

## Who Performs Memory Work

Two modes should be supported:

### Agent-Owned Memory

The same model that plays also compacts and commits its memory.

Pros:

- fairer for model-vs-model comparisons,
- each agent has its own interpretation of events,
- weaker models may have weaker memory, which is part of the evaluation.

Cons:

- lower-quality summaries,
- more hallucinated memory risk,
- harder debugging.

### Neutral Summarizer Memory

A separate harness model compacts and commits memory for all agents.

Pros:

- more reliable summaries,
- easier debugging,
- cheaper if using a small summarizer.

Cons:

- changes benchmark fairness,
- can give weaker agents better memory than they earned.

Default for benchmark fairness should be agent-owned memory. Neutral summarizer
mode is useful for development runs, but must be labeled clearly.

## Context Overflow Policy

Context overflow should not be treated as an emergency failure. It should be a
normal runner behavior.

Every activity runner should have:

```text
max_context_tokens
compaction_threshold = 0.75 * max_context_tokens
recent_steps_to_keep
max_summary_tokens
```

If the prompt approaches the threshold:

1. Compact older recent steps into session summary.
2. Keep the current screen.
3. Keep the last N steps verbatim.
4. Continue the activity.

If compaction fails repeatedly:

1. Save raw logs.
2. End the activity gracefully.
3. Commit a minimal failure memory.
4. Move to the next campaign phase.

## Event Logs

Every decision tick should emit a structured log.

Suggested JSONL event:

```json
{
  "match_id": "match-0001",
  "campaign_turn": 3,
  "phase": "tw2-game",
  "agent_id": "agent-001",
  "node": 1,
  "step": 42,
  "observation": {
    "model_text": "...",
    "pretty_screen": "...",
    "cursor": [12, 4],
    "stable_ms": 350,
    "matched_prompt": "tw2-command",
    "ready_reason": "stable",
    "transcript_path": "runtime/transcripts/agent-001.raw"
  },
  "prompt_path": "runtime/logs/match-0001/agent-001/step-0042.prompt.txt",
  "model_response_path": "runtime/logs/match-0001/agent-001/step-0042.response.txt",
  "parsed_action": {
    "action": "send",
    "text": "?"
  },
  "validation": {
    "accepted": true,
    "notes": []
  },
  "budget": {
    "decision_ticks_remaining": 41,
    "wall_seconds_remaining": 420
  },
  "timestamp": "..."
}
```

Large prompt/response bodies should be stored as separate files to keep JSONL
easy to scan.

## Social And Message-Board Phases

Social play is part of the environment. The harness should allow open-ended
text in social contexts, while still controlling rate, length, and policy.

Social phase budget examples:

```text
max_posts = 3
max_reply_lines = 40
max_chat_lines = 20
max_decision_ticks = 30
max_wall_time = 5 minutes
```

Policy should be explicit per run:

```text
friendly
competitive
flaming_allowed_with_limits
strict_safe_chat
```

Even local logs can contain abusive content, so social policy should be
configurable and recorded in match metadata.

## Scoring

Use separate scoring types.

### Skill Game Score

Extract native game score:

- BRE rankings,
- TW2/TW2002 assets,
- sectors,
- turns used,
- combat results,
- survival.

### Task Completion Score

For BBS workflows:

- posted a message,
- replied to a prompt,
- entered a door,
- found a specific message,
- sent mail.

### Social Score

For chat/message boards:

- coherence,
- role consistency,
- persuasion,
- helpfulness,
- policy compliance,
- judge preference.

Scoring can be post-hoc. It should not require direct game-state access during
agent decisions.

## Reset And Reproducibility

Filesystem snapshots are the baseline reset strategy.

```text
configure BBS
install doors
create accounts
stop container
snapshot runtime/sbbs
restore snapshot before match
```

This gives deterministic starting state, not deterministic trajectories.

Non-determinism remains from:

- model sampling,
- real-time delays,
- game RNG,
- inter-agent timing,
- maintenance events,
- concurrent sessions.

Record seeds and model parameters where possible.

## First Implementation Vertical Slice

Build in this order:

1. `Action` dataclass/schema and JSON parser. Done.
2. `ModelAdapter` interface. Done.
3. OpenAI-compatible text adapter. Done.
4. Anthropic text adapter. Done.
5. `ActivityBudget`. Done.
6. Single-agent `ActivityRunner`. Done.
7. JSONL step logs. Done.
8. Basic memory store with JSON files. Done.
9. Intra-session compaction call. Done.
10. End-of-session memory commit call. Done.
11. BBS main-menu activity profile. Done.
12. TW2 entry profile. Done.
13. Account/rlogin provisioning. Done.
14. Run and tune one real local/API model through TW2 entry.
15. Add real Synchronet node discovery/allocation.
16. Two-agent sequential campaign runner.

Do not start with a full async multi-agent scheduler. First prove:

```text
one agent can observe -> decide -> act -> recover from mistakes -> summarize
```

Then prove:

```text
two agents can take bounded sequential campaign turns
```

Then add concurrency for doors that need it.

## Open Questions

- Should compaction use each agent's own model by default, or a neutral
  summarizer for early development?
- Should social policy be enforced by validator only, by a judge model, or both?
- How much prompt history should local small models receive before compaction?
- Should screenshots be included in the first model loop, or added after the
  text-only runner is stable?
