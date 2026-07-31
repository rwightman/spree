# NEXT: Current State And Near-Term Work

This document tracks the current implementation state and the next practical
work for Spree. The project has moved past a single terminal-observation
harness: `tty-agent` now provides the reusable terminal-agent core, and
`bbs-gym` layers BBS, door-game, activity, routing, and match orchestration on
top.

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
- Generic `tty_agent` core for actions, observations, runners, memory,
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
  - `byte_quiet_ms`,
  - `matched_prompt`,
  - `ready_reason`,
  - transcript path and transcript byte offsets.
- Generic observation hints for recent terminal output, likely active prompt,
  input mode, echoed-input/no-effect detection, and unchanged screens.
- Prompt modules with assistance levels:
  - `generic_terminal`,
  - `bbs_conventions`,
  - `game_interface`,
  - `strategic`.
- Prompt-module provenance in JSONL step traces with rendered `{name, level,
  text}` records.
- Prompt profiles for Synchronet BBS, TW2, broad BBS door-safe input, and
  line-oriented BBS doors.
- BBS-specific `bbs_gym` shell for Synchronet CP437 defaults, profiles,
  activities, Docker config, and CLI commands.
- Structured actions and validation, including CP437 checks and one JSON repair
  retry with malformed-response logging.
- Model adapter interfaces.
- OpenAI-compatible and Anthropic HTTP adapters.
- Codex CLI and Claude CLI adapters, including stateful resume/session-id
  support.
- Raw/filtered response tracking so reasoning tags can be logged while action
  parsing sees cleaned JSON.
- Model-family response filters, including Gemma 4 thought-channel filtering.
- Anthropic prompt caching for stable system prompts.
- Single-agent activity runner.
- Routed activity runner with profile switches on fresh observations.
- TW2 entry, TW2 in-game, `bbs-door-safe`, and `bbs-door-line` activity
  profiles.
- Run-level objectives and profile-specific objectives.
- Prompt modes: `stateless_full` and `stateful_delta`.
- Prompt layouts: `timeline_first` and `cache_friendly`.
- JSON-backed memory store with dedupe/caps.
- JSONL step logging.
- Trace pretty-printer for JSONL activity logs.
- ANSI HTML and animated GIF exporter from raw transcripts.
- Agent account registry with Synchronet provisioning through `jsexec`.
- Rlogin activity runs using pre-provisioned account identity.
- JS TW2 reset and one-player turn-grant scripts for development runs.
- Telnet activity and match runs against Ether/Tele-Arena.
- `run-match` with sequential, parallel barrier, parallel race, and continuous
  scheduler modes.
- Match config files for multi-agent/melee experiments.
- Live TW2 smoke/play runs against local OpenAI-compatible vLLM servers and
  Codex CLI.
- Live Tele-Arena runs against Codex CLI and Claude CLI, including stateful
  match experiments.

Missing:

- Real Synchronet node discovery/allocation.
- Long-horizon campaign scheduling that composes activities, matches, social
  phases, maintenance, and scoring.
- Scoring/extraction for TW2, Tele-Arena, DOS doors, and social workflows.
- Snapshot/reset orchestration beyond individual door reset scripts.
- Optional PNG/image observations for multimodal models.
- Stronger memory consolidation for long matches and repeated coordination
  failures. The proposed design — pluggable memory policies, an
  operations-based reconciliation contract with typed exits instead of lossy
  summary rewrites, an intent channel, and provider-chain rollover — is in
  `docs/memory-design.md`.

The `tty_agent` package is intentionally separate from `bbs_gym`.
Terminal observation, action validation, model adapters, memory, and the
activity runner are generic enough to drive shells, TUI programs, SSH sessions,
and terminal games. BBS-specific account policy, door profiles, campaign
scheduling, reset, and scoring stay in `bbs_gym`.

## Model Adapter Strategy

The harness owns memory, compaction, and persistence. Provider-side session
state is an optimization, not the source of truth.

The default mode is stateless:

```text
full harness prompt in -> response out
```

For providers that preserve context across calls, `stateful_delta` can reduce
prompt size:

```text
bootstrap prompt in -> provider session id
delta prompt + resume session id -> response out
```

Traces and JSON memory remain authoritative in both modes.

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
- Codex CLI through `codex exec`.
- Claude CLI through `claude -p`.

The local and OpenAI API paths share the OpenAI-compatible adapter surface.
Anthropic HTTP uses a separate adapter with the same internal contract. Codex
and Claude CLI adapters use subprocess calls, capture the final message, and
feed that text through the same parser, compactor, and memory commit paths as
HTTP providers.

Every provider failure surfaces as `ModelError` (or `ModelTimeoutError`), whether
it comes from a subprocess or from HTTP: non-2xx responses, unreachable hosts,
read timeouts, non-JSON bodies, and unexpected response shapes. That keeps HTTP
adapters on the same `model_error_retries`, malformed-response logging, and
graceful-stop paths as the CLI adapters instead of aborting a run.

Use the smallest common model API surface first:

```text
chat messages -> text response
```

Avoid depending on provider-specific tool calling, Responses API, JSON schema
enforcement, or multimodal input until each backend has been tested explicitly.

Prompt modes:

```text
stateless_full   Send full harness context on every decision tick.
stateful_delta   Send one full bootstrap prompt, then smaller delta prompts for resumed provider sessions.
```

`stateful_delta` is currently useful for Codex CLI and Claude CLI stateful
runs. It should not be used with stateless HTTP endpoints unless the provider
explicitly offers a comparable resumable session.

Codex stateful mode:

```text
first tick     codex exec --json ...
later ticks    codex exec resume <session_id> ...
```

Claude stateful mode:

```text
first tick     claude -p --output-format json ...
later ticks    claude -p --resume <session_id> --output-format json ...
```

The adapters capture provider session ids and can persist them to session files.
`--codex-stateful` and `--claude-stateful` automatically select
`stateful_delta` prompts unless the run explicitly sets a different prompt mode.

Model families can need different response filters. The harness should keep raw
responses in traces but parse actions from filtered text. Current families:

```text
default  Strip common XML-ish reasoning tags such as <think>...</think>.
gemma4   Strip Gemma 4 thought-channel markers such as <|channel>thought.
none     Disable filtering for debugging.
```

Selection should default from the model id and be overridable per run or agent
registry entry.

## Action Model

The model should always return a structured JSON action. The JSON wrapper is for
validation, logging, replay, and policy enforcement. It should not prevent
free-form BBS input where free-form input is appropriate.

Example short command:

```json
{"action": "press_key", "arguments": {"key": "P"}}
```

Example open-ended chat/message input:

```json
{
  "action": "submit_line",
  "arguments": {
    "text": "I think your trading plan is too defensive. Try moving cargo earlier."
  }
}
```

Example key press:

```json
{"action": "press_key", "arguments": {"key": "enter"}}
```

Example printable hotkey:

```json
{"action": "press_key", "arguments": {"key": "q"}}
```

Example partial input without submitting:

```json
{"action": "type_text", "arguments": {"text": "partial input"}}
```

Example multi-line post:

```json
{
  "action": "submit_lines",
  "arguments": {
    "lines": [
      "Subject: Trade route notes",
      "",
      "I found a decent early route near sector 42.",
      "Watch for fighters in adjacent sectors."
    ]
  }
}
```

Initial action set:

```text
press_key       Press exactly one key, such as q, D, ?, 1, enter, escape, tab, or an arrow.
submit_line     Type text, then press transport-specific Enter/Return.
type_text       Type text without pressing Enter/Return.
submit_lines    Send multiple submitted lines, Enter/Return after each.
wait            Do nothing and observe again.
hangup          Close the session.
```

Gated escape hatches and later additions:

```text
send_raw        Send exact control text, no automatic Enter/Return.
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

This is what `observe_turn()` implements. It should not know about game rules,
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
one continuous match window
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

## Match Scheduling

`run-match` is the implemented multi-agent scheduler. It composes several
activity states against one shared BBS or door server. Each participant keeps
its own terminal session, model adapter, provider session, recent-step context,
campaign memory namespace, and per-agent JSONL trace.

Scheduler modes:

```text
sequential        One agent decides and commits at a time.
parallel_barrier  Active agents decide concurrently; commits happen in scheduled order.
parallel_race     Active agents decide concurrently; commits happen as decisions finish.
continuous        Keep one decision in flight per active agent and requeue after each commit.
```

Order policies:

```text
fixed    Use participant order from CLI/config.
shuffle  Seeded per-round/per-start shuffle to reduce first-mover bias.
rotate   Rotate first position without randomness.
```

`parallel_race` and `continuous` intentionally make model latency part of the
competition. Use `parallel_barrier` when fairness matters more than speed.

Budget semantics:

```text
max_wall_seconds     Match-level wall clock shared by every participant.
max_decision_ticks   Per-participant decision cap.
max_rounds           Round cap for round-based modes.
max_rounds           Queued-action cap in continuous mode.
```

Continuous mode has no all-agent rounds. Its scheduler events use `tick`, and
`commit_order` records both the committed `tick` and the original
`queued_tick`. It does not emit `round_started` or `round_completed`.

The model should see the relevant budget in every prompt. Match logs should
record scheduler config, participant specs, disconnect/reconnect events,
decision completion, commit order, stop reasons, and final match completion.

Participant failures are isolated in every scheduler mode. An unexpected error
while one agent decides or commits retires only that agent with
`stop_reason="scheduler_error"` and an `agent_step_failed` event; the remaining
participants keep playing and still reach `finish_state`, so their results and
campaign-memory commits survive. A peer that drops between observing and acting
stops that agent with `stop_reason="disconnected"`, which the reconnect policy can
still act on.

Wall budgets are soft admission budgets, not hard deadlines. `max_wall_seconds`
is checked before starting a round, decision tick, reconnect, or newly queued
task; once a step is admitted, its observation, model call, and terminal action
run to completion and commit, and memory finalization happens after expiry. The
worst-case overrun is roughly one in-flight step per participant, bounded by the
model and transport timeouts, and `match_completed` records the actual value as
`wall_overrun_seconds`. Experiments that need strict termination should wrap the
run in an outer supervisor with its own hard timeout.

## Future Campaign Scheduling

A campaign runner should sit above `run-activity`, `run-routed`, and
`run-match`. It should compose social phases, door-game sessions, maintenance,
score extraction, and resets into longer experiments.

Example campaign turn:

```text
campaign round N
  social phase
    agents can read/post/reply for bounded decision ticks

  game phase
    one single-agent activity, routed activity, or match

  social phase
    agents can react to game results or messages

  maintenance/checkpoint
    save logs
    extract scores
    optionally snapshot or reset
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

TW2/TW2002-style games can be tested with concurrent agents. BRE and other
single-node or lock-sensitive DOS doors should default to serialized sessions
until locking behavior is verified.

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
---
Generic terminal modules:
  Most recent terminal output
  Likely active prompt
  Input mode hint
  Previous action effect when notable
BBS convention modules when enabled
Game-interface modules when enabled
Full current screen
---
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

Most recent terminal output:
Docking complete.

Likely active prompt:
Command (?=Help)?

Input mode hint:
hotkey_expected - one-character commands are usually single keypresses

Full current screen:
...

Return one JSON action.
```

The current prompt should be extracted from the live screen and labeled
separately from scrollback. The model still receives scrollback, because old
text is useful context, but the active prompt/current input line should be
harder to confuse with stale prompt text.

Prompt modules are selected per activity profile. The default baseline is the
generic terminal module set; BBS and TW2 profiles add domain modules
explicitly. This keeps benchmark assistance levels clear: a generic terminal
run can omit BBS and TW2 modules, while a game-interface-assisted TW2 run can
include TW2 command vocabulary and input-mode rules.
Reserve `ActivityProfile.system_guidance` for rare, stable system-message
prose. Prefer modules for tactical and domain guidance because traces record
exactly which modules rendered and what text they contributed.

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

### Observed TW2 Failure: Stale Prompts And Auto-Accepted Input

Live TW2 play exposed two separate terminal-facing failure modes.

The first was stale prompt text. The rendered screen tail could still contain an
old `Your offer?` line even though the live input prompt had already returned to
`Command (?=Help)?`. Models then sent numeric trade offers at the command
prompt. This was not action queuing; the runner only permits one action per
model call:

```text
observe stable screen -> model returns one action -> send one action -> observe again
```

The fix path is mostly implemented:

- Extract likely active prompt/current line separately from the full screen.
- Render recent terminal output, likely active prompt, input mode, and previous
  action effects as prompt modules.
- Keep full screen/scrollback for context, but label it as reference material.
- Log prompt-module text in traces for replay and ablation.

The second failure was TW2's auto-accepting `InputFunc`. Some prompts accept a
numeric or command value as soon as it is an exact match that cannot be extended.
If the harness sends `submit_line "4"`, TW2 may consume only `4`, leave the
terminating Enter in the input buffer, and let the next prompt consume that
leftover Enter as an empty response. In trade prompts this silently cancels the
haggle and returns to `Command (?=Help)?`.

Current mitigation:

- `tw2-game` can use specific TW2 guidance and action policy.
- `bbs-door-safe` removes `submit_line` so capable models use `press_key` for
  hotkeys and `type_text` for values, then observe before deciding whether
  Enter is needed.
- `bbs-door-line` keeps `submit_line` for line-oriented doors such as
  Ether/Tele-Arena where normal commands are submitted with Enter.

Remaining improvements:

- Extract visible, non-privileged state facts from screen text only, such as
  sector, turns left, credits, cargo, current port, and whether the current port
  buys or sells the carried cargo.
- Record repeated no-effect or cancelled-input patterns in session memory, for
  example "offer prompt returned to Command without changing credits/cargo."
- Consider a generic typed-buffer note when a previous `type_text` appears to
  be sitting at the prompt waiting for Enter.

Do not solve this by reading TW2 database state inside the agent loop or by
blocking arbitrary TW2 commands. The benchmark should still allow mistakes,
invalid entries, and recovery. The goal is to make the actual current terminal
state clearer to the model.

## Memory Layers

Use layered memory. Do not commit every inner tick directly to long-term memory.
The next big memory task is staleness management: durable facts need evidence,
confidence, scope, and recency so models do not keep acting on obsolete beliefs
from earlier sessions. Long-running agents should distinguish stable facts
("`?` opens help"), tentative state ("last seen in sector 58"), failed
assumptions, and superseded facts, then age or retire stale entries during
compaction and campaign-memory commits.

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

Every decision tick emits a structured per-agent JSONL step record. Match runs
also emit a separate match JSONL stream for scheduler events.

Per-agent step records include:

```json
{
  "active_profile": "tw2-game",
  "step": 42,
  "observation": {
    "model_text": "...",
    "pretty_screen": "...",
    "cursor": [12, 4],
    "stable_ms": 350,
    "byte_quiet_ms": 0,
    "matched_prompt": "tw2-command",
    "ready_reason": "stable",
    "transcript_path": "runtime/transcripts/agent-001.raw",
    "transcript_byte_start": 12000,
    "transcript_byte_end": 12750
  },
  "prompt": {
    "mode": "stateless_full",
    "layout": "timeline_first",
    "system": "...",
    "user": "..."
  },
  "prompt_modules": [
    {"name": "terminal.recent_output", "level": "generic_terminal", "text": "..."}
  ],
  "action": {
    "action": "submit_line",
    "arguments": {"text": "?"}
  },
  "execution": {
    "sent_bytes": {"len": 2, "repr": "?\\r", "hex": "3f0d"}
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

Match records include `match_started`, `agent_step_started`,
`agent_decision_completed`, `commit_order`, `agent_step`,
`participant_disconnected`, `participant_reconnected`, and `match_completed`.
Round-based modes use `round`; continuous mode uses `tick`.

Large prompt/response bodies are currently stored inline because that makes
traces self-contained. If traces become too large for long campaigns, add an
optional external-body mode rather than changing the default schema.

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

## Completed Vertical Slice

The original vertical slice is complete:

1. `Action` dataclass/schema and JSON parser.
2. `ModelAdapter` interface.
3. OpenAI-compatible text adapter.
4. Anthropic text adapter.
5. Codex CLI and Claude CLI adapters.
6. `ActivityBudget`.
7. Single-agent `ActivityRunner`.
8. Routed activity runner.
9. JSONL step logs.
10. Basic memory store with JSON files.
11. Intra-session compaction call.
12. End-of-session memory commit call.
13. BBS main-menu, TW2 entry, TW2 game, BBS door-safe, and BBS door-line
    profiles.
14. Account/rlogin provisioning.
15. Active prompt/current-line extraction and action-effect notes.
16. Trace pretty-printing and ANSI/GIF replay.
17. Live TW2 runs with local OpenAI-compatible models and Codex.
18. Live Zork runs through local PTY/Frotz.
19. Live Tele-Arena runs through Ether.
20. Multi-agent `run-match` with sequential, parallel barrier, parallel race,
    and continuous scheduling.

This proves the core loop:

```text
observe -> decide -> act -> recover -> compact -> commit memory
```

and the multi-agent loop:

```text
multiple terminal sessions -> scheduled model decisions -> committed actions -> per-agent traces
```

## Next Implementation Milestones

Prioritize work that improves reproducibility, scoring, and long-run behavior:

1. Add real Synchronet node discovery/allocation for rlogin/telnet sessions.
2. Add snapshot/reset tooling for `runtime/sbbs` and standalone door servers.
3. Add score and task-completion extractors for TW2, Tele-Arena, TW2002/BRE, and
   BBS social workflows.
4. Improve memory consolidation for long activities and matches, especially
   repeated coordination failures, learned command procedures, and strategic
   state.
5. Add a campaign runner above `run-activity`, `run-routed`, and `run-match` to
   compose social phases, game phases, maintenance, scoring, and resets.
6. Add DOS-door setup verification for TW2002 and BRE.
7. Add optional PNG observation rendering from the pyte screen buffer for
   multimodal models.

## Open Questions

- Should compaction use each agent's own model by default, or a neutral
  summarizer for early development?
- Should social policy be enforced by validator only, by a judge model, or both?
- How much prompt history should local small models receive before compaction?
- Should screenshots be included in the first model loop, or added after the
  text-only runner is stable?
