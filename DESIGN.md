# BBS Gym Design

## Goal

Build a local sandbox where LLM-driven agents interact with a real BBS as
ordinary terminal users. Agents should be able to chat on message boards, enter
doors, and play games such as TradeWars-style games and Barren Realms Elite
against each other.

The key design constraint is that agents do not call game APIs directly. The
BBS terminal session is the environment boundary.

```text
model/harness
  -> terminal action text/keystrokes
  -> telnet/rlogin
  -> Synchronet
  -> message boards, chat, doors
  -> terminal observation bytes/screen
  -> model/harness
```

## Current Stack

- BBS software: Synchronet.
- Runtime: Docker Compose.
- Local telnet endpoint: `127.0.0.1:2323`.
- Local rlogin endpoint: `127.0.0.1:2513`.
- Persistent BBS state: `runtime/sbbs`.
- Agent transcripts: `runtime/transcripts`.
- Initial game target: Synchronet's bundled JavaScript `tw2` door.
- Original DOS door target: BRE and TW2002 via the optional DOSEMU image.
- Optional local model server: vLLM, Ollama, or llama.cpp through an
  OpenAI-compatible `/v1/chat/completions` endpoint.

Synchronet was chosen because it is actively maintained, has Docker support,
ships with useful JavaScript doors, and supports classic BBS door dropfiles.

## Package Boundary

The project is split into a reusable terminal-agent core and a BBS-specific
shell.

`terminal_agent` owns behavior that applies to any interactive terminal target:

- structured terminal actions and validation,
- model adapters for OpenAI-compatible endpoints, Anthropic, and scripted tests,
- raw and parsed model-response tracking,
- JSON-backed memory, compaction, and memory commits,
- pyte-backed terminal rendering and quiescence observation,
- bounded activity runners,
- telnet, rlogin, and local PTY transports,
- generic prompt profiles such as stability-only and shell prompts.

`bbs_gym` owns Synchronet and BBS policy:

- Docker/Compose runtime defaults,
- CP437 defaults and BBS prompt profiles,
- BBS and TW2 activity profiles,
- agent account registry and Synchronet user provisioning,
- `BbsGym` connection wiring,
- future campaign/match scheduling, door resets, and score extraction.

The split is intentional. The terminal boundary, action schema, quiescence
observer, and model loop are useful for shells, SSH sessions, TUI applications,
and terminal games outside BBSs. The campaign layer is more BBS-shaped and
should stay in `bbs_gym` until another domain needs it.

## Agent Interface

Each agent receives a separate terminal connection and BBS account.

The minimal runtime interface is:

```python
with BbsGym() as gym:
    agent = gym.connect("agent-001", transport="rlogin")

    observation = agent.observe_turn()
    action = model(observation.model_text)
    agent.act(action)
```

Current implementation:

- `terminal_agent` contains the generic terminal-agent core: structured
  actions, model adapters, memory, runners, pyte-backed observations, and
  telnet/rlogin/local-PTY transports.
- `bbs_gym` contains the Synchronet/BBS shell: CP437 defaults, BBS/TW2 prompt
  profiles, activity profiles, account provisioning, Docker config, and CLI
  commands.
- Raw CP437/ANSI bytes are stored for replay/debugging.
- `observe_turn()` uses quiescence-first turn boundaries with prompt matches
  recorded as guardrail metadata.
- `BbsGym` maps `agent_id` to an optional registry record containing the BBS
  alias, password source, model config, and metadata.
- Rlogin is available for deterministic automated login after accounts are
  provisioned.
- Requested node values are recorded in observation metadata; true Synchronet
  node pinning/allocation remains a later BBS-layer feature.

## Observation Model

There should be three observation forms:

1. Raw transcript bytes.
2. Pretty rendered terminal screen state for humans/replay.
3. Model-oriented rendered text with ANSI/color noise stripped or normalized.

The raw transcript is authoritative and should always be preserved. The rendered
model text is what we pass to models most of the time. The pretty render is for
debugging, replay, and screenshots.

Current trace tooling:

- `scripts/trace_pretty.py` renders JSONL activity logs into a readable text
  transcript with optional model responses and control-character markers.
- `scripts/ansi_screencap.py` replays raw transcript bytes through the virtual
  terminal and emits ANSI HTML frames or animated GIFs for human debugging.
- GIF frame timing is configurable with `--duration-ms`; color depends on the
  raw terminal bytes actually sent by Synchronet.

Suggested structured observation:

```python
{
    "agent_id": "agent-001",
    "pretty_screen": "...80x24 rendered text...",
    "model_text": "...normalized screen text...",
    "new_text": "...recent appended text when useful...",
    "cursor": [row, col],
    "stable_ms": 350,
    "matched_prompt": "tw2-command",
    "ready_reason": "stable",
    "metadata": {"requested_node": 1, "transport": "telnet"},
    "timestamp": "...",
}
```

### Active Prompt Versus Scrollback

The model-oriented observation should distinguish the live input prompt from
older scrollback. Live TW2 testing showed a failure mode where the screen still
contained stale text such as `Your offer?`, but the current prompt had already
returned to `Command (?=Help)?`. The model then sent numeric trade offers at
the command prompt because old prompt text remained in the screen tail.

The harness should keep the input space open, but shape observations so the
current state is harder to misread:

- preserve normal scrollback for context,
- label the detected current prompt or active input line separately,
- put the current prompt near the end of the model prompt,
- include the last action and the visible result/delta before the next
  decision,
- optionally extract visible facts such as sector, turns left, cargo, and
  credits from rendered text only.

This is not a privileged game API and should not reject open-ended actions. It
is observation hygiene: the model should still be able to make mistakes, but it
should not have to infer the active prompt from stale terminal history.

## Optional Multimodal Observations

Multimodal agents can receive a PNG render alongside cleaned terminal text.
This should be optional because image observations are more expensive to
generate, store, and send to models.

Suggested multimodal observation:

```python
{
    "agent_id": "agent-001",
    "model_text": "...normalized screen text...",
    "screen_png": "runtime/observations/match-001/agent-001/step-0042.png",
    "cursor": [row, col],
    "stable_ms": 350,
    "matched_prompt": "tw2-command",
    "ready_reason": "stable",
    "timestamp": "...",
}
```

The PNG should be rendered from the same virtual terminal buffer used to create
`model_text`, not captured from a real terminal window. That keeps the harness
headless, deterministic, and able to run many agents in parallel.

The PNG is useful for information that cleaned text loses:

- ANSI layout, boxes, maps, and split panes,
- color emphasis and highlighted menu choices,
- cursor position,
- status bars and fixed-position regions,
- human replay/debugging.

The default observation should still include cleaned text. PNG-only observations
are harder to search, diff, prompt-match, and debug. A good default for
multimodal agents is cleaned text plus PNG, with raw bytes retained as the
authoritative transcript.

Implementation notes:

- Use `pyte` or an equivalent terminal buffer as the source of truth.
- Render PNGs with a fixed terminal font, cell size, and palette.
- Store PNGs under ignored runtime artifacts such as `runtime/observations`.
- Make image rendering configurable per run so text-only agents pay no cost.

## Timing And Turn Boundaries

Fixed reads such as `observe(seconds=2.0)` are only a low-level primitive. Door
games have variable output pacing: animated combat, file displays, explicit
pauses, and prompts that appear after delays.

The default benchmark harness should use quiescence as the primary signal before
asking the model for the next action:

- the virtual terminal screen has been stable for a configurable interval,
- then record whether a door/BBS prompt regex matched,
- then record whether a profile-specific input state was detected,
- otherwise return on a hard timeout.

The first robust implementation should use `pyte` to detect screen changes and
combine that with per-profile prompt regexes as metadata. For example, TW2 can
have a different prompt set than Synchronet's main menu or message editor.

Prompt matching is a guardrail, not a required turn boundary. We should avoid
making normal play depend on exact prompt strings because BBS text varies across
versions, themes, and local configuration. Profiles may opt into a prompt
fast-path for special cases, but the default should wait for screen stability.

Quiescence is not perfect by itself. Some games pause on screens that are not
asking for input, and some prompts update a clock/status line. Each door profile
can add "safe to act" hints, but those hints should annotate observations rather
than becoming the only way to make progress.

## Action Model

Actions are terminal input, not privileged game commands. The model returns one
JSON action per decision tick so the harness can validate mechanics, log
behavior, and replay runs without narrowing the BBS input space too much.

Examples:

```json
{"action": "send_line", "text": "P"}
```

```json
{"action": "key", "key": "enter"}
```

```json
{"action": "send_text", "text": "partial input"}
```

```json
{
  "action": "send_multiline",
  "lines": [
    "Subject: Trade route notes",
    "",
    "I found a decent early route near sector 42."
  ]
}
```

The action payload may contain open-ended text when the current activity allows
it, such as message-board posts or chat. Validation rejects malformed JSON,
encoding failures, overlong input, and disallowed action types. It should not
reject ordinary game mistakes such as wrong menu choices, invalid commands, or
bad quantities; those should flow through to the BBS and be recoverable on later
decision ticks.

`send_line` types text and submits it with the transport-specific Enter/Return
key; `send_line` with empty text is equivalent to pressing Enter/Return.
`send_text` types without submitting. `key` presses exactly one key without an
automatic Enter/Return; that can be a printable key such as `q`, `D`, `?`, or
`1`, or a named key such as `enter`, `escape`, `tab`, or an arrow. `send_raw`
is a gated escape hatch for exact control text and should not be available in
normal game/menu phases.

The action schema shown to the model must be rendered from the active
`ActionPolicy`, not hardcoded globally. A BBS menu profile should not advertise
multi-line posting actions, and a message editor should not force the same
limits as a one-key door menu.

## Model Responses And Reasoning Traces

Model adapters are stateless: each call receives the full prompt context the
harness wants the provider to see. The harness owns memory and compaction rather
than relying on provider-side memory.

For text-generation backends, the adapter keeps two versions of each response:

- the raw response, preserved in JSONL traces for debugging,
- the parsed response, after provider/model-specific output filters.

The default output filter removes closed or truncated reasoning blocks such as
`<think>...</think>`, `<thinking>...</thinking>`, `<reasoning>...</reasoning>`,
and `<analysis>...</analysis>` before action parsing. This lets Qwen-style local
models expose reasoning in traces without feeding hidden reasoning text back
into the terminal action parser.

Response filters are selected by model family when the model id is known.
Gemma 4 models can emit thought-channel markers such as `<|channel>thought`;
the Gemma 4 filter strips those thought channels and final-channel markers
before JSON action parsing. Runs can override the inferred family with
`--response-filter` or `response_filter` in the agent registry.

Provider-specific optimizations are allowed when they do not change the
internal contract. Anthropic prompt caching is enabled for the stable system
prompt/action schema. OpenAI-compatible local servers can receive extra request
body fields from the agent registry when needed.

## Runner, Clocks, And Memory

The model loop has three timing layers:

- I/O tick: read bytes, update the virtual terminal, wait for quiescence, and
  produce an observation.
- Decision tick: build a prompt, call the model, parse/validate one JSON action,
  send terminal input, and record a step.
- Campaign turn: a higher-level schedule such as one social phase, one door-game
  session, one BBS day, or one match round.

`ActivityRunner` currently implements bounded single-agent activity sessions on
top of the first two clocks. A future campaign runner should compose activities
into fair model-vs-model schedules instead of replacing `ActivityRunner`.

Memory is harness-owned:

- raw logs and transcripts are permanent audit artifacts,
- recent steps stay in the decision prompt for short-horizon recovery,
- structured session summaries compact older in-activity context,
- durable campaign memory is committed only at activity boundaries.

Session summaries use a small JSON shape with fields such as `current_state`,
`last_error`, `open_subgoals`, `discovered_facts`, `failed_actions`, and
`strategy_notes`. Campaign memory is also JSON and is portable across model
adapters. For fair scoring, model swaps should happen only at campaign-turn
boundaries and must be recorded in match metadata.

Malformed JSON receives one repair attempt with the validation error and action
schema. If repair fails, the failure consumes a decision tick and the raw bad
model response is preserved in the trace.

## Connection And Login

Telnet should remain available because it is the most human-like path through
the BBS. It is useful for realism tests and for debugging what an ordinary user
would see.

Automated benchmark runs should prefer rlogin. Synchronet rlogin can pass user
identity during connection setup, avoiding fragile new-user and login prompts.
That makes runs easier to reproduce across Synchronet versions and local config
changes.

Implemented transports:

- `telnet`: realistic user path, prompt-driven login required.
- `rlogin`: default automation path, pre-provisioned user identity, configurable
  terminal type for ANSI/color negotiation.
- `pty`: generic local subprocess path used to validate the terminal core
  outside BBSs.

The transport choice should be explicit in `BbsGym.connect(...)`.

## Accounts

Agents need deterministic BBS identities because door games usually key player
state by BBS user number or alias.

Implemented account flow:

1. Define agent identities in `config/agents.local.json` or another registry
   file.
2. Keep passwords in environment variables or ignored local config.
3. Run `python -m bbs_gym.cli accounts check` to validate the registry.
4. Run `python -m bbs_gym.cli accounts provision` to create/update Synchronet
   users through `jsexec`.
5. Run activities with `--transport rlogin --agent-id ...` so the BBS account
   identity is attached before the model loop starts.

Models should not need to complete first-time signup during normal benchmark
runs.

## Nodes

Synchronet exposes a fixed pool of nodes. Door games often use per-node
dropfiles and lock files, so node allocation is part of the environment state.
An agent is not just a model session; it is a model session assigned to a BBS
user and a BBS node.

`BbsGym` should make this explicit:

```python
agent = gym.connect("agent-001", node=1, transport="rlogin")
```

If `node` is omitted, the harness can allocate from a configured pool. The event
log should record the node because DOS door issues often reduce to node-specific
paths, locks, or stale dropfiles.

Current status: `BbsGym.connect(..., node=N)` records the requested node in
metadata, but the rlogin transport does not yet prove or force the actual
Synchronet node assigned by the server. The next BBS-layer work should discover
or pin the real node before relying on node-specific scoring/debugging.

Concurrency policy belongs to the game profile:

- TW2/TW2002-style games can be tested with concurrent agents.
- BRE should default to serialized sessions until locking behavior is verified.
- Single-node doors should reject multi-agent matches unless explicitly forced.

## Door Entry

Each game should have a small profile that knows how to reach it from a logged
in BBS session.

Example profile responsibilities:

- wait for login completion,
- navigate to the external programs menu,
- select the Games section,
- launch `TW2`, `BRE`, or `TW2002`,
- detect that the game has started,
- optionally run game-specific reset/setup commands outside normal play.

Once inside the game, the model receives terminal observations and sends
keystrokes.

## Multi-Agent Runner

For model-vs-model play:

1. Create one BBS account per model.
2. Open one terminal session per account.
3. Drive sessions concurrently.
4. Tick each session independently:
   - read bytes,
   - update virtual terminal,
   - build model observation,
   - request model action,
   - send action,
   - record event log.

The runner should record enough information to reproduce/debug a match:

```python
{
    "match_id": "...",
    "agent_id": "agent-001",
    "node": 1,
    "step": 42,
    "raw_bytes_path": "...",
    "pretty_screen": "...",
    "model_text": "...",
    "matched_prompt": "tw2-command",
    "ready_reason": "stable",
    "prompt": "...",
    "action": "...",
    "timestamp": "...",
}
```

## Game State And Reset

The BBS is stateful. Door games add more state.

Preferred reset strategy:

1. Build a clean baseline BBS state under `runtime/sbbs`.
2. Install/configure desired doors.
3. Stop the container.
4. Snapshot the baseline directory.
5. Restore that snapshot before each match.

Door-specific reset programs can be used when they are reliable, but filesystem
snapshots are the simplest reproducible baseline.

A snapshot restore gives a deterministic starting state. It does not guarantee
a bit-identical trajectory. Door games may advance RNG, timers, maintenance
events, or inter-player state differently depending on exact action timing.

## Scoring And Evaluation

Different BBS activities need different scoring signals.

Suggested taxonomy:

- Skill-game score: use native game scores where available, such as BRE rankings
  or TW-style assets, sectors, turns, and combat results.
- Task completion: score whether an agent completed a requested BBS workflow,
  such as posting a message, replying to mail, or joining a specific door.
- Social interaction: score conversation quality, policy adherence, role
  consistency, or human/LLM judge preferences over message-board/chat logs.

The event log should preserve enough context to support all three: terminal
screen observations, model actions, timestamps, BBS user identity, node, game
profile, and any extracted scoreboard snapshots.

## Door Game Notes

### Synchronet TW2

Use this first. It is already local, scriptable, and avoids DOS/DOSEMU issues
while the harness is still evolving.

Current development tooling:

- `make reset-js-tw2` reinitializes the JS TW2 universe.
- `make grant-js-tw2-turns PLAYER=RLoginSmoke TURNS=30` grants turns to one
  player without resetting the world.
- `run-activity --activity tw2-game --transport rlogin ...` has been exercised
  against a local OpenAI-compatible vLLM/Qwen server through a complete
  turn-exhaustion session.

### TradeWars 2002

Target path:

- supply original door files in `doors/tw2002`,
- stage into `runtime/sbbs/xtrn/tw2002`,
- run with the DOSEMU-capable image,
- configure node/dropfile paths in TW2002's setup tool.

TradeWars-style games can support concurrent players, so they are a good target
for model-vs-model experiments.

### Barren Realms Elite

Target path:

- supply original door files in `doors/bre`,
- stage into `runtime/sbbs/xtrn/bre`,
- run with the DOSEMU-capable image,
- configure dropfile paths and game resets.

BRE should be treated cautiously for concurrency until the specific version's
locking behavior is verified.

## Non-Goals

- Do not expose privileged game APIs to agents during normal play.
- Do not read or mutate door game state directly for normal scoring or
  decision-making.
- Do not commit proprietary door binaries, registration keys, or generated game
  state.
- Do not make model prompts depend on Synchronet internals unless the task is
  explicitly BBS administration.

## Next Implementation Milestones

1. Add active-prompt/current-line extraction to reduce stale scrollback
   confusion while preserving open-ended terminal input.
2. Tune TW2 in-game profile/memory from live traces, especially repeated trade
   offer mistakes and command-prompt phase confusion.
3. Add a sequential two-agent campaign runner that composes existing
   `ActivityRunner` sessions.
4. Add real Synchronet node discovery/allocation for rlogin/telnet sessions.
5. Add snapshot/reset tooling for `runtime/sbbs`.
6. Add score and task-completion extractors for TW2, TW2002/BRE, and BBS social
   workflows.
7. Add DOS-door setup verification for TW2002 and BRE.
8. Add optional PNG observation rendering from the pyte screen buffer.
