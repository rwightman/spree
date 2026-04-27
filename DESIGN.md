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

Synchronet was chosen because it is actively maintained, has Docker support,
ships with useful JavaScript doors, and supports classic BBS door dropfiles.

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

- `TelnetSession` handles telnet connection and basic option negotiation.
- Raw CP437/ANSI bytes are stored for replay/debugging.
- `strip_ansi()` gives a rough plain-text observation.
- `BbsGym` manages multiple named agent sessions.

Near-term improvements:

- Use `pyte` as a virtual terminal screen so observations reflect cursor
  movement, cleared regions, and overwritten ANSI menu text.
- Add rlogin sessions for automated benchmark runs.
- Add quiescence/prompt-based turn boundaries instead of fixed sleep windows.

## Observation Model

There should be three observation forms:

1. Raw transcript bytes.
2. Pretty rendered terminal screen state for humans/replay.
3. Model-oriented rendered text with ANSI/color noise stripped or normalized.

The raw transcript is authoritative and should always be preserved. The rendered
model text is what we pass to models most of the time. The pretty render is for
debugging, replay, and screenshots.

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
    "timestamp": "...",
}
```

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

The benchmark harness should wait for one of these conditions before asking the
model for the next action:

- the virtual terminal screen has been stable for a configurable interval,
- a door/BBS prompt regex has matched,
- a profile-specific input state has been detected,
- a hard timeout has elapsed.

The first robust implementation should use `pyte` to detect screen changes and
combine that with per-profile prompt regexes. For example, TW2 can have a
different prompt set than Synchronet's main menu or message editor.

Quiescence is not enough by itself. Some games pause on screens that are not
asking for input, and some prompts update a clock/status line. Each door profile
needs its own prompt and "safe to act" rules.

## Action Model

Actions should be terminal input, not game commands.

Examples:

```python
agent.act("agent001")
agent.act("P")
agent.act("1")
agent.act("\x1b", newline=False)
```

The harness can expose convenience helpers for Enter, Escape, arrow keys, and
menu macros, but these should still compile down to terminal keystrokes.

## Connection And Login

Telnet should remain available because it is the most human-like path through
the BBS. It is useful for realism tests and for debugging what an ordinary user
would see.

Automated benchmark runs should prefer rlogin. Synchronet rlogin can pass user
identity during connection setup, avoiding fragile new-user and login prompts.
That makes runs easier to reproduce across Synchronet versions and local config
changes.

Planned transports:

- `telnet`: realistic user path, prompt-driven login required.
- `rlogin`: default automation path, pre-provisioned user identity.

The transport choice should be explicit in `BbsGym.connect(...)`.

## Accounts

Agents need deterministic BBS identities because door games usually key player
state by BBS user number or alias.

Planned account flow:

1. Generate account definitions for `agent001`, `agent002`, etc.
2. Provision those accounts through Synchronet tooling or a controlled setup
   script.
3. Store generated credentials under ignored runtime state.
4. Script login for each agent before handing control to the model.

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

1. Add a `pyte`-backed rendered terminal screen.
2. Add one scripted end-to-end path into Synchronet TW2.
3. Add quiescence and prompt-based `observe_turn()` handling.
4. Add structured event logs.
5. Add rlogin login/account provisioning helpers.
6. Add explicit node allocation in `BbsGym`.
7. Add an async multi-agent runner.
8. Add snapshot/reset tooling for `runtime/sbbs`.
9. Add DOS-door setup verification for TW2002 and BRE.
