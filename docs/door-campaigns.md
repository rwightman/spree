# Accelerated DOS-door campaigns

Classic shared-world doors such as SRE and BRE meter each player in game turns
and replenish those turns on a daily maintenance boundary. A competitive model
match should advance that boundary explicitly instead of waiting for real days
or rewriting the door's world files after every round.

The generic epoch scheduler and the SRE 0.994b adapter are implemented. BRE can
use the same scheduler after it gets its own reset, launch, maintenance, and
score adapter.

## Run an SRE campaign

Build the pinned Ubuntu 24.04/Synchronet/DOSEMU image and fetch SRE first. The
Synchronet service does not need to be running: campaign sessions launch the
door directly in short-lived, network-isolated containers.

```bash
make build-dos
make fetch-sre

uv run bbs-gym run-campaign \
  --campaign-dir runtime/campaigns/sre-claude-vs-codex \
  --start-time 2026-07-28T12:00:00Z \
  --epochs 30 \
  --agent-id claude-blue \
  --agent-id codex-debug \
  --campaign-order rotate \
  --max-decision-ticks 50 \
  --max-wall-seconds 600 \
  --social-rounds 2
```

`--agent-id` loads the stable SRE player name and model binding from
`config/agents.local.json`; BBS passwords are neither needed nor exposed. For
an inline run, use `--no-agents-config` and repeat
`--participant agent_id:provider:model`.

Resume from the last committed epoch with the same participant identities,
clock, order, and social settings:

```bash
uv run bbs-gym run-campaign \
  --campaign-dir runtime/campaigns/sre-claude-vs-codex \
  --resume \
  --start-time 2026-07-28T12:00:00Z \
  --epochs 30 \
  --agent-id claude-blue \
  --agent-id codex-debug \
  --campaign-order rotate \
  --social-rounds 2
```

The campaign directory owns the isolated world, DOS launch files, model memory,
terminal and evaluator transcripts, JSONL journal, per-session recovery
snapshots, committed epoch snapshots, score logs, forum records, and manifests.
It is never the live `runtime/sbbs` world.

On first creation, the SRE adapter drives the door's native `SRE RESET` UI at
1999-12-31, using explicit reproducible game settings. It then applies the
narrow SRE 0.994b Y2K validation patch, advances the protected reset clocks to
the requested start time, and proves the guest `DATE`/`TIME` before admitting a
player. Every later epoch uses SRE's native `SRE MAINTENANCE` command.

## Runtime boundary

The campaign orchestrator owns one isolated world shared by every participant:

```text
campaign orchestrator
├── world snapshot and epoch journal
├── virtual DOS date for epoch N
├── player A → DOSEMU → door → shared world
├── player B → DOSEMU → door → shared world
├── ...
└── barrier → advance one day → native maintenance → scores/snapshot
```

SRE and BRE sessions are serialized even if model inference happens in
parallel. The game process and files remain the authority, and each participant
gets a distinct door identity/dropfile in the same world. Separate per-player
worlds would remove diplomacy, trade, attacks, and other multiplayer effects.

For SRE, `DOORFILE.SR` line 1 is the stable BBS handle and line 8 is the real
name; it does not carry a numeric BBS user ID. The campaign `player_name` is
written unchanged on every launch, while `agent_id` only identifies the
orchestrator/model. SRE persists the handle-to-ruler association in `PLAYER.II`
and the ruler's current empire in `EMPIRE.II`.

This is a campaign-level scheduler, not the existing decision-level
`run-match` scheduler. `run-match` keeps all participant terminal sessions open
while interleaving individual decisions; a single-node DOS door instead needs
one complete player session to close before the next participant is admitted.

## Virtual time

The pinned DOSEMU2 image defaults to `$_timemode = "bios"`. Its configuration
documents that this clock is monotonic within an emulator run, does not follow
later host clock changes, and can be set from DOS. The launcher therefore sets
the DOS date and time before every door or maintenance invocation.

The campaign epoch maps to a fixed UTC instant safely inside each game day,
for example noon:

```text
epoch 0 → 2026-07-28 12:00:00 UTC
epoch 1 → 2026-07-29 12:00:00 UTC
epoch 2 → 2026-07-30 12:00:00 UTC
```

Every participant and the maintenance process for an epoch sees the same
virtual date. Choosing noon avoids crossing a date boundary during a long
session. `TZ=UTC0` keeps calendar conversion deterministic. The Linux host and
orchestrator clocks remain untouched, and monotonic host time still enforces
real wall-clock budgets.

SRE's encrypted timestamp patch remains useful once when bootstrapping a reset
1999 world into the supported modern range. It is not the epoch mechanism.
After bootstrap, the game observes the injected DOS date and its own native
maintenance updates the persisted cursors and player turns.

Before admitting players, the SRE adapter uses both its legacy reset date and
the requested campaign date, confirms DOS `DATE`/`TIME` output, and validates
the door's protected maintenance state. Other doors may read file timestamps or
the RTC differently and require an adapter-specific clock helper.

## Epoch transaction

Each epoch runs as a recoverable transaction:

1. Restore or open the last committed world snapshot and acquire its exclusive
   host lock.
2. Set the epoch's virtual date. Choose a fixed or seeded rotating participant
   order and write it to the journal.
3. For each participant, generate its dropfile, launch one DOSEMU door process,
   expose its terminal through the local PTY/telnet/rlogin adapter, and run until
   the game-turn limit, session budget, normal exit, or failure policy ends the
   session.
4. Ask the player door to return cleanly to the BBS, confirm the process is
   gone, clear only the adapter's known stale lock, then run the adapter's
   out-of-band final scorer and checkpoint the world. The scorer may use an
   offline score command, a lossless data inspector, or a fresh evaluator-only
   terminal session. SRE uses its global F5 return-to-BBS path here because a
   forced PTY/container teardown can discard its in-memory player and empire
   updates. If F5 does not exit within the bounded grace period, the session
   fails and is restored rather than being treated as safely persisted.
5. At an all-participant barrier, advance the virtual date by one day and run
   the door's native daily maintenance exactly once.
6. Poll scores again, emit a lossless state snapshot where supported, hash the
   binary world files, and atomically commit the new epoch manifest. If the
   active poll fails but a previously extracted leaderboard remains valid, the
   epoch records an explicit `fallback` score-probe status and commits that last
   snapshot rather than failing otherwise successful game maintenance.

The orchestrator never advances the date merely because one participant
finished early. A timeout or crash is recorded as that participant's outcome;
the configured retry/forfeit policy runs before the barrier. On a host crash or
partial file write, restore the most recent per-session checkpoint rather than
trying to infer which writes completed.

## Fairness and accounting

- Give every participant the same model-decision and real wall-clock budget per
  game day. Door game turns are a separate observed metric, not model decisions.
- Rotate or deterministically shuffle first-player order across epochs. A fixed
  order can create systematic first-mover advantages even though sessions are
  serialized correctly.
- Hidden status/score probes do not consume model decision ticks, do not enter
  model context, and must be demonstrated not to consume game turns. An
  in-session probe may be skipped when the current prompt is unsafe; the
  post-close scorer is what guarantees the end-of-session sample.
- SRE's post-close scorer reads both Empire Status and the live Scores menu.
  It checkpoints that parsed leaderboard as `.spree-scoreboard.json` inside
  the isolated world, because the game's optional `SRESCORE.TXT` export can be
  stale even while its live menu is correct.
- After SRE maintenance, a separate evaluator-only probe opens the live Galaxy
  Scores menu against a disposable copy of the world. Only the parsed JSON
  leaderboard is committed back to the canonical world, so incidental SRE
  metadata writes cannot affect the campaign. SRE 0.994b no longer provides a
  usable unattended `SRE SCORES` command despite that command appearing in
  older bundled documentation.
- Record the injected timestamp, participant order, binary hashes, maintenance
  result, score records, terminal transcript paths, and model metadata in every
  epoch manifest.
- Use one process-admission lock owned by the orchestrator. Door-specific lock
  files are secondary evidence and cleanup targets, not the concurrency
  primitive.

## Optional campaign forum

SRE itself exposes Read Messages and Send Messages in its Galaxy menu. The
campaign forum is intentionally separate: it provides the same public social
surface to games with no native messaging, and it remains available after a
player has spent all game turns or left the door.

`--social-rounds N` enables zero or more synchronized social rounds after every
participant has completed the epoch's game session and before maintenance.
Each participant may post once or pass in each round, so `N` is also the maximum
number of messages per participant per epoch. Every participant in one round
sees the same snapshot; drafts commit together and become visible in the next
round. This permits proposal/reply/counteroffer sequences without giving a
later model an advantage from inference order.

Forum calls have their own accounting and do not increment game decision ticks.
Messages are length-limited, stored as append-only epoch records, labeled as
untrusted player speech in model prompts, and never receive hidden evaluator
scores. Posts from an incomplete epoch are not considered committed and are
ignored on resume. Committed posts are included in the next epoch's game
objective so stateless as well as stateful models can act on agreements.

## Door adapter surface

The shared orchestration should be generic while each game supplies a small
adapter with these responsibilities:

- create/reset and validate a world;
- generate a participant dropfile and launch command;
- set and verify the guest clock;
- identify turn exhaustion and a clean terminal exit;
- invoke native hourly/daily maintenance;
- provide passive and active score probes with declared game-turn cost;
- inspect or export state, preserving unknown bytes losslessly;
- list safe stale-lock cleanup targets.

A direct local PTY is the simplest transport. A thin telnet or rlogin bridge can
sit in front of it when an agent profile expects those protocols; it should not
own game state or time. Synchronet remains valuable as the human-play/reference
adapter and as a compatibility oracle, but the accelerated benchmark should not
depend on its wall clock or login menus.

Initially, native SRE/BRE maintenance is the behavioral authority. A future
Python engine can replace it only after differential tests show identical state
transitions from the same snapshots, choices, clock, and random inputs.
