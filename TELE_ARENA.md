# Tele-Arena Through Ether

This is the experimental path for running Tele-Arena as a standalone telnet
target and driving it with Spree's `bbs-door-line` profile. It does not use
Synchronet. Ether serves the game directly on telnet port `3000`.

The important harness detail is line endings: Ether accepts LF for Enter. Keep
`--telnet-enter lf` when using `bbs-gym` against Ether. The default remains CR
because that is what the Synchronet/TW2 path currently expects.

## Status

- Tested with Tele-Arena 5.6d data converted by AreaBuilder.
- Tested with the Ether Java telnet server on `127.0.0.1:3000`.
- Tested with Codex and Claude CLI via `examples/tele_arena_activity.py`.
- Ether runtime fixes are maintained outside this repo in
  <https://github.com/rwightman/ether-arena>.
- Game data, Ether archives, generated data, player files, and logs belong
  under ignored `runtime/tele-arena/`.

This repo does not ship Tele-Arena, Ether, converted data, player files, or
registration material. Use files you are allowed to run. Ether's own install
notes state that Ether does not include the copyrighted Tele-Arena data files.
For Java/runtime fixes, use the `ether-arena` fork.

## Sources

- Ether fork with local runtime fixes: <https://github.com/rwightman/ether-arena>
- Original Ether runtime: <https://sourceforge.net/projects/jether/files/ether1.00b56.zip/download>
- Original Ether source: <https://sourceforge.net/projects/jether/files/src/ether_src1.00b53.zip/download>
- Ether/AreaBuilder install notes: <https://tdod.org/ether/install.html>
- Tele-Arena module page: <https://www.mbbsemu.com/Module/TSGARN>
- Tele-Arena wiki notes: <https://wiki.mbbsemu.com/doku.php?id=modules%3Atsgarn>

## 1. Install Local Prerequisites

```bash
sudo apt install unzip ant default-jdk
```

The original Ether tooling was built for old Java. For Java 21, use the
`ether-arena` fork. It carries the Java/runtime fixes separately from Spree's
terminal-agent harness code.

## 2. Put Archives Under Runtime

Create the ignored workspace:

```bash
mkdir -p runtime/tele-arena/downloads
```

Download or place these archives there:

```text
runtime/tele-arena/downloads/ether1.00b56.zip
runtime/tele-arena/downloads/AreaBuilder1.00b1.zip
runtime/tele-arena/downloads/TSGARN_MBBSEmu.zip
```

`TSGARN_MBBSEmu.zip` is one convenient source for the `.MSG` files AreaBuilder
needs. The original installer can also work if you extract the same message
files yourself.

## 3. Prepare Ether And AreaBuilder

```bash
mkdir -p runtime/tele-arena/ether \
  runtime/tele-arena/area-builder

unzip -q runtime/tele-arena/downloads/ether1.00b56.zip \
  -d runtime/tele-arena/ether

unzip -q runtime/tele-arena/downloads/AreaBuilder1.00b1.zip \
  -d runtime/tele-arena/area-builder
```

For the patched Java source tree, clone the fork under ignored runtime storage:

```bash
git clone https://github.com/rwightman/ether-arena.git \
  runtime/tele-arena/ether-arena
```

The resulting paths should include:

```text
runtime/tele-arena/ether/ether/
runtime/tele-arena/ether-arena/
runtime/tele-arena/area-builder/AreaBuilder/
```

## 4. Stage Tele-Arena Message Files

AreaBuilder expects these six files in its `data/` directory:

```text
TSGARN-C.MSG
TSGARN-D.MSG
TSGARN-M.MSG
TSGARN-T.MSG
TSGARNDD.MSG
TSGARNDT.MSG
```

For the MBBSEmu-ready archive, extract them with:

```bash
mkdir -p runtime/tele-arena/area-builder/AreaBuilder/data

unzip -j runtime/tele-arena/downloads/TSGARN_MBBSEmu.zip \
  'TSGARN*.MSG' \
  -d runtime/tele-arena/area-builder/AreaBuilder/data
```

Confirm:

```bash
ls runtime/tele-arena/area-builder/AreaBuilder/data/TSGARN*.MSG
```

## 5. Convert The Area Data

Run AreaBuilder from its own directory:

```bash
cd runtime/tele-arena/area-builder/AreaBuilder

java \
  --add-opens java.base/java.util=ALL-UNNAMED \
  --add-opens java.base/java.text=ALL-UNNAMED \
  --add-opens java.desktop/java.awt.font=ALL-UNNAMED \
  -DTaConfigFile=config/ta.properties \
  -jar AreaBuilder.jar
```

The generated files land in `build/`. A successful local conversion produced
`town.xml`, `world1.xml`, room descriptions, item/mob/spell data, help data,
and `tamessages.properties`.

## 6. Copy Converted Data Into Ether

From the repository root:

```bash
ETHER_HOME=runtime/tele-arena/ether-arena

cp runtime/tele-arena/area-builder/AreaBuilder/build/town.xml \
  runtime/tele-arena/area-builder/AreaBuilder/build/world1.xml \
  runtime/tele-arena/area-builder/AreaBuilder/build/town_room_desc.xml \
  runtime/tele-arena/area-builder/AreaBuilder/build/world_room_desc.xml \
  "$ETHER_HOME/area/"

cp runtime/tele-arena/area-builder/AreaBuilder/build/armor.dat \
  runtime/tele-arena/area-builder/AreaBuilder/build/barriers.dat \
  runtime/tele-arena/area-builder/AreaBuilder/build/cmd_triggers.xml \
  runtime/tele-arena/area-builder/AreaBuilder/build/emotes.dat \
  runtime/tele-arena/area-builder/AreaBuilder/build/equipment.dat \
  runtime/tele-arena/area-builder/AreaBuilder/build/help.dat \
  runtime/tele-arena/area-builder/AreaBuilder/build/mob_weapons.dat \
  runtime/tele-arena/area-builder/AreaBuilder/build/mobs.dat \
  runtime/tele-arena/area-builder/AreaBuilder/build/npcs.dat \
  runtime/tele-arena/area-builder/AreaBuilder/build/spells.dat \
  runtime/tele-arena/area-builder/AreaBuilder/build/tamessages.properties \
  runtime/tele-arena/area-builder/AreaBuilder/build/teleporters.dat \
  runtime/tele-arena/area-builder/AreaBuilder/build/traps.dat \
  runtime/tele-arena/area-builder/AreaBuilder/build/treasures.dat \
  runtime/tele-arena/area-builder/AreaBuilder/build/weapons.dat \
  "$ETHER_HOME/data/"
```

Set `ETHER_HOME=runtime/tele-arena/ether/ether` instead if you intentionally
want to run the original unpacked runtime.

## 7. Start Ether

```bash
cd runtime/tele-arena/ether-arena
ant jar
java -DTaConfigFile=config/ta.properties -jar ether.jar
```

Expected startup includes:

```text
PortListener ;Listening to Port 3,000
Genesis ;Server up
```

If the original Ether jar fails on a modern JDK with XStream or reflection
errors, use the `ether-arena` fork above.

## 8. Smoke Test The Telnet Prompt

In another shell:

```bash
telnet 127.0.0.1 3000
```

You should see:

```text
Welcome to the Java port of Tele-Arena 5.6d!
Enter your character's name or type NEW:
```

## 9. Let An Agent Play Through The Wrapper

From the repository root:

```bash
uv run python examples/tele_arena_activity.py \
  --activity bbs-door-line \
  --provider codex \
  --model gpt-5.5 \
  --max-decision-ticks 100
```

The wrapper delegates to:

```bash
uv run bbs-gym run-activity \
  --host 127.0.0.1 \
  --port 3000 \
  --transport telnet \
  --telnet-enter lf \
  --activity bbs-door-line \
  --provider codex \
  --model gpt-5.5
```

`bbs-door-line` is the preferred Tele-Arena profile. It allows `submit_line`
for complete line-oriented commands while keeping `type_text` and `press_key`
available for prompts that behave like single-key BBS inputs. Use
`bbs-door-safe` when testing a door that auto-accepts values before Enter, such
as the TW2 JavaScript door.

Claude Code can run through the same wrapper in stateful mode:

```bash
uv run python examples/tele_arena_activity.py \
  --activity bbs-door-line \
  --provider claude \
  --model sonnet \
  --agent-id tele-arena-claude \
  --claude-stateful \
  --claude-session-file runtime/claude-sessions/tele-arena-claude.session \
  --max-decision-ticks 100
```

The trace defaults to:

```text
runtime/logs/tele-arena-codex-bbs-door-line-lf.jsonl
```

Pretty-print it with:

```bash
python scripts/trace_pretty.py \
  runtime/logs/tele-arena-codex-bbs-door-line-lf.jsonl \
  --show-new-text \
  --out runtime/logs/tele-arena-codex-bbs-door-line-lf.pretty.txt
```

## Observed Result

The first successful Codex run created `ArenaCodex`, completed character
creation, entered the north plaza, used `HELP`, `STATUS`, `INVENTORY`, `EXITS`,
and `LOOK`, navigated to the equipment shop, bought a torch/waterskin/food,
visited the guild hall, returned to the plaza, and hung up cleanly at step 100.

A stateful Claude run with `bbs-door-line` created `ArenaLine`, completed
character creation, bought starter supplies, entered the arena, fought a giant
bat, died, recovered in the temple, and continued until the 100-step budget.
The same run used `submit_line` for most complete commands and had no action
validation failures.

## 10. Let Two Agents Play A Match

`run-match` opens one telnet session per participant. The default `sequential`
scheduler alternates one decision tick per active agent. `parallel_barrier`
collects decisions concurrently and commits them in the scheduled order;
`parallel_race` commits actions as model decisions finish. `continuous` keeps
one decision in flight per active agent and immediately requeues that agent
after each committed action, so faster models get more chances to act during the
same match wall-clock budget. Inline participant specs use
`agent_id:provider:model`; each participant still gets its own per-agent JSONL
trace and model state.

```bash
uv run bbs-gym run-match \
  --host 127.0.0.1 \
  --port 3000 \
  --transport telnet \
  --telnet-enter lf \
  --no-agents-config \
  --activity bbs-door-line \
  --participant arena-codex:codex:gpt-5.5 \
  --participant arena-claude:claude:sonnet \
  --codex-stateful \
  --claude-stateful \
  --prompt-layout cache_friendly \
  --log-path runtime/logs/tele-arena-match.jsonl \
  --disable-action hangup \
  --run-objective "Play Tele-Arena as {agent_id}. If asked for a character name, create or log in as {agent_id}. Stay connected; do not hang up or quit. Other active agents: {opponents}. Survive, gain experience and gold, buy and equip useful supplies, spend gold wisely, recover when hurt, find opponents, and defeat them when prepared." \
  --max-rounds 100 \
  --max-decision-ticks 100
```

The match trace goes to `runtime/logs/tele-arena-match.jsonl`. Per-agent traces
use the same stem, for example `tele-arena-match.arena-codex.jsonl` and
`tele-arena-match.arena-claude.jsonl`.

For match runs, `--max-wall-seconds` is match-level. `--max-decision-ticks`
still applies per participant. In `continuous` mode, `--max-rounds` caps the
total queued action decisions for the whole match rather than full all-agent
rounds. Continuous traces use `tick` instead of `round` for scheduler events and
do not emit `round_started` / `round_completed` lifecycle events.

## Notes

- Use `--telnet-enter lf` for Ether. CR-only caused repeated delayed submits.
- `bbs-door-line` is the right starting profile for Tele-Arena because most
  gameplay commands are typed lines that expect Enter.
- `bbs-door-safe` is still useful for doors that often auto-accept typed values
  before Enter.
- The wrapper is intentionally thin; pass any extra `bbs-gym run-activity`
  arguments after the wrapper arguments and they will be forwarded.
- Match-specific objectives should carry game strategy. Add
  `--disable-action hangup` for competitive runs so agents cannot leave the
  match with the harness-level hangup action.
