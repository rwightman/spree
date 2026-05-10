# Tele-Arena Through Ether

This is the experimental path for running Tele-Arena as a standalone telnet
target and driving it with Spree's `bbs-door-safe` profile. It does not use
Synchronet. Ether serves the game directly on telnet port `3000`.

The important harness detail is line endings: Ether accepts LF for Enter. Keep
`--telnet-enter lf` when using `bbs-gym` against Ether. The default remains CR
because that is what the Synchronet/TW2 path currently expects.

## Status

- Tested with Tele-Arena 5.6d data converted by AreaBuilder.
- Tested with the Ether Java telnet server on `127.0.0.1:3000`.
- Tested with Codex via `examples/tele_arena_activity.py`.
- Game data, Ether archives, generated data, player files, and logs belong
  under ignored `runtime/tele-arena/`.

This repo does not ship Tele-Arena, Ether, converted data, player files, or
registration material. Use files you are allowed to run. Ether's own install
notes state that Ether does not include the copyrighted Tele-Arena data files.

## Sources

- Ether runtime: <https://sourceforge.net/projects/jether/files/ether1.00b56.zip/download>
- Ether source: <https://sourceforge.net/projects/jether/files/src/ether_src1.00b53.zip/download>
- Ether/AreaBuilder install notes: <https://tdod.org/ether/install.html>
- Tele-Arena module page: <https://www.mbbsemu.com/Module/TSGARN>
- Tele-Arena wiki notes: <https://wiki.mbbsemu.com/doku.php?id=modules%3Atsgarn>

## 1. Install Local Prerequisites

```bash
sudo apt install unzip ant default-jdk
```

The original Ether tooling was built for old Java. The path below worked in
the local development tree with Java 21 after patching/rebuilding Ether from
source and using a modern XStream jar. If you are setting this up from scratch,
an older JDK is usually the lower-friction path. Java 21 notes are included
where they mattered during the local setup.

## 2. Put Archives Under Runtime

Create the ignored workspace:

```bash
mkdir -p runtime/tele-arena/downloads
```

Download or place these archives there:

```text
runtime/tele-arena/downloads/ether1.00b56.zip
runtime/tele-arena/downloads/ether_src1.00b53.zip
runtime/tele-arena/downloads/AreaBuilder1.00b1.zip
runtime/tele-arena/downloads/TSGARN_MBBSEmu.zip
```

`TSGARN_MBBSEmu.zip` is one convenient source for the `.MSG` files AreaBuilder
needs. The original installer can also work if you extract the same message
files yourself.

## 3. Unpack Ether And AreaBuilder

```bash
mkdir -p runtime/tele-arena/ether \
  runtime/tele-arena/ether-src \
  runtime/tele-arena/area-builder

unzip -q runtime/tele-arena/downloads/ether1.00b56.zip \
  -d runtime/tele-arena/ether

unzip -q runtime/tele-arena/downloads/ether_src1.00b53.zip \
  -d runtime/tele-arena/ether-src

unzip -q runtime/tele-arena/downloads/AreaBuilder1.00b1.zip \
  -d runtime/tele-arena/area-builder
```

The resulting paths should include:

```text
runtime/tele-arena/ether/ether/
runtime/tele-arena/ether-src/ether/
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
cp runtime/tele-arena/area-builder/AreaBuilder/build/town.xml \
  runtime/tele-arena/area-builder/AreaBuilder/build/world1.xml \
  runtime/tele-arena/area-builder/AreaBuilder/build/town_room_desc.xml \
  runtime/tele-arena/area-builder/AreaBuilder/build/world_room_desc.xml \
  runtime/tele-arena/ether/ether/area/

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
  runtime/tele-arena/ether/ether/data/
```

## 7. Start Ether

```bash
cd runtime/tele-arena/ether/ether
java -DTaConfigFile=config/ta.properties -jar ether.jar
```

Expected startup includes:

```text
PortListener ;Listening to Port 3,000
Genesis ;Server up
```

If the original Ether jar fails on a modern JDK with XStream or reflection
errors, either run it with an older JDK or rebuild the Ether source with a
modern XStream jar. The repo includes a source-only Java 21 patch for the two
issues seen locally: modern XStream type permissions and the removed
`Thread.suspend`/`Thread.resume` APIs.

```bash
cd runtime/tele-arena/ether-src/ether
patch -p2 < ../../../../docs/patches/ether-java21.patch
ant
```

Then copy the rebuilt `ether.jar` plus the newer XStream jar into
`runtime/tele-arena/ether/ether/`.

Keep this as a source patch rather than committing rebuilt jars or game data. A
patch keeps the setup reproducible while preserving the boundary between
Spree's harness code and third-party game assets. A separate Ether fork only
makes sense if the patch grows into ongoing runtime maintenance.

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

## 9. Let Codex Play Through The Wrapper

From the repository root:

```bash
uv run python examples/tele_arena_activity.py \
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
  --activity bbs-door-safe \
  --provider codex \
  --model gpt-5.5
```

The trace defaults to:

```text
runtime/logs/tele-arena-codex-bbs-door-safe-lf.jsonl
```

Pretty-print it with:

```bash
python scripts/trace_pretty.py \
  runtime/logs/tele-arena-codex-bbs-door-safe-lf.jsonl \
  --show-new-text \
  --out runtime/logs/tele-arena-codex-bbs-door-safe-lf.pretty.txt
```

## Observed Result

The first successful Codex run created `ArenaCodex`, completed character
creation, entered the north plaza, used `HELP`, `STATUS`, `INVENTORY`, `EXITS`,
and `LOOK`, navigated to the equipment shop, bought a torch/waterskin/food,
visited the guild hall, returned to the plaza, and hung up cleanly at step 100.

## Notes

- Use `--telnet-enter lf` for Ether. CR-only caused repeated delayed submits.
- `bbs-door-safe` is the right starting profile because Tele-Arena often
  accepts one-character choices but also has typed command lines.
- The wrapper is intentionally thin; pass any extra `bbs-gym run-activity`
  arguments after the wrapper arguments and they will be forwarded.
- The current objective is conservative. For more exploratory runs, override
  `--run-objective`.
