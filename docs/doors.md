# Door Setup

## Bundled JS TradeWars-Like Door

Synchronet ships a JavaScript door named `tw2`. It is not the original TW2002
DOS binary, but it is useful for proving that agents can discover and drive a
door game through the terminal server.

```bash
docker compose up -d
make install-js-tw2
```

## Barren Realms Elite

1. Put the extracted BRE files in `doors/bre`.
2. Run `make stage-dos-doors`.
3. Start the DOSEMU-capable container with `make up-dos`.
4. Install the external-program config:

   ```bash
   docker compose exec bbs jsexec install-xtrn.js ../xtrn/bre -auto
   ```

5. In `scfg`, inspect the generated program and set/confirm:

   ```text
   Name: Barren Realms Elite
   Internal Code: BRE
   Start-up Directory: ../xtrn/bre
   Command Line: bre.exe
   Multiple Concurrent Users: No or door-managed
   I/O Method: DOSEMU / socket depending on Synchronet version
   Native Executable/Script: No
   BBS Drop File Type: Solar Realms DOORFILE.SR
   ```

BRE is not truly simultaneous multi-node gameplay in the same way as TW2002.
For experiments, serialize BRE sessions unless you have verified the specific
version you run handles lock contention cleanly.

## Solar Realms Elite

The original author hosts a DOSBox-ready SRE 0.994b archive. Fetch, verify,
stage, and register its Synchronet entry with:

```bash
make install-sre
```

Run `SPREG.BAT` once inside the door directory to generate the author's free
registration, then reset the game with `SPRERST.BAT`. The reset batch uses a
31-Dec-1999 DOS clock, as recommended by the author, because the unmodified
game rejects later maintenance timestamps.

Immediately after a reset, while the world is idle and before any players
join, advance it to the current hour with:

```bash
make patch-sre-reset-time
```

This makes a timestamped backup under `runtime/backups`, raises only SRE
0.994b's four hard-coded 1999 timestamp comparisons, advances the two
validation timestamps in `DATA/SYSTEM.II`, decrypts and advances all three
maintenance clocks in `DATA/GALAXY.II`, and sets the tournament start to the
beginning of the current UTC day. SRE compares that field with a date
normalized to midnight, so using the current hour would leave the game closed
until the following day. The helper then re-encrypts the Galaxy and rebuilds
its integrity descriptor. The backup contains `SRE.EXE`, `SYSTEM.II`, and
`GALAXY.II`. Override `SRE_BACKUP_DIRECTORY` if a specific destination is
desired.

The staged `external.bat` also sets `TZ=UTC0`. This keeps SRE's DOS runtime
on the same clock as the reset-time helper and avoids a seasonal four- or
five-hour offset. Apply the patch only to a freshly reset, idle world; SRE's
`inuse.sr` lock must not exist.

Synchronet runs a host-side JavaScript cleanup module after the door exits so a
carrier-loss or budget disconnect cannot strand SRE's `inuse.sr` lock. The
module is only a recovery path; SRE/BRE sessions still require serialized
admission. Accelerated daily-turn campaigns are described in
[accelerated DOS-door campaigns](door-campaigns.md).

The recovered record encryption, integrity layout, and a lossless-first JSON
plan for a clean Python implementation are documented in
[SRE reverse engineering](sre-reverse-engineering.md).

## TradeWars 2002

1. Put the extracted TW2002 files in `doors/tw2002`.
2. Run `make stage-dos-doors`.
3. Start the DOSEMU-capable container with `make up-dos`.
4. Install the external-program config:

   ```bash
   docker compose exec bbs jsexec install-xtrn.js ../xtrn/tw2002 -auto
   ```

5. In `scfg`, inspect the generated program and set/confirm:

   ```text
   Name: TradeWars 2002
   Internal Code: TW2002
   Start-up Directory: ../xtrn/tw2002
   Command Line: tw2002.exe TWNODE=%N NOXMS NOEMS BUFFER=16500 MULTITASK=YES
   Multiple Concurrent Users: Yes
   I/O Method: DOSEMU / socket depending on Synchronet version
   Native Executable/Script: No
   BBS Drop File Type: GAP DOOR.SYS
   ```

6. Run TW2002's `TEDIT.EXE` inside the DOS environment and configure each node's
   dropfile path. Use DOS paths as seen by DOSEMU, not Linux host paths.

The DOSEMU config in this repo uses memory settings commonly needed by TW2002:
640K conventional memory, 4096K XMS, 4096K EMS, and DPMI enabled.

## Registration And Licensing

This repo does not ship BRE, TW2002, registration files, cracks, or archives.
Drop your own legally usable distributions into `doors/` and keep them out of
git. The `.gitignore` is set up to enforce that by default.

## Fallback: Remote TWGS/RLogin

If local DOS execution is unstable, configure a Synchronet internet gateway
door to an rlogin/TWGS server. That is less reproducible than local execution
but useful for testing the agent driver against a working TradeWars endpoint.
