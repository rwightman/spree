# Spree BBS Gym

Containerized BBS sandbox for agent experiments: LLMs connect as terminal users,
play door games, and use BBS message areas/chat through normal telnet/rlogin
interfaces.

## Current Shape

- BBS runtime: Synchronet in Docker, with persistent state in `runtime/sbbs`.
- Local service ports: telnet `127.0.0.1:2323`, web `127.0.0.1:8080`,
  rlogin `127.0.0.1:2513`, NNTP `127.0.0.1:1119`, IRC `127.0.0.1:6667`.
- Agent client: `python -m bbs_gym.cli smoke` for raw telnet/ANSI transcripts.
- Door strategy:
  - Use Synchronet's bundled JS doors first for immediate smoke tests.
  - Stage original DOS doors from `doors/bre` and `doors/tw2002`.
  - Build the optional DOSEMU image only when original DOS doors are needed.

Synchronet was chosen because its maintained Docker image already exposes BBS
services and its external-program system supports common door dropfiles,
including `DOOR.SYS`, `DORINFO#.DEF`, `DOORFILE.SR`, and `DOOR32.SYS`.

## Quick Start

```bash
make init
docker compose up -d
python -m bbs_gym.cli smoke
```

If Docker does not start, run:

```bash
make doctor
```

For initial sysop configuration:

```bash
make scfg
```

For a shell inside the BBS container:

```bash
make shell
```

## Immediate Playable TradeWars-Like Door

Synchronet includes a JavaScript door named `tw2`. Install it into the BBS
configuration after the container has initialized:

```bash
make install-js-tw2
```

This gives you a fast local target for agent-session plumbing before dealing
with original DOS door setup and registration.

## Original DOS Doors

Put your legally obtained/extracted door files here:

```text
doors/bre/
doors/tw2002/
```

Then stage them into Synchronet's external-program tree:

```bash
make stage-dos-doors
```

Build and run with DOSEMU support:

```bash
make up-dos
```

Install the staged external-program configs:

```bash
docker compose exec bbs jsexec install-xtrn.js ../xtrn/bre -auto
docker compose exec bbs jsexec install-xtrn.js ../xtrn/tw2002 -auto
```

You still need to run each door's own setup editor to set node/dropfile paths,
game resets, and registration keys. See [docs/doors.md](docs/doors.md).

## Agent Smoke Test

```bash
python -m bbs_gym.cli smoke \
  --host 127.0.0.1 \
  --port 2323 \
  --transcript runtime/transcripts/smoke.raw
```

The transcript is stored as raw CP437/ANSI bytes. The CLI prints a plain-text
view with ANSI control sequences removed.

## Notes

- The default Compose file binds to loopback only. Change `BBS_HOST` in `.env`
  if you intentionally want to expose the BBS.
- Apple Silicon Macs are a later target for the BBS and JS doors. Original DOS
  doors through DOSEMU should be treated as x86_64 Linux-first.
- Door binaries are ignored by git by default.

## Sources

- [bbsio/synchronet Docker image](https://hub.docker.com/r/bbsio/synchronet)
- [Synchronet external program/dropfile docs](https://www.synchro.net/docs/external_programs.html)
- [Synchronet install-xtrn module](https://wiki.synchro.net/module:install-xtrn)
- [TradeWars on Linux/DOSEMU notes](https://www.arcadiabbs.com/setting-up-tradewars-on-linux/)
