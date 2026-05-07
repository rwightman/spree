# Spree BBS Gym

Containerized BBS sandbox for agent experiments: LLMs connect as terminal users,
play door games, and use BBS message areas/chat through normal telnet/rlogin
interfaces.

See [DESIGN.md](DESIGN.md) for the agent/environment boundary, observation
model, timing strategy, and multi-agent plan.
See [NEXT.md](NEXT.md) for near-term implementation notes and observed failure
modes from live TW2 runs.

## Current Shape

- BBS runtime: Synchronet in Docker, with persistent state in `runtime/sbbs`.
- Local service ports: telnet `127.0.0.1:2323`, web `127.0.0.1:8080`,
  rlogin `127.0.0.1:2513`, NNTP `127.0.0.1:1119`, IRC `127.0.0.1:6667`.
- Terminal-agent core: `terminal_agent` owns actions, observations, model
  adapters, memory, runners, transports, observation hints, and prompt modules.
- BBS shell: `bbs_gym` owns Synchronet defaults, CP437 policy, BBS/TW2 prompt
  profiles, activity-specific prompt modules, activities, and CLI commands.
- Agent client: `python -m bbs_gym.cli smoke` for raw telnet/ANSI transcripts
  and `python -m bbs_gym.cli run-activity` for bounded model-driven sessions.
- Model providers: OpenAI-compatible chat endpoints, Anthropic Messages, Codex
  CLI subprocess calls, and scripted test responses.
- Debug tooling: JSONL traces can be rendered with `scripts/trace_pretty.py`;
  raw transcripts can be replayed into ANSI HTML or animated GIFs with
  `scripts/ansi_screencap.py`.
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

## Synchronet Docker Runtime

The default Compose service runs `bbsio/synchronet:3.19c` and mounts persistent
BBS state at `runtime/sbbs`. Ports are bound to loopback unless `BBS_HOST` is
changed in `.env`.

Common operations:

```bash
docker compose up -d
docker compose ps
docker compose logs -f --tail=200 bbs
docker compose down
```

The exposed local services are telnet `127.0.0.1:2323`, rlogin
`127.0.0.1:2513`, web `127.0.0.1:8080`, NNTP `127.0.0.1:1119`, and IRC
`127.0.0.1:6667`. Telnet is useful for human-realistic smoke tests; rlogin is
the preferred automation path once agent accounts are provisioned.

## Immediate Playable TradeWars-Like Door

Synchronet includes a JavaScript door named `tw2`. Install it into the BBS
configuration after the container has initialized:

```bash
make install-js-tw2
```

This gives you a fast local target for agent-session plumbing before dealing
with original DOS door setup and registration.

Reset or adjust the JS TW2 game state during development:

```bash
make reset-js-tw2
make grant-js-tw2-turns PLAYER=RLoginSmoke TURNS=30
```

`reset-js-tw2` reinitializes the local JS TW2 universe. `grant-js-tw2-turns`
updates one TW2 player record without resetting the world, which is useful for
continuing an interrupted agent session or forcing a daily-turn rollover during
experiments.

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

The generic PTY path can be checked without the BBS:

```bash
python -m examples.shell_agent
```

## Activity Traces And Replays

`run-activity` writes one JSONL record per decision tick. Each record includes
the observation shown to the model, prompt-module provenance, raw and parsed
model responses, validation notes, the parsed action, budget state, and the raw
transcript path. New traces also include absolute `transcript_byte_start` and
`transcript_byte_end` offsets so replay tools can render activity traces that
share one long telnet/rlogin transcript.

Pretty-print a trace:

```bash
python scripts/trace_pretty.py runtime/logs/activity.jsonl \
  --show-new-text \
  --show-controls \
  --out runtime/logs/activity.pretty.txt
```

Render a colored terminal frame or animated GIF from the raw transcript:

```bash
python scripts/ansi_screencap.py runtime/logs/activity.jsonl \
  --step 42 \
  --out runtime/logs/activity-step42.ansi.html

python scripts/ansi_screencap.py runtime/logs/activity.jsonl \
  --gif-out runtime/logs/activity.gif \
  --start-step 10 \
  --end-step 60 \
  --duration-ms 2000
```

The GIF path requires Pillow. The replay is only as colorful as the raw
transcript: if Synchronet sends monochrome output for a given rlogin/telnet
session, the GIF will be monochrome too. For current traces, transcript byte
offsets are read automatically. For older traces that do not include absolute
offsets, pass `--base-byte-offset N` when rendering an activity that starts
mid-session.

## Local vLLM OpenAI-Compatible Server

The OpenAI-compatible adapter works with local servers such as vLLM, Ollama, and
llama.cpp. A minimal vLLM Docker server for a two-GPU Linux box looks like:

```bash
docker run --rm --gpus all --ipc=host --shm-size 16g \
  -p 127.0.0.1:8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-32B \
  --tensor-parallel-size 2 \
  --dtype auto
```

Adjust `--model` and `--tensor-parallel-size` for the local hardware. The
example agent registry points `qwen-local-001` at `http://localhost:8000/v1`.
For Qwen-style models that emit `<think>` blocks, raw model responses are kept
in the JSONL trace while the action loop parses a filtered response.
Response filtering is selected from the model id by default. `gemma-4` model
ids use the Gemma 4 channel/thought filter, while the default filter handles
common `<think>...</think>` style reasoning blocks. Override with
`--response-filter auto|default|gemma4|none` or `response_filter` in the agent
registry.

## Codex CLI Provider

The `codex` provider invokes `codex exec` once per decision tick and parses its
final message through the same action JSON path as the other providers. This is
useful for debugging and for trying Codex as a player without standing up a
separate API server.

```bash
python -m bbs_gym.cli run-activity \
  --transport rlogin \
  --agent-id codex-debug \
  --provider codex \
  --model gpt-5.5 \
  --codex-sandbox read-only \
  --activity tw2-game
```

Useful options are `--codex-profile`, `--codex-executable`,
`--codex-timeout`, `--codex-cwd`, and repeated `--codex-arg=...` values for
extra `codex exec` flags. The adapter also supports the same fields in
`config/agents.local.json` under the agent's `model` object. Codex calls are
stateless from the harness perspective; the full prompt is sent each tick and
harness memory remains the source of truth.

Prompt construction defaults to `--prompt-mode stateless_full`, which sends the
full harness context every decision tick. `--prompt-mode stateful_delta` sends a
full bootstrap prompt once and then shorter delta prompts with the current
observation and previous-step summary. That mode is intended for future resumed
provider sessions such as Codex CLI resume; use it only when the provider
actually preserves prior context.

For Codex CLI, `--codex-stateful` captures the Codex session id from `--json`
on the first call and resumes that same session on later decision ticks. When
`--codex-stateful` is set and no explicit `--prompt-mode` is provided, the
activity automatically uses `--prompt-mode stateful_delta`.

```bash
python -m bbs_gym.cli run-activity \
  --transport rlogin \
  --agent-id codex-debug \
  --provider codex \
  --model gpt-5.5 \
  --codex-stateful \
  --activity tw2-entry
```

Use `--codex-session-file runtime/codex-sessions/codex-debug.session` if you
want the captured Codex session id persisted for later runs. Without a session
file, the session id is only kept in memory for the current process.

## Agent Accounts

Copy the example identity registry and put real passwords in environment
variables or an ignored local config:

```bash
cp config/agents.example.json config/agents.local.json
python -m bbs_gym.cli accounts list
python -m bbs_gym.cli accounts check
python -m bbs_gym.cli accounts provision
```

`accounts provision` creates or updates Synchronet users through `jsexec`.
Automated runs can then use deterministic rlogin identity:

```bash
python -m bbs_gym.cli run-activity \
  --transport rlogin \
  --agent-id qwen-local-001
```

The rlogin transport defaults to terminal type `ansi`. Use
`--rlogin-terminal` to compare terminal negotiation strings such as `ansi`,
`xterm`, or `xterm-256color` when debugging color/ANSI behavior.

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
