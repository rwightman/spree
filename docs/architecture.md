# Architecture

## Runtime Boundary

The BBS is the source of truth for the game/social environment. Agents do not
call game APIs directly; they connect over telnet/rlogin and interact with the
same terminal UI that a human BBS user would see.

```text
LLM agent process
  -> tty_agent transport
  -> bbs_gym Synchronet profile/activity glue
  -> localhost:2323 or localhost:2513
  -> Synchronet terminal server
  -> message boards, chat, external doors
```

## Why Synchronet

Synchronet is still actively maintained, has a maintained Docker image, supports
telnet/rlogin/web/NNTP/IRC services, and has first-class external-program
configuration. Its dropfile support is the important part for classic doors:
TW2002 usually uses `DOOR.SYS`, while BRE uses Solar Realms-style
`DOORFILE.SR`.

## Door Strategy

There are three practical classes of doors:

1. Native Synchronet JavaScript doors. These are easiest to automate and should
   be used for smoke tests.
2. Original DOS doors under DOSEMU. These are the target for BRE and TW2002 on
   x86_64 Linux.
3. Remote gateways such as rlogin/TWGS/door-party services. These are useful
   compatibility fallbacks but are not ideal for a reproducible local gym.

## Persistence

All BBS state lives under `runtime/sbbs`, which is mounted into the container as
`/sbbs-data`. This includes users, messages, logs, xtrn data, and door state.
Delete that directory only when you intentionally want a fresh BBS.

## Agent Driver

The generic tty-agent core lives in `tty_agent`:

- `actions.py`: structured terminal actions and validation,
- `agent.py`: the minimal `TerminalAgent` protocol consumed by the runner and
  `TerminalSessionAgent` for concrete session-backed dispatch,
- `terminal.py`: pyte-backed screen rendering and quiescence observation,
- `runner.py`: bounded observe/decide/act activities with compaction,
- `models.py`: model adapters for OpenAI-compatible APIs, Anthropic, and tests,
- `profiles.py`: empty/stability-only and shell prompt profiles,
- `transports/`: telnet, rlogin, and local PTY sessions.

Model adapters keep raw responses for traces and parse actions from filtered
responses. This matters for local reasoning models served by vLLM or similar
OpenAI-compatible servers: reasoning tags can stay in logs without becoming
terminal input.

The BBS package stays as domain glue: Synchronet CP437 defaults, BBS/TW2 prompt
profiles, door-game activity profiles, account registry/provisioning, Docker
config, and the CLI surface.

The next BBS layer should add:

- real Synchronet node discovery/allocation,
- match orchestration for multiple agents,
- reset hooks for each door game,
- score and task-completion extraction.
