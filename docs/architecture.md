# Architecture

## Runtime Boundary

The BBS is the source of truth for the game/social environment. Agents do not
call game APIs directly; they connect over telnet/rlogin and interact with the
same terminal UI that a human BBS user would see.

```text
LLM agent process
  -> bbs_gym TelnetSession
  -> localhost:2323
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

The Python package starts with a low-level telnet client. The next layer should
add:

- account creation/login scripts,
- per-agent transcript directories,
- action/observation step records,
- match orchestration for multiple agents,
- reset hooks for each door game.

