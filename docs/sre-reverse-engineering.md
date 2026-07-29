# Solar Realms Elite recovery plan

The SRE work in this repository targets the author's DOS release 0.994b. The
pristine `SRE.EXE` used for analysis has SHA-256
`b4381bb2326580ca2d0ee0bf28997e95dc8d6be9893f2f3d345b396ac627acb9`.
Offsets and structures below must be treated as version-specific until another
release is tested.

## Feasibility

A clean, inspectable Python implementation is feasible. It is not equivalent
to automatically recovering the original source: names, abstractions,
comments, and some intent disappeared during compilation. The remaining game
rules can be reconstructed from file formats, visible behavior, documentation,
and narrow disassembly of the DOS executable. Differential tests can then run
the same initial world and choices through DOS SRE and Python and compare the
resulting state.

The work should proceed losslessly. A field is named only after evidence
supports it, while every unknown byte remains available for exact round trips.
This lets a useful JSON inspector arrive well before every gameplay rule is
understood.

The original game executable, prose, and artwork should remain user-supplied
unless their redistribution terms are separately established. The Python
engine and format documentation can be original, reviewable code. The
[author's SRE documentation][author-docs] is an important behavioral oracle.

[author-docs]: http://www-cs-students.stanford.edu/~amitp/Articles/SRE-Documentation.html

## Recovered protected-record format

SRE protects world records with a 24-byte descriptor:

| Offset | Size | Meaning |
| ---: | ---: | --- |
| `0x00` | 4 | little-endian cumulative 16-bit checksum in a 32-bit field |
| `0x04` | 4 | little-endian reflected CRC-32 state before the final XOR |
| `0x08` | 8 | initial cipher state/nonce |
| `0x10` | 8 | final cipher state and verifier |

Encryption is an evolving eight-byte XOR stream. For plaintext byte `p` at
index `i`, ciphertext is `p XOR state[i & 7]`. That state byte is then rotated
left by three and XORed with an eight-bit counter and `p`; the counter advances
by seven for every byte. `decode_sre_record` and `encode_sre_record` in
`bbs_gym.sre_data` implement and validate this format.

The descriptor hierarchy in 0.994b is now known:

```text
SYSTEM.II (322 bytes)
└── descriptor at 0x12a → GALAXY.II (966-byte protected record)
    ├── descriptors at 0x054 + slot*0x18 → 25 EMPIRE.II records (764 bytes each)
    └── descriptors at 0x2ac + slot*0x18 → 9 PIRATE.II records (262 bytes each)
```

All 35 links above have been validated against the live reset world. Known
Galaxy scheduler fields are the 32-bit daily-maintenance cursor at `0x40`, the
tournament start at `0x44`, and the hourly-maintenance cursor at `0x3ba`. SRE's
play gate compares the tournament start with a current-day value normalized to
midnight, so a reset patch uses the beginning of the current UTC day for that
field and the current hour for both cursors. Other files and most entity fields
still need names.

## Proposed JSON state

The first schema should be a versioned inspection and interchange format, not
an assertion that every field is understood:

```json
{
  "schema_version": 1,
  "format": "sre-0.994b-world",
  "source": {
    "executable_sha256": "b4381bb2326580ca2d0ee0bf28997e95dc8d6be9893f2f3d345b396ac627acb9",
    "files": {}
  },
  "system": {
    "raw_plaintext_base64": "...",
    "fields": {}
  },
  "galaxy": {
    "raw_plaintext_base64": "...",
    "integrity": {},
    "fields": {
      "hourly_maintenance_at": 1785268800,
      "daily_maintenance_at": 1785268800,
      "tournament_started_at": 1785196800
    }
  },
  "empires": [
    {
      "slot": "A",
      "raw_plaintext_base64": "...",
      "integrity": {},
      "fields": {}
    }
  ],
  "pirates": [],
  "append_only_files": {}
}
```

Each named field should also have a machine-readable layout definition in
Python: byte offset, width, scalar/string encoding, and any enum mapping. An
importer starts from `raw_plaintext_base64`, applies named-field changes, and
rebuilds descriptors upward from Empire/Pirate to Galaxy to System. This makes
unknown fields round-trip unchanged and lets later schema revisions add names
without invalidating old snapshots.

For reproducibility, `source.files` should contain byte length and SHA-256 for
every input. Human-friendly UTC strings may accompany timestamps, but the raw
integer remains authoritative. Large append-only files such as messages,
battles, and reports can initially be preserved as base64 blobs, then promoted
to arrays of records when their boundaries are known.

## Implementation stages

1. Build a read-only world loader that validates the complete descriptor tree
   and emits lossless JSON with known scheduler fields.
2. Map player, empire, pirate, message, battle, economy, and configuration
   fields. Add binary-to-JSON-to-binary round-trip fixtures after every mapping.
3. Record small DOS reference scenarios from fixed snapshots and choices. Map
   each state delta to one rule: turn income, food, population, markets,
   research, combat, diplomacy, maintenance, and victory/scoring.
4. Implement those rules as pure Python transitions over typed state. Use a
   deterministic injected clock and random-number stream.
5. Differentially replay scenarios against DOS SRE until state deltas agree,
   then add a terminal/UI adapter independently of the engine.

This structure produces useful artifacts at every stage: an inspector first,
safe editors and reset tooling next, and eventually a standalone game engine.
