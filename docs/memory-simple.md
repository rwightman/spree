# A simplified memory subsystem behind a common seam

Status: companion to `docs/memory-design.md`. That document remains the full
target design (the "causal" system below) and is not modified by this one.
This document does two things: defines a **common subsystem interface** so
memory implementations can be swapped per run and compared under identical
metrics, and distills a **simplified subsystem ("structured")** that keeps
the highest-value ideas from the full design at a fraction of its build cost.

## Why a parallel simplified system

The full design is an event-sourced store with causal effectiveness
projection, progression authorities, and a commit protocol. Its correctness
story is strong, but two costs precede any experimental signal:

1. **Build cost.** The storage engine, authority protocol, and implementation
   gates must exist before the first `reconciling` arm runs.
2. **Model-grammar risk.** Support predicates, dependency edges, and
   proposal ids raise the structural failure rate of utility responses; a
   high dead-letter rate would measure format compliance, not memory quality.

The simplified system inverts the bet: it keeps the anti-drop operations
contract, typed sections, caps-in-code, journaling, and mechanical cursor
rules — the parts that directly attack the observed failure modes — and drops
everything whose value is contingent on campaigns, concurrency, or crash
windows we have not yet hit in practice. If the structured arm already beats
`legacy`, we learn that cheaply; if it plateaus where the causal design's
extra machinery would help, the limitation table below says exactly where and
why, and the causal build is then justified by data.

Both systems (and `legacy`, and `raw-window-only`) sit behind one seam, so
trials are apples-to-apples and a run can swap systems by configuration.

## The common seam: `MemorySubsystem`

The seam is deliberately **coarser** than the full design's
`MemoryPolicy`/`PersonaMemoryStore` split. That split presumes the causal
internals (batches, prefixes, progressions). Swapping wildly different
internals requires the runner and campaign to depend only on a small
service-level surface:

```python
class MemorySubsystem(Protocol):
    name: str                                  # "structured" | "causal" | "raw-window-only"
    def fingerprints(self) -> dict[str, str]:  # schema/mutation/prompt fingerprints
        ...
    def open_context(self, agent_id: str, context_id: str) -> MemoryHandle: ...
    # The subsystem owns its root; both ids are path-validated. context_id is
    # e.g. "zork" or "sre-campaign-7" — flat, runtime-issued.

class MemoryHandle(Protocol):
    def observe(self, event: MemoryEvent) -> None: ...
    def render_context(self, budget_chars: int | None = None) -> str: ...
    def render_bootstrap(self, budget_chars: int | None = None) -> str: ...
    def maybe_reconcile(self, model: ModelAdapter, *, force: bool = False) -> ReconcileOutcome | None: ...
    def commit(self, model: ModelAdapter, extra_evidence: str = "") -> CommitOutcome: ...
    def close(self) -> None: ...
```

The `legacy` arm does **not** sit behind this interface: it stays inline in
the runner (selected when no subsystem is passed) and satisfies only the
measurement contract below through instrumentation.

- `observe` feeds activity as it happens (terminal steps today; a normalized
  event later). Implementations own their window/queue internally.
- `render_context` returns the decision-prompt memory section;
  `render_bootstrap` returns the full view a fresh provider chain needs.
  Rollover and stateful bootstraps call the latter, after a forced catch-up
  reconciliation folds in any pending events (free when nothing is pending;
  on failure the bootstrap degrades to committed memory with the failure
  journaled). Nothing else about provider lifecycle lives in the subsystem.
- `maybe_reconcile` is cadence-gated internally (`force` for rollover, flush,
  and pre-commit); it may call the model's utility surface and returns what
  happened, or `None` when cadence declined.
- `commit` durably folds the session into the persistent context store,
  with campaign-supplied `extra_evidence` (forum text) in its input.

The runner calls `observe` per step, `render_context` per decision,
`maybe_reconcile` per tick, and `commit` at `finish_state`. The campaign
calls `commit` at its post-social boundary instead. `ActivityRunner` keeps
retry/backoff mechanics; the subsystem decides representation, prompts, and
parsing.

### The interface carries the measurement contract

Swappability is worthless if arms cannot be compared. Every implementation
must therefore:

- **Journal every mutation** in a common record shape (`op`, section, item
  id, origin `model|system`, accepted/rejected + reason, source step range,
  timestamps) to a JSONL file under the context root, fsynced before any
  boundary advances. Each applied batch is one JSONL line
  (`{"records": [...]}`) with records in exact application order — system
  capacity exits sit at the point they occurred, and a torn trailing line is
  a torn batch, so truncation on recovery preserves batch atomicity. Metrics
  (capture rate, closure accuracy, churn, repeated-failure correlation) read
  this journal identically for every arm.
- **Report fingerprints** (schema, mutation rules, prompt text): the handle
  writes a `manifest.json` beside its journal, and the runner logs a
  `memory_context` record (subsystem name + fingerprints) into the activity
  log at open. Campaign-level manifests arrive with campaign wiring.
- **Never silently drop**: whatever the internal representation, an item may
  leave the active set only through a journaled, typed exit.
- **Run in an isolated root** per experiment arm.

The contract is honest about its limit: it makes journals **structurally**
identical, not semantically. A `legacy_replace_summary` pseudo-op and a set
of item-level `add`/`contradict` operations answer capture/closure/churn
questions differently, so cross-arm metrics need a small arm-aware
normalization layer in the analyzer — that layer does not exist yet, and no
claim of metric identity should be made until it does.

`legacy` satisfies this by wrapping its summary rewrite and merge patch as
two journaled pseudo-ops, exactly as the full design specifies. Subsystems
call the model through the typed `utility_text(messages, operation)` surface
on `ModelAdapter`, which owns trace reset, output filtering, per-operation
settings, and truncation/incomplete/empty detection; CLI adapters run utility
calls stateless so they never pollute the resumed gameplay session. The rest
of the provider-lifecycle work (call contexts, the stage handshake,
`reset_decision_state`) lives in `docs/memory-design.md` and is not
duplicated here.

## The `structured` subsystem

One store per `(agent_id, context_id)` — deliberately *not* a cross-activity
persona store. A JSON document plus an append-only ops journal, guarded by
the existing `flock` + atomic-replace machinery from `tty_agent/memory.py`.
Single writer by construction (the handle holds the lock); no revisions, no
CAS, no scopes, no progressions.

### Representation

```python
@dataclass(frozen=True)
class SimpleGoal:
    id: str                    # code-issued m1, m2, ...
    text: str
    status: Literal["open", "blocked", "done", "abandoned"]
    priority: int
    context: str = ""          # for blocked: what blocks it

@dataclass(frozen=True)
class SimpleItem:              # facts and hypotheses share a shape
    id: str
    text: str
    source_steps: tuple[int, ...]   # code-recorded, never model-authored

@dataclass(frozen=True)
class StructuredMemory:        # the whole store document
    schema_version: int        # 1
    next_id: int
    state: dict[str, str]      # small keyed current-state map, capped
    goals: tuple[SimpleGoal, ...]
    facts: tuple[SimpleItem, ...]
    hypotheses: tuple[SimpleItem, ...]
    archive: tuple[dict, ...]  # every exited item + typed exit reason
```

Kept from the full design: typed sections with the fact/hypothesis
distinction, goal statuses including `blocked`-with-context, keyed state
instead of prose summary blobs, code-issued ids, caps per section enforced in
code, archive-with-typed-exits (the in-store archive is itself capped,
oldest-evicted; the journal retains every exit forever). Dropped: claims and
corrections as sections (a
contradicted item exits to the archive with reason `contradicted`, and the
replacement — if any — is an ordinary `add`; another player's assertion is
recorded as a fact *about the speech act* or a hypothesis, at the model's
discretion), provenance beyond step numbers, versioned references, temporal
kinds, staleness horizons.

### Operations contract

The model's reconciliation response is a JSON list of operations —
**nothing else**. No envelope, no support predicates, no dependencies, no
proposal ids. Six verbs:

| Verb | Target | Notes |
| --- | --- | --- |
| `add` | goal/fact/hypothesis | code issues id, records current step range |
| `revise` | any active item | text (goals: also priority/context); id kept |
| `set` | state key | create or overwrite one bounded keyed value |
| `promote` | hypothesis → fact | id kept, section moves |
| `contradict` | fact/hypothesis | exits to archive; optional replacement text becomes an `add` |
| `transition` | goal | status change per the legal-move table; `abandoned` needs a reason |

There is no model-facing `demote` and no `drop`: **capacity is entirely
code-owned**. When an `add` crosses a cap, code archives the lowest-priority
goal or the oldest item (numeric-id tiebreak) with exit `capacity` and
journals it as `origin="system"`. Unmentioned items persist — the
default-persist rule is identical to the full design.

Validation is **per-operation, not all-or-nothing**: without dependency
edges, operations are independent, so valid ops apply and invalid ops are
individually rejected with journaled reasons. This deletes the full design's
repair-ladder/partition/dead-letter machinery in one stroke — its purpose was
protecting cross-op atomicity that cannot arise here.

### Mechanical advancement, unchanged

The rules that killed the attestation flaw carry over verbatim: the covered
boundary advances exactly when the response parses, accepted ops are applied,
the journal record is fsynced, and no truncation signal fired
(`finish_reason`, `output_tokens` near the utility budget). Zero ops against
a non-trivial window → one retry → advance with a `no_memory_change` marker.
Each pass reconciles a **bounded contiguous prefix** of pending events
(`max_events_per_reconcile`), so a backlog — especially one built up across
failure retries — cannot grow the prompt without limit; the remainder stays
pending and the next pass (or the commit's drain loop) continues from where
coverage stopped. Overlap re-presents the last few covered steps as context
(the existing `summarized_recent_steps` mechanics). Semantic omission stays
measurable, not preventable — same honest position, same metrics.

### Crash safety, proportionate

The journal is the source of truth; the store document is a cache. One
property test carries the whole guarantee:

```text
replay(ops_journal) == store_document
```

Journal appends are fsynced before the covered boundary advances; the store
document is rewritten atomically after each accepted batch. Recovery is:
truncate a torn trailing journal line (one line = one batch, so a torn line
is a never-applied batch), replay, continue. That is the entire recovery
design. The property is held by a randomized replay test, not a single
happy-path example. Stated volatility, deliberately accepted: pending
un-reconciled events and the covered/overlap boundary live only in process
memory — a crash loses evidence not yet folded in, never anything journaled.
Durable evidence logs are full-design scope.

### Commit and campaigns

`commit` first drains the pending backlog in bounded prefixes using the same
extraction prompt and model settings as ordinary reconciliation. It then
presents `extra_evidence` (forum text, injected by the campaign at its
post-social boundary exactly as the full design schedules it) **exactly
once** through that extraction path — repeating it per prefix would duplicate
updates and inflate churn metrics.

After extraction succeeds, one independent eventless audit examines the
complete bounded active store. The audit is cleanup-only: validation permits
`revise`, replacement-free `contradict`, and `transition`, so it cannot add
items, set state, or trigger capacity eviction. It is journaled as its own
`audit` batch. An audit provider failure is recorded but nonfatal; successful
extraction remains committed. HTTP adapters let `audit_temperature` /
`--audit-temperature` override sampling for this pass and otherwise inherit
the decision temperature (the Zork example therefore remains at `0.6` by
default). The audit reuses the final-memory reasoning, extra-body, token, and
retry-ceiling settings.

There is no separate campaign representation and no `_merge_memory`.
Bounded extraction retry and a journaled commit event replace the
silent-empty-patch behavior, as required of every arm. (The campaign runner
does not yet call `commit` with forum evidence — that wiring, like the intent
channel, is still open.)

### Accepted limitations — the experiment's hypotheses

Each row is a deliberate omission and names what the causal system buys; if
the structured arm's failures cluster on a row, that row justifies the build.

| Limitation | Consequence | What the causal design buys |
| --- | --- | --- |
| No effect-dependent evidence | After a crash/abandoned progression, beliefs from discarded world effects persist unmarked | `experience_occurred` vs `external_effect_committed`; projection suppression |
| Per-context stores | No cross-game persona identity; opponent knowledge does not follow the agent between doors | persona store, scopes, related-scope retrieval |
| Single writer, no revisions | One activity per context at a time; concurrent activities need separate contexts | revision vectors, CAS cursors, serializable reads |
| No pending overlay | Mid-session commits are immediately authoritative; an interrupted epoch keeps them | progression outcomes, pending/committed folds |
| No history search / resurrect | Archived items recoverable only by post-hoc analysis | `search_history` + `resurrect` |
| Flat provenance (step numbers) | Weaker audit; claims/speaker attribution unavailable | typed sources, actor ids, claims section |

### What is shared, not duplicated

The two live bug fixes (prose-fallback wipe, silent empty commit), the
provider lifecycle normalization, the intent channel, typed initial
`compaction_max_tokens` / `memory_max_tokens` budgets plus their adaptive retry
ceilings, and the raw-window tuning all live in `docs/memory-design.md` and
apply to every arm equally. The `structured` subsystem assumes them; it does
not re-specify them. The inline legacy control additionally uses a selective
compaction prompt and code-enforced working-summary/campaign-document bounds;
those safeguards improve the control without changing its wholesale-summary
representation.

## Measurement and the updated matrix

The common journal schema means every metric in the full design's
Measurement section computes identically for `structured`: new-fact capture
rate, stale-facts-active, goal-closure accuracy, repeated failed actions,
churn, plus score/cost. The matrix gains an arm and an ordering:

`{legacy, structured, raw-window-only}` are buildable immediately and run the
Zork matrix first; `causal` joins the matrix when built, and the decision to
build it is informed by where `structured` fails. The pre-registered gate
discipline (trial set, minimum effect, uncertainty criterion, cost ceiling
declared before results) applies unchanged.

## Coexistence and migration

A `structured` store can be imported into a causal persona store later: its
ops journal replays as model-origin operations with `experience_occurred`
support derived from recorded step ranges, into the scope matching its
context id. The reverse direction is deliberately unsupported — projection
semantics cannot be flattened without loss. Fingerprints keep imported
history attributable to the system that produced it.
