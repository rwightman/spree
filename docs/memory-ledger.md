# The `ledger` memory subsystem: durable evidence behind the common seam

Status: proposed third arm, not yet implemented, revised after external
review (which replaced stage 3's item-level tombstones with
progression-scoped batch disposition — see below). It does not replace
`structured` — that arm stays as the running experiment baseline — and it
does not resurrect the full causal machinery of `docs/memory-design.md`. It
is `structured`'s store core wrapped in durable evidence: an append-only
event ledger, durable reconciliation cursors, and progression-scoped batch
disposition. Build order is gated: stage 1 starts only after the Zork policy
loop on `structured` (consistency/audit prompt variants, paired runs) has
run its course, and later stages start only when the campaign features they
serve are actually wired.

## Why a third arm, and why "ledger"

Two independent analyses of the qwen Zork runs converged on the same split:

- The **dominant** failure modes in solo play are semantic — stale mutable
  snapshots recorded as timeless facts, under-use of closure operations —
  and are being addressed at the prompt-policy layer (the state-consistency
  appendix, the cleanup-only audit), measurably, through the point-replay
  harness. (Not the *only* failures: oldest-item capacity eviction is a
  visible code-policy weakness, which is why the heuristic stays pluggable
  and fingerprinted.) No storage machinery fixes a model that writes "the
  gates are now closed" as an invariant.
- The failure modes the full design targets have not occurred in Zork and
  structurally cannot: evidence from abandoned epochs treated as
  authoritative, crash loss of un-reconciled evidence, multi-channel
  evidence with no total order. Those are **campaign** phenomena. For
  SRE/BRE campaigns with accelerated epochs, repaired generations, forums,
  and provider rollovers, that correctness core is worth building.

The name follows the same convention as `raw-window-only`: it names what the
arm *is*. Its defining property is that events, cursors, mutations, and
progression outcomes are durable append-only records — ledgers — and memory
validity follows them. The name deliberately does not promise fine-grained
causal semantics: support predicates, proposal ids, and dependency edges are
omitted from the model grammar. Weaker models already struggle with much
simpler utility calls, so every provenance and disposition relationship in
this arm is code-derived.

## What `ledger` reuses from `structured`, unchanged

- The six-verb model-facing grammar and per-operation validation. Nothing in
  this arm makes the utility model's job harder. (The first 500-tick qwen
  run produced 25/25 parseable, advancing reconciliations with one rejected
  op — mechanically clean, even though the semantic results still contained
  stale facts; the two layers fail independently and are fixed
  independently.)
- The canonical record reducer: materialized state is a cache of the mutation
  journal fold, with one mutation code path shared by live application and
  replay-on-open. Stage 3 qualifies that fold by progression disposition.
- The mutation journal (one fsynced JSONL line per batch, records in exact
  application order, torn trailing line = torn batch), the
  manifest/fingerprint compatibility check on open, caps-in-code with
  journaled typed exits, and the cross-arm measurement contract.
- The extraction / cleanup-audit lifecycle split, including the audit
  operation allowlist (`revise`, replacement-free `contradict`,
  `transition`) and nonfatal audit failure. The offline point replays are
  promising — the constrained audit revised or contradicted stale entries
  without growing memory, where mixed extraction+cleanup encouraged
  additions — but one set of replays is evidence of direction, not a
  demonstration of correctness; the paired-run matrix is what settles it.
- Rendering (fair-share section budgets, reserved pending share in
  bootstrap) and the reconcile/audit prompt structure with configurable
  appendices.

Implementation-wise this means extracting `structured`'s store core for
reuse, not forking it: `ledger` replaces the volatile pending/overlap layer
and adds hooks around the same reducer and journal writer. `structured`
itself is not modified beyond that extraction.

## Stage 1 — the event ledger and durable cursors

One `events.jsonl` per context beside the ops journal: every observed
`MemoryEvent` is normalized and appended (fsynced per step — negligible at
BBS pace) with a store-issued monotonic `sequence` **and a runtime-issued
stable `event_id`**. The sequence gives multi-channel evidence a total order
the moment forum and message events join; the event id makes a retried
append idempotent after the original append succeeded but its caller
crashed — a sequence alone cannot. Repeating an `event_id` with the same
normalized payload succeeds idempotently; reusing it with different content
is corruption and fails.

```jsonc
{"sequence": 41, "event_id": "run7-step37", "kind": "terminal_step",
 "step": 37, "session_id": "run-...", "progression_id": "epoch-3-p2",
 "channel": "terminal", "observation": "...", "action": "...", "intent": "",
 "warnings": [], "actor_id": null, "thread_id": null, "message_id": null}
```

Stable **actor, thread, and message ids are populated from stage 1**
whenever the runtime knows them, even though nothing consumes them until the
claims stage: historical events cannot be reliably retrofitted with identity
later. `events.jsonl` also contains ordered code-owned control records:
`session_started`, `progression_started`, and `progression_resolved`. They
make replay boundaries and progression disposition durable without pretending
that control metadata is model evidence. Reconciliation selection never sends
control-record payloads to the utility model; it consumes them through
code-only cursor batches and never selects activity records across a
progression boundary. This retires the point-replay tool's one accepted blind
spot (crashed sessions leaving no boundary marker). A resolution control
record carries the code-owned `progression_id`, `outcome`, and `authority_ref`;
it is the only local disposition artifact.

**Cursors ride in `ops.jsonl`, not in a separate file.** Each
applied batch's envelope records both `covered_from_exclusive` and
`covered_through_inclusive`. The from-side is derivable for a single writer,
but recording and validating both catches skipped prefixes and corruption
cheaply, and batch application plus cursor advancement become one fsynced
line — they cannot disagree after a crash, and replay recomputes the cursor
with the store. Cursor-only system batches advance across control records;
model batches contain a bounded contiguous prefix of activity records from
exactly one progression. **Prefixes never cross a progression boundary.**
Pending is derived (`cursor → event-ledger head`), the overlap tail is derived
from recently covered activity records in the same progression, and both
survive a process crash. Overlap resets at every progression boundary; it
cannot smuggle prior-progression evidence into a new prefix. The two standing
volatility caveats in `structured` — pending evidence and the covered boundary
living only in process memory — are deleted, not documented.

**Provenance is bounded and honestly named.** The batch envelope's presented
range *is* the provenance record: exact, append-only, and free. What it
records is which events were **presented with** an operation — every
operation in a batch receives the whole prefix — not verified semantic
support; the design must not claim more. Materialized items carry a small
bounded set of recent provenance references (the qwen store accumulated
hundreds of step numbers on single facts through `revise`'s source-append
behavior — an unbounded attic in miniature), and goals and state entries
carry provenance and version metadata too, which `structured` today records
only for facts and hypotheses.

Once an `observe` append returns successfully, stage 1 buys crash-safe
pending, exact replay boundaries, multi-channel `observe`, idempotent ingest,
and campaign forum/message events entering the same evidence stream as
terminal steps. Atomicity between an external action and construction of its
`MemoryEvent` remains a runtime concern; the subsystem cannot preserve an
event it was never asked to append.

## Stage 2 — extraction and audit

Carried over from `structured` as-is; the lifecycle split is the shape the
replay evidence favors, and its operation allowlist is mechanical rather
than trusted. The only `ledger` change is mechanical: prefixes come from the
durable cursor rather than the in-memory queue, and stop at progression
boundaries.

## Stage 3 — progression-scoped batch disposition

The SRE correctness core, and the reason this arm exists. The first draft of
this stage used item-level tombstones ("demote items whose evidence lies
entirely within the abandoned progression") and was **wrong**: it handles
abandoned *additions* only. A revision of a committed fact merges sources
and would survive carrying abandoned text; a contradiction archives the
committed original with nothing active to demote; goal transitions, promote,
capacity evictions triggered by abandoned adds, and state overwrites all
mutate committed memory in ways an item scan cannot undo. Applying mutations
immediately and later scanning current items cannot be made correct.

The replacement leans on the reducer property instead. There are two derived
views: a committed projection and, only for the handle explicitly bound to the
active progression, a provisional projection that also folds that whole
progression. In notation:

```text
dispositions = verified_dispositions(events.jsonl)
committed_store = fold(ops.jsonl, dispositions)
active_store(P) = fold(ops.jsonl, dispositions, include_unresolved=P)
```

`verified_dispositions` contains only `progression_resolved` control records
admitted through the trusted scheduler boundary described below. The
materialized store and disposition index are caches of these folds.

- Every mutation batch envelope carries a **code-owned `progression_id`**
  (model batches and system batches alike; standalone batches use `null`).
- **At most one unresolved progression may affect a context** at a time —
  the memory-side mirror of the scheduler's durability-domain invariant.
- `start_progression` durably installs the context's active progression.
  During play, only `open_progression_context(..., progression_id=P)` may
  open that provisional branch, and it must match the installed id. Ordinary
  `open_context` refuses while a progression is unresolved. On process
  recovery, the scheduler resolves or repairs external authority before any
  ordinary handle can render memory. This prevents a fresh session from
  silently inheriting an indeterminate pre-crash branch.
- The bound handle sees the fold of committed batches plus all batches from
  its one unresolved progression, so it sees its own memory updates. This is
  one progression-wide provisional fold, not a per-item or per-operation
  support overlay.
- `progression_resolved` outcomes are control records in `events.jsonl`, not
  a second disposition artifact in `ops.jsonl`. On **commit**, the same
  batches move from the provisional to the committed fold with no content
  change. On **abandonment**, the store is re-folded excluding every batch of
  that progression. The exclusion reverts
  every mutation kind exactly: revisions revert to committed text,
  contradicted items return from the archive, goal transitions and
  promotions revert, capacity demotions triggered by abandoned additions
  restore their victims, and overwritten state entries recover their
  committed values.
- Replay-on-open derives the verified disposition index from control records,
  then folds `ops.jsonl` with the same exclusions — open-after-crash and live
  abandonment produce identical stores.
- **Issued ids are never reused**: `next_id` tracking runs across excluded
  batches too, so an id spent by an abandoned progression stays spent.
- Cursor coverage stands (the events were genuinely presented), raw events
  remain in the ledger and searchable, and the voided batches remain in the
  journal as measurable extraction work — excluded from the fold, not
  erased.

Runs without an external scheduler use `progression_id=null`; their batches
are immediately effective and need no resolution record. Session markers
remain useful for replay, but the subsystem does not invent a durability
protocol for a world that has none.

This is a very small causal projection — batch-granular, code-owned, no
model-authored support graph — and it still fits the name `ledger`. The
fully-simple alternative, if even the provisional branch is unwanted, is to
defer reconciliation until the progression outcome is authoritative
(reconcile committed progressions only; mark abandoned ones presented
without applying changes) — correct, but it makes memory lag an entire epoch
behind play; take it only if the provisional branch proves troublesome.

### Campaign authority: what exists and what this stage requires

Current campaign durability is **epoch-level only**: a successful session
checkpoint updates an in-memory variable, the durable `campaign-state.json`
pointer advances only after the complete epoch, and resume restores the
epoch-level checkpoint — a crash late in an epoch currently discards earlier
successful sessions from that epoch. Stage 3 therefore either (a) defines
the durability progression as the **whole epoch** initially — matching what
is actually durable today, at the cost of coarse abandonment — or (b) first
implements durable per-session generation publication with a resumable
intra-epoch position. Option (a) is the honest starting point; (b) is
scheduler work, not memory work. With option (a), recovery repairs and marks
the partial epoch interrupted, then advances; it never replays paid activity
from the abandoned epoch.

The resolution protocol is normative, in this order:

1. Under the durability-domain lock, fsync the scheduler's progression-start
   record.
2. Idempotently call `start_progression` and fsync its control record in
   `events.jsonl`.
3. Execute activity.
4. Publish the external checkpoint/state pointer containing the outcome —
   this publication is the sole commit authority.
5. Through the trusted scheduler API, mirror the outcome as a
   `progression_resolved` control record in `events.jsonl` idempotently.
6. On recovery, repair a missing start or resolution mirror from scheduler
   state and the authoritative pointer before activity resumes.

The scheduler validates that the authority reference is published before it
calls the memory subsystem; the subsystem deliberately does not interpret BBS
or campaign manifests. `resolve_progression` is therefore a trusted boundary,
not an arbitrary model- or caller-facing append. Repeating the same
progression/outcome/reference succeeds; a conflicting outcome or reference
fails. On recovery, the scheduler consults external authority and calls this
same idempotent path before permitting an ordinary open. These ordering and
recovery rules are what guarantee that an unpublished reference never becomes
effective. A future runtime that cannot provide this trust boundary may inject
an authority resolver without changing the journal format.

### Seam shape

`ActivityRunner.finish_state()` commits and closes its memory handle before
the campaign creates its checkpoint, so progression resolution cannot be a
handle method. Instead the arm exposes a small optional capability on the
**subsystem**:

```python
class ProgressionMemorySubsystem(Protocol):  # optional; ledger implements it
    def start_progression(self, agent_id: str, context_id: str, progression_id: str) -> None: ...
    def open_progression_context(
        self, agent_id: str, context_id: str, progression_id: str,
    ) -> MemoryHandle: ...
    def resolve_progression(
        self, agent_id: str, context_id: str, progression_id: str,
        outcome: Literal["committed", "abandoned"], authority_ref: str,
    ) -> None: ...
```

`start_progression` and `resolve_progression` operate on
`(agent_id, context_id)`, opening the context internally under the normal
lock. The scheduler starts the progression before the runner opens its bound
handle, then resolves it after `finish_state` and checkpoint publication
without keeping runner handles alive or probing handle attributes.
`resolve_progression` is the one internal path allowed to open an unresolved
context without a play binding. `MemorySubsystem` itself stays coarse;
runners that know nothing about progressions keep using ordinary contexts.

### Escalation path, with trigger

Batch disposition is progression-granular: an abandoned progression voids
*all* of its memory work, including extractions of genuine experience
("I attempted X and observed Y" remains true even when the epoch's world
effects are discarded). If campaign data shows valuable experience-history
being lost this way — or provisional visibility mattering under genuinely
concurrent progressions — the escalation is the full design's
`experience_occurred`/`external_effect_committed` distinction and pending
overlay (`docs/memory-design.md`). Do not build it ahead of that evidence;
one salvage-shaped alternative (re-presenting an abandoned progression's
events for re-extraction under the committed store) should be tried first
since it needs no new machinery.

## Implementation gates

Stage 1 is not complete until deterministic tests cover:

- an event append that returns before a process crash and remains pending on
  reopen;
- an identical duplicate `event_id` succeeding idempotently and a conflicting
  payload failing;
- a torn trailing event or ops line being ignored/truncated at its batch
  boundary;
- a durable ops batch with a missing/stale cache replaying to the same store
  and cursor;
- a cursor `from` mismatch, skipped sequence, or non-contiguous prefix being
  rejected;
- a model prefix stopping at a progression boundary while code-only cursor
  batches consume control records;
- overlap resetting rather than crossing into the next progression; and
- no control-record payload ever reaching a utility prompt.

Stage 3 additionally requires:

- abandonment restoring the committed predecessor after each mutation class:
  add plus capacity exit, revise, contradict, promote, goal transition, and
  state overwrite;
- an authority publication with a missing memory mirror being repaired
  idempotently before ordinary open;
- a crash between the scheduler start and the memory start being repaired
  without executing or replaying paid activity;
- an unresolved context refusing ordinary open, but accepting the matching
  explicit progression binding;
- duplicate identical starts succeeding while a conflicting start or second
  unresolved progression fails;
- duplicate identical resolutions succeeding and conflicting resolutions
  failing;
- a resolution fsynced with a stale materialized cache reopening to the same
  effective projection; and
- excluded additions still advancing the id high-water mark.

Property tests assert the independent folds:

```text
materialized_store == fold(ops.jsonl, verified_dispositions(events.jsonl))
materialized_cursor == fold(batch_envelopes(ops.jsonl))
```

## Stage 4 — components with their own triggers

- **Claims memory section.** Build when forum evidence enters commits (SRE
  social rounds are live). Actor/thread/message ids are already present from
  stage 1; add a verification lifecycle only if facts about speech acts prove
  insufficient.
- **Scopes / shared persona memory.** Build when a real cross-context
  knowledge feature is wanted. First render a second read-only context as its
  own labeled prompt section.
- **Retrieval / `resurrect`.** Build when capacity-demote churn appears
  (demoted items are re-added within a few cycles). The journal retains the
  history; this adds an index and one verb.
- **Concurrency (revisions/CAS).** Build when two writers genuinely need one
  context. Until then use the single-writer lock and per-participant contexts.
- **Manual/evaluation rollover.** Build once durable catch-up exists in stage
  1. Forced rollover in designated trials is required to measure whether
  memory can replace provider state. Keep automatic rollover disabled until
  an arm passes the quality gate; the gate is on automation, not measurement.

## Measurement contract

Identical to `structured`: same mutation-record schema, same batch-envelope
journal (envelope gains `progression_id` and the covered range), same
manifest and `memory_context` logging, plus the event ledger as a new
fingerprinted artifact. The point-replay tooling gains exact
session/progression boundaries and loses its approximations. Cross-arm
metrics still require the arm-aware normalization layer; nothing here changes
that honesty clause.

## Deliberately omitted, permanently or until triggered

Model-authored support predicates, proposal ids, dependency edges, and the
repair-ladder/partition/dead-letter machinery (per-operation independence
held up in real runs; all-or-nothing atomicity is reserved for
*code-derived* groups such as "close goal + record outcome"). Mandatory
`Correction` objects on contradiction (a cleanup contradiction usually means
"archive this stale statement"; the journal and archive already hold the
evidence — an active correction item is created only when the corrected
proposition is itself useful, which the existing `contradict`+replacement
already expresses). Near-budget output-token heuristics as rejection signals
(explicit length/incomplete metadata or malformed output means truncation;
token usage near a ceiling is telemetry — reasoning models approach large
ceilings routinely without semantic loss).
