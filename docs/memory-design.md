# Working memory, goals, and compaction

Status: proposed design for experimentation, revised after external review.
This describes one complete target implementation, not a staged plan; nothing
is implemented yet. The control arm ("legacy" below) is derived from the
current behavior in `packages/tty-agent/src/tty_agent/runner.py` and
`packages/tty-agent/src/tty_agent/models.py`.

## Problem

Stateful provider play (a Responses `previous_response_id` chain, or stateful
codex/claude CLI sessions) currently outperforms runs that rely on the
harness's internal memory. That gap is not an argument for abandoning internal
memory — campaigns and rollovers may deliberately start fresh provider
sessions, and cross-session state must then flow through our representation —
it is an argument that the internal representation is weak. Two diseases, one
at each layer:

1. **Session summaries accumulate instead of reconciling.** The compaction
   prompt says "preserve observed facts ... and unresolved goals", which
   encourages accumulation but never requires closing completed goals,
   prioritizing, or separating belief from observation. The flat
   `SessionSummary` (six untyped string lists) cannot distinguish a confirmed
   fact from a hypothesis, a completed task from a blocked one, or a strategy
   history from a current strategy.
2. **Campaign memory accumulates instead of reconciling.** `_merge_memory` in
   `tty_agent/memory.py` is append + dedupe + cap-at-100. `durable_facts` and
   `open_tasks` grow monotonically across epochs; nothing ever closes or
   corrects them, and every epoch's bootstrap eats the whole attic.

On top of the accumulation problem, compaction **silently drops facts** it was
supposed to preserve. The mechanical causes, all in current code:

- Every compaction is a full lossy rewrite: the previous summary is only *in
  the prompt*, so the model must re-emit every fact to keep it, and nothing in
  code carries anything forward. Compaction re-fires every
  `compact_every_steps` / `compact_recent_chars`, so each fact re-rolls the
  survival dice every cycle.
- The prose fallback wipes structure: when a compaction response is not
  parseable JSON, `TextChatAdapter.compact` degrades to
  `{"current_state": <the prose>}` — a summary with N facts becomes one prose
  blob and zero facts, and the empty-summary guard does not fire because
  `current_state` is non-empty.
- Missing keys silently mean zero: `SessionSummary.from_mapping` maps an
  absent `discovered_facts` to `()`, so a partial response wipes sections with
  no signal.
- Output-budget pressure causes preemptive shrinking: compaction historically
  ran under the decision `max_tokens` (default 512). HTTP adapters now expose
  typed initial `compaction_max_tokens` and `memory_max_tokens` budgets, each
  inheriting the decision budget only when omitted, plus per-operation adaptive
  retry ceilings. An explicit hard-truncation signal doubles the applicable
  budget, makes the increase sticky, and is traced; a model that pre-shrinks to
  fit still produces no signal.
- A failed memory commit is silently empty: `_commit_memory` catches
  `ModelError` and returns `MemoryPatch()` with no retry and no journal
  event, so a whole session's learning can vanish without trace.
- Loss propagates: the end-of-run memory commit sees only summary + recent
  steps, so a fact dropped mid-run never reaches `campaign.json` either.

Finally, **provider lifecycles are not uniform**, which both causes a live
problem and would confound any experiment run on top of them:

- The Responses adapter isolates utility traffic (`store: false`, no
  `previous_response_id` for compaction/memory calls) and starts a fresh chain
  on every bootstrap.
- The stateful codex and claude CLI adapters do neither: they retain their
  session id across `runner.run()` calls, and because they do not override
  `compaction_chat`/`memory_chat`, **compaction and memory-commit prompts are
  appended to the resumed gameplay session today** — control-plane utility
  traffic inside the agent's own conversation. Clearing
  `decision_prompts_sent` re-sends a bootstrap but does not start a fresh CLI
  session, so "rollover" would be a no-op for them.
- Campaign-social handling is divergent by accident: a CLI adapter's social
  `decide` resumes the gameplay session, while the Responses adapter treats
  social prompts as stateless one-shots. Neither is a bug in isolation, but
  the divergence is resolved deliberately below: social is agent-facing
  *activity*, not utility, and joins the agent's context on every provider.

## Principles

1. **Persistence is the code's job; the model proposes changes.** Reconciliation
   emits operations against the previous state; anything unmentioned persists.
   Forgetting to mention a fact keeps it instead of losing it.
2. **There is no unqualified drop.** Every exit from working memory is typed,
   reasoned, journaled, and code-validated. Capacity pressure demotes to an
   archive tier; it never deletes.
3. **Everything is measurable.** Every mutation is journaled as an accepted
   operation record (not a reconstructed diff), so memory policies can be
   compared empirically. What cannot be mechanically prevented — a model
   omitting new facts — must at least be measurable.
4. **Experimentation is a first-class requirement.** Representation, prompts,
   and parsing are pluggable per persona-store policy; rendering and cadence
   are pluggable per activity view. The control arm is **normalized legacy**:
   the current representation and prompts, running under the same lifecycle
   and failure normalization as every other arm — bit-for-bit preservation is
   impossible once live bugs are fixed, and un-normalized lifecycles would
   confound the comparison anyway. Every arm records its schema, mutation,
   prompt, and view fingerprints, and uses an isolated memory root, so prior
   runs cannot contaminate later arms.
5. **Isolate by purpose and authority, not reflexively by channel.** Gameplay,
   forum rounds, and in-game messaging may share one decision conversation
   inside a configured authority domain -- one persona on one BBS -- while the
   persona memory remains common across every domain. Only control-plane
   traffic is always isolated: compaction/commit utility calls and hidden
   evaluation probes.
6. **Provider lifecycle is normalized before policies are compared.** Utility
   and evaluation calls never touch the decision-session state on any
   provider, and every stateful provider exposes the same reset semantics.
   Without this the matrix compares lifecycle semantics, not memory policies.
7. **In stateful modes, internal memory is the parachute, not the pilot.**
   Provider-side state is the primary context; internal memory only has to be
   good at chain-rollover boundaries and the post-social campaign commit, so
   it can run on a sparser cadence there.
8. **Activity happens once; recovery is bookkeeping.** Durable agent activity
   is never re-executed. The scheduler owns opaque progression boundaries — a
   progression may be a BBS session, player turn, social round, epoch, whole
   standalone run, or an immediately committed action. Its effects are pending
   until the scheduler's durability authority publishes a committed or
   abandoned outcome, mirrored append-only into the persona store. A crash
   yields a progression that ended badly, not a do-over: completed work counts,
   torn external state is repaired to the last consistent snapshot,
   salvageable memory is carried forward without loss, and execution moves
   forward even when reconciliation must be deferred. The operation journal
   never rolls back; its effective view resolves causal support through those
   progression outcomes.

## Provider lifecycle normalization

The foundation the rest of the design assumes — and a standalone fix for the
live utility-pollution issue above. Stateful adapters (Responses, codex CLI,
claude CLI) gain a small uniform capability surface:

- **An explicit call-context API.** Adapters currently infer lifecycle from
  which method happened to be called; instead, every model call carries a
  typed context:

  ```python
  CallPurpose = Literal["activity", "utility", "evaluation"]
  ActivityChannel = Literal["terminal", "forum", "message"]

  @dataclass(frozen=True)
  class ConversationKey:
      persona_id: str
      authority_domain: str

  @dataclass(frozen=True)
  class CallContext:
      purpose: CallPurpose
      channel: ActivityChannel | None
      scope: ScopeKey | None
      conversation_key: ConversationKey | None
  ```

  `channel`, `scope`, and `conversation_key` are required for activity. A
  utility or evaluation call may carry tracing tags, but they never opt it into
  decision state.
  The rules are uniform across providers: **activity participates in
  provider-side state** (when the adapter is stateful) regardless of channel;
  **utility and evaluation never do**. Bootstrap/delta stage belongs to the
  shared activity conversation, not to the terminal runner: the first activity
  call after a reset bootstraps, and every later activity call is a
  continuation even when the channel changes. A forum call has
  channel-explicit framing and its own restricted action policy so
  terminal-action JSON and forum messages cannot cross-contaminate. Context
  tags land in the response trace so journals can separate channels and scopes.

  Provider state is shared only inside one runtime-owned `ConversationKey`.
  The default BBS profile may use `environment_id` as its authority domain so
  doors, forums, and messages retain the human-like continuity intended here;
  an evaluator may instead key by campaign or another stricter boundary. The
  persona memory store remains common across every key. Sharing a key is an
  explicit threat-model decision: raw turns can influence later channels and
  cannot be hidden by a narrower `MemoryView`. It is allowed only when the
  scopes have the same authority boundary, actions are sandboxed, no secret or
  capability boundary separates them, and cross-channel influence is intended.
  Channel framing mitigates prompt confusion; it is not security isolation.
  Stateful APIs may retain every activity turn server-side (`store: true` or
  its provider equivalent); that retention mode is explicit configuration and
  run metadata, with stateless mode available for stricter data policies.
- **A runner-facing stage handshake.** Prompt stage cannot be hidden entirely
  inside the adapter because bootstrap and delta prompts have different
  structures. Every adapter therefore exposes:

  ```python
  PromptStage = Literal["full", "bootstrap", "delta"]

  class ModelAdapter(Protocol):
      def activity_prompt_stage(self, context: CallContext) -> PromptStage: ...
  ```

  A stateless adapter returns `full`; a stateful adapter returns `bootstrap`
  until it has usable provider conversation state and `delta` afterwards. The
  runner queries immediately before building every activity prompt, includes
  that stage in the prompt, and serializes the query/build/call sequence per
  adapter. The adapter validates the supplied stage and advances to `delta`
  once the provider call establishes or advances conversation state — even if
  the returned action later fails action-schema validation. A **determinate**
  failure before provider state is established leaves the stage unchanged. An
  **indeterminate** outcome — a timeout or transport error after the request
  may have reached the provider — is treated as state loss: the adapter
  atomically clears its state, resets its stage, and the next activity call
  bootstraps. Guessing continuation against a maybe-advanced conversation is
  worse than paying for one bootstrap. If prior state is explicitly rejected
  by the provider, the adapter likewise clears it and raises
  `ModelStateError`; the runner records `requires_bootstrap` for the
  trace/retry decision and re-queries the adapter before rebuilding the next
  prompt. The runner keeps no competing prompt-count stage.

  Live call failure also has an explicit scheduler outcome. A paid call either
  belongs to an already-open durability progression or causes a call-level
  progression to be durably started; it never creates a second implicit
  progression. A timeout or determinate failure that yields no executable
  action closes a newly created call-level progression as `abandoned` through
  its durability authority, records `no_action` or `indeterminate_call`, and
  retries under a new id. If the call
  belongs to a wider session progression, the call-attempt trace is closed and
  the scheduler either continues that progression or abandons and repairs it
  when external state is indeterminate. A received provider turn that fails
  action-schema validation is appended as `activity_no_action`: its experience
  is durable, it creates no effect-dependent evidence, and the scheduler
  commits a no-effect outcome at its normal boundary. No live error path defers
  resolution of a newly started progression until process recovery.

  Chain loss does **not** immediately use an ordinary bootstrap. Before the
  next activity call, the runner attempts forced catch-up reconciliation over
  contiguous pending-event prefixes for the affected conversation key until
  they reach the durable log head. If a utility failure prevents catch-up, the
  bootstrap carries the current memory view plus the largest contiguous
  pending prefix that fits a dedicated recovery-bootstrap budget, rather than
  only the ordinary last-K decision window. That bootstrap is traced as
  `degraded_state_recovery`; pending events remain queued and reconciliation
  keeps retrying. Thus planned rollover and unplanned state loss share the
  invariant that a fresh chain receives every **not-yet-presented** event
  either through successful reconciliation or directly as bounded pending
  context. A bounded unresolved-history section additionally cites recent
  `no_memory_change` and dead-letter records so semantically omitted, already
  presented events are discoverable without pretending they remain pending;
  it obeys the same authority and scope permissions as every raw window.
- **Isolated utility execution.** `compaction_chat`/`memory_chat` (purpose
  `utility`) must never resume the decision session. For the CLI adapters
  this means running utility calls stateless (`--ephemeral` /
  `--no-session-persistence`) regardless of the adapter's decision-session
  mode. The Responses adapter already conforms for utility; under the
  call-context rules its *social* handling flips — social is `activity`, so
  a stateful Responses adapter appends forum turns to the chain instead of
  treating them as stateless one-shots.
- **`reset_decision_state(conversation_key)`.** Discard provider-side
  conversation state for one key so
  the next bootstrap genuinely starts fresh: clear `response_id` for
  Responses (already the effect of a new bootstrap), clear `session_id` (and
  the session file) for codex/claude, and reset the adapter's shared activity
  stage. This is the primitive for rollover and for **configured** lifecycle
  boundaries. A campaign may deliberately reset at each epoch for fairness or
  cost control, but an epoch or channel change does not imply a reset by
  itself; a fresh chain always bootstraps from the persona's current memory
  view. **Abnormal session end and campaign recovery always reset**: provider
  conversations are never assumed to survive a crash, and there is no
  provider-side rollback to match anything to. Losing campaign A's state does
  not clear an unrelated conversation key.
- **Chain-usage reporting where available.** Responses `usage` is already
  captured per call; CLI adapters report what their result envelopes expose.
  Absence is allowed and recorded, since rollover triggers must not assume
  token counts exist.

## Architecture: the `MemoryPolicy` seam

Same move that worked for scoring (`EvaluationProfile`): the representation,
prompts, and parsing become pluggable, while `ActivityRunner` keeps the
mechanics — cadence triggers, retry backoff, `model_utility` events, bounded
rendering. The policy belongs to the **persona store**, not to an individual
door or forum profile: one store cannot safely alternate schemas when the
persona changes activities. An `ActivityProfile` instead supplies a
`MemoryViewProfile` (active scope, parent/related-scope visibility and ranking,
prompt budget, reconciliation cadence, and raw-window size).

Parsing today lives inside `TextChatAdapter.compact()`/`commit_memory()`,
which return concrete `SessionSummary`/`MemoryPatch` objects, so a policy
cannot own parsing without an adapter-layer change. Adapters therefore gain a
generic utility call returning raw output plus trace metadata, and all
parsing moves into the policy:

```python
@dataclass(frozen=True)
class UtilityResult:
    raw_text: str              # unfiltered, mirrors last_response
    filtered_text: str         # after output filters, mirrors last_parsed_response
    reasoning: str             # provider-reported reasoning, for journals only
    finish_reason: str | None  # typed: cursor advancement legally depends on it
    output_tokens: int | None  # typed: compared against compaction_max_tokens
    usage: dict[str, Any] | None
    provider_metadata: dict[str, Any] | None
    response_id: str | None

class PersonaMemoryState(Protocol):
    schema_version: int
    def is_empty(self) -> bool: ...
    def to_dict(self) -> dict[str, Any]: ...

class MemoryView(Protocol):
    schema_version: int
    source_revisions: tuple[ScopeRevision, ...]
    def is_empty(self) -> bool: ...
    def to_prompt_dict(self) -> dict[str, Any]: ...

@dataclass(frozen=True)
class ProposedBatch:
    proposed_ops: tuple[Op, ...]                 # parsed from the model's response
    read_revisions: tuple[ScopeRevision, ...]    # every scope in the rendered view
    write_scopes: tuple[ScopeKey, ...]           # code-derived from proposed_ops
    presented_prefixes: tuple[PresentedPrefix, ...]  # code-derived cursor CAS envelope

class MemoryPolicy(Protocol):
    name: str
    schema_version: int
    def store_schema_fingerprint(self) -> str: ...    # serialization compatibility
    def mutation_policy_fingerprint(self) -> str: ... # caps + code-owned semantics
    def prompt_fingerprint(self) -> str: ...          # per-run metadata
    def empty_store(self) -> PersonaMemoryState: ...
    def render_view(self, state, context, view_profile) -> MemoryView: ...
    def reconcile_prompt(self, previous_view, events_text, observation, objective, context) -> CompactionPrompt: ...
    def parse_reconciliation(self, result: UtilityResult, previous_view) -> tuple[Op, ...]: ...
    def commit_prompt(
        self,
        state,
        session_view,
        events_text,
        observation,
        objective,
        context,
    ) -> MemoryCommitPrompt: ...
    def parse_commit(self, result: UtilityResult, state) -> tuple[Op, ...]: ...
    def validate_op(self, op, state, context) -> str | None: ...

class PersonaMemoryStore(Protocol):
    def load(self) -> PersonaMemoryState: ...
    def start_progression(self, started: ProgressionStarted) -> None: ...
    def append_events(self, events: tuple[ActivityEventDraft, ...]) -> tuple[ActivityEvent, ...]: ...
    def finish_progression(self, disposition: ProgressionDisposition) -> None: ...
    def pending_scopes(self, conversation_key: ConversationKey) -> tuple[ScopeKey, ...]: ...
    def pending_events(
        self,
        conversation_key: ConversationKey,
        scope: ScopeKey,
        *,
        max_events: int,
        max_chars: int,
    ) -> tuple[ActivityEvent, ...]: ...
    def search_history(
        self,
        query: HistoryQuery,
        context: CallContext,
        view_profile: MemoryViewProfile,
    ) -> tuple[HistoryHit, ...]: ...
    def apply_batch(self, proposal: ProposedBatch, policy, context) -> ReconciliationResult: ...
```

The store owns the durable per-persona `ActivityEvent` log and the
reconciliation cursors: cursors are positions in that log, not fields of the
memory state, and are keyed by `(ConversationKey, ScopeKey)`. `pending_events`
returns an oldest-first bounded contiguous prefix of that key's cursor→head
queue. The store issues sequence numbers while appending drafts and returns the
immutable persisted events; callers never race to pre-allocate persona-wide
sequences. `event_id` is an idempotency key, so replaying an append after a
crash returns the existing event rather than duplicating it.

Policies parse **operations only**; the runner composes `ProposedBatch` from
the parsed operations plus the envelope it already owns: the complete rendered
view's `source_revisions`, code-derived writable scopes, and cursors for the
contiguous pending prefixes it actually placed in the prompt. This keeps the
parse signature honest: nothing in the model's response could populate the
envelope. `MemoryView.source_revisions` always includes the active scope and
every runtime-permitted write scope, even when that scope currently renders no
items, as well as every related scope whose records were rendered.

Cadence has one runtime owner. `MemoryPolicy` may provide default cadence
values while an `ActivityProfile` is resolved, but the resulting
`MemoryViewProfile` is authoritative and `ActivityRunner` evaluates its step,
event-size, and provider-usage thresholds. Policies do not get a second veto.
Rollover, explicit flush, and the post-social campaign commit are forced
reconciliations and override ordinary cadence.

`PersonaMemoryStore.apply_batch` takes the store lock, checks the full read
revision vector and writable scopes, validates every operation, issues ids,
enforces caps, and journals the batch atomically before returning. Here
"applied" means durably accepted into the operation history, not necessarily
visible in the committed projection. Pending operations feed a non-destructive
provisional overlay; only operations whose support predicates are effective
feed committed memory. The journal never removes or rewrites an accepted
record, and abandoned operations are excluded by projection rather than
rolled back.
Reconciliation returns more than the new state, so journaling and metrics
have a reliable source instead of reconstructing mutations afterwards:

The append-only operation/event journals are the memory crash-recovery source
of truth. One memory transaction record contains accepted operation records,
resulting revisions, presented prefixes and resulting cursors, progression
metadata, and fingerprints; code fsyncs that record before replacing derived
state/index snapshots. Event appends follow the same write-ahead rule. A
progression-disposition mirror may
lag its external commit authority across a crash and is reconciled from that
authority before effective projection; a store-local authority writes its
ledger outcome first. On open, the store truncates a torn final record, replays
committed records past the latest snapshot, and repairs disposition indexes.
Thus operation state, cursor advancement, and the audit record cannot disagree
after a crash, while external effect validity still has exactly one authority.

`ScopeRevision` is the conservative read version for everything that can alter
a rendered scope, not only item writes. Appending an event or a progression
disposition advances every affected scope's revision and invalidates its view
cache. Consequently a disposition or concurrent observation cannot change the
effective evidence behind a prompt while leaving its `read_revisions` valid.

```python
@dataclass(frozen=True)
class ReconciliationResult:
    proposed_ops: tuple[Op, ...]      # parsed or policy-derived operations
    journaled_ops: tuple[OperationRecord, ...]  # durable accepted records
    rejected_ops: tuple[RejectedOp, ...]  # with rejection reasons
    read_revisions: tuple[ScopeRevision, ...]  # validated rendered-view vector
    write_scopes: tuple[ScopeKey, ...]
    resulting_revisions: tuple[ScopeRevision, ...]
    presented_prefixes: tuple[PresentedPrefix, ...]  # validated cursor CAS inputs
    resulting_cursors: tuple[ScopeCursor, ...] # mechanical durable outcome
    batch_errors: tuple[str, ...]     # envelope/revision-level failures
```

The store journal is authoritative for memory history, while the configured
durability ledger is authoritative for external-effect validity.
`PersonaMemoryState` is the policy-owned materialization at a recorded
journal/authority revision, while `MemoryView` is an immutable bounded
effective projection for one prompt.
`ReconciliationResult` therefore returns resulting revisions, not an
ambiguously named memory object. Callers load or refresh the materialization
and re-render a view after application.

Validation is two-pass. The first implementation validates **every revision in
the rendered view**, not only scopes being written: an operation based on a
related-scope claim must not commit after that claim changes. This conservative
read set provides serializable decisions. A later optimization may let
operations declare versioned `depends_on` references and derive a smaller read
set, but write atomicity alone is insufficient. If any read revision fails or
any model-proposed operation is rejected, no operation in that proposal
is accepted and the cursors do not advance. The result records batch errors or
rejected operations with `journaled_ops=()`. A valid proposal journals all of
its operations atomically and may gain deterministic code-generated operations
such as a capacity demotion. Those appear as full `OperationRecord`s with
`origin="system"` and causal links to the operation that required them.

The conservative read vector can otherwise starve under a steady stream of
concurrent event appends. The first implementation therefore serializes
activity-event append and reconciliation from view capture through apply for
one persona. That is intentionally coarse but covers related scopes as well as
the active conversation key. A later MVCC implementation may use fixed
snapshots and split item revisions from event-head validation; it must preserve
the same serializable result.

The campaign commit parses into the same operations and uses the same store
transaction, with the resulting `ReconciliationResult` tagged as a commit in
the journal. It never returns the legacy `MemoryPatch` directly, so campaign
mutations are revision-checked and journaled identically to session
mutations. The `legacy` parser constructs two policy-internal operations that
the model never emits: `LegacyReplaceSummary` for its full-rewrite compaction
response and `LegacyMergePatch` for its campaign commit. Only the legacy policy
validator accepts them, but both still pass through `apply_batch` for locking,
revision checks, before/after journaling, and cursor handling. The revision
vector recorded by the store is what epoch manifests record. Legacy summary
replacements also retain their progression-supported version chain: the
effective legacy view selects the newest committed replacement and falls back
to its predecessor when a newer progression is pending or abandoned. This is
coarse-grained, but gives interrupted campaign comparisons the same recovery
semantics as the reconciling policy.

Policies are selected per agent/persona runtime, exposed as
`--memory-policy` and via registry participant/model config. The policy name
and **store-schema fingerprint** are persisted with the store; opening an
incompatible schema requires an explicit migration or a new root. A separate
**mutation-policy fingerprint** records every state-affecting code rule — caps,
provisional caps, support/dependency defaults, causal-projection algorithm,
dedupe and source normalization, lifecycle horizons, goal transitions, cursor
advancement, archive behavior, automatic-demotion order, and canonical-key
handling.
Changing it need not always force migration, but every transaction and run
records it and incompatible changes require explicit revalidation. Prompt text
has its own fingerprint and never forces migration. Activity profiles add the
view fingerprint. Three initial policies:

- **`legacy`** — the current `SessionSummary` representation, prompts, and
  merge semantics, extracted verbatim (tests pin the representation) but
  running under the normalized lifecycle and failure handling like every
  other arm ("normalized legacy", not bit-for-bit).
  It remains intentionally flat and exists as a compatibility/control policy,
  not as the recommended multi-activity persona store.
- **`reconciling`** — the schema and operations contract below.
- **`raw-window-only`** — no internal summary at all; the memory commit is
  built from `_bounded_recent_events_text` alone. (Named for what it *is* —
  the arm also runs in stateless modes, where there is no provider state to
  lean on.) Today compaction cannot actually be disabled
  (`compact_every_steps=0` leaves the char trigger live); this arm makes
  "off" first-class and is the null hypothesis that an internal summary
  helps at all.

## Persona memory schema v2

Every item carries an **immutable, store-wide, code-issued id** (`m1`, `m2`,
…). Positional indices dangle across mutations and content hashes break on
revision, so neither is acceptable. The next-id counter is persisted in the
persona store.

Scope, progression, identity, and provenance are concrete types, not
conventions embedded in a step number or free-form string. A scope identifies
the item's home partition; source references identify the activity evidence
from which it was learned. One item may cite several sources, including
evidence from another scope.

Code normalizes terminal steps, forum activity, and direct messages into
**one ordered `ActivityEvent` log per persona**: the store issues a single
monotonic `sequence` across all channels and scopes, so the merged stream has
a total order and reconciliation needs no terminal-only special case.
Per-scope cursors are positions in that total order, filtered by scope — two
concurrent activities on one persona each track their own coverage without a
global cursor marking the other's events presented. The event and progression
logs are append-only: cursor advancement changes what reconciliation
re-presents, never what exists.

`ScopeKey`s are constructed by the runtime, not invented by the model.
`environment_id` is the stable root (a BBS instance or standalone TTY world),
and the optional fields identify the most-specific activity beneath it.
Scope ancestry is positional: an ancestor drops the most-specific populated
field, so `(env, campaign, game, participant)` → `(env, campaign, game)` →
`(env, campaign)` → `(env)`. `participant_id` is the persona's **own**
account identity in that context, from the orchestrator's stable mapping —
never a display name, door slot, or model-chosen handle. Other actors are
runtime-issued metadata on events and records, not scopes or names extracted
from prose. That makes claim validation, related-scope retrieval, and identity
continuity mechanical across display-name changes.

```python
@dataclass(frozen=True)
class ScopeKey:
    environment_id: str         # BBS instance or standalone TTY world
    campaign_id: str | None = None
    game_id: str | None = None
    participant_id: str | None = None

@dataclass(frozen=True)
class ScopeRevision:
    scope: ScopeKey
    revision: int

@dataclass(frozen=True)
class ScopeCursor:
    conversation_key: ConversationKey
    scope: ScopeKey
    presented_through_sequence: int  # persona-log sequence, scope-filtered

@dataclass(frozen=True)
class PresentedPrefix:
    conversation_key: ConversationKey
    scope: ScopeKey
    from_exclusive: int
    through_inclusive: int

@dataclass(frozen=True)
class SourceRef:
    conversation_key: ConversationKey
    scope: ScopeKey
    channel: ActivityChannel
    sequence: int               # store-issued, monotonic per persona (total order)
    event_id: str               # runtime-issued idempotency key
    epoch: int | None = None

ActivityEventKind = Literal[
    "terminal_step", "forum_read", "forum_post",
    "message_read", "message_sent", "activity_no_action",
]

@dataclass(frozen=True)
class ProgressionStarted:
    progression_id: str
    conversation_key: ConversationKey | None
    durability_domain: str       # scheduler-owned recovery/isolation domain
    base_generation: str         # authority generation active at start

@dataclass(frozen=True)
class ProgressionDisposition:
    progression_id: str         # opaque scheduler-owned identity
    outcome: Literal["committed", "abandoned"]
    authority_ref: str           # must resolve through the domain's commit authority
    reason: str = ""

@dataclass(frozen=True)
class AuthorityHead:
    durability_domain: str
    authority_ref: str           # ledger revision used for this materialization

@dataclass(frozen=True)
class ActivityEventDraft:       # submitted to the store before sequence allocation
    scope: ScopeKey
    channel: ActivityChannel
    event_id: str
    progression_id: str
    conversation_key: ConversationKey
    kind: ActivityEventKind
    observation: str = ""
    action: str = ""
    intent: str = ""
    warnings: tuple[str, ...] = ()
    actor_id: str | None = None
    related_actor_ids: tuple[str, ...] = ()
    thread_id: str | None = None
    message_id: str | None = None

@dataclass(frozen=True)
class ActivityEvent:
    ref: SourceRef
    progression_id: str
    kind: ActivityEventKind
    observation: str = ""
    action: str = ""
    intent: str = ""
    warnings: tuple[str, ...] = ()
    actor_id: str | None = None         # stable runtime identity, never model-authored
    related_actor_ids: tuple[str, ...] = ()
    thread_id: str | None = None
    message_id: str | None = None

EvidenceRole = Literal[
    "supports_current", "prior_state", "assertion",
    "verification", "contradiction", "outcome",
]
EvidenceDependency = Literal[
    "experience_occurred",
    "external_effect_committed",
]

@dataclass(frozen=True)
class EvidenceRef:
    source: SourceRef
    role: EvidenceRole
    dependency: EvidenceDependency

@dataclass(frozen=True)
class SupportClause:             # every member is required (logical AND)
    all_of: tuple[EvidenceRef, ...]

@dataclass(frozen=True)
class SupportPredicate:          # any clause is sufficient (logical OR)
    any_of: tuple[SupportClause, ...]

OperationOrigin = Literal["model", "policy", "system"]
SystemReason = Literal[
    "capacity", "hypothesis_stale", "goal_pruned",
    "state_expired", "salvage", "progression_abandoned",
]

@dataclass(frozen=True)
class OperationRecord:
    operation_id: str            # store-issued and durable
    batch_id: str                # store-issued transaction identity
    scope_revision: ScopeRevision
    origin: OperationOrigin
    operation: Op
    support: SupportPredicate
    required_progression_ids: tuple[str, ...]  # derived from effect-dependent atoms
    depends_on_operation_ids: tuple[str, ...]
    caused_by_operation_ids: tuple[str, ...]   # mandatory for dependent system ops
    system_reason: SystemReason | None = None

@dataclass(frozen=True)
class ItemVersionRef:
    id: str
    updated_revision: int

@dataclass(frozen=True)
class HistoryQuery:
    text: str
    scopes: tuple[ScopeKey, ...]
    subject_ids: tuple[str, ...] = ()
    limit: int = 10

@dataclass(frozen=True)
class HistoryHit:
    kind: Literal["event", "item_version", "archive"]
    reference: SourceRef | ItemVersionRef
    summary: str

@dataclass(frozen=True)
class ItemMeta:
    id: str
    scope: ScopeKey             # partition that owns this record
    support: SupportPredicate
    last_updated_operation_id: str
    updated_revision: int       # comparable within the owning scope
    subject_ids: tuple[str, ...] = ()
    canonical_key: str | None = None
    derived_from: tuple[ItemVersionRef, ...] = ()

@dataclass(frozen=True)
class StateEntry:               # independently addressable mutable current state
    meta: ItemMeta
    key: str
    value: str
    stale_after_events: int | None = None

@dataclass(frozen=True)
class Goal:
    meta: ItemMeta
    text: str
    status: Literal["open", "blocked", "done", "abandoned"]
    priority: int
    context: str = ""          # for blocked: what blocks it, and where
    outcome_refs: tuple[ItemVersionRef | SourceRef, ...] = ()

@dataclass(frozen=True)
class Fact:
    meta: ItemMeta
    text: str
    temporal_kind: Literal["invariant", "historical"] = "invariant"

@dataclass(frozen=True)
class Hypothesis:
    meta: ItemMeta
    text: str
    untouched_cycles: int = 0  # drives code-enforced test-or-demote

@dataclass(frozen=True)
class Correction:
    meta: ItemMeta
    text: str                  # corrected proposition plus concise reason
    corrects: ItemVersionRef

@dataclass(frozen=True)
class Claim:                   # another actor's assertion; never a fact on arrival
    meta: ItemMeta
    speaker_actor_id: str      # validated from source-event actor metadata
    text: str

@dataclass(frozen=True)
class ArchivedItem:            # archive holds every exited item kind
    kind: Literal["state", "goal", "fact", "hypothesis", "claim", "correction"]
    item: StateEntry | Goal | Fact | Hypothesis | Claim | Correction
    exit: Literal[
        "capacity", "hypothesis_stale", "goal_closed",
        "state_expired", "promoted", "contradicted",
        "progression_abandoned",
    ]
    exited_by_operation_id: str

@dataclass(frozen=True)
class ReconciledPersonaState:  # implements PersonaMemoryState; schema_version = 2
    schema_version: int
    next_id: int
    operation_journal_sequence: int
    authority_heads: tuple[AuthorityHead, ...]
    revisions: tuple[ScopeRevision, ...]
    state_entries: tuple[StateEntry, ...]
    goals: tuple[Goal, ...]
    facts: tuple[Fact, ...]
    hypotheses: tuple[Hypothesis, ...]
    claims: tuple[Claim, ...]
    corrections: tuple[Correction, ...]
    archive: tuple[ArchivedItem, ...]  # on disk, never in prompts
```

Hard caps (state entries, open goals ~3–5, facts, hypotheses, claims, and
corrections) are part of the persona policy configuration, apply **per scope
and section**, and are enforced when operations are accepted and projected,
not merely requested in the prompt.
Provisional sections have separate caps so pending work cannot consume or evict
committed capacity; both cap sets belong to the mutation-policy fingerprint.
Every text field, support clause, support alternative list, subject list,
lineage list, actor list, and canonical key also has a code-validated
character/item bound. A single string or provenance structure can never become
an unbounded replacement attic.
Prompt budgets and scope-ranking weights belong to `MemoryViewProfile` and do
not mutate stored data. Dedupe is scope-aware. `is_empty`/`to_dict` keep
`ActivityResult` (whose `session_summary` carries the `PersonaMemoryState`),
match results, and trace tooling working across schema versions.

`StateEntry` replaces the former `ScopedState` summary blobs. Keys are
independently addressable and normalized by code; profiles may reserve keys
such as `location` or `last_error`. Current-state and tactical-summary prose is
an ephemeral deterministic formatting/ranking of active state, goals, facts,
and hypotheses, never a durable field that can silently erase omitted facts.
Historical events belong
in `Fact` or the event log; mutable keyed values belong in `StateEntry`.

Rendering constructs a bounded `MemoryView`: the exact active scope first,
then its campaign/environment ancestors and relevance-selected related scopes
(including records whose `subject_ids` match actors in the current activity).
Every rendered record retains a visible scope and channel label; forum text is
quoted as another
participant's data, never spliced into control instructions. The initial
implementation never retrieves across `environment_id` unless the view
explicitly opts in, and performs no automatic cross-scope promotion. New
records belong to the most-specific active scope, while a later explicit copy
operation may promote selected knowledge to an ancestor without moving or
rewriting its source record. Related-scope visibility is additionally bounded
by the current `ConversationKey`'s authority policy; sharing one persona store
does not grant raw-event access across authority domains.

Progression validity is an append-only overlay, not a mutable event Boolean.
The scheduler fsyncs `ProgressionStarted` before paid activity, assigns the
opaque id to every resulting event, and later mirrors one committed or
abandoned `ProgressionDisposition` from the configured durability authority.
Absence of an authority-resolved terminal outcome means pending. Different
concurrent activities use different ids, so a global event sequence never
invalidates unrelated work.

The memory layer assigns no meaning to the id and stores no world/forum/message
commit domain. A campaign scheduler may group an entire play epoch; another
runtime may use one session, turn, or social round; a non-scheduled runner may
commit one whole run or each successfully executed action. If effects become
durable independently, the scheduler uses separate progression ids.

### Operation-causal effective projection

Progression status qualifies an operation, not merely its current item or flat
evidence list. Every accepted operation becomes a durable `OperationRecord`.
The store maps call-local proposal ids to store-issued operation ids and keeps
the semantic dependency edges after transaction planning; dependency cycles
are rejected. Dependencies may name an earlier journal record or another
operation in the same batch; same-batch records are folded in a deterministic
topological order and cannot name a future batch. `required_progression_ids`
is a code-derived index over the operation's support and is never model-authored.

Support is an OR of AND clauses. Every `SupportClause` is one sufficient set
whose evidence members are all required; any satisfied clause is sufficient.
Code evaluates each evidence atom according to its declared dependency:

- `experience_occurred` is satisfied when the immutable event exists. It
  remains true when the containing progression is abandoned: an attempted
  command or observed screen is still confirmed history.
- `external_effect_committed` is committed only when the event's progression
  has a committed outcome resolved by its durability authority, pending while
  that outcome is unresolved, and false when it is authoritatively abandoned.
  Current state, delivered-message claims, and outcome-based goal transitions
  use this dependency.

The policy derives or validates dependency by operation field; the model
cannot preserve an uncommitted world effect merely by labeling it experience.
Historical attempts and observed output default to `experience_occurred`.
`StateEntry` writes, delivery/visibility claims, and outcome-driven goal status
default to `external_effect_committed`. `Fact.temporal_kind` describes the
proposition's time shape, while evidence dependency describes what makes it
true; neither substitutes for the other.

An empty support clause is never legal model output. Policy/system operations
may use one only after code validates and serializes their deterministic
precondition (for example a staleness threshold); causal system operations
still name the operation ids that caused them.

A support predicate is **committed** when at least one clause is fully
satisfied without pending atoms, **pending** when no committed clause exists
but at least one could become satisfied, and **ineffective** otherwise. An
operation enters the committed projection only when its support is committed
and every `depends_on_operation_id` and `caused_by_operation_id` is itself
effective. It enters the pending overlay when the same graph is potentially
satisfiable but not yet committed. An abandoned prerequisite suppresses every
dependent operation mechanically. A model-proposed operation may use the full
support form, though the first parser should reject needlessly complex clauses
and require all cited uncommitted effect evidence to belong to one progression
unless a declared dependency genuinely needs more.

Rendering is therefore two related folds over the immutable operation journal:

1. **Committed projection:** fold only effective committed operations in
   journal order. This is authoritative memory.
2. **Pending overlay:** render potentially effective operations alongside that
   projection without destructively changing it. Pending additions use a
   separately bounded provisional section; a pending exit, contradiction,
   promotion, rewrite, goal closure, or prune leaves its committed predecessor
   visible and labels the prospective result as provisional.

Every code-generated consequence is an operation with a causal edge. A
capacity demotion names the add or mutation that required it; goal pruning,
expiry, and exit-and-birth successors likewise retain their causes. A system
operation cannot become effective before its cause and becomes ineffective if
that cause is abandoned. Thus a pending addition can never evict a committed
fact, and an abandoned contradiction automatically reveals the last effective
fact version without a compensating model call. When the progression commits,
the add and its deterministic demotion enter the committed fold together.

The on-disk active/archive collections are materialized caches of the committed
projection plus pending-overlay indexes, not independent truth. Replay of the
journal and resolved dispositions must reproduce them exactly. Cleanup may
append explicit operations for storage or prompt hygiene, but correctness
never depends on cleanup and no journal entry is rewritten. This is a
forward-only causal projection, not rollback.

## Reconciliation: an operations contract, not a rewrite

The reconciliation prompt presents the previous memory view (items with their
ids and scopes) plus the unsummarized `ActivityEvent`s, and asks for
**operations only**. The model's response carries no envelope fields: a model
cannot attest to what it failed to notice, so nothing durable is ever gated on
its self-report. Every envelope field is code-supplied:

- `read_revisions` — the complete rendered view's `source_revisions`. Every
  revision must still match when the persona-store lock is acquired or the
  whole batch is rejected.
- `write_scopes` — derived from operation targets and checked against the
  runtime-owned writable set. Every write scope must also appear in
  `read_revisions`; multi-scope writes enter the journal atomically.
- `presented_prefixes` — one code-built `PresentedPrefix` per
  `(ConversationKey, ScopeKey)`, recording both `from_exclusive` and
  `through_inclusive` for the bounded pending prefix actually placed in the
  prompt. The model never echoes cursors, so there is nothing to validate or
  hallucinate.

For each cursor key, new events in a reconciliation prompt are always an
oldest-first **contiguous prefix** after its cursor: no event in that scope and
conversation may be skipped before `through_inclusive`. Already-presented
overlap is a separately labeled context section and never changes the
boundary. Token or relevance selection may shorten a prefix but may not punch
holes in it.

Cursor advancement is a compare-and-swap under the persona-store lock. The
store verifies that the current cursor still equals `from_exclusive`, that the
immutable log's exact authority-filtered, scope-filtered prefix ends at
`through_inclusive`, and that no event in between was omitted. Applying,
accepting no memory change, or dead-lettering the prefix advances the cursor
and the owning scope revision even when no item changed. A stale no-op proposal
therefore conflicts instead of journaling a second advancement.

Cursor advancement is **mechanical**. Cursors advance exactly when the
response parses, every operation validates and the batch is accepted, the
`ReconciliationResult` has been durably journaled (fsync, the same discipline
as the campaign journal), and no truncation signal fired
(`UtilityResult.finish_reason` length/incomplete, or `output_tokens` within a
configured margin of `compaction_max_tokens`). Any failed condition routes to
the normal retry/backoff path and leaves the events in the pending queue;
there is no model-controlled outcome between "accepted and advanced" and
"failed and retried". One mechanical heuristic guards the lazy-empty case:
zero proposed operations against a non-trivial event window triggers a single
retry, then advances anyway with an explicit `no_memory_change` journal marker.
`through_inclusive` means only what the utility model was given, never what it
captured. The store separately records extraction-attempt counts per event,
operation source references, and a `no_memory_change` outcome when advancing
without a durable mutation; raw events remain searchable.

Two mechanisms stand in for the attestation this design deliberately omits:

- **Overlap, not attestation.** The next reconciliation re-presents the last
  few presented events as already-presented context — a mechanical second
  extraction chance, mirroring the existing post-compaction carryover
  (`recent_steps_to_keep` retained with `summarized_recent_steps` marking
  them context-only). Scope-aware dedupe keeps the overlap from duplicating
  additions.
- **Recoverability, not prevention.** The event log is append-only and
  transcripts remain the authoritative substrate, so a missed fact stays
  recoverable by re-observation, retrieval, or post-hoc analysis. Operations
  protect previously stored items mechanically, but a valid, lazily sparse
  operation list can still omit *new* facts from the latest events. Such
  omission is silent at application time; the durable raw-event log and the
  metrics below make it measurable after the fact. Semantic new-fact omission
  is measurable, not mechanically preventable.

The operation vocabulary follows one rule: **rewrites keep the id and stay
active; kind-changing or removing operations are exits** — the item moves to
the archive with a typed exit, and any successor links back through
versioned `derived_from` references. The ops journal is the audit trail for rewrites; the
archive holds every exit.

- **`add`** applies to goals, facts, hypotheses, and claims. It requires a
  target scope, section, text, support predicate, and any section-specific fields.
  Code validates the writable scope, bounds, cap, canonical-key dedupe, actor
  metadata, and evidence, then issues the id. A fact whose effect-dependent
  support is already false is rejected or represented as a hypothesis;
  experience-dependent history may remain confirmed after abandonment.
  Pending support creates a provisionally rendered item that becomes effective
  only when its predicate commits.
- **`revise`** is field-specific, not a generic object rewrite. For a goal it
  may change text, priority, or context but never status or outcome; for a
  hypothesis it may refine text while preserving its canonical proposition
  key. Fact text and state values cannot use `revise`. Claims and corrections
  are immutable. The id, home scope, prior version, and allowed-field mask
  remain journaled.
- **`add_support`** adds one bounded, validated alternative support clause to
  an existing item's current version without changing its proposition. This is
  how a fact gains independent confirmation without abusing `revise`.
- **`set_state`** creates or rewrites one bounded `StateEntry` selected by its
  normalized key. It cannot replace an aggregate summary blob.
- **`promote`** is an exit-and-birth: the hypothesis or claim moves to the
  archive (exit `promoted`) and a new `Fact` is created with
  a versioned `derived_from` link. A claim promotion requires the agent's own
  committed verification evidence; `speaker_actor_id` must match source-event
  actor metadata.
- **`supersede`** replaces a fact's text under the same id and requires
  current evidence and an unchanged code-normalized canonical proposition key
  — a rewrite, journal-audited, no exit. It means a more precise statement of
  the same proposition. Changed mutable state belongs in `set_state`; evidence
  that the old proposition was false uses `contradict`.
- **`contradict`** is an exit-and-birth: the fact, hypothesis, or claim moves
  to the archive (exit `contradicted`) and a traced `Correction` is created
  from current evidence with a versioned `corrects` reference.
- **`transition`** changes a goal status according to the transition table
  below. A `done` transition must cite an existing outcome version/event or
  create an outcome fact/state entry in the same atomic group.
- **`demote`** moves any bounded item kind to the archive. It is rejected
  unless its section is at cap or a code-owned lifecycle rule applies.
- **`resurrect`** creates a new active item from an archived/journal version,
  with a versioned lineage link; archived ids themselves remain immutable.

There is deliberately **no `drop`**. Unmentioned items persist. When an `add`
would cross a cap and the batch contains no paired `demote`, code demotes
within that section and scope deterministically — lowest priority for goals,
oldest `meta.updated_revision` for other item kinds, ties broken by
the numeric id counter (`m2` before `m10`), never lexical string order — and
journals the system operation with its trigger's operation id; automatic
expiry, stale-hypothesis demotion, and goal pruning are likewise explicit
system operations with code-derived support and causal provenance. Legal goal
moves are open↔blocked, open→done, open/blocked→abandoned, and
blocked/done/abandoned→open (reopen);
abandonment requires a reason. Capacity behavior is never left to model
discretion.

Provenance survives mutation without making lineage ambiguous. Every reference
to a mutable item is an `ItemVersionRef(id, updated_revision)`, evidence is
role- and dependency-labeled, and every mutation points to its durable
`OperationRecord`. An old observation can therefore remain `prior_state`
without appearing to support replacement text. `revise` and `supersede` retain
history under the same id and create a new version; exit-and-birth operations
link exact versions through `derived_from`. The batch stamps
`updated_revision` from the owning scope. Support and lineage structures are
bounded in active state; overflow remains in the journal and searchable
history index.

Lifecycle rules, per category:

- **Facts are invariant or historical.** Confirmed invariant world knowledge
  does not decay; historical facts record that an event occurred. Mutable
  current values such as location, ownership, prices, planet counts, and door
  state use canonical keyed `StateEntry` records with profile-configured
  staleness. `stale_after_events` is assigned by code from the key/profile and
  counts committed scope events, never model mood or persona-global traffic. A
  free-form model-assigned volatility Boolean is not a state model.
- **Hypotheses are not droppable for being untested** — untested means
  still-valuable. They carry a tighter cap and a *test-or-demote* discipline:
  untouched for K reconciliation cycles → auto-demoted by code, not by model
  discretion. A cycle means a successful cursor advance in the owning scope,
  not a retry or unrelated-scope reconciliation. Tested-true promotes;
  tested-false converts to a correction.
- **Claims are epistemically distinct from observations.** "Player X proposed
  a truce" is a *fact* — the speech act was observed in the forum transcript.
  "X will honor the truce" is a *claim* by an interested party: tracked with
  speaker attribution, never merged into facts without the agent's own
  verification, and usable as evidence either way — kept promises and broken
  ones are both informative.
- **Goals:** `done`/`abandoned` remain visible for a configured recent-outcome
  window and carry outcome references before code archives them. `blocked`
  persists indefinitely with its context (blocked-with-reason is information);
  `reopened` is legal so an abandoned approach can return cheaply. If the sole
  support for closure is an effect-dependent atom from an abandoned
  progression, the effective view mechanically exposes the prior committed
  status.
- **Capacity demotes, never deletes.** The archive tier lives in the state
  file and the journal, costs zero ordinary prompt tokens, and is operationally
  recoverable through bounded `search_history` plus `resurrect`. Archive-cap
  eviction removes the state-file copy only; the indexed journal entry remains
  searchable. Oldest-update demotion is the deterministic baseline, not a
  claim of optimality; pinning, retrieval frequency, or learned importance are
  experimental policy variants rather than implementation gates.

Failure containment: reconciliation always receives the previous state, so an
omitted operation cannot erase a prior committed item; at worst it fails to
extract new information from the presented cycle. A non-JSON
response **raises** (the prose fallback is removed — compaction has retry plus
`compaction_retry_after_step` backoff, so failing is safe where degrading is
not). A revision mismatch, invalid operation, or truncation signal rejects
the batch without advancing the boundary. The memory commit gets the same
treatment it currently lacks:
**a bounded retry and an explicit journal event** replace today's silent
`except ModelError: return MemoryPatch()`. That fix and the prose-fallback
removal are ordinary bug fixes independent of everything else here, worth
landing ahead of the rest. Periodic reconciliation and final durable commits
get their own optional output budgets (`compaction_max_tokens` and
`memory_max_tokens`, each inheriting the decision `max_tokens` when unset) as
**typed adapter options**, not `extra_body` overrides of protected request
fields. Decisions and both utility operations also have separate adaptive
retry ceilings: explicit token-limit termination retries the same request at a
doubled, sticky budget. Stateful Responses retries remain children of the last
complete response; an incomplete response id is never committed locally.

Each proposed operation has a call-local `proposal_op_id` and optional
`depends_on_proposal_op_ids`. The store maps them to durable operation ids and
persists the translated semantic edges in each `OperationRecord`; they are not
item ids, but they remain necessary for later causal projection. Whole-batch
atomicity has a bounded liveness escape. A rejected proposal first receives a
structured repair prompt containing per-operation reasons. After N identical
failures, code partitions operations only where a dependency graph proves
independence (disjoint existing targets/scopes and no temporary-id or declared
dependency edge), journals valid groups atomically, and re-prompts
against the updated view. The cursor remains at the pending prefix until every
dependent group is accepted. After a final configured limit, the unresolved
group receives an explicit `reconciliation_dead_letter` record with its source
events and rejection reasons; the prefix advances so compaction and rollover
cannot remain blocked forever. Dead-lettered events stay searchable and are
eligible for later explicit retrieval/reconciliation. Atomicity remains the
default and is never guessed across dependent operations.

## The intent channel

A structured `intent` (settled name; not `next_intent`) may ride on every
agent-facing action: each channel's `ActionPolicy` gains `request_intent` and
`max_intent_chars`; `render_action_schema` documents the optional `"intent"`
field only when enabled. Intent is **best-effort**: an absent, over-long, or
malformed intent never rejects an otherwise valid action — code strips it,
logs a warning in the `ActivityEvent`, and executes the action. The activity
event window keeps a small rolling subset (last 2–3 intents with source
references) rendered near the previous-activity section, so multi-step plans
("after taking the lantern, move the rug") survive both the next terminal tick
and a channel change without feeding raw reasoning back into context — an
intent is a deliberate public commitment, not chain-of-thought. At
reconciliation, intents in the presented events are evidence for goal status.
Terminal intents also remain in `StepRecord.action` for backward-compatible
traces.

## Raw window

`recent_steps_to_keep` is already per-profile configuration, decoupled from
compaction cadence. It becomes the backward-compatible terminal alias for a
`recent_events_to_keep` view setting that also covers forum/message activity.
The **decision window and the reconciliation queue are independent
structures over the same persona event log**: the window is the last K events
authorized for the current `ConversationKey` and resolved view profile; the
queue is that key and scope's cursor→head prefix, consumed by reconciliation.
Advancing one never affects the other, and both read from the durable log. Raw
windows never cross authority domains merely because their events are recent.
Likewise, `pending_events` requires a conversation key, and `search_history`
intersects requested scopes with code-owned view permissions and rejects an
unauthorized request rather than trusting `HistoryQuery.scopes`. Text-adventure
profiles should carry a larger window (8–12 for Zork) so one imperfect
reconciliation is less consequential. This is tuning, not design, and rides
along in the experiment matrix.

## Stateful providers and chain rollover

For Responses chains and stateful codex/claude sessions, provider state is
primary and the resolved `MemoryViewProfile` may configure a sparser cadence —
the internal state only needs to be current at rollover and before a configured
lifecycle reset or durable commit.

Rollover is a forced reconciliation followed by
`reset_decision_state(conversation_key)`, which atomically clears that
provider conversation and its activity stage. The next activity call —
terminal, forum, or message — is therefore a full bootstrap carrying the
reconciled view on a genuinely fresh provider session. Clearing terminal
prompt counters alone is
insufficient because a codex/claude call would still resume the old session.
The trigger is a per-profile `ChainRolloverPolicy`
(`max_chain_decisions`, and `max_chain_tokens` where the adapter reports
usage). Two behaviors are defined rather than left implicit:

- **Failure:** if the forced reconciliation fails, the provider session is
  **not** reset — the old chain keeps playing, the normal compaction backoff
  applies, and rollover re-arms on the next trigger. Provider state is never
  discarded without a fresh parachute in hand.
- **Token accounting:** chain size is the **most recent call's input token
  count** where usage is reported — each Responses call's input already spans
  the whole chain, so summing per-call totals double-counts. Where usage is
  unavailable (CLI adapters may not report it), `max_chain_decisions` is the
  only trigger.

Rollover is **off by default** and stays off until the `reconciling`
policy beats `legacy` in the harness — rolling over earlier would convert
strong provider state into our weaker representation.

## Campaign memory

Under `reconciling`, the end-of-epoch commit becomes the same operations
contract against the campaign scopes in the participant's persona store
(keyed state/goals/facts/hypotheses/claims/corrections replace the flat
`durable_facts`/`open_tasks` lists), instead of `_merge_memory`'s
append-and-cap.

**The commit boundary sits after socialization.** A participant's logical
epoch is *game session → social rounds → reconcile/commit*: agreements,
threats, and deception from the forum must be able to shape next-epoch
memory through the reconciliation contract, not only through raw
forum-context injection. Mechanically this splits today's `finish_state`:
session end still drains the terminal and runs the evaluation probe, but the
commit reconciliation runs once socialization closes, with the epoch's typed
forum/message `ActivityEvent`s in its input. This is **scheduling, not
transactional atomicity**. Each game session and social visibility unit still
resolves through its own scheduler-owned progression boundary. Mid-session
reconciliations enter that session's pending overlay and become effective when
its safe generation publishes; social evidence follows its configured
visibility boundary. The post-social commit is a forced reconciliation over
those resolved events, not a catch-all external checkpoint. Provider decision
state remains available throughout the social rounds; only after that
reconciliation succeeds may a configured epoch-boundary policy reset it.

Memory is **one logical store per persona with scoped records, not one store
per campaign**. `ScopeKey` and `SourceRef` let rendering weight the active
scope while cross-game relationships and general BBS knowledge remain
available — an agent that plays multiple doors and posts in forums accumulates
one coherent identity. A campaign may mutate only its most-specific campaign
scopes; there is no automatic write into an ancestor or unrelated activity
scope. The ownership rule is hygiene and auditability — every mutation is
attributable to the activity that made it — while the common store and
composed views still provide shared memory. Per-scope revision vectors and
the persona-store lock keep all-or-nothing application coherent when several
activities share the store.

## Progression commit authority and durability domains

`ConversationKey.authority_domain` is an input trust/visibility boundary;
`ProgressionStarted.durability_domain` is an external checkpoint and rollback
isolation boundary. They may map one-to-one in a simple runtime but are not the
same concept.

Every external durability domain has exactly one normative commit authority.
For campaigns, it is the atomic active-generation pointer and the progression
outcome ledger inside the referenced generation manifest — not an independently
appended persona-store disposition. A runtime with no external mutable state
may use one store-local ledger record as its authority, but it still designates
one source of truth. `ProgressionDisposition` in the persona store is an
idempotent mirror/index used by projection and search.

The active generation exposes a cumulative outcome ledger, either by carrying
prior entries forward or by referencing an immutable ancestor chain. An
`authority_ref` therefore remains resolvable after later generations publish;
checkpoint retention may compact snapshots but cannot discard the authoritative
progression outcome history.

The campaign commit protocol is ordered as follows:

1. Under the durability-domain scheduler lock, read active generation `G0` and
   fsync `ProgressionStarted(progression_id, durability_domain,
   base_generation=G0)` before executing activity that may affect the domain.
2. Stage generation `G1`, including the external-state snapshot or visibility
   state, the progression's committed/abandoned outcome, its base generation,
   and relevant persona operation/event-log heads. Fsync every artifact and
   the staging directory.
3. Atomically publish `G1` through the domain's active-generation pointer.
   The compare-and-swap requires that the active pointer still names `G0`;
   otherwise publication fails and the scheduler re-resolves the domain.
   Successful publication is the commit point and sole authority for both
   external effect validity and progression outcome.
4. Idempotently call `finish_progression` to append/index the matching
   `ProgressionDisposition` in the persona store. Projection must resolve its
   `authority_ref` through the published ledger before treating it as
   effective.

Commit uses a generation containing the progression's durable effects.
Abandonment first publishes a repaired/restored safe generation containing the
abandoned outcome, then mirrors that outcome to the persona store. Repeating
`start_progression` with the same id and fields succeeds idempotently, while a
field mismatch fails. Repeating `finish_progression` with the same outcome and
authority reference succeeds;
attempting an opposite effective outcome fails. The normal API rejects a
committed mirror whose reference is not already published.

This ordering closes both split-brain windows. If publication succeeds but the
persona mirror is missing, recovery synthesizes it idempotently from the active
generation manifest. A raw store record that references an unpublished
generation is never effective; recovery journals it as an invalid mirror,
restores or confirms the safe external generation, publishes an authoritative
abandoned outcome, and indexes that outcome. Invalid raw records remain
forensic history but do not count as a terminal disposition, so there is still
exactly one authority-resolved outcome.

External rollback isolation is a scheduler invariant, not a property of the
persona event log. At most one progression may be pending in a durability
domain unless the domain supplies disjoint state, separate checkpoint domains,
independently committable copy-on-write branches/deltas, or an explicitly
grouped recovery unit. Overlapping progressions that mutate one monolithic
world are one recovery unit. Restoring a torn progression may be described as
leaving unrelated progression ids untouched only when this invariant proves
their external effects are isolated. The current sequential campaign scheduler
should publish a safe generation between player/social durability units and
enforce the domain lock even if future execution becomes concurrent.

Writable memory scopes are mapped to the same scheduler domain for the
duration of effect-dependent pending work. The active progression may append
several reconciliation batches to its own provisional overlay, but another
progression cannot mutate those scopes until the first resolves. This makes the
predecessor chosen by a causal capacity/exit operation stable. Disjoint domains
are logically safe to append to the persona-wide event journal concurrently
because their cursors, scopes, and external restore points are independent;
the first implementation's coarse persona gate may still serialize them for
liveness simplicity.

## Recovery

**Recovery finishes bookkeeping; it never re-executes agent activity.** An
epoch is a day that happens once: sessions that ran spent their turns and
their tokens, and a crash produces an epoch that ended badly, not a do-over.
This replaces the current `campaign.py` resume semantics — which restore the
last committed checkpoint and *replay* the whole epoch, re-running sessions
that already burned real model calls — as a deliberate semantic change, not a
refinement.

The scheduler persists a small progress state machine. It fsyncs
`ProgressionStarted` **before** the first paid call or external mutation in the
chosen durability unit, and commits that progression only through the
authority protocol above. A session is completed only when its session-safe
generation is active; a social progression is completed only when its
configured visibility state is active. A missing persona-store mirror is never
used to guess: recovery first consults the authoritative generation ledger.
The manifest records the current epoch phase, participant position, social
position, progression ids, durability domains/base generations, and event-log
heads, so every crash window maps to one recovery action.

On restart with an unfinished epoch:

1. **Resolve authority, then repair — never replay.** If the active generation
   records the progression as committed, synthesize any missing persona-store
   mirror and retain its effects. Otherwise restore externally mutable state
   to the scheduler-owned safe generation, publish a repaired generation that
   records the progression as abandoned, then mirror that outcome. Previously
   committed progressions retain their effects only under the durability-domain
   isolation invariant above; an overlapping group is repaired as one unit.
   An epoch with no started paid activity may start normally.
2. **Resolve evidence mechanically.** Events from the abandoned progression
   remain durable experience. Experience-dependent historical operations stay
   confirmed, while operations requiring committed external effects become
   ineffective or need re-verification. Facts about current effects, goal
   transitions, and keyed state updates therefore cannot remain authoritative
   merely because reconciliation preceded the crash. The causal operation fold
   also suppresses dependent exits, successors, demotions, and pruning.
   Salvage-time code mutations are `OperationRecord`s with
   `origin="system"`, `system_reason="salvage"`, and explicit causal support.
3. **Catch up when possible.** Folding already-journaled operations is
   deterministic; generating outstanding reconciliation operations is not.
   Recovery attempts the same contiguous-prefix catch-up used for live chain
   loss. If the utility provider still fails after bounded retries, it records
   `salvage_deferred`, leaves the events pending, and continues. The next
   bootstrap uses the degraded-recovery pending-prefix policy, so campaign
   progress never depends on pretending an external model call is
   deterministic or always available.
4. **Finish the epoch as it happened.** Non-agent maintenance may be restored
   and retried from its safe boundary. The epoch commits as-played — some
   participants or social progressions absent/truncated — and is marked
   `interrupted` in its manifest. The next epoch begins.

Because the operation journal never rolls back, there is no slice restore, no
restore-time revision conflict, and no journal entry that is ever obliterated.
Changing a progression outcome only changes the causal projection. Provider
conversations are never assumed to survive: recovery (like any abnormal
session end) forces `reset_decision_state(conversation_key)` for affected
keys, whose next activity call follows catch-up/degraded-bootstrap recovery.

Checkpoints therefore provide scheduler-defined restore points and the durable
campaign record, not memory rollback. A safe generation contains the external
state snapshot, visible social state, and manifest (progress state,
progression outcomes, policy/view fingerprints, scope revisions, and persona
operation/event-log heads). Multi-artifact atomicity follows the normative
generation publication protocol above. This extends the
immutable-checkpoint/atomic-pointer pattern already present in
`campaign.py`; the current code does not yet make session progress or all
artifacts one generation.

Arbitrary-position replay is explicitly not assumed anywhere: memory supports
deterministic *forensic* reconstruction at any revision (fold the ops
journal) and transcripts reproduce any screen, but the world restores only at
checkpoint positions and provider conversations restore nowhere. Every
recovery path above is forward-only.

## Implementation gates

The design is not described as crash-consistent until deterministic tests cover
these boundaries:

- A pending add selects a capacity demotion, then its progression abandons;
  the committed predecessor remains active and neither causal operation is
  effective.
- A pending contradiction archives a committed fact, then abandons; the fact
  remains in the committed projection.
- A goal closes and is pruned, then its outcome progression abandons; the prior
  effective status and visibility return.
- A generation is published but its persona-store disposition mirror is
  missing; recovery synthesizes the mirror without replaying activity.
- A store disposition references an unpublished generation; it never becomes
  effective and recovery publishes the repaired authoritative outcome.
- Two stale no-op reconciliations race on one cursor; exactly one
  compare-and-swap succeeds.
- Two progressions attempt to overlap in one non-branching durability domain;
  the scheduler rejects or groups the second before external mutation.
- Raw-window, pending-prefix, and history retrieval each attempt to cross an
  unauthorized authority domain and are rejected.
- A provider timeout occurs after `ProgressionStarted` but before an event is
  appended; the call/progression receives a terminal no-action outcome and the
  retry uses a new attempt id.
- A dependent batch cites different progressions, one committed and one
  abandoned; only the causally supported projection is effective.
- Experience-dependent history remains confirmed after abandonment while an
  effect-dependent state update from the same event becomes ineffective.
- A legacy summary replacement belongs to an abandoned progression; rendering
  falls back to the previous committed summary version.

Property-based journal tests independently assert:

```text
replay(operation_journal) == materialized_operation_state

effective_view(operation_journal, authority_ledger, disposition_index)
    == materialized_effective_projection
```

The first property covers the raw version graph, archive, indexes, revisions,
and cursors; the second independently covers disposition-sensitive visibility.
Randomized cases include journal truncation at every record boundary,
duplicate idempotent appends, AND/OR support predicates, causal DAGs, cursor
races, and every progression outcome ordering permitted by the commit protocol.

## Measurement

Because unmentioned items persist by construction, raw retention rate is
~100% under `reconciling` and measures the code, not the model. The
discriminating metrics, computable from the operation-record journal plus
activity event logs:

- **New-fact capture rate** — facts recorded vs. facts available in the
  presented activity events. Needs ground truth. Zork's existing extractor is
  ground truth for *score and moves only*, not facts or goals; capture and
  closure metrics need either a hand-curated fact/goal inventory for the
  opening map (feasible: static world, walkthrough-known) or judge-based
  post-hoc scoring.
- **Incorrect or stale facts remaining active** — audit of the active set.
- **Goal-closure accuracy** — goals closed when the evidence says done, kept
  when not (same ground-truth caveat).
- **Repeated failed actions** — from existing step logs; the downstream
  symptom bad memory causes.
- **Post-rollover performance** — score/progress in the window immediately
  after a forced fresh bootstrap, the direct test of whether reconciled
  memory can replace provider state.
- **Active-memory precision vs. archive churn** — how often demoted items are
  resurrected (an item exiting via `demote` or state expiry and re-added
  within a few cycles is a churn event).
- Game score, moves-to-score, token cost per decision — from the existing
  evaluation framework and `usage` capture.

Interrupted epochs are first-class data, not noise: metrics filter on epoch
status (`committed` vs `interrupted`) rather than pretending every epoch
completed.

Experiment isolation: every arm runs with an **isolated memory root** and
records the **store-schema, mutation-policy, prompt, and view fingerprints**,
not just the policy name. Manifests also record the decision and utility model
ids, provider versions where available, reasoning settings, output budgets,
retention/stateful mode, and recovery/rollover policy. Shared `runtime/memory`
across arms with the same `agent_id` would contaminate every later arm, and a
name alone cannot distinguish tuned variants of one policy.

Experiment matrix: `{legacy, reconciling, raw-window-only}` ×
`{stateless_full, responses-chain, claude/codex stateful}` × window
`{4, 12}` × intent `{on, off}`. The full cross is 36 arms; screen the intent
factor on a subset (one policy per mode) rather than crossing everything.
Zork first (free, deterministic, score extractor wired), winning arms to an
SRE campaign where cross-epoch memory compounds. Zork arms need only the core
design with isolated per-run memory roots; campaign arms additionally require
the forum/message ingestion and the recovery semantics described above.

These are configurations, not observations. Each selected cell runs repeated
paired trials with a predeclared primary metric and reports dispersion or a
confidence interval; seeds are recorded where a provider exposes them. Normal
play may leave rollover disabled, but designated trials force rollover at a
controlled decision/token boundary so post-rollover performance is actually
measured. The gate "reconciling beats legacy" names its trial set, minimum
effect, uncertainty criterion, and cost ceiling before results are examined.

## Open questions

Whether reconciliation may resurrect archived items unprompted or only via
explicit retrieval; default keyed-state horizons per game; how much
judge-based scoring the capture/closure metrics need before the Zork
extractors alone are trustworthy ground truth; how aggressively rendering
should weight the active scope of the persona store, and when a future
explicit copy-to-ancestor operation is justified; event-log retention and
archival policy for long-lived personas; and how to score claim handling —
whether an agent tracks, verifies, and exploits other players' kept and broken
promises.
