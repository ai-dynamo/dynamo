# Harphies Improvements — Batch 1 (Quick Wins)

Scope: `components/src/dynamo/frontend` and `components/src/dynamo/router`.
Four low-risk, high-value fixes from the engineering review. Each is reversible
and respects the repo's `requires-python = ">=3.10"` floor.

| # | Area | File | Change |
|---|------|------|--------|
| 1 | Security — validate at the boundary | `frontend/prepost.py` | Request validation now defaults to **fail-closed** |
| 2 | Concurrency — structured shutdown | `router/__main__.py` | Endpoint serving is now **fail-fast** (cancels siblings) |
| 3 | Concurrency — task lifecycle | `frontend/main.py` | Shutdown task is **retained** (GC-safe) |
| 4 | Correctness — cache key identity | `frontend/prepost.py` | Tokenizer cache uses **weak keys**, not `id()` |

---

## 1. Request validation fails closed by default

**Change.** `DYN_VLLM_SKIP_REQUEST_VALIDATION` now defaults to `0` (validate)
instead of `1` (skip). The trusted fast path (`model_construct`, no validation)
becomes an explicit opt-in.

**Design / reasoning.** The HTTP edge is exactly where untrusted, client-shaped
input must become a validated domain object (*Secure by Design*). A secure
default must fail **closed**; previously the default skipped Pydantic validation
and a malformed payload reached sampling-param construction and tokenization
unchecked.

**Trade-off & rollback.** Validation costs a little latency. Callers that have
already validated upstream restore the fast path with
`DYN_VLLM_SKIP_REQUEST_VALIDATION=1` — no code change, fully reversible.

## 2. Router endpoint serving is fail-fast

**Change.** Replaced `asyncio.gather(...)` over the three `serve_endpoint`
coroutines with an explicit scope: `asyncio.wait(FIRST_EXCEPTION)`, re-raise the
first real failure, and a `finally` that cancels **all** siblings and awaits
their settlement.

**Design / reasoning.** Plain `gather` does **not** cancel its peers when one
task raises, so a single endpoint failure left two orphaned servers holding
sockets while the service logged "shutting down." This is the structured-
concurrency / `errgroup` principle (*Concurrency in Go*): a fatal error in one
sibling cancels the scope.

**Why not `TaskGroup`.** `asyncio.TaskGroup` gives this for free but is 3.11+;
the project supports 3.10, so the scope is built manually. The `finally` also
covers the **graceful-shutdown** path — if the worker is cancelled, all three
endpoints are cancelled and drained, which `gather` previously handled but the
naive `wait` rewrite would have regressed.

## 3. Shutdown task is retained

**Change.** The signal handler now stores the `graceful_shutdown` task in a
module-level set and clears it via `add_done_callback(...discard)`.

**Design / reasoning.** The event loop holds only a **weak** reference to tasks
(*Concurrency in Go* — own the lifecycle of every spawned unit of work). An
unreferenced `create_task(...)` can be garbage-collected mid-execution; dropping
the *shutdown* task is the worst place for that to happen. Low probability, high
consequence — the one-line guard is the standard idiom.

## 4. Tokenizer cache keyed by weak reference

**Change.** `_ASYNC_TOKENIZER_POOL` is now a `weakref.WeakKeyDictionary` keyed
on the tokenizer object, with a `TypeError` fallback to an uncached wrapper for
non-weak-referenceable tokenizers.

**Design / reasoning.** The cache keyed on `id(tokenizer)` but held no reference
to the tokenizer. CPython reuses an `id()` after the object is collected, so the
cache could alias two distinct tokenizers onto one stale wrapper — an identity
bug (*Clean Code*; *DDIA* — cache keys must be stable identities). Weak keys also
make the cache self-evicting instead of an unbounded global. In practice
tokenizers are long-lived singletons so the bug rarely fired, but the pattern was
a latent trap.

---

## Verification

- All four edited files parse cleanly (`ast.parse`).
- Changes are import-clean and use only stdlib (`weakref`, `asyncio.wait`).
- No public signatures changed; behaviour change in #1 is gated by an env var.

---

# Batch 2 — Reliability hardening

Four fixes that bound failure and tighten resource/error handling. All default
to **no behavioural change** unless an operator opts in.

| # | Area | File | Change |
|---|------|------|--------|
| 5 | Reliability — bound the wait | `router/__main__.py` | Optional per-call **worker timeout** → typed `ConnectionTimeout` |
| 6 | Resource lifecycle | `frontend/sglang_processor.py` | Worker pool reaped via **`atexit` + idempotent `shutdown()`** |
| 7 | Concurrency — cancellation semantics | `frontend/sglang_processor.py` | **Documented** non-interruptible pool jobs + semaphore buffer |
| 8 | Error provenance | `frontend/sglang_processor.py`, `frontend/vllm_processor.py` | Broad `except` now **passes through typed `DynamoException`s** |

## 5. Bounded worker calls in the standalone router

**Change.** Added `StandaloneRouterHandler._await_worker(...)`, gated by
`DYN_ROUTER_WORKER_TIMEOUT_SECS` (default `0` = disabled). All three worker
call sites (`generate` streaming, `best_worker_id`, `get_overlap_scores`) route
through it. On stall it raises `ConnectionTimeout` (a typed `DynamoException`).

**Design / reasoning.** Every cross-process call in a routing tier needs a
timeout (*Release It!* — Timeouts; SRE — bound tail latency); the router had
none, so a hung worker stalled a request forever. For streaming the timeout
bounds the gap **between chunks**, *not* total generation time — implemented by
iterating the stream manually (`__anext__` under `asyncio.wait_for`) so long,
legitimate generations are unaffected.

**Why an env var, not a CLI flag.** Kept the surface minimal and the default
off. Trade-offs: the breaker half of the original finding (per-`worker_id`
circuit state) is **deferred** — it belongs closer to the Rust `KvRouter`, which
owns worker health; a Python-side breaker here would duplicate that state. When
enabled, each chunk is wrapped in a Task (`wait_for`), a deliberate
correctness-over-micro-perf choice that costs nothing when disabled.

## 6. Preprocess worker pool is reaped deterministically

**Change.** `SglangProcessor.shutdown()` (idempotent) tears the pool down with
`cancel_futures=True`; the factory registers it via `atexit` whenever a pool
exists.

**Design / reasoning.** The pool spawns child processes; it was created and
stored but never closed on the happy path (only on warmup failure). Whoever
spawns workers owns their teardown (*Concurrency in Go*); leaking them risks
process/memory buildup (*Release It!* — resource exhaustion). `shutdown()` is
idempotent so a future runtime teardown hook can call it directly without
racing the `atexit` handler.

> Review refinement (CodeRabbit): the `atexit` callback captures **only
> `preprocess_pool`**, not `gen`. Registering the bound `gen.shutdown` would pin
> the whole `SglangProcessor` (and its tokenizer / routed engine) until
> interpreter exit, so processors from earlier model reloads couldn't be GC'd.
> Capturing the pool directly (`preprocess_pool.shutdown(wait=False,
> cancel_futures=True)`) still guarantees exit-time reaping while leaving the
> processor collectible. A weakref-only callback was rejected: it would let the
> processor be GC'd before exit and skip pool shutdown entirely.

## 7. Non-interruptible pool cancellation, documented and bounded

**Change.** Comments on the semaphore and in `_generator_inner_pool` now state
the contract: `asyncio.wrap_future` cancellation **cannot** stop an already-
running pool job, so the worker runs to completion; the `+2` semaphore buffer
absorbs those cancelled-but-running jobs so they can't oversubscribe the pool.

**Design / reasoning.** Cancellation that stops at a process boundary is a known
trap (*Concurrency in Go* — cancellation must reach the leaf). The guard (the
existing `_worker_semaphore`) was already correct; the gap was that the
invariant was undocumented and easy to break. Making it explicit is the fix —
true interruption would require a cancellable IPC channel, out of scope here.

## 8. Typed errors survive the broad `except`

**Change.** The three catch-all sites in `sglang_processor.py` now
`except DynamoException: raise` before the generic handler, and the wrapped
`Unknown` message includes the original exception's class name. The **parallel
site in `vllm_processor.py`** (`_generate_and_stream`) gets the same pass-through
— previously it flattened *every* error, typed or not, into a generic
`make_internal_error` chunk; its `finally` still aborts registered requests.

**Design / reasoning.** Collapsing every failure to `Unknown` /
`internal_error` erased the distinction between transient (`ConnectionTimeout`,
`Disconnected`, `EngineShutdown`) and permanent (`InvalidArgument`) errors —
exactly what a caller needs to decide retry vs. fail (*Clean Code* — exceptions
must carry context; *Release It!* — distinguishing failure modes enables
breakers). The typed pass-through composes with #5: a `ConnectionTimeout` from a
stalled worker now propagates intact instead of being masked. The vllm path
keeps yielding `internal_error` for genuinely unexpected exceptions (unchanged
client contract); only typed Dynamo errors now propagate.

> Note: `except Exception` already does **not** catch `asyncio.CancelledError`
> (a `BaseException` since 3.8), so request cancellation propagated correctly
> before and after this change.

## Verification (Batch 2)

- Both edited files parse cleanly (`ast.parse`); all error imports
  (`DynamoException`, `Unknown`, `InvalidArgument`, `ConnectionTimeout`) are used.
- Defaults unchanged: timeout off (`0`), `atexit` only registered when a pool
  exists, `except` order preserves prior behaviour for non-Dynamo exceptions.

---

# Next — planned batches

Remaining review findings, grouped by the work they require. Numbers in
parentheses map back to the original review findings. Nothing here is started.

## Batch 3 — Structural refactors (larger, behind tests)

| # | Area | File(s) | Why | Effort |
|---|------|---------|-----|--------|
| 1 | Decompose `process_output` state machine (170–250 lines) | `frontend/prepost.py`, `frontend/sglang_prepost.py` | One method interleaves fast-path/reasoning/tool-buffer/streaming modes; extract per-mode strategies so each transition is unit-testable (*Clean Code*; *Refactoring* — Long Method) | High |
| 11 | Typed transport structs for router request/response | `router/__main__.py` | Two ~13-field dicts hand-mirrored with a wall of `# type: ignore`; a `TypedDict`/`msgspec.Struct` makes the boundary checkable and drift-proof (*API Design Patterns*) | Medium |
| 10 | `EndpointPath` value object | `router/args.py`, `router/__main__.py` | `namespace.component.endpoint` is split/validated in 3 places; parse once into a value object (*Refactoring* — Duplicated Code; *DDIA*) | Low–Medium |

**Sequencing.** Do #1 first behind the existing test suite (biggest
maintainability payoff, zero intended behaviour change). #11 and #10 are
mechanical once a test harness covers the router transport.

## Batch 4 — Deferred / optional reliability

| # | Area | File(s) | Why deferred |
|---|------|---------|--------------|
| 6b | Per-`worker_id` circuit breaker | `router/__main__.py` (or Rust `KvRouter`) | Breaker state belongs next to the layer that owns worker health — the Rust `KvRouter`. A Python-side breaker would duplicate that state; needs a cross-language design decision first. |
| 6c | Inter-chunk timeout on the Python processors' `routed_engine` stream | `frontend/vllm_processor.py`, `frontend/sglang_processor.py` | The embedded Rust router already owns timeouts on this path, so it's partly redundant. Add only if profiling shows stalls slipping through. |
| 7 | Hot-path allocation in the streaming loop | `frontend/sglang_processor.py` | *Measure first* with `--dyn-debug-perf`; CPython list realloc may make this a no-op. Not worth a blind change. |

## Out of scope (reinforced as already-correct)

The review also confirmed several patterns as already strong — kept here so they
aren't "fixed" by mistake: the `router/CLAUDE.md` boundary doc, per-choice
post-processor isolation for `n > 1`, typed config `validate()` methods, NVTX +
`--dyn-debug-perf` instrumentation, and graceful endpoint shutdown.
