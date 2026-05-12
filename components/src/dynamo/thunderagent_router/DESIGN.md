# `dynamo.thunderagent_router` — design and parity notes

> Status: **experimental.** This package implements program-level pause/resume
> scheduling on top of Dynamo's native KV router, modelled on the ThunderAgent
> paper and its public reference implementation. The mechanism is validated
> against live workloads (mini-SWE-bench Lite, GLM-4.6-FP8 TP8) but the
> contract is not stable; treat the CLI flags and `nvext.agent_context` schema
> as in-flux.

This document is the entry point for anyone reading this package for the
first time. It exists alongside [`README.md`](README.md) (usage / launch
instructions) and explains *why* the code is shaped the way it is.

We walk through four things:

1. **What ThunderAgent proposes** — the paper's design.
2. **What the upstream reference repo actually wires up** — and what it
   doesn't. There is meaningful drift between paper and code.
3. **Our hypothesis** for what produces the observed scheduling behavior, and
   why a naive port doesn't reproduce it.
4. **What this Dynamo implementation does**, where it matches upstream
   intentionally, and where it deviates.

The audience is someone who has read the paper or skimmed the upstream
repo and is now trying to understand or extend the Dynamo port. If you
just want to run the thing, [`README.md`](README.md) is enough; come back
here when something is surprising.

---

## 1. What ThunderAgent proposes

[ThunderAgent](https://arxiv.org/abs/2602.13692) introduces **program-level**
scheduling for agentic LLM serving. The core insight: an agent run consists
of many LLM turns interleaved with tool execution, and serving systems that
schedule at request granularity (vLLM, SGLang, TRT-LLM) lose throughput when
many such agents share an engine because each agent's KV cache sits idle
between turns. ThunderAgent groups requests under a `program_id` and
schedules **the program**, not the request.

The schedulable unit is a `Program` with two orthogonal axes:

- **status** ∈ `{REASONING, ACTING}` — REASONING means an LLM request is
  on the GPU; ACTING means the agent is between turns (tool execution,
  subagent dispatch, waiting on input).
- **state** ∈ `{ACTIVE, PAUSED, TERMINATED}` — ACTIVE means the program
  is registered with a backend and admissible; PAUSED means the next
  request will block on the admission gate until the scheduler resumes it.

The two paper-defined operations are **Pause** (unbind program from
backend, release its KV slot) and **Restore** (re-bind to a backend with
available capacity). Pause is never invoked mid-decode: REASONING programs
are marked-for-pause and actually pause at the next request boundary, so
in-flight generations always complete.

### The paper's formal scheduler

Periodic monitor at fixed interval Δt (default 5 s). Each tick:

- **Thrashing condition** (equation 7 in the paper):
  ```
  C_total < Σ c_p (reasoning)  +  Σ c_q × f(t_q) (acting)
  ```
  where `f(t) = 2^(-t)` decays the contribution of ACTING programs by
  tool-execution time. The motivation: a program idle for 20 s has likely
  had its KV paged out, so counting it at full weight overcounts pressure.
- **Eviction policy** (Lemma 4.1, Eq. 8–9): recomputation cost scales
  quadratically with context length, so pause the **shortest-context**
  programs first. Specifically, prefer ACTING (idle, easy to displace)
  over REASONING (mid-turn, would forfeit a partial decode).
- **Restore policy** (Section 4.3.2): a single global program-aware
  waiting queue across all backends, packed via Best-Fit-Decreasing —
  largest paused program goes to the worker with the most remaining
  capacity. Prioritise REASONING > NEW > ACTING within the queue so
  pending requests resume before idle ones.

These three primitives — periodic check, shortest-first eviction, global
BFD restore — are the entire scheduler in the paper. The decay function
`f(t)` is presented as an optimisation that "balances `Cost_caching`
against `Cost_recompute`", not as the load-bearing mechanism.

---

## 2. What the upstream reference repo actually wires up

We read the public `ThunderAgent/scheduler/router.py` and `backend/state.py`
end-to-end against the paper. The drift between them is significant. Three
things stand out.

### 2.1 The polled metrics don't feed the pause decision

The reference repo polls each backend's Prometheus `/metrics` endpoint
every 5 s, populates a 12-deep `metrics_history` ring buffer per backend,
and exposes the latest values on its own `/health` and `/metrics` HTTP
endpoints. You'd reasonably assume these polled values inform the pause
trigger.

They don't.

The trigger is `BackendState.remaining_capacity()`, defined as:

```python
used = self.active_program_tokens - self.shared_tokens + buffer
return self.cache_config.total_tokens_capacity - used
```

where:

- `total_tokens_capacity` is parsed **once at startup** from
  `vllm:cache_config_info` (block_size × num_gpu_blocks). It is static
  after the first fetch.
- `active_program_tokens = reasoning_tokens + tool_coefficient * acting_tokens`,
  computed from the in-memory program registry (`backend._programs`).
- `shared_tokens` starts at 0 and only changes inside
  `BackendState.update_shared_tokens()`. **A repo-wide grep shows
  `update_shared_tokens` is defined but never called.** Therefore
  `shared_tokens` stays 0 for the life of the process.

So the trigger reduces to:

```
remaining = total_tokens_capacity - (active_program_tokens + 100 * N)
```

where `total_tokens_capacity` is static and `active_program_tokens` is
the scheduler's own program-table sum. The polled values
(`kv_cache_usage_perc`, `num_requests_running`, `num_requests_waiting`)
feed only `calculate_shared_tokens` (the dead path) and the HTTP API
responses. The only live side-effect of polling on scheduling is the
`backend.healthy` flag, used by `_greedy_resume` to skip unhealthy
backends.

**Implication.** A faithful port could skip Prometheus polling entirely
and lose nothing on the pause-decision path. We discovered this the hard
way after wondering whether the upstream proxy used engine-load data we
weren't using ourselves.

### 2.2 The decay function lives on the wrong side from the paper

The paper's equation 7 puts `f(t)` on the **pause-trigger** side:

```
C_total < Σ c_p (reasoning) + Σ c_q × f(t_q) (acting)
```

Effect: acting programs that have been idle long enough decay out of the
thrashing detection, so the trigger fires less aggressively. The paper
motivates this as recapture-cost balancing.

The reference code does not implement this. The pause trigger
(`remaining_capacity()`) uses a flat `tool_coefficient * acting_tokens`,
no decay.

Instead, the code offers `--use-acting-token-decay` as an **opt-in flag**
that switches the **resume** gate (`_greedy_resume`'s capacity test) from
`remaining_capacity()` to `remaining_capacity_with_decay()`:

```python
remaining = (backend.remaining_capacity_with_decay()
             if backend.use_acting_token_decay
             else backend.remaining_capacity())
```

This is a *different* mechanism. Decay-on-trigger (paper) reduces pause
aggression. Decay-on-resume (code, when flag is on) increases resume
aggression. They share the `2^(-t)` primitive but produce opposite
operational effects.

The default is **off**, and our benchmark baseline (Arm A_v2, mini-SWE-Lite
on GLM-4.6) runs with decay disabled and still produces meaningful pause
dwell.

### 2.3 The `_pause_until_safe` loop's exit condition is technically incomplete

```python
while backend.remaining_capacity() < 0:
    acting = smallest unmarked ACTING program
    if acting:
        _pause_program(acting)   # unregisters from backend, drops its tokens
        continue
    reasoning = smallest unmarked REASONING program
    if reasoning:
        _mark_program_for_pause(reasoning)  # sets a flag, does NOT drop tokens
        continue
    break
```

Marking a REASONING program sets `marked_for_pause = True` and bumps
`backend.future_paused_tokens`, but it does **not** unregister the program
from `backend._programs`. Its tokens stay in `active_program_tokens`. So
`remaining_capacity()` does not improve from marking, and the loop will
mark every unmarked REASONING program (until the candidate list is empty)
when no ACTING programs are available to pause.

In practice the lazy actual-pause happens at each marked program's next
request boundary (in `update_program_after_request`), so the deferred
debt clears naturally. But the symptom is that during deep saturation the
scheduler logs may show many more "marked" events than expected.

### Bottom line of section 2

The reference implementation does **not** match the paper's equation 7,
does **not** use polled engine load in any scheduling decision, and has a
quirky pause-loop terminator. What does work — and works well empirically
— is the basic structure: periodic check, smallest-first eviction, global
BFD restore, with `shared_tokens=0` and no decay anywhere unless you
explicitly opt in via a flag that defaults off.

---

## 3. Our hypothesis

Given the above, what actually produces the observed dwell behavior in
upstream ThunderAgent (mean pause-time ~15 s per program, max 1800 s in
our A_v2 measurements)?

The mechanism, as far as we can determine empirically, is the
**non-decayed resume gate**:

```
remaining_capacity = pool - (Σ acting_tokens × tool_coef + Σ reasoning_tokens + 100 × N)
resume program if remaining_capacity > program_required_tokens + 100
```

At saturation, every ACTING program contributes at full weight. With
~125 programs × ~9 k tokens each, the program-table sum sits at or above
the pool size. `remaining_capacity` is near zero or negative. Only the
1-2 programs whose required tokens fit can be re-admitted per tick.
Everyone else waits.

The "waiting" is what produces dwell. A program paused at tick N stays
paused at tick N+1 unless other ACTING programs have advanced to their
next turn (releasing capacity by transitioning to a new acting period or
by finishing entirely). The dwell duration is gated by the *natural*
turnover rate of the workload, not by any decay function.

If you flip `--use-acting-token-decay` on (i.e. switch to
`remaining_capacity_with_decay` on the resume side), this gate collapses.
ACTING programs decay to ~0 weight within seconds of going idle, so
`remaining` jumps to ~pool, and every paused program is immediately
resumable. **No dwell.** That's what we observed in `dynamo.thunderagent_router`
B13 before the fix, which always used decay on the resume side regardless
of any flag.

So:

- **Dwell is produced by the non-decayed resume gate** under sustained
  pool saturation.
- **The paper's equation-7 decay** would make pauses fire *less* often
  by discounting idle ACTING contribution to the trigger. It is a
  *separate* optimisation, not the dwell driver.
- **`shared_tokens`** is dead in upstream, and the polled metrics don't
  contribute to either trigger or gate. They are observability surface
  with a vestigial control-path attached.

The corollary for the Dynamo port: the cheapest thing to match the
upstream behavior is to (a) use a non-decayed resume gate by default,
(b) skip Prometheus / FPM engine-load polling on the scheduling path,
and (c) keep the basic periodic + shortest-first + BFD structure.

---

## 4. What this Dynamo implementation does

`dynamo.thunderagent_router` is a standalone Dynamo service that
registers an OpenAI-compatible model endpoint. Requests for that model
flow:

```
client
  → dynamo.frontend (any router-mode)
    → dynamo.thunderagent_router.generate   ◄── this package
      → KvRouter.generate_from_request      (Dynamo's native KV-aware router)
        → dynamo.vllm worker
```

When a request carries `nvext.agent_context.trajectory_id` (our analogue
of `program_id`), the router applies pause/resume gating, KV-aware
resume placement, and exact token accounting. Requests without
`agent_context` pass through unchanged, so the service is safe to put
in front of mixed traffic.

### Algorithm — what we port faithfully

We port the paper's basic algorithm verbatim:

- **Program lifecycle.** Same two-axis model: `ProgramStatus`
  (REASONING/ACTING) × `ProgramLifecycle` (ACTIVE/PAUSED/TERMINATED).
- **Periodic scheduler tick.** Default 5 s, same as upstream.
- **Pause trigger.** Per-worker `worker_used / capacity ≥ pause_threshold`.
  Default 0.95 (upstream effectively 1.0 since `remaining_capacity < 0`).
  With `--pause-threshold 1.0` the trigger is identical to upstream's.
- **Pause selection.** Smallest ACTING by token count first; if none,
  mark smallest REASONING.
- **Mark/clear semantics.** REASONING programs marked-for-pause clear the
  mark and actually pause in `after_request`, identical to upstream.
- **Restore order.** Tick runs `_apply_soft_demotes → _greedy_resume →
  _pause_until_safe`. Programs paused in a tick cannot be resumed until
  the next tick — same as upstream.
- **BFD restore.** Priority groups REASONING > NEW (step=1) > ACTING;
  cumulative selection within total capacity; largest-first placement
  onto highest-remaining worker.
- **Forced resume.** 1800 s timeout per paused program, identical.
- **Non-decayed resume gate.** As of the decay fix
  ([`router.py:833`](router.py)), `_greedy_resume` uses
  `_worker_used(decayed=False)`. This matches upstream's default
  (`use_acting_token_decay=False`).

### Algorithm — where we deviate

Three deviations, all in directions where Dynamo has a strictly stronger
primitive available:

| Concern | Upstream | This package | Why |
|---|---|---|---|
| Token accounting | `len(json.dumps(payload)) / char_to_token_ratio` (EWMA heuristic) | Exact `prompt_tokens + completion_tokens` from response `usage` | We have the preprocessor's exact token count; using a 5-char heuristic is strictly worse |
| Resume placement | Whichever backend BFD selected | `KvRouter.best_worker(last_prefix_token_ids)` | Land the paused program on the worker whose KV cache is warmest for its last-turn prefix; only meaningful with >1 worker |
| Soft demote rung | None | Programs in `[soft_demote_threshold, pause_threshold)` get a negative `priority_jump` for one tick | Optional softer admission control; not measurably useful on single-worker so far. Disabled by setting `--soft-demote-threshold 1.0` |

### Implementation correctness

`dynamo.thunderagent_router` does **not** use engine-runtime load fields
in its pause math. The `FpmCapacityProvider` subscribes to Dynamo's
forward-pass-metrics event plane and populates a `WorkerCapacity` dataclass
with `active_decode_kv_tokens`, `active_prefill_tokens`, `num_running`,
etc., but the scheduler reads only `cap.kv_pool_tokens` (which is the
static MDC-published pool size). This mirrors upstream's pattern: the
runtime fields exist for observability, the static pool size is what
drives the trigger.

**This is currently a known gap.** Previous design notes claimed an
"engine-true FPM signal" advantage over upstream's polled Prometheus.
That claim is misleading — neither implementation actually uses engine
load in the pause decision, and our FPM subscription pays a cost for
data the scheduler never reads. Two principled resolutions:

1. **Drop the FPM subscription** entirely; replace with a one-shot MDC
   lookup for `kv_pool_tokens` at worker discovery. Cleanest and matches
   upstream behavior.
2. **Actually use engine load** as the pause-trigger denominator:
   `(cap.active_decode_kv_tokens + cap.active_prefill_tokens) / cap.kv_pool_tokens`.
   This would be a deliberate deviation from upstream and a real "engine-true"
   advantage, with the trade-off that engine load lags admission by FPM
   cadence.

We are currently in the worst-of-both state. Resolving this is in the
backlog (see section 7).

---

## 5. Empirical results

Reference workload: mini-SWE-bench Lite, 128 concurrent workers,
GLM-4.6-FP8 on TP8 vLLM 0.20.1.

### A_v2 (upstream ThunderAgent reference, target)

`mini-swe-agent → upstream ThunderAgent (--router tr) → bare vLLM TP8`.

| Metric | Value |
|---|---:|
| Pool size (`block_size * num_gpu_blocks`) | 999,888 |
| `active_program_tokens` at resume time, p50 | 979,750 (98.0% util) |
| Pauses fired over 60 min canonical window | 576 |
| Resumes | 553 |
| `pause_s` sum across window | 8,800 s |
| `pause_s > 1 s` count | 334 |
| `pause_s` mean per pause | ~15 s |
| `pause_s` max | 1,800 s (forced-resume cap) |
| Throughput (steps/min, 10–70 window) | 31.93 |

### armB13 (Dynamo port, pre-decay-fix)

Same workload through `dynamo.frontend → dynamo.thunderagent_router →
dynamo.vllm TP8`, with `_greedy_resume` using `decayed=True` on the
resume gate.

| Metric | Value |
|---|---:|
| Pauses fired | 753 |
| Resumes | 739 |
| `pause_s` sum across window | 1.9 s |
| `pause_s > 1 s` count | 0 |
| `pause_s` mean per pause | ~2.5 ms |

The router was *firing pauses* at the right rate, but each pause
re-admitted on the very next tick because decayed ACTING programs
contributed ~0 to the resume gate. Net: zero effective dwell, no
admission backpressure on the engine, sustained over-subscription
(engine `Running` climbed to 125 vs A_v2's ~100).

### armB15 (Dynamo port, decay fix)

Same stack, with `_greedy_resume` switched to `decayed=False` (matching
upstream default) and `--pause-threshold 1.0 --pause-target 1.0
--resume-hysteresis 0.0 --soft-demote-threshold 1.0`.

Partial measurement, 10 min into saturation phase:

| Metric | Value |
|---|---:|
| Pauses fired so far | 105 |
| Resumes | 101 |
| Admission decisions with non-zero wait | 98 |
| `waited_seconds` mean | 7.9 s |
| `waited_seconds` p50 | 3.07 s |
| `waited_seconds` p90 | 18.6 s |
| `waited_seconds` max | 149 s |
| `waited_seconds > 1 s` count | 77 |

**Real dwell.** The mean grew from 2.5 ms to 7.9 s — a ~3,000× increase
— putting the Dynamo port in the same operating regime as upstream A_v2
on the same workload. Run is ongoing; the full 60-min window has not yet
been captured.

---

## 6. What's experimental

- **CLI flag surface is provisional.** Knob names (`--pause-threshold`,
  `--pause-target`, `--resume-hysteresis`, `--soft-demote-threshold`,
  `--acting-token-weight`, `--acting-decay-tau-seconds`,
  `--scheduling-disabled`) reflect debugging history, not a curated
  user-facing contract. Expect renames before this package is
  promoted out of `experimental`.
- **Single-process service.** Pause state lives in-memory in the router
  process. A restart loses every paused program's bookkeeping. In-flight
  trajectories will reconnect cleanly but their `step_count` and
  `last_prefix` are gone.
- **Multi-worker correctness.** Per-worker pause math works, but
  multi-worker resume placement via `KvRouter.best_worker` has only
  been smoke-tested. The first multi-worker bench arm is on the
  near-term roadmap.
- **Soft-demote.** The `[soft_demote_threshold, pause_threshold)` band
  has not shown a measurable effect on throughput in any single-worker
  benchmark to date. We keep it for completeness; consider it inert
  until a multi-worker run finds it useful.
- **Acting decay knob.** `--acting-decay-tau-seconds` is currently
  unused for the resume decision (since we hard-code `decayed=False`
  in `_greedy_resume`). It remains for future experiments around the
  paper's equation-7 decay, which we have not yet implemented.

---

## 7. Open work

In rough priority order:

1. **Decide on FPM subscription.** Either drop it (matches upstream
   behavior, deletes ~80 lines) or commit to using engine-runtime
   load fields as the pause-trigger denominator (real "engine-true"
   advantage). The current half-state is wasteful.
2. **Paper-faithful equation-7 ablation.** Add an optional decay on
   the pause-trigger side and measure whether it lifts throughput by
   admitting more programs into the gentle-overload region without
   triggering re-prefill cascades.
3. **`decision.waited_seconds` into the per-step profile.** Currently
   logged at `__main__.py:196` and counted in `stats()` but not
   exported as a structured per-request artifact. Wire it into the
   FPM event or a sidecar CSV so offline analysis can compute dwell
   distributions without log-grepping.
4. **Verify worker pinning under load.** Confirm that
   `routing.backend_instance_id` from `select_worker` is actually
   honored by the KV router's admission gate at saturation, not silently
   overridden. Today the only check is "the chunk we got back came from
   the worker we asked for" — log this explicitly per request.
5. **Drop the static pool-size discovery race.** `kv_pool_tokens` is
   read from the MDC at FPM-subscription startup; if discovery has
   not surfaced the worker yet, capacity is 0 and the trigger never
   fires. We currently work around this by skipping pause cycles when
   capacity is unknown; a deterministic worker-registered hook would
   be cleaner.
6. **Move the heavy hot-path off Python.** Per-response-chunk
   `Python::with_gil` + `pythonize` cost is real at 128-concurrency.
   A Rust port of `KvThunderAgentRouter` is the eventual destination
   once the algorithm stabilises; the Rust scaffold in
   `lib/llm/src/kv_router/program_controller.rs` is the starting point.

---

## 8. References

- ThunderAgent paper: <https://arxiv.org/abs/2602.13692>
- Public ThunderAgent repo: <https://github.com/HaoKang-Timmy/ThunderAgent>
- Dynamo KV router design: `docs/components/router/router-guide.md`
- `nvext.agent_context` schema: `docs/features/agentic_workloads.md`
- Companion HTML walkthrough (study guide, not part of the PR):
  `thunderagent_code_walkthrough.html` in the repo root.

## 9. Citation

If you use this package for research, please cite the original ThunderAgent
paper:

```bibtex
@misc{kang2026thunderagentsimplefastprogramaware,
      title={ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System},
      author={Hao Kang and Ziyang Li and Xinyu Yang and Weili Xu and Yinfang Chen and Junxiong Wang and Beidi Chen and Tushar Krishna and Chenfeng Xu and Simran Arora},
      year={2026},
      eprint={2602.13692},
      archivePrefix={arXiv},
      primaryClass={cs.OS},
      url={https://arxiv.org/abs/2602.13692},
}
```
