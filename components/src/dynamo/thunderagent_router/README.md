# `dynamo.thunderagent_router` (experimental)

> **Status: experimental.** CLI flags, the `nvext.agent_context` schema, and
> the lifecycle hooks in this package are not stable. Defaults reflect what
> we have measured so far; expect them to move.

A standalone Dynamo routing service that adds **program-level scheduling**
with tool-boundary pause/resume on top of Dynamo's native KV router. It
treats an entire agent run (LLM turn → tool execution → next LLM turn …)
as the schedulable unit, not individual requests.

---

## 1. The problem

Agentic LLM workloads (SWE-bench, browser-use, anything with a tool loop)
make many short LLM calls interleaved with non-GPU work — `docker exec`,
`pytest`, `curl`, waiting on subagent output. Between turns the agent's
KV cache sits on the GPU contributing zero progress while still occupying
blocks. At scale this caps useful throughput well below what the engine
can sustain on raw request volume.

Request-level routers (vLLM's, SGLang's, Dynamo's stock `KvRouter`)
schedule one request at a time. They see the LLM turn but not the agent
behind it. Two failure modes follow:

1. **Cache-occupancy explosion.** With N concurrent agents at conversation
   step K, the working set is `N × step_K_context`. Most of that KV is
   between turns, doing nothing. The engine evicts useful blocks under
   memory pressure or refuses admission, and every "next turn" pays a
   re-prefill tax.
2. **No tool-boundary backpressure.** The router can't slow down a hot
   trajectory at a natural pause point; it can only cancel in-flight
   requests or queue them. Either choice is worse than "wait until this
   agent is between turns and then defer."

This package addresses both.

---

## 2. ThunderAgent in one paragraph

[ThunderAgent](https://arxiv.org/abs/2602.13692) (Kang et al., 2026)
groups requests under a `program_id` and runs an outer scheduler that
moves a program through `(REASONING | ACTING) × (ACTIVE | PAUSED)`.
At a tool boundary the program goes to ACTING; under memory pressure the
scheduler **pauses** ACTING programs (logically — no decode preemption)
so that their KV blocks are eligible for eviction by the engine. When
working-set util drops, the scheduler **resumes** the smallest-token
programs first, BFD-packing them back below threshold. The result is
boundary-aware admission that never preempts active decode.

The mechanism is simple. The wins reported in the paper come from two
places: working-set accounting that knows about programs (not requests),
and pause/resume targeting tool boundaries (not arbitrary tokens).

---

## 3. How we built on ThunderAgent

The scheduler algorithm here is upstream's: same lifecycle model
(`REASONING / ACTING × ACTIVE / PAUSED`), same "pause smallest ACTING
first" selection, same BFD restore, same `2^(-t/τ)` decay applied only
on the resume side. v0 makes two real mechanical changes:

1. **Real-token accounting.** Upstream estimates `total_tokens` from
   `len(json.dumps(payload)) / chars_per_token_ratio`. We read
   `prompt_tokens + completion_tokens` from the chat-completions
   response. No estimator state, no error compounding across long
   conversations.
2. **Multi-worker BFD packing.** Upstream is single-backend; Dynamo is
   not. We extend the BFD restore to pick a worker per resumed program.
   This is the extension that required the experimental
   `kv_aware_resume_enabled` flag (default off — see §4).

What makes both possible is the integration shape: this is a Dynamo
router *inside* the request path, not a Python OpenAI proxy in front of
the engine. Same shape gives us `nvext.agent_context` propagation and
an opt-in `dynamo.agent.trace.v1` event stream for offline analysis.
The scheduler knobs in the table below are upstream values exposed as
flags, not new mechanisms.

Single in-memory service for now; pause state is lost on restart, a
Rust port is on the roadmap, and the more substantial deviations
(blended cost function for worker selection, workflow-profile-aware
pause selection, KV demote/prefetch) are explicitly future work.

### Knobs (full table)

| Flag | Env var | Default | Description |
|---|---|---|---|
| `--endpoint` | `DYN_ROUTER_ENDPOINT` | – | Worker endpoint (e.g. `dynamo.vllm.generate`) |
| `--router-block-size` | `DYN_ROUTER_BLOCK_SIZE` | 128 | KV cache block size |
| `--pause-threshold` | `DYN_THUNDERAGENT_PAUSE_THRESHOLD` | 0.95 | Working-set fraction of KV pool that fires a pause cycle. |
| `--pause-target` | `DYN_THUNDERAGENT_PAUSE_TARGET` | 0.80 | Setpoint that pause cycles drive util back down to. |
| `--soft-demote-threshold` | `DYN_THUNDERAGENT_SOFT_DEMOTE_THRESHOLD` | 0.80 | Soft-demote band start (negative priority jump in `[soft, pause)`). |
| `--soft-demote-priority-jump` | `DYN_THUNDERAGENT_SOFT_DEMOTE_PRIORITY_JUMP` | -2.0 | Priority seconds applied to soft-demoted programs. |
| `--resume-priority-boost` | `DYN_THUNDERAGENT_RESUME_PRIORITY_BOOST` | 1.0 | Priority seconds added to a request that just resumed. |
| `--resume-timeout-seconds` | `DYN_THUNDERAGENT_RESUME_TIMEOUT_SECONDS` | 1800.0 | Forced-resume cap. Mirrors ThunderAgent's `_wait_for_resume`. |
| `--resume-hysteresis` | `DYN_THUNDERAGENT_RESUME_HYSTERESIS` | 0.10 | Headroom below `pause_threshold` required before any resume. |
| `--acting-token-weight` | `DYN_THUNDERAGENT_ACTING_TOKEN_WEIGHT` | 1.0 | Multiplier on `token_total` for ACTING programs in the **pause-side** working set. |
| `--acting-decay-tau-seconds` | `DYN_THUNDERAGENT_ACTING_DECAY_TAU_SECONDS` | 1.0 | Tau for exponential decay of ACTING tokens in the **resume-side** working set. |
| `--scheduler-interval-seconds` | `DYN_THUNDERAGENT_SCHEDULER_INTERVAL_SECONDS` | 5.0 | Scheduler tick period. |
| `--scheduling-disabled` | `DYN_THUNDERAGENT_SCHEDULING_DISABLED` | false | Record lifecycle state but skip pause/resume/soft-demote. Useful for attribution. |
| `--kv-aware-resume-enabled` | `DYN_THUNDERAGENT_KV_AWARE_RESUME_ENABLED` | **false** | Ablation flag. When true, hard-overrides BFD's resume worker assignment with `KvRouter.best_worker(last_prefix)`. Default false because the override hurts spm (§4). |
| `--model-name` | `DYN_THUNDERAGENT_MODEL_NAME` | – | Frontend-visible model name. Triggers `register_model`. |
| `--model-path` | `DYN_THUNDERAGENT_MODEL_PATH` | – | Path or HF repo ID for tokenizer + model card. |

All `KvRouter` flags from `dynamo.router` (`--router-temperature`,
`--use-kv-events`, `--router-track-output-blocks`, …) are also accepted
and forwarded.

---

## 4. Initial experimental results

Benchmark stack — public repro:

- Model: `MiniMaxAI/MiniMax-M2.7` (FP8), 2× TP4 on 8× H100.
- Frontend: `dynamo.frontend --router-mode round-robin`.
- Workers: 2× `dynamo.vllm` at TP4, KV events on.
- Router: this package, knobs as below.
- Bench: `mini-swe-agent` v1.14.4 against SWE-bench-Lite at 128 concurrent
  workers, driven by [ishandhanani/ThunderAgent](https://github.com/ishandhanani/ThunderAgent)'s
  `mini-extra swebench`. The fork carries a minimal `nvext.agent_context`
  injector so each LLM call carries a stable `trajectory_id` and
  `session_id`; everything else is upstream.
- Window: bench start +10 min to +70 min (steady-state).
- Metric: successful `chat_completions` per minute at the frontend.

Three arms run with identical worker config and bench settings, varying
only the router behaviour:

| Arm | Workers | KV-aware override on resume | spm (10–67) | Pauses fired |
|---|---:|---|---:|---:|
| Dynamo + stock `KvRouter` (no program scheduler) | 128 | n/a | ≈ 23.7 | 0 |
| `thunderagent_router`, override **on** | 128 | yes | 25.93 | 823 |
| **`thunderagent_router`, override off (BFD)** | 128 | no | **27.54** | 651 |

**Headline:** program-aware pause/resume + working-set projection beats
the stock KV router by **+16%** when the override is off, and beats
itself-with-override by **+6.2%**. The override concentrates resumed
programs on whichever worker holds their warm prefix — which is the
same worker that just over-pressured into a pause — and on multi-worker
benchmarks that is strictly worse than letting BFD spread the load.

Mechanism, confirmed by per-worker metrics:

- With override on, W0 holds 92% KV util while W1 sits at 80% — ~12 pp
  of stranded capacity.
- With override off, both workers track at ~93% util with symmetric
  ~76% prefix-cache hit rates. The cache locality argument doesn't
  hold up at 128-concurrency on this workload: mini-SWE programs
  share large system-prompt prefixes that *both* workers see within a
  few turns.
- Pause rate drops 21% (823 → 651) once BFD spreads the load, because
  fewer programs cross threshold on a single worker.

The flag is kept for reproducibility but should be considered
experimental.

A separate `workers=64` arm (same override-on config) lost an additional
16% to the 128-worker baseline. Under-saturation does not rescue the
override; the asymmetric load concentration is what costs throughput.

---

## 5. Roadmap

Next, in rough priority:

1. **Blended worker-selection cost function.** Replace both the hard
   override on resume and the load-only admission path in
   `_select_worker_for_new_program_locked` with a single `KvRouter`
   call configured with `overlap_score_weight ∈ (0, 1)`. Dynamo's
   KvRouter already supports λ-blending of load and prefix overlap;
   we just need to call the right scoring API with a tuned weight.
   Sweep λ on captured traces, then live-validate the top candidate.
2. **Frontend in `--router-mode kv`.** The richer per-request timing
   fields (`prefill_wait_time_ms`, `prefill_time_ms`, `ttft_ms`,
   `avg_itl_ms`, `kv_hit_rate`, prefill/decode worker IDs) are only
   populated by `push_router`'s `record_prefill_start/complete` markers.
   Today's round-robin frontend skips them. Switching modes adds zero
   schema work and unlocks prefill/decode breakdown for offline replay.
3. **Workflow-profile-aware pause selection.** Profile per-session-type
   tool-gap distributions from captured traces (P50/P90 acting seconds).
   At pause time, prefer programs whose predicted idle exceeds the
   resume cost. This is where the trace work pays off: pick the right
   program to pause based on what it's likely to do next.
4. **KV demote / prefetch primitives.** Today pause is logical; the
   engine evicts on its own schedule. A `demote_kv(program_id, tier)`
   RPC lets the scheduler deterministically offload paused KV to
   HiCache CPU/disk, freeing GPU pool predictably. Paired with a
   `prefetch_kv(prefix, worker)` call on subagent close, the parent
   resume turn skips re-prefill.
5. **Rust port of the hot path.** Per-response-chunk `Python::with_gil`
   + `pythonize` cost is real at 128-concurrency. The Rust scaffold in
   `lib/llm/src/kv_router/program_controller.rs` is the starting point
   once the algorithm stabilises.
6. **Stronger correctness coverage.** Multi-worker resume placement,
   restart durability of pause state, and the
   `routing.backend_instance_id` honouring path are smoke-tested today;
   they need per-request log assertions before this package is
   promoted out of `experimental`.

---

## Tracing

When the frontend has agent-trace enabled
(`DYN_AGENT_TRACE_SINKS=jsonl`, `DYN_AGENT_TRACE_OUTPUT_PATH=...`), every
LLM call lands a `request_end` record carrying `trajectory_id`,
`session_id`, `input_tokens`, `output_tokens`, `cached_tokens`,
`request_received_ms`, `total_time_ms`, and the block-level
`input_sequence_hashes` — enough for offline replay against this router.
With a producer-side ZMQ publisher (set
`DYN_AGENT_TOOL_EVENTS_ZMQ_ENDPOINT` on the harness), `tool_start` /
`tool_end` / `tool_error` events come through with the same
`trajectory_id` and matching `tool_call_id` pairs, giving you the full
LLM-turn ↔ tool-gap timeline per agent.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ dynamo.frontend  (HTTP + auth + tracing sink)               │
└────────────────────┬────────────────────────────────────────┘
                     │  chat completions, with nvext.agent_context
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ dynamo.thunderagent_router  (this service)                  │
│  - ProgramTable: trajectory_id → ProgramState               │
│  - admission gate: before_request → was_paused?             │
│  - scheduler loop (every scheduler_interval_seconds):       │
│      _apply_soft_demotes → _pause_until_safe → _greedy_resume│
│  - select_worker: BFD on resume; passthrough on new req     │
│  - after_request: real-token accounting                     │
└────────────────────┬────────────────────────────────────────┘
                     │  KvRouter.generate
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ KvRouter  (in-process; subscribes to KV events + FPM)       │
└────────────────────┬────────────────────────────────────────┘
                     │  per-worker dispatch
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ dynamo.vllm  (N workers; FPM publisher, KV events publisher)│
└─────────────────────────────────────────────────────────────┘
```

---

## Usage

### Launching

```bash
# 1. Start your Dynamo workers (vLLM example, with KV events on)
python -m dynamo.vllm \
    --model <model> --tensor-parallel-size <N> \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events",
                         "endpoint":"tcp://*:20080",
                         "enable_kv_cache_events":true}'

# 2. Start the ThunderAgent router pointing at the worker endpoint
python -m dynamo.thunderagent_router \
    --endpoint dynamo.backend.generate \
    --model-name <model> \
    --router-block-size 16 \
    --router-reset-states

# 3. Start the frontend (any router mode -- the frontend just needs to find
#    a model handler, which our service registered)
python -m dynamo.frontend --router-mode round-robin --router-reset-states
```

### Sending requests

The router expects `nvext.agent_context.trajectory_id` (and optionally
`session_id`, `session_type_id`) on each chat-completions request so it
can group turns under the same program. The
[ishandhanani/ThunderAgent](https://github.com/ishandhanani/ThunderAgent)
fork of `mini-swe-agent` injects these directly via OpenAI client
`extra_body`; any other harness can do the same.

```json
{
  "model": "MiniMaxAI/MiniMax-M2",
  "messages": [...],
  "stream": true,
  "nvext": {
    "agent_context": {
      "trajectory_id": "astropy__astropy-14365",
      "session_id":    "mswea-...",
      "session_type_id": "swebench-lite"
    }
  }
}
```

Requests without `agent_context` are passed through as one-off (no
program admission, no pause/resume). This is the safe fallback for
non-agentic traffic sharing the same workers.

---

## Testing

```
pytest components/src/dynamo/thunderagent_router/tests/test_router.py
```

The unit tests exercise admission, after-request token accounting, the
default BFD-on-resume path, and the opt-in KV-aware-override path.

---

## Citation

If you use this package for research, please cite the original
ThunderAgent paper:

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

## References

- ThunderAgent paper: <https://arxiv.org/abs/2602.13692>
- Upstream ThunderAgent reference: <https://github.com/HaoKang-Timmy/ThunderAgent>
- Repro fork (mini-swe-agent + agent_context injector): <https://github.com/ishandhanani/ThunderAgent>
- Dynamo KV router: `docs/components/router/router-guide.md`
- `nvext.agent_context` schema: `docs/features/agentic_workloads.md`
