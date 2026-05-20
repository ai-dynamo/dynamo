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

This is a port, not a redesign. The scheduler algorithm is upstream's:
same lifecycle model (`REASONING / ACTING × ACTIVE / PAUSED`), same
"pause smallest ACTING first" selection, same BFD restore, same
`2^(-t/τ)` decay applied only on the resume side, same per-backend
capacity bookkeeping. The scheduler knobs in the table below are
upstream values exposed as flags, not new mechanisms.

Two implementation choices worth flagging:

1. **Native Dynamo router service.** Upstream's reference is a Python
   OpenAI proxy. We re-implemented the algorithm as a Dynamo router
   service that owns a `KvRouter` instance directly and registers as a
   model handler — no external proxy in the path.
2. **Real-token accounting.** Because the router runs in-path, we read
   `prompt_tokens + completion_tokens` straight off the chat-completions
   response. Upstream's proxy estimates from
   `len(json.dumps(payload)) / chars_per_token_ratio` since it sits
   in front of the engine and only sees raw bytes.

Single in-memory service for now; pause state is lost on restart, a
Rust port is on the roadmap, and the substantial deviations from
upstream (blended cost function for worker selection,
workflow-profile-aware pause selection, KV demote/prefetch) are
explicitly future work, not part of v0.

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

Two arms run with identical worker config and bench settings, varying
only the router:

| Arm | spm (10–67) | Pauses fired |
|---|---:|---:|
| Dynamo + stock `KvRouter` (no program scheduler) | ≈ 23.7 | 0 |
| **`thunderagent_router`** | **27.54** | 651 |

**Headline: program-aware pause/resume + working-set projection beats
the stock KV router by +16%.** Both workers track at ~93% KV util with
symmetric ~76% prefix-cache hit rates; BFD spreads paused programs
evenly without losing meaningful cache locality, because mini-SWE
programs share large system-prompt prefixes that both workers see
within a few turns.

---

## 5. Roadmap

Next, in rough priority:

1. **Blended worker selection.** The admission path picks the
   lightest-loaded worker and ignores cache state. Configure Dynamo's
   `KvRouter` with `overlap_score_weight ∈ (0, 1)` so worker selection
   blends load and prefix overlap. Sweep the weight on captured traces.
2. **Frontend in `--router-mode kv`.** Richer per-request timing
   (`prefill_wait_time_ms`, `ttft_ms`, `kv_hit_rate`, prefill/decode
   worker IDs) is only populated by `push_router`'s markers; the
   round-robin frontend skips them. Switching modes unlocks
   prefill/decode breakdown for offline replay.
3. **Workflow-profile-aware pause selection.** Profile per-session-type
   tool-gap distributions from captured traces (P50/P90 acting seconds).
   At pause time, prefer programs whose predicted idle exceeds the
   resume cost.
4. **Rust port of the hot path.** Per-response-chunk `Python::with_gil`
   + `pythonize` cost is real at 128-concurrency. Port once the
   algorithm stabilises.
5. **Stronger correctness coverage.** Multi-worker resume placement,
   restart durability of pause state, and `routing.backend_instance_id`
   honouring are smoke-tested today; they need per-request log
   assertions before this package leaves `experimental`.

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
│  - sticky worker pin from program.assigned_worker_id        │
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

The unit tests exercise admission, after-request token accounting, and
the default BFD-on-resume path.

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
