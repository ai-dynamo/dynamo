# `dynamo.thunderagent_router` (experimental)

> **Experimental — not a released component.** Run it from a source checkout
> (see [Install](#install)), not from a `pip install ai-dynamo`. The CLI
> flags, the `nvext.agent_context` schema, and the lifecycle hooks are all
> unstable and will change.

A standalone Dynamo router that schedules at the granularity of an agent run
— the whole `LLM turn → tool call → next turn` loop — instead of individual
requests. It wraps Dynamo's native KV router and adds tool-boundary
pause/resume, porting the scheduler from the ThunderAgent paper.

## The problem

Agentic workloads (SWE-bench, browser-use, anything with a tool loop) make
many short LLM calls separated by non-GPU work: `docker exec`, `pytest`,
`curl`, waiting on a subagent. Between turns the agent's KV cache stays
resident, holding blocks while doing nothing. A request-level router
(vLLM's, SGLang's, Dynamo's stock `KvRouter`) sees each turn but not the
agent behind it, which costs you two ways:

- **Cache-occupancy blowup.** With N agents at step K, the working set is
  `N × step_K_context`, most of it idle between turns. The engine evicts
  useful blocks under pressure or refuses admission, and every next turn
  pays a re-prefill tax.
- **No tool-boundary backpressure.** The router can't defer a hot trajectory
  at a natural pause point — it can only cancel in-flight requests or queue
  them, both worse than waiting until the agent is between turns.

## The scheduler

The algorithm comes from [ThunderAgent](https://arxiv.org/abs/2602.13692)
(Kang et al., 2026). It groups requests by `program_id` and runs an outer
scheduler that moves each program through `(REASONING | ACTING) × (ACTIVE |
PAUSED)`. A program enters ACTING at a tool boundary. Under memory pressure
the scheduler pauses ACTING programs — logically, with no decode preemption —
so the engine is free to evict their KV. When utilization drops it resumes
the smallest-token programs first, BFD-packing them back under threshold. The
payoff is working-set accounting that counts programs rather than requests,
plus pause/resume aimed at tool boundaries rather than arbitrary tokens.

## What this port changes

The scheduler is upstream's, unchanged: same lifecycle, same "pause smallest
ACTING first" selection, same BFD restore, same `2^(-t/τ)` decay on the
resume side, same per-backend capacity bookkeeping. The knobs in the table
below expose upstream's values as flags; none are new mechanisms.

Two things differ from the reference implementation:

- **In-path Dynamo service, not a proxy.** Upstream ships a Python OpenAI
  proxy in front of the engine. This runs as a Dynamo router that owns a
  `KvRouter` directly and registers as a model handler, so there's no extra
  proxy hop.
- **Real token counts.** Running in-path, it reads `prompt_tokens +
  completion_tokens` off each response. The upstream proxy only sees raw
  bytes, so it estimates from `len(json.dumps(payload)) / chars_per_token`.

v0 is a single in-memory service, so pause state is lost on restart. A Rust
port and the larger deviations from upstream — blended load/overlap worker
selection, workflow-profile-aware pause selection, KV demote/prefetch — are
future work, not part of this version.

### Knobs

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

## Roadmap

Roughly in priority order:

1. **Blended worker selection.** Admission currently picks the
   lightest-loaded worker. Configure `KvRouter` with
   `overlap_score_weight ∈ (0, 1)` so selection blends load and prefix
   overlap.
2. **Workflow-profile-aware pause selection.** Profile per-session-type
   tool-gap distributions and prefer pausing programs whose predicted
   idle exceeds the resume cost.
3. **Rust port of the hot path.** Per-response-chunk `Python::with_gil`
   + `pythonize` cost is real at 128-concurrency.
4. **Stronger correctness coverage.** Multi-worker resume placement,
   restart durability of pause state, and `routing.backend_instance_id`
   honouring need per-request log assertions before this package leaves
   `experimental`.

---

## Tracing

Enable agent tracing on the frontend with the master switch
`DYN_AGENT_TRACE=1`. That turns on sane defaults: the `jsonl_gz` sink at
`/tmp/dynamo-agent-trace`, the tool-events ZMQ socket bound at
`tcp://127.0.0.1:20390`, and replay hashes. Override any of them with
`DYN_AGENT_TRACE_SINKS` (e.g. `jsonl`, `stderr`),
`DYN_AGENT_TRACE_OUTPUT_PATH`, and `DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT`.

Every LLM call then lands a `request_end` record carrying `trajectory_id`,
`session_id`, `input_tokens`, `output_tokens`, `cached_tokens`,
`request_received_ms`, `total_time_ms`, and the block-level
`input_sequence_hashes` — enough for offline replay against this router.
Dynamo owns the ZMQ bind side, so point your harness's tool-event publisher
at that endpoint (producers connect) and `tool_start` / `tool_end` /
`tool_error` events arrive with the same `trajectory_id` and matching
`tool_call_id` pairs, giving you the full LLM-turn ↔ tool-gap timeline per
agent.

---

## Architecture

```text
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

## Install

This is experimental and not shipped as a supported entrypoint. Run it from a
source checkout:

```bash
git clone https://github.com/ai-dynamo/dynamo
cd dynamo
# build the Rust bindings, then install the Python components editable
(cd lib/bindings/python && maturin develop --uv)
uv pip install -e .
```

`python -m dynamo.thunderagent_router` then resolves against that checkout.

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

## Reproducing the MiniMax-M2 results

The headline numbers — program-aware scheduling vs KV-routing-only on the
same hardware — come from driving SWE-bench-Lite through pi via a Harbor
adapter, against two TP4 MiniMax-M2 replicas on a single 8×H100 node.

### 1. Bring up Dynamo (2× TP4 MiniMax-M2)

```bash
# Two TP4 vLLM workers, KV events on, native MiniMax tool-call parser.
for i in 0 1; do
  CUDA_VISIBLE_DEVICES=$((i*4)),$((i*4+1)),$((i*4+2)),$((i*4+3)) \
  python -m dynamo.vllm \
      --model MiniMaxAI/MiniMax-M2 --tensor-parallel-size 4 \
      --dyn-tool-call-parser minimax_m2 \
      --kv-events-config '{"publisher":"zmq","topic":"kv-events",
                           "endpoint":"tcp://*:2008'"$i"'",
                           "enable_kv_cache_events":true}' &
done

# Router. DYN_TOOL_CALL_PARSER is forwarded to register_model so MiniMax's
# <minimax:tool_call> XML reaches pi as real OpenAI tool_calls, not raw text.
DYN_TOOL_CALL_PARSER=minimax_m2 python -m dynamo.thunderagent_router \
    --endpoint dynamo.vllm.generate \
    --model-name MiniMaxAI/MiniMax-M2 \
    --router-block-size 16

# Frontend on :8100 (the router already registered the model handler).
python -m dynamo.frontend --http-port 8100 --router-mode round-robin
```

For the **KV-routing-only baseline** arm, drop the `thunderagent_router`
process and run the frontend in KV-router mode against the same two workers
(`python -m dynamo.frontend --http-port 8100 --router-mode kv`).

### 2. Run pi + Harbor

The Harbor pi adapter lives on the `feat/harbor-pi-adapter` branch of the
fork. It installs `pi-dynamo-provider` into each trial container and injects
`nvext.agent_context` so each Harbor trial maps to one ThunderAgent program.

```bash
# Clone the fork and the provider side by side.
git clone -b feat/harbor-pi-adapter https://github.com/ishandhanani/ThunderAgent
git clone https://github.com/ai-dynamo/pi-dynamo-provider
export PI_DYNAMO_PROVIDER_PATH="$PWD/pi-dynamo-provider"

cd ThunderAgent/examples/datagen/harbor
uv venv && source .venv/bin/activate && uv pip install -e .

harbor run \
  --path datasets/swebench \
  --agent pi \
  --model 'dynamo/MiniMaxAI/MiniMax-M2' \
  --ak api_base='http://127.0.0.1:8100/v1' \
  --n-tasks 30 --n-concurrent 10 \
  --network-mode host --override-cpus 2 --override-memory-mb 8192 \
  -v "$PI_DYNAMO_PROVIDER_PATH:/opt/pi-dynamo-provider:ro" \
  --jobs-dir /tmp/harbor-jobs --job-name ta-run --quiet
```

### Expected

On 8×H100 with 30 SWE-bench-Lite tasks (20 astropy + 10 django) at
`n_concurrent=10`, program-aware scheduling runs at ~1.34 trials/min vs ~0.84
for the KV-routing-only baseline — roughly **+60% faster wall-clock at matched
pass rate** (pass-rate deltas on this slice are within run-to-run noise). The
advantage widens as agents make more LLM calls per task.

Pointing `--ak api_base` at a stock `vllm serve` instead of the Dynamo
frontend also works — `nvext.agent_context` is silently dropped — which is how
the non-Dynamo control arm is run.

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
