# `dynamo.thunderagent_router` (experimental)

> **Experimental — not a released component.** Run it from a source checkout
> (see [Install](#install)), not from a `pip install ai-dynamo`. The CLI
> flags, the `nvext.agent_context` schema, and the lifecycle hooks are all
> unstable and will change.

A standalone Dynamo router that schedules at the granularity of an agent run —
the whole `LLM turn → tool call → next turn` loop — instead of individual
requests. It wraps Dynamo's native KV router and adds a program-level scheduler
with tool-boundary pause/resume, porting the scheduler from the ThunderAgent
paper.

**Conceptual docs live in [docs/agents/thunderagent-router.md](/docs/agents/thunderagent-router.md)** —
the scheduler model, the 5s scheduler tick (resume → pause), tool-boundary
pause/resume semantics, the utilization-driven control loop and its full knob
table, the architecture diagram, and the scheduler observability logs. This
README keeps only the build/run/repro specifics that belong next to the code.

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

The control-loop knobs (`--pause-threshold`, `--pause-target`,
`--resume-hysteresis`, `--scheduler-interval-seconds`, …) and their defaults are
documented in [docs/agents/thunderagent-router.md](/docs/agents/thunderagent-router.md#utilization-driven-control-loop).
All `KvRouter` flags from `dynamo.router` (`--router-temperature`,
`--use-kv-events`, `--router-track-output-blocks`, …) are also accepted and
forwarded.

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

## Tracing

Enable request tracing on the frontend with the master switch
`DYN_REQUEST_TRACE=1`. That turns on sane defaults: the `jsonl_gz` sink at
`/tmp/dynamo-request-trace` and replay hashes. Bind the optional tool-events ZMQ
socket with `DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT` when the harness
publishes explicit tool spans. Override sink behavior with
`DYN_REQUEST_TRACE_SINKS` (e.g. `jsonl`, `stderr`) and
`DYN_REQUEST_TRACE_OUTPUT_PATH`.
See [Agent Tracing](/docs/agents/agent-tracing.md) for the record schema.

Every LLM call then lands a `request_end` record carrying `trajectory_id`,
`session_id`, `input_tokens`, `output_tokens`, `cached_tokens`,
`request_received_ms`, `total_time_ms`, and the block-level
`input_sequence_hashes` — enough for offline replay against this router.
Dynamo owns the ZMQ bind side, so point your harness's tool-event publisher
at that endpoint (producers connect) and `tool_start` / `tool_end` /
`tool_error` events arrive with the same `trajectory_id` and matching
`tool_call_id` pairs, giving you the full LLM-turn ↔ tool-gap timeline per
agent.

## Reproducing the MiniMax-M2 results

The headline numbers — program-aware scheduling vs KV-routing-only on the
same hardware — come from driving SWE-bench-Lite through **mini-SWE-agent** at
128 concurrent workers, against two TP4 MiniMax-M2 replicas on a single 8×H100
node.

### 1. Bring up Dynamo (2× TP4 MiniMax-M2)

One script brings up both TP4 workers, the program-aware router, and the
frontend on `:8100`:

```bash
bash components/src/dynamo/thunderagent_router/run_minimax_8xh100.sh
```

First launch JIT-warms the FP8 kernels — wait for `curl localhost:8100/v1/models`
to list the model before starting the client.

For the **KV-routing-only baseline** arm, drop the `thunderagent_router` line
from the script and run the frontend in KV-router mode against the same two
workers (`--router-mode kv`).

### 2. Run mini-SWE-agent

The mini-SWE-agent variant we reference is bundled in the ThunderAgent fork on
the `feat/mini-swe-direct-dynamo` branch: it patches mini-SWE-agent so that with
`MSWEA_BACKEND=dynamo` it injects `nvext.agent_context` natively (no proxy in the
loop), mapping each SWE-bench instance to one ThunderAgent program. Clone it,
point it at the frontend on `:8100`, and drive SWE-bench-Lite at 128 concurrent
workers:

```bash
git clone -b feat/mini-swe-direct-dynamo https://github.com/ishandhanani/ThunderAgent
cd ThunderAgent/examples/inference/mini-swe-agent
# [full] pulls the swebench extras (datasets + swe-rex); a bare `-e .` can't run swebench.
uv venv && source .venv/bin/activate && uv pip install -e ".[full]"

# The bundled config defaults base_url to :8000; point it at step 1's frontend (:8100).
sed -i 's#http://localhost:8000/v1#http://localhost:8100/v1#' \
  src/minisweagent/config/extra/swebench.yaml

export OPENAI_API_KEY="DUMMY"          # any non-empty value; the frontend ignores it
export MSWEA_BACKEND="dynamo"          # inject nvext.agent_context natively
export MSWEA_SESSION_TYPE_ID="swebench-lite"

mini-extra swebench \
  --config src/minisweagent/config/extra/swebench.yaml \
  --subset lite --split test --workers 128 \
  --model 'MiniMaxAI/MiniMax-M2' \
  --output ./swebench_out --redo-existing
```

For the **KV-routing-only baseline** arm, bring Dynamo up in KV-router mode
(step 1, `--router-mode kv`, no `thunderagent_router`) and rerun the same
command against the same two workers.

### Expected

Over the 10–67 min steady-state window, program-aware routing
(`thunderagent_router`) sustains **≈27.5 steps/min** versus **≈23.7** for the
KV-routing-only baseline on the same two workers — an **8–16% throughput
improvement** (≈16% at the steady-state peak; ≈8.8% in the stricter matched-A/B
framing). Throughput (steps/min over the window), not resolved-rate, is the
metric at this concurrency; resolved-rate deltas are within run-to-run noise.

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

- Conceptual docs: [docs/agents/thunderagent-router.md](/docs/agents/thunderagent-router.md)
- ThunderAgent paper: <https://arxiv.org/abs/2602.13692>
- Upstream ThunderAgent reference: <https://github.com/HaoKang-Timmy/ThunderAgent>
- Repro fork (mini-swe-agent + agent_context injector): <https://github.com/ishandhanani/ThunderAgent>
- Dynamo KV router: [Router Guide](/docs/components/router/router-guide.md)
- `nvext.agent_context` schema: [nvext reference](/docs/components/frontend/nvext.md#agent-context)
