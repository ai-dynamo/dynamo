# `dynamo.thunderagent_router` (experimental)

> **Experimental — not a released component.** Run it from a source checkout
> (see [Install](#install)), not from a `pip install ai-dynamo`. The CLI
> flags, session headers, and the lifecycle hooks are all unstable and will
> change.

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

The router expects header-derived `session_id` on each chat-completions
request so it can group turns under the same program. Custom harnesses can send
`x-dynamo-session-id` and, for subagents, `x-dynamo-parent-session-id`.

Requests without session identity are passed through as one-off (no program
admission, no pause/resume). This is the safe fallback for non-agentic traffic
sharing the same workers.

### SGLang HiCache retention budget

`dynamo.sglang` publishes the authoritative GPU KV and HiCache host capacities in each worker's model deployment card. The scheduler automatically uses their sum as its retention budget, so `--pause-threshold 0.95` means 95% of the combined GPU + host pool; there is no ThunderAgent HiCache flag to set. This lets SGLang spill from GPU to its native host tier before ThunderAgent starts holding programs at tool boundaries.

Mooncake capacity is deliberately excluded. It is a content-addressed storage tier whose contents may be evicted or may not match the next request, not an unconditional program-retention budget. ThunderAgent does not call HiCache eviction, restore, prefetch, or Mooncake APIs; SGLang remains the admission and materialization authority.

## Tracing

Enable request tracing on the frontend with the master switch
`DYN_REQUEST_TRACE=1`. That turns on sane defaults: the `jsonl_gz` sink at
`/tmp/dynamo-request-trace` and replay hashes. Bind the optional tool-events ZMQ
socket with `DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT` when the harness
publishes explicit tool spans. Override sink behavior with
`DYN_REQUEST_TRACE_SINKS` (e.g. `jsonl`, `stderr`) and
`DYN_REQUEST_TRACE_OUTPUT_PATH`.
See [Agent Tracing](/docs/agents/agent-tracing.md) for the record schema.

Every LLM call then lands a `request_end` record carrying `session_id`,
`input_tokens`, `output_tokens`, `cached_tokens`,
`request_received_ms`, `total_time_ms`, and the block-level
`input_sequence_hashes` — enough for offline replay against this router.
Dynamo owns the ZMQ bind side, so point your harness's tool-event publisher
at that endpoint (producers connect) and `tool_start` / `tool_end` /
`tool_error` events arrive with the same `session_id` and matching
`tool_call_id` pairs, giving you the full LLM-turn ↔ tool-gap timeline per
agent.

## Reproducing with Pi

The maintained smoke path is Pi through the Dynamo provider, not the old private mini-SWE-agent fork. Start the workers, this router, and the frontend as described above, then wait for the frontend to advertise the served model:

```bash
curl -fsS http://127.0.0.1:8000/v1/models
```

Install the [Pi Dynamo provider](https://github.com/ai-dynamo/agent-plugins/tree/main/pi-plugin) from a checkout, then send a sessionized turn through the frontend:

```bash
git clone https://github.com/ai-dynamo/agent-plugins.git
cd agent-plugins/pi-plugin
npm install && npm run build
pi install "$PWD"

export DYNAMO_BASE_URL=http://127.0.0.1:8000/v1
export DYNAMO_API_KEY=dummy
export DYN_AGENT_SESSION_ID=ta-repro-$(uuidgen)
pi --model dynamo/<served-model-name> -p 'Reply exactly READY.'
```

The provider sends `x-dynamo-session-id` on every LLM request. `DYN_REQUEST_TRACE` is optional and controls Dynamo trace collection/tool-event relay only; it does not control session headers. With `DYN_REQUEST_TRACE=1` on the frontend, confirm the trace row carries the same session ID before running a larger Harbor/SWE-bench cohort. See [Agent Harnesses](/docs/agents/agent-harnesses.md#pi) for the provider contract.

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
- Pi Dynamo provider: <https://github.com/ai-dynamo/agent-plugins/tree/main/pi-plugin>
- Dynamo KV router: [Router Guide](/docs/components/router/router-guide.md)
