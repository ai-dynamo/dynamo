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

**Conceptual docs live in [docs/agents/thunderagent-router.md](/docs/agents/thunderagent-router.md)** — the scheduler model, tool-boundary pause/resume semantics, the utilization-driven control loop, and observability. This README contains the source build and the complete Harbor/Pi A/B walkthrough.

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
    --router-block-size 16

# 3. Start the frontend (any router mode -- the frontend just needs to find
#    a model handler, which our service registered)
python -m dynamo.frontend --router-mode round-robin
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

## Harbor/Pi A/B walkthrough

This walkthrough runs the same SWE-bench Verified task through ThunderAgent and the stock Dynamo KV router. Harbor owns the task container, Pi runs inside it, and the model stack runs on the host. ThunderAgent is backend-agnostic and works with Dynamo's vLLM and SGLang backends; this example uses one 8-GPU node and two TP4 vLLM workers loading `MiniMaxAI/MiniMax-M2.7` from Hugging Face and serving the API alias `MiniMaxAI/MiniMax-M2`. There is no HiCache, Mooncake, shared cache, or frontend admission control in either arm.

The [agent-plugins `DynamoPi` adapter](https://github.com/ai-dynamo/agent-plugins/blob/main/pi-plugin/harbor/dynamo_pi.py) is required. It installs the Dynamo provider in each Harbor task container and maps Harbor's per-trial ID to one stable `x-dynamo-session-id` across all Pi turns. The ThunderAgent arm also sends one terminal session request so the router can release the completed program; the stock KV arm disables that request because it has no lifecycle consumer.

### 1. Install the three source trees

Use Python 3.12 and a machine with Docker and eight visible GPUs. Build Dynamo from source rather than installing a released wheel. Pi and its Node.js runtime are installed inside each Harbor task container.

```bash
git clone https://github.com/ai-dynamo/dynamo ~/src/dynamo

git clone --branch v0.16.0 --depth 1 https://github.com/harbor-framework/harbor ~/src/harbor
git clone https://github.com/ai-dynamo/agent-plugins ~/src/agent-plugins
git -C ~/src/agent-plugins checkout 223a0b8823610d042e7479bfff9b93eeac4a23ec

cd ~/src/dynamo
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
uv pip install pip 'maturin[patchelf]'
(cd lib/bindings/python && maturin develop --uv)
uv pip install -e '.[vllm]'

cd ~/src/harbor
uv sync
```

These commands were validated with Harbor `0.16.0` and Pi `0.72.1`. Record the Dynamo, Harbor, and agent-plugins revisions with benchmark artifacts.

### 2. Launch ThunderAgent or stock KV

Choose one arm. The launcher starts both TP4 workers and the matching router, then prints readiness after both workers and the public model register.

```bash
cd ~/src/dynamo
source .venv/bin/activate
export HF_HOME=/home/nvidia/hf_cache

# ThunderAgent
export ARM=ta

# Stock KV instead
# export ARM=kv

./components/src/dynamo/thunderagent_router/run_minimax_8xh100.sh "$ARM"
```

Stop the launcher with `Ctrl-C` before starting the other arm. Always use fresh server processes for each arm.

### 3. Run one Verified task

In a second terminal, set `ARM` and `DYN_AGENT_SESSION_FINAL` to match the launched stack. Use a host address reachable from Docker rather than `127.0.0.1`.

```bash
cd ~/src/harbor
source .venv/bin/activate

# ThunderAgent
export ARM=ta
export DYN_AGENT_SESSION_FINAL=1

# Stock KV instead
# export ARM=kv
# export DYN_AGENT_SESSION_FINAL=0

export PI_PLUGIN_DIR=~/src/agent-plugins/pi-plugin
export PYTHONPATH=$PI_PLUGIN_DIR/harbor
export DYNAMO_BASE_URL=http://$(ip route get 1.1.1.1 | awk '{print $7; exit}'):8100/v1
curl -fsS "$DYNAMO_BASE_URL/models"

harbor run \
  -d swebench-verified@1.0 -i astropy__astropy-12907 \
  -a dynamo_pi:DynamoPi -m dynamo/MiniMaxAI/MiniMax-M2 \
  --ak version=0.72.1 --ae "DYNAMO_BASE_URL=$DYNAMO_BASE_URL" \
  --ae "DYN_AGENT_SESSION_FINAL=$DYN_AGENT_SESSION_FINAL" \
  --mounts "[{\"type\":\"bind\",\"source\":\"$PI_PLUGIN_DIR\",\"target\":\"/opt/pi-dynamo-provider\",\"read_only\":true}]" \
  -n 1 --agent-setup-timeout-multiplier 10 \
  --job-name "$ARM-verified-one" -y
```

No pause/resume lines are expected from a one-task smoke; those only appear after the working set reaches the configured thresholds.

### 4. Run the full Verified dataset

On a fresh host, authenticate with Docker Hub and prepare every task image outside the measured interval. Verified uses 500 unique images, which exceeds the anonymous pull limit. Keeping containers until the job finishes avoids hundreds of concurrent per-trial Compose teardowns; prune them once afterward.

```bash
docker login

harbor run \
  -d swebench-verified@1.0 \
  -a nop \
  --extra-docker-compose ~/src/agent-plugins/pi-plugin/harbor/host-network.yml \
  -n 32 --n-concurrent-agents 32 \
  --memory ignore \
  --no-delete --environment-kwarg keep_containers=true \
  --job-name verified-prepull --install-only -y

docker container prune -f
```

The host-network overlay avoids creating one Docker bridge network per trial, which exhausts Docker's default address pools during the measured run.

```bash
harbor run \
  -d swebench-verified@1.0 \
  -a dynamo_pi:DynamoPi -m dynamo/MiniMaxAI/MiniMax-M2 \
  --ak version=0.72.1 --ae "DYNAMO_BASE_URL=$DYNAMO_BASE_URL" \
  --ae "DYN_AGENT_SESSION_FINAL=$DYN_AGENT_SESSION_FINAL" \
  --mounts "[{\"type\":\"bind\",\"source\":\"$PI_PLUGIN_DIR\",\"target\":\"/opt/pi-dynamo-provider\",\"read_only\":true}]" \
  --extra-docker-compose ~/src/agent-plugins/pi-plugin/harbor/host-network.yml \
  -n 256 --n-concurrent-agents 256 \
  --agent-setup-timeout-multiplier 10 --memory ignore \
  --no-delete --environment-kwarg keep_containers=true \
  --job-name "$ARM-verified-full" -y
```

This runs every task in `swebench-verified@1.0` with verification enabled. Remove the stopped task containers, start the other arm from fresh processes, set its matching `ARM` and `DYN_AGENT_SESSION_FINAL`, and rerun the same command.

Stable session headers are sent in both arms. `DYN_AGENT_SESSION_FINAL=0` only prevents the stock KV arm from receiving ThunderAgent's terminal lifecycle request.

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
