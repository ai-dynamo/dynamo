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

## Harbor/Pi A/B walkthrough

This walkthrough runs the same SWE-bench Verified task through ThunderAgent and the stock Dynamo KV router. Harbor owns the task container, Pi runs inside it, and the model stack runs on the host. The example uses one 8-GPU node, two TP4 vLLM workers, and MiniMax-M2. There is no HiCache, Mooncake, shared cache, or frontend admission control in either arm.

The [agent-plugins `DynamoPi` adapter](https://github.com/ai-dynamo/agent-plugins/blob/main/pi-plugin/harbor_dynamo_pi.py) is required. It installs the Dynamo provider in each Harbor task container and maps Harbor's per-trial ID to one stable `x-dynamo-session-id` across all Pi turns. The ThunderAgent arm also sends one terminal session request so the router can release the completed program; the stock KV arm disables that request because it has no lifecycle consumer.

### 1. Check out and install Dynamo, Harbor, and the Pi adapter

Use Python 3.12 and a machine with Docker, Node.js, and eight visible GPUs. Build Dynamo from the branch under test rather than installing a released wheel.

```bash
export DYNAMO_DIR=$HOME/src/dynamo
export HARBOR_DIR=$HOME/src/harbor
export PLUGINS_DIR=$HOME/src/agent-plugins

git clone https://github.com/ai-dynamo/dynamo "$DYNAMO_DIR"
git clone https://github.com/harbor-framework/harbor "$HARBOR_DIR"
git clone https://github.com/ai-dynamo/agent-plugins "$PLUGINS_DIR"

cd "$DYNAMO_DIR"
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
uv pip install pip 'maturin[patchelf]'
(cd lib/bindings/python && maturin develop --uv)
uv pip install -e lib/gpu_memory_service
uv pip install -e '.[vllm]'

cd "$HARBOR_DIR"
uv sync

git -C "$DYNAMO_DIR" rev-parse HEAD
git -C "$HARBOR_DIR" rev-parse HEAD
git -C "$PLUGINS_DIR" rev-parse HEAD
```

The Harbor commands below were validated with Harbor `0.16.0` and Pi `0.72.1`. Record all three revisions with the run artifacts.

### 2. Start the ThunderAgent arm

Run the following from one terminal. The public model name is registered by ThunderAgent; the two workers use an internal name so the frontend cannot bypass it.

```bash
export DYNAMO_DIR=$HOME/src/dynamo
cd "$DYNAMO_DIR"
source .venv/bin/activate

export MODEL_PATH=MiniMaxAI/MiniMax-M2
export PUBLIC_MODEL=MiniMaxAI/MiniMax-M2
export WORKER_MODEL=dyn-internal-minimax-m2
export RUN_DIR=$DYNAMO_DIR/runs/ta-$(date -u +%Y%m%dT%H%M%SZ)
export DYN_DISCOVERY_BACKEND=file
export DYN_FILE_KV=$RUN_DIR/file-kv
export DYN_REQUEST_PLANE=tcp
export DYN_EVENT_PLANE=zmq
mkdir -p "$RUN_DIR/logs" "$DYN_FILE_KV"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
DYN_SYSTEM_PORT=8181 \
DYN_FORWARDPASS_METRIC_PORT=20081 \
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
python -m dynamo.vllm \
  --model "$MODEL_PATH" \
  --served-model-name "$WORKER_MODEL" \
  --tensor-parallel-size 4 \
  --kv-cache-dtype fp8 \
  --block-size 16 \
  --enable-prefix-caching \
  --dyn-tool-call-parser minimax_m2 \
  --dyn-reasoning-parser minimax_append_think \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}' \
  >"$RUN_DIR/logs/worker-0.log" 2>&1 &
WORKER_0_PID=$!

CUDA_VISIBLE_DEVICES=4,5,6,7 \
DYN_SYSTEM_PORT=8182 \
DYN_FORWARDPASS_METRIC_PORT=20082 \
VLLM_NIXL_SIDE_CHANNEL_PORT=20098 \
python -m dynamo.vllm \
  --model "$MODEL_PATH" \
  --served-model-name "$WORKER_MODEL" \
  --tensor-parallel-size 4 \
  --kv-cache-dtype fp8 \
  --block-size 16 \
  --enable-prefix-caching \
  --dyn-tool-call-parser minimax_m2 \
  --dyn-reasoning-parser minimax_append_think \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20090","enable_kv_cache_events":true}' \
  >"$RUN_DIR/logs/worker-1.log" 2>&1 &
WORKER_1_PID=$!

DYN_SYSTEM_PORT=8183 python -m dynamo.thunderagent_router \
  --endpoint dynamo.backend.generate \
  --model-name "$PUBLIC_MODEL" \
  --model-path "$MODEL_PATH" \
  --dyn-tool-call-parser minimax_m2 \
  --dyn-reasoning-parser minimax_append_think \
  --router-block-size 16 \
  --router-reset-states \
  --shared-cache-type none \
  --pause-threshold 0.95 \
  --pause-target 0.80 \
  --soft-demote-threshold 0.80 \
  --resume-hysteresis 0.10 \
  --resume-timeout-seconds 1800 \
  --scheduler-interval-seconds 5 \
  >"$RUN_DIR/logs/thunderagent.log" 2>&1 &
THUNDERAGENT_PID=$!

DYN_SYSTEM_PORT=8184 python -m dynamo.frontend \
  --http-host 0.0.0.0 \
  --http-port 8100 \
  --router-mode round-robin \
  --router-reset-states \
  --shared-cache-type none \
  --admission-control none \
  >"$RUN_DIR/logs/frontend.log" 2>&1 &
FRONTEND_PID=$!

until [ "$(curl -fsS "http://127.0.0.1:8100/v1/models/$WORKER_MODEL/ready" | jq '[.namespaces[].worker_types.aggregated.workers // 0] | add')" = 2 ]; do sleep 5; done
until curl -fsS http://127.0.0.1:8100/v1/models | grep -Fq "$PUBLIC_MODEL"; do sleep 5; done
curl -fsS "http://127.0.0.1:8100/v1/models/$WORKER_MODEL/ready" | jq
curl -fsS http://127.0.0.1:8100/v1/models | jq
echo "$RUN_DIR"
```

The four processes remain attached to this shell as background jobs. If readiness never succeeds, inspect the four files in `$RUN_DIR/logs` instead of restarting blindly.

### 3. Run the ThunderAgent task through Harbor

Run this from a second terminal. `DYNAMO_BASE_URL` must use an address reachable from the task container, not `127.0.0.1` under Docker's default bridge network.

```bash
export HARBOR_DIR=$HOME/src/harbor
export PLUGINS_DIR=$HOME/src/agent-plugins
cd "$HARBOR_DIR"
source .venv/bin/activate

export PUBLIC_MODEL=MiniMaxAI/MiniMax-M2
export RUN_DIR=/absolute/path/printed/by/the/stack/terminal
export PI_PLUGIN_DIR=$PLUGINS_DIR/pi-plugin
export PYTHONPATH=$PI_PLUGIN_DIR${PYTHONPATH:+:$PYTHONPATH}
export DYNAMO_HOST=$(ip route get 1.1.1.1 | awk '{print $7; exit}')
export DYNAMO_BASE_URL=http://$DYNAMO_HOST:8100/v1

curl -fsS "$DYNAMO_BASE_URL/models" | jq
docker run --rm curlimages/curl:8.11.1 -fsS "$DYNAMO_BASE_URL/models"

harbor run \
  --dataset swebench-verified@1.0 \
  --include-task-name astropy__astropy-12907 \
  --agent harbor_dynamo_pi:DynamoPi \
  --model "dynamo/$PUBLIC_MODEL" \
  --agent-kwarg version=0.72.1 \
  --agent-env "DYNAMO_BASE_URL=$DYNAMO_BASE_URL" \
  --agent-env DYNAMO_API_KEY=dynamo-local \
  --agent-env DYN_AGENT_SESSION_FINAL=1 \
  --mounts "[{\"type\":\"bind\",\"source\":\"$PI_PLUGIN_DIR\",\"target\":\"/opt/pi-dynamo-provider\",\"read_only\":true}]" \
  --n-concurrent 1 \
  --n-concurrent-agents 1 \
  --max-retries 0 \
  --agent-setup-timeout-multiplier 10 \
  --no-force-build \
  --no-delete \
  --environment-kwarg keep_containers=true \
  --jobs-dir "$RUN_DIR/harbor" \
  --job-name ta-verified-one \
  --yes
```

`--no-delete --environment-kwarg keep_containers=true` stops completed task containers without running `docker compose down` on the hot path. This avoids completed containers holding Harbor concurrency slots under load and preserves them for inspection.

Check that the task produced Pi output and that ThunderAgent observed the lifecycle:

```bash
find "$RUN_DIR/harbor/ta-verified-one" -name pi.txt -o -name results.json
rg 'scheduler\.tick|Paused program|Resumed program|terminal' "$RUN_DIR/logs/thunderagent.log"
```

No pause/resume lines are expected from a one-task smoke; those only appear after the working set reaches the configured thresholds.

### 4. Stop ThunderAgent and start the stock KV arm

Return to the first terminal and stop every process from the first arm. A fair comparison starts fresh workers and an empty file-discovery directory.

```bash
kill "$FRONTEND_PID" "$THUNDERAGENT_PID" "$WORKER_0_PID" "$WORKER_1_PID"
wait "$FRONTEND_PID" "$THUNDERAGENT_PID" "$WORKER_0_PID" "$WORKER_1_PID" 2>/dev/null || true
docker ps -a --filter status=exited
# On a dedicated benchmark host, remove the stopped containers from the TA job.
docker container prune -f

export WORKER_MODEL=MiniMaxAI/MiniMax-M2
export RUN_DIR=$DYNAMO_DIR/runs/kv-$(date -u +%Y%m%dT%H%M%SZ)
export DYN_FILE_KV=$RUN_DIR/file-kv
mkdir -p "$RUN_DIR/logs" "$DYN_FILE_KV"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
DYN_SYSTEM_PORT=8181 \
DYN_FORWARDPASS_METRIC_PORT=20081 \
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
python -m dynamo.vllm \
  --model "$MODEL_PATH" \
  --served-model-name "$WORKER_MODEL" \
  --tensor-parallel-size 4 \
  --kv-cache-dtype fp8 \
  --block-size 16 \
  --enable-prefix-caching \
  --dyn-tool-call-parser minimax_m2 \
  --dyn-reasoning-parser minimax_append_think \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}' \
  >"$RUN_DIR/logs/worker-0.log" 2>&1 &
WORKER_0_PID=$!

CUDA_VISIBLE_DEVICES=4,5,6,7 \
DYN_SYSTEM_PORT=8182 \
DYN_FORWARDPASS_METRIC_PORT=20082 \
VLLM_NIXL_SIDE_CHANNEL_PORT=20098 \
python -m dynamo.vllm \
  --model "$MODEL_PATH" \
  --served-model-name "$WORKER_MODEL" \
  --tensor-parallel-size 4 \
  --kv-cache-dtype fp8 \
  --block-size 16 \
  --enable-prefix-caching \
  --dyn-tool-call-parser minimax_m2 \
  --dyn-reasoning-parser minimax_append_think \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20090","enable_kv_cache_events":true}' \
  >"$RUN_DIR/logs/worker-1.log" 2>&1 &
WORKER_1_PID=$!

DYN_SYSTEM_PORT=8184 python -m dynamo.frontend \
  --http-host 0.0.0.0 \
  --http-port 8100 \
  --router-mode kv \
  --router-reset-states \
  --shared-cache-type none \
  --admission-control none \
  >"$RUN_DIR/logs/frontend.log" 2>&1 &
FRONTEND_PID=$!

until [ "$(curl -fsS "http://127.0.0.1:8100/v1/models/$PUBLIC_MODEL/ready" | jq '[.namespaces[].worker_types.aggregated.workers // 0] | add')" = 2 ]; do sleep 5; done
curl -fsS "http://127.0.0.1:8100/v1/models/$PUBLIC_MODEL/ready" | jq
curl -fsS http://127.0.0.1:8100/v1/models | jq
echo "$RUN_DIR"
```

### 5. Run the same task through stock KV routing

In the Harbor terminal, point `RUN_DIR` at the stock arm and run the same command with terminal control disabled:

```bash
export RUN_DIR=/absolute/path/printed/by/the/stock/stack/terminal

harbor run \
  --dataset swebench-verified@1.0 \
  --include-task-name astropy__astropy-12907 \
  --agent harbor_dynamo_pi:DynamoPi \
  --model "dynamo/$PUBLIC_MODEL" \
  --agent-kwarg version=0.72.1 \
  --agent-env "DYNAMO_BASE_URL=$DYNAMO_BASE_URL" \
  --agent-env DYNAMO_API_KEY=dynamo-local \
  --agent-env DYN_AGENT_SESSION_FINAL=0 \
  --mounts "[{\"type\":\"bind\",\"source\":\"$PI_PLUGIN_DIR\",\"target\":\"/opt/pi-dynamo-provider\",\"read_only\":true}]" \
  --n-concurrent 1 \
  --n-concurrent-agents 1 \
  --max-retries 0 \
  --agent-setup-timeout-multiplier 10 \
  --no-force-build \
  --no-delete \
  --environment-kwarg keep_containers=true \
  --jobs-dir "$RUN_DIR/harbor" \
  --job-name kv-verified-one \
  --yes

find "$RUN_DIR/harbor/kv-verified-one" -name pi.txt -o -name results.json
```

The two Harbor commands differ only in the job name and `DYN_AGENT_SESSION_FINAL`. Stable session headers are still sent in the stock arm; they are identity metadata and do not enable ThunderAgent or sticky routing.

### 6. Scale the comparison

After both one-task smokes pass, use the same frozen task list and concurrency for both arms. For an exploratory 32-task run, replace `--include-task-name ...` with `--n-tasks 32` and set both concurrency options to 32. For a reported result, use repeated `--include-task-name <task-id>` arguments from a checked-in manifest because registry order is not a stable cohort. Add `--disable-verification` only when measuring serving throughput rather than solve rate.

Start Tachometer before each Harbor run if you need server-side metrics:

```bash
tachometer-scraper \
  --endpoint frontend=http://127.0.0.1:8100/metrics \
  --endpoint worker0=http://127.0.0.1:8181/metrics \
  --endpoint worker1=http://127.0.0.1:8182/metrics \
  --freq 1.0 \
  --storage "$RUN_DIR/tachometer"
```

Add `--endpoint thunderagent=http://127.0.0.1:8183/metrics` for the ThunderAgent arm. Run each arm from fresh server processes, pre-pull task images before the measured interval, and preserve the Harbor job directory and server logs. Stop the stock processes when finished:

```bash
kill "$FRONTEND_PID" "$WORKER_0_PID" "$WORKER_1_PID"
wait "$FRONTEND_PID" "$WORKER_0_PID" "$WORKER_1_PID" 2>/dev/null || true
docker ps -a --filter status=exited
```

Inspect the stopped Harbor containers before removing the ones belonging to the completed jobs.

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
