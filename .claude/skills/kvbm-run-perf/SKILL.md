---
name: kvbm-run-perf
description: Run performance benchmarks for KVBM-enabled serving with aiperf
user-invocable: true
disable-model-invocation: true
---

# Run KVBM Performance Benchmarks

Run aiperf benchmarks against a KVBM-enabled Dynamo serving stack inside the container.

## Arguments

`/dynamo:kvbm:run-perf <framework> [options...]`

- **framework** (required): Only `vllm` is supported. Reject others: "Only vllm is supported for KVBM benchmarks currently."
- **--topology TOPO** (default: `agg`): `agg`, `disagg`, `agg-router`, `disagg-router`, `disagg-2p2d`
- **--model MODEL** (default: `Qwen/Qwen3-0.6B`)
- **--concurrency N** (default: `10`)
- **--isl N** (default: `1024`): Input sequence length mean
- **--osl N** (default: `256`): Output sequence length mean
- **--requests N** (default: `100`): Total request count
- **--url URL**: Remote endpoint URL. If provided, skip container launch.
- **--image IMAGE** (default: `dynamo:latest-vllm`): Container image
- **--artifact-dir DIR** (default: `artifacts/kvbm-perf`): Where to save results

## Step 1: Validate Framework

If the first argument is not `vllm`:
> "Only `vllm` is currently supported for KVBM perf benchmarks. TensorRT-LLM and SGLang support is planned."

## Step 2: Parse Configuration and Show Plan

Collect all options, fill defaults, display:

```
KVBM Performance Benchmark Plan
────────────────────────────────
Framework:   vllm
Topology:    <topology>
Model:       <model>
Concurrency: <N>
ISL/OSL:     <ISL>/<OSL>
Requests:    <N>
Target:      <local container | remote URL>
Image:       <image>
GPUs needed: <from topology table>
Artifact dir: <dir>
```

GPU requirements:

| Topology | Launch script | GPUs |
|----------|--------------|------|
| `agg` | `examples/backends/vllm/launch/agg_kvbm.sh` | 1 |
| `disagg` | `examples/backends/vllm/launch/disagg_kvbm.sh` | 2 |
| `agg-router` | `examples/backends/vllm/launch/agg_kvbm_router.sh` | 2 |
| `disagg-router` | `examples/backends/vllm/launch/disagg_kvbm_router.sh` | 4 |
| `disagg-2p2d` | `examples/backends/vllm/launch/disagg_kvbm_2p2d.sh` | 4 |

Confirm with the user before proceeding.

## Step 3: Check Prerequisites

```bash
nvidia-smi --query-gpu=name,count --format=csv,noheader 2>/dev/null || echo "ERROR: No GPU detected"
docker run --rm <image> which aiperf || echo "WARNING: aiperf not found in container"
docker run --rm <image> python -c "import vllm, kvbm; print('OK')" || echo "ERROR: vllm/kvbm not in image"
```

Verify GPU count meets the topology requirement.

## Step 4: Launch Serving Stack (Local Only)

Skip this step if `--url` was provided.

Launch the serving container with the appropriate launch script. The container uses `--network host`, so the serving port (8000) is directly accessible from the host.

```bash
container/run.sh \
    --framework vllm \
    --image <image> \
    --gpus all \
    --mount-workspace \
    --rm FALSE \
    --name kvbm-serve \
    -- \
    bash examples/backends/vllm/launch/<script>.sh
```

**Note:** The launch scripts default to `Qwen/Qwen3-0.6B`. To override the model, the user would need to edit the script or set the MODEL variable. Tell the user if they specified a non-default model.

Wait for the server to become ready (up to 600 seconds):

```bash
for i in $(seq 1 120); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "Server is ready"
        break
    fi
    echo "Waiting for server... ($i/120)"
    sleep 5
done
```

Verify the model is loaded:

```bash
curl -s http://localhost:8000/v1/models | python -m json.tool
```

If the server doesn't become ready, report failure and suggest checking: `docker logs kvbm-serve`

## Step 5: Run aiperf Benchmark

Run aiperf from inside a second container (or from the host if aiperf is installed). Since the serving container uses `--network host`, the endpoint is accessible at `localhost:8000`.

```bash
container/run.sh \
    --framework vllm \
    --image <image> \
    --gpus none \
    --mount-workspace \
    -it \
    -v $(pwd)/<artifact-dir>:/workspace/<artifact-dir> \
    -- \
    aiperf profile \
        --model "<model>" \
        --url "http://localhost:8000" \
        --endpoint-type chat \
        --endpoint /v1/chat/completions \
        --streaming \
        --concurrency <concurrency> \
        --request-count <requests> \
        --synthetic-input-tokens-mean <isl> \
        --synthetic-input-tokens-stddev 0 \
        --output-tokens-mean <osl> \
        --output-tokens-stddev 0 \
        --extra-inputs ignore_eos:true \
        --extra-inputs temperature:0.0 \
        --artifact-dir "<artifact-dir>" \
        --random-seed 100
```

Stream output so the user can see progress.

## Step 6: Report Results

After aiperf completes, present key metrics from the summary table:

```
KVBM Performance Results
────────────────────────
Topology:          <topology>
Model:             <model>
Concurrency:       <N>
ISL/OSL:           <ISL>/<OSL>

Key Metrics:
  TTFT (avg/p50/p99):          X / Y / Z ms
  ITL (avg/p50/p99):           X / Y / Z ms
  Request Latency (avg/p99):   X / Y ms
  Output Throughput:            X tokens/s
  Request Throughput:           X req/s

Artifacts saved to: <artifact-dir>
```

Suggest visualization:
```bash
aiperf plot --artifact-dir <artifact-dir>
```

## Step 7: Cleanup (Local Only)

If a local serving container was launched:

```bash
docker stop kvbm-serve
```

Tell the user the serving stack has been shut down.

## Concurrency Sweep

If the user asks for a sweep or Pareto analysis, run multiple concurrency levels sequentially. For each level, use a separate artifact subdirectory:

```bash
for c in 1 2 5 10 25 50; do
    aiperf profile \
        --model "<model>" \
        --url "http://localhost:8000" \
        --endpoint-type chat \
        --endpoint /v1/chat/completions \
        --streaming \
        --concurrency $c \
        --request-count $(( c * 5 > 20 ? c * 5 : 20 )) \
        --synthetic-input-tokens-mean <isl> \
        --output-tokens-mean <osl> \
        --extra-inputs ignore_eos:true \
        --artifact-dir "<artifact-dir>/c${c}" \
        --random-seed 100
done
```

Then plot all results: `aiperf plot --artifact-dir <artifact-dir>`

## Reference: Launch Scripts

| Script | Topology | Components | GPUs | Default model |
|--------|----------|------------|------|---------------|
| `agg_kvbm.sh` | agg | frontend + 1 worker (KVBM) | 1 | Qwen/Qwen3-0.6B |
| `disagg_kvbm.sh` | disagg | frontend + 1 decode (GPU 0) + 1 prefill+KVBM (GPU 1) | 2 | Qwen/Qwen3-0.6B |
| `agg_kvbm_router.sh` | agg-router | frontend (KV router) + 2 workers (KVBM) | 2 | Qwen/Qwen3-0.6B |
| `disagg_kvbm_router.sh` | disagg-router | frontend (KV router) + 2 decode + 2 prefill+KVBM | 4 | Qwen/Qwen3-0.6B |
| `disagg_kvbm_2p2d.sh` | disagg-2p2d | frontend (KV router) + 2 decode + 2 prefill+KVBM | 4 | Qwen/Qwen3-0.6B |

All scripts listen on port 8000 (configurable via `DYN_HTTP_PORT`). They use `--enforce-eager` by default (remove for production perf).

## Reference: Key aiperf Flags

| Flag | Description |
|------|-------------|
| `--concurrency N` | Max concurrent requests |
| `--request-count N` | Total requests to send |
| `--request-rate N` | Target requests/second (alternative to concurrency) |
| `--benchmark-duration N` | Run for N seconds instead of request count |
| `--warmup-request-count N` | Warmup requests before measurement |
| `--gpu-telemetry` | Collect GPU metrics via DCGM |
| `--input-file FILE` | Use trace dataset instead of synthetic |
| `--custom-dataset-type mooncake_trace` | For Mooncake trace replay |
| `--fixed-schedule` | Replay at original timestamps |
