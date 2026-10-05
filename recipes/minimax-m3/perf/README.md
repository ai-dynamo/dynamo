<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# MiniMax M3 Benchmark Recipe

[`perf.yaml`](perf.yaml) defines a trace-replay Job targeting the MiniMax M3
disaggregated DGD. The Job waits for the model at `/v1/models`, runs a warmup,
replays the configured Mooncake trace at one `CONCURRENCY` value, and writes
JSON summaries and JSONL records to the `shared-model-cache` PVC.

Restart the DGD pods and use a unique `ARTIFACT_DIR` between independent
concurrency points so server state and result files are not reused.

## Dataset

The benchmark replays a
[Mooncake-format](https://github.com/kvcache-ai/Mooncake) trace. Each JSONL line
describes one request with `input_length`, `output_length`, and `hash_ids`.
AIPerf reads the trace sequentially, caps synthesized requests at `MAX_ISL` and
`CAP_OSL`, streams responses, ignores EOS, and uses server token counts.

The recipe uses the 64K-ISL / 400-OSL / 90%-KV-reuse agentic trace. The Git
LFS file is referenced from the Kimi-K2.6 recipe through a symlink under
[`traces`](traces):

```text
traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl
  -> ../../../kimi-k2.6/perf/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl
```

The default 15% trace contains 3,541 rows. Its SHA-256 is
`f20d3f2bc83dd1306cda659fbe34e7c4d85ca5497626c98bc0b1c4d2211379d0`.

## Workflow

```bash
export NAMESPACE=your-namespace
```

### 1. Deploy the DGD

See the deployment instructions in the [MiniMax M3 recipe README](../README.md).

Both deployment profiles default to real EAGLE3 verification. This benchmark
Job does not enable synthetic acceptance. To run a synthetic-acceptance
experiment, follow the explicit opt-in instructions for
[AGG](../vllm/agg-gb200-agentic/README.md#synthetic-acceptance-for-benchmarks) or
[DISAGG](../vllm/disagg-gb200-agentic/README.md#synthetic-acceptance-for-benchmarks)
and record that setting with the image and rendered deployment.
The historical synthetic-acceptance figures on the Fern recipe page do not
qualify the current image or real-verification defaults.

### 2. Stage the trace on the PVC

Materialize the Git LFS trace and copy it through a helper pod that mounts the
`shared-model-cache` PVC:

```bash
git lfs pull --include='recipes/kimi-k2.6/perf/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl'

kubectl run pvc-helper -n "${NAMESPACE}" \
  --image=busybox:1.36 --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"helper","image":"busybox:1.36","command":["sleep","3600"],"volumeMounts":[{"name":"shared-model-cache","mountPath":"/model-cache"}]}],"volumes":[{"name":"shared-model-cache","persistentVolumeClaim":{"claimName":"shared-model-cache"}}]}}' \
  --command -- sleep 3600

kubectl wait --for=condition=Ready pod/pvc-helper \
  -n "${NAMESPACE}" --timeout=120s

TRACE_SOURCE="$(git rev-parse --show-toplevel)/recipes/kimi-k2.6/perf/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl"
kubectl exec -n "${NAMESPACE}" pvc-helper -- mkdir -p /model-cache/traces
kubectl cp "${TRACE_SOURCE}" \
  "${NAMESPACE}/pvc-helper:/model-cache/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl"
```

Keep `pvc-helper` until you fetch the benchmark artifacts in step 4.

### 3. Run the benchmark

`perf.yaml` defaults to the GB200 disaggregated target at concurrency 64. If
the DGD name changes, update both `INFERENCE_URL` and the pod-affinity DGD name.
The required affinity places the benchmark pod with the selected frontend.
For the aggregated profile, set `INFERENCE_URL` to
`http://minimax-m3-agg-gb200-agentic-frontend:8000/v1/chat/completions` and the
affinity's `nvidia.com/dynamo-graph-deployment-name` value to
`minimax-m3-agg-gb200-agentic`.

```bash
kubectl apply -f perf.yaml -n "${NAMESPACE}"
kubectl logs -n "${NAMESPACE}" \
  -l job-name=minimax-m3-disagg-gb200-agentic-bench -f
kubectl wait --for=condition=Complete \
  job/minimax-m3-disagg-gb200-agentic-bench \
  -n "${NAMESPACE}" --timeout=10800s
```

### 4. Fetch artifacts

With the default `ARTIFACT_DIR`:

```bash
kubectl cp \
  "${NAMESPACE}/pvc-helper:/model-cache/aiperf-artifacts" \
  ./results
```

### 5. Cleanup

```bash
kubectl delete job minimax-m3-disagg-gb200-agentic-bench -n "${NAMESPACE}"
kubectl delete configmap minimax-m3-aiperf-config -n "${NAMESPACE}"
kubectl delete pod pvc-helper -n "${NAMESPACE}"
```

## Running a concurrency sweep

`perf.yaml` runs one `CONCURRENCY` value. Between points, delete the completed
Job, restart the DGD pods, and set a unique `ARTIFACT_DIR`:

```bash
kubectl delete job minimax-m3-disagg-gb200-agentic-bench \
  -n "${NAMESPACE}" --ignore-not-found

DGD=minimax-m3-disagg-gb200-agentic
kubectl delete pods -n "${NAMESPACE}" \
  -l nvidia.com/dynamo-graph-deployment-name="${DGD}"
kubectl wait --for=condition=Ready pod -n "${NAMESPACE}" \
  -l nvidia.com/dynamo-graph-deployment-name="${DGD}" \
  --timeout=7200s

# Update CONCURRENCY and ARTIFACT_DIR in perf.yaml before each run.
kubectl apply -f perf.yaml -n "${NAMESPACE}"
kubectl wait --for=condition=Complete \
  job/minimax-m3-disagg-gb200-agentic-bench \
  -n "${NAMESPACE}" --timeout=10800s
```

## Tunable environment variables

| Variable | Default | Notes |
| --- | --- | --- |
| `TARGET_MODEL` | `nvidia/MiniMax-M3-NVFP4` | Must match the served model name |
| `TOKENIZER` | `nvidia/MiniMax-M3-NVFP4` | Tokenizer repository or cached path |
| `INFERENCE_URL` | `http://minimax-m3-disagg-gb200-agentic-frontend:8000/v1/chat/completions` | DGD chat-completions endpoint |
| `TRACE_FILE` | `/model-cache/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl` | Mooncake trace on the PVC |
| `CONCURRENCY` | `64` | Warmup and profiling concurrency |
| `MAX_ISL` | `1000000` | Maximum synthesized input length |
| `CAP_OSL` | `12000` | Maximum synthesized output length cap |
| `ARTIFACT_DIR` | `/model-cache/aiperf-artifacts` | Use a unique directory per run |

## Artifacts

AIPerf writes a JSON summary and JSONL request records beneath
`ARTIFACT_DIR`.
