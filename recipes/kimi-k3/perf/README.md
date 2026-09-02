<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Kimi-K3 Benchmark Recipe

[`perf.yaml`](perf.yaml) defines a trace-replay Job targeting a deployed DGD.
The Job waits for the model at `/v1/models`, runs a small warmup, replays the
configured Mooncake trace at a given `CONCURRENCY` value, and writes JSON
summaries and JSONL records to the `shared-model-cache` PVC.

Restart the DGD pods and use a unique `ARTIFACT_DIR` between independent
concurrency points so server state and result files are not reused.

## Dataset

The benchmark replays a
[Mooncake-format](https://github.com/kvcache-ai/Mooncake) trace. Each JSONL line
describes one request with `input_length`, `output_length`, and `hash_ids`.
AIPerf reads the trace sequentially, caps synthesized requests at `MAX_ISL` and
`CAP_OSL`, streams responses, ignores EOS, and uses server token counts.

The recipe uses the same 64K-ISL / 400-OSL / 90%-KV-reuse agentic trace as the
other agentic recipes. The Git LFS file is referenced from the Kimi-K2.6 recipe
through a symlink under [`traces`](traces):

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

See the [Kimi-K3 documentation](https://docs.nvidia.com/dynamo/dev/recipes/kimi-k3)
for deployment instructions.

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

Keep `pvc-helper` to fetch artifacts, or delete it after staging.

### 3. Run the benchmark

Before applying, set `INFERENCE_URL` to
`http://YOUR_DGD_NAME-frontend:8000/v1/chat/completions`, set the pod-affinity
`nvidia.com/dynamo-graph-deployment-name` value to the same `YOUR_DGD_NAME`, and
edit `CONCURRENCY` and `ARTIFACT_DIR` for the target run. The affinity
co-locates the benchmark pod with the selected frontend.

```bash
kubectl apply -f perf.yaml -n "${NAMESPACE}"
kubectl logs -n "${NAMESPACE}" -l job-name=kimi-k3-sglang-gb300-agg-bench -f
kubectl wait --for=condition=Complete job/kimi-k3-sglang-gb300-agg-bench \
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
kubectl delete job kimi-k3-sglang-gb300-agg-bench -n "${NAMESPACE}"
kubectl delete pod pvc-helper -n "${NAMESPACE}"
```

## Running a concurrency sweep

`perf.yaml` runs one `CONCURRENCY` value. Between points, delete the completed
Job, restart the DGD pods, and set a unique `ARTIFACT_DIR`:

```bash
kubectl delete job kimi-k3-sglang-gb300-agg-bench \
  -n "${NAMESPACE}" --ignore-not-found

DGD=kimi-k3-sglang-gb300-agg-agentic
kubectl delete pods -n "${NAMESPACE}" \
  -l nvidia.com/dynamo-graph-deployment-name="${DGD}"
kubectl wait --for=condition=Ready pod -n "${NAMESPACE}" \
  -l nvidia.com/dynamo-graph-deployment-name="${DGD}" \
  --timeout=7200s

# Update CONCURRENCY and ARTIFACT_DIR in perf.yaml before each run.
kubectl apply -f perf.yaml -n "${NAMESPACE}"
kubectl wait --for=condition=Complete job/kimi-k3-sglang-gb300-agg-bench \
  -n "${NAMESPACE}" --timeout=10800s
```

## Tunable environment variables

| Variable | Default | Notes |
| --- | --- | --- |
| `TARGET_MODEL` | `moonshotai/Kimi-K3` | Must match the served model name |
| `TOKENIZER` | `moonshotai/Kimi-K3` | Tokenizer repository or path |
| `INFERENCE_URL` | `http://kimi-k3-sglang-gb300-agg-agentic-frontend:8000/v1/chat/completions` | DGD chat-completions endpoint |
| `TRACE_FILE` | `/model-cache/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl` | Mooncake trace on the PVC |
| `CONCURRENCY` | `64` | Profiling concurrency |
| `MAX_ISL` | `1048576` | Maximum synthesized input length |
| `CAP_OSL` | `12000` | Maximum synthesized output length cap|
| `ARTIFACT_DIR` | `/model-cache/aiperf-artifacts` | Use a unique directory per run |

## Artifacts

AIPerf writes a JSON summary and JSONL request records beneath
`ARTIFACT_DIR`.
