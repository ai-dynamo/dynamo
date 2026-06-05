<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Nemotron-3-Super Benchmark Recipe

A single [AIPerf](https://github.com/ai-dynamo/aiperf) trace-replay Job — [perf.yaml](perf.yaml) — covers every Nemotron-3-Super DGD variant (B200/H200 × chat/agent traces). The benchmark is identical across variants; only `ENDPOINT`, `TRACE_FILE`, and `TARGET_MODEL` need to change.

The Job waits for `GET /v1/models` on the DGD frontend to return the configured `TARGET_MODEL` (up to ~1h by default), runs a short warmup, then replays the configured trace at a single `CONCURRENCY` value and writes raw artifacts to the shared `model-cache` PVC.

The bench pod is **co-located with the DGD frontend** (`podAffinity` on the frontend's host) so client → server traffic stays on a single node.

## Targeting a variant

Edit the `env` block in [perf.yaml](perf.yaml):

| Variant target           | `ENDPOINT`                                            | `TARGET_MODEL`                                            | `TRACE_FILE` (chat / agent)                                             |
| ------------------------ | ----------------------------------------------------- | --------------------------------------------------------- | ----------------------------------------------------------------------- |
| B200 agg, chat workload  | `nemotron-3-super-b200-chat-frontend:8000`      | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | `/model-cache/traces/8k_1k_70kv_chat_new_noschedule.jsonl`    |
| B200 agg, agent workload | `nemotron-3-super-b200-agentic-frontend:8000`   | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | `/model-cache/traces/64k_400_90kv_agent_new_noschedule.jsonl` |
| H200 agg, chat workload  | `nemotron-3-super-h200-chat-frontend:8000`      | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8`   | `/model-cache/traces/8k_1k_70kv_chat_new_noschedule.jsonl`    |
| H200 agg, agent workload | `nemotron-3-super-h200-agentic-frontend:8000`   | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8`   | `/model-cache/traces/64k_400_90kv_agent_new_noschedule.jsonl` |

Both DGDs of a given SKU serve the same `--served-model-name`, so either trace can be replayed against either DGD by swapping `TRACE_FILE`. `TARGET_MODEL` only changes between B200 (NVFP4) and H200 (FP8).

If you run more than one benchmark in the same namespace, also update `metadata.name` / `labels.app` so jobs and artifact directories stay distinct.

## Dataset

The benchmark replays a [Mooncake-format](https://github.com/kvcache-ai/Mooncake) trace via `aiperf --custom-dataset-type mooncake_trace`. Each JSONL line describes one request (`input_length`, `output_length`, `hash_ids`).

Trace flavours expected on the PVC:

- **Chat** — `/model-cache/traces/8k_1k_70kv_chat_new_noschedule.jsonl`
- **Agent** — `/model-cache/traces/64k_400_90kv_agent_new_noschedule.jsonl`

For shorter runs (smoke tests, faster iteration), point `TRACE_FILE` at a smaller variant of the same trace rather than capping run time. Typical staging:

```
/model-cache/traces/<flavour>.jsonl                 # full
/model-cache/traces/<flavour>_short_30perc.jsonl    # ~30% subset
/model-cache/traces/<flavour>_short_15perc.jsonl    # ~15% subset
```

These are workload-shape traces (not model-specific). Stage your own Mooncake-format JSONLs at the path you set in `TRACE_FILE`.

## Workflow

```bash
export NAMESPACE=your-namespace
```

### 1. Deploy the DGD

See instructions in the [recipe README](../README.md).

Before deploying, do these adjustments to the DGD:
- (H200 only) Change the `SPECULATIVE_CONFIG` env to point to `key: speculative-config-synthetic` if you want a fixed-AL synthetic MTP run.
- Modify the worker `replicas` to match your desired target.

### 2. Stage the trace on the PVC

Spin up a short-lived helper pod that mounts `model-cache`, then `kubectl cp` the traces in:

```bash
kubectl run pvc-helper -n ${NAMESPACE} \
  --image=busybox:1.36 --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"helper","image":"busybox:1.36","command":["sleep","3600"],"volumeMounts":[{"name":"model-cache","mountPath":"/model-cache"}]}],"volumes":[{"name":"model-cache","persistentVolumeClaim":{"claimName":"model-cache"}}]}}' \
  --command -- sleep 3600

kubectl cp ./traces ${NAMESPACE}/pvc-helper:/model-cache/
```

Keep `pvc-helper` around for fetching artifacts later, or `kubectl delete pod pvc-helper -n ${NAMESPACE}` once you're done staging.

### 3. Run the benchmark

```bash
kubectl apply -f perf.yaml -n ${NAMESPACE}

# Stream logs
kubectl logs -n ${NAMESPACE} -l job-name=nemotron-3-super-bench -f

# Wait for completion (2h hard cap on the Job)
kubectl wait --for=condition=Complete \
  job/nemotron-3-super-bench \
  -n ${NAMESPACE} --timeout=7200s
```

### 4. Fetch artifacts

```bash
kubectl cp ${NAMESPACE}/pvc-helper:/model-cache/perf/<epoch>_nemotron-3-super-bench ./results
```

### 5. Cleanup

```bash
kubectl delete job nemotron-3-super-bench -n ${NAMESPACE}
kubectl delete pod pvc-helper -n ${NAMESPACE}   # if you kept it around
```

## Running a concurrency sweep

`perf.yaml` runs a **single** `CONCURRENCY` value. To measure multiple concurrencies you must clear server state between runs — otherwise residual KV cache / prefix-cache hits from the previous run skew results.

For each concurrency value you want to measure:

```bash
# 1. Delete the previous bench job
kubectl delete job nemotron-3-super-bench -n ${NAMESPACE} --ignore-not-found

# 2. Drop KV / prefix-cache by deleting the worker pods; Grove respawns them
#    (Dynamo workers are PodClique pods, not k8s Deployments — `kubectl rollout
#    restart deployment ...` is a silent no-op against the Grove resource chain.)
DGD=nemotron-3-super-b200-chat   # or any of the four variants
kubectl delete pods -n ${NAMESPACE} \
  -l nvidia.com/dynamo-graph-deployment-name=${DGD},nvidia.com/dynamo-component-type=worker

# 3. Bump CONCURRENCY in perf.yaml, then re-apply
kubectl apply -f perf.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/nemotron-3-super-bench -n ${NAMESPACE} --timeout=7200s
```

(The bench Job's `wait_for_model_ready` loop handles the worker restart window — it re-polls `/v1/models` until the frontend reports ready again.)

## Tunable environment variables

Edit the `env` block on the `Job` to adjust:

| Variable       | Default                                                          | Notes                                                                           |
| -------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `ENDPOINT`     | `nemotron-3-super-b200-chat-frontend:8000`                 | DGD frontend service:port — change per variant                                  |
| `TRACE_FILE`   | `/model-cache/traces/8k_1k_70kv_chat_new_noschedule.jsonl` | Swap to agent or to a smaller subset (`...short_15perc.jsonl`) for shorter runs |
| `CONCURRENCY`  | `24`                                                             | Single value — see [Running a concurrency sweep](#running-a-concurrency-sweep)  |
| `TARGET_MODEL` | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`                 | Must match `--served-model-name` on the DGD frontend (NVFP4 for B200, FP8 for H200) |

## Artifacts

Results are written to:

```
/model-cache/perf/<epoch>_<job-name>/
  warmup/
  NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4_trace_c<concurrency>_<timestamp>/
    profile_export.json
    inputs.json
    ...
```
