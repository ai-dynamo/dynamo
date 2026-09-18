<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# K-EXAONE 2.0 Benchmark Recipe

A single [AIPerf](https://github.com/ai-dynamo/aiperf) trace-replay Job —
[`perf.yaml`](perf.yaml) — covers both K-EXAONE 2.0 DGDs. Set `ENDPOINT` and
`CONCURRENCY` for the target DGD.

The Job waits for the target model on the DGD frontend, runs a short warmup,
replays the configured trace at one `CONCURRENCY` value, and writes raw
artifacts to the shared `model-cache` PVC. The benchmark pod is co-located with
a DGD frontend through `podAffinity`.

## Targeting a variant

Edit the `env` block in [`perf.yaml`](perf.yaml) and update the `podAffinity` `values` list to contain only the target DGD name, so the benchmark pod is co-located with the correct frontend:

| Variant target | `ENDPOINT` | `TRACE_FILE` |
| --- | --- | --- |
| B200 aggregated (4 GPU) | `k-exaone-2-agg-frontend:8000` | `/model-cache/traces/8k_1k_70kv_chat_new_noschedule_short_15perc.jsonl` |
| B200 disaggregated (8 GPU) | `k-exaone-2-disagg-frontend:8000` | `/model-cache/traces/8k_1k_70kv_chat_new_noschedule_short_15perc.jsonl` |

If you run more than one benchmark in the same namespace, also update
`metadata.name` and `labels.app` so Jobs and artifact directories stay
distinct.

## Dataset

The benchmark replays a
[Mooncake-format](https://github.com/kvcache-ai/Mooncake) trace through
`--custom-dataset-type mooncake_trace`. Each JSONL line describes one request
with `input_length`, `output_length`, and `hash_ids`.

This recipe benchmarks the **8K-ISL / 1K-OSL chat trace**, designed for ~70% prefix
reuse. (The 64K / 400 / 90% shape is the *agentic* trace used by other recipes -- not
this one.) On this topology the achieved reuse is ~8.8%, because the working set is
~42x oversubscribed against TP4's KV capacity. The Git-LFS blob is shared with the
Kimi-K2.6 recipe rather than duplicated, and referenced via a symlink under
[`traces`](traces):

```text
traces/8k_1k_70kv_chat_new_noschedule_short_15perc.jsonl
  -> ../../../kimi-k2.6/perf/traces/8k_1k_70kv_chat_new_noschedule_short_15perc.jsonl
```

The default 15% trace contains 1,805 requests. Its SHA-256 is
`b1221bca72b69f842897f339624306a84857f1b55ea0d866525f94d9ceb9b871`.

> [!IMPORTANT]
> `CONCURRENCY` is **per variant**: `7` for `agg-b200-chat`, `14` for `disagg-b200-chat`.
> The shipped default is 7. Running the disaggregated target at 7 under-loads it and
> reproduces 55 tok/s/GPU rather than the published 85 -- set it to match the variant
> that `ENDPOINT` points at.

## Workflow

```bash
export NAMESPACE=your-namespace
```

### 1. Deploy the DGD

See the deployment instructions in the [recipe README](../README.md).

### 2. Stage the trace on the PVC

Materialize the Git LFS trace files, then copy them through a helper pod that
mounts `model-cache`:

```bash
git lfs pull --include='recipes/kimi-k2.6/perf/traces/8k_1k_70kv_chat_new_noschedule_short_15perc.jsonl'

kubectl run pvc-helper -n ${NAMESPACE} \
  --image=busybox:1.36 --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"helper","image":"busybox:1.36","command":["sleep","3600"],"volumeMounts":[{"name":"model-cache","mountPath":"/model-cache"}]}],"volumes":[{"name":"model-cache","persistentVolumeClaim":{"claimName":"model-cache"}}]}}' \
  --command -- sleep 3600

TRACE_SOURCE="$(git rev-parse --show-toplevel)/recipes/kimi-k2.6/perf/traces/8k_1k_70kv_chat_new_noschedule_short_15perc.jsonl"
kubectl exec -n "${NAMESPACE}" pvc-helper -- mkdir -p /model-cache/traces
kubectl cp "${TRACE_SOURCE}" \
  "${NAMESPACE}/pvc-helper:/model-cache/traces/8k_1k_70kv_chat_new_noschedule_short_15perc.jsonl"
```

Keep `pvc-helper` for fetching artifacts, or delete it after staging.

### 3. Run the benchmark

```bash
kubectl apply -f perf.yaml -n ${NAMESPACE}
kubectl logs -n ${NAMESPACE} -l job-name=k-exaone-2-bench -f
kubectl wait --for=condition=Complete job/k-exaone-2-bench \
  -n ${NAMESPACE} --timeout=10800s
```

The Job runs `python:3.12-slim` and installs `aiperf==0.12.0` at start, together
with `transformers==5.15.1` -- 5.x is required, because this model's tokenizer_config
declares `tokenizer_class: TokenizersBackend`, which does not exist in the transformers
4.x line and makes aiperf fail before it issues a request. The exact version matches the
serving image, so the bench tokenizes the way the server does; aiperf uses the tokenizer
to synthesize the trace prompts, so a different one is a different workload.

### 4. Fetch artifacts

```bash
kubectl cp \
  ${NAMESPACE}/pvc-helper:/model-cache/perf/<epoch>_k-exaone-2-bench \
  ./results
```

### 5. Cleanup

```bash
kubectl delete job k-exaone-2-bench -n ${NAMESPACE}
kubectl delete pod pvc-helper -n ${NAMESPACE}
```

## Running a concurrency sweep

`perf.yaml` runs one `CONCURRENCY` value. Restart the workers to clear engine KV
state and Dynamo frontend/router state between independent runs:

```bash
kubectl delete job k-exaone-2-bench -n ${NAMESPACE} --ignore-not-found

DGD=k-exaone-2-agg            # or k-exaone-2-disagg
kubectl delete pods -n ${NAMESPACE} \
  -l nvidia.com/dynamo-graph-deployment-name=${DGD}
kubectl wait --for=condition=Ready pod -n ${NAMESPACE} \
  -l nvidia.com/dynamo-graph-deployment-name=${DGD} \
  --timeout=7200s

# Update CONCURRENCY in perf.yaml before each run.
kubectl apply -f perf.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/k-exaone-2-bench \
  -n ${NAMESPACE} --timeout=10800s
```

Do not compare partial runs. A completed run must account for successful,
errored, and unfinished requests before reporting aggregate throughput.

## Tunable environment variables

| Variable | Default | Notes |
| --- | --- | --- |
| `ENDPOINT` | `k-exaone-2-agg-frontend:8000` | Change per DGD variant |
| `TRACE_FILE` | `/model-cache/traces/8k_1k_70kv_chat_new_noschedule_short_15perc.jsonl` | 1,805-request 15% chat trace |
| `CONCURRENCY` | `7` | per-variant operating point: **7 for `agg-b200-chat`, 14 for `disagg-b200-chat`**. Set it to match the variant `ENDPOINT` points at, or the result will not reproduce the published row. |
| `TARGET_MODEL` | `LGAI-EXAONE/K-EXAONE-2.0-750B-A37B-NVFP4` | Must match `--served-model-name` |

## Artifacts

Results are written to:

```text
/model-cache/perf/<epoch>_<job-name>/
  warmup/
  K-EXAONE-2.0-750B-A37B-NVFP4_trace_c<concurrency>_<timestamp>/
    profile_export_aiperf.json
    inputs.json
    ...
```
