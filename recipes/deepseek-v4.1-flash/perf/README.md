<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4.1-Flash vLLM benchmark

A single [AIPerf](https://github.com/ai-dynamo/aiperf) trace-replay Job —
[`perf.yaml`](perf.yaml) — covers all four vLLM DGDs. Set `ENDPOINT` and
`CONCURRENCY` for the target.

The Job waits for the target model on the DGD frontend, then replays the trace
at one `CONCURRENCY` value and writes raw artifacts to the shared
`shared-model-cache` PVC. The benchmark pod is co-located with the target DGD
frontend through `podAffinity`.

This is the SGLang lane's sibling; the SGLang targets of this model are day-0
and carry no benchmark.

## Targeting a variant

Edit the `env` block in [`perf.yaml`](perf.yaml). The `podAffinity` already
lists every deployment name, so it needs no edit: only one target is deployed
at a time and the Job lands beside whichever frontend exists.

| Variant | `ENDPOINT` | Validated `CONCURRENCY` |
| --- | --- | --- |
| B200 aggregated | `dsv41-flash-vllm-b200-agg-frontend:8000` | `88` |
| B200 disaggregated | `dsv41-flash-vllm-b200-disagg-frontend:8000` | `160` |
| GB200 aggregated | `dsv41-flash-vllm-gb200-agg-frontend:8000` | `88` |
| GB200 disaggregated | `dsv41-flash-vllm-gb200-disagg-frontend:8000` | `168` |

Each concurrency is that variant's best SLA-clearing rung, not the peak of the
throughput curve. Both lanes end on the per-user token rate rather than on
TTFT, so a higher concurrency produces more total tokens while missing the
floor. Running more than one benchmark in the same namespace also needs a
distinct `metadata.name` and `labels.app`, so Jobs and artifact directories
stay separate.

## SLA

The recipes were tuned against two floors, both measured at p50:

| Metric | Floor |
| --- | --- |
| Output token throughput per user | >= 50 tok/s |
| Time to first token | < 5000 ms |

> [!NOTE]
> `total_output_tokens` counts the non-reasoning subset only. Add
> `total_reasoning_tokens` to get what the GPU actually produced.

> [!IMPORTANT]
> Throughput on this workload repeats to about 3 percent across identical runs.
> Do not read a throughput difference smaller than that as a result. Latency
> percentiles are far more stable and resolve differences throughput cannot.

## Dataset

The benchmark replays a
[Mooncake-format](https://github.com/kvcache-ai/Mooncake) trace through
`--custom-dataset-type mooncake_trace`. Each JSONL line describes one request
with `input_length`, `output_length`, and `hash_ids`.

This is the same 64K-ISL / 400-OSL / 90%-KV-reuse agentic trace the other
agentic recipes use, so rather than duplicate the Git LFS blob it is referenced
from the DeepSeek-V4 family recipes through a symlink under [`traces`](traces):

```text
traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl
  -> ../../../deepseek-v4/perf/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl
```

The trace contains 3,541 requests. Its SHA-256 is
`f20d3f2bc83dd1306cda659fbe34e7c4d85ca5497626c98bc0b1c4d2211379d0`.

Measured against this corpus, the input median is about 67,600 tokens and the
output median is about 400. The input p99 is above 450,000 and the longest row
exceeds 900,000, which is why the prefill knobs in the disaggregated recipes
target the tail rather than the median.

## Workflow

```bash
export NAMESPACE=your-namespace
```

### 1. Deploy the DGD

See the deployment instructions in [the recipe README](../README.md).

### 2. Stage the trace on the PVC

Materialize the Git LFS trace file, then copy it through a helper pod that
mounts `shared-model-cache`:

```bash
git lfs pull --include='recipes/deepseek-v4/perf/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl'

kubectl run pvc-helper -n ${NAMESPACE} \
  --image=busybox:1.36 --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"helper","image":"busybox:1.36","command":["sleep","86400"],"volumeMounts":[{"name":"shared-model-cache","mountPath":"/shared-model-cache"}]}],"volumes":[{"name":"shared-model-cache","persistentVolumeClaim":{"claimName":"shared-model-cache"}}]}}' \
  --command -- sleep 86400

kubectl wait --for=condition=Ready pod/pvc-helper -n "${NAMESPACE}" --timeout=300s
TRACE_SOURCE="$(git rev-parse --show-toplevel)/recipes/deepseek-v4/perf/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl"
kubectl exec -n "${NAMESPACE}" pvc-helper -- mkdir -p /shared-model-cache/traces
kubectl cp "${TRACE_SOURCE}" \
  "${NAMESPACE}/pvc-helper:/shared-model-cache/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl"
```

Keep `pvc-helper` for fetching artifacts afterwards, or delete it once staging
is done. It sleeps for 24 h so it outlives the benchmark Job.

### 3. Run the benchmark

```bash
kubectl apply -f perf.yaml -n ${NAMESPACE}
kubectl logs -n ${NAMESPACE} -l job-name=dsv41-flash-vllm-bench -f
kubectl wait --for=condition=Complete job/dsv41-flash-vllm-bench -n ${NAMESPACE} --timeout=86400s
```

Results land under `/shared-model-cache/perf/<epoch>_<job-name>/trace_c<CONCURRENCY>/`.

### 4. Between runs

Every run replays the same corpus, so a second run against a warm deployment
inherits the first run's cache instead of the trace's own reuse and the two are
not comparable. Clear worker and router KV state between independent runs,
either by deleting and re-applying the DGD or by resetting the prefix cache on
every worker.

> [!WARNING]
> A disaggregated prefill worker may refuse a prefix-cache reset with "some
> blocks are in use" while a transfer to decode is still outstanding. That is a
> property of disaggregation, not a fault; retry once the transfer drains.
