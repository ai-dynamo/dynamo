<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Nemotron 3.5 Super VL Benchmark Recipe

A single [AIPerf](https://github.com/ai-dynamo/aiperf) Job,
[`perf.yaml`](perf.yaml), covers the Nemotron 3.5 Super VL DGD. Set `ENDPOINT`,
`TARGET_MODEL`, and `CONCURRENCY` for the target variant.

The Job waits for the target model on the Dynamo frontend, generates the
synthetic agentic workload in-job, runs a short warmup, benchmarks at one
`CONCURRENCY` value, and writes raw artifacts to the shared `model-cache`
persistent volume claim (PVC). The benchmark pod is co-located with a DGD
frontend through `podAffinity`.

No trace file has to be staged on the PVC.

## Targeting a Variant

Edit the `env` block in [`perf.yaml`](perf.yaml) with the values from the target
row. Also update the `podAffinity` `values` list to contain only the target DGD
name so the benchmark pod is co-located with the correct frontend.

| Variant | DGD affinity | `ENDPOINT` | `TARGET_MODEL` | `CONCURRENCY` |
| --- | --- | --- | --- | --- |
| B200 aggregate agentic | `nemotron-35-super-vl-agg-b200` | `nemotron-35-super-vl-agg-b200-frontend:8000` | `nvidia/nemotron_35_super_conservative_fp8kv` | `192` |
| GB200 aggregate agentic | `nemotron-35-super-vl-agg-b200` | `nemotron-35-super-vl-agg-b200-frontend:8000` | `nvidia/nemotron_35_super_conservative_fp8kv` | `192` |
| H200 aggregate agentic | `nemotron-35-super-vl-agg-b200` | `nemotron-35-super-vl-agg-b200-frontend:8000` | `nvidia/nemotron_35_super_conservative_fp8kv` | `32` |

<!--
When adding a variant, add its values to this table and keep the matching
defaults in `perf.yaml` on the recommended target. If multiple benchmark Jobs
run in one namespace, give each Job a distinct `metadata.name` and
`metadata.labels.app` so their logs and artifact directories remain separate.
-->

## Workload

The benchmark is fully synthetic -- a single `aiperf profile` invocation, with no
trace file to stage and no dataset generation step.

The parameters approximate the Terminal Bench 2.1 mooncake trace
(`qwen_3_6_35b_terminal_bench_2_1_base_mooncake`, 32,887 requests across 712
sessions), whose median request is 42,684 input tokens of which 42,240 are a
cache hit (99.2% token-weighted), with a median 417-token output:

| Component | Tokens | Flag |
| --- | --- | --- |
| Shared, cacheable prefix | 42,240 | `--shared-system-prompt-length` |
| Fresh tokens per request | 444 | `--isl` |
| Output | 417 | `--osl` |

One prefix shared by every request, the same approximation the gemma4-31b recipe
uses. It reproduces the TTFT, throughput and latency of a heavily-cached agentic
workload without replaying 32,887 requests. It does not reproduce the trace's KV
footprint, whose reuse is spread across many growing per-session contexts rather
than one prefix, nor its input-length tail (real p90 is 149K tokens) or its
output-length tail (2.5% of requests carry 78% of all output tokens).

`SHARED_PROMPT_LEN` must stay consistent with
`kv_cache_config.mamba_state_config.additional_snapshot_offsets_from_start` in
the deployment: a Nemotron-H hybrid can only reuse a prefix it has snapshotted
the Mamba state for, and prompts shorter than the offset are not cached at all.

Measured on 2x B200 at `CONCURRENCY=64`, 192 requests, with the MTP
configuration this recipe ships:

| Metric | MTP on | MTP off |
| --- | --- | --- |
| Prefix cache read | 96.86% | 96.86% |
| Output token throughput | 1,449 tok/s | 1,447 tok/s |
| Output tokens/sec/user | 79.5 | 30.6 |
| Inter-token latency (mean) | 15.1 ms | 15.9 ms |
| TTFT (mean / p99) | 10.8 s / 30.3 s | 9.5 s / 21.3 s |
| Request latency (mean) | 17.1 s | 16.1 s |

MTP buys per-user token rate, not aggregate throughput, and costs tail latency
at this concurrency.

## Workflow

```bash
export NAMESPACE=your-namespace
```

### 1. Deploy Nemotron 3.5 Super VL

Follow the deployment instructions in the [Nemotron 3.5 Super VL recipe README](../README.md)
and wait for the selected DGD to become ready. Use the DGD affinity value from
the target table when configuring `perf.yaml`.

### 2. Start a helper pod

Used to fetch the benchmark artifacts afterwards:

```bash
kubectl run pvc-helper -n ${NAMESPACE} \
  --image=busybox:1.36 --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"helper","image":"busybox:1.36","command":["sleep","3600"],"volumeMounts":[{"name":"model-cache","mountPath":"/model-cache"}]}],"volumes":[{"name":"model-cache","persistentVolumeClaim":{"claimName":"model-cache"}}]}}' \
  --command -- sleep 3600
```

### 3. Run the benchmark

Delete any previous benchmark Job before creating a run. Kubernetes does not
allow updates to a Job's pod template after the Job is created.

```bash
kubectl delete job nemotron-35-super-vl-bench -n ${NAMESPACE} --ignore-not-found
kubectl create -f perf.yaml -n ${NAMESPACE}
kubectl logs -n ${NAMESPACE} -l job-name=nemotron-35-super-vl-bench -f
kubectl wait --for=condition=Complete job/nemotron-35-super-vl-bench \
  -n ${NAMESPACE} --timeout=10800s
```

The Job uses `nvcr.io/nvidia/ai-dynamo/aiperf:0.11.0` directly. It does not
install or patch AIPerf at runtime.

### 4. Fetch the artifacts

```bash
kubectl cp \
  ${NAMESPACE}/pvc-helper:/model-cache/perf/<epoch>_nemotron-35-super-vl-bench \
  ./results
```

### 5. Clean up

```bash
kubectl delete job nemotron-35-super-vl-bench -n ${NAMESPACE}
kubectl delete pod pvc-helper -n ${NAMESPACE}
```

## Concurrency Sweep

`perf.yaml` runs one concurrency value at a time. Restart the TensorRT-LLM
workers and Dynamo frontend between independent points to clear KV-cache and
router state:

```bash
kubectl delete job nemotron-35-super-vl-bench -n ${NAMESPACE} --ignore-not-found

DGD=nemotron-35-super-vl-agg-b200 # Choose a DGD affinity value from the target table.
kubectl delete pods -n ${NAMESPACE} \
  -l nvidia.com/dynamo-graph-deployment-name=${DGD}
kubectl wait --for=condition=Ready pod -n ${NAMESPACE} \
  -l nvidia.com/dynamo-graph-deployment-name=${DGD} \
  --timeout=7200s

# Update CONCURRENCY in perf.yaml before each run.
kubectl create -f perf.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/nemotron-35-super-vl-bench \
  -n ${NAMESPACE} --timeout=10800s
```

Do not compare partial runs. A completed run must account for successful,
errored, and unfinished requests before reporting aggregate throughput.

## Tunable Environment Variables

| Variable | Initial value | Notes |
| --- | --- | --- |
| `ENDPOINT` | `nemotron-35-super-vl-agg-b200-frontend:8000` | Change per DGD variant |
| `SHARED_PROMPT_LEN` | `42240` | Cacheable prefix; keep in sync with the Mamba snapshot offset |
| `FRESH_ISL` | `444` | New tokens per request |
| `OSL` | `417` | Median output length of the source trace |
| `CONCURRENCY` | `192` | B200 starting point; sweep and record a value for GB200 or H200 |
| `TARGET_MODEL` | `nvidia/nemotron_35_super_conservative_fp8kv` | Change per DGD variant; must match `--served-model-name` |

## Artifacts

Results are written under:

```text
/model-cache/perf/<epoch>_nemotron-35-super-vl-bench/
  warmup/
  <model-name>_trace_c<concurrency>_<timestamp>/
    profile_export_aiperf.json
    inputs.json
    ...
```
