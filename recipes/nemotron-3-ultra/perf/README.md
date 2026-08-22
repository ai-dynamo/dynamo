<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Nemotron-3-Ultra Benchmark Recipe

The Day-0 profiles share one AIPerf trace-replay Job, [perf.yaml](perf.yaml). The Refresh profiles share a second Job, [refresh-perf.yaml](refresh-perf.yaml), and its [runner ConfigMap](refresh-runner.configmap.yaml). There is no profile-specific perf manifest.

The Day-0 Job waits for `GET /v1/models` on the DGD frontend to return `TARGET_MODEL`, runs a short warmup, then replays the configured Mooncake-format trace at one `CONCURRENCY` value. Its artifacts are written to the shared model-cache PVC under `/opt/models/perf`.

Benchmark rows assume the vLLM DGD manifests in this recipe, including the CUDA13 image and worker runtime settings:

```text
image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0-nemotron-ultra-dev.1
VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel
--no-enable-flashinfer-autotune
```

The Day-0 bench pod is co-located with the DGD frontend through pod affinity on the frontend host. If you run more than one benchmark in the same namespace, also update `metadata.name` and `labels.app` so Jobs and artifact directories stay distinct.

## Targeting a Day-0 Variant

Edit the `env` block in [perf.yaml](perf.yaml):

| Variant target | `ENDPOINT` | `TARGET_MODEL` | `TRACE_FILE` | Typical `CONCURRENCY` |
|---|---|---|---|---:|
| B200 AGG chat MTP | `ultra-agg-b200-chat-mtp-frontend:8000` | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` | `/opt/models/traces/nim_turbo_8k_1k_70kv_chat_new_noschedule_short_15perc.jsonl` | 18 |
| B200 AGG chat no-MTP | `ultra-agg-b200-chat-nomtp-frontend:8000` | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` | `/opt/models/traces/nim_turbo_8k_1k_70kv_chat_new_noschedule_short_15perc.jsonl` | 16 |
| B200 AGG agentic MTP | `ultra-agg-b200-agentic-mtp-frontend:8000` | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` | `/opt/models/traces/nim_turbo_64k_400_90kv_agent_new_noschedule_short_15perc.jsonl` | 20 |
| B200 AGG agentic no-MTP | `ultra-agg-b200-agentic-nomtp-frontend:8000` | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` | `/opt/models/traces/nim_turbo_64k_400_90kv_agent_new_noschedule_short_15perc.jsonl` | 8 |
| B200 1P1D agentic no-MTP | `ultra-disagg-b200-1p1d-agentic-nomtp-frontend:8000` | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` | `/opt/models/traces/nim_turbo_64k_400_90kv_agent_new_noschedule_short_15perc.jsonl` | 32 |
| H200 AGG chat MTP | `ultra-agg-h200-chat-mtp-frontend:8000` | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` | `/opt/models/traces/nim_turbo_8k_1k_70kv_chat_new_noschedule_short_15perc.jsonl` | 10 |
| H200 AGG chat no-MTP | `ultra-agg-h200-chat-nomtp-frontend:8000` | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` | `/opt/models/traces/nim_turbo_8k_1k_70kv_chat_new_noschedule_short_15perc.jsonl` | 8 |
| H200 AGG agentic MTP | `ultra-agg-h200-agentic-mtp-frontend:8000` | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` | `/opt/models/traces/nim_turbo_64k_400_90kv_agent_new_noschedule_short_15perc.jsonl` | 8 |
| H200 AGG agentic no-MTP | `ultra-agg-h200-agentic-nomtp-frontend:8000` | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` | `/opt/models/traces/nim_turbo_64k_400_90kv_agent_new_noschedule_short_15perc.jsonl` | 8 |

The default Job is configured for the B200 AGG chat MTP 15% trace at concurrency 18. For other release-style benchmark rows, use the trace/concurrency pair that matches the recipe row being reported.

## Targeting a Refresh Profile

The shared Refresh Job replays all 3541 rows of the 15% agentic Mooncake trace through AIPerf 0.12. Edit its `env` block with one row from this table before applying it:

| Profile | `DGD_NAME` | `CONTEXT` | `CONCURRENCY` | `GPU_COUNT` | `REQUEST_TIMEOUT` | `WARMUP_BURSTS` | Expected requests | Expected errors |
|---|---|---|---:|---:|---:|---:|---:|---:|
| B200 256K | `ultra-agg-b200-256k-2w-kv` | `256k` | 94 | 8 | 3600 | 2 | 3411 | 130 |
| B200 1M | `ultra-agg-b200-1m-2w-kv` | `1m` | 46 | 8 | 3600 | 2 | 3528 | 13 |
| GB200 256K | `ultra-agg-gb200-256k-2w-kv` | `256k` | 96 | 8 | 1200 | 1 | 3411 | 130 |
| GB200 1M | `ultra-agg-gb200-1m-2w-kv` | `1m` | 48 | 8 | 3600 | 2 | 3528 | 13 |
| H200 256K | `ultra-agg-h200-256k-2w-kv` | `256k` | 64 | 16 | 1200 | 2 | 3411 | 130 |
| H200 1M | `ultra-agg-h200-1m-2w-kv` | `1m` | 32 | 16 | 3600 | 2 | 3528 | 13 |
| B200 256K 1P2D | `ultra-disagg-b200-256k-1p2d` | `256k` | 144 | 12 | 3600 | 2 | 3411 | 130 |
| B200 1M 1P1D | `ultra-disagg-b200-1m-1p1d` | `1m` | 38 | 8 | 3600 | 2 | 3528 | 13 |
| GB200 256K 1P2D | `ultra-disagg-gb200-256k-1p2d` | `256k` | 140 | 12 | 1200 | 1 | 3411 | 130 |
| GB200 1M 1P1D | `ultra-disagg-gb200-1m-1p1d` | `1m` | 48 | 8 | 3600 | 2 | 3528 | 13 |
| H200 256K 1P2D | `ultra-disagg-h200-256k-1p2d` | `256k` | 72 | 24 | 1200 | 2 | 3411 | 130 |
| H200 1M 1P1D | `ultra-disagg-h200-1m-1p1d` | `1m` | 30 | 16 | 3600 | 2 | 3528 | 13 |

Set `ENDPOINT` to `<DGD_NAME>-frontend:8000`, `EXPECTED_REQUEST_COUNT` and `EXPECTED_ERROR_COUNT` to the final two columns, and leave `JOB_NAME=perf-nemotron-ultra-refresh`. Then run:

All numeric rows are selected released-1.4.0 reference points.

```bash
kubectl apply -f refresh-runner.configmap.yaml -n ${NAMESPACE}
kubectl apply -f refresh-perf.yaml -n ${NAMESPACE}
kubectl logs -f job/perf-nemotron-ultra-refresh -n ${NAMESPACE}
kubectl wait --for=condition=Complete \
  job/perf-nemotron-ultra-refresh \
  -n ${NAMESPACE} --timeout=21600s
```

Refresh artifacts are written to `/artifacts/perf-dynamo/perf-nemotron-ultra-refresh/<UTC-run-id>/` on `shared-model-cache`. The result contains the AIPerf output, `run_manifest.json`, and `summary.json`.

## Dataset

The benchmark replays a Mooncake-format trace through `aiperf --custom-dataset-type mooncake_trace`. Each JSONL line describes one request with fields such as `input_length`, `output_length`, and `hash_ids`.

Trace files included in this recipe:

| Trace | Rows |
|---|---:|
| `traces/nim_turbo_8k_1k_70kv_chat_new_noschedule_short_15perc.jsonl` | 1805 |
| `traces/nim_turbo_8k_1k_70kv_chat_new_noschedule_short_30perc.jsonl` | 3609 |
| `traces/nim_turbo_8k_1k_70kv_chat_new_noschedule.jsonl` | 12031 |
| `traces/nim_turbo_64k_400_90kv_agent_new_noschedule_short_15perc.jsonl` | 3541 |
| `traces/nim_turbo_64k_400_90kv_agent_new_noschedule_short_30perc.jsonl` | 7082 |
| `traces/nim_turbo_64k_400_90kv_agent_new_noschedule.jsonl` | 23608 |

The 15% and 30% traces are prefix slices, not random samples, so they preserve trace order and cache warmup behavior.

## Replay Policy

The intended replay policy is raw direct Moontrace replay:

- No context-length filtering.
- No OSL clipping.
- No synthetic output-length substitution.
- No `--export-http-trace`.
- No `bad_words` guard.
- `ignore_eos:true` is the only extra input.
- HTTP400 and no-content rows remain benchmark failure evidence.

## Workflow

```bash
export NAMESPACE=your-namespace
```

### 1. Deploy the DGD

See instructions in the [recipe README](../README.md).

### 2. Stage the Traces on the PVC

Spin up a short-lived helper pod that mounts `shared-model-cache` at `/opt/models`, then copy the bundled traces in:

```bash
kubectl run pvc-helper -n ${NAMESPACE} \
  --image=busybox:1.36 --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"helper","image":"busybox:1.36","command":["sleep","3600"],"volumeMounts":[{"name":"model-cache","mountPath":"/opt/models"}]}],"volumes":[{"name":"model-cache","persistentVolumeClaim":{"claimName":"shared-model-cache"}}]}}' \
  --command -- sleep 3600

kubectl exec -n ${NAMESPACE} pvc-helper -- mkdir -p /opt/models/traces
kubectl cp traces/. ${NAMESPACE}/pvc-helper:/opt/models/traces/
```

Keep `pvc-helper` around for fetching artifacts later, or delete it once staging is complete.

### 3. Run the Benchmark

```bash
kubectl apply -f perf.yaml -n ${NAMESPACE}

kubectl logs -n ${NAMESPACE} -l job-name=ultra-bench -f

kubectl wait --for=condition=Complete \
  job/ultra-bench \
  -n ${NAMESPACE} --timeout=7200s
```

### 4. Fetch Artifacts

```bash
kubectl cp ${NAMESPACE}/pvc-helper:/opt/models/perf/<epoch>_ultra-bench ./results
```

### 5. Cleanup

```bash
kubectl delete job ultra-bench -n ${NAMESPACE}
kubectl delete pod pvc-helper -n ${NAMESPACE}
```

## Running a Concurrency Sweep

`perf.yaml` runs a single `CONCURRENCY` value. To measure multiple concurrencies, clear server state between runs so residual KV cache and prefix-cache hits from the previous run do not skew results.

For each concurrency value:

```bash
kubectl delete job ultra-bench -n ${NAMESPACE} --ignore-not-found

DGD=ultra-agg-b200-chat-mtp
kubectl delete pods -n ${NAMESPACE} \
  -l nvidia.com/dynamo-graph-deployment-name=${DGD},nvidia.com/dynamo-component-type=worker

kubectl apply -f perf.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/ultra-bench -n ${NAMESPACE} --timeout=7200s
```

The Job's `wait_for_model_ready` loop handles the worker restart window by polling `/v1/models` until the frontend reports the target model.

## Tunable Environment Variables

Edit the `env` block on the Job:

| Variable | Default | Notes |
|---|---|---|
| `ENDPOINT` | `ultra-agg-b200-chat-mtp-frontend:8000` | DGD frontend service:port |
| `TRACE_FILE` | `/opt/models/traces/nim_turbo_8k_1k_70kv_chat_new_noschedule_short_15perc.jsonl` | Swap to chat or agentic 15%, 30%, or full trace |
| `CONCURRENCY` | `18` | Single value; see [Running a Concurrency Sweep](#running-a-concurrency-sweep) |
| `TARGET_MODEL` | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` | Must match the served model name on the DGD frontend |
| `TOKENIZER_PATH` | `/opt/models/patched/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` | Used by AIPerf tokenization |
| `AIPERF_VERSION` | `0.10.0` | aiperf version used for benchmarking |
| `ROOT_ARTIFACT_DIR` | `/opt/models/perf` | Shared PVC artifact root |

## Artifacts

Results are written to:

```text
/opt/models/perf/<epoch>_<job-name>/
  trace_replay_manifest.json
  warmup/
  NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4_trace_c<concurrency>_<timestamp>/
    profile_export.json
    inputs.json
    ...
```

For release evidence, preserve the AIPerf output directory, Job logs, DGD manifest, image digest, trace SHA, server-shape evidence, and model-cache validation proof.
