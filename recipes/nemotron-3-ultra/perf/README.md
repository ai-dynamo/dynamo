<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Nemotron-3-Ultra Benchmark Recipe

A single AIPerf trace-replay Job — [perf.yaml](perf.yaml) — covers the Nemotron-3-Ultra DGD variants. The benchmark is identical across variants; change `ENDPOINT`, `TRACE_FILE`, `CONCURRENCY`, and `TARGET_MODEL` in the Job env block to target a specific server.

The Job waits for `GET /v1/models` on the DGD frontend to return `TARGET_MODEL`, runs a short warmup, then replays the configured Mooncake-format trace at one `CONCURRENCY` value. Artifacts are written to the shared model-cache PVC under `/opt/models/perf`.

Benchmark rows assume the vLLM DGD manifests in this recipe, including the CUDA13 image and worker runtime settings:

```text
image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0-nemotron-ultra-dev.1
VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel
--no-enable-flashinfer-autotune
```

The bench pod is co-located with the DGD frontend through pod affinity on the frontend host. If you run more than one benchmark in the same namespace, also update `metadata.name` and `labels.app` so Jobs and artifact directories stay distinct.

## Targeting a Variant

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
| `AIPERF_VERSION` | `0.8.0` | Matches the saved Ultra benchmark profile exports |
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
