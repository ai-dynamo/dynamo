<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Motif-3 chat-trace benchmarks

Complete the [recipe prerequisites and deployment](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/recipes/model-recipes/motif-3.mdx#deploy)
before benchmarking. Run commands from the repository root, with `NAMESPACE`
set to the deployment namespace.

Stage the complete
[`8k_1k_70kv_chat_new_noschedule_short_15perc.jsonl` trace](https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k2.6/perf/traces/8k_1k_70kv_chat_new_noschedule_short_15perc.jsonl)
at `/shared-model-cache/traces/8k_1k_70kv_chat_new_noschedule_short_15perc.jsonl`.
The disaggregated runner checks its SHA-256 and preserves all 1,805 rows for
measurement.

Both targets default to **real MTP verification**. For synthetic acceptance
length 2.13, change the worker `SPECULATIVE_CONFIG` ConfigMap key from
`speculative-config` to `speculative-config-synthetic` in the deployment source,
regenerate any derived manifest, and redeploy. Change **both worker roles** for
disaggregation; see the [disaggregated recipe](../vllm/disagg-b200-chat/README.md)
for regeneration commands. Use real MTP verification for quality evaluation.

## Aggregated

The [AIPerf Job](perf.yaml) uses concurrency 11 and a six-request warmup:

```bash
kubectl apply -f recipes/motif-3/perf/perf.yaml -n "${NAMESPACE}"
kubectl logs -f job/motif3-aiperf-job -n "${NAMESPACE}"
```

## Disaggregated 3P1D

Use AIPerf 0.13.0 with Transformers 5.19.0, Tokenizers 0.23.2, and
huggingface-hub 1.33.0. The benchmark client needs the model cache mounted at
`/shared-model-cache` and network access to the frontend. Prepare a separate
tokenizer-only cache with the [client helper](prepare_disagg_client.py):

```bash
export HF_HOME=/tmp/motif3-disagg-client-hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python3 recipes/motif-3/perf/prepare_disagg_client.py
```

Before each independent run, clear worker and router state from a terminal
with cluster access:

```bash
kubectl delete pods -n "${NAMESPACE}" \
  -l nvidia.com/dynamo-graph-deployment-name=motif3-disagg-b200
kubectl get pods -n "${NAMESPACE}" \
  -l nvidia.com/dynamo-graph-deployment-name=motif3-disagg-b200 \
  --watch
```

Wait for all five replacement pods (one frontend, three prefill workers, and
one decode worker) to show Ready, then stop the watch with Ctrl+C.

Keep the tokenizer environment variables above in the benchmark client and
run the [benchmark runner](run_disagg.py):

```bash
export INFERENCE_URL=http://motif3-disagg-b200-frontend:8000/v1/chat/completions
export AIPERF_ARTIFACT_DIR="/tmp/motif3-3p1d-c64-$(date -u +%Y%m%dT%H%M%SZ)"
python3 recipes/motif-3/perf/run_disagg.py
```

The runner sends the first 32 eligible trace records as a separate warm-cache
burst at concurrency 32. After all 32 complete without errors, it starts the
complete 1,805-row trace from the beginning at concurrency 64, using
[disagg.yaml](disagg.yaml). Preserve worker and router caches between warmup
and measurement. Warmup and measurement use the same complete trace source
and random seed so their synthetic prefixes match.

`AIPERF_ARTIFACT_DIR` must be a new directory. The runner saves `warmup.yaml`,
`measurement.yaml`, and `protocol.json` there, with separate `warmup/` and
`measurement/` result directories. Warmup requests are excluded from the
measured results.

Divide system output throughput by all eight serving GPUs. Report context-limit
rejections separately from unexpected errors; the measured trace retains
over-limit requests. Preserve the deployment manifest and benchmark artifacts
with the reported results.
