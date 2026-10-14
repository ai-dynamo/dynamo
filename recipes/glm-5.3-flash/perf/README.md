<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## GLM-5.3-Flash benchmark

[perf.yaml](perf.yaml) contains an AIPerf ConfigMap and a CPU-only Kubernetes
Job for text trace replay against the SGLang recipes. It follows the
[Kimi-K3 benchmark](../../kimi-k3/perf/perf.yaml) configuration format and shares
the agentic trace used by [GLM-5.2](../../glm-5.2/perf/README.md).

The default target is GB200 disaggregated serving: two TP4/EP4 prefill workers
and one TP4 decode worker, using 12 GPUs. The Job waits for model
registration, runs `CONCURRENCY` warmup requests at that concurrency, then replays
the trace at the same concurrency (default `64`). Results are saved on the
`shared-model-cache` PVC. This workload measures text serving, including prefix
reuse; it does not send images or video. Benchmark warmup is routed normally and
does not guarantee that every worker receives a warmup request.

## Select a target

Update `INFERENCE_URL` and the DGD name in the Job's frontend pod-affinity selector:

| SGLang target | GPUs | DGD name | `INFERENCE_URL` |
| --- | --- | --- | --- |
| GB200 disaggregated (default) | 12 (2P1D) | `glm53-flash-sglang-disagg-gb200` | `http://glm53-flash-sglang-disagg-gb200-frontend:8000/v1/chat/completions` |
| GB200 aggregated | 4 | `glm53-flash-sglang-agg-gb200` | `http://glm53-flash-sglang-agg-gb200-frontend:8000/v1/chat/completions` |
| B200 aggregated | 4 | `glm53-flash-sglang-agg-b200` | `http://glm53-flash-sglang-agg-b200-frontend:8000/v1/chat/completions` |

All three use `RadixArk/GLM-5.3-Flash-NVFP4`. Both the served checkpoint and
benchmark tokenizer are pinned to revision
`f46cf340d35a22d0d83d0c1dac8957cf2b1bcd35`. The deployment workers and
[SGLang model-download Job](../sglang/model-cache/model-download.yaml) set
`--revision`; AIPerf sets `benchmark.tokenizer.revision` through
`TOKENIZER_REVISION`. AIPerf sends the served model name to the endpoint;
the deployment controls which checkpoint is loaded.
The benchmark uses AIPerf 0.13.0, bootstraps pip with the bundled `ensurepip`,
and installs Transformers 5.19.0 with Tokenizers 0.23.2 to read this checkpoint's `TokenizersBackend` tokenizer.
Package installation requires network access. Model assets must already be on
the shared cache as described in the [SGLang deployment guide](../README.md#sglang).

The benchmark pod requests no GPUs and is co-located with the selected frontend.
It tolerates the GB200 frontend's `kubernetes.io/arch=arm64:NoSchedule` taint and
GPU `NoSchedule` taints for either target. Adjust tolerations for additional
cluster-specific taints.

The deployment bases use real speculative verification by default. The perf Job
does not enable synthetic acceptance or change serving flags. Record any optional
synthetic-acceptance component separately when comparing results.

For B200 or GB200 synthetic measurements, first follow the
[synthetic deployment procedure](../README.md#synthetic-acceptance) and wait for
the replacement workers to become ready. It sets acceptance length `2.5657`,
method `match-expected`, and token mode `real-draft-token`.

Before creating the Job, set its `CONCURRENCY` environment variable in
[perf.yaml](perf.yaml) for the corresponding published synthetic result:

| Target | `CONCURRENCY` |
| --- | ---: |
| B200 aggregated | `56` |
| GB200 aggregated | `56` |
| GB200 disaggregated (2P1D) | `128` |

The default `64` does not match those rows. Change `INFERENCE_URL` and the
frontend pod-affinity selector as described above for the selected target.

## Trace and load settings

The [trace symlink](traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl)
references the shared 64K-input / 400-output / 90%-KV-reuse agentic profile's
15% subset. These are workload profile names, not fixed lengths: individual
requests have different input and output lengths. The source has 3,541 records
and SHA-256 `f20d3f2bc83dd1306cda659fbe34e7c4d85ca5497626c98bc0b1c4d2211379d0`.

- `CONCURRENCY`: warmup request count and concurrency for both phases; default `64`.
- `MAX_ISL`: input-length limit for trace synthesis; default `500000`.
- `CAP_OSL`: output-length cap; default `12000`.
- `TRACE_FILE`: staged trace path on the shared PVC.
- `ARTIFACT_DIR`: output directory; default `/model-cache/aiperf-artifacts`. Set a unique directory for each run.

The request count is computed from records within `MAX_ISL` (3,493 with the
default limit). The trace has no
timestamps, so replay uses concurrency-based timing. Keep the input and output
limits within the target deployment's configured context length.

## Run

Run commands from the repository root. Set your context and namespace, and deploy
the target from the [SGLang guide](../README.md#sglang) first:

```bash
export CONTEXT=your-context
export NAMESPACE=your-namespace
export DGD=glm53-flash-sglang-disagg-gb200
kubectl --context "$CONTEXT" -n "$NAMESPACE" wait --for=condition=Ready pod \
  -l "nvidia.com/dynamo-graph-deployment-name=$DGD" --timeout=7200s
```

Stage the Git LFS trace through a helper pod mounting `shared-model-cache`:

```bash
git lfs pull --include='recipes/kimi-k2.6/perf/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl'
kubectl --context "$CONTEXT" -n "$NAMESPACE" run glm53-perf-helper \
  --image=busybox:1.36 --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"helper","image":"busybox:1.36","command":["sleep","10800"],"volumeMounts":[{"name":"cache","mountPath":"/model-cache"}]}],"volumes":[{"name":"cache","persistentVolumeClaim":{"claimName":"shared-model-cache"}}]}}' \
  --command -- sleep 10800
kubectl --context "$CONTEXT" -n "$NAMESPACE" wait --for=condition=Ready \
  pod/glm53-perf-helper --timeout=300s
kubectl --context "$CONTEXT" -n "$NAMESPACE" exec glm53-perf-helper -- \
  mkdir -p /model-cache/traces
kubectl --context "$CONTEXT" -n "$NAMESPACE" cp \
  recipes/kimi-k2.6/perf/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl \
  glm53-perf-helper:/model-cache/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl

kubectl --context "$CONTEXT" -n "$NAMESPACE" apply -f recipes/glm-5.3-flash/perf/perf.yaml
kubectl --context "$CONTEXT" -n "$NAMESPACE" logs -f job/glm53-flash-bench
kubectl --context "$CONTEXT" -n "$NAMESPACE" wait --for=condition=Complete \
  job/glm53-flash-bench --timeout=10800s
```

The helper also needs any tolerations required by the storage's eligible nodes.
The Job has no automatic retries, so a failed run is not silently replayed
against warmed caches. Inspect its logs if the completion wait times out.

## Results and repeat runs

Results are written under `ARTIFACT_DIR` on the shared PVC. With the default
`/model-cache/aiperf-artifacts`:

```bash
kubectl --context "$CONTEXT" -n "$NAMESPACE" cp \
  glm53-perf-helper:/model-cache/aiperf-artifacts ./glm53-perf-results
kubectl --context "$CONTEXT" -n "$NAMESPACE" delete job glm53-flash-bench
kubectl --context "$CONTEXT" -n "$NAMESPACE" delete configmap glm53-flash-aiperf-config
kubectl --context "$CONTEXT" -n "$NAMESPACE" delete pod glm53-perf-helper
```

For a concurrency sweep, change `CONCURRENCY`, set a unique `ARTIFACT_DIR`, and
recreate the Job for each point.
Reset worker KV caches and frontend/router state between independent runs, and
wait for all replacement workers to be ready before submitting the next Job.
Keep the warmup policy, trace, tokenizer revision, synthesis limits, and speculative
verification mode identical across comparisons. Preserve the deployed manifest
alongside the AIPerf artifacts so GPU counts and runtime settings are recorded.
