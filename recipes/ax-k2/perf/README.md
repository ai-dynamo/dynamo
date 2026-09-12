<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# A.X-K2-NVFP4 KV-router benchmark

This benchmark uses AIPerf 0.12.0 to replay the 8K-input / 1K-output,
70%-KV-reuse Mooncake no-schedule chat trace against the two-worker TP4
aggregate recipe at concurrency 32.

The source trace contains 12,031 requests and no timestamp fields, so AIPerf
uses concurrency timing rather than fixed-schedule replay. A.X-K2 supports a
262,144-token context. The benchmark applies `max_isl: 256000`, which removes
186 over-limit requests and leaves 11,845 requests; every retained request's
input plus requested output fits the model context.

The trace is shared with the Nemotron-3-Ultra recipe:

```text
recipes/nemotron-3-ultra/perf/traces/
  nim_turbo_8k_1k_70kv_chat_new_noschedule.jsonl
```

Its SHA-256 is
`5f369eb75ce639ad8b05cc209bb534bfedd627e9f7b923de32888155b4c9085a`.
The runner reads the staged Git LFS asset from `TRACE_FILE` on the model-cache
PVC. It verifies the checksum, line count, eligible-request count, and absence
of timestamps before sending any traffic.

## Stage the trace

Run these commands from the repository root after creating the model-cache
PVC. Set `CONTEXT` and `NAMESPACE` to your cluster context and namespace. If
you use an existing PVC, replace `model-cache` in the helper's `claimName`
and in `perf.yaml` with that PVC name.

Materialize the trace from your checked-out Git revision, then copy it through
a helper pod that mounts the PVC:

```bash
git lfs pull --include='recipes/nemotron-3-ultra/perf/traces/nim_turbo_8k_1k_70kv_chat_new_noschedule.jsonl'

kubectl --context "${CONTEXT}" -n "${NAMESPACE}" run axk2-trace-helper \
  --image=busybox:1.36 --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"helper","image":"busybox:1.36","command":["sleep","3600"],"volumeMounts":[{"name":"model-cache","mountPath":"/model-cache"}]}],"volumes":[{"name":"model-cache","persistentVolumeClaim":{"claimName":"model-cache"}}]}}' \
  --command -- sleep 3600
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" wait --for=condition=Ready \
  pod/axk2-trace-helper --timeout=300s

TRACE_SOURCE="$(git rev-parse --show-toplevel)/recipes/nemotron-3-ultra/perf/traces/nim_turbo_8k_1k_70kv_chat_new_noschedule.jsonl"
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" exec axk2-trace-helper -- mkdir -p /model-cache/traces
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" cp "${TRACE_SOURCE}" \
  axk2-trace-helper:/model-cache/traces/nim_turbo_8k_1k_70kv_chat_new_noschedule.jsonl
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" delete pod axk2-trace-helper
```

The benchmark job sets `TRACE_FILE` to this PVC path. It does not download the
trace at runtime.

## Run KV-aware routing

Deploy the DGD as described in the
[A.X-K2 recipe](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/recipes/model-recipes/ax-k2.mdx#deploy), then apply the
runner and Job from the repository root:

For throughput-only speculative-decoding measurements at the measured
acceptance proxy, change the worker `SPECULATIVE_CONFIG` ConfigMap key in
`recipes/ax-k2/vllm/agg-b200-chat/kustomize/base/deploy.yaml` from
`speculative-config` to `speculative-config-synthetic`, then regenerate the
manifest before deploying as described in the
[aggregate instructions](../vllm/agg-b200-chat/README.md#edit-and-render).
This enables vLLM's
`rejection_sample_method: synthetic` with
`synthetic_acceptance_length: 2.12`. Keep the production key for functional or
quality validation because synthetic rejection sampling intentionally forces
acceptance behavior.

```bash
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" apply -f recipes/ax-k2/perf/runner.configmap.yaml
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" apply -f recipes/ax-k2/perf/perf.yaml
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" logs -l job-name=axk2-kv-bench -f
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" wait --for=condition=Complete \
  job/axk2-kv-bench --timeout=21600s
```

The Job first replays the first 32 eligible trace records as a warm-cache
burst, then starts the measured replay from the beginning. Results and
frontend metric snapshots are written under:

```text
/model-cache/perf/ax-k2/kv/<UTC-run-id>/
```

## Round-robin baseline

A routing comparison must start from empty worker and router caches. Delete
the DGD pods, change the frontend command from `--router-mode kv` to
`--router-mode round_robin`, and wait for the replacement frontend and both
workers to become Ready. Then change `ROUTING_MODE` in `perf.yaml` to
`round_robin`, use a distinct Job name, and run the same benchmark unchanged.

Do not compare a warm KV-aware run with a cold round-robin run. Preserve the
DGD manifest, pod logs, AIPerf expanded configs, raw reports, and frontend
`/metrics` snapshots for both runs.

## AIPerf tokenizer workaround

The AIPerf-side Transformers build does not recognize the custom `axk2` model
configuration. The runner therefore downloads only four tokenizer assets from
the same pinned model revision. It places them in an isolated, tokenizer-only
Hugging Face snapshot under `/tmp/axk2-tokenizer-hf`, verifies the 163,840-token
vocabulary, and points AIPerf at the revision-pinned repo ID through that
offline snapshot.

The synthetic snapshot layout is necessary for AIPerf 0.12.0 Mooncake traces:
parallel prompt synthesis forces offline mode and resolves tokenizers through
the Hugging Face cache API, so an arbitrary absolute tokenizer path fails in
the child processes. Keeping `config.json` out of this isolated snapshot also
avoids the unsupported `axk2` model-type lookup. The snapshot exists only in
the benchmark container's `/tmp`; the server still loads the full model from
the shared Hugging Face cache.
