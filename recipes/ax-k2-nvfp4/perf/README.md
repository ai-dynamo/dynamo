<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
The runner downloads that exact Git-LFS asset and verifies its checksum, line
count, eligible-request count, and absence of timestamps before sending any
traffic.

## Run KV-aware routing

Deploy the DGD as described in the [model README](../README.md), then apply the
runner and Job:

```bash
export NAMESPACE=your-namespace
kubectl apply -f runner.configmap.yaml -n "${NAMESPACE}"
kubectl apply -f perf.yaml -n "${NAMESPACE}"
kubectl logs -n "${NAMESPACE}" -l job-name=axk2-kv-bench -f
kubectl wait --for=condition=Complete job/axk2-kv-bench \
  -n "${NAMESPACE}" --timeout=21600s
```

The Job first replays the first 32 eligible trace records as a warm-cache
burst, then starts the measured replay from the beginning. Results and
frontend metric snapshots are written under:

```text
/model-cache/perf/ax-k2-nvfp4/kv/<UTC-run-id>/
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
