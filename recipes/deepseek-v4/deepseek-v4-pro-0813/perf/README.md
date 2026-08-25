<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4-Pro-0813 — performance benchmark

AIPerf **Mooncake trace replay** against a deployed 0813 DGD, using the shared
DeepSeek-V4 agentic traces in [`../../perf/traces/`](../../perf/traces/).
The traces are not duplicated here.

## Variants

| Target | `ENDPOINT` | Validated `CONCURRENCY` |
|---|---|---|
| AGG H200 (agentic) | `deepseek-v4-pro-0813-vllm-h200-agg-agentic-frontend:8000` | **2** |
| DisAgg H200 (agentic) | `deepseek-v4-pro-0813-vllm-h200-disagg-agentic-frontend:8000` | **7** (gate crossing ≈7.95) |
| AGG H200 (1M) | `deepseek-v4-pro-0813-vllm-h200-agg-1m-frontend:8000` | 1 |
| DisAgg H200 (1M) | `deepseek-v4-pro-0813-vllm-h200-disagg-1m-frontend:8000` | 1 |

`TARGET_MODEL` is `deepseek-ai/DeepSeek-V4-Pro-0813` for all four.

When switching variants, change **both** `ENDPOINT` and the `podAffinity`
`nvidia.com/dynamo-graph-deployment-name` in `perf.yaml` — they must name the
same DGD, or the Job will not co-locate with the frontend it is measuring.

## Stage traces

A fresh clone has only LFS pointers:

```bash
git lfs pull --include="recipes/deepseek-v4/perf/traces/*.jsonl"
```

Copy them onto the model-cache PVC at `/model-cache/traces/` — see
[`../../perf/README.md`](../../perf/README.md) ("Stage Traces").

## Run

```bash
kubectl apply -f perf.yaml
kubectl logs -f job/dsv4-pro-0813-bench
```

One concurrency per Job. For a sweep, run them sequentially and reset DGD
worker/frontend state between independent rows so prefix-cache state does not
carry over.

## SLA gate

**E2E ≥ 50 tok/s/user AND TTFT p50 < 5 s, jointly**, where

```
E2E tok/s/user = OSL / (TTFT + OSL × ITL)
```

i.e. the per-user token rate *including* time-to-first-token. The looser
decode-only rate (`1000 / ITL`) is not the gate — at 64K input, prefill is
latency the user actually experiences.

The 1M variants are a **batch capability, not interactive** (TTFT is minutes at
that length) and are not gated on throughput.

## Results

| Workload | Recipe | SKU | Concurrency | System output tok/s/gpu | User output tok/s (P50) | TTFT P50 (ms) |
| -------- | ------ | --- | ----------- | ----------------------- | ----------------------- | ------------- |

> [!NOTE]
> Not yet populated -- no Mooncake trace-replay results have been collected for this
> checkpoint. Run the Job above and fill this in, then mirror it into the top-level
> [`../README.md`](../README.md).

## Note on comparability

The operating points quoted in the deploy configs were measured with AIPerf's
**synthetic** agentic profile (`--shared-system-prompt-length 57600`,
`--synthetic-input-tokens-mean 6400`, `--output-tokens-mean 400`, verified ISL
64,004, 90% measured prefix reuse), not with these Mooncake traces. The traces
replay a different request-length distribution, so absolute numbers from this
Job will not reproduce those figures exactly. Use the traces for
variant-to-variant comparison on equal footing; use the synthetic profile to
reproduce the published operating points.

Run-to-run variance measured on this setup is **17%**. Treat any delta smaller
than that as noise, and repeat to N≥3 before quoting a threshold crossing.
