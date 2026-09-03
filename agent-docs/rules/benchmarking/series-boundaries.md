<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Benchmark Series Boundaries

A benchmark should answer a specific question about the current candidate's performance. Do not reuse one frozen
benchmark for every candidate by default. Change the load shape, operating range, or metric focus when a different
experiment would better expose the behavior under investigation.

## Comparable Runs

A benchmark series is the set of runs used for one direct comparison. Freeze the benchmark only for those runs. Keep
fixed:

- request content, shape, formatting, scheduling, and cache-reuse behavior;
- load policy and its concurrency, request-rate, request-count, or duration contract;
- endpoint behavior, served model, tokenizer, warmup, and measured phases; and
- metric definitions, SLOs, run policy, and benchmark-tool behavior.

Treat requested token counts as workload-shape targets unless the user explicitly requires exact counts. Small,
explainable differences from chat templates, headers, special tokens, tokenizer accounting, or natural stopping remain
comparable when they do not materially change the performance question. Record the difference and continue; do not
invalidate or rerun a benchmark solely to force an exact count.

If a question requires a different benchmark, start a new series and rerun any reference needed for that comparison.
Do not calculate gains, losses, or Pareto relationships across series. Multiple series may inform promotion, but each
claim must be supported by comparable runs. Cross-series results may motivate the next experiment.
