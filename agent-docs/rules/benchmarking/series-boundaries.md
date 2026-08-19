<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Benchmark Series Boundaries

A benchmark series is the set of candidate runs governed by one frozen evaluation contract. Same-series gain, loss,
Pareto, and promotion claims require that contract to remain comparable.

## Fixed Comparison Semantics

Freeze these fields in `benchmark_plan.json` for a series:

- workload source, content hash, and any declared transformation;
- trace timestamps or other schedule semantics;
- load policy and its concurrency, request-rate, request-count, or duration contract;
- input/output shape, prompt formatting, prefix/cache-reuse semantics, and random seed policy;
- endpoint API, streaming behavior, served model identity, and tokenizer;
- warmup, measured phases, repetitions, and confidence policy;
- metric definitions, percentiles, SLO thresholds, and optimization directions; and
- AIPerf runtime or source version when a version change could alter request generation or measurement semantics.

Changing any of these semantics starts a new benchmark series. Fixed-schedule trace replay, concurrency capacity
search, and request-rate capacity search are separate series.

## Candidate Changes Within A Series

Endpoint addresses, Kubernetes object names, and artifact paths may change to reach a newly deployed candidate. DGD,
router, topology, replica, cache, and backend-engine settings are candidate variables—not automatic series boundaries—
when the target workload permits them and the benchmark contract remains frozen. Record active GPU count and required
normalization whenever resource allocation changes.

A speculative-decoding, routing, batching, or cache setting may therefore be an optimization knob in this workflow. Do
not classify it as a series boundary merely because another benchmark program treats it as a separate submission
track.

## Target And Cross-Series Changes

Changing the served model, framework family, precision, hardware class, or another target-fixed constraint is outside
the current optimization series. If the user explicitly requests such a comparison, create and label a separate series
with its own baseline.

Cross-series results may guide target selection or a future hypothesis. Never present them as a measured gain over the
current series, splice them into its history, or use them to promote its candidate.
