<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Batch Gateway Planner POC Experiment

## Goal

Measure the stock Batch Gateway path, then prove that Planner can ingest job and
dispatcher state and safely control llm-d Async batch admission with renewable,
fail-closed leases.

## Hypothesis

The native Planner tick can preserve durable Batch Gateway demand when optional
serving telemetry is unavailable, recover the owned worker from zero replicas,
and safely control llm-d Async admission with renewable fail-closed leases. The
same evidence harness can preserve enough request, progress, Kubernetes,
metric, and log state to distinguish Planner actions from experiment setup and
support later online-traffic comparisons.

## Success Criteria

- Every run has a unique UTC identifier and a write-once run directory under
  `results/raw/`.
- The submitted JSONL contains a deterministic GSM8K slice with one model,
  temperature, and output-token limit.
- Progress observations preserve total, completed, and failed counts until a
  terminal state.
- Optional online traffic records per-request HTTP status, Time To First Token
  (TTFT), and end-to-end latency.
- The run captures the effective pod images, referenced ConfigMaps, selected
  pod logs, Kubernetes client/server versions, and any configured metric
  endpoints.
- The command exits nonzero when preflight, workload execution, online load, or
  required result retrieval fails.
- A native scale-from-zero treatment begins with the authoritative DGDSA at
  zero, correlates the Planner decision/scale log with its `0 -> 1` transition
  while the experiment driver performs no in-window mutation, keeps admission
  closed through readiness, drains every request, and returns the lease to zero.
- Native evidence records continuous DGDSA, worker readiness, Planner decision,
  Redis lease, and Async counter transitions and is checked by a machine
  verifier.
- Credential values are neither read from the Hugging Face environment nor
  written to artifacts.

## Scope and Environment

The experiment targets a caller-selected namespace, model `Qwen/Qwen3-0.6B`,
and the existing Batch Gateway, Valkey, llm-d Async, and Dynamo deployment. The
workload harness itself never mutates Kubernetes resources. In native mode, the
deployed Planner is the only experiment control plane: it publishes leased
Redis decisions and scales the exact owned worker DGDSA through its normal tick.

The author's workspace used a converted GSM8K test split outside this
repository. A fresh checkout must prepare an OpenAI Batch JSONL input and pass
its absolute path with `--dataset`. The source file remains read-only; each run
writes its exact normalized slice into that run's raw artifact directory.

## Current Status

The August 2026 end-to-end native POC run completed successfully. Canonical run
`20260828T213549Z-planner-native-1e3ff8` began with worker DGDSA spec/status
`0/0`, no worker pod, and a Grove/KAI-gated frontend. A new durable Gateway job
caused Planner to publish floor 1/cap 0, scale the owned DGDSA `0 -> 1`, wait for
readiness, open cap 5, drain 100/100 with zero failures, and return floor/cap to
zero. Async dispatched and successful counters both increased by exactly 100;
terminal backlog, in-flight, and queue depth were zero. The machine verifier
passed all 15 assertions and the compiler reported zero data-quality issues.

That result predates the September rebase and does not establish that the
current stack was rerun live. Current-head validation is limited to the focused
unit, render, lint, and checked-in fixture-consistency checks recorded with the
rebased PR stack.

The earlier stock, standalone-controlled, and warm native runs remain useful
comparators. The standalone runner's replica recommendation remains advisory;
only the native Planner path owns Kubernetes scaling.

The September 19 CPU-only Mocker campaign is complete and decision-ready. All
15 reportable runs (five each of online-only, stock Batch drain, and native
Planner control) are valid with no data-quality issues. Stock drain reduced
online goodput by 61.95% versus online-only. Planner-native retained 99.00% of
online-only goodput, improved goodput 160.31% over stock, and reduced p99 TTFT
95.92% versus stock while completing every Batch request. Native Batch rate was
15.49% lower than stock, the intended tradeoff for preserving the online SLO.
See the [Planner Gym impact report](reports/20260919-planner-gym-batch-impact.md).

## Key Evidence

- [Baseline input research](research/20260828-baseline-inputs.md)
- [Local preflight report](reports/20260828-local-preflight.md)
- [Control-loop runner report](reports/20260828-control-loop-runner.md)
- [Stock live baseline report](reports/20260828-stock-live-baseline.md)
- [Planner-controlled live report](reports/20260828-controlled-live.md)
- [Native autonomous E2E report](reports/20260828-native-planner-e2e.md)
- [Planner Gym Batch impact report](reports/20260919-planner-gym-batch-impact.md)

## Next Action

Repeat the same balanced design on the target GPU/model stack, calibrating
admission in tokens or measured service time rather than copying the Mocker's
5.5-RPS mixed-work envelope. Keep the CPU-Mocker campaign as the deterministic
control-plane regression and use the GPU run for production capacity settings.
