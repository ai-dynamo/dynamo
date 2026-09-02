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

- Every run has a unique UTC identifier and an immutable directory under
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
  zero, observes Planner alone change it to one, keeps admission closed through
  readiness, drains every request, and returns the lease to zero.
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

The default dataset is the converted GSM8K test split at
`../../../../../../datasets/gsm8k/batch-gateway/gsm8k-main-test.jsonl`.
The source file remains read-only; each run writes its exact normalized slice
into that run's raw artifact directory.
