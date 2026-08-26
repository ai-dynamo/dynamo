<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GPU Hold Discipline

GPU-hours accrue from allocation to teardown, not from benchmark duration. Analysis, hypothesis generation,
challenger review, plan authoring, artifact writing, and report work are CPU-only: they never justify holding an
allocated deployment by themselves.

## Break-Even Rule

Hold a deployment through offline (non-GPU) work only when BOTH hold:

- the expected remaining offline time times the allocated GPU count is less than the expected cost of tearing down
  and recreating the deployment (estimate the recreate cost from this engagement's own observed deploy and
  weight-load times); and
- a next GPU-dependent action on that deployment is already authorized (an approved candidate awaiting benchmark, a
  required confirmation, or a pre-authorized pilot). Pending review or open-ended analysis is not an authorized
  next action.

Otherwise tear down before starting the offline work and record `torn_down_at` in the deployment ledger.

## Hold TTL

Every justified hold carries a recorded time-to-live derived from the break-even arithmetic. When the TTL expires
before the next GPU action starts, tear down; re-justify any further hold with fresh arithmetic. Warm-state parity
for a planned comparison may set the TTL but never makes a hold indefinite.

## Accounting

Record each hold decision (arithmetic, TTL, outcome) in the deployment ledger or `reasoning_transcript.md`. Idle
allocation — GPUs held with no GPU-dependent action running — is a named cost: report cumulative idle GPU-hours
alongside total spend at every iteration boundary and in every stop-request.
