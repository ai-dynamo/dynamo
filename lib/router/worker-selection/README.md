<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-worker-selection

`dynamo-worker-selection` picks one worker rank for a request from a table of candidates. A policy is a chain of filters and scorers followed by a picker, each of which a plugin can supply; the crate also holds the candidate rows and views those plugins read, conditional disaggregation (whether a request needs a separate prefill worker), and the LoRA filter. Plugins reach it through `dynamo_kv_router::plugins::prelude`.

## Worker-weight control

`WorkerPicker::worker_weights` opts a picker into in-memory traffic-weight control.
`WorkerWeights` stores current membership, non-default overrides, and a revision.
`WorkerWeightUpdate` applies an atomic batch to current members. `WorkerWeightConfig`
reports effective weights. The host calls control methods between selections and excludes
zero-weight members from queue eligibility. The picker reads `weight(worker_id)` and must
also exclude zero-weight candidates. These weights are relative request shares, have no units,
and last until an override reset, observed removal, or router rebuild. Unspecified IDs read 1.
The host reconciles discovered membership, not request-specific candidate membership.
