<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-worker-selection

Candidate rows in, a pick out. The router's worker selection stage: the plugin API (`WorkerFilter`, `WorkerScorer`, `WorkerPicker`) and the composed `WorkerSelectionPolicy` that runs them over the rows a `CandidateRowSource` yields and returns a `RowPick` or a `SelectionFailure`; the `WorkerSelectionContext` view of the request; the candidate rows (`CandidateData`, `WorkerInputRows`) and their gated views; `CandidatePreparer` and `PreparedRows` for rows built before the request reaches the actor; runtime parameter updates (`RuntimePolicy*`); the selection slice of the router config (`SelectionConfig`); the bounded selection trace; the conditional-disaggregation policies; the LoRA narrowing filter; and the kv-transfer source summary (`kv_transfer`). `lib/kv-router` re-exports the crate at `plugins::worker_selection` and keeps the kernel side: `WorkerSelector`, the request-to-row source, the config- and projection-derived columns, the policy factory and registry, and the reference selector.

Dependencies: `dynamo-router-protocol` and nothing router-side beyond it; no axum, reqwest, zmq, Prometheus, or `dynamo-runtime` under any feature. The `bench` and `testing` features add the reference picker (`parking_lot`) for parity tests and benchmarks.

Part of the router split (see `lib/router/README.md`).
