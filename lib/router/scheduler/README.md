<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-scheduler

The router's scheduler kernel: it runs the routing stages (index lookup, slot tracking,
session affinity, worker selection, admission) atomically for one request. It owns the
request and response vocabulary, eligibility, the queue actor and `LocalScheduler`, the
overlap and prefill-load analysis, the selection kernel, the validated kernel configuration,
and the tracking hash. It depends on the five stage crates, `dynamo-router-protocol`,
`dynamo-kv-hashing`, and `dynamo-kv-hints`, and on nothing that opens a socket or serves
HTTP; `dynamo-kv-router` composes it with those adapters. Its `protocols` module holds the
kernel's own request, response, and config-like types (`RouterRequest`, `RouterResponse`,
`PotentialLoad`, `RoutingConstraints`, `WorkerConfigLike`, `WorkerSelectionResult`,
`RouterEventSink`) and re-exports nothing; the kernel imports stage and wire types from their
owners.
