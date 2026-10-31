<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-scheduler

The router's scheduler kernel: it runs the routing stages (index lookup, slot tracking,
session affinity, worker selection, admission) atomically for one request. It owns the
request and response vocabulary, eligibility, the queue actor and `LocalScheduler`, the
overlap and prefill-load analysis, the selection kernel, the validated kernel configuration,
and the tracking hash. It depends on the five stage crates, `dynamo-router-protocol`, and
`dynamo-kv-hashing`, and on nothing that opens a socket or serves HTTP; `dynamo-kv-router`
composes it with those adapters.
