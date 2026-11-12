<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-scheduler

`dynamo-scheduler` is the router's kernel. For each request it reads cached-prefix overlap from the indexer, checks slot load and session affinity, runs worker selection, and admits or queues the request as one atomic step, then books the chosen worker; an advisory selection returns a pick without booking it. It owns the request and response types, the queue actor, `LocalScheduler`, `SelectionCore` (the entry point hosts and the router service call), and the tracking hash. It opens no sockets; `dynamo-kv-router` adds the transports.
