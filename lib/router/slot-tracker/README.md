<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-slot-tracker

Bookings per rank in, load and fullness per request out. The router's slot tracker stage: the active-sequence trackers (`ActiveSequences`, `ActiveSequencesMultiWorker`), the prompt registry with its membership trie and load projection (`WorkerLoadProjection`, `WorkerLoadReader`), the block and prefill trackers, the request index that holds the one `AttemptIdMinter`, worker topology, the replication effects, the replication wire (`wire`), per-rank fullness (`busy::BusyThresholds`, `busy::FullRankCounts`, the `RankLoad` reading trait), and the booking the scheduler hands in (`booking::SchedulerBookingDescriptor`). `lib/kv-router` re-exports every item at its old path and keeps the adapters: the replica mesh, the slot-tracker registry, and the HTTP server.

Dependencies: `dynamo-router-protocol` and nothing router-side beyond it; no axum, reqwest, zmq, Prometheus, or `dynamo-runtime` under any feature. Tokio stays for channels and timers; the stage rule forbids sockets and HTTP, not async.

Part of the router split (see `lib/router/README.md`).
