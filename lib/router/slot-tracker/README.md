<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-slot-tracker

Bookings per rank in, load and fullness per request out. The router's slot tracker stage: the active-sequence trackers (`ActiveSequences`, `ActiveSequencesMultiWorker`), the prompt registry with its membership trie and load projection (`WorkerLoadProjection`, `WorkerLoadReader`), the block and prefill trackers, the request index that holds the one `AttemptIdMinter`, worker topology, the replication effects, the replication wire (`wire`, including the scheduler-load report `SchedulerLoad` and `SchedulerGroup`), per-rank fullness (`busy::BusyThresholds`, `busy::FullRankCounts`, the `RankLoad` reading trait; admission and worker selection depend on this crate for that rule), and the booking the scheduler hands in (`booking::SchedulerBookingDescriptor`). `lib/router/kv-router` re-exports every item at its old path and keeps the adapters: the replica mesh, the slot-tracker registry, and the HTTP server.

Dependencies: `dynamo-router-protocol` and nothing router-side beyond it; no axum, reqwest, zmq, Prometheus, or `dynamo-runtime` under any feature. Tokio stays for channels and timers; the stage rule forbids sockets and HTTP, not async.

Part of the router split (see `lib/router/README.md`).

## Modeled prefill load

`ActiveSequences` owns request and prefill state; mutations project derived load
into `PromptRegistry`. Reads are advisory and may lag authoritative writes.

For fully modeled prefills, elapsed progress reduces the aggregate backlog. Before
inserting a new request, progress beyond previously available work is discarded,
independently for tokens and modeled time. This prevents pre-arrival progress from
consuming newly booked work. The counters belong to the effective anchor, are
bounded when a non-front completion shifts it, and reset when the front changes.

If any prefill lacks a duration estimate, token load uses anchor-only decay and
modeled remaining time is unavailable. A zero-duration anchor completes only its
own tokens. Token spillover assumes approximately uniform modeled tokens/sec;
modeled time is an advisory estimate, not an engine timing measurement.
