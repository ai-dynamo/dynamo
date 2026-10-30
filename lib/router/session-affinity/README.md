<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-session-affinity

Session bindings in, constraints out. The router's session affinity stage: the replicated table from session id to worker target (`SessionAffinity`, `WeakSessionAffinity`), the acquire protocol (`Hold`, `AffinityInitialization`, `AffinityLease`), versioned bindings (`AffinityVersion`) published through the `AffinityReplicaSink` port, the config and mode (`SessionAffinityConfig`, `SessionAffinityMode`), the limits, and the replication wire (`wire::AffinityBindingEvent`). `lib/kv-router` re-exports the crate at `services::selection::affinity` and keeps the replica mesh that carries the wire.

Dependencies: `dynamo-router-protocol` and nothing router-side beyond it; no axum, reqwest, zmq, Prometheus, or `dynamo-runtime` under any feature. Tokio stays for the notify and timers; the stage rule forbids sockets and HTTP, not async.

Part of the router split (see `lib/router/README.md`).
