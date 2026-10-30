// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Everything about KV cache events: the event types and the [`RouterEvent`](event::RouterEvent)
//! envelope, the engine ZMQ msgpack codec (`zmq_wire`), the replay cursor, and the publisher's
//! placement helpers. Both ends of the KV-event wire depend on this crate: workers and sidecars
//! publish, the router's indexer ingests.
//!
//! No sockets, no tasks, no servers live here; the ZMQ socket itself is `dynamo-zmq` and the
//! listener that drives it is a `dynamo-kv-router` adapter.

pub mod cursor;
pub mod event;
pub mod placement;
pub mod zmq_wire;
