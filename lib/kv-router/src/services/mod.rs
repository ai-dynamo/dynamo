// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Standalone services built from brokerless transport primitives.

pub(crate) mod common;

// The ZMQ listener registry; its HTTP server needs `http` and its peer recovery `peer-recovery`.
#[cfg(feature = "ingest-zmq")]
pub mod indexer;

#[deprecated(
    since = "1.7.0",
    note = "dynamo_scheduler::overlap_summary; see lib/kv-router/README.md, Deprecated paths"
)]
pub mod overlap;

// The selection service: its composition needs no adapter. The ZMQ-direct ingress needs
// `ingest-zmq`, indexer peer recovery `ingest-zmq` + `peer-recovery`, the replica mesh
// `replica-sync-zmq`, and its HTTP server and CLI config all three plus `http`.
pub mod selection;

// The slot-tracker service: an HTTP server over the registry, replicated through the mesh.
#[cfg(all(feature = "http", feature = "replica-sync-zmq"))]
pub mod slot_tracker;
