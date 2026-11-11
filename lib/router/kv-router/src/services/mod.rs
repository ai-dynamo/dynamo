// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Standalone services built from brokerless transport primitives.

pub(crate) mod common;

// The ZMQ listener registry; its HTTP server needs `http` and its peer recovery `peer-recovery`.
#[cfg(feature = "ingest-zmq")]
pub mod indexer;

// The selection service: its composition needs no adapter. The ZMQ-direct ingress needs
// `ingest-zmq`, indexer peer recovery `ingest-zmq` + `peer-recovery`, the replica mesh
// `replica-sync-zmq`, and its HTTP server and CLI config all three plus `http`.
pub mod selection;

// The slot-tracker service: an HTTP server over the registry, replicated through the mesh.
#[cfg(all(feature = "http", feature = "replica-sync-zmq"))]
pub mod slot_tracker;

// Pre-split paths kept through 1.9 and removed in 1.10; the named list is the inventory in compat.rs.
#[allow(deprecated)]
pub use crate::compat::services::overlap;
