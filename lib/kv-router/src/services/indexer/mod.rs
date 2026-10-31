// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Standalone HTTP KV-cache indexer.
//!
//! Hosts an Axum HTTP server with `/register`, `/unregister`, `/query`,
//! `/query_by_hash`, and peer-discovery routes that workers / gateways can
//! call to drive cache-aware routing decisions. Each registered worker spawns
//! a ZMQ listener that ingests its KV events into a per-(model, routing group)
//! `backend::Indexer`.
//!
//! [`RoutingPartitionId`](crate::identity::RoutingPartitionId) remains this service's sole registry
//! authority for `(model_name, routing_group)`.
//! Resolving it to `IndexerDomainId` is intentionally deferred until registration can carry
//! authoritative explicit identity material; hashing local defaults here would create a second
//! authority without enabling safe cross-service identity.
//!
//! ## Multi-tier responses
//!
//! `/query` and `/query_by_hash` return both:
//! - the legacy flat `scores`/`frequencies`/`tree_sizes` (device-tier overlap),
//!   for backward compatibility, and
//! - a per-instance `instances` map keyed by `instance_id` with `gpu`, `cpu`,
//!   `disk`, per-`dp_rank` device counts, and `longest_matched`.
//!
//! The `instances` shape is intended to align with Mooncake's
//! "\[RFC\]: KV-Store Indexer API Standardization"
//! (<https://github.com/kvcache-ai/Mooncake/issues/1403>).
//! Tier counts are CUMULATIVE through each tier's walk — see the doc on the
//! response struct in `server` for the exact semantics.

pub mod listener;
#[cfg(feature = "http")]
pub mod logging;
#[cfg(feature = "http")]
pub mod metrics;
#[cfg(feature = "peer-recovery")]
pub mod recovery;
pub mod registry;
/// The service entry: CLI-shaped config, worker list parsing, and `run_server`.
#[cfg(all(feature = "http", feature = "peer-recovery"))]
mod run;
#[cfg(feature = "http")]
pub mod server;
#[cfg(all(feature = "http", feature = "peer-recovery"))]
pub use run::{IndexerConfig, parse_workers, run_server};

pub use dynamo_kv_indexer::{backend, lookup, recording, session_updates};

use crate::services::common::zmq::validate_endpoint as validate_zmq_endpoint;

/// Both listener endpoints a worker registers must be valid ZMQ endpoints; the HTTP server
/// checks them at registration.
#[cfg_attr(not(feature = "http"), allow(dead_code))]
pub(super) fn validate_listener_endpoints(
    endpoint: &str,
    replay_endpoint: Option<&str>,
) -> anyhow::Result<()> {
    validate_zmq_endpoint(endpoint)?;
    if let Some(replay_endpoint) = replay_endpoint {
        validate_zmq_endpoint(replay_endpoint).map_err(|error| {
            anyhow::anyhow!("invalid replay endpoint `{replay_endpoint}`: {error}")
        })?;
    }
    Ok(())
}
