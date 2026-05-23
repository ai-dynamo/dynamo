// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal ZMQ helper for the KV indexer's ingest side.
//!
//! Topology is inverted relative to the usual pub/sub: the hub **binds** a
//! `SUB` socket and connectors **connect** their `PUB` sockets to it. ZMQ
//! supports binding on either end. Mirrors the knobs in
//! `kvbm-consolidator/src/zmq_util.rs` without depending on that crate.

use anyhow::{Context as _, Result};
use tmq::{
    AsZmqSocket, Context,
    subscribe::{Subscribe, subscribe},
};

const ZMQ_LINGER_MS: i32 = 0;

/// Binds a `SUB` socket to `endpoint` (e.g. `tcp://0.0.0.0:0` for an
/// OS-assigned port) and subscribes to all topics.
pub fn bind_sub_socket(endpoint: &str) -> Result<Subscribe> {
    let ctx = Context::new();
    let socket = subscribe(&ctx)
        .set_linger(ZMQ_LINGER_MS)
        .bind(endpoint)
        .with_context(|| format!("binding indexer SUB socket to {endpoint}"))?
        .subscribe(b"")
        .context("subscribing indexer SUB socket to all topics")?;
    Ok(socket)
}

/// Reads back the concrete endpoint the socket bound to — resolves the
/// OS-assigned port when bound to `:0`.
pub fn bound_endpoint(sub: &Subscribe) -> Result<String> {
    sub.get_socket()
        .get_last_endpoint()
        .context("zmq get_last_endpoint")?
        .map_err(|_| anyhow::anyhow!("zmq last endpoint is not valid UTF-8"))
}

/// Extracts the port from a `tcp://host:port` endpoint string.
pub fn port_of(endpoint: &str) -> Result<u16> {
    endpoint
        .rsplit(':')
        .next()
        .and_then(|p| p.parse::<u16>().ok())
        .with_context(|| format!("no port in endpoint {endpoint}"))
}
