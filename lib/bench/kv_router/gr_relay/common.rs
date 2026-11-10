// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What the pool relays and the selection core share: the velo setup and the
//! handshake that opens one stream per worker. The core writes its velo peer
//! info to a shared file that the relays read.
//!
//! Each stream carries msgpack `Vec<RouterEvent>` records, one per engine
//! batch, encoded with field names as Dynamo's event plane encodes them.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use velo::streaming::{AutoFlush, FlushPolicy, MuxConfig};
use velo::transports::tcp::TcpTransportBuilder;
use velo::{Velo, VeloMetrics};

/// Unary handler on the core that opens one anchor per worker.
pub const OPEN_HANDLER: &str = "gr_relay_open";

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenRequest {
    pub worker_ids: Vec<u64>,
}

/// One anchor handle per requested worker, in request order.
#[derive(Debug, Serialize, Deserialize)]
pub struct OpenReply {
    pub handles: Vec<u128>,
}

#[derive(Debug, Clone, Copy)]
pub struct VeloOptions {
    /// Bound and advertised; use the NIC address peers reach.
    pub bind: SocketAddr,
    /// TCP connections per peer (each direction).
    pub lanes: u16,
    /// Mux write linger; 0 writes as soon as the transport takes the last batch.
    pub linger_ms: u64,
}

/// A mux-only TCP instance, configured as Dynamo's velo response plane is
/// (`lib/runtime/src/pipeline/network/velo_response.rs`), with default credit.
pub async fn build_velo(
    options: VeloOptions,
    metrics: Option<Arc<VeloMetrics>>,
) -> anyhow::Result<Velo> {
    let mut builder = Velo::builder();
    if let Some(metrics) = metrics {
        builder = builder.metrics(metrics);
    }
    let mux = MuxConfig {
        enabled: true,
        flush_policy: FlushPolicy::Auto(AutoFlush {
            on_admission: options.linger_ms == 0,
            max_linger: (options.linger_ms != 0).then(|| Duration::from_millis(options.linger_ms)),
        }),
        ..Default::default()
    };
    let tcp = TcpTransportBuilder::new()
        .bind_addr(options.bind)
        .lanes(options.lanes)
        .build()?;
    builder
        .mux_only()
        .messenger_mux(mux)?
        .add_transport(Arc::new(tcp))
        .build()
        .await
}
