// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Velo runtime support for delegated response streams.
//!
//! This module owns the process-wide Velo instance that will back direct
//! response streams. The current milestone only introduces peer discovery and
//! local runtime initialization; request dispatch still uses the existing
//! request-plane transports.

pub mod dynamo_peer_discovery;

use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

use anyhow::{Context as _, Result, anyhow};

use ::velo::backend::tcp::TcpTransportBuilder;
use ::velo::discovery::PeerDiscovery;
pub use ::velo::{InstanceId, PeerInfo, StreamAnchorHandle, Velo, WorkerId};

pub use self::dynamo_peer_discovery::{DynamoPeerDiscovery, DynamoPeerRegistrationGuard};
use crate::discovery::Discovery;

/// Process-wide Velo instance shared by frontend, router, and worker response
/// stream code in this process.
static GLOBAL_VELO: tokio::sync::OnceCell<VeloHandle> = tokio::sync::OnceCell::const_new();

/// Configuration for the per-process Velo runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VeloRuntimeConfig {
    /// IP for the Velo active-message TCP transport.
    pub messenger_bind_host: IpAddr,
    /// TCP port for the Velo active-message transport. `None` lets the OS choose.
    pub messenger_bind_port: Option<u16>,
    /// IP for Velo frame streaming anchors.
    pub stream_bind_host: IpAddr,
}

impl Default for VeloRuntimeConfig {
    fn default() -> Self {
        let host = IpAddr::from([0, 0, 0, 0]);
        Self {
            messenger_bind_host: host,
            messenger_bind_port: None,
            stream_bind_host: host,
        }
    }
}

impl VeloRuntimeConfig {
    /// Read Velo network configuration from environment variables.
    ///
    /// `DYN_VELO_HOST` is used for both active messages and stream anchors.
    /// `DYN_VELO_PORT` only applies to the active-message TCP transport; stream
    /// anchors allocate per-anchor endpoints through Velo.
    pub fn from_env() -> Self {
        let default = Self::default();
        let host = std::env::var("DYN_VELO_HOST")
            .ok()
            .and_then(|s| s.parse::<IpAddr>().ok())
            .unwrap_or(default.messenger_bind_host);
        let port = std::env::var("DYN_VELO_PORT")
            .ok()
            .and_then(|s| s.parse::<u16>().ok());

        Self {
            messenger_bind_host: host,
            messenger_bind_port: port,
            stream_bind_host: host,
        }
    }

    fn messenger_bind_addr(&self) -> SocketAddr {
        SocketAddr::new(
            self.messenger_bind_host,
            self.messenger_bind_port.unwrap_or(0),
        )
    }
}

/// Owning handle for the process-wide Velo instance.
///
/// The registration guard keeps this process' Velo peer entry visible in the
/// Dynamo discovery plane for the lifetime of the handle.
pub struct VeloHandle {
    velo: Arc<Velo>,
    _registration: DynamoPeerRegistrationGuard,
}

impl VeloHandle {
    /// The shared Velo runtime.
    pub fn velo(&self) -> &Arc<Velo> {
        &self.velo
    }

    /// Velo peer info registered in discovery.
    pub fn peer_info(&self) -> PeerInfo {
        self.velo.peer_info()
    }

    /// Velo instance ID for this process.
    pub fn instance_id(&self) -> InstanceId {
        self.velo.instance_id()
    }
}

/// Initialize or fetch the process-wide Velo runtime.
pub async fn global_velo(
    discovery: Arc<dyn Discovery>,
    config: VeloRuntimeConfig,
) -> Result<&'static VeloHandle> {
    GLOBAL_VELO
        .get_or_try_init(|| async move { build_velo_handle(discovery, config).await })
        .await
}

/// Return the Velo instance ID if the process-wide runtime is initialized.
pub fn current_velo_instance_id() -> Result<InstanceId> {
    GLOBAL_VELO
        .get()
        .map(|h| h.instance_id())
        .ok_or_else(|| anyhow!("Velo runtime has not been initialized"))
}

async fn build_velo_handle(
    discovery: Arc<dyn Discovery>,
    config: VeloRuntimeConfig,
) -> Result<VeloHandle> {
    let bind_addr = config.messenger_bind_addr();
    tracing::info!(
        messenger_bind_addr = %bind_addr,
        stream_bind_host = %config.stream_bind_host,
        port_source = if config.messenger_bind_port.is_some() { "DYN_VELO_PORT" } else { "OS-assigned" },
        "Initializing Velo runtime"
    );

    let transport: Arc<dyn ::velo::Transport> = Arc::new(
        TcpTransportBuilder::new()
            .bind_addr(bind_addr)
            .build()
            .with_context(|| format!("building Velo TCP transport on {bind_addr}"))?,
    );

    let discovery = Arc::new(DynamoPeerDiscovery::new(discovery));
    let discovery_for_velo: Arc<dyn PeerDiscovery> = discovery.clone();

    let velo = Velo::builder()
        .add_transport(transport)
        .discovery(discovery_for_velo)
        .stream_bind_addr(config.stream_bind_host)
        .build()
        .await
        .with_context(|| "building Velo runtime")?;

    let registration = discovery
        .register(velo.peer_info())
        .await
        .with_context(|| "registering local Velo peer")?;

    tracing::info!(
        velo_instance = %velo.instance_id().as_uuid(),
        "Velo runtime ready"
    );

    Ok(VeloHandle {
        velo,
        _registration: registration,
    })
}
