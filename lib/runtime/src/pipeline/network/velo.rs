// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Velo-backed request plane support.
//!
//! This module owns the per-process [`Velo`] instance shared between the
//! velo-mode `RequestPlaneServer` and `RequestPlaneClient`, builds it with the
//! TCP transport, wires Dynamo's KV-backed [`PeerDiscovery`] adapter, and
//! defines the wire address scheme used in service discovery.
//!
//! # Address scheme
//!
//! The string published into Dynamo's discovery KV (and consumed by clients) is:
//!
//! ```text
//! velo://<velo_instance_id_uuid>/<dynamo_instance_id_hex>/<endpoint_name>
//! ```
//!
//! - `<velo_instance_id_uuid>` is the velo [`InstanceId`] used by the client to
//!   resolve and dial the peer via [`Velo::discover_and_register_peer`].
//! - `<dynamo_instance_id_hex>` is the Dynamo `instance_id` (`u64`) — preserved
//!   so multiple workers on the same velo node remain disambiguated, exactly
//!   the same convention used by the TCP request plane.
//! - `<endpoint_name>` matches the user-defined endpoint (e.g. `generate`).

pub mod kv_discovery;

use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

use anyhow::{Context as _, Result, anyhow};
use tokio::sync::OnceCell;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

pub use ::velo::Velo;
pub use ::velo::InstanceId;
use ::velo::transports::tcp::TcpTransportBuilder;

pub use self::kv_discovery::{KvPeerDiscovery, KvPeerRegistrationGuard, VELO_PEERS_BUCKET};

use crate::storage::kv;

/// Scheme prefix for velo addresses.
pub const VELO_SCHEME: &str = "velo://";

/// Header carrying the demux key (`{instance_id_hex}/{endpoint_name}`) on velo
/// requests so a single registered velo handler can dispatch to many Dynamo
/// endpoints.
pub const ENDPOINT_HEADER: &str = "x-dynamo-endpoint";

/// Header carrying the Dynamo request id (matches the existing TCP header naming).
pub const REQUEST_ID_HEADER: &str = "x-dynamo-request-id";

/// Name of the single multiplexed velo handler all Dynamo endpoints register through.
pub const REQUEST_PLANE_HANDLER: &str = "dynamo-request-plane";

/// Configuration for the per-process velo instance.
#[derive(Debug, Clone)]
pub struct VeloRuntimeConfig {
    /// IP to bind the velo TCP transport on. Defaults to `0.0.0.0`.
    pub bind_host: IpAddr,
    /// TCP port for the velo transport. `None` lets the OS assign one.
    pub bind_port: Option<u16>,
}

impl Default for VeloRuntimeConfig {
    fn default() -> Self {
        Self {
            bind_host: IpAddr::from([0, 0, 0, 0]),
            bind_port: None,
        }
    }
}

impl VeloRuntimeConfig {
    /// Read [`VeloRuntimeConfig`] from `DYN_VELO_HOST` / `DYN_VELO_PORT`.
    pub fn from_env() -> Self {
        let bind_host = std::env::var("DYN_VELO_HOST")
            .ok()
            .and_then(|s| s.parse::<IpAddr>().ok())
            .unwrap_or_else(|| IpAddr::from([0, 0, 0, 0]));
        let bind_port = std::env::var("DYN_VELO_PORT")
            .ok()
            .and_then(|s| s.parse::<u16>().ok());
        Self {
            bind_host,
            bind_port,
        }
    }

    fn bind_socket(&self) -> SocketAddr {
        SocketAddr::new(self.bind_host, self.bind_port.unwrap_or(0))
    }
}

/// Decoded view of a `velo://...` address published in service discovery.
#[derive(Debug, Clone)]
pub struct VeloAddress {
    /// Velo instance to dial.
    pub velo_instance: InstanceId,
    /// Demux key: `{dynamo_instance_id_hex}/{endpoint_name}`.
    pub endpoint_key: String,
}

/// Encode a velo address for service discovery.
pub fn encode_velo_address(
    velo_instance: InstanceId,
    dynamo_instance_id: u64,
    endpoint_name: &str,
) -> String {
    format!(
        "{}{}/{:x}/{}",
        VELO_SCHEME,
        velo_instance.as_uuid(),
        dynamo_instance_id,
        endpoint_name,
    )
}

/// Encode the address prefix for a velo node (no endpoint suffix).
///
/// Returned by [`super::ingress::velo_endpoint::VeloRequestPlaneServer::address`]
/// for logging / debugging purposes; full addresses for clients to dial are
/// constructed by `build_transport_type` using [`encode_velo_address`].
pub fn encode_velo_node_prefix(velo_instance: InstanceId) -> String {
    format!("{}{}", VELO_SCHEME, velo_instance.as_uuid())
}

/// Parse a velo address. The endpoint key is `<dynamo_instance_id_hex>/<endpoint_name>`
/// and is preserved verbatim so it matches the demux DashMap on the server.
pub fn decode_velo_address(address: &str) -> Result<VeloAddress> {
    let rest = address
        .strip_prefix(VELO_SCHEME)
        .ok_or_else(|| anyhow!("address {address} is missing `{VELO_SCHEME}` prefix"))?;
    let (uuid_str, endpoint_key) = rest
        .split_once('/')
        .ok_or_else(|| anyhow!("velo address {address} missing endpoint key"))?;
    if endpoint_key.is_empty() {
        return Err(anyhow!("velo address {address} has empty endpoint key"));
    }
    let velo_instance = Uuid::parse_str(uuid_str)
        .map(InstanceId::from)
        .with_context(|| format!("invalid velo InstanceId in {address}"))?;
    Ok(VeloAddress {
        velo_instance,
        endpoint_key: endpoint_key.to_string(),
    })
}

// ---------------------------------------------------------------------------
// Per-process velo instance
// ---------------------------------------------------------------------------

/// Process-wide velo instance shared between the velo `RequestPlaneServer` and
/// `RequestPlaneClient`. Built lazily on first access.
static GLOBAL_VELO: OnceCell<VeloHandle> = OnceCell::const_new();

/// Process-wide cancellation token for the global velo instance, decoupled from
/// per-runtime cancellation tokens (mirrors `GLOBAL_TCP_SERVER_TOKEN`).
pub(crate) static GLOBAL_VELO_TOKEN: std::sync::LazyLock<CancellationToken> =
    std::sync::LazyLock::new(CancellationToken::new);

/// Owning handle for the per-process velo instance. The discovery guard keeps
/// the local peer entry alive in the KV for the lifetime of the process.
pub struct VeloHandle {
    velo: Arc<Velo>,
    discovery: Arc<KvPeerDiscovery>,
    // Held to keep the local peer registered for the lifetime of the process.
    _registration: KvPeerRegistrationGuard,
}

impl VeloHandle {
    /// The shared velo instance.
    pub fn velo(&self) -> &Arc<Velo> {
        &self.velo
    }

    /// The KV-backed discovery adapter wired into this velo instance.
    pub fn discovery(&self) -> &Arc<KvPeerDiscovery> {
        &self.discovery
    }
}

/// Return the velo `InstanceId` of the per-process velo node, or an error if
/// it has not been initialized yet (the velo `RequestPlaneServer` must be
/// created before this is called — that is the contract that
/// `build_transport_type` upholds).
pub fn current_velo_instance_id() -> Result<InstanceId> {
    GLOBAL_VELO
        .get()
        .map(|h| h.velo.instance_id())
        .ok_or_else(|| {
            anyhow!(
                "velo request-plane node not yet initialized; \
                 ensure DistributedRuntime started the velo server before constructing a transport address"
            )
        })
}

/// Borrow the per-process velo handle, or error if it has not been
/// initialized yet. Used by the bidi client/server paths to call
/// `velo.create_anchor` / `velo.attach_anchor` / `velo.unary(...)`.
pub fn current_velo() -> Result<&'static Arc<Velo>> {
    GLOBAL_VELO.get().map(|h| &h.velo).ok_or_else(|| {
        anyhow!(
            "velo request-plane node not yet initialized; \
             ensure DistributedRuntime started the velo server before issuing bidi traffic"
        )
    })
}

/// Build (or fetch) the process-wide velo instance.
///
/// `kv_manager` is used as the backing store for the `PeerDiscovery` adapter so
/// that velo peer lookups share a single discovery surface with the rest of
/// the runtime (memory / file / etcd, depending on `DYN_DISCOVERY_BACKEND`).
pub async fn global_velo(
    kv_manager: Arc<kv::Manager>,
    config: VeloRuntimeConfig,
) -> Result<&'static VeloHandle> {
    GLOBAL_VELO
        .get_or_try_init(|| async move { build_velo_handle(kv_manager, config).await })
        .await
}

async fn build_velo_handle(
    kv_manager: Arc<kv::Manager>,
    config: VeloRuntimeConfig,
) -> Result<VeloHandle> {
    let bind_addr = config.bind_socket();
    tracing::info!(
        bind_addr = %bind_addr,
        port_source = if config.bind_port.is_some() { "DYN_VELO_PORT" } else { "OS-assigned" },
        "Initializing velo request-plane node"
    );

    let transport = TcpTransportBuilder::new()
        .bind_addr(bind_addr)
        .build()
        .with_context(|| format!("building velo TCP transport on {bind_addr}"))?;

    let discovery = Arc::new(KvPeerDiscovery::new(kv_manager));
    let discovery_for_velo: Arc<dyn ::velo::discovery::PeerDiscovery> = discovery.clone();

    let velo = Velo::builder()
        .add_transport(Arc::new(transport))
        .discovery(discovery_for_velo)
        .build()
        .await
        .with_context(|| "building velo instance")?;

    let registration = discovery
        .register(velo.peer_info())
        .await
        .with_context(|| "registering local peer in velo KV discovery")?;

    tracing::info!(
        velo_instance = %velo.instance_id().as_uuid(),
        "Velo request-plane node ready"
    );

    Ok(VeloHandle {
        velo,
        discovery,
        _registration: registration,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_address() {
        let id = InstanceId::new_v4();
        let s = encode_velo_address(id, 0xdeadbeef, "generate");
        let parsed = decode_velo_address(&s).expect("decode");
        assert_eq!(parsed.velo_instance, id);
        assert_eq!(parsed.endpoint_key, "deadbeef/generate");
    }

    #[test]
    fn rejects_non_velo_scheme() {
        assert!(decode_velo_address("tcp://1.2.3.4:5/0/foo").is_err());
    }

    #[test]
    fn rejects_missing_endpoint_key() {
        let s = format!("{}{}", VELO_SCHEME, InstanceId::new_v4().as_uuid());
        assert!(decode_velo_address(&s).is_err());
    }

    #[test]
    fn rejects_empty_endpoint_key() {
        let s = format!("{}{}/", VELO_SCHEME, InstanceId::new_v4().as_uuid());
        assert!(decode_velo_address(&s).is_err());
    }
}
