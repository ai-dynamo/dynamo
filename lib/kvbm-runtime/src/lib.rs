// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Build KVBM runtime *primitives* from a [`kvbm_config`].
//!
//! This crate exists to keep [`kvbm_config`] pure: the config structs there
//! carry no `velo` or `dynamo-memory` (CUDA) dependency, so config-only
//! consumers — the `kvbm_hub` server, `kvbmctl` validation — link without
//! pulling a transport stack or CUDA. The actual construction of velo
//! transports/messengers and NIXL backends from that config lives here, as
//! free functions (the orphan rule forbids re-attaching them as inherent
//! methods / `From` impls on the foreign config types).
//!
//! Not to be confused with `kvbm_engine`'s `KvbmRuntime`; this crate only
//! builds the lower-level primitives a runtime is assembled from.

use std::sync::Arc;

use anyhow::{Context, Result, bail};
use kvbm_config::{DiscoveryConfig, MessengerConfig, NixlConfig};

use dynamo_memory::nixl::NixlBackendConfig;

/// Build a [`velo::Velo`] instance from a [`MessengerConfig`].
///
/// This is the production constructor — Velo wraps a Messenger plus the
/// streaming AnchorManager and RendezvousManager, which are required by the
/// disagg session machinery. Callers that only need the underlying Messenger
/// should call [`velo::Velo::messenger`] on the returned instance.
pub async fn build_velo(cfg: &MessengerConfig) -> Result<Arc<velo::Velo>> {
    build_velo_with_discovery(cfg, None).await
}

/// Build a [`velo::Velo`] instance, optionally overriding (or filling in) the
/// peer-discovery backend.
///
/// When `discovery_override` is `Some`, it takes precedence over the
/// `discovery` field on `cfg`. Use this when the caller already holds a
/// `PeerDiscovery` impl that can't be constructed from static config — e.g.
/// `kvbm_hub::HubClient`, which the connector builds at startup from the hub
/// URL. Wiring it here lets velo's standard
/// `messenger.discover_and_register_peer` path resolve cross-instance peers via
/// the hub, instead of relying on each subsystem's bespoke resolver.
pub async fn build_velo_with_discovery(
    cfg: &MessengerConfig,
    discovery_override: Option<Arc<dyn velo::discovery::PeerDiscovery>>,
) -> Result<Arc<velo::Velo>> {
    use std::net::TcpListener;

    use velo::Velo;
    use velo::transports::tcp::TcpTransportBuilder;
    use velo::transports::uds::UdsTransportBuilder;

    // 1. Build TCP transport
    let bind_addr = cfg.backend.resolve_bind_addr()?;
    let listener = TcpListener::bind(bind_addr)
        .with_context(|| format!("Failed to bind TCP listener to {}", bind_addr))?;
    let actual_addr = listener
        .local_addr()
        .context("Failed to get local address from listener")?;
    tracing::info!("Built TCP transport bound to {}", actual_addr);

    let tcp_transport = TcpTransportBuilder::new()
        .from_listener(listener)?
        .build()
        .context("Failed to build TCP transport")?;
    let tcp_transport = Arc::new(tcp_transport);

    // 2. Configure VeloBuilder. Insertion order seeds the default transport
    //    priority list (see VeloBackend::new in velo); UDS is added first so
    //    same-host peers prefer it. Host-affinity in velo automatically falls
    //    back to TCP when a peer's UDS path is not visible on this host's
    //    filesystem.
    let mut builder = Velo::builder();

    if cfg.backend.uds_enabled {
        let dir = cfg
            .backend
            .uds_dir
            .clone()
            .unwrap_or_else(std::env::temp_dir);
        let socket_path = dir.join(format!("velo-kvbm-{}.sock", uuid::Uuid::new_v4()));
        let uds_transport = UdsTransportBuilder::new()
            .socket_path(&socket_path)
            .build()
            .context("Failed to build UDS transport")?;
        tracing::info!("Built UDS transport bound to {}", socket_path.display());
        builder = builder.add_transport(Arc::new(uds_transport));
    }

    builder = builder.add_transport(tcp_transport);

    if let Some(discovery) = discovery_override {
        // The injected backend (e.g. the hub's HubClient) wins. If the config
        // also specified a static discovery backend, it is silently superseded
        // — warn so an operator whose `messenger.discovery` is being ignored
        // isn't surprised.
        if let Some(superseded) = &cfg.discovery {
            tracing::warn!(
                superseded = ?superseded,
                "messenger.discovery is configured but superseded by the injected \
                 discovery backend (hub); the configured backend will not be used"
            );
        }
        builder = builder.discovery(discovery);
        tracing::info!("Using injected discovery backend (override)");
    } else if let Some(discovery_config) = &cfg.discovery {
        match discovery_config {
            DiscoveryConfig::Etcd(_cfg) => {
                bail!("Etcd discovery not yet supported in velo");
            }
            DiscoveryConfig::P2p(_cfg) => {
                bail!("P2P discovery not yet supported in velo");
            }
            DiscoveryConfig::Filesystem(cfg) => {
                use velo::discovery::FilesystemPeerDiscovery;

                let peer_discovery = FilesystemPeerDiscovery::new(&cfg.path)
                    .context("Failed to build filesystem discovery")?;

                builder = builder.discovery(Arc::new(peer_discovery));
                tracing::info!("Built filesystem discovery from: {:?}", cfg.path);
            }
        }
    }

    // 3. Build Velo
    let velo = builder.build().await.context("Failed to build Velo")?;
    Ok(velo)
}

/// Build a [`velo::Messenger`] instance from a [`MessengerConfig`].
///
/// Convenience wrapper around [`build_velo`] that returns just the messenger.
pub async fn build_messenger(cfg: &MessengerConfig) -> Result<Arc<velo::Messenger>> {
    let velo = build_velo(cfg).await?;
    Ok(velo.messenger().clone())
}

/// Convert a [`NixlConfig`] into a [`NixlBackendConfig`] (the dynamo-memory
/// type the NIXL agent is built from).
///
/// A free function rather than a `From` impl: the orphan rule forbids
/// `impl From<NixlConfig> for NixlBackendConfig` here since both types are
/// foreign to this crate.
pub fn nixl_backend_config(cfg: &NixlConfig) -> NixlBackendConfig {
    NixlBackendConfig::new(cfg.backends.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nixl_backend_config_carries_backends() {
        // The NixlConfig → NixlBackendConfig bridge (moved out of kvbm-config to
        // keep that crate CUDA-free) preserves the configured backends.
        let cfg = NixlConfig::empty()
            .with_backend("UCX")
            .with_backend("GDS_MT");
        let backend = nixl_backend_config(&cfg);
        let names: Vec<String> = backend.iter().map(|(b, _)| b.to_string()).collect();
        assert!(names.contains(&"UCX".to_string()));
        assert!(names.contains(&"GDS_MT".to_string()));
    }
}
