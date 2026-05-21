// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! P2P foundation wiring.
//!
//! [`wire_p2p`] registers `Feature::P2P` (+ any `extra_features`, e.g.
//! ConditionalDisagg, and Indexer when effective) with the hub, builds the
//! [`HubPeerResolver`] + [`VeloSessionFactory`], installs the session factory
//! on the leader's control-plane `transfer` module, and spawns the
//! describe-push. It is the shared foundation for **both** a standalone P2P
//! instance (remote-controllable block-copy peer) and ConditionalDisagg, which
//! builds its decode/prefill flows on top of the returned [`P2pFoundation`].

use std::sync::Arc;

use anyhow::{Context, Result};
use kvbm_engine::disagg::session::{PeerResolver as EnginePeerResolver, SessionFactory};
use kvbm_engine::leader::InstanceLeader;
use kvbm_hub::protocol::LayoutCompatPayload;
use kvbm_hub::{Feature, HubClient, IndexerFeatureConfig, P2pConfig};
use velo::{InstanceId, Velo};

use super::peer_resolver::HubPeerResolver;
use crate::connector::leader::ConnectorLeader;
use crate::connector::leader::disagg::build_hub_client;
use crate::connector::leader::hub_handshake::HubHandshake;

/// Shared P2P wiring produced by [`wire_p2p`] and consumed by CD.
pub struct P2pFoundation {
    /// Hub client holding this instance's registration (keep alive for the
    /// leader's lifetime so the RAII guard doesn't fire a premature `DELETE`).
    pub hub: Arc<HubClient>,
    /// Hub's velo `InstanceId` (when the hub runs a velo participant).
    pub hub_velo_id: Option<InstanceId>,
    /// Shared hub-backed peer resolver (de-dup cache shared across paths).
    pub peer_resolver: Arc<HubPeerResolver>,
    /// Session factory installed on the leader's `transfer` module.
    pub session_factory: Arc<dyn SessionFactory>,
    /// The leader's velo runtime.
    pub velo_runtime: Arc<Velo>,
    /// Tokio handle for spawning transfer/coordinator work.
    pub tokio_handle: tokio::runtime::Handle,
}

/// Register `Feature::P2P` (+ `extra_features`, + Indexer when effective) and
/// build the P2P transfer foundation. `layout_compat` is the mandatory P2P
/// payload (built by the caller from worker metadata). `engine_leader` is the
/// built [`InstanceLeader`]; its control-plane `transfer` module receives the
/// session factory here.
pub(crate) async fn wire_p2p(
    inner: &Arc<ConnectorLeader>,
    engine_leader: &Arc<InstanceLeader>,
    handshake: &HubHandshake,
    layout_compat: LayoutCompatPayload,
    extra_features: Vec<Feature>,
) -> Result<P2pFoundation> {
    let velo_runtime = inner
        .runtime
        .velo()
        .context("P2P features require a KvbmRuntime built with a Velo (got bare Messenger only)")?
        .clone();
    let tokio_handle = inner.runtime.tokio().clone();

    let hub = build_hub_client(&handshake.url)?;
    // Install hub velo handlers (heartbeat) so the hub's liveness probe doesn't
    // unregister us (which would drop our P2P/transfer participation).
    hub.register_handlers_messenger(velo_runtime.messenger())
        .context("installing hub velo handlers for P2P registration")?;

    // Single registration: P2P + caller extras (+ Indexer when effective).
    let mut features = vec![Feature::P2P(P2pConfig { layout_compat })];
    features.extend(extra_features);
    if handshake.has(kvbm_hub::FeatureKey::Indexer) {
        features.push(Feature::Indexer(IndexerFeatureConfig {
            max_seq_len: inner.runtime.config().max_seq_len,
        }));
    }
    let hub_velo_id = hub
        .register_instance_with_features_and_runtime(
            velo_runtime.peer_info(),
            features,
            handshake.runtime_summary.clone(),
        )
        .await
        .with_context(|| {
            format!(
                "registering P2P features with kvbm-hub at {}",
                handshake.url
            )
        })?;

    // One hub-backed peer resolver shared by the session factory (and, for CD,
    // the coordinator) so its de-dup cache works across both paths.
    let peer_resolver = HubPeerResolver::new(Arc::clone(&hub), Arc::clone(&velo_runtime));
    let session_factory: Arc<dyn SessionFactory> =
        kvbm_engine::disagg::session::VeloSessionFactory::with_peer_resolver(
            Arc::clone(&velo_runtime),
            Arc::clone(engine_leader),
            tokio_handle.clone(),
            Arc::clone(&peer_resolver) as Arc<dyn EnginePeerResolver>,
        );
    // Hand the factory to the engine control plane's `transfer` module
    // (registered earlier with an empty cell). Idempotent.
    engine_leader.set_session_factory(Arc::clone(&session_factory));

    spawn_describe_push(inner, engine_leader, &hub, hub_velo_id);

    Ok(P2pFoundation {
        hub,
        hub_velo_id,
        peer_resolver,
        session_factory,
        velo_runtime,
        tokio_handle,
    })
}

/// Push the leader's `InstanceDescription` to the hub (steady-state describe
/// path). Injects `hub_instance_id` + `config_blob`, then spawns a task that
/// briefly settles, calls `describe()`, and POSTs the result; failures fall
/// back to the hub's pull path.
fn spawn_describe_push(
    inner: &Arc<ConnectorLeader>,
    engine_leader: &Arc<InstanceLeader>,
    hub: &Arc<HubClient>,
    hub_velo_id: Option<InstanceId>,
) {
    if let Some(hub_id) = hub_velo_id {
        engine_leader.set_hub_instance_id(hub_id);
    }
    match serde_json::to_value(inner.runtime.config()) {
        Ok(blob) => {
            engine_leader.set_config_blob(blob);
        }
        Err(e) => tracing::warn!(
            error = %e,
            "failed to serialise KvbmConfig for describe push; continuing without config"
        ),
    }
    let hub = Arc::clone(hub);
    let leader = Arc::clone(engine_leader);
    let instance_id = inner.runtime.messenger().instance_id();
    inner.runtime.tokio().spawn(async move {
        // Brief settle so workers can stamp layouts before the first push.
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        let describe =
            tokio::time::timeout(std::time::Duration::from_secs(5), leader.describe()).await;
        match describe {
            Ok(Ok(payload)) => {
                if let Err(e) = hub.push_describe(instance_id, &payload).await {
                    tracing::warn!(error = %e, "initial describe push failed; hub can pull via ?force=true");
                } else {
                    tracing::info!(instance = %instance_id, workers = payload.workers.len(), "describe pushed to hub");
                }
            }
            Ok(Err(e)) => tracing::warn!(error = %e, "leader.describe() failed; hub describe stays pending until forced pull"),
            Err(_) => tracing::warn!("leader.describe() timed out after 5s; hub describe stays pending until forced pull"),
        }
    });
}
