// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::{Context, Result};
use dashmap::DashMap;
use upstream_velo_messenger::{Messenger, PeerInfo};
use upstream_velo_transports::tcp::TcpTransportBuilder;

use crate::component::Component;
use crate::traits::DistributedRuntimeProvider;

use super::{DiscoveryQuery, DiscoverySpec};

pub async fn build_default_messenger() -> Result<Arc<Messenger>> {
    let transport = Arc::new(
        TcpTransportBuilder::new()
            .build()
            .context("Failed to build Velo TCP transport")?,
    );

    Messenger::builder()
        .add_transport(transport)
        .build()
        .await
        .context("Failed to build Velo messenger")
}

pub async fn register_local_component_peer(
    component: &Component,
    peer_info: &PeerInfo,
) -> Result<()> {
    let spec = DiscoverySpec::from_velo_peer(
        component.namespace().name(),
        component.name().to_string(),
        peer_info,
    )?;

    component
        .drt()
        .discovery()
        .register(spec)
        .await
        .context("Failed to register Velo peer in discovery")?;

    Ok(())
}

pub struct DiscoveredVeloMessenger {
    component: Component,
    peer_info_cache: DashMap<u64, PeerInfo>,
    messenger: Arc<Messenger>,
}

impl DiscoveredVeloMessenger {
    pub async fn new(component: Component) -> Result<Self> {
        Ok(Self {
            component,
            peer_info_cache: DashMap::new(),
            messenger: build_default_messenger().await?,
        })
    }

    pub fn messenger(&self) -> &Arc<Messenger> {
        &self.messenger
    }

    pub fn invalidate_peer_info(&self, instance_id: u64) {
        self.peer_info_cache.remove(&instance_id);
    }

    pub async fn resolve_peer_info(&self, instance_id: u64) -> Result<PeerInfo> {
        if let Some(peer_info) = self.peer_info_cache.get(&instance_id) {
            return Ok(peer_info.clone());
        }

        let peers = self
            .component
            .drt()
            .discovery()
            .list(DiscoveryQuery::ComponentVeloPeers {
                namespace: self.component.namespace().name(),
                component: self.component.name().to_string(),
            })
            .await
            .context("Failed to list discovered Velo peers")?;

        let Some(peer_instance) = peers
            .into_iter()
            .find(|peer| peer.instance_id() == instance_id)
        else {
            anyhow::bail!("No discovered Velo peer for runtime instance {instance_id}");
        };

        let peer_info = peer_instance
            .deserialize_velo_peer::<PeerInfo>()
            .with_context(|| {
                format!(
                    "Failed to deserialize Velo peer metadata for runtime instance {instance_id}"
                )
            })?;

        self.messenger
            .register_peer(peer_info.clone())
            .context("Failed to register discovered Velo peer")?;
        self.peer_info_cache.insert(instance_id, peer_info.clone());

        Ok(peer_info)
    }
}
