// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Client-side wrapper around [`HubClient`] for the ConditionalDisagg feature.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use velo::discovery::PeerDiscovery;
use velo_common::{InstanceId, PeerInfo};

use crate::client::HubClient;
use crate::protocol::{
    self, ConditionalDisaggConfig, ConditionalDisaggInstancesResponse, ConditionalDisaggRole,
    Feature,
};

/// Thin wrapper that registers an instance under the ConditionalDisagg
/// feature and exposes helpers for discovering peers by role.
pub struct ConditionalDisaggClient {
    hub: Arc<HubClient>,
    role: ConditionalDisaggRole,
}

impl std::fmt::Debug for ConditionalDisaggClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConditionalDisaggClient")
            .field("role", &self.role)
            .finish()
    }
}

impl ConditionalDisaggClient {
    /// Wrap a [`HubClient`] and declare a role for this participant.
    pub fn new(hub: Arc<HubClient>, role: ConditionalDisaggRole) -> Arc<Self> {
        Arc::new(Self { hub, role })
    }

    /// Role this participant was built with.
    pub fn role(&self) -> ConditionalDisaggRole {
        self.role
    }

    /// Underlying [`HubClient`] — useful for peer lookups that aren't
    /// feature-scoped.
    pub fn hub(&self) -> &Arc<HubClient> {
        &self.hub
    }

    /// Register the participant with the hub, declaring the CD feature.
    pub async fn register(&self, peer_info: PeerInfo) -> Result<Option<InstanceId>> {
        self.hub
            .register_instance_with_features(
                peer_info,
                vec![Feature::ConditionalDisagg(Some(ConditionalDisaggConfig {
                    role: self.role,
                }))],
            )
            .await
    }

    /// Fetch the full CD role split from the hub (uses the discovery port).
    pub async fn list_instances(&self) -> Result<ConditionalDisaggInstancesResponse> {
        let url = self
            .hub
            .config()
            .discovery_url
            .join(protocol::paths::CD_INSTANCES)
            .context("joining CD list path")?;
        let resp = reqwest::get(url)
            .await
            .context("GET /v1/features/conditional-disagg/instances")?;
        if !resp.status().is_success() {
            return Err(anyhow!("CD list endpoint returned {}", resp.status()));
        }
        resp.json::<ConditionalDisaggInstancesResponse>()
            .await
            .context("decoding CD list response")
    }

    /// Poll the CD list endpoint until an instance of `role` is present,
    /// then resolve its [`PeerInfo`] via the hub's `PeerDiscovery` surface.
    ///
    /// Returns an error if no such peer appears within `timeout`.
    pub async fn await_peer_of_role(
        &self,
        role: ConditionalDisaggRole,
        poll: Duration,
        timeout: Duration,
    ) -> Result<PeerInfo> {
        let deadline = Instant::now() + timeout;
        loop {
            let snap = self.list_instances().await?;
            let ids = match role {
                ConditionalDisaggRole::Prefill => &snap.prefill,
                ConditionalDisaggRole::Decode => &snap.decode,
            };
            if let Some(first) = ids.first().copied() {
                return self
                    .hub
                    .discover_by_instance_id(first)
                    .await
                    .with_context(|| format!("resolving PeerInfo for {first}"));
            }
            if Instant::now() >= deadline {
                return Err(anyhow!(
                    "timed out waiting for a ConditionalDisagg peer in role {role:?}"
                ));
            }
            tokio::time::sleep(poll).await;
        }
    }
}
