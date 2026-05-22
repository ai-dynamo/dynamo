// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Client-side wrapper around [`HubClient`] for the ConditionalDisagg feature.

use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use tokio::sync::OnceCell;
use velo::Messenger;
use velo::discovery::PeerDiscovery;
use velo::queue::NextOptions;
use velo::queue::backends::messenger::{MessengerQueueBackend, MessengerQueueConfig};
use velo_ext::{InstanceId, PeerInfo};

use super::protocol::ConditionalDisaggInstancesResponse;
use crate::client::HubClient;
use crate::protocol::{
    self, ConditionalDisaggConfig, ConditionalDisaggRole, Feature, LayoutCompatPayload, P2pConfig,
    PrefillRequest, RuntimeConfigSummary,
};

/// Thin wrapper that registers an instance under the ConditionalDisagg
/// feature and exposes helpers for peer discovery and prefill-queue IO.
///
/// Construction requires the [`HubClient`] (HTTP discovery surface) and an
/// [`Arc<Messenger>`] (used to open a [`MessengerQueueBackend`] targeting the
/// hub for prefill enqueue / dequeue). Use [`ConditionalDisaggClient::new`]
/// when you have a full [`velo::Velo`], or
/// [`ConditionalDisaggClient::with_messenger`] when only a bare [`Messenger`]
/// is available (e.g. from [`kvbm_engine::runtime::KvbmRuntime`]).
pub struct ConditionalDisaggClient {
    hub: Arc<HubClient>,
    messenger: Arc<Messenger>,
    role: ConditionalDisaggRole,
    /// Hub's velo `InstanceId`, learned on `register()`. Queue RPCs need
    /// this to address the hub-side queue service.
    hub_velo_id: OnceLock<InstanceId>,
    /// Lazily-built queue backend targeting the hub. Constructed on first
    /// prefill-queue call.
    queue_backend: OnceCell<Arc<MessengerQueueBackend>>,
}

impl std::fmt::Debug for ConditionalDisaggClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConditionalDisaggClient")
            .field("role", &self.role)
            .field("hub_velo_id", &self.hub_velo_id.get())
            .finish()
    }
}

impl ConditionalDisaggClient {
    /// Wrap a [`HubClient`] + [`velo::Velo`] and declare a role.
    pub fn new(
        hub: Arc<HubClient>,
        velo: Arc<velo::Velo>,
        role: ConditionalDisaggRole,
    ) -> Arc<Self> {
        Self::with_messenger(hub, velo.messenger().clone(), role)
    }

    /// Wrap a [`HubClient`] + [`Arc<Messenger>`] and declare a role.
    ///
    /// Prefer this constructor when only a bare [`Messenger`] is available.
    /// The [`Messenger`] is the only piece of [`velo::Velo`] this client
    /// actually uses (for the hub-targeted prefill queue backend), so a full
    /// [`velo::Velo`] instance is not required.
    pub fn with_messenger(
        hub: Arc<HubClient>,
        messenger: Arc<Messenger>,
        role: ConditionalDisaggRole,
    ) -> Arc<Self> {
        Arc::new(Self {
            hub,
            messenger,
            role,
            hub_velo_id: OnceLock::new(),
            queue_backend: OnceCell::new(),
        })
    }

    /// Role this participant was built with.
    pub fn role(&self) -> ConditionalDisaggRole {
        self.role
    }

    /// Seed the hub's velo `InstanceId` when registration was performed
    /// elsewhere (e.g. the shared P2P foundation registered `[P2P, CD, ...]` in
    /// one call). Idempotent; needed before prefill-queue calls. No-op on `None`.
    pub fn set_hub_velo_id(&self, hub_velo_id: Option<InstanceId>) {
        if let Some(id) = hub_velo_id {
            let _ = self.hub_velo_id.set(id);
        }
    }

    /// Underlying [`HubClient`] — useful for peer lookups that aren't
    /// feature-scoped.
    pub fn hub(&self) -> &Arc<HubClient> {
        &self.hub
    }

    /// Register the participant with the hub, declaring the CD feature
    /// alongside the mandatory P2P feature carrying `layout_compat`.
    ///
    /// On success, caches the hub's velo `InstanceId` internally so
    /// subsequent prefill-queue calls can target it without re-reading the
    /// response.
    ///
    /// `layout_compat` is mandatory — c2 removed the opt-in bypass path.
    pub async fn register(
        &self,
        peer_info: PeerInfo,
        layout_compat: LayoutCompatPayload,
    ) -> Result<Option<InstanceId>> {
        self.register_with(peer_info, layout_compat, Vec::new(), None)
            .await
    }

    /// Like [`register`](Self::register) but additionally declares
    /// `extra_features` (e.g. `Feature::Indexer`) and an optional must-match
    /// [`RuntimeConfigSummary`], all in a single `POST /v1/instances`. The
    /// mandatory `P2P` + `ConditionalDisagg` features are always prepended.
    pub async fn register_with(
        &self,
        peer_info: PeerInfo,
        layout_compat: LayoutCompatPayload,
        extra_features: Vec<Feature>,
        runtime: Option<RuntimeConfigSummary>,
    ) -> Result<Option<InstanceId>> {
        let mut features = vec![
            Feature::P2P(P2pConfig { layout_compat }),
            Feature::ConditionalDisagg(ConditionalDisaggConfig { role: self.role }),
        ];
        features.extend(extra_features);
        let hub_id = match runtime {
            Some(rt) => {
                self.hub
                    .register_instance_with_features_and_runtime(peer_info, features, rt)
                    .await?
            }
            None => {
                self.hub
                    .register_instance_with_features(peer_info, features)
                    .await?
            }
        };
        if let Some(id) = hub_id {
            let _ = self.hub_velo_id.set(id);
        }
        Ok(hub_id)
    }

    /// Fetch the full CD role split from the hub (uses the discovery port).
    pub async fn list_instances(&self) -> Result<ConditionalDisaggInstancesResponse> {
        let url = self
            .hub
            .config()
            .discovery_url
            .join(&format!(
                "/v1/features/{}{}",
                super::protocol::ROUTE_PREFIX,
                super::protocol::paths::INSTANCES
            ))
            .context("joining CD list path")?;
        let resp = reqwest::get(url)
            .await
            .context("GET /v1/features/disagg/instances")?;
        if !resp.status().is_success() {
            return Err(anyhow!("CD list endpoint returned {}", resp.status()));
        }
        resp.json::<ConditionalDisaggInstancesResponse>()
            .await
            .context("decoding CD list response")
    }

    /// Poll the CD list endpoint until an instance of `role` is present,
    /// then resolve its [`PeerInfo`] via the hub's `PeerDiscovery` surface.
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

    fn hub_velo_id(&self) -> Result<InstanceId> {
        self.hub_velo_id.get().copied().ok_or_else(|| {
            anyhow!(
                "hub velo instance id unknown — call register() against a hub configured with a velo transport first"
            )
        })
    }

    async fn queue_backend(&self) -> Result<&Arc<MessengerQueueBackend>> {
        self.queue_backend
            .get_or_try_init(|| async {
                let hub_id = self.hub_velo_id()?;
                Ok::<_, anyhow::Error>(Arc::new(MessengerQueueBackend::new(
                    self.messenger.clone(),
                    hub_id,
                    MessengerQueueConfig::default(),
                )))
            })
            .await
    }

    /// Enqueue a [`PrefillRequest`] on the hub's CD prefill queue.
    ///
    /// Only valid for Decode participants — returns an error if this client's
    /// role is not [`ConditionalDisaggRole::Decode`].
    pub async fn push_prefill_request(&self, req: &PrefillRequest) -> Result<()> {
        if self.role != ConditionalDisaggRole::Decode {
            return Err(anyhow!(
                "push_prefill_request is only valid for Decode participants (this client is {:?})",
                self.role
            ));
        }
        let encoded = serde_json::to_vec(req).context("encoding PrefillRequest")?;
        let backend = self.queue_backend().await?;
        let tx = velo::queue::sender::<Vec<u8>>(backend.as_ref(), protocol::CD_PREFILL_QUEUE)
            .await
            .context("building CD prefill queue sender")?;
        tx.enqueue(&encoded)
            .await
            .context("enqueue PrefillRequest")?;
        Ok(())
    }

    /// Dequeue a [`PrefillRequest`] from the hub's CD prefill queue,
    /// blocking for up to `timeout`. Returns `Ok(None)` when the window
    /// elapses with no item available.
    ///
    /// Only valid for Prefill participants — returns an error if this
    /// client's role is not [`ConditionalDisaggRole::Prefill`].
    pub async fn pull_prefill_request(&self, timeout: Duration) -> Result<Option<PrefillRequest>> {
        if self.role != ConditionalDisaggRole::Prefill {
            return Err(anyhow!(
                "pull_prefill_request is only valid for Prefill participants (this client is {:?})",
                self.role
            ));
        }
        let backend = self.queue_backend().await?;
        let rx = velo::queue::receiver::<Vec<u8>>(backend.as_ref(), protocol::CD_PREFILL_QUEUE)
            .await
            .context("building CD prefill queue receiver")?;
        let batch = rx
            .next_with_options(NextOptions::new().batch_size(1).timeout(timeout))
            .await
            .context("dequeue PrefillRequest")?;
        batch
            .into_iter()
            .next()
            .map(|bytes| serde_json::from_slice(&bytes).context("decoding PrefillRequest"))
            .transpose()
    }
}
