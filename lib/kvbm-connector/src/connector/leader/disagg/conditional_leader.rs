// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Role-dispatching conditional-disaggregation leader.
//!
//! `ConditionalDisaggLeader` is a thin [`ConnectorLeaderApi`]
//! wrapper that delegates every call to a role-specific inner
//! implementation: [`super::decode_leader::DecodeDisaggLeader`]
//! for decode participants, [`super::prefill_leader::PrefillDisaggLeader`]
//! for prefill participants. The split keeps the per-role state
//! machines independent and lets `init.rs` construct the right
//! inner type based on `DisaggConfig::role`.
//!
//! This module also owns the hub-registration helper used by both
//! roles (`register_with_hub`).

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use kvbm_config::{DisaggConfig, DisaggregationRole};
use kvbm_hub::{ConditionalDisaggClient, ConditionalDisaggRole, HubClient, HubClientBuilder};
use url::Url;
use velo::{InstanceId, Messenger};

use crate::BlockId;
use crate::connector::leader::scheduler::{KvConnectorMetadata, SchedulerOutput};
use crate::connector::leader::{FinishedStatus, Request};

use super::ConnectorLeaderApi;

// ============================================================================
// Hub registration (shared)
// ============================================================================

pub async fn register_with_hub(
    config: &DisaggConfig,
    messenger: Arc<Messenger>,
) -> Result<(
    Arc<HubClient>,
    Arc<ConditionalDisaggClient>,
    Option<InstanceId>,
)> {
    let hub = build_hub_client(&config.hub_url)?;

    hub.register_handlers_messenger(&messenger)
        .context("installing hub velo handlers on leader messenger")?;

    let cd_role = role_to_hub(config.role);
    let client =
        ConditionalDisaggClient::with_messenger(Arc::clone(&hub), messenger.clone(), cd_role);

    let peer_info = messenger.peer_info();
    let hub_velo_id = client
        .register(peer_info)
        .await
        .with_context(|| format!("registering with kvbm-hub at {}", config.hub_url))?;

    tracing::info!(
        role = ?config.role,
        hub_url = %config.hub_url,
        hub_velo_id = ?hub_velo_id,
        "Registered with kvbm-hub"
    );

    Ok((hub, client, hub_velo_id))
}

fn role_to_hub(role: DisaggregationRole) -> ConditionalDisaggRole {
    match role {
        DisaggregationRole::Prefill => ConditionalDisaggRole::Prefill,
        DisaggregationRole::Decode => ConditionalDisaggRole::Decode,
    }
}

fn build_hub_client(hub_url: &str) -> Result<Arc<HubClient>> {
    let url =
        Url::parse(hub_url).with_context(|| format!("parsing disagg hub_url: {}", hub_url))?;
    let scheme = url.scheme().to_string();
    let host = match url.host_str() {
        Some(h) if !h.is_empty() => h.to_string(),
        _ => return Err(anyhow!("disagg hub_url has no host: {}", hub_url)),
    };
    let discovery_port = url.port_or_known_default().unwrap_or(1337);

    HubClientBuilder::new()
        .scheme(scheme)
        .host(host)
        .discovery_port(discovery_port)
        .build()
}

// ============================================================================
// Dispatcher
// ============================================================================

/// Role-dispatching `ConnectorLeaderApi` wrapper.
///
/// Holds an `Arc<dyn ConnectorLeaderApi>` constructed at init
/// time from the role-specific implementation
/// (`DecodeDisaggLeader` or `PrefillDisaggLeader`) and forwards
/// every method to it. Carries shared cross-role hub
/// registration handles for diagnostics and future use.
pub struct ConditionalDisaggLeader {
    role: DisaggregationRole,
    inner: Arc<dyn ConnectorLeaderApi>,
    hub: Option<Arc<HubClient>>,
    client: Option<Arc<ConditionalDisaggClient>>,
    hub_velo_id: Option<InstanceId>,
}

impl std::fmt::Debug for ConditionalDisaggLeader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConditionalDisaggLeader")
            .field("role", &self.role)
            .field("hub_velo_id", &self.hub_velo_id)
            .finish()
    }
}

impl ConditionalDisaggLeader {
    pub fn new(
        role: DisaggregationRole,
        inner: Arc<dyn ConnectorLeaderApi>,
        hub: Option<Arc<HubClient>>,
        client: Option<Arc<ConditionalDisaggClient>>,
        hub_velo_id: Option<InstanceId>,
    ) -> Arc<Self> {
        Arc::new(Self {
            role,
            inner,
            hub,
            client,
            hub_velo_id,
        })
    }

    pub fn role(&self) -> DisaggregationRole {
        self.role
    }

    pub fn hub(&self) -> Option<&Arc<HubClient>> {
        self.hub.as_ref()
    }

    pub fn client(&self) -> Option<&Arc<ConditionalDisaggClient>> {
        self.client.as_ref()
    }

    pub fn hub_velo_id(&self) -> Option<InstanceId> {
        self.hub_velo_id
    }
}

impl ConnectorLeaderApi for ConditionalDisaggLeader {
    fn create_slot(&self, request: Request) -> Result<()> {
        self.inner.create_slot(request)
    }

    fn has_slot(&self, request_id: &str) -> bool {
        self.inner.has_slot(request_id)
    }

    fn extend_slot_tokens(&self, request_id: &str, tokens: Vec<u32>) -> Result<()> {
        self.inner.extend_slot_tokens(request_id, tokens)
    }

    fn get_num_new_matched_tokens(
        &self,
        request_id: &str,
        num_computed_tokens: usize,
    ) -> Result<(Option<usize>, bool)> {
        self.inner
            .get_num_new_matched_tokens(request_id, num_computed_tokens)
    }

    fn update_state_after_alloc(
        &self,
        request_id: &str,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> Result<()> {
        self.inner
            .update_state_after_alloc(request_id, block_ids, num_external_tokens)
    }

    fn build_connector_meta(&self, output: SchedulerOutput) -> Result<KvConnectorMetadata> {
        self.inner.build_connector_meta(output)
    }

    fn update_connector_output(
        &self,
        finished_sending: HashSet<String>,
        finished_recving: HashSet<String>,
    ) -> Result<()> {
        self.inner
            .update_connector_output(finished_sending, finished_recving)
    }

    fn request_finished(&self, request_id: &str) -> FinishedStatus {
        self.inner.request_finished(request_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_maps_to_hub_role() {
        assert_eq!(
            role_to_hub(DisaggregationRole::Prefill),
            ConditionalDisaggRole::Prefill
        );
        assert_eq!(
            role_to_hub(DisaggregationRole::Decode),
            ConditionalDisaggRole::Decode
        );
    }

    #[test]
    fn build_hub_client_accepts_explicit_port() {
        let client = build_hub_client("http://127.0.0.1:1337").unwrap();
        assert_eq!(client.config().discovery_url.port(), Some(1337));
    }

    #[test]
    fn build_hub_client_rejects_malformed_url() {
        assert!(build_hub_client("not a url").is_err());
    }

    #[test]
    fn build_hub_client_defaults_control_port() {
        let client = build_hub_client("http://127.0.0.1:1337").unwrap();
        assert_eq!(client.config().control_url.port(), Some(8337));
    }
}
