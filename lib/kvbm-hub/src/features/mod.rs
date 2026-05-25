// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Feature-scoped managers that plug into the hub.
//!
//! A [`FeatureManager`] owns state for one feature (e.g. ConditionalDisagg),
//! contributes axum routes to both listeners, and is notified when instances
//! register or unregister. Managers are attached to the hub via
//! [`HubServerBuilder::add_feature_manager`](crate::HubServerBuilder::add_feature_manager).

use std::sync::Arc;

use axum::Router;
use futures::future::BoxFuture;
use tokio_util::sync::CancellationToken;
use velo_ext::{InstanceId, PeerInfo};

pub use crate::protocol::FeatureConfigRequirements;
use crate::protocol::{Feature, FeatureKey, PrimaryConfig};
use crate::registry::PeerRegistry;

/// Client-side per-feature `kvbmctl` CLI surface. Gated behind the `kvbmctl`
/// feature (it pulls in `clap` usage + the client trait); the default hub build
/// excludes it.
#[cfg(feature = "kvbmctl")]
pub mod cli;
pub mod control_plane;
pub mod disagg;
pub(crate) mod http;
pub mod indexer;
pub mod p2p;
pub mod prefill_router;

/// Context handed to a [`FeatureManager`] at hub startup so it can stash any
/// references it needs (e.g. the hub's Velo handle for active messaging).
#[derive(Clone)]
pub struct HubContext {
    /// The hub's Velo instance — present only when the hub was configured
    /// with at least one transport.
    pub velo: Option<Arc<velo::Velo>>,
    /// The shared registry backing peer discovery.
    pub registry: Arc<dyn PeerRegistry>,
    /// Child of the hub's master shutdown token. Managers spawning background
    /// work (refresh tasks, watchers) should hang it off this so they exit on
    /// hub shutdown. Existing managers may ignore it.
    pub cancel: CancellationToken,
}

impl std::fmt::Debug for HubContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HubContext")
            .field("velo_attached", &self.velo.is_some())
            .finish()
    }
}

/// Errors a [`FeatureManager`] may return during registration dispatch.
#[derive(Debug, thiserror::Error)]
pub enum FeatureError {
    /// The feature payload is missing or malformed for this manager.
    #[error("feature config invalid: {0}")]
    InvalidConfig(String),
    /// The manager was handed a [`Feature`] whose key doesn't match its own.
    /// Indicates a routing bug in the server dispatcher.
    #[error("feature key mismatch: manager={manager:?} payload={payload:?}")]
    KeyMismatch {
        /// The manager's declared key.
        manager: FeatureKey,
        /// The payload's actual key.
        payload: FeatureKey,
    },
    /// Any other failure.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Trait implemented by per-feature managers.
///
/// Managers are type-erased behind `Arc<dyn FeatureManager>` — the hub
/// dispatches by [`FeatureKey`] at registration time and merges each
/// manager's `Router`s into the appropriate listener.
pub trait FeatureManager: Send + Sync + 'static {
    /// Stable discriminant this manager handles.
    fn key(&self) -> FeatureKey;

    /// Other features this one depends on. A registration declaring this
    /// feature MUST also declare every dependency (transitively); the server
    /// enforces the closure pre-dispatch in
    /// [`register_instance`](crate::server). Default: no dependencies.
    ///
    /// Example: `ConditionalDisagg` returns `&[FeatureKey::P2P]`.
    fn dependencies(&self) -> &'static [FeatureKey] {
        &[]
    }

    /// Which [`PrimaryConfig`] must-match fields a registrant must declare and
    /// match for this feature. Default: none.
    fn config_requirements(&self) -> FeatureConfigRequirements {
        FeatureConfigRequirements::default()
    }

    /// The authoritative must-match `block_size` this feature owns, if any. The
    /// hub reconciles it into [`PrimaryConfig::block_size`] at startup
    /// ([`HubServerBuilder::serve`](crate::HubServerBuilder::serve)) — filling
    /// an unset primary and rejecting an explicit primary that conflicts — so
    /// registrant validation always has a source of truth even when the
    /// operator did not set `primary` explicitly.
    ///
    /// Default `None` (the feature owns no sizing). KV-index returns its index
    /// block size. (`max_seq_len` is *not* reconciled — it is advisory and
    /// grows dynamically per registrant.)
    fn authoritative_block_size(&self) -> Option<usize> {
        None
    }

    /// Whether a registrant declaring this feature MUST include a
    /// [`RuntimeConfigSummary`](crate::protocol::RuntimeConfigSummary) — i.e.
    /// must-match validation may not be skipped for it.
    ///
    /// Default `false`: features that predate the runtime-summary field (P2P,
    /// ConditionalDisagg) tolerate a missing summary so legacy clients keep
    /// registering (their fields are still validated when a summary *is*
    /// present). Features introduced together with the summary (KV-index)
    /// override to `true` so a misconfigured publisher cannot bypass the
    /// block-size / sequence-length consistency check by omitting it.
    fn requires_runtime_summary(&self) -> bool {
        false
    }

    /// Feature-specific config view for the aggregate `GET /v1/config`
    /// response, given the resolved hub `primary`. Default: `null` (the feature
    /// exposes nothing beyond its key + dependencies). KV-index overrides this
    /// to advertise its ZMQ ingest endpoint.
    fn descriptor(&self, _primary: &PrimaryConfig) -> serde_json::Value {
        serde_json::Value::Null
    }

    /// Called exactly once during [`HubServerBuilder::serve`](crate::HubServerBuilder::serve)
    /// after the registry and (optional) hub Velo are built. Implementations
    /// may stash references from the context and perform any async
    /// initialization (e.g. registering velo handlers).
    fn attach<'a>(&'a self, ctx: HubContext) -> BoxFuture<'a, Result<(), FeatureError>>;

    /// Called after base registration succeeds, for every [`Feature`] in the
    /// [`RegisterRequest`](crate::protocol::RegisterRequest) that matches
    /// this manager's [`FeatureKey`].
    ///
    /// Returning `Err` causes the hub to unregister the base entry and
    /// return an error to the client (all-or-nothing semantics).
    fn on_register<'a>(
        &'a self,
        instance_id: InstanceId,
        feature: &'a Feature,
    ) -> BoxFuture<'a, Result<(), FeatureError>>;

    /// Called once per successful `register_instance` for *every* attached
    /// manager, regardless of which [`Feature`] payloads the client declared.
    /// Default no-op — managers opt in to discover post-registration state
    /// (e.g. query the leader's control plane).
    ///
    /// **Errors here MUST NOT roll back base registration.** The leader may
    /// be briefly unreachable for follow-up control queries; that is not a
    /// registration failure. Implementations return `()` and handle their own
    /// retry/backoff internally.
    fn on_register_any<'a>(
        &'a self,
        _instance_id: InstanceId,
        _peer: &'a PeerInfo,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async {})
    }

    /// Called when an instance leaves the registry — either explicitly via
    /// HTTP `DELETE` or implicitly via TTL reaper. Must be idempotent.
    fn on_unregister(&self, instance_id: InstanceId);

    /// Feature-owned URL segment. When `Some("x")`, the server nests this
    /// manager's `control_router`/`public_router` under `/v1/features/x` —
    /// the manager declares **relative** routes (e.g. `/config`) and owns its
    /// whole namespace. Default `None` keeps the legacy behavior where the
    /// manager declares full absolute paths itself (p2p, disagg,
    /// control_plane).
    fn route_prefix(&self) -> Option<&'static str> {
        None
    }

    /// Axum routes mounted on the control-plane listener (port 8337).
    fn control_router(self: Arc<Self>) -> Router;

    /// Axum routes mounted on the public/discovery listener (port 1337).
    fn public_router(self: Arc<Self>) -> Router;
}
