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
use velo_common::InstanceId;

use crate::protocol::{Feature, FeatureKey};
use crate::registry::PeerRegistry;

pub mod conditional_disagg;

/// Context handed to a [`FeatureManager`] at hub startup so it can stash any
/// references it needs (e.g. the hub's Velo handle for active messaging).
#[derive(Clone)]
pub struct HubContext {
    /// The hub's Velo instance — present only when the hub was configured
    /// with at least one transport.
    pub velo: Option<Arc<velo::Velo>>,
    /// The shared registry backing peer discovery.
    pub registry: Arc<dyn PeerRegistry>,
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

    /// Called exactly once during [`HubServerBuilder::serve`](crate::HubServerBuilder::serve)
    /// after the registry and (optional) hub Velo are built. Implementations
    /// may stash references from the context for later use.
    fn attach(&self, ctx: HubContext) -> Result<(), FeatureError>;

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

    /// Called when an instance leaves the registry — either explicitly via
    /// HTTP `DELETE` or implicitly via TTL reaper. Must be idempotent.
    fn on_unregister(&self, instance_id: InstanceId);

    /// Axum routes mounted on the control-plane listener (port 8337).
    fn control_router(self: Arc<Self>) -> Router;

    /// Axum routes mounted on the public/discovery listener (port 1337).
    fn public_router(self: Arc<Self>) -> Router;
}
