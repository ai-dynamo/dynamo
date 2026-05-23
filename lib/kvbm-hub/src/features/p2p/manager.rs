// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hub-side manager for the P2P feature.

use std::collections::HashSet;
use std::sync::{Arc, OnceLock};

use axum::Json;
use axum::Router;
use axum::extract::{Path, State};
use axum::response::Response;
use axum::routing::post;
use futures::future::BoxFuture;
use parking_lot::RwLock;
use velo_ext::InstanceId;

use crate::features::http::{control_error_response, leader_client, ok_response};
use crate::features::{FeatureError, FeatureManager, HubContext};
use crate::protocol::{self, Feature, FeatureKey, LayoutCompatPayload};
use crate::registry::PeerRegistry;
use kvbm_protocols::control::layout_compat::check_layout_compat;
use kvbm_protocols::control::{
    CloseTransferSessionRequest, OpenTransferSessionRequest, PullFromSessionRequest, SearchRequest,
};

/// Tracks P2P registrations and enforces the layout-compatibility baseline.
///
/// The first P2P registration whose payload passes `validate_self` becomes
/// the baseline. Every subsequent registration is checked against it via
/// `check_layout_compat`. When the last P2P-registered instance unregisters,
/// the baseline clears so a new group can adopt a different layout without
/// bouncing the hub.
pub struct P2pManager {
    inner: RwLock<P2pInner>,
    /// Hub velo handle — needed to issue `/control/transfer/*` RPCs to the
    /// target leader. Set in [`FeatureManager::attach`]; absent when the hub
    /// has no velo transport (transfer routes then return 503).
    velo: OnceLock<Arc<velo::Velo>>,
    /// Shared peer registry — used to confirm a transfer target is currently
    /// registered before issuing the velo call. Set in `attach`.
    registry: OnceLock<Arc<dyn PeerRegistry>>,
}

struct P2pInner {
    /// Instances currently registered with `Feature::P2P`.
    instances: HashSet<InstanceId>,
    /// Layout-compat baseline established by the first valid P2P
    /// registration. Cleared when `instances` becomes empty.
    layout_baseline: Option<LayoutCompatPayload>,
}

impl std::fmt::Debug for P2pManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.read();
        f.debug_struct("P2pManager")
            .field("instance_count", &inner.instances.len())
            .field("has_baseline", &inner.layout_baseline.is_some())
            .field("velo_attached", &self.velo.get().is_some())
            .field("registry_attached", &self.registry.get().is_some())
            .finish()
    }
}

impl Default for P2pManager {
    fn default() -> Self {
        Self::new()
    }
}

impl P2pManager {
    /// Create an empty manager with no baseline.
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(P2pInner {
                instances: HashSet::new(),
                layout_baseline: None,
            }),
            velo: OnceLock::new(),
            registry: OnceLock::new(),
        }
    }

    /// Number of P2P-registered instances currently tracked. Exposed
    /// for tests + diagnostics.
    pub fn instance_count(&self) -> usize {
        self.inner.read().instances.len()
    }

    /// Whether a baseline is currently set. Exposed for tests.
    pub fn has_baseline(&self) -> bool {
        self.inner.read().layout_baseline.is_some()
    }

    /// Validate a describe-push `layout_compat` candidate against the stored
    /// baseline.
    ///
    /// Returns `Ok(())` when:
    /// - The baseline matches the candidate under `check_layout_compat`.
    ///
    /// Returns `Err(FeatureError::InvalidConfig)` when:
    /// - `instance_id` is not a P2P-registered member of this hub — a leader
    ///   that registered without `Feature::P2P` has no group baseline to be
    ///   validated against, so any `layout_compat` it pushes is unverifiable.
    ///   Without this check, a non-member's payload would either get silently
    ///   accepted (matches the unrelated baseline of some other P2P group) or
    ///   rejected with a misleading "diverges from baseline" message.
    /// - No baseline is present (defensive — should be unreachable when the
    ///   membership check above passes, since baseline and `instances` are
    ///   populated together under the same lock).
    /// - The candidate diverges from the baseline (mode, canonical, or
    ///   per-worker fields differ under the operative mode's predicate).
    pub fn check_describe_layout(
        &self,
        instance_id: velo_ext::InstanceId,
        candidate: &LayoutCompatPayload,
    ) -> Result<(), FeatureError> {
        let inner = self.inner.read();
        if !inner.instances.contains(&instance_id) {
            return Err(FeatureError::InvalidConfig(format!(
                "describe for instance {instance_id} carries layout_compat but \
                 this instance is not registered with Feature::P2P; only P2P \
                 members are checked against the group's layout baseline"
            )));
        }
        let baseline = inner.layout_baseline.as_ref().ok_or_else(|| {
            FeatureError::InvalidConfig(format!(
                "describe for instance {instance_id} carries layout_compat but \
                 the P2P baseline is unset (hub restart between register and \
                 describe?)"
            ))
        })?;
        check_layout_compat(baseline, candidate).map_err(|e| {
            FeatureError::InvalidConfig(format!(
                "describe-push layout_compat for {instance_id} diverges from \
                 P2P baseline: {e}"
            ))
        })
    }
}

impl FeatureManager for P2pManager {
    fn key(&self) -> FeatureKey {
        FeatureKey::P2P
    }

    fn config_requirements(&self) -> crate::features::FeatureConfigRequirements {
        // P2P transfers require both peers to agree on block size and the
        // cross-leader layout-compat mode. (The detailed layout_compat payload
        // is still validated separately in `on_register`; this is the cheap
        // primary-consistency gate.) CD inherits these via its P2P dependency.
        crate::features::FeatureConfigRequirements {
            block_size: true,
            block_layout: true,
        }
    }

    fn attach<'a>(&'a self, ctx: HubContext) -> BoxFuture<'a, Result<(), FeatureError>> {
        // Stash the registry + (optional) velo so the `/control/transfer/*`
        // handlers can proxy to the target leader. Mirrors `ControlPlaneManager`.
        Box::pin(async move {
            let _ = self.registry.set(ctx.registry.clone());
            if let Some(v) = ctx.velo.clone() {
                let _ = self.velo.set(v);
            } else {
                tracing::warn!(
                    "P2pManager: hub has no velo transport — \
                     /control/transfer/* routes will return 503"
                );
            }
            Ok(())
        })
    }

    fn on_register<'a>(
        &'a self,
        instance_id: InstanceId,
        feature: &'a Feature,
    ) -> BoxFuture<'a, Result<(), FeatureError>> {
        Box::pin(async move {
            let Feature::P2P(cfg) = feature else {
                return Err(FeatureError::KeyMismatch {
                    manager: FeatureKey::P2P,
                    payload: feature.key(),
                });
            };
            let candidate = &cfg.layout_compat;

            let mut inner = self.inner.write();

            // Idempotent re-register: an instance that's already in the set
            // does not re-validate (the baseline already accepted it once).
            if inner.instances.contains(&instance_id) {
                return Ok(());
            }

            match inner.layout_baseline.as_ref() {
                None => {
                    candidate.validate_self().map_err(|e| {
                        FeatureError::InvalidConfig(format!(
                            "P2P layout_compat payload from instance {instance_id} is \
                             internally inconsistent: {e}"
                        ))
                    })?;
                    inner.layout_baseline = Some(candidate.clone());
                }
                Some(baseline) => check_layout_compat(baseline, candidate).map_err(|e| {
                    FeatureError::InvalidConfig(format!(
                        "P2P layout_compat incompatibility for instance {instance_id}: {e}"
                    ))
                })?,
            }

            inner.instances.insert(instance_id);
            Ok(())
        })
    }

    fn on_unregister(&self, instance_id: InstanceId) {
        let mut inner = self.inner.write();
        inner.instances.remove(&instance_id);
        // Clear the baseline once the last P2P instance leaves so a fresh
        // group can adopt a different mode/shape without bouncing the hub.
        if inner.instances.is_empty() {
            inner.layout_baseline = None;
        }
    }

    fn control_router(self: Arc<Self>) -> Router {
        transfer_routes(self)
    }

    fn public_router(self: Arc<Self>) -> Router {
        // Mirror `ControlPlaneManager`: the transfer surface is mounted on
        // both listeners so callers can reach it on either port.
        transfer_routes(self)
    }
}

/// The `/v1/instances/{id}/control/transfer/*` surface. Block-copy session
/// management is a P2P concern, so these routes only exist when the P2P
/// feature is enabled on the hub.
fn transfer_routes(manager: Arc<P2pManager>) -> Router {
    use protocol::paths::*;
    Router::new()
        .route(CONTROL_TRANSFER_SEARCH_PREFIX, post(transfer_search_prefix))
        .route(
            CONTROL_TRANSFER_SEARCH_SCATTER,
            post(transfer_search_scatter),
        )
        .route(CONTROL_TRANSFER_OPEN_SESSION, post(transfer_open_session))
        .route(
            CONTROL_TRANSFER_PULL_FROM_SESSION,
            post(transfer_pull_from_session),
        )
        .route(CONTROL_TRANSFER_CLOSE_SESSION, post(transfer_close_session))
        .with_state(manager)
}

/// Build a `LeaderControlClient` for `instance_id` via the shared
/// [`http::leader_client`](crate::features::http::leader_client) helper,
/// supplying this manager's own velo + registry handles.
#[allow(clippy::result_large_err)]
fn p2p_leader_client(
    mgr: &P2pManager,
    instance_id: InstanceId,
) -> Result<kvbm_protocols::control::LeaderControlClient, Response> {
    leader_client(&mgr.velo, &mgr.registry, instance_id)
}

/// `POST /control/transfer/search_prefix` — always-on (when P2P is enabled).
async fn transfer_search_prefix(
    State(mgr): State<Arc<P2pManager>>,
    Path(instance_id): Path<InstanceId>,
    Json(req): Json<SearchRequest>,
) -> Response {
    let client = match p2p_leader_client(&mgr, instance_id) {
        Ok(c) => c,
        Err(resp) => return resp,
    };
    match client.transfer().search_prefix(req).await {
        Ok(resp) => ok_response(&resp),
        Err(err) => control_error_response(instance_id, "search_prefix", err),
    }
}

/// `POST /control/transfer/search_scatter` — always-on (when P2P is enabled).
async fn transfer_search_scatter(
    State(mgr): State<Arc<P2pManager>>,
    Path(instance_id): Path<InstanceId>,
    Json(req): Json<SearchRequest>,
) -> Response {
    let client = match p2p_leader_client(&mgr, instance_id) {
        Ok(c) => c,
        Err(resp) => return resp,
    };
    match client.transfer().search_scatter(req).await {
        Ok(resp) => ok_response(&resp),
        Err(err) => control_error_response(instance_id, "search_scatter", err),
    }
}

/// `POST /control/transfer/open_session` — dispatched at the holder. Returns
/// the attach triple in
/// [`kvbm_protocols::control::OpenTransferSessionResponse`].
async fn transfer_open_session(
    State(mgr): State<Arc<P2pManager>>,
    Path(instance_id): Path<InstanceId>,
    Json(req): Json<OpenTransferSessionRequest>,
) -> Response {
    let client = match p2p_leader_client(&mgr, instance_id) {
        Ok(c) => c,
        Err(resp) => return resp,
    };
    match client.transfer().open_session(req).await {
        Ok(resp) => ok_response(&resp),
        Err(err) => control_error_response(instance_id, "open_session", err),
    }
}

/// `POST /control/transfer/pull_from_session` — dispatched at the puller.
/// Long-poll: returns when the pull is complete.
async fn transfer_pull_from_session(
    State(mgr): State<Arc<P2pManager>>,
    Path(instance_id): Path<InstanceId>,
    Json(req): Json<PullFromSessionRequest>,
) -> Response {
    let client = match p2p_leader_client(&mgr, instance_id) {
        Ok(c) => c,
        Err(resp) => return resp,
    };
    match client.transfer().pull_from_session(req).await {
        Ok(resp) => ok_response(&resp),
        Err(err) => control_error_response(instance_id, "pull_from_session", err),
    }
}

/// `POST /control/transfer/close_session` — idempotent.
async fn transfer_close_session(
    State(mgr): State<Arc<P2pManager>>,
    Path(instance_id): Path<InstanceId>,
    Json(req): Json<CloseTransferSessionRequest>,
) -> Response {
    let client = match p2p_leader_client(&mgr, instance_id) {
        Ok(c) => c,
        Err(resp) => return resp,
    };
    match client.transfer().close_session(req).await {
        Ok(resp) => ok_response(&resp),
        Err(err) => control_error_response(instance_id, "close_session", err),
    }
}
