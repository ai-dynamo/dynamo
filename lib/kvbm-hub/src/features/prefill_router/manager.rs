// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hub-side manager for the prefill-router feature.
//!
//! Owns a [`PrefillRouter`] / [`Selector`] pair and translates
//! `Feature::PrefillRouter` registrations into [`PrefillExecutionBackend`]
//! instances added to the fleet at runtime. `on_unregister` removes the
//! worker — in-flight tasks against it complete normally; new picks see
//! it gone.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::StatusCode,
    routing::{get, post},
};
use futures::future::BoxFuture;
use parking_lot::RwLock;
use serde::Deserialize;
use velo_ext::InstanceId;

use super::calibration::{CALIBRATE_HANDLER, CalibrationRequest, CalibrationResponse};
use super::dispatcher::PrefillRequestDispatcher;
use super::execution::{HttpExecutionBackend, PrefillExecutionBackend, VeloExecutionBackend};
use super::protocol::{
    self, CountersResponse, PrefillBackendAdvertisement, PrefillTargetSummary, TargetsResponse,
    WorkerCountersSnapshot,
};
use super::router::PrefillRouter;
use super::selection::{Selector, SelectorConfig};
use crate::features::{FeatureError, FeatureManager, HubContext};
use crate::protocol::{Feature, FeatureKey};

/// Wall-clock guard on a single calibration unary call. Generous because
/// a full sweep up to 32k tokens can take several minutes; the worker
/// runs single-stream and produces an OSL of typically 64 tokens per
/// ISL step.
const CALIBRATE_TIMEOUT: Duration = Duration::from_secs(900);

/// FeatureManager for the prefill router. Owns the [`PrefillRouter`] +
/// [`Selector`]; exposes the router via [`Self::dispatcher`] so the hub
/// binary can install it on the disagg manager as the
/// [`PrefillRequestDispatcher`].
pub struct PrefillRouterManager {
    router: Arc<PrefillRouter>,
    selector: Arc<Selector>,
    /// Side-table mirroring the selector's fleet so `GET /targets` can
    /// surface the original advertisement (the selector itself only
    /// stores the backend trait object — it has no concept of "what was
    /// in the wire payload"). Kept in lock-step with `on_register` /
    /// `on_unregister`.
    advertisements: RwLock<HashMap<InstanceId, PrefillBackendAdvertisement>>,
    /// Hub's own velo handle, stashed by [`Self::attach`] when the hub was
    /// configured with a transport. Required to build velo execution
    /// backends — `on_register` rejects velo advertisements when this is
    /// unset (discovery-only hub).
    velo: OnceLock<Arc<velo::Velo>>,
}

impl std::fmt::Debug for PrefillRouterManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefillRouterManager")
            .field("selector", &self.selector)
            .finish()
    }
}

impl PrefillRouterManager {
    /// Build a manager with the given selector config.
    pub fn new(config: SelectorConfig) -> Arc<Self> {
        let selector = Selector::new(config);
        let router = PrefillRouter::new(Arc::clone(&selector));
        Arc::new(Self {
            router,
            selector,
            advertisements: RwLock::new(HashMap::new()),
            velo: OnceLock::new(),
        })
    }

    /// The [`PrefillRequestDispatcher`] the hub binary hands to the
    /// disagg manager's `start_dispatcher`.
    pub fn dispatcher(self: &Arc<Self>) -> Arc<dyn PrefillRequestDispatcher> {
        Arc::clone(&self.router) as Arc<dyn PrefillRequestDispatcher>
    }

    /// Underlying selector (mainly for tests and the HTTP introspection
    /// endpoints).
    pub fn selector(&self) -> &Arc<Selector> {
        &self.selector
    }

    /// Snapshot of every registered prefill target.
    pub fn targets(&self) -> TargetsResponse {
        let snapshot = self.selector.snapshot();
        let advertisements = self.advertisements.read();
        let mut targets: Vec<PrefillTargetSummary> = snapshot
            .iter()
            .filter_map(|slot| {
                let advertisement = advertisements.get(&slot.instance_id)?.clone();
                Some(PrefillTargetSummary {
                    instance_id: slot.instance_id,
                    backend: slot.backend.label().to_string(),
                    advertisement,
                })
            })
            .collect();
        targets.sort_by(|a, b| a.instance_id.to_string().cmp(&b.instance_id.to_string()));
        TargetsResponse { targets }
    }

    /// Snapshot of per-worker counters and remaining fleet capacity.
    pub fn counters(&self) -> CountersResponse {
        let snapshot = self.selector.snapshot();
        let mut workers: Vec<WorkerCountersSnapshot> = snapshot
            .iter()
            .map(|slot| {
                let c = slot.counters();
                WorkerCountersSnapshot {
                    instance_id: slot.instance_id,
                    backend: slot.backend.label().to_string(),
                    inflight: c.inflight,
                    load_net_new: c.load_net_new,
                }
            })
            .collect();
        workers.sort_by(|a, b| a.instance_id.to_string().cmp(&b.instance_id.to_string()));
        CountersResponse {
            workers,
            available_permits: self.selector.available_permits(),
        }
    }
}

impl FeatureManager for PrefillRouterManager {
    fn key(&self) -> FeatureKey {
        FeatureKey::PrefillRouter
    }

    fn attach<'a>(&'a self, ctx: HubContext) -> BoxFuture<'a, Result<(), FeatureError>> {
        Box::pin(async move {
            // Stash the hub's velo handle (if any) so on_register can build
            // velo execution backends. A discovery-only hub leaves `velo`
            // unset and velo advertisements are rejected at register time.
            if let Some(velo) = ctx.velo {
                let _ = self.velo.set(velo);
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
            let Feature::PrefillRouter(cfg) = feature else {
                return Err(FeatureError::KeyMismatch {
                    manager: FeatureKey::PrefillRouter,
                    payload: feature.key(),
                });
            };
            let backend: Arc<dyn PrefillExecutionBackend> = match &cfg.backend {
                PrefillBackendAdvertisement::Http(endpoint) => {
                    HttpExecutionBackend::new(instance_id, endpoint.clone()).map_err(|e| {
                        FeatureError::InvalidConfig(format!("HTTP backend build failed: {e}"))
                    })?
                }
                PrefillBackendAdvertisement::Velo {
                    instance_id: target,
                } => {
                    let velo = self.velo.get().ok_or_else(|| {
                        FeatureError::InvalidConfig(
                            "velo prefill backend requires the hub to have a velo transport \
                             (start kvbm_hub with --velo-port)"
                                .to_string(),
                        )
                    })?;
                    VeloExecutionBackend::new(*target, velo.messenger().clone())
                }
            };
            let newly_added = self.selector.add_worker(instance_id, backend);
            self.advertisements
                .write()
                .insert(instance_id, cfg.backend.clone());
            tracing::info!(
                instance_id = %instance_id,
                backend = cfg.backend.label(),
                newly_added,
                "PrefillRouter: target registered"
            );
            Ok(())
        })
    }

    fn on_unregister(&self, instance_id: InstanceId) {
        self.selector.remove_worker(instance_id);
        self.advertisements.write().remove(&instance_id);
        tracing::info!(
            instance_id = %instance_id,
            "PrefillRouter: target removed"
        );
    }

    fn route_prefix(&self) -> Option<&'static str> {
        Some(protocol::ROUTE_PREFIX)
    }

    fn control_router(self: Arc<Self>) -> Router {
        routes(self)
    }

    fn public_router(self: Arc<Self>) -> Router {
        routes(self)
    }
}

fn routes(manager: Arc<PrefillRouterManager>) -> Router {
    Router::new()
        .route(protocol::paths::TARGETS, get(get_targets))
        .route(protocol::paths::COUNTERS, get(get_counters))
        .route(protocol::paths::CALIBRATE, post(post_calibrate))
        .with_state(manager)
}

async fn get_targets(State(mgr): State<Arc<PrefillRouterManager>>) -> Json<TargetsResponse> {
    Json(mgr.targets())
}

async fn get_counters(State(mgr): State<Arc<PrefillRouterManager>>) -> Json<CountersResponse> {
    Json(mgr.counters())
}

/// Query params for `POST /calibrate/:instance_id`. `force` is the only
/// query knob; everything else is body fields on the JSON request.
#[derive(Debug, Default, Deserialize)]
struct CalibrateQuery {
    #[serde(default)]
    force: Option<bool>,
}

/// HTTP proxy that forwards a `CalibrationRequest` to the worker's velo
/// `CALIBRATE_HANDLER` and returns the `CalibrationResponse` body.
///
/// Error mapping:
/// - 400 if the instance_id path segment isn't a valid UUID.
/// - 404 if the named worker isn't registered with the prefill router.
/// - 409 if the worker is already calibrating or has in-flight prefill
///   requests (the worker raises `already_calibrating` / `prefill_busy`).
/// - 504 on velo unary timeout.
/// - 500 on every other transport / handler failure (body carries the
///   formatted reason).
async fn post_calibrate(
    State(mgr): State<Arc<PrefillRouterManager>>,
    Path(instance_id_str): Path<String>,
    Query(query): Query<CalibrateQuery>,
    Json(mut body): Json<CalibrationRequest>,
) -> Result<Json<CalibrationResponse>, (StatusCode, String)> {
    let uuid = uuid::Uuid::parse_str(&instance_id_str).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            format!("instance_id is not a valid uuid: {e}"),
        )
    })?;
    let instance_id = InstanceId::from(uuid);

    if !mgr.advertisements.read().contains_key(&instance_id) {
        return Err((
            StatusCode::NOT_FOUND,
            format!("no prefill target registered for instance_id={instance_id}"),
        ));
    }

    if let Some(force) = query.force {
        body.force = force;
    }

    let velo = mgr.velo.get().ok_or_else(|| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "hub has no velo transport; calibrate proxy requires a velo-equipped hub \
             (start kvbm_hub with --velo-port)"
                .to_string(),
        )
    })?;

    let call = velo
        .messenger()
        .typed_unary::<CalibrationResponse>(CALIBRATE_HANDLER)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("typed_unary({CALIBRATE_HANDLER}) builder: {e}"),
            )
        })?
        .payload(&body)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("encode CalibrationRequest: {e}"),
            )
        })?
        .instance(instance_id)
        .send();

    match tokio::time::timeout(CALIBRATE_TIMEOUT, call).await {
        Ok(Ok(resp)) => Ok(Json(resp)),
        Ok(Err(err)) => {
            let msg = err.to_string();
            let code = if msg.contains("already_calibrating") || msg.contains("prefill_busy") {
                StatusCode::CONFLICT
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            Err((code, msg))
        }
        Err(_) => Err((
            StatusCode::GATEWAY_TIMEOUT,
            format!("velo unary to {instance_id} timed out after {CALIBRATE_TIMEOUT:?}"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{Feature, PrefillRouterConfig};

    fn cfg() -> SelectorConfig {
        SelectorConfig {
            per_worker_concurrency: 4,
            block_size: 16,
        }
    }

    fn http_feature(url: &str, model: &str) -> Feature {
        Feature::PrefillRouter(PrefillRouterConfig {
            backend: PrefillBackendAdvertisement::Http(super::protocol::VllmHttpEndpoint {
                base_url: url.into(),
                model: model.into(),
            }),
        })
    }

    fn velo_feature(target: InstanceId) -> Feature {
        Feature::PrefillRouter(PrefillRouterConfig {
            backend: PrefillBackendAdvertisement::Velo {
                instance_id: target,
            },
        })
    }

    #[tokio::test]
    async fn on_register_http_adds_to_fleet() {
        let mgr = PrefillRouterManager::new(cfg());
        let id = InstanceId::new_v4();
        mgr.on_register(id, &http_feature("http://x:8000", "m"))
            .await
            .unwrap();
        assert_eq!(mgr.selector.worker_count(), 1);
        let targets = mgr.targets();
        assert_eq!(targets.targets.len(), 1);
        assert_eq!(targets.targets[0].instance_id, id);
        assert_eq!(targets.targets[0].backend, "http");
    }

    async fn attached_mgr_with_velo() -> (Arc<PrefillRouterManager>, Arc<velo::Velo>) {
        let mgr = PrefillRouterManager::new(cfg());
        let velo = velo::Velo::builder()
            .add_transport({
                let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
                Arc::new(
                    velo::transports::tcp::TcpTransportBuilder::new()
                        .from_listener(listener)
                        .unwrap()
                        .build()
                        .unwrap(),
                )
            })
            .build()
            .await
            .unwrap();
        let ctx = HubContext {
            velo: Some(Arc::clone(&velo)),
            registry: Arc::new(crate::registry::InMemoryRegistry::builder().build())
                as Arc<dyn crate::registry::PeerRegistry>,
            cancel: tokio_util::sync::CancellationToken::new(),
        };
        mgr.attach(ctx).await.unwrap();
        (mgr, velo)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn on_register_velo_adds_to_fleet() {
        let (mgr, _velo) = attached_mgr_with_velo().await;
        let id = InstanceId::new_v4();
        let target = InstanceId::new_v4();
        mgr.on_register(id, &velo_feature(target)).await.unwrap();
        let targets = mgr.targets();
        assert_eq!(targets.targets[0].backend, "velo");
        match &targets.targets[0].advertisement {
            PrefillBackendAdvertisement::Velo { instance_id } => assert_eq!(*instance_id, target),
            other => panic!("expected velo, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn velo_on_register_without_attached_velo_fails() {
        // Discovery-only hub: attach with no velo. Registering a velo
        // backend must hard-fail with a clear reason.
        let mgr = PrefillRouterManager::new(cfg());
        let ctx = HubContext {
            velo: None,
            registry: Arc::new(crate::registry::InMemoryRegistry::builder().build())
                as Arc<dyn crate::registry::PeerRegistry>,
            cancel: tokio_util::sync::CancellationToken::new(),
        };
        mgr.attach(ctx).await.unwrap();
        let err = mgr
            .on_register(InstanceId::new_v4(), &velo_feature(InstanceId::new_v4()))
            .await
            .unwrap_err();
        match err {
            FeatureError::InvalidConfig(msg) => {
                assert!(msg.contains("velo"), "expected velo error, got: {msg}");
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn on_unregister_drains_from_fleet() {
        let mgr = PrefillRouterManager::new(cfg());
        let id = InstanceId::new_v4();
        mgr.on_register(id, &http_feature("http://x:8000", "m"))
            .await
            .unwrap();
        mgr.on_unregister(id);
        assert_eq!(mgr.selector.worker_count(), 0);
        assert!(mgr.targets().targets.is_empty());
    }

    #[tokio::test]
    async fn re_register_replaces_advertisement() {
        let mgr = PrefillRouterManager::new(cfg());
        let id = InstanceId::new_v4();
        mgr.on_register(id, &http_feature("http://old:8000", "m"))
            .await
            .unwrap();
        mgr.on_register(id, &http_feature("http://new:8000", "m"))
            .await
            .unwrap();
        assert_eq!(mgr.selector.worker_count(), 1);
        let targets = mgr.targets();
        assert_eq!(targets.targets.len(), 1);
        match &targets.targets[0].advertisement {
            PrefillBackendAdvertisement::Http(ep) => assert_eq!(ep.base_url, "http://new:8000"),
            other => panic!("expected http, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn on_register_rejects_wrong_feature() {
        use crate::protocol::IndexerFeatureConfig;
        let mgr = PrefillRouterManager::new(cfg());
        let id = InstanceId::new_v4();
        let err = mgr
            .on_register(id, &Feature::Indexer(IndexerFeatureConfig::default()))
            .await
            .unwrap_err();
        match err {
            FeatureError::KeyMismatch { manager, .. } => {
                assert_eq!(manager, FeatureKey::PrefillRouter);
            }
            other => panic!("expected KeyMismatch, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn counters_snapshot_reflects_workers() {
        let mgr = PrefillRouterManager::new(cfg());
        let id = InstanceId::new_v4();
        mgr.on_register(id, &http_feature("http://x:8000", "m"))
            .await
            .unwrap();
        let snap = mgr.counters();
        assert_eq!(snap.workers.len(), 1);
        assert_eq!(snap.workers[0].inflight, 0);
        assert_eq!(snap.available_permits, cfg().per_worker_concurrency);
    }
}
