// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hub-side manager for the ConditionalDisagg feature.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use axum::{Json, Router, extract::State, routing::get};
use futures::future::BoxFuture;
use tokio::task::JoinHandle;
use velo::queue::NextOptions;
use velo::queue::backends::messenger::{MessengerQueueBackend, MessengerQueueConfig};
use velo_common::InstanceId;

use super::dispatcher::{DispatchOutcome, PrefillRequestDispatcher};
use super::registry::CdPeerRegistry;
use super::selector::{PrefillPeerSource, PrefillWorkerSelector};
use crate::features::{FeatureError, FeatureManager, HubContext};
use crate::protocol::{
    self, ConditionalDisaggInstancesResponse, ConditionalDisaggRole, Feature, FeatureKey,
    PrefillRequest,
};

/// Orchestrates the ConditionalDisagg feature: holds per-peer state
/// (delegated to [`CdPeerRegistry`]), the velo-backed prefill queue, and the
/// drainer task that selects + dispatches each dequeued request.
///
/// Per-peer state (role membership, engine URLs) lives in
/// [`Self::cd_registry`] — both this manager and the prefill drainer hold
/// an `Arc` to the same registry, keeping the manager
/// orchestration-focused and avoiding reference cycles between the manager
/// and any selector / drainer.
///
/// Mirrors dynamo's split between `Client` (discovery) and `PushRouter`
/// (policy) at `lib/runtime/src/pipeline/network/egress/push_router.rs`:
/// state and orchestration in separate types.
pub struct ConditionalDisaggManager {
    /// Per-peer CD state. Written by `on_register` / `on_unregister`,
    /// read by the prefill drainer via [`PrefillPeerSource`].
    cd_registry: Arc<CdPeerRegistry>,
    velo: OnceLock<Arc<velo::Velo>>,
    /// Hub-local queue backend owning the CD prefill queue. Lazily created
    /// during [`FeatureManager::attach`] when the hub has a Velo instance —
    /// `None` when the hub is discovery-only.
    queue_backend: OnceLock<Arc<MessengerQueueBackend>>,
    /// Optional bound on the prefill queue depth. `None` = unbounded.
    queue_capacity: Option<usize>,
    /// Dispatcher and selector are set together via
    /// [`Self::with_dispatch_pipeline`]. When `Some`,
    /// [`FeatureManager::attach`] spawns a drainer that selects a worker
    /// per request and ships it via the dispatcher.
    dispatcher: Option<Arc<dyn PrefillRequestDispatcher>>,
    /// Selection policy used by the prefill drainer. Must be `Some`
    /// whenever `dispatcher` is.
    selector: Option<Arc<dyn PrefillWorkerSelector>>,
    /// Worker task handle (set once spawned during `attach`).
    dispatcher_task: OnceLock<JoinHandle<()>>,
}

impl std::fmt::Debug for ConditionalDisaggManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let snap = self.cd_registry.snapshot();
        f.debug_struct("ConditionalDisaggManager")
            .field("prefill_count", &snap.prefill.len())
            .field("decode_count", &snap.decode.len())
            .field("velo_attached", &self.velo.get().is_some())
            .field("dispatch_pipeline", &self.dispatcher.is_some())
            .finish()
    }
}

impl ConditionalDisaggManager {
    /// Create an empty manager wired to the given registry, with no attached
    /// Velo and an unbounded prefill queue.
    pub fn new(cd_registry: Arc<CdPeerRegistry>) -> Self {
        Self::with_queue_capacity(cd_registry, None)
    }

    /// Create a manager with an explicit capacity bound on the prefill queue.
    pub fn with_queue_capacity(
        cd_registry: Arc<CdPeerRegistry>,
        capacity: Option<usize>,
    ) -> Self {
        Self {
            cd_registry,
            velo: OnceLock::new(),
            queue_backend: OnceLock::new(),
            queue_capacity: capacity,
            dispatcher: None,
            selector: None,
            dispatcher_task: OnceLock::new(),
        }
    }

    /// Builder: install the prefill dispatch pipeline. Both must be set
    /// together — the drainer needs a selector to pick a worker and a
    /// dispatcher to ship the request.
    pub fn with_dispatch_pipeline(
        mut self,
        dispatcher: Arc<dyn PrefillRequestDispatcher>,
        selector: Arc<dyn PrefillWorkerSelector>,
    ) -> Self {
        self.dispatcher = Some(dispatcher);
        self.selector = Some(selector);
        self
    }

    /// Read-only handle to the CD peer registry — useful for binaries that
    /// need to wire other components (e.g. a metrics exporter) against the
    /// same registry the manager uses.
    pub fn cd_registry(&self) -> &Arc<CdPeerRegistry> {
        &self.cd_registry
    }

    /// Current snapshot of the role split. Delegates to the registry.
    pub fn snapshot(&self) -> ConditionalDisaggInstancesResponse {
        self.cd_registry.snapshot()
    }

    /// Hub Velo handle stashed during [`FeatureManager::attach`], if any.
    pub fn velo_handle(&self) -> Option<&Arc<velo::Velo>> {
        self.velo.get()
    }

    /// Hub-local queue backend for the CD prefill queue, if the hub was
    /// configured with a Velo instance.
    pub fn queue_backend(&self) -> Option<&Arc<MessengerQueueBackend>> {
        self.queue_backend.get()
    }
}

impl FeatureManager for ConditionalDisaggManager {
    fn key(&self) -> FeatureKey {
        FeatureKey::ConditionalDisagg
    }

    fn attach<'a>(&'a self, ctx: HubContext) -> BoxFuture<'a, Result<(), FeatureError>> {
        Box::pin(async move {
            let Some(velo) = ctx.velo else {
                // Discovery-only hub: no Velo, so no queue surface. The CD
                // list endpoints still work — only the queue handlers are
                // skipped.
                return Ok(());
            };

            let backend = Arc::new(MessengerQueueBackend::new(
                velo.messenger().clone(),
                velo.instance_id(),
                MessengerQueueConfig {
                    capacity: self.queue_capacity,
                },
            ));

            // Eagerly instantiate a local receiver. The first `.receiver()`
            // / `.sender()` call is what registers the `velo.queue.rpc`
            // handler on the hub's Velo, so without this the handler is
            // absent and remote clients get a "handler not found" error on
            // their first enqueue. Dropping the receiver is fine — the
            // underlying queue service stays alive as long as the backend
            // is held.
            velo::queue::receiver::<Vec<u8>>(backend.as_ref(), protocol::CD_PREFILL_QUEUE)
                .await
                .map_err(|e| FeatureError::Other(anyhow::anyhow!("CD queue init: {e}")))?;

            let _ = self.queue_backend.set(Arc::clone(&backend));
            let _ = self.velo.set(velo);

            // Spawn the dispatcher worker if the dispatch pipeline is
            // configured. The drainer captures an `Arc<dyn PrefillPeerSource>`
            // cloned from the registry, so it can fetch a fresh peer list
            // each iteration without holding any reference back to the
            // manager.
            if let (Some(dispatcher), Some(selector)) =
                (self.dispatcher.clone(), self.selector.clone())
            {
                let peer_source: Arc<dyn PrefillPeerSource> = self.cd_registry.clone();
                let task = tokio::spawn(prefill_dispatcher_loop(
                    Arc::clone(&backend),
                    dispatcher,
                    selector,
                    peer_source,
                ));
                let _ = self.dispatcher_task.set(task);
                tracing::info!("CD prefill dispatcher worker started");
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
            let Feature::ConditionalDisagg(cfg) = feature;
            let cfg = cfg.as_ref().ok_or_else(|| {
                FeatureError::InvalidConfig(
                    "ConditionalDisagg requires a config with a role".to_string(),
                )
            })?;

            // Manager-side invariant: a Prefill peer must advertise an
            // engine_url at registration. The registry would otherwise just
            // filter URL-less prefills out of `prefill_peers()` silently;
            // rejecting at register time is louder and matches the prior
            // behavior.
            if cfg.role == ConditionalDisaggRole::Prefill && cfg.engine_url.is_none() {
                return Err(FeatureError::InvalidConfig(
                    "ConditionalDisagg Prefill requires an engine_url".to_string(),
                ));
            }

            self.cd_registry
                .insert(instance_id, cfg.role, cfg.engine_url.clone())
                .map_err(|e| FeatureError::InvalidConfig(e.to_string()))?;
            Ok(())
        })
    }

    fn on_unregister(&self, instance_id: InstanceId) {
        self.cd_registry.remove(instance_id);
    }

    fn control_router(self: Arc<Self>) -> Router {
        routes(self)
    }

    fn public_router(self: Arc<Self>) -> Router {
        routes(self)
    }
}

fn routes(manager: Arc<ConditionalDisaggManager>) -> Router {
    Router::new()
        .route(protocol::paths::CD_INSTANCES, get(list_instances))
        .with_state(manager)
}

async fn list_instances(
    State(mgr): State<Arc<ConditionalDisaggManager>>,
) -> Json<ConditionalDisaggInstancesResponse> {
    Json(mgr.snapshot())
}

/// Long-running task that drains the CD prefill queue and hands each
/// dequeued [`PrefillRequest`] to the configured dispatcher.
///
/// The loop terminates if the queue receiver fails to be (re)created —
/// that signals the underlying messenger backend has shut down. Per-
/// iteration `next_with_options` errors and dispatcher errors are
/// logged and skipped; we never want one bad request to take down the
/// pump.
async fn prefill_dispatcher_loop(
    backend: Arc<MessengerQueueBackend>,
    dispatcher: Arc<dyn PrefillRequestDispatcher>,
    selector: Arc<dyn PrefillWorkerSelector>,
    peer_source: Arc<dyn PrefillPeerSource>,
) {
    // Long-poll window. The receiver returns as soon as it has a full
    // batch OR the timeout fires — for the dispatcher's purposes we
    // want each request handed off ASAP, so use batch_size=1 (return on
    // first item) with a long idle timeout so wakeups are cheap when
    // nothing's flowing.
    const POLL_TIMEOUT: Duration = Duration::from_secs(30);
    const BATCH_SIZE: usize = 1;

    loop {
        // Re-create the receiver each iteration. The backend caches the
        // underlying handler; this is cheap.
        let receiver = match velo::queue::receiver::<Vec<u8>>(
            backend.as_ref(),
            protocol::CD_PREFILL_QUEUE,
        )
        .await
        {
            Ok(r) => r,
            Err(err) => {
                tracing::error!(error = %err, "CD dispatcher: receiver build failed; shutting down loop");
                return;
            }
        };

        let batch = match receiver
            .next_with_options(
                NextOptions::new()
                    .batch_size(BATCH_SIZE)
                    .timeout(POLL_TIMEOUT),
            )
            .await
        {
            Ok(b) => b,
            Err(err) => {
                tracing::warn!(error = %err, "CD dispatcher: dequeue failed; retrying");
                continue;
            }
        };

        if batch.is_empty() {
            // Idle window — long-poll timed out. Loop back.
            continue;
        }

        for bytes in batch {
            let req: PrefillRequest = match serde_json::from_slice(&bytes) {
                Ok(r) => r,
                Err(err) => {
                    tracing::error!(
                        error = %err,
                        bytes_len = bytes.len(),
                        "CD dispatcher: undecodable PrefillRequest; dropping"
                    );
                    continue;
                }
            };
            let request_id = req.request_id.clone();
            tracing::info!(
                request_id = request_id,
                session_id = %req.session_id,
                initiator = %req.initiator_instance_id,
                "CD dispatcher: dispatching PrefillRequest"
            );
            // Snapshot the live prefill peers and ask the selector to pick
            // one. Selector errors (e.g. no peers registered) skip this
            // request — never tear down the loop.
            let peers = peer_source.prefill_peers();
            let selected = match selector.select(&req, &peers).await {
                Ok(s) => s,
                Err(err) => {
                    tracing::warn!(request_id, error = %err, "CD dispatcher: select failed");
                    continue;
                }
            };
            let engine_url = selected.engine_url.clone();
            match dispatcher.dispatch(req, engine_url).await {
                Ok(DispatchOutcome::Accepted) => {
                    tracing::info!(request_id, "CD dispatcher: accepted");
                }
                Ok(DispatchOutcome::Rejected { reason }) => {
                    tracing::warn!(request_id, reason, "CD dispatcher: rejected");
                }
                Err(err) => {
                    tracing::error!(request_id, error = %err, "CD dispatcher: error");
                }
            }
            // `selected` drops here at end of iteration → its `permit` drops
            // → the worker's in-flight count decrements (for load-aware
            // selectors; no-op for round-robin).
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::ConditionalDisaggConfig;

    /// Build a CD `Feature` for the given role. Prefill peers get a stub
    /// `engine_url` (manager rejects URL-less prefills); decode gets `None`.
    fn cd(role: ConditionalDisaggRole) -> Feature {
        let engine_url = match role {
            ConditionalDisaggRole::Prefill => Some("http://test:8000".to_string()),
            ConditionalDisaggRole::Decode => None,
        };
        Feature::ConditionalDisagg(Some(ConditionalDisaggConfig { role, engine_url }))
    }

    #[tokio::test]
    async fn register_prefill_appears_in_snapshot() {
        let mgr = ConditionalDisaggManager::new(Arc::new(CdPeerRegistry::new()));
        let id = InstanceId::new_v4();
        mgr.on_register(id, &cd(ConditionalDisaggRole::Prefill))
            .await
            .unwrap();
        let snap = mgr.snapshot();
        assert_eq!(snap.prefill, vec![id]);
        assert!(snap.decode.is_empty());
    }

    #[tokio::test]
    async fn register_decode_appears_in_snapshot() {
        let mgr = ConditionalDisaggManager::new(Arc::new(CdPeerRegistry::new()));
        let id = InstanceId::new_v4();
        mgr.on_register(id, &cd(ConditionalDisaggRole::Decode))
            .await
            .unwrap();
        let snap = mgr.snapshot();
        assert_eq!(snap.decode, vec![id]);
        assert!(snap.prefill.is_empty());
    }

    #[tokio::test]
    async fn register_without_config_is_invalid() {
        let mgr = ConditionalDisaggManager::new(Arc::new(CdPeerRegistry::new()));
        let err = mgr
            .on_register(InstanceId::new_v4(), &Feature::ConditionalDisagg(None))
            .await
            .unwrap_err();
        assert!(matches!(err, FeatureError::InvalidConfig(_)));
    }

    #[tokio::test]
    async fn register_prefill_without_engine_url_is_invalid() {
        let mgr = ConditionalDisaggManager::new(Arc::new(CdPeerRegistry::new()));
        let f = Feature::ConditionalDisagg(Some(ConditionalDisaggConfig {
            role: ConditionalDisaggRole::Prefill,
            engine_url: None,
        }));
        let err = mgr
            .on_register(InstanceId::new_v4(), &f)
            .await
            .unwrap_err();
        assert!(matches!(err, FeatureError::InvalidConfig(_)));
    }

    #[tokio::test]
    async fn reregister_same_role_is_idempotent() {
        let mgr = ConditionalDisaggManager::new(Arc::new(CdPeerRegistry::new()));
        let id = InstanceId::new_v4();
        mgr.on_register(id, &cd(ConditionalDisaggRole::Prefill))
            .await
            .unwrap();
        mgr.on_register(id, &cd(ConditionalDisaggRole::Prefill))
            .await
            .unwrap();
        assert_eq!(mgr.snapshot().prefill.len(), 1);
    }

    #[tokio::test]
    async fn reregister_different_role_rejected() {
        let mgr = ConditionalDisaggManager::new(Arc::new(CdPeerRegistry::new()));
        let id = InstanceId::new_v4();
        mgr.on_register(id, &cd(ConditionalDisaggRole::Prefill))
            .await
            .unwrap();
        let err = mgr
            .on_register(id, &cd(ConditionalDisaggRole::Decode))
            .await
            .unwrap_err();
        assert!(matches!(err, FeatureError::InvalidConfig(_)));
    }

    #[tokio::test]
    async fn unregister_removes_from_snapshot() {
        let mgr = ConditionalDisaggManager::new(Arc::new(CdPeerRegistry::new()));
        let id = InstanceId::new_v4();
        mgr.on_register(id, &cd(ConditionalDisaggRole::Prefill))
            .await
            .unwrap();
        mgr.on_unregister(id);
        assert!(mgr.snapshot().prefill.is_empty());
    }

    #[test]
    fn unregister_unknown_is_noop() {
        let mgr = ConditionalDisaggManager::new(Arc::new(CdPeerRegistry::new()));
        mgr.on_unregister(InstanceId::new_v4());
    }
}
