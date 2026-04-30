// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hub-side manager for the ConditionalDisagg feature.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use axum::{Json, Router, extract::State, routing::get};
use futures::future::BoxFuture;
use parking_lot::RwLock;
use tokio::task::JoinHandle;
use velo::queue::NextOptions;
use velo::queue::backends::messenger::{MessengerQueueBackend, MessengerQueueConfig};
use velo_common::InstanceId;

use super::dispatcher::{DispatchOutcome, PrefillRequestDispatcher};
use crate::features::{FeatureError, FeatureManager, HubContext};
use crate::protocol::{
    self, ConditionalDisaggInstancesResponse, ConditionalDisaggRole, Feature, FeatureKey,
    PrefillRequest,
};

/// Tracks which instances participate in ConditionalDisagg and under what role.
///
/// State is kept behind a single `RwLock` — lookups are O(1) via the
/// `by_instance` map, and role-filtered listings iterate the matching set.
pub struct ConditionalDisaggManager {
    inner: RwLock<CdInner>,
    velo: OnceLock<Arc<velo::Velo>>,
    /// Hub-local queue backend owning the CD prefill queue. Lazily created
    /// during [`FeatureManager::attach`] when the hub has a Velo instance —
    /// `None` when the hub is discovery-only.
    queue_backend: OnceLock<Arc<MessengerQueueBackend>>,
    /// Optional bound on the prefill queue depth. `None` = unbounded.
    queue_capacity: Option<usize>,
    /// Optional dispatcher for the prefill queue. When set,
    /// [`FeatureManager::attach`] spawns a background worker that
    /// drains the queue and hands each request to this dispatcher.
    dispatcher: Option<Arc<dyn PrefillRequestDispatcher>>,
    /// Worker task handle (set once spawned during `attach`).
    dispatcher_task: OnceLock<JoinHandle<()>>,
}

struct CdInner {
    prefill: HashSet<InstanceId>,
    decode: HashSet<InstanceId>,
    by_instance: HashMap<InstanceId, ConditionalDisaggRole>,
}

impl std::fmt::Debug for ConditionalDisaggManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.read();
        f.debug_struct("ConditionalDisaggManager")
            .field("prefill_count", &inner.prefill.len())
            .field("decode_count", &inner.decode.len())
            .field("velo_attached", &self.velo.get().is_some())
            .finish()
    }
}

impl Default for ConditionalDisaggManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ConditionalDisaggManager {
    /// Create an empty manager with no attached Velo and an unbounded
    /// prefill queue.
    pub fn new() -> Self {
        Self::with_queue_capacity(None)
    }

    /// Create a manager with an explicit capacity bound on the prefill queue.
    pub fn with_queue_capacity(capacity: Option<usize>) -> Self {
        Self {
            inner: RwLock::new(CdInner {
                prefill: HashSet::new(),
                decode: HashSet::new(),
                by_instance: HashMap::new(),
            }),
            velo: OnceLock::new(),
            queue_backend: OnceLock::new(),
            queue_capacity: capacity,
            dispatcher: None,
            dispatcher_task: OnceLock::new(),
        }
    }

    /// Builder: install a [`PrefillRequestDispatcher`]. When set, the
    /// hub spawns a background worker (in [`FeatureManager::attach`])
    /// that drains the prefill queue and hands each item to the
    /// dispatcher.
    pub fn with_dispatcher(mut self, dispatcher: Arc<dyn PrefillRequestDispatcher>) -> Self {
        self.dispatcher = Some(dispatcher);
        self
    }

    /// Current snapshot of the role split, sorted deterministically.
    pub fn snapshot(&self) -> ConditionalDisaggInstancesResponse {
        let inner = self.inner.read();
        ConditionalDisaggInstancesResponse {
            prefill: inner.prefill.iter().copied().collect(),
            decode: inner.decode.iter().copied().collect(),
        }
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

fn insert_role(inner: &mut CdInner, id: InstanceId, role: ConditionalDisaggRole) {
    match role {
        ConditionalDisaggRole::Prefill => {
            inner.prefill.insert(id);
        }
        ConditionalDisaggRole::Decode => {
            inner.decode.insert(id);
        }
    }
}

fn remove_role(inner: &mut CdInner, id: InstanceId, role: ConditionalDisaggRole) {
    match role {
        ConditionalDisaggRole::Prefill => {
            inner.prefill.remove(&id);
        }
        ConditionalDisaggRole::Decode => {
            inner.decode.remove(&id);
        }
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

            // Spawn the dispatcher worker if one is configured.
            if let Some(dispatcher) = self.dispatcher.clone() {
                let task =
                    tokio::spawn(prefill_dispatcher_loop(Arc::clone(&backend), dispatcher));
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
            let role = cfg.role;

            let mut inner = self.inner.write();
            if let Some(prior) = inner.by_instance.get(&instance_id).copied() {
                if prior != role {
                    return Err(FeatureError::InvalidConfig(format!(
                        "instance {instance_id} already registered as {:?}, cannot switch to {:?}",
                        prior, role
                    )));
                }
                // Same role re-registration is idempotent.
                return Ok(());
            }
            inner.by_instance.insert(instance_id, role);
            insert_role(&mut inner, instance_id, role);
            Ok(())
        })
    }

    fn on_unregister(&self, instance_id: InstanceId) {
        let mut inner = self.inner.write();
        if let Some(role) = inner.by_instance.remove(&instance_id) {
            remove_role(&mut inner, instance_id, role);
        }
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
        let receiver =
            match velo::queue::receiver::<Vec<u8>>(backend.as_ref(), protocol::CD_PREFILL_QUEUE)
                .await
            {
                Ok(r) => r,
                Err(err) => {
                    tracing::error!(error = %err, "CD dispatcher: receiver build failed; shutting down loop");
                    return;
                }
            };

        let batch = match receiver
            .next_with_options(NextOptions::new().batch_size(BATCH_SIZE).timeout(POLL_TIMEOUT))
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
            match dispatcher.dispatch(req).await {
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::ConditionalDisaggConfig;

    fn cd(role: ConditionalDisaggRole) -> Feature {
        Feature::ConditionalDisagg(Some(ConditionalDisaggConfig { role }))
    }

    #[tokio::test]
    async fn register_prefill_appears_in_snapshot() {
        let mgr = ConditionalDisaggManager::new();
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
        let mgr = ConditionalDisaggManager::new();
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
        let mgr = ConditionalDisaggManager::new();
        let err = mgr
            .on_register(InstanceId::new_v4(), &Feature::ConditionalDisagg(None))
            .await
            .unwrap_err();
        assert!(matches!(err, FeatureError::InvalidConfig(_)));
    }

    #[tokio::test]
    async fn reregister_same_role_is_idempotent() {
        let mgr = ConditionalDisaggManager::new();
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
        let mgr = ConditionalDisaggManager::new();
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
        let mgr = ConditionalDisaggManager::new();
        let id = InstanceId::new_v4();
        mgr.on_register(id, &cd(ConditionalDisaggRole::Prefill))
            .await
            .unwrap();
        mgr.on_unregister(id);
        assert!(mgr.snapshot().prefill.is_empty());
    }

    #[test]
    fn unregister_unknown_is_noop() {
        let mgr = ConditionalDisaggManager::new();
        mgr.on_unregister(InstanceId::new_v4());
    }
}
