// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hub-side manager for the ConditionalDisagg feature.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};

use axum::{Json, Router, extract::State, routing::get};
use futures::future::BoxFuture;
use parking_lot::RwLock;
use velo::queue::backends::messenger::{MessengerQueueBackend, MessengerQueueConfig};
use velo_common::InstanceId;

use crate::features::{FeatureError, FeatureManager, HubContext};
use crate::protocol::{
    self, ConditionalDisaggInstancesResponse, ConditionalDisaggRole, Feature, FeatureKey,
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
        }
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

            let _ = self.queue_backend.set(backend);
            let _ = self.velo.set(velo);
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
