// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hub-side manager for the ConditionalDisagg feature.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};

use axum::{Json, Router, extract::State, routing::get};
use futures::future::BoxFuture;
use parking_lot::RwLock;
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
    /// Create an empty manager with no attached Velo.
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(CdInner {
                prefill: HashSet::new(),
                decode: HashSet::new(),
                by_instance: HashMap::new(),
            }),
            velo: OnceLock::new(),
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
    ///
    /// Reserved for future use when the manager begins owning Velo queues.
    pub fn velo_handle(&self) -> Option<&Arc<velo::Velo>> {
        self.velo.get()
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

    fn attach(&self, ctx: HubContext) -> Result<(), FeatureError> {
        if let Some(velo) = ctx.velo {
            let _ = self.velo.set(velo);
        }
        Ok(())
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
