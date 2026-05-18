// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hub-side manager for the P2P feature.

use std::collections::HashSet;

use axum::Router;
use futures::future::BoxFuture;
use parking_lot::RwLock;
use velo_ext::InstanceId;

use crate::features::{FeatureError, FeatureManager, HubContext};
use crate::protocol::{Feature, FeatureKey, LayoutCompatPayload};
use kvbm_protocols::control::layout_compat::check_layout_compat;

/// Tracks P2P registrations and enforces the layout-compatibility baseline.
///
/// The first P2P registration whose payload passes `validate_self` becomes
/// the baseline. Every subsequent registration is checked against it via
/// `check_layout_compat`. When the last P2P-registered instance unregisters,
/// the baseline clears so a new group can adopt a different layout without
/// bouncing the hub.
pub struct P2pManager {
    inner: RwLock<P2pInner>,
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
    /// - No baseline is present (instance registered without `Feature::P2P`
    ///   or the hub restarted since last register) — this is a protocol
    ///   violation (describe-before-register or missing re-register after
    ///   hub restart).
    /// - The candidate diverges from the baseline (mode, canonical, or
    ///   per-worker fields differ under the operative mode's predicate).
    pub fn check_describe_layout(
        &self,
        instance_id: velo_ext::InstanceId,
        candidate: &LayoutCompatPayload,
    ) -> Result<(), FeatureError> {
        let inner = self.inner.read();
        let baseline = inner.layout_baseline.as_ref().ok_or_else(|| {
            FeatureError::InvalidConfig(format!(
                "describe for instance {instance_id} carries layout_compat but \
                 this instance has no P2P baseline (describe before register, \
                 or hub restarted without re-register?)"
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

    fn attach<'a>(&'a self, _ctx: HubContext) -> BoxFuture<'a, Result<(), FeatureError>> {
        // P2P is a gate-only feature; nothing to attach.
        Box::pin(async { Ok(()) })
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

    fn control_router(self: std::sync::Arc<Self>) -> Router {
        Router::new()
    }

    fn public_router(self: std::sync::Arc<Self>) -> Router {
        Router::new()
    }
}
