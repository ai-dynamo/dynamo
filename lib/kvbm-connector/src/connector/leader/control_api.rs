// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Connector-leader control API — transport-agnostic.
//!
//! Wire types and the precondition logic live in
//! [`kvbm_control_protocol`] so that `kvbm-hub` can map velo replies to
//! HTTP status codes without taking a dep on `kvbm-connector` (avoiding
//! the cycle).
//!
//! This module re-exports those types for backward-compat at the
//! original module path and adds the [`ConnectorControlApi`] trait
//! impl for `Arc<ConnectorLeader>`.

use std::collections::HashSet;
use std::sync::Arc;

// Re-exports for backward compatibility.
pub use kvbm_control_protocol::{
    ControlError, ControlReply, REGISTER_LEADER_HANDLER, RESET_HANDLER, RegisterLeaderRequest,
    RegisterLeaderResponse, RegisterLeaderStatus, ResetRequest, ResetResponse, Tier, TierError,
    plan_reset,
};

use super::ConnectorLeader;

/// Connector-leader control surface, exposed transport-agnostically.
///
/// Implementations live alongside `ConnectorLeader`. The trait is
/// `async` to accommodate `register_leader`'s velo-discovery call;
/// `reset` is also `async fn` for uniformity even though its body is
/// synchronous today.
pub trait ConnectorControlApi: Send + Sync {
    fn reset(
        &self,
        req: ResetRequest,
    ) -> impl std::future::Future<Output = Result<ResetResponse, ControlError>> + Send;

    fn register_leader(
        &self,
        req: RegisterLeaderRequest,
    ) -> impl std::future::Future<Output = Result<RegisterLeaderResponse, ControlError>> + Send;
}

impl ConnectorControlApi for Arc<ConnectorLeader> {
    async fn reset(&self, req: ResetRequest) -> Result<ResetResponse, ControlError> {
        let il = self.instance_leader().ok_or(ControlError::NotInitialized)?;

        let mut available = HashSet::new();
        // G2 is always present once InstanceLeader is up.
        available.insert(Tier::G2);
        if il.g3_manager().is_some() {
            available.insert(Tier::G3);
        }

        let (to_reset, skipped) = plan_reset(&req, &available)?;

        let mut reset = Vec::with_capacity(to_reset.len());
        let mut failed = Vec::new();
        for tier in to_reset {
            let result = match tier {
                Tier::G2 => il
                    .g2_manager()
                    .reset_inactive_pool()
                    .map_err(|e| e.to_string()),
                Tier::G3 => il
                    .g3_manager()
                    .expect("plan_reset already verified G3 is configured")
                    .reset_inactive_pool()
                    .map_err(|e| e.to_string()),
            };
            match result {
                Ok(()) => {
                    tracing::info!(?tier, "tier reset succeeded");
                    reset.push(tier);
                }
                Err(message) => {
                    tracing::warn!(?tier, %message, "tier reset failed");
                    failed.push(TierError { tier, message });
                }
            }
        }

        Ok(ResetResponse {
            reset,
            failed,
            skipped_unconfigured: skipped,
        })
    }

    async fn register_leader(
        &self,
        req: RegisterLeaderRequest,
    ) -> Result<RegisterLeaderResponse, ControlError> {
        let il = self.instance_leader().ok_or(ControlError::NotInitialized)?;
        let instance_id = req.instance_id;

        if il.remote_leaders().contains(&instance_id) {
            return Ok(RegisterLeaderResponse {
                status: RegisterLeaderStatus::AlreadyRegistered,
                remote_leaders: il.remote_leaders(),
            });
        }

        self.runtime
            .messenger()
            .discover_and_register_peer(instance_id)
            .await
            .map_err(|e| ControlError::PeerNotFound {
                instance_id,
                reason: format!("{e:#}"),
            })?;

        il.add_remote_leader(instance_id);

        Ok(RegisterLeaderResponse {
            status: RegisterLeaderStatus::Registered,
            remote_leaders: il.remote_leaders(),
        })
    }
}
