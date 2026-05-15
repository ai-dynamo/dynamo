// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The opt-in `dev` control module: `reset`.
//!
//! Operator/debug tooling. Off by default, enabled via
//! `control.dev = true`. Safe to run in production — no warning is logged
//! when enabled. Migrated from the connector's `ConnectorControlApi`.

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::Result;
use velo::{Handler, Messenger};

use kvbm_protocols::control::{
    ControlError, ControlReply, ModuleId, RESET_HANDLER, ResetRequest, ResetResponse, Tier,
    TierError, plan_reset,
};

use super::ControlModule;
use crate::leader::InstanceLeader;

/// The `dev` control module — opt-in.
pub struct DevModule {
    leader: Arc<InstanceLeader>,
}

impl DevModule {
    pub fn new(leader: Arc<InstanceLeader>) -> Self {
        Self { leader }
    }
}

impl ControlModule for DevModule {
    fn id(&self) -> ModuleId {
        ModuleId::Dev
    }

    fn register(&self, messenger: &Arc<Messenger>) -> Result<()> {
        let leader = self.leader.clone();
        let handler = Handler::typed_unary_async(RESET_HANDLER, move |ctx| {
            let leader = Arc::clone(&leader);
            async move {
                let req: ResetRequest = ctx.input;
                let reply: ControlReply<ResetResponse> = reset(&leader, req).into();
                Ok::<ControlReply<ResetResponse>, anyhow::Error>(reply)
            }
        })
        .build();
        messenger
            .register_handler(handler)
            .map_err(|e| anyhow::anyhow!("velo register_handler({RESET_HANDLER}): {e}"))?;
        Ok(())
    }
}

/// Reset the inactive pools of the requested (or all configured) tiers.
///
/// Synchronous — `BlockManager::reset_inactive_pool` does not await.
fn reset(leader: &InstanceLeader, req: ResetRequest) -> Result<ResetResponse, ControlError> {
    let mut available = HashSet::new();
    // G2 is always present once an InstanceLeader is up.
    available.insert(Tier::G2);
    if leader.g3_manager().is_some() {
        available.insert(Tier::G3);
    }

    let (to_reset, skipped) = plan_reset(&req, &available)?;

    let mut reset = Vec::with_capacity(to_reset.len());
    let mut failed = Vec::new();
    for tier in to_reset {
        let result = match tier {
            Tier::G2 => leader
                .g2_manager()
                .reset_inactive_pool()
                .map_err(|e| e.to_string()),
            Tier::G3 => leader
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
