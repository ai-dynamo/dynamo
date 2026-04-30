// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Velo handlers for the connector-leader control plane.
//!
//! Mirror of the axum routes in `control.rs` — same business logic via
//! `super::control_api::ConnectorControlApi`, different transport. The
//! hub's HTTP→velo proxy invokes these handlers; operators do not call
//! them directly.

use std::sync::Arc;

use anyhow::{Context, Result};
use kvbm_control_protocol::{
    ControlReply, REGISTER_LEADER_HANDLER, RESET_HANDLER, RegisterLeaderRequest,
    RegisterLeaderResponse, ResetRequest, ResetResponse,
};
use velo::{Handler, Messenger};

use super::ConnectorLeader;
use super::control_api::ConnectorControlApi;

// Re-export the handler-name constants so existing call sites keep working.
pub use kvbm_control_protocol::{
    REGISTER_LEADER_HANDLER as REGISTER_LEADER_HANDLER_NAME, RESET_HANDLER as RESET_HANDLER_NAME,
};

/// Register both connector-leader control handlers on the given messenger.
///
/// Called from `initialize_async` after workers are up. Failure to
/// register is fatal — the hub-proxy path depends on these handlers
/// being present.
pub fn register_handlers(messenger: &Arc<Messenger>, leader: Arc<ConnectorLeader>) -> Result<()> {
    register_reset(messenger, leader.clone())
        .context("registering kvbm.connector.leader.reset velo handler")?;
    register_register_leader(messenger, leader)
        .context("registering kvbm.connector.leader.register_leader velo handler")?;
    Ok(())
}

fn register_reset(messenger: &Arc<Messenger>, leader: Arc<ConnectorLeader>) -> Result<()> {
    let leader = Arc::clone(&leader);
    let handler = Handler::typed_unary_async(RESET_HANDLER, move |ctx| {
        let leader = Arc::clone(&leader);
        async move {
            let req: ResetRequest = ctx.input;
            let reply: ControlReply<ResetResponse> = leader.reset(req).await.into();
            // The handler must return `Result<Reply, _>`; `ControlReply`
            // already carries the typed error variant — wrap as Ok to
            // signal "transport delivered the message," not "the call
            // succeeded."
            Ok::<ControlReply<ResetResponse>, anyhow::Error>(reply)
        }
    })
    .build();
    messenger
        .register_handler(handler)
        .map_err(|e| anyhow::anyhow!("velo register_handler({RESET_HANDLER}): {e}"))?;
    tracing::debug!(handler = RESET_HANDLER, "registered velo handler");
    Ok(())
}

fn register_register_leader(
    messenger: &Arc<Messenger>,
    leader: Arc<ConnectorLeader>,
) -> Result<()> {
    let leader = Arc::clone(&leader);
    let handler = Handler::typed_unary_async(REGISTER_LEADER_HANDLER, move |ctx| {
        let leader = Arc::clone(&leader);
        async move {
            let req: RegisterLeaderRequest = ctx.input;
            let reply: ControlReply<RegisterLeaderResponse> =
                leader.register_leader(req).await.into();
            Ok::<ControlReply<RegisterLeaderResponse>, anyhow::Error>(reply)
        }
    })
    .build();
    messenger
        .register_handler(handler)
        .map_err(|e| anyhow::anyhow!("velo register_handler({REGISTER_LEADER_HANDLER}): {e}"))?;
    tracing::debug!(handler = REGISTER_LEADER_HANDLER, "registered velo handler");
    Ok(())
}

// Wire-shape tests live in `kvbm-control-protocol`; this module's tests
// would only re-prove serde derives — covered upstream.
