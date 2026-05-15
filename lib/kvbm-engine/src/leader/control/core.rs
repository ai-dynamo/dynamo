// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The always-on `core` control module: `register_leader`.
//!
//! Migrated from the connector's `ConnectorControlApi`. The logic is
//! pass-thru to [`InstanceLeader`]; the `NotInitialized` precondition is gone
//! because a [`CoreModule`] only exists once its `InstanceLeader` does.

use std::sync::Arc;

use anyhow::Result;
use velo::{Handler, Messenger};

use kvbm_protocols::control::{
    ControlError, ControlReply, DESCRIBE_INSTANCE_HANDLER, DescribeInstanceRequest,
    InstanceDescription, ModuleId, REGISTER_LEADER_HANDLER, RegisterLeaderRequest,
    RegisterLeaderResponse, RegisterLeaderStatus,
};

use super::ControlModule;
use crate::leader::InstanceLeader;

/// The `core` control module — always enabled.
pub struct CoreModule {
    leader: Arc<InstanceLeader>,
}

impl CoreModule {
    pub fn new(leader: Arc<InstanceLeader>) -> Self {
        Self { leader }
    }
}

impl ControlModule for CoreModule {
    fn id(&self) -> ModuleId {
        ModuleId::Core
    }

    fn register(&self, messenger: &Arc<Messenger>) -> Result<()> {
        register_register_leader(messenger, self.leader.clone())?;
        register_describe_instance(messenger, self.leader.clone())?;
        Ok(())
    }
}

fn register_register_leader(messenger: &Arc<Messenger>, leader: Arc<InstanceLeader>) -> Result<()> {
    let handler = Handler::typed_unary_async(REGISTER_LEADER_HANDLER, move |ctx| {
        let leader = Arc::clone(&leader);
        async move {
            let req: RegisterLeaderRequest = ctx.input;
            let reply: ControlReply<RegisterLeaderResponse> =
                register_leader(&leader, req).await.into();
            Ok::<ControlReply<RegisterLeaderResponse>, anyhow::Error>(reply)
        }
    })
    .build();
    messenger
        .register_handler(handler)
        .map_err(|e| anyhow::anyhow!("velo register_handler({REGISTER_LEADER_HANDLER}): {e}"))?;
    Ok(())
}

/// Register the `describe_instance` velo handler.
///
/// This is the fallback-pull surface — the steady-state flow has the leader
/// pushing [`InstanceDescription`] to the hub via HTTP. The handler stays
/// available so the hub can recover after a cold restart, and so operators
/// can force-refresh via `POST /control/core/describe_instance`.
fn register_describe_instance(
    messenger: &Arc<Messenger>,
    leader: Arc<InstanceLeader>,
) -> Result<()> {
    let handler = Handler::typed_unary_async(DESCRIBE_INSTANCE_HANDLER, move |ctx| {
        let leader = Arc::clone(&leader);
        async move {
            let _req: DescribeInstanceRequest = ctx.input;
            let reply: ControlReply<InstanceDescription> = leader.describe().await.into();
            Ok::<ControlReply<InstanceDescription>, anyhow::Error>(reply)
        }
    })
    .build();
    messenger
        .register_handler(handler)
        .map_err(|e| anyhow::anyhow!("velo register_handler({DESCRIBE_INSTANCE_HANDLER}): {e}"))?;
    Ok(())
}

/// Discover and register a remote leader by instance id.
async fn register_leader(
    leader: &InstanceLeader,
    req: RegisterLeaderRequest,
) -> Result<RegisterLeaderResponse, ControlError> {
    let instance_id = req.instance_id;

    if leader.remote_leaders().contains(&instance_id) {
        return Ok(RegisterLeaderResponse {
            status: RegisterLeaderStatus::AlreadyRegistered,
            remote_leaders: leader.remote_leaders(),
        });
    }

    leader
        .messenger()
        .discover_and_register_peer(instance_id)
        .await
        .map_err(|e| ControlError::PeerNotFound {
            instance_id,
            reason: format!("{e:#}"),
        })?;

    leader.add_remote_leader(instance_id);

    Ok(RegisterLeaderResponse {
        status: RegisterLeaderStatus::Registered,
        remote_leaders: leader.remote_leaders(),
    })
}
