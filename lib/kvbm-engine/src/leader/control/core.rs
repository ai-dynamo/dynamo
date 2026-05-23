// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The always-on `core` control module: `describe_instance`.
//!
//! Migrated from the connector's `ConnectorControlApi`. The logic is
//! pass-thru to [`InstanceLeader`]; the `NotInitialized` precondition is gone
//! because a [`CoreModule`] only exists once its `InstanceLeader` does.

use std::sync::Arc;

use anyhow::Result;
use velo::{Handler, Messenger};

use kvbm_protocols::control::{
    ControlReply, DESCRIBE_INSTANCE_HANDLER, DescribeInstanceRequest, InstanceDescription, ModuleId,
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
        register_describe_instance(messenger, self.leader.clone())?;
        Ok(())
    }
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
