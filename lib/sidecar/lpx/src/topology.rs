// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Allocation snapshots: Agent ServerInfo -> dedicated DWM field -> local startup API.

mod agent;
mod consumer;
mod identity;
mod payload;
pub mod proto;
mod publisher;
mod socket;
mod validate;

#[cfg(test)]
mod tests;

use std::path::PathBuf;

use dynamo_runtime::DistributedRuntime;
use tokio_util::sync::CancellationToken;

use agent::AgentClient;
use consumer::Consumer;
use identity::{check_expected, check_identity, require_expected};
pub(crate) use socket::check_socket_path;
use validate::check_snapshot;

/// A sidecar's topology role, independent of whether it serves inference.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Role {
    /// Beside the Agent allocation's leader: register the allocation's
    /// snapshot from OpenEngine `GetServerInfo` on `agent_socket` in DWM.
    Publisher { agent_socket: PathBuf },
    /// In a conductor pod: serve the published DWM snapshot for the requested
    /// identity through `TopologyService` on `socket`.
    Consumer { socket: PathBuf },
}

/// Runs `role` in the Dynamo `namespace` until `shutdown`.
///
/// The publisher waits, with no deadline, for the Agent's first valid
/// snapshot before it registers, and fails on an invalid one. The consumer
/// fails only when it cannot bind its socket.
pub async fn run(
    role: &Role,
    drt: &DistributedRuntime,
    namespace: &str,
    shutdown: CancellationToken,
) -> anyhow::Result<()> {
    match role {
        Role::Publisher { agent_socket } => {
            let agent = AgentClient::new(agent_socket).map_err(anyhow::Error::msg)?;
            publisher::publish(drt.discovery(), namespace, agent, shutdown).await
        }
        Role::Consumer { socket } => {
            let consumer = Consumer::new(
                drt.discovery(),
                namespace.to_string(),
                shutdown.child_token(),
            );
            consumer::serve(consumer, socket, shutdown).await
        }
    }
}
