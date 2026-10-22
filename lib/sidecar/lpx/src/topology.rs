// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The LPX allocation's partition topology (`lpx.topology.v1`).
//!
//! The leader of an LPX Agent allocation is the only authority for where the
//! allocation's partitions execute. Its Agent serves `TopologyService` on a
//! pod-local Unix socket. The sidecar beside it, the publisher, registers a
//! dedicated non-LLM Dynamo endpoint scoped by the allocation's identity and
//! answers each request on it with a live answer from the Agent. The sidecar
//! in each conductor pod, the consumer, serves `TopologyService` on its own
//! pod-local Unix socket and answers each request with a live answer from the
//! publisher of the identity the conductor expects: of every publisher
//! registered for it, exactly one must answer OK. No hop caches a snapshot.

mod agent;
mod consumer;
mod envelope;
mod identity;
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
use envelope::{RelayRequest, RelayResponse};
use identity::{
    allocation_prefix, check_expected, check_identity, endpoint_name, require_expected,
};
pub(crate) use socket::check_socket_path;
use validate::check_snapshot;

/// The Dynamo component of every topology endpoint. Never a model component,
/// so the endpoint is never mistaken for an inference worker.
pub(crate) const COMPONENT: &str = "lpx-topology";

/// A sidecar's topology role, independent of whether it serves inference.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Role {
    /// Beside the Agent allocation's leader: register the allocation's
    /// topology endpoint and relay each request on it to the Agent's
    /// `TopologyService` on `agent_socket`.
    Publisher { agent_socket: PathBuf },
    /// In a conductor pod: serve `TopologyService` on `socket`, relaying each
    /// request to the authority registered for the identity it expects.
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
            publisher::publish(drt, namespace, agent, shutdown).await
        }
        Role::Consumer { socket } => {
            let consumer =
                Consumer::new(drt.clone(), namespace.to_string(), shutdown.child_token());
            consumer::serve(consumer, socket, shutdown).await
        }
    }
}
