// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Agent's topology service, which the publisher reads.

use std::path::Path;
use std::time::Duration;

use tonic::Status;
use tonic::codegen::http::Uri;
use tonic::transport::{Channel, Endpoint};

use super::proto::topology_service_client::TopologyServiceClient;
use super::proto::{GetSnapshotRequest, GetSnapshotResponse};

/// The `:authority` of requests over a Unix socket, which has no host of its
/// own.
const UNIX_SOCKET_ORIGIN: &str = "http://localhost";

/// The Agent's `TopologyService` on its pod-local Unix socket.
#[derive(Clone)]
pub(crate) struct AgentClient {
    client: TopologyServiceClient<Channel>,
    /// The socket's path, which failures name: the conductor that finally
    /// reads them runs in another pod.
    socket: String,
}

impl AgentClient {
    /// A client that connects on first use, so the socket need not exist yet,
    /// and reconnects after the Agent restarts.
    pub(crate) fn new(socket: &Path) -> Result<Self, String> {
        let socket = socket
            .to_str()
            .ok_or_else(|| format!("the Agent socket {} is not UTF-8", socket.display()))?;
        let endpoint = Endpoint::from_shared(format!("unix://{socket}"))
            .map_err(|error| format!("invalid Agent socket {socket}: {error}"))?
            .origin(Uri::from_static(UNIX_SOCKET_ORIGIN));
        Ok(Self {
            client: TopologyServiceClient::new(endpoint.connect_lazy()),
            socket: socket.to_string(),
        })
    }

    /// One live answer, bounded by `timeout`. Failing to reach the Agent,
    /// including a socket that does not exist yet, is UNAVAILABLE.
    pub(crate) async fn get_snapshot(
        &self,
        request: GetSnapshotRequest,
        timeout: Duration,
    ) -> Result<GetSnapshotResponse, Status> {
        let mut client = self.client.clone();
        match tokio::time::timeout(timeout, client.get_snapshot(request)).await {
            Ok(Ok(response)) => Ok(response.into_inner()),
            Ok(Err(status)) => Err(self.transport_unavailable(status)),
            Err(_) => Err(Status::unavailable(format!(
                "the Agent's topology socket {} did not answer within {timeout:?}",
                self.socket
            ))),
        }
    }

    /// Tonic gives the transport's own failures various codes: UNAVAILABLE
    /// for a socket it cannot connect to, UNKNOWN for a connection lost
    /// during the call, CANCELLED for a call the connection dropped or a
    /// stream reset with CANCEL, INTERNAL for an HTTP/2 GOAWAY. Such a status
    /// carries the local error as its source and a status the Agent sent has
    /// none, so every status with a source becomes UNAVAILABLE naming the
    /// socket and the cause, and the Agent's own statuses keep their code and
    /// message.
    fn transport_unavailable(&self, status: Status) -> Status {
        if std::error::Error::source(&status).is_none() {
            return status;
        }
        Status::unavailable(format!(
            "the Agent's topology socket {} is unreachable: {}",
            self.socket,
            cause(&status)
        ))
    }
}

/// The status's source chain, outermost first, each text once: the
/// transport's wrappers often repeat their inner error, and the status's own
/// message only restates one of them.
fn cause(status: &Status) -> String {
    let mut cause = String::new();
    let mut source = std::error::Error::source(status);
    while let Some(error) = source {
        let text = error.to_string();
        if !text.is_empty() && !cause.contains(&text) {
            if !cause.is_empty() {
                cause.push_str(": ");
            }
            cause.push_str(&text);
        }
        source = error.source();
    }
    cause
}
