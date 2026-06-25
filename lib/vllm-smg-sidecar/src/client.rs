// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thin SMG vLLM gRPC client: connect-with-backoff, discovery, and error
//! mapping into [`DynamoError`].

use std::sync::atomic::{AtomicUsize, Ordering};

use dynamo_backend_common::{BackendError, DynamoError, ErrorType};
use tokio::time::Instant;
use tonic::transport::{Channel, Endpoint};

use crate::args::TransportConfig;
use crate::proto::engine as pb;
use crate::proto::engine::vllm_engine_client::VllmEngineClient;

/// Connected SMG vLLM client over a tonic [`Channel`].
pub type Client = VllmEngineClient<Channel>;

/// Engine discovery snapshot: model metadata plus server / KV role metadata.
#[derive(Clone, Debug)]
pub struct Discovery {
    pub model: pb::GetModelInfoResponse,
    pub server: pb::GetServerInfoResponse,
}

/// Dial the engine, retrying with backoff until reachable or the deadline
/// elapses.
pub async fn connect(uri: &str, cfg: &TransportConfig) -> Result<Client, DynamoError> {
    let deadline = Instant::now() + cfg.deadline;
    let mut last_err;
    loop {
        match try_connect_once(uri, cfg).await {
            Ok(client) => return Ok(client),
            Err(e) => {
                last_err = e;
                if Instant::now() >= deadline {
                    return Err(cannot_connect(format!(
                        "could not reach SMG vLLM engine at {uri} within {:?}: {last_err}",
                        cfg.deadline
                    )));
                }
                tokio::time::sleep(cfg.poll_interval).await;
            }
        }
    }
}

async fn try_connect_once(uri: &str, cfg: &TransportConfig) -> Result<Client, String> {
    let endpoint = Endpoint::from_shared(uri.to_string())
        .map_err(|e| format!("invalid endpoint `{uri}`: {e}"))?
        .connect_timeout(cfg.connect_timeout);
    let channel = endpoint.connect().await.map_err(|e| e.to_string())?;
    Ok(VllmEngineClient::new(channel))
}

/// A fixed-size pool of independent SMG gRPC connections.
pub struct Pool {
    clients: Vec<Client>,
    next: AtomicUsize,
}

impl Pool {
    /// Dial `size` (clamped to >=1) independent connections to `uri`.
    pub async fn connect(
        uri: &str,
        cfg: &TransportConfig,
        size: usize,
    ) -> Result<Self, DynamoError> {
        let size = size.max(1);
        let mut clients = Vec::with_capacity(size);
        for _ in 0..size {
            clients.push(connect(uri, cfg).await?);
        }
        Ok(Self {
            clients,
            next: AtomicUsize::new(0),
        })
    }

    /// Number of connections in the pool (always >= 1).
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.clients.len()
    }

    /// Round-robin a client for a streaming `Generate` call.
    pub fn stream_client(&self) -> Client {
        let i = self.next.fetch_add(1, Ordering::Relaxed) % self.clients.len();
        self.clients[i].clone()
    }

    /// A stable client for low-frequency control RPCs.
    pub fn control_client(&self) -> Client {
        self.clients[0].clone()
    }
}

/// Fetch model + server metadata in one shot.
pub async fn discover(client: &mut Client) -> Result<Discovery, DynamoError> {
    let model = client
        .get_model_info(pb::GetModelInfoRequest {})
        .await
        .map_err(|s| status_to_dynamo("GetModelInfo", s))?
        .into_inner();
    let server = client
        .get_server_info(pb::GetServerInfoRequest {})
        .await
        .map_err(|s| status_to_dynamo("GetServerInfo", s))?
        .into_inner();
    Ok(Discovery { model, server })
}

// ============================================================================
// Error mapping
// ============================================================================

fn backend(kind: BackendError, msg: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(kind))
        .message(msg)
        .build()
}

/// Client-supplied bad input.
pub fn invalid_arg(msg: impl Into<String>) -> DynamoError {
    backend(BackendError::InvalidArgument, msg)
}

/// The engine is gone / never came up / used before start.
pub fn engine_shutdown(msg: impl Into<String>) -> DynamoError {
    backend(BackendError::EngineShutdown, msg)
}

/// Could not establish the transport to the engine.
pub fn cannot_connect(msg: impl Into<String>) -> DynamoError {
    backend(BackendError::CannotConnect, msg)
}

/// Map a tonic transport-level [`Status`](tonic::Status) to a typed error.
pub fn status_to_dynamo(rpc: &str, status: tonic::Status) -> DynamoError {
    let kind = match status.code() {
        tonic::Code::InvalidArgument | tonic::Code::NotFound | tonic::Code::OutOfRange => {
            BackendError::InvalidArgument
        }
        tonic::Code::Unavailable => BackendError::CannotConnect,
        tonic::Code::Cancelled => BackendError::Cancelled,
        tonic::Code::DeadlineExceeded => BackendError::ConnectionTimeout,
        _ => BackendError::Unknown,
    };
    backend(
        kind,
        format!("{rpc}: {} ({:?})", status.message(), status.code()),
    )
}
