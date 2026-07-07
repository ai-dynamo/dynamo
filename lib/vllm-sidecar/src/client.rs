// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thin engine RPC v1 gRPC client: connect-with-backoff, two-RPC discovery,
//! and error mapping into [`DynamoError`].
//!
//! A single tonic [`Channel`] multiplexes every concurrent stream over **one**
//! HTTP/2 connection — one socket, one codec task. Under overhead-bound load
//! (low/zero per-token engine compute, high concurrency) that one codec task
//! becomes the throughput ceiling: streams queue behind it, latency grows
//! unbounded, and requests start timing out. [`Pool`] sidesteps this by holding
//! several independent [`Channel`]s and round-robining streaming calls across
//! them, so concurrent requests spread over multiple sockets + codec tasks.

use std::sync::atomic::{AtomicUsize, Ordering};

use dynamo_backend_common::{BackendError, DynamoError, ErrorType};
use tokio::time::Instant;
use tonic::transport::{Channel, Endpoint};

use crate::args::TransportConfig;
use crate::proto as pb;
use crate::proto::engine_client::EngineClient;

/// Connected engine RPC client over a tonic [`Channel`].
pub type Client = EngineClient<Channel>;

/// Engine discovery snapshot: identity / role / parallelism plus model caps.
#[derive(Clone, Debug)]
pub struct Discovery {
    pub engine: pb::EngineInfo,
    pub model: pb::ModelInfo,
}

/// Dial the engine, retrying with backoff until reachable or the deadline
/// elapses.
///
/// Each attempt is bounded by [`TransportConfig::connect_timeout`]; failed
/// attempts are retried every [`TransportConfig::poll_interval`] until
/// [`TransportConfig::deadline`].
pub async fn connect(uri: &str, cfg: &TransportConfig) -> Result<Client, DynamoError> {
    connect_channel(uri, cfg).await.map(EngineClient::new)
}

async fn connect_channel(uri: &str, cfg: &TransportConfig) -> Result<Channel, DynamoError> {
    let deadline = Instant::now() + cfg.deadline;
    let mut last_err;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(cannot_connect(format!(
                "could not reach engine RPC at {uri} within {:?}",
                cfg.deadline
            )));
        }
        match try_connect_once(uri, cfg, remaining.min(cfg.connect_timeout)).await {
            Ok(client) => return Ok(client),
            Err(e) => {
                last_err = e;
                if Instant::now() >= deadline {
                    return Err(cannot_connect(format!(
                        "could not reach engine RPC at {uri} within {:?}: {last_err}",
                        cfg.deadline
                    )));
                }
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    return Err(cannot_connect(format!(
                        "could not reach engine RPC at {uri} within {:?}: {last_err}",
                        cfg.deadline
                    )));
                }
                tokio::time::sleep(cfg.poll_interval.min(remaining)).await;
            }
        }
    }
}

async fn try_connect_once(
    uri: &str,
    cfg: &TransportConfig,
    connect_timeout: std::time::Duration,
) -> Result<Channel, String> {
    let endpoint = Endpoint::from_shared(uri.to_string())
        .map_err(|e| format!("invalid endpoint `{uri}`: {e}"))?
        .connect_timeout(connect_timeout)
        .timeout(cfg.connect_timeout);
    let channel = endpoint.connect().await.map_err(|e| e.to_string())?;
    Ok(channel)
}

/// A fixed-size pool of independent engine RPC connections.
///
/// Each [`Client`] wraps its own tonic [`Channel`] — its own HTTP/2 connection,
/// socket, and codec task on both ends. Streaming `generate` calls are
/// round-robined via [`stream_client`](Pool::stream_client) so concurrent
/// requests are not all serialized through a single connection. Low-frequency
/// control RPCs (discovery / health / abort / drain / kv-event-sources) use a
/// stable connection via [`control_client`](Pool::control_client).
pub struct Pool {
    channels: Vec<Channel>,
    next: AtomicUsize,
}

impl Pool {
    /// Dial `size` (clamped to ≥1) independent connections to `uri`. Every
    /// connection uses the same connect-with-backoff budget; the first blocks
    /// until the engine is reachable, the rest then connect against the now-up
    /// server (typically on their first attempt).
    pub async fn connect(
        uri: &str,
        cfg: &TransportConfig,
        size: usize,
    ) -> Result<Self, DynamoError> {
        let size = size.max(1);
        let mut channels = Vec::with_capacity(size);
        for _ in 0..size {
            channels.push(connect_channel(uri, cfg).await?);
        }
        Ok(Self {
            channels,
            next: AtomicUsize::new(0),
        })
    }

    /// Number of connections in the pool (always ≥ 1).
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.channels.len()
    }

    /// Round-robin a client for a streaming `generate` call. The returned
    /// [`Client`] is a cheap clone sharing one of the pool's connections.
    pub fn stream_client(&self) -> Client {
        let i = self.next.fetch_add(1, Ordering::Relaxed) % self.channels.len();
        EngineClient::new(self.channels[i].clone())
    }

    /// A stable client (the first connection) for low-frequency control RPCs.
    pub fn control_client(&self) -> Client {
        EngineClient::new(self.channels[0].clone())
    }
}

/// Fetch engine + model metadata within one shared deadline.
pub async fn discover(
    client: &mut Client,
    timeout: std::time::Duration,
) -> Result<Discovery, DynamoError> {
    let deadline = Instant::now() + timeout;
    let engine = tokio::time::timeout(
        deadline.saturating_duration_since(Instant::now()),
        client.get_engine_info(pb::GetEngineInfoRequest {}),
    )
    .await
    .map_err(|_| engine_shutdown("GetEngineInfo timed out"))?
    .map_err(|s| status_to_dynamo("GetEngineInfo", s))?
    .into_inner();
    let model = tokio::time::timeout(
        deadline.saturating_duration_since(Instant::now()),
        client.get_model_info(pb::GetModelInfoRequest {}),
    )
    .await
    .map_err(|_| engine_shutdown("GetModelInfo timed out"))?
    .map_err(|s| status_to_dynamo("GetModelInfo", s))?
    .into_inner();
    Ok(Discovery { engine, model })
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

/// The engine returned a response that violates the private RPC contract.
pub fn protocol_error(msg: impl Into<String>) -> DynamoError {
    backend(
        BackendError::Unknown,
        format!("Generate returned an invalid response: {}", msg.into()),
    )
}

/// Could not establish the transport to the engine.
pub fn cannot_connect(msg: impl Into<String>) -> DynamoError {
    backend(BackendError::CannotConnect, msg)
}

/// Map a tonic transport-level [`Status`](tonic::Status) to a typed error.
pub fn status_to_dynamo(rpc: &str, status: tonic::Status) -> DynamoError {
    let kind = match status.code() {
        tonic::Code::InvalidArgument
        | tonic::Code::NotFound
        | tonic::Code::OutOfRange
        | tonic::Code::FailedPrecondition
        | tonic::Code::AlreadyExists => BackendError::InvalidArgument,
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

/// Map a structured [`pb::EngineError`] stream event to a typed error.
pub fn engine_error_to_dynamo(err: &pb::EngineError) -> DynamoError {
    let code = pb::ErrorCode::try_from(err.code).unwrap_or(pb::ErrorCode::Unspecified);
    let kind = match code {
        pb::ErrorCode::InvalidArgument
        | pb::ErrorCode::UnsupportedFeature
        | pb::ErrorCode::RoleMismatch
        | pb::ErrorCode::ModelNotFound
        | pb::ErrorCode::KvSessionNotFound => BackendError::InvalidArgument,
        pb::ErrorCode::Cancelled => BackendError::Cancelled,
        pb::ErrorCode::Draining => BackendError::EngineShutdown,
        pb::ErrorCode::KvTransferFailed => BackendError::Disconnected,
        _ => BackendError::Unknown,
    };
    backend(kind, format!("engine error [{code:?}]: {}", err.message))
}
