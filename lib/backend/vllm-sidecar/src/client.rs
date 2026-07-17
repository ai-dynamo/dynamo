// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicUsize, Ordering};

use dynamo_backend_common::{BackendError, DynamoError, ErrorType, GrpcTransportConfig};
use futures::future::try_join_all;
use tokio::time::{Instant, sleep_until, timeout_at};
use tonic::transport::{Channel, Endpoint};

use crate::proto as pb;

const MAX_MESSAGE_SIZE: usize = 64 * 1024 * 1024;

pub(crate) struct VllmClient {
    channels: Vec<Channel>,
    next: AtomicUsize,
}

impl VllmClient {
    pub(crate) async fn connect(
        endpoint: &str,
        transport: GrpcTransportConfig,
    ) -> Result<Self, DynamoError> {
        let endpoint = Endpoint::from_shared(endpoint.to_string())
            .map_err(|error| invalid_argument(format!("invalid vLLM endpoint: {error}")))?;
        let deadline = Instant::now() + transport.startup_deadline;
        let first = connect_until_ready(endpoint.clone(), transport, deadline).await?;
        let mut channels = vec![first];
        let remaining = try_join_all((1..transport.connections.get()).map(|_| {
            let endpoint = endpoint.clone();
            async move { connect_until_ready(endpoint, transport, deadline).await }
        }))
        .await?;
        channels.extend(remaining);
        Ok(Self {
            channels,
            next: AtomicUsize::new(0),
        })
    }

    pub(crate) fn connection_count(&self) -> usize {
        self.channels.len()
    }

    pub(crate) async fn generate_stream(
        &self,
        request: pb::GenerateRequest,
    ) -> Result<tonic::Streaming<pb::GenerateResponse>, DynamoError> {
        let index = self.next.fetch_add(1, Ordering::Relaxed) % self.channels.len();
        let mut client = pb::generate_client::GenerateClient::new(self.channels[index].clone())
            .max_encoding_message_size(MAX_MESSAGE_SIZE)
            .max_decoding_message_size(MAX_MESSAGE_SIZE);
        client
            .generate_stream(request)
            .await
            .map(tonic::Response::into_inner)
            .map_err(|status| status_to_dynamo("GenerateStream", status))
    }
}

async fn connect_until_ready(
    endpoint: Endpoint,
    transport: GrpcTransportConfig,
    deadline: Instant,
) -> Result<Channel, DynamoError> {
    let mut last_error = None;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(startup_timeout(transport, last_error.as_deref()));
        }

        let attempt_endpoint = endpoint
            .clone()
            .connect_timeout(transport.connect_attempt_timeout.min(remaining));
        let attempt = attempt_endpoint.connect();
        match timeout_at(deadline, attempt).await {
            Ok(Ok(channel)) => return Ok(channel),
            Ok(Err(error)) => {
                tracing::debug!(%error, "waiting for vLLM gRPC");
                last_error = Some(error.to_string());
            }
            Err(_) => return Err(startup_timeout(transport, last_error.as_deref())),
        }

        if Instant::now() >= deadline {
            return Err(startup_timeout(transport, last_error.as_deref()));
        }
        sleep_until((Instant::now() + transport.retry_interval).min(deadline)).await;
    }
}

fn startup_timeout(transport: GrpcTransportConfig, last_error: Option<&str>) -> DynamoError {
    let cause = last_error.unwrap_or("the connection attempt exceeded the startup deadline");
    cannot_connect(format!(
        "failed to establish the vLLM gRPC connection pool within {:?}: {cause}",
        transport.startup_deadline
    ))
}

fn backend(kind: BackendError, message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(kind))
        .message(message)
        .build()
}

pub(crate) fn invalid_argument(message: impl Into<String>) -> DynamoError {
    backend(BackendError::InvalidArgument, message)
}

pub(crate) fn protocol_error(message: impl Into<String>) -> DynamoError {
    backend(
        BackendError::Unknown,
        format!("invalid vLLM gRPC response: {}", message.into()),
    )
}

pub(crate) fn engine_shutdown(message: impl Into<String>) -> DynamoError {
    backend(BackendError::EngineShutdown, message)
}

fn cannot_connect(message: impl Into<String>) -> DynamoError {
    backend(BackendError::CannotConnect, message)
}

pub(crate) fn status_to_dynamo(rpc: &str, status: tonic::Status) -> DynamoError {
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
