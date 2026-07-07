// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use dynamo_backend_common::{BackendError, DynamoError, ErrorType};
use futures::future::try_join_all;
use tokio::time::Instant;
use tonic::transport::{Channel, Endpoint};

use crate::proto as pb;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const STARTUP_TIMEOUT: Duration = Duration::from_secs(300);
const RETRY_INTERVAL: Duration = Duration::from_secs(1);

pub(crate) struct VllmClient {
    channels: Vec<Channel>,
    next: AtomicUsize,
}

impl VllmClient {
    pub(crate) async fn connect(endpoint: &str, connections: usize) -> Result<Self, DynamoError> {
        if connections == 0 {
            return Err(invalid_argument(
                "vLLM gRPC connection count must be greater than zero",
            ));
        }
        let endpoint = Endpoint::from_shared(endpoint.to_string())
            .map_err(|error| invalid_argument(format!("invalid vLLM endpoint: {error}")))?
            .connect_timeout(CONNECT_TIMEOUT);
        let first = connect_until_ready(endpoint.clone()).await?;
        let mut channels = vec![first];
        let remaining = try_join_all((1..connections).map(|_| {
            let endpoint = endpoint.clone();
            async move { endpoint.connect().await }
        }))
        .await
        .map_err(|error| cannot_connect(format!("failed to connect to vLLM: {error}")))?;
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
        let mut client = pb::generate_client::GenerateClient::new(self.channels[index].clone());
        client
            .generate_stream(request)
            .await
            .map(tonic::Response::into_inner)
            .map_err(|status| status_to_dynamo("GenerateStream", status))
    }
}

async fn connect_until_ready(endpoint: Endpoint) -> Result<Channel, DynamoError> {
    let deadline = Instant::now() + STARTUP_TIMEOUT;
    loop {
        match endpoint.clone().connect().await {
            Ok(channel) => return Ok(channel),
            Err(error) if Instant::now() < deadline => {
                tracing::debug!(%error, "waiting for vLLM gRPC");
                tokio::time::sleep(RETRY_INTERVAL).await;
            }
            Err(error) => {
                return Err(cannot_connect(format!(
                    "failed to connect to vLLM within {STARTUP_TIMEOUT:?}: {error}"
                )));
            }
        }
    }
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
