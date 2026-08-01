// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::DynamoError;
use dynamo_sidecar_common::{
    DEFAULT_MAX_GRPC_MESSAGE_SIZE, GrpcChannelPool, GrpcEndpoint, GrpcTransportConfig,
};

pub(crate) use dynamo_sidecar_common::{
    cannot_connect, engine_shutdown, invalid_argument, status_to_dynamo,
};

use crate::proto as pb;

pub(crate) struct VllmClient {
    pool: GrpcChannelPool,
}

impl VllmClient {
    pub(crate) async fn connect(
        endpoint: &GrpcEndpoint,
        transport: GrpcTransportConfig,
    ) -> Result<Self, DynamoError> {
        let pool = GrpcChannelPool::connect("vLLM", endpoint, transport).await?;
        Ok(Self { pool })
    }

    pub(crate) fn connection_count(&self) -> usize {
        self.pool.len()
    }

    pub(crate) async fn generate_stream(
        &self,
        request: pb::GenerateRequest,
    ) -> Result<tonic::Streaming<pb::GenerateResponse>, DynamoError> {
        let mut client = pb::generate_client::GenerateClient::new(self.pool.next_channel())
            .max_encoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
            .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE);
        client
            .generate_stream(request)
            .await
            .map(tonic::Response::into_inner)
            .map_err(|status| status_to_dynamo("GenerateStream", status))
    }

    /// Open the `GenerateStream` server-stream returning the raw `tonic::Status`
    /// on failure (no `status_to_dynamo` mapping). The direct-gRPC dispatch maps
    /// statuses to TOP-LEVEL `ErrorType`s itself so `PushRouter` fault detection
    /// fires — see `crate::direct::status_to_top_level`.
    pub(crate) async fn generate_stream_raw(
        &self,
        request: pb::GenerateRequest,
    ) -> Result<tonic::Streaming<pb::GenerateResponse>, tonic::Status> {
        let mut client = pb::generate_client::GenerateClient::new(self.pool.next_channel())
            .max_encoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
            .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE);
        client
            .generate_stream(request)
            .await
            .map(tonic::Response::into_inner)
    }
}

/// Cheap liveness probe for the direct-mode health loop.
///
/// vLLM's released gRPC surface exposes only `Generate` / `GenerateStream` — no
/// health or model-info RPC (its `HealthCheck` and `grpc.health` both return
/// UNIMPLEMENTED), so there is no RPC to poll. Instead we open a single fresh
/// gRPC channel to the engine endpoint: it succeeds while the server is
/// listening and fails fast (connection refused) once the engine dies, which is
/// all the direct orchestrator needs to unregister and later re-register on
/// recovery. The channel is dropped immediately; this never runs inference.
pub(crate) async fn probe_liveness(
    endpoint: &GrpcEndpoint,
    transport: GrpcTransportConfig,
) -> Result<(), DynamoError> {
    let channel = tonic::transport::Endpoint::from_shared(endpoint.as_str().to_string())
        .map_err(|error| invalid_argument(format!("invalid vLLM endpoint: {error}")))?
        .connect_timeout(transport.connect_attempt_timeout);
    channel
        .connect()
        .await
        .map(|_channel| ())
        .map_err(|error| cannot_connect(format!("vLLM gRPC liveness probe failed: {error}")))
}

pub(crate) fn protocol_error(message: impl Into<String>) -> DynamoError {
    dynamo_sidecar_common::protocol_error("vLLM", message)
}
