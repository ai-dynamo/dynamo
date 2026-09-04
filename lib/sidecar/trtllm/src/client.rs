// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thin client for TensorRT-LLM's OpenEngine gRPC services (`openengine.v1`).

use std::time::Duration;

use dynamo_backend_common::{BackendError, DynamoError, ErrorType};
use dynamo_sidecar_common::{
    DEFAULT_MAX_GRPC_MESSAGE_SIZE, GrpcChannelPool, GrpcEndpoint, GrpcTransportConfig,
    connection_timeout,
};
use tonic::transport::Channel;

pub(crate) use dynamo_sidecar_common::{engine_shutdown, invalid_argument, status_to_dynamo};

use crate::proto as pb;
use crate::proto::control_client::ControlClient;
use crate::proto::inference_client::InferenceClient;

/// Deadline for the one-shot `Control` RPCs issued at startup / on cancel, so a
/// connected-but-unresponsive server cannot hang `start` or `abort`.
const RPC_TIMEOUT: Duration = Duration::from_secs(30);

pub(crate) struct TrtllmClient {
    pool: GrpcChannelPool,
}

impl TrtllmClient {
    pub(crate) async fn connect(
        endpoint: &GrpcEndpoint,
        transport: GrpcTransportConfig,
    ) -> Result<Self, DynamoError> {
        let pool = GrpcChannelPool::connect("TensorRT-LLM", endpoint, transport).await?;
        Ok(Self { pool })
    }

    pub(crate) fn connection_count(&self) -> usize {
        self.pool.len()
    }

    fn inference(&self) -> InferenceClient<Channel> {
        InferenceClient::new(self.pool.next_channel())
            .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
            .max_encoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
    }

    fn control(&self) -> ControlClient<Channel> {
        ControlClient::new(self.pool.next_channel())
            .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
            .max_encoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
    }

    pub(crate) async fn generate(
        &self,
        request: pb::GenerateRequest,
    ) -> Result<tonic::Streaming<pb::GenerateResponse>, DynamoError> {
        self.inference()
            .generate(request)
            .await
            .map(tonic::Response::into_inner)
            .map_err(|status| status_to_dynamo("Generate", status))
    }

    /// Queries `GetModelInfo` and returns the reported maximum context length
    /// (input + output) as the registration context length, if positive.
    ///
    /// The server is the only source of this value, so the OpenEngine `Control`
    /// service is required: `engine::start` turns both an RPC error and a
    /// missing `max_context_length` into a startup failure rather than serving
    /// with an unknown context window.
    pub(crate) async fn model_info(&self, model: &str) -> Result<Option<u32>, DynamoError> {
        let info = tokio::time::timeout(
            RPC_TIMEOUT,
            self.control().get_model_info(pb::GetModelInfoRequest {
                model: model.to_string(),
            }),
        )
        .await
        .map_err(|_| {
            connection_timeout(format!(
                "GetModelInfo did not respond within {RPC_TIMEOUT:?}"
            ))
        })?
        .map(tonic::Response::into_inner)
        .map_err(|status| status_to_dynamo("GetModelInfo", status))?;
        Ok(info.max_context_length.filter(|len| *len > 0))
    }

    pub(crate) async fn abort(&self, request_id: String) -> Result<(), DynamoError> {
        let response = tokio::time::timeout(
            RPC_TIMEOUT,
            self.control().abort(pb::AbortRequest {
                target: Some(pb::abort_request::Target::RequestId(request_id)),
            }),
        )
        .await
        .map_err(|_| connection_timeout(format!("Abort did not respond within {RPC_TIMEOUT:?}")))?
        .map(tonic::Response::into_inner)
        .map_err(|status| status_to_dynamo("Abort", status))?;
        // A request already gone (finished/never seen) is not an error; only an
        // unspecified status is protocol drift.
        match pb::AbortStatus::try_from(response.status) {
            Ok(pb::AbortStatus::Aborted) | Ok(pb::AbortStatus::AlreadyFinished) => Ok(()),
            _ => Err(protocol_error(format!(
                "TensorRT-LLM returned an unexpected abort status: {}",
                response.message
            ))),
        }
    }
}

pub(crate) fn protocol_error(message: impl Into<String>) -> DynamoError {
    dynamo_sidecar_common::protocol_error("TensorRT-LLM", message)
}

/// A generation failure the engine reported in-band via an `EngineError` event
/// (as opposed to a transport/validation gRPC status).
pub(crate) fn engine_error(message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(BackendError::Unknown))
        .message(message)
        .build()
}
