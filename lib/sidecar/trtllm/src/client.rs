// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thin client for TensorRT-LLM's native `trtllm.TrtllmService`.

use std::time::Duration;

use dynamo_backend_common::DynamoError;
use dynamo_sidecar_common::{
    DEFAULT_MAX_GRPC_MESSAGE_SIZE, GrpcChannelPool, GrpcEndpoint, GrpcTransportConfig,
    connection_timeout,
};
use tonic::transport::Channel;

pub(crate) use dynamo_sidecar_common::{engine_shutdown, invalid_argument, status_to_dynamo};

use crate::proto as pb;
use crate::proto::trtllm_service_client::TrtllmServiceClient;

/// Deadline for the one-shot control RPCs issued at startup / on cancel, so a
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

    fn client(&self) -> TrtllmServiceClient<Channel> {
        TrtllmServiceClient::new(self.pool.next_channel())
            .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
            .max_encoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
    }

    pub(crate) async fn generate(
        &self,
        request: pb::GenerateRequest,
    ) -> Result<tonic::Streaming<pb::GenerateResponse>, DynamoError> {
        self.client()
            .generate(request)
            .await
            .map(tonic::Response::into_inner)
            .map_err(|status| status_to_dynamo("Generate", status))
    }

    /// Queries `GetModelInfo` and returns the reported maximum sequence length
    /// (input + output) as the registration context length, if it is usable.
    ///
    /// TensorRT-LLM answers `max_seq_len` with `args.max_seq_len or
    /// args.max_input_len`, and on the PyTorch path `max_seq_len` is unset
    /// while `max_input_len` holds its 1024 default. An equal pair is that
    /// fallback's signature: the number describes the argument defaults rather
    /// than the model, so it is discarded in favour of `--context-length` or
    /// the context length the frontend reads from the model itself.
    pub(crate) async fn model_info(&self) -> Result<Option<u32>, DynamoError> {
        let info = tokio::time::timeout(
            RPC_TIMEOUT,
            self.client().get_model_info(pb::GetModelInfoRequest {}),
        )
        .await
        .map_err(|_| {
            connection_timeout(format!(
                "GetModelInfo did not respond within {RPC_TIMEOUT:?}"
            ))
        })?
        .map(tonic::Response::into_inner)
        .map_err(|status| status_to_dynamo("GetModelInfo", status))?;
        let max_seq_len = u32::try_from(info.max_seq_len).ok().filter(|len| *len > 0);
        let max_input_len = u32::try_from(info.max_input_len)
            .ok()
            .filter(|len| *len > 0);
        if max_seq_len.is_some() && max_seq_len == max_input_len {
            tracing::warn!(
                reported_context_length = max_seq_len,
                "TensorRT-LLM reported max_seq_len == max_input_len, which is how it \
                 answers when the engine was started without --max_seq_len; ignoring \
                 the report"
            );
            return Ok(None);
        }
        Ok(max_seq_len)
    }

    pub(crate) async fn abort(&self, request_id: String) -> Result<(), DynamoError> {
        let response = tokio::time::timeout(
            RPC_TIMEOUT,
            self.client().abort(pb::AbortRequest { request_id }),
        )
        .await
        .map_err(|_| connection_timeout(format!("Abort did not respond within {RPC_TIMEOUT:?}")))?
        .map(tonic::Response::into_inner)
        .map_err(|status| status_to_dynamo("Abort", status))?;
        if !response.success {
            return Err(protocol_error(format!(
                "TensorRT-LLM rejected abort: {}",
                response.message
            )));
        }
        Ok(())
    }
}

pub(crate) fn protocol_error(message: impl Into<String>) -> DynamoError {
    dynamo_sidecar_common::protocol_error("TensorRT-LLM", message)
}
