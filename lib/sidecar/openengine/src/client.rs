// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thin client for the OpenEngine Inference and Control services.

use std::time::Duration;

use dynamo_backend_common::DynamoError;
use dynamo_sidecar_common::{
    DEFAULT_MAX_GRPC_MESSAGE_SIZE, GrpcChannelPool, GrpcEndpoint, GrpcTransportConfig,
    connection_timeout,
};
use tonic::transport::Channel;

pub(crate) use dynamo_sidecar_common::{
    engine_shutdown, invalid_argument, protocol_error as sidecar_protocol_error, status_to_dynamo,
};

use crate::proto as pb;
use crate::proto::control_client::ControlClient;
use crate::proto::inference_client::InferenceClient;

pub(crate) struct OpenEngineClient {
    pool: GrpcChannelPool,
}

impl OpenEngineClient {
    pub(crate) async fn connect(
        endpoint: &GrpcEndpoint,
        transport: GrpcTransportConfig,
    ) -> Result<Self, DynamoError> {
        let pool = GrpcChannelPool::connect("OpenEngine", endpoint, transport).await?;
        Ok(Self { pool })
    }

    pub(crate) fn connection_count(&self) -> usize {
        self.pool.len()
    }

    fn control_client(&self) -> ControlClient<Channel> {
        ControlClient::new(self.pool.next_channel())
            .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
            .max_encoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
    }

    fn inference_client(&self) -> InferenceClient<Channel> {
        InferenceClient::new(self.pool.next_channel())
            .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
            .max_encoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
    }

    pub(crate) async fn server_info(
        &self,
        timeout: Duration,
    ) -> Result<pb::ServerInfo, DynamoError> {
        tokio::time::timeout(
            timeout,
            self.control_client()
                .get_server_info(pb::GetServerInfoRequest {}),
        )
        .await
        .map_err(|_| {
            connection_timeout(format!(
                "OpenEngine GetServerInfo did not respond within {timeout:?}"
            ))
        })?
        .map(tonic::Response::into_inner)
        .map_err(|status| status_to_dynamo("GetServerInfo", status))
    }

    pub(crate) async fn model_info(
        &self,
        model: &str,
        timeout: Duration,
    ) -> Result<pb::ModelInfo, DynamoError> {
        tokio::time::timeout(
            timeout,
            self.control_client()
                .get_model_info(pb::GetModelInfoRequest {
                    model: model.to_string(),
                }),
        )
        .await
        .map_err(|_| {
            connection_timeout(format!(
                "OpenEngine GetModelInfo did not respond within {timeout:?}"
            ))
        })?
        .map(tonic::Response::into_inner)
        .map_err(|status| status_to_dynamo("GetModelInfo", status))
    }

    pub(crate) async fn generate(
        &self,
        request: tonic::Request<pb::GenerateRequest>,
    ) -> Result<tonic::Streaming<pb::GenerateResponse>, DynamoError> {
        self.inference_client()
            .generate(request)
            .await
            .map(tonic::Response::into_inner)
            .map_err(|status| status_to_dynamo("Generate", status))
    }
}

pub(crate) fn protocol_error(message: impl Into<String>) -> DynamoError {
    sidecar_protocol_error("OpenEngine", message)
}
