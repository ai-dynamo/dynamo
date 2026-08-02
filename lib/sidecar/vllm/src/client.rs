// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::DynamoError;
use dynamo_sidecar_common::{
    DEFAULT_MAX_GRPC_MESSAGE_SIZE, GrpcChannelPool, GrpcEndpoint, GrpcTransportConfig,
    connection_timeout,
};

pub(crate) use dynamo_sidecar_common::{engine_shutdown, invalid_argument, status_to_dynamo};

use crate::proto as pb;

#[derive(Clone, Debug)]
pub(crate) struct Discovery {
    pub(crate) server: pb::ServerInfo,
    pub(crate) model: pb::ModelInfo,
}

pub(crate) struct VllmClient {
    pool: GrpcChannelPool,
}

impl VllmClient {
    pub(crate) async fn connect_and_discover(
        endpoint: &GrpcEndpoint,
        transport: GrpcTransportConfig,
    ) -> Result<(Self, Discovery), DynamoError> {
        tokio::time::timeout(transport.startup_deadline, async {
            let client = Self::connect(endpoint, transport).await?;
            let discovery = client.discover().await?;
            Ok((client, discovery))
        })
        .await
        .map_err(|_| {
            connection_timeout(format!(
                "vLLM gRPC startup deadline {:?} expired during connection or Control discovery",
                transport.startup_deadline
            ))
        })?
    }

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
        let mut client = pb::inference_client::InferenceClient::new(self.pool.next_channel())
            .max_encoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
            .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE);
        client
            .generate_stream(request)
            .await
            .map(tonic::Response::into_inner)
            .map_err(|status| status_to_dynamo("GenerateStream", status))
    }

    pub(crate) async fn discover(&self) -> Result<Discovery, DynamoError> {
        let mut client = pb::control_client::ControlClient::new(self.pool.next_channel())
            .max_encoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
            .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE);
        let server = client
            .get_server_info(pb::GetServerInfoRequest {})
            .await
            .map(tonic::Response::into_inner)
            .map_err(|status| status_to_dynamo("GetServerInfo", status))?;
        let model = client
            .get_model_info(pb::GetModelInfoRequest {})
            .await
            .map(tonic::Response::into_inner)
            .map_err(|status| status_to_dynamo("GetModelInfo", status))?;
        Ok(Discovery { server, model })
    }

    pub(crate) async fn load_lora(
        &self,
        adapter: pb::LoraAdapter,
        load_inplace: bool,
    ) -> Result<pb::LoadLoraResponse, DynamoError> {
        let mut client = pb::control_client::ControlClient::new(self.pool.next_channel())
            .max_encoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
            .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE);
        client
            .load_lora(pb::LoadLoraRequest {
                adapter: Some(adapter),
                load_inplace,
            })
            .await
            .map(tonic::Response::into_inner)
            .map_err(|status| status_to_dynamo("LoadLora", status))
    }

    pub(crate) async fn list_loras(&self) -> Result<Vec<pb::LoraAdapter>, DynamoError> {
        let mut client = pb::control_client::ControlClient::new(self.pool.next_channel())
            .max_encoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
            .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE);
        client
            .list_loras(pb::ListLorasRequest {})
            .await
            .map(tonic::Response::into_inner)
            .map(|response| response.adapters)
            .map_err(|status| status_to_dynamo("ListLoras", status))
    }
}

pub(crate) fn protocol_error(message: impl Into<String>) -> DynamoError {
    dynamo_sidecar_common::protocol_error("vLLM", message)
}
