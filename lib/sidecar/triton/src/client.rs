// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use std::sync::Arc;

use dynamo_backend_common::DynamoError;
use dynamo_llm::grpc::service::kserve::inference;
use dynamo_sidecar_common::{
    DEFAULT_MAX_GRPC_MESSAGE_SIZE, GrpcChannelPool, GrpcEndpoint, GrpcTransportConfig,
    connection_timeout, protocol_error, status_to_dynamo,
};
use tokio::time::{Instant, sleep_until, timeout_at};

#[derive(Clone)]
pub(crate) struct TritonClient {
    pool: Arc<GrpcChannelPool>,
}

impl TritonClient {
    pub(crate) async fn connect(
        endpoint: &GrpcEndpoint,
        transport: GrpcTransportConfig,
    ) -> Result<Self, DynamoError> {
        let pool = GrpcChannelPool::connect("Triton", endpoint, transport, false).await?;
        let client = Self {
            pool: Arc::new(pool),
        };
        client.wait_until_ready(transport).await?;
        Ok(client)
    }

    fn grpc(
        &self,
    ) -> inference::grpc_inference_service_client::GrpcInferenceServiceClient<
        tonic::transport::Channel,
    > {
        inference::grpc_inference_service_client::GrpcInferenceServiceClient::new(
            self.pool.next_channel(),
        )
        .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
        .max_encoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
    }

    async fn wait_until_ready(&self, transport: GrpcTransportConfig) -> Result<(), DynamoError> {
        let deadline = dynamo_sidecar_common::startup_deadline(transport.startup_deadline)?;
        let mut last_error = None;
        loop {
            let mut grpc = self.grpc();
            match timeout_at(
                deadline,
                grpc.server_ready(inference::ServerReadyRequest {}),
            )
            .await
            {
                Ok(Ok(response)) => {
                    if response.into_inner().ready {
                        return Ok(());
                    }
                    last_error = Some("Triton reported ready=false".to_string());
                }
                Ok(Err(status)) => last_error = Some(status.to_string()),
                Err(_) => {
                    let detail = last_error
                        .as_deref()
                        .map(|error| format!("; last error: {error}"))
                        .unwrap_or_default();
                    return Err(connection_timeout(format!(
                        "Triton did not become ready within {:?}{detail}",
                        transport.startup_deadline
                    )));
                }
            }
            let retry_at = Instant::now()
                .checked_add(transport.retry_interval)
                .ok_or_else(|| protocol_error("Triton", "retry interval overflowed"))?;
            sleep_until(retry_at.min(deadline)).await;
        }
    }

    pub(crate) async fn model_ready(&self, model: &str) -> Result<bool, DynamoError> {
        self.grpc()
            .model_ready(inference::ModelReadyRequest {
                name: model.to_string(),
                version: String::new(),
            })
            .await
            .map(|response| response.into_inner().ready)
            .map_err(|status| status_to_dynamo("ModelReady", status))
    }

    /// Names of the models Triton reports as READY, sorted and de-duplicated.
    pub(crate) async fn ready_models(&self) -> Result<Vec<String>, DynamoError> {
        let response = self
            .grpc()
            .repository_index(inference::RepositoryIndexRequest {
                repository_name: String::new(),
                ready: true,
            })
            .await
            .map_err(|status| status_to_dynamo("RepositoryIndex", status))?
            .into_inner();
        Ok(response
            .models
            .into_iter()
            .filter(|entry| entry.state.eq_ignore_ascii_case("READY"))
            .map(|entry| entry.name)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect())
    }

    pub(crate) async fn model_config(
        &self,
        model: &str,
    ) -> Result<inference::ModelConfig, DynamoError> {
        self.grpc()
            .model_config(inference::ModelConfigRequest {
                name: model.to_string(),
                version: String::new(),
            })
            .await
            .map_err(|status| status_to_dynamo("ModelConfig", status))?
            .into_inner()
            .config
            .ok_or_else(|| {
                protocol_error("Triton", format!("returned no config for model '{model}'"))
            })
    }

    pub(crate) async fn infer(
        &self,
        request: inference::ModelInferRequest,
    ) -> Result<inference::ModelInferResponse, DynamoError> {
        self.grpc()
            .model_infer(request)
            .await
            .map(|response| response.into_inner())
            .map_err(|status| status_to_dynamo("ModelInfer", status))
    }
}
