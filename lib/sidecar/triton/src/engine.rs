// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use async_trait::async_trait;
use dynamo_backend_common::{DynamoError, EngineConfig, GenerateContext, RawEngine};
use dynamo_llm::protocols::tensor::{NvCreateTensorRequest, TensorModelConfig};
use futures::stream::BoxStream;

use crate::client::TritonClient;
use crate::convert::{from_triton_response, to_triton_request};

pub(crate) struct TritonModelEngine {
    model: String,
    client: TritonClient,
    tensor_model_config: TensorModelConfig,
}

impl TritonModelEngine {
    pub(crate) fn new(
        model: String,
        client: TritonClient,
        tensor_model_config: TensorModelConfig,
    ) -> Self {
        Self {
            model,
            client,
            tensor_model_config,
        }
    }
}

#[async_trait]
impl RawEngine for TritonModelEngine {
    async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
        if !self.client.model_ready(&self.model).await? {
            return Err(dynamo_sidecar_common::cannot_connect(format!(
                "Triton model '{}' is not ready",
                self.model
            )));
        }
        Ok(EngineConfig {
            model: self.model.clone(),
            served_model_name: Some(self.model.clone()),
            ..Default::default()
        })
    }

    async fn generate(
        &self,
        request: serde_json::Value,
        ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<serde_json::Value, DynamoError>>, DynamoError> {
        let request: NvCreateTensorRequest = serde_json::from_value(request).map_err(|error| {
            dynamo_sidecar_common::invalid_argument(format!(
                "invalid Dynamo tensor request: {error}"
            ))
        })?;
        let request = to_triton_request(request, &self.model)?;
        let client = self.client.clone();
        let context = ctx.inner_arc();
        Ok(Box::pin(async_stream::stream! {
            let response = tokio::select! {
                _ = context.stopped() => return,
                _ = context.killed() => return,
                response = client.infer(request) => response,
            };
            match response.and_then(from_triton_response) {
                Ok(response) => match serde_json::to_value(response) {
                    Ok(response) => yield Ok(response),
                    Err(error) => yield Err(dynamo_sidecar_common::protocol_error(
                        "Triton",
                        format!("could not serialize Triton response: {error}")
                    )),
                },
                Err(error) => yield Err(error),
            }
        }))
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        Ok(())
    }

    fn tensor_model_config(&self) -> Option<TensorModelConfig> {
        Some(self.tensor_model_config.clone())
    }
}
