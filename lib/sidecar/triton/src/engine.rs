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

#[cfg(test)]
mod tests {
    use dynamo_backend_common::{BackendError, ErrorType, GenerateContext, RawEngine};
    use dynamo_llm::protocols::tensor::{
        DataType, FlattenTensor, NvCreateTensorRequest, NvCreateTensorResponse, Tensor,
        TensorMetadata,
    };
    use futures::StreamExt;
    use serde_json::json;

    use super::TritonModelEngine;
    use crate::client::TritonClient;
    use crate::http::tests::{Reply, infer_reply, triton};

    fn engine(client: TritonClient) -> TritonModelEngine {
        TritonModelEngine::new("model".to_string(), client, Default::default())
    }

    fn request(model: &str) -> serde_json::Value {
        serde_json::to_value(NvCreateTensorRequest {
            id: Some("request-1".to_string()),
            model: model.to_string(),
            tensors: vec![Tensor {
                metadata: TensorMetadata {
                    name: "INPUT0".to_string(),
                    data_type: DataType::Int32,
                    shape: vec![1],
                    parameters: Default::default(),
                },
                data: FlattenTensor::Int32(vec![5]),
            }],
            outputs: Vec::new(),
            parameters: Default::default(),
            nvext: None,
        })
        .unwrap()
    }

    fn context() -> GenerateContext {
        GenerateContext::new(dynamo_backend_common::testing::mock_context(), None)
    }

    #[tokio::test]
    async fn start_requires_a_ready_model() {
        let (client, server) = triton(vec![
            Reply::json("400 Bad Request", json!({"error": "unknown model"})),
            Reply::json("200 OK", json!({})),
        ])
        .await;
        let engine = engine(client);
        let error = engine.start(0).await.unwrap_err().to_string();
        assert!(
            error.contains("Triton model 'model' is not ready"),
            "{error}"
        );

        let config = engine.start(0).await.unwrap();
        assert_eq!(config.model, "model");
        assert_eq!(config.served_model_name.as_deref(), Some("model"));
        assert!(engine.tensor_model_config().is_some());
        server.await.unwrap();
    }

    #[tokio::test]
    async fn generate_rejects_invalid_requests_before_calling_triton() {
        // A client without transports panics if a request reaches Triton.
        let engine = engine(TritonClient::for_tests());
        for (request, expected) in [
            (json!({"model": "model"}), "invalid Dynamo tensor request"),
            (request("other"), "request model 'other' does not match"),
        ] {
            let error = match engine.generate(request, context()).await {
                Ok(_) => panic!("accepted an invalid request"),
                Err(error) => error,
            };
            assert_eq!(
                error.error_type(),
                ErrorType::Backend(BackendError::InvalidArgument)
            );
            assert!(error.to_string().contains(expected), "{error}");
        }
    }

    #[tokio::test]
    async fn generate_round_trips_through_triton() {
        let (client, server) = triton(vec![
            infer_reply("OUTPUT0", "INT32", &[1], 9_i32.to_le_bytes().to_vec()),
            Reply::json("400 Bad Request", json!({"error": "unexpected shape"})),
        ])
        .await;
        let engine = engine(client);

        let mut stream = engine.generate(request("model"), context()).await.unwrap();
        let response: NvCreateTensorResponse =
            serde_json::from_value(stream.next().await.unwrap().unwrap()).unwrap();
        assert!(stream.next().await.is_none());
        assert_eq!(response.tensors[0].metadata.name, "OUTPUT0");
        assert_eq!(response.tensors[0].data, FlattenTensor::Int32(vec![9]));

        // A Triton failure is yielded on the stream with its mapped error type.
        let mut stream = engine.generate(request("model"), context()).await.unwrap();
        let error = stream.next().await.unwrap().unwrap_err();
        assert!(stream.next().await.is_none());
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::InvalidArgument)
        );
        assert!(error.to_string().contains("unexpected shape"), "{error}");

        let requests = server.await.unwrap();
        assert_eq!(requests[1].0, "POST /triton/v2/models/model/infer HTTP/1.1");
        let header = String::from_utf8_lossy(&requests[1].1);
        assert!(header.contains(r#""id":"request-1""#), "{header}");
        assert!(requests[1].1.ends_with(&5_i32.to_le_bytes()));
    }

    #[tokio::test]
    async fn generate_ends_without_a_response_when_stopped() {
        let (client, server) = triton(vec![infer_reply(
            "OUTPUT0",
            "INT32",
            &[1],
            9_i32.to_le_bytes().to_vec(),
        )])
        .await;
        let engine = engine(client);
        let context = dynamo_backend_common::testing::mock_context();
        context.stop_generating();
        let mut stream = engine
            .generate(request("model"), GenerateContext::new(context, None))
            .await
            .unwrap();
        assert!(stream.next().await.is_none());
        server.abort();
    }
}
