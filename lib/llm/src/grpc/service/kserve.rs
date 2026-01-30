// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;
use std::sync::Arc;

use crate::grpc::service::kserve::inference::DataType;
use crate::grpc::service::kserve::inference::ModelInput;
use crate::grpc::service::kserve::inference::ModelOutput;
use crate::http::service::Metrics;
use crate::http::service::service_v2 as http_service;

use crate::discovery::ModelManager;
use crate::local_model::runtime_config::ModelRuntimeConfig;
use crate::protocols::tensor::TensorModelConfig;
use crate::protocols::tensor::{NvCreateTensorRequest, NvCreateTensorResponse};
use crate::request_template::RequestTemplate;
use anyhow::Result;
use derive_builder::Builder;
use futures::pin_mut;
use tokio::task::JoinHandle;
use tokio_stream::{Stream, StreamExt};
use tokio_util::sync::CancellationToken;

use crate::grpc::service::openai::completion_response_stream;
use crate::grpc::service::tensor::{ExtendedNvCreateTensorResponse, tensor_response_stream};
use std::convert::{TryFrom, TryInto};
use tonic::{Request, Response, Status, transport::Server};

use crate::protocols::openai::completions::{
    NvCreateCompletionRequest, NvCreateCompletionResponse,
};

pub mod inference {
    tonic::include_proto!("inference");
}
use inference::grpc_inference_service_server::{GrpcInferenceService, GrpcInferenceServiceServer};
use inference::{
    ModelConfig, ModelConfigRequest, ModelConfigResponse, ModelInferRequest, ModelInferResponse,
    ModelMetadataRequest, ModelMetadataResponse, ModelStreamInferResponse,
};

use prost::Message;

/// gRPC service state - shares metrics with HTTP service for unified metrics collection
pub struct State {
    metrics: Arc<Metrics>,
    manager: Arc<ModelManager>,
}

#[derive(Default, Builder)]
#[builder(
    pattern = "owned",
    build_fn(private, name = "build_internal"),
    name = "StateBuilder",
    vis = "pub"
)]
pub(crate) struct StateConfig {
    #[builder(default, setter(strip_option))]
    metrics: Option<Arc<Metrics>>,
    #[builder(default, setter(strip_option))]
    manager: Option<Arc<ModelManager>>,
}

impl State {
    pub fn builder() -> StateBuilder {
        StateBuilder::default()
    }

    /// Get the Prometheus [`Metrics`] object which tracks request counts and inflight requests
    pub fn metrics_clone(&self) -> Arc<Metrics> {
        self.metrics.clone()
    }

    pub fn manager(&self) -> &ModelManager {
        Arc::as_ref(&self.manager)
    }

    pub fn manager_clone(&self) -> Arc<ModelManager> {
        self.manager.clone()
    }

    fn is_tensor_model(&self, model: &String) -> bool {
        self.manager.list_tensor_models().contains(model)
    }
}

impl StateBuilder {
    pub fn build(self) -> Result<State, anyhow::Error> {
        let config = self.build_internal()?;

        Ok(State {
            manager: config
                .manager
                .unwrap_or_else(|| Arc::new(ModelManager::new())),
            metrics: config
                .metrics
                .unwrap_or_else(|| Arc::new(Metrics::default())),
        })
    }
}

#[derive(Clone)]
pub struct KserveService {
    // The state we share with every request handler
    state: Arc<State>,

    // HTTP service for metrics endpoint
    http_service: http_service::HttpService,

    port: u16,
    host: String,
    request_template: Option<RequestTemplate>,
}

#[derive(Clone, Builder)]
#[builder(pattern = "owned", build_fn(private, name = "build_internal"))]
pub struct KserveServiceConfig {
    #[builder(default = "8787")]
    port: u16,

    #[builder(setter(into), default = "String::from(\"0.0.0.0\")")]
    host: String,

    #[builder(default = "None")]
    request_template: Option<RequestTemplate>,

    #[builder(default = "8788")]
    http_metrics_port: u16,

    #[builder(setter(into), default = "String::from(\"0.0.0.0\")")]
    http_metrics_host: String,
}

impl KserveService {
    pub fn builder() -> KserveServiceConfigBuilder {
        KserveServiceConfigBuilder::default()
    }

    pub fn state_clone(&self) -> Arc<State> {
        self.state.clone()
    }

    pub fn state(&self) -> &State {
        Arc::as_ref(&self.state)
    }

    pub fn model_manager(&self) -> &ModelManager {
        self.state().manager()
    }

    pub fn http_service(&self) -> &http_service::HttpService {
        &self.http_service
    }

    pub async fn spawn(&self, cancel_token: CancellationToken) -> JoinHandle<Result<()>> {
        let this = self.clone();
        tokio::spawn(async move { this.run(cancel_token).await })
    }

    pub async fn run(&self, cancel_token: CancellationToken) -> Result<()> {
        let address = format!("{}:{}", self.host, self.port);
        tracing::info!(address, "Starting KServe gRPC service on: {address}");

        let observer = cancel_token.child_token();
        Server::builder()
            .add_service(GrpcInferenceServiceServer::new(self.clone()))
            .serve_with_shutdown(address.parse()?, observer.cancelled_owned())
            .await
            .inspect_err(|_| cancel_token.cancel())?;

        Ok(())
    }
}

impl KserveServiceConfigBuilder {
    pub fn build(self) -> Result<KserveService, anyhow::Error> {
        let config: KserveServiceConfig = self.build_internal()?;

        // Create HTTP service with only non-inference endpoints (metrics, health, models list)
        // This provides the metrics endpoint and shared metrics object
        let http_service = http_service::HttpService::builder()
            .port(config.http_metrics_port)
            .host(config.http_metrics_host.clone())
            // Disable all inference endpoints - only use for metrics/health
            .enable_chat_endpoints(false)
            .enable_cmpl_endpoints(false)
            .enable_embeddings_endpoints(false)
            .enable_responses_endpoints(false)
            .build()?;

        // Share the HTTP service's model manager and metrics object with gRPC state
        let state = Arc::new(
            State::builder()
                .manager(http_service.state().manager_clone())
                .metrics(http_service.state().metrics_clone())
                .build()?,
        );

        Ok(KserveService {
            state,
            http_service,
            port: config.port,
            host: config.host,
            request_template: config.request_template,
        })
    }

    pub fn with_request_template(mut self, request_template: Option<RequestTemplate>) -> Self {
        self.request_template = Some(request_template);
        self
    }
}

#[allow(clippy::large_enum_variant)]
enum Config {
    Dynamo(TensorModelConfig),
    Triton(ModelConfig),
}

impl Config {
    fn from_runtime_config(runtime_config: &ModelRuntimeConfig) -> Result<Config, anyhow::Error> {
        if let Some(tensor_model_config) = runtime_config.tensor_model_config.as_ref() {
            if let Some(triton_model_config) = tensor_model_config.triton_model_config.as_ref() {
                let model_config = ModelConfig::decode(triton_model_config.as_slice())?;
                Ok(Config::Triton(model_config))
            } else {
                Ok(Config::Dynamo(tensor_model_config.clone()))
            }
        } else {
            Err(anyhow::anyhow!("no model config is provided"))
        }
    }
}

#[tonic::async_trait]
impl GrpcInferenceService for KserveService {
    async fn model_infer(
        &self,
        request: Request<ModelInferRequest>,
    ) -> Result<Response<ModelInferResponse>, Status> {
        let model = request.get_ref().model_name.clone();
        let request = request.into_inner();
        let request_id = request.id.clone();

        // [gluo TODO] refactor to reuse code, inference logic is largely the same
        if self.state().is_tensor_model(&model) {
            let set_raw_output_contents = !request.raw_input_contents.is_empty();
            let tensor_request: NvCreateTensorRequest = NvCreateTensorRequest::try_from(request)
                .map_err(|e| Status::invalid_argument(format!("Failed to parse request: {}", e)))?;

            let stream = tensor_response_stream(self.state_clone(), tensor_request, false).await?;

            let tensor_response = ExtendedNvCreateTensorResponse {
                response: NvCreateTensorResponse::from_annotated_stream(stream)
                    .await
                    .map_err(|e| {
                        tracing::error!("Failed to fold completions stream: {:?}", e);
                        Status::internal(format!("Failed to fold completions stream: {}", e))
                    })?,
                set_raw_output_contents,
            };

            let mut reply: ModelInferResponse = tensor_response.try_into().map_err(|e| {
                Status::invalid_argument(format!("Failed to parse response: {}", e))
            })?;
            reply.id = request_id;

            return Ok(Response::new(reply));
        }

        // [gluo FIXME] check model existence first, otherwise the true error
        // is masked by "Failed to parse request" below.
        // Fallback handling by assuming the model is OpenAI Completions model
        let mut completion_request: NvCreateCompletionRequest = request
            .try_into()
            .map_err(|e| Status::invalid_argument(format!("Failed to parse request: {}", e)))?;

        if completion_request.inner.stream.unwrap_or(false) {
            // return error that streaming is not supported
            return Err(Status::invalid_argument(
                "Streaming is not supported for this endpoint",
            ));
        }

        // Apply template values if present
        if let Some(template) = self.request_template.as_ref() {
            if completion_request.inner.model.is_empty() {
                completion_request.inner.model = template.model.clone();
            }
            if completion_request.inner.temperature.unwrap_or(0.0) == 0.0 {
                completion_request.inner.temperature = Some(template.temperature);
            }
            if completion_request.inner.max_tokens.unwrap_or(0) == 0 {
                completion_request.inner.max_tokens = Some(template.max_completion_tokens);
            }
        }

        let model = completion_request.inner.model.clone();
        let parsing_options = self.state.manager.get_parsing_options(&model);

        let stream = completion_response_stream(self.state_clone(), completion_request).await?;

        let completion_response =
            NvCreateCompletionResponse::from_annotated_stream(stream, parsing_options)
                .await
                .map_err(|e| {
                    tracing::error!("Failed to fold completions stream: {:?}", e);
                    Status::internal(format!("Failed to fold completions stream: {}", e))
                })?;

        let mut reply: ModelInferResponse = completion_response
            .try_into()
            .map_err(|e| Status::invalid_argument(format!("Failed to parse response: {}", e)))?;
        reply.id = request_id;

        Ok(Response::new(reply))
    }

    type ModelStreamInferStream =
        Pin<Box<dyn Stream<Item = Result<ModelStreamInferResponse, Status>> + Send + 'static>>;

    async fn model_stream_infer(
        &self,
        request: Request<tonic::Streaming<ModelInferRequest>>,
    ) -> Result<Response<Self::ModelStreamInferStream>, Status> {
        let mut request_stream = request.into_inner();
        let state = self.state_clone();
        let template = self.request_template.clone();
        let output = async_stream::try_stream! {
            // [gluo FIXME] should be able to demux request / response streaming
            // await requests in a separate task until cancellation / completion,
            // and passing AsyncEngineStream for each request to the response stream
            // which will be collectively polling.
            while let Some(request) = request_stream.next().await {
                let request = match request {
                    Err(e) => {
                        tracing::error!("Unexpected gRPC failed to read request: {}", e);
                        yield ModelStreamInferResponse {
                            error_message: e.to_string(),
                            infer_response: None
                        };
                        continue;
                    }
                    Ok(request) => {
                        request
                    }
                };

                let model = request.model_name.clone();

                // [gluo TODO] refactor to reuse code, inference logic is largely the same
                if state.is_tensor_model(&model) {
                    // Must keep track of 'request_id' which will be returned in corresponding response
                    let request_id = request.id.clone();
                    let set_raw_output_contents = !request.raw_input_contents.is_empty();
                    let tensor_request: NvCreateTensorRequest = request.try_into().map_err(|e| {
                        Status::invalid_argument(format!("Failed to parse request: {}", e))
                    })?;

                    let stream = tensor_response_stream(state.clone(), tensor_request, true).await?;

                    pin_mut!(stream);
                    while let Some(delta) = stream.next().await {
                        let response = match delta.ok() {
                            Err(e) => {
                                yield ModelStreamInferResponse {
                                    error_message: e.to_string(),
                                    infer_response: None
                                };
                                continue;
                            }
                            Ok(response) => response,
                        };
                        match response.data {
                            Some(data) => {
                                let data = ExtendedNvCreateTensorResponse {response: data,
                                    set_raw_output_contents,
                                };
                                let mut reply = ModelStreamInferResponse::try_from(data).map_err(|e| {
                                    Status::invalid_argument(format!("Failed to parse response: {}", e))
                                })?;
                                if reply.infer_response.is_some() {
                                    reply.infer_response.as_mut().unwrap().id = request_id.clone();
                                }
                                yield reply;
                            },
                            None => {
                                // Skip if no data is present, the response is for annotation
                            },
                        }
                    }
                    continue;
                }

                // Fallback handling by assuming the model is OpenAI Completions model
                // Must keep track of 'request_id' which will be returned in corresponding response
                let request_id = request.id.clone();
                let mut completion_request: NvCreateCompletionRequest = request.try_into().map_err(|e| {
                    Status::invalid_argument(format!("Failed to parse request: {}", e))
                })?;

                // Apply template values if present
                if let Some(template) = &template {
                    if completion_request.inner.model.is_empty() {
                        completion_request.inner.model = template.model.clone();
                    }
                    if completion_request.inner.temperature.unwrap_or(0.0) == 0.0 {
                        completion_request.inner.temperature = Some(template.temperature);
                    }
                    if completion_request.inner.max_tokens.unwrap_or(0) == 0 {
                        completion_request.inner.max_tokens = Some(template.max_completion_tokens);
                    }
                }

                let model = completion_request.inner.model.clone();
                let parsing_options = state.manager.get_parsing_options(&model);

                let streaming = completion_request.inner.stream.unwrap_or(false);

                let stream = completion_response_stream(state.clone(), completion_request).await?;

                if streaming {
                    pin_mut!(stream);
                    while let Some(delta) = stream.next().await {
                        let response = match delta.ok() {
                            Err(e) => {
                                yield ModelStreamInferResponse {
                                    error_message: e.to_string(),
                                    infer_response: None
                                };
                                continue;
                            }
                            Ok(response) => response,
                        };
                        match response.data {
                            Some(data) => {
                                let mut reply = ModelStreamInferResponse::try_from(data).map_err(|e| {
                                    Status::invalid_argument(format!("Failed to parse response: {}", e))
                                })?;
                                if reply.infer_response.is_some() {
                                    reply.infer_response.as_mut().unwrap().id = request_id.clone();
                                }
                                yield reply;
                            },
                            None => {
                                // Skip if no data is present, the response is for annotation
                            },
                        }
                    }
                } else {
                    let completion_response = NvCreateCompletionResponse::from_annotated_stream(stream, parsing_options)
                        .await
                        .map_err(|e| {
                            tracing::error!(
                                "Failed to fold completions stream: {:?}",
                                e
                            );
                            Status::internal(format!("Failed to fold completions stream: {}", e))
                        })?;

                    let mut response: ModelStreamInferResponse = completion_response.try_into().map_err(|e| {
                        Status::invalid_argument(format!("Failed to parse response: {}", e))
                    })?;
                    if response.infer_response.is_some() {
                        response.infer_response.as_mut().unwrap().id = request_id.clone();
                    }
                    yield response;
                }
            }
        };

        Ok(Response::new(
            Box::pin(output) as Self::ModelStreamInferStream
        ))
    }

    async fn model_metadata(
        &self,
        request: Request<ModelMetadataRequest>,
    ) -> Result<Response<ModelMetadataResponse>, Status> {
        let cards = self.state.manager().get_model_cards();
        let request_model_name = &request.into_inner().name;
        if let Some(card) = cards
            .into_iter()
            .find(|card| request_model_name == &card.display_name)
        {
            if card.model_type.supports_tensor() {
                let config = Config::from_runtime_config(&card.runtime_config).map_err(|e| {
                    Status::invalid_argument(format!(
                        "Model '{}' has type Tensor but: {}",
                        request_model_name, e
                    ))
                })?;
                match config {
                    Config::Triton(model_config) => {
                        return Ok(Response::new(ModelMetadataResponse {
                            name: model_config.name,
                            versions: vec!["1".to_string()],
                            platform: model_config.platform,
                            inputs: model_config
                                .input
                                .iter()
                                .map(|input| inference::model_metadata_response::TensorMetadata {
                                    name: input.name.clone(),
                                    datatype: match inference::DataType::try_from(input.data_type) {
                                        Ok(dt) => dt.as_str_name().to_string(),
                                        Err(_) => "TYPE_INVALID".to_string(),
                                    },
                                    shape: input.dims.clone(),
                                })
                                .collect(),
                            outputs: model_config
                                .output
                                .iter()
                                .map(
                                    |output| inference::model_metadata_response::TensorMetadata {
                                        name: output.name.clone(),
                                        datatype: match inference::DataType::try_from(
                                            output.data_type,
                                        ) {
                                            Ok(dt) => dt.as_str_name().to_string(),
                                            Err(_) => "TYPE_INVALID".to_string(),
                                        },
                                        shape: output.dims.clone(),
                                    },
                                )
                                .collect(),
                        }));
                    }
                    Config::Dynamo(model_config) => {
                        return Ok(Response::new(ModelMetadataResponse {
                            name: model_config.name.clone(),
                            versions: vec!["1".to_string()],
                            platform: "dynamo".to_string(),
                            inputs: model_config
                                .inputs
                                .iter()
                                .map(|input| inference::model_metadata_response::TensorMetadata {
                                    name: input.name.clone(),
                                    datatype: input.data_type.to_string(),
                                    shape: input.shape.clone(),
                                })
                                .collect(),
                            outputs: model_config
                                .outputs
                                .iter()
                                .map(
                                    |output| inference::model_metadata_response::TensorMetadata {
                                        name: output.name.clone(),
                                        datatype: output.data_type.to_string(),
                                        shape: output.shape.clone(),
                                    },
                                )
                                .collect(),
                        }));
                    }
                }
            } else if card.model_type.supports_completions() {
                return Ok(Response::new(ModelMetadataResponse {
                    name: card.display_name,
                    versions: vec!["1".to_string()],
                    platform: "dynamo".to_string(),
                    inputs: vec![
                        inference::model_metadata_response::TensorMetadata {
                            name: "text_input".to_string(),
                            datatype: "BYTES".to_string(),
                            shape: vec![1],
                        },
                        inference::model_metadata_response::TensorMetadata {
                            name: "streaming".to_string(),
                            datatype: "BOOL".to_string(),
                            shape: vec![1],
                        },
                    ],
                    outputs: vec![
                        inference::model_metadata_response::TensorMetadata {
                            name: "text_output".to_string(),
                            datatype: "BYTES".to_string(),
                            shape: vec![-1],
                        },
                        inference::model_metadata_response::TensorMetadata {
                            name: "finish_reason".to_string(),
                            datatype: "BYTES".to_string(),
                            shape: vec![-1],
                        },
                    ],
                }));
            }
        }
        Err(Status::not_found(format!(
            "Model '{}' not found",
            request_model_name
        )))
    }

    async fn model_config(
        &self,
        request: Request<ModelConfigRequest>,
    ) -> Result<Response<ModelConfigResponse>, Status> {
        let cards = self.state.manager().get_model_cards();
        let request_model_name = &request.into_inner().name;
        if let Some(card) = cards
            .into_iter()
            .find(|card| request_model_name == &card.display_name)
        {
            if card.model_type.supports_tensor() {
                let config = Config::from_runtime_config(&card.runtime_config).map_err(|e| {
                    Status::invalid_argument(format!(
                        "Model '{}' has type Tensor but: {}",
                        request_model_name, e
                    ))
                })?;
                match config {
                    Config::Triton(model_config) => {
                        return Ok(Response::new(ModelConfigResponse {
                            config: Some(model_config),
                        }));
                    }
                    Config::Dynamo(tensor_model_config) => {
                        let model_config = ModelConfig {
                            name: tensor_model_config.name.clone(),
                            platform: "dynamo".to_string(),
                            backend: "dynamo".to_string(),
                            input: tensor_model_config
                                .inputs
                                .iter()
                                .map(|input| ModelInput {
                                    name: input.name.clone(),
                                    data_type: input.data_type.to_kserve(),
                                    dims: input.shape.clone(),
                                    ..Default::default()
                                })
                                .collect(),
                            output: tensor_model_config
                                .outputs
                                .iter()
                                .map(|output| ModelOutput {
                                    name: output.name.clone(),
                                    data_type: output.data_type.to_kserve(),
                                    dims: output.shape.clone(),
                                    ..Default::default()
                                })
                                .collect(),
                            ..Default::default()
                        };
                        return Ok(Response::new(ModelConfigResponse {
                            config: Some(model_config.clone()),
                        }));
                    }
                }
            } else if card.model_type.supports_completions() {
                let config = ModelConfig {
                    name: card.display_name,
                    platform: "dynamo".to_string(),
                    backend: "dynamo".to_string(),
                    input: vec![
                        ModelInput {
                            name: "text_input".to_string(),
                            data_type: DataType::TypeString as i32,
                            dims: vec![1],
                            ..Default::default()
                        },
                        ModelInput {
                            name: "streaming".to_string(),
                            data_type: DataType::TypeBool as i32,
                            dims: vec![1],
                            optional: true,
                            ..Default::default()
                        },
                    ],
                    output: vec![
                        ModelOutput {
                            name: "text_output".to_string(),
                            data_type: DataType::TypeString as i32,
                            dims: vec![-1],
                            ..Default::default()
                        },
                        ModelOutput {
                            name: "finish_reason".to_string(),
                            data_type: DataType::TypeString as i32,
                            dims: vec![-1],
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                };
                return Ok(Response::new(ModelConfigResponse {
                    config: Some(config),
                }));
            }
        }
        Err(Status::not_found(format!(
            "Model '{}' not found",
            request_model_name
        )))
    }

    async fn server_live(
        &self,
        _request: Request<inference::ServerLiveRequest>,
    ) -> Result<Response<inference::ServerLiveResponse>, Status> {
        // server is live if we can respond
        Ok(Response::new(inference::ServerLiveResponse { live: true }))
    }

    async fn server_ready(
        &self,
        _request: Request<inference::ServerReadyRequest>,
    ) -> Result<Response<inference::ServerReadyResponse>, Status> {
        let has_models = !self.state.manager().get_model_cards().is_empty();
        Ok(Response::new(inference::ServerReadyResponse {
            ready: has_models,
        }))
    }

    async fn model_ready(
        &self,
        request: Request<inference::ModelReadyRequest>,
    ) -> Result<Response<inference::ModelReadyResponse>, Status> {
        let request_model_name = &request.into_inner().name;
        let is_ready = self
            .state
            .manager()
            .get_model_cards()
            .into_iter()
            .any(|card| request_model_name == &card.display_name);
        Ok(Response::new(inference::ModelReadyResponse {
            ready: is_ready,
        }))
    }
}
