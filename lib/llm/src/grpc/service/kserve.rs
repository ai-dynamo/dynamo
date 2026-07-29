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
use crate::protocols::tensor::TensorModelConfig;
use crate::protocols::tensor::{NvCreateTensorRequest, NvCreateTensorResponse};
use crate::request_template::{RequestTemplate, resolve_request_model};
use anyhow::Result;
use derive_builder::Builder;
use dynamo_runtime::config::environment_names::llm::metrics as env_metrics;
use futures::pin_mut;
use tokio::task::JoinHandle;
use tokio_stream::{Stream, StreamExt};
use tokio_util::sync::CancellationToken;

/// Optional HTTP/2 window size configuration from environment variables.
///
/// # Environment Variables
///
/// - `DYN_GRPC_INITIAL_CONNECTION_WINDOW_SIZE`: HTTP/2 connection window size in bytes
/// - `DYN_GRPC_INITIAL_STREAM_WINDOW_SIZE`: HTTP/2 per-stream window size in bytes
///
/// If set, these override tonic defaults. If not set, tonic defaults are used.
#[derive(Debug, Clone, Default)]
pub struct GrpcTuningConfig {
    /// HTTP/2 connection-level flow control window size in bytes.
    /// If None, uses tonic default.
    pub initial_connection_window_size: Option<u32>,

    /// HTTP/2 stream-level flow control window size in bytes.
    /// If None, uses tonic default.
    pub initial_stream_window_size: Option<u32>,
}

impl GrpcTuningConfig {
    /// Create configuration from environment variables.
    ///
    /// Reads `DYN_GRPC_INITIAL_CONNECTION_WINDOW_SIZE` and `DYN_GRPC_INITIAL_STREAM_WINDOW_SIZE`.
    /// If not set, the values remain None and tonic defaults are used.
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(val) = std::env::var("DYN_GRPC_INITIAL_CONNECTION_WINDOW_SIZE")
            && let Ok(size) = val.parse::<u32>()
        {
            config.initial_connection_window_size = Some(size);
        }

        if let Ok(val) = std::env::var("DYN_GRPC_INITIAL_STREAM_WINDOW_SIZE")
            && let Ok(size) = val.parse::<u32>()
        {
            config.initial_stream_window_size = Some(size);
        }

        config
    }
}

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

    fn is_completions_model(&self, model: &String) -> bool {
        self.manager.list_completions_models().contains(model)
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

    // gRPC server tuning configuration
    grpc_tuning: GrpcTuningConfig,
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

    #[builder(default = "std::env::var(env_metrics::DYN_METRICS_PREFIX).ok()")]
    metrics_prefix: Option<String>,

    #[builder(setter(into), default = "String::from(\"0.0.0.0\")")]
    http_metrics_host: String,

    #[builder(default = "None")]
    http_cancel_token: Option<CancellationToken>,

    /// gRPC server tuning configuration.
    /// Default: GrpcTuningConfig::from_env() - reads from environment variables with fallback to defaults.
    #[builder(default = "GrpcTuningConfig::from_env()")]
    grpc_tuning: GrpcTuningConfig,
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

        let tuning = &self.grpc_tuning;

        // Log tuning settings if configured via environment variables
        if tuning.initial_connection_window_size.is_some()
            || tuning.initial_stream_window_size.is_some()
        {
            tracing::info!(
                "gRPC tuning: connection_window={:?}, stream_window={:?}",
                tuning.initial_connection_window_size,
                tuning.initial_stream_window_size
            );
        }

        let observer = cancel_token.child_token();

        // Build server - only override window sizes if set via env vars
        let mut builder = Server::builder();

        if let Some(size) = tuning.initial_connection_window_size {
            builder = builder.initial_connection_window_size(size);
        }
        if let Some(size) = tuning.initial_stream_window_size {
            builder = builder.initial_stream_window_size(size);
        }

        builder
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
            .metrics_prefix(config.metrics_prefix)
            .cancel_token(config.http_cancel_token)
            // Disable all inference endpoints - only use for metrics/health
            .enable_chat_endpoints(false)
            .enable_cmpl_endpoints(false)
            .enable_embeddings_endpoints(false)
            .enable_responses_endpoints(false)
            .enable_anthropic_endpoints(false)
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
            grpc_tuning: config.grpc_tuning,
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
    fn from_tensor_model_config(
        tensor_model_config: Option<&TensorModelConfig>,
    ) -> Result<Config, anyhow::Error> {
        if let Some(tensor_model_config) = tensor_model_config {
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

/// Apply a request template's defaults to a completions request, filling in the
/// model, temperature, and max-tokens fields only when the request leaves them
/// unset. Shared by the unary and streaming inference handlers so the merge
/// stays consistent between them.
fn apply_request_template(
    completion_request: &mut NvCreateCompletionRequest,
    template: Option<&RequestTemplate>,
) {
    if let Some(template) = template {
        if completion_request.inner.model.is_empty() {
            completion_request.inner.model = template.model.clone();
        }
        // Only fill truly-unset (`None`) fields: an explicit `temperature = 0.0`
        // (deterministic decoding) or `max_tokens = 0` is a deliberate caller
        // choice and must not be clobbered by the template. Mirrors the
        // `is_none()` checks used elsewhere (e.g. the Responses API handler).
        if completion_request.inner.temperature.is_none() {
            completion_request.inner.temperature = Some(template.temperature);
        }
        if completion_request.inner.max_tokens.is_none() {
            completion_request.inner.max_tokens = Some(template.max_completion_tokens);
        }
    }
}

/// A `ModelInferRequest` resolved to the concrete inference flavor it targets.
///
/// Centralises the tensor-vs-completions dispatch that both `model_infer` and
/// `model_stream_infer` need. Constructing an [`InferRequest`] also performs the
/// model-existence check up front (see [`InferRequest::from_model_infer`]) so a
/// missing model surfaces as a clear `not_found` status instead of being masked
/// by a downstream "Failed to parse request" parse error.
#[allow(clippy::large_enum_variant)]
enum InferRequest {
    /// Tensor model request. The boolean records whether the response should
    /// populate `raw_output_contents` (mirrors the input's `raw_input_contents`).
    Tensor {
        request: NvCreateTensorRequest,
        set_raw_output_contents: bool,
    },
    /// OpenAI Completions model request, with any request-template defaults applied.
    Completions(NvCreateCompletionRequest),
}

impl InferRequest {
    /// Dispatch a raw `ModelInferRequest` to the correct inference flavor.
    ///
    /// Tensor models are routed to [`InferRequest::Tensor`]. Otherwise the model
    /// must be a registered completions model: the existence check uses the
    /// template-resolved model name (so an empty request model that a template
    /// fills in is validated against the resolved name) and returns
    /// [`Status::not_found`] before any `try_into` parse step. This avoids
    /// masking a missing model behind a misleading parse error.
    #[allow(clippy::result_large_err)]
    fn from_model_infer(
        state: &State,
        request: ModelInferRequest,
        template: Option<&RequestTemplate>,
    ) -> Result<Self, Status> {
        let model = request.model_name.clone();

        if state.is_tensor_model(&model) {
            let set_raw_output_contents = !request.raw_input_contents.is_empty();
            let request = NvCreateTensorRequest::try_from(request)
                .map_err(|e| Status::invalid_argument(format!("Failed to parse request: {}", e)))?;
            return Ok(InferRequest::Tensor {
                request,
                set_raw_output_contents,
            });
        }

        // Not a tensor model: must be a registered completions model. Check
        // existence against the template-resolved model name *before* the
        // try_into parse below, otherwise a missing model is masked by a
        // misleading "Failed to parse request" error.
        let resolved_model = resolve_request_model(&model, template);
        if !state.is_completions_model(&resolved_model.to_string()) {
            return Err(Status::not_found(format!(
                "Model '{}' not found",
                resolved_model
            )));
        }

        let mut completion_request = NvCreateCompletionRequest::try_from(request)
            .map_err(|e| Status::invalid_argument(format!("Failed to parse request: {}", e)))?;
        apply_request_template(&mut completion_request, template);

        Ok(InferRequest::Completions(completion_request))
    }

    /// Run the request to completion and fold the stream into a single unary
    /// [`ModelInferResponse`]. Streaming completion requests are rejected here,
    /// matching the unary endpoint's contract.
    async fn unary_response(
        self,
        state: Arc<State>,
        metadata: &tonic::metadata::MetadataMap,
        request_id: String,
    ) -> Result<ModelInferResponse, Status> {
        let mut reply: ModelInferResponse = match self {
            InferRequest::Tensor {
                request,
                set_raw_output_contents,
            } => {
                let stream = tensor_response_stream(state, request, false, metadata).await?;
                let tensor_response = ExtendedNvCreateTensorResponse {
                    response: NvCreateTensorResponse::from_annotated_stream(stream)
                        .await
                        .map_err(|e| {
                            tracing::error!("Failed to fold completions stream: {:?}", e);
                            Status::internal(format!("Failed to fold completions stream: {}", e))
                        })?,
                    set_raw_output_contents,
                };
                tensor_response.try_into().map_err(|e| {
                    Status::invalid_argument(format!("Failed to parse response: {}", e))
                })?
            }
            InferRequest::Completions(completion_request) => {
                if completion_request.inner.stream.unwrap_or(false) {
                    return Err(Status::invalid_argument(
                        "Streaming is not supported for this endpoint",
                    ));
                }
                let (stream, parsing_options) =
                    completion_response_stream(state, completion_request, metadata).await?;
                let completion_response =
                    NvCreateCompletionResponse::from_annotated_stream(stream, parsing_options)
                        .await
                        .map_err(|e| {
                            tracing::error!("Failed to fold completions stream: {:?}", e);
                            Status::internal(format!("Failed to fold completions stream: {}", e))
                        })?;
                completion_response.try_into().map_err(|e| {
                    Status::invalid_argument(format!("Failed to parse response: {}", e))
                })?
            }
        };
        reply.id = request_id;
        Ok(reply)
    }

    /// Produce a stream of [`ModelStreamInferResponse`] for the streaming
    /// endpoint. Non-streaming completion requests are folded into a single
    /// response; streaming ones are forwarded delta-by-delta.
    async fn stream_response<'a>(
        self,
        state: Arc<State>,
        metadata: &'a tonic::metadata::MetadataMap,
        request_id: String,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<ModelStreamInferResponse, Status>> + Send + 'a>>,
        Status,
    > {
        match self {
            InferRequest::Tensor {
                request,
                set_raw_output_contents,
            } => {
                let stream = tensor_response_stream(state, request, true, metadata).await?;
                let output = async_stream::try_stream! {
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
                                let data = ExtendedNvCreateTensorResponse {
                                    response: data,
                                    set_raw_output_contents,
                                };
                                let mut reply = ModelStreamInferResponse::try_from(data).map_err(|e| {
                                    Status::invalid_argument(format!("Failed to parse response: {}", e))
                                })?;
                                if let Some(infer_response) = reply.infer_response.as_mut() {
                                    infer_response.id = request_id.clone();
                                }
                                yield reply;
                            },
                            None => {
                                // Skip if no data is present, the response is for annotation
                            },
                        }
                    }
                };
                Ok(Box::pin(output))
            }
            InferRequest::Completions(completion_request) => {
                let streaming = completion_request.inner.stream.unwrap_or(false);
                let (stream, parsing_options) =
                    completion_response_stream(state, completion_request, metadata).await?;
                let output = async_stream::try_stream! {
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
                                    if let Some(infer_response) = reply.infer_response.as_mut() {
                                        infer_response.id = request_id.clone();
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
                        if let Some(infer_response) = response.infer_response.as_mut() {
                            infer_response.id = request_id.clone();
                        }
                        yield response;
                    }
                };
                Ok(Box::pin(output))
            }
        }
    }
}

#[tonic::async_trait]
impl GrpcInferenceService for KserveService {
    async fn model_infer(
        &self,
        request: Request<ModelInferRequest>,
    ) -> Result<Response<ModelInferResponse>, Status> {
        let (metadata, _extensions, request) = request.into_parts();
        let request_id = request.id.clone();

        let infer_request =
            InferRequest::from_model_infer(self.state(), request, self.request_template.as_ref())?;
        let reply = infer_request
            .unary_response(self.state_clone(), &metadata, request_id)
            .await?;

        Ok(Response::new(reply))
    }

    type ModelStreamInferStream =
        Pin<Box<dyn Stream<Item = Result<ModelStreamInferResponse, Status>> + Send + 'static>>;

    async fn model_stream_infer(
        &self,
        request: Request<tonic::Streaming<ModelInferRequest>>,
    ) -> Result<Response<Self::ModelStreamInferStream>, Status> {
        let (metadata, _extensions, request_stream) = request.into_parts();
        let mut request_stream = request_stream;
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

                // Must keep track of 'request_id' which will be returned in corresponding response
                let request_id = request.id.clone();

                let infer_request = InferRequest::from_model_infer(
                    state.as_ref(),
                    request,
                    template.as_ref(),
                )?;

                let response_stream = infer_request
                    .stream_response(state.clone(), &metadata, request_id)
                    .await?;
                pin_mut!(response_stream);
                while let Some(response) = response_stream.next().await {
                    yield response?;
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
                let config = Config::from_tensor_model_config(card.tensor_model_config.as_ref())
                    .map_err(|e| {
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
                let config = Config::from_tensor_model_config(card.tensor_model_config.as_ref())
                    .map_err(|e| {
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
        // Only report ready when at least one model has a WorkerSet that can
        // actually serve an inference request — a registered ModelDeploymentCard
        // is not enough (the WorkerSet is wired up afterwards).
        Ok(Response::new(inference::ServerReadyResponse {
            ready: self.state.manager().has_any_ready_model(),
        }))
    }

    async fn model_ready(
        &self,
        request: Request<inference::ModelReadyRequest>,
    ) -> Result<Response<inference::ModelReadyResponse>, Status> {
        let request_model_name = &request.into_inner().name;
        Ok(Response::new(inference::ModelReadyResponse {
            ready: self
                .state
                .manager()
                .is_model_ready_to_serve(request_model_name),
        }))
    }
}

#[cfg(test)]
mod readiness_gate_tests {
    use super::inference::grpc_inference_service_server::GrpcInferenceService;
    use super::inference::{ModelReadyRequest, ServerReadyRequest};
    use super::*;
    use crate::discovery::WorkerSet;
    use crate::model_card::ModelDeploymentCard;
    use crate::worker_type::WorkerType;
    use tonic::Request;

    /// A WorkerSet with an explicit role/needs, a live worker, and a chat engine
    /// attached. `namespace` is the WorkerSet's own namespace (sets sharing it
    /// form one deployment); the caller passes a distinct DashMap key.
    fn chat_ws_with_role(
        namespace: &str,
        mdcsum: &str,
        worker_type: WorkerType,
        needs: Vec<Vec<WorkerType>>,
    ) -> WorkerSet {
        let mut card = ModelDeploymentCard::default();
        card.worker_type = Some(worker_type);
        card.needs = needs;
        // Watch receiver keeps its last value after the sender drops → count 1.
        let (_tx, rx) = tokio::sync::watch::channel(vec![1u64]);
        let mut ws = WorkerSet::new(namespace.to_string(), mdcsum.to_string(), card);
        ws.set_instance_watcher(rx);
        ws.chat_engine = Some(Arc::new(crate::engines::StreamingEngineAdapter::new(
            crate::engines::make_echo_engine(),
        )));
        ws
    }

    fn model_ready_req(name: &str) -> Request<ModelReadyRequest> {
        Request::new(ModelReadyRequest {
            name: name.to_string(),
            version: String::new(),
        })
    }

    /// KServe `ModelReady` / `ServerReady` must reflect the namespace
    /// completeness gate: a live decode-only WorkerSet with a chat engine but no
    /// prefill peer reports NOT ready (even though an engine is attached), and
    /// flips to ready once the prefill peer joins the same namespace. Drives the
    /// real gRPC handlers end to end through `is_model_ready_to_serve`.
    #[tokio::test]
    async fn model_ready_reflects_worker_set_completeness() {
        let svc = KserveService::builder().build().unwrap();
        let mm = svc.model_manager();

        // Incomplete deployment: decode-only (needs a prefill peer), live + chat engine.
        mm.add_worker_set(
            "llama",
            "dep1",
            chat_ws_with_role(
                "dep1",
                "mdc-d",
                WorkerType::Decode,
                vec![vec![WorkerType::Prefill]],
            ),
        );

        assert!(
            !svc.model_ready(model_ready_req("llama"))
                .await
                .unwrap()
                .get_ref()
                .ready,
            "decode-only (missing prefill) must report KServe ModelReady=false"
        );
        assert!(
            !svc.server_ready(Request::new(ServerReadyRequest {}))
                .await
                .unwrap()
                .get_ref()
                .ready,
            "no complete worker set → ServerReady=false"
        );

        // Prefill peer joins the SAME namespace (distinct DashMap key) → complete.
        mm.add_worker_set(
            "llama",
            "dep1:prefill",
            chat_ws_with_role(
                "dep1",
                "mdc-p",
                WorkerType::Prefill,
                vec![vec![WorkerType::Decode]],
            ),
        );

        assert!(
            svc.model_ready(model_ready_req("llama"))
                .await
                .unwrap()
                .get_ref()
                .ready,
            "completing the worker set must flip KServe ModelReady=true"
        );
        assert!(
            svc.server_ready(Request::new(ServerReadyRequest {}))
                .await
                .unwrap()
                .get_ref()
                .ready,
            "a complete worker set → ServerReady=true"
        );
    }
}

#[cfg(test)]
mod infer_dispatch_tests {
    use super::*;
    use crate::protocols::openai::completions::NvCreateCompletionRequest;
    use inference::ModelInferRequest;
    use inference::model_infer_request::InferInputTensor;

    fn template(model: &str, temperature: f32, max_completion_tokens: u32) -> RequestTemplate {
        RequestTemplate {
            model: model.to_string(),
            temperature,
            max_completion_tokens,
        }
    }

    /// Build a minimal, valid Completions-shaped `ModelInferRequest` carrying a
    /// single `text_input` BYTES tensor for the given model name.
    fn completion_infer_request(model_name: &str) -> ModelInferRequest {
        ModelInferRequest {
            model_name: model_name.to_string(),
            inputs: vec![InferInputTensor {
                name: "text_input".to_string(),
                datatype: "BYTES".to_string(),
                shape: vec![1],
                contents: Some(inference::InferTensorContents {
                    bytes_contents: vec![b"hello".to_vec()],
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    /// An "empty" completions request (model unset, no temperature/max_tokens),
    /// built through the real `ModelInferRequest` conversion so every other
    /// field holds a valid default. `NvCreateCompletionRequest` itself does not
    /// implement `Default`.
    fn empty_completion_request() -> NvCreateCompletionRequest {
        NvCreateCompletionRequest::try_from(completion_infer_request(""))
            .expect("build empty completion request")
    }

    /// Register a no-op Completions engine for `model` so it shows up in
    /// `list_completions_models()`.
    fn register_completions_model(state: &State, model: &str) {
        let engine = Arc::new(crate::engines::StreamingEngineAdapter::new(
            crate::engines::make_echo_engine(),
        ));
        state
            .manager()
            .add_completions_model(model, "mdc-test", engine)
            .expect("register completions model");
    }

    /// (a) Template application: defaults fill only the fields the request left
    /// unset; explicitly-provided values are preserved.
    #[test]
    fn apply_request_template_fills_only_unset_fields() {
        let t = template("template-model", 0.7, 128);

        // Empty/zeroed request picks up all template defaults.
        let mut req = empty_completion_request();
        apply_request_template(&mut req, Some(&t));
        assert_eq!(req.inner.model, "template-model");
        assert_eq!(req.inner.temperature, Some(0.7));
        assert_eq!(req.inner.max_tokens, Some(128));

        // Caller-supplied values win over the template.
        let mut req = empty_completion_request();
        req.inner.model = "user-model".to_string();
        req.inner.temperature = Some(0.1);
        req.inner.max_tokens = Some(42);
        apply_request_template(&mut req, Some(&t));
        assert_eq!(req.inner.model, "user-model");
        assert_eq!(req.inner.temperature, Some(0.1));
        assert_eq!(req.inner.max_tokens, Some(42));

        // Explicit zero values are deliberate (deterministic decoding /
        // zero-length) and must be preserved, not treated as "unset".
        let mut req = empty_completion_request();
        req.inner.temperature = Some(0.0);
        req.inner.max_tokens = Some(0);
        apply_request_template(&mut req, Some(&t));
        assert_eq!(req.inner.temperature, Some(0.0));
        assert_eq!(req.inner.max_tokens, Some(0));

        // No template is a no-op.
        let mut req = empty_completion_request();
        apply_request_template(&mut req, None);
        assert_eq!(req.inner.model, "");
    }

    /// (b) Not-found error: an unregistered model surfaces as a clear
    /// `not_found` status naming the model, instead of a masked
    /// "Failed to parse request" parse error.
    #[test]
    fn unregistered_model_returns_not_found() {
        let svc = KserveService::builder().build().unwrap();
        let request = completion_infer_request("missing-model");

        let err = InferRequest::from_model_infer(svc.state(), request, None)
            .err()
            .expect("expected not_found error");
        assert_eq!(err.code(), tonic::Code::NotFound);
        assert!(
            err.message().contains("missing-model"),
            "error message should name the missing model, got: {}",
            err.message()
        );
    }

    /// (c) Template-resolved name edge case: with an empty request model the
    /// existence check (and any resulting error) must use the
    /// template-resolved model name.
    #[test]
    fn empty_model_resolves_via_template() {
        let svc = KserveService::builder().build().unwrap();

        // Template points at a registered completions model: empty request
        // model resolves to it and dispatches as a Completions request.
        register_completions_model(svc.state(), "template-model");
        let t = template("template-model", 0.0, 0);
        let request = completion_infer_request("");
        let infer = InferRequest::from_model_infer(svc.state(), request, Some(&t))
            .expect("template-resolved model should dispatch");
        assert!(matches!(infer, InferRequest::Completions(_)));

        // Template points at an *unregistered* model: the not-found error names
        // the resolved template model, not the empty request model.
        let t = template("template-missing", 0.0, 0);
        let request = completion_infer_request("");
        let err = InferRequest::from_model_infer(svc.state(), request, Some(&t))
            .err()
            .expect("expected not_found error");
        assert_eq!(err.code(), tonic::Code::NotFound);
        assert!(
            err.message().contains("template-missing"),
            "error message should name the template-resolved model, got: {}",
            err.message()
        );
    }
}
