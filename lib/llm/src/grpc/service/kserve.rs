// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;
use std::sync::Arc;

use crate::grpc::service::kserve::inference::DataType;
use crate::grpc::service::kserve::inference::ModelInput;
use crate::grpc::service::kserve::inference::ModelOutput;
use crate::http::service::Metrics;
use crate::http::service::metrics;

use crate::discovery::ModelManager;
use crate::request_template::RequestTemplate;
use anyhow::Result;
use derive_builder::Builder;
use dynamo_async_openai::types::{CompletionFinishReason, CreateCompletionRequest, Prompt};
use dynamo_runtime::transports::etcd;
use futures::pin_mut;
use tokio::task::JoinHandle;
use tokio_stream::{Stream, StreamExt};
use tokio_util::sync::CancellationToken;

use crate::grpc::service::openai::completion_response_stream;
use tonic::{Request, Response, Status, transport::Server};

use crate::protocols::openai::completions::{
    NvCreateCompletionRequest, NvCreateCompletionResponse,
};

pub mod inference {
    tonic::include_proto!("inference");
}
use inference::grpc_inference_service_server::{GrpcInferenceService, GrpcInferenceServiceServer};
use inference::{
    InferParameter, ModelConfig, ModelConfigRequest, ModelConfigResponse, ModelInferRequest,
    ModelInferResponse, ModelMetadataRequest, ModelMetadataResponse, ModelStreamInferResponse,
};

/// [gluo TODO] 'metrics' are for HTTP service and there is HTTP endpoint
/// for it as part of HTTP service. Should we always start HTTP service up
/// for non-inference?
pub struct State {
    metrics: Arc<Metrics>,
    manager: Arc<ModelManager>,
    etcd_client: Option<etcd::Client>,
}

impl State {
    pub fn new(manager: Arc<ModelManager>) -> Self {
        Self {
            manager,
            metrics: Arc::new(Metrics::default()),
            etcd_client: None,
        }
    }

    pub fn new_with_etcd(manager: Arc<ModelManager>, etcd_client: Option<etcd::Client>) -> Self {
        Self {
            manager,
            metrics: Arc::new(Metrics::default()),
            etcd_client,
        }
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

    pub fn etcd_client(&self) -> Option<&etcd::Client> {
        self.etcd_client.as_ref()
    }
}

#[derive(Clone)]
pub struct KserveService {
    // The state we share with every request handler
    state: Arc<State>,

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

    #[builder(default = "None")]
    etcd_client: Option<etcd::Client>,
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

        let model_manager = Arc::new(ModelManager::new());
        let state = Arc::new(State::new_with_etcd(model_manager, config.etcd_client));

        // enable prometheus metrics
        let registry = metrics::Registry::new();
        state.metrics_clone().register(&registry)?;

        Ok(KserveService {
            state,
            port: config.port,
            host: config.host,
            request_template: config.request_template,
        })
    }

    pub fn with_request_template(mut self, request_template: Option<RequestTemplate>) -> Self {
        self.request_template = Some(request_template);
        self
    }

    pub fn with_etcd_client(mut self, etcd_client: Option<etcd::Client>) -> Self {
        self.etcd_client = Some(etcd_client);
        self
    }
}

#[tonic::async_trait]
impl GrpcInferenceService for KserveService {
    async fn model_infer(
        &self,
        request: Request<ModelInferRequest>,
    ) -> Result<Response<ModelInferResponse>, Status> {
        let request = request.into_inner();
        let request_id = request.id.clone();
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
                    Status::internal("Failed to fold completions stream")
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
                // Must keep track of 'request_id' which will be returned in corresponding response
                let request_id: String;
                let mut completion_request: NvCreateCompletionRequest = match request {
                    Err(e) => {
                        tracing::error!("Unexpected gRPC failed to read request: {}", e);
                        yield ModelStreamInferResponse {
                            error_message: e.to_string(),
                            infer_response: None
                        };
                        continue;
                    }
                    Ok(request) => {
                        request_id = request.id.clone();
                        request.try_into().map_err(|e| {
                            Status::invalid_argument(format!("Failed to parse request: {}", e))
                        })?
                    }
                };

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
                    while let Some(response) = stream.next().await {
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
                            Status::internal("Failed to fold completions stream")
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
        let models = self.state.manager().list_completions_models();
        let request_model_name = &request.into_inner().name;
        if let Some(model_name) = models.into_iter().find(|n| request_model_name == n) {
            return Ok(Response::new(ModelMetadataResponse {
                name: model_name,
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
        Err(Status::not_found(format!(
            "Model '{}' not found",
            request_model_name
        )))
    }

    async fn model_config(
        &self,
        request: Request<ModelConfigRequest>,
    ) -> Result<Response<ModelConfigResponse>, Status> {
        let models = self.state.manager().list_completions_models();
        let request_model_name = &request.into_inner().name;
        if let Some(model_name) = models.into_iter().find(|n| request_model_name == n) {
            let config = ModelConfig {
                name: model_name,
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
        Err(Status::not_found(format!(
            "Model '{}' not found",
            request_model_name
        )))
    }
}

impl TryFrom<ModelInferRequest> for NvCreateCompletionRequest {
    type Error = Status;

    fn try_from(request: ModelInferRequest) -> Result<Self, Self::Error> {
        // Protocol requires if `raw_input_contents` is used to hold input data,
        // it must be used for all inputs.
        if !request.raw_input_contents.is_empty()
            && request.inputs.len() != request.raw_input_contents.len()
        {
            return Err(Status::invalid_argument(
                "`raw_input_contents` must be used for all inputs",
            ));
        }

        // iterate through inputs
        let mut text_input = None;
        let mut stream = false;
        for (idx, input) in request.inputs.iter().enumerate() {
            match input.name.as_str() {
                "text_input" => {
                    if input.datatype != "BYTES" {
                        return Err(Status::invalid_argument(format!(
                            "Expected 'text_input' to be of type BYTES for string input, got {:?}",
                            input.datatype
                        )));
                    }
                    if input.shape != vec![1] && input.shape != vec![1, 1] {
                        return Err(Status::invalid_argument(format!(
                            "Expected 'text_input' to have shape [1], got {:?}",
                            input.shape
                        )));
                    }
                    match &input.contents {
                        Some(content) => {
                            let bytes = &content.bytes_contents[0];
                            text_input = Some(String::from_utf8_lossy(bytes).to_string());
                        }
                        None => {
                            let raw_input =
                                request.raw_input_contents.get(idx).ok_or_else(|| {
                                    Status::invalid_argument("Missing raw input for 'text_input'")
                                })?;
                            if raw_input.len() < 4 {
                                return Err(Status::invalid_argument(
                                    "'text_input' raw input must be length-prefixed (>= 4 bytes)",
                                ));
                            }
                            // We restrict the 'text_input' only contain one element, only need to
                            // parse the first element. Skip first four bytes that is used to store
                            // the length of the input.
                            text_input = Some(String::from_utf8_lossy(&raw_input[4..]).to_string());
                        }
                    }
                }
                "streaming" | "stream" => {
                    if input.datatype != "BOOL" {
                        return Err(Status::invalid_argument(format!(
                            "Expected '{}' to be of type BOOL, got {:?}",
                            input.name, input.datatype
                        )));
                    }
                    if input.shape != vec![1] {
                        return Err(Status::invalid_argument(format!(
                            "Expected 'stream' to have shape [1], got {:?}",
                            input.shape
                        )));
                    }
                    match &input.contents {
                        Some(content) => {
                            stream = content.bool_contents[0];
                        }
                        None => {
                            let raw_input =
                                request.raw_input_contents.get(idx).ok_or_else(|| {
                                    Status::invalid_argument("Missing raw input for 'stream'")
                                })?;
                            if raw_input.is_empty() {
                                return Err(Status::invalid_argument(
                                    "'stream' raw input must contain at least one byte",
                                ));
                            }
                            stream = raw_input[0] != 0;
                        }
                    }
                }
                _ => {
                    return Err(Status::invalid_argument(format!(
                        "Invalid input name: {}, supported inputs are 'text_input', 'stream'",
                        input.name
                    )));
                }
            }
        }

        // return error if text_input is None
        let text_input = match text_input {
            Some(input) => input,
            None => {
                return Err(Status::invalid_argument(
                    "Missing required input: 'text_input'",
                ));
            }
        };

        Ok(NvCreateCompletionRequest {
            inner: CreateCompletionRequest {
                model: request.model_name,
                prompt: Prompt::String(text_input),
                stream: Some(stream),
                user: if request.id.is_empty() {
                    None
                } else {
                    Some(request.id.clone())
                },
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
        })
    }
}

impl TryFrom<NvCreateCompletionResponse> for ModelInferResponse {
    type Error = anyhow::Error;

    fn try_from(response: NvCreateCompletionResponse) -> Result<Self, Self::Error> {
        let mut outputs = vec![];
        let mut text_output = vec![];
        let mut finish_reason = vec![];
        for choice in &response.inner.choices {
            text_output.push(choice.text.clone());
            if let Some(reason) = choice.finish_reason.as_ref() {
                match reason {
                    CompletionFinishReason::Stop => {
                        finish_reason.push("stop".to_string());
                    }
                    CompletionFinishReason::Length => {
                        finish_reason.push("length".to_string());
                    }
                    CompletionFinishReason::ContentFilter => {
                        finish_reason.push("content_filter".to_string());
                    }
                }
            }
        }
        outputs.push(inference::model_infer_response::InferOutputTensor {
            name: "text_output".to_string(),
            datatype: "BYTES".to_string(),
            shape: vec![text_output.len() as i64],
            contents: Some(inference::InferTensorContents {
                bytes_contents: text_output
                    .into_iter()
                    .map(|text| text.as_bytes().to_vec())
                    .collect(),
                ..Default::default()
            }),
            ..Default::default()
        });
        outputs.push(inference::model_infer_response::InferOutputTensor {
            name: "finish_reason".to_string(),
            datatype: "BYTES".to_string(),
            shape: vec![finish_reason.len() as i64],
            contents: Some(inference::InferTensorContents {
                bytes_contents: finish_reason
                    .into_iter()
                    .map(|text| text.as_bytes().to_vec())
                    .collect(),
                ..Default::default()
            }),
            ..Default::default()
        });

        Ok(ModelInferResponse {
            model_name: response.inner.model,
            model_version: "1".to_string(),
            id: response.inner.id,
            outputs,
            parameters: ::std::collections::HashMap::<String, InferParameter>::new(),
            raw_output_contents: vec![],
        })
    }
}

impl TryFrom<NvCreateCompletionResponse> for ModelStreamInferResponse {
    type Error = anyhow::Error;

    fn try_from(response: NvCreateCompletionResponse) -> Result<Self, Self::Error> {
        match ModelInferResponse::try_from(response) {
            Ok(response) => Ok(ModelStreamInferResponse {
                infer_response: Some(response),
                ..Default::default()
            }),
            Err(e) => Ok(ModelStreamInferResponse {
                infer_response: None,
                error_message: format!("Failed to convert response: {}", e),
            }),
        }
    }
}
