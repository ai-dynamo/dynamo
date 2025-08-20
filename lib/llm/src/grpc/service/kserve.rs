// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::env::var;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use crate::http::service::metrics;
use crate::http::service::Metrics;
use crate::http::service::RouteDoc;

use crate::discovery::ModelManager;
use crate::preprocessor::prompt;
use crate::request_template::RequestTemplate;
use anyhow::Result;
use async_nats::jetstream::stream;
use async_openai::types::CompletionFinishReason;
use async_openai::types::CreateCompletionRequest;
use async_openai::types::Model;
use async_openai::types::Stop;
use derive_builder::Builder;
use dynamo_runtime::transports::etcd;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tokio_stream::{wrappers::ReceiverStream, Stream, StreamExt};

use tonic::{transport::Server, Request, Response, Status};
use crate::grpc::service::openai::{model_infer_completions, completion_response_stream};

use crate::protocols::openai::completions::{NvCreateCompletionRequest, NvCreateCompletionResponse};

pub mod inference {
    tonic::include_proto!("inference");
}
use inference::grpc_inference_service_server::{GrpcInferenceService, GrpcInferenceServiceServer};
use inference::{ModelInferRequest, ModelInferResponse, InferParameter, ModelStreamInferResponse, ModelMetadataRequest, ModelMetadataResponse, ModelConfigRequest, ModelConfigResponse};

/// [WIP] Understand the State, central piece for dynamo logic
/// gRPC service shared state
/// [TODO] 'metrics' are for HTTP service, how to port to gRPC?
/// Or should we always have HTTP service up for non-inference?
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

    // TODO
    pub fn sse_keep_alive(&self) -> Option<Duration> {
        None
    }
}

// [WIP] rename to Kserve?
#[derive(Clone)]
pub struct GrpcService {
    // The state we share with every request handler
    state: Arc<State>,

    // router: axum::Router, // [WIP] should be the tonic server
    port: u16,
    host: String,
    // route_docs: Vec<RouteDoc>, // [WIP] does this apply for gRPC?
}

#[derive(Clone, Builder)]
#[builder(pattern = "owned", build_fn(private, name = "build_internal"))]
pub struct GrpcServiceConfig {
    #[builder(default = "8787")]
    port: u16,

    #[builder(setter(into), default = "String::from(\"0.0.0.0\")")]
    host: String,

     // [WIP] should apply to all request type.
    #[builder(default = "None")]
    request_template: Option<RequestTemplate>,

    #[builder(default = "None")]
    etcd_client: Option<etcd::Client>,
}

impl GrpcService {
    pub fn builder() -> GrpcServiceConfigBuilder {
        GrpcServiceConfigBuilder::default()
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

    // Documentation of exposed HTTP endpoints
    // pub fn route_docs(&self) -> &[RouteDoc] {
    //     &self.route_docs
    // }
}

impl TryFrom<ModelInferRequest> for NvCreateCompletionRequest {
    type Error = Status;

    fn try_from(request: ModelInferRequest) -> Result<Self, Self::Error> {
        // iterate through inputs
        let mut text_input = None;
        let mut stream = false;
        for input in request.inputs.iter() {
            match input.name.as_str() {
                "text_input" => {
                    if input.datatype != "BYTES" {
                        return Err(Status::invalid_argument(format!(
                            "Expected 'text_input' to be of type BYTES for string input, got {:?}",
                            input.datatype
                        )));
                    }
                    if input.shape != vec![1] {
                        return Err(Status::invalid_argument(format!(
                            "Expected 'text_input' to have shape [1], got {:?}",
                            input.shape
                        )));
                    }
                    match &input.contents {
                        Some(content) => {
                            let bytes = &content.bytes_contents[0];
                            text_input = Some(String::from_utf8_lossy(&bytes).to_string());
                        }
                        _ => {
                            // [gluo WIP] look for 'raw_input_contents'
                            return Err(Status::invalid_argument(format!(
                                "[gluo WIP] Currently expecting 'text_input' contents to be of type BYTES"
                            )));
                        }
                    }
                }
                "streaming" | "stream" => {
                    if input.datatype != "BOOL" {
                        return Err(Status::invalid_argument(format!(
                            "Expected '{}' to be of type BOOL, got {:?}",
                            input.name,
                            input.datatype
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
                        _ => {
                            return Err(Status::invalid_argument(format!(
                                "expected 'stream' contents to be of type BOOL"
                            )));
                        }
                    }
                }
                _ => {
                    return Err(Status::invalid_argument(format!("Invalid input name: {}, supported inputs are 'text_input', 'stream'", input.name)));
                }
            }
        }

        // return error if text_input is None
        let text_input = match text_input {
            Some(input) => input,
            None => {
                return Err(Status::invalid_argument(
                    "Missing required input: 'text_input'"
                ));
            }
        };
        

        // [gluo FIXME] directly construct the object
        Ok(NvCreateCompletionRequest {
            inner: CreateCompletionRequest {
                model: request.model_name,
                prompt: async_openai::types::Prompt::String(text_input),
                stream: Some(stream),
                ..Default::default()
            },
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
                    CompletionFinishReason::Stop => { finish_reason.push("stop".to_string()); }
                    CompletionFinishReason::Length => { finish_reason.push("length".to_string()); }
                    CompletionFinishReason::ContentFilter => { finish_reason.push("content_filter".to_string()); }
                }
            }
        }
        outputs.push(inference::model_infer_response::InferOutputTensor {
                    name: "text_output".to_string(),
                    datatype: "BYTES".to_string(),
                    shape: vec![text_output.len() as i64],
                contents: Some(inference::InferTensorContents {
                    bytes_contents: text_output.into_iter().map(|text| text.as_bytes().to_vec()).collect(),
                    ..Default::default()
                }),
                ..Default::default()
            });
        outputs.push(inference::model_infer_response::InferOutputTensor {
                    name: "finish_reason".to_string(),
                    datatype: "BYTES".to_string(),
                    shape: vec![finish_reason.len() as i64],
                contents: Some(inference::InferTensorContents {
                    bytes_contents: finish_reason.into_iter().map(|text| text.as_bytes().to_vec()).collect(),
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
            Ok(response) => {
                Ok(ModelStreamInferResponse {
                    infer_response: Some(response),
                    ..Default::default()
                })
            }
            Err(e) => {
                Ok(ModelStreamInferResponse {
                    infer_response: None,
                    error_message: format!("Failed to convert response: {}", e).into(),
                })
            }
        }
    }
}

#[tonic::async_trait]
impl GrpcInferenceService for GrpcService {
    async fn model_infer(
        &self,
        request: Request<ModelInferRequest>,
    ) -> Result<Response<ModelInferResponse>, Status> {
        let completion_request: NvCreateCompletionRequest = request.into_inner().try_into().map_err(|e| {
            Status::invalid_argument(format!("Failed to parse request: {}", e))
        })?;

        if completion_request.inner.stream.unwrap_or(false) {
            // return error that streaming is not supported
            return Err(Status::invalid_argument("Streaming is not supported for this endpoint"));
        }

        let completion_response = model_infer_completions(self.state_clone(), completion_request).await?;

        let reply = completion_response.try_into().map_err(|e| {
            Status::invalid_argument(format!("Failed to parse response: {}", e))
        })?;

        Ok(Response::new(reply))
    }

    type ModelStreamInferStream = Pin<Box<dyn Stream<Item = Result<ModelStreamInferResponse, Status>> + Send  + 'static>>;

    async fn model_stream_infer(
        &self,
        request: Request<tonic::Streaming<ModelInferRequest>>,
    ) -> Result<Response<Self::ModelStreamInferStream>, Status> {
        let mut stream = request.into_inner();
        let state = self.state_clone();
        let output = async_stream::try_stream! {
            // [gluo FIXME] should be able to demux request / response streaming 
            while let Some(request) = stream.next().await {
                // [gluo FIXME] request error handling
                let completion_request: NvCreateCompletionRequest = request.unwrap().try_into().map_err(|e| {
                    Status::invalid_argument(format!("Failed to parse request: {}", e))
                })?;

                let mut stream = completion_response_stream(state.clone(), completion_request).await?;

                while let Some(response) = stream.next().await {
                    match response.data {
                        Some(data) => {
                            let reply = ModelStreamInferResponse::try_from(data).map_err(|e| {
                                Status::invalid_argument(format!("Failed to parse response: {}", e))
                            })?;
                            yield reply;
                        },
                        None => {
                            // Handle the case where there is no data
                        }
                    }
                }
            }
        };

        Ok(Response::new(Box::pin(output) as Self::ModelStreamInferStream))
    }

    async fn model_metadata(
        &self,
        _request: Request<ModelMetadataRequest>,
    ) -> Result<Response<ModelMetadataResponse>, Status> {
        unimplemented!()
    }

    async fn model_config(
        &self,
        _request: Request<ModelConfigRequest>,
    ) -> Result<Response<ModelConfigResponse>, Status> {
        unimplemented!()
    }
}

impl GrpcServiceConfigBuilder {
    pub fn build(self) -> Result<GrpcService, anyhow::Error> {
        let config: GrpcServiceConfig = self.build_internal()?;

        let model_manager = Arc::new(ModelManager::new());
        let state = Arc::new(State::new_with_etcd(model_manager, config.etcd_client));

        // enable prometheus metrics
        let registry = metrics::Registry::new();
        state.metrics_clone().register(&registry)?;

        Ok(GrpcService {
            state,
            // router,
            port: config.port,
            host: config.host,
            // route_docs: all_docs,
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
