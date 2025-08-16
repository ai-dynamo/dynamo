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
use crate::request_template::RequestTemplate;
use anyhow::Result;
use derive_builder::Builder;
use dynamo_runtime::transports::etcd;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tokio_stream::{wrappers::ReceiverStream, Stream, StreamExt};

use tonic::{transport::Server, Request, Response, Status};

pub mod inference {
    tonic::include_proto!("inference");
}
use inference::grpc_inference_service_server::{GrpcInferenceService, GrpcInferenceServiceServer};
use inference::{ModelInferRequest, ModelInferResponse, InferParameter, ModelStreamInferResponse, ModelMetadataRequest, ModelMetadataResponse, ModelConfigRequest, ModelConfigResponse};

/// [WIP] Understand the State, central piece for dynamo logic
/// gRPC service shared state
/// [TODO] 'metrics' are not being shared
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

    #[builder(default = "None")]
    request_template: Option<RequestTemplate>, // [WIP] should this be required?

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

        // Part of the builder untill serve()
        Server::builder()
            .add_service(GrpcInferenceServiceServer::new(self.clone()))
            .serve(address.parse()?)
            .await?;

        // [WIP] Start tonic gRPC server here
        // let listener = tokio::net::TcpListener::bind(address.as_str())
        //     .await
        //     .unwrap_or_else(|_| panic!("could not bind to address: {address}"));

        // let router = self.router.clone();
        // let observer = cancel_token.child_token();

        // axum::serve(listener, router)
        //     .with_graceful_shutdown(observer.cancelled_owned())
        //     .await
        //     .inspect_err(|_| cancel_token.cancel())?;

        Ok(())
    }

    // Documentation of exposed HTTP endpoints
    // pub fn route_docs(&self) -> &[RouteDoc] {
    //     &self.route_docs
    // }
}

#[tonic::async_trait]
impl GrpcInferenceService for GrpcService {
    async fn model_infer(
        &self,
        request: Request<ModelInferRequest>,
    ) -> Result<Response<ModelInferResponse>, Status> {
        println!("Got a request: {:?}", request);

        let reply = ModelInferResponse {
            model_version: "1".to_string(),
            model_name: "mock".to_string(),
            id: "1234".to_string(),
            outputs: vec![],
            parameters: ::std::collections::HashMap::<String, InferParameter>::new(),
            raw_output_contents: vec![],
        };

        Ok(Response::new(reply))
    }

    type ModelStreamInferStream = Pin<Box<dyn Stream<Item = Result<ModelStreamInferResponse, Status>> + Send  + 'static>>;

    async fn model_stream_infer(
        &self,
        _request: Request<tonic::Streaming<ModelInferRequest>>,
    ) -> Result<Response<Self::ModelStreamInferStream>, Status> {
        unimplemented!()
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

        // let mut router = axum::Router::new();

        // let mut all_docs = Vec::new();

        // [WIP] add gRPC endpoints here
        // let mut routes = vec![
        //     metrics::router(registry, var(HTTP_SVC_METRICS_PATH_ENV).ok()),
        //     super::openai::list_models_router(state.clone(), var(HTTP_SVC_MODELS_PATH_ENV).ok()),
        //     super::health::health_check_router(state.clone(), var(HTTP_SVC_HEALTH_PATH_ENV).ok()),
        //     super::health::live_check_router(state.clone(), var(HTTP_SVC_LIVE_PATH_ENV).ok()),
        // ];

        // if config.enable_chat_endpoints {
        //     routes.push(super::openai::chat_completions_router(
        //         state.clone(),
        //         config.request_template.clone(), // TODO clone()? reference?
        //         var(HTTP_SVC_CHAT_PATH_ENV).ok(),
        //     ));
        // }

        // if config.enable_cmpl_endpoints {
        //     routes.push(super::openai::completions_router(
        //         state.clone(),
        //         var(HTTP_SVC_CMP_PATH_ENV).ok(),
        //     ));
        // }

        // if config.enable_embeddings_endpoints {
        //     routes.push(super::openai::embeddings_router(
        //         state.clone(),
        //         var(HTTP_SVC_EMB_PATH_ENV).ok(),
        //     ));
        // }

        // if config.enable_responses_endpoints {
        //     routes.push(super::openai::responses_router(
        //         state.clone(),
        //         config.request_template,
        //         var(HTTP_SVC_RESPONSES_PATH_ENV).ok(),
        //     ));
        // }

        // for (route_docs, route) in routes.into_iter().chain(self.routes.into_iter()) {
        //     router = router.merge(route);
        //     all_docs.extend(route_docs);
        // }

        // for (route_docs, route) in routes.into_iter() {
        //     router = router.merge(route);
        //     all_docs.extend(route_docs);
        // }

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
