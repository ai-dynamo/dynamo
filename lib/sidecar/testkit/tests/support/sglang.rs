// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::BackendError;
use dynamo_mocker::common::protocols::EngineType;
use dynamo_sglang_mocker::{MockerServerConfig, SglangMockerService};
use dynamo_sglang_sidecar::SglangSidecarEngine;
use dynamo_sglang_sidecar::proto::{
    self as pb,
    sglang_service_server::{SglangService, SglangServiceServer},
};
use dynamo_sidecar_testkit::control::{Controller, Protocol};
use dynamo_sidecar_testkit::server::TestServer;
use futures::stream::BoxStream;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{Request, Response, Status};

use super::{FixtureConfig, SidecarFixture, fast_engine_args};

pub struct Fixture {
    config: FixtureConfig,
    service: SglangMockerService,
    server: TestServer,
}

impl SidecarFixture for Fixture {
    type Engine = SglangSidecarEngine;
    type Protocol = Adapter;

    async fn start(control: Controller<Adapter>, config: FixtureConfig) -> Self {
        let service = SglangMockerService::new(
            MockerServerConfig {
                model: config.model.clone(),
                ..Default::default()
            },
            fast_engine_args(EngineType::Sglang),
        )
        .unwrap();
        let controlled = ControlledService {
            inner: service.clone(),
            control,
        };
        let server = TestServer::start(move |listener, shutdown| async move {
            tonic::transport::Server::builder()
                .add_service(SglangServiceServer::new(controlled))
                .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
                    let _ = shutdown.await;
                })
                .await?;
            Ok(())
        })
        .await
        .unwrap();
        Self {
            config,
            service,
            server,
        }
    }

    async fn engine(&self) -> Self::Engine {
        let argv = vec![
            "dynamo-sglang-sidecar".into(),
            "--grpc-endpoint".into(),
            self.server.endpoint(),
            "--grpc-connections".into(),
            self.config.connections.to_string(),
            "--grpc-connect-attempt-timeout-secs".into(),
            "1".into(),
            "--grpc-retry-interval-secs".into(),
            "1".into(),
            "--grpc-startup-deadline-secs".into(),
            "5".into(),
        ];
        tokio::task::spawn_blocking(move || SglangSidecarEngine::from_args(Some(argv)).unwrap().0)
            .await
            .unwrap()
    }

    fn eof_error() -> BackendError {
        BackendError::EngineShutdown
    }

    fn active_request_count(&self) -> usize {
        self.service.active_request_count()
    }

    async fn shutdown(&mut self) {
        self.server.shutdown().await.unwrap();
    }
}

#[derive(Clone)]
struct ControlledService {
    inner: SglangMockerService,
    control: Controller<Adapter>,
}

macro_rules! delegate_service {
    ($($method:ident($request:ty) -> $response:ty;)*) => {
        #[tonic::async_trait]
        impl SglangService for ControlledService {
            type TextGenerateStream = <SglangMockerService as SglangService>::TextGenerateStream;
            type GenerateStream = BoxStream<'static, Result<pb::GenerateResponse, Status>>;
            type ChatCompleteStream = <SglangMockerService as SglangService>::ChatCompleteStream;
            type CompleteStream = <SglangMockerService as SglangService>::CompleteStream;

            async fn generate(
                &self,
                request: Request<pb::GenerateRequest>,
            ) -> Result<Response<Self::GenerateStream>, Status> {
                let opened = self.control.open(request.get_ref()).await?;
                let response = self.inner.generate(request).await?;
                Ok(Response::new(opened.wrap(response.into_inner())))
            }

            $(
                async fn $method(
                    &self,
                    request: Request<$request>,
                ) -> Result<Response<$response>, Status> {
                    self.inner.$method(request).await
                }
            )*
        }
    };
}

delegate_service! {
    text_generate(pb::TextGenerateRequest) -> Self::TextGenerateStream;
    text_embed(pb::TextEmbedRequest) -> pb::TextEmbedResponse;
    embed(pb::EmbedRequest) -> pb::EmbedResponse;
    classify(pb::ClassifyRequest) -> pb::ClassifyResponse;
    tokenize(pb::TokenizeRequest) -> pb::TokenizeResponse;
    detokenize(pb::DetokenizeRequest) -> pb::DetokenizeResponse;
    health_check(pb::HealthCheckRequest) -> pb::HealthCheckResponse;
    get_model_info(pb::GetModelInfoRequest) -> pb::GetModelInfoResponse;
    get_server_info(pb::GetServerInfoRequest) -> pb::GetServerInfoResponse;
    list_models(pb::ListModelsRequest) -> pb::ListModelsResponse;
    get_load(pb::GetLoadRequest) -> pb::GetLoadResponse;
    abort(pb::AbortRequest) -> pb::AbortResponse;
    flush_cache(pb::FlushCacheRequest) -> pb::FlushCacheResponse;
    pause_generation(pb::PauseGenerationRequest) -> pb::PauseGenerationResponse;
    continue_generation(pb::ContinueGenerationRequest) -> pb::ContinueGenerationResponse;
    chat_complete(pb::OpenAiRequest) -> Self::ChatCompleteStream;
    complete(pb::OpenAiRequest) -> Self::CompleteStream;
    open_ai_embed(pb::OpenAiRequest) -> pb::OpenAiResponse;
    open_ai_classify(pb::OpenAiRequest) -> pb::OpenAiResponse;
    score(pb::OpenAiRequest) -> pb::OpenAiResponse;
    rerank(pb::OpenAiRequest) -> pb::OpenAiResponse;
    start_profile(pb::StartProfileRequest) -> pb::StartProfileResponse;
    stop_profile(pb::StopProfileRequest) -> pb::StopProfileResponse;
    update_weights_from_disk(pb::UpdateWeightsRequest) -> pb::UpdateWeightsResponse;
}

#[derive(Clone, Copy)]
pub struct Adapter;

impl Protocol for Adapter {
    type Request = pb::GenerateRequest;
    type Response = pb::GenerateResponse;
    type Error = Status;

    fn request_id(request: &Self::Request) -> &str {
        request.rid.as_deref().expect("sidecar request ID")
    }

    fn record_tokens(response: &Self::Response, tokens: &mut Vec<u32>) -> bool {
        tokens.extend(
            response
                .output_ids
                .iter()
                .map(|&id| u32::try_from(id).unwrap()),
        );
        !response.output_ids.is_empty()
    }

    fn is_terminal(response: &Self::Response) -> bool {
        response.finished
    }

    fn injected_error(message: &'static str) -> Self::Error {
        Status::unavailable(message)
    }
}
