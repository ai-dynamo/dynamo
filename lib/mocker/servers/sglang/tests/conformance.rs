// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::BackendError;
use dynamo_mocker::common::protocols::{EngineType, MockEngineArgs};
use dynamo_sglang_mocker::{MockerServerConfig, SglangMockerService};
use dynamo_sglang_sidecar::SglangSidecarEngine;
use dynamo_sglang_sidecar::proto::{
    self as pb,
    sglang_service_server::{SglangService, SglangServiceServer},
};
use futures::stream::BoxStream;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{Request, Response, Status};

#[path = "../../tests/common/mod.rs"]
mod common;

struct Fixture {
    endpoint: String,
    service: SglangMockerService,
    server: JoinHandle<()>,
}

impl common::SidecarFixture for Fixture {
    type Engine = SglangSidecarEngine;

    async fn start(control: common::Control) -> Self {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .block_size(4)
            .num_gpu_blocks(4_096)
            .max_num_seqs(Some(64))
            .max_num_batched_tokens(Some(1_024))
            .speedup_ratio(0.0)
            .dp_size(1)
            .build()
            .unwrap();
        let service = SglangMockerService::new(MockerServerConfig::default(), args).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let controlled = ControlledService {
            inner: service.clone(),
            control,
        };
        let server = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(SglangServiceServer::new(controlled))
                .serve_with_incoming(TcpListenerStream::new(listener))
                .await
                .unwrap();
        });
        Self {
            endpoint: format!("http://{address}"),
            service,
            server,
        }
    }

    async fn engine(&self) -> Self::Engine {
        let argv = vec![
            "dynamo-sglang-sidecar".into(),
            "--sglang-endpoint".into(),
            self.endpoint.clone(),
            "--sglang-connections".into(),
            "1".into(),
            "--connect-timeout-secs".into(),
            "1".into(),
            "--health-poll-interval-secs".into(),
            "1".into(),
            "--health-deadline-secs".into(),
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
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.server.abort();
    }
}

#[derive(Clone)]
struct ControlledService {
    inner: SglangMockerService,
    control: common::Control,
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
                let guard = self.control.open(request.get_ref().rid.as_deref().unwrap()).await?;
                let response = self.inner.generate(request).await?;
                Ok(Response::new(self.control.stream(response.into_inner(), guard)))
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

impl common::NativeResponse for pb::GenerateResponse {
    fn record_tokens(&self, tokens: &mut Vec<u32>) -> bool {
        tokens.clear();
        tokens.extend(self.output_ids.iter().map(|&id| u32::try_from(id).unwrap()));
        !self.output_ids.is_empty()
    }

    fn is_terminal(&self) -> bool {
        self.finished
    }
}
