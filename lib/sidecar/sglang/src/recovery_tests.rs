// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use clap::Parser;
use pb::sglang_service_server::{SglangService, SglangServiceServer};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use tonic::{Request, Response, Status};

#[derive(Default)]
struct TestEngine {
    instance: AtomicU64,
    healthy: AtomicBool,
    shutdowns: AtomicUsize,
    lose_ack: AtomicBool,
}

#[async_trait]
impl SglangService for Arc<TestEngine> {
    type WatchEngineStateStream = BoxStream<'static, Result<pb::EngineStateSnapshot, Status>>;
    async fn watch_engine_state(
        &self,
        _: Request<pb::WatchEngineStateRequest>,
    ) -> Result<Response<Self::WatchEngineStateStream>, Status> {
        let instance = self.instance.load(Ordering::SeqCst);
        if instance == 0 {
            return Err(Status::unavailable("engine stopped"));
        }
        Ok(Response::new(Box::pin(futures::stream::once(async move {
            Ok(pb::EngineStateSnapshot {
                instance_id: instance,
                ..Default::default()
            })
        }))))
    }
    async fn shutdown(
        &self,
        _: Request<pb::ShutdownRequest>,
    ) -> Result<Response<pb::ShutdownResponse>, Status> {
        self.shutdowns.fetch_add(1, Ordering::SeqCst);
        if self.lose_ack.swap(false, Ordering::SeqCst) {
            return Err(Status::unavailable("acknowledgement lost"));
        }
        Ok(Response::new(pb::ShutdownResponse {}))
    }
    async fn health_check(
        &self,
        _: Request<pb::HealthCheckRequest>,
    ) -> Result<Response<pb::HealthCheckResponse>, Status> {
        Ok(Response::new(pb::HealthCheckResponse {
            healthy: self.healthy.load(Ordering::SeqCst),
        }))
    }
    async fn get_model_info(
        &self,
        _: Request<pb::GetModelInfoRequest>,
    ) -> Result<Response<pb::GetModelInfoResponse>, Status> {
        Ok(Response::new(pb::GetModelInfoResponse {
            model_path: "test-model".into(),
            json_info: "{}".into(),
        }))
    }
    async fn get_server_info(
        &self,
        _: Request<pb::GetServerInfoRequest>,
    ) -> Result<Response<pb::GetServerInfoResponse>, Status> {
        let base = 5500 + self.instance.load(Ordering::SeqCst);
        Ok(Response::new(pb::GetServerInfoResponse { json_info: serde_json::json!({
            "incremental_streaming_output": true, "page_size": 16,
            "kv_events_config": {"replay_endpoint": format!("tcp://*:{}", base + 100), "buffer_steps": 100},
            "kv_events": {"publisher": "zmq", "endpoint_host": "*", "endpoint_port_base": base, "topic": "", "block_size": 16, "dp_size": 1}
        }).to_string() }))
    }
    async fn list_models(
        &self,
        _: Request<pb::ListModelsRequest>,
    ) -> Result<Response<pb::ListModelsResponse>, Status> {
        Ok(Response::new(pb::ListModelsResponse { models: vec![] }))
    }
    type TextGenerateStream = BoxStream<'static, Result<pb::TextGenerateResponse, Status>>;
    async fn text_generate(
        &self,
        _: Request<pb::TextGenerateRequest>,
    ) -> Result<Response<Self::TextGenerateStream>, Status> {
        Err(Status::unimplemented("unused in recovery test"))
    }
    type GenerateStream = BoxStream<'static, Result<pb::GenerateResponse, Status>>;
    async fn generate(
        &self,
        _: Request<pb::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        Err(Status::unimplemented("unused in recovery test"))
    }
    async fn text_embed(
        &self,
        _: Request<pb::TextEmbedRequest>,
    ) -> Result<Response<pb::TextEmbedResponse>, Status> {
        Err(Status::unimplemented("unused in recovery test"))
    }
    async fn embed(
        &self,
        _: Request<pb::EmbedRequest>,
    ) -> Result<Response<pb::EmbedResponse>, Status> {
        Err(Status::unimplemented("unused in recovery test"))
    }
    async fn classify(
        &self,
        _: Request<pb::ClassifyRequest>,
    ) -> Result<Response<pb::ClassifyResponse>, Status> {
        Err(Status::unimplemented("unused in recovery test"))
    }
    async fn tokenize(
        &self,
        _: Request<pb::TokenizeRequest>,
    ) -> Result<Response<pb::TokenizeResponse>, Status> {
        Err(Status::unimplemented("unused in recovery test"))
    }
    async fn detokenize(
        &self,
        _: Request<pb::DetokenizeRequest>,
    ) -> Result<Response<pb::DetokenizeResponse>, Status> {
        Err(Status::unimplemented("unused in recovery test"))
    }
    async fn get_load(
        &self,
        _: Request<pb::GetLoadRequest>,
    ) -> Result<Response<pb::GetLoadResponse>, Status> {
        Err(Status::unimplemented("unused in recovery test"))
    }
    async fn abort(
        &self,
        _: Request<pb::AbortRequest>,
    ) -> Result<Response<pb::AbortResponse>, Status> {
        Err(Status::unimplemented("unused in recovery test"))
    }
    async fn flush_cache(
        &self,
        _: Request<pb::FlushCacheRequest>,
    ) -> Result<Response<pb::FlushCacheResponse>, Status> {
        Err(Status::unimplemented("unused in recovery test"))
    }
    async fn pause_generation(
        &self,
        _: Request<pb::PauseGenerationRequest>,
    ) -> Result<Response<pb::PauseGenerationResponse>, Status> {
        Err(Status::unimplemented("unused in recovery test"))
    }
    async fn continue_generation(
        &self,
        _: Request<pb::ContinueGenerationRequest>,
    ) -> Result<Response<pb::ContinueGenerationResponse>, Status> {
        Err(Status::unimplemented("unused in recovery test"))
    }
    type ChatCompleteStream = BoxStream<'static, Result<pb::OpenAiStreamChunk, Status>>;
    async fn chat_complete(
        &self,
        _: Request<pb::OpenAiRequest>,
    ) -> Result<Response<Self::ChatCompleteStream>, Status> {
        Err(Status::unimplemented("unused in recovery test"))
    }
    type CompleteStream = BoxStream<'static, Result<pb::OpenAiStreamChunk, Status>>;
    async fn complete(
        &self,
        _: Request<pb::OpenAiRequest>,
    ) -> Result<Response<Self::CompleteStream>, Status> {
        Err(Status::unimplemented("unused in recovery test"))
    }
    async fn open_ai_embed(
        &self,
        _: Request<pb::OpenAiRequest>,
    ) -> Result<Response<pb::OpenAiResponse>, Status> {
        Err(Status::unimplemented("unused in recovery test"))
    }
    async fn open_ai_classify(
        &self,
        _: Request<pb::OpenAiRequest>,
    ) -> Result<Response<pb::OpenAiResponse>, Status> {
        Err(Status::unimplemented("unused in recovery test"))
    }
    async fn score(
        &self,
        _: Request<pb::OpenAiRequest>,
    ) -> Result<Response<pb::OpenAiResponse>, Status> {
        Err(Status::unimplemented("unused in recovery test"))
    }
    async fn rerank(
        &self,
        _: Request<pb::OpenAiRequest>,
    ) -> Result<Response<pb::OpenAiResponse>, Status> {
        Err(Status::unimplemented("unused in recovery test"))
    }
    async fn start_profile(
        &self,
        _: Request<pb::StartProfileRequest>,
    ) -> Result<Response<pb::StartProfileResponse>, Status> {
        Err(Status::unimplemented("unused in recovery test"))
    }
    async fn stop_profile(
        &self,
        _: Request<pb::StopProfileRequest>,
    ) -> Result<Response<pb::StopProfileResponse>, Status> {
        Err(Status::unimplemented("unused in recovery test"))
    }
    async fn update_weights_from_disk(
        &self,
        _: Request<pb::UpdateWeightsRequest>,
    ) -> Result<Response<pb::UpdateWeightsResponse>, Status> {
        Err(Status::unimplemented("unused in recovery test"))
    }
}

async fn fixture() -> (
    SglangSidecarEngine,
    Arc<TestEngine>,
    tokio::task::JoinHandle<()>,
) {
    let remote = Arc::new(TestEngine::default());
    remote.instance.store(1, Ordering::SeqCst);
    remote.healthy.store(true, Ordering::SeqCst);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let service = remote.clone();
    let server = tokio::spawn(async move {
        let incoming = async_stream::stream! {
            loop { yield listener.accept().await.map(|(stream, _)| stream); }
        };
        tonic::transport::Server::builder()
            .add_service(SglangServiceServer::new(service))
            .serve_with_incoming(incoming)
            .await
            .unwrap();
    });
    let args = Args::try_parse_from(["sidecar", "--grpc-endpoint", &address.to_string()]).unwrap();
    let (mut engine, _) = SglangSidecarEngine::from_discovered(
        args,
        Discovery {
            model_path: "test-model".into(),
            tokenizer_path: "test-model".into(),
            served_model_name: None,
            max_model_len: None,
            model_info: serde_json::json!({}),
            server_info: serde_json::json!({}),
        },
    )
    .unwrap();
    engine.transport.retry_interval = Duration::from_millis(10);
    engine.transport.connect_attempt_timeout = Duration::from_millis(200);
    engine.transport.startup_deadline = Duration::from_secs(2);
    engine.start(1).await.unwrap();
    (engine, remote, server)
}

#[tokio::test]
async fn missing_history_waits_for_new_instance_even_when_shutdown_ack_is_lost() {
    for lose_ack in [false, true] {
        let (engine, remote, server) = fixture().await;
        remote.lose_ack.store(lose_ack, Ordering::SeqCst);
        *engine.bootstrap_failure.lock().unwrap() = Some(BootstrapOutcome::MissingHistory {
            dp_rank: 0,
            expected: 0,
            got: 10,
        });
        let recovery = engine.recover_startup();
        tokio::pin!(recovery);
        assert!(
            tokio::time::timeout(Duration::from_millis(100), &mut recovery)
                .await
                .is_err()
        );
        assert_eq!(
            remote.shutdowns.load(Ordering::SeqCst),
            if lose_ack { 2 } else { 1 }
        );
        // Even a still-healthy old process is not a replacement.
        assert!(engine.started_state().is_none());
        remote.instance.store(0, Ordering::SeqCst);
        assert!(
            tokio::time::timeout(Duration::from_millis(50), &mut recovery)
                .await
                .is_err()
        );
        remote.instance.store(2, Ordering::SeqCst);
        remote.healthy.store(false, Ordering::SeqCst);
        assert!(
            tokio::time::timeout(Duration::from_millis(50), &mut recovery)
                .await
                .is_err()
        );
        remote.healthy.store(true, Ordering::SeqCst);
        tokio::time::timeout(Duration::from_secs(2), &mut recovery)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let state = engine.started_state().unwrap();
        assert_eq!(state.instance_id, Some(2));
        assert!(state.kv_event_sources[0].endpoint.ends_with(":5502"));
        assert!(state.kv_event_sources[0].replay_endpoint.ends_with(":5602"));
        assert_eq!(
            remote.shutdowns.load(Ordering::SeqCst),
            if lose_ack { 2 } else { 1 }
        );
        server.abort();
    }
}

#[tokio::test]
async fn uncertainty_retries_without_shutdown_and_changed_engine_is_not_shutdown() {
    for replacement in [false, true] {
        let (engine, remote, server) = fixture().await;
        let failure = if replacement {
            remote.instance.store(2, Ordering::SeqCst);
            BootstrapOutcome::MissingHistory {
                dp_rank: 0,
                expected: 0,
                got: 10,
            }
        } else {
            BootstrapOutcome::Uncertain {
                reason: "replay timed out".into(),
            }
        };
        *engine.bootstrap_failure.lock().unwrap() = Some(failure);
        tokio::time::timeout(Duration::from_secs(2), engine.recover_startup())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(remote.shutdowns.load(Ordering::SeqCst), 0);
        assert_eq!(
            engine.started_state().unwrap().instance_id,
            Some(if replacement { 2 } else { 1 })
        );
        server.abort();
    }
}

#[tokio::test]
async fn changed_or_unhealthy_engine_after_bootstrap_is_uncertain() {
    for changed in [false, true] {
        let (engine, remote, server) = fixture().await;
        if changed {
            remote.instance.store(2, Ordering::SeqCst);
        } else {
            remote.healthy.store(false, Ordering::SeqCst);
        }
        assert!(engine.wait_for_startup().await.is_err());
        assert!(matches!(
            *engine.bootstrap_failure.lock().unwrap(),
            Some(BootstrapOutcome::Uncertain { .. })
        ));
        assert_eq!(remote.shutdowns.load(Ordering::SeqCst), 0);
        server.abort();
    }
}
