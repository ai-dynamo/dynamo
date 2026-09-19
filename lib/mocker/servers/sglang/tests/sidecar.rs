// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use dynamo_backend_common::{
    AsyncEngineContext, DisaggregationMode, FinishReason, GenerateContext, LLMEngine,
    PrefillResult, PreprocessedRequest,
};
use dynamo_mocker::common::protocols::{EngineType, MockEngineArgs};
use dynamo_sglang_mocker::{MockerServerConfig, ServerMode, SglangMockerService};
use dynamo_sglang_sidecar::{
    SglangSidecarEngine, proto::sglang_service_server::SglangServiceServer,
};
use dynamo_sidecar_testkit::{self as testkit, request};
use futures::StreamExt;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tokio::time::timeout;
use tokio_stream::wrappers::TcpListenerStream;

struct RunningServer {
    endpoint: String,
    service: SglangMockerService,
    shutdown: Option<oneshot::Sender<()>>,
    task: JoinHandle<()>,
}

impl RunningServer {
    async fn start(mode: ServerMode, engine_args: MockEngineArgs) -> Self {
        let service = SglangMockerService::new(
            MockerServerConfig {
                mode,
                ..Default::default()
            },
            engine_args,
        )
        .unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (shutdown, shutdown_rx) = oneshot::channel();
        let server_service = service.clone();
        let task = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(SglangServiceServer::new(server_service))
                .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
                    let _ = shutdown_rx.await;
                })
                .await
                .unwrap();
        });
        Self {
            endpoint: format!("http://{address}"),
            service,
            shutdown: Some(shutdown),
            task,
        }
    }

    async fn close(mut self, engine: impl LLMEngine) {
        timeout(Duration::from_secs(5), async {
            engine.cleanup().await.unwrap();
            drop(engine);
            self.service.shutdown().await.unwrap();
            if let Some(shutdown) = self.shutdown.take() {
                let _ = shutdown.send(());
            }
            (&mut self.task).await.expect("gRPC server task failed");
        })
        .await
        .expect("sidecar and mocker teardown timed out");
    }
}

impl Drop for RunningServer {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        self.task.abort();
    }
}

fn fast_engine_args() -> MockEngineArgs {
    MockEngineArgs::builder()
        .engine_type(EngineType::Sglang)
        .block_size(4)
        .num_gpu_blocks(4_096)
        .max_num_seqs(Some(64))
        .max_num_batched_tokens(Some(1_024))
        .speedup_ratio(0.0)
        .dp_size(1)
        .build()
        .unwrap()
}

async fn sidecar(endpoint: &str, mode: DisaggregationMode) -> SglangSidecarEngine {
    let mut argv = vec![
        "dynamo-sglang-sidecar".to_string(),
        "--grpc-endpoint".to_string(),
        endpoint.to_string(),
        "--grpc-connections".to_string(),
        "1".to_string(),
        "--grpc-connect-attempt-timeout-secs".to_string(),
        "1".to_string(),
        "--grpc-retry-interval-secs".to_string(),
        "1".to_string(),
        "--grpc-startup-deadline-secs".to_string(),
        "5".to_string(),
    ];
    if mode.is_prefill() {
        argv.extend(["--bootstrap-host".to_string(), "127.0.0.1".to_string()]);
    }
    tokio::task::spawn_blocking(move || SglangSidecarEngine::from_args(Some(argv)).unwrap().0)
        .await
        .unwrap()
}

async fn native_output(endpoint: &str, request_id: &str) -> (Vec<u32>, Vec<f64>) {
    use dynamo_sglang_sidecar::proto as pb;

    timeout(Duration::from_secs(5), async {
        let mut client =
            pb::sglang_service_client::SglangServiceClient::connect(endpoint.to_string())
                .await
                .unwrap();
        let mut stream = client
            .generate(pb::GenerateRequest {
                input_ids: vec![11, 22, 33, 44],
                sampling_params: Some(pb::SamplingParams {
                    max_new_tokens: Some(3),
                    n: Some(1),
                    ..Default::default()
                }),
                stream: Some(true),
                return_logprob: Some(true),
                rid: Some(request_id.to_string()),
                ..Default::default()
            })
            .await
            .unwrap()
            .into_inner();
        let mut tokens = Vec::new();
        let mut logprobs = Vec::new();
        while let Some(response) = stream.message().await.unwrap() {
            let selected: Vec<(f64, u32, Option<String>)> =
                serde_json::from_str(&response.meta_info["output_token_logprobs"]).unwrap();
            assert_eq!(selected.len(), response.output_ids.len());
            logprobs.extend(selected.into_iter().map(|(logprob, _, _)| logprob));
            tokens.extend(
                response
                    .output_ids
                    .into_iter()
                    .map(|token| u32::try_from(token).unwrap()),
            );
        }
        assert_eq!(tokens.len(), 3);
        assert_eq!(logprobs.len(), 3);
        (tokens, logprobs)
    })
    .await
    .expect("native SGLang reference stream timed out")
}

async fn collect(
    engine: &SglangSidecarEngine,
    request: PreprocessedRequest,
) -> Vec<dynamo_backend_common::LLMEngineOutput> {
    collect_with_context(
        engine,
        request,
        dynamo_backend_common::testing::mock_context(),
    )
    .await
}

async fn collect_with_context(
    engine: &SglangSidecarEngine,
    request: PreprocessedRequest,
    context: Arc<dyn AsyncEngineContext>,
) -> Vec<dynamo_backend_common::LLMEngineOutput> {
    engine
        .generate(request, GenerateContext::new(context, None))
        .await
        .unwrap()
        .map(|item| item.unwrap())
        .collect()
        .await
}

#[tokio::test]
async fn shared_streaming_preserves_native_tokens_logprobs_and_usage() {
    timeout(Duration::from_secs(30), async {
        let server = RunningServer::start(ServerMode::Aggregated, fast_engine_args()).await;
        let engine = sidecar(&server.endpoint, DisaggregationMode::Aggregated).await;
        let config = engine.start(0).await.unwrap();
        let registration = config.llm.unwrap();
        assert_eq!(registration.context_length, Some(32_768));
        assert_eq!(registration.kv_cache_block_size, Some(4));
        assert_eq!(registration.total_kv_blocks, Some(4_096));
        assert_eq!(registration.max_num_seqs, Some(64));
        assert_eq!(registration.max_num_batched_tokens, Some(1_024));

        let context = dynamo_backend_common::testing::mock_context();
        let (expected, expected_logprobs) = native_output(&server.endpoint, context.id()).await;
        testkit::wait_idle(server.service.metrics_receiver(), || {
            server.service.active_request_count()
        })
        .await;
        testkit::streaming(&engine, context, &expected, &expected_logprobs, 2).await;
        testkit::wait_idle(server.service.metrics_receiver(), || {
            server.service.active_request_count()
        })
        .await;
        server.close(engine).await;
    })
    .await
    .expect("sglang shared_streaming_preserves_native_tokens_logprobs_and_usage timed out");
}

#[tokio::test]
async fn shared_native_rejection_preserves_error_and_recovers() {
    timeout(Duration::from_secs(30), async {
        let server = RunningServer::start(ServerMode::Aggregated, fast_engine_args()).await;
        let engine = sidecar(&server.endpoint, DisaggregationMode::Aggregated).await;
        engine.start(0).await.unwrap();
        testkit::rejection(&engine).await;
        testkit::wait_idle(server.service.metrics_receiver(), || {
            server.service.active_request_count()
        })
        .await;
        server.close(engine).await;
    })
    .await
    .expect("sglang shared_native_rejection_preserves_error_and_recovers timed out");
}

#[tokio::test]
async fn shared_cancellation_releases_scheduler_work_and_recovers() {
    timeout(Duration::from_secs(30), async {
        let mut args = fast_engine_args();
        args.speedup_ratio = 0.1;
        let server = RunningServer::start(ServerMode::Aggregated, args).await;
        let engine = sidecar(&server.endpoint, DisaggregationMode::Aggregated).await;
        engine.start(0).await.unwrap();
        testkit::cancellation(&engine, || server.service.active_request_count()).await;
        testkit::wait_idle(server.service.metrics_receiver(), || {
            server.service.active_request_count()
        })
        .await;
        testkit::recovery(&engine).await;
        server.close(engine).await;
    })
    .await
    .expect("sglang shared_cancellation_releases_scheduler_work_and_recovers timed out");
}

#[tokio::test]
async fn shared_consumer_drop_releases_scheduler_work_and_recovers() {
    timeout(Duration::from_secs(30), async {
        let mut args = fast_engine_args();
        args.speedup_ratio = 0.1;
        let server = RunningServer::start(ServerMode::Aggregated, args).await;
        let engine = sidecar(&server.endpoint, DisaggregationMode::Aggregated).await;
        engine.start(0).await.unwrap();
        testkit::consumer_drop(&engine, || server.service.active_request_count()).await;
        testkit::wait_idle(server.service.metrics_receiver(), || {
            server.service.active_request_count()
        })
        .await;
        testkit::recovery(&engine).await;
        server.close(engine).await;
    })
    .await
    .expect("sglang shared_consumer_drop_releases_scheduler_work_and_recovers timed out");
}

#[tokio::test]
async fn prefill_handoff_round_trips_through_a_decode_server() {
    let prefill_server = RunningServer::start(ServerMode::Prefill, fast_engine_args()).await;
    let decode_server = RunningServer::start(ServerMode::Decode, fast_engine_args()).await;
    let prefill = sidecar(&prefill_server.endpoint, DisaggregationMode::Prefill).await;
    let decode = sidecar(&decode_server.endpoint, DisaggregationMode::Decode).await;
    prefill.start(0).await.unwrap();
    decode.start(1).await.unwrap();

    let prefill_outputs = collect(&prefill, request(3)).await;
    assert_eq!(prefill_outputs.len(), 2);
    assert!(prefill_outputs[0].token_ids.is_empty());
    assert!(prefill_outputs[0].finish_reason.is_none());
    let handoff = prefill_outputs[0]
        .disaggregated_params
        .clone()
        .expect("prefill response should carry SGLang rendezvous metadata");
    assert_eq!(handoff["bootstrap_host"], "127.0.0.1");
    assert_eq!(handoff["bootstrap_port"], 8_998);
    assert!(handoff["bootstrap_room"].is_number());
    assert_eq!(prefill_outputs[1].finish_reason, Some(FinishReason::Length));
    assert!(prefill_outputs[1].disaggregated_params.is_none());

    let mut decode_request = request(3);
    decode_request.prefill_result = Some(PrefillResult {
        disaggregated_params: handoff,
        prompt_tokens_details: None,
    });
    let decode_outputs = collect(&decode, decode_request).await;
    assert_eq!(decode_outputs.len(), 3);
    assert_eq!(
        decode_outputs.last().unwrap().finish_reason,
        Some(FinishReason::Length)
    );
    prefill_server.close(prefill).await;
    decode_server.close(decode).await;
}

#[tokio::test]
async fn sidecar_abort_releases_mocker_work() {
    let mut args = fast_engine_args();
    args.speedup_ratio = 0.001;
    let server = RunningServer::start(ServerMode::Aggregated, args).await;
    let engine = sidecar(&server.endpoint, DisaggregationMode::Aggregated).await;
    engine.start(0).await.unwrap();

    let context = dynamo_backend_common::testing::mock_context();
    let stream = engine
        .generate(
            request(10_000),
            GenerateContext::new(Arc::clone(&context), None),
        )
        .await
        .unwrap();
    let consumer = tokio::spawn(async move { stream.collect::<Vec<_>>().await });

    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while server.service.active_request_count() == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("request should reach the Mocker scheduler");

    engine.abort(Arc::clone(&context)).await;
    testkit::wait_idle(server.service.metrics_receiver(), || {
        server.service.active_request_count()
    })
    .await;
    consumer.abort();
    let _ = consumer.await;
    server.close(engine).await;
}

#[tokio::test]
async fn request_cancellation_is_isolated_and_shutdown_reaches_grpc_streams() {
    let server = RunningServer::start(ServerMode::Aggregated, fast_engine_args()).await;
    let engine = sidecar(&server.endpoint, DisaggregationMode::Aggregated).await;
    engine.start(0).await.unwrap();

    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        let first_context = dynamo_backend_common::testing::mock_context();
        let mut streams = Vec::new();
        for context in [
            first_context.clone(),
            dynamo_backend_common::testing::mock_context(),
        ] {
            let mut stream = engine
                .generate(request(64), GenerateContext::new(context, None))
                .await
                .unwrap();
            let first = stream.next().await.unwrap().unwrap();
            assert!(!first.token_ids.is_empty());
            assert!(first.finish_reason.is_none());
            streams.push(stream);
        }

        first_context.stop_generating();
        assert_eq!(
            streams[0].next().await.unwrap().unwrap().finish_reason,
            Some(FinishReason::Cancelled)
        );
        assert!(streams[0].next().await.is_none());
        for stream in &mut streams[1..] {
            let next = stream.next().await.unwrap().unwrap();
            assert!(next.finish_reason.is_none());
        }

        engine.cleanup().await.unwrap();
        for stream in &mut streams[1..] {
            assert_eq!(
                stream.next().await.unwrap().unwrap().finish_reason,
                Some(FinishReason::Cancelled)
            );
            assert!(stream.next().await.is_none());
        }
        let late = collect(&engine, request(1)).await;
        assert_eq!(late.len(), 1);
        assert_eq!(late[0].finish_reason, Some(FinishReason::Cancelled));
    })
    .await
    .expect("request cancellation and engine shutdown must finish promptly");
    server.close(engine).await;
}
