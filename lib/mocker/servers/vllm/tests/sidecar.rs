// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use tonic_health_v14 as tonic_health;
use tonic_v14 as tonic;

use std::time::Duration;

use dynamo_backend_common::{
    DisaggregationMode, FinishReason, GenerateContext, LLMEngine, PrefillResult,
    PreprocessedRequest,
};
use dynamo_mocker::common::protocols::MockEngineArgs;
use dynamo_sidecar_testkit::{self as testkit, request};
use dynamo_vllm_mocker::{MockerServerConfig, ServerMode, VllmMockerService};
use dynamo_vllm_sidecar::VllmSidecarEngine;
use dynamo_vllm_sidecar::proto::control_server::ControlServer;
use dynamo_vllm_sidecar::proto::inference_server::InferenceServer;
use futures::StreamExt;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tokio::time::timeout;
use tokio_stream::wrappers::TcpListenerStream;

struct RunningServer {
    endpoint: String,
    service: VllmMockerService,
    shutdown: Option<oneshot::Sender<()>>,
    task: JoinHandle<()>,
}

impl RunningServer {
    async fn start(mode: ServerMode, engine_args: MockEngineArgs) -> Self {
        let service = VllmMockerService::new(
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
        let inference_service = service.clone();
        let control_service = service.clone();
        let (health, health_service) = tonic_health::server::health_reporter();
        health
            .set_serving::<ControlServer<VllmMockerService>>()
            .await;
        health
            .set_serving::<InferenceServer<VllmMockerService>>()
            .await;
        let task = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(InferenceServer::new(inference_service))
                .add_service(ControlServer::new(control_service))
                .add_service(health_service)
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
        .block_size(4)
        .num_gpu_blocks(4096)
        .max_num_seqs(Some(64))
        .max_num_batched_tokens(Some(1024))
        .speedup_ratio(0.0)
        .dp_size(1)
        .build()
        .unwrap()
}

async fn sidecar(endpoint: &str, mode: DisaggregationMode) -> VllmSidecarEngine {
    let mut argv = vec![
        "dynamo-vllm-sidecar".to_string(),
        "--grpc-endpoint".to_string(),
        endpoint.to_string(),
        "--grpc-connections".to_string(),
        "1".to_string(),
        "--grpc-startup-deadline-secs".to_string(),
        "5".to_string(),
        "--grpc-connect-attempt-timeout-secs".to_string(),
        "1".to_string(),
    ];
    if mode != DisaggregationMode::Aggregated {
        argv.extend(["--disaggregation-mode".to_string(), mode.to_string()]);
    }
    tokio::task::spawn_blocking(move || VllmSidecarEngine::from_args(Some(argv)))
        .await
        .unwrap()
        .unwrap()
        .0
}

async fn native_output(endpoint: &str, request_id: &str) -> (Vec<u32>, Vec<f64>) {
    use dynamo_vllm_sidecar::proto as pb;

    timeout(Duration::from_secs(5), async {
        let mut client = pb::inference_client::InferenceClient::connect(endpoint.to_string())
            .await
            .unwrap();
        let mut stream = client
            .generate_stream(pb::GenerateRequest {
                request_id: request_id.to_string(),
                prompt: Some(pb::generate_request::Prompt::TokenIds(pb::TokenIds {
                    ids: vec![11, 22, 33, 44],
                })),
                stopping: Some(pb::StoppingCriteria {
                    max_new_tokens: 3,
                    ..Default::default()
                }),
                response: Some(pb::ResponseOptions {
                    output_token_ids: true,
                    output_logprobs: true,
                    ..Default::default()
                }),
                ..Default::default()
            })
            .await
            .unwrap()
            .into_inner();
        let mut tokens = Vec::new();
        let mut logprobs = Vec::new();
        while let Some(response) = stream.message().await.unwrap() {
            if let Some(output) = response.outputs {
                assert_eq!(output.token_ids.len(), output.logprobs.len());
                tokens.extend(output.token_ids);
                logprobs.extend(output.logprobs.into_iter().map(f64::from));
            }
        }
        assert_eq!(tokens.len(), 3);
        assert_eq!(logprobs.len(), 3);
        (tokens, logprobs)
    })
    .await
    .expect("native vLLM reference stream timed out")
}

async fn collect(
    engine: &VllmSidecarEngine,
    request: PreprocessedRequest,
) -> Vec<dynamo_backend_common::LLMEngineOutput> {
    let context = dynamo_backend_common::testing::mock_context();
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
        engine.start(0).await.unwrap();

        let context = dynamo_backend_common::testing::mock_context();
        let (expected, expected_logprobs) = native_output(&server.endpoint, context.id()).await;
        testkit::wait_idle(server.service.metrics_receiver(), || {
            server.service.active_request_count()
        })
        .await;
        testkit::streaming(&engine, context, &expected, &expected_logprobs, 3).await;
        testkit::wait_idle(server.service.metrics_receiver(), || {
            server.service.active_request_count()
        })
        .await;
        server.close(engine).await;
    })
    .await
    .expect("vllm shared_streaming_preserves_native_tokens_logprobs_and_usage timed out");
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
    .expect("vllm shared_native_rejection_preserves_error_and_recovers timed out");
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
    .expect("vllm shared_cancellation_releases_scheduler_work_and_recovers timed out");
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
    .expect("vllm shared_consumer_drop_releases_scheduler_work_and_recovers timed out");
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
    assert_eq!(prefill_outputs.len(), 1);
    assert!(prefill_outputs[0].token_ids.is_empty());
    let handoff = prefill_outputs[0]
        .disaggregated_params
        .clone()
        .expect("prefill response should carry an opaque KV handoff");
    assert_eq!(handoff["do_remote_prefill"], true);
    assert!(handoff["remote_engine_id"].is_string());
    // The non-rendezvous sentinel proves the sidecar preserved opaque handoff
    // fields rather than reconstructing only the keys it recognizes.
    assert!(
        handoff["mocker_request_id"].is_string(),
        "sidecar must forward opaque KV-transfer fields verbatim"
    );

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

mod common;

#[tokio::test]
async fn sidecar_relays_stored_and_evicted_blocks() {
    let mut args = fast_engine_args();
    args.num_gpu_blocks = 8;
    args.max_num_seqs = Some(1);
    let block_size = u32::try_from(args.block_size).unwrap();
    let server = RunningServer::start(ServerMode::Aggregated, args).await;
    let engine = sidecar(&server.endpoint, DisaggregationMode::Aggregated).await;
    engine.start(0).await.unwrap();
    common::check_kv_events(&engine, block_size).await;
    server.close(engine).await;
}
