// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use tonic_health_v14 as tonic_health;
use tonic_v14 as tonic;

use std::sync::Arc;

use dynamo_backend_common::{
    AsyncEngineContext, DisaggregationMode, FinishReason, GenerateContext, LLMEngine,
    OutputOptions, PrefillResult, PreprocessedRequest, SamplingOptions, StopConditions, StopReason,
};
use dynamo_mocker::common::protocols::MockEngineArgs;
use dynamo_vllm_mocker::{MockerServerConfig, ServerMode, VllmMockerService};
use dynamo_vllm_sidecar::VllmSidecarEngine;
use dynamo_vllm_sidecar::proto::AbortRequest;
use dynamo_vllm_sidecar::proto::control_client::ControlClient;
use dynamo_vllm_sidecar::proto::control_server::ControlServer;
use dynamo_vllm_sidecar::proto::inference_server::InferenceServer;
use futures::StreamExt;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio_stream::wrappers::TcpListenerStream;

struct RunningServer {
    endpoint: String,
    service: VllmMockerService,
    shutdown: Option<oneshot::Sender<()>>,
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
        tokio::spawn(async move {
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
        }
    }
}

impl Drop for RunningServer {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
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

fn request(max_tokens: u32) -> PreprocessedRequest {
    PreprocessedRequest::builder()
        .model("mocker-model".to_string())
        .token_ids(vec![11, 22, 33, 44])
        .stop_conditions(StopConditions {
            max_tokens: Some(max_tokens),
            ignore_eos: Some(true),
            ..Default::default()
        })
        .sampling_options(SamplingOptions {
            temperature: Some(0.0),
            ..Default::default()
        })
        .output_options(OutputOptions {
            logprobs: Some(2),
            prompt_logprobs: Some(1),
            ..Default::default()
        })
        .build()
        .unwrap()
}

async fn collect(
    engine: &VllmSidecarEngine,
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
    engine: &VllmSidecarEngine,
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
async fn sidecar_streams_mocker_tokens_logprobs_and_usage() {
    let server = RunningServer::start(ServerMode::Aggregated, fast_engine_args()).await;
    let engine = sidecar(&server.endpoint, DisaggregationMode::Aggregated).await;
    engine.start(0).await.unwrap();

    let outputs = collect(&engine, request(3)).await;
    assert_eq!(outputs.len(), 3);
    assert!(outputs.iter().all(|output| output.token_ids.len() == 1));
    assert!(
        outputs
            .iter()
            .all(|output| output.log_probs.as_ref().unwrap().len() == 1)
    );
    assert!(
        outputs
            .iter()
            .all(|output| output.top_logprobs.as_ref().unwrap()[0].len() == 3)
    );
    let terminal = outputs.last().unwrap();
    assert_eq!(terminal.finish_reason, Some(FinishReason::Length));
    let usage = terminal.completion_usage.as_ref().unwrap();
    assert_eq!((usage.prompt_tokens, usage.completion_tokens), (4, 3));
    assert!(terminal.engine_data.as_ref().unwrap()["prompt_logprobs"].is_array());
    assert_eq!(server.service.active_request_count(), 0);
}

#[tokio::test]
async fn sidecar_preserves_vllm_stop_token_controls() {
    let server = RunningServer::start(ServerMode::Aggregated, fast_engine_args()).await;
    let engine = sidecar(&server.endpoint, DisaggregationMode::Aggregated).await;
    engine.start(0).await.unwrap();
    let context = dynamo_backend_common::testing::mock_context();
    let baseline = collect_with_context(&engine, request(4), Arc::clone(&context)).await;
    let baseline_tokens: Vec<_> = baseline
        .iter()
        .flat_map(|output| &output.token_ids)
        .copied()
        .collect();
    assert_eq!(baseline_tokens.len(), 4);
    let stop_token = baseline_tokens[0];

    for (max_tokens, min_tokens, is_ignore_eos, expected_tokens, expected_finish) in [
        (4, 0, false, 1, FinishReason::Stop),
        (1, 0, false, 1, FinishReason::Stop),
        (4, 4, false, 4, FinishReason::Length),
        (4, 0, true, 1, FinishReason::Stop),
        (1, 0, true, 1, FinishReason::Stop),
        (1, 1, true, 1, FinishReason::Length),
    ] {
        let mut stopped = request(max_tokens);
        stopped.stop_conditions.min_tokens = Some(min_tokens);
        stopped.stop_conditions.ignore_eos = Some(is_ignore_eos);
        stopped.stop_conditions.stop_token_ids = Some(vec![stop_token]);
        let outputs = collect_with_context(&engine, stopped, Arc::clone(&context)).await;
        let tokens: Vec<_> = outputs
            .iter()
            .flat_map(|output| &output.token_ids)
            .copied()
            .collect();
        assert_eq!(tokens, baseline_tokens[..expected_tokens]);
        let terminal = outputs.last().unwrap();
        assert_eq!(
            outputs
                .iter()
                .filter(|output| output.finish_reason.is_some())
                .count(),
            1
        );
        assert_eq!(terminal.finish_reason.as_ref(), Some(&expected_finish));
        assert_eq!(
            terminal.stop_reason,
            (expected_finish == FinishReason::Stop)
                .then_some(StopReason::Int(i64::from(stop_token)))
        );
        let usage = terminal.completion_usage.as_ref().unwrap();
        assert_eq!(
            (usage.prompt_tokens, usage.completion_tokens as usize),
            (4, expected_tokens)
        );
        assert_eq!(
            usage.total_tokens,
            usage.prompt_tokens + usage.completion_tokens
        );
        assert_eq!(server.service.active_request_count(), 0);
    }
}

#[tokio::test]
async fn prefill_handoff_round_trips_through_a_decode_server() {
    let prefill_server = RunningServer::start(ServerMode::Prefill, fast_engine_args()).await;
    let decode_server = RunningServer::start(ServerMode::Decode, fast_engine_args()).await;
    let prefill = sidecar(&prefill_server.endpoint, DisaggregationMode::Prefill).await;
    let decode = sidecar(&decode_server.endpoint, DisaggregationMode::Decode).await;
    prefill.start(0).await.unwrap();
    decode.start(1).await.unwrap();

    let mut prefill_request = request(3);
    prefill_request.token_ids = vec![11, 22, 33, 44, 55].into();
    let prefill_outputs = collect(&prefill, prefill_request.clone()).await;
    assert_eq!(prefill_outputs.len(), 1);
    assert!(prefill_outputs[0].token_ids.is_empty());
    let handoff = prefill_outputs[0]
        .disaggregated_params
        .clone()
        .expect("prefill response should carry an opaque KV handoff");
    assert_eq!(handoff["do_remote_prefill"], true);
    assert!(handoff["remote_engine_id"].is_string());
    assert!(handoff["remote_request_id"].is_string());
    let groups = handoff["remote_block_ids"].as_array().unwrap();
    assert_eq!(groups.len(), 1);
    let blocks = groups[0].as_array().unwrap();
    assert_eq!(blocks.len(), 2);
    assert!(blocks.iter().all(|block| block.as_u64().is_some()));
    // The non-rendezvous sentinel proves the sidecar preserved opaque handoff
    // fields rather than reconstructing only the keys it recognizes.
    assert!(
        handoff["mocker_request_id"].is_string(),
        "sidecar must forward opaque KV-transfer fields verbatim"
    );

    let mut decode_request = prefill_request;
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
}

#[tokio::test]
async fn dropping_sidecar_stream_cancels_mocker_work() {
    let mut args = fast_engine_args();
    args.speedup_ratio = 0.1;
    let server = RunningServer::start(ServerMode::Aggregated, args).await;
    let engine = sidecar(&server.endpoint, DisaggregationMode::Aggregated).await;
    engine.start(0).await.unwrap();

    let context = dynamo_backend_common::testing::mock_context();
    let mut stream = engine
        .generate(
            request(10_000),
            GenerateContext::new(Arc::clone(&context), None),
        )
        .await
        .unwrap();
    let first = stream.next().await.unwrap().unwrap();
    assert!(first.finish_reason.is_none());
    context.stop_generating();
    let terminal = stream.next().await.unwrap().unwrap();
    assert_eq!(terminal.finish_reason, Some(FinishReason::Cancelled));
    drop(stream);

    let mut metrics = server.service.metrics_receiver();
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            let snapshot = metrics.borrow_and_update().clone();
            if server.service.active_request_count() == 0
                && snapshot.running_requests == 0
                && snapshot.waiting_requests == 0
            {
                break;
            }
            metrics.changed().await.unwrap();
        }
    })
    .await
    .expect("dropping the gRPC stream should cancel scheduler work promptly");
}

#[tokio::test]
async fn native_abort_maps_to_cancelled_and_allows_recovery() {
    let mut args = fast_engine_args();
    args.speedup_ratio = 0.1;
    let server = RunningServer::start(ServerMode::Aggregated, args).await;
    let engine = sidecar(&server.endpoint, DisaggregationMode::Aggregated).await;
    engine.start(0).await.unwrap();

    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        let context = dynamo_backend_common::testing::mock_context();
        let mut stream = engine
            .generate(
                request(10_000),
                GenerateContext::new(Arc::clone(&context), None),
            )
            .await
            .unwrap();
        let first = stream.next().await.unwrap().unwrap();
        assert!(!first.token_ids.is_empty());
        assert!(first.finish_reason.is_none());
        ControlClient::connect(server.endpoint.clone())
            .await
            .unwrap()
            .abort(AbortRequest {
                request_ids: vec![context.id().to_owned()],
            })
            .await
            .unwrap();
        let mut outputs = vec![first];
        outputs.extend(stream.map(|item| item.unwrap()).collect::<Vec<_>>().await);
        let terminal = outputs.last().unwrap();
        assert_eq!(
            outputs
                .iter()
                .filter(|output| output.finish_reason.is_some())
                .count(),
            1
        );
        assert_eq!(terminal.finish_reason, Some(FinishReason::Cancelled));
        let usage = terminal.completion_usage.as_ref().unwrap();
        let generated: usize = outputs.iter().map(|output| output.token_ids.len()).sum();
        assert_eq!(
            (usage.prompt_tokens, usage.completion_tokens as usize),
            (4, generated)
        );
        assert_eq!(
            usage.total_tokens,
            usage.prompt_tokens + usage.completion_tokens
        );

        let mut metrics = server.service.metrics_receiver();
        loop {
            let snapshot = metrics.borrow_and_update().clone();
            if server.service.active_request_count() == 0
                && snapshot.running_requests == 0
                && snapshot.waiting_requests == 0
            {
                break;
            }
            metrics.changed().await.unwrap();
        }
        let recovered = collect(&engine, request(1)).await;
        assert_eq!(recovered[0].token_ids.len(), 1);
        assert_eq!(
            recovered.last().unwrap().finish_reason,
            Some(FinishReason::Length)
        );
        assert_eq!(server.service.active_request_count(), 0);
    })
    .await
    .expect("native Abort should cancel scheduler work and permit another request");
}

#[path = "../../tests/common/mod.rs"]
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
}
