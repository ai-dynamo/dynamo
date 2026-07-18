// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_backend_common::{
    DisaggregationMode, FinishReason, GenerateContext, LLMEngine, OutputOptions, PrefillResult,
    PreprocessedRequest, SamplingOptions, StopConditions,
};
use dynamo_mocker::common::protocols::{EngineType, MockEngineArgs};
use dynamo_sglang_grpc::sglang_service_server::SglangServiceServer;
use dynamo_sglang_mocker::{MockerServerConfig, ServerMode, SglangMockerService};
use dynamo_sglang_sidecar::SglangSidecarEngine;
use futures::StreamExt;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio_stream::wrappers::TcpListenerStream;

struct RunningServer {
    endpoint: String,
    service: SglangMockerService,
    shutdown: Option<oneshot::Sender<()>>,
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
        tokio::spawn(async move {
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
        "--sglang-endpoint".to_string(),
        endpoint.to_string(),
        "--sglang-connections".to_string(),
        "1".to_string(),
        "--connect-timeout-secs".to_string(),
        "1".to_string(),
        "--health-poll-interval-secs".to_string(),
        "1".to_string(),
        "--health-deadline-secs".to_string(),
        "5".to_string(),
    ];
    if mode.is_prefill() {
        argv.extend(["--bootstrap-host".to_string(), "127.0.0.1".to_string()]);
    }
    tokio::task::spawn_blocking(move || SglangSidecarEngine::from_args(Some(argv)).unwrap().0)
        .await
        .unwrap()
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
    engine: &SglangSidecarEngine,
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
async fn sidecar_discovers_and_streams_mocker_tokens_logprobs_and_usage() {
    let server = RunningServer::start(ServerMode::Aggregated, fast_engine_args()).await;
    let engine = sidecar(&server.endpoint, DisaggregationMode::Aggregated).await;
    let config = engine.start(0).await.unwrap();
    let registration = config.llm.unwrap();
    assert_eq!(registration.context_length, Some(32_768));
    assert_eq!(registration.kv_cache_block_size, Some(4));
    assert_eq!(registration.total_kv_blocks, Some(4_096));

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
            .all(|output| output.top_logprobs.as_ref().unwrap()[0].len() == 2)
    );
    let terminal = outputs.last().unwrap();
    assert_eq!(terminal.finish_reason, Some(FinishReason::Length));
    let usage = terminal.completion_usage.as_ref().unwrap();
    assert_eq!((usage.prompt_tokens, usage.completion_tokens), (4, 3));
    assert!(terminal.engine_data.as_ref().unwrap()["prompt_logprobs"].is_array());
    assert_eq!(server.service.active_request_count(), 0);
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
        .expect("prefill response should carry SGLang rendezvous metadata");
    assert_eq!(handoff["bootstrap_host"], "127.0.0.1");
    assert_eq!(handoff["bootstrap_port"], 8_998);
    assert!(handoff["bootstrap_room"].is_number());

    let expected_handoff = handoff.clone();
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
    let observed = decode_server
        .service
        .last_disaggregated_params()
        .expect("decode server should observe rendezvous metadata");
    assert_eq!(
        observed.bootstrap_host,
        expected_handoff["bootstrap_host"].as_str().unwrap()
    );
    assert_eq!(
        i64::from(observed.bootstrap_port),
        expected_handoff["bootstrap_port"].as_i64().unwrap()
    );
    assert_eq!(
        observed.bootstrap_room,
        expected_handoff["bootstrap_room"].as_i64().unwrap()
    );
}

#[tokio::test]
async fn sidecar_abort_releases_mocker_work() {
    let mut args = fast_engine_args();
    args.speedup_ratio = 0.001;
    let server = RunningServer::start(ServerMode::Aggregated, args).await;
    let engine = Arc::new(sidecar(&server.endpoint, DisaggregationMode::Aggregated).await);
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
    .expect("Abort should release scheduler work promptly");
    consumer.abort();
}
