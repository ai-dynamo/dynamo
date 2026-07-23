// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_backend_common::{
    DisaggregationMode, FinishReason, GenerateContext, LLMEngine, OutputOptions, PrefillResult,
    PreprocessedRequest, SamplingOptions, StopConditions,
};
use dynamo_mocker::common::protocols::MockEngineArgs;
use dynamo_vllm_mocker::{MockerServerConfig, ServerMode, VllmMockerService};
use dynamo_vllm_sidecar::VllmSidecarEngine;
use dynamo_vllm_sidecar::proto::generate_server::GenerateServer;
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
        let server_service = service.clone();
        tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(GenerateServer::new(server_service))
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

fn sidecar(endpoint: &str, mode: DisaggregationMode) -> VllmSidecarEngine {
    let mut argv = vec![
        "dynamo-vllm-sidecar".to_string(),
        "--vllm-endpoint".to_string(),
        endpoint.to_string(),
        "--model-path".to_string(),
        "mocker-model".to_string(),
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
    VllmSidecarEngine::from_args(Some(argv)).unwrap().0
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
async fn sidecar_streams_mocker_tokens_logprobs_and_usage() {
    let server = RunningServer::start(ServerMode::Aggregated, fast_engine_args()).await;
    let engine = sidecar(&server.endpoint, DisaggregationMode::Aggregated);
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
async fn prefill_handoff_round_trips_through_a_decode_server() {
    let prefill_server = RunningServer::start(ServerMode::Prefill, fast_engine_args()).await;
    let decode_server = RunningServer::start(ServerMode::Decode, fast_engine_args()).await;
    let prefill = sidecar(&prefill_server.endpoint, DisaggregationMode::Prefill);
    let decode = sidecar(&decode_server.endpoint, DisaggregationMode::Decode);
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
}

#[tokio::test]
async fn dropping_sidecar_stream_cancels_mocker_work() {
    let mut args = fast_engine_args();
    args.speedup_ratio = 0.1;
    let server = RunningServer::start(ServerMode::Aggregated, args).await;
    let engine = sidecar(&server.endpoint, DisaggregationMode::Aggregated);
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
