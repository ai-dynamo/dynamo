// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Drives the real TensorRT-LLM sidecar against the Mocker server over a real
//! socket. Nothing here stubs the sidecar: if the mocker violates the OpenEngine
//! contract the sidecar actually enforces, these fail.

use dynamo_backend_common::{
    DisaggregationMode, FinishReason, GenerateContext, LLMEngine, OutputOptions, PrefillResult,
    PreprocessedRequest, SamplingOptions, StopConditions,
};
use dynamo_mocker::common::protocols::{EngineType, MockEngineArgs};
use dynamo_trtllm_mocker::{MockerServerConfig, ServerMode, TrtllmMockerService};
use dynamo_trtllm_sidecar::TrtllmSidecarEngine;
use dynamo_trtllm_sidecar::proto::control_server::ControlServer;
use dynamo_trtllm_sidecar::proto::inference_server::InferenceServer;
use futures::StreamExt;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio_stream::wrappers::TcpListenerStream;

const MODEL: &str = "mocker-model";

struct RunningServer {
    endpoint: String,
    service: TrtllmMockerService,
    shutdown: Option<oneshot::Sender<()>>,
}

impl RunningServer {
    async fn start(mode: ServerMode, engine_args: MockEngineArgs) -> Self {
        Self::start_with(
            MockerServerConfig {
                mode,
                model: MODEL.to_string(),
                context_length: 4_096,
                ..Default::default()
            },
            engine_args,
        )
        .await
    }

    async fn start_with(config: MockerServerConfig, engine_args: MockEngineArgs) -> Self {
        let service = TrtllmMockerService::new(config, engine_args).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (shutdown, shutdown_rx) = oneshot::channel();
        let inference_service = service.clone();
        let control_service = service.clone();
        let (health, health_service) = tonic_health::server::health_reporter();
        health
            .set_serving::<ControlServer<TrtllmMockerService>>()
            .await;
        health
            .set_serving::<InferenceServer<TrtllmMockerService>>()
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
        .engine_type(EngineType::Trtllm)
        .block_size(4)
        .num_gpu_blocks(4096)
        .max_num_seqs(Some(64))
        .max_num_batched_tokens(Some(1024))
        .speedup_ratio(0.0)
        .dp_size(1)
        .build()
        .unwrap()
}

/// `--model-path` is mandatory and becomes `GenerateRequest.model`, so it has to
/// match the server's `--model` or every request is NOT_FOUND.
async fn sidecar(endpoint: &str, mode: DisaggregationMode) -> TrtllmSidecarEngine {
    let mut argv = vec![
        "dynamo-trtllm-sidecar".to_string(),
        "--trtllm-endpoint".to_string(),
        endpoint.to_string(),
        "--model-path".to_string(),
        MODEL.to_string(),
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
    tokio::task::spawn_blocking(move || TrtllmSidecarEngine::from_args(argv))
        .await
        .unwrap()
        .unwrap()
        .0
}

/// Note the absence of `prompt_logprobs`: the TensorRT-LLM sidecar rejects it
/// client-side, so a request carrying it never reaches the server.
fn request(max_tokens: u32) -> PreprocessedRequest {
    PreprocessedRequest::builder()
        .model(MODEL.to_string())
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
            ..Default::default()
        })
        .build()
        .unwrap()
}

async fn collect(
    engine: &TrtllmSidecarEngine,
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
    let engine = sidecar(&server.endpoint, DisaggregationMode::Aggregated).await;
    // Starting at all proves Control.GetModelInfo returned a usable context
    // length; the sidecar refuses to start otherwise.
    engine.start(0).await.unwrap();

    // Three token deltas plus the terminal chunk the `finished` event becomes.
    let outputs = collect(&engine, request(3)).await;

    // What the sidecar put on the wire, not just what it made of the reply:
    // build_generate_request is where the sidecar's mapping logic lives.
    let sent = server.service.received_requests();
    assert_eq!(sent.len(), 1);
    let sent = &sent[0];
    assert_eq!(sent.model, MODEL);
    assert!(matches!(
        sent.input,
        Some(dynamo_trtllm_sidecar::proto::generate_request::Input::TokenIds(_))
    ));
    assert_eq!(sent.stopping.as_ref().unwrap().max_tokens, Some(3));
    assert_eq!(sent.stopping.as_ref().unwrap().ignore_eos, Some(true));
    assert_eq!(sent.sampling.as_ref().unwrap().temperature, Some(0.0));
    assert_eq!(sent.sampling.as_ref().unwrap().num_sequences, Some(1));
    assert_eq!(
        sent.response.as_ref().unwrap().return_output_logprobs,
        Some(true)
    );
    assert!(sent.extra.is_none());

    assert_eq!(outputs.len(), 4);
    let (deltas, terminal) = outputs.split_at(3);
    assert!(deltas.iter().all(|output| output.token_ids.len() == 1));
    assert!(
        deltas
            .iter()
            .all(|output| output.log_probs.as_ref().unwrap().len() == 1)
    );
    assert!(
        deltas
            .iter()
            .all(|output| output.top_logprobs.as_ref().unwrap()[0].len() == 2)
    );
    let terminal = &terminal[0];
    assert!(terminal.token_ids.is_empty());
    assert_eq!(terminal.finish_reason, Some(FinishReason::Length));
    let usage = terminal.completion_usage.as_ref().unwrap();
    assert_eq!((usage.prompt_tokens, usage.completion_tokens), (4, 3));
    assert_eq!(server.service.active_request_count(), 0);
}

#[tokio::test]
async fn sidecar_start_fails_when_the_model_is_not_served() {
    let server = RunningServer::start_with(
        MockerServerConfig {
            model: "some-other-model".to_string(),
            context_length: 4_096,
            ..Default::default()
        },
        fast_engine_args(),
    )
    .await;
    let engine = sidecar(&server.endpoint, DisaggregationMode::Aggregated).await;
    assert!(engine.start(0).await.is_err());
}

/// The flagship: a real prefill sidecar hands off to a real decode sidecar, and
/// the decode server verifies the payload arrived byte-for-byte.
#[tokio::test]
async fn prefill_handoff_round_trips_through_a_decode_server() {
    let prefill_server = RunningServer::start(ServerMode::Prefill, fast_engine_args()).await;
    let decode_server = RunningServer::start(ServerMode::Decode, fast_engine_args()).await;
    let prefill = sidecar(&prefill_server.endpoint, DisaggregationMode::Prefill).await;
    let decode = sidecar(&decode_server.endpoint, DisaggregationMode::Decode).await;
    prefill.start(0).await.unwrap();
    decode.start(1).await.unwrap();

    let outputs = collect(&prefill, request(8)).await;
    assert_eq!(outputs.len(), 1);
    let prefill_output = &outputs[0];
    assert!(prefill_output.token_ids.is_empty());
    // The frontend's prefill router treats any other terminal reason as
    // "already complete" and never runs the decode leg.
    assert_eq!(prefill_output.finish_reason, Some(FinishReason::Length));
    let usage = prefill_output.completion_usage.as_ref().unwrap();
    assert_eq!((usage.prompt_tokens, usage.completion_tokens), (4, 0));

    let sent = prefill_server.service.received_requests();
    assert_eq!(sent.len(), 1);
    // The mocker enforces this, so the assertion is belt-and-braces -- but it
    // names the sidecar behaviour the rest of this test depends on.
    assert_eq!(sent[0].stopping.as_ref().unwrap().max_tokens, Some(1));
    assert!(sent[0].extra.is_some());
    assert!(
        sent[0]
            .kv
            .as_ref()
            .and_then(|kv| kv.session.as_ref())
            .is_none()
    );

    let handoff = prefill_output.disaggregated_params.clone().unwrap();
    assert!(
        handoff["session_id"]
            .as_str()
            .unwrap()
            .starts_with("mocker-prefill-")
    );
    assert_eq!(handoff["transfer_backend"], "MOCKER");
    assert_eq!(handoff["endpoints"][0]["port"], 5600);
    assert_eq!(handoff["dp_rank"], 0);
    // Opaque attributes the sidecar cannot interpret must survive verbatim.
    assert!(handoff["attributes"]["mocker_request_id"].is_string());
    assert!(handoff["attributes"]["mocker_first_gen_tokens"].is_array());
    // A whole double must arrive as an integer, a fractional one unrounded.
    assert_eq!(handoff["attributes"]["mocker_prompt_tokens"], 4);
    assert_eq!(handoff["attributes"]["mocker_ttft_ms"], 12.5);

    let mut decode_request = request(3);
    decode_request.prefill_result = Some(PrefillResult {
        disaggregated_params: handoff,
        prompt_tokens_details: None,
    });
    let outputs = collect(&decode, decode_request).await;
    assert_eq!(outputs.len(), 4);
    assert_eq!(
        outputs.last().unwrap().finish_reason,
        Some(FinishReason::Length)
    );

    let sent = decode_server.service.received_requests();
    let replayed = sent[0]
        .kv
        .as_ref()
        .and_then(|kv| kv.session.as_ref())
        .expect("the decode leg must carry the session");
    assert_eq!(replayed.transfer_backend, "MOCKER");
    // The context phase's first token is the decode leg's first output, as a
    // real generation worker would replay it.
    assert_eq!(
        outputs[0].token_ids[0],
        handoff_first_token(replayed).expect("handoff carries a first token")
    );
}

fn handoff_first_token(session: &dynamo_trtllm_sidecar::proto::KvSessionRef) -> Option<u32> {
    use prost_types::value::Kind;
    let attributes = session.attributes_struct.as_ref()?;
    let Some(Kind::ListValue(list)) = attributes
        .fields
        .get("mocker_first_gen_tokens")?
        .kind
        .as_ref()
    else {
        return None;
    };
    match list.values.first()?.kind.as_ref()? {
        Kind::NumberValue(value) => Some(*value as u32),
        _ => None,
    }
}

/// The decode server must reject a handoff that lost an opaque field, or the
/// round trip above would prove nothing about verbatim forwarding.
#[tokio::test]
async fn decode_rejects_a_handoff_with_a_dropped_opaque_field() {
    let prefill_server = RunningServer::start(ServerMode::Prefill, fast_engine_args()).await;
    let decode_server = RunningServer::start(ServerMode::Decode, fast_engine_args()).await;
    let prefill = sidecar(&prefill_server.endpoint, DisaggregationMode::Prefill).await;
    let decode = sidecar(&decode_server.endpoint, DisaggregationMode::Decode).await;
    prefill.start(0).await.unwrap();
    decode.start(1).await.unwrap();

    let outputs = collect(&prefill, request(8)).await;
    let mut handoff = outputs[0].disaggregated_params.clone().unwrap();
    handoff["attributes"]
        .as_object_mut()
        .unwrap()
        .remove("mocker_request_id");

    let mut decode_request = request(3);
    decode_request.prefill_result = Some(PrefillResult {
        disaggregated_params: handoff,
        prompt_tokens_details: None,
    });
    let context = dynamo_backend_common::testing::mock_context();
    let failed = match decode
        .generate(decode_request, GenerateContext::new(context, None))
        .await
    {
        Err(_) => true,
        Ok(stream) => {
            stream
                .map(|item| item.is_err())
                .any(|failed| async move { failed })
                .await
        }
    };
    assert!(failed, "a mangled handoff must fail the decode request");
}

#[tokio::test]
async fn dropping_the_sidecar_stream_cancels_mocker_work() {
    let server = RunningServer::start(
        ServerMode::Aggregated,
        MockEngineArgs::builder()
            .engine_type(EngineType::Trtllm)
            .block_size(4)
            .num_gpu_blocks(4096)
            .max_num_seqs(Some(64))
            .max_num_batched_tokens(Some(1024))
            .speedup_ratio(0.1)
            .dp_size(1)
            .build()
            .unwrap(),
    )
    .await;
    let engine = sidecar(&server.endpoint, DisaggregationMode::Aggregated).await;
    engine.start(0).await.unwrap();

    let context = dynamo_backend_common::testing::mock_context();
    let mut stream = engine
        .generate(request(2_000), GenerateContext::new(context, None))
        .await
        .unwrap();
    let _first = stream.next().await.unwrap().unwrap();
    drop(stream);

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        if server.service.active_request_count() == 0 {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "dropping the stream must cancel the scheduler's work"
        );
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
}

#[tokio::test]
async fn capacity_rejection_surfaces_as_a_sidecar_error() {
    let server = RunningServer::start(
        ServerMode::Aggregated,
        MockEngineArgs::builder()
            .engine_type(EngineType::Trtllm)
            .block_size(4)
            .num_gpu_blocks(1)
            .max_num_seqs(Some(8))
            .max_num_batched_tokens(Some(64))
            .speedup_ratio(0.0)
            .dp_size(1)
            .build()
            .unwrap(),
    )
    .await;
    let engine = sidecar(&server.endpoint, DisaggregationMode::Aggregated).await;
    engine.start(0).await.unwrap();

    let mut oversized = request(4);
    oversized.token_ids = vec![1, 2, 3, 4, 5];
    let context = dynamo_backend_common::testing::mock_context();
    let stream = engine
        .generate(oversized, GenerateContext::new(context, None))
        .await
        .unwrap();
    let failed = stream
        .map(|item| item.is_err())
        .any(|failed| async move { failed })
        .await;
    assert!(failed, "an in-band EngineError must fail the request");
}
