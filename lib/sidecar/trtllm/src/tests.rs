// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use std::net::SocketAddr;
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use dynamo_backend_common::{
    FinishReason, GenerateContext, LLMEngine, OutputOptions, PreprocessedRequest, SamplingOptions,
    StopConditions, StopReason,
};
use dynamo_sidecar_common::{GrpcEndpoint, GrpcTransportConfig};
use futures::{Stream, StreamExt};
use serde_json::json;
use tokio::net::TcpListener;
use tokio::sync::{Mutex, oneshot};
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{Request, Response, Status};

use crate::client::TrtllmClient;
use crate::convert::{ResponseState, build_generate_request};
use crate::engine::TrtllmSidecarEngine;
use crate::model::ConfiguredModel;
use crate::proto as pb;

// ---------------------------------------------------------------------------
// Fake TensorRT-LLM gRPC service
// ---------------------------------------------------------------------------

#[derive(Clone, Default)]
struct FakeTrtllm {
    requests: Arc<Mutex<Vec<pb::GenerateRequest>>>,
    aborts: Arc<Mutex<Vec<String>>>,
    peers: Arc<Mutex<Vec<SocketAddr>>>,
    reject: Arc<AtomicBool>,
    hang: Arc<AtomicBool>,
}

#[tonic::async_trait]
impl pb::trtllm_service_server::TrtllmService for FakeTrtllm {
    type GenerateStream = Pin<Box<dyn Stream<Item = Result<pb::GenerateResponse, Status>> + Send>>;

    async fn generate(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        if let Some(peer) = request.remote_addr() {
            self.peers.lock().await.push(peer);
        }
        let request = request.into_inner();
        self.requests.lock().await.push(request.clone());
        if self.reject.load(Ordering::SeqCst) {
            return Err(Status::invalid_argument("rejected by fake TensorRT-LLM"));
        }

        let request_id = request.request_id.clone();
        let prompt_tokens = request
            .tokenized
            .as_ref()
            .map(|t| t.input_token_ids.len() as u32)
            .unwrap_or(0);
        let wants_logprobs = request
            .output_config
            .as_ref()
            .and_then(|o| o.logprobs)
            .is_some();
        let hang = self.hang.load(Ordering::SeqCst);

        let stream = async_stream::try_stream! {
            let logprobs = if wants_logprobs {
                vec![pb::TokenLogprob {
                    token_id: 42,
                    logprob: -0.25,
                    top_logprobs: vec![pb::TopLogprob { token_id: 43, logprob: -0.5 }],
                }]
            } else {
                Vec::new()
            };

            yield pb::GenerateResponse {
                request_id: request_id.clone(),
                response: Some(pb::generate_response::Response::Chunk(pb::GenerateStreamChunk {
                    token_ids: vec![42],
                    sequence_index: 0,
                    prompt_tokens,
                    completion_tokens: 1,
                    cached_tokens: 0,
                    logprobs,
                })),
            };

            if hang {
                loop {
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                }
            }

            yield pb::GenerateResponse {
                request_id,
                response: Some(pb::generate_response::Response::Complete(pb::GenerateComplete {
                    output_token_ids: vec![42],
                    sequence_index: 0,
                    finish_reason: "stop".to_string(),
                    matched_stop: Some(pb::generate_complete::MatchedStop::MatchedTokenId(2)),
                    prompt_tokens,
                    completion_tokens: 1,
                    cached_tokens: 0,
                    ..Default::default()
                })),
            };
        };
        Ok(Response::new(Box::pin(stream)))
    }

    async fn embed(
        &self,
        _request: Request<pb::EmbedRequest>,
    ) -> Result<Response<pb::EmbedResponse>, Status> {
        Err(Status::unimplemented("embed is not used"))
    }

    async fn health_check(
        &self,
        _request: Request<pb::HealthCheckRequest>,
    ) -> Result<Response<pb::HealthCheckResponse>, Status> {
        Ok(Response::new(pb::HealthCheckResponse {
            status: "OK".to_string(),
        }))
    }

    async fn abort(
        &self,
        request: Request<pb::AbortRequest>,
    ) -> Result<Response<pb::AbortResponse>, Status> {
        let request_id = request.into_inner().request_id;
        self.aborts.lock().await.push(request_id.clone());
        Ok(Response::new(pb::AbortResponse {
            success: true,
            message: format!("aborted {request_id}"),
        }))
    }

    async fn get_model_info(
        &self,
        _request: Request<pb::GetModelInfoRequest>,
    ) -> Result<Response<pb::GetModelInfoResponse>, Status> {
        Ok(Response::new(pb::GetModelInfoResponse {
            model_id: "fake-model".to_string(),
            max_seq_len: 4096,
            vocab_size: 32000,
            ..Default::default()
        }))
    }

    async fn get_server_info(
        &self,
        _request: Request<pb::GetServerInfoRequest>,
    ) -> Result<Response<pb::GetServerInfoResponse>, Status> {
        Ok(Response::new(pb::GetServerInfoResponse {
            version: "1.3.0rc21".to_string(),
            backend: "pytorch".to_string(),
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
            context_parallel_size: 1,
            world_size: 1,
        }))
    }
}

struct FakeServer {
    endpoint: String,
    service: FakeTrtllm,
    shutdown: Option<oneshot::Sender<()>>,
}

impl FakeServer {
    async fn start(service: FakeTrtllm) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let address = listener.local_addr().expect("address");
        let (shutdown, shutdown_rx) = oneshot::channel();
        let server_service = service.clone();
        tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(pb::trtllm_service_server::TrtllmServiceServer::new(
                    server_service,
                ))
                .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
                    let _ = shutdown_rx.await;
                })
                .await
                .expect("serve fake TensorRT-LLM");
        });
        Self {
            endpoint: format!("http://{address}"),
            service,
            shutdown: Some(shutdown),
        }
    }
}

impl Drop for FakeServer {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn request() -> PreprocessedRequest {
    PreprocessedRequest::builder()
        .model("served-model".to_string())
        .token_ids(vec![11, 22, 33])
        .stop_conditions(StopConditions {
            max_tokens: Some(16),
            min_tokens: Some(1),
            stop: Some(vec!["done".to_string()]),
            stop_token_ids_hidden: Some(vec![2]),
            ignore_eos: Some(true),
            ..Default::default()
        })
        .sampling_options(SamplingOptions {
            temperature: Some(0.2),
            top_p: Some(0.9),
            top_k: Some(4),
            min_p: Some(0.1),
            seed: Some(123),
            presence_penalty: Some(0.3),
            frequency_penalty: Some(0.4),
            repetition_penalty: Some(1.1),
            include_stop_str_in_output: Some(true),
            guided_decoding: Some(dynamo_backend_common::GuidedDecodingOptions {
                json: Some(json!({"type": "object"})),
                ..Default::default()
            }),
            ..Default::default()
        })
        .output_options(OutputOptions {
            logprobs: Some(1),
            ..Default::default()
        })
        .build()
        .expect("request")
}

fn transport(connections: usize) -> GrpcTransportConfig {
    GrpcTransportConfig {
        connections: NonZeroUsize::new(connections).expect("nonzero connections"),
        ..Default::default()
    }
}

fn engine(endpoint: &str, connections: usize) -> TrtllmSidecarEngine {
    TrtllmSidecarEngine::new(
        GrpcEndpoint::parse(endpoint, "--trtllm-endpoint").expect("valid test endpoint"),
        transport(connections),
        ConfiguredModel {
            source: "model-source".to_string(),
            context_length: None,
        },
    )
}

async fn collect(
    engine: &TrtllmSidecarEngine,
    request: PreprocessedRequest,
) -> Vec<dynamo_backend_common::LLMEngineOutput> {
    let context = dynamo_backend_common::testing::mock_context();
    engine
        .generate(request, GenerateContext::new(context, None))
        .await
        .expect("generate")
        .map(|item| item.expect("stream item"))
        .collect()
        .await
}

// ---------------------------------------------------------------------------
// Request-building unit tests
// ---------------------------------------------------------------------------

#[test]
fn request_maps_sampling_stop_and_output_fields() {
    let proto = build_generate_request(&request(), "req-1", None).expect("build request");
    assert_eq!(proto.request_id, "req-1");
    assert_eq!(
        proto.tokenized.as_ref().unwrap().input_token_ids,
        [11, 22, 33]
    );
    assert_eq!(proto.max_tokens, 16);
    assert!(proto.streaming);
    assert!(proto.ignore_eos);
    assert!(proto.include_stop_token_in_output);
    assert_eq!(proto.stop, ["done"]);
    assert_eq!(proto.stop_token_ids, [2]);

    let sampling = proto.sampling_config.as_ref().unwrap();
    assert_eq!(sampling.top_k, Some(4));
    assert_eq!(sampling.top_p, Some(0.9));
    assert_eq!(sampling.min_p, Some(0.1));
    assert_eq!(sampling.temperature, Some(0.2));
    assert_eq!(sampling.seed, Some(123));
    assert_eq!(sampling.repetition_penalty, Some(1.1));
    assert_eq!(sampling.min_tokens, Some(1));

    let output = proto.output_config.as_ref().unwrap();
    assert_eq!(output.logprobs, Some(1));
    assert!(output.exclude_input_from_output);

    let guided = proto.guided_decoding.as_ref().unwrap();
    assert_eq!(
        guided.guide_type,
        pb::guided_decoding_params::GuideType::JsonSchema as i32
    );
    assert!(guided.guide.contains("object"));
}

#[test]
fn omitted_max_tokens_without_context_length_is_rejected() {
    let mut req = request();
    req.stop_conditions.max_tokens = None;
    let error = build_generate_request(&req, "req", None).expect_err("must require max_tokens");
    assert!(error.to_string().contains("max_tokens"));
}

#[test]
fn omitted_max_tokens_defaults_to_remaining_context() {
    let mut req = request();
    req.stop_conditions.max_tokens = None;
    // request() carries three prompt tokens ([11, 22, 33]); the default fills the
    // remaining context: max(1, context_length - prompt_len).
    let proto = build_generate_request(&req, "req", Some(100)).expect("build with fallback");
    assert_eq!(proto.max_tokens, 97);
}

#[test]
fn omitted_max_tokens_default_is_floored_at_one() {
    let mut req = request();
    req.stop_conditions.max_tokens = None;
    // Prompt already fills (or exceeds) the context: default must not underflow to 0.
    let proto = build_generate_request(&req, "req", Some(2)).expect("build with fallback");
    assert_eq!(proto.max_tokens, 1);
}

#[test]
fn top_k_all_tokens_is_left_unset() {
    let mut req = request();
    req.sampling_options.top_k = Some(-1);
    let proto = build_generate_request(&req, "req", None).expect("build");
    assert_eq!(proto.sampling_config.unwrap().top_k, None);
}

#[test]
fn oversized_logprob_count_is_rejected() {
    let mut req = request();
    req.output_options.logprobs = Some(i32::MAX as u32 + 1);
    let error =
        build_generate_request(&req, "req", None).expect_err("oversized logprobs must fail");
    assert!(error.to_string().contains("must fit in i32"));
}

// ---------------------------------------------------------------------------
// Response-conversion unit tests
// ---------------------------------------------------------------------------

#[test]
fn chunk_then_complete_produces_delta_then_terminal_usage() {
    let req = request();
    let mut state = ResponseState::new(&req);

    let chunk = pb::GenerateResponse {
        request_id: "r".to_string(),
        response: Some(pb::generate_response::Response::Chunk(
            pb::GenerateStreamChunk {
                token_ids: vec![7, 8],
                sequence_index: 0,
                prompt_tokens: 3,
                completion_tokens: 2,
                cached_tokens: 0,
                logprobs: vec![
                    pb::TokenLogprob {
                        token_id: 7,
                        logprob: -0.1,
                        top_logprobs: vec![],
                    },
                    pb::TokenLogprob {
                        token_id: 8,
                        logprob: -0.2,
                        top_logprobs: vec![],
                    },
                ],
            },
        )),
    };
    let delta = state.convert(chunk).expect("convert chunk").expect("delta");
    assert_eq!(delta.token_ids, [7, 8]);
    assert!(delta.finish_reason.is_none());
    let log_probs = delta.log_probs.as_ref().expect("log_probs");
    assert_eq!(log_probs.len(), 2);
    assert!((log_probs[0] - f64::from(-0.1_f32)).abs() < 1e-9);
    assert!((log_probs[1] - f64::from(-0.2_f32)).abs() < 1e-9);
    assert_eq!(delta.top_logprobs.as_ref().unwrap().len(), 2);

    let complete = pb::GenerateResponse {
        request_id: "r".to_string(),
        response: Some(pb::generate_response::Response::Complete(
            pb::GenerateComplete {
                output_token_ids: vec![7, 8],
                sequence_index: 0,
                finish_reason: "length".to_string(),
                prompt_tokens: 3,
                completion_tokens: 2,
                ..Default::default()
            },
        )),
    };
    let terminal = state
        .convert(complete)
        .expect("convert complete")
        .expect("terminal");
    assert!(terminal.token_ids.is_empty());
    assert_eq!(terminal.finish_reason, Some(FinishReason::Length));
    let usage = terminal.completion_usage.as_ref().expect("usage");
    assert_eq!((usage.prompt_tokens, usage.completion_tokens), (3, 2));
}

#[test]
fn unsupported_sequence_index_is_rejected() {
    let req = request();
    let mut state = ResponseState::new(&req);
    let chunk = pb::GenerateResponse {
        request_id: "r".to_string(),
        response: Some(pb::generate_response::Response::Chunk(
            pb::GenerateStreamChunk {
                token_ids: vec![1],
                sequence_index: 1,
                ..Default::default()
            },
        )),
    };
    assert!(state.convert(chunk).is_err());
}

// ---------------------------------------------------------------------------
// Integration tests against the fake server
// ---------------------------------------------------------------------------

#[tokio::test]
async fn aggregated_generation_streams_delta_then_terminal() {
    let server = FakeServer::start(FakeTrtllm::default()).await;
    let engine = engine(&server.endpoint, 2);
    let config = engine.start(0).await.expect("start");
    assert_eq!(config.model, "model-source");
    // GetModelInfo reports max_seq_len 4096.
    assert_eq!(config.llm.unwrap().context_length, Some(4096));

    let outputs = collect(&engine, request()).await;
    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].token_ids, [42]);
    assert!(outputs[0].finish_reason.is_none());
    assert_eq!(outputs[0].log_probs.as_deref(), Some(&[-0.25][..]));

    let terminal = &outputs[1];
    assert!(terminal.token_ids.is_empty());
    assert_eq!(terminal.finish_reason, Some(FinishReason::Stop));
    assert_eq!(terminal.stop_reason, Some(StopReason::Int(2)));
    let usage = terminal.completion_usage.as_ref().expect("usage");
    assert_eq!((usage.prompt_tokens, usage.completion_tokens), (3, 1));

    let requests = server.service.requests.lock().await;
    let sent = requests.first().expect("recorded request");
    assert!(sent.streaming);
    assert_eq!(
        sent.tokenized.as_ref().unwrap().input_token_ids,
        [11, 22, 33]
    );
}

#[tokio::test]
async fn grpc_request_errors_are_propagated() {
    let service = FakeTrtllm::default();
    service.reject.store(true, Ordering::SeqCst);
    let server = FakeServer::start(service).await;
    let engine = engine(&server.endpoint, 1);
    engine.start(0).await.expect("start");

    // TRT-LLM surfaces an invalid-argument on the initial response header, so
    // opening the stream fails rather than yielding an error item.
    let context = dynamo_backend_common::testing::mock_context();
    let result = engine
        .generate(request(), GenerateContext::new(context, None))
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn cancellation_yields_a_cancelled_terminal() {
    let service = FakeTrtllm::default();
    service.hang.store(true, Ordering::SeqCst);
    let server = FakeServer::start(service).await;
    let engine = engine(&server.endpoint, 1);
    engine.start(0).await.expect("start");

    let context = dynamo_backend_common::testing::mock_context();
    let mut stream = engine
        .generate(request(), GenerateContext::new(context.clone(), None))
        .await
        .expect("generate");
    let first = stream.next().await.unwrap().unwrap();
    assert_eq!(first.token_ids, [42]);
    context.stop_generating();
    let terminal = tokio::time::timeout(std::time::Duration::from_secs(2), stream.next())
        .await
        .expect("terminal within deadline")
        .unwrap()
        .unwrap();
    assert_eq!(terminal.finish_reason, Some(FinishReason::Cancelled));
}

#[tokio::test]
async fn abort_sends_the_abort_rpc_to_the_server() {
    let server = FakeServer::start(FakeTrtllm::default()).await;
    let engine = engine(&server.endpoint, 1);
    engine.start(0).await.expect("start");

    let context = dynamo_backend_common::testing::mock_context();
    let request_id = context.id().to_string();
    engine.abort(context).await;

    // The cancelled generation's ID must reach TensorRT-LLM, not just produce a
    // local terminal, or the server keeps generating.
    assert_eq!(server.service.aborts.lock().await.as_slice(), [request_id]);
}

#[tokio::test]
async fn unsupported_features_fail_before_rpc_submission() {
    let server = FakeServer::start(FakeTrtllm::default()).await;
    let engine = engine(&server.endpoint, 1);
    engine.start(0).await.expect("start");

    let mut multiple = request();
    multiple.sampling_options.n = Some(2);

    let mut beam = request();
    beam.sampling_options.use_beam_search = Some(true);

    let mut embeds = request();
    embeds.prompt_embeds = Some("encoded".to_string());

    let mut prompt_logprobs = request();
    prompt_logprobs.output_options.prompt_logprobs = Some(1);

    let mut visible_stops = request();
    visible_stops.stop_conditions.stop_token_ids_visible = Some(vec![7]);

    for unsupported in [multiple, beam, embeds, prompt_logprobs, visible_stops] {
        let context = dynamo_backend_common::testing::mock_context();
        let result = engine
            .generate(unsupported, GenerateContext::new(context, None))
            .await;
        assert!(result.is_err());
    }
    assert!(server.service.requests.lock().await.is_empty());
}

#[tokio::test]
async fn pool_uses_each_configured_connection() {
    let server = FakeServer::start(FakeTrtllm::default()).await;
    let endpoint =
        GrpcEndpoint::parse(&server.endpoint, "--trtllm-endpoint").expect("valid endpoint");
    let client = TrtllmClient::connect(&endpoint, transport(2))
        .await
        .expect("connect pool");
    assert_eq!(client.connection_count(), 2);

    for index in 0..4 {
        let mut stream = client
            .generate(pb::GenerateRequest {
                request_id: format!("request-{index}"),
                tokenized: Some(pb::TokenizedInput {
                    input_token_ids: vec![1, 2],
                    ..Default::default()
                }),
                max_tokens: 4,
                streaming: true,
                ..Default::default()
            })
            .await
            .expect("start stream");
        while stream.message().await.expect("message").is_some() {}
    }

    let ports: BTreeSet<_> = server
        .service
        .peers
        .lock()
        .await
        .iter()
        .map(SocketAddr::port)
        .collect();
    assert_eq!(ports.len(), 2);
}
