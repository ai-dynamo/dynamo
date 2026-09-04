// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use std::net::SocketAddr;
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use dynamo_backend_common::{
    DisaggregationMode, FinishReason, GenerateContext, LLMEngine, OutputOptions,
    PreprocessedRequest, SamplingOptions, StopConditions, StopReason,
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

/// Most tests exercise aggregated serving; the disaggregation tests name their
/// mode explicitly.
const AGG: DisaggregationMode = DisaggregationMode::Aggregated;

// ---------------------------------------------------------------------------
// Fake TensorRT-LLM OpenEngine services
// ---------------------------------------------------------------------------

#[derive(Clone, Default)]
struct FakeTrtllm {
    requests: Arc<Mutex<Vec<pb::GenerateRequest>>>,
    aborts: Arc<Mutex<Vec<String>>>,
    peers: Arc<Mutex<Vec<SocketAddr>>>,
    reject: Arc<AtomicBool>,
    hang: Arc<AtomicBool>,
    /// Simulates a server whose Control service is not implemented.
    no_control: Arc<AtomicBool>,
    /// Simulates a server that answers GetModelInfo without a context length.
    empty_model_info: Arc<AtomicBool>,
}

fn prompt_len(request: &pb::GenerateRequest) -> u32 {
    match request.input.as_ref() {
        Some(pb::generate_request::Input::TokenIds(tokens)) => tokens.ids.len() as u32,
        _ => 0,
    }
}

/// Mirrors the OpenEngine servicer: a request is prefill-only when its `extra`
/// Struct carries `request_type = "context_only"`.
fn is_context_only(request: &pb::GenerateRequest) -> bool {
    request
        .extra
        .as_ref()
        .and_then(|extra| extra.fields.get("request_type"))
        .and_then(|value| value.kind.as_ref())
        .is_some_and(|kind| {
            matches!(kind, prost_types::value::Kind::StringValue(value) if value == "context_only")
        })
}

fn wants_logprobs(request: &pb::GenerateRequest) -> bool {
    request
        .response
        .as_ref()
        .and_then(|response| response.return_output_logprobs)
        .unwrap_or(false)
}

#[tonic::async_trait]
impl pb::inference_server::Inference for FakeTrtllm {
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
        let prompt_tokens = prompt_len(&request);
        let wants_logprobs = wants_logprobs(&request);
        let context_only = is_context_only(&request);
        let hang = self.hang.load(Ordering::SeqCst);

        let stream = async_stream::try_stream! {
            let tokens = if wants_logprobs {
                vec![pb::TokenInfo {
                    token_id: 42,
                    token: String::new(),
                    logprob: Some(-0.25),
                    rank: Some(1),
                    candidates: vec![pb::LogProb {
                        token_id: 43,
                        logprob: -0.5,
                        token: String::new(),
                        rank: Some(2),
                    }],
                }]
            } else {
                vec![pb::TokenInfo {
                    token_id: 42,
                    token: String::new(),
                    logprob: None,
                    rank: None,
                    candidates: Vec::new(),
                }]
            };

            yield pb::GenerateResponse {
                request_id: request_id.clone(),
                event: Some(pb::generate_response::Event::Token(pb::TokenOutput {
                    output_index: Some(0),
                    tokens,
                    text: String::new(),
                })),
                usage: None,
            };

            // A context_only request terminates on PrefillReady: the servicer
            // suppresses the `finished` event because the engine reports the
            // sequence as unfinished.
            if context_only {
                yield pb::GenerateResponse {
                    request_id,
                    event: Some(pb::generate_response::Event::PrefillReady(pb::PrefillReady {
                        kv_session: Some(fake_session()),
                    })),
                    usage: None,
                };
                return;
            }

            if hang {
                loop {
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                }
            }

            yield pb::GenerateResponse {
                request_id,
                event: Some(pb::generate_response::Event::Finished(pb::GenerationFinished {
                    output_index: Some(0),
                    reason: pb::FinishReason::Stop as i32,
                    message: String::new(),
                    stop_match: Some(pb::StopMatch {
                        r#match: Some(pb::stop_match::Match::StopTokenId(2)),
                    }),
                })),
                usage: Some(pb::Usage {
                    prompt_tokens,
                    completion_tokens: 1,
                    total_tokens: prompt_tokens + 1,
                    cached_prompt_tokens: None,
                    reasoning_tokens: None,
                }),
            };
        };
        Ok(Response::new(Box::pin(stream)))
    }
}

/// The handoff a context worker returns, shaped like TensorRT-LLM's: the
/// session id is the context request id and the engine-specific state rides in
/// `attributes_struct`.
fn fake_session() -> pb::KvSessionRef {
    pb::KvSessionRef {
        session_id: "12345".to_string(),
        transfer_backend: "NIXL".to_string(),
        endpoints: vec![pb::KvEndpoint {
            host: "10.0.0.7".to_string(),
            port: 5601,
            protocol: "grpc".to_string(),
        }],
        dp_rank: 0,
        attributes_struct: Some(prost_types::Struct {
            fields: [
                (
                    "opaque_state".to_string(),
                    prost_types::Value {
                        kind: Some(prost_types::value::Kind::StringValue(
                            "c3RhdGU=".to_string(),
                        )),
                    },
                ),
                (
                    "first_gen_tokens".to_string(),
                    prost_types::Value {
                        kind: Some(prost_types::value::Kind::ListValue(
                            prost_types::ListValue {
                                values: vec![prost_types::Value {
                                    kind: Some(prost_types::value::Kind::NumberValue(42.0)),
                                }],
                            },
                        )),
                    },
                ),
            ]
            .into_iter()
            .collect(),
        }),
    }
}

#[tonic::async_trait]
impl pb::control_server::Control for FakeTrtllm {
    async fn get_model_info(
        &self,
        _request: Request<pb::GetModelInfoRequest>,
    ) -> Result<Response<pb::ModelInfo>, Status> {
        if self.no_control.load(Ordering::SeqCst) {
            return Err(Status::unimplemented("Control is not implemented"));
        }
        Ok(Response::new(pb::ModelInfo {
            model_id: "fake-model".to_string(),
            max_context_length: if self.empty_model_info.load(Ordering::SeqCst) {
                None
            } else {
                Some(4096)
            },
            ..Default::default()
        }))
    }

    async fn abort(
        &self,
        request: Request<pb::AbortRequest>,
    ) -> Result<Response<pb::AbortResponse>, Status> {
        let request_id = match request.into_inner().target {
            Some(pb::abort_request::Target::RequestId(id)) => id,
            other => {
                return Err(Status::invalid_argument(format!(
                    "unexpected abort target {other:?}"
                )));
            }
        };
        self.aborts.lock().await.push(request_id.clone());
        Ok(Response::new(pb::AbortResponse {
            status: pb::AbortStatus::Aborted as i32,
            message: format!("aborted {request_id}"),
        }))
    }

    async fn get_server_info(
        &self,
        _request: Request<pb::GetServerInfoRequest>,
    ) -> Result<Response<pb::ServerInfo>, Status> {
        Err(Status::unimplemented("GetServerInfo is not used"))
    }

    async fn get_load(
        &self,
        _request: Request<pb::GetLoadRequest>,
    ) -> Result<Response<pb::LoadInfo>, Status> {
        Err(Status::unimplemented("GetLoad is not used"))
    }

    async fn health(
        &self,
        _request: Request<pb::HealthRequest>,
    ) -> Result<Response<pb::HealthResponse>, Status> {
        Err(Status::unimplemented("Health is not used"))
    }

    async fn load_lora(
        &self,
        _request: Request<pb::LoadLoraRequest>,
    ) -> Result<Response<pb::LoadLoraResponse>, Status> {
        Err(Status::unimplemented("LoadLora is not used"))
    }

    async fn unload_lora(
        &self,
        _request: Request<pb::UnloadLoraRequest>,
    ) -> Result<Response<pb::UnloadLoraResponse>, Status> {
        Err(Status::unimplemented("UnloadLora is not used"))
    }

    async fn list_loras(
        &self,
        _request: Request<pb::ListLorasRequest>,
    ) -> Result<Response<pb::ListLorasResponse>, Status> {
        Err(Status::unimplemented("ListLoras is not used"))
    }

    async fn get_kv_event_sources(
        &self,
        _request: Request<pb::GetKvEventSourcesRequest>,
    ) -> Result<Response<pb::GetKvEventSourcesResponse>, Status> {
        Err(Status::unimplemented("GetKvEventSources is not used"))
    }

    type SubscribeKvEventsStream =
        Pin<Box<dyn Stream<Item = Result<pb::SubscribeKvEventsResponse, Status>> + Send>>;

    async fn subscribe_kv_events(
        &self,
        _request: Request<pb::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        Err(Status::unimplemented("SubscribeKvEvents is not used"))
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
                .add_service(pb::inference_server::InferenceServer::new(
                    server_service.clone(),
                ))
                .add_service(pb::control_server::ControlServer::new(server_service))
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
    engine_in_mode(endpoint, connections, AGG)
}

fn engine_in_mode(
    endpoint: &str,
    connections: usize,
    mode: DisaggregationMode,
) -> TrtllmSidecarEngine {
    TrtllmSidecarEngine::new(
        GrpcEndpoint::parse(endpoint, "--trtllm-endpoint").expect("valid test endpoint"),
        transport(connections),
        ConfiguredModel {
            source: "model-source".to_string(),
            context_length: None,
        },
        mode,
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

/// Applies `mutate` to a baseline request and asserts `build_generate_request`
/// rejects it with a message mentioning `expect`.
fn assert_rejected(mutate: impl FnOnce(&mut PreprocessedRequest), expect: &str) {
    let mut req = request();
    mutate(&mut req);
    let error = build_generate_request(&req, "req", "model", None, AGG)
        .expect_err("request must be rejected");
    assert!(
        error.to_string().contains(expect),
        "error {error:?} should mention {expect:?}"
    );
}

fn token_response(tokens: Vec<pb::TokenInfo>) -> pb::GenerateResponse {
    pb::GenerateResponse {
        request_id: "r".to_string(),
        event: Some(pb::generate_response::Event::Token(pb::TokenOutput {
            output_index: Some(0),
            tokens,
            text: String::new(),
        })),
        usage: None,
    }
}

fn logprob_token(token_id: u32, logprob: f64) -> pb::TokenInfo {
    pb::TokenInfo {
        token_id,
        token: String::new(),
        logprob: Some(logprob),
        rank: Some(1),
        candidates: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Request-building unit tests
// ---------------------------------------------------------------------------

#[test]
fn request_maps_sampling_stop_and_output_fields() {
    let proto = build_generate_request(&request(), "req-1", "served-model", None, AGG)
        .expect("build request");
    assert_eq!(proto.request_id, "req-1");
    assert_eq!(proto.model, "served-model");
    match proto.input.as_ref().expect("input") {
        pb::generate_request::Input::TokenIds(tokens) => assert_eq!(tokens.ids, [11, 22, 33]),
        other => panic!("expected token IDs input, got {other:?}"),
    }

    let stopping = proto.stopping.as_ref().unwrap();
    assert_eq!(stopping.max_tokens, Some(16));
    assert_eq!(stopping.min_tokens, Some(1));
    assert_eq!(stopping.ignore_eos, Some(true));
    // Stop-string retention is never enabled from `include_stop_str_in_output`.
    assert_eq!(stopping.include_stop_in_output, None);
    let conditions: Vec<_> = stopping
        .conditions
        .iter()
        .map(|condition| condition.condition.clone().unwrap())
        .collect();
    assert!(
        conditions.contains(&pb::stop_condition::Condition::StopText("done".to_string())),
        "missing stop text: {conditions:?}"
    );
    assert!(
        conditions.contains(&pb::stop_condition::Condition::StopTokenId(2)),
        "missing stop token id: {conditions:?}"
    );

    let sampling = proto.sampling.as_ref().unwrap();
    assert_eq!(sampling.top_k, Some(4));
    assert_eq!(sampling.top_p, Some(f64::from(0.9_f32)));
    assert_eq!(sampling.min_p, Some(f64::from(0.1_f32)));
    assert_eq!(sampling.temperature, Some(f64::from(0.2_f32)));
    assert_eq!(sampling.seed, Some(123));
    assert_eq!(sampling.repetition_penalty, Some(f64::from(1.1_f32)));
    assert_eq!(sampling.num_sequences, Some(1));

    let response = proto.response.as_ref().unwrap();
    assert_eq!(response.return_output_logprobs, Some(true));
    match response
        .output_candidates
        .as_ref()
        .unwrap()
        .selection
        .as_ref()
        .unwrap()
    {
        pb::candidate_token_selection::Selection::TopN(n) => assert_eq!(*n, 1),
        other => panic!("expected top_n selection, got {other:?}"),
    }

    match proto.guided.as_ref().unwrap().guide.as_ref().unwrap() {
        pb::guided_decoding::Guide::JsonSchema(guide) => assert!(guide.contains("object")),
        other => panic!("expected JSON schema guide, got {other:?}"),
    }
}

#[test]
fn omitted_max_tokens_without_context_length_is_rejected() {
    let mut req = request();
    req.stop_conditions.max_tokens = None;
    let error = build_generate_request(&req, "req", "model", None, AGG)
        .expect_err("must require max_tokens");
    assert!(error.to_string().contains("max_tokens"));
}

#[test]
fn omitted_max_tokens_defaults_to_remaining_context() {
    let mut req = request();
    req.stop_conditions.max_tokens = None;
    // request() carries three prompt tokens ([11, 22, 33]); the default fills the
    // remaining context: max(1, context_length - prompt_len).
    let proto =
        build_generate_request(&req, "req", "model", Some(100), AGG).expect("build with fallback");
    assert_eq!(proto.stopping.unwrap().max_tokens, Some(97));
}

#[test]
fn omitted_max_tokens_default_is_floored_at_one() {
    let mut req = request();
    req.stop_conditions.max_tokens = None;
    // Prompt already fills (or exceeds) the context: default must not underflow to 0.
    let proto =
        build_generate_request(&req, "req", "model", Some(2), AGG).expect("build with fallback");
    assert_eq!(proto.stopping.unwrap().max_tokens, Some(1));
}

#[test]
fn top_k_all_tokens_is_left_unset() {
    let mut req = request();
    req.sampling_options.top_k = Some(-1);
    let proto = build_generate_request(&req, "req", "model", None, AGG).expect("build");
    assert_eq!(proto.sampling.unwrap().top_k, None);
}

#[test]
fn unsupported_request_controls_are_rejected() {
    // Controls the OpenEngine contract can neither forward nor faithfully honor:
    // reject rather than fail open.
    assert_rejected(
        |r| r.sampling_options.include_stop_str_in_output = Some(true),
        "include_stop_str_in_output",
    );
    assert_rejected(
        |r| r.stop_conditions.max_thinking_tokens = Some(32),
        "max_thinking_tokens",
    );
    assert_rejected(
        |r| {
            r.routing = Some(dynamo_backend_common::engine::RoutingHints {
                cache_namespace: Some("tenant-a".to_string()),
                ..Default::default()
            })
        },
        "cache namespace",
    );
    assert_rejected(
        |r| {
            r.routing = Some(dynamo_backend_common::engine::RoutingHints {
                priority: Some(5),
                ..Default::default()
            })
        },
        "priority",
    );
    // A negative top_k other than -1/0 (the "all tokens" sentinels) is invalid,
    // not a silent widening to "all tokens".
    assert_rejected(|r| r.sampling_options.top_k = Some(-5), "top_k must be");
    assert_rejected(
        |r| r.sampling_options.seed = Some(-1),
        "seed must be non-negative",
    );
}

#[test]
fn logprobs_zero_keeps_selected_without_alternatives() {
    let mut req = request();
    req.output_options.logprobs = Some(0);
    // The wire request must still ask TRT-LLM for one candidate so the selected
    // token's logprob is computed.
    let proto = build_generate_request(&req, "req", "model", None, AGG).expect("build");
    match proto
        .response
        .unwrap()
        .output_candidates
        .unwrap()
        .selection
        .unwrap()
    {
        pb::candidate_token_selection::Selection::TopN(n) => assert_eq!(n, 1),
        other => panic!("expected top_n selection, got {other:?}"),
    }

    let mut state = ResponseState::new(&req, AGG);
    let delta = state
        .convert(token_response(vec![logprob_token(7, -0.1)]))
        .expect("convert")
        .expect("delta");
    assert_eq!(delta.log_probs.as_deref(), Some(&[-0.1_f64][..]));
    // logprobs=0 surfaces the selected-token logprob but no top alternatives.
    assert!(delta.top_logprobs.is_none());
}

// ---------------------------------------------------------------------------
// Response-conversion unit tests
// ---------------------------------------------------------------------------

#[test]
fn token_then_finished_produces_delta_then_terminal_usage() {
    let req = request();
    let mut state = ResponseState::new(&req, AGG);

    let delta = state
        .convert(token_response(vec![
            logprob_token(7, -0.1),
            logprob_token(8, -0.2),
        ]))
        .expect("convert token")
        .expect("delta");
    assert_eq!(delta.token_ids, [7, 8]);
    assert!(delta.finish_reason.is_none());
    let log_probs = delta.log_probs.as_ref().expect("log_probs");
    assert_eq!(log_probs, &[-0.1, -0.2]);
    assert_eq!(delta.top_logprobs.as_ref().unwrap().len(), 2);

    let finished = pb::GenerateResponse {
        request_id: "r".to_string(),
        event: Some(pb::generate_response::Event::Finished(
            pb::GenerationFinished {
                output_index: Some(0),
                reason: pb::FinishReason::Length as i32,
                message: String::new(),
                stop_match: None,
            },
        )),
        usage: Some(pb::Usage {
            prompt_tokens: 3,
            completion_tokens: 2,
            total_tokens: 5,
            cached_prompt_tokens: None,
            reasoning_tokens: None,
        }),
    };
    let terminal = state
        .convert(finished)
        .expect("convert finished")
        .expect("terminal");
    assert!(terminal.token_ids.is_empty());
    assert_eq!(terminal.finish_reason, Some(FinishReason::Length));
    let usage = terminal.completion_usage.as_ref().expect("usage");
    assert_eq!((usage.prompt_tokens, usage.completion_tokens), (3, 2));
}

#[test]
fn unsupported_output_index_is_rejected() {
    let req = request();
    let mut state = ResponseState::new(&req, AGG);
    let response = pb::GenerateResponse {
        request_id: "r".to_string(),
        event: Some(pb::generate_response::Event::Token(pb::TokenOutput {
            output_index: Some(1),
            tokens: vec![logprob_token(1, -0.1)],
            text: String::new(),
        })),
        usage: None,
    };
    assert!(state.convert(response).is_err());
}

#[test]
fn missing_event_is_rejected() {
    let req = request();
    let mut state = ResponseState::new(&req, AGG);
    let empty = pb::GenerateResponse {
        request_id: "r".to_string(),
        event: None,
        usage: None,
    };
    assert!(state.convert(empty).is_err());
}

#[test]
fn unspecified_finish_reason_is_rejected() {
    let req = request();
    let mut state = ResponseState::new(&req, AGG);
    let finished = pb::GenerateResponse {
        request_id: "r".to_string(),
        event: Some(pb::generate_response::Event::Finished(
            pb::GenerationFinished {
                output_index: Some(0),
                reason: pb::FinishReason::Unspecified as i32,
                message: String::new(),
                stop_match: None,
            },
        )),
        usage: None,
    };
    assert!(state.convert(finished).is_err());
}

#[test]
fn engine_error_event_is_surfaced_as_error() {
    let req = request();
    let mut state = ResponseState::new(&req, AGG);
    let error = pb::GenerateResponse {
        request_id: "r".to_string(),
        event: Some(pb::generate_response::Event::Error(pb::EngineError {
            code: pb::ErrorCode::Internal as i32,
            message: "boom".to_string(),
            retryable: false,
        })),
        usage: None,
    };
    let error = state.convert(error).expect_err("engine error must surface");
    assert!(error.to_string().contains("boom"));
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
    // GetModelInfo reports max_context_length 4096.
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
    assert_eq!(sent.model, "model-source");
    match sent.input.as_ref().expect("input") {
        pb::generate_request::Input::TokenIds(tokens) => assert_eq!(tokens.ids, [11, 22, 33]),
        other => panic!("expected token IDs input, got {other:?}"),
    }
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
                model: "model-source".to_string(),
                input: Some(pb::generate_request::Input::TokenIds(pb::TokenIds {
                    ids: vec![1, 2],
                })),
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

/// The context length now comes only from the server. Without it the sidecar
/// would reject every request that omits `max_tokens`, so refuse to start
/// instead of registering a worker that cannot serve.
#[tokio::test]
async fn start_fails_when_control_get_model_info_is_unavailable() {
    let service = FakeTrtllm::default();
    service.no_control.store(true, Ordering::SeqCst);
    let server = FakeServer::start(service).await;
    let engine = engine(&server.endpoint, 1);

    let error = engine
        .start(0)
        .await
        .expect_err("start must fail without a context length");
    assert!(
        error.to_string().contains("Control.GetModelInfo failed"),
        "unexpected error: {error}"
    );
}

/// A server that answers GetModelInfo but reports no context length is equally
/// unusable, and must fail the same way rather than register with `None`.
#[tokio::test]
async fn start_fails_when_the_server_reports_no_context_length() {
    let service = FakeTrtllm::default();
    service.empty_model_info.store(true, Ordering::SeqCst);
    let server = FakeServer::start(service).await;
    let engine = engine(&server.endpoint, 1);

    let error = engine
        .start(0)
        .await
        .expect_err("start must fail without a context length");
    assert!(
        error.to_string().contains("no max_context_length"),
        "unexpected error: {error}"
    );
}

// ---------------------------------------------------------------------------
// Disaggregated prefill / decode
// ---------------------------------------------------------------------------

/// The prefill worker marks its request `context_only` and caps generation at
/// the single token the context phase produces.
#[test]
fn prefill_request_is_marked_context_only() {
    let mut req = request();
    req.stop_conditions.max_tokens = Some(128);
    req.stop_conditions.min_tokens = Some(8);
    req.output_options.logprobs = Some(1);
    let proto = build_generate_request(&req, "req", "model", None, DisaggregationMode::Prefill)
        .expect("build prefill request");

    assert!(
        is_context_only(&proto),
        "prefill must set extra.request_type"
    );
    let stopping = proto.stopping.expect("stopping options");
    assert_eq!(stopping.max_tokens, Some(1));
    assert_eq!(stopping.min_tokens, None, "a minimum would force decoding");
    // The prefill worker surfaces no tokens, but it must still compute
    // logprobs: the first generated token comes from the context phase and its
    // logprob only reaches the decode worker through the handoff.
    assert_eq!(
        proto
            .response
            .expect("response options")
            .return_output_logprobs,
        Some(true)
    );
    assert!(proto.kv.is_none(), "prefill carries no session to replay");
}

/// `PrefillReady` is the prefill worker's terminal chunk: no tokens, and the
/// handoff the decode worker will replay.
#[tokio::test]
async fn prefill_ready_is_the_terminal_handoff() {
    let server = FakeServer::start(FakeTrtllm::default()).await;
    let engine = engine_in_mode(&server.endpoint, 1, DisaggregationMode::Prefill);
    engine.start(0).await.expect("start");

    let outputs = collect(&engine, request()).await;
    let terminal = outputs
        .iter()
        .find(|output| output.finish_reason.is_some())
        .expect("a terminal chunk");
    // The frontend's prefill router chains into decode only for `Length`; any
    // other terminal reason is returned to the caller as a finished request.
    assert_eq!(terminal.finish_reason, Some(FinishReason::Length));

    assert!(
        outputs.iter().all(|output| output.token_ids.is_empty()),
        "a prefill worker must not stream tokens to the client"
    );
    let handoff = terminal
        .disaggregated_params
        .as_ref()
        .expect("terminal carries the prefill handoff");
    assert_eq!(handoff["session_id"], json!("12345"));
    assert_eq!(handoff["transfer_backend"], json!("NIXL"));
    assert_eq!(handoff["endpoints"][0]["port"], json!(5601));
    assert_eq!(handoff["attributes"]["first_gen_tokens"], json!([42]));
}

/// The decode worker replays the prefill handoff verbatim in `kv.session`.
#[tokio::test]
async fn decode_request_replays_the_prefill_session() {
    let server = FakeServer::start(FakeTrtllm::default()).await;

    // Phase 1: prefill produces the handoff.
    let prefill = engine_in_mode(&server.endpoint, 1, DisaggregationMode::Prefill);
    prefill.start(0).await.expect("start prefill");
    let handoff = collect(&prefill, request())
        .await
        .into_iter()
        .find_map(|output| output.disaggregated_params)
        .expect("prefill handoff");

    // Phase 2: decode replays it.
    let decode = engine_in_mode(&server.endpoint, 1, DisaggregationMode::Decode);
    decode.start(0).await.expect("start decode");
    let mut req = request();
    req.prefill_result = Some(dynamo_backend_common::PrefillResult {
        disaggregated_params: handoff,
        prompt_tokens_details: None,
    });
    let outputs = collect(&decode, req).await;
    assert!(
        outputs.iter().any(|output| !output.token_ids.is_empty()),
        "the decode worker streams the completion"
    );

    let session = server.service.requests.lock().await[1]
        .kv
        .as_ref()
        .and_then(|kv| kv.session.as_ref())
        .expect("decode request carries kv.session")
        .clone();
    assert_eq!(
        session,
        fake_session(),
        "the handoff must round-trip intact"
    );
}

#[test]
fn decode_without_a_prefill_result_is_rejected() {
    let error =
        build_generate_request(&request(), "req", "model", None, DisaggregationMode::Decode)
            .expect_err("decode requires a handoff");
    assert!(
        error.to_string().contains("missing the prefill_result"),
        "unexpected error: {error}"
    );
}

/// A handoff on a non-decode worker means the frontend routed the request to
/// the wrong role - fail loudly rather than silently prefill it again.
#[test]
fn prefill_result_on_a_non_decode_worker_is_rejected() {
    let mut req = request();
    req.prefill_result = Some(dynamo_backend_common::PrefillResult {
        disaggregated_params: json!({"session_id": "1"}),
        prompt_tokens_details: None,
    });
    for mode in [DisaggregationMode::Aggregated, DisaggregationMode::Prefill] {
        let error = build_generate_request(&req, "req", "model", None, mode)
            .expect_err("handoff must be rejected");
        assert!(
            error
                .to_string()
                .contains("must be routed to a decode worker"),
            "unexpected error for {mode}: {error}"
        );
    }
}

/// An aggregated worker never asks for a context handoff, so receiving one is
/// protocol drift rather than a silent no-op.
#[test]
fn unexpected_prefill_ready_on_an_aggregated_worker_is_rejected() {
    let mut state = ResponseState::new(&request(), AGG);
    let error = state
        .convert(pb::GenerateResponse {
            request_id: "req".to_string(),
            event: Some(pb::generate_response::Event::PrefillReady(
                pb::PrefillReady {
                    kv_session: Some(fake_session()),
                },
            )),
            usage: None,
        })
        .expect_err("prefill_ready must be rejected");
    assert!(
        error.to_string().contains("not running in prefill mode"),
        "unexpected error: {error}"
    );
}

// ---------------------------------------------------------------------------
// End-to-end against a live TensorRT-LLM OpenEngine server
// ---------------------------------------------------------------------------

/// Drives a real disaggregated prefill -> decode handoff against two live
/// OpenEngine servers, each backed by a TensorRT-LLM engine with a KV cache
/// transceiver. Ignored by default; run explicitly with both endpoints:
///
/// ```text
/// TRTLLM_E2E_PREFILL_ENDPOINT=http://127.0.0.1:50051 \
/// TRTLLM_E2E_DECODE_ENDPOINT=http://127.0.0.1:50052 \
///   cargo test -p dynamo-trtllm-sidecar e2e_real_disagg -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore = "requires two live TensorRT-LLM OpenEngine servers with a KV transceiver"]
async fn e2e_real_disagg_handoff() {
    let prefill_endpoint = std::env::var("TRTLLM_E2E_PREFILL_ENDPOINT")
        .expect("set TRTLLM_E2E_PREFILL_ENDPOINT, e.g. http://127.0.0.1:50051");
    let decode_endpoint = std::env::var("TRTLLM_E2E_DECODE_ENDPOINT")
        .expect("set TRTLLM_E2E_DECODE_ENDPOINT, e.g. http://127.0.0.1:50052");
    let model = std::env::var("TRTLLM_E2E_MODEL")
        .unwrap_or_else(|_| "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string());

    // `start` resolves the context length from `Control.GetModelInfo`.
    let configured = |source: String| ConfiguredModel {
        source,
        context_length: None,
    };
    let prefill = TrtllmSidecarEngine::new(
        GrpcEndpoint::parse(&prefill_endpoint, "--trtllm-endpoint").expect("valid endpoint"),
        transport(1),
        configured(model.clone()),
        DisaggregationMode::Prefill,
    );
    let decode = TrtllmSidecarEngine::new(
        GrpcEndpoint::parse(&decode_endpoint, "--trtllm-endpoint").expect("valid endpoint"),
        transport(1),
        configured(model.clone()),
        DisaggregationMode::Decode,
    );
    prefill.start(0).await.expect("start prefill worker");
    decode.start(0).await.expect("start decode worker");

    // "<s> Hello, my name is" in the Llama tokenizer.
    let base = || {
        PreprocessedRequest::builder()
            .model(model.clone())
            .token_ids(vec![1, 15043, 29892, 590, 1024, 338])
            .stop_conditions(StopConditions {
                max_tokens: Some(16),
                ..Default::default()
            })
            .sampling_options(SamplingOptions {
                temperature: Some(0.0),
                ..Default::default()
            })
            .output_options(OutputOptions::default())
            .build()
            .expect("request")
    };

    // Phase 1: prefill.
    let prefill_outputs = collect(&prefill, base()).await;
    eprintln!(
        "[e2e-disagg] prefill produced {} chunk(s)",
        prefill_outputs.len()
    );
    assert!(
        prefill_outputs
            .iter()
            .all(|output| output.token_ids.is_empty()),
        "the prefill worker must not stream tokens"
    );
    let handoff = prefill_outputs
        .iter()
        .find_map(|output| output.disaggregated_params.clone())
        .expect("prefill terminal carries the KV handoff");
    eprintln!("[e2e-disagg] handoff = {handoff}");

    // Phase 2: decode replays the handoff.
    let mut decode_request = base();
    decode_request.prefill_result = Some(dynamo_backend_common::PrefillResult {
        disaggregated_params: handoff,
        prompt_tokens_details: None,
    });
    let decode_outputs = collect(&decode, decode_request).await;

    let mut generated = Vec::new();
    let mut terminal = None;
    for output in &decode_outputs {
        generated.extend(output.token_ids.iter().copied());
        if output.finish_reason.is_some() {
            terminal = Some(output);
        }
    }
    eprintln!("[e2e-disagg] decode generated token IDs: {generated:?}");
    let terminal = terminal.expect("a terminal output carrying a finish_reason");
    eprintln!("[e2e-disagg] finish_reason = {:?}", terminal.finish_reason);
    let usage = terminal.completion_usage.as_ref().expect("terminal usage");
    eprintln!(
        "[e2e-disagg] usage: prompt={}, completion={}",
        usage.prompt_tokens, usage.completion_tokens
    );

    assert!(
        !generated.is_empty(),
        "the decode worker must generate tokens from the prefill handoff"
    );
}

/// Drives the real `TrtllmSidecarEngine` against a live OpenEngine gRPC server
/// (a real TensorRT-LLM engine on a GPU). Ignored by default; run explicitly with
/// a reachable endpoint:
///
/// ```text
/// TRTLLM_E2E_ENDPOINT=http://127.0.0.1:50051 \
///   cargo test -p dynamo-trtllm-sidecar e2e_real_openengine -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore = "requires a live TensorRT-LLM OpenEngine server; set TRTLLM_E2E_ENDPOINT"]
async fn e2e_real_openengine_server() {
    let endpoint = std::env::var("TRTLLM_E2E_ENDPOINT")
        .expect("set TRTLLM_E2E_ENDPOINT, e.g. http://127.0.0.1:50051");
    let model = std::env::var("TRTLLM_E2E_MODEL")
        .unwrap_or_else(|_| "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string());

    let engine = TrtllmSidecarEngine::new(
        GrpcEndpoint::parse(&endpoint, "--trtllm-endpoint").expect("valid endpoint"),
        transport(2),
        // `start` resolves the context length from `Control.GetModelInfo`.
        ConfiguredModel {
            source: model.clone(),
            context_length: None,
        },
        AGG,
    );
    let config = engine
        .start(0)
        .await
        .expect("start against the live server");
    eprintln!("[e2e] connected; registered model = {}", config.model);

    // "<s> Hello, my name is" in the Llama tokenizer; any valid token IDs work.
    let request = PreprocessedRequest::builder()
        .model(model)
        .token_ids(vec![1, 15043, 29892, 590, 1024, 338])
        .stop_conditions(StopConditions {
            max_tokens: Some(16),
            ..Default::default()
        })
        .sampling_options(SamplingOptions {
            temperature: Some(0.0),
            ..Default::default()
        })
        .output_options(OutputOptions {
            logprobs: Some(1),
            ..Default::default()
        })
        .build()
        .expect("request");

    let outputs = collect(&engine, request).await;
    eprintln!("[e2e] received {} stream item(s)", outputs.len());

    let mut generated = Vec::new();
    let mut terminal = None;
    let mut saw_logprobs = false;
    for output in &outputs {
        generated.extend(output.token_ids.iter().copied());
        saw_logprobs |= output.log_probs.is_some();
        if output.finish_reason.is_some() {
            terminal = Some(output);
        }
    }
    eprintln!("[e2e] generated token IDs: {generated:?}");

    let terminal = terminal.expect("a terminal output carrying a finish_reason");
    eprintln!("[e2e] finish_reason = {:?}", terminal.finish_reason);
    let usage = terminal.completion_usage.as_ref().expect("terminal usage");
    eprintln!(
        "[e2e] usage: prompt={}, completion={}",
        usage.prompt_tokens, usage.completion_tokens
    );

    assert!(
        !generated.is_empty(),
        "expected at least one generated token"
    );
    assert_eq!(
        usage.prompt_tokens, 6,
        "prompt token count should echo the input"
    );
    assert!(
        usage.completion_tokens > 0,
        "expected nonzero completion usage"
    );
    assert!(
        saw_logprobs,
        "logprobs were requested but none were surfaced"
    );
}
