// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use std::net::SocketAddr;
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::time::Duration;

use dynamo_backend_common::{
    DisaggregationMode, ErrorType, FinishReason, GenerateContext, LLMEngine, OutputOptions,
    PreprocessedRequest, SamplingOptions, StopConditions, StopReason,
};
use dynamo_sidecar_common::{GrpcEndpoint, GrpcTransportConfig};
use futures::{Stream, StreamExt};
use serde_json::json;
use tokio::net::TcpListener;
use tokio::sync::{Mutex, oneshot};
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{Request, Response, Status};

use crate::client::{ModelLimits, TrtllmClient};
use crate::convert::{ResponseState, build_generate_request, engine_error};
use crate::engine::TrtllmSidecarEngine;
use crate::model::ConfiguredModel;
use crate::proto as pb;

/// Most tests exercise aggregated serving; the disaggregation tests name their
/// mode explicitly.
const AGG: DisaggregationMode = DisaggregationMode::Aggregated;

// The tests themselves live in `tests/`, grouped by the surface they cover;
// this file holds only the fakes and fixtures they share.
mod convert_request;
mod convert_response;
mod disagg;
mod e2e;
mod engine;

// ---------------------------------------------------------------------------
// Fake TensorRT-LLM OpenEngine services
// ---------------------------------------------------------------------------

#[derive(Clone, Default)]
struct FakeTrtllm {
    requests: Arc<Mutex<Vec<pb::GenerateRequest>>>,
    target_dp_ranks: Arc<Mutex<Vec<Option<String>>>>,
    aborts: Arc<Mutex<Vec<String>>>,
    peers: Arc<Mutex<Vec<SocketAddr>>>,
    reject: Arc<AtomicBool>,
    hang: Arc<AtomicBool>,
    /// Simulates a server that has not yet accepted the request, so the RPC
    /// itself is still in flight.
    hang_before_stream: Arc<AtomicBool>,
    /// Simulates a server whose Control service is not implemented.
    no_control: Arc<AtomicBool>,
    /// Simulates a server that answers GetModelInfo without a context length.
    empty_model_info: Arc<AtomicBool>,
    model_info_calls: Arc<AtomicUsize>,
    kv_routing: Arc<AtomicBool>,
    disable_events: Arc<AtomicBool>,
    disable_load: Arc<AtomicBool>,
    dp_size: Arc<AtomicU32>,
    hang_load: Arc<AtomicBool>,
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
        self.target_dp_ranks.lock().await.push(
            request
                .metadata()
                .get("openengine-target-dp-rank")
                .and_then(|value| value.to_str().ok())
                .map(str::to_string),
        );
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
        if self.hang_before_stream.load(Ordering::SeqCst) {
            std::future::pending::<()>().await;
        }

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
        self.model_info_calls.fetch_add(1, Ordering::SeqCst);
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
        if !self.kv_routing.load(Ordering::SeqCst) {
            return Err(Status::unimplemented("GetServerInfo is not used"));
        }
        Ok(Response::new(pb::ServerInfo {
            extra: Some(prost_types::Struct {
                fields: [
                    (
                        "trtllm_supports_dp_rank_targeting".to_string(),
                        prost_types::Value {
                            kind: Some(prost_types::value::Kind::BoolValue(true)),
                        },
                    ),
                    (
                        "kv_event_heartbeat_interval_ms".to_string(),
                        prost_types::Value {
                            kind: Some(prost_types::value::Kind::NumberValue(5000.0)),
                        },
                    ),
                ]
                .into(),
            }),
            parallelism: Some(pb::ParallelismInfo {
                data_parallel_size: Some(self.dp_size.load(Ordering::SeqCst).max(1)),
                ..Default::default()
            }),
            capacity: Some(pb::DeploymentCapacity {
                kv_block_size: Some(32),
                total_kv_blocks: Some(100),
                max_running_requests: Some(16),
                max_batched_tokens: Some(2048),
                ..Default::default()
            }),
            ..Default::default()
        }))
    }

    async fn get_load(
        &self,
        _request: Request<pb::GetLoadRequest>,
    ) -> Result<Response<pb::LoadInfo>, Status> {
        if !self.kv_routing.load(Ordering::SeqCst) {
            return Err(Status::unimplemented("GetLoad is not used"));
        }
        if self.disable_load.load(Ordering::SeqCst) {
            return Ok(Response::new(pb::LoadInfo::default()));
        }
        if self.hang_load.load(Ordering::SeqCst) {
            std::future::pending::<()>().await;
        }
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time after epoch")
            .as_nanos() as u64;
        let dp_size = self.dp_size.load(Ordering::SeqCst).max(1);
        Ok(Response::new(pb::LoadInfo {
            timestamp_unix_nanos: Some(timestamp),
            used_kv_blocks: Some(10),
            total_kv_blocks: Some(100),
            ranks: (0..dp_size)
                .map(|rank| pb::RankLoadInfo {
                    data_parallel_rank: Some(rank),
                    used_kv_blocks: Some(10 + rank as u64),
                    total_kv_blocks: Some(100),
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }))
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
        if !self.kv_routing.load(Ordering::SeqCst) {
            return Err(Status::unimplemented("GetKvEventSources is not used"));
        }
        if self.disable_events.load(Ordering::SeqCst) {
            return Ok(Response::new(pb::GetKvEventSourcesResponse::default()));
        }
        let dp_size = self.dp_size.load(Ordering::SeqCst).max(1);
        Ok(Response::new(pb::GetKvEventSourcesResponse {
            sources: (0..dp_size)
                .map(|rank| pb::KvEventSource {
                    transport: "zmq".to_string(),
                    endpoint_addr: Some(pb::KvEndpoint {
                        host: "127.0.0.1".to_string(),
                        port: 5557 + rank,
                        protocol: "tcp".to_string(),
                    }),
                    topic: "kv-events".to_string(),
                    data_parallel_rank: Some(rank),
                    encoding: "msgpack".to_string(),
                    schema_version: Some(1),
                    ..Default::default()
                })
                .collect(),
        }))
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

/// A transport that gives up on startup quickly, for the paths that retry until
/// the deadline rather than failing on the first answer.
fn impatient_transport() -> GrpcTransportConfig {
    GrpcTransportConfig {
        retry_interval: Duration::from_millis(10),
        startup_deadline: Duration::from_millis(200),
        ..transport(1)
    }
}

/// Engine limits as `Control.GetModelInfo` would report them, with no
/// separate output cap.
fn limits(context_length: u32) -> Option<ModelLimits> {
    Some(ModelLimits {
        context_length: Some(context_length),
        max_output_tokens: None,
    })
}

fn engine(endpoint: &str, connections: usize) -> TrtllmSidecarEngine {
    engine_in_mode(endpoint, connections, AGG)
}

fn engine_in_mode(
    endpoint: &str,
    connections: usize,
    mode: DisaggregationMode,
) -> TrtllmSidecarEngine {
    engine_with(endpoint, transport(connections), None, mode)
}

/// The one place a test engine is built. Everything a test varies -- the
/// transport, whether `--context-length` was supplied, the disaggregation role
/// -- is a parameter here.
fn engine_with(
    endpoint: &str,
    transport: GrpcTransportConfig,
    context_length: Option<u32>,
    mode: DisaggregationMode,
) -> TrtllmSidecarEngine {
    TrtllmSidecarEngine::new(
        GrpcEndpoint::parse(endpoint, "--grpc-endpoint").expect("valid test endpoint"),
        transport,
        ConfiguredModel {
            source: "model-source".to_string(),
            context_length,
            ..Default::default()
        },
        mode,
    )
}

fn engine_with_kv_routing(endpoint: &str) -> TrtllmSidecarEngine {
    TrtllmSidecarEngine::new(
        GrpcEndpoint::parse(endpoint, "--grpc-endpoint").expect("valid test endpoint"),
        transport(1),
        ConfiguredModel {
            source: "model-source".to_string(),
            context_length: Some(4096),
            ..Default::default()
        },
        AGG,
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
