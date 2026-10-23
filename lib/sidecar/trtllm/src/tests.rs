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

use crate::client::TrtllmClient;
use crate::engine::TrtllmSidecarEngine;
use crate::model::ConfiguredModel;
use crate::proto as pb;

use crate::test_fixtures::*;

mod e2e;
mod kv_routing;

// ---------------------------------------------------------------------------
// Fake TensorRT-LLM OpenEngine services
// ---------------------------------------------------------------------------

#[derive(Clone, Default)]
struct FakeTrtllm {
    requests: Arc<Mutex<Vec<pb::GenerateRequest>>>,
    target_dp_ranks: Arc<Mutex<Vec<Option<String>>>>,
    affinity_headers: Arc<Mutex<Vec<(String, String)>>>,
    aborts: Arc<Mutex<Vec<String>>>,
    peers: Arc<Mutex<Vec<SocketAddr>>>,
    hang: Arc<AtomicBool>,
    /// Simulates a server that has not yet accepted the request, so the RPC
    /// itself is still in flight.
    hang_before_stream: Arc<AtomicBool>,
    /// Simulates a server whose Control service is not implemented.
    no_control: Arc<AtomicBool>,
    /// Simulates a server that answers GetModelInfo without a context length.
    empty_model_info: Arc<AtomicBool>,
    /// Answers `UNAVAILABLE`, the shape of an engine that is not serving yet.
    unavailable_model_info: Arc<AtomicBool>,
    model_info_calls: Arc<AtomicUsize>,
    kv_routing: Arc<AtomicBool>,
    server_info_error: Arc<Mutex<Option<tonic::Code>>>,
    kv_sources_error: Arc<Mutex<Option<tonic::Code>>>,
    subagent_affinity: Arc<AtomicBool>,
    disable_events: Arc<AtomicBool>,
    disable_load: Arc<AtomicBool>,
    dp_size: Arc<AtomicU32>,
    hang_load: Arc<AtomicBool>,
    node_metadata: Arc<Mutex<Option<crate::node::NodeMetadata>>>,
}

fn prompt_len(request: &pb::GenerateRequest) -> u32 {
    match request.input.as_ref() {
        Some(pb::generate_request::Input::TokenIds(tokens)) => tokens.ids.len() as u32,
        _ => 0,
    }
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
        if let Some(parent) = request.metadata().get("x-trtllm-subagent-affinity-id") {
            let auth = request
                .metadata()
                .get("x-trtllm-subagent-affinity-auth")
                .expect("signed parent affinity")
                .to_str()
                .expect("ASCII signature")
                .to_string();
            self.affinity_headers
                .lock()
                .await
                .push((parent.to_str().expect("valid parent ID").to_string(), auth));
        }
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

        let request_id = request.request_id.clone();
        let prompt_tokens = prompt_len(&request);
        let wants_logprobs = wants_logprobs(&request);
        let context_only = is_context_only(&request);
        let hang = self.hang.load(Ordering::SeqCst);
        if self.hang_before_stream.load(Ordering::SeqCst) {
            std::future::pending::<()>().await;
        }
        // The fake always samples 42. Like TensorRT-LLM, it stops on a stop
        // word and returns it only under `include_stop_in_output`.
        let stopping = request.stopping.clone().unwrap_or_default();
        let stops_on_42 = stopping.conditions.iter().any(|condition| {
            condition.condition == Some(pb::stop_condition::Condition::StopTokenId(42))
        });
        let returns_42 = !stops_on_42 || stopping.include_stop_in_output == Some(true);
        let matched_stop = if stops_on_42 { 42 } else { 2 };

        let stream = async_stream::try_stream! {
            let tokens = if !returns_42 {
                Vec::new()
            } else if wants_logprobs {
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
                        r#match: Some(pb::stop_match::Match::StopTokenId(matched_stop)),
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
        if self.unavailable_model_info.load(Ordering::SeqCst) {
            return Err(Status::unavailable("not serving yet"));
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
        if let Some(code) = *self.server_info_error.lock().await {
            return Err(Status::new(code, "temporary metadata failure"));
        }
        if !self.kv_routing.load(Ordering::SeqCst) {
            return Err(Status::unimplemented("GetServerInfo is not used"));
        }
        let mut fields = [
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
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>();
        if self.subagent_affinity.load(Ordering::SeqCst) {
            fields.insert(
                "trtllm_supports_subagent_affinity".to_string(),
                prost_types::Value {
                    kind: Some(prost_types::value::Kind::BoolValue(true)),
                },
            );
        }
        if let Some(node) = self.node_metadata.lock().await.as_ref() {
            fields.extend(
                dynamo_sidecar_common::json_to_struct(json!({"trtllm_node": node}), "test")
                    .unwrap()
                    .fields,
            );
        }
        Ok(Response::new(pb::ServerInfo {
            extra: Some(prost_types::Struct { fields }),
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
        if let Some(code) = *self.kv_sources_error.lock().await {
            return Err(Status::new(code, "temporary metadata failure"));
        }
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

/// `PrefillReady` is the prefill worker's terminal chunk: no tokens, and the
/// handoff the decode worker will replay.
#[tokio::test]
async fn prefill_ready_is_the_terminal_handoff() {
    let server = FakeServer::start(FakeTrtllm::default()).await;
    let engine = engine_in_mode(&server.endpoint, 1, DisaggregationMode::Prefill);
    engine.start(0).await.expect("start");

    let mut req = request();
    req.routing = Some(dynamo_backend_common::engine::RoutingHints {
        prefill_dp_rank: Some(0),
        dp_rank: Some(7),
        ..Default::default()
    });
    let outputs = collect(&engine, req).await;
    assert_eq!(
        server.service.target_dp_ranks.lock().await.as_slice(),
        [None]
    );
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

    let mut agent_request = request();
    agent_request.agent_context = Some(
        serde_json::from_value(json!({"session_id": "agentx-session"}))
            .expect("valid agent context"),
    );

    // Phase 1: prefill produces the handoff.
    let prefill = engine_in_mode(&server.endpoint, 1, DisaggregationMode::Prefill);
    prefill.start(0).await.expect("start prefill");
    let handoff = collect(&prefill, agent_request.clone())
        .await
        .into_iter()
        .find_map(|output| output.disaggregated_params)
        .expect("prefill handoff");

    // Phase 2: decode replays it.
    let decode = engine_in_mode(&server.endpoint, 1, DisaggregationMode::Decode);
    decode.start(0).await.expect("start decode");
    let mut req = agent_request;
    req.routing = Some(dynamo_backend_common::engine::RoutingHints {
        dp_rank: Some(0),
        prefill_dp_rank: Some(7),
        ..Default::default()
    });
    req.prefill_result = Some(dynamo_backend_common::PrefillResult {
        disaggregated_params: handoff,
        prompt_tokens_details: None,
    });
    let outputs = collect(&decode, req).await;
    assert_eq!(
        server.service.target_dp_ranks.lock().await.as_slice(),
        [None, None]
    );
    assert!(
        outputs.iter().any(|output| !output.token_ids.is_empty()),
        "the decode worker streams the completion"
    );

    let requests = server.service.requests.lock().await;
    for request in requests.iter() {
        let extra = request.extra.as_ref().expect("request extensions");
        assert_eq!(
            extra.fields["detokenize"].kind,
            Some(prost_types::value::Kind::BoolValue(false))
        );
        assert_eq!(
            extra.fields["conversation_id"].kind,
            Some(prost_types::value::Kind::StringValue(
                "agentx-session".into()
            )),
            "both RPC legs must retain the same conversation identity"
        );
    }
    let session = requests[1]
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

#[tokio::test]
async fn aggregated_generation_streams_delta_then_terminal() {
    let server = FakeServer::start(FakeTrtllm::default()).await;
    let engine = engine(&server.endpoint, 2);
    let config = engine.start(0).await.expect("start");
    assert_eq!(config.model, "model-source");
    // GetModelInfo reports max_context_length 4096.
    assert_eq!(config.llm.unwrap().context_length, Some(4096));

    let mut req = request();
    req.routing = Some(dynamo_backend_common::engine::RoutingHints {
        dp_rank: Some(0),
        ..Default::default()
    });
    let outputs = collect(&engine, req).await;
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

/// A request cancelled before dispatch must not reach the engine.
#[tokio::test]
async fn an_aggregated_request_cancelled_before_dispatch_never_reaches_the_engine() {
    let server = FakeServer::start(FakeTrtllm::default()).await;
    let engine = engine(&server.endpoint, 1);
    engine.start(0).await.expect("start");

    let context = dynamo_backend_common::testing::mock_context();
    context.stop_generating();

    let mut stream = engine
        .generate(request(), GenerateContext::new(context.clone(), None))
        .await
        .expect("generate");
    let terminal = tokio::time::timeout(std::time::Duration::from_secs(2), stream.next())
        .await
        .expect("terminal within deadline")
        .unwrap()
        .unwrap();
    assert_eq!(terminal.finish_reason, Some(FinishReason::Cancelled));
    assert!(
        server.service.requests.lock().await.is_empty(),
        "a request cancelled before dispatch must not be sent to the engine"
    );
}

/// Nonzero logical ranks are not advertised and must be rejected before dispatch.
#[tokio::test]
async fn a_data_parallel_rank_hint_is_rejected_before_dispatch() {
    let server = FakeServer::start(FakeTrtllm::default()).await;

    for (mode, hints) in [
        (
            DisaggregationMode::Prefill,
            dynamo_backend_common::engine::RoutingHints {
                prefill_dp_rank: Some(7),
                ..Default::default()
            },
        ),
        (
            DisaggregationMode::Decode,
            dynamo_backend_common::engine::RoutingHints {
                dp_rank: Some(3),
                ..Default::default()
            },
        ),
        (
            AGG,
            dynamo_backend_common::engine::RoutingHints {
                dp_rank: Some(3),
                ..Default::default()
            },
        ),
    ] {
        let engine = engine_in_mode(&server.endpoint, 1, mode);
        engine.start(0).await.expect("start");
        let mut req = request();
        req.routing = Some(hints);
        let context = dynamo_backend_common::testing::mock_context();
        let result = engine
            .generate(req, GenerateContext::new(context, None))
            .await;
        assert!(result.is_err(), "{mode:?} must refuse a rank hint");
    }
    assert!(
        server.service.requests.lock().await.is_empty(),
        "nothing may reach the engine"
    );
}

/// A decode request holding transferred KV blocks must outlive its client until
/// the first token proves the transfer landed -- dropping the stream earlier
/// strands the prefill worker's blocks. It must not outlive it any longer than
/// that, or a cancelled request generates its whole budget with no consumer.
#[tokio::test]
async fn a_cancelled_decode_request_survives_only_until_the_transfer_lands() {
    let service = FakeTrtllm::default();
    service.hang.store(true, Ordering::SeqCst);
    let server = FakeServer::start(service).await;
    let engine = engine_in_mode(&server.endpoint, 1, DisaggregationMode::Decode);
    engine.start(0).await.expect("start");

    let context = dynamo_backend_common::testing::mock_context();
    let mut req = request();
    req.prefill_result = Some(dynamo_backend_common::PrefillResult {
        disaggregated_params: crate::disagg::session_to_json(fake_session()).expect("handoff"),
        prompt_tokens_details: None,
    });
    let mut stream = engine
        .generate(req, GenerateContext::new(context.clone(), None))
        .await
        .expect("generate");

    context.stop_generating();
    // Still deferring: nothing has confirmed the KV transfer yet.
    let first = stream.next().await.unwrap().unwrap();
    assert_eq!(
        first.finish_reason, None,
        "the deferral outlasts the client"
    );
    assert!(!first.token_ids.is_empty());

    let terminal = tokio::time::timeout(std::time::Duration::from_secs(2), stream.next())
        .await
        .expect("the deferral lifts once a token lands")
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

    for unsupported in [multiple, beam, embeds, prompt_logprobs] {
        let context = dynamo_backend_common::testing::mock_context();
        let result = engine
            .generate(unsupported, GenerateContext::new(context, None))
            .await;
        assert!(result.is_err());
    }
    assert!(server.service.requests.lock().await.is_empty());
}

#[tokio::test]
async fn visible_stop_token_is_returned_and_hidden_stop_token_is_not() {
    let server = FakeServer::start(FakeTrtllm::default()).await;
    let engine = engine(&server.endpoint, 1);
    engine.start(0).await.expect("start");
    let token_ids = |outputs: &[dynamo_backend_common::LLMEngineOutput]| {
        outputs
            .iter()
            .flat_map(|output| output.token_ids.clone())
            .collect::<Vec<_>>()
    };

    let mut visible = request();
    visible.stop_conditions.stop_token_ids_visible = Some(vec![42]);
    let outputs = collect(&engine, visible).await;
    assert_eq!(token_ids(&outputs), [42]);
    let terminal = outputs.last().unwrap();
    assert_eq!(terminal.token_ids, [42], "held for the terminal's usage");
    assert!(terminal.completion_usage.is_some());
    assert_eq!(terminal.stop_reason, Some(StopReason::Int(42)));

    // Retaining a visible stop makes the engine return hidden stops too.
    let mut hidden = request();
    hidden.stop_conditions.stop_token_ids_hidden = Some(vec![42]);
    hidden.stop_conditions.stop_token_ids_visible = Some(vec![7]);
    let outputs = collect(&engine, hidden).await;
    assert!(token_ids(&outputs).is_empty());
    let terminal = outputs.last().unwrap();
    assert_eq!(terminal.finish_reason, Some(FinishReason::Stop));
    assert_eq!(terminal.stop_reason, Some(StopReason::Int(42)));
}

#[tokio::test]
async fn pool_uses_each_configured_connection() {
    let server = FakeServer::start(FakeTrtllm::default()).await;
    let endpoint =
        GrpcEndpoint::parse(&server.endpoint, "--grpc-endpoint").expect("valid endpoint");
    let client = TrtllmClient::connect(&endpoint, transport(2))
        .await
        .expect("connect pool");
    assert_eq!(client.connection_count(), 2);

    for index in 0..4 {
        let mut stream = client
            .generate(
                pb::GenerateRequest {
                    request_id: format!("request-{index}"),
                    model: "model-source".to_string(),
                    input: Some(pb::generate_request::Input::TokenIds(pb::TokenIds {
                        ids: vec![1, 2],
                    })),
                    ..Default::default()
                },
                None,
                None,
            )
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

/// A server with no Control service leaves the window unknown. Registering
/// anyway would advertise capacity and then reject, with a non-migratable 4xx,
/// every request that omits `max_tokens` -- most chat traffic. Fail at startup
/// instead, naming the flag that fixes it.
#[tokio::test]
async fn start_without_a_context_length_fails() {
    let service = FakeTrtllm::default();
    service.no_control.store(true, Ordering::SeqCst);
    let server = FakeServer::start(service).await;
    let engine = engine_with(&server.endpoint, impatient_transport(), None, AGG);

    let error = engine
        .start(0)
        .await
        .expect_err("a worker with no window must not register");
    assert!(
        error.to_string().contains("--context-length"),
        "the error must name the flag that supplies a window: {error}"
    );
}

/// An engine that answers has finished loading, so an answer carrying no
/// context length means it was started without `--max_seq_len` and will not
/// change. Fail on the first answer rather than polling to the deadline and
/// then blaming a slow load.
#[tokio::test]
async fn start_fails_at_once_when_the_server_answers_without_a_context_length() {
    let service = FakeTrtllm::default();
    service.empty_model_info.store(true, Ordering::SeqCst);
    let server = FakeServer::start(service).await;
    let engine = engine_with(&server.endpoint, impatient_transport(), None, AGG);

    let error = engine
        .start(0)
        .await
        .expect_err("a window is required to register");
    assert!(
        error.to_string().contains("--max_seq_len"),
        "the error must name the fix: {error}"
    );
    assert_eq!(
        server.service.model_info_calls.load(Ordering::SeqCst),
        1,
        "an answer that cannot change must not be retried"
    );
}

/// The wait exists for an engine that is not serving yet: it keeps asking
/// through transport failures and picks up the window once the engine answers.
#[tokio::test]
async fn start_waits_for_a_server_that_is_not_serving_yet() {
    let service = FakeTrtllm::default();
    service.unavailable_model_info.store(true, Ordering::SeqCst);
    let server = FakeServer::start(service).await;
    let ready = Arc::clone(&server.service.unavailable_model_info);
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(40)).await;
        ready.store(false, Ordering::SeqCst);
    });
    let engine = engine_with(&server.endpoint, impatient_transport(), None, AGG);

    let config = engine.start(0).await.expect("start once the engine serves");
    assert_eq!(
        config.llm.expect("llm registration").context_length,
        Some(4096)
    );
}

/// `--context-length` wins over what the engine reports. The engine is still
/// asked, so the disagreement can be logged and its output cap picked up, but
/// its answer does not decide the window.
#[tokio::test]
async fn a_configured_context_length_outranks_the_servers() {
    let server = FakeServer::start(FakeTrtllm::default()).await;
    let engine = engine_with(&server.endpoint, transport(1), Some(8192), AGG);

    let config = engine
        .start(0)
        .await
        .expect("start with a configured window");
    assert_eq!(
        config.llm.expect("llm registration").context_length,
        Some(8192),
        "the configured window must outrank the server's 4096"
    );
    assert_eq!(
        server.service.model_info_calls.load(Ordering::SeqCst),
        1,
        "the engine is still consulted, to cross-check and to learn its output cap"
    );
}

/// A configured window also means a server with no Control service is no longer
/// a startup problem: there is nothing left to ask it for.
#[tokio::test]
async fn a_configured_context_length_survives_a_server_without_control() {
    let service = FakeTrtllm::default();
    service.no_control.store(true, Ordering::SeqCst);
    let server = FakeServer::start(service).await;
    let engine = engine_with(&server.endpoint, transport(1), Some(8192), AGG);

    let config = engine
        .start(0)
        .await
        .expect("start with a configured window");
    assert_eq!(
        config.llm.expect("llm registration").context_length,
        Some(8192)
    );
}

/// Shutdown has to terminate in-flight requests: the worker cannot drain if a
/// stream waits forever on an engine that will never answer.
#[tokio::test]
async fn cleanup_terminates_an_in_flight_request() {
    let service = FakeTrtllm::default();
    service.hang.store(true, Ordering::SeqCst);
    let server = FakeServer::start(service).await;
    let engine = Arc::new(engine(&server.endpoint, 1));
    engine.start(0).await.expect("start");

    let context = dynamo_backend_common::testing::mock_context();
    let mut stream = engine
        .generate(request(), GenerateContext::new(context, None))
        .await
        .expect("generate");
    let first = stream.next().await.expect("first item").expect("delta");
    assert_eq!(first.token_ids, [42]);

    engine.cleanup().await.expect("cleanup");
    let terminal = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("shutdown must terminate the stream")
        .expect("terminal item")
        .expect("terminal");
    assert_eq!(terminal.finish_reason, Some(FinishReason::Cancelled));
}

/// A decode leg's cancellation deferral has to cover the dispatch itself, not
/// just the streaming loop. `generate` sends the request and then awaits
/// response headers, so a cancellation that wins that race abandons a request
/// the engine has already accepted and begun pulling KV for -- stranding the
/// prefill worker's blocks with no leg left to claim them. Shutdown still wins.
#[tokio::test]
async fn a_cancelled_decode_dispatch_is_not_abandoned_mid_flight() {
    let service = FakeTrtllm::default();
    service.hang_before_stream.store(true, Ordering::SeqCst);
    let server = FakeServer::start(service).await;
    let engine = Arc::new(engine_in_mode(
        &server.endpoint,
        1,
        DisaggregationMode::Decode,
    ));
    engine.start(0).await.expect("start");

    let mut decode_request = request();
    decode_request.prefill_result = Some(dynamo_backend_common::PrefillResult {
        disaggregated_params: crate::disagg::session_to_json(fake_session()).expect("handoff"),
        prompt_tokens_details: None,
    });
    let context = dynamo_backend_common::testing::mock_context();
    let mut dispatch = tokio::spawn({
        let engine = Arc::clone(&engine);
        let context = context.clone();
        async move {
            engine
                .generate(decode_request, GenerateContext::new(context, None))
                .await
        }
    });

    // The fake records the request before it withholds response headers, which
    // is exactly the window this test is about: the engine has the request and
    // the sidecar does not know it yet.
    while server.service.requests.lock().await.is_empty() {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    context.stop_generating();
    assert!(
        tokio::time::timeout(Duration::from_millis(200), &mut dispatch)
            .await
            .is_err(),
        "the dispatch must outlive the client's cancellation"
    );

    engine.cleanup().await.expect("cleanup");
    let mut stream = tokio::time::timeout(Duration::from_secs(5), dispatch)
        .await
        .expect("shutdown must release the dispatch")
        .expect("dispatch task")
        .expect("generate");
    let terminal = stream
        .next()
        .await
        .expect("a terminal item")
        .expect("terminal");
    assert_eq!(terminal.finish_reason, Some(FinishReason::Cancelled));
}

// Regression: eager discovery must neither bypass Worker start nor reconnect
// during that non-cancellable phase after bootstrap has already succeeded.
#[tokio::test]
async fn bootstrap_discovers_once_but_requires_start_before_serving() {
    let server = FakeServer::start(FakeTrtllm::default()).await;
    let bootstrap = TrtllmSidecarEngine::try_from_args_async(vec![
        "dynamo-trtllm-sidecar".into(),
        "--grpc-endpoint".into(),
        server.endpoint.clone(),
        "--model-path".into(),
        "model-source".into(),
    ])
    .unwrap();
    let (engine, _) = bootstrap.1.await.unwrap();
    assert_eq!(server.service.model_info_calls.load(Ordering::SeqCst), 1);
    let context = dynamo_backend_common::testing::mock_context();
    let result = engine
        .generate(request(), GenerateContext::new(context, None))
        .await;
    let error = result.err().expect("bootstrap must not enable generation");
    assert_eq!(
        error.error_type(),
        ErrorType::Backend(dynamo_backend_common::BackendError::EngineShutdown)
    );
    assert!(server.service.requests.lock().await.is_empty());

    // Metadata was resolved already; a transient Control failure cannot make
    // Worker repeat discovery and hang in its non-cancellable start phase.
    server
        .service
        .unavailable_model_info
        .store(true, Ordering::SeqCst);
    let config = engine.start(0).await.unwrap();
    assert_eq!(config.llm.unwrap().context_length, Some(4096));
    assert_eq!(server.service.model_info_calls.load(Ordering::SeqCst), 1);
    let outputs = collect(&engine, request()).await;
    assert_eq!(outputs[0].token_ids, [42]);
    assert_eq!(
        outputs.last().unwrap().finish_reason,
        Some(FinishReason::Stop)
    );
    engine.cleanup().await.unwrap();
}
