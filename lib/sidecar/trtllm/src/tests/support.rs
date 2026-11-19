// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared fake OpenEngine service and engine fixtures.

use std::net::SocketAddr;
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize};

use futures::Stream;
use tokio::net::TcpListener;
use tokio::sync::{Mutex, oneshot};
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{Request, Response, Status};

use super::*;

// ---------------------------------------------------------------------------
// Fake TensorRT-LLM OpenEngine services
// ---------------------------------------------------------------------------

#[derive(Clone, Default)]
pub(super) struct FakeTrtllm {
    pub(super) requests: Arc<Mutex<Vec<pb::GenerateRequest>>>,
    pub(super) encode_requests: Arc<Mutex<Vec<pb::EncodeRequest>>>,
    pub(super) encoder_role: Arc<Mutex<Option<String>>>,
    pub(super) encoder_schema_release: Arc<Mutex<Option<String>>>,
    pub(super) model_info_override: Arc<Mutex<Option<pb::ModelInfo>>>,
    pub(super) target_dp_ranks: Arc<Mutex<Vec<Option<String>>>>,
    pub(super) affinity_headers: Arc<Mutex<Vec<(String, String)>>>,
    pub(super) aborts: Arc<Mutex<Vec<String>>>,
    pub(super) released_encoder_handoffs: Arc<Mutex<Vec<pb::EncodedMediaRef>>>,
    pub(super) encoder_release_error: Arc<Mutex<Option<tonic::Code>>>,
    pub(super) invalid_encode_response: Arc<AtomicBool>,
    pub(super) encoder_release_started: Arc<tokio::sync::Notify>,
    pub(super) encoder_release_resume: Arc<tokio::sync::Notify>,
    pub(super) encoder_release_completions: Arc<Mutex<Vec<oneshot::Receiver<()>>>>,
    pub(super) hang_encoder_release: Arc<AtomicBool>,
    pub(super) peers: Arc<Mutex<Vec<SocketAddr>>>,
    pub(super) hang: Arc<AtomicBool>,
    /// Simulates a server that has not yet accepted the request, so the RPC
    /// itself is still in flight.
    pub(super) hang_before_stream: Arc<AtomicBool>,
    /// Simulates a server whose Control service is not implemented.
    pub(super) no_control: Arc<AtomicBool>,
    /// Simulates a server that answers GetModelInfo without a context length.
    pub(super) empty_model_info: Arc<AtomicBool>,
    /// Answers `UNAVAILABLE`, the shape of an engine that is not serving yet.
    pub(super) unavailable_model_info: Arc<AtomicBool>,
    pub(super) model_info_unavailable: Arc<tokio::sync::Notify>,
    pub(super) model_info_calls: Arc<AtomicUsize>,
    pub(super) served_model_name: Arc<Mutex<Option<String>>>,
    pub(super) kv_routing: Arc<AtomicBool>,
    pub(super) server_info_error: Arc<Mutex<Option<tonic::Code>>>,
    pub(super) server_info_calls: Arc<AtomicUsize>,
    pub(super) kv_sources_error: Arc<Mutex<Option<tonic::Code>>>,
    /// Injected metadata errors returned by either RPC.
    pub(super) metadata_errors_served: Arc<AtomicUsize>,
    /// `GetKvEventSources` calls answered successfully.
    pub(super) kv_sources_served: Arc<AtomicUsize>,
    pub(super) subagent_affinity: Arc<AtomicBool>,
    pub(super) disable_events: Arc<AtomicBool>,
    pub(super) disable_load: Arc<AtomicBool>,
    pub(super) dp_size: Arc<AtomicU32>,
    pub(super) hang_load: Arc<AtomicBool>,
    pub(super) node_metadata: Arc<Mutex<Option<crate::node::NodeMetadata>>>,
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

    async fn encode(
        &self,
        request: Request<pb::EncodeRequest>,
    ) -> Result<Response<pb::EncodeResponse>, Status> {
        let request = request.into_inner();
        self.encode_requests.lock().await.push(request.clone());
        if self.hang_before_stream.load(Ordering::SeqCst) {
            std::future::pending::<()>().await;
        }
        Ok(Response::new(pb::EncodeResponse {
            request_id: if self.invalid_encode_response.load(Ordering::SeqCst) {
                "wrong-request".into()
            } else {
                request.request_id
            },
            encoded_media: Some(pb::EncodedMediaRef {
                opaque_state: vec![1, 2, 3],
            }),
        }))
    }

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
            self.model_info_unavailable.notify_one();
            return Err(Status::unavailable("not serving yet"));
        }
        if let Some(info) = self.model_info_override.lock().await.as_ref() {
            return Ok(Response::new(info.clone()));
        }
        if self.encoder_role.lock().await.is_some() {
            return Ok(Response::new(encoder_model_info()));
        }
        Ok(Response::new(pb::ModelInfo {
            model_id: "fake-model".to_string(),
            served_model_name: self
                .served_model_name
                .lock()
                .await
                .clone()
                .unwrap_or_else(|| "served-model".into()),
            max_context_length: if self.empty_model_info.load(Ordering::SeqCst) {
                None
            } else {
                Some(4096)
            },
            supports_multimodal: Some(false),
            ..Default::default()
        }))
    }

    async fn release_encoded_media(
        &self,
        request: Request<pb::ReleaseEncodedMediaRequest>,
    ) -> Result<Response<pb::ReleaseEncodedMediaResponse>, Status> {
        let cache = request
            .into_inner()
            .encoded_media
            .ok_or_else(|| Status::invalid_argument("missing encoded_media"))?;
        let (_handler, completion) = oneshot::channel();
        self.encoder_release_completions
            .lock()
            .await
            .push(completion);
        self.released_encoder_handoffs.lock().await.push(cache);
        self.encoder_release_started.notify_one();
        if self.hang_encoder_release.load(Ordering::SeqCst) {
            self.encoder_release_resume.notified().await;
        }
        if let Some(code) = self.encoder_release_error.lock().await.take() {
            return Err(Status::new(code, "encoder release failed"));
        }
        Ok(Response::new(pb::ReleaseEncodedMediaResponse {}))
    }

    async fn abort(
        &self,
        request: Request<pb::AbortRequest>,
    ) -> Result<Response<pb::AbortResponse>, Status> {
        match request.into_inner().target {
            Some(pb::abort_request::Target::RequestId(id)) => {
                self.aborts.lock().await.push(id);
            }
            other => {
                return Err(Status::invalid_argument(format!(
                    "unexpected abort target {other:?}"
                )));
            }
        }
        Ok(Response::new(pb::AbortResponse {
            status: pb::AbortStatus::Aborted as i32,
            message: "aborted".into(),
        }))
    }

    async fn get_server_info(
        &self,
        _request: Request<pb::GetServerInfoRequest>,
    ) -> Result<Response<pb::ServerInfo>, Status> {
        self.server_info_calls.fetch_add(1, Ordering::SeqCst);
        if let Some(code) = *self.server_info_error.lock().await {
            self.metadata_errors_served.fetch_add(1, Ordering::SeqCst);
            return Err(Status::new(code, "temporary metadata failure"));
        }
        let encoder_role = self.encoder_role.lock().await.clone();
        let mut identity = pb::ServerInfo::default();
        if let Some(role) = encoder_role.as_deref() {
            identity.engine_role = if role == "encode" {
                pb::EngineRole::Encoder as i32
            } else {
                pb::EngineRole::Aggregated as i32
            };
            identity.schema_release = self
                .encoder_schema_release
                .lock()
                .await
                .clone()
                .unwrap_or_else(|| include_str!("../../proto/REVISION").trim().into());
        }
        if !self.kv_routing.load(Ordering::SeqCst) {
            return if encoder_role.is_some() {
                Ok(Response::new(identity))
            } else {
                Err(Status::unimplemented("GetServerInfo is not used"))
            };
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
            ..identity
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

    async fn shutdown(
        &self,
        _request: Request<pb::ShutdownRequest>,
    ) -> Result<Response<pb::ShutdownResponse>, Status> {
        Err(Status::unimplemented("engine shutdown is not implemented"))
    }

    async fn get_kv_event_sources(
        &self,
        _request: Request<pb::GetKvEventSourcesRequest>,
    ) -> Result<Response<pb::GetKvEventSourcesResponse>, Status> {
        if let Some(code) = *self.kv_sources_error.lock().await {
            self.metadata_errors_served.fetch_add(1, Ordering::SeqCst);
            return Err(Status::new(code, "temporary metadata failure"));
        }
        if !self.kv_routing.load(Ordering::SeqCst) {
            return Err(Status::unimplemented("GetKvEventSources is not used"));
        }
        self.kv_sources_served.fetch_add(1, Ordering::SeqCst);
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

pub(super) struct FakeServer {
    pub(super) endpoint: String,
    pub(super) service: FakeTrtllm,
    shutdown: Option<oneshot::Sender<()>>,
}

impl FakeServer {
    pub(super) async fn start(service: FakeTrtllm) -> Self {
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

pub(super) fn transport(connections: usize) -> GrpcTransportConfig {
    GrpcTransportConfig {
        connections: NonZeroUsize::new(connections).expect("nonzero connections"),
        ..Default::default()
    }
}

/// A transport that gives up on startup quickly, for the paths that retry until
/// the deadline rather than failing on the first answer.
pub(super) fn impatient_transport() -> GrpcTransportConfig {
    GrpcTransportConfig {
        retry_interval: Duration::from_millis(10),
        startup_deadline: Duration::from_millis(200),
        ..transport(1)
    }
}

pub(super) fn engine(endpoint: &str, connections: usize) -> TrtllmSidecarEngine {
    engine_in_mode(endpoint, connections, AGG)
}

pub(super) fn engine_in_mode(
    endpoint: &str,
    connections: usize,
    mode: DisaggregationMode,
) -> TrtllmSidecarEngine {
    engine_with(endpoint, transport(connections), None, mode)
}

pub(super) fn encoder_consumer_engine(endpoint: &str) -> TrtllmSidecarEngine {
    TrtllmSidecarEngine::from_args(
        [
            "sidecar",
            "--grpc-endpoint",
            endpoint,
            "--model-path",
            "model-source",
            "--route-to-encoder",
        ]
        .map(str::to_string)
        .to_vec(),
    )
    .unwrap()
    .0
}

pub(super) fn request_with_encoder_result() -> PreprocessedRequest {
    let mut req = request();
    add_images(&mut req, ["https://example.com/image.png"]);
    req.encoder_result = Some(json!({
        "opaque_state": "AQID"
    }));
    req
}

/// The one place a test engine is built. Everything a test varies -- the
/// transport, whether `--context-length` was supplied, the disaggregation role
/// -- is a parameter here.
pub(super) fn engine_with(
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

pub(super) fn engine_with_kv_routing(endpoint: &str) -> TrtllmSidecarEngine {
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
    .with_supervision(RoutingSupervision::default())
}

pub(super) async fn collect(
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
