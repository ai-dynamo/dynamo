// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit + conformance tests for [`VllmSidecarEngine`].
//!
//! Everything runs against an **in-process fake OpenEngine server** — a tonic
//! service bound to an ephemeral loopback port on its own runtime thread — so
//! no real vLLM engine or GPU is involved. The fake is configurable by role so
//! the same harness drives the aggregated, prefill, and decode paths.

use std::collections::BTreeMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use dynamo_backend_common::{
    BackendError, DisaggregationMode, ErrorType, FinishReason, GenerateContext, LLMEngine,
    LLMEngineOutput, LoraAdapter, MultimodalData, MultimodalDataMap, PrefillResult,
    PreprocessedRequest, RoutingHints, SamplingOptions, StopConditions,
};
use dynamo_runtime::engine::AsyncEngineContext;
use dynamo_runtime::pipeline::{AsyncEngineContextProvider, Context};
use futures::{Stream, StreamExt};
use tonic::{Request, Response, Status};

use crate::args::TransportConfig;
use crate::client::PrimeRlClient;
use crate::engine::{
    VllmSidecarEngine, build_generate_request, disagg_json_to_kv_session, json_to_prost_struct,
    kv_session_to_disagg_json, prost_struct_to_json,
};
use crate::proto as pb;
use crate::proto::open_engine_server::{OpenEngine, OpenEngineServer};
use crate::proto::prime_rl as prime_pb;
use crate::proto::prime_rl::prime_rl_engine_server::{PrimeRlEngine, PrimeRlEngineServer};

// ============================================================================
// Fake OpenEngine server
// ============================================================================

type RespStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send>>;

/// How the fake engine should behave for a test.
#[derive(Clone)]
struct FakeConfig {
    role: pb::EngineRole,
    /// Number of `Token` events the agg/decode path emits before `Finished`.
    tokens: u32,
    model_id: String,
    served_model_name: String,
    reasoning_parser: String,
    tool_call_parser: String,
    supports_lora: bool,
    serve_prime_rl: bool,
    text_only_flush: bool,
    drain_error: bool,
}

impl Default for FakeConfig {
    fn default() -> Self {
        Self {
            role: pb::EngineRole::Aggregated,
            tokens: 4,
            model_id: "fake-model".to_string(),
            served_model_name: "fake-served".to_string(),
            reasoning_parser: "deepseek_r1".to_string(),
            tool_call_parser: "hermes".to_string(),
            supports_lora: false,
            serve_prime_rl: true,
            text_only_flush: false,
            drain_error: false,
        }
    }
}

struct FakeOpenEngine {
    cfg: FakeConfig,
    /// Incremented once per `Abort` RPC — lets tests assert abort was sent.
    abort_count: Arc<AtomicUsize>,
    /// Captures the `kv_session` of the most recent `Generate` request so the
    /// decode round-trip test can assert the handoff was lifted onto the wire.
    last_kv_session: Arc<Mutex<Option<pb::KvSessionRef>>>,
    /// Captures the `data_parallel_rank` of the most recent `Generate` request
    /// so the KV-routing test can assert the router's forced rank is forwarded.
    last_dp_rank: Arc<Mutex<Option<u32>>>,
    last_lora_name: Arc<Mutex<Option<String>>>,
    adapters: Arc<Mutex<BTreeMap<String, pb::LoraAdapter>>>,
}

#[derive(Default)]
struct FakePrimeRlEngine {
    last_distributed_update: Arc<Mutex<Option<prime_pb::UpdateWeightsFromDistributedRequest>>>,
}

fn admin_ok() -> Response<prime_pb::AdminResponse> {
    Response::new(prime_pb::AdminResponse {
        status: "ok".to_string(),
        message: String::new(),
    })
}

#[tonic::async_trait]
impl PrimeRlEngine for FakePrimeRlEngine {
    async fn liveness_probe(
        &self,
        _request: Request<prime_pb::LivenessProbeRequest>,
    ) -> Result<Response<prime_pb::AdminResponse>, Status> {
        Ok(admin_ok())
    }

    async fn pause_generation(
        &self,
        _request: Request<prime_pb::PauseGenerationRequest>,
    ) -> Result<Response<prime_pb::AdminResponse>, Status> {
        Ok(admin_ok())
    }

    async fn resume_generation(
        &self,
        _request: Request<prime_pb::ResumeGenerationRequest>,
    ) -> Result<Response<prime_pb::AdminResponse>, Status> {
        Ok(admin_ok())
    }

    async fn flush_cache(
        &self,
        _request: Request<prime_pb::FlushCacheRequest>,
    ) -> Result<Response<prime_pb::AdminResponse>, Status> {
        Ok(admin_ok())
    }

    async fn abort_request(
        &self,
        _request: Request<prime_pb::AbortRequestRequest>,
    ) -> Result<Response<prime_pb::AdminResponse>, Status> {
        Ok(admin_ok())
    }

    async fn init_weights_update_group(
        &self,
        _request: Request<prime_pb::InitWeightsUpdateGroupRequest>,
    ) -> Result<Response<prime_pb::AdminResponse>, Status> {
        Ok(admin_ok())
    }

    async fn destroy_weights_update_group(
        &self,
        _request: Request<prime_pb::DestroyWeightsUpdateGroupRequest>,
    ) -> Result<Response<prime_pb::AdminResponse>, Status> {
        Ok(admin_ok())
    }

    async fn update_weights_from_disk(
        &self,
        request: Request<prime_pb::UpdateWeightsFromDiskRequest>,
    ) -> Result<Response<prime_pb::WeightUpdateResponse>, Status> {
        let request = request.into_inner();
        Ok(Response::new(prime_pb::WeightUpdateResponse {
            status: "ok".to_string(),
            message: String::new(),
            weight_version: request.weight_version,
        }))
    }

    async fn update_weights_from_distributed(
        &self,
        request: Request<prime_pb::UpdateWeightsFromDistributedRequest>,
    ) -> Result<Response<prime_pb::WeightUpdateResponse>, Status> {
        let request = request.into_inner();
        let weight_version = request.weight_version.clone();
        *self.last_distributed_update.lock().unwrap() = Some(request);
        Ok(Response::new(prime_pb::WeightUpdateResponse {
            status: "ok".to_string(),
            message: String::new(),
            weight_version,
        }))
    }

    async fn get_weight_version(
        &self,
        _request: Request<prime_pb::GetWeightVersionRequest>,
    ) -> Result<Response<prime_pb::GetWeightVersionResponse>, Status> {
        Ok(Response::new(prime_pb::GetWeightVersionResponse {
            status: "ok".to_string(),
            weight_version: "v-test".to_string(),
        }))
    }
}

#[tonic::async_trait]
impl OpenEngine for FakeOpenEngine {
    type GenerateStream = RespStream<pb::GenerateResponse>;
    type DrainStream = RespStream<pb::DrainResponse>;
    type SubscribeKvEventsStream = RespStream<pb::SubscribeKvEventsResponse>;
    type SubscribeRuntimeEventsStream = RespStream<pb::SubscribeRuntimeEventsResponse>;

    async fn generate(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let req = request.into_inner();
        *self.last_kv_session.lock().unwrap() = req.kv.as_ref().and_then(|kv| kv.session.clone());
        *self.last_dp_rank.lock().unwrap() = req.kv.as_ref().and_then(|kv| kv.data_parallel_rank);
        *self.last_lora_name.lock().unwrap() = Some(req.lora_name.clone());
        let request_id = req.request_id;
        let cfg = self.cfg.clone();

        let stream = async_stream::stream! {
            match cfg.role {
                pb::EngineRole::Prefill => {
                    // Prefill returns a single KV handoff and no decoded tokens.
                    // Mirror real vLLM prefill: typed params via attributes_struct.
                    let kv = pb::KvSessionRef {
                        session_id: request_id.clone(),
                        transfer_backend: "NixlConnector".to_string(),
                        endpoints: Vec::new(),
                        dp_rank: 0,
                        attributes_struct: json_to_prost_struct(&serde_json::json!({
                            "remote_engine_id": "engine-fake",
                            "remote_block_ids": [1, 2, 3],
                        })),
                    };
                    yield Ok(pb::GenerateResponse {
                        request_id: request_id.clone(),
                        event: Some(pb::generate_response::Event::PrefillReady(
                            pb::PrefillReady { kv_session: Some(kv) },
                        )),
                        usage: Some(pb::Usage {
                            prompt_tokens: 3,
                            completion_tokens: 1,
                            total_tokens: 4,
                            ..Default::default()
                        }),
                    });
                }
                _ => {
                    for i in 0..cfg.tokens {
                        // A small per-token delay so a mid-stream cancel can be
                        // interleaved deterministically.
                        tokio::time::sleep(Duration::from_millis(10)).await;
                        yield Ok(pb::GenerateResponse {
                            request_id: request_id.clone(),
                            event: Some(pb::generate_response::Event::Token(pb::TokenOutput {
                                output_index: Some(0),
                                tokens: vec![pb::TokenInfo {
                                    token_id: 1000 + i,
                                    token: format!("t{i}"),
                                    logprob: Some(-0.25),
                                    rank: Some(1),
                                    candidates: vec![pb::LogProb {
                                        token_id: 1000 + i,
                                        logprob: -0.25,
                                        token: format!("t{i}"),
                                        rank: Some(1),
                                    }],
                                }],
                                text: format!("t{i}"),
                            })),
                            usage: None,
                        });
                    }
                    if cfg.text_only_flush {
                        yield Ok(pb::GenerateResponse {
                            request_id: request_id.clone(),
                            event: Some(pb::generate_response::Event::Token(pb::TokenOutput {
                                output_index: Some(0),
                                tokens: Vec::new(),
                                text: "<flush>".to_string(),
                            })),
                            usage: None,
                        });
                    }
                    yield Ok(pb::GenerateResponse {
                        request_id: request_id.clone(),
                        event: Some(pb::generate_response::Event::Finished(
                            pb::GenerationFinished {
                                output_index: Some(0),
                                reason: pb::FinishReason::Stop as i32,
                                message: String::new(),
                                stop_match: None,
                            },
                        )),
                        usage: Some(pb::Usage {
                            prompt_tokens: 3,
                            completion_tokens: cfg.tokens,
                            total_tokens: 3 + cfg.tokens,
                            ..Default::default()
                        }),
                    });
                }
            }
        };
        Ok(Response::new(Box::pin(stream)))
    }

    async fn get_engine_info(
        &self,
        _request: Request<pb::GetEngineInfoRequest>,
    ) -> Result<Response<pb::EngineInfo>, Status> {
        Ok(Response::new(pb::EngineInfo {
            engine_name: "vllm".to_string(),
            engine_version: "0.0.0-fake".to_string(),
            role: self.cfg.role as i32,
            instance_id: "fake-instance".to_string(),
            supported_models: vec![self.cfg.model_id.clone()],
            parallelism: Some(pb::ParallelismInfo {
                tensor_parallel_size: Some(1),
                pipeline_parallel_size: Some(1),
                data_parallel_size: Some(1),
                data_parallel_rank: Some(0),
                data_parallel_start_rank: Some(0),
            }),
            kv_connector: None,
            schema_revision: 1,
            minimum_client_revision: 1,
            schema_release: "openengine-test-r1".to_string(),
        }))
    }

    async fn get_model_info(
        &self,
        _request: Request<pb::GetModelInfoRequest>,
    ) -> Result<Response<pb::ModelInfo>, Status> {
        Ok(Response::new(pb::ModelInfo {
            model_id: self.cfg.model_id.clone(),
            served_model_name: self.cfg.served_model_name.clone(),
            served_model_aliases: Vec::new(),
            max_context_length: Some(4096),
            max_output_tokens: Some(2048),
            kv_block_size: Some(16),
            total_kv_blocks: Some(1000),
            max_running_requests: Some(256),
            max_batched_tokens: Some(8192),
            tokenizer_modes: Vec::new(),
            supports_text_input: Some(true),
            supports_token_ids_input: Some(true),
            generation: Some(pb::GenerationCapabilities {
                prompt_logprobs: Some(pb::LogprobCapabilities {
                    supported: Some(true),
                    ..Default::default()
                }),
                output_logprobs: Some(pb::LogprobCapabilities {
                    supported: Some(true),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            supports_lora: Some(self.cfg.supports_lora),
            supports_multimodal: Some(false),
            reasoning_parser: self.cfg.reasoning_parser.clone(),
            tool_call_parser: self.cfg.tool_call_parser.clone(),
        }))
    }

    async fn get_load(
        &self,
        _request: Request<pb::GetLoadRequest>,
    ) -> Result<Response<pb::LoadInfo>, Status> {
        Ok(Response::new(pb::LoadInfo {
            instance_id: "fake-instance".to_string(),
            total_kv_blocks: Some(1000),
            ..Default::default()
        }))
    }

    async fn health(
        &self,
        _request: Request<pb::HealthRequest>,
    ) -> Result<Response<pb::HealthResponse>, Status> {
        Ok(Response::new(pb::HealthResponse {
            state: pb::HealthState::Ready as i32,
            checks: Vec::new(),
        }))
    }

    async fn abort(
        &self,
        _request: Request<pb::AbortRequest>,
    ) -> Result<Response<pb::AbortResponse>, Status> {
        self.abort_count.fetch_add(1, Ordering::SeqCst);
        Ok(Response::new(pb::AbortResponse {
            status: pb::AbortStatus::Aborted as i32,
            message: String::new(),
        }))
    }

    async fn drain(
        &self,
        _request: Request<pb::DrainRequest>,
    ) -> Result<Response<Self::DrainStream>, Status> {
        let drain_error = self.cfg.drain_error;
        let stream = async_stream::stream! {
            yield Ok(pb::DrainResponse {
                event: Some(pb::drain_response::Event::State(
                    pb::DrainState::Started as i32,
                )),
                ..Default::default()
            });
            if drain_error {
                yield Ok(pb::DrainResponse {
                    event: Some(pb::drain_response::Event::Error(pb::EngineError {
                        code: pb::ErrorCode::Overloaded as i32,
                        message: "drain deadline elapsed".to_string(),
                        retryable: true,
                        ..Default::default()
                    })),
                    ..Default::default()
                });
            } else {
                yield Ok(pb::DrainResponse {
                    event: Some(pb::drain_response::Event::State(
                        pb::DrainState::Complete as i32,
                    )),
                    ..Default::default()
                });
            }
        };
        Ok(Response::new(Box::pin(stream)))
    }

    async fn load_lora(
        &self,
        request: Request<pb::LoadLoraRequest>,
    ) -> Result<Response<pb::LoadLoraResponse>, Status> {
        if !self.cfg.supports_lora {
            return Err(Status::failed_precondition("LoRA disabled"));
        }
        let adapter = request
            .into_inner()
            .adapter
            .ok_or_else(|| Status::invalid_argument("adapter required"))?;
        let mut adapters = self.adapters.lock().unwrap();
        let already_loaded = adapters.get(&adapter.lora_name) == Some(&adapter);
        adapters.insert(adapter.lora_name.clone(), adapter.clone());
        Ok(Response::new(pb::LoadLoraResponse {
            adapter: Some(adapter),
            already_loaded,
        }))
    }

    async fn unload_lora(
        &self,
        request: Request<pb::UnloadLoraRequest>,
    ) -> Result<Response<pb::UnloadLoraResponse>, Status> {
        let name = request.into_inner().lora_name;
        let adapter = self
            .adapters
            .lock()
            .unwrap()
            .remove(&name)
            .ok_or_else(|| Status::not_found("adapter not loaded"))?;
        Ok(Response::new(pb::UnloadLoraResponse {
            adapter: Some(adapter),
        }))
    }

    async fn list_loras(
        &self,
        _request: Request<pb::ListLorasRequest>,
    ) -> Result<Response<pb::ListLorasResponse>, Status> {
        Ok(Response::new(pb::ListLorasResponse {
            adapters: self.adapters.lock().unwrap().values().cloned().collect(),
        }))
    }

    async fn get_kv_connector_info(
        &self,
        _request: Request<pb::GetKvConnectorInfoRequest>,
    ) -> Result<Response<pb::KvConnectorInfo>, Status> {
        Ok(Response::new(pb::KvConnectorInfo {
            schema_version: Some(1),
            ..Default::default()
        }))
    }

    async fn get_kv_event_sources(
        &self,
        _request: Request<pb::GetKvEventSourcesRequest>,
    ) -> Result<Response<pb::GetKvEventSourcesResponse>, Status> {
        Ok(Response::new(pb::GetKvEventSourcesResponse {
            sources: Vec::new(),
        }))
    }

    async fn subscribe_kv_events(
        &self,
        _request: Request<pb::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        Err(Status::unimplemented(
            "subscribe_kv_events not supported by fake",
        ))
    }

    async fn subscribe_runtime_events(
        &self,
        _request: Request<pb::SubscribeRuntimeEventsRequest>,
    ) -> Result<Response<Self::SubscribeRuntimeEventsStream>, Status> {
        Err(Status::unimplemented(
            "subscribe_runtime_events not supported by fake",
        ))
    }
}

struct FakeHandle {
    endpoint: String,
    abort_count: Arc<AtomicUsize>,
    last_kv_session: Arc<Mutex<Option<pb::KvSessionRef>>>,
    last_dp_rank: Arc<Mutex<Option<u32>>>,
    last_lora_name: Arc<Mutex<Option<String>>>,
    last_distributed_update: Arc<Mutex<Option<prime_pb::UpdateWeightsFromDistributedRequest>>>,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
}

impl FakeHandle {
    fn shutdown(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
    }
}

impl Drop for FakeHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Bind a fake OpenEngine server to an ephemeral port on a dedicated runtime
/// thread and return its `http://host:port` endpoint once it is listening.
fn spawn_fake_engine(cfg: FakeConfig) -> FakeHandle {
    let abort_count = Arc::new(AtomicUsize::new(0));
    let last_kv_session = Arc::new(Mutex::new(None));
    let last_dp_rank = Arc::new(Mutex::new(None));
    let last_lora_name = Arc::new(Mutex::new(None));
    let adapters = Arc::new(Mutex::new(BTreeMap::new()));
    let last_distributed_update = Arc::new(Mutex::new(None));
    let (tx, rx) = std::sync::mpsc::channel();
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();

    let svc_abort = abort_count.clone();
    let svc_kv = last_kv_session.clone();
    let svc_dp = last_dp_rank.clone();
    let svc_lora_name = last_lora_name.clone();
    let svc_adapters = adapters.clone();
    let svc_distributed_update = last_distributed_update.clone();
    let serve_prime_rl = cfg.serve_prime_rl;
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("fake engine runtime");
        rt.block_on(async move {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("bind fake engine");
            let addr = listener.local_addr().expect("local_addr");
            tx.send(addr).expect("send addr");

            let svc = FakeOpenEngine {
                cfg,
                abort_count: svc_abort,
                last_kv_session: svc_kv,
                last_dp_rank: svc_dp,
                last_lora_name: svc_lora_name,
                adapters: svc_adapters,
            };
            let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
            if serve_prime_rl {
                tonic::transport::Server::builder()
                    .add_service(OpenEngineServer::new(svc))
                    .add_service(PrimeRlEngineServer::new(FakePrimeRlEngine {
                        last_distributed_update: svc_distributed_update,
                    }))
                    .serve_with_incoming_shutdown(incoming, async move {
                        let _ = shutdown_rx.await;
                    })
                    .await
                    .expect("serve fake engine");
            } else {
                tonic::transport::Server::builder()
                    .add_service(OpenEngineServer::new(svc))
                    .serve_with_incoming_shutdown(incoming, async move {
                        let _ = shutdown_rx.await;
                    })
                    .await
                    .expect("serve fake engine");
            }
        });
    });

    let addr = rx.recv().expect("fake engine never bound");
    FakeHandle {
        endpoint: format!("http://{addr}"),
        abort_count,
        last_kv_session,
        last_dp_rank,
        last_lora_name,
        last_distributed_update,
        shutdown_tx: Some(shutdown_tx),
    }
}

// ============================================================================
// Test helpers
// ============================================================================

/// Short, test-friendly timeouts: fail fast instead of hanging. Uses a 2-conn
/// pool so the round-robin / multi-connection `start()` path is exercised.
fn test_transport() -> TransportConfig {
    TransportConfig {
        connect_timeout: Duration::from_secs(5),
        poll_interval: Duration::from_millis(50),
        deadline: Duration::from_secs(10),
        connections: 2,
    }
}

fn engine_for(handle: &FakeHandle, mode: DisaggregationMode) -> VllmSidecarEngine {
    VllmSidecarEngine::new(handle.endpoint.clone(), test_transport(), mode, true)
}

fn engine_without_rl(handle: &FakeHandle, mode: DisaggregationMode) -> VllmSidecarEngine {
    VllmSidecarEngine::new(handle.endpoint.clone(), test_transport(), mode, false)
}

fn fresh_ctx() -> Arc<dyn AsyncEngineContext> {
    Context::new(()).context()
}

fn gen_ctx(ctx: Arc<dyn AsyncEngineContext>) -> GenerateContext {
    GenerateContext::new(ctx, None)
}

fn request(max_tokens: Option<u32>) -> PreprocessedRequest {
    PreprocessedRequest::builder()
        .model("fake-model".to_string())
        .token_ids(vec![1, 2, 3])
        .stop_conditions(StopConditions {
            max_tokens,
            ..Default::default()
        })
        .sampling_options(SamplingOptions::default())
        .output_options(Default::default())
        .build()
        .expect("build request")
}

fn request_with_prefill_result(disagg: serde_json::Value) -> PreprocessedRequest {
    let mut req = request(Some(8));
    req.prefill_result = Some(PrefillResult {
        disaggregated_params: disagg,
        prompt_tokens_details: None,
        generate_metadata: None,
    });
    req
}

async fn collect_ok(
    stream: futures::stream::BoxStream<
        'static,
        Result<LLMEngineOutput, dynamo_backend_common::DynamoError>,
    >,
) -> Vec<LLMEngineOutput> {
    stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(|item| item.expect("engine yielded Err in test"))
        .collect()
}

#[tokio::test]
async fn prime_rl_admin_callback_forwards_typed_distributed_update() {
    let handle = spawn_fake_engine(FakeConfig::default());
    let channel = tonic::transport::Endpoint::from_shared(handle.endpoint.clone())
        .expect("valid fake endpoint")
        .connect()
        .await
        .expect("connect to fake engine");
    let callback = crate::prime_rl::callback(
        crate::prime_rl::Route::UpdateWeightsFromDistributed,
        PrimeRlClient::new(channel),
        tokio_util::sync::CancellationToken::new(),
    );

    let response = callback(serde_json::json!({
        "weight_dir": "/weights/v7",
        "weight_version": "v7",
        "engine_rpc": "update_weights_from_path",
        "allow_unpaused": false,
        "reset_prefix_cache": true,
    }))
    .await
    .expect("admin callback response");

    assert_eq!(response["status"], "ok");
    assert_eq!(response["version"], "v7");
    let observed = handle
        .last_distributed_update
        .lock()
        .unwrap()
        .clone()
        .expect("forwarded request");
    assert_eq!(observed.weight_dir, "/weights/v7");
    assert_eq!(observed.weight_version, "v7");
    assert!(!observed.allow_unpaused);
    assert!(observed.reset_prefix_cache);
}

// ============================================================================
// Conformance
// ============================================================================

#[tokio::test]
async fn sidecar_passes_conformance() {
    let handle = spawn_fake_engine(FakeConfig::default());
    dynamo_backend_common::testing::run_conformance(|| {
        engine_for(&handle, DisaggregationMode::Aggregated)
    })
    .await
    .expect("vllm sidecar must satisfy conformance");
}

// ============================================================================
// Lifecycle + discovery
// ============================================================================

#[tokio::test]
async fn start_advertises_discovered_metadata() {
    let handle = spawn_fake_engine(FakeConfig::default());
    let engine = engine_for(&handle, DisaggregationMode::Aggregated);

    let cfg = engine.start(0).await.expect("start");
    assert_eq!(cfg.model, "fake-model");
    assert_eq!(cfg.served_model_name.as_deref(), Some("fake-served"));
    assert_eq!(
        cfg.runtime_data.get("engine_generate_protocol_version"),
        Some(&serde_json::json!(1)),
        "versioned route publication must be paired with its model-card capability"
    );
    let llm = cfg.llm.expect("sidecar must advertise LLM registration");
    assert_eq!(llm.context_length, Some(4096));
    assert_eq!(llm.kv_cache_block_size, Some(16));
    assert_eq!(llm.total_kv_blocks, Some(1000));
    assert_eq!(llm.max_num_seqs, Some(256));
    assert_eq!(llm.max_num_batched_tokens, Some(8192));
    assert!(llm.bootstrap_host.is_none());

    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn rl_enabled_start_rejects_openengine_without_prime_extension() {
    let handle = spawn_fake_engine(FakeConfig {
        serve_prime_rl: false,
        ..FakeConfig::default()
    });
    let engine = engine_for(&handle, DisaggregationMode::Aggregated);

    let error = engine
        .start(0)
        .await
        .expect_err("RL-enabled sidecar must require PrimeRlEngine");

    assert!(error.message().contains("prime_rl.engine.v1.PrimeRlEngine"));
}

#[tokio::test]
async fn rl_disabled_start_accepts_canonical_openengine_only() {
    let handle = spawn_fake_engine(FakeConfig {
        serve_prime_rl: false,
        ..FakeConfig::default()
    });
    let engine = engine_without_rl(&handle, DisaggregationMode::Aggregated);

    let config = engine.start(0).await.expect("canonical OpenEngine start");

    assert_eq!(config.model, "fake-model");
    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn double_start_is_rejected() {
    let handle = spawn_fake_engine(FakeConfig::default());
    let engine = engine_for(&handle, DisaggregationMode::Aggregated);
    engine.start(0).await.expect("first start");

    let err = engine.start(0).await.expect_err("second start must fail");
    assert_eq!(
        err.error_type(),
        ErrorType::Backend(BackendError::EngineShutdown)
    );
    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn watch_reports_engine_shutdown_when_health_endpoint_disappears() {
    let mut handle = spawn_fake_engine(FakeConfig::default());
    let engine = engine_for(&handle, DisaggregationMode::Aggregated);
    engine.start(0).await.expect("start");

    handle.shutdown();

    let err = tokio::time::timeout(Duration::from_secs(3), engine.watch())
        .await
        .expect("watch should notice fake engine shutdown")
        .expect_err("engine shutdown should fail watch");
    assert_eq!(
        err.error_type(),
        ErrorType::Backend(BackendError::EngineShutdown)
    );

    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn start_rejects_role_mismatch() {
    // Engine registered as aggregated, but the live engine reports prefill.
    let handle = spawn_fake_engine(FakeConfig {
        role: pb::EngineRole::Prefill,
        ..FakeConfig::default()
    });
    let engine = engine_for(&handle, DisaggregationMode::Aggregated);

    let err = engine.start(0).await.expect_err("role mismatch must fail");
    assert_eq!(
        err.error_type(),
        ErrorType::Backend(BackendError::InvalidArgument)
    );
}

#[tokio::test]
async fn generate_before_start_errors() {
    let handle = spawn_fake_engine(FakeConfig::default());
    let engine = engine_for(&handle, DisaggregationMode::Aggregated);

    let result = engine
        .generate(request(Some(2)), gen_ctx(fresh_ctx()))
        .await;
    let Err(err) = result else {
        panic!("generate before start must fail");
    };
    assert_eq!(
        err.error_type(),
        ErrorType::Backend(BackendError::EngineShutdown)
    );
}

// ============================================================================
// Generate happy path + cancellation
// ============================================================================

#[tokio::test]
async fn generate_streams_tokens_then_stop_terminal() {
    let handle = spawn_fake_engine(FakeConfig {
        tokens: 4,
        ..FakeConfig::default()
    });
    let engine = engine_for(&handle, DisaggregationMode::Aggregated);
    engine.start(0).await.unwrap();

    let stream = engine
        .generate(request(Some(64)), gen_ctx(fresh_ctx()))
        .await
        .expect("stream");
    let chunks = collect_ok(stream).await;

    // 4 token chunks + 1 terminal.
    let token_chunks = &chunks[..chunks.len() - 1];
    let total: usize = token_chunks.iter().map(|c| c.token_ids.len()).sum();
    assert_eq!(total, 4);
    assert!(token_chunks.iter().all(|c| c.finish_reason.is_none()));
    assert!(
        token_chunks
            .iter()
            .all(|chunk| chunk.log_probs.as_deref() == Some(&[-0.25]))
    );
    assert!(token_chunks.iter().all(|chunk| {
        chunk.top_logprobs.as_ref().is_some_and(|positions| {
            positions.len() == 1
                && positions[0].len() == 1
                && positions[0][0].token_id == chunk.token_ids[0]
        })
    }));

    let terminal = chunks.last().unwrap();
    assert!(matches!(terminal.finish_reason, Some(FinishReason::Stop)));
    let usage = terminal.completion_usage.as_ref().unwrap();
    assert_eq!(usage.completion_tokens, 4);

    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn generate_preserves_text_only_decoder_flush() {
    let handle = spawn_fake_engine(FakeConfig {
        tokens: 1,
        text_only_flush: true,
        ..FakeConfig::default()
    });
    let engine = engine_for(&handle, DisaggregationMode::Aggregated);
    engine.start(0).await.unwrap();

    let chunks = collect_ok(
        engine
            .generate(request(Some(8)), gen_ctx(fresh_ctx()))
            .await
            .expect("stream"),
    )
    .await;
    assert!(
        chunks.iter().any(|chunk| {
            chunk.token_ids.is_empty() && chunk.text.as_deref() == Some("<flush>")
        })
    );

    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn generate_observes_midstream_cancellation() {
    // Enough tokens (with per-token delay) that the stream can't finish before
    // we cancel after the first token.
    let handle = spawn_fake_engine(FakeConfig {
        tokens: 50,
        ..FakeConfig::default()
    });
    let engine = engine_for(&handle, DisaggregationMode::Aggregated);
    engine.start(0).await.unwrap();

    let ctx = fresh_ctx();
    let mut stream = engine
        .generate(request(Some(10_000)), gen_ctx(ctx.clone()))
        .await
        .expect("stream");

    let first = stream.next().await.expect("first item").expect("ok");
    assert!(
        first.finish_reason.is_none(),
        "first item should be a token"
    );

    ctx.stop_generating();

    let rest: Vec<_> = stream.collect().await;
    let last = rest
        .last()
        .expect("terminal after cancel")
        .as_ref()
        .unwrap();
    assert!(
        matches!(last.finish_reason, Some(FinishReason::Cancelled)),
        "expected Cancelled terminal, got {:?}",
        last.finish_reason
    );

    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn generate_cancelled_before_first_poll_yields_cancelled() {
    let handle = spawn_fake_engine(FakeConfig::default());
    let engine = engine_for(&handle, DisaggregationMode::Aggregated);
    engine.start(0).await.unwrap();

    let ctx = fresh_ctx();
    ctx.stop_generating();
    let stream = engine
        .generate(request(Some(10_000)), gen_ctx(ctx))
        .await
        .expect("stream");
    let chunks = collect_ok(stream).await;

    assert_eq!(chunks.len(), 1, "pre-stopped request must short-circuit");
    assert!(matches!(
        chunks[0].finish_reason,
        Some(FinishReason::Cancelled)
    ));

    engine.cleanup().await.unwrap();
}

// ============================================================================
// Abort + drain
// ============================================================================

#[tokio::test]
async fn abort_sends_abort_rpc() {
    let handle = spawn_fake_engine(FakeConfig::default());
    let engine = engine_for(&handle, DisaggregationMode::Aggregated);
    engine.start(0).await.unwrap();

    let ctx = fresh_ctx();
    engine.abort(ctx).await;
    assert_eq!(handle.abort_count.load(Ordering::SeqCst), 1);

    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn abort_before_start_is_noop() {
    let handle = spawn_fake_engine(FakeConfig::default());
    let engine = engine_for(&handle, DisaggregationMode::Aggregated);

    // No client yet — must not panic or call the engine.
    engine.abort(fresh_ctx()).await;
    assert_eq!(handle.abort_count.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn lora_lifecycle_proxies_to_openengine() {
    let handle = spawn_fake_engine(FakeConfig {
        supports_lora: true,
        ..FakeConfig::default()
    });
    let engine = engine_for(&handle, DisaggregationMode::Aggregated);
    let config = engine.start(0).await.unwrap();
    assert!(config.llm.as_ref().is_some_and(|llm| llm.supports_lora));

    let adapter = LoraAdapter {
        id: 17,
        name: "adapter-a".to_string(),
        path: "/tmp/adapter-a".to_string(),
    };
    assert_eq!(engine.load_lora(adapter.clone()).await.unwrap(), adapter);
    assert_eq!(engine.list_loras().await.unwrap(), vec![adapter.clone()]);
    assert_eq!(engine.unload_lora("adapter-a").await.unwrap(), adapter);
    assert!(engine.list_loras().await.unwrap().is_empty());

    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn generate_forwards_lora_routing_hint() {
    let handle = spawn_fake_engine(FakeConfig {
        supports_lora: true,
        ..FakeConfig::default()
    });
    let engine = engine_for(&handle, DisaggregationMode::Aggregated);
    engine.start(0).await.unwrap();

    let mut req = request(Some(1));
    req.routing = Some(RoutingHints {
        lora_name: Some("adapter-a".to_string()),
        ..Default::default()
    });
    let stream = engine
        .generate(req, gen_ctx(fresh_ctx()))
        .await
        .expect("stream");
    let _ = collect_ok(stream).await;

    assert_eq!(
        handle.last_lora_name.lock().unwrap().as_deref(),
        Some("adapter-a")
    );
    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn drain_consumes_stream_to_completion() {
    let handle = spawn_fake_engine(FakeConfig::default());
    let engine = engine_for(&handle, DisaggregationMode::Aggregated);
    engine.start(0).await.unwrap();

    engine.begin_drain().await.expect("drain must succeed");

    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn drain_propagates_terminal_engine_error() {
    let handle = spawn_fake_engine(FakeConfig {
        drain_error: true,
        ..FakeConfig::default()
    });
    let engine = engine_for(&handle, DisaggregationMode::Aggregated);
    engine.start(0).await.unwrap();

    let error = engine.begin_drain().await.expect_err("drain must fail");
    assert!(error.message().contains("drain deadline elapsed"));

    engine.cleanup().await.unwrap();
}

// ============================================================================
// Disaggregation
// ============================================================================

#[tokio::test]
async fn prefill_emits_terminal_with_disaggregated_params() {
    let handle = spawn_fake_engine(FakeConfig {
        role: pb::EngineRole::Prefill,
        ..FakeConfig::default()
    });
    let engine = engine_for(&handle, DisaggregationMode::Prefill);
    engine.start(0).await.unwrap();

    let stream = engine
        .generate(request(Some(5)), gen_ctx(fresh_ctx()))
        .await
        .expect("stream");
    let chunks = collect_ok(stream).await;

    assert_eq!(chunks.len(), 1, "prefill must emit a single terminal");
    let terminal = &chunks[0];
    assert!(matches!(terminal.finish_reason, Some(FinishReason::Stop)));
    let params = terminal
        .disaggregated_params
        .as_ref()
        .expect("prefill terminal must carry disaggregated_params");
    assert_eq!(params["transfer_backend"], "NixlConnector");
    // Typed params pass through as native JSON with types preserved.
    assert_eq!(
        params["attributes_struct"]["remote_engine_id"],
        "engine-fake"
    );
    assert_eq!(
        params["attributes_struct"]["remote_block_ids"],
        serde_json::json!([1, 2, 3])
    );

    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn decode_rejects_request_without_prefill_result() {
    let handle = spawn_fake_engine(FakeConfig {
        role: pb::EngineRole::Decode,
        ..FakeConfig::default()
    });
    let engine = engine_for(&handle, DisaggregationMode::Decode);
    engine.start(0).await.unwrap();

    let result = engine
        .generate(request(Some(2)), gen_ctx(fresh_ctx()))
        .await;
    let Err(err) = result else {
        panic!("decode without prefill_result must fail");
    };
    assert_eq!(
        err.error_type(),
        ErrorType::Backend(BackendError::InvalidArgument)
    );

    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn decode_lifts_prefill_result_onto_kv_session() {
    let handle = spawn_fake_engine(FakeConfig {
        role: pb::EngineRole::Decode,
        ..FakeConfig::default()
    });
    let engine = engine_for(&handle, DisaggregationMode::Decode);
    engine.start(0).await.unwrap();

    let disagg = serde_json::json!({
        "session_id": "sess-xyz",
        "transfer_backend": "NixlConnector",
        "dp_rank": 2,
        "attributes_struct": { "remote_engine_id": "engine-fake", "remote_port": 20097 },
    });
    let stream = engine
        .generate(request_with_prefill_result(disagg), gen_ctx(fresh_ctx()))
        .await
        .expect("stream");
    let _ = collect_ok(stream).await;

    let captured = handle
        .last_kv_session
        .lock()
        .unwrap()
        .clone()
        .expect("decode request must carry a kv_session");
    assert_eq!(captured.session_id, "sess-xyz");
    assert_eq!(captured.transfer_backend, "NixlConnector");
    assert_eq!(captured.dp_rank, 2);
    // The typed handoff is forwarded on attributes_struct with types intact.
    let attrs = prost_struct_to_json(
        captured
            .attributes_struct
            .as_ref()
            .expect("attributes_struct forwarded"),
    );
    assert_eq!(attrs["remote_engine_id"], "engine-fake");
    assert_eq!(attrs["remote_port"], serde_json::json!(20097));

    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn generate_forwards_router_forced_dp_rank() {
    let handle = spawn_fake_engine(FakeConfig::default());
    let engine = engine_for(&handle, DisaggregationMode::Aggregated);
    engine.start(0).await.unwrap();

    let mut req = request(Some(4));
    req.routing = Some(RoutingHints {
        dp_rank: Some(3),
        ..Default::default()
    });
    let stream = engine
        .generate(req, gen_ctx(fresh_ctx()))
        .await
        .expect("stream");
    let _ = collect_ok(stream).await;

    assert_eq!(
        *handle.last_dp_rank.lock().unwrap(),
        Some(3),
        "the router's forced dp_rank must be forwarded on the OpenEngine request"
    );

    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn generate_without_routing_leaves_dp_rank_unset() {
    let handle = spawn_fake_engine(FakeConfig::default());
    let engine = engine_for(&handle, DisaggregationMode::Aggregated);
    engine.start(0).await.unwrap();

    let stream = engine
        .generate(request(Some(4)), gen_ctx(fresh_ctx()))
        .await
        .expect("stream");
    let _ = collect_ok(stream).await;

    assert_eq!(
        *handle.last_dp_rank.lock().unwrap(),
        None,
        "with no routing hint the engine selects its own rank (field stays unset)"
    );

    engine.cleanup().await.unwrap();
}

// ============================================================================
// Media forwarding (build_generate_request)
// ============================================================================

fn request_with_media(map: MultimodalDataMap) -> PreprocessedRequest {
    let mut req = request(Some(16));
    req.multi_modal_data = Some(map);
    req
}

/// Build a `MultimodalData::Url` (the parsed-`url::Url` variant) via serde, so
/// the test exercises the real typed variant without pulling the `url` crate
/// into the sidecar's deliberately lean dependency set.
fn url_media(s: &str) -> MultimodalData {
    serde_json::from_value(serde_json::json!({ "Url": s })).expect("parse MultimodalData::Url")
}

/// Synthesize a `MultimodalData::Decoded` (the RDMA-descriptor variant) via
/// serde. Its inner `RdmaMediaDataDescriptor` fields are `pub(crate)` to
/// `dynamo-llm`, so the sidecar test crate cannot build one with struct
/// literal syntax — deserialization is the only route. The exact payload is
/// irrelevant: `build_media` rejects on the variant alone, before reading any
/// field.
fn decoded_media() -> MultimodalData {
    serde_json::from_value(serde_json::json!({
        "Decoded": {
            "nixl_metadata": "",
            "nixl_descriptor": { "addr": 0, "size": 0, "mem_type": "Dram", "device_id": 0 },
            "shape": [1, 1, 1],
            "dtype": "UINT8",
            "metadata": null,
        }
    }))
    .expect("synthesize MultimodalData::Decoded")
}

#[test]
fn build_media_passes_through_http_url_as_url_source() {
    let map = MultimodalDataMap::from([(
        "image_url".to_string(),
        vec![url_media("http://example.com/cat.png")],
    )]);
    let req = build_generate_request(&request_with_media(map), "req-1", false).unwrap();

    assert_eq!(req.media.len(), 1);
    let item = &req.media[0];
    assert_eq!(item.modality, pb::Modality::Image as i32);
    assert_eq!(
        item.source,
        Some(pb::media_item::Source::Url(
            "http://example.com/cat.png".to_string()
        ))
    );
}

#[test]
fn build_media_passes_through_data_uri_as_data_uri_source() {
    // A `data:` string — whether it arrives as a parsed `Url` or a `RawUrl`
    // string — must land on the `data_uri` source so the engine decodes it
    // rather than trying to fetch it over the network.
    let data_uri = "data:image/png;base64,AAAA";
    let map = MultimodalDataMap::from([(
        "image_url".to_string(),
        vec![MultimodalData::RawUrl(data_uri.to_string())],
    )]);
    let req = build_generate_request(&request_with_media(map), "req-2", false).unwrap();

    assert_eq!(req.media.len(), 1);
    assert_eq!(
        req.media[0].source,
        Some(pb::media_item::Source::DataUri(data_uri.to_string()))
    );
}

#[test]
fn build_media_rejects_mixed_modalities_without_global_order() {
    let map = MultimodalDataMap::from([
        (
            "image_url".to_string(),
            vec![MultimodalData::RawUrl("http://h/a.png".to_string())],
        ),
        (
            "video_url".to_string(),
            vec![MultimodalData::RawUrl("http://h/b.mp4".to_string())],
        ),
        (
            "audio_url".to_string(),
            vec![MultimodalData::RawUrl("http://h/c.wav".to_string())],
        ),
    ]);
    let error = build_generate_request(&request_with_media(map), "req-3", false).unwrap_err();
    assert!(error.message().contains("mixed media modalities"));
}

#[test]
fn build_media_maps_single_modality_from_map_key() {
    let map = MultimodalDataMap::from([(
        "audio_url".to_string(),
        vec![MultimodalData::RawUrl("http://h/c.wav".to_string())],
    )]);
    let req = build_generate_request(&request_with_media(map), "req-audio", false).unwrap();
    assert_eq!(req.media[0].modality, pb::Modality::Audio as i32);
}

#[test]
fn build_media_is_empty_for_text_only_request() {
    let req = build_generate_request(&request(Some(8)), "req-4", false).unwrap();
    assert!(req.media.is_empty());
}

#[test]
fn build_generate_request_uses_canonical_nested_options() {
    let mut input = request(Some(12));
    input.sampling_options.temperature = Some(0.7);
    input.sampling_options.top_p = Some(0.9);
    input.sampling_options.n = Some(1);
    input.stop_conditions.min_tokens = Some(2);
    input.stop_conditions.ignore_eos = Some(true);
    input.stop_conditions.stop = Some(vec!["done".to_string()]);
    input.output_options.logprobs = Some(3);
    input.output_options.prompt_logprobs = Some(2);
    input.routing = Some(RoutingHints {
        dp_rank: Some(3),
        cache_namespace: Some("tenant-a".to_string()),
        priority: Some(7),
        ..Default::default()
    });

    let req = build_generate_request(&input, "req-canonical", false).unwrap();
    let sampling = req.sampling.expect("sampling");
    assert_eq!(sampling.temperature, Some(0.7_f32 as f64));
    assert_eq!(sampling.top_p, Some(0.9_f32 as f64));
    assert_eq!(sampling.num_sequences, Some(1));

    let stopping = req.stopping.expect("stopping");
    assert_eq!(stopping.max_tokens, Some(12));
    assert_eq!(stopping.min_tokens, Some(2));
    assert_eq!(stopping.ignore_eos, Some(true));
    assert_eq!(stopping.conditions.len(), 1);

    let response = req.response.expect("response");
    assert_eq!(response.return_output_logprobs, Some(true));
    assert_eq!(response.return_prompt_logprobs, Some(true));
    assert!(matches!(
        response
            .output_candidates
            .and_then(|selection| selection.selection),
        Some(pb::candidate_token_selection::Selection::TopN(3))
    ));
    assert!(matches!(
        response
            .prompt_candidates
            .and_then(|selection| selection.selection),
        Some(pb::candidate_token_selection::Selection::TopN(2))
    ));

    let kv = req.kv.expect("kv options");
    assert_eq!(kv.data_parallel_rank, Some(3));
    assert_eq!(kv.cache_salt.as_deref(), Some("tenant-a"));
    assert_eq!(req.priority, Some(7));
}

#[test]
fn build_generate_request_uses_engine_native_worker_envelope() {
    let mut input = request(None);
    input.generate_request = Some(
        serde_json::from_value(serde_json::json!({
            "request_id": "request-native",
            "model": "fake-model",
            "sampling_params": {
                "temperature": 0.25,
                "top_p": 0.9,
                "max_tokens": 17,
                "min_tokens": 2,
                "ignore_eos": true,
                "logprobs": 1,
                "skip_special_tokens": false,
                "skip_reading_prefix_cache": true
            },
            "cache_salt": "tenant-native",
            "priority": -4
        }))
        .expect("valid engine-native worker envelope"),
    );
    input.routing = Some(RoutingHints {
        cache_namespace: Some("tenant-native".to_string()),
        priority: Some(4),
        ..Default::default()
    });

    let req = build_generate_request(&input, "request-native", false).unwrap();
    let sampling = req.sampling.expect("sampling");
    assert_eq!(sampling.temperature, Some(0.25));
    assert_eq!(sampling.top_p, Some(0.9_f32 as f64));

    let stopping = req.stopping.expect("stopping");
    assert_eq!(stopping.max_tokens, Some(17));
    assert_eq!(stopping.min_tokens, Some(2));
    assert_eq!(stopping.ignore_eos, Some(true));

    let response = req.response.expect("response");
    assert_eq!(response.return_output_logprobs, Some(true));
    assert!(matches!(
        response
            .output_candidates
            .and_then(|selection| selection.selection),
        Some(pb::candidate_token_selection::Selection::TopN(1))
    ));

    let kv = req.kv.expect("kv options");
    assert_eq!(kv.cache_salt.as_deref(), Some("tenant-native"));
    assert_eq!(kv.bypass_prefix_cache, Some(true));
    assert_eq!(req.priority, Some(-4));
}

#[test]
fn build_generate_request_rejects_unrepresented_native_sampling_fields() {
    let mut input = request(None);
    input.generate_request = Some(
        serde_json::from_value(serde_json::json!({
            "model": "fake-model",
            "sampling_params": {"logit_bias": {"7": -1.0}}
        }))
        .expect("valid worker envelope"),
    );

    let error = build_generate_request(&input, "request-native", false).unwrap_err();
    assert!(error.message().contains("sampling_params.logit_bias"));
}

#[test]
fn build_generate_request_rejects_native_cache_salt_routing_mismatch() {
    let mut input = request(None);
    input.generate_request = Some(
        serde_json::from_value(serde_json::json!({
            "model": "fake-model",
            "sampling_params": {},
            "cache_salt": "native-salt"
        }))
        .expect("valid worker envelope"),
    );
    input.routing = Some(RoutingHints {
        cache_namespace: Some("routed-salt".to_string()),
        ..Default::default()
    });

    let error = build_generate_request(&input, "request-native", false).unwrap_err();
    assert!(
        error
            .message()
            .contains("does not match routing cache_salt")
    );
}

#[test]
fn build_generate_request_rejects_unrepresentable_output_semantics() {
    let mut input = request(Some(12));
    input.output_options.skip_special_tokens = Some(false);
    assert!(build_generate_request(&input, "req-special", false).is_err());

    input.output_options.skip_special_tokens = Some(true);
    input.output_options.return_tokens_as_token_ids = Some(true);
    assert!(build_generate_request(&input, "req-token-format", false).is_err());
}

#[test]
fn build_generate_request_rejects_only_mixed_stop_visibility() {
    let mut input = request(Some(12));
    input.stop_conditions.stop = Some(vec!["visible".to_string()]);
    input.stop_conditions.stop_token_ids_visible = Some(vec![42]);
    input.sampling_options.include_stop_str_in_output = Some(true);
    let request = build_generate_request(&input, "req-visible-stops", false).unwrap();
    assert_eq!(request.stopping.unwrap().include_stop_in_output, Some(true));

    input.stop_conditions.stop_token_ids_hidden = Some(vec![43]);
    assert!(build_generate_request(&input, "req-mixed-stops", false).is_err());
}

#[test]
fn canonical_stop_match_maps_to_backend_stop_reason() {
    let text = super::engine::stop_match_to_reason(pb::StopMatch {
        r#match: Some(pb::stop_match::Match::StopText("done".to_string())),
    });
    assert_eq!(
        text,
        Some(dynamo_backend_common::StopReason::String(
            "done".to_string()
        ))
    );

    let token = super::engine::stop_match_to_reason(pb::StopMatch {
        r#match: Some(pb::stop_match::Match::StopTokenId(42)),
    });
    assert_eq!(token, Some(dynamo_backend_common::StopReason::Int(42)));
}

#[test]
fn missing_per_token_text_does_not_fabricate_empty_token_metadata() {
    let output = super::engine::token_output_to_engine(pb::TokenOutput {
        output_index: Some(0),
        tokens: vec![pb::TokenInfo {
            token_id: 42,
            token: String::new(),
            ..Default::default()
        }],
        text: "decoded".to_string(),
    });
    assert_eq!(output.token_ids, vec![42]);
    assert_eq!(output.text.as_deref(), Some("decoded"));
    assert!(output.tokens.is_none());
}

#[test]
fn build_media_rejects_decoded_rdma_descriptor() {
    // Fail-closed: the sidecar has no NIXL agent to dereference a pre-decoded
    // descriptor, so a request carrying one must error rather than silently
    // drop the media and degrade to a text request.
    let map = MultimodalDataMap::from([("image_url".to_string(), vec![decoded_media()])]);
    let err = build_generate_request(&request_with_media(map), "req-5", false).unwrap_err();

    assert_eq!(
        err.error_type(),
        ErrorType::Backend(BackendError::InvalidArgument)
    );
    assert!(
        err.message().contains("media_decoder"),
        "error should point operators at URL-passthrough mode, got: {}",
        err.message()
    );
}

#[test]
fn disagg_json_round_trips_through_kv_session() {
    // Typed params survive the prefill -> JSON -> decode round-trip with their
    // JSON types intact (numbers stay numbers, arrays stay arrays).
    let attrs = serde_json::json!({
        "remote_engine_id": "engine-7",
        "remote_port": 20097,
        "tp_size": 1,
        "do_remote_prefill": true,
        "remote_block_ids": [4, 5, 6],
    });
    let original = pb::KvSessionRef {
        session_id: "sess-1".to_string(),
        transfer_backend: "NixlConnector".to_string(),
        endpoints: Vec::new(),
        dp_rank: 3,
        attributes_struct: json_to_prost_struct(&attrs),
    };

    let json = kv_session_to_disagg_json(original.clone());
    let restored = disagg_json_to_kv_session(&json, "fallback-id");

    assert_eq!(restored.session_id, original.session_id);
    assert_eq!(restored.transfer_backend, original.transfer_backend);
    assert_eq!(restored.dp_rank, original.dp_rank);
    let restored_attrs = prost_struct_to_json(
        restored
            .attributes_struct
            .as_ref()
            .expect("attributes_struct round-trips"),
    );
    assert_eq!(restored_attrs, attrs);
}

#[test]
fn disagg_json_falls_back_to_request_id_when_session_id_absent() {
    let restored = disagg_json_to_kv_session(&serde_json::json!({}), "req-42");
    assert_eq!(restored.session_id, "req-42");
    assert!(restored.transfer_backend.is_empty());
    assert_eq!(restored.dp_rank, 0);
}

// ============================================================================
// from_args bootstrap discovery (sync — mirrors real startup)
// ============================================================================

#[test]
fn from_args_discovers_aggregated_role() {
    let handle = spawn_fake_engine(FakeConfig::default());
    let (_engine, config) = VllmSidecarEngine::from_args(Some(vec![
        "dynamo-vllm-sidecar".to_string(),
        "--openengine-endpoint".to_string(),
        handle.endpoint.clone(),
    ]))
    .expect("from_args");

    assert_eq!(config.component, "backend");
    assert_eq!(config.disaggregation_mode, DisaggregationMode::Aggregated);
    assert_eq!(config.model_name, "fake-model");
    assert_eq!(config.served_model_name.as_deref(), Some("fake-served"));
    assert_eq!(config.reasoning_parser.as_deref(), Some("deepseek_r1"));
    assert_eq!(config.tool_call_parser.as_deref(), Some("hermes"));
    assert_eq!(config.endpoint_aliases, ["engine_generate_v1"]);
}

#[test]
fn from_args_discovers_prefill_role_and_component() {
    let handle = spawn_fake_engine(FakeConfig {
        role: pb::EngineRole::Prefill,
        ..FakeConfig::default()
    });
    let (_engine, config) = VllmSidecarEngine::from_args(Some(vec![
        "dynamo-vllm-sidecar".to_string(),
        "--openengine-endpoint".to_string(),
        handle.endpoint.clone(),
    ]))
    .expect("from_args");

    assert_eq!(config.component, "prefill");
    assert_eq!(config.disaggregation_mode, DisaggregationMode::Prefill);
    assert_eq!(config.endpoint_aliases, ["engine_generate_prefill_v1"]);
}
