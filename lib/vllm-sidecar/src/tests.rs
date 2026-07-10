// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit + conformance tests for [`VllmSidecarEngine`].
//!
//! Everything runs against an in-process fake vLLM gRPC server — a tonic
//! service bound to an ephemeral loopback port on its own runtime thread — so
//! no real vLLM engine or GPU is involved. The fake is configurable by role so
//! the same harness drives the aggregated, prefill, and decode paths.

mod bootstrap;
mod conversion;
mod disaggregation;
mod generation;
mod lifecycle;

use std::collections::BTreeMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use dynamo_backend_common::{
    BackendError, DisaggregationMode, ErrorType, FinishReason, GenerateContext, LLMEngine,
    LLMEngineOutput, LoraAdapter, PrefillResult, PreprocessedRequest, RoutingHints,
    SamplingOptions, StopConditions,
};
use dynamo_runtime::engine::AsyncEngineContext;
use dynamo_runtime::pipeline::{AsyncEngineContextProvider, Context};
use futures::{Stream, StreamExt};
use tonic::{Request, Response, Status};

use crate::args::TransportConfig;
use crate::engine::VllmSidecarEngine;
use crate::proto as pb;
use crate::proto::generate_server::{Generate, GenerateServer};
use crate::request::build_generate_request;
use crate::wire::{json_to_prost_struct, prost_struct_to_json};

// ============================================================================
// Fake vLLM gRPC server
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
    supports_lora: bool,
    max_loras: u32,
    mismatch_lora_load_reply: bool,
    hang_list_loras: bool,
    reasoning_parser: String,
    tool_call_parser: String,
}

impl Default for FakeConfig {
    fn default() -> Self {
        Self {
            role: pb::EngineRole::Aggregated,
            tokens: 4,
            model_id: "fake-model".to_string(),
            served_model_name: "fake-served".to_string(),
            supports_lora: false,
            max_loras: 0,
            mismatch_lora_load_reply: false,
            hang_list_loras: false,
            reasoning_parser: String::new(),
            tool_call_parser: String::new(),
        }
    }
}

struct FakeEngine {
    cfg: FakeConfig,
    /// Incremented once per `Abort` RPC — lets tests assert abort was sent.
    abort_count: Arc<AtomicUsize>,
    /// Captures the KV transfer parameters on the most recent request.
    last_kv_transfer_params: Arc<Mutex<Option<prost_types::Struct>>>,
    /// Captures the `data_parallel_rank` of the most recent `Generate` request
    /// so the KV-routing test can assert the router's forced rank is forwarded.
    last_dp_rank: Arc<Mutex<Option<u32>>>,
    last_lora_name: Arc<Mutex<Option<String>>>,
    adapters: Arc<Mutex<BTreeMap<String, pb::LoraAdapter>>>,
}

#[tonic::async_trait]
impl Generate for FakeEngine {
    type GenerateStreamStream = RespStream<pb::GenerateResponse>;

    async fn generate(
        &self,
        _request: Request<pb::GenerateRequest>,
    ) -> Result<Response<pb::GenerateResponse>, Status> {
        Err(Status::unimplemented(
            "unary generation is not used by these tests",
        ))
    }

    async fn generate_stream(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStreamStream>, Status> {
        let req = request.into_inner();
        *self.last_kv_transfer_params.lock().unwrap() =
            req.kv.as_ref().and_then(|kv| kv.kv_transfer_params.clone());
        *self.last_dp_rank.lock().unwrap() = req.data_parallel_rank;
        *self.last_lora_name.lock().unwrap() = Some(req.lora_name.clone());
        let cfg = self.cfg.clone();

        let stream = async_stream::stream! {
            match cfg.role {
                pb::EngineRole::Prefill => {
                    let kv_transfer_params = json_to_prost_struct(&serde_json::json!({
                            "remote_engine_id": "engine-fake",
                            "remote_block_ids": [1, 2, 3],
                        }));
                    yield Ok(pb::GenerateResponse {
                        prompt_info: Some(pb::PromptInfo {
                            num_prompt_tokens: 3,
                            ..Default::default()
                        }),
                        outputs: Some(pb::SequenceOutput {
                            finish_info: Some(pb::FinishInfo {
                                num_output_tokens: 1,
                                finish_reason: pb::finish_info::FinishReason::Stop as i32,
                                kv_transfer_params,
                                ..Default::default()
                            }),
                            ..Default::default()
                        }),
                    });
                }
                _ => {
                    yield Ok(pb::GenerateResponse {
                        prompt_info: Some(pb::PromptInfo {
                            num_prompt_tokens: 3,
                            ..Default::default()
                        }),
                        outputs: None,
                    });
                    for i in 0..cfg.tokens {
                        tokio::time::sleep(Duration::from_millis(10)).await;
                        yield Ok(pb::GenerateResponse {
                            prompt_info: None,
                            outputs: Some(pb::SequenceOutput {
                                token_ids: vec![1000 + i],
                                text: format!("t{i}"),
                                num_tokens: 1,
                                ..Default::default()
                            }),
                        });
                    }
                    yield Ok(pb::GenerateResponse {
                        prompt_info: None,
                        outputs: Some(pb::SequenceOutput {
                            finish_info: Some(pb::FinishInfo {
                                num_output_tokens: cfg.tokens,
                                finish_reason: pb::finish_info::FinishReason::Stop as i32,
                                ..Default::default()
                            }),
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
            api_version: "vllm".to_string(),
            role: self.cfg.role as i32,
            instance_id: "fake-instance".to_string(),
            supported_models: vec![self.cfg.model_id.clone()],
            parallelism: Some(pb::ParallelismInfo {
                tensor_parallel_size: 1,
                pipeline_parallel_size: 1,
                data_parallel_size: 1,
                data_parallel_rank: 0,
                data_parallel_start_rank: 0,
            }),
            kv_connector: None,
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
            max_context_length: 4096,
            max_output_tokens: 2048,
            kv_block_size: 16,
            total_kv_blocks: 1000,
            max_running_requests: 256,
            max_batched_tokens: 8192,
            tokenizer_modes: Vec::new(),
            supports_text_input: true,
            supports_token_ids_input: true,
            supports_lora: self.cfg.supports_lora,
            max_loras: self.cfg.max_loras,
            supports_multimodal: false,
            reasoning_parser: self.cfg.reasoning_parser.clone(),
            tool_call_parser: self.cfg.tool_call_parser.clone(),
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
    ) -> Result<Response<pb::DrainResponse>, Status> {
        Ok(Response::new(pb::DrainResponse {
            state: pb::DrainState::Complete as i32,
            ..Default::default()
        }))
    }

    async fn get_kv_connector_info(
        &self,
        _request: Request<pb::GetKvConnectorInfoRequest>,
    ) -> Result<Response<pb::KvConnectorInfo>, Status> {
        Ok(Response::new(pb::KvConnectorInfo {
            schema_version: 1,
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
        let response_adapter = if self.cfg.mismatch_lora_load_reply {
            pb::LoraAdapter {
                lora_name: "different-adapter".to_string(),
                ..adapter
            }
        } else {
            adapter
        };
        Ok(Response::new(pb::LoadLoraResponse {
            adapter: Some(response_adapter),
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
        if self.cfg.hang_list_loras {
            return std::future::pending().await;
        }
        Ok(Response::new(pb::ListLorasResponse {
            adapters: self.adapters.lock().unwrap().values().cloned().collect(),
        }))
    }
}

struct FakeHandle {
    endpoint: String,
    abort_count: Arc<AtomicUsize>,
    last_kv_transfer_params: Arc<Mutex<Option<prost_types::Struct>>>,
    last_dp_rank: Arc<Mutex<Option<u32>>>,
    last_lora_name: Arc<Mutex<Option<String>>>,
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

/// Bind a fake vLLM gRPC server to an ephemeral port on a dedicated runtime
/// thread and return its `http://host:port` endpoint once it is listening.
fn spawn_fake_engine(cfg: FakeConfig) -> FakeHandle {
    let abort_count = Arc::new(AtomicUsize::new(0));
    let last_kv_transfer_params = Arc::new(Mutex::new(None));
    let last_dp_rank = Arc::new(Mutex::new(None));
    let last_lora_name = Arc::new(Mutex::new(None));
    let adapters = Arc::new(Mutex::new(BTreeMap::new()));
    let (tx, rx) = std::sync::mpsc::channel();
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();

    let svc_abort = abort_count.clone();
    let svc_kv = last_kv_transfer_params.clone();
    let svc_dp = last_dp_rank.clone();
    let svc_lora_name = last_lora_name.clone();
    let svc_adapters = adapters.clone();
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

            let svc = FakeEngine {
                cfg,
                abort_count: svc_abort,
                last_kv_transfer_params: svc_kv,
                last_dp_rank: svc_dp,
                last_lora_name: svc_lora_name,
                adapters: svc_adapters,
            };
            tonic::transport::Server::builder()
                .add_service(GenerateServer::new(svc))
                .serve_with_incoming_shutdown(
                    tokio_stream::wrappers::TcpListenerStream::new(listener),
                    async move {
                        let _ = shutdown_rx.await;
                    },
                )
                .await
                .expect("serve fake engine");
        });
    });

    let addr = rx.recv().expect("fake engine never bound");
    FakeHandle {
        endpoint: format!("http://{addr}"),
        abort_count,
        last_kv_transfer_params,
        last_dp_rank,
        last_lora_name,
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
    VllmSidecarEngine::new(handle.endpoint.clone(), test_transport(), mode)
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
