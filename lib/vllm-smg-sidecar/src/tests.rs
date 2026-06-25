// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit + conformance tests for [`VllmSmgSidecarEngine`].
//!
//! Everything runs against an in-process fake SMG vLLM server bound to an
//! ephemeral loopback port on its own runtime thread.

use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use dynamo_backend_common::{
    BackendError, DisaggregationMode, ErrorType, FinishReason, GenerateContext, LLMEngine,
    LLMEngineOutput, MultimodalDataMap, OutputOptions, PrefillResult, PreprocessedRequest,
    RoutingHints, SamplingOptions, StopConditions,
};
use dynamo_runtime::engine::AsyncEngineContext;
use dynamo_runtime::pipeline::{AsyncEngineContextProvider, Context};
use futures::{Stream, StreamExt};
use tonic::{Request, Response, Status};

use crate::args::TransportConfig;
use crate::engine::{VllmSmgSidecarEngine, build_generate_request, validate_supported_request};
use crate::proto::common as common_pb;
use crate::proto::engine as pb;
use crate::proto::engine::vllm_engine_server::{VllmEngine, VllmEngineServer};

type RespStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send>>;

// ============================================================================
// Fake SMG vLLM server
// ============================================================================

#[derive(Clone)]
struct FakeConfig {
    tokens: u32,
    model_path: String,
    served_model_name: String,
    is_generation: bool,
    healthy: bool,
    kv_role: String,
    data_parallel_size: i32,
}

impl Default for FakeConfig {
    fn default() -> Self {
        Self {
            tokens: 4,
            model_path: "fake-model".to_string(),
            served_model_name: "fake-served".to_string(),
            is_generation: true,
            healthy: true,
            kv_role: String::new(),
            data_parallel_size: 1,
        }
    }
}

struct FakeSmgVllm {
    cfg: FakeConfig,
    abort_count: Arc<AtomicUsize>,
    abort_ids: Arc<Mutex<Vec<String>>>,
    last_generate: Arc<Mutex<Option<pb::GenerateRequest>>>,
}

#[tonic::async_trait]
impl VllmEngine for FakeSmgVllm {
    type GenerateStream = RespStream<pb::GenerateResponse>;
    type GetTokenizerStream = RespStream<common_pb::GetTokenizerChunk>;
    type SubscribeKvEventsStream = RespStream<common_pb::KvEventBatch>;

    async fn generate(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let req = request.into_inner();
        *self.last_generate.lock().unwrap() = Some(req.clone());
        let request_tokens = match req.input.as_ref() {
            Some(pb::generate_request::Input::Tokenized(t)) => t.input_ids.len() as u32,
            Some(pb::generate_request::Input::Text(t)) => t.len() as u32,
            None => 0,
        };
        let tokens = self.cfg.tokens;

        let stream = async_stream::stream! {
            for i in 0..tokens {
                tokio::time::sleep(Duration::from_millis(10)).await;
                yield Ok(pb::GenerateResponse {
                    response: Some(pb::generate_response::Response::Chunk(
                        pb::GenerateStreamChunk {
                            token_ids: vec![1000 + i],
                            prompt_tokens: request_tokens,
                            completion_tokens: 1,
                            cached_tokens: 0,
                            output_logprobs: None,
                            input_logprobs: None,
                            index: 0,
                        },
                    )),
                });
            }
            yield Ok(pb::GenerateResponse {
                response: Some(pb::generate_response::Response::Complete(
                    pb::GenerateComplete {
                        output_ids: Vec::new(),
                        finish_reason: "stop".to_string(),
                        prompt_tokens: request_tokens,
                        completion_tokens: tokens,
                        cached_tokens: 0,
                        output_logprobs: None,
                        input_logprobs: None,
                        index: 0,
                        kv_transfer_params: None,
                        matched_stop: None,
                        kv_transfer_params_json: None,
                    },
                )),
            });
        };
        Ok(Response::new(Box::pin(stream)))
    }

    async fn embed(
        &self,
        _request: Request<pb::EmbedRequest>,
    ) -> Result<Response<pb::EmbedResponse>, Status> {
        Err(Status::unimplemented("fake embed is not implemented"))
    }

    async fn health_check(
        &self,
        _request: Request<pb::HealthCheckRequest>,
    ) -> Result<Response<pb::HealthCheckResponse>, Status> {
        Ok(Response::new(pb::HealthCheckResponse {
            healthy: self.cfg.healthy,
            message: if self.cfg.healthy {
                "ok".to_string()
            } else {
                "not healthy".to_string()
            },
        }))
    }

    async fn abort(
        &self,
        request: Request<pb::AbortRequest>,
    ) -> Result<Response<pb::AbortResponse>, Status> {
        self.abort_count.fetch_add(1, Ordering::SeqCst);
        self.abort_ids
            .lock()
            .unwrap()
            .extend(request.into_inner().request_ids);
        Ok(Response::new(pb::AbortResponse {}))
    }

    async fn get_model_info(
        &self,
        _request: Request<pb::GetModelInfoRequest>,
    ) -> Result<Response<pb::GetModelInfoResponse>, Status> {
        Ok(Response::new(pb::GetModelInfoResponse {
            model_path: self.cfg.model_path.clone(),
            is_generation: self.cfg.is_generation,
            max_context_length: 4096,
            vocab_size: 32000,
            supports_vision: false,
            served_model_name: self.cfg.served_model_name.clone(),
            tokenizer_path: String::new(),
            model_type: "fake".to_string(),
            architectures: vec!["FakeForCausalLM".to_string()],
            eos_token_ids: vec![2],
            pad_token_id: 0,
            bos_token_id: 1,
            max_req_input_len: 4096,
            default_sampling_params_json: String::new(),
        }))
    }

    async fn get_server_info(
        &self,
        _request: Request<pb::GetServerInfoRequest>,
    ) -> Result<Response<pb::GetServerInfoResponse>, Status> {
        Ok(Response::new(pb::GetServerInfoResponse {
            active_requests: 0,
            is_paused: false,
            last_receive_timestamp: 0.0,
            uptime_seconds: 1.0,
            server_type: "vllm-grpc".to_string(),
            kv_connector: String::new(),
            kv_role: self.cfg.kv_role.clone(),
            kv_engine_id: String::new(),
            data_parallel_size: self.cfg.data_parallel_size,
        }))
    }

    async fn get_loads(
        &self,
        _request: Request<pb::GetLoadsRequest>,
    ) -> Result<Response<pb::GetLoadsResponse>, Status> {
        Err(Status::unimplemented("fake loads are not implemented"))
    }

    async fn get_tokenizer(
        &self,
        _request: Request<common_pb::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        Err(Status::unimplemented("fake tokenizer is not implemented"))
    }

    async fn subscribe_kv_events(
        &self,
        _request: Request<common_pb::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        Err(Status::unimplemented("fake KV events are not implemented"))
    }
}

struct FakeHandle {
    endpoint: String,
    abort_count: Arc<AtomicUsize>,
    abort_ids: Arc<Mutex<Vec<String>>>,
    last_generate: Arc<Mutex<Option<pb::GenerateRequest>>>,
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

fn spawn_fake_engine(cfg: FakeConfig) -> FakeHandle {
    let abort_count = Arc::new(AtomicUsize::new(0));
    let abort_ids = Arc::new(Mutex::new(Vec::new()));
    let last_generate = Arc::new(Mutex::new(None));
    let (tx, rx) = std::sync::mpsc::channel();
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();

    let svc_abort_count = abort_count.clone();
    let svc_abort_ids = abort_ids.clone();
    let svc_generate = last_generate.clone();
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

            let svc = FakeSmgVllm {
                cfg,
                abort_count: svc_abort_count,
                abort_ids: svc_abort_ids,
                last_generate: svc_generate,
            };
            tonic::transport::Server::builder()
                .add_service(VllmEngineServer::new(svc))
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
        abort_ids,
        last_generate,
        shutdown_tx: Some(shutdown_tx),
    }
}

// ============================================================================
// Test helpers
// ============================================================================

fn test_transport() -> TransportConfig {
    TransportConfig {
        connect_timeout: Duration::from_secs(5),
        poll_interval: Duration::from_millis(50),
        deadline: Duration::from_secs(10),
        connections: 2,
    }
}

fn engine_for(handle: &FakeHandle) -> VllmSmgSidecarEngine {
    VllmSmgSidecarEngine::new(handle.endpoint.clone(), test_transport())
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
        .output_options(OutputOptions::default())
        .build()
        .expect("build request")
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

fn assert_invalid_arg(err: dynamo_backend_common::DynamoError) {
    assert_eq!(
        err.error_type(),
        ErrorType::Backend(BackendError::InvalidArgument)
    );
}

// ============================================================================
// Conformance + lifecycle
// ============================================================================

#[tokio::test]
async fn smg_sidecar_passes_conformance() {
    let handle = spawn_fake_engine(FakeConfig::default());
    dynamo_backend_common::testing::run_conformance(|| engine_for(&handle))
        .await
        .expect("SMG sidecar must satisfy conformance");
}

#[tokio::test]
async fn from_args_builds_aggregated_worker_config() {
    let handle = spawn_fake_engine(FakeConfig::default());
    let endpoint = handle.endpoint.clone();
    let cfg = std::thread::spawn(move || {
        let (_engine, cfg) = VllmSmgSidecarEngine::from_args(Some(vec![
            "dynamo-vllm-smg-sidecar".to_string(),
            "--smg-endpoint".to_string(),
            endpoint,
            "--endpoint-types".to_string(),
            "chat".to_string(),
        ]))
        .expect("from_args");
        cfg
    })
    .join()
    .expect("from_args thread");

    assert_eq!(cfg.component, "backend");
    assert_eq!(cfg.disaggregation_mode, DisaggregationMode::Aggregated);
    assert_eq!(cfg.model_name, "fake-model");
    assert_eq!(cfg.served_model_name.as_deref(), Some("fake-served"));
    assert_eq!(cfg.endpoint_types, "chat");
}

#[tokio::test]
async fn start_advertises_smg_metadata() {
    let handle = spawn_fake_engine(FakeConfig {
        data_parallel_size: 3,
        ..FakeConfig::default()
    });
    let engine = engine_for(&handle);

    let cfg = engine.start(0).await.expect("start");
    assert_eq!(cfg.model, "fake-model");
    assert_eq!(cfg.served_model_name.as_deref(), Some("fake-served"));
    assert_eq!(cfg.context_length, Some(4096));
    assert_eq!(cfg.data_parallel_size, Some(3));
    assert!(cfg.kv_cache_block_size.is_none());
    assert!(cfg.total_kv_blocks.is_none());
    assert!(cfg.bootstrap_host.is_none());

    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn start_rejects_disaggregated_kv_role() {
    let handle = spawn_fake_engine(FakeConfig {
        kv_role: "kv_producer".to_string(),
        ..FakeConfig::default()
    });
    let engine = engine_for(&handle);

    let err = engine
        .start(0)
        .await
        .expect_err("disaggregated role must fail");
    assert_invalid_arg(err);
}

#[tokio::test]
async fn watch_reports_engine_shutdown_when_health_endpoint_disappears() {
    let mut handle = spawn_fake_engine(FakeConfig::default());
    let engine = engine_for(&handle);
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

// ============================================================================
// Generate happy path + cancellation
// ============================================================================

#[tokio::test]
async fn generate_maps_request_and_streams_tokens_then_stop_terminal() {
    let handle = spawn_fake_engine(FakeConfig {
        tokens: 4,
        ..FakeConfig::default()
    });
    let engine = engine_for(&handle);
    engine.start(0).await.unwrap();

    let mut req = request(Some(64));
    req.stop_conditions.min_tokens = Some(2);
    req.stop_conditions.stop = Some(vec!["done".to_string()]);
    req.stop_conditions.stop_token_ids = Some(vec![7]);
    req.stop_conditions.ignore_eos = Some(true);
    req.sampling_options.temperature = Some(0.7);
    req.sampling_options.top_p = Some(0.9);
    req.sampling_options.top_k = Some(50);
    req.sampling_options.min_p = Some(0.01);
    req.sampling_options.frequency_penalty = Some(0.2);
    req.sampling_options.presence_penalty = Some(0.3);
    req.sampling_options.repetition_penalty = Some(1.05);
    req.sampling_options.seed = Some(123);
    req.sampling_options.include_stop_str_in_output = Some(true);
    req.output_options.skip_special_tokens = Some(false);

    let stream = engine
        .generate(req, gen_ctx(fresh_ctx()))
        .await
        .expect("stream");
    let chunks = collect_ok(stream).await;

    let token_chunks = &chunks[..chunks.len() - 1];
    let total: usize = token_chunks.iter().map(|c| c.token_ids.len()).sum();
    assert_eq!(total, 4);
    assert!(token_chunks.iter().all(|c| c.finish_reason.is_none()));

    let terminal = chunks.last().unwrap();
    assert!(matches!(terminal.finish_reason, Some(FinishReason::Stop)));
    let usage = terminal.completion_usage.as_ref().unwrap();
    assert_eq!(usage.prompt_tokens, 3);
    assert_eq!(usage.completion_tokens, 4);
    assert_eq!(usage.total_tokens, 7);

    let sent = handle
        .last_generate
        .lock()
        .unwrap()
        .clone()
        .expect("captured generate request");
    assert!(sent.stream);
    match sent.input.unwrap() {
        pb::generate_request::Input::Tokenized(t) => assert_eq!(t.input_ids, vec![1, 2, 3]),
        pb::generate_request::Input::Text(_) => panic!("expected tokenized input"),
    }
    let sampling = sent.sampling_params.expect("sampling params");
    assert_eq!(sampling.temperature, Some(0.7));
    assert_eq!(sampling.top_p, 0.9);
    assert_eq!(sampling.top_k, 50);
    assert_eq!(sampling.min_p, 0.01);
    assert_eq!(sampling.frequency_penalty, 0.2);
    assert_eq!(sampling.presence_penalty, 0.3);
    assert_eq!(sampling.repetition_penalty, 1.05);
    assert_eq!(sampling.max_tokens, Some(64));
    assert_eq!(sampling.min_tokens, 2);
    assert_eq!(sampling.stop, vec!["done".to_string()]);
    assert_eq!(sampling.stop_token_ids, vec![7]);
    assert!(sampling.ignore_eos);
    assert_eq!(sampling.seed, Some(123));
    assert!(sampling.include_stop_str_in_output);
    assert!(!sampling.skip_special_tokens);

    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn build_generate_request_maps_guided_decoding_regex() {
    let mut req = request(Some(5));
    req.sampling_options.guided_decoding =
        Some(dynamo_llm::protocols::common::GuidedDecodingOptions::new(
            None,
            Some("[a-z]+".to_string()),
            None,
            None,
            None,
            None,
            None,
        ));

    let sent = build_generate_request(&req, "req-1").expect("build");
    let sampling = sent.sampling_params.expect("sampling");
    assert_eq!(
        sampling.constraint,
        Some(pb::sampling_params::Constraint::Regex("[a-z]+".to_string()))
    );
}

#[tokio::test]
async fn generate_observes_midstream_cancellation() {
    let handle = spawn_fake_engine(FakeConfig {
        tokens: 50,
        ..FakeConfig::default()
    });
    let engine = engine_for(&handle);
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
async fn abort_sends_request_id_to_smg() {
    let handle = spawn_fake_engine(FakeConfig::default());
    let engine = engine_for(&handle);
    engine.start(0).await.unwrap();

    let ctx = fresh_ctx();
    let request_id = ctx.id().to_string();
    engine.abort(ctx).await;

    assert_eq!(handle.abort_count.load(Ordering::SeqCst), 1);
    assert_eq!(*handle.abort_ids.lock().unwrap(), vec![request_id]);
    engine.cleanup().await.unwrap();
}

// ============================================================================
// Unsupported feature gates
// ============================================================================

#[test]
fn unsupported_features_fail_explicitly() {
    let mut req = request(Some(4));
    req.prompt_embeds = Some("abc".to_string());
    assert_invalid_arg(validate_supported_request(&req).unwrap_err());

    let mut req = request(Some(4));
    req.multi_modal_data = Some(MultimodalDataMap::default());
    assert_invalid_arg(validate_supported_request(&req).unwrap_err());

    let mut req = request(Some(4));
    req.prefill_result = Some(PrefillResult {
        disaggregated_params: serde_json::json!({"remote_host": "127.0.0.1"}),
        prompt_tokens_details: None,
    });
    assert_invalid_arg(validate_supported_request(&req).unwrap_err());

    let mut req = request(Some(4));
    req.output_options.logprobs = Some(1);
    assert_invalid_arg(validate_supported_request(&req).unwrap_err());

    let mut req = request(Some(4));
    req.sampling_options.n = Some(2);
    assert_invalid_arg(validate_supported_request(&req).unwrap_err());

    let mut req = request(Some(4));
    req.sampling_options.best_of = Some(2);
    assert_invalid_arg(validate_supported_request(&req).unwrap_err());

    let mut req = request(Some(4));
    req.sampling_options.use_beam_search = Some(true);
    assert_invalid_arg(validate_supported_request(&req).unwrap_err());

    let mut req = request(Some(4));
    req.routing = Some(RoutingHints {
        dp_rank: Some(0),
        ..Default::default()
    });
    assert_invalid_arg(validate_supported_request(&req).unwrap_err());
}
