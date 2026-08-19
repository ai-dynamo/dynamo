// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use std::net::SocketAddr;
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};

use dynamo_backend_common::engine::RoutingHints;
use dynamo_backend_common::{
    DisaggregationMode, FinishReason, GenerateContext, LLMEngine, OutputOptions, PrefillResult,
    PreprocessedRequest, SamplingOptions, StopConditions,
};
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_runtime::discovery::{DiscoveryInstance, DiscoveryQuery, DiscoverySpec};
use dynamo_runtime::distributed::DistributedConfig;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::{DistributedRuntime, Runtime};
use dynamo_sidecar_common::{GrpcEndpoint, GrpcTransportConfig};
use futures::{Stream, StreamExt};
use serde_json::json;
use tokio::net::TcpListener;
use tokio::sync::{Mutex, Notify, oneshot};
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{Request, Response, Status};
use tonic_health::ServingStatus as HealthServingStatus;

use crate::client::{CONTROL_SERVICE, INFERENCE_SERVICE, VllmClient};
use crate::convert::{ResponseState, build_generate_request};
use crate::engine::VllmSidecarEngine;
use crate::json::{json_to_struct, struct_to_json};
use crate::model::DiscoveredModel;
use crate::proto as pb;

#[derive(Clone, Default)]
struct FakeVllm {
    requests: Arc<Mutex<Vec<pb::GenerateRequest>>>,
    data_parallel_rank_metadata: Arc<Mutex<Vec<Option<String>>>>,
    loras: Arc<Mutex<Vec<pb::LoraAdapter>>>,
    next_lora_id: Arc<AtomicI64>,
    peers: Arc<Mutex<Vec<SocketAddr>>>,
    model_info_override: Arc<Mutex<Option<pb::ModelInfo>>>,
    reject: Arc<AtomicBool>,
    hang: Arc<AtomicBool>,
    hang_before_headers: Arc<AtomicBool>,
    headers_pending: Arc<AtomicBool>,
    release_headers: Arc<Notify>,
    hold_before_first_token: Arc<AtomicBool>,
    close_before_first_token: Arc<AtomicBool>,
    first_token_pending: Arc<AtomicBool>,
    release_first_token: Arc<Notify>,
    server_stream_dropped: Arc<AtomicBool>,
    load_commit_error: Arc<AtomicBool>,
    unload_commit_error: Arc<AtomicBool>,
}

struct DropSignal(Arc<AtomicBool>);

impl Drop for DropSignal {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

#[tonic::async_trait]
impl pb::inference_server::Inference for FakeVllm {
    type GenerateStreamStream =
        Pin<Box<dyn Stream<Item = Result<pb::GenerateResponse, Status>> + Send>>;

    async fn generate(
        &self,
        _request: Request<pb::GenerateRequest>,
    ) -> Result<Response<pb::GenerateResponse>, Status> {
        Err(Status::unimplemented("unary generation is not used"))
    }

    async fn generate_stream(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStreamStream>, Status> {
        if let Some(peer) = request.remote_addr() {
            self.peers.lock().await.push(peer);
        }
        let data_parallel_rank = request
            .metadata()
            .get("x-data-parallel-rank")
            .map(|value| value.to_str().map(str::to_owned))
            .transpose()
            .map_err(|error| Status::invalid_argument(error.to_string()))?;
        self.data_parallel_rank_metadata
            .lock()
            .await
            .push(data_parallel_rank);
        let request = request.into_inner();
        self.requests.lock().await.push(request.clone());
        if self.hang_before_headers.load(Ordering::SeqCst) {
            self.headers_pending.store(true, Ordering::SeqCst);
            self.release_headers.notified().await;
            self.headers_pending.store(false, Ordering::SeqCst);
        }
        if self.reject.load(Ordering::SeqCst) {
            return Err(Status::invalid_argument("rejected by fake vLLM"));
        }

        let prompt_tokens = match request.prompt.as_ref() {
            Some(pb::generate_request::Prompt::TokenIds(ids)) => ids.ids.len() as u32,
            Some(pb::generate_request::Prompt::Text(text)) => {
                text.split_whitespace().count() as u32
            }
            None => return Err(Status::invalid_argument("prompt required")),
        };
        let wants_logprobs = request
            .response
            .as_ref()
            .is_some_and(|response| response.output_logprobs);
        let wants_prompt_logprobs = request
            .response
            .as_ref()
            .is_some_and(|response| response.prompt_logprobs);
        let request_kv = request
            .kv
            .as_ref()
            .and_then(|kv| kv.kv_transfer_params.clone())
            .map(struct_to_json)
            .transpose()
            .map_err(|error| Status::invalid_argument(error.to_string()))?;
        let is_prefill = request_kv
            .as_ref()
            .and_then(|kv| kv.get("do_remote_decode"))
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
            && request_kv
                .as_ref()
                .and_then(|kv| kv.get("remote_engine_id"))
                .is_none();
        let handoff = json!({
            "do_remote_decode": false,
            "do_remote_prefill": true,
            "remote_engine_id": "prefill-0",
            "remote_host": "127.0.0.1",
            "remote_port": 20097,
            "remote_block_ids": [7, 8],
            "nested": {"flags": [true, null, "opaque"]},
        });
        let hang = self.hang.load(Ordering::SeqCst);
        let hold_before_first_token = self.hold_before_first_token.load(Ordering::SeqCst);
        let close_before_first_token = self.close_before_first_token.load(Ordering::SeqCst);
        let first_token_pending = self.first_token_pending.clone();
        let release_first_token = self.release_first_token.clone();
        let dropped = self.server_stream_dropped.clone();

        let stream = async_stream::try_stream! {
            let _drop_signal = DropSignal(dropped);
            let prompt_info = if wants_prompt_logprobs {
                pb::PromptInfo {
                    num_prompt_tokens: prompt_tokens,
                    token_ids: vec![11, 22, 33],
                    logprobs: vec![0.0, -0.2, -0.3],
                    ranks: vec![0, 1, 2],
                    candidate_tokens: vec![
                        pb::CandidateTokenInfo { tokens: vec![] },
                        pb::CandidateTokenInfo { tokens: vec![] },
                        pb::CandidateTokenInfo { tokens: vec![] },
                    ],
                }
            } else {
                pb::PromptInfo {
                    num_prompt_tokens: prompt_tokens,
                    ..Default::default()
                }
            };
            yield pb::GenerateResponse {
                prompt_info: Some(prompt_info),
                outputs: None,
            };

            if hold_before_first_token {
                first_token_pending.store(true, Ordering::SeqCst);
                release_first_token.notified().await;
                first_token_pending.store(false, Ordering::SeqCst);
            }
            if close_before_first_token {
                return;
            }

            if hang {
                loop {
                    yield sequence_response(false, wants_logprobs, None);
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                }
            } else {
                let kv = is_prefill.then(|| {
                    json_to_struct(handoff.clone()).expect("encode handoff")
                });
                yield sequence_response(true, wants_logprobs, kv);
            }
        };
        Ok(Response::new(Box::pin(stream)))
    }
}

#[tonic::async_trait]
impl pb::control_server::Control for FakeVllm {
    async fn get_server_info(
        &self,
        _request: Request<pb::GetServerInfoRequest>,
    ) -> Result<Response<pb::ServerInfo>, Status> {
        Ok(Response::new(server_info()))
    }

    async fn get_model_info(
        &self,
        _request: Request<pb::GetModelInfoRequest>,
    ) -> Result<Response<pb::ModelInfo>, Status> {
        let model = self
            .model_info_override
            .lock()
            .await
            .clone()
            .unwrap_or_else(model_info);
        Ok(Response::new(model))
    }

    async fn abort(
        &self,
        _request: Request<pb::AbortRequest>,
    ) -> Result<Response<pb::AbortResponse>, Status> {
        Ok(Response::new(pb::AbortResponse {}))
    }

    async fn load_lora(
        &self,
        request: Request<pb::LoadLoraRequest>,
    ) -> Result<Response<pb::LoadLoraResponse>, Status> {
        let request = request.into_inner();
        let mut loras = self.loras.lock().await;
        if let Some(existing) = loras
            .iter()
            .find(|loaded| loaded.lora_name == request.lora_name)
        {
            return Err(Status::already_exists(format!(
                "adapter `{}` is already loaded with id {}",
                existing.lora_name, existing.lora_id
            )));
        }
        let adapter = pb::LoraAdapter {
            lora_id: self.next_lora_id.fetch_add(1, Ordering::SeqCst) + 1,
            lora_name: request.lora_name,
            source_path: request.source_path,
        };
        loras.push(adapter.clone());
        if self.load_commit_error.swap(false, Ordering::SeqCst) {
            return Err(Status::unavailable("injected error after load commit"));
        }
        Ok(Response::new(pb::LoadLoraResponse {
            adapter: Some(adapter),
        }))
    }

    async fn unload_lora(
        &self,
        request: Request<pb::UnloadLoraRequest>,
    ) -> Result<Response<pb::UnloadLoraResponse>, Status> {
        let name = request.into_inner().lora_name;
        let mut loras = self.loras.lock().await;
        let index = loras
            .iter()
            .position(|adapter| adapter.lora_name == name)
            .ok_or_else(|| Status::not_found("adapter not found"))?;
        let adapter = loras.remove(index);
        if self.unload_commit_error.swap(false, Ordering::SeqCst) {
            return Err(Status::unavailable("injected error after unload commit"));
        }
        Ok(Response::new(pb::UnloadLoraResponse {
            adapter: Some(adapter),
        }))
    }

    async fn list_loras(
        &self,
        _request: Request<pb::ListLorasRequest>,
    ) -> Result<Response<pb::ListLorasResponse>, Status> {
        Ok(Response::new(pb::ListLorasResponse {
            adapters: self.loras.lock().await.clone(),
        }))
    }

    async fn get_kv_event_sources(
        &self,
        _request: Request<pb::GetKvEventSourcesRequest>,
    ) -> Result<Response<pb::GetKvEventSourcesResponse>, Status> {
        Ok(Response::new(pb::GetKvEventSourcesResponse {
            sources: (0..2)
                .map(|rank| pb::KvEventSource {
                    transport: "zmq".to_string(),
                    endpoint: format!("tcp://*:{}", 20081 + rank),
                    topic: String::new(),
                    replay_endpoint: String::new(),
                    data_parallel_rank: Some(rank),
                    encoding: "msgpack".to_string(),
                    schema_version: 1,
                    buffer_steps: 0,
                    hwm: 0,
                    max_queue_size: 0,
                })
                .collect(),
        }))
    }
}

fn model_info() -> pb::ModelInfo {
    pb::ModelInfo {
        model_id: "model-source".to_string(),
        served_model_name: "served-model".to_string(),
        served_model_aliases: vec!["model-alias".to_string()],
        supports_text_input: true,
        supports_token_ids_input: true,
        supports_lora: true,
        supports_multimodal: false,
        reasoning_parser: "deepseek_r1".to_string(),
        tool_call_parser: "hermes".to_string(),
    }
}

fn server_info() -> pb::ServerInfo {
    pb::ServerInfo {
        engine_version: "test-vllm".to_string(),
        api_version: "vllm".to_string(),
        instance_id: "test-instance".to_string(),
        parallelism: Some(pb::ParallelismInfo {
            tensor_parallel_size: 2,
            pipeline_parallel_size: 1,
            data_parallel_size: 2,
            data_parallel_rank: 0,
            decode_context_parallel_size: 1,
        }),
        max_model_len: 8192,
        kv_block_size: 16,
        total_kv_blocks: 4096,
        max_running_requests: 128,
        max_batched_tokens: 2048,
        max_loras: 4,
    }
}

fn sequence_response(
    terminal: bool,
    logprobs: bool,
    kv_transfer_params: Option<prost_types::Struct>,
) -> pb::GenerateResponse {
    pb::GenerateResponse {
        prompt_info: None,
        outputs: Some(pb::SequenceOutput {
            index: 0,
            text: " token".to_string(),
            num_tokens: 1,
            token_ids: vec![42],
            logprobs: logprobs.then_some(vec![-0.25]).unwrap_or_default(),
            ranks: logprobs.then_some(vec![1]).unwrap_or_default(),
            candidate_tokens: logprobs
                .then_some(vec![pb::CandidateTokenInfo {
                    tokens: vec![pb::candidate_token_info::TokenInfo {
                        id: 43,
                        logprob: -0.5,
                        rank: 2,
                    }],
                }])
                .unwrap_or_default(),
            finish_info: terminal.then_some(pb::FinishInfo {
                num_output_tokens: 1,
                finish_reason: pb::finish_info::FinishReason::Stop as i32,
                stop_reason: Some(pb::finish_info::StopReason::StopTokenId(2)),
                kv_transfer_params,
                ec_transfer_params: None,
            }),
        }),
    }
}

#[test]
fn prompt_logprobs_are_retained_for_the_terminal_chunk() {
    let request = request();
    let mut state = ResponseState::new(&request, DisaggregationMode::Aggregated);
    let mut first_response = sequence_response(false, true, None);
    first_response.prompt_info = Some(pb::PromptInfo {
        num_prompt_tokens: 3,
        token_ids: vec![11, 22, 33],
        logprobs: vec![0.0, -0.2, -0.3],
        ranks: vec![0, 1, 2],
        candidate_tokens: vec![pb::CandidateTokenInfo::default(); 3],
    });

    let first = state
        .convert(first_response)
        .expect("convert first chunk")
        .expect("first chunk");
    assert!(first.finish_reason.is_none());
    assert!(first.engine_data.is_none());

    let mut terminal_response = sequence_response(true, true, None);
    terminal_response
        .outputs
        .as_mut()
        .unwrap()
        .finish_info
        .as_mut()
        .unwrap()
        .num_output_tokens = 2;
    let terminal = state
        .convert(terminal_response)
        .expect("convert terminal chunk")
        .expect("terminal chunk");
    assert!(terminal.finish_reason.is_some());
    assert!(terminal.engine_data.as_ref().unwrap()["prompt_logprobs"].is_array());
}

#[test]
fn negative_infinity_logprobs_are_normalized() {
    let request = request();
    let mut state = ResponseState::new(&request, DisaggregationMode::Aggregated);
    let mut response = sequence_response(true, true, None);
    response.prompt_info = Some(pb::PromptInfo {
        num_prompt_tokens: 3,
        token_ids: vec![11, 22, 33],
        logprobs: vec![0.0, f32::NEG_INFINITY, -0.3],
        ranks: vec![0, 1, 2],
        candidate_tokens: vec![
            pb::CandidateTokenInfo::default(),
            pb::CandidateTokenInfo {
                tokens: vec![pb::candidate_token_info::TokenInfo {
                    id: 23,
                    logprob: f32::NEG_INFINITY,
                    rank: 2,
                }],
            },
            pb::CandidateTokenInfo::default(),
        ],
    });
    let output = response.outputs.as_mut().unwrap();
    output.logprobs[0] = f32::NEG_INFINITY;
    output.candidate_tokens[0].tokens[0].logprob = f32::NEG_INFINITY;

    let mapped = state
        .convert(response)
        .expect("convert response")
        .expect("terminal output");
    assert_eq!(mapped.log_probs.as_deref(), Some(&[-9999.0][..]));
    assert!(
        mapped.top_logprobs.as_ref().unwrap()[0]
            .iter()
            .all(|entry| entry.logprob == -9999.0)
    );
    let prompt = &mapped.engine_data.as_ref().unwrap()["prompt_logprobs"][1];
    assert_eq!(prompt["22"]["logprob"], json!(-9999.0));
    assert_eq!(prompt["23"]["logprob"], json!(-9999.0));
}

#[test]
fn zero_output_logprobs_omits_top_logprobs() {
    let mut request = request();
    request.output_options.logprobs = Some(0);
    let mut state = ResponseState::new(&request, DisaggregationMode::Aggregated);
    let mapped = state
        .convert(sequence_response(true, true, None))
        .expect("convert response")
        .expect("terminal output");

    assert_eq!(mapped.log_probs.as_deref(), Some(&[-0.25][..]));
    assert!(mapped.top_logprobs.is_none());
}

#[test]
fn oversized_logprob_counts_are_rejected() {
    let oversized = i32::MAX as u32 + 1;

    let mut output_request = request();
    output_request.output_options.logprobs = Some(oversized);
    let output_error = build_generate_request(
        output_request,
        "output-logprobs".to_string(),
        DisaggregationMode::Aggregated,
    )
    .expect_err("oversized output logprobs must fail");
    assert!(output_error.to_string().contains("must fit in i32"));

    let mut prompt_request = request();
    prompt_request.output_options.prompt_logprobs = Some(oversized);
    let prompt_error = build_generate_request(
        prompt_request,
        "prompt-logprobs".to_string(),
        DisaggregationMode::Aggregated,
    )
    .expect_err("oversized prompt logprobs must fail");
    assert!(prompt_error.to_string().contains("must fit in i32"));
}

struct FakeServer {
    endpoint: String,
    service: FakeVllm,
    shutdown: Option<oneshot::Sender<()>>,
}

impl FakeServer {
    async fn start(service: FakeVllm) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let address = listener.local_addr().expect("address");
        let (shutdown, shutdown_rx) = oneshot::channel();
        let inference_service = service.clone();
        let control_service = service.clone();
        let (health, health_service) = tonic_health::server::health_reporter();
        health
            .set_service_status(CONTROL_SERVICE, HealthServingStatus::Serving)
            .await;
        health
            .set_service_status(INFERENCE_SERVICE, HealthServingStatus::Serving)
            .await;
        tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(
                    pb::inference_server::InferenceServer::new(inference_service)
                        .max_encoding_message_size(64 * 1024 * 1024)
                        .max_decoding_message_size(64 * 1024 * 1024),
                )
                .add_service(pb::control_server::ControlServer::new(control_service))
                .add_service(health_service)
                .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
                    let _ = shutdown_rx.await;
                })
                .await
                .expect("serve fake vLLM");
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

fn request() -> PreprocessedRequest {
    PreprocessedRequest::builder()
        .model("served-model".to_string())
        .token_ids(vec![11, 22, 33])
        .stop_conditions(StopConditions {
            max_tokens: Some(1),
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
            prompt_logprobs: Some(1),
            ..Default::default()
        })
        .mdc_sum(Some("model-checksum".to_string()))
        .routing(Some(RoutingHints {
            cache_namespace: Some("cache-salt".to_string()),
            ..Default::default()
        }))
        .extra_args(Some(json!({
            "nvext": {"cache_salt": "cache-salt", "token_in": true},
            "bypass_prefix_cache": true,
            "kv_transfer_params": {
                "connector_data": {"values": [1, true, null]}
            }
        })))
        .build()
        .expect("request")
}

fn decode_request() -> PreprocessedRequest {
    let mut request = request();
    request.prefill_result = Some(PrefillResult {
        disaggregated_params: json!({
            "do_remote_decode": false,
            "do_remote_prefill": true,
            "remote_engine_id": "prefill-0",
            "remote_host": "127.0.0.1",
            "remote_port": 20097,
            "remote_block_ids": [7, 8],
        }),
        prompt_tokens_details: None,
    });
    request
}

fn engine(endpoint: &str, mode: DisaggregationMode, connections: usize) -> VllmSidecarEngine {
    let transport = GrpcTransportConfig {
        connections: NonZeroUsize::new(connections).expect("non-zero connection count"),
        ..Default::default()
    };
    VllmSidecarEngine::new(
        GrpcEndpoint::parse(endpoint, "--vllm-endpoint").expect("valid test endpoint"),
        DiscoveredModel::from_proto(model_info(), server_info()).expect("valid discovery"),
        mode,
        transport,
    )
}

async fn runtime_endpoint(namespace: &str) -> dynamo_runtime::component::Endpoint {
    let runtime = Runtime::from_current().expect("current runtime");
    let drt = DistributedRuntime::new(runtime, DistributedConfig::process_local())
        .await
        .expect("process-local DRT");
    let endpoint = drt
        .namespace(namespace)
        .expect("namespace")
        .component("backend")
        .expect("component")
        .endpoint("generate");
    let mut base = ModelDeploymentCard::with_name_only("model-source");
    base.source_path = Some("model-source".to_string());
    endpoint
        .drt()
        .discovery()
        .register(
            DiscoverySpec::from_model(
                namespace.to_string(),
                "backend".to_string(),
                "generate".to_string(),
                &base,
            )
            .expect("base discovery spec"),
        )
        .await
        .expect("register base model");
    endpoint
}

async fn engine_from_args(
    endpoint: &str,
) -> (VllmSidecarEngine, dynamo_backend_common::WorkerConfig) {
    let argv = vec![
        "dynamo-vllm-sidecar".to_string(),
        "--vllm-endpoint".to_string(),
        endpoint.to_string(),
        "--grpc-connections".to_string(),
        "2".to_string(),
        "--grpc-startup-deadline-secs".to_string(),
        "5".to_string(),
        "--grpc-connect-attempt-timeout-secs".to_string(),
        "1".to_string(),
    ];
    tokio::task::spawn_blocking(move || VllmSidecarEngine::from_args(Some(argv)))
        .await
        .expect("bootstrap task")
        .expect("bootstrap discovery")
}

async fn collect(
    engine: &VllmSidecarEngine,
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

#[test]
fn discovery_rejects_incompatible_model_metadata() {
    let mut unsupported_api = server_info();
    unsupported_api.api_version = "unsupported".to_string();

    let mut missing_model_id = model_info();
    missing_model_id.model_id.clear();

    let mut missing_served_name = model_info();
    missing_served_name.served_model_name.clear();

    let mut unsupported_input = model_info();
    unsupported_input.supports_token_ids_input = false;

    for (case, model, server) in [
        ("unsupported API", model_info(), unsupported_api),
        ("missing model ID", missing_model_id, server_info()),
        ("missing served name", missing_served_name, server_info()),
        ("unsupported input", unsupported_input, server_info()),
    ] {
        assert!(
            DiscoveredModel::from_proto(model, server).is_err(),
            "{case} metadata should be rejected"
        );
    }
}

#[tokio::test]
async fn startup_rejects_model_identity_change_after_bootstrap() {
    let server = FakeServer::start(FakeVllm::default()).await;
    let (engine, _) = engine_from_args(&server.endpoint).await;

    let mut changed = model_info();
    changed.served_model_name = "changed-served-model".to_string();
    *server.service.model_info_override.lock().await = Some(changed);

    assert!(engine.start(0).await.is_err());
}

#[tokio::test]
async fn aggregated_generation_converts_request_stream_and_usage() {
    let server = FakeServer::start(FakeVllm::default()).await;
    let (engine, worker) = engine_from_args(&server.endpoint).await;
    assert_eq!(worker.model_name, "model-source");
    assert_eq!(worker.served_model_name.as_deref(), Some("served-model"));
    assert!(worker.reasoning_parser.is_none());
    assert!(worker.tool_call_parser.is_none());
    let config = engine.start(0).await.expect("start");
    assert_eq!(config.model, "model-source");
    assert_eq!(config.served_model_name.as_deref(), Some("served-model"));
    assert_eq!(config.model_aliases, ["model-alias"]);
    let registration = config.llm.expect("LLM registration");
    assert_eq!(registration.context_length, Some(8192));
    assert_eq!(registration.kv_cache_block_size, Some(16));
    assert_eq!(registration.total_kv_blocks, Some(4096));
    assert_eq!(registration.max_num_seqs, Some(128));
    assert_eq!(registration.max_num_batched_tokens, Some(2048));
    assert_eq!(registration.max_gpu_lora_count, Some(4));
    assert_eq!(registration.data_parallel_size, Some(2));
    assert_eq!(registration.data_parallel_start_rank, Some(0));

    let sources = engine.kv_event_sources().await.expect("KV event sources");
    assert_eq!(sources.len(), 2);
    assert_eq!(
        sources
            .iter()
            .map(|source| source.dp_rank())
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([0, 1])
    );
    assert!(sources.iter().all(|source| matches!(
        source,
        dynamo_backend_common::KvEventSource::Zmq { topic, .. } if topic.is_empty()
    )));
    assert_eq!(
        sources
            .iter()
            .map(|source| match source {
                dynamo_backend_common::KvEventSource::Zmq { endpoint, .. } => endpoint.as_str(),
                dynamo_backend_common::KvEventSource::Push { .. } => unreachable!(),
            })
            .collect::<Vec<_>>(),
        ["tcp://127.0.0.1:20081", "tcp://127.0.0.1:20082"]
    );

    let mut routed_request = serde_json::to_value(request()).expect("serialize request");
    routed_request["routing"] = json!({"dp_rank": 1, "cache_salt": "cache-salt"});
    let outputs = collect(
        &engine,
        serde_json::from_value(routed_request).expect("deserialize routed request"),
    )
    .await;
    assert_eq!(outputs.len(), 1);
    let terminal = &outputs[0];
    assert_eq!(terminal.token_ids, [42]);
    assert_eq!(terminal.text.as_deref(), Some(" token"));
    assert_eq!(terminal.finish_reason, Some(FinishReason::Stop));
    assert_eq!(terminal.log_probs.as_deref(), Some(&[-0.25][..]));
    assert_eq!(terminal.top_logprobs.as_ref().unwrap()[0].len(), 2);
    let usage = terminal.completion_usage.as_ref().expect("usage");
    assert_eq!((usage.prompt_tokens, usage.completion_tokens), (3, 1));
    assert!(terminal.engine_data.as_ref().unwrap()["prompt_logprobs"].is_array());

    let requests = server.service.requests.lock().await;
    let sent = requests.first().expect("recorded request");
    assert_eq!(sent.model, "served-model");
    assert_eq!(sent.priority, 0);
    assert_eq!(
        server.service.data_parallel_rank_metadata.lock().await[0],
        Some("1".to_string())
    );
    let sampling = sent.sampling.as_ref().unwrap();
    assert_eq!(
        (sampling.top_k, sampling.top_p, sampling.min_p),
        (4, 0.9, 0.1)
    );
    assert_eq!(sampling.seed, Some(123));
    let decoding = sent.decoding.as_ref().unwrap();
    assert_eq!(
        (
            decoding.presence_penalty,
            decoding.frequency_penalty,
            decoding.repetition_penalty,
        ),
        (0.3, 0.4, 1.1)
    );
    assert!(matches!(
        decoding.structured_output,
        Some(pb::decoding_parameters::StructuredOutput::Json(_))
    ));
    let stopping = sent.stopping.as_ref().unwrap();
    assert_eq!((stopping.max_new_tokens, stopping.min_new_tokens), (1, 1));
    assert_eq!(stopping.stop_strings, ["done"]);
    assert!(stopping.include_stop_strings);
    assert!(stopping.ignore_eos);
    let kv = sent.kv.as_ref().unwrap();
    assert!(kv.bypass_prefix_cache);
    assert_eq!(kv.cache_salt, "dynamo-cache-salt:cache-salt");
    assert_eq!(
        struct_to_json(kv.kv_transfer_params.clone().unwrap()).unwrap(),
        json!({"connector_data": {"values": [1, true, null]}})
    );
}

#[tokio::test]
async fn lora_request_selection_reaches_the_grpc_boundary() {
    let server = FakeServer::start(FakeVllm::default()).await;
    let engine = engine(&server.endpoint, DisaggregationMode::Aggregated, 1);
    engine.start(0).await.expect("start");
    let mut value = serde_json::to_value(request()).expect("serialize request");
    value["routing"]["lora_name"] = json!("math-r8");
    let request = serde_json::from_value(value).expect("deserialize request");

    let outputs = collect(&engine, request).await;

    assert_eq!(outputs.len(), 1);
    assert_eq!(server.service.requests.lock().await[0].lora_name, "math-r8");
}

#[tokio::test]
async fn native_lora_lifecycle_reconciles_committed_rpc_errors() {
    let service = FakeVllm::default();
    service.load_commit_error.store(true, Ordering::SeqCst);
    service.unload_commit_error.store(true, Ordering::SeqCst);
    let server = FakeServer::start(service).await;
    let engine = engine(&server.endpoint, DisaggregationMode::Aggregated, 1);
    engine.start(0).await.expect("start");
    let endpoint = runtime_endpoint("lora_lifecycle").await;
    engine
        .on_endpoint_ready(endpoint.clone())
        .await
        .expect("endpoint ready");
    assert_eq!(
        engine.supported_updates().await.unwrap(),
        ["load_lora", "unload_lora", "list_loras"]
    );

    let adapter_dir = tempfile::tempdir().expect("adapter tempdir");
    std::fs::write(adapter_dir.path().join("adapter_config.json"), "{}").unwrap();
    std::fs::write(adapter_dir.path().join("adapter_model.safetensors"), []).unwrap();
    let load = engine
        .engine_update(
            "load_lora".to_string(),
            json!({
                "lora_name": "math-r8",
                "source": {"uri": format!("file://{}", adapter_dir.path().display())},
            }),
        )
        .await
        .unwrap();
    assert_eq!(load["status"], "success");
    assert_eq!(load["lora_id"], 1);
    assert_eq!(server.service.loras.lock().await.len(), 1);
    let models = endpoint
        .drt()
        .discovery()
        .list(DiscoveryQuery::EndpointModels {
            namespace: "lora_lifecycle".to_string(),
            component: "backend".to_string(),
            endpoint: "generate".to_string(),
        })
        .await
        .unwrap();
    let lora_card = models
        .iter()
        .find(|instance| {
            matches!(
                instance,
                DiscoveryInstance::Model {
                    model_suffix: Some(_),
                    ..
                }
            )
        })
        .expect("LoRA discovery sibling")
        .deserialize_model::<ModelDeploymentCard>()
        .unwrap();
    assert_eq!(lora_card.name(), "math-r8");
    assert_eq!(lora_card.lora.unwrap().max_gpu_lora_count, Some(4));

    let unload = engine
        .engine_update("unload_lora".to_string(), json!({"lora_name": "math-r8"}))
        .await
        .unwrap();
    assert_eq!(unload["status"], "success");
    assert!(server.service.loras.lock().await.is_empty());
    assert_eq!(
        endpoint
            .drt()
            .discovery()
            .list(DiscoveryQuery::EndpointModels {
                namespace: "lora_lifecycle".to_string(),
                component: "backend".to_string(),
                endpoint: "generate".to_string(),
            })
            .await
            .unwrap()
            .len(),
        1
    );
}

#[tokio::test]
async fn startup_republishes_loras_loaded_before_sidecar_restart() {
    let service = FakeVllm::default();
    service.loras.lock().await.push(pb::LoraAdapter {
        lora_id: 7,
        lora_name: "math-r8".to_string(),
        source_path: "/shared/loras/math-r8".to_string(),
    });
    let server = FakeServer::start(service).await;
    let engine = engine(&server.endpoint, DisaggregationMode::Aggregated, 1);
    engine.start(0).await.expect("start");
    let endpoint = runtime_endpoint("lora_restart").await;
    engine.on_endpoint_ready(endpoint.clone()).await.unwrap();

    engine.supported_updates().await.unwrap();

    let models = endpoint
        .drt()
        .discovery()
        .list(DiscoveryQuery::EndpointModels {
            namespace: "lora_restart".to_string(),
            component: "backend".to_string(),
            endpoint: "generate".to_string(),
        })
        .await
        .unwrap();
    assert!(models.iter().any(|instance| {
        matches!(
            instance,
            DiscoveryInstance::Model {
                model_suffix: Some(_),
                ..
            }
        )
    }));
}

#[tokio::test]
async fn lora_updates_are_capability_gated() {
    let mut base_only = model_info();
    base_only.supports_lora = false;
    let disabled = VllmSidecarEngine::new(
        GrpcEndpoint::parse("http://127.0.0.1:1", "test endpoint").unwrap(),
        DiscoveredModel::from_proto(base_only, server_info()).unwrap(),
        DisaggregationMode::Aggregated,
        GrpcTransportConfig::default(),
    );
    assert!(disabled.supported_updates().await.unwrap().is_empty());
}

#[tokio::test]
async fn grpc_request_errors_are_propagated() {
    let service = FakeVllm::default();
    service.reject.store(true, Ordering::SeqCst);
    let server = FakeServer::start(service).await;
    let engine = engine(&server.endpoint, DisaggregationMode::Aggregated, 1);
    engine.start(0).await.expect("start");

    let context = dynamo_backend_common::testing::mock_context();
    let result = engine
        .generate(request(), GenerateContext::new(context, None))
        .await;
    assert!(result.is_err());
    assert_eq!(server.service.requests.lock().await.len(), 1);
}

#[tokio::test]
async fn prefill_decode_handoff_is_opaque_and_repeatable() {
    let server = FakeServer::start(FakeVllm::default()).await;
    let prefill = engine(&server.endpoint, DisaggregationMode::Prefill, 1);
    let decode = engine(&server.endpoint, DisaggregationMode::Decode, 1);
    prefill.start(0).await.expect("start prefill");
    decode.start(1).await.expect("start decode");

    for _ in 0..2 {
        let prefill_outputs = collect(&prefill, request()).await;
        let handoff = prefill_outputs[0]
            .disaggregated_params
            .clone()
            .expect("handoff");
        assert_eq!(prefill_outputs[0].token_ids, Vec::<u32>::new());
        assert_eq!(handoff["nested"]["flags"], json!([true, null, "opaque"]));

        let mut decode_request = request();
        decode_request.prefill_result = Some(PrefillResult {
            disaggregated_params: handoff.clone(),
            prompt_tokens_details: None,
        });
        let decode_outputs = collect(&decode, decode_request).await;
        assert_eq!(decode_outputs[0].token_ids, [42]);

        let requests = server.service.requests.lock().await;
        let decode_wire = requests.last().unwrap().kv.as_ref().unwrap();
        let decoded = struct_to_json(decode_wire.kv_transfer_params.clone().unwrap()).unwrap();
        // Every field round-trips opaquely except remote_port, which the sidecar
        // stringifies so vLLM builds a valid NIXL side-channel URL (a protobuf
        // Struct number would reach the engine as `20097.0`).
        let mut expected = handoff.clone();
        expected["remote_port"] = json!("20097");
        assert_eq!(decoded, expected);
    }
}

#[tokio::test]
async fn component_honors_config_for_aggregated_but_fixes_disagg_roles() {
    let server = FakeServer::start(FakeVllm::default()).await;
    for (extra, expected) in [
        (Vec::<&str>::new(), "custom"),
        (vec!["--disaggregation-mode", "prefill"], "prefill"),
        (vec!["--disaggregation-mode", "decode"], "backend"),
    ] {
        let mut argv = vec![
            "dynamo-vllm-sidecar".to_string(),
            "--vllm-endpoint".to_string(),
            server.endpoint.clone(),
            "--component".to_string(),
            "custom".to_string(),
        ];
        argv.extend(extra.into_iter().map(str::to_string));
        let component =
            tokio::task::spawn_blocking(move || VllmSidecarEngine::from_args(Some(argv)))
                .await
                .expect("bootstrap task")
                .expect("from_args")
                .1
                .component;
        assert_eq!(component, expected);
    }
}

#[tokio::test]
async fn pool_uses_each_configured_connection() {
    let server = FakeServer::start(FakeVllm::default()).await;
    let transport = GrpcTransportConfig {
        connections: NonZeroUsize::new(2).unwrap(),
        ..Default::default()
    };
    let endpoint = GrpcEndpoint::parse(&server.endpoint, "--vllm-endpoint").unwrap();
    let deadline = crate::client::startup_deadline(transport.startup_deadline).unwrap();
    let client = VllmClient::connect(&endpoint, transport, deadline)
        .await
        .expect("connect pool");
    assert_eq!(client.connection_count(), 2);

    for index in 0..4 {
        let mut stream = client
            .generate_stream(
                pb::GenerateRequest {
                    request_id: format!("request-{index}"),
                    prompt: Some(pb::generate_request::Prompt::Text("hello".to_string())),
                    ..Default::default()
                },
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
    assert!(
        server
            .service
            .data_parallel_rank_metadata
            .lock()
            .await
            .iter()
            .all(Option::is_none)
    );
}

#[tokio::test]
async fn cancellation_drops_the_remote_stream() {
    let service = FakeVllm::default();
    service.hang.store(true, Ordering::SeqCst);
    let server = FakeServer::start(service).await;
    let engine = engine(&server.endpoint, DisaggregationMode::Aggregated, 1);
    engine.start(0).await.expect("start");

    let context = dynamo_backend_common::testing::mock_context();
    let mut stream = engine
        .generate(request(), GenerateContext::new(context.clone(), None))
        .await
        .expect("generate");
    let first = stream.next().await.unwrap().unwrap();
    assert_eq!(first.token_ids, [42]);
    context.stop_generating();
    let terminal = stream.next().await.unwrap().unwrap();
    assert_eq!(terminal.finish_reason, Some(FinishReason::Cancelled));
    drop(stream);

    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while !server.service.server_stream_dropped.load(Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("server stream dropped");
}

#[tokio::test]
async fn cancellation_interrupts_pending_response_headers() {
    let service = FakeVllm::default();
    service.hang_before_headers.store(true, Ordering::SeqCst);
    let server = FakeServer::start(service).await;
    let engine = engine(&server.endpoint, DisaggregationMode::Aggregated, 1);
    engine.start(0).await.expect("start");

    let context = dynamo_backend_common::testing::mock_context();
    let generate = engine.generate(request(), GenerateContext::new(context.clone(), None));
    tokio::pin!(generate);

    tokio::select! {
        _ = &mut generate => panic!("generate returned before cancellation"),
        _ = async {
            while !server.service.headers_pending.load(Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
        } => {}
    }

    context.stop_generating();
    let mut stream = tokio::time::timeout(std::time::Duration::from_secs(2), &mut generate)
        .await
        .expect("cancel pending headers")
        .expect("generate cancellation stream");
    let terminal = stream.next().await.unwrap().unwrap();
    assert_eq!(terminal.finish_reason, Some(FinishReason::Cancelled));
    server.service.release_headers.notify_waiters();
}

#[tokio::test]
async fn decode_cancellation_waits_for_submission_and_first_token() {
    let service = FakeVllm::default();
    service.hang_before_headers.store(true, Ordering::SeqCst);
    service
        .hold_before_first_token
        .store(true, Ordering::SeqCst);
    let server = FakeServer::start(service).await;
    let engine = engine(&server.endpoint, DisaggregationMode::Decode, 1);
    engine.start(0).await.expect("start");

    let context = dynamo_backend_common::testing::mock_context();
    let generate = engine.generate(
        decode_request(),
        GenerateContext::new(context.clone(), None),
    );
    tokio::pin!(generate);

    tokio::select! {
        _ = &mut generate => panic!("decode returned before response headers were gated"),
        _ = async {
            while !server.service.headers_pending.load(Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
        } => {}
    }
    assert_eq!(server.service.requests.lock().await.len(), 1);
    context.stop_generating();
    tokio::select! {
        _ = &mut generate => panic!("decode cancellation returned before response headers"),
        _ = tokio::time::sleep(std::time::Duration::from_millis(50)) => {}
    }

    server.service.release_headers.notify_one();
    let mut stream = tokio::time::timeout(std::time::Duration::from_secs(2), &mut generate)
        .await
        .expect("decode response headers")
        .expect("decode stream");
    let next = stream.next();
    tokio::pin!(next);
    tokio::select! {
        _ = &mut next => panic!("decode returned before the first token was gated"),
        _ = async {
            while !server.service.first_token_pending.load(Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
        } => {}
    }
    assert!(
        !server.service.server_stream_dropped.load(Ordering::SeqCst),
        "decode stream dropped before the first token"
    );
    tokio::select! {
        _ = &mut next => panic!("decode cancellation completed before the first token"),
        _ = tokio::time::sleep(std::time::Duration::from_millis(50)) => {}
    }

    server.service.release_first_token.notify_one();
    let terminal = tokio::time::timeout(std::time::Duration::from_secs(2), &mut next)
        .await
        .expect("first token did not release decode cancellation")
        .expect("cancelled terminal")
        .expect("cancelled output");
    assert_eq!(terminal.finish_reason, Some(FinishReason::Cancelled));
    drop(stream);

    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while !server.service.server_stream_dropped.load(Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("server stream dropped after first token");
}

#[tokio::test]
async fn decode_cancellation_maps_premature_eof_to_cancelled() {
    let service = FakeVllm::default();
    service
        .close_before_first_token
        .store(true, Ordering::SeqCst);
    let server = FakeServer::start(service).await;
    let engine = engine(&server.endpoint, DisaggregationMode::Decode, 1);
    engine.start(0).await.expect("start");

    let context = dynamo_backend_common::testing::mock_context();
    let mut stream = engine
        .generate(
            decode_request(),
            GenerateContext::new(context.clone(), None),
        )
        .await
        .expect("decode stream");
    context.stop_generating();
    let terminal = tokio::time::timeout(std::time::Duration::from_secs(2), stream.next())
        .await
        .expect("premature EOF did not release decode cancellation")
        .expect("cancelled terminal")
        .expect("cancelled output");
    assert_eq!(terminal.finish_reason, Some(FinishReason::Cancelled));
}

#[tokio::test]
async fn unsupported_features_fail_before_rpc_submission() {
    let server = FakeServer::start(FakeVllm::default()).await;
    let engine = engine(&server.endpoint, DisaggregationMode::Aggregated, 1);
    engine.start(0).await.expect("start");

    let mut requests = Vec::new();

    let mut multiple = request();
    multiple.sampling_options.n = Some(2);
    requests.push(multiple);

    let mut embeddings = request();
    embeddings.prompt_embeds = Some("encoded".to_string());
    requests.push(embeddings);

    let mut multimodal = request();
    multimodal.mm_processor_kwargs = Some(json!({"use_audio_in_video": true}));
    requests.push(multimodal);

    let mut mismatched_cache_salt = request();
    mismatched_cache_salt.extra_args.as_mut().unwrap()["nvext"]["cache_salt"] =
        json!("different-cache-salt");
    requests.push(mismatched_cache_salt);

    for unsupported in requests {
        let context = dynamo_backend_common::testing::mock_context();
        let result = engine
            .generate(unsupported, GenerateContext::new(context, None))
            .await;
        assert!(result.is_err());
    }
    assert!(server.service.requests.lock().await.is_empty());
}
