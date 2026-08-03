// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use std::net::SocketAddr;
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use dynamo_backend_common::{
    DisaggregationMode, FinishReason, GenerateContext, LLMEngine, OutputOptions, PrefillResult,
    PreprocessedRequest, SamplingOptions, StopConditions,
};
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_runtime::discovery::{
    Discovery, DiscoveryInstance, DiscoveryQuery, DiscoverySpec, MockDiscovery, SharedMockRegistry,
};
use dynamo_sidecar_common::{GrpcEndpoint, GrpcTransportConfig};
use futures::{Stream, StreamExt};
use serde_json::json;
use tokio::net::TcpListener;
use tokio::sync::{Mutex, Notify, oneshot};
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{Request, Response, Status};

use crate::client::VllmClient;
use crate::convert::{ResponseState, build_generate_request};
use crate::engine::{VllmSidecarEngine, rl_worker_metadata};
use crate::json::{json_to_struct, struct_to_json};
use crate::lora::{next_lora_id, parse_load_lora, publish_lora_model_to_discovery};
use crate::model::ConfiguredModel;
use crate::proto as pb;

#[derive(Clone, Default)]
struct FakeInference {
    requests: Arc<Mutex<Vec<pb::GenerateRequest>>>,
    peers: Arc<Mutex<Vec<SocketAddr>>>,
    reject: Arc<AtomicBool>,
    hang: Arc<AtomicBool>,
    hang_before_headers: Arc<AtomicBool>,
    headers_pending: Arc<AtomicBool>,
    release_headers: Arc<Notify>,
    server_stream_dropped: Arc<AtomicBool>,
}

struct DropSignal(Arc<AtomicBool>);

impl Drop for DropSignal {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

#[tonic::async_trait]
impl pb::inference_server::Inference for FakeInference {
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

#[derive(Clone)]
struct FakeControl {
    server_info: pb::ServerInfo,
    model_info: pb::ModelInfo,
    server_info_calls: Arc<AtomicUsize>,
    model_info_calls: Arc<AtomicUsize>,
    hang_server_info: Arc<AtomicBool>,
    loras: Arc<Mutex<Vec<pb::LoraAdapter>>>,
}

impl Default for FakeControl {
    fn default() -> Self {
        Self {
            server_info: pb::ServerInfo {
                engine_version: "test".to_string(),
                api_version: "vllm".to_string(),
                instance_id: "instance-1".to_string(),
                parallelism: Some(pb::ParallelismInfo {
                    tensor_parallel_size: 1,
                    pipeline_parallel_size: 1,
                    data_parallel_size: 1,
                    data_parallel_rank: 0,
                    decode_context_parallel_size: 1,
                    data_parallel_start_rank: 0,
                    managed_data_parallel_size: 1,
                }),
                max_model_len: 4096,
                kv_block_size: 16,
                total_kv_blocks: 1024,
                max_running_requests: 64,
                max_batched_tokens: 8192,
                max_loras: 0,
                capabilities: vec![
                    "generate.sampling.v2".to_string(),
                    "generate.preprocessed_mm.v1".to_string(),
                    "generate.routed_experts.v1".to_string(),
                ],
            },
            model_info: pb::ModelInfo {
                model_id: "model-source".to_string(),
                served_model_name: "served-model".to_string(),
                served_model_aliases: vec!["model-alias".to_string()],
                tokenizer_modes: vec!["auto".to_string()],
                supports_text_input: true,
                supports_token_ids_input: true,
                supports_lora: false,
                supports_multimodal: false,
                reasoning_parser: String::new(),
                tool_call_parser: String::new(),
            },
            server_info_calls: Arc::new(AtomicUsize::new(0)),
            model_info_calls: Arc::new(AtomicUsize::new(0)),
            hang_server_info: Arc::new(AtomicBool::new(false)),
            loras: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[tonic::async_trait]
impl pb::control_server::Control for FakeControl {
    async fn get_server_info(
        &self,
        _request: Request<pb::GetServerInfoRequest>,
    ) -> Result<Response<pb::ServerInfo>, Status> {
        self.server_info_calls.fetch_add(1, Ordering::SeqCst);
        if self.hang_server_info.load(Ordering::SeqCst) {
            std::future::pending::<()>().await;
        }
        Ok(Response::new(self.server_info.clone()))
    }

    async fn get_model_info(
        &self,
        _request: Request<pb::GetModelInfoRequest>,
    ) -> Result<Response<pb::ModelInfo>, Status> {
        self.model_info_calls.fetch_add(1, Ordering::SeqCst);
        Ok(Response::new(self.model_info.clone()))
    }

    async fn abort(
        &self,
        _request: Request<pb::AbortRequest>,
    ) -> Result<Response<pb::AbortResponse>, Status> {
        Err(Status::unimplemented("abort"))
    }

    async fn drain(
        &self,
        _request: Request<pb::DrainRequest>,
    ) -> Result<Response<pb::DrainResponse>, Status> {
        Err(Status::unimplemented("drain"))
    }

    async fn load_lora(
        &self,
        request: Request<pb::LoadLoraRequest>,
    ) -> Result<Response<pb::LoadLoraResponse>, Status> {
        let request = request.into_inner();
        let adapter = request
            .adapter
            .ok_or_else(|| Status::invalid_argument("adapter required"))?;
        let mut loras = self.loras.lock().await;
        let already_loaded = loras.iter().any(|item| item.lora_name == adapter.lora_name);
        if already_loaded && !request.load_inplace {
            return Err(Status::already_exists("adapter already loaded"));
        }
        loras.retain(|item| item.lora_name != adapter.lora_name);
        loras.push(adapter.clone());
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
        let mut loras = self.loras.lock().await;
        let index = loras
            .iter()
            .position(|item| item.lora_name == name)
            .ok_or_else(|| Status::not_found("adapter not found"))?;
        Ok(Response::new(pb::UnloadLoraResponse {
            adapter: Some(loras.remove(index)),
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
        Err(Status::unimplemented("get_kv_event_sources"))
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
            }),
            routed_experts: None,
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

#[test]
fn token_native_compatibility_envelope_is_accepted_when_fields_are_projected() {
    let mut request = request();
    request.extra_args = Some(json!({
        "vllm_tito": {
            "request_id": "request-1",
            "sampling_params": {
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 4,
                "min_p": 0.1,
                "seed": 123,
                "max_tokens": 1,
                "min_tokens": 1,
                "presence_penalty": 0.3,
                "frequency_penalty": 0.4,
                "repetition_penalty": 1.1,
                "stop_token_ids": [2],
                "ignore_eos": true,
                "logprobs": 1,
                "prompt_logprobs": 1,
                "cache_salt": "cache-salt",
                "skip_special_tokens": false,
                "return_token_ids": true
            },
            "model": "served-model",
            "stream": false,
            "cache_salt": "cache-salt",
            "priority": 0
        },
        "bypass_prefix_cache": true,
        "kv_transfer_params": {
            "connector_data": {"values": [1, true, null]}
        }
    }));

    let wire = build_generate_request(
        request,
        "request-1".to_string(),
        DisaggregationMode::Aggregated,
    )
    .expect("projected compatibility envelope");

    assert_eq!(wire.temperature, Some(0.2));
    assert_eq!(wire.stopping.as_ref().unwrap().stop_token_ids, [2]);
    assert!(wire.response.as_ref().unwrap().output_logprobs);
    assert_eq!(wire.kv.as_ref().unwrap().cache_salt, "cache-salt");
}

#[test]
fn token_native_compatibility_envelope_rejects_disabled_token_ids() {
    let mut request = request();
    request.extra_args = Some(json!({
        "vllm_tito": {
            "request_id": "request-1",
            "sampling_params": {
                "max_tokens": 1,
                "return_token_ids": false
            },
            "stream": false,
            "priority": 0
        }
    }));

    let error = build_generate_request(
        request,
        "request-1".to_string(),
        DisaggregationMode::Aggregated,
    )
    .expect_err("the token-native sidecar always returns token IDs");

    assert!(
        error
            .to_string()
            .contains("sampling_params.return_token_ids must be true")
    );
}

#[test]
fn token_native_compatibility_envelope_rejects_conflicting_cache_salt() {
    let mut request = request();
    request.extra_args = Some(json!({
        "vllm_tito": {
            "request_id": "request-1",
            "sampling_params": {
                "max_tokens": 1,
                "cache_salt": "different-policy-version"
            },
            "cache_salt": "cache-salt",
            "stream": false,
            "priority": 0
        }
    }));

    let error = build_generate_request(
        request,
        "request-1".to_string(),
        DisaggregationMode::Aggregated,
    )
    .expect_err("duplicate cache salts must match the canonical routing salt");

    assert!(
        error
            .to_string()
            .contains("sampling_params.cache_salt must match the canonical cache_salt")
    );
}

#[test]
fn token_native_compatibility_envelope_rejects_unprojected_fields() {
    let mut request = request();
    request.extra_args = Some(json!({
        "vllm_tito": {
            "request_id": "request-1",
            "sampling_params": {
                "max_tokens": 1,
                "future_sampling_field": true
            },
            "stream": false,
            "priority": 0
        }
    }));

    let error = build_generate_request(
        request,
        "request-1".to_string(),
        DisaggregationMode::Aggregated,
    )
    .expect_err("unprojected compatibility fields must fail closed");

    assert!(
        error
            .to_string()
            .contains("extra_args.vllm_tito.sampling_params.future_sampling_field")
    );
}

#[test]
fn rl_metadata_accepts_only_engine_advertised_model_names() {
    let model = FakeControl::default().model_info;
    let default = rl_worker_metadata("http://worker:8120".to_string(), 2, &model, None).unwrap();
    assert_eq!(default.model, "model-source");

    let alias = rl_worker_metadata(
        "http://worker:8120".to_string(),
        2,
        &model,
        Some("model-alias"),
    )
    .unwrap();
    assert_eq!(alias.model, "model-alias");
    assert!(
        rl_worker_metadata("http://worker:8120".to_string(), 2, &model, Some("unknown"),).is_err()
    );
}

struct FakeServer {
    endpoint: String,
    service: FakeInference,
    control: FakeControl,
    shutdown: Option<oneshot::Sender<()>>,
}

impl FakeServer {
    async fn start(service: FakeInference) -> Self {
        Self::start_with_control(service, FakeControl::default()).await
    }

    async fn start_with_control(service: FakeInference, control: FakeControl) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let address = listener.local_addr().expect("address");
        let (shutdown, shutdown_rx) = oneshot::channel();
        let server_service = service.clone();
        let server_control = control.clone();
        tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(
                    pb::inference_server::InferenceServer::new(server_service)
                        .max_encoding_message_size(64 * 1024 * 1024)
                        .max_decoding_message_size(64 * 1024 * 1024),
                )
                .add_service(pb::control_server::ControlServer::new(server_control))
                .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
                    let _ = shutdown_rx.await;
                })
                .await
                .expect("serve fake vLLM");
        });
        Self {
            endpoint: format!("http://{address}"),
            service,
            control,
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

#[test]
fn load_lora_payload_requires_a_local_file_uri() {
    let update = parse_load_lora(&json!({
        "lora_name": "math-r8",
        "source": {"uri": "file:///shared/adapter/step_1"},
        "load_inplace": true,
    }))
    .expect("valid local LoRA update");
    assert_eq!(update.name, "math-r8");
    assert_eq!(update.path.to_string_lossy(), "/shared/adapter/step_1");
    assert!(update.load_inplace);

    for uri in [
        "https://example.com/adapter",
        "file://remote/adapter",
        "file:///adapter?version=1",
        "file:///adapter#fragment",
        "relative",
    ] {
        let error = parse_load_lora(&json!({
            "lora_name": "math-r8",
            "source": {"uri": uri},
        }))
        .expect_err("non-local LoRA URI must be rejected");
        assert!(error.to_string().contains("local file URI"));
    }
}

#[test]
fn next_lora_id_reuses_existing_name_and_allocates_after_the_highest_id() {
    let loaded = vec![
        pb::LoraAdapter {
            lora_id: 3,
            lora_name: "other".to_string(),
            source_path: "/other".to_string(),
        },
        pb::LoraAdapter {
            lora_id: 7,
            lora_name: "math-r8".to_string(),
            source_path: "/old".to_string(),
        },
    ];
    assert_eq!(next_lora_id(&loaded, "math-r8").unwrap(), 7);
    assert_eq!(next_lora_id(&loaded, "new-adapter").unwrap(), 8);
}

#[tokio::test]
async fn native_lora_client_loads_in_place_and_lists_the_adapter() {
    let server = FakeServer::start(FakeInference::default()).await;
    let client = VllmClient::connect(
        &GrpcEndpoint::parse(&server.endpoint, "test").unwrap(),
        GrpcTransportConfig::default(),
    )
    .await
    .unwrap();
    let adapter = pb::LoraAdapter {
        lora_id: 1,
        lora_name: "math-r8".to_string(),
        source_path: "/shared/adapter/step_1".to_string(),
    };

    let loaded = client.load_lora(adapter.clone(), true).await.unwrap();
    assert_eq!(loaded.adapter, Some(adapter.clone()));
    assert!(!loaded.already_loaded);
    let loaded_again = client.load_lora(adapter.clone(), true).await.unwrap();
    assert!(loaded_again.already_loaded);
    assert_eq!(client.list_loras().await.unwrap(), vec![adapter]);
}

#[tokio::test]
async fn loaded_lora_is_published_as_a_sibling_of_the_base_model() {
    let discovery = MockDiscovery::new(Some(42), SharedMockRegistry::new());
    let mut base = ModelDeploymentCard::with_name_only("Qwen/Qwen3-0.6B");
    base.source_path = Some("Qwen/Qwen3-0.6B".to_string());
    discovery
        .register(
            DiscoverySpec::from_model(
                "test".to_string(),
                "backend".to_string(),
                "generate".to_string(),
                &base,
            )
            .unwrap(),
        )
        .await
        .unwrap();
    let adapter = pb::LoraAdapter {
        lora_id: 7,
        lora_name: "math-r8".to_string(),
        source_path: "/shared/adapter/step_1".to_string(),
    };

    publish_lora_model_to_discovery(&discovery, "test", "backend", "generate", 42, &adapter, 4)
        .await
        .unwrap();

    let models = discovery
        .list(DiscoveryQuery::EndpointModels {
            namespace: "test".to_string(),
            component: "backend".to_string(),
            endpoint: "generate".to_string(),
        })
        .await
        .unwrap();
    assert_eq!(models.len(), 2);
    let lora = models
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
        .unwrap()
        .deserialize_model::<ModelDeploymentCard>()
        .unwrap();
    assert_eq!(lora.name(), "math-r8");
    assert_eq!(lora.source_path(), "Qwen/Qwen3-0.6B");
    assert_eq!(lora.lora.as_ref().unwrap().name, "math-r8");
    assert_eq!(lora.lora.as_ref().unwrap().max_gpu_lora_count, Some(4));
    assert_eq!(lora.user_data.as_ref().unwrap()["lora_id"], json!(7));
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
        .mdc_sum(Some("cache-salt".to_string()))
        .extra_args(Some(json!({
            "bypass_prefix_cache": true,
            "kv_transfer_params": {
                "connector_data": {"values": [1, true, null]}
            }
        })))
        .build()
        .expect("request")
}

fn engine(endpoint: &str, mode: DisaggregationMode, connections: usize) -> VllmSidecarEngine {
    VllmSidecarEngine::new(
        GrpcEndpoint::parse(endpoint, "--vllm-endpoint").expect("valid test endpoint"),
        ConfiguredModel {
            source: "model-source".to_string(),
        },
        mode,
        GrpcTransportConfig {
            connections: NonZeroUsize::new(connections).expect("non-zero connection count"),
            ..Default::default()
        },
    )
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

#[tokio::test]
async fn aggregated_generation_converts_request_stream_and_usage() {
    let server = FakeServer::start(FakeInference::default()).await;
    let engine = engine(&server.endpoint, DisaggregationMode::Aggregated, 2);
    let config = engine.start(0).await.expect("start");
    assert_eq!(config.model, "model-source");
    assert_eq!(config.served_model_name, Some("served-model".to_string()));
    assert_eq!(server.control.server_info_calls.load(Ordering::SeqCst), 1);
    assert_eq!(server.control.model_info_calls.load(Ordering::SeqCst), 1);

    let outputs = collect(&engine, request()).await;
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
    assert!(sent.model.is_empty());
    assert_eq!(sent.priority, 0);
    let sampling = sent.sampling.as_ref().unwrap();
    assert_eq!(
        (sampling.top_k, sampling.top_p, sampling.min_p),
        (Some(4), Some(0.9), Some(0.1))
    );
    assert_eq!(sampling.seed, Some(123));
    let decoding = sent.decoding.as_ref().unwrap();
    assert_eq!(
        (
            decoding.presence_penalty,
            decoding.frequency_penalty,
            decoding.repetition_penalty,
        ),
        (Some(0.3), Some(0.4), Some(1.1))
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
    assert_eq!(kv.cache_salt, "cache-salt");
    assert_eq!(
        struct_to_json(kv.kv_transfer_params.clone().unwrap()).unwrap(),
        json!({"connector_data": {"values": [1, true, null]}})
    );
}

#[tokio::test]
async fn lora_routing_is_forwarded_to_native_grpc() {
    let server = FakeServer::start(FakeInference::default()).await;
    let engine = engine(&server.endpoint, DisaggregationMode::Aggregated, 1);
    engine.start(0).await.expect("start");

    let mut value = serde_json::to_value(request()).expect("serialize request");
    value["routing"] = json!({"lora_name": "math-r8"});
    let request = serde_json::from_value(value).expect("deserialize request");
    collect(&engine, request).await;

    let requests = server.service.requests.lock().await;
    assert_eq!(
        requests.first().expect("recorded request").lora_name,
        "math-r8"
    );
}

#[tokio::test]
async fn grpc_request_errors_are_propagated() {
    let service = FakeInference::default();
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
    let server = FakeServer::start(FakeInference::default()).await;
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
    let server = FakeServer::start(FakeInference::default()).await;
    let component = |extra: &[&str]| {
        let endpoint = server.endpoint.clone();
        let mut argv = vec![
            "dynamo-vllm-sidecar".to_string(),
            "--vllm-endpoint".to_string(),
            endpoint,
            "--model-path".to_string(),
            "test-model".to_string(),
            "--component".to_string(),
            "custom".to_string(),
        ];
        argv.extend(extra.iter().map(ToString::to_string));
        tokio::task::spawn_blocking(move || {
            VllmSidecarEngine::from_args(Some(argv))
                .expect("from_args")
                .1
                .component
        })
    };
    // Aggregated keeps the operator-configured component.
    assert_eq!(component(&[]).await.expect("component task"), "custom");
    // Disaggregated roles override to fixed names so the frontend can route.
    assert_eq!(
        component(&["--disaggregation-mode", "prefill"])
            .await
            .expect("component task"),
        "prefill"
    );
    assert_eq!(
        component(&["--disaggregation-mode", "decode"])
            .await
            .expect("component task"),
        "backend"
    );
}

#[tokio::test]
async fn sidecar_advertises_load_lora_only_when_native_server_supports_it() {
    let mut control = FakeControl::default();
    control.model_info.supports_lora = true;
    control.server_info.max_loras = 4;
    let server = FakeServer::start_with_control(FakeInference::default(), control).await;
    let endpoint = server.endpoint.clone();
    let engine = tokio::task::spawn_blocking(move || {
        VllmSidecarEngine::from_args(Some(vec![
            "dynamo-vllm-sidecar".to_string(),
            "--vllm-endpoint".to_string(),
            endpoint,
            "--model-path".to_string(),
            "test-model".to_string(),
        ]))
        .expect("LoRA-capable sidecar arguments")
        .0
    })
    .await
    .unwrap();
    assert_eq!(engine.supported_updates().await.unwrap(), ["load_lora"]);

    let server = FakeServer::start(FakeInference::default()).await;
    let endpoint = server.endpoint.clone();
    let engine = tokio::task::spawn_blocking(move || {
        VllmSidecarEngine::from_args(Some(vec![
            "dynamo-vllm-sidecar".to_string(),
            "--vllm-endpoint".to_string(),
            endpoint,
            "--model-path".to_string(),
            "test-model".to_string(),
        ]))
        .expect("base-only sidecar arguments")
        .0
    })
    .await
    .unwrap();
    assert!(engine.supported_updates().await.unwrap().is_empty());
}

#[tokio::test]
async fn pool_uses_each_configured_connection() {
    let server = FakeServer::start(FakeInference::default()).await;
    let transport = GrpcTransportConfig {
        connections: NonZeroUsize::new(2).unwrap(),
        ..Default::default()
    };
    let endpoint = GrpcEndpoint::parse(&server.endpoint, "--vllm-endpoint").unwrap();
    let client = VllmClient::connect(&endpoint, transport)
        .await
        .expect("connect pool");
    assert_eq!(client.connection_count(), 2);

    for index in 0..4 {
        let mut stream = client
            .generate_stream(pb::GenerateRequest {
                request_id: format!("request-{index}"),
                prompt: Some(pb::generate_request::Prompt::Text("hello".to_string())),
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

#[tokio::test]
async fn control_discovery_respects_the_startup_deadline() {
    let control = FakeControl::default();
    control.hang_server_info.store(true, Ordering::SeqCst);
    let server = FakeServer::start_with_control(FakeInference::default(), control).await;
    let transport = GrpcTransportConfig {
        connections: NonZeroUsize::new(1).unwrap(),
        startup_deadline: std::time::Duration::from_millis(100),
        ..Default::default()
    };
    let endpoint = GrpcEndpoint::parse(&server.endpoint, "--vllm-endpoint").unwrap();
    let result = VllmClient::connect_and_discover(&endpoint, transport).await;
    let error = result.err().expect("discovery must time out");
    assert!(format!("{error}").contains("startup deadline"));
}

#[tokio::test]
async fn cancellation_drops_the_remote_stream() {
    let service = FakeInference::default();
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
    let service = FakeInference::default();
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
async fn unsupported_features_fail_before_rpc_submission() {
    let server = FakeServer::start(FakeInference::default()).await;
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

    let mut value = serde_json::to_value(request()).expect("serialize request");
    value["routing"] = json!({"dp_rank": 1});
    requests.push(serde_json::from_value(value).expect("deserialize request"));

    for unsupported in requests {
        let context = dynamo_backend_common::testing::mock_context();
        let result = engine
            .generate(unsupported, GenerateContext::new(context, None))
            .await;
        assert!(result.is_err());
    }
    assert!(server.service.requests.lock().await.is_empty());
}
