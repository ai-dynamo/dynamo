// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::net::SocketAddr;
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};

use dynamo_backend_common::engine::RoutingHints;
use dynamo_backend_common::{
    DisaggregationMode, FinishReason, GenerateContext, LLMEngine, MultimodalData, OutputOptions,
    PrefillResult, PreprocessedRequest, SamplingOptions, StopConditions,
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
    control_calls: Arc<Mutex<Vec<(String, serde_json::Value)>>>,
    paused: Arc<AtomicBool>,
    sleeping_tags: Arc<Mutex<BTreeSet<String>>>,
    weight_version: Arc<Mutex<String>>,
    load_commit_error: Arc<AtomicBool>,
    unload_commit_error: Arc<AtomicBool>,
    /// Mirrors a vLLM started without LoRA: every lifecycle RPC fails the precondition.
    lora_disabled: Arc<AtomicBool>,
    /// Block inside `LoadLora` until released, to exercise concurrent lifecycle calls.
    hold_load: Arc<AtomicBool>,
    load_pending: Arc<AtomicBool>,
    release_load: Arc<Notify>,
}

impl FakeVllm {
    #[allow(clippy::result_large_err)]
    fn ensure_lora_enabled(&self) -> Result<(), Status> {
        if self.lora_disabled.load(Ordering::SeqCst) {
            return Err(Status::failed_precondition(
                "engine was not started with LoRA enabled",
            ));
        }
        Ok(())
    }

    async fn record_control(&self, name: &str, body: serde_json::Value) {
        self.control_calls
            .lock()
            .await
            .push((name.to_string(), body));
    }
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
        // Upstream resolves the adapter before generating; reproduce that second
        // safety layer so the sidecar's own guard is not the only thing tested.
        if !request.lora_name.is_empty() {
            if self.lora_disabled.load(Ordering::SeqCst) {
                return Err(Status::failed_precondition(
                    "engine was not started with LoRA enabled",
                ));
            }
            if !self
                .loras
                .lock()
                .await
                .iter()
                .any(|adapter| adapter.lora_name == request.lora_name)
            {
                return Err(Status::not_found(format!(
                    "LoRA adapter `{}` is not loaded",
                    request.lora_name
                )));
            }
        }

        let prompt_tokens = match request.prompt.as_ref() {
            Some(pb::generate_request::Prompt::TokenIds(ids)) => ids.ids.len() as u32,
            Some(pb::generate_request::Prompt::Text(text)) => {
                text.split_whitespace().count() as u32
            }
            None => return Err(Status::invalid_argument("prompt required")),
        };
        let prompt_tokens = if request.media.is_empty() {
            prompt_tokens
        } else {
            601
        };
        let wants_logprobs = request
            .response
            .as_ref()
            .is_some_and(|response| response.output_logprobs);
        let wants_prompt_token_ids = request
            .response
            .as_ref()
            .is_some_and(|response| response.prompt_token_ids);
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
            let prompt_info = pb::PromptInfo {
                num_prompt_tokens: prompt_tokens,
                token_ids: if wants_prompt_token_ids {
                    (0..prompt_tokens).collect()
                } else {
                    Vec::new()
                },
                logprobs: if wants_prompt_logprobs {
                    vec![-0.2; prompt_tokens as usize]
                } else {
                    Vec::new()
                },
                ranks: if wants_prompt_logprobs {
                    vec![1; prompt_tokens as usize]
                } else {
                    Vec::new()
                },
                candidate_tokens: if wants_prompt_logprobs {
                    vec![pb::CandidateTokenInfo::default(); prompt_tokens as usize]
                } else {
                    Vec::new()
                },
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
        self.ensure_lora_enabled()?;
        if self.hold_load.load(Ordering::SeqCst) {
            self.load_pending.store(true, Ordering::SeqCst);
            self.release_load.notified().await;
            self.load_pending.store(false, Ordering::SeqCst);
        }
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
        self.ensure_lora_enabled()?;
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
        self.ensure_lora_enabled()?;
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

    async fn pause_generation(
        &self,
        request: Request<pb::PauseGenerationRequest>,
    ) -> Result<Response<pb::PauseGenerationResponse>, Status> {
        let request = request.into_inner();
        self.record_control(
            "pause_generation",
            json!({"mode": request.mode, "clear_cache": request.clear_cache}),
        )
        .await;
        self.paused.store(true, Ordering::SeqCst);
        Ok(Response::new(pb::PauseGenerationResponse {}))
    }

    async fn resume_generation(
        &self,
        _request: Request<pb::ResumeGenerationRequest>,
    ) -> Result<Response<pb::ResumeGenerationResponse>, Status> {
        self.record_control("resume_generation", json!({})).await;
        self.paused.store(false, Ordering::SeqCst);
        Ok(Response::new(pb::ResumeGenerationResponse {}))
    }

    async fn is_paused(
        &self,
        _request: Request<pb::IsPausedRequest>,
    ) -> Result<Response<pb::IsPausedResponse>, Status> {
        Ok(Response::new(pb::IsPausedResponse {
            paused: self.paused.load(Ordering::SeqCst),
        }))
    }

    async fn sleep(
        &self,
        request: Request<pb::SleepRequest>,
    ) -> Result<Response<pb::SleepResponse>, Status> {
        let request = request.into_inner();
        self.record_control(
            "sleep",
            json!({"level": request.level, "mode": request.mode}),
        )
        .await;
        let mut sleeping_tags = self.sleeping_tags.lock().await;
        *sleeping_tags = if request.level == Some(0) {
            BTreeSet::from(["scheduling".to_string()])
        } else {
            BTreeSet::from(["kv_cache".to_string(), "weights".to_string()])
        };
        Ok(Response::new(pb::SleepResponse {}))
    }

    async fn wake_up(
        &self,
        request: Request<pb::WakeUpRequest>,
    ) -> Result<Response<pb::WakeUpResponse>, Status> {
        let tags = request.into_inner().tags;
        self.record_control("wake_up", json!({"tags": tags.clone()}))
            .await;
        let mut sleeping_tags = self.sleeping_tags.lock().await;
        if tags.is_empty() {
            sleeping_tags.clear();
        } else {
            for tag in tags {
                sleeping_tags.remove(&tag);
            }
        }
        Ok(Response::new(pb::WakeUpResponse {}))
    }

    async fn is_sleeping(
        &self,
        _request: Request<pb::IsSleepingRequest>,
    ) -> Result<Response<pb::IsSleepingResponse>, Status> {
        Ok(Response::new(pb::IsSleepingResponse {
            sleeping: !self.sleeping_tags.lock().await.is_empty(),
        }))
    }

    async fn init_weight_transfer_engine(
        &self,
        request: Request<pb::InitWeightTransferEngineRequest>,
    ) -> Result<Response<pb::InitWeightTransferEngineResponse>, Status> {
        let body = serde_json::from_slice(&request.into_inner().init_info_json)
            .map_err(|error| Status::invalid_argument(error.to_string()))?;
        self.record_control("init_weight_transfer_engine", body)
            .await;
        Ok(Response::new(pb::InitWeightTransferEngineResponse {}))
    }

    async fn start_weight_update(
        &self,
        _request: Request<pb::StartWeightUpdateRequest>,
    ) -> Result<Response<pb::StartWeightUpdateResponse>, Status> {
        self.record_control("start_weight_update", json!({})).await;
        Ok(Response::new(pb::StartWeightUpdateResponse {}))
    }

    async fn start_draft_weight_update(
        &self,
        _request: Request<pb::StartDraftWeightUpdateRequest>,
    ) -> Result<Response<pb::StartDraftWeightUpdateResponse>, Status> {
        self.record_control("start_draft_weight_update", json!({}))
            .await;
        Ok(Response::new(pb::StartDraftWeightUpdateResponse {}))
    }

    async fn update_weights(
        &self,
        request: Request<pb::UpdateWeightsRequest>,
    ) -> Result<Response<pb::UpdateWeightsResponse>, Status> {
        let body = serde_json::from_slice(&request.into_inner().update_info_json)
            .map_err(|error| Status::invalid_argument(error.to_string()))?;
        self.record_control("update_weights", body).await;
        Ok(Response::new(pb::UpdateWeightsResponse {}))
    }

    async fn finish_weight_update(
        &self,
        request: Request<pb::FinishWeightUpdateRequest>,
    ) -> Result<Response<pb::FinishWeightUpdateResponse>, Status> {
        let version = request.into_inner().weight_version;
        if let Some(version) = &version {
            self.weight_version.lock().await.clone_from(version);
        }
        self.record_control("finish_weight_update", json!({"weight_version": version}))
            .await;
        Ok(Response::new(pb::FinishWeightUpdateResponse {}))
    }

    async fn update_weight_version(
        &self,
        request: Request<pb::UpdateWeightVersionRequest>,
    ) -> Result<Response<pb::UpdateWeightVersionResponse>, Status> {
        let version = request.into_inner().weight_version;
        self.weight_version.lock().await.clone_from(&version);
        self.record_control("update_weight_version", json!({"weight_version": version}))
            .await;
        Ok(Response::new(pb::UpdateWeightVersionResponse {}))
    }

    async fn get_weight_version(
        &self,
        _request: Request<pb::GetWeightVersionRequest>,
    ) -> Result<Response<pb::GetWeightVersionResponse>, Status> {
        Ok(Response::new(pb::GetWeightVersionResponse {
            weight_version: self.weight_version.lock().await.clone(),
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
            world_size: 1,
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
        rl_capabilities: Some(pb::RlCapabilities {
            weight_transfer_enabled: true,
            weight_transfer_backend: "nccl".to_string(),
            sleep_mode_enabled: true,
            draft_weight_updates_enabled: true,
        }),
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

fn engine(
    endpoint: &str,
    mode: DisaggregationMode,
    connections: usize,
    model: pb::ModelInfo,
) -> VllmSidecarEngine {
    engine_with_server_info(endpoint, mode, connections, model, server_info())
}

fn engine_with_server_info(
    endpoint: &str,
    mode: DisaggregationMode,
    connections: usize,
    model: pb::ModelInfo,
    server: pb::ServerInfo,
) -> VllmSidecarEngine {
    let transport = GrpcTransportConfig {
        connections: NonZeroUsize::new(connections).expect("non-zero connection count"),
        ..Default::default()
    };
    VllmSidecarEngine::new(
        GrpcEndpoint::parse(endpoint, "--grpc-endpoint").expect("valid test endpoint"),
        DiscoveredModel::from_proto(model, server).expect("valid discovery"),
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
        "--grpc-endpoint".to_string(),
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
async fn rl_engine_routes_preserve_lifecycle_payloads_and_version() {
    let server = FakeServer::start(FakeVllm::default()).await;
    let engine = engine(
        &server.endpoint,
        DisaggregationMode::Aggregated,
        1,
        model_info(),
    );
    engine.start(0).await.expect("start");

    assert_eq!(
        engine
            .supported_controls()
            .await
            .unwrap()
            .into_iter()
            .collect::<BTreeSet<_>>(),
        [
            "get_weight_version",
            "is_paused",
            "is_sleeping",
            "pause_generation",
            "resume_generation",
            "sleep",
            "wake_up",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    );
    assert_eq!(
        engine
            .supported_updates()
            .await
            .unwrap()
            .into_iter()
            .collect::<BTreeSet<_>>(),
        [
            "finish_weight_update",
            "init_weight_transfer_engine",
            "start_draft_weight_update",
            "start_weight_update",
            "update_weight_version",
            "update_weights",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    );

    // Regression: unsupported sleep levels or wake tags can enter vLLM's
    // destructive/partial sleep paths while still returning gRPC success.
    for (control, body, expected) in [
        ("sleep", json!({"level": 3}), "one of 0, 1, or 2"),
        (
            "wake_up",
            json!({"tags": ["unknown"]}),
            "weights, kv_cache, or scheduling",
        ),
    ] {
        let error = engine
            .engine_control(control.to_string(), body)
            .await
            .expect_err("unsupported lifecycle value must fail before gRPC");
        assert!(error.to_string().contains(expected), "unexpected {error}");
    }

    for (control, body, expected) in [
        ("is_paused", json!({}), json!({"is_paused": false})),
        (
            "pause_generation",
            json!({"mode": "keep", "clear_cache": false}),
            json!({"status": "paused"}),
        ),
        ("is_paused", json!({}), json!({"is_paused": true})),
        ("resume_generation", json!({}), json!({"status": "resumed"})),
        ("is_paused", json!({}), json!({"is_paused": false})),
        ("is_sleeping", json!({}), json!({"is_sleeping": false})),
        (
            "sleep",
            json!({"level": 2, "mode": "wait"}),
            json!({"status": "sleeping"}),
        ),
        ("is_sleeping", json!({}), json!({"is_sleeping": true})),
        (
            "wake_up",
            json!({"tags": ["weights"]}),
            json!({"status": "partially_awake", "is_sleeping": true}),
        ),
        ("is_sleeping", json!({}), json!({"is_sleeping": true})),
    ] {
        assert_eq!(
            engine
                .engine_control(control.to_string(), body)
                .await
                .unwrap(),
            expected,
            "unexpected {control} response"
        );
    }

    for (update, body, expected) in [
        (
            "init_weight_transfer_engine",
            json!({"init_info": {"master_addr": "trainer", "master_port": 1234}}),
            json!({"message": "Weight transfer initialized"}),
        ),
        (
            "start_weight_update",
            json!({}),
            json!({"message": "Weight update started"}),
        ),
        (
            "start_draft_weight_update",
            json!({}),
            json!({"message": "Draft weight update started"}),
        ),
        (
            "update_weights",
            json!({"update_info": {"names": ["layer.weight"], "shape": [4, 8]}}),
            json!({"message": "Weights updated"}),
        ),
        (
            "finish_weight_update",
            json!({"weight_version": "step-42"}),
            json!({"message": "Weight update finished"}),
        ),
    ] {
        assert_eq!(
            engine
                .engine_update(update.to_string(), body)
                .await
                .unwrap(),
            expected,
            "unexpected {update} response"
        );
    }
    assert_eq!(
        engine
            .engine_control("get_weight_version".to_string(), json!({}))
            .await
            .unwrap(),
        json!({"weight_version": "step-42"})
    );
    assert_eq!(
        engine
            .engine_update(
                "update_weight_version".to_string(),
                json!({"new_version": "step-43"}),
            )
            .await
            .unwrap(),
        json!({"success": true, "new_version": "step-43"})
    );
    assert_eq!(
        engine
            .engine_control("get_weight_version".to_string(), json!({}))
            .await
            .unwrap(),
        json!({"weight_version": "step-43"})
    );

    let calls = server.service.control_calls.lock().await;
    let actual = calls.iter().cloned().collect::<BTreeMap<_, _>>();
    let expected = BTreeMap::from([
        (
            "pause_generation".to_string(),
            json!({"mode": pb::PauseMode::Keep as i32, "clear_cache": false}),
        ),
        ("resume_generation".to_string(), json!({})),
        (
            "sleep".to_string(),
            json!({"level": 2, "mode": pb::PauseMode::Wait as i32}),
        ),
        ("wake_up".to_string(), json!({"tags": ["weights"]})),
        (
            "init_weight_transfer_engine".to_string(),
            json!({"master_addr": "trainer", "master_port": 1234}),
        ),
        ("start_weight_update".to_string(), json!({})),
        ("start_draft_weight_update".to_string(), json!({})),
        (
            "update_weights".to_string(),
            json!({"names": ["layer.weight"], "shape": [4, 8]}),
        ),
        (
            "finish_weight_update".to_string(),
            json!({"weight_version": "step-42"}),
        ),
        (
            "update_weight_version".to_string(),
            json!({"weight_version": "step-43"}),
        ),
    ]);
    assert_eq!(calls.len(), expected.len(), "each mutating RPC runs once");
    assert_eq!(actual, expected);
}

/// Regression: vLLM exposes sleep status independently of CUDA sleep-mode
/// allocation support, so capability discovery must not hide the status RPC
/// when only the mutating sleep/wake operations are disabled.
#[tokio::test]
async fn sleep_status_remains_advertised_without_sleep_mode() {
    let mut server = server_info();
    server
        .rl_capabilities
        .as_mut()
        .expect("RL capabilities")
        .sleep_mode_enabled = false;
    let engine = engine_with_server_info(
        "http://127.0.0.1:1",
        DisaggregationMode::Aggregated,
        1,
        model_info(),
        server,
    );

    let controls = engine
        .supported_controls()
        .await
        .unwrap()
        .into_iter()
        .collect::<BTreeSet<_>>();
    assert!(controls.contains("is_sleeping"));
    assert!(!controls.contains("sleep"));
    assert!(!controls.contains("wake_up"));
}

#[tokio::test]
async fn multimodal_image_is_forwarded_with_uuid() {
    let service = FakeVllm::default();
    let mut discovered = model_info();
    discovered.supports_multimodal = true;
    *service.model_info_override.lock().await = Some(discovered.clone());
    let server = FakeServer::start(service).await;
    let (aggregate, _) = engine_from_args(&server.endpoint).await;
    aggregate.start(0).await.expect("start");

    let mut image_request = request();
    image_request.multi_modal_data = Some(std::collections::HashMap::from([(
        "image_url".to_string(),
        vec![MultimodalData::RawUrl(
            "data:image/png;base64,iVBORw0KGgo=".to_string(),
        )],
    )]));
    image_request.output_options.prompt_logprobs = None;
    image_request
        .extra_args
        .as_mut()
        .and_then(serde_json::Value::as_object_mut)
        .expect("object extra_args")
        .extend([
            (
                "messages".to_string(),
                json!([{"role": "user", "content": [{"type": "image_url"}]}]),
            ),
            ("formatted_prompt".to_string(), json!("<image>\nDescribe.")),
            ("mm_hashes".to_string(), json!(["0123456789abcdef"])),
        ]);

    let outputs = collect(&aggregate, image_request.clone()).await;
    assert_eq!(outputs[0].finish_reason, Some(FinishReason::Stop));
    assert_eq!(
        outputs[0]
            .completion_usage
            .as_ref()
            .expect("usage")
            .prompt_tokens,
        601
    );

    let requests = server.service.requests.lock().await;
    let media = &requests.last().expect("recorded request").media;
    assert_eq!(media.len(), 1);
    assert_eq!(media[0].modality(), pb::Modality::Image);
    assert_eq!(
        media[0].uuid,
        "0123456789abcdef000000000000000000000000000000000000000000000000"
    );
    assert!(matches!(
        media[0].source.as_ref(),
        Some(pb::media_item::Source::DataUri(_))
    ));
    drop(requests);

    let prefill = engine(
        &server.endpoint,
        DisaggregationMode::Prefill,
        1,
        discovered.clone(),
    );
    let decode = engine(&server.endpoint, DisaggregationMode::Decode, 1, discovered);
    prefill.start(1).await.expect("start prefill");
    decode.start(2).await.expect("start decode");

    let prefill_outputs = collect(&prefill, image_request.clone()).await;
    let handoff = prefill_outputs[0]
        .disaggregated_params
        .clone()
        .expect("multimodal handoff");
    assert_eq!(
        handoff["_dynamo_sidecar_multimodal_prompt_token_ids"]
            .as_array()
            .expect("expanded prompt token IDs")
            .len(),
        601
    );

    let mut decode_request = image_request;
    decode_request.prefill_result = Some(PrefillResult {
        disaggregated_params: handoff,
        prompt_tokens_details: None,
    });
    let decode_outputs = collect(&decode, decode_request).await;
    assert_eq!(
        decode_outputs[0]
            .completion_usage
            .as_ref()
            .expect("decode usage")
            .prompt_tokens,
        601
    );

    let requests = server.service.requests.lock().await;
    let prefill_wire = &requests[requests.len() - 2];
    let decode_wire = &requests[requests.len() - 1];
    assert_eq!(prefill_wire.media.len(), 1);
    assert!(
        prefill_wire
            .response
            .as_ref()
            .expect("prefill response options")
            .prompt_token_ids
    );
    assert!(decode_wire.media.is_empty());
    assert_eq!(
        decode_wire.prompt.as_ref(),
        Some(&pb::generate_request::Prompt::TokenIds(pb::TokenIds {
            ids: (0..601).collect(),
        }))
    );
    let decode_kv = struct_to_json(
        decode_wire
            .kv
            .as_ref()
            .and_then(|kv| kv.kv_transfer_params.clone())
            .expect("decode KV handoff"),
    )
    .expect("decode KV JSON");
    assert!(
        decode_kv["_dynamo_sidecar_multimodal_prompt_token_ids"].is_null(),
        "sidecar metadata must not reach vLLM"
    );
}

// ======================================================================================
// LoRA lifecycle
// ======================================================================================

/// Build an engine with the LoRA surface enabled, bypassing the process-global
/// `DYN_LORA_ENABLED` so these tests stay independent of ambient environment.
fn lora_engine(endpoint: &str) -> VllmSidecarEngine {
    engine(endpoint, DisaggregationMode::Aggregated, 1, model_info()).with_lora_enabled(true)
}

/// A directory that passes the adapter-layout validation.
fn adapter_dir() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("adapter tempdir");
    std::fs::write(dir.path().join("adapter_config.json"), "{}").unwrap();
    std::fs::write(dir.path().join("adapter_model.safetensors"), []).unwrap();
    dir
}

fn load_body(name: &str, dir: &tempfile::TempDir) -> serde_json::Value {
    json!({
        "lora_name": name,
        "source": {"uri": format!("file://{}", dir.path().display())},
    })
}

async fn load(
    engine: &VllmSidecarEngine,
    name: &str,
    dir: &tempfile::TempDir,
) -> serde_json::Value {
    engine
        .engine_update("load_lora".to_string(), load_body(name, dir))
        .await
        .expect("load_lora envelope")
}

async fn unload(engine: &VllmSidecarEngine, name: &str) -> serde_json::Value {
    engine
        .engine_update("unload_lora".to_string(), json!({"lora_name": name}))
        .await
        .expect("unload_lora envelope")
}

async fn lora_siblings(endpoint: &dynamo_runtime::component::Endpoint) -> Vec<String> {
    let endpoint_id = endpoint.id();
    endpoint
        .drt()
        .discovery()
        .list(DiscoveryQuery::EndpointModels {
            namespace: endpoint_id.namespace.clone(),
            component: endpoint_id.component.clone(),
            endpoint: endpoint_id.name.clone(),
        })
        .await
        .unwrap()
        .into_iter()
        .filter_map(|instance| match instance {
            DiscoveryInstance::Model {
                model_suffix: Some(suffix),
                ..
            } => Some(suffix),
            _ => None,
        })
        .collect()
}

/// Start an engine that is ready to serve lifecycle calls against `service`.
async fn started_lora_engine(
    service: FakeVllm,
    namespace: &str,
) -> (
    FakeServer,
    VllmSidecarEngine,
    dynamo_runtime::component::Endpoint,
) {
    let server = FakeServer::start(service).await;
    let engine = lora_engine(&server.endpoint);
    engine.start(0).await.expect("start");
    let endpoint = runtime_endpoint(namespace).await;
    engine
        .on_endpoint_ready(endpoint.clone())
        .await
        .expect("endpoint ready");
    (server, engine, endpoint)
}

// --- protocol and compatibility ------------------------------------------------------

#[test]
fn vendored_protos_match_the_merged_vllm_release() {
    // Pinned to vllm-project/vllm@1f9444a34ff4ebfba4d65c68971bb5306a11aa92
    // (vllm-project/vllm#52840). Update `proto/README.md` and these digests
    // together when resyncing; a mismatch means the vendored copy drifted.
    for (path, expected) in [
        (
            "proto/control.proto",
            "1a050496e7d0f919f398d150d4bff1660d5a5eac57951137aeb0ca5970436696",
        ),
        (
            "proto/inference.proto",
            "078a3d2a94bd03a96fdfdfa31c13a805d00575b365dec5b3f8ed82d36f065e85",
        ),
    ] {
        let full = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(path);
        let bytes = std::fs::read(&full).unwrap_or_else(|e| panic!("read {path}: {e}"));
        let digest = <sha2::Sha256 as sha2::Digest>::digest(&bytes);
        assert_eq!(
            format!("{digest:x}"),
            expected,
            "{path} drifted from the vendored vLLM revision"
        );
    }
}

#[test]
fn lora_wire_fields_keep_their_upstream_numbers() {
    // Field numbers are the wire contract. Renumbering them silently breaks
    // interoperability with a vLLM built from the pinned revision, so encode a
    // known value and assert the tag bytes rather than trusting the generated code.
    use prost::Message;

    let request = pb::GenerateRequest {
        lora_name: "math-r8".to_string(),
        ..Default::default()
    };
    // field 15, wire type 2 -> tag byte 0x7a
    assert!(
        request.encode_to_vec().starts_with(&[0x7a, 0x07]),
        "GenerateRequest.lora_name must stay field 15"
    );

    let adapter = pb::LoraAdapter {
        lora_id: 1,
        lora_name: "n".to_string(),
        source_path: "p".to_string(),
    };
    assert_eq!(
        adapter.encode_to_vec(),
        vec![0x08, 0x01, 0x12, 0x01, b'n', 0x1a, 0x01, b'p'],
        "LoraAdapter fields must stay 1/2/3"
    );

    let server = pb::ServerInfo {
        max_loras: 4,
        ..Default::default()
    };
    // field 10, wire type 0 -> tag byte 0x50
    assert_eq!(
        server.encode_to_vec(),
        vec![0x50, 0x04],
        "ServerInfo.max_loras must stay field 10"
    );

    let model = pb::ModelInfo {
        supports_lora: true,
        ..Default::default()
    };
    // field 22, wire type 0 -> tag bytes 0xb0 0x01
    assert_eq!(
        model.encode_to_vec(),
        vec![0xb0, 0x01, 0x01],
        "ModelInfo.supports_lora must stay field 22"
    );
}

#[test]
fn lora_statuses_map_to_stable_dynamo_meanings() {
    use crate::client::LoraRpcError;
    use dynamo_backend_common::{BackendError, ErrorType};

    let error = |code| LoraRpcError {
        rpc: "LoadLora",
        code,
        message: "boom".to_string(),
    };

    // vLLM answered definitively, so its state is known and no reconciliation is needed.
    for code in [
        tonic::Code::InvalidArgument,
        tonic::Code::AlreadyExists,
        tonic::Code::NotFound,
        tonic::Code::FailedPrecondition,
    ] {
        assert!(error(code).is_definitive(), "{code:?} must be definitive");
    }

    // The operation may have committed before failing; the inventory has to decide.
    for code in [
        tonic::Code::Internal,
        tonic::Code::Unknown,
        tonic::Code::DeadlineExceeded,
        tonic::Code::Unavailable,
        tonic::Code::Aborted,
    ] {
        assert!(
            !error(code).is_definitive(),
            "{code:?} must trigger reconciliation"
        );
    }

    assert_eq!(
        error(tonic::Code::NotFound).into_dynamo().error_type(),
        ErrorType::Backend(BackendError::InvalidArgument)
    );
    assert_eq!(
        error(tonic::Code::DeadlineExceeded)
            .into_dynamo()
            .error_type(),
        ErrorType::Backend(BackendError::ConnectionTimeout)
    );
    assert_eq!(
        error(tonic::Code::Internal).into_dynamo().error_type(),
        ErrorType::Backend(BackendError::Unknown)
    );
}

#[test]
fn inventory_validation_rejects_unusable_entries_and_sorts_by_name() {
    let adapter = |id: i64, name: &str, path: &str| pb::LoraAdapter {
        lora_id: id,
        lora_name: name.to_string(),
        source_path: path.to_string(),
    };

    let sorted =
        crate::lora::validate_inventory(vec![adapter(2, "zeta", "/z"), adapter(1, "alpha", "/a")])
            .expect("valid inventory");
    assert_eq!(
        sorted
            .iter()
            .map(|a| a.lora_name.as_str())
            .collect::<Vec<_>>(),
        ["alpha", "zeta"]
    );

    for (case, inventory) in [
        ("empty name", vec![adapter(1, "", "/a")]),
        ("non-positive id", vec![adapter(0, "alpha", "/a")]),
        ("missing path", vec![adapter(1, "alpha", "")]),
        (
            "duplicate name",
            vec![adapter(1, "alpha", "/a"), adapter(2, "alpha", "/b")],
        ),
        (
            "duplicate id",
            vec![adapter(1, "alpha", "/a"), adapter(1, "beta", "/b")],
        ),
    ] {
        assert!(
            crate::lora::validate_inventory(inventory).is_err(),
            "{case} must be rejected"
        );
    }
}

#[tokio::test]
async fn lora_surface_requires_flag_capability_and_capacity() {
    let server = FakeServer::start(FakeVllm::default()).await;

    // The operator flag is off, even though vLLM advertises support and capacity.
    let flag_off = engine(
        &server.endpoint,
        DisaggregationMode::Aggregated,
        1,
        model_info(),
    )
    .with_lora_enabled(false);
    assert!(
        !flag_off
            .supported_updates()
            .await
            .unwrap()
            .iter()
            .any(|u| u == "load_lora")
    );

    // vLLM does not advertise adapter support.
    let mut no_support = model_info();
    no_support.supports_lora = false;
    let unsupported = engine(
        &server.endpoint,
        DisaggregationMode::Aggregated,
        1,
        no_support,
    )
    .with_lora_enabled(true);
    assert!(
        !unsupported
            .supported_updates()
            .await
            .unwrap()
            .iter()
            .any(|u| u == "load_lora")
    );

    // Support is advertised but there is no GPU capacity for an adapter.
    let mut no_capacity = server_info();
    no_capacity.max_loras = 0;
    let capacity_zero = engine_with_server_info(
        &server.endpoint,
        DisaggregationMode::Aggregated,
        1,
        model_info(),
        no_capacity,
    )
    .with_lora_enabled(true);
    assert!(
        !capacity_zero
            .supported_updates()
            .await
            .unwrap()
            .iter()
            .any(|u| u == "load_lora")
    );

    // All three hold.
    let enabled = lora_engine(&server.endpoint);
    let updates = enabled.supported_updates().await.unwrap();
    for expected in ["load_lora", "unload_lora", "list_loras"] {
        assert!(
            updates.contains(&expected.to_string()),
            "missing {expected}"
        );
    }
}

#[tokio::test]
async fn lora_and_rl_update_surfaces_coexist() {
    let server = FakeServer::start(FakeVllm::default()).await;
    let engine = lora_engine(&server.endpoint);
    let updates = engine.supported_updates().await.unwrap();
    assert!(updates.contains(&"load_lora".to_string()));
    assert!(updates.contains(&"update_weight_version".to_string()));
}

#[tokio::test]
async fn base_model_advertises_lora_capacity() {
    let server = FakeServer::start(FakeVllm::default()).await;
    let engine = lora_engine(&server.endpoint);
    let config = engine.start(0).await.expect("start");
    assert_eq!(config.llm.unwrap().max_gpu_lora_count, Some(4));
}

// --- lifecycle semantics -------------------------------------------------------------

#[tokio::test]
async fn first_load_publishes_a_discovery_sibling_with_the_server_assigned_id() {
    let (server, engine, endpoint) =
        started_lora_engine(FakeVllm::default(), "lora_first_load").await;
    engine.supported_updates().await.unwrap();
    let dir = adapter_dir();

    let response = load(&engine, "math-r8", &dir).await;

    assert_eq!(response["status"], "success");
    assert_eq!(response["lora_name"], "math-r8");
    assert_eq!(response["hot_swap"], false);
    // The ID comes from vLLM, never from a Dynamo-side derivation of the name.
    let assigned = server.service.loras.lock().await[0].lora_id;
    assert_eq!(response["lora_id"], assigned);

    let card = endpoint
        .drt()
        .discovery()
        .list(DiscoveryQuery::EndpointModels {
            namespace: "lora_first_load".to_string(),
            component: "backend".to_string(),
            endpoint: "generate".to_string(),
        })
        .await
        .unwrap()
        .into_iter()
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
    assert_eq!(card.name(), "math-r8");
    assert_eq!(card.lora.as_ref().unwrap().max_gpu_lora_count, Some(4));
    assert_eq!(card.user_data.as_ref().unwrap()["lora_adapter"], true);
    assert_eq!(card.user_data.as_ref().unwrap()["lora_id"], assigned);
}

#[tokio::test]
async fn lora_sibling_preserves_base_topology() {
    let (_server, engine, endpoint) =
        started_lora_engine(FakeVllm::default(), "lora_topology").await;
    engine.supported_updates().await.unwrap();
    let dir = adapter_dir();
    load(&engine, "math-r8", &dir).await;

    let models = endpoint
        .drt()
        .discovery()
        .list(DiscoveryQuery::EndpointModels {
            namespace: "lora_topology".to_string(),
            component: "backend".to_string(),
            endpoint: "generate".to_string(),
        })
        .await
        .unwrap();
    let mut base = None;
    let mut sibling = None;
    for instance in models {
        let is_sibling = matches!(
            &instance,
            DiscoveryInstance::Model {
                model_suffix: Some(_),
                ..
            }
        );
        let card = instance.deserialize_model::<ModelDeploymentCard>().unwrap();
        if is_sibling {
            sibling = Some(card);
        } else {
            base = Some(card);
        }
    }
    let base = base.expect("base card");
    let sibling = sibling.expect("sibling card");

    // Only adapter-specific fields may differ; routing topology must carry over.
    assert_eq!(sibling.model_type, base.model_type);
    assert_eq!(sibling.model_input, base.model_input);
    assert_eq!(sibling.worker_type, base.worker_type);
    assert_eq!(sibling.needs, base.needs);
    assert_eq!(sibling.kv_cache_block_size, base.kv_cache_block_size);
    assert_eq!(sibling.runtime_config, base.runtime_config);
    assert_eq!(sibling.migration_limit, base.migration_limit);
    // Adapter-specific.
    assert_eq!(sibling.name(), "math-r8");
    assert_eq!(sibling.source_path, Some(base.name().to_string()));
    assert!(sibling.aliases.is_empty());
    assert!(sibling.lora.is_some());
    assert!(base.lora.is_none());
}

#[tokio::test]
async fn repeated_load_returns_the_existing_id_even_for_a_different_uri() {
    let (server, engine, _endpoint) =
        started_lora_engine(FakeVllm::default(), "lora_idempotent").await;
    engine.supported_updates().await.unwrap();
    let first_dir = adapter_dir();
    let first = load(&engine, "math-r8", &first_dir).await;

    let same = load(&engine, "math-r8", &first_dir).await;
    // A different URI is still idempotent while hot swap is unsupported.
    let other_dir = adapter_dir();
    let different_uri = load(&engine, "math-r8", &other_dir).await;

    for response in [&same, &different_uri] {
        assert_eq!(response["status"], "success");
        assert_eq!(response["lora_id"], first["lora_id"]);
        assert_eq!(response["hot_swap"], false);
    }
    assert_eq!(server.service.loras.lock().await.len(), 1);
}

#[tokio::test]
async fn load_rejects_names_that_collide_with_the_base_model_or_reserved_suffixes() {
    let (_server, engine, _endpoint) = started_lora_engine(FakeVllm::default(), "lora_names").await;
    engine.supported_updates().await.unwrap();
    let dir = adapter_dir();

    for name in ["model-source", "served-model", "model-alias"] {
        let response = load(&engine, name, &dir).await;
        assert_eq!(
            response["status"], "error",
            "name `{name}` must be rejected"
        );
    }

    // Only an exact `_base` slug collides with the base-sibling sentinel, and `Slug`
    // trims leading underscores, so `_base` derives `base` rather than the sentinel.
    // Both of these are legitimate distinct keys.
    for name in ["anything_base", "_base"] {
        assert_eq!(
            load(&engine, name, &dir).await["status"],
            "success",
            "name `{name}` derives its own key and must be accepted"
        );
    }

    // `_BASE` also derives `base`, so it now collides with the adapter just loaded.
    let response = load(&engine, "_BASE", &dir).await;
    assert_eq!(response["status"], "error");
    assert!(
        response["message"]
            .as_str()
            .unwrap()
            .contains("discovery suffix"),
        "{response}"
    );
}

#[tokio::test]
async fn load_rejects_a_name_that_collides_with_a_loaded_adapter_discovery_suffix() {
    // `Slug::slugify` lowercases, so these names all derive the suffix `math-r8`.
    // Publishing a second one would overwrite the first adapter's sibling, and
    // unloading either would remove the wrong record.
    let (server, engine, endpoint) =
        started_lora_engine(FakeVllm::default(), "lora_suffix_collision").await;
    engine.supported_updates().await.unwrap();
    let dir = adapter_dir();
    assert_eq!(load(&engine, "math-r8", &dir).await["status"], "success");

    for colliding in ["Math-R8", "MATH-R8"] {
        let response = load(&engine, colliding, &dir).await;
        assert_eq!(
            response["status"], "error",
            "`{colliding}` collides with `math-r8` in discovery"
        );
        let message = response["message"].as_str().unwrap();
        assert!(message.contains("discovery suffix"), "{message}");
    }

    // The original adapter is untouched: still loaded, still the only sibling.
    assert_eq!(server.service.loras.lock().await.len(), 1);
    assert_eq!(lora_siblings(&endpoint).await.len(), 1);
    assert_eq!(
        collect(&engine, request_selecting("math-r8")).await.len(),
        1
    );
}

#[tokio::test]
async fn reconciliation_rejects_an_inventory_whose_names_share_a_discovery_suffix() {
    // vLLM keys adapters by raw name and would accept both; Dynamo cannot publish them
    // as distinct siblings, so reconciliation must refuse rather than silently collapse
    // two adapters onto one discovery record.
    let service = FakeVllm::default();
    {
        let mut loras = service.loras.lock().await;
        loras.push(pb::LoraAdapter {
            lora_id: 1,
            lora_name: "math-r8".to_string(),
            source_path: "/shared/loras/a".to_string(),
        });
        loras.push(pb::LoraAdapter {
            lora_id: 2,
            lora_name: "Math-R8".to_string(),
            source_path: "/shared/loras/b".to_string(),
        });
    }
    let (_server, engine, endpoint) = started_lora_engine(service, "lora_suffix_inventory").await;

    // Best effort: the base model still serves.
    engine
        .supported_updates()
        .await
        .expect("startup must survive");
    assert!(lora_siblings(&endpoint).await.is_empty());

    let listed = engine
        .engine_update("list_loras".to_string(), json!({}))
        .await
        .unwrap();
    assert_eq!(listed["status"], "error");
    assert!(
        listed["message"]
            .as_str()
            .unwrap()
            .contains("discovery suffix"),
        "{listed}"
    );
}

#[tokio::test]
async fn capacity_is_preserved_under_concurrent_loads_of_distinct_adapters() {
    // `server_info()` advertises max_loras = 4.
    let (server, engine, _endpoint) =
        started_lora_engine(FakeVllm::default(), "lora_capacity").await;
    engine.supported_updates().await.unwrap();
    let dirs: Vec<_> = (0..6).map(|_| adapter_dir()).collect();
    let names: Vec<String> = (0..6).map(|index| format!("adapter-{index}")).collect();

    let responses = futures::future::join_all(
        names
            .iter()
            .zip(&dirs)
            .map(|(name, dir)| load(&engine, name, dir)),
    )
    .await;

    let succeeded = responses
        .iter()
        .filter(|response| response["status"] == "success")
        .count();
    assert_eq!(
        succeeded, 4,
        "capacity must cap concurrent loads: {responses:?}"
    );
    assert_eq!(server.service.loras.lock().await.len(), 4);
}

#[tokio::test]
async fn same_name_lifecycle_operations_serialize() {
    let service = FakeVllm::default();
    service.hold_load.store(true, Ordering::SeqCst);
    let (server, engine, _endpoint) = started_lora_engine(service, "lora_serialize").await;
    engine.supported_updates().await.unwrap();
    let dir = adapter_dir();

    let loading = load(&engine, "math-r8", &dir);
    tokio::pin!(loading);
    // Wait until the load is parked inside vLLM.
    loop {
        if server.service.load_pending.load(Ordering::SeqCst) {
            break;
        }
        tokio::task::yield_now().await;
        futures::future::poll_immediate(&mut loading).await;
    }

    // The unload must not observe or mutate state until the load releases the key.
    let unloading = unload(&engine, "math-r8");
    tokio::pin!(unloading);
    assert!(
        futures::future::poll_immediate(&mut unloading)
            .await
            .is_none(),
        "unload must block behind the in-flight load for the same name"
    );

    server.service.hold_load.store(false, Ordering::SeqCst);
    server.service.release_load.notify_waiters();
    let loaded = loading.await;
    assert_eq!(loaded["status"], "success");
    let unloaded = unloading.await;
    assert_eq!(unloaded["status"], "success");
    assert!(server.service.loras.lock().await.is_empty());
}

#[tokio::test]
async fn ambiguous_load_and_unload_outcomes_reconcile_against_list_loras() {
    let service = FakeVllm::default();
    // Both RPCs commit and then fail, so only `ListLoras` reveals the truth.
    service.load_commit_error.store(true, Ordering::SeqCst);
    service.unload_commit_error.store(true, Ordering::SeqCst);
    let (server, engine, endpoint) = started_lora_engine(service, "lora_ambiguous").await;
    engine.supported_updates().await.unwrap();
    let dir = adapter_dir();

    let loaded = load(&engine, "math-r8", &dir).await;
    assert_eq!(
        loaded["status"], "success",
        "committed load must be recognized"
    );
    assert_eq!(server.service.loras.lock().await.len(), 1);
    assert_eq!(lora_siblings(&endpoint).await.len(), 1);

    let unloaded = unload(&engine, "math-r8").await;
    assert_eq!(
        unloaded["status"], "success",
        "committed unload must be recognized"
    );
    assert!(server.service.loras.lock().await.is_empty());
    assert!(lora_siblings(&endpoint).await.is_empty());
}

#[tokio::test]
async fn load_rolls_back_the_native_adapter_when_discovery_publication_fails() {
    let server = FakeServer::start(FakeVllm::default()).await;
    let engine = lora_engine(&server.endpoint);
    engine.start(0).await.expect("start");
    // No `on_endpoint_ready`, so publication cannot succeed.
    let dir = adapter_dir();

    let response = load(&engine, "math-r8", &dir).await;

    assert_eq!(response["status"], "error");
    assert!(
        server.service.loras.lock().await.is_empty(),
        "a load that cannot be published must not leave the adapter resident"
    );
}

#[tokio::test]
async fn unload_restores_discovery_when_the_adapter_survives() {
    let (server, engine, endpoint) =
        started_lora_engine(FakeVllm::default(), "lora_unload_restore").await;
    engine.supported_updates().await.unwrap();
    let dir = adapter_dir();
    load(&engine, "math-r8", &dir).await;
    assert_eq!(lora_siblings(&endpoint).await.len(), 1);

    // vLLM refuses the removal but keeps the adapter loaded.
    server.service.lora_disabled.store(true, Ordering::SeqCst);
    let response = unload(&engine, "math-r8").await;
    server.service.lora_disabled.store(false, Ordering::SeqCst);

    assert_eq!(response["status"], "error");
    assert_eq!(server.service.loras.lock().await.len(), 1);
    assert_eq!(
        lora_siblings(&endpoint).await.len(),
        1,
        "an adapter vLLM still holds must stay routable"
    );
}

#[tokio::test]
async fn unload_reports_available_adapters_when_the_name_is_unknown() {
    let (_server, engine, _endpoint) =
        started_lora_engine(FakeVllm::default(), "lora_unknown_unload").await;
    engine.supported_updates().await.unwrap();
    let dir = adapter_dir();
    load(&engine, "math-r8", &dir).await;

    let response = unload(&engine, "absent").await;

    assert_eq!(response["status"], "error");
    let message = response["message"].as_str().unwrap();
    assert!(message.contains("not found"), "{message}");
    assert!(message.contains("math-r8"), "{message}");
}

#[tokio::test]
async fn hot_swap_is_refused_with_a_clear_message() {
    let server = FakeServer::start(FakeVllm::default()).await;
    let engine = lora_engine(&server.endpoint).with_hot_swap_requested(true);
    engine.start(0).await.expect("start");
    let endpoint = runtime_endpoint("lora_hot_swap").await;
    engine
        .on_endpoint_ready(endpoint)
        .await
        .expect("endpoint ready");
    engine.supported_updates().await.unwrap();
    let dir = adapter_dir();
    load(&engine, "math-r8", &dir).await;

    let response = load(&engine, "math-r8", &dir).await;

    assert_eq!(response["status"], "error");
    let message = response["message"].as_str().unwrap();
    assert!(message.contains("hot swap is not supported"), "{message}");
}

#[tokio::test]
async fn list_loras_returns_a_deterministic_sorted_map() {
    let (_server, engine, _endpoint) = started_lora_engine(FakeVllm::default(), "lora_list").await;
    engine.supported_updates().await.unwrap();
    let dirs: Vec<_> = (0..3).map(|_| adapter_dir()).collect();
    for (name, dir) in ["zeta", "alpha", "mu"].iter().zip(&dirs) {
        load(&engine, name, dir).await;
    }

    let response = engine
        .engine_update("list_loras".to_string(), json!({}))
        .await
        .unwrap();

    assert_eq!(response["status"], "success");
    assert_eq!(response["count"], 3);
    let names: Vec<&str> = response["loras"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(names, ["alpha", "mu", "zeta"]);
}

#[tokio::test]
async fn startup_republishes_loras_loaded_before_sidecar_restart() {
    let service = FakeVllm::default();
    service.loras.lock().await.push(pb::LoraAdapter {
        lora_id: 7,
        lora_name: "math-r8".to_string(),
        source_path: "/shared/loras/math-r8".to_string(),
    });
    let (_server, engine, endpoint) = started_lora_engine(service, "lora_restart").await;

    engine.supported_updates().await.unwrap();

    assert_eq!(lora_siblings(&endpoint).await.len(), 1);
}

#[tokio::test]
async fn startup_reconciliation_failure_leaves_the_base_model_serving() {
    let service = FakeVllm::default();
    // Every lifecycle RPC fails, so reconciliation cannot complete.
    service.lora_disabled.store(true, Ordering::SeqCst);
    let (_server, engine, _endpoint) = started_lora_engine(service, "lora_recon_fail").await;

    // The worker treats an error here as fatal, so reconciliation must not raise one.
    let updates = engine
        .supported_updates()
        .await
        .expect("reconciliation failure must not abort worker startup");

    assert!(updates.contains(&"load_lora".to_string()));
}

#[tokio::test]
async fn shutdown_unpublishes_lora_siblings() {
    let (_server, engine, endpoint) =
        started_lora_engine(FakeVllm::default(), "lora_shutdown").await;
    engine.supported_updates().await.unwrap();
    let dir = adapter_dir();
    load(&engine, "math-r8", &dir).await;
    assert_eq!(lora_siblings(&endpoint).await.len(), 1);

    engine.cleanup().await.expect("cleanup");

    assert!(
        lora_siblings(&endpoint).await.is_empty(),
        "the sidecar publishes siblings itself, so it must remove them itself"
    );
}

// --- inference behavior --------------------------------------------------------------

fn request_selecting(lora_name: &str) -> PreprocessedRequest {
    let mut value = serde_json::to_value(request()).expect("serialize request");
    value["routing"]["lora_name"] = json!(lora_name);
    serde_json::from_value(value).expect("deserialize request")
}

fn generate_context() -> GenerateContext {
    GenerateContext::new(dynamo_backend_common::testing::mock_context(), None)
}

/// `generate` returns a stream, which has no `Debug`, so `expect_err` cannot be used.
async fn generate_error(
    engine: &VllmSidecarEngine,
    lora_name: &str,
) -> dynamo_backend_common::DynamoError {
    match engine
        .generate(request_selecting(lora_name), generate_context())
        .await
    {
        Ok(_) => panic!("generate unexpectedly succeeded for `{lora_name}`"),
        Err(error) => error,
    }
}

#[tokio::test]
async fn a_loaded_adapter_reaches_vllm_as_lora_name() {
    let (server, engine, _endpoint) =
        started_lora_engine(FakeVllm::default(), "lora_generate").await;
    engine.supported_updates().await.unwrap();
    let dir = adapter_dir();
    load(&engine, "math-r8", &dir).await;

    let outputs = collect(&engine, request_selecting("math-r8")).await;

    assert_eq!(outputs.len(), 1);
    let sent = server.service.requests.lock().await;
    assert_eq!(sent.last().unwrap().lora_name, "math-r8");
}

#[tokio::test]
async fn base_model_names_and_aliases_select_the_base_model() {
    let (server, engine, _endpoint) =
        started_lora_engine(FakeVllm::default(), "lora_base_names").await;
    engine.supported_updates().await.unwrap();

    for name in ["model-source", "served-model", "model-alias"] {
        let outputs = collect(&engine, request_selecting(name)).await;
        assert_eq!(outputs.len(), 1, "{name} must serve from the base model");
        assert_eq!(
            server
                .service
                .requests
                .lock()
                .await
                .last()
                .unwrap()
                .lora_name,
            "",
            "{name} must clear lora_name rather than select an adapter"
        );
    }
}

#[tokio::test]
async fn an_unknown_adapter_never_falls_back_to_the_base_model() {
    let (server, engine, _endpoint) =
        started_lora_engine(FakeVllm::default(), "lora_unknown_request").await;
    engine.supported_updates().await.unwrap();

    let error = generate_error(&engine, "absent").await;

    assert!(
        error.to_string().contains("unknown model or LoRA adapter"),
        "{error}"
    );
    assert!(
        server.service.requests.lock().await.is_empty(),
        "the request must never reach vLLM as a base-model generation"
    );
}

#[tokio::test]
async fn an_adapter_vllm_holds_but_dynamo_never_published_is_not_admitted() {
    // Discovery is authoritative for what routers may target. An adapter present in
    // vLLM but absent from discovery is not routable, and admission must say so
    // rather than quietly generating from the base model.
    let service = FakeVllm::default();
    service.loras.lock().await.push(pb::LoraAdapter {
        lora_id: 7,
        lora_name: "math-r8".to_string(),
        source_path: "/shared/loras/math-r8".to_string(),
    });
    let (server, engine, _endpoint) = started_lora_engine(service, "lora_unpublished").await;

    let error = generate_error(&engine, "math-r8").await;
    assert!(
        error.to_string().contains("unknown model or LoRA adapter"),
        "{error}"
    );
    assert!(server.service.requests.lock().await.is_empty());

    // Once reconciliation publishes it, the same request is admitted.
    engine.supported_updates().await.unwrap();
    let outputs = collect(&engine, request_selecting("math-r8")).await;
    assert_eq!(outputs.len(), 1);
    assert_eq!(
        server
            .service
            .requests
            .lock()
            .await
            .last()
            .unwrap()
            .lora_name,
        "math-r8"
    );
}

#[tokio::test]
async fn selecting_an_adapter_without_engine_support_fails_clearly() {
    let mut no_support = model_info();
    no_support.supports_lora = false;
    let service = FakeVllm::default();
    *service.model_info_override.lock().await = Some(no_support.clone());
    let server = FakeServer::start(service).await;
    let engine = engine(
        &server.endpoint,
        DisaggregationMode::Aggregated,
        1,
        no_support,
    );
    engine.start(0).await.expect("start");

    let error = generate_error(&engine, "math-r8").await;

    assert!(
        error.to_string().contains("did not advertise LoRA support"),
        "{error}"
    );
    assert!(server.service.requests.lock().await.is_empty());
}

#[tokio::test]
async fn request_admission_and_unload_cannot_race() {
    let (server, engine, _endpoint) =
        started_lora_engine(FakeVllm::default(), "lora_admission").await;
    engine.supported_updates().await.unwrap();
    let dir = adapter_dir();
    load(&engine, "math-r8", &dir).await;

    // Park the sidecar inside the generation RPC, after admission succeeded.
    server
        .service
        .hang_before_headers
        .store(true, Ordering::SeqCst);
    let mut generating =
        Box::pin(engine.generate(request_selecting("math-r8"), generate_context()));
    loop {
        if server.service.headers_pending.load(Ordering::SeqCst) {
            break;
        }
        tokio::task::yield_now().await;
        futures::future::poll_immediate(&mut generating).await;
    }

    // The unload must wait for the in-flight admission to release the key.
    let unloading = unload(&engine, "math-r8");
    tokio::pin!(unloading);
    assert!(
        futures::future::poll_immediate(&mut unloading)
            .await
            .is_none(),
        "unload must not proceed while a request is still being admitted"
    );

    server
        .service
        .hang_before_headers
        .store(false, Ordering::SeqCst);
    server.service.release_headers.notify_waiters();
    let _stream = generating.await.expect("stream established");
    assert_eq!(unloading.await["status"], "success");
}

#[tokio::test]
async fn grpc_request_errors_are_propagated() {
    let service = FakeVllm::default();
    service.reject.store(true, Ordering::SeqCst);
    let server = FakeServer::start(service).await;
    let engine = engine(
        &server.endpoint,
        DisaggregationMode::Aggregated,
        1,
        model_info(),
    );
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
    let prefill = engine(
        &server.endpoint,
        DisaggregationMode::Prefill,
        1,
        model_info(),
    );
    let decode = engine(
        &server.endpoint,
        DisaggregationMode::Decode,
        1,
        model_info(),
    );
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
            "--grpc-endpoint".to_string(),
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
    let endpoint = GrpcEndpoint::parse(&server.endpoint, "--grpc-endpoint").unwrap();
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
    let engine = engine(
        &server.endpoint,
        DisaggregationMode::Aggregated,
        1,
        model_info(),
    );
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
    let engine = engine(
        &server.endpoint,
        DisaggregationMode::Aggregated,
        1,
        model_info(),
    );
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
    let engine = engine(
        &server.endpoint,
        DisaggregationMode::Decode,
        1,
        model_info(),
    );
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
    let engine = engine(
        &server.endpoint,
        DisaggregationMode::Decode,
        1,
        model_info(),
    );
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
    let engine = engine(
        &server.endpoint,
        DisaggregationMode::Aggregated,
        1,
        model_info(),
    );
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
