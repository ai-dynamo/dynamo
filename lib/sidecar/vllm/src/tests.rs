// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::net::SocketAddr;
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use dynamo_backend_common::engine::RoutingHints;
use dynamo_backend_common::{
    DisaggregationMode, FinishReason, GenerateContext, LLMEngine, MultimodalData, OutputOptions,
    PrefillResult, PreprocessedRequest, RlWorkerMetadata, SamplingOptions, StopConditions,
};
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
}

impl FakeVllm {
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
        rl_capabilities: Some(pb::RlCapabilities {
            weight_transfer_enabled: true,
            weight_transfer_backend: "nccl".to_string(),
            sleep_mode_enabled: true,
            draft_weight_updates_enabled: true,
        }),
    }
}

#[test]
fn discovered_model_advertises_token_native_generate() {
    let model = DiscoveredModel::from_proto(model_info(), server_info()).expect("valid discovery");
    assert_eq!(
        model
            .engine_config()
            .runtime_data
            .get("vllm_inference_v1_generate"),
        Some(&serde_json::Value::Bool(true))
    );
    assert!(
        !model
            .engine_config()
            .runtime_data
            .contains_key("vllm_exact_mm_routing"),
        "feature transport must not advertise exact routing"
    );
}

#[test]
fn discovered_model_reports_rl_worker_metadata() {
    let model = DiscoveredModel::from_proto(model_info(), server_info()).expect("valid discovery");
    assert_eq!(
        model
            .rl_worker_metadata(Some("http://worker:8120".to_string()))
            .expect("valid RL metadata"),
        RlWorkerMetadata::new(
            4,
            Some("nccl".to_string()),
            Some("http://worker:8120".to_string()),
        )
        .expect("valid expected metadata")
    );
}

#[test]
fn startup_compatibility_rejects_tensor_or_pipeline_parallelism_change() {
    let bootstrap = DiscoveredModel::from_proto(model_info(), server_info())
        .expect("valid bootstrap discovery");

    for dimension in ["tensor", "pipeline"] {
        let mut changed_server = server_info();
        let parallelism = changed_server
            .parallelism
            .as_mut()
            .expect("parallelism metadata");
        match dimension {
            "tensor" => parallelism.tensor_parallel_size += 1,
            "pipeline" => parallelism.pipeline_parallel_size += 1,
            _ => unreachable!(),
        }
        let observed = DiscoveredModel::from_proto(model_info(), changed_server)
            .expect("valid startup discovery");

        assert!(
            bootstrap.ensure_startup_compatible(&observed).is_err(),
            "{dimension} parallelism change should be rejected"
        );
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
fn all_logprobs_use_the_proto_all_selector() {
    let mut request = request();
    request.output_options.logprobs = Some(u32::MAX);
    request.output_options.prompt_logprobs = Some(u32::MAX);

    let wire = build_generate_request(
        request,
        "all-logprobs".to_string(),
        DisaggregationMode::Aggregated,
    )
    .expect("build request");
    let response = wire.response.expect("response options");

    assert!(matches!(
        response.output_candidates.and_then(|value| value.select),
        Some(pb::candidate_tokens::Select::All(true))
    ));
    assert!(matches!(
        response.prompt_candidates.and_then(|value| value.select),
        Some(pb::candidate_tokens::Select::All(true))
    ));
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
    request.output_options.skip_special_tokens = Some(false);
    request.routing.as_mut().unwrap().priority = Some(-7);
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
                "stop": ["done"],
                "stop_token_ids": [2],
                "ignore_eos": true,
                "logprobs": 1,
                "prompt_logprobs": 1,
                "cache_salt": "cache-salt",
                "skip_reading_prefix_cache": true,
                "skip_special_tokens": false,
                "include_stop_str_in_output": true,
                "return_token_ids": true
            },
            "model": "served-model",
            "stream": false,
            "cache_salt": "cache-salt",
            "priority": 7,
            "kv_transfer_params": {
                "connector_data": {"values": [1, true, null]}
            }
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
    assert_eq!(wire.stopping.as_ref().unwrap().stop_strings, ["done"]);
    assert!(wire.stopping.as_ref().unwrap().include_stop_strings);
    assert_eq!(
        wire.response.as_ref().unwrap().skip_special_tokens,
        Some(false)
    );
    assert!(wire.response.as_ref().unwrap().output_logprobs);
    assert_eq!(
        wire.kv.as_ref().unwrap().cache_salt,
        "dynamo-cache-salt:cache-salt"
    );
    assert!(wire.kv.as_ref().unwrap().bypass_prefix_cache);
    assert!(wire.kv.as_ref().unwrap().kv_transfer_params.is_some());
    assert_eq!(wire.priority, 7);
}

#[test]
fn token_native_compatibility_envelope_rejects_disabled_token_ids() {
    let mut request = request();
    request.extra_args = Some(json!({
        "vllm_tito": {
            "request_id": "request-1",
            "sampling_params": {"max_tokens": 1, "return_token_ids": false},
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
            "sampling_params": {"max_tokens": 1, "cache_salt": "different-policy-version"},
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
fn token_native_compatibility_envelope_rejects_conflicting_prefix_cache_aliases() {
    let mut request = request();
    request.extra_args = Some(json!({
        "bypass_prefix_cache": true,
        "vllm_tito": {
            "request_id": "request-1",
            "sampling_params": {"skip_reading_prefix_cache": false},
            "stream": false,
            "priority": 0
        }
    }));

    let error = build_generate_request(
        request,
        "request-1".to_string(),
        DisaggregationMode::Aggregated,
    )
    .expect_err("prefix-cache aliases must agree");

    assert!(
        error
            .to_string()
            .contains("conflicts with extra_args.bypass_prefix_cache")
    );
}

#[test]
fn token_native_compatibility_envelope_treats_optional_nulls_as_omitted() {
    let mut request = request();
    request.extra_args = Some(json!({
        "vllm_tito": {
            "request_id": "request-1",
            "sampling_params": {
                "skip_reading_prefix_cache": null,
                "skip_special_tokens": null
            },
            "stream": false,
            "priority": 0
        }
    }));

    build_generate_request(
        request,
        "request-1".to_string(),
        DisaggregationMode::Aggregated,
    )
    .expect("optional nulls should be omitted");
}

#[test]
fn token_native_compatibility_envelope_rejects_unprojected_fields() {
    let mut request = request();
    request.extra_args = Some(json!({
        "vllm_tito": {
            "request_id": "request-1",
            "sampling_params": {"max_tokens": 1, "future_sampling_field": true},
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

fn request_with_preprocessed_features(features: serde_json::Value) -> PreprocessedRequest {
    let mut request = request();
    request.extra_args = Some(json!({
        "vllm_tito": {
            "request_id": "request-1",
            "sampling_params": {},
            "stream": false,
            "priority": 0,
            "features": features
        }
    }));
    request
}

const VALID_MM_KWARGS_BASE64: &str =
    "gaxwaXhlbF92YWx1ZXOCpGRhdGGTpXVpbnQ4kQPHAwMBAgOlZmllbGSSp2JhdGNoZWSBq2tlZXBfb25fY3B1wg==";

fn image_features(kwargs: serde_json::Value) -> serde_json::Value {
    json!({
        "mm_hashes": {"image": ["producer-controlled-hash"]},
        "mm_placeholders": {"image": [{"offset": 1, "length": 2}]},
        "kwargs_data": {"image": [kwargs]}
    })
}

#[test]
fn preprocessed_multimodal_features_use_vllm_derived_identifiers() {
    let request = request_with_preprocessed_features(image_features(json!(VALID_MM_KWARGS_BASE64)));

    let wire = build_generate_request(
        request,
        "request-1".to_string(),
        DisaggregationMode::Aggregated,
    )
    .expect("preprocessed feature should be forwarded");

    let feature = match wire.media[0].source.as_ref() {
        Some(pb::media_item::Source::Features(feature)) => feature,
        other => panic!("expected preprocessed features, got {other:?}"),
    };
    assert_eq!(feature.kwargs.as_ref().map(Vec::len), Some(64));
    assert_eq!(
        feature.identifier,
        "grpc-mm:eb917f9e320304a5c5fc538e0de431ccc7237ba302a2aa8b6f0dd39bde78509f"
    );
    assert_eq!(
        feature.mm_hash.as_deref(),
        Some(feature.identifier.as_str())
    );
    assert_eq!((feature.offset, feature.length), (1, 2));
}

#[test]
fn preprocessed_multimodal_features_reject_unknown_nested_fields() {
    let mut features = image_features(json!(VALID_MM_KWARGS_BASE64));
    features["future_field"] = json!(true);
    let root_error = build_generate_request(
        request_with_preprocessed_features(features),
        "request-1".to_string(),
        DisaggregationMode::Aggregated,
    )
    .expect_err("unknown feature fields must fail closed");
    assert!(root_error.to_string().contains("future_field"));

    let mut features = image_features(json!(VALID_MM_KWARGS_BASE64));
    features["mm_placeholders"]["image"][0]["future_field"] = json!(true);
    let placeholder_error = build_generate_request(
        request_with_preprocessed_features(features),
        "request-1".to_string(),
        DisaggregationMode::Aggregated,
    )
    .expect_err("unknown placeholder fields must fail closed");
    assert!(placeholder_error.to_string().contains("future_field"));
}

#[test]
fn preprocessed_multimodal_features_require_inline_kwargs() {
    for kwargs in [
        serde_json::Value::Null,
        serde_json::Value::String(String::new()),
    ] {
        let error = build_generate_request(
            request_with_preprocessed_features(image_features(kwargs)),
            "request-1".to_string(),
            DisaggregationMode::Aggregated,
        )
        .expect_err("cache-only or empty feature kwargs must be rejected");
        assert!(error.to_string().contains("kwargs_data"));
    }
}

#[test]
fn preprocessed_multimodal_features_bound_producer_hashes() {
    let mut features = image_features(json!(VALID_MM_KWARGS_BASE64));
    features["mm_hashes"]["image"][0] = json!("h".repeat(257));
    let error = build_generate_request(
        request_with_preprocessed_features(features),
        "request-1".to_string(),
        DisaggregationMode::Aggregated,
    )
    .expect_err("unused producer hashes must still be bounded");
    assert!(error.to_string().contains("between 1 and 256 bytes"));
}

#[test]
fn preprocessed_multimodal_features_enforce_count_and_byte_limits() {
    let hashes = vec!["producer-controlled-hash"; 65];
    let placeholders = (0..65)
        .map(|offset| json!({"offset": offset, "length": 1}))
        .collect::<Vec<_>>();
    let kwargs = vec![VALID_MM_KWARGS_BASE64; 65];
    let too_many = json!({
        "mm_hashes": {"image": hashes},
        "mm_placeholders": {"image": placeholders},
        "kwargs_data": {"image": kwargs}
    });
    let count_error = build_generate_request(
        request_with_preprocessed_features(too_many),
        "request-1".to_string(),
        DisaggregationMode::Aggregated,
    )
    .expect_err("feature count must be bounded");
    assert!(count_error.to_string().contains("at most 64"));

    let oversized = "AAAA".repeat((16 * 1024 * 1024) / 3 + 1);
    let byte_error = build_generate_request(
        request_with_preprocessed_features(image_features(json!(oversized))),
        "request-1".to_string(),
        DisaggregationMode::Aggregated,
    )
    .expect_err("decoded feature bytes must be bounded");
    assert!(byte_error.to_string().contains("16 MiB"));
}

#[test]
fn preprocessed_multimodal_features_accept_transport_boundaries() {
    let hashes = vec!["producer-controlled-hash"; 64];
    let placeholders = (0..64)
        .map(|offset| json!({"offset": offset, "length": 1}))
        .collect::<Vec<_>>();
    let kwargs = vec![VALID_MM_KWARGS_BASE64; 64];
    let mut count_request = request_with_preprocessed_features(json!({
        "mm_hashes": {"image": hashes},
        "mm_placeholders": {"image": placeholders},
        "kwargs_data": {"image": kwargs}
    }));
    count_request.token_ids = vec![1; 64];
    let count_wire = build_generate_request(
        count_request,
        "request-1".to_string(),
        DisaggregationMode::Aggregated,
    )
    .expect("64 features should be accepted");
    assert_eq!(count_wire.media.len(), 64);

    let exact_limit = format!("{}AA==", "AAAA".repeat((16 * 1024 * 1024) / 3));
    build_generate_request(
        request_with_preprocessed_features(image_features(json!(exact_limit))),
        "request-1".to_string(),
        DisaggregationMode::Aggregated,
    )
    .expect("exactly 16 MiB of decoded kwargs should be accepted");
}

#[test]
fn preprocessed_multimodal_features_validate_alignment_and_ranges() {
    let cases = [
        (
            json!({
                "mm_hashes": {"image": ["hash"]},
                "mm_placeholders": {"video": [{"offset": 1, "length": 1}]},
                "kwargs_data": {"image": [VALID_MM_KWARGS_BASE64]}
            }),
            "same modalities",
        ),
        (
            json!({
                "mm_hashes": {"image": ["hash", "hash-2"]},
                "mm_placeholders": {"image": [{"offset": 1, "length": 1}]},
                "kwargs_data": {"image": [VALID_MM_KWARGS_BASE64]}
            }),
            "equal lengths",
        ),
        (image_features(json!("not base64")), "not valid base64"),
        (
            json!({
                "mm_hashes": {"image": ["hash", "hash-2"]},
                "mm_placeholders": {"image": [
                    {"offset": 0, "length": 2},
                    {"offset": 1, "length": 2}
                ]},
                "kwargs_data": {"image": [VALID_MM_KWARGS_BASE64, VALID_MM_KWARGS_BASE64]}
            }),
            "cannot overlap",
        ),
        (
            json!({
                "mm_hashes": {"image": ["hash"]},
                "mm_placeholders": {"image": [{"offset": 2, "length": 2}]},
                "kwargs_data": {"image": [VALID_MM_KWARGS_BASE64]}
            }),
            "exceeds the prompt token count",
        ),
        (
            json!({
                "mm_hashes": {"image": ["hash"]},
                "mm_placeholders": {"image": [{
                    "offset": 1,
                    "length": 2,
                    "is_embed": [true]
                }]},
                "kwargs_data": {"image": [VALID_MM_KWARGS_BASE64]}
            }),
            "is_embed must match length",
        ),
    ];

    for (features, expected) in cases {
        let error = build_generate_request(
            request_with_preprocessed_features(features),
            "request-1".to_string(),
            DisaggregationMode::Aggregated,
        )
        .expect_err(expected);
        assert!(
            error.to_string().contains(expected),
            "expected {expected:?}, got {error}"
        );
    }
}

#[test]
fn preprocessed_multimodal_features_cannot_mix_with_raw_media() {
    let mut request =
        request_with_preprocessed_features(image_features(json!(VALID_MM_KWARGS_BASE64)));
    request.multi_modal_data = Some(std::collections::HashMap::from([(
        "image_url".to_string(),
        vec![MultimodalData::RawUrl(
            "data:image/png;base64,iVBORw0KGgo=".to_string(),
        )],
    )]));

    let error = build_generate_request(
        request,
        "request-1".to_string(),
        DisaggregationMode::Aggregated,
    )
    .expect_err("raw media and preprocessed features must not be mixed");

    assert!(error.to_string().contains("cannot be mixed"));
}

#[tokio::test]
async fn preprocessed_multimodal_features_require_model_support() {
    let engine = engine(
        "http://127.0.0.1:9",
        DisaggregationMode::Aggregated,
        1,
        model_info(),
    );
    let context = dynamo_backend_common::testing::mock_context();
    let result = engine
        .generate(
            request_with_preprocessed_features(image_features(json!(VALID_MM_KWARGS_BASE64))),
            GenerateContext::new(context, None),
        )
        .await;
    let error = match result {
        Ok(_) => panic!("a text-only model must reject preprocessed media before RPC submission"),
        Err(error) => error,
    };

    assert!(
        error
            .to_string()
            .contains("does not advertise multimodal support")
    );
}

#[test]
fn skip_special_tokens_false_is_forwarded_to_vllm() {
    let mut request = request();
    request.output_options.skip_special_tokens = Some(false);

    let mapped = build_generate_request(
        request,
        "skip-special-tokens".to_string(),
        DisaggregationMode::Aggregated,
    )
    .expect("skip_special_tokens=false should be supported");

    assert_eq!(
        mapped
            .response
            .expect("response options")
            .skip_special_tokens,
        Some(false)
    );
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
        GrpcEndpoint::parse(endpoint, "--vllm-endpoint").expect("valid test endpoint"),
        DiscoveredModel::from_proto(model, server).expect("valid discovery"),
        mode,
        transport,
    )
}

async fn engine_from_args(
    endpoint: &str,
) -> (VllmSidecarEngine, dynamo_backend_common::WorkerConfig) {
    let argv = vec![
        "dynamo-vllm-sidecar".to_string(),
        "--vllm-endpoint".to_string(),
        endpoint.to_string(),
        "--vllm-http-endpoint".to_string(),
        "http://worker:8120".to_string(),
        "--enable-rl".to_string(),
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
    assert_eq!(
        worker.rl_metadata,
        Some(
            RlWorkerMetadata::new(
                4,
                Some("nccl".to_string()),
                Some("http://worker:8120".to_string()),
            )
            .expect("valid RL metadata")
        )
    );
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

#[tokio::test]
async fn preprocessed_multimodal_features_preserve_disaggregated_handoff() {
    let service = FakeVllm::default();
    let mut discovered = model_info();
    discovered.supports_multimodal = true;
    *service.model_info_override.lock().await = Some(discovered.clone());
    let server = FakeServer::start(service).await;
    let prefill = engine(
        &server.endpoint,
        DisaggregationMode::Prefill,
        1,
        discovered.clone(),
    );
    let decode = engine(&server.endpoint, DisaggregationMode::Decode, 1, discovered);
    prefill.start(1).await.expect("start prefill");
    decode.start(2).await.expect("start decode");

    let mut feature_request =
        request_with_preprocessed_features(image_features(json!(VALID_MM_KWARGS_BASE64)));
    feature_request.output_options.prompt_logprobs = None;
    let prefill_context = dynamo_backend_common::testing::mock_context();
    feature_request.extra_args.as_mut().unwrap()["vllm_tito"]["request_id"] =
        json!(prefill_context.id());
    let prefill_outputs = prefill
        .generate(
            feature_request.clone(),
            GenerateContext::new(prefill_context, None),
        )
        .await
        .expect("prefill generate")
        .map(|item| item.expect("prefill output"))
        .collect::<Vec<_>>()
        .await;
    let handoff = prefill_outputs[0]
        .disaggregated_params
        .clone()
        .expect("preprocessed multimodal handoff");
    assert_eq!(
        handoff["_dynamo_sidecar_multimodal_prompt_token_ids"]
            .as_array()
            .expect("expanded prompt token IDs")
            .len(),
        601
    );

    let mut decode_request = feature_request;
    decode_request.prefill_result = Some(PrefillResult {
        disaggregated_params: handoff,
        prompt_tokens_details: None,
    });
    let decode_context = dynamo_backend_common::testing::mock_context();
    decode_request.extra_args.as_mut().unwrap()["vllm_tito"]["request_id"] =
        json!(decode_context.id());
    let decode_outputs = decode
        .generate(decode_request, GenerateContext::new(decode_context, None))
        .await
        .expect("decode generate")
        .map(|item| item.expect("decode output"))
        .collect::<Vec<_>>()
        .await;
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
    assert!(matches!(
        prefill_wire.media[0].source.as_ref(),
        Some(pb::media_item::Source::Features(_))
    ));
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

    let mut lora_request = serde_json::to_value(request()).expect("serialize request");
    lora_request["routing"] = json!({"lora_name": "adapter"});
    requests.push(serde_json::from_value(lora_request).expect("deserialize request"));

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
