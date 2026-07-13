// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicI32, AtomicU8, AtomicU64, Ordering};

use dynamo_backend_common::{GenerateContext, LLMEngine, PreprocessedRequest};
use dynamo_llm::protocols::common::preprocessor::{MultimodalData, RoutingHints};
use parking_lot::Mutex;

use crate::convert;
use crate::proto as pb;

type GrpcStream<T> = Pin<Box<dyn futures::Stream<Item = Result<T, tonic::Status>> + Send>>;

fn request() -> PreprocessedRequest {
    PreprocessedRequest::builder()
        .model("model".to_string())
        .token_ids(vec![1, 2, 3])
        .stop_conditions(Default::default())
        .sampling_options(Default::default())
        .output_options(Default::default())
        .build()
        .unwrap()
}

fn trt_handoff(
    dp_rank: u32,
    ctx_request_id: &str,
    opaque_state: Option<&str>,
) -> serde_json::Value {
    let profile = serde_json::json!({
        "first_gen_tokens": null,
        "first_gen_log_probs": null,
        "ctx_request_id": ctx_request_id,
        "disagg_request_id": "9223372036854775808",
        "ctx_dp_rank": dp_rank,
        "ctx_info_endpoint": null,
        "draft_tokens": null,
        "ctx_usage": null,
        "conversation_id": null,
        "schedule_style": "context_first",
        "requires_decode_media": true,
        "opaque_state": opaque_state,
    });
    serde_json::json!({
        "session_id": "ctx",
        "transfer_backend": "tensorrt_llm",
        "endpoints": [{"host": "127.0.0.1", "port": 1234, "protocol": "nixl"}],
        "dp_rank": dp_rank,
        "attributes_struct": {
            "tensorrt_llm.disaggregated_params.v1": serde_json::to_string(&profile).unwrap()
        }
    })
}

#[test]
fn request_conversion_forwards_sampling_routing_and_stopping() {
    let mut request = request();
    request.sampling_options.temperature = Some(0.25);
    request.sampling_options.top_k = Some(7);
    request.sampling_options.n = Some(2);
    request.stop_conditions.max_tokens = Some(16);
    request.stop_conditions.stop = Some(vec!["done".to_string()]);
    request.routing = Some(RoutingHints {
        dp_rank: Some(3),
        priority: Some(9),
        lora_name: Some("adapter".to_string()),
        cache_namespace: Some("tenant".to_string()),
        ..Default::default()
    });
    let converted =
        convert::build_generate_request(&request, "r1", "served", false, false).unwrap();
    assert_eq!(converted.model, "served");
    assert_eq!(converted.priority, Some(9));
    assert_eq!(converted.lora_name, "adapter");
    assert_eq!(converted.sampling.unwrap().num_sequences, Some(2));
    assert_eq!(converted.stopping.unwrap().max_tokens, Some(16));
    let kv = converted.kv.unwrap();
    assert_eq!(kv.data_parallel_rank, Some(3));
    assert_eq!(kv.cache_salt.as_deref(), Some("tenant"));
}

#[test]
fn multimodal_order_and_media_options_follow_original_messages() {
    let mut request = request();
    request.multi_modal_data = Some(HashMap::from([
        (
            "image_url".to_string(),
            vec![MultimodalData::RawUrl("https://host/image.png".to_string())],
        ),
        (
            "audio_url".to_string(),
            vec![MultimodalData::RawUrl(
                "data:audio/wav;base64,AAAA".to_string(),
            )],
        ),
    ]));
    request.extra_args = Some(serde_json::json!({
        "messages": [{"content": [
            {"type": "audio_url", "audio_url": {"url": "ignored"}},
            {"type": "image_url", "image_url": {"url": "ignored"}}
        ]}],
        "formatted_prompt": "<audio><image>describe them"
    }));
    request.mm_processor_kwargs = Some(serde_json::json!({"num_frames": 8}));

    let converted = convert::build_generate_request(&request, "r1", "served", false, true).unwrap();
    assert_eq!(
        pb::Modality::try_from(converted.media[0].modality).unwrap(),
        pb::Modality::Audio
    );
    assert_eq!(
        pb::Modality::try_from(converted.media[1].modality).unwrap(),
        pb::Modality::Image
    );
    assert!(matches!(
        converted.media[0].source,
        Some(pb::media_item::Source::DataUri(_))
    ));
    assert!(matches!(
        converted.input,
        Some(pb::generate_request::Input::Prompt(ref prompt))
            if prompt == "<audio><image>describe them"
    ));
    let token_only =
        convert::build_generate_request(&request, "r2", "served", false, false).unwrap();
    assert!(matches!(
        token_only.input,
        Some(pb::generate_request::Input::TokenIds(_))
    ));
    let options = convert::prost_struct_to_json(converted.media_options.as_ref().unwrap());
    assert_eq!(options["audio"]["num_frames"], 8);
    assert_eq!(options["image"]["num_frames"], 8);
}

#[test]
fn decoded_media_fails_closed() {
    let mut request = request();
    let descriptor = serde_json::from_value(serde_json::json!({
        "nixl_metadata": "opaque-agent-metadata",
        "nixl_descriptor": {
            "addr": 0,
            "size": 1,
            "mem_type": "Dram",
            "device_id": 0
        },
        "shape": [1],
        "dtype": "UINT8",
        "metadata": null
    }))
    .expect("construct a real serialized RDMA media descriptor");
    request.multi_modal_data = Some(HashMap::from([(
        "image_url".to_string(),
        vec![MultimodalData::Decoded(descriptor)],
    )]));
    assert!(convert::build_generate_request(&request, "r1", "m", false, false).is_err());
}

#[test]
fn decode_handoff_preserves_media_and_media_options_for_mrope() {
    let mut request = request();
    request.multi_modal_data = Some(HashMap::from([(
        "image_url".to_string(),
        vec![MultimodalData::RawUrl("https://host/image.png".to_string())],
    )]));
    request.mm_processor_kwargs = Some(serde_json::json!({"min_pixels": 64}));
    request.prefill_result = Some(dynamo_backend_common::PrefillResult {
        disaggregated_params: trt_handoff(2, "42", None),
        prompt_tokens_details: None,
    });
    let converted =
        convert::build_generate_request(&request, "r1", "served", false, false).unwrap();
    assert_eq!(converted.media.len(), 1);
    assert!(converted.media_options.is_some());
    assert_eq!(converted.kv.unwrap().session.unwrap().session_id, "ctx");
}

#[test]
fn trt_handoff_preserves_large_ids_and_binary_as_strings() {
    let large_id = "18446744073709551615";
    let opaque = "AP8QICo=";
    let json = trt_handoff(2, large_id, Some(opaque));
    let restored = convert::disagg_json_to_kv_session(&json).unwrap();
    let restored_attributes =
        convert::prost_struct_to_json(restored.attributes_struct.as_ref().unwrap());
    let profile: serde_json::Value = serde_json::from_str(
        restored_attributes["tensorrt_llm.disaggregated_params.v1"]
            .as_str()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(profile["ctx_request_id"], large_id);
    assert_eq!(profile["opaque_state"], opaque);
    assert_eq!(restored.endpoints[0].port, 1234);
}

#[test]
fn malformed_or_generation_first_trt_handoffs_fail_closed() {
    let mut missing_profile = trt_handoff(0, "42", None);
    missing_profile["attributes_struct"] = serde_json::json!({});
    assert!(convert::disagg_json_to_kv_session(&missing_profile).is_err());

    let mut generation_first = trt_handoff(0, "42", None);
    let encoded = generation_first["attributes_struct"]["tensorrt_llm.disaggregated_params.v1"]
        .as_str()
        .unwrap();
    let mut profile: serde_json::Value = serde_json::from_str(encoded).unwrap();
    profile["schedule_style"] = serde_json::json!("generation_first");
    generation_first["attributes_struct"]["tensorrt_llm.disaggregated_params.v1"] =
        serde_json::json!(serde_json::to_string(&profile).unwrap());
    assert!(convert::disagg_json_to_kv_session(&generation_first).is_err());

    let mut numeric_id = trt_handoff(0, "42", None);
    let encoded = numeric_id["attributes_struct"]["tensorrt_llm.disaggregated_params.v1"]
        .as_str()
        .unwrap();
    let mut profile: serde_json::Value = serde_json::from_str(encoded).unwrap();
    profile["ctx_request_id"] = serde_json::json!(u64::MAX);
    numeric_id["attributes_struct"]["tensorrt_llm.disaggregated_params.v1"] =
        serde_json::json!(serde_json::to_string(&profile).unwrap());
    assert!(convert::disagg_json_to_kv_session(&numeric_id).is_err());

    let mut bad_base64 = trt_handoff(0, "42", None);
    let encoded = bad_base64["attributes_struct"]["tensorrt_llm.disaggregated_params.v1"]
        .as_str()
        .unwrap();
    let mut profile: serde_json::Value = serde_json::from_str(encoded).unwrap();
    profile["opaque_state"] = serde_json::json!("not-base64");
    bad_base64["attributes_struct"]["tensorrt_llm.disaggregated_params.v1"] =
        serde_json::json!(serde_json::to_string(&profile).unwrap());
    assert!(convert::disagg_json_to_kv_session(&bad_base64).is_err());

    let mut unsupported_logits = trt_handoff(0, "42", None);
    let encoded = unsupported_logits["attributes_struct"]["tensorrt_llm.disaggregated_params.v1"]
        .as_str()
        .unwrap();
    let mut profile: serde_json::Value = serde_json::from_str(encoded).unwrap();
    profile["first_gen_logits"] = serde_json::json!([0.5]);
    unsupported_logits["attributes_struct"]["tensorrt_llm.disaggregated_params.v1"] =
        serde_json::json!(serde_json::to_string(&profile).unwrap());
    assert!(convert::disagg_json_to_kv_session(&unsupported_logits).is_err());

    let mut incoherent_rank = trt_handoff(0, "42", None);
    incoherent_rank["dp_rank"] = serde_json::json!(1);
    assert!(convert::disagg_json_to_kv_session(&incoherent_rank).is_err());

    let mut bad_endpoint = trt_handoff(0, "42", None);
    bad_endpoint["endpoints"][0]["port"] = serde_json::json!(70000);
    assert!(convert::disagg_json_to_kv_session(&bad_endpoint).is_err());

    let mut empty_endpoints = trt_handoff(0, "42", None);
    empty_endpoints["endpoints"] = serde_json::json!([]);
    assert!(
        convert::disagg_json_to_kv_session(&empty_endpoints).is_ok(),
        "endpoint list is optional; entries are validated only when present"
    );
}

#[test]
fn token_delta_preserves_text_logprobs_and_output_index() {
    let output = convert::token_output(pb::TokenOutput {
        output_index: Some(4),
        tokens: vec![pb::TokenInfo {
            token_id: 9,
            token: "hello".into(),
            logprob: Some(-0.25),
            rank: Some(1),
            candidates: vec![pb::LogProb {
                token_id: 10,
                logprob: -1.5,
                token: "world".into(),
                rank: Some(2),
            }],
        }],
        text: "hello".into(),
    });
    assert_eq!(output.index, Some(4));
    assert_eq!(output.token_ids, vec![9]);
    assert_eq!(output.text.as_deref(), Some("hello"));
    assert_eq!(output.log_probs.as_ref().unwrap(), &vec![-0.25]);
    assert_eq!(output.top_logprobs.as_ref().unwrap()[0][0].token_id, 10);
}

#[test]
fn runtime_trace_metadata_is_merged_into_generate_request() {
    let mut converted =
        convert::build_generate_request(&request(), "r1", "served", false, false).unwrap();
    convert::merge_context_metadata(
        &mut converted,
        &BTreeMap::from([
            ("traceparent".to_string(), "00-abcd-1234-01".to_string()),
            ("tracestate".to_string(), "vendor=value".to_string()),
        ]),
    );
    assert_eq!(converted.metadata["traceparent"], "00-abcd-1234-01");
    assert_eq!(converted.metadata["tracestate"], "vendor=value");
}

const AGGREGATE: u8 = 0;
const PREFILL: u8 = 1;
const PENDING: u8 = 2;
const MULTI_OUTPUT: u8 = 3;
const PROMPT_LOGPROBS: u8 = 4;
const PREFILL_NO_USAGE: u8 = 5;
const PREFILL_BAD_HANDOFF: u8 = 6;

struct FakeState {
    engine: Mutex<pb::EngineInfo>,
    model: Mutex<pb::ModelInfo>,
    health: AtomicI32,
    behavior: AtomicU8,
    requests: Mutex<Vec<pb::GenerateRequest>>,
    aborts: Mutex<Vec<String>>,
    load: Mutex<pb::LoadInfo>,
    loras: Mutex<HashMap<String, pb::LoraAdapter>>,
    subscriptions: Mutex<Vec<pb::SubscribeKvEventsRequest>>,
    discovery_delay_ms: AtomicU64,
}

impl Default for FakeState {
    fn default() -> Self {
        Self {
            engine: Mutex::new(pb::EngineInfo {
                engine_name: "tensorrt_llm".into(),
                role: pb::EngineRole::Aggregated as i32,
                supported_models: vec!["model".into()],
                schema_revision: openengine_proto::SCHEMA_REVISION,
                minimum_client_revision: 1,
                schema_release: crate::OPENENGINE_PROTO_COMMIT.into(),
                parallelism: Some(pb::ParallelismInfo {
                    data_parallel_size: Some(2),
                    data_parallel_start_rank: Some(0),
                    ..Default::default()
                }),
                kv_connector: Some(pb::KvConnectorInfo {
                    enabled: Some(true),
                    transfer_backend: "tensorrt_llm".into(),
                    supported_protocols: vec!["tcp".into()],
                    supports_remote_prefill: Some(true),
                    supports_decode_pull: Some(true),
                    supports_abort_cleanup: Some(true),
                    supports_drain: Some(true),
                    schema_version: Some(1),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            model: Mutex::new(pb::ModelInfo {
                model_id: "model".into(),
                served_model_name: "model".into(),
                supports_token_ids_input: Some(true),
                supports_text_input: Some(true),
                supports_lora: Some(true),
                kv_block_size: Some(16),
                generation: Some(pb::GenerationCapabilities {
                    prompt_logprobs: Some(pb::LogprobCapabilities {
                        supported: Some(true),
                        candidate_selection_modes: vec![
                            pb::CandidateTokenSelectionMode::TopN as i32,
                        ],
                        max_top_n: Some(4),
                    }),
                    max_num_sequences: Some(4),
                    ..Default::default()
                }),
                multimodal_capabilities: Some(pb::MultimodalCapabilities {
                    aggregate_modalities: vec![pb::Modality::Image as i32],
                    prefill_decode_modalities: vec![pb::Modality::Image as i32],
                    source_types: vec![
                        pb::MediaSourceType::Url as i32,
                        pb::MediaSourceType::DataUri as i32,
                    ],
                    supports_per_request_media_options: Some(true),
                }),
                ..Default::default()
            }),
            health: AtomicI32::new(pb::HealthState::Ready as i32),
            behavior: AtomicU8::new(AGGREGATE),
            requests: Mutex::new(Vec::new()),
            aborts: Mutex::new(Vec::new()),
            load: Mutex::new(pb::LoadInfo {
                running_requests: Some(0),
                used_kv_blocks: Some(1),
                total_kv_blocks: Some(10),
                ..Default::default()
            }),
            loras: Mutex::new(HashMap::new()),
            subscriptions: Mutex::new(Vec::new()),
            discovery_delay_ms: AtomicU64::new(0),
        }
    }
}

#[derive(Clone)]
struct FakeOpenEngine(Arc<FakeState>);

#[tonic::async_trait]
impl pb::open_engine_server::OpenEngine for FakeOpenEngine {
    type GenerateStream = GrpcStream<pb::GenerateResponse>;

    async fn generate(
        &self,
        request: tonic::Request<pb::GenerateRequest>,
    ) -> Result<tonic::Response<Self::GenerateStream>, tonic::Status> {
        let request = request.into_inner();
        let request_id = request.request_id.clone();
        self.0.requests.lock().push(request);
        let behavior = self.0.behavior.load(Ordering::SeqCst);
        if matches!(behavior, PREFILL | PREFILL_NO_USAGE | PREFILL_BAD_HANDOFF) {
            let mut handoff = trt_handoff(0, "18446744073709551615", Some("AP8QICo="));
            if behavior == PREFILL_BAD_HANDOFF {
                let encoded = handoff["attributes_struct"]["tensorrt_llm.disaggregated_params.v1"]
                    .as_str()
                    .unwrap();
                let mut profile: serde_json::Value = serde_json::from_str(encoded).unwrap();
                profile["schedule_style"] = serde_json::json!("generation_first");
                handoff["attributes_struct"]["tensorrt_llm.disaggregated_params.v1"] =
                    serde_json::json!(serde_json::to_string(&profile).unwrap());
            }
            let attributes = handoff["attributes_struct"].clone();
            let usage = (behavior != PREFILL_NO_USAGE).then_some(pb::Usage {
                prompt_tokens: 3,
                total_tokens: 3,
                ..Default::default()
            });
            return Ok(tonic::Response::new(Box::pin(futures::stream::once(
                async move {
                    Ok(pb::GenerateResponse {
                        request_id: request_id.clone(),
                        event: Some(pb::generate_response::Event::PrefillReady(
                            pb::PrefillReady {
                                kv_session: Some(pb::KvSessionRef {
                                    session_id: request_id,
                                    transfer_backend: "tensorrt_llm".into(),
                                    endpoints: vec![pb::KvEndpoint {
                                        host: "127.0.0.1".into(),
                                        port: 9000,
                                        protocol: "tcp".into(),
                                    }],
                                    dp_rank: 0,
                                    attributes_struct: convert::json_to_prost_struct(&attributes),
                                }),
                            },
                        )),
                        usage,
                    })
                },
            ))));
        }
        if self.0.behavior.load(Ordering::SeqCst) == PENDING {
            return Ok(tonic::Response::new(Box::pin(async_stream::try_stream! {
                yield pb::GenerateResponse {
                    request_id,
                    event: Some(pb::generate_response::Event::Token(pb::TokenOutput {
                        output_index: Some(0),
                        tokens: vec![pb::TokenInfo { token_id: 42, token: "x".into(), ..Default::default() }],
                        text: "x".into(),
                    })),
                    usage: None,
                };
                std::future::pending::<()>().await;
            })));
        }
        if self.0.behavior.load(Ordering::SeqCst) == MULTI_OUTPUT {
            return Ok(tonic::Response::new(Box::pin(futures::stream::iter([
                Ok(pb::GenerateResponse {
                    request_id: request_id.clone(),
                    event: Some(pb::generate_response::Event::Finished(
                        pb::GenerationFinished {
                            output_index: Some(0),
                            reason: pb::FinishReason::Length as i32,
                            ..Default::default()
                        },
                    )),
                    usage: None,
                }),
                Ok(pb::GenerateResponse {
                    request_id,
                    event: Some(pb::generate_response::Event::Finished(
                        pb::GenerationFinished {
                            output_index: Some(1),
                            reason: pb::FinishReason::Stop as i32,
                            ..Default::default()
                        },
                    )),
                    usage: Some(pb::Usage {
                        prompt_tokens: 5,
                        completion_tokens: 7,
                        total_tokens: 99,
                        ..Default::default()
                    }),
                }),
            ]))));
        }
        if self.0.behavior.load(Ordering::SeqCst) == PROMPT_LOGPROBS {
            return Ok(tonic::Response::new(Box::pin(futures::stream::iter([
                Ok(pb::GenerateResponse {
                    request_id: request_id.clone(),
                    event: Some(pb::generate_response::Event::Prompt(pb::PromptOutput {
                        tokens: vec![
                            pb::TokenInfo {
                                token_id: 1,
                                token: "<bos>".into(),
                                ..Default::default()
                            },
                            pb::TokenInfo {
                                token_id: 2,
                                token: "hello".into(),
                                logprob: Some(-0.25),
                                rank: Some(1),
                                candidates: vec![pb::LogProb {
                                    token_id: 3,
                                    token: "world".into(),
                                    logprob: -1.5,
                                    rank: Some(2),
                                }],
                            },
                        ],
                    })),
                    usage: None,
                }),
                Ok(pb::GenerateResponse {
                    request_id,
                    event: Some(pb::generate_response::Event::Finished(
                        pb::GenerationFinished {
                            output_index: Some(0),
                            reason: pb::FinishReason::Stop as i32,
                            ..Default::default()
                        },
                    )),
                    usage: Some(pb::Usage {
                        prompt_tokens: 2,
                        total_tokens: 2,
                        ..Default::default()
                    }),
                }),
            ]))));
        }
        let responses = vec![
            Ok(pb::GenerateResponse {
                request_id: request_id.clone(),
                event: Some(pb::generate_response::Event::Token(pb::TokenOutput {
                    output_index: Some(0),
                    tokens: vec![pb::TokenInfo {
                        token_id: 42,
                        token: "x".into(),
                        ..Default::default()
                    }],
                    text: "x".into(),
                })),
                usage: None,
            }),
            Ok(pb::GenerateResponse {
                request_id,
                event: Some(pb::generate_response::Event::Finished(
                    pb::GenerationFinished {
                        output_index: Some(0),
                        reason: pb::FinishReason::Stop as i32,
                        ..Default::default()
                    },
                )),
                usage: Some(pb::Usage {
                    prompt_tokens: 3,
                    completion_tokens: 1,
                    total_tokens: 4,
                    ..Default::default()
                }),
            }),
        ];
        Ok(tonic::Response::new(Box::pin(futures::stream::iter(
            responses,
        ))))
    }

    async fn embed(
        &self,
        _: tonic::Request<pb::EmbedRequest>,
    ) -> Result<tonic::Response<pb::EmbedResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("embed"))
    }

    async fn classify(
        &self,
        _: tonic::Request<pb::ClassifyRequest>,
    ) -> Result<tonic::Response<pb::ClassifyResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("classify"))
    }

    async fn score(
        &self,
        _: tonic::Request<pb::ScoreRequest>,
    ) -> Result<tonic::Response<pb::ScoreResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("score"))
    }

    async fn get_engine_info(
        &self,
        _: tonic::Request<pb::GetEngineInfoRequest>,
    ) -> Result<tonic::Response<pb::EngineInfo>, tonic::Status> {
        let delay_ms = self.0.discovery_delay_ms.load(Ordering::SeqCst);
        if delay_ms > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
        }
        Ok(tonic::Response::new(self.0.engine.lock().clone()))
    }

    async fn get_model_info(
        &self,
        _: tonic::Request<pb::GetModelInfoRequest>,
    ) -> Result<tonic::Response<pb::ModelInfo>, tonic::Status> {
        Ok(tonic::Response::new(self.0.model.lock().clone()))
    }

    async fn get_load(
        &self,
        _: tonic::Request<pb::GetLoadRequest>,
    ) -> Result<tonic::Response<pb::LoadInfo>, tonic::Status> {
        Ok(tonic::Response::new(self.0.load.lock().clone()))
    }

    async fn health(
        &self,
        _: tonic::Request<pb::HealthRequest>,
    ) -> Result<tonic::Response<pb::HealthResponse>, tonic::Status> {
        Ok(tonic::Response::new(pb::HealthResponse {
            state: self.0.health.load(Ordering::SeqCst),
            checks: Vec::new(),
        }))
    }

    async fn abort(
        &self,
        request: tonic::Request<pb::AbortRequest>,
    ) -> Result<tonic::Response<pb::AbortResponse>, tonic::Status> {
        if let Some(pb::abort_request::Target::RequestId(request_id)) = request.into_inner().target
        {
            self.0.aborts.lock().push(request_id);
        }
        Ok(tonic::Response::new(pb::AbortResponse {
            status: pb::AbortStatus::Aborted as i32,
            message: String::new(),
        }))
    }

    type DrainStream = GrpcStream<pb::DrainResponse>;

    async fn drain(
        &self,
        _: tonic::Request<pb::DrainRequest>,
    ) -> Result<tonic::Response<Self::DrainStream>, tonic::Status> {
        if self.0.behavior.load(Ordering::SeqCst) == PENDING {
            return Ok(tonic::Response::new(Box::pin(async_stream::try_stream! {
                yield pb::DrainResponse {
                    event: Some(pb::drain_response::Event::State(pb::DrainState::Started as i32)),
                    ..Default::default()
                };
                std::future::pending::<()>().await;
            })));
        }
        Ok(tonic::Response::new(Box::pin(futures::stream::once(
            async {
                Ok(pb::DrainResponse {
                    event: Some(pb::drain_response::Event::State(
                        pb::DrainState::Complete as i32,
                    )),
                    ..Default::default()
                })
            },
        ))))
    }

    async fn load_lora(
        &self,
        request: tonic::Request<pb::LoadLoraRequest>,
    ) -> Result<tonic::Response<pb::LoadLoraResponse>, tonic::Status> {
        let adapter = request
            .into_inner()
            .adapter
            .ok_or_else(|| tonic::Status::invalid_argument("adapter required"))?;
        let mut loras = self.0.loras.lock();
        let already_loaded = loras.get(&adapter.lora_name) == Some(&adapter);
        if let Some(existing) = loras.get(&adapter.lora_name)
            && existing != &adapter
        {
            return Err(tonic::Status::already_exists("conflicting adapter"));
        }
        loras.insert(adapter.lora_name.clone(), adapter.clone());
        Ok(tonic::Response::new(pb::LoadLoraResponse {
            adapter: Some(adapter),
            already_loaded,
        }))
    }

    async fn unload_lora(
        &self,
        request: tonic::Request<pb::UnloadLoraRequest>,
    ) -> Result<tonic::Response<pb::UnloadLoraResponse>, tonic::Status> {
        let adapter = self.0.loras.lock().remove(&request.into_inner().lora_name);
        Ok(tonic::Response::new(pb::UnloadLoraResponse { adapter }))
    }

    async fn list_loras(
        &self,
        _: tonic::Request<pb::ListLorasRequest>,
    ) -> Result<tonic::Response<pb::ListLorasResponse>, tonic::Status> {
        Ok(tonic::Response::new(pb::ListLorasResponse {
            adapters: self.0.loras.lock().values().cloned().collect(),
        }))
    }

    async fn get_kv_connector_info(
        &self,
        _: tonic::Request<pb::GetKvConnectorInfoRequest>,
    ) -> Result<tonic::Response<pb::KvConnectorInfo>, tonic::Status> {
        Ok(tonic::Response::new(
            self.0
                .engine
                .lock()
                .kv_connector
                .clone()
                .unwrap_or_default(),
        ))
    }

    async fn get_kv_event_sources(
        &self,
        _: tonic::Request<pb::GetKvEventSourcesRequest>,
    ) -> Result<tonic::Response<pb::GetKvEventSourcesResponse>, tonic::Status> {
        Ok(tonic::Response::new(pb::GetKvEventSourcesResponse {
            sources: vec![
                pb::KvEventSource {
                    transport: "zmq".into(),
                    endpoint_addr: Some(pb::KvEndpoint {
                        host: "127.0.0.1".into(),
                        port: 5555,
                        protocol: "tcp".into(),
                    }),
                    topic: "kv".into(),
                    data_parallel_rank: Some(0),
                    ..Default::default()
                },
                pb::KvEventSource {
                    transport: "grpc".into(),
                    data_parallel_rank: Some(1),
                    ..Default::default()
                },
            ],
        }))
    }

    type SubscribeKvEventsStream = GrpcStream<pb::SubscribeKvEventsResponse>;

    async fn subscribe_kv_events(
        &self,
        request: tonic::Request<pb::SubscribeKvEventsRequest>,
    ) -> Result<tonic::Response<Self::SubscribeKvEventsStream>, tonic::Status> {
        self.0.subscriptions.lock().push(request.into_inner());
        Ok(tonic::Response::new(Box::pin(futures::stream::once(
            async {
                Ok(pb::SubscribeKvEventsResponse {
                    event: Some(pb::subscribe_kv_events_response::Event::Batch(
                        pb::KvEventBatch {
                            data_parallel_rank: 1,
                            events: vec![pb::KvEvent {
                                event: Some(pb::kv_event::Event::AllBlocksCleared(
                                    pb::AllBlocksCleared {},
                                )),
                                ..Default::default()
                            }],
                            ..Default::default()
                        },
                    )),
                })
            },
        ))))
    }

    type SubscribeRuntimeEventsStream = GrpcStream<pb::SubscribeRuntimeEventsResponse>;

    async fn subscribe_runtime_events(
        &self,
        _: tonic::Request<pb::SubscribeRuntimeEventsRequest>,
    ) -> Result<tonic::Response<Self::SubscribeRuntimeEventsStream>, tonic::Status> {
        Err(tonic::Status::unimplemented("runtime events"))
    }
}

struct FakeServer {
    address: std::net::SocketAddr,
    shutdown: tokio::sync::oneshot::Sender<()>,
    task: tokio::task::JoinHandle<()>,
}

impl FakeServer {
    async fn start(state: Arc<FakeState>) -> Self {
        use tokio_stream::wrappers::TcpListenerStream;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind fake tonic server; these integration tests must never be skipped");
        let address = listener.local_addr().unwrap();
        let (shutdown, shutdown_rx) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(pb::open_engine_server::OpenEngineServer::new(
                    FakeOpenEngine(state),
                ))
                .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
                    let _ = shutdown_rx.await;
                })
                .await
                .unwrap();
        });
        Self {
            address,
            shutdown,
            task,
        }
    }

    async fn stop(self) {
        let _ = self.shutdown.send(());
        self.task.await.unwrap();
    }
}

async fn build_sidecar(
    address: std::net::SocketAddr,
    expected_engine: &str,
) -> Result<
    (
        crate::OpenEngineSidecar,
        dynamo_backend_common::WorkerConfig,
    ),
    dynamo_backend_common::DynamoError,
> {
    let expected_engine = expected_engine.to_string();
    tokio::task::spawn_blocking(move || {
        crate::OpenEngineSidecar::from_args(Some(vec![
            "dynamo-openengine-sidecar".to_string(),
            "--openengine-endpoint".to_string(),
            address.to_string(),
            "--expected-engine".to_string(),
            expected_engine,
            "--openengine-connections".to_string(),
            "1".to_string(),
            "--health-poll-interval-secs".to_string(),
            "1".to_string(),
            "--health-deadline-secs".to_string(),
            "5".to_string(),
            "--drain-timeout-secs".to_string(),
            "1".to_string(),
        ]))
    })
    .await
    .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fake_tonic_server_discovery_and_aggregate_stream() {
    use futures::StreamExt;

    let state = Arc::new(FakeState::default());
    let server = FakeServer::start(state.clone()).await;
    let (engine, config) = build_sidecar(server.address, "tensorrt_llm").await.unwrap();
    assert_eq!(config.model_name, "model");
    let started = engine.start(1).await.unwrap();
    assert_eq!(started.model, "model");
    let outputs = engine
        .generate(
            request(),
            GenerateContext::new(dynamo_backend_common::testing::mock_context(), None),
        )
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].as_ref().unwrap().token_ids, vec![42]);
    let terminal = outputs[1].as_ref().unwrap();
    assert!(terminal.finish_reason.is_some());
    assert_eq!(terminal.completion_usage.as_ref().unwrap().total_tokens, 4);

    // Some multimodal processors (notably Phi-4 audio) require the rendered
    // text prompt so they can expand media placeholders. Preserve it when the
    // frontend supplies it instead of forcing token-only input.
    let mut multimodal = request();
    multimodal.multi_modal_data = Some(HashMap::from([(
        "image_url".to_string(),
        vec![MultimodalData::RawUrl("https://host/image.png".to_string())],
    )]));
    multimodal.extra_args = Some(serde_json::json!({
        "messages": [{"content": [
            {"type": "image_url", "image_url": {"url": "https://host/image.png"}}
        ]}],
        "formatted_prompt": "<image>describe it"
    }));
    let formatted_outputs = engine
        .generate(
            multimodal,
            GenerateContext::new(dynamo_backend_common::testing::mock_context(), None),
        )
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert_eq!(formatted_outputs.len(), 2);
    let forwarded = state.requests.lock();
    assert_eq!(forwarded.last().unwrap().media.len(), 1);
    assert!(matches!(
        forwarded.last().unwrap().input,
        Some(pb::generate_request::Input::Prompt(ref prompt))
            if prompt == "<image>describe it"
    ));
    drop(forwarded);

    let mut bypass = request();
    bypass.extra_args = Some(serde_json::json!({"bypass_prefix_cache": true}));
    assert!(
        engine
            .generate(
                bypass,
                GenerateContext::new(dynamo_backend_common::testing::mock_context(), None),
            )
            .await
            .is_err()
    );
    engine.cleanup().await.unwrap();
    server.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fake_tonic_health_probe_bypasses_decode_handoff() {
    use futures::StreamExt;

    let state = Arc::new(FakeState::default());
    state.engine.lock().role = pb::EngineRole::Decode as i32;
    let server = FakeServer::start(state.clone()).await;
    let (engine, _) = build_sidecar(server.address, "tensorrt_llm").await.unwrap();
    engine.start(1).await.unwrap();

    assert!(engine.health_check_payload().await.unwrap().is_some());
    let mut probe = request();
    probe.is_probe = true;
    let outputs = engine
        .generate(
            probe,
            GenerateContext::new(dynamo_backend_common::testing::mock_context(), None),
        )
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert_eq!(outputs.len(), 1);
    assert!(outputs[0].is_ok());
    assert!(state.requests.lock().is_empty());

    engine.cleanup().await.unwrap();
    server.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fake_tonic_token_only_multimodal_uses_token_ids() {
    use futures::StreamExt;

    let state = Arc::new(FakeState::default());
    state.model.lock().supports_text_input = None;
    let server = FakeServer::start(state.clone()).await;
    let (engine, _) = build_sidecar(server.address, "tensorrt_llm").await.unwrap();
    engine.start(1).await.unwrap();

    let mut multimodal = request();
    multimodal.multi_modal_data = Some(HashMap::from([(
        "image_url".to_string(),
        vec![MultimodalData::RawUrl("https://host/image.png".to_string())],
    )]));
    multimodal.extra_args = Some(serde_json::json!({
        "messages": [{"content": [
            {"type": "image_url", "image_url": {"url": "https://host/image.png"}}
        ]}],
        "formatted_prompt": "<image>describe it"
    }));
    let outputs = engine
        .generate(
            multimodal,
            GenerateContext::new(dynamo_backend_common::testing::mock_context(), None),
        )
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert_eq!(outputs.len(), 2);
    assert!(matches!(
        state.requests.lock().last().unwrap().input.as_ref(),
        Some(pb::generate_request::Input::TokenIds(_))
    ));

    engine.cleanup().await.unwrap();
    server.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fake_tonic_rejects_unadvertised_request_semantics_before_scheduling() {
    let state = Arc::new(FakeState::default());
    state.model.lock().supports_lora = Some(false);
    let server = FakeServer::start(state.clone()).await;
    let (engine, _) = build_sidecar(server.address, "tensorrt_llm").await.unwrap();
    engine.start(1).await.unwrap();

    let mut requests = Vec::new();
    let mut priority = request();
    priority.routing = Some(RoutingHints {
        priority: Some(1),
        ..Default::default()
    });
    requests.push(priority);
    let mut salted = request();
    salted.routing = Some(RoutingHints {
        cache_namespace: Some("tenant".into()),
        ..Default::default()
    });
    requests.push(salted);
    let mut lora = request();
    lora.routing = Some(RoutingHints {
        lora_name: Some("adapter".into()),
        ..Default::default()
    });
    requests.push(lora);
    let mut guided = request();
    guided.sampling_options.guided_decoding = Some(dynamo_backend_common::GuidedDecodingOptions {
        regex: Some("a+".into()),
        ..Default::default()
    });
    requests.push(guided);
    let mut too_many = request();
    too_many.sampling_options.n = Some(5);
    requests.push(too_many);
    let mut too_many_logprobs = request();
    too_many_logprobs.output_options.prompt_logprobs = Some(5);
    requests.push(too_many_logprobs);

    for request in requests {
        assert!(
            engine
                .generate(
                    request,
                    GenerateContext::new(dynamo_backend_common::testing::mock_context(), None),
                )
                .await
                .is_err()
        );
    }
    assert!(
        state.requests.lock().is_empty(),
        "rejected requests reached Generate"
    );
    engine.cleanup().await.unwrap();
    server.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fake_tonic_rejects_schema_engine_and_role_mismatches() {
    let state = Arc::new(FakeState::default());
    let server = FakeServer::start(state.clone()).await;
    assert!(build_sidecar(server.address, "vllm").await.is_err());
    state.engine.lock().schema_release = "main".into();
    assert!(build_sidecar(server.address, "tensorrt_llm").await.is_err());
    state.engine.lock().schema_release = crate::OPENENGINE_PROTO_COMMIT.into();
    state.engine.lock().role = pb::EngineRole::Unspecified as i32;
    assert!(build_sidecar(server.address, "tensorrt_llm").await.is_err());
    state.engine.lock().role = pb::EngineRole::Aggregated as i32;
    let (engine, _) = build_sidecar(server.address, "tensorrt_llm").await.unwrap();
    state.engine.lock().role = pb::EngineRole::Decode as i32;
    assert!(engine.start(1).await.is_err());
    engine.cleanup().await.unwrap();
    server.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fake_tonic_prefill_decode_preserves_media_options_and_handoff() {
    use futures::StreamExt;

    let state = Arc::new(FakeState::default());
    state.engine.lock().role = pb::EngineRole::Prefill as i32;
    state.behavior.store(PREFILL, Ordering::SeqCst);
    let server = FakeServer::start(state.clone()).await;
    let mut media_request = request();
    media_request.multi_modal_data = Some(HashMap::from([(
        "image_url".to_string(),
        vec![MultimodalData::RawUrl("https://host/image.png".into())],
    )]));
    media_request.mm_processor_kwargs = Some(serde_json::json!({"image": {"min_pixels": 64}}));
    let (prefill, _) = build_sidecar(server.address, "tensorrt_llm").await.unwrap();
    prefill.start(1).await.unwrap();
    let mut output = prefill
        .generate(
            media_request.clone(),
            GenerateContext::new(dynamo_backend_common::testing::mock_context(), None),
        )
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    let handoff = output.remove(0).unwrap().disaggregated_params.unwrap();
    prefill.cleanup().await.unwrap();

    state.engine.lock().role = pb::EngineRole::Decode as i32;
    state.behavior.store(AGGREGATE, Ordering::SeqCst);
    state.requests.lock().clear();
    media_request.prefill_result = Some(dynamo_backend_common::PrefillResult {
        disaggregated_params: handoff,
        prompt_tokens_details: None,
    });
    let (decode, _) = build_sidecar(server.address, "tensorrt_llm").await.unwrap();
    decode.start(2).await.unwrap();
    let _ = decode
        .generate(
            media_request,
            GenerateContext::new(dynamo_backend_common::testing::mock_context(), None),
        )
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    let decoded = state.requests.lock().last().cloned().unwrap();
    assert!(decoded.kv.unwrap().session.is_some());
    assert_eq!(decoded.media.len(), 1);
    assert!(decoded.media_options.is_some());
    decode.cleanup().await.unwrap();
    server.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fake_tonic_prefill_rejects_missing_usage_and_malformed_server_handoff() {
    use futures::StreamExt;

    for behavior in [PREFILL_NO_USAGE, PREFILL_BAD_HANDOFF] {
        let state = Arc::new(FakeState::default());
        state.engine.lock().role = pb::EngineRole::Prefill as i32;
        state.behavior.store(behavior, Ordering::SeqCst);
        let server = FakeServer::start(state).await;
        let (engine, _) = build_sidecar(server.address, "tensorrt_llm").await.unwrap();
        engine.start(1).await.unwrap();
        let outputs = engine
            .generate(
                request(),
                GenerateContext::new(dynamo_backend_common::testing::mock_context(), None),
            )
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].is_err());
        engine.cleanup().await.unwrap();
        server.stop().await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fake_tonic_health_load_kv_discovery_and_watch_failure() {
    let state = Arc::new(FakeState::default());
    let server = FakeServer::start(state.clone()).await;
    let (engine, _) = build_sidecar(server.address, "tensorrt_llm").await.unwrap();
    engine.start(1).await.unwrap();
    assert_eq!(engine.is_quiescent().await.unwrap(), Some(true));
    state.load.lock().running_requests = Some(2);
    assert_eq!(engine.is_quiescent().await.unwrap(), Some(false));
    let sources = engine.kv_event_sources().await.unwrap();
    assert_eq!(sources.len(), 2);
    assert!(matches!(
        sources[0],
        dynamo_backend_common::KvEventSource::Zmq { dp_rank: 0, .. }
    ));
    assert!(matches!(
        sources[1],
        dynamo_backend_common::KvEventSource::Push { dp_rank: 1, .. }
    ));
    state
        .health
        .store(pb::HealthState::Draining as i32, Ordering::SeqCst);
    assert!(
        tokio::time::timeout(std::time::Duration::from_secs(3), engine.watch())
            .await
            .unwrap()
            .is_err()
    );
    engine.cleanup().await.unwrap();
    server.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fake_tonic_rejects_kv_source_rank_outside_discovered_parallelism() {
    let state = Arc::new(FakeState::default());
    state
        .engine
        .lock()
        .parallelism
        .as_mut()
        .unwrap()
        .data_parallel_size = Some(1);
    let server = FakeServer::start(state).await;
    let (engine, _) = build_sidecar(server.address, "tensorrt_llm").await.unwrap();
    engine.start(1).await.unwrap();
    let error = engine
        .kv_event_sources()
        .await
        .err()
        .expect("out-of-range KV source must fail discovery");
    assert!(
        error
            .to_string()
            .contains("outside its data-parallel range")
    );
    engine.cleanup().await.unwrap();
    server.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bootstrap_discovery_rpc_timeout_is_typed_and_bounded() {
    let state = Arc::new(FakeState::default());
    state.discovery_delay_ms.store(100, Ordering::SeqCst);
    let server = FakeServer::start(state).await;
    let transport = crate::args::TransportConfig {
        connect_timeout: std::time::Duration::from_secs(1),
        poll_interval: std::time::Duration::from_millis(10),
        deadline: std::time::Duration::from_secs(1),
        drain_timeout: std::time::Duration::from_secs(1),
        load_poll_interval: std::time::Duration::from_secs(1),
        connections: 1,
    };
    let mut grpc_client = crate::client::connect(&format!("http://{}", server.address), &transport)
        .await
        .unwrap();
    let error = crate::client::discover(
        &mut grpc_client,
        None,
        Some("tensorrt_llm"),
        std::time::Duration::from_millis(20),
    )
    .await
    .unwrap_err();
    assert_eq!(
        error.error_type(),
        dynamo_backend_common::ErrorType::Backend(
            dynamo_backend_common::BackendError::CannotConnect
        )
    );
    assert!(error.to_string().contains("GetEngineInfo"));
    server.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fake_tonic_blackholed_drain_stream_times_out() {
    let state = Arc::new(FakeState::default());
    state.behavior.store(PENDING, Ordering::SeqCst);
    let server = FakeServer::start(state).await;
    let (engine, _) = build_sidecar(server.address, "tensorrt_llm").await.unwrap();
    engine.start(1).await.unwrap();
    let result = tokio::time::timeout(std::time::Duration::from_secs(2), engine.drain())
        .await
        .expect("sidecar drain must have its own timeout");
    assert!(result.is_err());
    engine.cleanup().await.unwrap();
    server.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fake_tonic_dropped_stream_sends_abort() {
    use futures::StreamExt;

    let state = Arc::new(FakeState::default());
    state.behavior.store(PENDING, Ordering::SeqCst);
    let server = FakeServer::start(state.clone()).await;
    let (engine, _) = build_sidecar(server.address, "tensorrt_llm").await.unwrap();
    engine.start(1).await.unwrap();
    let mut stream = engine
        .generate(
            request(),
            GenerateContext::new(dynamo_backend_common::testing::mock_context(), None),
        )
        .await
        .unwrap();
    assert!(stream.next().await.is_some());
    drop(stream);
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            if !state.aborts.lock().is_empty() {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    engine.cleanup().await.unwrap();
    server.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fake_tonic_forwards_every_output_terminal_and_exact_final_usage() {
    use dynamo_backend_common::FinishReason;
    use futures::StreamExt;

    let state = Arc::new(FakeState::default());
    state.behavior.store(MULTI_OUTPUT, Ordering::SeqCst);
    let server = FakeServer::start(state).await;
    let (engine, _) = build_sidecar(server.address, "tensorrt_llm").await.unwrap();
    engine.start(1).await.unwrap();
    let mut generate_request = request();
    generate_request.sampling_options.n = Some(2);
    let outputs = engine
        .generate(
            generate_request,
            GenerateContext::new(dynamo_backend_common::testing::mock_context(), None),
        )
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].index, Some(0));
    assert_eq!(outputs[0].finish_reason, Some(FinishReason::Length));
    assert!(outputs[0].completion_usage.is_none());
    assert_eq!(outputs[1].index, Some(1));
    assert_eq!(outputs[1].finish_reason, Some(FinishReason::Stop));
    let usage = outputs[1].completion_usage.as_ref().unwrap();
    assert_eq!(usage.prompt_tokens, 5);
    assert_eq!(usage.completion_tokens, 7);
    assert_eq!(usage.total_tokens, 99);
    engine.cleanup().await.unwrap();
    server.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fake_tonic_attaches_exact_prompt_logprobs_to_terminal_engine_data() {
    use dynamo_backend_common::PromptLogprobs;
    use futures::StreamExt;

    let state = Arc::new(FakeState::default());
    state.behavior.store(PROMPT_LOGPROBS, Ordering::SeqCst);
    let server = FakeServer::start(state).await;
    let (engine, _) = build_sidecar(server.address, "tensorrt_llm").await.unwrap();
    engine.start(1).await.unwrap();
    let mut generate_request = request();
    generate_request.output_options.prompt_logprobs = Some(2);
    let outputs = engine
        .generate(
            generate_request,
            GenerateContext::new(dynamo_backend_common::testing::mock_context(), None),
        )
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(
        outputs.len(),
        1,
        "PromptOutput is metadata, not a Dynamo chunk"
    );
    let engine_data = outputs[0].engine_data.as_ref().unwrap();
    assert!(outputs[0].extra_args.is_none());
    let prompt: PromptLogprobs =
        serde_json::from_value(engine_data["prompt_logprobs"].clone()).unwrap();
    assert_eq!(prompt.len(), 2);
    assert!(prompt[0].is_none());
    let second = prompt[1].as_ref().unwrap();
    assert_eq!(second[&2].decoded_token.as_deref(), Some("hello"));
    assert_eq!(second[&3].rank, Some(2));
    engine.cleanup().await.unwrap();
    server.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fake_tonic_lora_lifecycle_publishes_and_removes_model_cards() {
    use dynamo_llm::local_model::LocalModelBuilder;
    use dynamo_llm::model_type::ModelType;
    use dynamo_llm::worker_type::WorkerType;
    use dynamo_runtime::distributed::DistributedConfig;
    use dynamo_runtime::{DistributedRuntime, Runtime};

    let state = Arc::new(FakeState::default());
    let server = FakeServer::start(state).await;
    let (mut engine, _) = build_sidecar(server.address, "tensorrt_llm").await.unwrap();
    engine.enable_local_lora_for_test();
    engine.start(1).await.unwrap();

    let runtime = Runtime::from_current().unwrap();
    let drt = DistributedRuntime::new(runtime, DistributedConfig::process_local())
        .await
        .unwrap();
    let endpoint = drt
        .namespace("lora-test")
        .unwrap()
        .component("backend")
        .unwrap()
        .endpoint("generate");
    let base_model = LocalModelBuilder::default()
        .model_name(Some("model".into()))
        .build()
        .await
        .unwrap();
    engine
        .on_model_ready(
            endpoint,
            base_model,
            ModelType::Chat | ModelType::Completions,
            WorkerType::Aggregated,
            Vec::new(),
        )
        .await
        .unwrap();

    let adapter_dir = std::env::temp_dir().join(format!(
        "dynamo-openengine-lora-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&adapter_dir).unwrap();
    std::fs::write(adapter_dir.join("adapter_config.json"), "{}").unwrap();
    let load = serde_json::json!({
        "lora_name": "adapter",
        "source": {"uri": format!("file://{}", adapter_dir.display())}
    });
    let first = engine
        .engine_update("load_lora".into(), load.clone())
        .await
        .unwrap();
    assert_eq!(first["status"], "ok");
    assert_eq!(engine.lora_card_count().await, 1);
    assert_eq!(
        engine.lora_card_display_name("adapter").await.as_deref(),
        Some("adapter")
    );
    let repeated = engine
        .engine_update("load_lora".into(), load)
        .await
        .unwrap();
    assert_eq!(repeated["already_loaded"], true);
    let listed = engine
        .engine_update("list_loras".into(), serde_json::Value::Null)
        .await
        .unwrap();
    assert_eq!(listed["adapters"].as_array().unwrap().len(), 1);
    engine
        .engine_update(
            "unload_lora".into(),
            serde_json::json!({"lora_name": "adapter"}),
        )
        .await
        .unwrap();
    assert_eq!(engine.lora_card_count().await, 0);
    engine.cleanup().await.unwrap();
    std::fs::remove_dir_all(adapter_dir).unwrap();
    server.stop().await;
}
