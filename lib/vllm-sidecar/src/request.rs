// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::{
    DynamoError, LLMEngineOutput, LLMEngineOutputExt, MultimodalData, PreprocessedRequest, usage,
};

use crate::client;
use crate::proto as pb;
use crate::wire::json_to_prost_struct;

const CACHE_SALT_PREFIX: &str = "dynamo-cache-salt:";

#[derive(Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TitoSampling {
    max_tokens: Option<u32>,
    min_tokens: Option<u32>,
    temperature: Option<f32>,
    top_k: Option<i32>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    seed: Option<i64>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    repetition_penalty: Option<f32>,
    stop: Option<Vec<String>>,
    stop_token_ids: Option<Vec<u32>>,
    ignore_eos: Option<bool>,
    include_stop_str_in_output: Option<bool>,
    logprobs: Option<u32>,
    prompt_logprobs: Option<u32>,
    skip_special_tokens: Option<bool>,
    n: Option<u8>,
    best_of: Option<u8>,
    use_beam_search: Option<bool>,
    length_penalty: Option<f32>,
    // Prime's renderer adds this routing-only hint on subsequent turns. The
    // native protocol does not yet carry expert traces, but dense models can
    // safely accept the hint without changing generation semantics.
    routed_experts_prompt_start: Option<u32>,
}

#[derive(Default, serde::Deserialize)]
struct TitoEnvelope {
    #[serde(default)]
    sampling_params: TitoSampling,
    cache_salt: Option<String>,
    priority: Option<i32>,
    kv_transfer_params: Option<serde_json::Value>,
}

fn tito_envelope(request: &PreprocessedRequest) -> Result<Option<TitoEnvelope>, DynamoError> {
    let Some(value) = request
        .extra_args
        .as_ref()
        .and_then(|value| value.get("vllm_tito"))
    else {
        return Ok(None);
    };
    serde_json::from_value(value.clone())
        .map(Some)
        .map_err(|error| client::invalid_arg(format!("invalid extra_args.vllm_tito: {error}")))
}

pub(crate) fn build_generate_request(
    request: &PreprocessedRequest,
    request_id: &str,
    is_prefill: bool,
) -> Result<pb::GenerateRequest, DynamoError> {
    validate_generate_request(request)?;
    let tito = tito_envelope(request)?;
    let tito_sampling = tito.as_ref().map(|value| &value.sampling_params);
    if let Some(options) = tito_sampling {
        if options.prompt_logprobs.is_some() {
            return Err(client::invalid_arg(
                "TITO prompt_logprobs are not yet supported by the vLLM sidecar",
            ));
        }
        if options.n.unwrap_or(1) != 1 || options.best_of.unwrap_or(1) != 1 {
            return Err(client::invalid_arg("TITO n and best_of must both be 1"));
        }
        if options.use_beam_search.unwrap_or(false) {
            return Err(client::invalid_arg("TITO beam search is not supported"));
        }
        if options
            .length_penalty
            .is_some_and(|value| (value - 1.0).abs() > f32::EPSILON)
        {
            return Err(client::invalid_arg(
                "TITO non-default length_penalty is not supported",
            ));
        }
        let _ = options.skip_special_tokens;
        let _ = options.routed_experts_prompt_start;
    }
    let sampling = &request.sampling_options;
    let max_tokens = if is_prefill {
        Some(1)
    } else {
        tito_sampling
            .and_then(|value| value.max_tokens)
            .or(request.stop_conditions.max_tokens)
    };
    let min_tokens = if is_prefill {
        Some(1)
    } else {
        tito_sampling
            .and_then(|value| value.min_tokens)
            .or(request.stop_conditions.min_tokens)
    };

    let mut stop_token_ids = Vec::new();
    for ids in [
        request.stop_conditions.stop_token_ids.as_ref(),
        request.stop_conditions.stop_token_ids_hidden.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        stop_token_ids.extend(ids.iter().copied());
    }
    if let Some(ids) = tito_sampling.and_then(|value| value.stop_token_ids.as_ref()) {
        stop_token_ids = ids.clone();
    }

    let framework_kv_transfer_params = request
        .prefill_result
        .as_ref()
        .map(|result| {
            json_to_prost_struct(&result.disaggregated_params).ok_or_else(|| {
                client::invalid_arg("prefill_result.disaggregated_params must be an object")
            })
        })
        .transpose()?;
    let caller_kv_transfer_params = tito
        .as_ref()
        .and_then(|value| value.kv_transfer_params.as_ref())
        .map(|value| {
            json_to_prost_struct(value).ok_or_else(|| {
                client::invalid_arg("extra_args.vllm_tito.kv_transfer_params must be an object")
            })
        })
        .transpose()?;
    if framework_kv_transfer_params.is_some() && caller_kv_transfer_params.is_some() {
        return Err(client::invalid_arg(
            "kv_transfer_params cannot be supplied by both TITO and the prefill handoff",
        ));
    }
    let kv_transfer_params = framework_kv_transfer_params.or(caller_kv_transfer_params);
    let priority = request
        .routing
        .as_ref()
        .and_then(|routing| routing.priority);
    let priority = tito.as_ref().and_then(|value| value.priority).or(priority);
    let lora_name = request
        .routing
        .as_ref()
        .and_then(|routing| routing.lora_name.clone())
        .unwrap_or_default();

    Ok(pb::GenerateRequest {
        request_id: request_id.to_string(),
        model: request.model.clone(),
        prompt: Some(pb::generate_request::Prompt::TokenIds(pb::TokenIds {
            ids: request.token_ids.clone(),
        })),
        temperature: tito_sampling
            .and_then(|value| value.temperature)
            .or(sampling.temperature),
        sampling: Some(pb::RandomSampling {
            num_sequences: u32::from(
                tito_sampling
                    .and_then(|value| value.n)
                    .or(sampling.n)
                    .unwrap_or(1),
            ),
            top_k: tito_sampling
                .and_then(|value| value.top_k)
                .or(sampling.top_k)
                .filter(|value| *value > 0)
                .and_then(|value| u32::try_from(value).ok())
                .unwrap_or(0),
            top_p: tito_sampling
                .and_then(|value| value.top_p)
                .or(sampling.top_p)
                .unwrap_or(0.0),
            min_p: tito_sampling
                .and_then(|value| value.min_p)
                .or(sampling.min_p)
                .unwrap_or(0.0),
            seed: tito_sampling.and_then(|value| value.seed).or(sampling.seed),
        }),
        decoding: Some(pb::DecodingParameters {
            presence_penalty: tito_sampling
                .and_then(|value| value.presence_penalty)
                .or(sampling.presence_penalty)
                .unwrap_or(0.0),
            frequency_penalty: tito_sampling
                .and_then(|value| value.frequency_penalty)
                .or(sampling.frequency_penalty)
                .unwrap_or(0.0),
            repetition_penalty: tito_sampling
                .and_then(|value| value.repetition_penalty)
                .or(sampling.repetition_penalty)
                .unwrap_or(0.0),
            logit_bias: Default::default(),
            allowed_token_ids: Vec::new(),
            structured_output: None,
        }),
        stopping: Some(pb::StoppingCriteria {
            max_new_tokens: max_tokens.unwrap_or(0),
            min_new_tokens: min_tokens.unwrap_or(0),
            stop_token_ids,
            stop_strings: tito_sampling
                .and_then(|value| value.stop.clone())
                .or_else(|| request.stop_conditions.stop.clone())
                .unwrap_or_default(),
            include_stop_strings: tito_sampling
                .and_then(|value| value.include_stop_str_in_output)
                .or(sampling.include_stop_str_in_output)
                .unwrap_or(false),
            ignore_eos: tito_sampling
                .and_then(|value| value.ignore_eos)
                .or(request.stop_conditions.ignore_eos)
                .unwrap_or(false),
        }),
        response: Some(pb::ResponseOptions {
            prompt_token_ids: false,
            prompt_logprobs: false,
            prompt_candidates: None,
            output_text: Some(false),
            output_token_ids: true,
            output_logprobs: tito_sampling
                .and_then(|value| value.logprobs)
                .or(request.output_options.logprobs)
                .is_some(),
            output_candidates: tito_sampling
                .and_then(|value| value.logprobs)
                .or(request.output_options.logprobs)
                .map(|top_n| pb::CandidateTokens {
                    select: Some(pb::candidate_tokens::Select::TopN(top_n)),
                }),
        }),
        kv: Some(pb::KvCacheParameters {
            bypass_prefix_cache: false,
            cache_salt: tito
                .as_ref()
                .and_then(|value| value.cache_salt.as_ref())
                .or_else(|| {
                    request
                        .routing
                        .as_ref()
                        .and_then(|routing| routing.cache_namespace.as_ref())
                })
                .map(|value| format!("{CACHE_SALT_PREFIX}{value}"))
                .unwrap_or_default(),
            kv_transfer_params,
            ec_transfer_params: None,
        }),
        truncate_prompt_tokens: 0,
        priority: priority.unwrap_or(0),
        media: build_media(request)?,
        lora_name,
    })
}

fn validate_generate_request(request: &PreprocessedRequest) -> Result<(), DynamoError> {
    if request.prompt_embeds.is_some() {
        return Err(client::invalid_arg(
            "prompt_embeds are not supported by the vLLM sidecar",
        ));
    }
    if request
        .stop_conditions
        .stop_token_ids_visible
        .as_ref()
        .is_some_and(|ids| !ids.is_empty())
    {
        return Err(client::invalid_arg(
            "visible stop token IDs are not supported by the vLLM sidecar",
        ));
    }
    let sampling = &request.sampling_options;
    if sampling.n.unwrap_or(1) != 1 {
        return Err(client::invalid_arg("n must be 1 for the vLLM sidecar"));
    }
    if sampling.best_of.unwrap_or(1) != 1 {
        return Err(client::invalid_arg(
            "best_of must be 1 for the vLLM sidecar",
        ));
    }
    if sampling.use_beam_search.unwrap_or(false) {
        return Err(client::invalid_arg(
            "beam search is not supported by the vLLM sidecar",
        ));
    }
    if let Some(length_penalty) = sampling.length_penalty
        && (length_penalty - 1.0).abs() > f32::EPSILON
    {
        return Err(client::invalid_arg(
            "non-default length_penalty is not supported by the vLLM sidecar",
        ));
    }
    if sampling.guided_decoding.is_some() {
        return Err(client::invalid_arg(
            "guided decoding is not supported by the vLLM sidecar",
        ));
    }
    if request.output_options.prompt_logprobs.is_some() {
        return Err(client::invalid_arg(
            "prompt_logprobs are not supported by the vLLM sidecar",
        ));
    }
    if request.output_options.skip_special_tokens == Some(false) {
        return Err(client::invalid_arg(
            "skip_special_tokens=false is not supported by the vLLM sidecar",
        ));
    }
    if request.stop_conditions.max_thinking_tokens.is_some() {
        return Err(client::invalid_arg(
            "max_thinking_tokens is not supported by the vLLM sidecar",
        ));
    }
    Ok(())
}

fn modality_for_key(key: &str) -> pb::Modality {
    match key {
        "image_url" => pb::Modality::Image,
        "video_url" => pb::Modality::Video,
        "audio_url" => pb::Modality::Audio,
        _ => pb::Modality::Unspecified,
    }
}

fn media_source_from_str(source: &str) -> pb::media_item::Source {
    if source.starts_with("data:") {
        pb::media_item::Source::DataUri(source.to_string())
    } else {
        pb::media_item::Source::Url(source.to_string())
    }
}

fn build_media(request: &PreprocessedRequest) -> Result<Vec<pb::MediaItem>, DynamoError> {
    let Some(map) = request.multi_modal_data.as_ref() else {
        return Ok(Vec::new());
    };
    let mut media = Vec::new();
    for key in ["image_url", "video_url", "audio_url"] {
        let Some(items) = map.get(key) else { continue };
        let modality = modality_for_key(key);
        for item in items {
            let source = match item {
                MultimodalData::Url(url) => media_source_from_str(url.as_str()),
                MultimodalData::RawUrl(source) => media_source_from_str(source),
                MultimodalData::Decoded(_) => {
                    return Err(client::invalid_arg(
                        "vllm-sidecar received a pre-decoded RDMA media descriptor; the sidecar \
                         has no NIXL agent to dereference it. Run the frontend in URL-passthrough \
                         mode (set the model's media_decoder to null).",
                    ));
                }
            };
            media.push(pb::MediaItem {
                modality: modality as i32,
                source: Some(source),
                mime_type: String::new(),
                uuid: String::new(),
            });
        }
    }
    Ok(media)
}

pub(crate) fn finish_output(
    reason: pb::finish_info::FinishReason,
    prompt_tokens: u32,
    generated: u32,
    disaggregated_params: Option<serde_json::Value>,
) -> LLMEngineOutput {
    let mut output = match reason {
        pb::finish_info::FinishReason::Length => LLMEngineOutput::length(),
        pb::finish_info::FinishReason::Aborted => LLMEngineOutput::cancelled(),
        _ => LLMEngineOutput::stop(),
    }
    .with_usage(usage(prompt_tokens, generated));
    output.disaggregated_params = disaggregated_params;
    output
}
