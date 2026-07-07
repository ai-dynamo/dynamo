// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::{
    DynamoError, LLMEngineOutput, LLMEngineOutputExt, MultimodalData, PreprocessedRequest, usage,
};

use crate::client;
use crate::proto as pb;
use crate::wire::disagg_json_to_kv_session;

pub(crate) fn build_generate_request(
    request: &PreprocessedRequest,
    request_id: &str,
    is_prefill: bool,
) -> Result<pb::GenerateRequest, DynamoError> {
    validate_generate_request(request)?;
    let sampling = &request.sampling_options;
    let max_tokens = if is_prefill {
        Some(1)
    } else {
        request.stop_conditions.max_tokens
    };
    let min_tokens = if is_prefill {
        Some(1)
    } else {
        request.stop_conditions.min_tokens
    };

    let proto_sampling = pb::SamplingParams {
        temperature: sampling.temperature.map(f64::from),
        top_p: sampling.top_p.map(f64::from),
        top_k: sampling.top_k,
        frequency_penalty: sampling.frequency_penalty.map(f64::from),
        presence_penalty: sampling.presence_penalty.map(f64::from),
        max_tokens,
        seed: sampling.seed,
        ignore_eos: request.stop_conditions.ignore_eos.unwrap_or(false),
        min_p: sampling.min_p.map(f64::from),
        repetition_penalty: sampling.repetition_penalty.map(f64::from),
        min_tokens,
    };

    let mut stop = Vec::new();
    if let Some(strings) = &request.stop_conditions.stop {
        for text in strings {
            stop.push(pb::StopCondition {
                condition: Some(pb::stop_condition::Condition::StopText(text.clone())),
            });
        }
    }
    for ids in [
        request.stop_conditions.stop_token_ids.as_ref(),
        request.stop_conditions.stop_token_ids_hidden.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        for id in ids {
            stop.push(pb::StopCondition {
                condition: Some(pb::stop_condition::Condition::StopTokenId(*id)),
            });
        }
    }

    let kv_session = request
        .prefill_result
        .as_ref()
        .map(|result| disagg_json_to_kv_session(&result.disaggregated_params))
        .transpose()?;
    let data_parallel_rank = request.routing.as_ref().and_then(|routing| {
        if is_prefill {
            routing.prefill_dp_rank.or(routing.dp_rank)
        } else {
            routing.dp_rank
        }
    });
    let priority = request
        .routing
        .as_ref()
        .and_then(|routing| routing.priority);
    let lora_name = request
        .routing
        .as_ref()
        .and_then(|routing| routing.lora_name.clone())
        .unwrap_or_default();

    Ok(pb::GenerateRequest {
        request_id: request_id.to_string(),
        model: request.model.clone(),
        input: Some(pb::generate_request::Input::TokenIds(pb::TokenIds {
            ids: request.token_ids.clone(),
        })),
        sampling: Some(proto_sampling),
        stop,
        stream: true,
        media: build_media(request)?,
        lora_name,
        data_parallel_rank,
        priority,
        kv_session,
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
    if sampling.include_stop_str_in_output.unwrap_or(false) {
        return Err(client::invalid_arg(
            "include_stop_str_in_output is not supported by the vLLM sidecar",
        ));
    }
    if sampling.guided_decoding.is_some() {
        return Err(client::invalid_arg(
            "guided decoding is not supported by the vLLM sidecar",
        ));
    }
    if request.output_options.logprobs.is_some() || request.output_options.prompt_logprobs.is_some()
    {
        return Err(client::invalid_arg(
            "logprobs are not supported by the vLLM sidecar",
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
    reason: pb::FinishReason,
    prompt_tokens: u32,
    generated: u32,
    disaggregated_params: Option<serde_json::Value>,
) -> LLMEngineOutput {
    let mut output = match reason {
        pb::FinishReason::Length => LLMEngineOutput::length(),
        pb::FinishReason::Cancelled => LLMEngineOutput::cancelled(),
        pb::FinishReason::Error => {
            LLMEngineOutput::error("engine reported error finish reason".to_string())
        }
        _ => LLMEngineOutput::stop(),
    }
    .with_usage(usage(prompt_tokens, generated));
    output.disaggregated_params = disaggregated_params;
    output
}
