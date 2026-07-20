// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use base64::Engine as _;
use dynamo_backend_common::{
    DynamoError, LLMEngineOutput, LLMEngineOutputExt, MAX_PREPROCESSED_MM_BYTES,
    MAX_PREPROCESSED_MM_FEATURES, MAX_PREPROCESSED_MM_MODALITY_BYTES,
    MAX_PREPROCESSED_MM_ROUTING_HASH_BYTES, MultimodalData, PreprocessedRequest,
    preprocessed_mm_cache_identifier, usage,
};

use crate::client;
use crate::proto as pb;
use crate::wire::json_to_prost_struct;

const CACHE_SALT_PREFIX: &str = "dynamo-cache-salt:";
const PROMPT_LOGPROBS_HANDOFF_KEY: &str = "prompt_logprobs";

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
    logprobs: Option<i32>,
    prompt_logprobs: Option<i32>,
    logit_bias: Option<BTreeMap<u32, f32>>,
    allowed_token_ids: Option<Vec<u32>>,
    bad_words: Option<Vec<String>>,
    logprob_token_ids: Option<Vec<u32>>,
    structured_outputs: Option<TitoStructuredOutputs>,
    skip_reading_prefix_cache: Option<bool>,
    thinking_token_budget: Option<i64>,
    vllm_xargs: Option<BTreeMap<String, serde_json::Value>>,
    skip_special_tokens: Option<bool>,
    // Prime's renderer requests token IDs explicitly. TITO always returns
    // them, so this is a compatibility hint rather than a wire-level option.
    return_token_ids: Option<bool>,
    // Dynamo's OpenAI adapter currently places Prime's cache salt inside the
    // sampling object. Preserve it through the native vLLM KV parameters.
    cache_salt: Option<String>,
    n: Option<u8>,
    best_of: Option<u8>,
    use_beam_search: Option<bool>,
    length_penalty: Option<f32>,
    // Prompt prefix excluded from returned routed-expert traces.
    routed_experts_prompt_start: Option<u32>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TitoStructuredOutputs {
    json: Option<serde_json::Value>,
    regex: Option<String>,
    choice: Option<Vec<String>>,
    grammar: Option<String>,
    json_object: Option<bool>,
    #[serde(default)]
    disable_any_whitespace: bool,
    #[serde(default)]
    disable_additional_properties: bool,
    whitespace_pattern: Option<String>,
    structural_tag: Option<String>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TitoPlaceholder {
    offset: u64,
    length: u64,
    is_embed: Option<Vec<bool>>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TitoFeatures {
    mm_hashes: BTreeMap<String, Vec<String>>,
    mm_placeholders: BTreeMap<String, Vec<TitoPlaceholder>>,
    kwargs_data: Option<BTreeMap<String, Vec<Option<String>>>>,
}

#[derive(Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)]
struct TitoEnvelope {
    #[serde(default)]
    request_id: String,
    #[serde(default)]
    sampling_params: TitoSampling,
    model: Option<String>,
    #[serde(default)]
    stream: bool,
    stream_options: Option<serde_json::Value>,
    cache_salt: Option<String>,
    priority: Option<i32>,
    kv_transfer_params: Option<serde_json::Value>,
    features: Option<TitoFeatures>,
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
    if let Some(envelope) = tito.as_ref() {
        if envelope.stream {
            return Err(client::invalid_arg(
                "TITO streaming requests are not supported by the sidecar",
            ));
        }
        if envelope.stream_options.is_some() {
            return Err(client::invalid_arg(
                "TITO stream_options require streaming and are not supported",
            ));
        }
    }
    let tito_sampling = tito.as_ref().map(|value| &value.sampling_params);
    if let Some(options) = tito_sampling {
        if options.logprobs.is_some_and(|value| value < -1) {
            return Err(client::invalid_arg(
                "TITO logprobs must be non-negative or -1",
            ));
        }
        if options.prompt_logprobs.is_some_and(|value| value < -1) {
            return Err(client::invalid_arg(
                "TITO prompt_logprobs must be non-negative or -1",
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
        if let Some(token_ids) = options.logprob_token_ids.as_ref()
            && options
                .logprobs
                .is_some_and(|count| count != token_ids.len() as i32)
        {
            return Err(client::invalid_arg(
                "when both TITO logprobs and logprob_token_ids are set, logprobs must equal the token-ID count",
            ));
        }
        if options
            .vllm_xargs
            .as_ref()
            .is_some_and(|args| args.contains_key("kv_transfer_params"))
        {
            return Err(client::invalid_arg(
                "TITO vllm_xargs must not contain reserved key kv_transfer_params",
            ));
        }
        if options
            .allowed_token_ids
            .as_ref()
            .is_some_and(Vec::is_empty)
        {
            return Err(client::invalid_arg(
                "TITO allowed_token_ids must not be empty",
            ));
        }
        if options
            .routed_experts_prompt_start
            .is_some_and(|start| start as usize >= request.token_ids.len())
        {
            return Err(client::invalid_arg(
                "TITO routed_experts_prompt_start must be less than the prompt length",
            ));
        }
        let _ = options.skip_special_tokens;
        let _ = options.return_token_ids;
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
            let mut params = result.disaggregated_params.clone();
            let fields = params.as_object_mut().ok_or_else(|| {
                client::invalid_arg("prefill_result.disaggregated_params must be an object")
            })?;
            fields.remove(PROMPT_LOGPROBS_HANDOFF_KEY);
            json_to_prost_struct(&params).ok_or_else(|| {
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
    let requested_prompt_logprobs = match tito_sampling.and_then(|value| value.prompt_logprobs) {
        Some(value) => Some(value),
        None => request
            .output_options
            .prompt_logprobs
            .map(i32::try_from)
            .transpose()
            .map_err(|_| client::invalid_arg("prompt_logprobs exceeds the supported range"))?,
    };
    let prompt_logprobs = if !is_prefill && request.prefill_result.is_some() {
        if requested_prompt_logprobs.is_some()
            && request
                .prefill_result
                .as_ref()
                .and_then(|result| result.disaggregated_params.get(PROMPT_LOGPROBS_HANDOFF_KEY))
                .is_none()
        {
            return Err(client::invalid_arg(
                "decode request is missing the requested prefill prompt_logprobs handoff",
            ));
        }
        None
    } else {
        requested_prompt_logprobs
    };
    let prompt_candidates = prompt_logprobs.map(|count| pb::CandidateTokens {
        select: Some(if count == -1 {
            pb::candidate_tokens::Select::All(true)
        } else {
            pb::candidate_tokens::Select::TopN(count as u32)
        }),
    });

    let structured_output = tito_sampling
        .and_then(|value| value.structured_outputs.as_ref())
        .map(build_structured_output)
        .transpose()?;
    let vllm_xargs_json = tito_sampling
        .and_then(|value| value.vllm_xargs.as_ref())
        .map(|value| {
            serde_json::to_vec(value)
                .map_err(|error| client::invalid_arg(format!("invalid TITO vllm_xargs: {error}")))
        })
        .transpose()?;
    let media = build_media(request)?;
    let mm_features = tito
        .as_ref()
        .and_then(|value| value.features.as_ref())
        .map(|features| build_mm_features(features, request.token_ids.len()))
        .transpose()?
        .unwrap_or_default();
    if !media.is_empty() && !mm_features.is_empty() {
        return Err(client::invalid_arg(
            "raw media and preprocessed multimodal features are mutually exclusive",
        ));
    }
    let top_k = tito_sampling
        .and_then(|value| value.top_k)
        .or(sampling.top_k);

    if !lora_name.is_empty() && (!media.is_empty() || !mm_features.is_empty()) {
        return Err(client::invalid_arg(
            "native gRPC does not yet advertise tower-LoRA multimodal cache semantics; multimodal requests with LoRA are unsupported",
        ));
    }

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
            top_k,
            top_p: tito_sampling
                .and_then(|value| value.top_p)
                .or(sampling.top_p),
            min_p: tito_sampling
                .and_then(|value| value.min_p)
                .or(sampling.min_p),
            seed: tito_sampling.and_then(|value| value.seed).or(sampling.seed),
        }),
        decoding: Some(pb::DecodingParameters {
            presence_penalty: tito_sampling
                .and_then(|value| value.presence_penalty)
                .or(sampling.presence_penalty),
            frequency_penalty: tito_sampling
                .and_then(|value| value.frequency_penalty)
                .or(sampling.frequency_penalty),
            repetition_penalty: tito_sampling
                .and_then(|value| value.repetition_penalty)
                .or(sampling.repetition_penalty),
            logit_bias: tito_sampling
                .and_then(|value| value.logit_bias.as_ref())
                .map(|bias| bias.iter().map(|(token, value)| (*token, *value)).collect())
                .unwrap_or_default(),
            allowed_token_ids: tito_sampling
                .and_then(|value| value.allowed_token_ids.clone())
                .unwrap_or_default(),
            structured_output,
            structured_output_disable_any_whitespace: tito_sampling
                .and_then(|value| value.structured_outputs.as_ref())
                .is_some_and(|value| value.disable_any_whitespace),
            structured_output_disable_additional_properties: tito_sampling
                .and_then(|value| value.structured_outputs.as_ref())
                .is_some_and(|value| value.disable_additional_properties),
            structured_output_whitespace_pattern: tito_sampling
                .and_then(|value| value.structured_outputs.as_ref())
                .and_then(|value| value.whitespace_pattern.clone()),
            bad_words: tito_sampling
                .and_then(|value| value.bad_words.clone())
                .unwrap_or_default(),
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
            thinking_token_budget: tito_sampling
                .and_then(|value| value.thinking_token_budget)
                .or(request.stop_conditions.max_thinking_tokens.map(i64::from)),
        }),
        response: Some(pb::ResponseOptions {
            prompt_token_ids: prompt_logprobs.is_some(),
            prompt_logprobs: prompt_logprobs.is_some(),
            prompt_candidates,
            output_text: Some(false),
            output_token_ids: true,
            output_logprobs: tito_sampling
                .and_then(|value| value.logprobs)
                .or(request.output_options.logprobs.map(|value| value as i32))
                .or_else(|| {
                    tito_sampling
                        .and_then(|value| value.logprob_token_ids.as_ref())
                        .map(|value| value.len() as i32)
                })
                .is_some(),
            output_candidates: output_candidates(
                tito_sampling.and_then(|value| value.logprobs),
                tito_sampling.and_then(|value| value.logprob_token_ids.as_deref()),
                request.output_options.logprobs,
            ),
        }),
        kv: Some(pb::KvCacheParameters {
            bypass_prefix_cache: tito_sampling
                .and_then(|value| value.skip_reading_prefix_cache)
                .unwrap_or(false),
            cache_salt: tito
                .as_ref()
                .and_then(|value| {
                    value
                        .cache_salt
                        .as_ref()
                        .or(value.sampling_params.cache_salt.as_ref())
                })
                .or_else(|| {
                    request
                        .routing
                        .as_ref()
                        .and_then(|routing| routing.cache_namespace.as_ref())
                })
                .map(|value| format!("{CACHE_SALT_PREFIX}{value}"))
                .unwrap_or_default(),
            kv_transfer_params,
        }),
        truncate_prompt_tokens: 0,
        priority: priority.unwrap_or(0),
        media,
        lora_name,
        vllm_xargs_json,
        mm_features,
        routed_experts_prompt_start: tito_sampling
            .and_then(|value| value.routed_experts_prompt_start)
            .unwrap_or_default(),
    })
}

fn build_structured_output(
    value: &TitoStructuredOutputs,
) -> Result<pb::decoding_parameters::StructuredOutput, DynamoError> {
    use pb::decoding_parameters::StructuredOutput;

    let mut constraints = Vec::new();
    if let Some(json) = value.json.as_ref() {
        let schema = match json {
            serde_json::Value::String(schema) => schema.clone(),
            schema => serde_json::to_string(schema).map_err(|error| {
                client::invalid_arg(format!("invalid TITO structured_outputs.json: {error}"))
            })?,
        };
        constraints.push(StructuredOutput::Json(schema));
    }
    if let Some(regex) = value.regex.as_ref() {
        constraints.push(StructuredOutput::Regex(regex.clone()));
    }
    if let Some(choice) = value.choice.as_ref() {
        constraints.push(StructuredOutput::Choice(
            pb::decoding_parameters::StringChoices {
                choices: choice.clone(),
            },
        ));
    }
    if let Some(grammar) = value.grammar.as_ref() {
        constraints.push(StructuredOutput::Grammar(grammar.clone()));
    }
    if let Some(json_object) = value.json_object {
        if !json_object {
            return Err(client::invalid_arg(
                "TITO structured_outputs.json_object must be true when set",
            ));
        }
        constraints.push(StructuredOutput::JsonObject(true));
    }
    if let Some(tag) = value.structural_tag.as_ref() {
        constraints.push(StructuredOutput::StructuralTag(tag.clone()));
    }
    if constraints.len() != 1 {
        return Err(client::invalid_arg(
            "TITO structured_outputs must contain exactly one constraint",
        ));
    }
    Ok(constraints.pop().expect("one constraint checked above"))
}

fn output_candidates(
    requested: Option<i32>,
    token_ids: Option<&[u32]>,
    framework_requested: Option<u32>,
) -> Option<pb::CandidateTokens> {
    let select = if let Some(token_ids) = token_ids {
        pb::candidate_tokens::Select::TokenIds(pb::TokenIds {
            ids: token_ids.to_vec(),
        })
    } else if requested == Some(-1) {
        pb::candidate_tokens::Select::All(true)
    } else if let Some(top_n) = requested.and_then(|value| u32::try_from(value).ok()) {
        pb::candidate_tokens::Select::TopN(top_n)
    } else if let Some(top_n) = framework_requested {
        pb::candidate_tokens::Select::TopN(top_n)
    } else {
        return None;
    };
    Some(pb::CandidateTokens {
        select: Some(select),
    })
}

fn build_mm_features(
    features: &TitoFeatures,
    prompt_len: usize,
) -> Result<Vec<pb::PreprocessedMultimodalFeature>, DynamoError> {
    if features
        .mm_hashes
        .keys()
        .ne(features.mm_placeholders.keys())
    {
        return Err(client::invalid_arg(
            "features.mm_hashes and features.mm_placeholders must have identical modality keys",
        ));
    }
    let kwargs_data = features.kwargs_data.as_ref().ok_or_else(|| {
        client::invalid_arg(
            "features.kwargs_data is required; unverified client-asserted multimodal cache hits are not supported",
        )
    })?;
    if features.mm_hashes.keys().ne(kwargs_data.keys()) {
        return Err(client::invalid_arg(
            "features.kwargs_data must have the same modality keys as features.mm_hashes",
        ));
    }

    let feature_count = features.mm_hashes.values().map(Vec::len).sum::<usize>();
    if feature_count == 0 || feature_count > MAX_PREPROCESSED_MM_FEATURES {
        return Err(client::invalid_arg(format!(
            "features must contain between 1 and {MAX_PREPROCESSED_MM_FEATURES} multimodal items"
        )));
    }

    let mut total_bytes = 0usize;
    let mut result = Vec::with_capacity(feature_count);
    let mut ranges = Vec::with_capacity(feature_count);
    for (modality, hashes) in &features.mm_hashes {
        if modality.is_empty() || modality.len() > MAX_PREPROCESSED_MM_MODALITY_BYTES {
            return Err(client::invalid_arg(
                "feature modality names must contain between 1 and 64 bytes",
            ));
        }
        let placeholders = &features.mm_placeholders[modality];
        let encoded_items = &kwargs_data[modality];
        if hashes.len() != placeholders.len() || hashes.len() != encoded_items.len() {
            return Err(client::invalid_arg(format!(
                "feature lists for modality {modality:?} must have equal lengths"
            )));
        }
        for ((routing_hash, position), encoded) in
            hashes.iter().zip(placeholders).zip(encoded_items)
        {
            if routing_hash.is_empty()
                || routing_hash.len() > MAX_PREPROCESSED_MM_ROUTING_HASH_BYTES
            {
                return Err(client::invalid_arg(
                    "multimodal hashes must contain between 1 and 512 bytes",
                ));
            }
            let offset = usize::try_from(position.offset)
                .map_err(|_| client::invalid_arg("multimodal feature offset is too large"))?;
            let length = usize::try_from(position.length)
                .map_err(|_| client::invalid_arg("multimodal feature length is too large"))?;
            if length == 0 {
                return Err(client::invalid_arg(
                    "multimodal feature length must be positive",
                ));
            }
            let end = offset
                .checked_add(length)
                .filter(|end| *end <= prompt_len)
                .ok_or_else(|| client::invalid_arg("multimodal feature range exceeds token_ids"))?;
            if position
                .is_embed
                .as_ref()
                .is_some_and(|mask| mask.len() != length)
            {
                return Err(client::invalid_arg(
                    "multimodal feature is_embed length must match its range length",
                ));
            }
            let encoded = encoded.as_ref().ok_or_else(|| {
                client::invalid_arg(
                    "each multimodal feature must carry inline kwargs_data; cache-hit nulls are not accepted",
                )
            })?;
            let kwargs_msgpack = base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .map_err(|error| {
                    client::invalid_arg(format!("invalid multimodal kwargs_data base64: {error}"))
                })?;
            total_bytes = total_bytes
                .checked_add(kwargs_msgpack.len())
                .ok_or_else(|| client::invalid_arg("multimodal feature payload is too large"))?;
            if total_bytes > MAX_PREPROCESSED_MM_BYTES {
                return Err(client::invalid_arg(format!(
                    "multimodal feature payload exceeds {} MiB",
                    MAX_PREPROCESSED_MM_BYTES / (1024 * 1024)
                )));
            }
            let cache_identifier = preprocessed_mm_cache_identifier(modality, &kwargs_msgpack);
            ranges.push((offset, end));
            result.push(pb::PreprocessedMultimodalFeature {
                modality: modality.clone(),
                // The renderer hash is only a routing hint. Bind vLLM's receiver
                // cache to the canonical payload identity instead.
                mm_hash: cache_identifier.clone(),
                position: Some(pb::MultimodalPlaceholder {
                    offset: position.offset,
                    length: position.length,
                    is_embed: position.is_embed.clone().unwrap_or_default(),
                }),
                kwargs_msgpack: Some(kwargs_msgpack),
                cache_identifier,
            });
        }
    }

    ranges.sort_unstable_by_key(|range| range.0);
    if ranges.windows(2).any(|pair| pair[0].1 > pair[1].0) {
        return Err(client::invalid_arg(
            "multimodal feature ranges must not overlap",
        ));
    }
    Ok(result)
}

pub(crate) fn handed_off_prompt_logprobs(
    request: &PreprocessedRequest,
) -> Result<Option<serde_json::Value>, DynamoError> {
    let Some(value) = request
        .prefill_result
        .as_ref()
        .and_then(|result| result.disaggregated_params.get(PROMPT_LOGPROBS_HANDOFF_KEY))
    else {
        return Ok(None);
    };
    let positions = value
        .as_array()
        .ok_or_else(|| client::invalid_arg("prefill prompt_logprobs handoff must be an array"))?;
    if positions.is_empty() || !positions[0].is_null() {
        return Err(client::invalid_arg(
            "prefill prompt_logprobs handoff must begin with null",
        ));
    }
    Ok(Some(value.clone()))
}

pub(crate) fn attach_prompt_logprobs_to_handoff(
    mut handoff: serde_json::Value,
    prompt_logprobs: serde_json::Value,
) -> Result<serde_json::Value, DynamoError> {
    let fields = handoff.as_object_mut().ok_or_else(|| {
        client::protocol_error("prefill kv_transfer_params must be a JSON object")
    })?;
    fields.insert(PROMPT_LOGPROBS_HANDOFF_KEY.to_string(), prompt_logprobs);
    Ok(handoff)
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
    if request.output_options.skip_special_tokens == Some(false) {
        return Err(client::invalid_arg(
            "skip_special_tokens=false is not supported by the vLLM sidecar",
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
