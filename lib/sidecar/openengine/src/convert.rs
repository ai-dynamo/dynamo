// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::collections::{HashMap, VecDeque};

use dynamo_backend_common::{
    DynamoError, GuidedDecodingOptions, LLMEngineOutput, PreprocessedRequest, TopLogprob,
};
use dynamo_llm::protocols::common::preprocessor::MultimodalData;
use tonic::metadata::{Ascii, MetadataKey, MetadataMap, MetadataValue};

use crate::client;
use crate::proto as pb;

pub fn generate_metadata(
    request: &PreprocessedRequest,
    context: &BTreeMap<String, String>,
    is_prefill: bool,
) -> Result<MetadataMap, DynamoError> {
    let mut metadata = MetadataMap::new();
    for (key, value) in context {
        insert_metadata(&mut metadata, key, value)?;
    }
    for annotation in &request.annotations {
        if let Some((key, value)) = annotation.split_once(':') {
            insert_metadata(&mut metadata, key, value)?;
        }
    }
    if let Some(routing) = request.routing.as_ref() {
        if let Some(priority) = routing.priority {
            insert_metadata(&mut metadata, "openengine-priority", &priority.to_string())?;
        }
        let rank = if is_prefill {
            routing.prefill_dp_rank
        } else {
            routing.dp_rank
        };
        if let Some(rank) = rank {
            insert_metadata(
                &mut metadata,
                "openengine-target-dp-rank",
                &rank.to_string(),
            )?;
        }
    }
    Ok(metadata)
}

fn insert_metadata(metadata: &mut MetadataMap, key: &str, value: &str) -> Result<(), DynamoError> {
    let key = MetadataKey::<Ascii>::from_bytes(key.as_bytes()).map_err(|error| {
        client::invalid_arg(format!("invalid OpenEngine metadata key `{key}`: {error}"))
    })?;
    let value = MetadataValue::<Ascii>::try_from(value).map_err(|error| {
        client::invalid_arg(format!("invalid OpenEngine metadata value: {error}"))
    })?;
    metadata.insert(key, value);
    Ok(())
}

pub fn build_generate_request(
    request: &PreprocessedRequest,
    request_id: &str,
    model: &str,
    is_prefill: bool,
    supports_text_input: bool,
) -> Result<pb::GenerateRequest, DynamoError> {
    if request.prompt_embeds.is_some() {
        return Err(client::invalid_arg(
            "OpenEngine sidecar cannot forward prompt_embeds; use text/token input",
        ));
    }
    if request.encoder_result.is_some() {
        return Err(client::invalid_arg(
            "OpenEngine sidecar does not support encoder-stage handoffs",
        ));
    }

    // Context-first multimodal decode still needs the original ordered media
    // and processor options (for example, TRT-LLM recomputes mRoPE metadata
    // locally). Capability validation in the engine rejects unsupported P/D
    // modalities before this conversion runs.
    let media = build_media(request)?;
    let media_options = build_media_options(request, &media)?;
    let input = request
        .extra_args
        .as_ref()
        .filter(|_| supports_text_input && !media.is_empty())
        .and_then(|args| args.get("formatted_prompt"))
        .and_then(serde_json::Value::as_str)
        .map(|prompt| pb::generate_request::Input::Prompt(prompt.to_string()))
        .unwrap_or_else(|| {
            pb::generate_request::Input::TokenIds(pb::TokenIds {
                ids: request.token_ids.clone(),
            })
        });
    let sampling = &request.sampling_options;
    let stopping = &request.stop_conditions;
    let conditions = stopping
        .stop
        .iter()
        .flatten()
        .cloned()
        .map(|value| pb::StopCondition {
            condition: Some(pb::stop_condition::Condition::StopText(value)),
        })
        .chain(
            stopping
                .stop_token_ids
                .iter()
                .flatten()
                .copied()
                .map(|value| pb::StopCondition {
                    condition: Some(pb::stop_condition::Condition::StopTokenId(value)),
                }),
        )
        .chain(
            stopping
                .stop_token_ids_hidden
                .iter()
                .flatten()
                .copied()
                .map(|value| pb::StopCondition {
                    condition: Some(pb::stop_condition::Condition::StopTokenId(value)),
                }),
        )
        .collect();

    let kv_session = request
        .prefill_result
        .as_ref()
        .map(|result| disagg_json_to_kv_session(&result.disaggregated_params))
        .transpose()?;
    let routing = request.routing.as_ref();
    let output = &request.output_options;

    Ok(pb::GenerateRequest {
        request_id: request_id.to_string(),
        model: model.to_string(),
        input: Some(input),
        sampling: Some(pb::SamplingParams {
            temperature: sampling.temperature.map(f64::from),
            top_p: sampling.top_p.map(f64::from),
            top_k: sampling.top_k,
            min_p: sampling.min_p.map(f64::from),
            frequency_penalty: sampling.frequency_penalty.map(f64::from),
            presence_penalty: sampling.presence_penalty.map(f64::from),
            repetition_penalty: sampling.repetition_penalty.map(f64::from),
            seed: sampling.seed.and_then(|seed| u64::try_from(seed).ok()),
            num_sequences: sampling.n.map(u32::from),
        }),
        stopping: Some(pb::StoppingOptions {
            max_tokens: if is_prefill {
                Some(1)
            } else {
                stopping.max_tokens
            },
            min_tokens: stopping.min_tokens,
            conditions,
            ignore_eos: stopping.ignore_eos,
            include_stop_in_output: sampling.include_stop_str_in_output,
        }),
        response: Some(pb::ResponseOptions {
            return_prompt_logprobs: output.prompt_logprobs.map(|_| true),
            prompt_candidates: output.prompt_logprobs.map(top_n_candidates),
            return_output_logprobs: output.logprobs.map(|_| true),
            output_candidates: output.logprobs.map(top_n_candidates),
            prompt_logprob_start: None,
        }),
        kv: Some(pb::KvOptions {
            session: kv_session,
            bypass_prefix_cache: prefix_cache_bypass(request),
            cache_salt: routing.and_then(|value| value.cache_namespace.clone()),
        }),
        guided: sampling
            .guided_decoding
            .as_ref()
            .map(build_guided)
            .transpose()?,
        media,
        lora_name: routing
            .and_then(|value| value.lora_name.clone())
            .unwrap_or_default(),
        extra: None,
        media_options,
    })
}

fn top_n_candidates(top_n: u32) -> pb::CandidateTokenSelection {
    pb::CandidateTokenSelection {
        selection: Some(pb::candidate_token_selection::Selection::TopN(top_n)),
    }
}

fn prefix_cache_bypass(request: &PreprocessedRequest) -> Option<bool> {
    request.extra_args.as_ref().and_then(|args| {
        args.get("bypass_prefix_cache")
            .and_then(serde_json::Value::as_bool)
            .or_else(|| {
                args.get("disable_prefix_cache")
                    .and_then(serde_json::Value::as_bool)
            })
    })
}

fn build_guided(value: &GuidedDecodingOptions) -> Result<pb::GuidedDecoding, DynamoError> {
    let guide = if let Some(json) = value.json.as_ref() {
        Some(if json.is_null() {
            pb::guided_decoding::Guide::JsonObject(pb::JsonObjectConstraint {})
        } else {
            pb::guided_decoding::Guide::JsonSchema(
                serde_json::to_string(json)
                    .map_err(|error| client::invalid_arg(format!("guided JSON: {error}")))?,
            )
        })
    } else if let Some(regex) = value.regex.as_ref() {
        Some(pb::guided_decoding::Guide::Regex(regex.clone()))
    } else if let Some(grammar) = value.grammar.as_ref() {
        Some(pb::guided_decoding::Guide::EbnfGrammar(grammar.clone()))
    } else if let Some(tag) = value.structural_tag.as_ref() {
        Some(pb::guided_decoding::Guide::StructuralTag(
            serde_json::to_string(tag)
                .map_err(|error| client::invalid_arg(format!("structural tag: {error}")))?,
        ))
    } else {
        value.choice.as_ref().map(|choices| {
            pb::guided_decoding::Guide::Choice(pb::ChoiceConstraint {
                choices: choices.clone(),
            })
        })
    };
    Ok(pb::GuidedDecoding {
        guide,
        backend: value.backend.clone().unwrap_or_default(),
    })
}

fn modality_for_key(key: &str) -> Option<pb::Modality> {
    match key {
        "image_url" => Some(pb::Modality::Image),
        "video_url" => Some(pb::Modality::Video),
        "audio_url" => Some(pb::Modality::Audio),
        _ => None,
    }
}

fn key_for_content_part(value: &serde_json::Value) -> Option<&'static str> {
    match value.get("type")?.as_str()? {
        "image_url" | "input_image" => Some("image_url"),
        "video_url" | "input_video" => Some("video_url"),
        "audio_url" | "input_audio" => Some("audio_url"),
        _ => None,
    }
}

/// Recover cross-modality order from the original message content preserved by
/// Dynamo's preprocessor, while taking media values from the authoritative map.
fn build_media(request: &PreprocessedRequest) -> Result<Vec<pb::MediaItem>, DynamoError> {
    let Some(map) = request.multi_modal_data.as_ref() else {
        return Ok(Vec::new());
    };
    let mut image_uuids = request
        .extra_args
        .as_ref()
        .and_then(|args| args.get("mm_hashes"))
        .map(|hashes| {
            hashes
                .as_array()
                .ok_or_else(|| client::invalid_arg("extra_args.mm_hashes must be an array"))?
                .iter()
                .map(|hash| {
                    let hash = hash.as_str().ok_or_else(|| {
                        client::invalid_arg("extra_args.mm_hashes entries must be strings")
                    })?;
                    if hash.len() != 16 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                        return Err(client::invalid_arg(
                            "extra_args.mm_hashes entries must be canonical 16-character hexadecimal strings",
                        ));
                    }
                    Ok(hash.to_ascii_lowercase())
                })
                .collect::<Result<VecDeque<_>, _>>()
        })
        .transpose()?
        .unwrap_or_default();
    let image_count = map.get("image_url").map_or(0, Vec::len);
    if !image_uuids.is_empty() && image_uuids.len() != image_count {
        return Err(client::invalid_arg(format!(
            "extra_args.mm_hashes has {} entries for {image_count} images",
            image_uuids.len()
        )));
    }

    let mut queues: HashMap<&str, VecDeque<&MultimodalData>> = map
        .iter()
        .filter_map(|(key, values)| {
            modality_for_key(key).map(|_| (key.as_str(), values.iter().collect()))
        })
        .collect();
    let mut order = Vec::new();
    let mut represented = HashMap::<&str, usize>::new();
    if let Some(messages) = request
        .extra_args
        .as_ref()
        .and_then(|args| args.get("messages"))
        .and_then(serde_json::Value::as_array)
    {
        for part in messages
            .iter()
            .filter_map(|message| message.get("content"))
            .filter_map(serde_json::Value::as_array)
            .flatten()
        {
            if let Some(key) = key_for_content_part(part) {
                let available = queues.get(key).map_or(0, VecDeque::len);
                let used = represented.entry(key).or_default();
                if *used >= available {
                    return Err(client::invalid_arg(format!(
                        "original messages contain more `{key}` items than multi_modal_data"
                    )));
                }
                order.push(key);
                *used += 1;
            }
        }
    }
    let populated_modalities = queues
        .iter()
        .filter(|(_, values)| !values.is_empty())
        .count();
    if populated_modalities > 1 {
        let missing = queues
            .iter()
            .any(|(key, values)| represented.get(key).copied().unwrap_or_default() != values.len());
        if missing {
            return Err(client::invalid_arg(
                "mixed-modality media order is unavailable; preserve every original message content part",
            ));
        }
    } else {
        // A modality-keyed map loses only cross-modality order. For a single
        // modality it is safe to append entries absent from legacy messages.
        for key in ["image_url", "video_url", "audio_url"] {
            let remaining = queues.get(key).map_or(0, VecDeque::len)
                - represented.get(key).copied().unwrap_or_default();
            order.extend(std::iter::repeat_n(key, remaining));
        }
    }

    let mut result = Vec::with_capacity(order.len());
    for key in order {
        let Some(value) = queues.get_mut(key).and_then(VecDeque::pop_front) else {
            continue;
        };
        let source = match value {
            MultimodalData::Url(value) => media_source(value.as_str()),
            MultimodalData::RawUrl(value) => media_source(value),
            MultimodalData::Decoded(_) => {
                return Err(client::invalid_arg(
                    "OpenEngine sidecar cannot dereference decoded/RDMA media; configure the frontend for URL/data passthrough",
                ));
            }
        };
        let modality = modality_for_key(key).expect("filtered modality");
        let uuid = if modality == pb::Modality::Image {
            image_uuids.pop_front().unwrap_or_default()
        } else {
            String::new()
        };
        result.push(pb::MediaItem {
            modality: modality as i32,
            source: Some(source),
            mime_type: String::new(),
            uuid,
        });
    }
    Ok(result)
}

fn media_source(value: &str) -> pb::media_item::Source {
    if value.starts_with("data:") {
        pb::media_item::Source::DataUri(value.to_string())
    } else {
        pb::media_item::Source::Url(value.to_string())
    }
}

fn build_media_options(
    request: &PreprocessedRequest,
    media: &[pb::MediaItem],
) -> Result<Option<prost_types::Struct>, DynamoError> {
    let Some(options) = request.mm_processor_kwargs.as_ref() else {
        return Ok(None);
    };
    if !options.is_object() {
        return Err(client::invalid_arg("mm_processor_kwargs must be an object"));
    }
    let already_keyed = ["image", "video", "audio"]
        .iter()
        .any(|key| options.get(key).is_some());
    let value = if already_keyed {
        options.clone()
    } else {
        let mut keyed = serde_json::Map::new();
        for modality in media
            .iter()
            .filter_map(|item| pb::Modality::try_from(item.modality).ok())
        {
            let key = match modality {
                pb::Modality::Image | pb::Modality::Unspecified => "image",
                pb::Modality::Video => "video",
                pb::Modality::Audio => "audio",
            };
            keyed
                .entry(key.to_string())
                .or_insert_with(|| options.clone());
        }
        serde_json::Value::Object(keyed)
    };
    json_to_prost_struct(&value)
        .map(Some)
        .ok_or_else(|| client::invalid_arg("media options must be an object"))
}

pub fn token_output(value: pb::TokenOutput) -> LLMEngineOutput {
    let token_ids = value.tokens.iter().map(|token| token.token_id).collect();
    let tokens = Some(
        value
            .tokens
            .iter()
            .map(|token| (!token.token.is_empty()).then(|| token.token.clone()))
            .collect(),
    );
    let has_logprobs = value.tokens.iter().any(|token| token.logprob.is_some());
    let log_probs = has_logprobs.then(|| {
        value
            .tokens
            .iter()
            .map(|token| token.logprob.unwrap_or(f64::NEG_INFINITY))
            .collect()
    });
    let top_logprobs = has_logprobs.then(|| {
        value
            .tokens
            .iter()
            .map(|token| {
                token
                    .candidates
                    .iter()
                    .map(|candidate| TopLogprob {
                        rank: candidate.rank.unwrap_or(0),
                        token_id: candidate.token_id,
                        token: (!candidate.token.is_empty()).then(|| candidate.token.clone()),
                        logprob: candidate.logprob,
                        bytes: None,
                    })
                    .collect()
            })
            .collect()
    });
    LLMEngineOutput {
        token_ids,
        tokens,
        text: (!value.text.is_empty()).then_some(value.text),
        log_probs,
        top_logprobs,
        index: value.output_index,
        ..Default::default()
    }
}

pub fn kv_session_to_disagg_json(value: pb::KvSessionRef) -> serde_json::Value {
    let bootstrap = value.bootstrap.map(|bootstrap| {
        serde_json::json!({
            "endpoint": bootstrap.endpoint.map(|endpoint| serde_json::json!({
                "host": endpoint.host,
                "port": endpoint.port,
                "protocol": endpoint.protocol,
            })),
            // This value can exceed JavaScript's exact integer range. Keep the
            // opaque Dynamo handoff lossless by carrying it as decimal text.
            "room_id": bootstrap.room_id.to_string(),
        })
    });
    serde_json::json!({
        "session_id": value.session_id,
        "transfer_backend": value.transfer_backend,
        "endpoints": value.endpoints.into_iter().map(|endpoint| serde_json::json!({
            "host": endpoint.host,
            "port": endpoint.port,
            "protocol": endpoint.protocol,
        })).collect::<Vec<_>>(),
        "dp_rank": value.dp_rank,
        "attributes_struct": value.attributes_struct.as_ref().map(prost_struct_to_json),
        "handoff_profile": value.handoff_profile,
        "bootstrap": bootstrap,
    })
}

pub fn disagg_json_to_kv_session(
    value: &serde_json::Value,
) -> Result<pb::KvSessionRef, DynamoError> {
    let object = value.as_object().ok_or_else(|| {
        client::invalid_arg("prefill_result.disaggregated_params must be an object")
    })?;
    let required_string = |key: &str| -> Result<String, DynamoError> {
        object
            .get(key)
            .and_then(serde_json::Value::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .ok_or_else(|| {
                client::invalid_arg(format!("handoff `{key}` must be a non-empty string"))
            })
    };
    let session_id = required_string("session_id")?;
    let transfer_backend = required_string("transfer_backend")?;
    let handoff_profile = required_string("handoff_profile")?;
    let dp_rank = object
        .get("dp_rank")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| client::invalid_arg("handoff `dp_rank` must be a uint32"))?;
    let endpoints_value = object
        .get("endpoints")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| client::invalid_arg("handoff `endpoints` must be an array"))?;
    let endpoints = endpoints_value
        .iter()
        .map(|endpoint| endpoint_from_json(endpoint, "handoff endpoint"))
        .collect::<Result<Vec<_>, _>>()?;
    let attributes_struct = match object.get("attributes_struct") {
        None | Some(serde_json::Value::Null) => None,
        Some(value @ serde_json::Value::Object(_)) => json_to_prost_struct(value),
        Some(_) => {
            return Err(client::invalid_arg(
                "handoff `attributes_struct` must be an object or null",
            ));
        }
    };
    let bootstrap = match object.get("bootstrap") {
        None | Some(serde_json::Value::Null) => None,
        Some(value) => {
            let bootstrap = value
                .as_object()
                .ok_or_else(|| client::invalid_arg("handoff `bootstrap` must be an object"))?;
            let endpoint = bootstrap
                .get("endpoint")
                .ok_or_else(|| client::invalid_arg("handoff bootstrap omitted `endpoint`"))?;
            let room_id = bootstrap
                .get("room_id")
                .and_then(|value| {
                    value.as_u64().or_else(|| {
                        value
                            .as_str()
                            .filter(|value| canonical_u64(value))
                            .and_then(|value| value.parse().ok())
                    })
                })
                .ok_or_else(|| {
                    client::invalid_arg(
                        "handoff bootstrap `room_id` must be a uint64 or canonical decimal string",
                    )
                })?;
            Some(pb::KvBootstrap {
                endpoint: Some(endpoint_from_json(endpoint, "handoff bootstrap endpoint")?),
                room_id,
            })
        }
    };

    Ok(pb::KvSessionRef {
        session_id,
        transfer_backend,
        endpoints,
        dp_rank,
        attributes_struct,
        handoff_profile,
        bootstrap,
    })
}

fn endpoint_from_json(
    value: &serde_json::Value,
    field: &str,
) -> Result<pb::KvEndpoint, DynamoError> {
    let endpoint = value
        .as_object()
        .ok_or_else(|| client::invalid_arg(format!("{field} must be an object")))?;
    let host = endpoint
        .get("host")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.is_empty() && value.trim() == *value)
        .ok_or_else(|| client::invalid_arg(format!("{field} host must be non-empty")))?;
    let port = endpoint
        .get("port")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| u16::try_from(value).ok())
        .filter(|value| *value > 0)
        .ok_or_else(|| client::invalid_arg(format!("{field} port must be in 1..=65535")))?;
    let protocol = endpoint
        .get("protocol")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| client::invalid_arg(format!("{field} protocol must be non-empty")))?;
    Ok(pb::KvEndpoint {
        host: host.to_owned(),
        port: u32::from(port),
        protocol: protocol.to_owned(),
    })
}

fn canonical_u64(value: &str) -> bool {
    value
        .parse::<u64>()
        .is_ok_and(|parsed| parsed.to_string() == value)
}

pub fn json_to_prost_struct(value: &serde_json::Value) -> Option<prost_types::Struct> {
    let serde_json::Value::Object(fields) = value else {
        return None;
    };
    Some(prost_types::Struct {
        fields: fields
            .iter()
            .map(|(key, value)| (key.clone(), json_to_prost_value(value)))
            .collect(),
    })
}

fn json_to_prost_value(value: &serde_json::Value) -> prost_types::Value {
    use prost_types::value::Kind;
    let kind = match value {
        serde_json::Value::Null => Kind::NullValue(0),
        serde_json::Value::Bool(value) => Kind::BoolValue(*value),
        serde_json::Value::Number(value) => Kind::NumberValue(value.as_f64().unwrap_or_default()),
        serde_json::Value::String(value) => Kind::StringValue(value.clone()),
        serde_json::Value::Array(values) => Kind::ListValue(prost_types::ListValue {
            values: values.iter().map(json_to_prost_value).collect(),
        }),
        serde_json::Value::Object(fields) => Kind::StructValue(prost_types::Struct {
            fields: fields
                .iter()
                .map(|(key, value)| (key.clone(), json_to_prost_value(value)))
                .collect(),
        }),
    };
    prost_types::Value { kind: Some(kind) }
}

pub fn prost_struct_to_json(value: &prost_types::Struct) -> serde_json::Value {
    serde_json::Value::Object(
        value
            .fields
            .iter()
            .map(|(key, value)| (key.clone(), prost_value_to_json(value)))
            .collect(),
    )
}

fn prost_value_to_json(value: &prost_types::Value) -> serde_json::Value {
    use prost_types::value::Kind;
    match value.kind.as_ref() {
        None | Some(Kind::NullValue(_)) => serde_json::Value::Null,
        Some(Kind::BoolValue(value)) => serde_json::Value::Bool(*value),
        Some(Kind::NumberValue(value))
            if value.is_finite()
                && value.fract() == 0.0
                && *value >= i64::MIN as f64
                && *value <= i64::MAX as f64 =>
        {
            serde_json::Value::Number((*value as i64).into())
        }
        Some(Kind::NumberValue(value)) => serde_json::Number::from_f64(*value)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        Some(Kind::StringValue(value)) => serde_json::Value::String(value.clone()),
        Some(Kind::ListValue(value)) => {
            serde_json::Value::Array(value.values.iter().map(prost_value_to_json).collect())
        }
        Some(Kind::StructValue(value)) => prost_struct_to_json(value),
    }
}
