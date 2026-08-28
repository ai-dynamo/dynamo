// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conversion between Dynamo requests and OpenEngine v0.1.0 messages.

use std::collections::{BTreeMap, BTreeSet};

use dynamo_backend_common::{
    DisaggregationMode, DynamoError, FinishReason, LLMEngineOutput, PreprocessedRequest,
    StopReason, TopLogprob, usage,
};
use tonic::metadata::{Ascii, MetadataKey, MetadataMap, MetadataValue};

use crate::client;
use crate::proto as pb;

pub(crate) fn generate_metadata(
    request: &PreprocessedRequest,
    context: &BTreeMap<String, String>,
    mode: DisaggregationMode,
) -> Result<MetadataMap, DynamoError> {
    let mut metadata = MetadataMap::new();
    for key in ["traceparent", "tracestate"] {
        if let Some(value) = context.get(key) {
            insert_metadata(&mut metadata, key, value)?;
        }
    }
    if let Some(routing) = request.routing.as_ref() {
        if let Some(priority) = routing.priority {
            insert_metadata(&mut metadata, "openengine-priority", &priority.to_string())?;
        }
        // A decode KvSessionRef carries the authoritative rank. Omitting the
        // metadata lets the server route from that opaque handoff.
        let rank = match mode {
            DisaggregationMode::Prefill => routing.prefill_dp_rank.or(routing.dp_rank),
            DisaggregationMode::Aggregated => routing.dp_rank,
            DisaggregationMode::Decode | DisaggregationMode::Encode => None,
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
        client::invalid_argument(format!("invalid OpenEngine metadata key `{key}`: {error}"))
    })?;
    let value = MetadataValue::<Ascii>::try_from(value).map_err(|error| {
        client::invalid_argument(format!("invalid OpenEngine metadata value: {error}"))
    })?;
    metadata.insert(key, value);
    Ok(())
}

pub(crate) fn build_generate_request(
    request: &PreprocessedRequest,
    request_id: &str,
    model: &str,
    mode: DisaggregationMode,
) -> Result<pb::GenerateRequest, DynamoError> {
    validate_request(request, mode)?;

    let sampling = &request.sampling_options;
    let stopping = &request.stop_conditions;
    let output = &request.output_options;
    let mut stop_token_ids = BTreeSet::new();
    for ids in [
        stopping.stop_token_ids.as_ref(),
        stopping.stop_token_ids_hidden.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        stop_token_ids.extend(ids.iter().copied());
    }
    let conditions = stopping
        .stop
        .iter()
        .flatten()
        .cloned()
        .map(|value| pb::StopCondition {
            condition: Some(pb::stop_condition::Condition::StopText(value)),
        })
        .chain(stop_token_ids.into_iter().map(|value| pb::StopCondition {
            condition: Some(pb::stop_condition::Condition::StopTokenId(value)),
        }))
        .collect();

    let session = match mode {
        DisaggregationMode::Decode => Some(disagg_json_to_kv_session(
            &request
                .prefill_result
                .as_ref()
                .ok_or_else(|| {
                    client::invalid_argument(
                        "decode request is missing the prefill_result KV session",
                    )
                })?
                .disaggregated_params,
        )?),
        DisaggregationMode::Aggregated | DisaggregationMode::Prefill => None,
        DisaggregationMode::Encode => unreachable!("encode rejected by validate_request"),
    };
    let routing = request.routing.as_ref();

    Ok(pb::GenerateRequest {
        request_id: request_id.to_string(),
        model: model.to_string(),
        input: Some(pb::generate_request::Input::TokenIds(pb::TokenIds {
            ids: request.token_ids.clone(),
        })),
        sampling: Some(pb::SamplingParams {
            temperature: sampling.temperature.map(f64::from),
            top_p: sampling.top_p.map(f64::from),
            top_k: normalize_top_k(sampling.top_k)?,
            min_p: sampling.min_p.map(f64::from),
            frequency_penalty: sampling.frequency_penalty.map(f64::from),
            presence_penalty: sampling.presence_penalty.map(f64::from),
            repetition_penalty: sampling.repetition_penalty.map(f64::from),
            seed: normalize_seed(sampling.seed)?,
            num_sequences: sampling.n.map(u32::from),
        }),
        stopping: Some(pb::StoppingOptions {
            max_tokens: if mode.is_prefill() {
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
            return_prompt_logprobs: None,
            prompt_candidates: None,
            return_output_logprobs: output.logprobs.map(|_| true),
            output_candidates: output.logprobs.map(top_n_candidates),
            prompt_logprob_start: None,
        }),
        kv: Some(pb::KvOptions {
            session,
            bypass_prefix_cache: prefix_cache_bypass(request),
            cache_salt: routing.and_then(|value| value.cache_namespace.clone()),
        }),
        guided: None,
        media: Vec::new(),
        lora_name: String::new(),
        extra: None,
    })
}

fn top_n_candidates(top_n: u32) -> pb::CandidateTokenSelection {
    pb::CandidateTokenSelection {
        selection: Some(pb::candidate_token_selection::Selection::TopN(top_n)),
    }
}

fn normalize_top_k(top_k: Option<i32>) -> Result<Option<i32>, DynamoError> {
    match top_k {
        None | Some(-1) | Some(0) => Ok(None),
        Some(value) if value > 0 => Ok(Some(value)),
        Some(value) => Err(client::invalid_argument(format!(
            "top_k must be -1, 0, or positive; got {value}"
        ))),
    }
}

fn normalize_seed(seed: Option<i64>) -> Result<Option<u64>, DynamoError> {
    seed.map(|value| {
        u64::try_from(value).map_err(|_| {
            client::invalid_argument(format!("seed must be non-negative; got {value}"))
        })
    })
    .transpose()
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

fn validate_request(
    request: &PreprocessedRequest,
    mode: DisaggregationMode,
) -> Result<(), DynamoError> {
    if mode.is_encode() {
        return Err(client::invalid_argument(
            "encode mode is not supported by the OpenEngine sidecar",
        ));
    }
    if request.token_ids.is_empty() {
        return Err(client::invalid_argument("token_ids must not be empty"));
    }
    if request.prompt_embeds.is_some() {
        return Err(client::invalid_argument(
            "prompt embeddings are not supported by this OpenEngine sidecar",
        ));
    }
    if request.multi_modal_data.is_some()
        || request.mm_routing_info.is_some()
        || request.encoder_result.is_some()
    {
        return Err(client::invalid_argument(
            "multimodal and encoder handoff inputs are not supported by this OpenEngine sidecar",
        ));
    }
    if mode.is_decode() != request.prefill_result.is_some() {
        return Err(client::invalid_argument(if mode.is_decode() {
            "decode requests require a prefill_result KV session"
        } else {
            "only decode requests may carry a prefill_result KV session"
        }));
    }
    if request.output_options.prompt_logprobs.is_some() {
        return Err(client::invalid_argument(
            "prompt logprobs are not supported by this OpenEngine sidecar",
        ));
    }
    if request.sampling_options.guided_decoding.is_some() {
        return Err(client::invalid_argument(
            "guided decoding is not implemented by the TensorRT-LLM OpenEngine server",
        ));
    }
    if request
        .routing
        .as_ref()
        .and_then(|routing| routing.lora_name.as_deref())
        .is_some_and(|name| !name.is_empty())
    {
        return Err(client::invalid_argument(
            "LoRA selection is not implemented by the TensorRT-LLM OpenEngine server",
        ));
    }
    if request
        .stop_conditions
        .stop_token_ids_visible
        .as_ref()
        .is_some_and(|ids| !ids.is_empty())
    {
        return Err(client::invalid_argument(
            "visible stop token IDs cannot be represented by OpenEngine's global stop-output flag",
        ));
    }
    if prefix_cache_bypass(request) == Some(true) {
        return Err(client::invalid_argument(
            "prefix-cache bypass is not implemented by the TensorRT-LLM OpenEngine server",
        ));
    }
    if request.stop_conditions.max_thinking_tokens.is_some() {
        return Err(client::invalid_argument(
            "max_thinking_tokens is not supported by OpenEngine v0.1.0",
        ));
    }
    let sampling = &request.sampling_options;
    if sampling.n.unwrap_or(1) != 1 {
        return Err(client::invalid_argument("n must be 1"));
    }
    if sampling.best_of.unwrap_or(1) != 1 {
        return Err(client::invalid_argument("best_of must be 1"));
    }
    if sampling.use_beam_search.unwrap_or(false) {
        return Err(client::invalid_argument("beam search is not supported"));
    }
    Ok(())
}

pub(crate) struct ResponseState {
    mode: DisaggregationMode,
    prompt_tokens: u32,
    completion_tokens: u32,
}

impl ResponseState {
    pub(crate) fn new(mode: DisaggregationMode, prompt_tokens: u32) -> Self {
        Self {
            mode,
            prompt_tokens,
            completion_tokens: 0,
        }
    }

    pub(crate) fn prompt_tokens(&self) -> u32 {
        self.prompt_tokens
    }

    pub(crate) fn completion_tokens(&self) -> u32 {
        self.completion_tokens
    }

    pub(crate) fn convert(
        &mut self,
        response: pb::GenerateResponse,
        request_id: &str,
    ) -> Result<Option<LLMEngineOutput>, DynamoError> {
        if response.request_id != request_id {
            return Err(client::protocol_error(format!(
                "Generate returned request_id `{}` on stream `{request_id}`",
                response.request_id
            )));
        }
        if let Some(value) = response.usage.as_ref() {
            self.prompt_tokens = value.prompt_tokens;
            self.completion_tokens = value.completion_tokens;
        }
        match response.event {
            Some(pb::generate_response::Event::Token(token)) => {
                if self.mode.is_prefill() || response.usage.is_some() {
                    return Err(client::protocol_error(
                        "Generate returned a token event for prefill or with terminal usage",
                    ));
                }
                let output_index = token.output_index.ok_or_else(|| {
                    client::protocol_error("token event omitted required output_index")
                })?;
                if output_index != 0 {
                    return Err(client::protocol_error(format!(
                        "received unsupported output index {output_index}"
                    )));
                }
                self.completion_tokens = self
                    .completion_tokens
                    .saturating_add(token.tokens.len() as u32);
                Ok(token_output(token))
            }
            Some(pb::generate_response::Event::PrefillReady(prefill)) => {
                if !self.mode.is_prefill() {
                    return Err(client::protocol_error(
                        "Generate returned PrefillReady for a non-prefill worker",
                    ));
                }
                let session = prefill.kv_session.ok_or_else(|| {
                    client::protocol_error("PrefillReady omitted required kv_session")
                })?;
                let wire_usage = response
                    .usage
                    .ok_or_else(|| client::protocol_error("PrefillReady omitted terminal usage"))?;
                let mut output = LLMEngineOutput::stop();
                output.completion_usage = Some(usage(
                    wire_usage.prompt_tokens,
                    wire_usage.completion_tokens,
                ));
                output.disaggregated_params = Some(kv_session_to_disagg_json(session));
                Ok(Some(output))
            }
            Some(pb::generate_response::Event::Finished(finished)) => {
                if self.mode.is_prefill() {
                    return Err(client::protocol_error(
                        "Generate returned GenerationFinished for a prefill worker",
                    ));
                }
                let output_index = finished.output_index.ok_or_else(|| {
                    client::protocol_error("GenerationFinished omitted required output_index")
                })?;
                if output_index != 0 {
                    return Err(client::protocol_error(format!(
                        "received unsupported output index {output_index}"
                    )));
                }
                let wire_usage = response.usage.ok_or_else(|| {
                    client::protocol_error("GenerationFinished omitted terminal usage")
                })?;
                let reason = pb::FinishReason::try_from(finished.reason).map_err(|_| {
                    client::protocol_error(format!(
                        "unknown OpenEngine finish reason {}",
                        finished.reason
                    ))
                })?;
                let mut output = finished_output(reason, finished.stop_match)?;
                output.index = Some(output_index);
                output.completion_usage = Some(usage(
                    wire_usage.prompt_tokens,
                    wire_usage.completion_tokens,
                ));
                Ok(Some(output))
            }
            Some(pb::generate_response::Event::Error(error)) => Err(client::protocol_error(
                format!("server error {:?}: {}", error.code(), error.message),
            )),
            Some(pb::generate_response::Event::Prompt(_)) => Err(client::protocol_error(
                "Generate returned prompt logprobs that were not requested",
            )),
            None => Err(client::protocol_error("Generate response carried no event")),
        }
    }
}

fn token_output(value: pb::TokenOutput) -> Option<LLMEngineOutput> {
    if value.tokens.is_empty() && value.text.is_empty() {
        return None;
    }
    let has_logprobs = value.tokens.iter().any(|token| token.logprob.is_some());
    Some(LLMEngineOutput {
        token_ids: value.tokens.iter().map(|token| token.token_id).collect(),
        tokens: Some(
            value
                .tokens
                .iter()
                .map(|token| (!token.token.is_empty()).then(|| token.token.clone()))
                .collect(),
        ),
        text: (!value.text.is_empty()).then_some(value.text),
        log_probs: has_logprobs.then(|| {
            value
                .tokens
                .iter()
                .map(|token| token.logprob.unwrap_or(f64::NEG_INFINITY))
                .collect()
        }),
        top_logprobs: has_logprobs.then(|| {
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
        }),
        index: value.output_index,
        ..Default::default()
    })
}

fn finished_output(
    reason: pb::FinishReason,
    stop_match: Option<pb::StopMatch>,
) -> Result<LLMEngineOutput, DynamoError> {
    let finish_reason = match reason {
        pb::FinishReason::Stop => FinishReason::Stop,
        pb::FinishReason::Length => FinishReason::Length,
        pb::FinishReason::Cancelled => FinishReason::Cancelled,
        pb::FinishReason::Unspecified => {
            return Err(client::protocol_error(
                "GenerationFinished used an unspecified finish reason",
            ));
        }
    };
    let stop_reason = stop_match.and_then(|value| match value.r#match {
        Some(pb::stop_match::Match::StopTokenId(id))
        | Some(pb::stop_match::Match::EosTokenId(id)) => Some(StopReason::Int(i64::from(id))),
        Some(pb::stop_match::Match::StopText(value)) => Some(StopReason::String(value)),
        None => None,
    });
    Ok(LLMEngineOutput {
        finish_reason: Some(finish_reason),
        stop_reason,
        ..Default::default()
    })
}

pub(crate) fn kv_session_to_disagg_json(value: pb::KvSessionRef) -> serde_json::Value {
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
    })
}

pub(crate) fn disagg_json_to_kv_session(
    value: &serde_json::Value,
) -> Result<pb::KvSessionRef, DynamoError> {
    let object = value.as_object().ok_or_else(|| {
        client::invalid_argument("prefill_result.disaggregated_params must be an object")
    })?;
    let required_string = |key: &str| -> Result<String, DynamoError> {
        object
            .get(key)
            .and_then(serde_json::Value::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .ok_or_else(|| {
                client::invalid_argument(format!("handoff `{key}` must be a non-empty string"))
            })
    };
    let endpoints = object
        .get("endpoints")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| client::invalid_argument("handoff `endpoints` must be an array"))?
        .iter()
        .map(endpoint_from_json)
        .collect::<Result<Vec<_>, _>>()?;
    let attributes_struct = match object.get("attributes_struct") {
        None | Some(serde_json::Value::Null) => None,
        Some(value @ serde_json::Value::Object(_)) => json_to_prost_struct(value),
        Some(_) => {
            return Err(client::invalid_argument(
                "handoff `attributes_struct` must be an object or null",
            ));
        }
    };
    let dp_rank = object
        .get("dp_rank")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| client::invalid_argument("handoff `dp_rank` must be a uint32"))?;
    Ok(pb::KvSessionRef {
        session_id: required_string("session_id")?,
        transfer_backend: required_string("transfer_backend")?,
        endpoints,
        dp_rank,
        attributes_struct,
    })
}

fn endpoint_from_json(value: &serde_json::Value) -> Result<pb::KvEndpoint, DynamoError> {
    let endpoint = value
        .as_object()
        .ok_or_else(|| client::invalid_argument("handoff endpoint must be an object"))?;
    let host = endpoint
        .get("host")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| client::invalid_argument("handoff endpoint host must be non-empty"))?;
    let port = endpoint
        .get("port")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| client::invalid_argument("handoff endpoint port must be a uint32"))?;
    let protocol = endpoint
        .get("protocol")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| client::invalid_argument("handoff endpoint protocol must be non-empty"))?;
    Ok(pb::KvEndpoint {
        host: host.to_string(),
        port,
        protocol: protocol.to_string(),
    })
}

fn json_to_prost_struct(value: &serde_json::Value) -> Option<prost_types::Struct> {
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

fn prost_struct_to_json(value: &prost_types::Struct) -> serde_json::Value {
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
        Some(Kind::NumberValue(value)) if value.fract() == 0.0 => {
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

#[cfg(test)]
mod tests {
    use super::{disagg_json_to_kv_session, json_to_prost_struct, kv_session_to_disagg_json};
    use crate::proto as pb;

    #[test]
    fn preserves_opaque_trtllm_kv_session_across_dynamo_handoff() {
        let session = pb::KvSessionRef {
            session_id: "42".to_string(),
            transfer_backend: "tensorrt_llm".to_string(),
            endpoints: Vec::new(),
            dp_rank: 3,
            attributes_struct: json_to_prost_struct(&serde_json::json!({
                "tensorrt_llm.disaggregated_params.v1": {
                    "ctx_request_id": "42",
                    "ctx_dp_rank": 3,
                    "schedule_style": "context_first",
                    "draft_tokens": [7, 8]
                }
            })),
        };

        let handoff = kv_session_to_disagg_json(session.clone());
        assert_eq!(disagg_json_to_kv_session(&handoff).unwrap(), session);
    }
}
