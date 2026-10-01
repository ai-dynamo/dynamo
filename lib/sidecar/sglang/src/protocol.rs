// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure request lowering and response conversion for SGLang's native gRPC protocol.

use std::collections::HashMap;

use dynamo_backend_common::{
    DisaggregationMode, DynamoError, LLMEngineOutput, LLMEngineOutputExt, PreprocessedRequest,
    StopConditions, StopReason, TopLogprob, usage,
};
use serde_json::{Map, Value};

use crate::client;
use crate::proto as pb;

pub(crate) fn build_generate_request(
    request: &PreprocessedRequest,
    request_id: &str,
    mode: DisaggregationMode,
    bootstrap_host: Option<&str>,
    bootstrap_port: Option<u16>,
) -> Result<pb::GenerateRequest, DynamoError> {
    validate_request(request)?;
    let input_ids = request
        .token_ids
        .iter()
        .map(|token| {
            i32::try_from(*token).map_err(|_| client::invalid_request("token ids must fit in i32"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let max_new_tokens = if mode.is_prefill() {
        Some(1)
    } else {
        request
            .stop_conditions
            .max_tokens
            .map(i32::try_from)
            .transpose()
            .map_err(|_| client::invalid_request("max_tokens does not fit in i32"))?
    };
    let min_new_tokens = if mode.is_prefill() {
        None
    } else {
        request
            .stop_conditions
            .min_tokens
            .map(i32::try_from)
            .transpose()
            .map_err(|_| client::invalid_request("min_tokens does not fit in i32"))?
    };

    let mut stop_token_ids = Vec::new();
    for tokens in [
        request.stop_conditions.stop_token_ids.as_ref(),
        request.stop_conditions.stop_token_ids_hidden.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        for token in tokens {
            let token = i32::try_from(*token)
                .map_err(|_| client::invalid_request("stop token ids must fit in i32"))?;
            if !stop_token_ids.contains(&token) {
                stop_token_ids.push(token);
            }
        }
    }

    let guided = request.sampling_options.guided_decoding.as_ref();
    let sampling_params = pb::SamplingParams {
        temperature: request.sampling_options.temperature,
        top_p: request.sampling_options.top_p,
        top_k: request.sampling_options.top_k,
        min_p: request.sampling_options.min_p,
        frequency_penalty: request.sampling_options.frequency_penalty,
        presence_penalty: request.sampling_options.presence_penalty,
        repetition_penalty: request.sampling_options.repetition_penalty,
        max_new_tokens,
        min_new_tokens,
        stop: request.stop_conditions.stop.clone().unwrap_or_default(),
        stop_token_ids,
        ignore_eos: request.stop_conditions.ignore_eos,
        n: request.sampling_options.n.map(i32::from),
        json_schema: guided
            .and_then(|value| value.json.as_ref())
            .map(json_value_to_string),
        regex: guided.and_then(|value| value.regex.clone()),
    };

    let output_options = &request.output_options;
    let return_logprob = !mode.is_prefill()
        && (output_options.logprobs.is_some() || output_options.prompt_logprobs.is_some());
    let top_logprobs_num = if mode.is_prefill() {
        0
    } else {
        output_options
            .logprobs
            .unwrap_or(0)
            .max(output_options.prompt_logprobs.unwrap_or(0))
    };
    let top_logprobs_num = i32::try_from(top_logprobs_num)
        .map_err(|_| client::invalid_request("requested logprobs does not fit in i32"))?;
    let logprob_start_len = if mode.is_prefill() {
        -1
    } else {
        output_options.prompt_logprobs.map(|_| 0).unwrap_or(-1)
    };
    let routed_dp_rank = routed_dp_rank(request, mode)
        .map(i32::try_from)
        .transpose()
        .map_err(|_| client::invalid_request("routed dp_rank does not fit in i32"))?;
    let lora_path = request
        .routing
        .as_ref()
        .and_then(|routing| routing.lora_name.clone());

    let mut trace_headers = HashMap::new();
    dynamo_runtime::logging::inject_trace_headers_into_map(&mut trace_headers);

    Ok(pb::GenerateRequest {
        input_ids,
        sampling_params: Some(sampling_params),
        stream: Some(true),
        return_logprob: Some(return_logprob),
        top_logprobs_num: Some(top_logprobs_num),
        logprob_start_len: Some(logprob_start_len),
        rid: Some(request_id.to_string()),
        lora_path,
        routing_key: request.mdc_sum.clone(),
        routed_dp_rank,
        trace_headers,
        session_id: None,
        disaggregated_params: resolve_disaggregated_params(
            request,
            mode,
            bootstrap_host,
            bootstrap_port,
        )?,
    })
}

pub(crate) fn routed_dp_rank(
    request: &PreprocessedRequest,
    mode: DisaggregationMode,
) -> Option<u32> {
    request.routing.as_ref().and_then(|routing| {
        if mode.is_prefill() {
            routing.prefill_dp_rank.or(routing.dp_rank)
        } else {
            routing.dp_rank
        }
    })
}

fn validate_request(request: &PreprocessedRequest) -> Result<(), DynamoError> {
    // prompt_embeds requests arrive with empty token_ids, so check them first.
    if request.prompt_embeds.is_some() {
        return Err(client::invalid_request(
            "prompt_embeds are not supported by SGLang's native gRPC proto",
        ));
    }
    if request.token_ids.is_empty() {
        return Err(client::invalid_request("token_ids must not be empty"));
    }
    if request.multi_modal_data.is_some() || request.mm_processor_kwargs.is_some() {
        return Err(client::invalid_request(
            "multimodal payloads are not supported by SGLang's native Generate RPC",
        ));
    }
    if request.sampling_options.n.unwrap_or(1) != 1 {
        return Err(client::invalid_request(
            "n must be 1 for the SGLang sidecar",
        ));
    }
    if request.sampling_options.best_of.unwrap_or(1) != 1 {
        return Err(client::invalid_request(
            "best_of is not represented by SGLang's native gRPC proto",
        ));
    }
    if request.sampling_options.use_beam_search.unwrap_or(false) {
        return Err(client::invalid_request(
            "beam search is not represented by SGLang's native gRPC proto",
        ));
    }
    if let Some(penalty) = request.sampling_options.length_penalty
        && (penalty - 1.0).abs() > f32::EPSILON
    {
        return Err(client::invalid_request(
            "length_penalty is not represented by SGLang's native gRPC proto",
        ));
    }
    if request.sampling_options.seed.is_some() {
        return Err(client::invalid_request(
            "seed is not represented by SGLang's native gRPC proto",
        ));
    }
    if request.stop_conditions.max_thinking_tokens.is_some() {
        return Err(client::invalid_request(
            "thinking_token_budget (max_thinking_tokens) is not represented by SGLang's native gRPC proto",
        ));
    }
    if request
        .sampling_options
        .include_stop_str_in_output
        .unwrap_or(false)
    {
        return Err(client::invalid_request(
            "include_stop_str_in_output is not represented by SGLang's native gRPC proto",
        ));
    }
    if request
        .stop_conditions
        .stop_token_ids_visible
        .as_ref()
        .is_some_and(|tokens| !tokens.is_empty())
    {
        return Err(client::invalid_request(
            "visible stop-token semantics are not represented by SGLang's native gRPC proto",
        ));
    }
    if let Some(guided) = request.sampling_options.guided_decoding.as_ref()
        && (guided
            .choice
            .as_ref()
            .is_some_and(|value| !value.is_empty())
            || guided.grammar.is_some()
            || guided
                .backend
                .as_ref()
                .is_some_and(|value| !value.is_empty())
            || guided.whitespace_pattern.is_some()
            || guided.structural_tag.is_some())
    {
        return Err(client::invalid_request(
            "the native SGLang gRPC proto currently supports only JSON-schema and regex guided decoding",
        ));
    }
    if request
        .routing
        .as_ref()
        .and_then(|routing| routing.priority)
        .unwrap_or(0)
        != 0
    {
        return Err(client::invalid_request(
            "engine priority is not represented by SGLang's native gRPC proto",
        ));
    }
    Ok(())
}

pub(crate) fn resolve_disaggregated_params(
    request: &PreprocessedRequest,
    mode: DisaggregationMode,
    bootstrap_host: Option<&str>,
    bootstrap_port: Option<u16>,
) -> Result<Option<pb::DisaggregatedParams>, DynamoError> {
    if mode == DisaggregationMode::Aggregated {
        return Ok(None);
    }
    if let Some(info) = request.bootstrap_info.as_ref() {
        return bootstrap_values_to_proto(
            &info.bootstrap_host,
            u64::from(info.bootstrap_port),
            info.bootstrap_room,
        )
        .map(Some);
    }
    if let Some(prefill) = request.prefill_result.as_ref() {
        return disaggregated_json_to_proto(&prefill.disaggregated_params).map(Some);
    }
    if mode.is_prefill() {
        let host = bootstrap_host.ok_or_else(|| {
            client::invalid_arg("prefill request has no bootstrap host from discovery")
        })?;
        let port = bootstrap_port.ok_or_else(|| {
            client::invalid_arg("prefill request has no bootstrap port from discovery")
        })?;
        let room = rand::random::<u64>() & (i64::MAX as u64);
        return bootstrap_values_to_proto(host, u64::from(port), room).map(Some);
    }
    Err(client::invalid_arg(
        "decode request has neither bootstrap_info nor prefill_result",
    ))
}

fn disaggregated_json_to_proto(value: &Value) -> Result<pb::DisaggregatedParams, DynamoError> {
    let host = value
        .get("bootstrap_host")
        .and_then(Value::as_str)
        .ok_or_else(|| client::invalid_arg("disaggregated_params.bootstrap_host is missing"))?;
    let port = value
        .get("bootstrap_port")
        .and_then(Value::as_u64)
        .ok_or_else(|| client::invalid_arg("disaggregated_params.bootstrap_port is missing"))?;
    let room = value
        .get("bootstrap_room")
        .and_then(Value::as_u64)
        .ok_or_else(|| client::invalid_arg("disaggregated_params.bootstrap_room is missing"))?;
    bootstrap_values_to_proto(host, port, room)
}

fn bootstrap_values_to_proto(
    host: &str,
    port: u64,
    room: u64,
) -> Result<pb::DisaggregatedParams, DynamoError> {
    if host.trim().is_empty() {
        return Err(client::invalid_arg("bootstrap_host must not be empty"));
    }
    let bootstrap_port = i32::try_from(port)
        .map_err(|_| client::invalid_arg(format!("bootstrap_port is out of range: {port}")))?;
    let bootstrap_room = i64::try_from(room).map_err(|_| {
        client::invalid_arg(format!(
            "bootstrap_room must fit SGLang's signed int64 field: {room}"
        ))
    })?;
    Ok(pb::DisaggregatedParams {
        bootstrap_host: host.to_string(),
        bootstrap_port,
        bootstrap_room,
    })
}

pub(crate) fn disaggregated_params_to_json(params: &pb::DisaggregatedParams) -> Value {
    serde_json::json!({
        "bootstrap_host": params.bootstrap_host,
        "bootstrap_port": params.bootstrap_port,
        "bootstrap_room": params.bootstrap_room,
    })
}

fn json_value_to_string(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        value => value.to_string(),
    }
}

pub(crate) fn output_ids_to_u32(ids: &[i32]) -> Result<Vec<u32>, DynamoError> {
    ids.iter()
        .map(|id| {
            u32::try_from(*id).map_err(|_| {
                client::protocol_error(format!("SGLang returned a negative token id: {id}"))
            })
        })
        .collect()
}

fn meta_value(meta: &HashMap<String, String>, key: &str) -> Option<Value> {
    meta.get(key)
        .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
}

pub(crate) fn meta_u32(meta: &HashMap<String, String>, key: &str) -> Option<u32> {
    meta_value(meta, key)
        .and_then(|value| value.as_u64())
        .and_then(|value| u32::try_from(value).ok())
}

pub(crate) fn terminal_from_meta(
    meta: &HashMap<String, String>,
    prompt_tokens: u32,
    generated: u32,
    stop_conditions: &StopConditions,
) -> Result<LLMEngineOutput, DynamoError> {
    let finish = meta_value(meta, "finish_reason")
        .ok_or_else(|| client::protocol_error("SGLang terminal is missing finish_reason"))?;
    let finish_type = finish
        .get("type")
        .and_then(Value::as_str)
        .or_else(|| finish.as_str())
        .ok_or_else(|| client::protocol_error("SGLang finish_reason is missing a type"))?;
    let mut output = match finish_type {
        "stop" => LLMEngineOutput::stop(),
        "length" => LLMEngineOutput::length(),
        "cancelled" => LLMEngineOutput::cancelled(),
        "abort" | "error" => return Err(terminal_failure(finish_type, &finish)),
        other => {
            return Err(client::protocol_error(format!(
                "SGLang returned unsupported finish_reason type `{other}`"
            )));
        }
    }
    .with_usage(usage(prompt_tokens, generated));
    output.stop_reason = finish.get("matched").and_then(|matched| match matched {
        Value::String(value) => Some(StopReason::String(value.clone())),
        Value::Number(value) => value
            .as_u64()
            .and_then(|id| u32::try_from(id).ok())
            .filter(|id| {
                stop_conditions
                    .stop_token_ids
                    .as_ref()
                    .is_some_and(|ids| ids.contains(id))
            })
            .map(|id| StopReason::Int(i64::from(id))),
        _ => None,
    });
    Ok(output)
}

pub(crate) fn terminal_failure(finish_type: &str, finish: &Value) -> DynamoError {
    let message = finish
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("SGLang generation failed");
    let status_code = finish.get("status_code").and_then(Value::as_i64);
    let err_type = finish.get("err_type").and_then(Value::as_str);
    let detail = format!(
        "SGLang generation {finish_type}: {message} (status_code={}, err_type={})",
        status_code
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unknown".to_string()),
        err_type.unwrap_or("unknown")
    );
    if matches!(status_code, Some(400..=499)) {
        client::invalid_arg(detail)
    } else {
        client::protocol_error(detail)
    }
}

pub(crate) fn engine_data_from_meta(
    meta: &HashMap<String, String>,
    include_prompt_logprobs: bool,
) -> Result<Option<Value>, DynamoError> {
    let mut data = Map::new();
    if let Some(routed_experts) = meta_value(meta, "routed_experts") {
        data.insert("routed_experts".to_string(), routed_experts);
    }
    if include_prompt_logprobs && let Some(prompt_logprobs) = prompt_logprobs_from_meta(meta)? {
        data.insert("prompt_logprobs".to_string(), prompt_logprobs);
    }
    Ok((!data.is_empty()).then_some(Value::Object(data)))
}

fn prompt_logprobs_from_meta(meta: &HashMap<String, String>) -> Result<Option<Value>, DynamoError> {
    let Some(Value::Array(input_logprobs)) = meta_value(meta, "input_token_logprobs") else {
        return Ok(None);
    };
    if input_logprobs.is_empty() {
        return Ok(None);
    }
    let input_top_logprobs = match meta_value(meta, "input_top_logprobs") {
        Some(Value::Array(values)) if !values.is_empty() => Some(values),
        _ => None,
    };

    // Current SGLang encodes the first requested prompt position as
    // `[null, token_id, null]`; older releases omitted it entirely.
    let has_native_sentinel = input_logprobs.first().is_some_and(|entry| {
        entry.is_null()
            || entry
                .as_array()
                .and_then(|parts| parts.first())
                .is_some_and(Value::is_null)
    });
    if let Some(input_top_logprobs) = input_top_logprobs.as_ref() {
        let top_has_native_sentinel = input_top_logprobs.first().is_some_and(Value::is_null);
        if input_top_logprobs.len() != input_logprobs.len()
            || top_has_native_sentinel != has_native_sentinel
        {
            return Err(client::protocol_error(
                "input_token_logprobs and input_top_logprobs use inconsistent position encoding",
            ));
        }
    }
    let mut payload = Vec::with_capacity(input_logprobs.len() + usize::from(!has_native_sentinel));
    if !has_native_sentinel {
        payload.push(Value::Null);
    }
    for (index, selected) in input_logprobs.iter().enumerate() {
        if index == 0 && has_native_sentinel {
            payload.push(Value::Null);
            continue;
        }
        let (token_id, entry) = prompt_logprob_entry(selected, "input_token_logprobs")?;
        let mut position = Map::new();
        position.insert(token_id, entry);
        if let Some(Value::Array(alternatives)) = input_top_logprobs
            .as_ref()
            .and_then(|values| values.get(index))
        {
            for alternative in alternatives {
                let (token_id, entry) = prompt_logprob_entry(alternative, "input_top_logprobs")?;
                position.entry(token_id).or_insert(entry);
            }
        }
        payload.push(Value::Object(position));
    }
    Ok(Some(Value::Array(payload)))
}

fn prompt_logprob_entry(value: &Value, label: &str) -> Result<(String, Value), DynamoError> {
    let parts = value
        .as_array()
        .ok_or_else(|| client::protocol_error(format!("invalid {label} entry from SGLang")))?;
    let logprob = parts
        .first()
        .and_then(Value::as_f64)
        .ok_or_else(|| client::protocol_error(format!("missing logprob in {label}")))?;
    let token_id = parts
        .get(1)
        .and_then(Value::as_i64)
        .ok_or_else(|| client::protocol_error(format!("missing token id in {label}")))?;
    let mut entry = Map::new();
    entry.insert("logprob".to_string(), Value::from(logprob));
    if let Some(decoded) = parts.get(2).and_then(Value::as_str) {
        entry.insert(
            "decoded_token".to_string(),
            Value::String(decoded.to_string()),
        );
    }
    Ok((token_id.to_string(), Value::Object(entry)))
}

pub(crate) type ExtractedLogprobs = (Option<Vec<f64>>, Option<Vec<Vec<TopLogprob>>>);

pub(crate) fn extract_logprobs(
    meta: &HashMap<String, String>,
    return_tokens_as_ids: bool,
) -> Result<ExtractedLogprobs, DynamoError> {
    let Some(Value::Array(all_logprobs)) = meta_value(meta, "output_token_logprobs") else {
        return Ok((None, None));
    };

    let mut log_probs = Vec::with_capacity(all_logprobs.len());
    for entry in &all_logprobs {
        let value = entry
            .as_array()
            .and_then(|parts| parts.first())
            .and_then(Value::as_f64)
            .ok_or_else(|| {
                client::protocol_error("invalid output_token_logprobs entry from SGLang")
            })?;
        log_probs.push(value);
    }

    let top_logprobs = match meta_value(meta, "output_top_logprobs") {
        Some(Value::Array(all_top)) => {
            let mut positions = Vec::new();
            for position in &all_top {
                let Some(entries) = position.as_array() else {
                    positions.push(Vec::new());
                    continue;
                };
                let mut mapped = Vec::with_capacity(entries.len());
                for (index, entry) in entries.iter().enumerate() {
                    let parts = entry.as_array().ok_or_else(|| {
                        client::protocol_error("invalid output_top_logprobs entry from SGLang")
                    })?;
                    let logprob = parts.first().and_then(Value::as_f64).ok_or_else(|| {
                        client::protocol_error("missing top-logprob value from SGLang")
                    })?;
                    let token_id = parts.get(1).and_then(Value::as_u64).ok_or_else(|| {
                        client::protocol_error("missing top-logprob token id from SGLang")
                    })?;
                    let token_id = u32::try_from(token_id).map_err(|_| {
                        client::protocol_error("top-logprob token id does not fit u32")
                    })?;
                    let token = if return_tokens_as_ids {
                        Some(format!("token_id:{token_id}"))
                    } else {
                        parts.get(2).and_then(Value::as_str).map(str::to_string)
                    };
                    mapped.push(TopLogprob {
                        rank: u32::try_from(index + 1).unwrap_or(u32::MAX),
                        token_id,
                        token,
                        logprob,
                        bytes: None,
                    });
                }
                positions.push(mapped);
            }
            Some(positions)
        }
        _ => None,
    };

    Ok((Some(log_probs), top_logprobs))
}

#[cfg(test)]
mod request_tests;

#[cfg(test)]
mod response_tests;
