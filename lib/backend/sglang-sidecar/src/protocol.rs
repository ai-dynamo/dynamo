// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure request lowering and response conversion for SGLang's native gRPC protocol.

use std::collections::HashMap;

use dynamo_backend_common::{
    DisaggregationMode, DynamoError, LLMEngineOutput, LLMEngineOutputExt, PreprocessedRequest,
    StopReason, TopLogprob, usage,
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
            i32::try_from(*token)
                .map_err(|_| client::invalid_arg(format!("token id {token} does not fit in i32")))
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
            .map_err(|_| client::invalid_arg("max_tokens does not fit in i32"))?
    };
    let min_new_tokens = if mode.is_prefill() {
        None
    } else {
        request
            .stop_conditions
            .min_tokens
            .map(i32::try_from)
            .transpose()
            .map_err(|_| client::invalid_arg("min_tokens does not fit in i32"))?
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
            let token = i32::try_from(*token).map_err(|_| {
                client::invalid_arg(format!("stop token id {token} does not fit in i32"))
            })?;
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
        .map_err(|_| client::invalid_arg("requested logprobs does not fit in i32"))?;
    let logprob_start_len = if mode.is_prefill() {
        -1
    } else {
        output_options.prompt_logprobs.map(|_| 0).unwrap_or(-1)
    };
    let routed_dp_rank = request
        .routing
        .as_ref()
        .and_then(|routing| routing.dp_rank)
        .map(i32::try_from)
        .transpose()
        .map_err(|_| client::invalid_arg("routed dp_rank does not fit in i32"))?;
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

fn validate_request(request: &PreprocessedRequest) -> Result<(), DynamoError> {
    if request.token_ids.is_empty() {
        return Err(client::invalid_arg("token_ids must not be empty"));
    }
    if request.prompt_embeds.is_some() {
        return Err(client::invalid_arg(
            "prompt_embeds are not supported by SGLang's native gRPC proto",
        ));
    }
    if request.multi_modal_data.is_some() || request.mm_processor_kwargs.is_some() {
        return Err(client::invalid_arg(
            "multimodal payloads are not supported by SGLang's native Generate RPC",
        ));
    }
    if request.sampling_options.n.unwrap_or(1) != 1 {
        return Err(client::invalid_arg("n must be 1 for the SGLang sidecar"));
    }
    if request.sampling_options.best_of.unwrap_or(1) != 1 {
        return Err(client::invalid_arg(
            "best_of is not represented by SGLang's native gRPC proto",
        ));
    }
    if request.sampling_options.use_beam_search.unwrap_or(false) {
        return Err(client::invalid_arg(
            "beam search is not represented by SGLang's native gRPC proto",
        ));
    }
    if let Some(penalty) = request.sampling_options.length_penalty
        && (penalty - 1.0).abs() > f32::EPSILON
    {
        return Err(client::invalid_arg(
            "length_penalty is not represented by SGLang's native gRPC proto",
        ));
    }
    if request.sampling_options.seed.is_some() {
        return Err(client::invalid_arg(
            "seed is not represented by SGLang's native gRPC proto",
        ));
    }
    if request.stop_conditions.max_thinking_tokens.is_some() {
        return Err(client::invalid_arg(
            "max_thinking_tokens is not represented by SGLang's native gRPC proto",
        ));
    }
    if request
        .sampling_options
        .include_stop_str_in_output
        .unwrap_or(false)
    {
        return Err(client::invalid_arg(
            "include_stop_str_in_output is not represented by SGLang's native gRPC proto",
        ));
    }
    if request
        .stop_conditions
        .stop_token_ids_visible
        .as_ref()
        .is_some_and(|tokens| !tokens.is_empty())
    {
        return Err(client::invalid_arg(
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
        return Err(client::invalid_arg(
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
        return Err(client::invalid_arg(
            "engine priority is not represented by SGLang's native gRPC proto",
        ));
    }
    Ok(())
}

fn resolve_disaggregated_params(
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
        Value::Number(value) => value.as_i64().map(StopReason::Int),
        _ => None,
    });
    Ok(output)
}

fn terminal_failure(finish_type: &str, finish: &Value) -> DynamoError {
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
        Some(Value::Array(values)) => values,
        _ => Vec::new(),
    };

    let mut payload = Vec::with_capacity(input_logprobs.len() + 1);
    payload.push(Value::Null);
    for (index, selected) in input_logprobs.iter().enumerate() {
        let (token_id, entry) = prompt_logprob_entry(selected, "input_token_logprobs")?;
        let mut position = Map::new();
        position.insert(token_id, entry);
        if let Some(Value::Array(alternatives)) = input_top_logprobs.get(index) {
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

pub(crate) type ExtractedLogprobs = (Option<Vec<f64>>, Option<Vec<Vec<TopLogprob>>>, usize);

pub(crate) fn extract_logprobs(
    meta: &HashMap<String, String>,
    offset: usize,
    return_tokens_as_ids: bool,
) -> Result<ExtractedLogprobs, DynamoError> {
    let Some(Value::Array(all_logprobs)) = meta_value(meta, "output_token_logprobs") else {
        return Ok((None, None, offset));
    };
    if offset >= all_logprobs.len() {
        return Ok((None, None, all_logprobs.len()));
    }

    let mut log_probs = Vec::with_capacity(all_logprobs.len() - offset);
    for entry in &all_logprobs[offset..] {
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
            for position in all_top.iter().skip(offset) {
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

    Ok((Some(log_probs), top_logprobs, all_logprobs.len()))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use dynamo_backend_common::{
        BootstrapInfo, DisaggregationMode, FinishReason, OutputOptions, PrefillResult,
        PreprocessedRequest, SamplingOptions, StopConditions,
    };
    use serde_json::json;

    use super::{
        build_generate_request, disaggregated_params_to_json, engine_data_from_meta,
        extract_logprobs, terminal_from_meta,
    };

    fn request() -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("Qwen/Qwen3-0.6B".to_string())
            .token_ids(vec![1, 2, 3])
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .stop_conditions(StopConditions {
                max_tokens: Some(8),
                ..Default::default()
            })
            .build()
            .unwrap()
    }

    #[test]
    fn request_maps_native_fields_and_full_width_room() {
        let mut request = request();
        request.bootstrap_info = Some(BootstrapInfo {
            bootstrap_host: "prefill".to_string(),
            bootstrap_port: 5000,
            bootstrap_room: i64::MAX as u64,
            handoff_id: None,
        });
        let mapped =
            build_generate_request(&request, "rid-1", DisaggregationMode::Decode, None, None)
                .unwrap();
        assert_eq!(mapped.input_ids, vec![1, 2, 3]);
        assert_eq!(mapped.rid.as_deref(), Some("rid-1"));
        assert_eq!(mapped.sampling_params.unwrap().max_new_tokens, Some(8));
        assert_eq!(
            mapped.disaggregated_params.unwrap().bootstrap_room,
            i64::MAX
        );
    }

    #[test]
    fn prefill_clamps_generation_and_disables_decode_only_options() {
        let mut request = request();
        request.stop_conditions.min_tokens = Some(4);
        request.output_options = OutputOptions {
            logprobs: Some(2),
            prompt_logprobs: Some(3),
            ..Default::default()
        };
        let mapped = build_generate_request(
            &request,
            "rid-2",
            DisaggregationMode::Prefill,
            Some("prefill"),
            Some(5001),
        )
        .unwrap();
        let sampling = mapped.sampling_params.unwrap();
        assert_eq!(sampling.max_new_tokens, Some(1));
        assert_eq!(sampling.min_new_tokens, None);
        assert_eq!(mapped.return_logprob, Some(false));
        assert_eq!(mapped.top_logprobs_num, Some(0));
        assert_eq!(mapped.logprob_start_len, Some(-1));
        assert_eq!(mapped.disaggregated_params.unwrap().bootstrap_port, 5001);
    }

    #[test]
    fn prefill_handoff_round_trips_to_decode_request() {
        let prefill = build_generate_request(
            &request(),
            "rid-prefill",
            DisaggregationMode::Prefill,
            Some("prefill.internal"),
            Some(5001),
        )
        .unwrap();
        let handoff = prefill.disaggregated_params.unwrap();

        let mut decode_request = request();
        decode_request.prefill_result = Some(PrefillResult {
            disaggregated_params: disaggregated_params_to_json(&handoff),
            prompt_tokens_details: None,
        });
        let decode = build_generate_request(
            &decode_request,
            "rid-decode",
            DisaggregationMode::Decode,
            None,
            None,
        )
        .unwrap();

        assert_eq!(decode.disaggregated_params, Some(handoff));
    }

    #[test]
    fn logprobs_are_sliced_from_cumulative_metadata() {
        let meta = HashMap::from([
            (
                "output_token_logprobs".to_string(),
                json!([[-0.1, 10, "a"], [-0.2, 11, "b"]]).to_string(),
            ),
            (
                "output_top_logprobs".to_string(),
                json!([[[-0.1, 10, "a"]], [[-0.2, 11, "b"]]]).to_string(),
            ),
        ]);
        let (logprobs, top, next) = extract_logprobs(&meta, 1, false).unwrap();
        assert_eq!(logprobs.unwrap(), vec![-0.2]);
        assert_eq!(top.unwrap()[0][0].token_id, 11);
        assert_eq!(next, 2);
    }

    #[test]
    fn terminal_maps_finish_reason_and_usage() {
        let meta = HashMap::from([(
            "finish_reason".to_string(),
            json!({"type": "length"}).to_string(),
        )]);
        let terminal = terminal_from_meta(&meta, 4, 3).unwrap();
        assert_eq!(terminal.finish_reason, Some(FinishReason::Length));
        assert_eq!(terminal.completion_usage.unwrap().total_tokens, 7);
    }

    #[test]
    fn abort_terminal_preserves_failure_metadata_as_error() {
        let meta = HashMap::from([(
            "finish_reason".to_string(),
            json!({
                "type": "abort",
                "message": "prefill allocation failed",
                "status_code": 503,
                "err_type": "KVTransferError"
            })
            .to_string(),
        )]);
        let error = terminal_from_meta(&meta, 4, 0).unwrap_err().to_string();
        assert!(error.contains("prefill allocation failed"));
        assert!(error.contains("status_code=503"));
        assert!(error.contains("KVTransferError"));
    }

    #[test]
    fn malformed_terminal_is_rejected() {
        assert!(terminal_from_meta(&HashMap::new(), 4, 0).is_err());
        let meta = HashMap::from([(
            "finish_reason".to_string(),
            json!({"type": "mystery"}).to_string(),
        )]);
        assert!(terminal_from_meta(&meta, 4, 0).is_err());
    }

    #[test]
    fn terminal_engine_data_contains_prompt_logprobs_and_routed_experts() {
        let meta = HashMap::from([
            (
                "input_token_logprobs".to_string(),
                json!([[-0.1, 10, "a"], [-0.2, 11, "b"]]).to_string(),
            ),
            (
                "input_top_logprobs".to_string(),
                json!([[[-0.3, 12, "c"]], []]).to_string(),
            ),
            ("routed_experts".to_string(), json!([1, 2]).to_string()),
        ]);
        let data = engine_data_from_meta(&meta, true).unwrap().unwrap();
        let prompt = data["prompt_logprobs"].as_array().unwrap();
        assert!(prompt[0].is_null());
        assert_eq!(prompt[1]["10"]["logprob"], json!(-0.1));
        assert_eq!(prompt[1]["12"]["decoded_token"], json!("c"));
        assert_eq!(data["routed_experts"], json!([1, 2]));
    }

    #[test]
    fn prompt_logprobs_are_terminal_only() {
        let meta = HashMap::from([(
            "input_token_logprobs".to_string(),
            json!([[-0.1, 10, "a"]]).to_string(),
        )]);
        assert!(engine_data_from_meta(&meta, false).unwrap().is_none());
    }

    #[test]
    fn decode_requires_rendezvous_params() {
        assert!(
            build_generate_request(&request(), "rid-3", DisaggregationMode::Decode, None, None,)
                .is_err()
        );
    }

    #[test]
    fn room_above_signed_int64_is_rejected() {
        let mut request = request();
        request.bootstrap_info = Some(BootstrapInfo {
            bootstrap_host: "prefill".to_string(),
            bootstrap_port: 5000,
            bootstrap_room: i64::MAX as u64 + 1,
            handoff_id: None,
        });
        assert!(
            build_generate_request(&request, "rid-4", DisaggregationMode::Decode, None, None,)
                .is_err()
        );
    }
}
