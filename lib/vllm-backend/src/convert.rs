// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request and response conversion between Dynamo backend-common and vLLM LLM types.

use std::collections::{BTreeSet, HashMap};

use dynamo_backend_common::{
    DynamoError, GuidedDecodingOptions, LLMEngineOutput, LLMEngineOutputExt, PreprocessedRequest,
    StopReason as DynamoStopReason, TopLogprob, usage,
};
use vllm_engine_core_client::protocol::{
    EngineCoreSamplingParams, Logprobs as VllmLogprobs, StopReason as VllmStopReason,
    StructuredOutputsParams,
};
use vllm_llm::{FinishReason as VllmFinishReason, GenerateOutput, GenerateRequest};

use crate::error::invalid_arg;

/// Converts a preprocessed Dynamo request into a token-level vLLM generate request.
///
/// This function intentionally assumes rendering and tokenization have already
/// happened in Dynamo preprocessing. It forwards only fields represented by the
/// Rust vLLM engine-core generate path and rejects unsupported payload shapes
/// before the request reaches the engine.
pub(crate) fn lower_request(
    request_id: String,
    request: PreprocessedRequest,
    max_model_len: Option<u32>,
) -> Result<GenerateRequest, DynamoError> {
    validate_request(&request)?;

    let prompt_len = request.token_ids.len() as u32;
    let max_tokens = request.stop_conditions.max_tokens.unwrap_or_else(|| {
        max_model_len
            .map(|limit| limit.saturating_sub(prompt_len).max(1))
            .unwrap_or(u32::MAX)
    });
    let ignore_eos = request.stop_conditions.ignore_eos.unwrap_or(false);
    let eos_token_id = if ignore_eos {
        None
    } else {
        request.eos_token_ids.first().copied()
    };

    let stop_token_ids = request
        .stop_conditions
        .stop_token_ids_hidden
        .clone()
        .unwrap_or_default();

    let mut all_stop_token_ids: BTreeSet<u32> = stop_token_ids.iter().copied().collect();
    if !ignore_eos {
        all_stop_token_ids.extend(request.eos_token_ids.iter().copied());
    }

    let sampling = request.sampling_options;
    let sampling_params = EngineCoreSamplingParams {
        temperature: sampling.temperature.unwrap_or(1.0),
        top_p: sampling.top_p.unwrap_or(1.0),
        top_k: normalize_top_k(sampling.top_k)?,
        seed: sampling.seed,
        max_tokens,
        min_tokens: request.stop_conditions.min_tokens.unwrap_or(0),
        logprobs: request
            .output_options
            .logprobs
            .map(u32_to_i32)
            .transpose()?,
        prompt_logprobs: request
            .output_options
            .prompt_logprobs
            .map(u32_to_i32)
            .transpose()?,
        min_p: sampling.min_p.unwrap_or(0.0),
        frequency_penalty: sampling.frequency_penalty.unwrap_or(0.0),
        presence_penalty: sampling.presence_penalty.unwrap_or(0.0),
        repetition_penalty: sampling.repetition_penalty.unwrap_or(1.0),
        stop_token_ids,
        eos_token_id,
        all_stop_token_ids,
        logit_bias: None,
        allowed_token_ids: None,
        bad_words_token_ids: None,
        structured_outputs: sampling
            .guided_decoding
            .as_ref()
            .map(structured_outputs_from_guided_decoding),
        logprob_token_ids: None,
        skip_reading_prefix_cache: None,
        extra_args: extra_args_as_object(request.extra_args)?,
    };

    let priority = request
        .routing
        .as_ref()
        .and_then(|routing| routing.priority)
        .unwrap_or(0);
    let data_parallel_rank = request.routing.as_ref().and_then(|routing| routing.dp_rank);

    Ok(GenerateRequest {
        request_id,
        prompt_token_ids: request.token_ids,
        sampling_params,
        arrival_time: request.request_timestamp_ms.map(|ms| ms / 1000.0),
        cache_salt: request.mdc_sum,
        trace_headers: None,
        priority,
        data_parallel_rank,
        reasoning_ended: None,
        lora_request: None,
    })
}

fn validate_request(request: &PreprocessedRequest) -> Result<(), DynamoError> {
    if request.token_ids.is_empty() {
        return Err(invalid_arg("token_ids must not be empty"));
    }
    if request.prompt_embeds.is_some() {
        return Err(invalid_arg(
            "prompt_embeds are not supported by this backend",
        ));
    }
    if request.multi_modal_data.is_some() || request.mm_processor_kwargs.is_some() {
        return Err(invalid_arg(
            "multimodal execution payloads are not supported by this backend",
        ));
    }

    let sampling = &request.sampling_options;
    if sampling.n.unwrap_or(1) != 1 {
        return Err(invalid_arg("n must be 1 for this backend"));
    }
    if sampling.best_of.unwrap_or(1) != 1 {
        return Err(invalid_arg("best_of must be 1 for this backend"));
    }
    if sampling.use_beam_search.unwrap_or(false) {
        return Err(invalid_arg("beam search is not supported by this backend"));
    }
    if let Some(length_penalty) = sampling.length_penalty
        && (length_penalty - 1.0).abs() > f32::EPSILON
    {
        return Err(invalid_arg(
            "non-default length_penalty is not supported by this backend",
        ));
    }
    Ok(())
}

fn normalize_top_k(top_k: Option<i32>) -> Result<u32, DynamoError> {
    match top_k {
        None | Some(-1) | Some(0) => Ok(0),
        Some(value) if value > 0 => Ok(value as u32),
        Some(value) => Err(invalid_arg(format!(
            "top_k must be -1, 0, or a positive integer; got {value}"
        ))),
    }
}

fn structured_outputs_from_guided_decoding(
    guided: &GuidedDecodingOptions,
) -> StructuredOutputsParams {
    StructuredOutputsParams {
        json: guided.json.clone(),
        regex: guided.regex.clone(),
        choice: guided.choice.clone(),
        grammar: guided.grammar.clone(),
        whitespace_pattern: guided.whitespace_pattern.clone(),
        structural_tag: guided.structural_tag.as_ref().map(structural_tag_to_string),
        ..Default::default()
    }
}

fn structural_tag_to_string(tag: &serde_json::Value) -> String {
    match tag {
        serde_json::Value::String(tag) => tag.clone(),
        tag => tag.to_string(),
    }
}

fn extra_args_as_object(
    extra_args: Option<serde_json::Value>,
) -> Result<Option<HashMap<String, serde_json::Value>>, DynamoError> {
    match extra_args {
        None => Ok(None),
        Some(serde_json::Value::Object(map)) => Ok(Some(map.into_iter().collect())),
        Some(_) => Err(invalid_arg(
            "extra_args must be a JSON object for vLLM engine-core",
        )),
    }
}

/// Maps one vLLM output chunk into the Dynamo backend output contract.
///
/// `prompt_tokens` and `completion_tokens` are supplied by the caller because
/// vLLM chunks carry generated token deltas while Dynamo terminal outputs also
/// advertise cumulative usage.
pub(crate) fn map_output(
    output: GenerateOutput,
    prompt_tokens: u32,
    completion_tokens: u32,
) -> LLMEngineOutput {
    let mut mapped = match output.finish_reason {
        None => LLMEngineOutput::default(),
        Some(VllmFinishReason::Stop(reason)) => {
            let mut mapped =
                LLMEngineOutput::stop().with_usage(usage(prompt_tokens, completion_tokens));
            mapped.stop_reason = reason.as_ref().map(map_stop_reason);
            mapped
        }
        Some(VllmFinishReason::Length) => {
            LLMEngineOutput::length().with_usage(usage(prompt_tokens, completion_tokens))
        }
        Some(VllmFinishReason::Abort) => {
            LLMEngineOutput::cancelled().with_usage(usage(prompt_tokens, completion_tokens))
        }
        Some(VllmFinishReason::Error) => LLMEngineOutput::error(
            "vLLM backend generation finished with an engine error".to_string(),
        )
        .with_usage(usage(prompt_tokens, completion_tokens)),
        Some(VllmFinishReason::Repetition) => {
            LLMEngineOutput::stop().with_usage(usage(prompt_tokens, completion_tokens))
        }
    };

    mapped.token_ids = output.token_ids;
    if let Some(logprobs) = output.logprobs.as_ref() {
        let (log_probs, top_logprobs) = map_logprobs(logprobs);
        mapped.log_probs = Some(log_probs);
        mapped.top_logprobs = Some(top_logprobs);
    }
    mapped.index = Some(0);
    mapped.disaggregated_params = output.kv_transfer_params;
    mapped
}

fn map_stop_reason(reason: &VllmStopReason) -> DynamoStopReason {
    match reason {
        VllmStopReason::TokenId(token_id) => DynamoStopReason::Int(i64::from(*token_id)),
        VllmStopReason::Text(text) => DynamoStopReason::String(text.clone()),
    }
}

fn map_logprobs(logprobs: &VllmLogprobs) -> (Vec<f64>, Vec<Vec<TopLogprob>>) {
    let log_probs = logprobs
        .positions
        .iter()
        .filter_map(|position| position.entries.first())
        .map(|entry| f64::from(entry.logprob))
        .collect();
    let top_logprobs = logprobs
        .positions
        .iter()
        .map(|position| {
            position
                .entries
                .iter()
                .map(|entry| TopLogprob {
                    rank: entry.rank,
                    token_id: entry.token_id,
                    token: None,
                    logprob: f64::from(entry.logprob),
                    bytes: None,
                })
                .collect()
        })
        .collect();

    (log_probs, top_logprobs)
}

fn u32_to_i32(value: u32) -> Result<i32, DynamoError> {
    i32::try_from(value).map_err(|_| {
        invalid_arg(format!(
            "vLLM logprobs request must fit in i32; got {value}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use dynamo_backend_common::{
        BackendError, ErrorType, FinishReason, GuidedDecodingOptions, OutputOptions,
        PreprocessedRequest, SamplingOptions, StopConditions, StopReason as DynamoStopReason,
    };
    use serde_json::json;
    use vllm_engine_core_client::protocol::StopReason as VllmStopReason;
    use vllm_llm::{FinishReason as VllmFinishReason, GenerateOutput, Logprobs};
    use vllm_llm::{PositionLogprobs, TokenLogprob};

    use super::{lower_request, map_output};

    #[test]
    fn lower_request_forwards_supported_engine_core_fields() {
        let mut request = sample_request();
        request.stop_conditions.max_tokens = Some(7);
        request.stop_conditions.min_tokens = Some(2);
        request.stop_conditions.stop = Some(vec!["</done>".to_string()]);
        request.stop_conditions.stop_token_ids_hidden = Some(vec![99]);
        request.eos_token_ids = vec![2, 3];
        request.sampling_options = SamplingOptions {
            temperature: Some(0.5),
            top_p: Some(0.9),
            top_k: Some(-1),
            min_p: Some(0.1),
            frequency_penalty: Some(0.2),
            presence_penalty: Some(0.3),
            repetition_penalty: Some(1.1),
            seed: Some(123),
            guided_decoding: Some(GuidedDecodingOptions::new(
                Some(json!({"type": "object"})),
                None,
                None,
                None,
                Some("ignored-backend".to_string()),
                Some(r"\s*".to_string()),
                None,
            )),
            ..Default::default()
        };
        request.output_options = OutputOptions {
            logprobs: Some(5),
            prompt_logprobs: Some(2),
            ..Default::default()
        };
        let routing = request.routing_mut();
        routing.priority = Some(4);
        routing.dp_rank = Some(1);
        request.mdc_sum = Some("cache-salt".to_string());
        request.extra_args = Some(json!({"custom": true}));

        let generate = lower_request("req-1".to_string(), request, Some(1024)).unwrap();

        assert_eq!(generate.request_id, "req-1");
        assert_eq!(generate.prompt_token_ids, vec![11, 22, 33]);
        assert_eq!(generate.priority, 4);
        assert_eq!(generate.data_parallel_rank, Some(1));
        assert_eq!(generate.cache_salt.as_deref(), Some("cache-salt"));

        let sampling = generate.sampling_params;
        assert_eq!(sampling.max_tokens, 7);
        assert_eq!(sampling.min_tokens, 2);
        assert_eq!(sampling.top_k, 0);
        assert_eq!(sampling.logprobs, Some(5));
        assert_eq!(sampling.prompt_logprobs, Some(2));
        assert_eq!(sampling.stop_token_ids, vec![99]);
        assert_eq!(sampling.eos_token_id, Some(2));
        assert_eq!(sampling.all_stop_token_ids, [2, 3, 99].into());
        let structured_outputs = sampling.structured_outputs.unwrap();
        assert_eq!(structured_outputs.json, Some(json!({"type": "object"})));
        assert_eq!(
            structured_outputs.whitespace_pattern.as_deref(),
            Some(r"\s*")
        );
        assert_eq!(
            sampling.extra_args.unwrap().get("custom"),
            Some(&json!(true))
        );
    }

    #[test]
    fn lower_request_defaults_max_tokens_from_context_length() {
        let request = sample_request();

        let generate = lower_request("req-1".to_string(), request, Some(10)).unwrap();

        assert_eq!(generate.sampling_params.max_tokens, 7);
    }

    #[test]
    fn lower_request_ignore_eos_keeps_explicit_stop_tokens() {
        let mut request = sample_request();
        request.stop_conditions.ignore_eos = Some(true);
        request.stop_conditions.stop_token_ids_hidden = Some(vec![99]);
        request.eos_token_ids = vec![2, 3];

        let generate = lower_request("req-1".to_string(), request, Some(10)).unwrap();

        assert_eq!(generate.sampling_params.stop_token_ids, vec![99]);
        assert_eq!(generate.sampling_params.eos_token_id, None);
        assert_eq!(generate.sampling_params.all_stop_token_ids, [99].into());
    }

    #[test]
    fn lower_request_rejects_currently_unsupported_payloads() {
        let mut request = sample_request();
        request.prompt_embeds = Some("base64".to_string());
        assert_invalid(lower_request("req-1".to_string(), request, Some(10)));

        let mut request = sample_request();
        request.multi_modal_data = Some(HashMap::new());
        assert_invalid(lower_request("req-1".to_string(), request, Some(10)));

        let mut request = sample_request();
        request.mm_processor_kwargs = Some(json!({"use_audio_in_video": true}));
        assert_invalid(lower_request("req-1".to_string(), request, Some(10)));
    }

    #[test]
    fn lower_request_rejects_multi_choice_and_beam_fields() {
        let mut request = sample_request();
        request.sampling_options.n = Some(2);
        assert_invalid(lower_request("req-1".to_string(), request, Some(10)));

        let mut request = sample_request();
        request.sampling_options.best_of = Some(2);
        assert_invalid(lower_request("req-1".to_string(), request, Some(10)));

        let mut request = sample_request();
        request.sampling_options.use_beam_search = Some(true);
        assert_invalid(lower_request("req-1".to_string(), request, Some(10)));

        let mut request = sample_request();
        request.sampling_options.length_penalty = Some(0.7);
        assert_invalid(lower_request("req-1".to_string(), request, Some(10)));
    }

    #[test]
    fn lower_request_rejects_oversized_logprobs() {
        let mut request = sample_request();
        request.output_options.logprobs = Some(i32::MAX as u32 + 1);
        assert_invalid(lower_request("req-1".to_string(), request, Some(10)));

        let mut request = sample_request();
        request.output_options.prompt_logprobs = Some(i32::MAX as u32 + 1);
        assert_invalid(lower_request("req-1".to_string(), request, Some(10)));
    }

    #[test]
    fn map_output_preserves_tokens_logprobs_stop_reason_usage_and_disagg_params() {
        let output = GenerateOutput {
            request_id: "req-1".to_string(),
            prompt_info: None,
            token_ids: vec![42],
            logprobs: Some(Logprobs {
                positions: vec![PositionLogprobs {
                    entries: vec![
                        TokenLogprob {
                            token_id: 42,
                            logprob: -0.25,
                            rank: 3,
                        },
                        TokenLogprob {
                            token_id: 7,
                            logprob: -1.5,
                            rank: 1,
                        },
                    ],
                }],
            }),
            finish_reason: Some(VllmFinishReason::Stop(Some(VllmStopReason::TokenId(42)))),
            kv_transfer_params: Some(json!({"connector": "kv"})),
        };

        let mapped = map_output(output, 3, 1);

        assert_eq!(mapped.token_ids, vec![42]);
        assert_eq!(mapped.finish_reason, Some(FinishReason::Stop));
        assert_eq!(mapped.stop_reason, Some(DynamoStopReason::Int(42)));
        assert_eq!(mapped.log_probs, Some(vec![-0.25]));
        let top_logprobs = mapped.top_logprobs.unwrap();
        assert_eq!(top_logprobs[0][0].token_id, 42);
        assert_eq!(top_logprobs[0][1].rank, 1);
        assert_eq!(
            mapped.disaggregated_params,
            Some(json!({"connector": "kv"}))
        );
        let usage = mapped.completion_usage.unwrap();
        assert_eq!(usage.prompt_tokens, 3);
        assert_eq!(usage.completion_tokens, 1);
        assert_eq!(usage.total_tokens, 4);
    }

    #[test]
    fn map_output_maps_terminal_finish_reasons() {
        let eos_stop = map_output(finished(VllmFinishReason::Stop(None)), 3, 2);
        assert_eq!(eos_stop.finish_reason, Some(FinishReason::Stop));
        assert_eq!(eos_stop.stop_reason, None);

        let text_stop = map_output(
            finished(VllmFinishReason::Stop(Some(VllmStopReason::Text(
                "</stop>".to_string(),
            )))),
            3,
            2,
        );
        assert_eq!(text_stop.finish_reason, Some(FinishReason::Stop));
        assert_eq!(
            text_stop.stop_reason,
            Some(DynamoStopReason::String("</stop>".to_string()))
        );

        let length = map_output(finished(VllmFinishReason::Length), 3, 2);
        assert_eq!(length.finish_reason, Some(FinishReason::Length));
        assert_eq!(length.stop_reason, None);

        let abort = map_output(finished(VllmFinishReason::Abort), 3, 2);
        assert_eq!(abort.finish_reason, Some(FinishReason::Cancelled));

        let error = map_output(finished(VllmFinishReason::Error), 3, 2);
        assert!(matches!(error.finish_reason, Some(FinishReason::Error(_))));

        let repetition = map_output(finished(VllmFinishReason::Repetition), 3, 2);
        assert_eq!(repetition.finish_reason, Some(FinishReason::Stop));
    }

    fn sample_request() -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("Qwen/Qwen3-0.6B".to_string())
            .token_ids(vec![11, 22, 33])
            .stop_conditions(StopConditions::default())
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .build()
            .unwrap()
    }

    fn assert_invalid<T: std::fmt::Debug>(result: Result<T, dynamo_backend_common::DynamoError>) {
        let error = result.unwrap_err();
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::InvalidArgument)
        );
    }

    fn finished(reason: VllmFinishReason) -> GenerateOutput {
        GenerateOutput {
            request_id: "req-1".to_string(),
            prompt_info: None,
            token_ids: vec![1, 2],
            logprobs: None,
            finish_reason: Some(reason),
            kv_transfer_params: None,
        }
    }
}
