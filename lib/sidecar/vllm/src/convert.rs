// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::{
    DisaggregationMode, DynamoError, GuidedDecodingOptions, LLMEngineOutput, PrefillResult,
    PreprocessedRequest, StopReason, TopLogprob, usage,
};

use crate::client;
use crate::json::{json_to_struct, struct_to_json};
use crate::proto as pb;

const VLLM_LOGPROB_FLOOR: f64 = -9999.0;

pub(crate) fn build_generate_request(
    request: PreprocessedRequest,
    request_id: String,
    mode: DisaggregationMode,
) -> Result<pb::GenerateRequest, DynamoError> {
    validate_request(&request, mode)?;

    let prompt_logprobs = request.output_options.prompt_logprobs;
    let output_logprobs = request.output_options.logprobs;
    let max_new_tokens = if mode.is_prefill() {
        1
    } else {
        request.stop_conditions.max_tokens.unwrap_or(0)
    };
    let min_new_tokens = if mode.is_prefill() {
        1
    } else {
        request.stop_conditions.min_tokens.unwrap_or(0)
    };
    let mut routing = request.routing;
    let priority = routing
        .as_ref()
        .and_then(|routing| routing.priority)
        .unwrap_or(0);
    let cache_salt = routing
        .as_mut()
        .and_then(|routing| routing.cache_namespace.take())
        .or(request.mdc_sum);

    let sampling = request.sampling_options;
    let stop_conditions = request.stop_conditions;
    let kv = build_kv_parameters(request.extra_args, request.prefill_result, cache_salt, mode)?;

    Ok(pb::GenerateRequest {
        request_id,
        model: String::new(),
        prompt: Some(pb::generate_request::Prompt::TokenIds(pb::TokenIds {
            ids: request.token_ids,
        })),
        temperature: sampling.temperature,
        sampling: Some(pb::RandomSampling {
            num_sequences: 1,
            top_k: normalize_top_k(sampling.top_k)?,
            top_p: sampling.top_p.unwrap_or(0.0),
            min_p: sampling.min_p.unwrap_or(0.0),
            seed: sampling.seed,
        }),
        decoding: Some(pb::DecodingParameters {
            presence_penalty: sampling.presence_penalty.unwrap_or(0.0),
            frequency_penalty: sampling.frequency_penalty.unwrap_or(0.0),
            repetition_penalty: sampling.repetition_penalty.unwrap_or(0.0),
            logit_bias: Default::default(),
            allowed_token_ids: Vec::new(),
            structured_output: structured_output(sampling.guided_decoding)?,
        }),
        stopping: Some(pb::StoppingCriteria {
            max_new_tokens,
            min_new_tokens,
            stop_token_ids: stop_token_ids(
                stop_conditions.stop_token_ids,
                stop_conditions.stop_token_ids_hidden,
            ),
            stop_strings: stop_conditions.stop.unwrap_or_default(),
            include_stop_strings: sampling.include_stop_str_in_output.unwrap_or(false),
            ignore_eos: stop_conditions.ignore_eos.unwrap_or(false),
        }),
        response: Some(pb::ResponseOptions {
            prompt_token_ids: prompt_logprobs.is_some(),
            prompt_logprobs: prompt_logprobs.is_some(),
            prompt_candidates: prompt_logprobs.map(top_n_candidates).transpose()?,
            output_text: Some(true),
            output_token_ids: true,
            output_logprobs: output_logprobs.is_some(),
            output_candidates: output_logprobs.map(top_n_candidates).transpose()?,
        }),
        kv: Some(kv),
        truncate_prompt_tokens: 0,
        priority,
    })
}

fn top_n_candidates(count: u32) -> Result<pb::CandidateTokens, DynamoError> {
    i32::try_from(count).map_err(|_| {
        client::invalid_argument(format!(
            "vLLM logprobs request must fit in i32; got {count}"
        ))
    })?;
    Ok(pb::CandidateTokens {
        select: Some(pb::candidate_tokens::Select::TopN(count)),
    })
}

fn normalize_top_k(top_k: Option<i32>) -> Result<u32, DynamoError> {
    match top_k {
        None | Some(-1) | Some(0) => Ok(0),
        Some(value) if value > 0 => Ok(value as u32),
        Some(value) => Err(client::invalid_argument(format!(
            "top_k must be -1, 0, or positive; got {value}"
        ))),
    }
}

fn stop_token_ids(visible: Option<Vec<u32>>, hidden: Option<Vec<u32>>) -> Vec<u32> {
    let mut ids = visible.unwrap_or_default();
    if let Some(hidden) = hidden {
        ids.extend(hidden);
    }
    ids.sort_unstable();
    ids.dedup();
    ids
}

fn structured_output(
    guided: Option<GuidedDecodingOptions>,
) -> Result<Option<pb::decoding_parameters::StructuredOutput>, DynamoError> {
    let Some(guided) = guided else {
        return Ok(None);
    };
    if guided.backend.is_some() || guided.whitespace_pattern.is_some() {
        return Err(client::invalid_argument(
            "guided decoding backend and whitespace_pattern are not supported by vLLM gRPC v0.25.1",
        ));
    }

    use pb::decoding_parameters::StructuredOutput;
    let mut values = Vec::new();
    if let Some(json) = guided.json {
        values.push(StructuredOutput::Json(json.to_string()));
    }
    if let Some(regex) = guided.regex {
        values.push(StructuredOutput::Regex(regex));
    }
    if let Some(choice) = guided.choice {
        values.push(StructuredOutput::Choice(
            pb::decoding_parameters::StringChoices { choices: choice },
        ));
    }
    if let Some(grammar) = guided.grammar {
        values.push(StructuredOutput::Grammar(grammar));
    }
    if let Some(tag) = guided.structural_tag {
        values.push(StructuredOutput::StructuralTag(match tag {
            serde_json::Value::String(tag) => tag,
            tag => tag.to_string(),
        }));
    }
    if values.len() > 1 {
        return Err(client::invalid_argument(
            "only one structured output constraint may be set",
        ));
    }
    Ok(values.pop())
}

fn build_kv_parameters(
    extra_args: Option<serde_json::Value>,
    prefill_result: Option<PrefillResult>,
    cache_salt: Option<String>,
    mode: DisaggregationMode,
) -> Result<pb::KvCacheParameters, DynamoError> {
    let mut extra = match extra_args {
        None => None,
        Some(serde_json::Value::Object(extra)) => Some(extra),
        Some(_) => {
            return Err(client::invalid_argument("extra_args must be a JSON object"));
        }
    };
    if let Some(extra) = extra.as_ref() {
        for key in extra.keys() {
            if !matches!(
                key.as_str(),
                "bypass_prefix_cache" | "skip_reading_prefix_cache" | "kv_transfer_params"
            ) {
                return Err(client::invalid_argument(format!(
                    "extra_args.{key} is not supported by vLLM gRPC v0.25.1"
                )));
            }
        }
    }

    let bypass_prefix_cache = bool_extra(extra.as_ref(), "bypass_prefix_cache")?
        .or(bool_extra(extra.as_ref(), "skip_reading_prefix_cache")?)
        .unwrap_or(false);
    let caller_kv = extra
        .as_mut()
        .and_then(|extra| extra.remove("kv_transfer_params"));

    let kv_transfer_params = match mode {
        DisaggregationMode::Aggregated => caller_kv,
        DisaggregationMode::Prefill => {
            let mut params = match caller_kv {
                None => serde_json::Map::new(),
                Some(serde_json::Value::Object(params)) => params,
                Some(_) => {
                    return Err(client::invalid_argument(
                        "extra_args.kv_transfer_params must be a JSON object",
                    ));
                }
            };
            params.insert(
                "do_remote_decode".to_string(),
                serde_json::Value::Bool(true),
            );
            Some(serde_json::Value::Object(params))
        }
        DisaggregationMode::Decode => Some(
            prefill_result
                .ok_or_else(|| {
                    client::invalid_argument(
                        "decode request is missing the prefill_result KV payload",
                    )
                })?
                .disaggregated_params,
        ),
        DisaggregationMode::Encode => {
            return Err(client::invalid_argument(
                "encode mode is not supported by the vLLM sidecar",
            ));
        }
    };

    Ok(pb::KvCacheParameters {
        bypass_prefix_cache,
        cache_salt: cache_salt.unwrap_or_default(),
        kv_transfer_params: kv_transfer_params.map(json_to_struct).transpose()?,
    })
}

fn bool_extra(
    extra: Option<&serde_json::Map<String, serde_json::Value>>,
    key: &str,
) -> Result<Option<bool>, DynamoError> {
    match extra.and_then(|extra| extra.get(key)) {
        None => Ok(None),
        Some(serde_json::Value::Bool(value)) => Ok(Some(*value)),
        Some(_) => Err(client::invalid_argument(format!(
            "extra_args.{key} must be a boolean"
        ))),
    }
}

fn validate_request(
    request: &PreprocessedRequest,
    mode: DisaggregationMode,
) -> Result<(), DynamoError> {
    if request.token_ids.is_empty() {
        return Err(client::invalid_argument("token_ids must not be empty"));
    }
    if request.prompt_embeds.is_some() {
        return Err(client::invalid_argument(
            "prompt embeddings are not supported by vLLM gRPC v0.25.1",
        ));
    }
    if request.multi_modal_data.is_some()
        || request.mm_routing_info.is_some()
        || request.mm_processor_kwargs.is_some()
        || request.encoder_result.is_some()
    {
        return Err(client::invalid_argument(
            "multimodal requests are not supported by vLLM gRPC v0.25.1",
        ));
    }
    if mode.is_encode() {
        return Err(client::invalid_argument(
            "encode mode is not supported by the vLLM sidecar",
        ));
    }
    if request
        .routing
        .as_ref()
        .and_then(|routing| routing.lora_name.as_deref())
        .is_some_and(|name| !name.is_empty())
    {
        return Err(client::invalid_argument(
            "LoRA request selection is not supported by vLLM gRPC v0.25.1",
        ));
    }
    if request
        .routing
        .as_ref()
        .is_some_and(|routing| routing.dp_rank.is_some() || routing.prefill_dp_rank.is_some())
    {
        return Err(client::invalid_argument(
            "KV-aware data-parallel routing is not supported by vLLM gRPC v0.25.1",
        ));
    }
    if request.bootstrap_info.is_some() {
        return Err(client::invalid_argument(
            "Dynamo bootstrap handoff is not supported by the vLLM sidecar",
        ));
    }
    if request
        .stop_conditions
        .stop_token_ids_visible
        .as_ref()
        .is_some_and(|ids| !ids.is_empty())
    {
        return Err(client::invalid_argument(
            "visible stop token IDs are not supported by vLLM gRPC v0.25.1",
        ));
    }
    if request.stop_conditions.max_thinking_tokens.is_some() {
        return Err(client::invalid_argument(
            "max_thinking_tokens is not supported by vLLM gRPC v0.25.1",
        ));
    }
    if request.output_options.skip_special_tokens == Some(false) {
        return Err(client::invalid_argument(
            "skip_special_tokens=false is not supported by vLLM gRPC v0.25.1",
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
    if let Some(length_penalty) = sampling.length_penalty
        && (length_penalty - 1.0).abs() > f32::EPSILON
    {
        return Err(client::invalid_argument(
            "non-default length_penalty is not supported",
        ));
    }
    Ok(())
}

pub(crate) struct ResponseState {
    prompt_tokens: u32,
    completion_tokens: u32,
    is_prefill: bool,
    output_logprobs: Option<u32>,
    expect_prompt_logprobs: bool,
    prompt_info: Option<pb::PromptInfo>,
}

impl ResponseState {
    pub(crate) fn new(request: &PreprocessedRequest, mode: DisaggregationMode) -> Self {
        Self {
            prompt_tokens: request.token_ids.len() as u32,
            completion_tokens: 0,
            is_prefill: mode.is_prefill(),
            output_logprobs: request.output_options.logprobs,
            expect_prompt_logprobs: request.output_options.prompt_logprobs.is_some(),
            prompt_info: None,
        }
    }

    pub(crate) fn reported_completion_tokens(&self) -> u32 {
        if self.is_prefill {
            0
        } else {
            self.completion_tokens
        }
    }

    pub(crate) fn prompt_tokens(&self) -> u32 {
        self.prompt_tokens
    }

    pub(crate) fn convert(
        &mut self,
        response: pb::GenerateResponse,
    ) -> Result<Option<LLMEngineOutput>, DynamoError> {
        if let Some(prompt) = response.prompt_info {
            self.consume_prompt_info(prompt)?;
        }
        let Some(output) = response.outputs else {
            return Ok(None);
        };
        if output.index != 0 {
            return Err(client::protocol_error(format!(
                "received unsupported sequence index {}",
                output.index
            )));
        }
        if output.num_tokens as usize != output.token_ids.len() {
            return Err(client::protocol_error(format!(
                "num_tokens {} does not match {} token IDs",
                output.num_tokens,
                output.token_ids.len()
            )));
        }

        let mapped_logprobs = if let Some(count) = self.output_logprobs
            && !self.is_prefill
        {
            Some(map_output_logprobs(&output, count > 0)?)
        } else {
            None
        };
        let pb::SequenceOutput {
            text,
            num_tokens,
            token_ids,
            finish_info,
            ..
        } = output;

        self.completion_tokens = self.completion_tokens.saturating_add(num_tokens);
        let mut mapped = LLMEngineOutput {
            token_ids: if self.is_prefill {
                Vec::new()
            } else {
                token_ids
            },
            text: if self.is_prefill || text.is_empty() {
                None
            } else {
                Some(text)
            },
            index: Some(0),
            ..Default::default()
        };
        if let Some(logprobs) = mapped_logprobs {
            mapped.log_probs = Some(logprobs.selected);
            mapped.top_logprobs = logprobs.top;
        }

        let Some(finish) = finish_info else {
            return if self.is_prefill || num_tokens == 0 {
                Ok(None)
            } else {
                Ok(Some(mapped))
            };
        };
        if finish.num_output_tokens != self.completion_tokens {
            return Err(client::protocol_error(format!(
                "terminal num_output_tokens {} does not match streamed count {}",
                finish.num_output_tokens, self.completion_tokens
            )));
        }

        let reason =
            pb::finish_info::FinishReason::try_from(finish.finish_reason).map_err(|_| {
                client::protocol_error(format!("unknown finish reason {}", finish.finish_reason))
            })?;
        let completion_tokens = self.reported_completion_tokens();
        mapped.finish_reason = Some(match reason {
            pb::finish_info::FinishReason::Length => dynamo_backend_common::FinishReason::Length,
            pb::finish_info::FinishReason::Stop => dynamo_backend_common::FinishReason::Stop,
            pb::finish_info::FinishReason::Aborted => {
                dynamo_backend_common::FinishReason::Cancelled
            }
            pb::finish_info::FinishReason::NotFinished => {
                return Err(client::protocol_error(
                    "terminal response has NOT_FINISHED finish reason",
                ));
            }
        });
        mapped.stop_reason = finish.stop_reason.map(|reason| match reason {
            pb::finish_info::StopReason::StopTokenId(id)
            | pb::finish_info::StopReason::EosTokenId(id) => StopReason::Int(i64::from(id)),
            pb::finish_info::StopReason::StopString(value) => StopReason::String(value),
        });
        mapped.completion_usage = Some(usage(self.prompt_tokens, completion_tokens));
        mapped.disaggregated_params = finish.kv_transfer_params.map(struct_to_json).transpose()?;
        if self.is_prefill && mapped.disaggregated_params.is_none() {
            return Err(client::protocol_error(
                "prefill terminal is missing kv_transfer_params",
            ));
        }
        self.attach_prompt_data(&mut mapped);
        Ok(Some(mapped))
    }

    fn attach_prompt_data(&mut self, output: &mut LLMEngineOutput) {
        if output.engine_data.is_none() {
            output.engine_data = self.prompt_info.take().map(prompt_logprobs_to_json);
        }
    }

    fn consume_prompt_info(&mut self, prompt: pb::PromptInfo) -> Result<(), DynamoError> {
        if prompt.num_prompt_tokens != self.prompt_tokens {
            return Err(client::protocol_error(format!(
                "prompt token count {} does not match request count {}",
                prompt.num_prompt_tokens, self.prompt_tokens
            )));
        }
        if !self.expect_prompt_logprobs {
            return Ok(());
        }
        let count = prompt.num_prompt_tokens as usize;
        if prompt.token_ids.len() != count
            || prompt.logprobs.len() != count
            || prompt.ranks.len() != count
            || prompt.candidate_tokens.len() != count
        {
            return Err(client::protocol_error(format!(
                "prompt logprob array lengths do not match expected count {count}: token_ids={}, logprobs={}, ranks={}, candidate_tokens={}",
                prompt.token_ids.len(),
                prompt.logprobs.len(),
                prompt.ranks.len(),
                prompt.candidate_tokens.len(),
            )));
        }

        self.prompt_info = Some(prompt);
        Ok(())
    }
}

fn prompt_logprobs_to_json(prompt: pb::PromptInfo) -> serde_json::Value {
    let count = prompt.num_prompt_tokens as usize;
    let mut positions = Vec::with_capacity(count);
    let rows = prompt
        .token_ids
        .into_iter()
        .zip(prompt.logprobs)
        .zip(prompt.ranks)
        .zip(prompt.candidate_tokens);
    for (index, (((token_id, logprob), rank), candidates)) in rows.enumerate() {
        if index == 0 {
            positions.push(serde_json::Value::Null);
            continue;
        }
        let mut entries = serde_json::Map::with_capacity(candidates.tokens.len() + 1);
        entries.insert(token_id.to_string(), prompt_logprob_entry(logprob, rank));
        for candidate in candidates.tokens {
            entries.insert(
                candidate.id.to_string(),
                prompt_logprob_entry(candidate.logprob, candidate.rank),
            );
        }
        positions.push(serde_json::Value::Object(entries));
    }

    let mut engine_data = serde_json::Map::with_capacity(1);
    engine_data.insert(
        "prompt_logprobs".to_string(),
        serde_json::Value::Array(positions),
    );
    serde_json::Value::Object(engine_data)
}

fn prompt_logprob_entry(logprob: f32, rank: u32) -> serde_json::Value {
    let mut entry = serde_json::Map::with_capacity(2);
    entry.insert(
        "logprob".to_string(),
        serde_json::Value::from(normalize_logprob(logprob)),
    );
    entry.insert("rank".to_string(), serde_json::Value::from(rank));
    serde_json::Value::Object(entry)
}

struct OutputLogprobs {
    selected: Vec<f64>,
    top: Option<Vec<Vec<TopLogprob>>>,
}

fn map_output_logprobs(
    output: &pb::SequenceOutput,
    include_top_logprobs: bool,
) -> Result<OutputLogprobs, DynamoError> {
    let count = output.token_ids.len();
    if output.logprobs.len() != count
        || output.ranks.len() != count
        || output.candidate_tokens.len() != count
    {
        return Err(client::protocol_error(format!(
            "output logprob array lengths do not match expected count {count}: token_ids={}, logprobs={}, ranks={}, candidate_tokens={}",
            output.token_ids.len(),
            output.logprobs.len(),
            output.ranks.len(),
            output.candidate_tokens.len(),
        )));
    }
    let log_probs = output
        .logprobs
        .iter()
        .copied()
        .map(normalize_logprob)
        .collect();
    let top_logprobs = include_top_logprobs.then(|| {
        output
            .token_ids
            .iter()
            .enumerate()
            .map(|(index, token_id)| {
                let mut entries =
                    Vec::with_capacity(output.candidate_tokens[index].tokens.len() + 1);
                entries.push(TopLogprob {
                    rank: output.ranks[index],
                    token_id: *token_id,
                    token: None,
                    logprob: normalize_logprob(output.logprobs[index]),
                    bytes: None,
                });
                entries.extend(
                    output.candidate_tokens[index]
                        .tokens
                        .iter()
                        .map(|candidate| TopLogprob {
                            rank: candidate.rank,
                            token_id: candidate.id,
                            token: None,
                            logprob: normalize_logprob(candidate.logprob),
                            bytes: None,
                        }),
                );
                entries
            })
            .collect()
    });
    Ok(OutputLogprobs {
        selected: log_probs,
        top: top_logprobs,
    })
}

fn normalize_logprob(logprob: f32) -> f64 {
    if logprob.is_finite() {
        f64::from(logprob).max(VLLM_LOGPROB_FLOOR)
    } else {
        VLLM_LOGPROB_FLOOR
    }
}
