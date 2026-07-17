// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conversion between Dynamo's `PreprocessedRequest` / `LLMEngineOutput` and
//! the TensorRT-LLM `TrtllmService` protobuf messages.
//!
//! Scope: aggregated generation. Disaggregation, multimodal, LoRA, beam search,
//! and `n > 1` are rejected before dispatch — the `Generate` response contract
//! carries no disaggregation handoff, and the sidecar streams a single sequence.

use std::collections::BTreeSet;

use dynamo_backend_common::{
    DynamoError, FinishReason, LLMEngineOutput, PreprocessedRequest, StopReason, TopLogprob, usage,
};

use crate::client;
use crate::proto as pb;

/// Per-chunk logprobs: the selected-token logprob sequence plus the per-token
/// top-k alternatives, both aligned with the chunk's delta tokens.
type MappedLogprobs = (Option<Vec<f64>>, Option<Vec<Vec<TopLogprob>>>);

pub(crate) fn build_generate_request(
    request: &PreprocessedRequest,
    request_id: &str,
    context_length: Option<u32>,
) -> Result<pb::GenerateRequest, DynamoError> {
    validate_request(request)?;

    let sampling = &request.sampling_options;
    let stop = &request.stop_conditions;
    let output = &request.output_options;

    Ok(pb::GenerateRequest {
        request_id: request_id.to_string(),
        tokenized: Some(pb::TokenizedInput {
            original_text: String::new(),
            input_token_ids: request.token_ids.clone(),
            query_token_ids: Vec::new(),
        }),
        sampling_config: Some(pb::SamplingConfig {
            beam_width: 1,
            num_return_sequences: 1,
            top_k: normalize_top_k(sampling.top_k),
            top_p: sampling.top_p,
            temperature: sampling.temperature,
            min_p: sampling.min_p,
            seed: normalize_seed(sampling.seed),
            min_tokens: stop.min_tokens,
            repetition_penalty: sampling.repetition_penalty,
            presence_penalty: sampling.presence_penalty,
            frequency_penalty: sampling.frequency_penalty,
            ..Default::default()
        }),
        output_config: Some(pb::OutputConfig {
            logprobs: output.logprobs.map(logprob_count).transpose()?,
            // prompt_logprobs is intentionally not requested: it is rejected in
            // `validate_request` (no `LLMEngineOutput` field to surface it).
            // TRT-LLM streams delta tokens; input tokens must not be echoed.
            exclude_input_from_output: true,
            ..Default::default()
        }),
        max_tokens: max_tokens(request, context_length)?,
        streaming: true,
        guided_decoding: guided_decoding(request)?,
        stop: stop.stop.clone().unwrap_or_default(),
        stop_token_ids: stop_token_ids(request),
        ignore_eos: stop.ignore_eos.unwrap_or(false),
        include_stop_token_in_output: sampling.include_stop_str_in_output.unwrap_or(false),
        ..Default::default()
    })
}

// Temporary workaround: TensorRT-LLM's gRPC `GenerateRequest.max_tokens` is
// REQUIRED (see proto/trtllm_service.proto), so an omitted `max_tokens` — which
// the Dynamo frontend forwards as `None` for the backend to default — has no
// natural value. We mirror the in-process backend's text-only default,
// `max(1, context_length - prompt_len)` (components/src/dynamo/trtllm
// `_default_max_tokens`); the sidecar rejects multimodal before dispatch, so
// `token_ids.len()` is the true prompt length. `context_length` comes from
// `--context-length` (GetModelInfo returns zero on current releases).
//
// Remove when https://github.com/NVIDIA/TensorRT-LLM/issues/16549 lands (gRPC
// `max_tokens` made optional): drop this fallback, the `--context-length` arg,
// and the context-length plumbing in `engine.rs`, and forward an omitted
// `max_tokens` as unset.
fn max_tokens(
    request: &PreprocessedRequest,
    context_length: Option<u32>,
) -> Result<u32, DynamoError> {
    if let Some(max_tokens) = request.stop_conditions.max_tokens {
        return Ok(max_tokens);
    }
    let context_length = context_length.ok_or_else(|| {
        client::invalid_argument(
            "TensorRT-LLM requires max_tokens, and no model context length is available to \
             derive a default; specify max_tokens explicitly",
        )
    })?;
    let prompt_len = request.token_ids.len() as u32;
    Ok(context_length.saturating_sub(prompt_len).max(1))
}

fn normalize_top_k(top_k: Option<i32>) -> Option<i32> {
    // Dynamo uses -1 (or absence) for "consider all tokens"; TRT-LLM treats an
    // unset/0 top_k the same way. Only forward a positive cap.
    match top_k {
        Some(value) if value > 0 => Some(value),
        _ => None,
    }
}

fn normalize_seed(seed: Option<i64>) -> Option<u64> {
    seed.and_then(|seed| u64::try_from(seed).ok())
}

fn logprob_count(count: u32) -> Result<i32, DynamoError> {
    i32::try_from(count).map_err(|_| {
        client::invalid_argument(format!("logprobs request must fit in i32; got {count}"))
    })
}

fn stop_token_ids(request: &PreprocessedRequest) -> Vec<u32> {
    let stop = &request.stop_conditions;
    let mut ids = BTreeSet::new();
    for values in [
        stop.stop_token_ids.as_ref(),
        stop.stop_token_ids_hidden.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        ids.extend(values.iter().copied());
    }
    ids.into_iter().collect()
}

fn guided_decoding(
    request: &PreprocessedRequest,
) -> Result<Option<pb::GuidedDecodingParams>, DynamoError> {
    let Some(guided) = request.sampling_options.guided_decoding.as_ref() else {
        return Ok(None);
    };
    if guided.backend.is_some() || guided.whitespace_pattern.is_some() {
        return Err(client::invalid_argument(
            "guided decoding backend and whitespace_pattern are not supported by TensorRT-LLM gRPC",
        ));
    }

    use pb::guided_decoding_params::GuideType;
    let mut constraints = Vec::new();
    if let Some(json) = &guided.json {
        constraints.push((GuideType::JsonSchema, json_guide(json)));
    }
    if let Some(regex) = &guided.regex {
        constraints.push((GuideType::Regex, regex.clone()));
    }
    if let Some(grammar) = &guided.grammar {
        constraints.push((GuideType::EbnfGrammar, grammar.clone()));
    }
    if let Some(tag) = &guided.structural_tag {
        constraints.push((GuideType::StructuralTag, json_guide(tag)));
    }
    if guided.choice.is_some() {
        return Err(client::invalid_argument(
            "guided decoding `choice` is not supported by TensorRT-LLM gRPC",
        ));
    }
    if constraints.len() > 1 {
        return Err(client::invalid_argument(
            "only one guided decoding constraint may be set",
        ));
    }
    Ok(constraints
        .pop()
        .map(|(guide_type, guide)| pb::GuidedDecodingParams {
            guide_type: guide_type as i32,
            guide,
        }))
}

/// A guide is either a JSON string carried verbatim or a JSON value rendered to
/// its string form (schema / structural tag).
fn json_guide(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(guide) => guide.clone(),
        value => value.to_string(),
    }
}

fn validate_request(request: &PreprocessedRequest) -> Result<(), DynamoError> {
    if request.token_ids.is_empty() {
        return Err(client::invalid_argument("token_ids must not be empty"));
    }
    if request.prompt_embeds.is_some() {
        return Err(client::invalid_argument(
            "prompt embeddings are not supported by the TensorRT-LLM sidecar",
        ));
    }
    if request.multi_modal_data.is_some()
        || request.mm_routing_info.is_some()
        || request.encoder_result.is_some()
    {
        return Err(client::invalid_argument(
            "multimodal requests are not supported by the TensorRT-LLM sidecar",
        ));
    }
    if request.prefill_result.is_some() {
        return Err(client::invalid_argument(
            "disaggregated (prefill/decode) requests are not supported by the TensorRT-LLM sidecar",
        ));
    }
    if request.output_options.prompt_logprobs.is_some() {
        // TRT-LLM would compute these, but the terminal `complete` carries them
        // with no `LLMEngineOutput` field to surface — reject rather than pay for
        // and drop them.
        return Err(client::invalid_argument(
            "prompt logprobs are not supported by the TensorRT-LLM sidecar",
        ));
    }
    if request
        .stop_conditions
        .stop_token_ids_visible
        .as_ref()
        .is_some_and(|ids| !ids.is_empty())
    {
        // A visible stop token must halt generation *and* stay in the output.
        // TRT-LLM's single `include_stop_token_in_output` flag cannot honor that
        // per token, so reject rather than silently drop or mis-retain them.
        return Err(client::invalid_argument(
            "visible stop token IDs are not supported by the TensorRT-LLM sidecar",
        ));
    }
    if request
        .routing
        .as_ref()
        .and_then(|routing| routing.lora_name.as_deref())
        .is_some_and(|name| !name.is_empty())
    {
        return Err(client::invalid_argument(
            "LoRA request selection is not supported by the TensorRT-LLM sidecar",
        ));
    }
    if request
        .routing
        .as_ref()
        .is_some_and(|routing| routing.dp_rank.is_some() || routing.prefill_dp_rank.is_some())
    {
        return Err(client::invalid_argument(
            "KV-aware data-parallel routing is not supported by the TensorRT-LLM sidecar",
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

/// Streaming response reducer. TRT-LLM streams delta `chunk`s followed by a
/// single terminal `complete`; this maps each onto an `LLMEngineOutput`.
pub(crate) struct ResponseState {
    prompt_tokens: u32,
    completion_tokens: u32,
    expect_output_logprobs: bool,
}

impl ResponseState {
    pub(crate) fn new(request: &PreprocessedRequest) -> Self {
        Self {
            prompt_tokens: request.token_ids.len() as u32,
            completion_tokens: 0,
            expect_output_logprobs: request.output_options.logprobs.is_some(),
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
    ) -> Result<Option<LLMEngineOutput>, DynamoError> {
        match response.response {
            Some(pb::generate_response::Response::Chunk(chunk)) => self.convert_chunk(chunk),
            Some(pb::generate_response::Response::Complete(complete)) => {
                self.convert_complete(complete).map(Some)
            }
            None => Ok(None),
        }
    }

    /// Validates the single-sequence invariant and folds the response's usage
    /// counts into the running totals. Shared by chunk and terminal handling.
    fn accumulate(
        &mut self,
        sequence_index: u32,
        prompt_tokens: u32,
        completion_tokens: u32,
    ) -> Result<(), DynamoError> {
        if sequence_index != 0 {
            return Err(client::protocol_error(format!(
                "received unsupported sequence index {sequence_index}"
            )));
        }
        if prompt_tokens != 0 {
            self.prompt_tokens = prompt_tokens;
        }
        // TRT-LLM reports cumulative (not per-chunk-delta) usage counts, verified
        // against 1.3.0rc21 over multi-chunk streams; `max` tolerates chunks that
        // omit the count (report 0) without regressing the running total.
        self.completion_tokens = self.completion_tokens.max(completion_tokens);
        Ok(())
    }

    fn convert_chunk(
        &mut self,
        chunk: pb::GenerateStreamChunk,
    ) -> Result<Option<LLMEngineOutput>, DynamoError> {
        self.accumulate(
            chunk.sequence_index,
            chunk.prompt_tokens,
            chunk.completion_tokens,
        )?;
        if chunk.token_ids.is_empty() {
            return Ok(None);
        }
        let (log_probs, top_logprobs) = self.map_logprobs(&chunk.token_ids, &chunk.logprobs)?;
        Ok(Some(LLMEngineOutput {
            token_ids: chunk.token_ids,
            log_probs,
            top_logprobs,
            index: Some(0),
            ..Default::default()
        }))
    }

    fn convert_complete(
        &mut self,
        complete: pb::GenerateComplete,
    ) -> Result<LLMEngineOutput, DynamoError> {
        self.accumulate(
            complete.sequence_index,
            complete.prompt_tokens,
            complete.completion_tokens,
        )?;

        // The delta tokens were already streamed via chunks; the terminal
        // response only carries finish state and usage.
        let mut terminal = LLMEngineOutput {
            index: Some(0),
            finish_reason: Some(finish_reason(&complete.finish_reason)),
            completion_usage: Some(usage(self.prompt_tokens, self.completion_tokens)),
            ..Default::default()
        };
        terminal.stop_reason = complete.matched_stop.map(|matched| match matched {
            pb::generate_complete::MatchedStop::MatchedStopStr(value) => StopReason::String(value),
            pb::generate_complete::MatchedStop::MatchedTokenId(id) => {
                StopReason::Int(i64::from(id))
            }
        });
        Ok(terminal)
    }

    fn map_logprobs(
        &self,
        token_ids: &[u32],
        logprobs: &[pb::TokenLogprob],
    ) -> Result<MappedLogprobs, DynamoError> {
        if !self.expect_output_logprobs {
            return Ok((None, None));
        }
        // Logprobs were requested, so every delta token must carry one; an empty
        // or short slice is a protocol violation, not a silent no-op.
        if logprobs.len() != token_ids.len() {
            return Err(client::protocol_error(format!(
                "logprob count {} does not match {} delta tokens",
                logprobs.len(),
                token_ids.len()
            )));
        }
        let log_probs = logprobs.iter().map(|lp| f64::from(lp.logprob)).collect();
        let top_logprobs = logprobs
            .iter()
            .map(|lp| {
                if lp.top_logprobs.is_empty() {
                    vec![TopLogprob {
                        rank: 0,
                        token_id: lp.token_id,
                        token: None,
                        logprob: f64::from(lp.logprob),
                        bytes: None,
                    }]
                } else {
                    lp.top_logprobs
                        .iter()
                        .enumerate()
                        .map(|(rank, candidate)| TopLogprob {
                            rank: rank as u32,
                            token_id: candidate.token_id,
                            token: None,
                            logprob: f64::from(candidate.logprob),
                            bytes: None,
                        })
                        .collect()
                }
            })
            .collect();
        Ok((Some(log_probs), Some(top_logprobs)))
    }
}

fn finish_reason(reason: &str) -> FinishReason {
    match reason.trim().to_ascii_lowercase().as_str() {
        "length" => FinishReason::Length,
        "cancelled" | "canceled" | "aborted" | "timeout" => FinishReason::Cancelled,
        "error" => FinishReason::Error("TensorRT-LLM reported an error finish".to_string()),
        // "stop", "stop_word", "eos", or an empty string (finished implies stop).
        _ => FinishReason::Stop,
    }
}
