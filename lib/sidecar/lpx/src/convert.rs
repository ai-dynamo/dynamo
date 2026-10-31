// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conversion between Dynamo's `PreprocessedRequest` / `LLMEngineOutput` and
//! the LPX engines' OpenEngine (`openengine.v1`) messages.
//!
//! The sidecar forwards every field a client set that OpenEngine can express
//! and lets the engine decide what it supports; the engine's rejection
//! reaches the client as a 4xx. The LPU-only engine accepts and ignores
//! sampling penalties rather than rejecting them. The sidecar rejects a field
//! itself only when OpenEngine has no place for it, when dropping it would
//! silently change the request's meaning, or when the response could not
//! carry the result back.

use std::collections::BTreeSet;
use std::sync::Arc;

use dynamo_backend_common::{
    BackendError, CompletionUsage, DynamoError, ErrorType, FinishReason, LLMEngineOutput,
    PrefillResult, PreprocessedRequest, PromptTokensDetails, StopReason, usage,
};

use crate::client::{engine_shutdown, invalid_argument, protocol_error, status_to_dynamo};
use crate::handoff;
use crate::model::{DiscoveredEngine, engine_label};
use crate::pb;

pub(crate) fn build_generate_request(
    request: &PreprocessedRequest,
    request_id: &str,
    engine: &DiscoveredEngine,
) -> Result<pb::GenerateRequest, DynamoError> {
    validate_request(request)?;

    let sampling = &request.sampling_options;
    let stop = &request.stop_conditions;
    let kv = match request.prefill_result.as_ref() {
        Some(prefill) => Some(pb::KvOptions {
            session: Some(handoff::decode_session(prefill)?),
            ..Default::default()
        }),
        None => None,
    };

    Ok(pb::GenerateRequest {
        request_id: request_id.to_string(),
        // An LPX engine serves one model and ignores the name.
        model: engine.served_model_name.clone(),
        input: Some(pb::generate_request::Input::TokenIds(pb::TokenIds {
            ids: request.token_ids.as_ref().clone(),
        })),
        sampling: Some(pb::SamplingParams {
            temperature: sampling.temperature.map(f64::from),
            top_p: sampling.top_p.map(f64::from),
            top_k: normalize_top_k(sampling.top_k),
            min_p: sampling.min_p.map(f64::from),
            frequency_penalty: sampling.frequency_penalty.map(f64::from),
            presence_penalty: sampling.presence_penalty.map(f64::from),
            repetition_penalty: sampling.repetition_penalty.map(f64::from),
            seed: normalize_seed(sampling.seed)?,
            num_sequences: Some(1),
        }),
        stopping: Some(pb::StoppingOptions {
            max_tokens: Some(max_tokens(request, engine.context_length)),
            min_tokens: stop.min_tokens,
            conditions: stop_conditions(request),
            ignore_eos: stop.ignore_eos,
            include_stop_in_output: None,
        }),
        // Logprobs are rejected in `validate_request`.
        response: None,
        kv,
        guided: None,
        media: Vec::new(),
        lora_name: request
            .routing
            .as_ref()
            .and_then(|routing| routing.lora_name.clone())
            .unwrap_or_default(),
        extra: None,
    })
}

/// Builds a minimal one-token Generate that releases the staged KV of a decode
/// leg that is rejected or already cancelled.
pub(crate) fn build_handoff_cleanup_request(
    token_ids: &[u32],
    prefill: &PrefillResult,
    request_id: &str,
) -> Result<pb::GenerateRequest, DynamoError> {
    if token_ids.is_empty() {
        return Err(invalid_argument("token_ids must not be empty"));
    }
    let session = handoff::decode_session(prefill)?;
    Ok(pb::GenerateRequest {
        request_id: request_id.to_string(),
        input: Some(pb::generate_request::Input::TokenIds(pb::TokenIds {
            ids: token_ids.to_vec(),
        })),
        sampling: Some(pb::SamplingParams {
            num_sequences: Some(1),
            ..Default::default()
        }),
        stopping: Some(pb::StoppingOptions {
            max_tokens: Some(1),
            ..Default::default()
        }),
        kv: Some(pb::KvOptions {
            session: Some(session),
            ..Default::default()
        }),
        ..Default::default()
    })
}

/// The current span's W3C Trace Context, as `Generate` metadata.
pub(crate) fn trace_headers() -> Vec<(&'static str, String)> {
    let Some(context) = dynamo_runtime::logging::get_distributed_tracing_context() else {
        return Vec::new();
    };
    let mut headers = Vec::with_capacity(2);
    headers.push(("traceparent", context.create_traceparent()));
    if let Some(tracestate) = context.tracestate {
        headers.push(("tracestate", tracestate));
    }
    headers
}

/// -1 and 0 mean "all tokens" in Dynamo, as an unset top_k does in
/// OpenEngine. Any other value is forwarded; the engine rejects a negative
/// one.
fn normalize_top_k(top_k: Option<i32>) -> Option<i32> {
    top_k.filter(|top_k| !matches!(top_k, -1 | 0))
}

fn normalize_seed(seed: Option<i64>) -> Result<Option<u64>, DynamoError> {
    seed.map(|seed| {
        u64::try_from(seed)
            .map_err(|_| invalid_argument(format!("seed must be non-negative; got {seed}")))
    })
    .transpose()
}

/// The output budget, sent with every request because the LPU-only engine
/// rejects a request without a positive one.
///
/// An explicit budget is forwarded unchanged. An absent one is filled the way
/// the hybrid engine fills it, so it sees no difference: the rest of the
/// context window after the prompt, and at least one token. When the prompt
/// already fills the window, its scheduler allows that one boundary token; a
/// longer prompt fails the engine's own admission check.
fn max_tokens(request: &PreprocessedRequest, context_length: u32) -> u32 {
    request.stop_conditions.max_tokens.unwrap_or_else(|| {
        let prompt_tokens = u32::try_from(request.token_ids.len()).unwrap_or(u32::MAX);
        context_length.saturating_sub(prompt_tokens).max(1)
    })
}

/// Stop token IDs only. Stop strings stay with the Dynamo frontend, which
/// matches them on the detokenized stream. The LPU-only engine rejects stop
/// strings and the hybrid engine ignores them, so sending them would only fail
/// LPU-only requests.
fn stop_conditions(request: &PreprocessedRequest) -> Vec<pb::StopCondition> {
    let stop = &request.stop_conditions;
    // Both lists stop generation without returning the token.
    let ids: BTreeSet<u32> = [
        stop.stop_token_ids.as_ref(),
        stop.stop_token_ids_hidden.as_ref(),
    ]
    .into_iter()
    .flatten()
    .flatten()
    .copied()
    .collect();
    ids.into_iter()
        .map(|id| pb::StopCondition {
            condition: Some(pb::stop_condition::Condition::StopTokenId(id)),
        })
        .collect()
}

fn validate_request(request: &PreprocessedRequest) -> Result<(), DynamoError> {
    if request.token_ids.is_empty() {
        return Err(invalid_argument("token_ids must not be empty"));
    }
    if request.prompt_embeds.is_some() {
        return Err(invalid_argument(
            "prompt embeddings are not supported by the LPX sidecar",
        ));
    }
    if request.multi_modal_data.is_some()
        || request.mm_routing_info.is_some()
        || request.encoder_result.is_some()
    {
        return Err(invalid_argument(
            "multimodal requests are not supported by the LPX sidecar",
        ));
    }
    if request.bootstrap_info.is_some() {
        return Err(invalid_argument(
            "a bootstrap handoff is not supported by the LPX sidecar; it consumes vLLM NIXL \
             handoffs only",
        ));
    }
    let sampling = &request.sampling_options;
    if sampling.n.unwrap_or(1) != 1 {
        return Err(invalid_argument("n must be 1"));
    }
    if sampling.best_of.unwrap_or(1) != 1 {
        return Err(invalid_argument("best_of must be 1"));
    }
    if sampling.use_beam_search.unwrap_or(false) {
        return Err(invalid_argument("beam search is not supported"));
    }
    if sampling
        .logit_bias
        .as_ref()
        .is_some_and(|bias| !bias.is_empty())
    {
        return Err(invalid_argument(
            "logit_bias is not supported by the LPX sidecar",
        ));
    }
    if request.output_options.prompt_logprobs.is_some() {
        return Err(invalid_argument(
            "prompt logprobs are not supported by the LPX sidecar",
        ));
    }
    if request.output_options.logprobs.is_some() {
        return Err(invalid_argument(
            "logprobs are not supported by the LPX sidecar",
        ));
    }
    if sampling.guided_decoding.as_ref().is_some_and(|guided| {
        guided.json.is_some()
            || guided.regex.is_some()
            || guided.choice.is_some()
            || guided.grammar.is_some()
            || guided.structural_tag.is_some()
            || guided.backend.is_some()
            || guided.whitespace_pattern.is_some()
    }) {
        return Err(invalid_argument(
            "guided decoding is not supported by the LPX sidecar",
        ));
    }
    if let Some(extra_sampling) = request
        .extra_args
        .as_ref()
        .and_then(|extra| extra.get("sampling_options"))
    {
        // An empty allow-list allows nothing; an empty deny-list bans nothing.
        for key in ["allowed_token_ids", "logprob_token_ids"] {
            if extra_sampling.get(key).is_some() {
                return Err(invalid_argument(format!(
                    "{key} is not supported by the LPX sidecar"
                )));
            }
        }
        if extra_sampling
            .get("bad_words_token_ids")
            .and_then(|value| value.as_array())
            .is_some_and(|ids| !ids.is_empty())
        {
            return Err(invalid_argument(
                "bad_words_token_ids is not supported by the LPX sidecar",
            ));
        }
    }
    let stop = &request.stop_conditions;
    if stop
        .stop_token_ids_visible
        .as_ref()
        .is_some_and(|ids| !ids.is_empty())
    {
        // A visible stop token must end generation and stay in the output;
        // OpenEngine stop conditions cannot express the second half.
        return Err(invalid_argument(
            "visible stop token IDs are not supported by the LPX sidecar",
        ));
    }
    if request.output_options.no_stop_trim == Some(true)
        && (stop
            .stop_token_ids
            .as_ref()
            .is_some_and(|ids| !ids.is_empty())
            || request.output_options.skip_special_tokens == Some(false))
    {
        // The engine strips stop tokens, so the frontend cannot restore their text.
        return Err(invalid_argument(
            "no_stop_trim with client stop token IDs or skip_special_tokens=false is not \
             supported by the LPX sidecar",
        ));
    }
    if sampling.include_stop_str_in_output == Some(true) {
        return Err(invalid_argument(
            "include_stop_str_in_output is not supported by the LPX sidecar",
        ));
    }
    if stop.max_thinking_tokens.is_some() {
        return Err(invalid_argument(
            "max_thinking_tokens is not supported by the LPX sidecar",
        ));
    }
    if let Some(routing) = request.routing.as_ref() {
        if routing
            .cache_namespace
            .as_deref()
            .is_some_and(|namespace| !namespace.is_empty())
        {
            // Dropping it would let requests from different namespaces share
            // prefix-cache entries.
            return Err(invalid_argument(
                "cache namespace isolation is not supported by the LPX sidecar",
            ));
        }
        if routing.priority.is_some_and(|priority| priority != 0) {
            return Err(invalid_argument(
                "request priority is not supported by the LPX sidecar",
            ));
        }
        // The worker registers one data-parallel rank. `prefill_dp_rank`
        // targets the vLLM prefill worker, not this one, so it is ignored.
        if routing.dp_rank.is_some_and(|rank| rank != 0) {
            return Err(invalid_argument(
                "nonzero data-parallel rank targeting is not supported by the LPX sidecar",
            ));
        }
    }
    Ok(())
}

/// Streaming response reducer: the engine streams `token` events and ends
/// with a `finished` event carrying authoritative usage.
pub(crate) struct ResponseState {
    /// Names the engine in protocol and engine errors.
    engine_name: Arc<str>,
    prompt_tokens: u32,
    completion_tokens: u32,
    /// `Some` on a decode leg: the prefill worker's cache-hit count, which is
    /// the only meaningful one there. The decode engine would count the
    /// transferred blocks, a different quantity reported under the same name.
    handoff_cached_tokens: Option<Option<u32>>,
}

impl ResponseState {
    pub(crate) fn new(request: &PreprocessedRequest, engine: &DiscoveredEngine) -> Self {
        Self {
            engine_name: Arc::clone(&engine.engine_name),
            prompt_tokens: request.token_ids.len() as u32,
            completion_tokens: 0,
            handoff_cached_tokens: request.prefill_result.as_ref().map(|prefill| {
                prefill
                    .prompt_tokens_details
                    .as_ref()
                    .and_then(|details| details.cached_tokens)
            }),
        }
    }

    pub(crate) fn protocol_error(&self, message: impl Into<String>) -> DynamoError {
        protocol_error(engine_label(&self.engine_name), message)
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
        let pb::GenerateResponse { event, usage, .. } = response;
        let Some(event) = event else {
            return Err(self.protocol_error("response carried no event"));
        };
        match event {
            pb::generate_response::Event::Token(token) => self.convert_token(token),
            pb::generate_response::Event::Finished(finished) => {
                self.convert_finished(finished, usage).map(Some)
            }
            pb::generate_response::Event::Error(error) => {
                Err(engine_error(error, engine_label(&self.engine_name)))
            }
            pb::generate_response::Event::PrefillReady(_) => Err(self.protocol_error(
                "received a prefill_ready event; this sidecar serves aggregated and decode \
                 requests only",
            )),
            pb::generate_response::Event::Prompt(_) => Err(self.protocol_error(
                "received an unexpected prompt event; prompt logprobs are not requested",
            )),
        }
    }

    fn convert_token(
        &mut self,
        token: pb::TokenOutput,
    ) -> Result<Option<LLMEngineOutput>, DynamoError> {
        self.check_output_index(token.output_index)?;
        let token_ids: Vec<u32> = token.tokens.into_iter().map(|info| info.token_id).collect();
        if token_ids.is_empty() {
            return Ok(None);
        }
        self.completion_tokens = self
            .completion_tokens
            .saturating_add(token_ids.len() as u32);
        Ok(Some(LLMEngineOutput {
            token_ids,
            index: Some(0),
            ..Default::default()
        }))
    }

    fn convert_finished(
        &mut self,
        finished: pb::GenerationFinished,
        reported: Option<pb::Usage>,
    ) -> Result<LLMEngineOutput, DynamoError> {
        self.check_output_index(finished.output_index)?;
        let finish_reason = match pb::FinishReason::try_from(finished.reason).map_err(|_| {
            self.protocol_error(format!("unknown finish reason {}", finished.reason))
        })? {
            pb::FinishReason::Stop => FinishReason::Stop,
            pb::FinishReason::Length => FinishReason::Length,
            pb::FinishReason::Cancelled => FinishReason::Cancelled,
            pb::FinishReason::Unspecified => {
                return Err(
                    self.protocol_error("terminal response has an unspecified finish reason")
                );
            }
        };
        let mut cached_tokens = self.handoff_cached_tokens.flatten();
        // The final usage is authoritative over the streamed counts.
        if let Some(reported) = reported {
            if reported.prompt_tokens != 0 {
                self.prompt_tokens = reported.prompt_tokens;
            }
            if reported.completion_tokens != 0 {
                self.completion_tokens = reported.completion_tokens;
            }
            if self.handoff_cached_tokens.is_none() {
                cached_tokens = reported.cached_prompt_tokens;
            }
        }
        Ok(LLMEngineOutput {
            index: Some(0),
            finish_reason: Some(finish_reason),
            stop_reason: finished
                .stop_match
                .and_then(|stop_match| stop_match.r#match)
                .map(|matched| match matched {
                    pb::stop_match::Match::StopTokenId(id)
                    | pb::stop_match::Match::EosTokenId(id) => StopReason::Int(i64::from(id)),
                    pb::stop_match::Match::StopText(text) => StopReason::String(text),
                }),
            completion_usage: Some(CompletionUsage {
                prompt_tokens_details: cached_tokens.map(|cached| PromptTokensDetails {
                    cached_tokens: Some(cached.min(self.prompt_tokens)),
                    ..Default::default()
                }),
                ..usage(self.prompt_tokens, self.completion_tokens)
            }),
            ..Default::default()
        })
    }

    /// The engine streams one sequence and must name it, as the contract
    /// requires.
    fn check_output_index(&self, index: Option<u32>) -> Result<(), DynamoError> {
        match index {
            Some(0) => Ok(()),
            Some(index) => {
                Err(self.protocol_error(format!("received unsupported output index {index}")))
            }
            None => Err(self.protocol_error(
                "response carried no output index; the OpenEngine contract requires one",
            )),
        }
    }
}

/// How a `Generate` status received before the response stream opened
/// reaches the frontend.
pub(crate) enum DispatchFailure {
    /// Returned as an error from `generate`. The frontend may migrate it,
    /// which re-runs prefill on another pair of workers.
    Retryable(DynamoError),
    /// Delivered as the stream's first item, which the frontend never
    /// migrates: another worker would reject the request the same way.
    Rejected(DynamoError),
}

/// Only UNAVAILABLE (the engine starting, paused, full or draining, or the
/// transport down) can succeed elsewhere. Every other status rejects this
/// request itself.
pub(crate) fn dispatch_failure(status: tonic::Status) -> DispatchFailure {
    let unavailable = status.code() == tonic::Code::Unavailable;
    let error = status_to_dynamo("Generate", status);
    if unavailable {
        DispatchFailure::Retryable(error)
    } else {
        DispatchFailure::Rejected(error)
    }
}

/// A status that ended an open response stream. An engine ends open streams
/// with UNAVAILABLE when it restarts, drains or shuts down; the request can
/// succeed elsewhere, so it is reported as a migratable engine shutdown.
pub(crate) fn stream_status(status: tonic::Status) -> DynamoError {
    if status.code() == tonic::Code::Unavailable {
        engine_shutdown(format!(
            "Generate: {} ({:?})",
            status.message(),
            status.code()
        ))
    } else {
        status_to_dynamo("Generate", status)
    }
}

/// An in-band failure of a request the engine had accepted. Only the hybrid
/// engine sends these; the LPU-only engine reports every failure as a gRPC
/// status.
fn engine_error(error: pb::EngineError, engine: &str) -> DynamoError {
    let code = pb::ErrorCode::try_from(error.code).unwrap_or(pb::ErrorCode::Unspecified);
    let message = if error.message.trim().is_empty() {
        format!("{engine} reported engine error {code:?}")
    } else {
        format!("{engine} engine error: {}", error.message)
    };
    if error.retryable {
        return engine_shutdown(message);
    }
    match code {
        pb::ErrorCode::InvalidArgument
        | pb::ErrorCode::UnsupportedFeature
        | pb::ErrorCode::DuplicateRequest
        | pb::ErrorCode::RoleMismatch => invalid_argument(message),
        // A decode engine reports a failed KV pull as INTERNAL with a "KV
        // transfer failed: …" message, which passes through.
        _ => unknown(message),
    }
}

fn unknown(message: String) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(BackendError::Unknown))
        .message(message)
        .build()
}
