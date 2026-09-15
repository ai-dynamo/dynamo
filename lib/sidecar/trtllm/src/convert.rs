// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conversion between Dynamo's `PreprocessedRequest` / `LLMEngineOutput` and the
//! TensorRT-LLM OpenEngine (`openengine.v1`) protobuf messages.
//!
//! Scope: aggregated and disaggregated (prefill/decode) generation. Multimodal,
//! LoRA, beam search, and `n > 1` are rejected before dispatch — the sidecar
//! streams a single sequence.
//!
//! Disaggregation is driven by [`DisaggregationMode`]: a prefill worker marks
//! its request `context_only` and returns the `PrefillReady` handoff as its
//! terminal chunk, and a decode worker replays that handoff in `kv.session`.
//! See [`crate::disagg`] for the codec.

use std::collections::BTreeSet;

use dynamo_backend_common::{
    CompletionUsage, DisaggregationMode, DynamoError, FinishReason, LLMEngineOutput,
    PreprocessedRequest, PromptTokensDetails, StopReason, TopLogprob, usage,
};

use crate::client::{self, ModelLimits};
use crate::disagg;
use crate::proto as pb;

/// A chunk's delta token IDs, the selected-token logprob sequence, and the
/// per-token top-k alternatives, all aligned with each other.
type MappedTokens = (Vec<u32>, Option<Vec<f64>>, Option<Vec<Vec<TopLogprob>>>);

pub(crate) fn build_generate_request(
    request: &PreprocessedRequest,
    request_id: &str,
    model: &str,
    limits: Option<ModelLimits>,
    mode: DisaggregationMode,
) -> Result<pb::GenerateRequest, DynamoError> {
    validate_request(request, mode)?;

    let sampling = &request.sampling_options;
    let stop = &request.stop_conditions;
    let output = &request.output_options;

    // A prefill worker only needs the context phase; TensorRT-LLM still requires
    // a positive budget, and one token is what the context phase produces.
    let max_tokens = if mode.is_prefill() {
        1
    } else {
        max_tokens(request, limits)?
    };
    // The decode worker replays the prefill worker's session; the prefill worker
    // marks its request `context_only` through `extra`.
    let (kv, extra) = match mode {
        DisaggregationMode::Prefill => (None, Some(disagg::context_only_extra())),
        // A handoff is the normal case, but conditional disaggregation
        // dispatches straight to a decode worker with none, expecting it to run
        // the context phase itself. Requiring one would fail every bypassed
        // request.
        DisaggregationMode::Decode => match request.prefill_result.as_ref() {
            Some(handoff) => (
                Some(pb::KvOptions {
                    session: Some(disagg::session_from_json(&handoff.disaggregated_params)?),
                    ..Default::default()
                }),
                None,
            ),
            None => (None, None),
        },
        DisaggregationMode::Aggregated | DisaggregationMode::Encode => (None, None),
    };

    Ok(pb::GenerateRequest {
        request_id: request_id.to_string(),
        // The OpenEngine server rejects an empty model; any non-empty name is
        // served by the loaded model (single-model server).
        model: model.to_string(),
        input: Some(pb::generate_request::Input::TokenIds(pb::TokenIds {
            ids: request.token_ids.as_ref().clone(),
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
            num_sequences: Some(1),
        }),
        stopping: Some(pb::StoppingOptions {
            max_tokens: Some(max_tokens),
            // A prefill worker stops after the context phase; a minimum would
            // force it to decode.
            min_tokens: if mode.is_prefill() {
                None
            } else {
                stop.min_tokens
            },
            conditions: stop_conditions(request),
            ignore_eos: stop.ignore_eos,
            // `include_stop_in_output` retains matched stop *strings*; the
            // request-level flag is rejected in `validate_request`, so leave it
            // unset (the server strips them).
            include_stop_in_output: None,
        }),
        response: Some(pb::ResponseOptions {
            // A prefill worker streams no tokens to the client, but it must
            // still be asked for logprobs: the context phase produces the first
            // generated token, and its logprob only reaches the decode worker
            // if the context request computed it (the server carries it as
            // `first_gen_log_probs` in the handoff). Suppressing it here makes
            // the decode worker report that first token as missing its logprob.
            return_output_logprobs: Some(output.logprobs.is_some()),
            output_candidates: output.logprobs.map(output_candidates),
            // Prompt logprobs are rejected in `validate_request` (no
            // `LLMEngineOutput` field to surface them).
            return_prompt_logprobs: None,
            prompt_candidates: None,
            prompt_logprob_start: None,
        }),
        guided: guided_decoding(request)?,
        // Text generation only: no multimodal media or LoRA selection.
        media: Vec::new(),
        lora_name: String::new(),
        kv,
        extra,
    })
}

// Temporary workaround: the OpenEngine contract makes `stopping.max_tokens`
// optional, but an omitted value falls through to TensorRT-LLM's small
// `SamplingParams` default rather than filling the context. The Dynamo frontend
// forwards an omitted `max_tokens` as `None` expecting the backend to default,
// so we mirror the in-process backend's text-only default,
// `max(1, context_length - prompt_len)` (components/src/dynamo/trtllm
// `_default_max_tokens`); the sidecar rejects multimodal before dispatch, so
// `token_ids.len()` is the true prompt length. `context_length` is resolved in
// `engine::start`, where `--context-length` wins over the `GetModelInfo` report.
// Only `/v1/chat/completions` and `/v1/responses` reach this fallback: they set
// `PRESERVE_OMITTED_MAX_TOKENS_CONTEXT_KEY`, which stops the frontend supplying
// its own default (`preprocessor::omitted_max_tokens_default`). `/v1/completions`
// is already defaulted by the frontend and never lands here.
//
// Remove when https://github.com/NVIDIA/TensorRT-LLM/issues/16549 lands (gRPC
// `max_tokens` made optional): drop this fallback and forward an omitted
// `max_tokens` as unset. Keep `--context-length` and the plumbing in
// `engine.rs` — they also feed `LlmRegistration.context_length`, which the
// frontend registers as the served context window and which this fallback does
// not govern.
fn max_tokens(
    request: &PreprocessedRequest,
    limits: Option<ModelLimits>,
) -> Result<u32, DynamoError> {
    if let Some(max_tokens) = request.stop_conditions.max_tokens {
        return Ok(max_tokens);
    }
    let context_length = limits
        .and_then(|limits| limits.context_length)
        .ok_or_else(|| {
            client::invalid_argument(
                "TensorRT-LLM requires max_tokens, and no model context length is known to \
             derive a default from; specify max_tokens explicitly or start the sidecar \
             with --context-length",
            )
        })?;
    let limits = limits.unwrap_or_default();
    let prompt_len = request.token_ids.len() as u32;
    let remaining = context_length.saturating_sub(prompt_len).max(1);
    // The window is input + output, so on a short prompt the remainder can
    // exceed what the engine will actually generate and it would reject the
    // request we derived.
    Ok(match limits.max_output_tokens {
        Some(cap) => remaining.min(cap),
        None => remaining,
    })
}

fn normalize_top_k(top_k: Option<i32>) -> Result<Option<i32>, DynamoError> {
    // Dynamo uses -1/0 (or absence) for "consider all tokens"; the OpenEngine
    // server treats an unset top_k the same way. Forward only a positive cap;
    // reject other negatives rather than silently widening them to "all tokens".
    match top_k {
        None | Some(-1) | Some(0) => Ok(None),
        Some(value) if value > 0 => Ok(Some(value)),
        Some(value) => Err(client::invalid_argument(format!(
            "top_k must be -1, 0, or positive; got {value}"
        ))),
    }
}

fn normalize_seed(seed: Option<i64>) -> Result<Option<u64>, DynamoError> {
    // OpenEngine's seed is `uint64`; reject a negative seed rather than silently
    // dropping it and losing reproducibility.
    seed.map(|seed| {
        u64::try_from(seed)
            .map_err(|_| client::invalid_argument(format!("seed must be non-negative; got {seed}")))
    })
    .transpose()
}

fn output_candidates(count: u32) -> pb::CandidateTokenSelection {
    // TRT-LLM computes the selected-token logprob only when at least one
    // candidate is requested, so floor the wire value at 1: `logprobs=0`
    // (selected token, no alternatives) still yields the chosen-token logprob.
    // The original count is preserved in `ResponseState` to decide whether to
    // surface alternatives.
    pb::CandidateTokenSelection {
        selection: Some(pb::candidate_token_selection::Selection::TopN(count.max(1))),
    }
}

fn stop_conditions(request: &PreprocessedRequest) -> Vec<pb::StopCondition> {
    let stop = &request.stop_conditions;
    let mut conditions = Vec::new();
    if let Some(stop_strings) = stop.stop.as_ref() {
        for text in stop_strings {
            conditions.push(pb::StopCondition {
                condition: Some(pb::stop_condition::Condition::StopText(text.clone())),
            });
        }
    }
    for id in stop_token_ids(request) {
        conditions.push(pb::StopCondition {
            condition: Some(pb::stop_condition::Condition::StopTokenId(id)),
        });
    }
    conditions
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
) -> Result<Option<pb::GuidedDecoding>, DynamoError> {
    let Some(guided) = request.sampling_options.guided_decoding.as_ref() else {
        return Ok(None);
    };
    if guided.backend.is_some() || guided.whitespace_pattern.is_some() {
        return Err(client::invalid_argument(
            "guided decoding backend and whitespace_pattern are not supported by the TensorRT-LLM OpenEngine server",
        ));
    }

    use pb::guided_decoding::Guide;
    let mut guides = Vec::new();
    if let Some(json) = &guided.json {
        guides.push(Guide::JsonSchema(json_guide(json)));
    }
    if let Some(regex) = &guided.regex {
        guides.push(Guide::Regex(regex.clone()));
    }
    if let Some(grammar) = &guided.grammar {
        guides.push(Guide::EbnfGrammar(grammar.clone()));
    }
    if let Some(tag) = &guided.structural_tag {
        guides.push(Guide::StructuralTag(json_guide(tag)));
    }
    if guided.choice.is_some() {
        // `choice` is in the OpenEngine contract but the TensorRT-LLM server
        // rejects it; fail fast with a clear message.
        return Err(client::invalid_argument(
            "guided decoding `choice` is not supported by the TensorRT-LLM OpenEngine server",
        ));
    }
    if guides.len() > 1 {
        return Err(client::invalid_argument(
            "only one guided decoding constraint may be set",
        ));
    }
    Ok(guides.pop().map(|guide| pb::GuidedDecoding {
        guide: Some(guide),
        backend: String::new(),
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

fn validate_request(
    request: &PreprocessedRequest,
    mode: DisaggregationMode,
) -> Result<(), DynamoError> {
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
    // Only a decode worker consumes a prefill handoff. Seeing one anywhere else
    // means the frontend routed a decode request to the wrong worker role.
    if request.prefill_result.is_some() && !mode.is_decode() {
        return Err(client::invalid_argument(format!(
            "received a prefill_result on a worker running in '{mode}' mode; \
             disaggregated decode requests must be routed to a decode worker"
        )));
    }
    if request.output_options.prompt_logprobs.is_some() {
        // TRT-LLM would compute these, but the terminal carries them with no
        // `LLMEngineOutput` field to surface — reject rather than pay for and
        // drop them.
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
        // OpenEngine's stop conditions and single `include_stop_in_output` (stop
        // *strings*) cannot honor that per token, so reject rather than silently
        // drop or mis-retain them.
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
        .and_then(|routing| routing.cache_namespace.as_deref())
        .is_some_and(|namespace| !namespace.is_empty())
    {
        // `cache_namespace` is the request-scoped KV-cache isolation contract.
        // The OpenEngine request carries `kv.cache_salt`, but the sidecar does
        // not yet map the namespace onto it, so honoring the request would let
        // requests from different namespaces share prefix-cache entries. Reject
        // until the mapping is implemented.
        return Err(client::invalid_argument(
            "cache namespace isolation is not supported by the TensorRT-LLM sidecar",
        ));
    }
    if request
        .routing
        .as_ref()
        .and_then(|routing| routing.priority)
        .is_some_and(|priority| priority != 0)
    {
        // OpenEngine carries priority as request metadata, which the TensorRT-LLM
        // server rejects; a nonzero priority would otherwise be silently dropped
        // and change queue ordering. Reject until the server honors it.
        return Err(client::invalid_argument(
            "request priority is not supported by the TensorRT-LLM sidecar",
        ));
    }
    if request.stop_conditions.max_thinking_tokens.is_some() {
        // A reasoning-token budget the sidecar can neither forward nor enforce.
        return Err(client::invalid_argument(
            "max_thinking_tokens is not supported by the TensorRT-LLM sidecar",
        ));
    }
    let sampling = &request.sampling_options;
    if sampling.include_stop_str_in_output == Some(true) {
        // Retaining stop *strings* maps to `include_stop_in_output`, but the
        // sidecar rejects visible stop token IDs above; honoring string retention
        // alone would be inconsistent, so reject until both are supported
        // together.
        return Err(client::invalid_argument(
            "include_stop_str_in_output is not supported by the TensorRT-LLM sidecar",
        ));
    }
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

/// Streaming response reducer. The OpenEngine server streams `token` events
/// followed by a terminal `finished` event carrying authoritative usage; this
/// maps each onto an `LLMEngineOutput`.
pub(crate) struct ResponseState {
    prompt_tokens: u32,
    completion_tokens: u32,
    output_logprobs: Option<u32>,
    phase: Phase,
}

/// What this worker does with the tokens the engine streams back. Each arm owns
/// the state that only exists in that mode, so a prefill worker cannot reach a
/// decode worker's cache accounting and vice versa.
enum Phase {
    /// Aggregated, or a decode leg: tokens stream straight to the client.
    Stream {
        /// The handoff this leg replays, if any. Cache hits are measured during
        /// the context phase, so on a decode leg the count arrives here rather
        /// than from the local engine -- which counts the blocks transferred
        /// into it and reports a different quantity under the same name.
        handoff: Option<Handoff>,
    },
    /// A prefill leg: `PrefillReady` is the terminal event, and context tokens
    /// are held back because the decode leg replays them. They only reach the
    /// client through a context request that ends without a handoff, which has
    /// no decode leg to do the replaying.
    Prefill { held: Vec<u32> },
}

/// What a decode leg knows about the context phase that preceded it.
struct Handoff {
    /// The context phase's prefix-cache hit count, when it reported one. Absent
    /// means the context phase reported none -- not that the decode engine's
    /// own count should stand in for it.
    cached_tokens: Option<u32>,
}

impl ResponseState {
    pub(crate) fn new(request: &PreprocessedRequest, mode: DisaggregationMode) -> Self {
        Self {
            prompt_tokens: request.token_ids.len() as u32,
            completion_tokens: 0,
            output_logprobs: request.output_options.logprobs,
            phase: if mode.is_prefill() {
                Phase::Prefill { held: Vec::new() }
            } else {
                Phase::Stream {
                    handoff: request.prefill_result.as_ref().map(|prefill| Handoff {
                        cached_tokens: prefill
                            .prompt_tokens_details
                            .as_ref()
                            .and_then(|details| details.cached_tokens),
                    }),
                }
            },
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
        let pb::GenerateResponse { event, usage, .. } = response;
        // A response with no event is protocol drift, not an empty delta.
        let Some(event) = event else {
            return Err(client::protocol_error("response carried no event"));
        };
        match event {
            pb::generate_response::Event::Token(token) => self.convert_token(token),
            pb::generate_response::Event::Finished(finished) => {
                self.convert_finished(finished, usage).map(Some)
            }
            pb::generate_response::Event::PrefillReady(prefill) => {
                self.convert_prefill_ready(prefill, usage).map(Some)
            }
            pb::generate_response::Event::Error(error) => Err(engine_error(error)),
            // Prompt logprobs are never requested, so a prompt event is drift.
            pb::generate_response::Event::Prompt(_) => Err(client::protocol_error(
                "received an unexpected prompt event; prompt logprobs are not requested",
            )),
        }
    }

    /// Terminal chunk for a prefill worker.
    ///
    /// `PrefillReady` *is* the terminal event for a `context_only` request — the
    /// server suppresses `finished` because the engine reports the sequence as
    /// unfinished — so this synthesizes the terminal chunk, carrying the handoff
    /// the decode worker will replay.
    fn convert_prefill_ready(
        &mut self,
        prefill: pb::PrefillReady,
        reported: Option<pb::Usage>,
    ) -> Result<LLMEngineOutput, DynamoError> {
        let Phase::Prefill { .. } = self.phase else {
            return Err(client::protocol_error(
                "received a prefill_ready event on a worker that is not running in prefill mode",
            ));
        };
        let session = prefill
            .kv_session
            .ok_or_else(|| client::protocol_error("prefill_ready event carried no kv_session"))?;
        Ok(LLMEngineOutput {
            // The client sees no tokens from prefill; the decode worker emits
            // the full completion.
            token_ids: Vec::new(),
            index: Some(0),
            // `Length` is what the frontend's prefill router requires to chain
            // into decode: it returns any other terminal reason straight to the
            // caller as an already-complete request (see
            // `kv_router::prefill_router::admission`). It is also accurate —
            // the context request is capped at one token.
            finish_reason: Some(FinishReason::Length),
            // PrefillReady is the final response for a context request, so it
            // carries the engine's usage -- including cached_prompt_tokens,
            // which the frontend forwards to the decode leg and which cannot be
            // reconstructed client-side.
            completion_usage: Some(prefill_usage(reported, self.prompt_tokens)),
            disaggregated_params: Some(disagg::session_to_json(session)?),
            ..Default::default()
        })
    }

    fn convert_token(
        &mut self,
        token: pb::TokenOutput,
    ) -> Result<Option<LLMEngineOutput>, DynamoError> {
        check_output_index(token.output_index)?;
        if let Phase::Prefill { held } = &mut self.phase {
            held.extend(token.tokens.iter().map(|info| info.token_id));
            self.completion_tokens = held.len() as u32;
            return Ok(None);
        }
        let (token_ids, log_probs, top_logprobs) = self.map_tokens(token.tokens)?;
        self.completion_tokens = self
            .completion_tokens
            .saturating_add(token_ids.len() as u32);
        if token_ids.is_empty() {
            // A text-only delta (e.g. stop-string holdback): the sidecar streams
            // token IDs and lets the frontend detokenize, so there is nothing to
            // surface yet.
            return Ok(None);
        }
        Ok(Some(LLMEngineOutput {
            token_ids,
            log_probs,
            top_logprobs,
            index: Some(0),
            ..Default::default()
        }))
    }

    fn convert_finished(
        &mut self,
        finished: pb::GenerationFinished,
        reported: Option<pb::Usage>,
    ) -> Result<LLMEngineOutput, DynamoError> {
        check_output_index(finished.output_index)?;
        // A decode engine counts the blocks transferred into it as cache hits,
        // a different quantity from the context phase's prefix-cache hit that
        // would read as a ~100% hit rate on every disaggregated request. Once
        // this leg took a handoff the context phase is the only source, even
        // when it reported no hits. A leg without one ran its own context
        // phase, so its engine is the source.
        let (mut cached_tokens, engine_measured_the_cache) = match &self.phase {
            Phase::Stream {
                handoff: Some(handoff),
            } => (handoff.cached_tokens, false),
            Phase::Stream { handoff: None } | Phase::Prefill { .. } => (None, true),
        };
        // The final response carries authoritative usage; prefer it over the
        // counts accumulated while streaming.
        if let Some(reported) = reported {
            if reported.prompt_tokens != 0 {
                self.prompt_tokens = reported.prompt_tokens;
            }
            if reported.completion_tokens != 0 {
                self.completion_tokens = reported.completion_tokens;
            }
            if engine_measured_the_cache {
                cached_tokens = reported.cached_prompt_tokens;
            }
        }

        let finish_reason = match pb::FinishReason::try_from(finished.reason).map_err(|_| {
            client::protocol_error(format!("unknown finish reason {}", finished.reason))
        })? {
            pb::FinishReason::Stop => FinishReason::Stop,
            pb::FinishReason::Length => FinishReason::Length,
            pb::FinishReason::Cancelled => FinishReason::Cancelled,
            // Fail closed on an unspecified reason rather than reporting a clean
            // stop for a version-skewed or malformed terminal.
            pb::FinishReason::Unspecified => {
                return Err(client::protocol_error(
                    "terminal response has an unspecified finish reason",
                ));
            }
        };

        // A context request only reaches a `finished` event when it produced no
        // handoff: the server suppresses the terminal once it has sent
        // `PrefillReady`. `Length` there means the context phase burned its
        // one-token budget without ever transmitting KV -- disaggregation is
        // not working on this engine -- so returning that single token would
        // silently truncate the completion. Any other reason is a genuine stop
        // during the context phase, and with no decode leg to run, it is the
        // whole answer.
        let held = match &mut self.phase {
            Phase::Prefill { held } if finish_reason == FinishReason::Length => {
                return Err(client::protocol_error(format!(
                    "context request ended at its token budget without a kv_session handoff \
                     after {} token(s); the engine ran the context phase but never transmitted \
                     its KV, which usually means disaggregation is not configured on it (check \
                     cache_transceiver_config)",
                    held.len()
                )));
            }
            Phase::Prefill { held } => std::mem::take(held),
            Phase::Stream { .. } => Vec::new(),
        };

        let mut terminal = LLMEngineOutput {
            token_ids: held,
            index: Some(0),
            finish_reason: Some(finish_reason),
            completion_usage: Some(CompletionUsage {
                prompt_tokens_details: cached_prompt_tokens(cached_tokens, self.prompt_tokens),
                ..usage(self.prompt_tokens, self.completion_tokens)
            }),
            ..Default::default()
        };
        terminal.stop_reason = finished
            .stop_match
            .and_then(|stop_match| stop_match.r#match)
            .map(|matched| match matched {
                pb::stop_match::Match::StopTokenId(id) | pb::stop_match::Match::EosTokenId(id) => {
                    StopReason::Int(i64::from(id))
                }
                pb::stop_match::Match::StopText(text) => StopReason::String(text),
            });
        Ok(terminal)
    }

    /// One pass over the owned deltas. Token IDs, selected-token logprobs, and
    /// the candidate lists all come from the same `TokenInfo`s, and this runs on
    /// every token of every request.
    fn map_tokens(&self, tokens: Vec<pb::TokenInfo>) -> Result<MappedTokens, DynamoError> {
        let mut token_ids = Vec::with_capacity(tokens.len());
        let Some(count) = self.output_logprobs else {
            token_ids.extend(tokens.into_iter().map(|info| info.token_id));
            return Ok((token_ids, None, None));
        };
        let mut log_probs = Vec::with_capacity(tokens.len());
        // `logprobs=0` keeps the selected-token logprob but omits the top
        // alternatives, matching the vLLM sidecar contract.
        let wants_candidates = count != 0;
        let mut top_logprobs = Vec::with_capacity(if wants_candidates { tokens.len() } else { 0 });
        for info in tokens {
            let token_id = info.token_id;
            let rank = info.rank.unwrap_or(0);
            // Logprobs were requested, so every delta token must carry its
            // selected-token logprob; a missing value is protocol drift.
            let logprob = info.logprob.ok_or_else(|| {
                client::protocol_error(format!("token {token_id} is missing its logprob"))
            })?;
            if wants_candidates {
                top_logprobs.push(if info.candidates.is_empty() {
                    vec![TopLogprob {
                        rank,
                        token_id,
                        token: None,
                        logprob,
                        bytes: None,
                    }]
                } else {
                    info.candidates
                        .into_iter()
                        .map(|candidate| TopLogprob {
                            rank: candidate.rank.unwrap_or(0),
                            token_id: candidate.token_id,
                            token: None,
                            logprob: candidate.logprob,
                            bytes: None,
                        })
                        .collect()
                });
            }
            token_ids.push(token_id);
            log_probs.push(logprob);
        }
        Ok((
            token_ids,
            Some(log_probs),
            wants_candidates.then_some(top_logprobs),
        ))
    }
}

/// The OpenEngine contract requires an explicit index even for output zero, and
/// the sidecar streams a single sequence: anything else is drift.
fn check_output_index(index: Option<u32>) -> Result<(), DynamoError> {
    match index {
        Some(0) => Ok(()),
        Some(index) => Err(client::protocol_error(format!(
            "received unsupported output index {index}"
        ))),
        None => Err(client::protocol_error(
            "response carried no output index; the OpenEngine contract requires one",
        )),
    }
}

/// Usage for the prefill terminal. `PrefillReady` *is* the final response for a
/// context request, so it carries the engine's authoritative counts; the
/// client-side prompt length is only a fallback for a server that omits them.
fn prefill_usage(reported: Option<pb::Usage>, prompt_tokens: u32) -> CompletionUsage {
    let Some(reported) = reported else {
        return usage(prompt_tokens, 0);
    };
    let prompt_tokens = if reported.prompt_tokens != 0 {
        reported.prompt_tokens
    } else {
        prompt_tokens
    };
    CompletionUsage {
        prompt_tokens_details: cached_prompt_tokens(reported.cached_prompt_tokens, prompt_tokens),
        ..usage(prompt_tokens, 0)
    }
}

/// A server that counts cache hits against an expanded prompt would otherwise
/// report more cached tokens than the prompt the client actually sent.
fn cached_prompt_tokens(cached: Option<u32>, prompt_tokens: u32) -> Option<PromptTokensDetails> {
    cached.map(|cached_tokens| PromptTokensDetails {
        audio_tokens: None,
        cached_tokens: Some(cached_tokens.min(prompt_tokens)),
    })
}

pub(crate) fn engine_error(error: pb::EngineError) -> DynamoError {
    let code = pb::ErrorCode::try_from(error.code).unwrap_or(pb::ErrorCode::Unspecified);
    let message = if error.message.trim().is_empty() {
        format!("TensorRT-LLM reported engine error {code:?}")
    } else {
        format!("TensorRT-LLM engine error: {}", error.message)
    };
    match code {
        pb::ErrorCode::InvalidArgument | pb::ErrorCode::UnsupportedFeature => {
            client::invalid_argument(message)
        }
        // The router sheds and migrates on an overload; flattening it to a
        // generic engine error costs that and surfaces an opaque 500 instead.
        pb::ErrorCode::Overloaded => client::worker_overloaded(message),
        pb::ErrorCode::Cancelled => client::cancelled(message),
        _ => client::engine_error(message),
    }
}
