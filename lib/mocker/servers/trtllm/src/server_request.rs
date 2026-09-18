// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;

use dynamo_mocker::common::protocols::DirectRequest;
use dynamo_mocker::live::{deterministic_token_id, stable_request_uuid};
use dynamo_trtllm_sidecar::proto as pb;
use prost_types::{Struct, value::Kind};
use tonic::Status;
use uuid::Uuid;

use super::handoff;
use super::{BoxedStatusResult, MockerServerConfig, ServerMode};

/// Spelled out rather than imported from the sidecar: this pair is the one part
/// of the contract the real TensorRT-LLM servicer binds by literal string, so
/// sharing the constant with the code under test would let a rename pass the
/// whole integration suite and fail against a real engine.
const REQUEST_TYPE_KEY: &str = "request_type";
const CONTEXT_ONLY: &str = "context_only";

pub(super) const DEFAULT_MAX_NEW_TOKENS: u32 = 20;
// Bound the request-owned synthetic token plan independently of LiveEngine's
// fixed per-request delivery buffer.
pub(super) const MAX_NEW_TOKENS: u32 = 32_768;
pub(super) const MAX_CANDIDATES: usize = 20;

#[derive(Debug)]
pub(super) struct PreparedRequest {
    pub(super) uuid: Uuid,
    pub(super) request_id: String,
    pub(super) session_id: String,
    /// The session the client presented. Only a decode request has one it
    /// could name in an `Abort`; a prefill worker's session reaches the client
    /// inside the terminal event, by which point the request is already gone.
    pub(super) client_session_id: Option<String>,
    seed: u64,
    /// The context phase's first token, replayed as this leg's first output —
    /// what a real generation worker does with the handoff. The sidecar drops
    /// the prefill leg's tokens, so the token is delivered to the client once.
    replayed_first_token: Option<u32>,
    /// The replayed token's logprob, when the context phase computed one. A
    /// decode leg asked for logprobs after a context leg that was not reports
    /// its first token without one, exactly as a real engine does.
    replayed_first_logprob: Option<f64>,
    prompt_tokens: Vec<u32>,
    pub(super) max_output_tokens: usize,
    /// Token IDs that end the request with `STOP` instead of `LENGTH`. Stop
    /// *strings* are accepted and never match: this server has no tokenizer, so
    /// it has no text to match them against.
    stop_token_ids: BTreeSet<u32>,
    min_output_tokens: usize,
    return_output_logprobs: bool,
    return_prompt_logprobs: bool,
    output_candidates: Option<pb::CandidateTokenSelection>,
    prompt_candidates: Option<pb::CandidateTokenSelection>,
}

impl PreparedRequest {
    pub(super) fn new(
        request: pb::GenerateRequest,
        config: &MockerServerConfig,
    ) -> BoxedStatusResult<Self> {
        // Only emptiness is rejected. A real TensorRT-LLM server loads one
        // model and serves it under whatever non-empty name the request names
        // -- verified against 1.3.0rc26, which answers `Generate` and
        // `GetModelInfo` for an unrelated name with the loaded model's info.
        // Rejecting a mismatch here would fail requests the real engine serves,
        // which is the one thing this mocker must never do.
        if request.model.is_empty() {
            return Err(Box::new(Status::invalid_argument(
                "model must be non-empty",
            )));
        }
        reject_unsupported(&request)?;

        let prompt_tokens = match request.input {
            Some(pb::generate_request::Input::TokenIds(ids)) => ids.ids,
            Some(pb::generate_request::Input::Prompt(_)) => {
                return Err(Box::new(Status::unimplemented(
                    "the Mocker server has no tokenizer; send token_ids instead of prompt",
                )));
            }
            None => {
                return Err(Box::new(Status::invalid_argument(
                    "request carries no input",
                )));
            }
        };
        if prompt_tokens.is_empty() {
            return Err(Box::new(Status::invalid_argument(
                "token_ids must not be empty",
            )));
        }

        let kv = request.kv.unwrap_or_default();
        validate_role(
            is_context_only(request.extra.as_ref()),
            kv.session.as_ref(),
            config.mode,
        )?;
        if let Some(session) = kv.session.as_ref() {
            handoff::validate_session(session)?;
        }

        let stopping = request.stopping.unwrap_or_default();
        let max_output_tokens = max_output_tokens(&stopping, config.mode)?;
        if prompt_tokens.len().saturating_add(max_output_tokens) > config.context_length as usize {
            return Err(Box::new(Status::invalid_argument(format!(
                "prompt ({}) plus max_tokens ({}) exceeds the context length of {}",
                prompt_tokens.len(),
                max_output_tokens,
                config.context_length
            ))));
        }

        let request_id = if request.request_id.is_empty() {
            Uuid::new_v4().to_string()
        } else {
            request.request_id
        };
        let uuid = stable_request_uuid(config.seed, &request_id);
        let session_id = handoff::session_id(uuid);
        let response = request.response.unwrap_or_default();

        Ok(Self {
            uuid,
            client_session_id: kv
                .session
                .as_ref()
                .map(|session| session.session_id.clone()),
            session_id,
            seed: config.seed,
            replayed_first_token: kv.session.as_ref().and_then(handoff::first_gen_token),
            replayed_first_logprob: kv.session.as_ref().and_then(handoff::first_gen_logprob),
            request_id,
            prompt_tokens,
            max_output_tokens,
            stop_token_ids: stopping
                .conditions
                .iter()
                .filter_map(|condition| match condition.condition {
                    Some(pb::stop_condition::Condition::StopTokenId(id)) => Some(id),
                    _ => None,
                })
                .collect(),
            min_output_tokens: stopping.min_tokens.unwrap_or(0) as usize,
            return_output_logprobs: response.return_output_logprobs == Some(true),
            return_prompt_logprobs: response.return_prompt_logprobs == Some(true),
            output_candidates: response.output_candidates,
            prompt_candidates: response.prompt_candidates,
        })
    }

    pub(super) fn direct_request(&self) -> DirectRequest {
        DirectRequest {
            tokens: self.prompt_tokens.clone(),
            max_output_tokens: self.max_output_tokens,
            uuid: Some(self.uuid),
            dp_rank: super::DP_RANK,
            output_token_ids: Some(
                (0..self.max_output_tokens)
                    .map(|position| self.output_token(position))
                    .collect(),
            ),
            ..Default::default()
        }
    }

    /// Whether this token ends the request. A real engine keeps generating
    /// until `min_tokens`, so a stop condition before that does not fire.
    pub(super) fn is_stop_token(&self, token_id: u32, generated: usize) -> bool {
        generated >= self.min_output_tokens && self.stop_token_ids.contains(&token_id)
    }

    /// The terminal a stop condition produces. The matched token is reported
    /// but not streamed, which is what the engine does unless the client asks
    /// for it back with `include_stop_in_output`.
    pub(super) fn stopped(
        &self,
        token_id: u32,
        generated: usize,
        cached_tokens: Option<usize>,
    ) -> pb::GenerateResponse {
        let mut response = self.finished(pb::FinishReason::Stop, generated, cached_tokens);
        if let Some(pb::generate_response::Event::Finished(finished)) = response.event.as_mut() {
            finished.stop_match = Some(pb::StopMatch {
                r#match: Some(pb::stop_match::Match::StopTokenId(token_id)),
            });
        }
        response
    }

    pub(super) fn prompt_len(&self) -> usize {
        self.prompt_tokens.len()
    }

    pub(super) fn output_token(&self, position: usize) -> u32 {
        match (position, self.replayed_first_token) {
            (0, Some(token_id)) => token_id,
            _ => deterministic_token_id(self.seed, &self.request_id, position),
        }
    }

    /// `selection` is the candidate setting for the stream this token belongs
    /// to: output tokens and prompt tokens are configured separately, and using
    /// one for the other silently returns nothing when only the other was asked
    /// for.
    ///
    /// `logprob` is an override rather than a computation for one case only --
    /// the replayed first token of a decode request, whose value was produced
    /// by the context phase and arrives in the handoff.
    fn token_info(
        &self,
        token_id: u32,
        with_logprobs: bool,
        selection: Option<&pb::CandidateTokenSelection>,
        logprob: Option<f64>,
    ) -> pb::TokenInfo {
        pb::TokenInfo {
            token_id,
            token: token_text(token_id),
            logprob: with_logprobs.then(|| logprob.unwrap_or_else(|| selected_logprob(token_id))),
            rank: with_logprobs.then_some(1),
            candidates: if with_logprobs {
                candidates(token_id, selection)
            } else {
                Vec::new()
            },
        }
    }

    /// Output position 0 of a decode request is the token the context phase
    /// already produced, so its logprob belongs to the handoff rather than to
    /// this engine: replay the received value instead of regenerating one, and
    /// if the context phase computed none, leave the hole it left.
    ///
    /// Keyed on the position, not the token id. The same token can be sampled
    /// again later in the stream, and those occurrences are this engine's own
    /// -- they must not inherit the replayed token's logprob or its absence.
    fn replayed_logprob(&self, token_id: u32, position: usize) -> Option<Replayed> {
        if position != 0 || self.replayed_first_token != Some(token_id) {
            return None;
        }
        Some(Replayed(self.replayed_first_logprob))
    }

    pub(super) fn token_output(&self, token_id: u32, position: usize) -> pb::TokenOutput {
        let replayed = self.replayed_logprob(token_id, position);
        let with_logprobs =
            self.return_output_logprobs && !matches!(replayed, Some(Replayed(None)));
        let info = self.token_info(
            token_id,
            with_logprobs,
            self.output_candidates.as_ref(),
            replayed.and_then(|Replayed(logprob)| logprob),
        );
        pb::TokenOutput {
            output_index: Some(0),
            text: info.token.clone(),
            tokens: vec![info],
        }
    }

    pub(super) fn prompt_output(&self) -> Option<pb::PromptOutput> {
        self.return_prompt_logprobs.then(|| pb::PromptOutput {
            tokens: self
                .prompt_tokens
                .iter()
                .map(|token_id| {
                    self.token_info(*token_id, true, self.prompt_candidates.as_ref(), None)
                })
                .collect(),
        })
    }

    pub(super) fn usage(
        &self,
        completion_tokens: usize,
        cached_tokens: Option<usize>,
    ) -> pb::Usage {
        let prompt_tokens = self.prompt_len() as u32;
        let completion_tokens = completion_tokens as u32;
        pb::Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            // A real engine always recomputes the last prompt token, so a fully
            // cached prompt reports `prompt_len - 1`, never `prompt_len`
            // (measured against 1.3.0rc26: 95 of 96 on a repeated prompt).
            // Reporting the whole prompt would make a 100% hit rate look
            // reachable when it is not.
            cached_prompt_tokens: cached_tokens
                .map(|tokens| (tokens as u32).min(prompt_tokens.saturating_sub(1))),
            reasoning_tokens: None,
        }
    }

    /// The terminal event an aggregated or decode request ends with.
    pub(super) fn finished(
        &self,
        reason: pb::FinishReason,
        generated: usize,
        cached_tokens: Option<usize>,
    ) -> pb::GenerateResponse {
        super::response_with_usage(
            &self.request_id,
            pb::generate_response::Event::Finished(pb::GenerationFinished {
                output_index: Some(0),
                reason: reason as i32,
                message: String::new(),
                stop_match: None,
            }),
            Some(self.usage(generated, cached_tokens)),
        )
    }

    /// The terminal event a context request ends with instead of `finished`.
    pub(super) fn prefill_ready(&self, config: &MockerServerConfig) -> pb::PrefillReady {
        let first_token = self.output_token(0);
        pb::PrefillReady {
            kv_session: Some(handoff::build_session(
                config,
                self.session_id.clone(),
                &self.request_id,
                self.prompt_len(),
                first_token,
                self.return_output_logprobs
                    .then(|| selected_logprob(first_token)),
            )),
        }
    }
}

fn max_output_tokens(stopping: &pb::StoppingOptions, mode: ServerMode) -> BoxedStatusResult<usize> {
    if mode == ServerMode::Prefill {
        // The client is expected to ask for exactly one token on a context
        // request. Forcing it instead of checking would hide a client that
        // stopped doing so, which is the behaviour this leg exists to test.
        if stopping.max_tokens != Some(1) {
            return Err(Box::new(Status::invalid_argument(format!(
                "a context_only request must ask for exactly one token, got {:?}",
                stopping.max_tokens
            ))));
        }
        return Ok(1);
    }
    let requested = match stopping.max_tokens {
        // The field is `optional`, so an explicit zero is a real request rather
        // than "unset", and asking for no tokens is not satisfiable.
        Some(0) => {
            return Err(Box::new(Status::invalid_argument(
                "max_tokens must be greater than zero",
            )));
        }
        Some(max_tokens) => max_tokens,
        None => DEFAULT_MAX_NEW_TOKENS,
    };
    if requested > MAX_NEW_TOKENS {
        return Err(Box::new(Status::invalid_argument(format!(
            "max_tokens {requested} exceeds the Mocker limit of {MAX_NEW_TOKENS}"
        ))));
    }
    if stopping.min_tokens.unwrap_or(0) > requested {
        return Err(Box::new(Status::invalid_argument(
            "min_tokens must not exceed max_tokens",
        )));
    }
    Ok(requested as usize)
}

fn reject_unsupported(request: &pb::GenerateRequest) -> BoxedStatusResult<()> {
    fn unsupported(what: &str) -> BoxedStatusResult<()> {
        Err(Box::new(Status::unimplemented(format!(
            "{what} is not simulated by the Mocker server"
        ))))
    }
    if !request.media.is_empty() {
        return unsupported("multimodal media");
    }
    if !request.lora_name.is_empty() {
        return unsupported("LoRA selection");
    }
    if request
        .sampling
        .as_ref()
        .is_some_and(|sampling| !matches!(sampling.num_sequences, None | Some(0) | Some(1)))
    {
        return unsupported("num_sequences greater than one");
    }
    // The Mocker samples from its own scheduler, so it cannot honour a grammar.
    // Answering an unconstrained completion would look like success and quietly
    // invalidate whatever the caller was asserting -- the same reason the
    // features above are refused rather than ignored. The sidecar always sends
    // this field when the request carries a constraint
    // (`convert::guided_decoding`), so an unsimulated one is always visible here.
    if request.guided.is_some() {
        return unsupported("guided decoding");
    }
    if let Some(kv) = request.kv.as_ref() {
        if kv.bypass_prefix_cache == Some(true) {
            return unsupported("prefix cache bypass");
        }
        if kv.cache_salt.as_ref().is_some_and(|salt| !salt.is_empty()) {
            return unsupported("cache_salt");
        }
    }
    Ok(())
}

fn is_context_only(extra: Option<&Struct>) -> bool {
    extra
        .and_then(|extra| extra.fields.get(REQUEST_TYPE_KEY))
        .and_then(|value| value.kind.as_ref())
        .is_some_and(|kind| matches!(kind, Kind::StringValue(value) if value == CONTEXT_ONLY))
}

/// A request's disaggregation role has to agree with the role this process was
/// started as. Serving a decode request on a prefill server would "work" and
/// quietly invalidate whatever the test was asserting.
fn validate_role(
    context_only: bool,
    session: Option<&pb::KvSessionRef>,
    mode: ServerMode,
) -> BoxedStatusResult<()> {
    let (shape, accepted_by): (&str, &[ServerMode]) = match (context_only, session.is_some()) {
        (true, true) => {
            return Err(Box::new(Status::invalid_argument(
                "a request cannot be both context_only and carry a kv_session",
            )));
        }
        (true, false) => ("context_only", &[ServerMode::Prefill]),
        (false, true) => ("carrying a kv_session", &[ServerMode::Decode]),
        // Neither marker is the aggregated shape, and it is also what
        // conditional disaggregation dispatches straight to a decode worker
        // when it bypasses prefill -- the decode engine is then expected to run
        // the context phase itself. The sidecar builds exactly this request
        // (`convert::build_generate_request`, and the
        // `decode_without_a_prefill_result_runs_the_whole_request` test), so a
        // decode server that refused it would reject every bypassed request.
        (false, false) => (
            "neither context_only nor carrying a kv_session",
            &[ServerMode::Aggregated, ServerMode::Decode],
        ),
    };
    if !accepted_by.contains(&mode) {
        return Err(Box::new(Status::failed_precondition(format!(
            "this server runs in {mode} mode, but the request is {shape}"
        ))));
    }
    Ok(())
}

/// A replayed position's logprob: `None` inside means the context phase
/// computed none, which is different from this position not being a replay.
#[derive(Clone, Copy)]
struct Replayed(Option<f64>);

fn token_text(token_id: u32) -> String {
    format!("<token:{token_id}>")
}

fn selected_logprob(token_id: u32) -> f64 {
    -0.1 * f64::from((token_id % 10) + 1)
}

fn candidates(selected: u32, selection: Option<&pb::CandidateTokenSelection>) -> Vec<pb::LogProb> {
    let ids: Vec<u32> = match selection.and_then(|selection| selection.selection.as_ref()) {
        None => Vec::new(),
        Some(pb::candidate_token_selection::Selection::TopN(count)) => (0..(*count as usize)
            .min(MAX_CANDIDATES))
            .map(|offset| selected.wrapping_add(offset as u32))
            .collect(),
        Some(pb::candidate_token_selection::Selection::TokenIds(ids)) => {
            ids.ids.iter().copied().take(MAX_CANDIDATES).collect()
        }
        Some(pb::candidate_token_selection::Selection::All(_)) => (0..MAX_CANDIDATES)
            .map(|offset| selected.wrapping_add(offset as u32))
            .collect(),
    };
    ids.into_iter()
        .enumerate()
        .map(|(index, token_id)| pb::LogProb {
            token_id,
            logprob: selected_logprob(selected) - 0.1 * index as f64,
            token: token_text(token_id),
            rank: Some(index as u32 + 1),
        })
        .collect()
}
