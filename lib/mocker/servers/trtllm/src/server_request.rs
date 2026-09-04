// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_mocker::common::protocols::DirectRequest;
use dynamo_mocker::live::{deterministic_token_id, stable_request_uuid};
use dynamo_trtllm_sidecar::disagg::{CONTEXT_ONLY, REQUEST_TYPE_KEY};
use dynamo_trtllm_sidecar::proto as pb;
use prost_types::{Struct, value::Kind};
use tonic::Status;
use uuid::Uuid;

use super::handoff;
use super::{BoxedStatusResult, MockerServerConfig, ServerMode};

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
    /// On a decode request this is the session the client presented, so an
    /// `Abort` targeting it resolves. Otherwise it equals `session_id`.
    pub(super) client_session_id: String,
    seed: u64,
    /// The context phase's first token, replayed as this leg's first output so
    /// the two legs' token accounting adds up the way a real engine's does.
    replayed_first_token: Option<u32>,
    prompt_tokens: Vec<u32>,
    pub(super) max_output_tokens: usize,
    return_output_logprobs: bool,
    return_prompt_logprobs: bool,
    output_candidates: Option<pb::CandidateTokenSelection>,
}

impl PreparedRequest {
    pub(super) fn new(
        request: pb::GenerateRequest,
        config: &MockerServerConfig,
    ) -> BoxedStatusResult<Self> {
        if request.model.is_empty() {
            return Err(Box::new(Status::invalid_argument(
                "model must be non-empty",
            )));
        }
        if request.model != config.model {
            return Err(Box::new(Status::not_found(format!(
                "model '{}' is not served; this server serves '{}'",
                request.model, config.model
            ))));
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
                .map_or_else(|| session_id.clone(), |session| session.session_id.clone()),
            session_id,
            seed: config.seed,
            replayed_first_token: kv.session.as_ref().and_then(handoff::first_gen_token),
            request_id,
            prompt_tokens,
            max_output_tokens,
            return_output_logprobs: response.return_output_logprobs == Some(true),
            return_prompt_logprobs: response.return_prompt_logprobs == Some(true),
            output_candidates: response.output_candidates,
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

    pub(super) fn prompt_len(&self) -> usize {
        self.prompt_tokens.len()
    }

    pub(super) fn output_token(&self, position: usize) -> u32 {
        match (position, self.replayed_first_token) {
            (0, Some(token_id)) => token_id,
            _ => deterministic_token_id(self.seed, &self.request_id, position),
        }
    }

    fn token_info(&self, token_id: u32, with_logprobs: bool) -> pb::TokenInfo {
        pb::TokenInfo {
            token_id,
            token: token_text(token_id),
            logprob: with_logprobs.then(|| selected_logprob(token_id)),
            rank: with_logprobs.then_some(1),
            candidates: if with_logprobs {
                candidates(token_id, self.output_candidates.as_ref())
            } else {
                Vec::new()
            },
        }
    }

    pub(super) fn token_output(&self, token_id: u32) -> pb::TokenOutput {
        let info = self.token_info(token_id, self.return_output_logprobs);
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
                .map(|token_id| self.token_info(*token_id, true))
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
            cached_prompt_tokens: cached_tokens.map(|tokens| tokens as u32),
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
        pb::GenerateResponse {
            request_id: self.request_id.clone(),
            event: Some(pb::generate_response::Event::Finished(
                pb::GenerationFinished {
                    output_index: Some(0),
                    reason: reason as i32,
                    message: String::new(),
                    stop_match: None,
                },
            )),
            usage: Some(self.usage(generated, cached_tokens)),
        }
    }

    /// The terminal event a context request ends with instead of `finished`.
    pub(super) fn prefill_ready(&self, config: &MockerServerConfig) -> pb::PrefillReady {
        pb::PrefillReady {
            kv_session: Some(handoff::build_session(
                config,
                self.session_id.clone(),
                &self.request_id,
                self.prompt_len(),
                self.output_token(0),
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
    let requested = match (context_only, session.is_some()) {
        (true, true) => {
            return Err(Box::new(Status::invalid_argument(
                "a request cannot be both context_only and carry a kv_session",
            )));
        }
        (true, false) => ServerMode::Prefill,
        (false, true) => ServerMode::Decode,
        (false, false) => ServerMode::Aggregated,
    };
    if requested != mode {
        return Err(Box::new(Status::failed_precondition(format!(
            "this server runs in {mode} mode, but the request is {}",
            match requested {
                ServerMode::Aggregated => "neither context_only nor carrying a kv_session",
                ServerMode::Prefill => "context_only",
                ServerMode::Decode => "carrying a kv_session",
            }
        ))));
    }
    Ok(())
}

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
