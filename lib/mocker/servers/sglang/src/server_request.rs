// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Validation and frame building for one Generate request.

use std::time::Instant;

use dynamo_mocker::common::protocols::DirectRequest;
use dynamo_mocker::live::{deterministic_output_tokens, stable_request_uuid};
use dynamo_mocker::sglang::{LogprobOptions, ResponseMetadata};
use dynamo_sglang_sidecar::proto as pb;
use serde_json::Value;
use tonic::Status;
use uuid::Uuid;

use super::{BoxedStatusResult, DP_RANK, MockerServerConfig, ServerMode};

const DEFAULT_MAX_NEW_TOKENS: i64 = 128;
const MAX_NEW_TOKENS: i64 = 1_000_000;

#[derive(Debug)]
pub(super) struct PreparedRequest {
    uuid: Uuid,
    prompt_tokens: Vec<u32>,
    pub(super) max_output_tokens: usize,
    pub(super) has_decode_handoff: bool,
    output_token_ids: Vec<u32>,
    stop_token: Option<u32>,
    response_metadata: ResponseMetadata,
    /// SGLang's in-stream rejection of a request that cannot fit the context.
    pub(super) context_error: Option<String>,
    started: Instant,
}

/// The Mocker serves single requests; SGLang's list forms are batches.
fn batch_error(field: &str) -> Box<Status> {
    Box::new(Status::invalid_argument(format!(
        "{field} must be a single value; the Mocker does not serve batch requests"
    )))
}

fn one_int64(field: &str, value: Option<pb::Int64OrList>) -> BoxedStatusResult<Option<i64>> {
    match value.and_then(|value| value.value) {
        Some(pb::int64_or_list::Value::One(value)) => Ok(Some(value)),
        Some(pb::int64_or_list::Value::Many(_)) => Err(batch_error(field)),
        None => Ok(None),
    }
}

impl PreparedRequest {
    pub(super) fn new(
        request: pb::GenerateRequest,
        config: &MockerServerConfig,
    ) -> BoxedStatusResult<Self> {
        if request.text.is_some() {
            return Err(Status::invalid_argument(
                "text prompts require a tokenizer, which the Mocker does not load",
            )
            .into());
        }
        let input_ids = match request.input_ids.and_then(|ids| ids.value) {
            Some(pb::token_ids_or_list::Value::One(ids)) => ids.ids,
            Some(pb::token_ids_or_list::Value::Many(_)) => return Err(batch_error("input_ids")),
            None => Vec::new(),
        };
        if input_ids.is_empty() {
            return Err(Status::invalid_argument("input_ids must not be empty").into());
        }
        let prompt_tokens = input_ids
            .iter()
            .map(|token| {
                u32::try_from(*token).map_err(|_| {
                    Box::new(Status::invalid_argument(format!(
                        "input_ids contains an invalid token ID: {token}"
                    )))
                })
            })
            .collect::<BoxedStatusResult<Vec<_>>>()?;

        let sampling = match request.sampling_params.and_then(|params| params.value) {
            Some(pb::sampling_params_or_list::Value::One(params)) => params,
            Some(pb::sampling_params_or_list::Value::Many(_)) => {
                return Err(batch_error("sampling_params"));
            }
            None => pb::SamplingParams::default(),
        };
        if sampling.n.is_some_and(|n| n != 1) {
            return Err(Status::invalid_argument("sampling_params.n must be 1").into());
        }

        // An unset protobuf max_new_tokens takes SGLang's schema default.
        let requested_max = sampling.max_new_tokens.unwrap_or(DEFAULT_MAX_NEW_TOKENS);
        if requested_max <= 0 || requested_max > MAX_NEW_TOKENS {
            return Err(Status::invalid_argument(format!(
                "max_new_tokens must be between 1 and {MAX_NEW_TOKENS}"
            ))
            .into());
        }
        let min_new_tokens = sampling.min_new_tokens.unwrap_or(0);
        if !(0..=requested_max).contains(&min_new_tokens) {
            return Err(Status::invalid_argument(
                "min_new_tokens must be between 0 and max_new_tokens",
            )
            .into());
        }
        let max_output_tokens = if config.mode == ServerMode::Prefill {
            1
        } else {
            requested_max as usize
        };
        let context_error = context_error(prompt_tokens.len(), max_output_tokens, config);

        let has_handoff = validate_role(
            config,
            request.bootstrap_host,
            request.bootstrap_port,
            request.bootstrap_room,
        )?;

        let return_logprob = match request.return_logprob.and_then(|value| value.value) {
            Some(pb::bool_or_list::Value::One(value)) => value,
            Some(pb::bool_or_list::Value::Many(_)) => return Err(batch_error("return_logprob")),
            None => false,
        };
        let logprob_options = LogprobOptions::new(
            return_logprob,
            one_int64("top_logprobs_num", request.top_logprobs_num)?.unwrap_or(0),
            one_int64("logprob_start_len", request.logprob_start_len)?.unwrap_or(-1),
        )
        .map_err(|message| Box::new(Status::invalid_argument(message)))?;

        let request_id = match request.rid.and_then(|rid| rid.value) {
            Some(pb::string_or_list::Value::One(rid)) if !rid.trim().is_empty() => rid,
            Some(pb::string_or_list::Value::Many(_)) => return Err(batch_error("rid")),
            _ => Uuid::new_v4().to_string(),
        };
        let uuid = stable_request_uuid(config.seed, &request_id);
        let mut output_token_ids =
            deterministic_output_tokens(config.seed, &request_id, max_output_tokens);
        let stop_token_ids = sampling
            .stop_token_ids
            .map(|ids| ids.items)
            .unwrap_or_default();
        let stop_position = (!stop_token_ids.is_empty() && !sampling.ignore_eos.unwrap_or(false))
            .then(|| {
                output_token_ids
                    .iter()
                    .enumerate()
                    .skip(min_new_tokens as usize)
                    .find(|(_, token)| stop_token_ids.contains(&i64::from(**token)))
                    .map(|(position, _)| position)
            })
            .flatten();
        let stop_token = stop_position.map(|position| {
            output_token_ids.truncate(position + 1);
            output_token_ids[position]
        });
        let max_output_tokens = output_token_ids.len();
        let response_metadata = ResponseMetadata::new(request_id, &prompt_tokens, logprob_options);
        Ok(Self {
            uuid,
            prompt_tokens,
            max_output_tokens,
            has_decode_handoff: config.mode == ServerMode::Decode && has_handoff,
            output_token_ids,
            stop_token,
            response_metadata,
            context_error,
            started: Instant::now(),
        })
    }

    pub(super) fn direct_request(&self) -> DirectRequest {
        DirectRequest {
            tokens: self.prompt_tokens.clone(),
            max_output_tokens: self.max_output_tokens,
            output_token_ids: Some(self.output_token_ids.clone()),
            uuid: Some(self.uuid),
            dp_rank: DP_RANK,
            ..Default::default()
        }
    }

    /// One incremental frame. Like SGLang's Rust server, input logprobs ride
    /// the first frame only, later logprob frames carry an empty input list,
    /// and `completion_tokens` stays cumulative.
    pub(super) fn frame(
        &self,
        output_tokens: &[u32],
        completion_tokens: usize,
        is_first: bool,
        is_terminal: bool,
    ) -> pb::GenerateResponse {
        let json = self
            .response_metadata
            .meta_info(output_tokens, completion_tokens, None);
        let mut meta_info = pb::GenerateMetaInfo {
            id: self.response_metadata.request_id().to_string(),
            prompt_tokens: u32::try_from(self.prompt_tokens.len()).unwrap_or(u32::MAX),
            completion_tokens: completion_tokens as u64,
            output_token_logprobs: json.get("output_token_logprobs").map(logprob_entries),
            output_top_logprobs: json
                .get("output_top_logprobs")
                .map(logprob_rows)
                .unwrap_or_default(),
            ..Default::default()
        };
        if meta_info.output_token_logprobs.is_some() {
            meta_info.input_token_logprobs = Some(pb::LogprobEntries::default());
        }
        if is_first {
            let prompt = self.response_metadata.prompt_logprobs();
            if let Some(selected) = prompt.get("input_token_logprobs") {
                meta_info.input_token_logprobs = Some(logprob_entries(selected));
            }
            if let Some(top) = prompt.get("input_top_logprobs") {
                meta_info.input_top_logprobs = logprob_rows(top);
            }
        }
        if is_terminal {
            meta_info.finish_reason = Some(self.finish_reason());
            meta_info.e2e_latency = Some(self.started.elapsed().as_secs_f64());
        }
        pb::GenerateResponse {
            text: String::new(),
            meta_info: Some(meta_info),
            output_ids: (!output_tokens.is_empty()).then(|| pb::TokenIds {
                ids: output_tokens.iter().copied().map(i64::from).collect(),
            }),
            index: None,
        }
    }

    fn finish_reason(&self) -> pb::FinishReason {
        use pb::finish_reason::Kind;
        pb::FinishReason {
            kind: Some(match self.stop_token {
                Some(token) => Kind::Stop(pb::FinishStop {
                    matched: Some(pb::Matched {
                        value: Some(pb::matched::Value::Token(i64::from(token))),
                    }),
                }),
                None => Kind::Length(pb::FinishLength {
                    length: Some(self.max_output_tokens as u64),
                }),
            }),
        }
    }
}

/// SGLang's tokenizer-side context check, which fails the request inside an
/// open stream rather than rejecting the RPC.
fn context_error(
    prompt_tokens: usize,
    max_new_tokens: usize,
    config: &MockerServerConfig,
) -> Option<String> {
    let context_length = config.context_length as usize;
    if prompt_tokens >= context_length {
        return Some(format!(
            "The input ({prompt_tokens} tokens) is longer than the model's context length \
             ({context_length} tokens)."
        ));
    }
    let total = prompt_tokens.saturating_add(max_new_tokens);
    (total > context_length).then(|| {
        format!(
            "Requested token count exceeds the model's maximum context length of \
             {context_length} tokens. You requested a total of {total} tokens: \
             {prompt_tokens} tokens from the input messages and {max_new_tokens} tokens \
             for the completion. Please reduce the number of tokens in the input messages \
             or the completion to fit within the limit."
        )
    })
}

/// `[logprob, token_id, text]` JSON entries from the shared metadata builder.
fn logprob_entries(value: &Value) -> pb::LogprobEntries {
    pb::LogprobEntries {
        entries: value
            .as_array()
            .map(|entries| entries.iter().map(logprob_entry).collect())
            .unwrap_or_default(),
    }
}

fn logprob_entry(value: &Value) -> pb::LogprobEntry {
    pb::LogprobEntry {
        logprob: value.get(0).and_then(Value::as_f64),
        token_id: value.get(1).and_then(Value::as_i64).unwrap_or_default(),
        text: value.get(2).and_then(Value::as_str).map(str::to_string),
    }
}

/// Per-position alternatives; a null position has no row.
fn logprob_rows(value: &Value) -> Vec<pb::NullableTopLogprobs> {
    value
        .as_array()
        .map(|rows| {
            rows.iter()
                .map(|row| pb::NullableTopLogprobs {
                    row: row.as_array().map(|entries| pb::TopLogprobRow {
                        entries: entries.iter().map(logprob_entry).collect(),
                    }),
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Whether the request carries disaggregation rendezvous fields.
fn validate_role(
    config: &MockerServerConfig,
    host: Option<pb::OptionalStringOrList>,
    port: Option<pb::OptionalInt64OrList>,
    room: Option<pb::OptionalInt64OrList>,
) -> BoxedStatusResult<bool> {
    let has_params = host.is_some() || port.is_some() || room.is_some();
    let one_string = |value: Option<pb::OptionalStringOrList>| match value.and_then(|v| v.value) {
        Some(pb::optional_string_or_list::Value::One(value)) => value.value,
        _ => None,
    };
    let one_int = |value: Option<pb::OptionalInt64OrList>| match value.and_then(|v| v.value) {
        Some(pb::optional_int64_or_list::Value::One(value)) => value.value,
        _ => None,
    };
    match (config.mode, has_params) {
        (ServerMode::Aggregated, false) => Ok(false),
        (ServerMode::Aggregated, true) => Err(Status::failed_precondition(
            "aggregated mock server received disaggregated parameters",
        )
        .into()),
        (ServerMode::Prefill | ServerMode::Decode, false) => Err(Status::failed_precondition(
            "disaggregated mock server requires bootstrap_host, bootstrap_port, and bootstrap_room",
        )
        .into()),
        (ServerMode::Prefill | ServerMode::Decode, true) => {
            let (host, port, room) = (one_string(host), one_int(port), one_int(room));
            let (Some(host), Some(port), Some(room)) = (host, port, room) else {
                return Err(Status::invalid_argument(
                    "disaggregated parameters must contain a host, positive port, and non-negative room",
                )
                .into());
            };
            if host.trim().is_empty() || port <= 0 || room < 0 {
                return Err(Status::invalid_argument(
                    "disaggregated parameters must contain a host, positive port, and non-negative room",
                )
                .into());
            }
            if config.mode == ServerMode::Prefill && i64::from(config.bootstrap_port) != port {
                return Err(Status::failed_precondition(format!(
                    "prefill bootstrap_port {port} does not match discovered port {}",
                    config.bootstrap_port
                ))
                .into());
            }
            Ok(true)
        }
    }
}
