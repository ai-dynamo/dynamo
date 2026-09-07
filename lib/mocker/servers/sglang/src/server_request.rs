// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use dynamo_mocker::common::protocols::DirectRequest;
use dynamo_mocker::live::{deterministic_output_tokens, stable_request_uuid};
use dynamo_sglang_sidecar::proto as pb;
use serde_json::{Value, json};
use tonic::Status;
use uuid::Uuid;

use super::{BoxedStatusResult, DP_RANK, MockerServerConfig, ServerMode};

const DEFAULT_MAX_NEW_TOKENS: i32 = 20;
const MAX_NEW_TOKENS: i32 = 1_000_000;
const MAX_TOP_LOGPROBS: i32 = 20;

#[derive(Debug)]
pub(super) struct PreparedRequest {
    uuid: Uuid,
    request_id: String,
    prompt_tokens: Vec<u32>,
    pub(super) max_output_tokens: usize,
    output_token_ids: Vec<u32>,
    return_logprob: bool,
    top_logprobs_num: usize,
    logprob_start_len: i32,
}

impl PreparedRequest {
    pub(super) fn new(
        request: pb::GenerateRequest,
        config: &MockerServerConfig,
    ) -> BoxedStatusResult<Self> {
        if request.input_ids.is_empty() {
            return Err(Status::invalid_argument("input_ids must not be empty").into());
        }
        let prompt_tokens = request
            .input_ids
            .iter()
            .map(|token| {
                u32::try_from(*token).map_err(|_| {
                    Box::new(Status::invalid_argument(format!(
                        "input_ids contains a negative token ID: {token}"
                    )))
                })
            })
            .collect::<BoxedStatusResult<Vec<_>>>()?;

        if let Some(n) = request.sampling_params.as_ref().and_then(|params| params.n)
            && n != 1
        {
            return Err(Status::invalid_argument("sampling_params.n must be 1").into());
        }

        let requested_max = request
            .sampling_params
            .as_ref()
            .and_then(|params| params.max_new_tokens)
            .unwrap_or(DEFAULT_MAX_NEW_TOKENS);
        if requested_max <= 0 || requested_max > MAX_NEW_TOKENS {
            return Err(Status::invalid_argument(format!(
                "max_new_tokens must be between 1 and {MAX_NEW_TOKENS}"
            ))
            .into());
        }
        let max_output_tokens = if config.mode == ServerMode::Prefill {
            1
        } else {
            requested_max as usize
        };
        let total_tokens = prompt_tokens
            .len()
            .checked_add(max_output_tokens)
            .ok_or_else(|| Status::invalid_argument("prompt and output token count overflows"))?;
        if total_tokens > config.context_length as usize {
            return Err(Status::invalid_argument(format!(
                "prompt tokens ({}) plus max_new_tokens ({max_output_tokens}) exceed context_length {}",
                prompt_tokens.len(),
                config.context_length
            ))
            .into());
        }

        validate_role(config, request.disaggregated_params.as_ref())?;

        let top_logprobs_num = request.top_logprobs_num.unwrap_or(0);
        if !(0..=MAX_TOP_LOGPROBS).contains(&top_logprobs_num) {
            return Err(Status::invalid_argument(format!(
                "top_logprobs_num must be between 0 and {MAX_TOP_LOGPROBS}"
            ))
            .into());
        }
        let logprob_start_len = request.logprob_start_len.unwrap_or(-1);
        if logprob_start_len < -1 {
            return Err(Status::invalid_argument("logprob_start_len must be -1 or greater").into());
        }

        let request_id = request
            .rid
            .filter(|request_id| !request_id.trim().is_empty())
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let uuid = stable_request_uuid(config.seed, &request_id);
        let output_token_ids =
            deterministic_output_tokens(config.seed, &request_id, max_output_tokens);
        Ok(Self {
            uuid,
            request_id,
            prompt_tokens,
            max_output_tokens,
            output_token_ids,
            return_logprob: request.return_logprob.unwrap_or(false),
            top_logprobs_num: top_logprobs_num as usize,
            logprob_start_len,
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

    pub(super) fn meta_info(
        &self,
        output_tokens: &[u32],
        terminal: bool,
    ) -> HashMap<String, String> {
        let mut meta = HashMap::from([
            (
                "prompt_tokens".to_string(),
                Value::from(self.prompt_tokens.len()).to_string(),
            ),
            (
                "mocker_request_id".to_string(),
                Value::String(self.request_id.clone()).to_string(),
            ),
        ]);
        if self.return_logprob {
            insert_json(
                &mut meta,
                "output_token_logprobs",
                Value::Array(
                    output_tokens
                        .iter()
                        .map(|token| logprob_entry(*token))
                        .collect(),
                ),
            );
            if self.top_logprobs_num > 0 {
                insert_json(
                    &mut meta,
                    "output_top_logprobs",
                    Value::Array(
                        output_tokens
                            .iter()
                            .map(|token| top_logprob_entries(*token, self.top_logprobs_num))
                            .collect(),
                    ),
                );
            }
        }
        if terminal {
            insert_json(&mut meta, "finish_reason", json!({"type": "length"}));
            if self.return_logprob && self.logprob_start_len >= 0 {
                let start = (self.logprob_start_len as usize).min(self.prompt_tokens.len());
                let prompt_tokens = &self.prompt_tokens[start..];
                let mut input_token_logprobs = Vec::with_capacity(prompt_tokens.len());
                let mut input_top_logprobs = Vec::with_capacity(prompt_tokens.len());
                if let Some((first, remaining)) = prompt_tokens.split_first() {
                    // Native SGLang retains the first token ID but uses a null
                    // logprob because no preceding token predicts it.
                    input_token_logprobs.push(json!([null, first, null]));
                    input_token_logprobs
                        .extend(remaining.iter().map(|token| logprob_entry(*token)));
                    if self.top_logprobs_num > 0 {
                        input_top_logprobs.push(Value::Null);
                        input_top_logprobs.extend(
                            remaining
                                .iter()
                                .map(|token| top_logprob_entries(*token, self.top_logprobs_num)),
                        );
                    }
                }
                insert_json(
                    &mut meta,
                    "input_token_logprobs",
                    Value::Array(input_token_logprobs),
                );
                if self.top_logprobs_num > 0 {
                    insert_json(
                        &mut meta,
                        "input_top_logprobs",
                        Value::Array(input_top_logprobs),
                    );
                }
            }
        }
        meta
    }
}

fn validate_role(
    config: &MockerServerConfig,
    params: Option<&pb::DisaggregatedParams>,
) -> BoxedStatusResult<()> {
    match (config.mode, params) {
        (ServerMode::Aggregated, None) => Ok(()),
        (ServerMode::Aggregated, Some(_)) => Err(Status::failed_precondition(
            "aggregated mock server received disaggregated parameters",
        )
        .into()),
        (ServerMode::Prefill | ServerMode::Decode, None) => Err(Status::failed_precondition(
            "disaggregated mock server requires bootstrap_host, bootstrap_port, and bootstrap_room",
        )
        .into()),
        (ServerMode::Prefill | ServerMode::Decode, Some(params)) => {
            if params.bootstrap_host.trim().is_empty()
                || params.bootstrap_port <= 0
                || params.bootstrap_room < 0
            {
                return Err(
                    Status::invalid_argument(
                        "disaggregated parameters must contain a host, positive port, and non-negative room",
                    )
                    .into(),
                );
            }
            if config.mode == ServerMode::Prefill
                && i32::from(config.bootstrap_port) != params.bootstrap_port
            {
                return Err(Status::failed_precondition(format!(
                    "prefill bootstrap_port {} does not match discovered port {}",
                    params.bootstrap_port, config.bootstrap_port
                ))
                .into());
            }
            Ok(())
        }
    }
}

fn selected_logprob(token_id: u32) -> f64 {
    -0.1 * f64::from((token_id % 10) + 1)
}

fn logprob_entry(token_id: u32) -> Value {
    json!([
        selected_logprob(token_id),
        token_id,
        format!("<token:{token_id}>")
    ])
}

fn top_logprob_entries(token_id: u32, count: usize) -> Value {
    Value::Array(
        (0..count)
            .map(|offset| {
                let candidate = token_id.saturating_add(offset as u32);
                json!([
                    selected_logprob(candidate) - (offset as f64 * 0.01),
                    candidate,
                    format!("<token:{candidate}>")
                ])
            })
            .collect(),
    )
}

fn insert_json(meta: &mut HashMap<String, String>, key: &str, value: Value) {
    meta.insert(key.to_string(), value.to_string());
}
