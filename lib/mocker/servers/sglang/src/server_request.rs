// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use dynamo_mocker::common::protocols::DirectRequest;
use dynamo_mocker::live::{deterministic_output_tokens, stable_request_uuid};
use dynamo_mocker::sglang::{LogprobOptions, ResponseMetadata};
use dynamo_sglang_sidecar::proto as pb;
use serde_json::json;
use tonic::Status;
use uuid::Uuid;

use super::{BoxedStatusResult, DP_RANK, MockerServerConfig, ServerMode};

const DEFAULT_MAX_NEW_TOKENS: i32 = 128;
const MAX_NEW_TOKENS: i32 = 1_000_000;

#[derive(Debug)]
pub(super) struct PreparedRequest {
    uuid: Uuid,
    prompt_tokens: Vec<u32>,
    pub(super) max_output_tokens: usize,
    pub(super) has_decode_handoff: bool,
    output_token_ids: Vec<u32>,
    stop_token: Option<u32>,
    response_metadata: ResponseMetadata,
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
        let min_new_tokens = request
            .sampling_params
            .as_ref()
            .and_then(|params| params.min_new_tokens)
            .unwrap_or(0);
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

        let logprob_options = LogprobOptions::new(
            request.return_logprob.unwrap_or(false),
            i64::from(request.top_logprobs_num.unwrap_or(0)),
            i64::from(request.logprob_start_len.unwrap_or(-1)),
        )
        .map_err(|message| Box::new(Status::invalid_argument(message)))?;

        let request_id = request
            .rid
            .filter(|request_id| !request_id.trim().is_empty())
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let uuid = stable_request_uuid(config.seed, &request_id);
        let mut output_token_ids =
            deterministic_output_tokens(config.seed, &request_id, max_output_tokens);
        let stop_position = request.sampling_params.as_ref().and_then(|params| {
            if params.stop_token_ids.is_empty() || params.ignore_eos.unwrap_or(false) {
                return None;
            }
            output_token_ids
                .iter()
                .enumerate()
                .skip(min_new_tokens as usize)
                .find(|(_, token)| params.stop_token_ids.contains(&(**token as i32)))
                .map(|(position, _)| position)
        });
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
            has_decode_handoff: config.mode == ServerMode::Decode
                && request.disaggregated_params.is_some(),
            output_token_ids,
            stop_token,
            response_metadata,
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
        completion_tokens: usize,
        terminal: bool,
    ) -> HashMap<String, String> {
        let mut meta = self
            .response_metadata
            .meta_info(
                output_tokens,
                completion_tokens,
                terminal.then(|| match self.stop_token {
                    Some(token) => json!({"type": "stop", "matched": token}),
                    None => json!({"type": "length", "length": self.max_output_tokens}),
                }),
            )
            .into_iter()
            .map(|(key, value)| (key, value.to_string()))
            .collect::<HashMap<_, _>>();
        // Alias only this mock server emits; the rest of the fields are shared.
        meta.insert(
            "mocker_request_id".to_string(),
            json!(self.response_metadata.request_id()).to_string(),
        );
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
