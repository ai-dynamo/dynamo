// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use dynamo_mocker::common::protocols::DirectRequest;
use dynamo_vllm_sidecar::proto as pb;
use prost_types::{ListValue, Struct, Value, value::Kind};
use tonic::Status;
use uuid::Uuid;

use super::{BoxedStatusResult, DP_RANK, MockerServerConfig, ServerMode};

pub(super) const DEFAULT_MAX_NEW_TOKENS: u32 = 20;
// Bound the request-owned synthetic token plan independently of LiveEngine's
// fixed per-request delivery buffer.
pub(super) const MAX_NEW_TOKENS: u32 = 32_768;
const MAX_CANDIDATES: usize = 20;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum KvTransferRole {
    Aggregated,
    Prefill,
    Decode,
}

pub(super) const DECODE_RENDEZVOUS_FIELDS: [&str; 4] = [
    "remote_engine_id",
    "remote_host",
    "remote_port",
    "remote_block_ids",
];

/// Non-rendezvous field the prefill role stamps into every handoff. The decode
/// role explicitly requires it so the disaggregated round trip fails loudly if
/// the sidecar ever drops opaque KV-transfer fields instead of forwarding the
/// payload verbatim.
pub(super) const HANDOFF_SENTINEL_FIELD: &str = "mocker_request_id";

pub(super) trait SequenceOutputExt {
    fn with_total_output_tokens(self, count: usize) -> Self;
}

impl SequenceOutputExt for pb::SequenceOutput {
    fn with_total_output_tokens(mut self, count: usize) -> Self {
        if let Some(finish) = self.finish_info.as_mut() {
            finish.num_output_tokens = count as u32;
        }
        self
    }
}

#[derive(Debug)]
pub(super) struct PreparedRequest {
    pub(super) uuid: Uuid,
    request_id: String,
    output_token_seed: u64,
    prompt_tokens: Vec<u32>,
    pub(super) max_output_tokens: usize,
    priority: i32,
    response: pb::ResponseOptions,
    mode: ServerMode,
}

impl PreparedRequest {
    pub(super) fn new(
        mut request: pb::GenerateRequest,
        config: &MockerServerConfig,
    ) -> BoxedStatusResult<Self> {
        if !request.model.is_empty() && request.model != config.model {
            return Err(Status::not_found(format!(
                "model '{}' is not served; expected '{}'",
                request.model, config.model
            ))
            .into());
        }
        let mut prompt_tokens = match request.prompt.take() {
            Some(pb::generate_request::Prompt::TokenIds(tokens)) => tokens.ids,
            Some(pb::generate_request::Prompt::Text(_)) => {
                return Err(Status::unimplemented(
                    "text tokenization is not available in the CPU-only Mocker server; send token_ids",
                )
                .into());
            }
            None => return Err(Status::invalid_argument("prompt is required").into()),
        };
        if prompt_tokens.is_empty() {
            return Err(Status::invalid_argument("token_ids must not be empty").into());
        }
        if request.truncate_prompt_tokens > 0 {
            let keep = request.truncate_prompt_tokens as usize;
            if prompt_tokens.len() > keep {
                prompt_tokens.drain(..prompt_tokens.len() - keep);
            }
        }
        if request
            .sampling
            .as_ref()
            .is_some_and(|sampling| sampling.num_sequences > 1)
        {
            return Err(Status::invalid_argument("num_sequences must be 0 or 1").into());
        }

        if let Some(kv) = request.kv.as_ref() {
            if kv.bypass_prefix_cache {
                return Err(Status::invalid_argument(
                    "bypass_prefix_cache is not supported by the Mocker server",
                )
                .into());
            }
            if !kv.cache_salt.is_empty() {
                return Err(Status::invalid_argument(
                    "cache_salt is not supported by the Mocker server",
                )
                .into());
            }
        }
        let transfer_role = KvTransferRole::classify(
            request
                .kv
                .as_ref()
                .and_then(|kv| kv.kv_transfer_params.as_ref()),
        )?;
        match (config.mode, transfer_role) {
            (ServerMode::Aggregated, KvTransferRole::Prefill | KvTransferRole::Decode) => {
                return Err(Status::failed_precondition(
                    "aggregated mock server received disaggregated KV transfer parameters",
                )
                .into());
            }
            (ServerMode::Prefill, role) if role != KvTransferRole::Prefill => {
                return Err(Status::failed_precondition(
                    "prefill mock server requires do_remote_decode=true",
                )
                .into());
            }
            (ServerMode::Decode, role) if role != KvTransferRole::Decode => {
                return Err(Status::failed_precondition(
                    "decode mock server requires a prefill KV transfer payload",
                )
                .into());
            }
            _ => {}
        }

        let stopping = request.stopping.as_ref();
        let max_new_tokens = stopping
            .map(|stopping| stopping.max_new_tokens)
            .unwrap_or_default();
        let max_new_tokens = if max_new_tokens == 0 {
            DEFAULT_MAX_NEW_TOKENS
        } else {
            max_new_tokens
        };
        if max_new_tokens > MAX_NEW_TOKENS {
            return Err(Status::invalid_argument(format!(
                "max_new_tokens must not exceed {MAX_NEW_TOKENS}"
            ))
            .into());
        }
        let min_new_tokens = stopping
            .map(|stopping| stopping.min_new_tokens)
            .unwrap_or_default();
        if min_new_tokens > max_new_tokens {
            return Err(Status::invalid_argument(format!(
                "min_new_tokens ({min_new_tokens}) must not exceed effective max_new_tokens ({max_new_tokens})"
            ))
            .into());
        }

        let request_id = if request.request_id.is_empty() {
            Uuid::new_v4().to_string()
        } else {
            request.request_id
        };
        let uuid = stable_uuid(config.seed, &request_id);
        let output_token_seed = synthetic_token_seed(config.seed, &request_id);
        let max_output_tokens = max_new_tokens as usize;
        Ok(Self {
            uuid,
            request_id,
            output_token_seed,
            prompt_tokens,
            max_output_tokens,
            priority: request.priority,
            response: request.response.unwrap_or_default(),
            mode: config.mode,
        })
    }

    pub(super) fn direct_request(&self) -> DirectRequest {
        DirectRequest {
            tokens: self.prompt_tokens.clone(),
            max_output_tokens: self.max_output_tokens,
            output_token_ids: Some(
                (0..self.max_output_tokens)
                    .map(|position| self.output_token(position))
                    .collect(),
            ),
            uuid: Some(self.uuid),
            dp_rank: DP_RANK,
            priority: self.priority,
            ..Default::default()
        }
    }

    pub(super) fn output_token(&self, position: usize) -> u32 {
        deterministic_token_id(self.output_token_seed, position)
    }

    pub(super) fn prompt_info(&self) -> pb::PromptInfo {
        let wants_logprobs = self.response.prompt_logprobs;
        let wants_tokens = self.response.prompt_token_ids || wants_logprobs;
        let token_ids = if wants_tokens {
            self.prompt_tokens.clone()
        } else {
            Vec::new()
        };
        let (logprobs, ranks, candidate_tokens) = if wants_logprobs {
            let rows = self
                .prompt_tokens
                .iter()
                .enumerate()
                .map(|(index, token)| {
                    if index == 0 {
                        (0.0, 0, pb::CandidateTokenInfo::default())
                    } else {
                        (
                            selected_logprob(*token),
                            1,
                            candidate_info(*token, self.response.prompt_candidates.as_ref()),
                        )
                    }
                })
                .collect::<Vec<_>>();
            unzip_rows(rows)
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };
        pb::PromptInfo {
            num_prompt_tokens: self.prompt_tokens.len() as u32,
            token_ids,
            logprobs,
            ranks,
            candidate_tokens,
        }
    }

    pub(super) fn sequence_output(&self, token_ids: &[u32], terminal: bool) -> pb::SequenceOutput {
        let wants_logprobs = self.response.output_logprobs;
        let output_ids = if self.response.output_token_ids {
            token_ids.to_vec()
        } else {
            Vec::new()
        };
        let (logprobs, ranks, candidate_tokens) = if wants_logprobs {
            unzip_rows(
                token_ids
                    .iter()
                    .map(|token| {
                        (
                            selected_logprob(*token),
                            1,
                            candidate_info(*token, self.response.output_candidates.as_ref()),
                        )
                    })
                    .collect(),
            )
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };
        let text = if self.response.output_text.unwrap_or(true) {
            token_ids
                .iter()
                .map(|token| format!("<token:{token}>"))
                .collect::<String>()
        } else {
            String::new()
        };
        pb::SequenceOutput {
            index: 0,
            text,
            num_tokens: token_ids.len() as u32,
            token_ids: output_ids,
            logprobs,
            ranks,
            candidate_tokens,
            finish_info: terminal.then(|| pb::FinishInfo {
                num_output_tokens: token_ids.len() as u32,
                finish_reason: pb::finish_info::FinishReason::Length as i32,
                stop_reason: None,
                kv_transfer_params: (self.mode == ServerMode::Prefill).then(|| self.handoff()),
            }),
        }
    }

    pub(super) fn handoff(&self) -> Struct {
        let remote_block_ids = self
            .prompt_tokens
            .chunks(64)
            .enumerate()
            .map(|(index, _)| Value {
                kind: Some(Kind::NumberValue(index as f64)),
            })
            .collect();
        Struct {
            fields: BTreeMap::from([
                ("do_remote_decode".to_string(), bool_value(false)),
                ("do_remote_prefill".to_string(), bool_value(true)),
                (
                    "remote_engine_id".to_string(),
                    string_value(format!("mocker-prefill-{}", self.uuid)),
                ),
                ("remote_host".to_string(), string_value("127.0.0.1")),
                ("remote_port".to_string(), number_value(0.0)),
                (
                    "remote_block_ids".to_string(),
                    Value {
                        kind: Some(Kind::ListValue(ListValue {
                            values: remote_block_ids,
                        })),
                    },
                ),
                (
                    HANDOFF_SENTINEL_FIELD.to_string(),
                    string_value(self.request_id.clone()),
                ),
            ]),
        }
    }
}

impl KvTransferRole {
    fn classify(params: Option<&Struct>) -> BoxedStatusResult<Self> {
        let Some(params) = params else {
            return Ok(Self::Aggregated);
        };
        let remote_decode = optional_bool(params, "do_remote_decode")?;
        let remote_prefill = optional_bool(params, "do_remote_prefill")?;

        match (remote_decode, remote_prefill) {
            (Some(true), Some(true)) => Err(Status::invalid_argument(
                "KV transfer payload cannot request both remote prefill and remote decode",
            )
            .into()),
            (Some(true), _) => {
                if DECODE_RENDEZVOUS_FIELDS
                    .iter()
                    .any(|key| params.fields.contains_key(*key))
                {
                    return Err(Status::invalid_argument(
                        "prefill KV transfer payload must not include decode rendezvous fields",
                    )
                    .into());
                }
                Ok(Self::Prefill)
            }
            (_, Some(true)) => {
                require_string(params, "remote_engine_id")?;
                require_string(params, "remote_host")?;
                require_port(params, "remote_port")?;
                require_block_ids(params, "remote_block_ids")?;
                // The prefill role always stamps this opaque sentinel; requiring
                // it here proves the sidecar forwarded the handoff verbatim
                // rather than silently dropping non-rendezvous fields.
                require_string(params, HANDOFF_SENTINEL_FIELD)?;
                Ok(Self::Decode)
            }
            _ => {
                if DECODE_RENDEZVOUS_FIELDS
                    .iter()
                    .any(|key| params.fields.contains_key(*key))
                {
                    return Err(Status::invalid_argument(
                        "KV rendezvous fields require do_remote_prefill=true",
                    )
                    .into());
                }
                Ok(Self::Aggregated)
            }
        }
    }
}

fn stable_uuid(seed: u64, request_id: &str) -> Uuid {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&seed.to_le_bytes());
    hasher.update(request_id.as_bytes());
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&hasher.finalize().as_bytes()[..16]);
    // Mark the stable digest as an RFC 4122 variant/version-4 UUID. It remains
    // deterministic; these bits only make diagnostics parse cleanly.
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    Uuid::from_bytes(bytes)
}

fn synthetic_token_seed(seed: u64, request_id: &str) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&seed.to_le_bytes());
    hasher.update(request_id.as_bytes());
    let mut seed_bytes = [0u8; 8];
    seed_bytes.copy_from_slice(&hasher.finalize().as_bytes()[..8]);
    u64::from_le_bytes(seed_bytes)
}

fn deterministic_token_id(seed: u64, position: usize) -> u32 {
    let mut value = seed.wrapping_add((position as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    1_000 + ((value ^ (value >> 31)) as u32 % 31_000)
}

fn selected_logprob(token_id: u32) -> f32 {
    -0.1 * ((token_id % 10) + 1) as f32
}

fn candidate_info(selected: u32, request: Option<&pb::CandidateTokens>) -> pb::CandidateTokenInfo {
    let ids: Vec<u32> = match request.and_then(|request| request.select.as_ref()) {
        None => Vec::new(),
        Some(pb::candidate_tokens::Select::TopN(count)) => (1..=(*count as usize)
            .min(MAX_CANDIDATES))
            .map(|offset| selected.wrapping_add(offset as u32))
            .collect(),
        Some(pb::candidate_tokens::Select::TokenIds(ids)) => {
            ids.ids.iter().copied().take(MAX_CANDIDATES).collect()
        }
        Some(pb::candidate_tokens::Select::All(true)) => (1..=MAX_CANDIDATES)
            .map(|offset| selected.wrapping_add(offset as u32))
            .collect(),
        Some(pb::candidate_tokens::Select::All(false)) => Vec::new(),
    };
    pb::CandidateTokenInfo {
        tokens: ids
            .into_iter()
            .enumerate()
            .map(|(index, id)| pb::candidate_token_info::TokenInfo {
                id,
                logprob: selected_logprob(selected) - 0.1 * (index + 1) as f32,
                rank: index as u32 + 2,
            })
            .collect(),
    }
}

fn unzip_rows(
    rows: Vec<(f32, u32, pb::CandidateTokenInfo)>,
) -> (Vec<f32>, Vec<u32>, Vec<pb::CandidateTokenInfo>) {
    let mut logprobs = Vec::with_capacity(rows.len());
    let mut ranks = Vec::with_capacity(rows.len());
    let mut candidates = Vec::with_capacity(rows.len());
    for (logprob, rank, candidate) in rows {
        logprobs.push(logprob);
        ranks.push(rank);
        candidates.push(candidate);
    }
    (logprobs, ranks, candidates)
}

fn optional_bool(value: &Struct, key: &str) -> BoxedStatusResult<Option<bool>> {
    match value.fields.get(key) {
        None => Ok(None),
        Some(Value {
            kind: Some(Kind::BoolValue(value)),
        }) => Ok(Some(*value)),
        Some(_) => Err(Status::invalid_argument(format!(
            "KV transfer field '{key}' must be a boolean",
        ))
        .into()),
    }
}

fn require_string<'a>(value: &'a Struct, key: &str) -> BoxedStatusResult<&'a str> {
    match value.fields.get(key) {
        Some(Value {
            kind: Some(Kind::StringValue(value)),
        }) if !value.is_empty() => Ok(value),
        _ => Err(Status::invalid_argument(format!(
            "decode KV transfer field '{key}' must be a non-empty string",
        ))
        .into()),
    }
}

fn require_port(value: &Struct, key: &str) -> BoxedStatusResult<u16> {
    match value.fields.get(key) {
        Some(Value {
            kind: Some(Kind::NumberValue(value)),
        }) if value.is_finite()
            && value.fract() == 0.0
            && *value >= 0.0
            && *value <= f64::from(u16::MAX) =>
        {
            Ok(*value as u16)
        }
        _ => Err(Status::invalid_argument(format!(
            "decode KV transfer field '{key}' must be an integer port",
        ))
        .into()),
    }
}

fn require_block_ids(value: &Struct, key: &str) -> BoxedStatusResult<()> {
    match value.fields.get(key) {
        Some(Value {
            kind: Some(Kind::ListValue(values)),
        }) if values.values.iter().all(|value| {
            matches!(
                &value.kind,
                Some(Kind::NumberValue(id))
                    if id.is_finite() && id.fract() == 0.0 && *id >= 0.0
            )
        }) =>
        {
            Ok(())
        }
        _ => Err(Status::invalid_argument(format!(
            "decode KV transfer field '{key}' must be a list of non-negative integer IDs",
        ))
        .into()),
    }
}

pub(super) fn bool_value(value: bool) -> Value {
    Value {
        kind: Some(Kind::BoolValue(value)),
    }
}

fn string_value(value: impl Into<String>) -> Value {
    Value {
        kind: Some(Kind::StringValue(value.into())),
    }
}

pub(super) fn number_value(value: f64) -> Value {
    Value {
        kind: Some(Kind::NumberValue(value)),
    }
}
