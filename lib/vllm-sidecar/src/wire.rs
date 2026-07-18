// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! vLLM gRPC response validation and protobuf/JSON conversion.

use std::collections::HashSet;

use dynamo_backend_common::{DynamoError, TopLogprob, TopLogprobs};

use crate::client;
use crate::proto as pb;

const MIN_FINITE_LOGPROB: f64 = -1e30;

fn finite_logprob(value: f32) -> Result<f64, DynamoError> {
    if value.is_finite() {
        Ok(f64::from(value))
    } else if value == f32::NEG_INFINITY {
        Ok(MIN_FINITE_LOGPROB)
    } else {
        Err(client::protocol_error(
            "logprob metadata contains NaN or positive infinity",
        ))
    }
}

pub(crate) enum GenerateEvent {
    PromptLogprobs(serde_json::Value),
    Token {
        token_ids: Vec<u32>,
        logprobs: Option<Vec<f64>>,
        top_logprobs: Option<TopLogprobs>,
    },
    Finished(pb::finish_info::FinishReason),
    PrefillReady(serde_json::Value),
}

pub(crate) struct ValidatedGenerateResponse {
    pub prompt_tokens: Option<u32>,
    pub events: Vec<GenerateEvent>,
}

/// Validate one streamed response before exposing it to Dynamo.
pub(crate) fn validate_generate_response(
    response: pb::GenerateResponse,
    is_prefill: bool,
) -> Result<ValidatedGenerateResponse, DynamoError> {
    let mut events = Vec::new();
    let prompt_tokens = match response.prompt_info {
        Some(info) => {
            let prompt_tokens = info.num_prompt_tokens;
            if let Some(prompt_logprobs) = prompt_logprobs_to_json(&info)? {
                events.push(GenerateEvent::PromptLogprobs(prompt_logprobs));
            }
            Some(prompt_tokens)
        }
        None => None,
    };
    if let Some(output) = response.outputs {
        if output.index != 0 {
            return Err(client::protocol_error(
                "multiple output sequences are not supported",
            ));
        }
        let has_logprob_metadata = !output.logprobs.is_empty()
            || !output.ranks.is_empty()
            || !output.candidate_tokens.is_empty();
        let top_logprobs = if has_logprob_metadata {
            let position_count = output.token_ids.len();
            if output.logprobs.len() != position_count
                || output.ranks.len() != position_count
                || output.candidate_tokens.len() != position_count
            {
                return Err(client::protocol_error(
                    "token_ids, logprobs, ranks, and candidate_tokens are not positionally aligned",
                ));
            }
            let mut positions = Vec::with_capacity(position_count);
            for (((token_id, logprob), rank), candidates) in output
                .token_ids
                .iter()
                .zip(&output.logprobs)
                .zip(&output.ranks)
                .zip(&output.candidate_tokens)
            {
                let mut seen = HashSet::with_capacity(candidates.tokens.len() + 1);
                seen.insert(*token_id);
                let mut position = Vec::with_capacity(candidates.tokens.len() + 1);
                position.push(TopLogprob {
                    rank: *rank,
                    token_id: *token_id,
                    token: None,
                    logprob: finite_logprob(*logprob)?,
                    bytes: None,
                });
                for candidate in &candidates.tokens {
                    if !seen.insert(candidate.id) {
                        return Err(client::protocol_error(
                            "output logprob metadata contains duplicate token IDs at one position",
                        ));
                    }
                    position.push(TopLogprob {
                        rank: candidate.rank,
                        token_id: candidate.id,
                        token: None,
                        logprob: finite_logprob(candidate.logprob)?,
                        bytes: None,
                    });
                }
                positions.push(position);
            }
            Some(positions)
        } else {
            None
        };
        if !is_prefill && !output.token_ids.is_empty() {
            let logprobs = (!output.logprobs.is_empty()).then(|| {
                output
                    .logprobs
                    .into_iter()
                    .map(finite_logprob)
                    .collect::<Result<Vec<_>, _>>()
            });
            let logprobs = logprobs.transpose()?;
            events.push(GenerateEvent::Token {
                token_ids: output.token_ids,
                logprobs,
                top_logprobs,
            });
        }
        if let Some(finish) = output.finish_info {
            let reason =
                pb::finish_info::FinishReason::try_from(finish.finish_reason).map_err(|_| {
                    client::protocol_error(format!(
                        "unknown finish reason {}",
                        finish.finish_reason
                    ))
                })?;
            if reason == pb::finish_info::FinishReason::NotFinished {
                return Err(client::protocol_error("finish reason is not terminal"));
            }
            if is_prefill {
                let params = finish
                    .kv_transfer_params
                    .as_ref()
                    .filter(|params| !params.fields.is_empty())
                    .ok_or_else(|| {
                        client::protocol_error("prefill response is missing kv_transfer_params")
                    })?;
                events.push(GenerateEvent::PrefillReady(prost_struct_to_json(params)));
            } else {
                events.push(GenerateEvent::Finished(reason));
            }
        }
    }

    if prompt_tokens.is_none() && events.is_empty() {
        return Err(client::protocol_error(
            "response contains no prompt or output data",
        ));
    }

    Ok(ValidatedGenerateResponse {
        prompt_tokens,
        events,
    })
}

fn prompt_logprobs_to_json(
    info: &pb::PromptInfo,
) -> Result<Option<serde_json::Value>, DynamoError> {
    let has_logprob_metadata =
        !info.logprobs.is_empty() || !info.ranks.is_empty() || !info.candidate_tokens.is_empty();
    if !has_logprob_metadata {
        return Ok(None);
    }

    let position_count = info.num_prompt_tokens as usize;
    if info.token_ids.len() != position_count
        || info.logprobs.len() != position_count
        || info.ranks.len() != position_count
        || info.candidate_tokens.len() != position_count
    {
        return Err(client::protocol_error(
            "prompt token_ids, logprobs, ranks, and candidate_tokens are not positionally aligned",
        ));
    }
    if position_count == 0 {
        return Err(client::protocol_error(
            "prompt logprob metadata cannot describe an empty prompt",
        ));
    }
    if !info.candidate_tokens[0].tokens.is_empty() {
        return Err(client::protocol_error(
            "the first prompt token cannot have logprob candidates",
        ));
    }

    let mut positions = Vec::with_capacity(position_count);
    positions.push(serde_json::Value::Null);
    for position in 1..position_count {
        let selected_logprob = finite_logprob(info.logprobs[position])?;
        let mut entries = serde_json::Map::new();
        entries.insert(
            info.token_ids[position].to_string(),
            serde_json::json!({
                "logprob": selected_logprob,
                "rank": info.ranks[position],
            }),
        );
        for candidate in &info.candidate_tokens[position].tokens {
            if entries
                .insert(
                    candidate.id.to_string(),
                    serde_json::json!({
                        "logprob": finite_logprob(candidate.logprob)?,
                        "rank": candidate.rank,
                    }),
                )
                .is_some()
            {
                return Err(client::protocol_error(
                    "prompt logprob metadata contains duplicate token IDs at one position",
                ));
            }
        }
        positions.push(serde_json::Value::Object(entries));
    }
    Ok(Some(serde_json::Value::Array(positions)))
}

/// Convert a JSON object into a `google.protobuf.Struct`. Non-object inputs
/// yield `None`.
pub(crate) fn json_to_prost_struct(value: &serde_json::Value) -> Option<prost_types::Struct> {
    match value {
        serde_json::Value::Object(map) => Some(prost_types::Struct {
            fields: map
                .iter()
                .map(|(key, value)| (key.clone(), json_to_prost_value(value)))
                .collect(),
        }),
        _ => None,
    }
}

fn json_to_prost_value(value: &serde_json::Value) -> prost_types::Value {
    use prost_types::value::Kind;
    let kind = match value {
        serde_json::Value::Null => Kind::NullValue(prost_types::NullValue::NullValue as i32),
        serde_json::Value::Bool(value) => Kind::BoolValue(*value),
        serde_json::Value::Number(value) => Kind::NumberValue(value.as_f64().unwrap_or(0.0)),
        serde_json::Value::String(value) => Kind::StringValue(value.clone()),
        serde_json::Value::Array(values) => Kind::ListValue(prost_types::ListValue {
            values: values.iter().map(json_to_prost_value).collect(),
        }),
        serde_json::Value::Object(values) => Kind::StructValue(prost_types::Struct {
            fields: values
                .iter()
                .map(|(key, value)| (key.clone(), json_to_prost_value(value)))
                .collect(),
        }),
    };
    prost_types::Value { kind: Some(kind) }
}

/// Convert a `google.protobuf.Struct` back into a JSON object.
pub(crate) fn prost_struct_to_json(value: &prost_types::Struct) -> serde_json::Value {
    serde_json::Value::Object(
        value
            .fields
            .iter()
            .map(|(key, value)| (key.clone(), prost_value_to_json(value)))
            .collect(),
    )
}

fn prost_value_to_json(value: &prost_types::Value) -> serde_json::Value {
    use prost_types::value::Kind;
    match &value.kind {
        None | Some(Kind::NullValue(_)) => serde_json::Value::Null,
        Some(Kind::BoolValue(value)) => serde_json::Value::Bool(*value),
        Some(Kind::NumberValue(value)) => number_to_json(*value),
        Some(Kind::StringValue(value)) => serde_json::Value::String(value.clone()),
        Some(Kind::ListValue(values)) => {
            serde_json::Value::Array(values.values.iter().map(prost_value_to_json).collect())
        }
        Some(Kind::StructValue(value)) => prost_struct_to_json(value),
    }
}

fn number_to_json(value: f64) -> serde_json::Value {
    if value.is_finite()
        && value.fract() == 0.0
        && value >= i64::MIN as f64
        && value <= i64::MAX as f64
    {
        serde_json::Value::Number((value as i64).into())
    } else {
        serde_json::Number::from_f64(value)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null)
    }
}
