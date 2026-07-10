// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! vLLM gRPC response validation and protobuf/JSON conversion.

use dynamo_backend_common::DynamoError;

use crate::client;
use crate::proto as pb;

pub(crate) enum GenerateEvent {
    Token(Vec<u32>),
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
    let prompt_tokens = response.prompt_info.map(|info| info.num_prompt_tokens);
    let mut events = Vec::new();
    if let Some(output) = response.outputs {
        if output.index != 0 {
            return Err(client::protocol_error(
                "multiple output sequences are not supported",
            ));
        }
        if !is_prefill && !output.token_ids.is_empty() {
            events.push(GenerateEvent::Token(output.token_ids));
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
