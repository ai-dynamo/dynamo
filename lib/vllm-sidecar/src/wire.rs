// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV-session encoding for the private vLLM engine RPC contract.

use dynamo_backend_common::DynamoError;

use crate::client;
use crate::proto as pb;

pub(crate) enum GenerateEvent {
    Token(pb::TokenOutput),
    Finished(pb::FinishReason),
    PrefillReady(serde_json::Value),
    Error(pb::EngineError),
}

pub(crate) struct ValidatedGenerateResponse {
    pub event: GenerateEvent,
    pub usage: Option<pb::Usage>,
}

/// Validate one streamed response before exposing it to Dynamo.
pub(crate) fn validate_generate_response(
    response: pb::GenerateResponse,
    expected_request_id: &str,
    is_prefill: bool,
) -> Result<ValidatedGenerateResponse, DynamoError> {
    if response.request_id != expected_request_id {
        return Err(client::protocol_error(format!(
            "request_id `{}` does not match `{expected_request_id}`",
            response.request_id
        )));
    }

    let event = match response.event {
        Some(pb::generate_response::Event::Token(token)) => GenerateEvent::Token(token),
        Some(pb::generate_response::Event::Error(error)) => GenerateEvent::Error(error),
        Some(pb::generate_response::Event::Finished(finished)) if !is_prefill => {
            let reason = pb::FinishReason::try_from(finished.reason).map_err(|_| {
                client::protocol_error(format!("unknown finish reason {}", finished.reason))
            })?;
            if reason == pb::FinishReason::Unspecified {
                return Err(client::protocol_error("finish reason is unspecified"));
            }
            GenerateEvent::Finished(reason)
        }
        Some(pb::generate_response::Event::PrefillReady(ready)) if is_prefill => {
            let session = ready
                .kv_session
                .ok_or_else(|| client::protocol_error("PrefillReady is missing kv_session"))?;
            GenerateEvent::PrefillReady(kv_session_to_disagg_json(session)?)
        }
        Some(pb::generate_response::Event::Finished(_)) => {
            return Err(client::protocol_error(
                "prefill worker returned Finished instead of PrefillReady",
            ));
        }
        Some(pb::generate_response::Event::PrefillReady(_)) => {
            return Err(client::protocol_error(
                "aggregated/decode worker returned PrefillReady",
            ));
        }
        None => return Err(client::protocol_error("event is missing")),
    };

    Ok(ValidatedGenerateResponse {
        event,
        usage: response.usage,
    })
}

/// Encode a prefill `KvSessionRef` into the JSON the frontend's PrefillRouter
/// forwards to the decode peer.
pub(crate) fn kv_session_to_disagg_json(
    session: pb::KvSessionRef,
) -> Result<serde_json::Value, DynamoError> {
    if session.session_id.trim().is_empty() {
        return Err(client::protocol_error("KvSessionRef.session_id is empty"));
    }
    if session.transfer_backend.trim().is_empty() {
        return Err(client::protocol_error(
            "KvSessionRef.transfer_backend is empty",
        ));
    }
    let attributes = session
        .attributes_struct
        .as_ref()
        .filter(|attributes| !attributes.fields.is_empty())
        .ok_or_else(|| {
            client::protocol_error("KvSessionRef.attributes_struct is missing or empty")
        })?;
    Ok(serde_json::json!({
        "session_id": session.session_id,
        "transfer_backend": session.transfer_backend,
        "dp_rank": session.dp_rank,
        "attributes_struct": prost_struct_to_json(attributes),
    }))
}

/// Reconstruct a `KvSessionRef` from the prefill peer's forwarded JSON.
pub(crate) fn disagg_json_to_kv_session(
    params: &serde_json::Value,
) -> Result<pb::KvSessionRef, DynamoError> {
    let obj = params.as_object().ok_or_else(|| {
        client::invalid_arg("prefill_result.disaggregated_params must be an object")
    })?;
    let required_string = |key: &str| {
        obj.get(key)
            .and_then(serde_json::Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .map(str::to_string)
            .ok_or_else(|| {
                client::invalid_arg(format!(
                    "prefill_result.disaggregated_params.{key} is required"
                ))
            })
    };
    let session_id = required_string("session_id")?;
    let transfer_backend = required_string("transfer_backend")?;
    let dp_rank = obj
        .get("dp_rank")
        .and_then(serde_json::Value::as_u64)
        .and_then(|rank| u32::try_from(rank).ok())
        .ok_or_else(|| {
            client::invalid_arg("prefill_result.disaggregated_params.dp_rank must be a uint32")
        })?;
    let attributes = obj
        .get("attributes_struct")
        .and_then(serde_json::Value::as_object)
        .filter(|attributes| !attributes.is_empty())
        .ok_or_else(|| {
            client::invalid_arg(
                "prefill_result.disaggregated_params.attributes_struct must be a non-empty object",
            )
        })?;
    let attributes_struct = json_to_prost_struct(&serde_json::Value::Object(attributes.clone()));
    Ok(pb::KvSessionRef {
        session_id,
        transfer_backend,
        dp_rank,
        attributes_struct,
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
