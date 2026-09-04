// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `google.protobuf.Struct` <-> `serde_json::Value` conversion.
//!
//! Engine gRPC contracts carry their opaque, engine-specific payloads (vLLM's
//! `kv_transfer_params`, the OpenEngine `KvSessionRef.attributes_struct`) as a
//! `Struct`, while Dynamo carries them as JSON. Sidecars share this codec so a
//! fix to the number or null handling lands for all of them at once.
//!
//! `what` names the payload in error messages; `peer` names the engine whose
//! response failed to decode.

use dynamo_backend_common::DynamoError;
use prost_types::value::Kind;

use crate::error::{invalid_argument, protocol_error};

/// A `Struct` number is an IEEE-754 double, so integers above 2^53 cannot
/// round-trip exactly. Reject them rather than silently truncate.
const MAX_EXACT_INTEGER: u64 = 1_u64 << 53;

/// Encodes a JSON object as a protobuf `Struct`.
pub fn json_to_struct(
    value: serde_json::Value,
    what: &str,
) -> Result<prost_types::Struct, DynamoError> {
    let serde_json::Value::Object(fields) = value else {
        return Err(invalid_argument(format!("{what} must be a JSON object")));
    };
    Ok(prost_types::Struct {
        fields: fields
            .into_iter()
            .map(|(key, value)| Ok((key, json_to_value(value, what)?)))
            .collect::<Result<_, DynamoError>>()?,
    })
}

fn json_to_value(value: serde_json::Value, what: &str) -> Result<prost_types::Value, DynamoError> {
    let kind = match value {
        serde_json::Value::Null => Kind::NullValue(prost_types::NullValue::NullValue as i32),
        serde_json::Value::Bool(value) => Kind::BoolValue(value),
        serde_json::Value::String(value) => Kind::StringValue(value),
        serde_json::Value::Number(value) => Kind::NumberValue(number_to_f64(&value, what)?),
        serde_json::Value::Array(values) => Kind::ListValue(prost_types::ListValue {
            values: values
                .into_iter()
                .map(|value| json_to_value(value, what))
                .collect::<Result<_, DynamoError>>()?,
        }),
        serde_json::Value::Object(values) => Kind::StructValue(prost_types::Struct {
            fields: values
                .into_iter()
                .map(|(key, value)| Ok((key, json_to_value(value, what)?)))
                .collect::<Result<_, DynamoError>>()?,
        }),
    };
    Ok(prost_types::Value { kind: Some(kind) })
}

fn number_to_f64(value: &serde_json::Number, what: &str) -> Result<f64, DynamoError> {
    if let Some(value) = value.as_u64()
        && value > MAX_EXACT_INTEGER
    {
        return Err(invalid_argument(format!(
            "{what} integer {value} cannot be represented exactly by protobuf Struct"
        )));
    }
    if let Some(value) = value.as_i64()
        && value.unsigned_abs() > MAX_EXACT_INTEGER
    {
        return Err(invalid_argument(format!(
            "{what} integer {value} cannot be represented exactly by protobuf Struct"
        )));
    }
    value.as_f64().ok_or_else(|| {
        invalid_argument(format!(
            "{what} number {value} cannot be represented by protobuf Struct"
        ))
    })
}

/// Decodes a protobuf `Struct` from `peer` into a JSON object.
pub fn struct_to_json(
    value: prost_types::Struct,
    peer: &str,
    what: &str,
) -> Result<serde_json::Value, DynamoError> {
    Ok(serde_json::Value::Object(
        value
            .fields
            .into_iter()
            .map(|(key, value)| Ok((key, value_to_json(value, peer, what)?)))
            .collect::<Result<_, DynamoError>>()?,
    ))
}

fn value_to_json(
    value: prost_types::Value,
    peer: &str,
    what: &str,
) -> Result<serde_json::Value, DynamoError> {
    match value.kind {
        None | Some(Kind::NullValue(_)) => Ok(serde_json::Value::Null),
        Some(Kind::BoolValue(value)) => Ok(serde_json::Value::Bool(value)),
        Some(Kind::StringValue(value)) => Ok(serde_json::Value::String(value)),
        Some(Kind::NumberValue(value)) => {
            if !value.is_finite() {
                return Err(protocol_error(
                    peer,
                    format!("{what} contains NaN or infinity"),
                ));
            }
            // Render whole doubles as JSON integers so a round-trip does not
            // turn `dp_rank: 0` into `0.0`.
            let number = if value.fract() == 0.0 && value.abs() <= MAX_EXACT_INTEGER as f64 {
                if value.is_sign_negative() {
                    serde_json::Number::from(value as i64)
                } else {
                    serde_json::Number::from(value as u64)
                }
            } else {
                serde_json::Number::from_f64(value).expect("finite f64 is a JSON number")
            };
            Ok(serde_json::Value::Number(number))
        }
        Some(Kind::ListValue(values)) => Ok(serde_json::Value::Array(
            values
                .values
                .into_iter()
                .map(|value| value_to_json(value, peer, what))
                .collect::<Result<_, DynamoError>>()?,
        )),
        Some(Kind::StructValue(value)) => struct_to_json(value, peer, what),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{json_to_struct, struct_to_json};

    #[test]
    fn nested_payload_round_trips_without_shape_changes() {
        let payload = json!({
            "string": "value",
            "bool": true,
            "number": 42,
            "null": null,
            "list": [1, "two", false, {"nested": 3.5}],
        });
        let encoded = json_to_struct(payload.clone(), "payload").expect("encode");
        assert_eq!(
            struct_to_json(encoded, "test-engine", "payload").expect("decode"),
            payload
        );
    }

    #[test]
    fn rejects_non_objects_and_inexact_integers() {
        assert!(json_to_struct(json!([1, 2]), "payload").is_err());
        assert!(json_to_struct(json!({"value": 9_007_199_254_740_993_u64}), "payload").is_err());
    }
}
