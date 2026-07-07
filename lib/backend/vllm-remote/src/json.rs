// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::DynamoError;
use prost_types::value::Kind;

use crate::client;

const MAX_EXACT_INTEGER: u64 = 1_u64 << 53;

pub(crate) fn json_to_struct(
    value: &serde_json::Value,
) -> Result<prost_types::Struct, DynamoError> {
    let serde_json::Value::Object(fields) = value else {
        return Err(client::invalid_argument(
            "kv_transfer_params must be a JSON object",
        ));
    };
    Ok(prost_types::Struct {
        fields: fields
            .iter()
            .map(|(key, value)| Ok((key.clone(), json_to_value(value)?)))
            .collect::<Result<_, DynamoError>>()?,
    })
}

fn json_to_value(value: &serde_json::Value) -> Result<prost_types::Value, DynamoError> {
    let kind = match value {
        serde_json::Value::Null => Kind::NullValue(prost_types::NullValue::NullValue as i32),
        serde_json::Value::Bool(value) => Kind::BoolValue(*value),
        serde_json::Value::String(value) => Kind::StringValue(value.clone()),
        serde_json::Value::Number(value) => Kind::NumberValue(number_to_f64(value)?),
        serde_json::Value::Array(values) => Kind::ListValue(prost_types::ListValue {
            values: values
                .iter()
                .map(json_to_value)
                .collect::<Result<_, DynamoError>>()?,
        }),
        serde_json::Value::Object(values) => Kind::StructValue(prost_types::Struct {
            fields: values
                .iter()
                .map(|(key, value)| Ok((key.clone(), json_to_value(value)?)))
                .collect::<Result<_, DynamoError>>()?,
        }),
    };
    Ok(prost_types::Value { kind: Some(kind) })
}

fn number_to_f64(value: &serde_json::Number) -> Result<f64, DynamoError> {
    if let Some(value) = value.as_u64()
        && value > MAX_EXACT_INTEGER
    {
        return Err(client::invalid_argument(format!(
            "kv_transfer_params integer {value} cannot be represented exactly by protobuf Struct"
        )));
    }
    if let Some(value) = value.as_i64()
        && value.unsigned_abs() > MAX_EXACT_INTEGER
    {
        return Err(client::invalid_argument(format!(
            "kv_transfer_params integer {value} cannot be represented exactly by protobuf Struct"
        )));
    }
    value.as_f64().ok_or_else(|| {
        client::invalid_argument(format!(
            "kv_transfer_params number {value} cannot be represented by protobuf Struct"
        ))
    })
}

pub(crate) fn struct_to_json(
    value: &prost_types::Struct,
) -> Result<serde_json::Value, DynamoError> {
    Ok(serde_json::Value::Object(
        value
            .fields
            .iter()
            .map(|(key, value)| Ok((key.clone(), value_to_json(value)?)))
            .collect::<Result<_, DynamoError>>()?,
    ))
}

fn value_to_json(value: &prost_types::Value) -> Result<serde_json::Value, DynamoError> {
    match &value.kind {
        None | Some(Kind::NullValue(_)) => Ok(serde_json::Value::Null),
        Some(Kind::BoolValue(value)) => Ok(serde_json::Value::Bool(*value)),
        Some(Kind::StringValue(value)) => Ok(serde_json::Value::String(value.clone())),
        Some(Kind::NumberValue(value)) => {
            if !value.is_finite() {
                return Err(client::protocol_error(
                    "kv_transfer_params contains NaN or infinity",
                ));
            }
            let number = if value.fract() == 0.0 && value.abs() <= MAX_EXACT_INTEGER as f64 {
                if value.is_sign_negative() {
                    serde_json::Number::from(*value as i64)
                } else {
                    serde_json::Number::from(*value as u64)
                }
            } else {
                serde_json::Number::from_f64(*value).expect("finite f64 is a JSON number")
            };
            Ok(serde_json::Value::Number(number))
        }
        Some(Kind::ListValue(values)) => Ok(serde_json::Value::Array(
            values
                .values
                .iter()
                .map(value_to_json)
                .collect::<Result<_, DynamoError>>()?,
        )),
        Some(Kind::StructValue(value)) => struct_to_json(value),
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
        let encoded = json_to_struct(&payload).expect("encode");
        assert_eq!(struct_to_json(&encoded).expect("decode"), payload);
    }

    #[test]
    fn rejects_non_objects_and_inexact_integers() {
        assert!(json_to_struct(&json!([1, 2])).is_err());
        assert!(json_to_struct(&json!({"value": 9_007_199_254_740_993_u64})).is_err());
    }
}
