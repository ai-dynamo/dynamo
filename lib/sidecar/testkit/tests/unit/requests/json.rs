// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
    let encoded = json_to_struct(payload.clone()).expect("encode");
    assert_eq!(struct_to_json(encoded).expect("decode"), payload);
}

#[test]
fn rejects_non_objects_and_inexact_integers() {
    assert!(json_to_struct(json!([1, 2])).is_err());
    assert!(json_to_struct(json!({"value": 9_007_199_254_740_993_u64})).is_err());
}

#[test]
fn integer_boundaries_and_negative_inexact_values_are_checked() {
    for number in [-(1_i64 << 53), -1, 0, 1, 1_i64 << 53] {
        let value = json!({"nested": [number, {"fraction": -3.25}]});
        assert_eq!(
            struct_to_json(json_to_struct(value.clone()).unwrap()).unwrap(),
            value
        );
    }
    for number in [i64::MIN, -(1_i64 << 53) - 1, (1_i64 << 53) + 1, i64::MAX] {
        let error = json_to_struct(json!({"nested": [number]})).unwrap_err();
        assert_eq!(
            error.error_type(),
            dynamo_backend_common::ErrorType::Backend(
                dynamo_backend_common::BackendError::InvalidArgument
            )
        );
        assert!(error.to_string().contains("cannot be represented exactly"));
    }
}

#[test]
fn incoming_nonfinite_numbers_fail_inside_nested_payloads() {
    use super::{Kind, prost_types};
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let wire = prost_types::Struct {
            fields: [(
                "nested".into(),
                prost_types::Value {
                    kind: Some(Kind::ListValue(prost_types::ListValue {
                        values: vec![prost_types::Value {
                            kind: Some(Kind::NumberValue(value)),
                        }],
                    })),
                },
            )]
            .into_iter()
            .collect(),
        };
        let error = struct_to_json(wire).unwrap_err();
        assert_eq!(
            error.error_type(),
            dynamo_backend_common::ErrorType::Backend(dynamo_backend_common::BackendError::Unknown)
        );
        assert!(error.to_string().contains("NaN or infinity"));
    }
}
