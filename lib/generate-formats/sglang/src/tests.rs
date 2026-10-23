// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{ClientCodec, Encoded, SglangFormat};
use axum::http::{HeaderMap, StatusCode};
use dynamo_llm::generate_format::*;
use serde_json::{Value, json};
use std::sync::Arc;

struct Wrapped;
impl ClientCodec for Wrapped {
    fn decode_request(&self, body: &[u8]) -> Result<Option<Value>, ClientError> {
        let value: Value =
            serde_json::from_slice(body).map_err(|error| ClientError::new(error.to_string()))?;
        Ok(Some(value["request"].clone()))
    }
    fn encode_response(
        &self,
        response: &Value,
    ) -> Result<Encoded, dynamo_runtime::error::DynamoError> {
        Ok(Encoded::Replace(json!({"response": response})))
    }
    fn error_body(&self, message: &str) -> Value {
        json!({"failure": message})
    }
}

#[test]
fn catalog_format_owns_client_shape_and_reuses_native_exchange() {
    let mut registry = Registry::default();
    registry.register(Arc::new(SglangFormat::native())).unwrap();
    registry
        .register(Arc::new(SglangFormat::with_codec("wrapped", Wrapped)))
        .unwrap();
    assert!(registry.register(Arc::new(SglangFormat::native())).is_err());
    assert!(registry.get("missing").is_err());
    let format = registry.get("wrapped").unwrap();
    assert_eq!(format.capability(), crate::CAPABILITY);
    assert_eq!(
        format.error_body(StatusCode::BAD_REQUEST, "bad"),
        json!({"failure": "bad"})
    );
    let opened = format
        .open(ClientRequest {
            body: br#"{"request":{"input_ids":[1],"sampling_params":{"max_new_tokens":2}}}"#,
            headers: &HeaderMap::new(),
            default_request_id: "public",
        })
        .unwrap();
    let mut bound = (opened.bind)(&crate::testkit::worker(&json!({}))).unwrap();
    assert_eq!(bound.request.token_ids.as_slice(), &[1]);
    assert!(bound.migration.is_ok());
    for (ids, finish) in [
        (json!([2]), Value::Null),
        (json!([3]), json!({"type":"length"})),
    ] {
        let output = crate::inspect(
            json!({"output_ids": ids, "meta_info": {
                "id": "private", "finish_reason": finish, "custom_metadata": {"score": 7},
            }}),
            false,
        )
        .unwrap()
        .output;
        bound.exchange.chunk(output, false, &mut |_| {}).unwrap();
    }
    let response = bound.exchange.finish().unwrap().unwrap();
    assert_eq!(response["response"]["output_ids"], json!([2, 3]));
    assert_eq!(response["response"]["meta_info"]["id"], "public");
    assert_eq!(
        response["response"]["meta_info"]["custom_metadata"]["score"],
        7
    );
}

#[test]
fn codec_extends_native_field_rules_and_joins_structured_sequences() {
    use crate::response_merge::{FieldClass, PrefillFlags, RuleSet};
    struct Structured;
    impl ClientCodec for Structured {
        fn rules(&self) -> Arc<RuleSet> {
            Arc::new(crate::rules().extend(
                [("/meta_info/custom_scores".into(), FieldClass::Delta)],
                PrefillFlags::default(),
            ))
        }
        fn overlay_sequence(&self, path: &str, prefix: &Value, target: &mut Value) -> bool {
            if path != "/meta_info/routed_experts" {
                return crate::Native.overlay_sequence(path, prefix, target);
            }
            let (Some(prefix), Some(target)) =
                (prefix["data"].as_array(), target["data"].as_array_mut())
            else {
                return false;
            };
            target.splice(..prefix.len().min(target.len()), prefix.iter().cloned());
            true
        }
    }
    let format = SglangFormat::with_codec("structured", Structured);
    let opened = format
        .open(ClientRequest {
            body: br#"{"input_ids":[1],"return_routed_experts":true}"#,
            headers: &HeaderMap::from_iter([(
                axum::http::header::CONTENT_TYPE,
                axum::http::HeaderValue::from_static("application/json"),
            )]),
            default_request_id: "public",
        })
        .unwrap();
    let mut bound = (opened.bind)(&crate::testkit::worker(&serde_json::Value::Null)).unwrap();
    assert!(
        bound
            .request
            .annotations
            .iter()
            .any(|a| a == dynamo_llm::protocols::response_merge::PREFILL_OUTPUT_ANNOTATION)
    );
    for (response, prefill) in [
        (
            json!({"meta_info":{"routed_experts":{"data":[1],"dtype":"u8"}}}),
            true,
        ),
        (
            json!({"output_ids":[2],"meta_info":{"custom_scores":[0.25]}}),
            false,
        ),
        (
            json!({"output_ids":[3],"meta_info":{"custom_scores":[0.5],"routed_experts":{"data":[0,2,3],"dtype":"u8"}}}),
            false,
        ),
    ] {
        bound
            .exchange
            .chunk(
                crate::inspect(response, prefill).unwrap().output,
                prefill,
                &mut |_| {},
            )
            .unwrap();
    }
    let result = bound.exchange.finish().unwrap().unwrap();
    assert_eq!(result["output_ids"], json!([2, 3]));
    assert_eq!(result["meta_info"]["custom_scores"], json!([0.25, 0.5]));
    assert_eq!(
        result["meta_info"]["routed_experts"],
        json!({"data":[1,2,3],"dtype":"u8"})
    );
}
