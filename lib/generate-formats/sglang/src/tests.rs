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
    let mut bound = (opened.bind)(&crate::testkit::worker()).unwrap();
    assert!(
        bound
            .request
            .annotations
            .iter()
            .any(|a| a == crate::PRIVATE_RID_ANNOTATION)
    );
    assert_eq!(bound.request.token_ids.as_slice(), &[1]);
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
