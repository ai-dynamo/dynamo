// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Establish the Rust parsing expectations used by schema-fidelity checks.
//!
//! `fixtures/openapi/requests.json` supplies `body` and the expected Serde result
//! (`valid`). This test does not load or validate an OpenAPI document. The separate
//! offline composition suite consumes the same cases and their `schema_valid` / `gap`
//! metadata; known mismatches must not be mistaken for schema/Serde agreement.
//! Parsing success does not establish request validation, backend support, or inference.
//!
//! Run: `cargo test -p dynamo-llm --no-default-features --test openapi_request_fidelity`.

use dynamo_llm::protocols::openai::{
    chat_completions::NvCreateChatCompletionRequest, completions::NvCreateCompletionRequest,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    name: String,
    endpoint: String,
    valid: bool,
    body: serde_json::Value,
}

#[derive(Deserialize)]
struct Cases {
    cases: Vec<Case>,
}

#[test]
fn request_fixtures_match_rust_deserialization() {
    let cases: Cases =
        serde_json::from_str(include_str!("fixtures/openapi/requests.json")).unwrap();
    let mut failures = Vec::new();
    for case in cases.cases {
        let result = match case.endpoint.as_str() {
            "chat" => {
                serde_json::from_value::<NvCreateChatCompletionRequest>(case.body).map(|_| ())
            }
            "completion" => {
                serde_json::from_value::<NvCreateCompletionRequest>(case.body).map(|_| ())
            }
            endpoint => panic!("unknown fixture endpoint {endpoint}"),
        };
        if result.is_ok() != case.valid {
            failures.push(format!(
                "{}: expected {}, got {result:?}",
                case.name, case.valid
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
