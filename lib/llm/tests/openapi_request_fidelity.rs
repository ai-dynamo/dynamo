// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared cases exercise real Serde, separately from JSON Schema validation.
//! Passing these cases is representative evidence, not behavioral conformance.

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
fn composed_schema_cases_match_rust_deserialization() {
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
