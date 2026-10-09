// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Protect request deserialization acceptance and rejection against shared fixtures.
//!
//! `lib/llm/tests/fixtures/openapi/requests.json` supplies the expected Rust result
//! (`valid`). This suite does not load or validate an OpenAPI document. Its companion,
//! `scripts/protocol_compatibility/tests/composition/fidelity.py` (in the downstream
//! tooling layer), checks the composed schema against `schema_valid`, defaulting to
//! `valid`, with documented exceptions in `gap`.
//! Both suites must pass against the same source revision to establish agreement for
//! aligned examples. Passing known-gap cases reproduces documented disagreements;
//! it is not conformance. Neither suite proves HTTP admission, backend support,
//! inference, or Dynamo/framework parity.
//!
//! Run: `cargo test -p dynamo-llm --no-default-features --test protocols openapi_request_fidelity::`.

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
        serde_json::from_str(include_str!("../fixtures/openapi/requests.json")).unwrap();
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
