// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::{FixtureCase, ToolCallFixture};
use serde_json::Value;

/// Llama-3 JSON — `<|python_tag|>{"name":"NAME","arguments":{...}}` with no end marker.
pub struct Llama3JsonFixture;

impl ToolCallFixture for Llama3JsonFixture {
    fn parser_name(&self) -> &'static str {
        "llama3_json"
    }

    fn case_1_single_call(&self, function_name: &str, arguments: &Value) -> FixtureCase<String> {
        FixtureCase::Sample(format!(
            "<|python_tag|>{{\"name\":\"{function_name}\",\"arguments\":{arguments}}}"
        ))
    }

    fn case_5_missing_end_token_recovery(
        &self,
        _function_name: &str,
        _arguments: &Value,
    ) -> FixtureCase<String> {
        FixtureCase::NotApplicable(
            "Llama3-JSON has no end token (config end-token is empty string); \
             the start sentinel `<|python_tag|>` opens a JSON object and EOF \
             terminates it. CASE.4 covers truncation of the JSON itself.",
        )
    }
}
