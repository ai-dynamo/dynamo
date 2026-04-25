// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::{FixtureCase, ToolCallFixture};
use serde_json::Value;

/// Phi-4 — `functools{"name":"NAME","arguments":{...}}` with no end marker.
pub struct Phi4Fixture;

impl ToolCallFixture for Phi4Fixture {
    fn parser_name(&self) -> &'static str {
        "phi4"
    }

    fn case_1_single_call(&self, function_name: &str, arguments: &Value) -> FixtureCase<String> {
        FixtureCase::Sample(format!(
            "functools{{\"name\":\"{function_name}\",\"arguments\":{arguments}}}"
        ))
    }

    fn case_5_missing_end_token_recovery(
        &self,
        _function_name: &str,
        _arguments: &Value,
    ) -> FixtureCase<String> {
        FixtureCase::NotApplicable(
            "Phi-4 has no end token (config end-token is empty string); the \
             `functools` sentinel opens a JSON object and EOF terminates it. \
             CASE.4 covers truncation of the JSON itself.",
        )
    }
}
