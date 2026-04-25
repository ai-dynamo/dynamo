// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::{FixtureCase, ToolCallFixture};
use serde_json::Value;

/// Jamba — `<tool_calls>{"name":"NAME","arguments":{...}}</tool_calls>`
pub struct JambaFixture;

impl ToolCallFixture for JambaFixture {
    fn parser_name(&self) -> &'static str {
        "jamba"
    }

    fn case_1_single_call(&self, function_name: &str, arguments: &Value) -> FixtureCase<String> {
        FixtureCase::Sample(format!(
            "<tool_calls>{{\"name\":\"{function_name}\",\"arguments\":{arguments}}}</tool_calls>"
        ))
    }

    fn case_5_missing_end_token_recovery(
        &self,
        function_name: &str,
        arguments: &Value,
    ) -> FixtureCase<String> {
        // Drop </tool_calls>; JSON body is complete.
        FixtureCase::KnownBroken {
            input: format!(
                "<tool_calls>{{\"name\":\"{function_name}\",\"arguments\":{arguments}}}"
            ),
            reason: "jamba has no missing-end recovery yet; follow-up to generalize PR #8208.",
        }
    }
}
