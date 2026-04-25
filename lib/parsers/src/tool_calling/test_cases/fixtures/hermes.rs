// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::{FixtureCase, ToolCallFixture};
use serde_json::Value;

/// Hermes — `<tool_call>{"name": "NAME", "arguments": {...}}\n</tool_call>`
pub struct HermesFixture;

impl ToolCallFixture for HermesFixture {
    fn parser_name(&self) -> &'static str {
        "hermes"
    }

    fn case_1_single_call(&self, function_name: &str, arguments: &Value) -> FixtureCase<String> {
        FixtureCase::Sample(format!(
            "<tool_call>{{\"name\":\"{function_name}\",\"arguments\":{arguments}}}\n</tool_call>"
        ))
    }

    fn case_5_missing_end_token_recovery(
        &self,
        function_name: &str,
        arguments: &Value,
    ) -> FixtureCase<String> {
        // Drop the closing </tool_call>; JSON body is complete.
        FixtureCase::KnownBroken {
            input: format!(
                "<tool_call>{{\"name\":\"{function_name}\",\"arguments\":{arguments}}}\n"
            ),
            reason: "hermes has no missing-end recovery yet; follow-up to generalize PR #8208.",
        }
    }
}
