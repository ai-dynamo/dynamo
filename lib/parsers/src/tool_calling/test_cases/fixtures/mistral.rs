// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::{FixtureCase, ToolCallFixture};
use serde_json::Value;

/// Mistral — `[TOOL_CALLS]{"name":"NAME","arguments":{...}}[/TOOL_CALLS]`
pub struct MistralFixture;

impl ToolCallFixture for MistralFixture {
    fn parser_name(&self) -> &'static str {
        "mistral"
    }

    fn case_1_single_call(&self, function_name: &str, arguments: &Value) -> FixtureCase<String> {
        FixtureCase::Sample(format!(
            "[TOOL_CALLS]{{\"name\":\"{function_name}\",\"arguments\":{arguments}}}[/TOOL_CALLS]"
        ))
    }

    fn case_5_missing_end_token_recovery(
        &self,
        function_name: &str,
        arguments: &Value,
    ) -> FixtureCase<String> {
        FixtureCase::Sample(format!(
            "[TOOL_CALLS]{{\"name\":\"{function_name}\",\"arguments\":{arguments}}}"
        ))
    }
}
