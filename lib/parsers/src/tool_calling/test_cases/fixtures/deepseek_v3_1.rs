// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::{FixtureCase, ToolCallFixture};
use serde_json::Value;

/// DeepSeek V3.1.
///
/// Per `test_parse_tool_calls_deepseek_v3_1_basic`:
/// `<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>NAME<｜tool▁sep｜>{json}<｜tool▁call▁end｜><｜tool▁calls▁end｜>`
///
/// Per-call wrappers nest inside the section wrappers.
pub struct DeepseekV31Fixture;

impl ToolCallFixture for DeepseekV31Fixture {
    fn parser_name(&self) -> &'static str {
        "deepseek_v3_1"
    }

    fn case_1_single_call(&self, function_name: &str, arguments: &Value) -> FixtureCase<String> {
        FixtureCase::Sample(format!(
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>{function_name}<｜tool▁sep｜>{arguments}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
        ))
    }

    fn case_5_missing_end_token_recovery(
        &self,
        function_name: &str,
        arguments: &Value,
    ) -> FixtureCase<String> {
        // Per-call is complete (`<｜tool▁call▁end｜>` present); only the
        // outer section close `<｜tool▁calls▁end｜>` is missing — the
        // same condition that PR #8208 fixed for Kimi K2.
        FixtureCase::KnownBroken {
            input: format!(
                "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>{function_name}<｜tool▁sep｜>{arguments}<｜tool▁call▁end｜>"
            ),
            reason: "deepseek_v3_1 has no missing-section-end recovery yet; follow-up to generalize PR #8208.",
        }
    }
}
