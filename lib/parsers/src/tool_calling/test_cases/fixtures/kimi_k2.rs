// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::{FixtureCase, ToolCallFixture};
use serde_json::Value;

/// Kimi K2 / K2.5 / K2.6 — special-token XML-ish format.
///
/// `<|tool_calls_section_begin|>`
///   `<|tool_call_begin|>functions.NAME:INDEX<|tool_call_argument_begin|>{json}<|tool_call_end|>`
/// `<|tool_calls_section_end|>`
pub struct KimiK2Fixture;

impl ToolCallFixture for KimiK2Fixture {
    fn parser_name(&self) -> &'static str {
        "kimi_k2"
    }

    fn case_1_single_call(&self, function_name: &str, arguments: &Value) -> FixtureCase<String> {
        FixtureCase::Sample(format!(
            "<|tool_calls_section_begin|>\
             <|tool_call_begin|>functions.{function_name}:0\
             <|tool_call_argument_begin|>{arguments}\
             <|tool_call_end|>\
             <|tool_calls_section_end|>"
        ))
    }
}
