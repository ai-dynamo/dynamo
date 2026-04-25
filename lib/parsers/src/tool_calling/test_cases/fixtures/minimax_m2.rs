// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::{FixtureCase, ToolCallFixture};
use serde_json::Value;

/// MiniMax M2 — generic XML with `<minimax:tool_call>` outer wrapper.
///
/// `<minimax:tool_call><invoke name="NAME"><parameter name="K">V</parameter>...</invoke></minimax:tool_call>`
pub struct MinimaxM2Fixture;

fn render_body(function_name: &str, arguments: &Value) -> String {
    let mut params = String::new();
    if let Some(map) = arguments.as_object() {
        for (k, v) in map {
            let v_str = match v {
                Value::String(s) => s.clone(),
                _ => v.to_string(),
            };
            params.push_str(&format!("<parameter name=\"{k}\">{v_str}</parameter>"));
        }
    }
    format!("<invoke name=\"{function_name}\">{params}</invoke>")
}

impl ToolCallFixture for MinimaxM2Fixture {
    fn parser_name(&self) -> &'static str {
        "minimax_m2"
    }

    fn case_1_single_call(&self, function_name: &str, arguments: &Value) -> FixtureCase<String> {
        let body = render_body(function_name, arguments);
        FixtureCase::Sample(format!("<minimax:tool_call>{body}</minimax:tool_call>"))
    }

    fn case_5_missing_end_token_recovery(
        &self,
        function_name: &str,
        arguments: &Value,
    ) -> FixtureCase<String> {
        // Drop the closing </minimax:tool_call> wrapper.
        let body = render_body(function_name, arguments);
        FixtureCase::KnownBroken {
            input: format!("<minimax:tool_call>{body}"),
            reason: "minimax_m2 has no missing-end recovery yet; follow-up to generalize PR #8208.",
        }
    }
}
