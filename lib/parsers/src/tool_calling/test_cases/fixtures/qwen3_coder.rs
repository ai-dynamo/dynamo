// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::{FixtureCase, ToolCallFixture};
use serde_json::Value;

/// Qwen3-Coder — default `XmlParserConfig`.
///
/// `<tool_call><function=NAME><parameter=K>V</parameter>...</function></tool_call>`
pub struct Qwen3CoderFixture;

fn render_body(function_name: &str, arguments: &Value) -> String {
    let mut params = String::new();
    if let Some(map) = arguments.as_object() {
        for (k, v) in map {
            let v_str = match v {
                Value::String(s) => s.clone(),
                _ => v.to_string(),
            };
            params.push_str(&format!("<parameter={k}>\n{v_str}\n</parameter>\n"));
        }
    }
    format!("<function={function_name}>\n{params}</function>")
}

impl ToolCallFixture for Qwen3CoderFixture {
    fn parser_name(&self) -> &'static str {
        "qwen3_coder"
    }

    fn case_1_single_call(&self, function_name: &str, arguments: &Value) -> FixtureCase<String> {
        let body = render_body(function_name, arguments);
        FixtureCase::Sample(format!("<tool_call>\n{body}\n</tool_call>"))
    }

    fn case_5_missing_end_token_recovery(
        &self,
        function_name: &str,
        arguments: &Value,
    ) -> FixtureCase<String> {
        let body = render_body(function_name, arguments);
        // Drop </tool_call>; the inner </function> stays present so the call
        // is logically complete, only the outer wrapper is missing.
        FixtureCase::KnownBroken {
            input: format!("<tool_call>\n{body}"),
            reason: "qwen3_coder XML parser has no missing-end recovery yet; follow-up to generalize PR #8208.",
        }
    }
}
