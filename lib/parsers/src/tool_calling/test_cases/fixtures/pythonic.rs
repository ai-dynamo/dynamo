// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::{FixtureCase, ToolCallFixture};
use serde_json::Value;

/// Pythonic — Python-function-call syntax: `[name(k=v, ...)]`.
pub struct PythonicFixture;

fn render_args(arguments: &Value) -> String {
    let Some(map) = arguments.as_object() else {
        return String::new();
    };
    map.iter()
        .map(|(k, v)| {
            let v_str = match v {
                Value::String(s) => format!("\"{s}\""),
                _ => v.to_string(),
            };
            format!("{k}={v_str}")
        })
        .collect::<Vec<_>>()
        .join(", ")
}

impl ToolCallFixture for PythonicFixture {
    fn parser_name(&self) -> &'static str {
        "pythonic"
    }

    fn case_1_single_call(&self, function_name: &str, arguments: &Value) -> FixtureCase<String> {
        FixtureCase::Sample(format!("[{function_name}({})]", render_args(arguments)))
    }

    fn case_5_missing_end_token_recovery(
        &self,
        _function_name: &str,
        _arguments: &Value,
    ) -> FixtureCase<String> {
        FixtureCase::NotApplicable(
            "Pythonic grammar has no separable section-end token. The closing `]` \
             is integral to the call list itself; a truncated `]` is CASE.4 \
             (malformed args), not section_end recovery.",
        )
    }
}
