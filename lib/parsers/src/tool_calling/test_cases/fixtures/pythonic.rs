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
                Value::Bool(true) => "True".to_string(),
                Value::Bool(false) => "False".to_string(),
                Value::Null => "None".to_string(),
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
}
