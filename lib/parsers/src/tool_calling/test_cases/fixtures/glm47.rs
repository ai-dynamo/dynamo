// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::{FixtureCase, ToolCallFixture};
use serde_json::Value;

/// GLM-4.7 — `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>`.
///
/// Each parameter is encoded as a flat alternating <arg_key>/<arg_value> pair,
/// not as nested XML elements. Schema-driven type coercion (string→number,
/// string→array) happens inside the parser; the fixture only emits string
/// values here. Schema-aware tests live in CASE.xml2 (deferred to part 2).
pub struct Glm47Fixture;

impl ToolCallFixture for Glm47Fixture {
    fn parser_name(&self) -> &'static str {
        "glm47"
    }

    fn case_1_single_call(&self, function_name: &str, arguments: &Value) -> FixtureCase<String> {
        let mut body = String::new();
        if let Some(map) = arguments.as_object() {
            for (k, v) in map {
                let v_str = match v {
                    Value::String(s) => s.clone(),
                    _ => v.to_string(),
                };
                body.push_str(&format!(
                    "<arg_key>{k}</arg_key><arg_value>{v_str}</arg_value>"
                ));
            }
        }
        FixtureCase::Sample(format!("<tool_call>{function_name}{body}</tool_call>"))
    }
}
