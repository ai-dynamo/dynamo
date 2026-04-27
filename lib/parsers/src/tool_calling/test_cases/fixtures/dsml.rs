// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::{FixtureCase, ToolCallFixture};
use serde_json::Value;

/// Family fixture for the DeepSeek DSML grammar (V3.2 + V4). Both
/// variants share `<｜DSML｜invoke name="NAME">...</｜DSML｜invoke>`
/// with `<｜DSML｜parameter name="K" string="...">V</｜DSML｜parameter>`
/// children; only the outer wrapper tag differs.
pub struct DsmlFixture {
    pub name: &'static str,
    pub outer_tag: &'static str, // e.g. "function_calls" (V3.2) or "tool_calls" (V4)
}

impl DsmlFixture {
    fn render_invoke(&self, function_name: &str, arguments: &Value) -> String {
        let mut params = String::new();
        if let Some(map) = arguments.as_object() {
            for (k, v) in map {
                let (is_string, v_str) = match v {
                    Value::String(s) => (true, s.clone()),
                    _ => (false, v.to_string()),
                };
                params.push_str(&format!(
                    "<｜DSML｜parameter name=\"{k}\" string=\"{is_string}\">{v_str}</｜DSML｜parameter>"
                ));
            }
        }
        format!("<｜DSML｜invoke name=\"{function_name}\">{params}</｜DSML｜invoke>")
    }
}

impl ToolCallFixture for DsmlFixture {
    fn parser_name(&self) -> &'static str {
        self.name
    }

    fn case_1_single_call(&self, function_name: &str, arguments: &Value) -> FixtureCase<String> {
        let invoke = self.render_invoke(function_name, arguments);
        FixtureCase::Sample(format!(
            "<｜DSML｜{}>{invoke}</｜DSML｜{}>",
            self.outer_tag, self.outer_tag
        ))
    }
}
