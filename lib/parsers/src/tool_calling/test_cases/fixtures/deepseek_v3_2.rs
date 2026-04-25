// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::{FixtureCase, ToolCallFixture};
use serde_json::Value;

/// DeepSeek V3.2 (DSML) — `<｜DSML｜function_calls>...</｜DSML｜function_calls>`.
///
/// Each call is `<｜DSML｜invoke name="NAME"><｜DSML｜parameter name="K" string="true">V</｜DSML｜parameter>...</｜DSML｜invoke>`.
pub struct DeepseekV32Fixture;

fn render_invoke(function_name: &str, arguments: &Value) -> String {
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

impl ToolCallFixture for DeepseekV32Fixture {
    fn parser_name(&self) -> &'static str {
        "deepseek_v3_2"
    }

    fn case_1_single_call(&self, function_name: &str, arguments: &Value) -> FixtureCase<String> {
        let invoke = render_invoke(function_name, arguments);
        FixtureCase::Sample(format!(
            "<｜DSML｜function_calls>{invoke}</｜DSML｜function_calls>"
        ))
    }

    fn case_5_missing_end_token_recovery(
        &self,
        function_name: &str,
        arguments: &Value,
    ) -> FixtureCase<String> {
        let invoke = render_invoke(function_name, arguments);
        FixtureCase::KnownBroken {
            input: format!("<｜DSML｜function_calls>{invoke}"),
            reason: "DSML parser has no missing-end recovery yet; follow-up to generalize PR #8208.",
        }
    }
}
