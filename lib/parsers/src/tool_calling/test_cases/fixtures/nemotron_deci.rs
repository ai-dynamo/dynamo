// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::{FixtureCase, ToolCallFixture};
use serde_json::Value;

/// Nemotron — `<TOOLCALL>[{"name":"NAME","arguments":{...}}]</TOOLCALL>`
pub struct NemotronDeciFixture;

impl ToolCallFixture for NemotronDeciFixture {
    fn parser_name(&self) -> &'static str {
        "nemotron_deci"
    }

    fn case_1_single_call(&self, function_name: &str, arguments: &Value) -> FixtureCase<String> {
        FixtureCase::Sample(format!(
            "<TOOLCALL>[{{\"name\":\"{function_name}\",\"arguments\":{arguments}}}]</TOOLCALL>"
        ))
    }
}
