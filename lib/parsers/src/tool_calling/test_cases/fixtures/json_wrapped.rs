// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::{FixtureCase, ToolCallFixture};
use serde_json::Value;

/// Family fixture for parsers that emit `{start}{json}{end}` — a JSON
/// object body sandwiched between two literal token strings. Differences
/// across parsers in this family are entirely the start/end strings.
pub struct JsonWrappedFixture {
    pub name: &'static str,
    pub start: &'static str,
    pub end: &'static str,
}

impl ToolCallFixture for JsonWrappedFixture {
    fn parser_name(&self) -> &'static str {
        self.name
    }

    fn case_1_single_call(&self, function_name: &str, arguments: &Value) -> FixtureCase<String> {
        FixtureCase::Sample(format!(
            "{}{{\"name\":\"{function_name}\",\"arguments\":{arguments}}}{}",
            self.start, self.end
        ))
    }
}
