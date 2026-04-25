// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::{FixtureCase, ToolCallFixture};
use serde_json::Value;

/// DeepSeek V3 — token-wrapped JSON with code-fence body.
///
/// `<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME\n```json\n{...}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>`
pub struct DeepseekV3Fixture;

impl ToolCallFixture for DeepseekV3Fixture {
    fn parser_name(&self) -> &'static str {
        "deepseek_v3"
    }

    fn case_1_single_call(&self, function_name: &str, arguments: &Value) -> FixtureCase<String> {
        FixtureCase::Sample(format!(
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>{function_name}\n\
             ```json\n{arguments}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
        ))
    }
}
