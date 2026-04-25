// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::{FixtureCase, ToolCallFixture};
use super::qwen3_coder::Qwen3CoderFixture;
use serde_json::Value;

/// Nemotron Nano — registry-aliased to the Qwen3-Coder XML format.
/// Fixture delegates so the matrix shows a separate row, but the format
/// stays in lockstep with the parser registry's actual mapping.
pub struct NemotronNanoFixture;

impl ToolCallFixture for NemotronNanoFixture {
    fn parser_name(&self) -> &'static str {
        "nemotron_nano"
    }

    fn case_1_single_call(&self, function_name: &str, arguments: &Value) -> FixtureCase<String> {
        Qwen3CoderFixture.case_1_single_call(function_name, arguments)
    }

    fn case_5_missing_end_token_recovery(
        &self,
        function_name: &str,
        arguments: &Value,
    ) -> FixtureCase<String> {
        Qwen3CoderFixture.case_5_missing_end_token_recovery(function_name, arguments)
    }
}
