// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::{FixtureCase, ToolCallFixture};
use serde_json::Value;

/// Harmony (gpt-oss) — channel/recipient token stream parsed via the
/// external `openai_harmony` crate.
///
/// Non-streaming complete form:
///   `<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{...}<|call|>`
pub struct HarmonyFixture;

impl ToolCallFixture for HarmonyFixture {
    fn parser_name(&self) -> &'static str {
        "harmony"
    }

    fn case_1_single_call(&self, function_name: &str, arguments: &Value) -> FixtureCase<String> {
        FixtureCase::Sample(format!(
            "<|channel|>commentary to=functions.{function_name} <|constrain|>json<|message|>{arguments}<|call|>"
        ))
    }
}
