// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::{FixtureCase, ToolCallFixture};
use serde_json::Value;

/// `default` registry entry — `JsonParserConfig::default()` with no
/// configured start/end tokens. Used as a permissive fallback when the
/// caller didn't pick a specific parser. Treated separately so the
/// matrix surfaces it explicitly.
pub struct DefaultFixture;

impl ToolCallFixture for DefaultFixture {
    fn parser_name(&self) -> &'static str {
        "default"
    }

    fn case_1_single_call(&self, function_name: &str, arguments: &Value) -> FixtureCase<String> {
        // Default JSON parser accepts a bare `{"name":..., "arguments":...}`.
        FixtureCase::Sample(format!(
            "{{\"name\":\"{function_name}\",\"arguments\":{arguments}}}"
        ))
    }

    fn case_5_missing_end_token_recovery(
        &self,
        _function_name: &str,
        _arguments: &Value,
    ) -> FixtureCase<String> {
        FixtureCase::NotApplicable(
            "Default parser config has no start/end tokens; the bare JSON \
             object is the entire message. Truncation manifests as CASE.4 \
             (malformed JSON), not section-end recovery.",
        )
    }
}
