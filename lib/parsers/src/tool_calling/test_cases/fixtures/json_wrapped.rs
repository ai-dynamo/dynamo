// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::{FixtureCase, ToolCallFixture};
use serde_json::Value;

/// What this parser does when the closing wrapper is missing.
#[derive(Clone, Copy)]
pub enum MissingEndBehavior {
    /// Parser has a recovery path; missing end yields the same call.
    Recovers,
    /// Parser drops the call when the end token is missing.
    Drops { reason: &'static str },
    /// Format has no separable end token, so the case is meaningless.
    NotApplicable { reason: &'static str },
}

/// Family fixture for parsers that emit `{start}{json}{end}` — a JSON
/// object body sandwiched between two literal token strings. Differences
/// across parsers in this family are entirely the start/end strings and
/// what they do when the closing token is missing.
pub struct JsonWrappedFixture {
    pub name: &'static str,
    pub start: &'static str,
    pub end: &'static str,
    pub on_missing_end: MissingEndBehavior,
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

    fn case_5_missing_end_token_recovery(
        &self,
        function_name: &str,
        arguments: &Value,
    ) -> FixtureCase<String> {
        let truncated = format!(
            "{}{{\"name\":\"{function_name}\",\"arguments\":{arguments}}}",
            self.start
        );
        match self.on_missing_end {
            MissingEndBehavior::Recovers => FixtureCase::Sample(truncated),
            MissingEndBehavior::Drops { reason } => FixtureCase::KnownBroken {
                input: truncated,
                reason,
            },
            MissingEndBehavior::NotApplicable { reason } => FixtureCase::NotApplicable(reason),
        }
    }
}
