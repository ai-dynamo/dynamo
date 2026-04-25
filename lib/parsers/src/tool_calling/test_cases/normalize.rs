// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Output canonicalizer for cross-parser comparison.
//!
//! Different parsers legitimately disagree on cosmetic details of their
//! output (auto-generated tool-call IDs, the `functions.` prefix Kimi K2
//! sometimes leaves on names, JSON-string vs JSON-Value argument shape).
//! These differences are not bugs but they make raw equality assertions
//! fail spuriously. The canonicalizer strips them so the contract tests
//! compare *meaning*, not byte-level output.
//!
//! Rules currently applied (keep this list complete and short — every new
//! rule should answer "what real parser disagreement does this resolve?"):
//!
//! 1. Drop the auto-generated `id` field. Parsers vary (`tool_call_0` vs
//!    UUIDs); ID equality is never a correctness signal.
//! 2. Strip a leading `functions.` from the function name. Some parsers
//!    (Kimi K2 in particular) emit names with this OpenAI-style prefix
//!    intact; others strip it. The canonical form is the bare name.
//! 3. Parse `function.arguments` (a JSON-encoded string) into a
//!    `serde_json::Value` so equality is structural rather than textual.
//!    A parser may emit `{"a":1}` while another emits `{ "a":1 }`; both
//!    are correct.

use crate::tool_calling::ToolCallResponse;
use serde_json::Value;

/// Canonical, comparable form of a parsed tool call.
#[derive(Debug, PartialEq, Eq)]
pub struct CanonicalCall {
    pub name: String,
    pub arguments: Value,
}

impl CanonicalCall {
    pub fn from_response(call: &ToolCallResponse) -> Self {
        let name = call
            .function
            .name
            .strip_prefix("functions.")
            .unwrap_or(&call.function.name)
            .to_string();
        let arguments: Value =
            serde_json::from_str(&call.function.arguments).unwrap_or(Value::Null);
        Self { name, arguments }
    }
}

/// Canonicalize a vector of parser-emitted calls. Order is preserved
/// (callers compare ordered for single-call cases; CASE.2 parallel-calls
/// in part2 will sort by name before comparison).
pub fn canonicalize(calls: &[ToolCallResponse]) -> Vec<CanonicalCall> {
    calls.iter().map(CanonicalCall::from_response).collect()
}
