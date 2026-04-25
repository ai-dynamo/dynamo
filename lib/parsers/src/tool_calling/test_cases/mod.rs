// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared test-case framework for all tool-call parsers.
//!
//! Each parser provides a `ToolCallFixture` impl that emits raw bytes for
//! a given universal scenario (`case_N_*` methods), or declares the
//! scenario as `NotApplicable` with a documented reason.
//!
//! The contract test in `tests.rs` runs every fixture against the parser
//! registry and asserts on a normalized output. The matrix becomes
//! self-policing: `NotApplicable` is explicit, `Unimplemented` is a hard
//! CI fail, and a parser-specific bug surfaces as a single test failure.
//!
//! This first prototype only covers `case_1_single_call`. Future revisions
//! add more universal cases (missing-end-token recovery, parallel calls,
//! malformed args, streaming, etc.). Adding a new case is intentionally a
//! breaking change for every fixture so coverage stays explicit.

use serde_json::Value;

pub mod fixtures;
pub mod normalize;
pub mod tests;

/// One parser's answer to a single test case.
///
/// Three states, each surfaced in the matrix report as a distinct symbol:
///
/// - `Sample` → ✓ — parser handles this case correctly; assert canonical match
/// - `NotApplicable` → N/A — format genuinely has no such concept
/// - `Unimplemented` → CI hard-fail — fixture method wasn't written
///
/// The three-state design separates "format doesn't have this" from
/// "we forgot." `Option<String>` would conflate the two and re-introduce
/// the silent-coverage problem this framework is meant to solve.
#[derive(Debug)]
pub enum FixtureCase<T> {
    /// Parser handles this case correctly. Run the contract assertion.
    Sample(T),

    /// This parser's grammar has no concept matching this scenario.
    /// Reason is printed in the matrix so reviewers can verify the N/A
    /// claim is honest.
    NotApplicable(&'static str),

    /// Fixture method not yet written. CI hard-fails. Used only as the
    /// trait default — every concrete fixture must override every method
    /// before this PR can land. Adding a new case is a breaking change
    /// for every fixture, forcing coverage.
    #[allow(dead_code)]
    Unimplemented,
}

/// Behavioral contract every tool-call parser must satisfy.
///
/// Each method emits the raw string a model would produce for the named
/// scenario. The shared test runner feeds this string to the registered
/// parser via `try_tool_call_parse` and asserts on the normalized output.
///
/// No default implementations: when a new case is added to this trait,
/// every fixture must explicitly take a position on it.
pub trait ToolCallFixture: Send + Sync {
    /// Registry key in `get_tool_parser_map()`. Used to look up the
    /// `ToolCallConfig` the runner passes to the parser.
    fn parser_name(&self) -> &'static str;

    /// CASE.1 — Single tool call (happy path).
    ///
    /// One complete, well-formed call. `function_name` and `arguments`
    /// are the canonical inputs; the fixture wraps them in its grammar.
    fn case_1_single_call(&self, function_name: &str, arguments: &Value) -> FixtureCase<String>;
}
