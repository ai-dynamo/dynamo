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
//! This first revision covers two cases:
//!   - `case_1_single_call` — single tool call (happy path)
//!   - `case_5_missing_end_token_recovery` — recovers a complete tool
//!     call when the section/closing token is missing (max_tokens or
//!     EOS truncation). PR #8208 is the reference fix for `kimi_k2`.
//!
//! Future revisions will add more universal cases (parallel calls,
//! malformed args, finish_reason, streaming chunkings, etc.). Adding a
//! new case is intentionally a breaking change for every fixture so
//! coverage stays explicit.

use serde_json::Value;

pub mod fixtures;
pub mod normalize;
pub mod tests;

/// One parser's answer to a single test case.
///
/// Four states, each surfaced in the matrix report as a distinct symbol:
///
/// - `Sample` → ✓ — parser handles this case correctly; assert canonical match
/// - `KnownBroken` → 🔧 — parser drops the call today; the test asserts the
///   broken behavior so a future fix that adds recovery shows up as a failed
///   test (forcing the fixture to be upgraded to `Sample`)
/// - `NotApplicable` → N/A — format genuinely has no such concept
/// - `Unimplemented` → CI hard-fail — fixture method wasn't written
///
/// The four-state design separates "format doesn't have this" from
/// "format has this but parser doesn't recover yet" from "we forgot."
/// `Option<String>` would conflate the first two and re-introduce the
/// silent-coverage problem this framework is meant to solve.
#[derive(Debug)]
pub enum FixtureCase<T> {
    /// Parser handles this case correctly. Run the contract assertion.
    Sample(T),

    /// Parser does NOT recover from this case today; it returns 0 calls.
    /// The test asserts the parser returns 0 calls so any future recovery
    /// work shows up as a test failure that forces the fixture to be
    /// upgraded to `Sample`.
    KnownBroken { input: T, reason: &'static str },

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

    /// CASE.5 — Missing end-token recovery.
    ///
    /// The model emitted a complete individual tool call but stopped
    /// before the section/closing token (typical max_tokens or EOS
    /// truncation). Fixtures whose grammar has no separable section-end
    /// token return `NotApplicable` with a reason.
    ///
    /// Reference: PR #8208 fixed this for `kimi_k2`. This case generalizes
    /// the regression check to every parser whose grammar has a separable
    /// section/closing token.
    fn case_5_missing_end_token_recovery(
        &self,
        function_name: &str,
        arguments: &Value,
    ) -> FixtureCase<String>;
}
