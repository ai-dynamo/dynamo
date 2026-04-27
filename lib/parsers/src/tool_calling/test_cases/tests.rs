// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parametrized contract test — one per universal scenario.
//!
//! Each `case_N_*` function runs once per registered fixture, looks up the
//! parser by name, feeds the fixture-emitted bytes through `try_tool_call_parse`,
//! and asserts on the canonicalized output. The three `FixtureCase` variants
//! produce three distinct expected behaviors:
//!
//!   Sample        → parser must return a call matching the canonical form
//!   NotApplicable → short-circuit with a printed reason
//!   Unimplemented → test fails loudly

use super::FixtureCase;
use super::fixtures::all_fixtures;
use super::normalize::{CanonicalCall, canonicalize, normal_text_is_empty};
use crate::tool_calling::parsers::{get_tool_parser_map, try_tool_call_parse};
use serde_json::json;

fn config_for(parser_name: &str) -> &'static crate::tool_calling::ToolCallConfig {
    get_tool_parser_map()
        .get(parser_name)
        .unwrap_or_else(|| panic!("fixture references unregistered parser: {parser_name}"))
}

fn expected_single_call(name: &str, arguments: &serde_json::Value) -> Vec<CanonicalCall> {
    vec![CanonicalCall {
        name: name.to_string(),
        arguments: arguments.clone(),
    }]
}

/// Drive one fixture case through the parser and return a one-line matrix
/// row plus an optional failure message.
async fn run_case(
    parser_name: &str,
    case: FixtureCase<String>,
    expected: &[CanonicalCall],
    case_label: &str,
) -> (String, Option<String>) {
    match case {
        FixtureCase::Sample(input) => {
            let config = config_for(parser_name);
            match try_tool_call_parse(&input, config, None).await {
                Ok((calls, normal)) => {
                    let actual = canonicalize(&calls);
                    if actual != expected {
                        (
                            format!("  FAIL {parser_name:<16} {case_label}"),
                            Some(format!(
                                "[{parser_name}] {case_label}: expected {expected:?}, got {actual:?}\n  input: {input:?}"
                            )),
                        )
                    } else if !normal_text_is_empty(&normal) {
                        (
                            format!("  FAIL {parser_name:<16} {case_label}"),
                            Some(format!(
                                "[{parser_name}] {case_label}: tool call extracted but `normal_text` leaked: {normal:?}\n  input: {input:?}"
                            )),
                        )
                    } else {
                        (format!("  ✓   {parser_name:<16} {case_label}"), None)
                    }
                }
                Err(e) => (
                    format!("  ERR  {parser_name:<16} {case_label}"),
                    Some(format!(
                        "[{parser_name}] {case_label}: parser returned error: {e}\n  input: {input:?}"
                    )),
                ),
            }
        }
        FixtureCase::NotApplicable(reason) => (
            format!("  N/A {parser_name:<16} {case_label}  ({reason})"),
            None,
        ),
        FixtureCase::Unimplemented => (
            format!("  ??? {parser_name:<16} {case_label}  (UNIMPLEMENTED)"),
            Some(format!(
                "[{parser_name}] {case_label}: fixture method is Unimplemented; \
                 every fixture must take a position on every case (Sample / NotApplicable)"
            )),
        ),
    }
}

#[tokio::test]
async fn case_1_single_call_all_parsers() {
    let function_name = "get_weather";
    let arguments = json!({"location": "NYC"});
    let expected = expected_single_call(function_name, &arguments);

    let mut rows: Vec<String> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    for fx in all_fixtures() {
        let parser_name = fx.parser_name();
        let case = fx.case_1_single_call(function_name, &arguments);
        let (row, fail) = run_case(parser_name, case, &expected, "PARSER.1").await;
        rows.push(row);
        if let Some(f) = fail {
            failures.push(f);
        }
    }

    println!("\n=== case_1_single_call ===");
    for row in &rows {
        println!("{row}");
    }
    println!(
        "=== {} parsers, {} failures ===\n",
        rows.len(),
        failures.len()
    );

    if !failures.is_empty() {
        panic!("PARSER.1 failures:\n{}", failures.join("\n"));
    }
}
