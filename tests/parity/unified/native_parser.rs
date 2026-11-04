// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;

use anyhow::Result;
use dynamo_conformance_fixtures_v2::{RunOptions, run_unified_family};

#[test]
fn qwen3_matches_the_authored_corpus_through_the_external_runner() -> Result<()> {
    let case_ids: BTreeSet<_> = [
        "UNIFIED.1-tool_only",
        "UNIFIED.2-two_calls",
        "UNIFIED.10-reason_only",
        "UNIFIED.10-reason_then_content",
        "UNIFIED.30-guided_json_named_tool",
        "UNIFIED.30-guided_json_required_tool",
        "UNIFIED.40-prefilled_reasoning_with_tool",
        "UNIFIED.40-prefilled_reasoning_with_guided_json",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    let expected = case_ids.len();
    let report = run_unified_family(
        "qwen3",
        &RunOptions {
            case_ids: Some(case_ids),
            ..RunOptions::default()
        },
    )?;

    assert_eq!(report.exit_code(true), 0, "{report:#?}");
    assert_eq!(
        (report.selected, report.executed, report.passed),
        (expected, expected, expected),
        "{report:#?}"
    );
    assert_eq!(report.isolation_pairs, expected - 1, "{report:#?}");
    println!("{report:#?}");
    Ok(())
}

#[test]
fn generation_constraints_remain_explicitly_unsupported() -> Result<()> {
    let report = run_unified_family(
        "qwen3",
        &RunOptions {
            case_ids: Some(BTreeSet::from([
                "UNIFIED.36-structured_response_after_reasoning".to_owned(),
            ])),
            ..RunOptions::default()
        },
    )?;

    assert_eq!(report.selected, 1);
    assert_eq!(report.executed, 0);
    assert_eq!(report.unsupported.len(), 1, "{report:#?}");
    assert!(report.unsupported[0].reason.contains("structural-tag"));
    assert_eq!(report.exit_code(false), 0, "{report:#?}");
    assert_eq!(report.exit_code(true), 1, "{report:#?}");
    Ok(())
}
