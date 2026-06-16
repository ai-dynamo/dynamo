// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end regression against the vendored Pi trace fixture: guards schema
//! drift, tool-span attribution, and the convert-time summary.

use std::path::PathBuf;

use dynamo_bench::request_trace::agentic::{build_agentic_mooncake_rows, summarize_tools};
use dynamo_bench::request_trace::load::load_request_trace_records;

mod support;

fn pi_trace_path() -> PathBuf {
    PathBuf::from(support::fixture_path("pi_request_trace.jsonl.gz").expect("fixture path"))
}

#[test]
fn pi_trace_summary_has_expected_counts() {
    let loaded =
        load_request_trace_records(&[pi_trace_path()]).expect("Pi trace fixture should load");

    assert_eq!(loaded.requests.len(), 17, "request_end row count");
    assert_eq!(loaded.tools.len(), 22, "terminal tool event count");

    let summary = summarize_tools(&loaded.tools);
    assert_eq!(summary.total_spans, 22);
    assert_eq!(summary.trajectories, 4);
    assert_eq!(summary.by_status.get("succeeded").copied(), Some(20));
    assert_eq!(summary.by_status.get("error").copied(), Some(2));
    // ~71.8s subagent dominates; range allows minor harness rounding.
    assert!(
        (72_000.0..73_000.0).contains(&summary.total_wall_ms),
        "unexpected wall-time {}",
        summary.total_wall_ms,
    );
}

#[test]
fn pi_trace_agentic_rows_preserve_tool_events() {
    let loaded =
        load_request_trace_records(&[pi_trace_path()]).expect("Pi trace fixture should load");
    let (trace_block_size, rows) =
        build_agentic_mooncake_rows(loaded).expect("agentic lowering should succeed");

    assert_eq!(trace_block_size, 16);
    assert_eq!(rows.len(), 17);

    let attached_spans: usize = rows.iter().map(|row| row.tool_events.len()).sum();
    assert_eq!(attached_spans, 22, "all tool spans attributed to rows");

    let events: Vec<_> = rows.iter().flat_map(|row| row.tool_events.iter()).collect();
    assert_eq!(
        events.iter().filter(|e| e.tool_class == "subagent").count(),
        2
    );
    assert!(
        events
            .iter()
            .any(|e| e.status == "error" && e.error_type.as_deref() == Some("pi_tool_error")),
        "expected at least one pi_tool_error event",
    );
}
