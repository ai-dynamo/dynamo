// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration test against the vendored Pi agent-trace fixture.
//!
//! The fixture in `testdata/pi_agent_trace.jsonl.gz` is a real Dynamo agent
//! trace captured from a Pi coding-agent run on Qwen3-14B-FP8 (root task plus
//! 3 subagent trajectories). This test guards three regression classes that
//! the per-module unit tests cannot catch:
//!
//! - schema drift in the raw record deserializers,
//! - tool-span attribution from harness tool events to LLM rows, and
//! - the convert-time tool summary (counts and total wall-time).

use std::path::PathBuf;

use dynamo_bench::agent_trace::agentic::{build_agentic_mooncake_rows, summarize_tools};
use dynamo_bench::agent_trace::load::load_agent_trace_records;

mod support;

fn pi_trace_path() -> PathBuf {
    PathBuf::from(support::fixture_path("pi_agent_trace.jsonl.gz").expect("fixture path"))
}

#[test]
fn pi_trace_summary_has_expected_counts() {
    let loaded =
        load_agent_trace_records(&[pi_trace_path()]).expect("Pi trace fixture should load");

    assert_eq!(loaded.requests.len(), 17, "request_end row count");
    assert_eq!(loaded.tools.len(), 22, "terminal tool event count");

    let summary = summarize_tools(&loaded.tools);
    assert_eq!(summary.total_spans, 22);
    assert_eq!(summary.trajectories, 4);
    assert_eq!(summary.by_status.get("succeeded").copied(), Some(20));
    assert_eq!(summary.by_status.get("error").copied(), Some(2));
    assert_eq!(summary.by_class.get("read").copied(), Some(9));
    assert_eq!(summary.by_class.get("ls").copied(), Some(6));
    assert_eq!(summary.by_class.get("bash").copied(), Some(2));
    assert_eq!(summary.by_class.get("subagent").copied(), Some(2));
    assert_eq!(summary.by_class.get("write").copied(), Some(2));
    assert_eq!(summary.by_class.get("find").copied(), Some(1));
    // Total wall-time is the sum of every tool_end / tool_error duration.
    // The Pi run is dominated by a single ~71.8s subagent call.
    assert!(
        (72_000.0..73_000.0).contains(&summary.total_wall_ms),
        "unexpected wall-time {}",
        summary.total_wall_ms,
    );
}

#[test]
fn pi_trace_agentic_rows_preserve_tool_events() {
    let loaded =
        load_agent_trace_records(&[pi_trace_path()]).expect("Pi trace fixture should load");
    let (trace_block_size, rows) =
        build_agentic_mooncake_rows(loaded).expect("agentic lowering should succeed");

    assert_eq!(trace_block_size, 16);
    assert_eq!(rows.len(), 17);

    // Every captured tool span should land on exactly one row's tool_events.
    let attached_spans: usize = rows.iter().map(|row| row.tool_events.len()).sum();
    assert_eq!(attached_spans, 22, "all tool spans attributed to rows");

    // Pi's two subagent dispatches must round-trip with the right tool_class.
    let subagent_events: Vec<_> = rows
        .iter()
        .flat_map(|row| row.tool_events.iter())
        .filter(|event| event.tool_class == "subagent")
        .collect();
    assert_eq!(subagent_events.len(), 2);

    // One of the captured tool spans is the failed bash call in the reviewer
    // trajectory; the row that consumed it should carry an error status and an
    // error_type populated by the harness.
    let error_events: Vec<_> = rows
        .iter()
        .flat_map(|row| row.tool_events.iter())
        .filter(|event| event.status == "error")
        .collect();
    assert_eq!(error_events.len(), 2);
    assert!(
        error_events
            .iter()
            .any(|event| event.error_type.as_deref() == Some("pi_tool_error")),
        "expected at least one pi_tool_error event",
    );
}
