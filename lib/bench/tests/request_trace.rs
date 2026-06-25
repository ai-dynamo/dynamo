// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end regression against the vendored Pi trace fixture: guards schema
//! drift, tool-span attribution, and the convert-time summary.

use std::io::Write;
use std::path::{Path, PathBuf};

use dynamo_bench::request_trace::agentic::{build_agentic_mooncake_rows, summarize_tools};
use dynamo_bench::request_trace::load::load_request_trace_records;
use dynamo_bench::request_trace::mooncake::build_mooncake_rows;
use dynamo_data_gen::MooncakeJsonlWriter;
use dynamo_mocker::loadgen::{AgenticTrace, DynamoRequestTrace, Trace};
use flate2::Compression;
use flate2::write::GzEncoder;
use tempfile::tempdir;

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
    assert_eq!(summary.sessions, 4);
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

#[test]
fn pi_direct_dynamo_lowering_matches_agentic_mooncake_reload() {
    let trace_path = pi_trace_path();
    let loaded = load_request_trace_records(std::slice::from_ref(&trace_path)).unwrap();
    let (block_size, rows) = build_agentic_mooncake_rows(loaded).unwrap();
    let direct = AgenticTrace::from_agentic_mooncake_rows(rows.clone(), block_size).unwrap();
    let dir = tempdir().unwrap();
    let mooncake_path = dir.path().join("pi.agentic-mooncake.jsonl");
    let mut writer = MooncakeJsonlWriter::create(&mooncake_path, None).unwrap();
    for row in &rows {
        writer.write_agentic_row(row).unwrap();
    }
    writer.finish().unwrap();
    let reloaded = AgenticTrace::from_agentic_mooncake(&mooncake_path, block_size).unwrap();

    assert_eq!(direct, reloaded);
}

#[test]
fn context_free_multi_shard_dynamo_lowering_matches_mooncake_reload() {
    let dir = tempdir().unwrap();
    let later = dir.path().join("trace.0001.jsonl.gz");
    let earlier = dir.path().join("trace.0002.jsonl.gz");
    write_gzip_record(
        &later,
        &request_trace_record("req-b", 1_500, 1_600, &[11, 33]),
    );
    write_gzip_record(
        &earlier,
        &request_trace_record("req-a", 1_000, 1_100, &[11, 22]),
    );
    let paths = vec![later, earlier];

    let direct = DynamoRequestTrace::from_request_trace_files(&paths, None).unwrap();
    let DynamoRequestTrace::Standard(direct) = direct else {
        panic!("context-free request trace should lower as standard");
    };

    let loaded = load_request_trace_records(&paths).unwrap();
    let (block_size, rows) = build_mooncake_rows(loaded.requests).unwrap();
    let mooncake_path = dir.path().join("context-free.mooncake.jsonl");
    let mut writer = MooncakeJsonlWriter::create(&mooncake_path, None).unwrap();
    for row in &rows {
        writer.write_row(row).unwrap();
    }
    writer.finish().unwrap();
    let reloaded = Trace::from_mooncake(&mooncake_path, block_size).unwrap();

    assert_eq!(direct.block_size, 2);
    assert_eq!(direct, reloaded);
}

fn request_trace_record(
    request_id: &str,
    request_received_ms: u64,
    event_time_unix_ms: u64,
    input_sequence_hashes: &[u64],
) -> String {
    serde_json::json!({
        "schema": "dynamo.request.trace.v1",
        "event_type": "request_end",
        "event_time_unix_ms": event_time_unix_ms,
        "request": {
            "request_id": request_id,
            "request_received_ms": request_received_ms,
            "output_tokens": 4,
            "replay": {
                "trace_block_size": 2,
                "input_length": input_sequence_hashes.len() * 2,
                "input_sequence_hashes": input_sequence_hashes,
            }
        }
    })
    .to_string()
}

fn write_gzip_record(path: &Path, record: &str) {
    let file = std::fs::File::create(path).unwrap();
    let mut writer = GzEncoder::new(file, Compression::default());
    writeln!(writer, "{record}").unwrap();
    writer.finish().unwrap();
}
