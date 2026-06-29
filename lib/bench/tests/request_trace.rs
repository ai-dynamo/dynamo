// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end direct replay regressions for realistic agentic and multi-shard traces.

use std::io::Write;
use std::path::{Path, PathBuf};

use dynamo_mocker::loadgen::DynamoRequestTrace;
use flate2::Compression;
use flate2::write::GzEncoder;
use tempfile::tempdir;

mod support;

fn pi_trace_path() -> PathBuf {
    PathBuf::from(support::fixture_path("pi_request_trace.jsonl.gz").expect("fixture path"))
}

#[test]
fn pi_direct_dynamo_lowering_builds_agentic_trace() {
    let trace_path = pi_trace_path();
    let direct = DynamoRequestTrace::from_request_trace_files(&[trace_path], None).unwrap();
    let DynamoRequestTrace::Agentic(direct) = direct else {
        panic!("Pi request trace should lower as agentic");
    };

    assert_eq!(direct.block_size, 16);
    assert_eq!(direct.turns.len(), 17);
    assert!(direct.turns.iter().any(|turn| !turn.wait_for.is_empty()));
    assert!(
        direct
            .turns
            .iter()
            .any(|turn| turn.delay_after_dependencies_ms > 0.0)
    );
}

#[test]
fn context_free_multi_shard_dynamo_lowering_preserves_order() {
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

    assert_eq!(direct.block_size, 2);
    assert_eq!(direct.sessions.len(), 2);
    assert_eq!(direct.sessions[0].first_arrival_timestamp_ms, Some(0.0));
    assert_eq!(direct.sessions[1].first_arrival_timestamp_ms, Some(500.0));
    assert_eq!(direct.sessions[0].turns[0].input_length, 4);
    assert_eq!(direct.sessions[1].turns[0].input_length, 4);
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
