// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::Write;

use aisimulate_replay::loadgen::{ReplayRequestHashes, TraceFileFormat, validate_trace_files};
use dynamo_kv_router::protocols::{
    BlockHashOptions, compute_block_hash_for_seq, compute_seq_hash_for_block,
};
use tempfile::NamedTempFile;

use super::DynamoRequestTrace;

fn write_trace(lines: &[serde_json::Value]) -> NamedTempFile {
    let mut file = NamedTempFile::new().unwrap();
    for line in lines {
        writeln!(file, "{}", serde_json::to_string(line).unwrap()).unwrap();
    }
    file
}

fn request_trace_row(
    request_id: &str,
    block_size: usize,
    agent_context: Option<serde_json::Value>,
) -> serde_json::Value {
    let mut row = serde_json::json!({
        "schema": "dynamo.request.trace.v1",
        "event_type": "request_end",
        "event_time_unix_ms": 1_100,
        "request": {
            "request_id": request_id,
            "request_received_ms": 1_000,
            "output_tokens": 4,
            "replay": {
                "trace_block_size": block_size,
                "input_length": block_size,
                "input_sequence_hashes": [11],
            }
        }
    });
    if let Some(agent_context) = agent_context {
        row["agent_context"] = agent_context;
    }
    row
}

#[test]
fn neutral_replay_hashes_match_dynamo_router_vectors_exactly() {
    for (tokens, block_size) in [
        (vec![1, 2, 3, 4, 5, 6], 4),
        (vec![7, 7, 7, 7, 9, 9, 9, 9], 4),
        ((0..17).collect::<Vec<u32>>(), 3),
    ] {
        let neutral = ReplayRequestHashes::from_tokens(&tokens, block_size);
        let router_local =
            compute_block_hash_for_seq(&tokens, block_size, BlockHashOptions::default());
        let router_sequence = compute_seq_hash_for_block(&router_local);

        assert_eq!(
            neutral.local_block_hashes,
            router_local.iter().map(|hash| hash.0).collect::<Vec<_>>()
        );
        assert_eq!(neutral.sequence_hashes, router_sequence);
    }
}

#[test]
fn dynamo_trace_input_validation_errors_are_clear() {
    enum ValidationCase {
        Validate(TraceFileFormat, Vec<std::path::PathBuf>),
        Load(Vec<std::path::PathBuf>, Option<usize>),
    }

    let mixed = write_trace(&[
        request_trace_row(
            "contextual",
            2,
            Some(serde_json::json!({"session_id": "root"})),
        ),
        request_trace_row("context-free", 2, None),
    ]);
    let inconsistent = write_trace(&[
        request_trace_row("block-2", 2, None),
        request_trace_row("block-4", 4, None),
    ]);
    let block_size = write_trace(&[request_trace_row("block-2", 2, None)]);
    let extra = write_trace(&[serde_json::json!({
        "timestamp": 0,
        "input_length": 2,
        "output_length": 1,
        "hash_ids": [1],
    })]);

    let cases = [
        (
            "empty",
            ValidationCase::Validate(TraceFileFormat::Dynamo, vec![]),
            "at least one trace file",
        ),
        (
            "mixed context",
            ValidationCase::Load(vec![mixed.path().to_path_buf()], None),
            "cannot mix requests with and without agent_context",
        ),
        (
            "inconsistent block size",
            ValidationCase::Load(vec![inconsistent.path().to_path_buf()], None),
            "mixed replay trace_block_size values",
        ),
        (
            "explicit block size mismatch",
            ValidationCase::Load(vec![block_size.path().to_path_buf()], Some(4)),
            "does not match embedded Dynamo request trace block size 2",
        ),
        (
            "multiple non-Dynamo files",
            ValidationCase::Validate(
                TraceFileFormat::Mooncake,
                vec![block_size.path().to_path_buf(), extra.path().to_path_buf()],
            ),
            "requires exactly one trace file",
        ),
    ];

    for (name, case, expected) in cases {
        let error = match case {
            ValidationCase::Validate(format, paths) => validate_trace_files(format, &paths),
            ValidationCase::Load(paths, block_size) => {
                DynamoRequestTrace::from_request_trace_files(&paths, block_size).map(|_| ())
            }
        }
        .expect_err(name);
        assert!(
            error.to_string().contains(expected),
            "{name}: unexpected error: {error:#}"
        );
    }
}
