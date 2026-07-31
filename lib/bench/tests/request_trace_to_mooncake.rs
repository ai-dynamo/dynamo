// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

use tempfile::tempdir;

const FAILING_TRACE: &str = concat!(
    r#"{"schema":"dynamo.request.trace.v1","event_type":"request_end","event_time_unix_ms":1100,"request":{"request_id":"good","request_received_ms":1000,"output_tokens":4,"replay":{"trace_block_size":2,"input_length":2,"input_sequence_hashes":[11]}}}"#,
    "\n",
    r#"{"schema":"dynamo.request.trace.v1","event_type":"request_end","event_time_unix_ms":2100,"request":{"request_id":"bad","request_received_ms":2000,"replay":{"trace_block_size":2,"input_length":2,"input_sequence_hashes":[22]}}}"#,
    "\n",
);

fn convert(input: &std::path::Path, output: &std::path::Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_request_trace_to_mooncake"))
        .args(["--input-path"])
        .arg(input)
        .args(["--output-file"])
        .arg(output)
        .output()
        .expect("run request_trace_to_mooncake")
}

#[test]
fn failed_conversion_does_not_publish_partial_output() {
    let temp = tempdir().unwrap();
    let input = temp.path().join("trace.jsonl");
    let output = temp.path().join("mooncake.jsonl");
    std::fs::write(&input, FAILING_TRACE).unwrap();

    assert!(!convert(&input, &output).status.success());
    assert!(!output.exists());

    std::fs::write(&output, "preserve this output\n").unwrap();
    assert!(!convert(&input, &output).status.success());
    assert_eq!(
        std::fs::read_to_string(output).unwrap(),
        "preserve this output\n"
    );
}
