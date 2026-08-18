// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

use dynamo_data_gen::mooncake::{
    AGENTIC_MOONCAKE_SCHEMA, AGENTIC_MOONCAKE_VERSION, AgenticMooncakeHeader,
};
use tempfile::tempdir;

const FAILING_TRACE: &str = concat!(
    r#"{"schema":"dynamo.request.trace.v1","event_type":"request_end","event_time_unix_ms":1100,"request":{"request_id":"good","request_received_ms":1000,"output_tokens":4,"replay":{"trace_block_size":2,"input_length":2,"input_sequence_hashes":[11]}}}"#,
    "\n",
    r#"{"schema":"dynamo.request.trace.v1","event_type":"request_end","event_time_unix_ms":2100,"request":{"request_id":"bad","request_received_ms":2000,"replay":{"trace_block_size":2,"input_length":2,"input_sequence_hashes":[22]}}}"#,
    "\n",
);

fn convert(
    input: &std::path::Path,
    output: &std::path::Path,
    agentic: bool,
) -> std::process::Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_request_trace_to_mooncake"));
    command.args(["--input-path"]).arg(input);
    command.args(["--output-file"]).arg(output);
    if agentic {
        command.arg("--agentic");
    }
    command.output().expect("run request_trace_to_mooncake")
}

#[test]
fn failed_conversion_does_not_publish_partial_output() {
    let temp = tempdir().unwrap();
    let input = temp.path().join("trace.jsonl");
    let output = temp.path().join("mooncake.jsonl");
    std::fs::write(&input, FAILING_TRACE).unwrap();

    assert!(!convert(&input, &output, false).status.success());
    assert!(!output.exists());

    std::fs::write(&output, "preserve this output\n").unwrap();
    assert!(!convert(&input, &output, false).status.success());
    assert_eq!(
        std::fs::read_to_string(output).unwrap(),
        "preserve this output\n"
    );
}

#[test]
fn agentic_trace_requires_agentic_flag() {
    let temp = tempdir().unwrap();
    let input = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/pi_request_trace.jsonl.gz");
    let output = temp.path().join("mooncake.jsonl");

    let rejected = convert(&input, &output, false);
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("require --agentic"));
    assert!(!output.exists());

    assert!(convert(&input, &output, true).status.success());
    let contents = std::fs::read_to_string(output).unwrap();
    let mut lines = contents.lines();
    let header: AgenticMooncakeHeader = serde_json::from_str(lines.next().unwrap()).unwrap();
    assert_eq!(header.schema, AGENTIC_MOONCAKE_SCHEMA);
    assert_eq!(header.version, AGENTIC_MOONCAKE_VERSION);
    assert_eq!(lines.count(), 17);
}

#[cfg(unix)]
#[test]
fn published_output_preserves_normal_permissions() {
    use std::os::unix::fs::PermissionsExt;

    let temp = tempdir().unwrap();
    let input = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/pi_request_trace.jsonl.gz");
    let output = temp.path().join("mooncake.jsonl");
    let normal = temp.path().join("normal.jsonl");
    std::fs::File::create(&normal).unwrap();

    assert!(convert(&input, &output, true).status.success());
    let normal_mode = std::fs::metadata(normal).unwrap().permissions().mode() & 0o777;
    let output_mode = std::fs::metadata(&output).unwrap().permissions().mode() & 0o777;
    assert_eq!(output_mode, normal_mode);

    std::fs::set_permissions(&output, std::fs::Permissions::from_mode(0o640)).unwrap();
    assert!(convert(&input, &output, true).status.success());
    let output_mode = std::fs::metadata(output).unwrap().permissions().mode() & 0o777;
    assert_eq!(output_mode, 0o640);
}
