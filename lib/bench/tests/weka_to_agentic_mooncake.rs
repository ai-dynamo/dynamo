// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

use dynamo_mocker::loadgen::AgenticTrace;
use tempfile::tempdir;

fn convert(input: &std::path::Path, output: &std::path::Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_weka_to_agentic_mooncake"))
        .args(["--input", input.to_str().unwrap()])
        .args(["--output", output.to_str().unwrap()])
        .output()
        .expect("run weka_to_agentic_mooncake")
}

#[test]
fn conversion_publishes_valid_v2_without_clobbering_it() {
    let temp = tempdir().unwrap();
    let input = temp.path().join("weka.json");
    let output = temp.path().join("agentic.jsonl");
    let trace = serde_json::json!({
        "id": "play",
        "models": ["model"],
        "block_size": 4,
        "hash_id_scope": "local",
        "requests": [{
            "t": 0.0,
            "type": "s",
            "model": "model",
            "in": 4,
            "out": 1,
            "hash_ids": [1],
            "api_time": 0.1
        }]
    });
    std::fs::write(&input, serde_json::to_vec(&trace).unwrap()).unwrap();

    let converted = convert(&input, &output);
    assert!(
        converted.status.success(),
        "{}",
        String::from_utf8_lossy(&converted.stderr)
    );
    let graph = AgenticTrace::from_agentic_mooncake(&output).unwrap();
    assert_eq!(graph.node_count(), 1);
    assert_eq!(graph.play_count(), 1);

    let published = std::fs::read(&output).unwrap();
    let rejected = convert(&input, &output);
    assert!(!rejected.status.success());
    assert!(
        String::from_utf8_lossy(&rejected.stderr).contains("refusing to overwrite"),
        "{}",
        String::from_utf8_lossy(&rejected.stderr)
    );
    assert_eq!(std::fs::read(output).unwrap(), published);
}
