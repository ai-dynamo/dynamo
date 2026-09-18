// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

#[test]
fn executable_exposes_sglang_and_shared_sidecar_contracts() {
    let output = Command::new(env!("CARGO_BIN_EXE_dynamo-sglang-sidecar"))
        .arg("--help")
        .output()
        .expect("run dynamo-sglang-sidecar --help");

    assert!(
        output.status.success(),
        "--help failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("help output is UTF-8");
    for expected in [
        "--grpc-endpoint",
        "DYN_SIDECAR_GRPC_ENDPOINT",
        "--grpc-connections",
        "DYN_SIDECAR_GRPC_CONNECTIONS",
        "--grpc-connect-attempt-timeout-secs",
        "DYN_SIDECAR_GRPC_CONNECT_ATTEMPT_TIMEOUT_SECS",
        "--grpc-retry-interval-secs",
        "DYN_SIDECAR_GRPC_RETRY_INTERVAL_SECS",
        "--grpc-startup-deadline-secs",
        "DYN_SIDECAR_GRPC_STARTUP_DEADLINE_SECS",
        "--leader-discovery-timeout-secs",
    ] {
        assert!(stdout.contains(expected), "help omits {expected}");
    }
}

#[test]
fn sidecar_requires_a_local_grpc_endpoint() {
    let output = Command::new(env!("CARGO_BIN_EXE_dynamo-sglang-sidecar"))
        .env_remove("DYN_SIDECAR_GRPC_ENDPOINT")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("the following required arguments were not provided"));
    assert!(stderr.contains("--grpc-endpoint <GRPC_ENDPOINT>"));
}
