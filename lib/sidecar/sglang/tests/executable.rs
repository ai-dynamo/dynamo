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
    ] {
        assert!(stdout.contains(expected), "help omits {expected}");
    }
}

#[test]
fn invalid_arguments_fail_before_runtime_configuration() {
    let mut command = Command::new(env!("CARGO_BIN_EXE_dynamo-sglang-sidecar"));
    for (key, _) in std::env::vars().filter(|(key, _)| {
        key.starts_with("DYN_") || key.starts_with("ETCD_") || key.starts_with("NATS_")
    }) {
        command.env_remove(key);
    }
    // A runtime configuration error must not mask a local argument error.
    let output = command
        .args([
            "--grpc-endpoint",
            "http://127.0.0.1:0",
            "--route-to-encoder",
        ])
        .env("DYN_DISCOVERY_BACKEND", "invalid-backend")
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("route-to-encoder is not supported"),
        "{stderr}"
    );
}
