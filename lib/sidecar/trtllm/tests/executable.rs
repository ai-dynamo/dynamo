// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

#[test]
fn executable_exposes_native_grpc_configuration() {
    let output = Command::new(env!("CARGO_BIN_EXE_dynamo-trtllm-sidecar"))
        .arg("--help")
        .output()
        .expect("run dynamo-trtllm-sidecar --help");

    assert!(
        output.status.success(),
        "--help failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("help output is UTF-8");
    for flag in [
        "--grpc-endpoint",
        "--model-path",
        "--disaggregation-mode",
        "--discovery-backend",
        "--request-plane",
        "--response-plane",
        "--event-plane",
    ] {
        assert!(stdout.contains(flag), "missing {flag} in help output");
    }
    assert!(stdout.contains("DYN_SIDECAR_GRPC_ENDPOINT"));
}

#[test]
fn invalid_arguments_fail_before_runtime_configuration() {
    let mut command = Command::new(env!("CARGO_BIN_EXE_dynamo-trtllm-sidecar"));
    for (key, _) in std::env::vars().filter(|(key, _)| {
        key.starts_with("DYN_") || key.starts_with("ETCD_") || key.starts_with("NATS_")
    }) {
        command.env_remove(key);
    }
    // A runtime configuration error must not mask a local argument error.
    let output = command
        .args(["--grpc-endpoint", "http://127.0.0.1:0", "--model-path", ""])
        .env("ETCD_ENDPOINTS", "://invalid")
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("model-path must not be empty"), "{stderr}");
}
