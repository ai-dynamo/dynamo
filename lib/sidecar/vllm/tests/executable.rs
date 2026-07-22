// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

#[test]
fn executable_exposes_native_grpc_configuration() {
    let output = Command::new(env!("CARGO_BIN_EXE_dynamo-vllm-sidecar"))
        .arg("--help")
        .output()
        .expect("run dynamo-vllm-sidecar --help");

    assert!(
        output.status.success(),
        "--help failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("help output is UTF-8");
    for flag in [
        "--vllm-endpoint",
        "--grpc-connections",
        "--model-path",
        "--disaggregation-mode",
        "--grpc-connect-attempt-timeout-secs",
        "--grpc-retry-interval-secs",
        "--grpc-startup-deadline-secs",
    ] {
        assert!(stdout.contains(flag), "missing {flag} in help output");
    }

    for env in [
        "DYN_SIDECAR_GRPC_CONNECTIONS",
        "DYN_SIDECAR_GRPC_CONNECT_ATTEMPT_TIMEOUT_SECS",
        "DYN_SIDECAR_GRPC_RETRY_INTERVAL_SECS",
        "DYN_SIDECAR_GRPC_STARTUP_DEADLINE_SECS",
    ] {
        assert!(stdout.contains(env), "missing {env} in help output");
    }
}
