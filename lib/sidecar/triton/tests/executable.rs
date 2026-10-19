// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

#[test]
fn executable_exposes_triton_configuration() {
    let output = Command::new(env!("CARGO_BIN_EXE_dynamo-triton-sidecar"))
        .arg("--help")
        .output()
        .expect("run dynamo-triton-sidecar --help");

    assert!(
        output.status.success(),
        "--help failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("help output is UTF-8");
    for flag in [
        "--discovery-backend",
        "--request-plane",
        "--response-plane",
        "--event-plane",
        "--grpc-endpoint",
        "--grpc-connections",
        "--http-endpoint",
        "--model-name",
    ] {
        assert!(stdout.contains(flag), "missing {flag} in help output");
    }
}
