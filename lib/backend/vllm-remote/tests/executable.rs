// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

#[test]
fn executable_exposes_native_grpc_configuration() {
    let output = Command::new(env!("CARGO_BIN_EXE_dynamo-vllm-remote"))
        .arg("--help")
        .output()
        .expect("run dynamo-vllm-remote --help");

    assert!(
        output.status.success(),
        "--help failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("help output is UTF-8");
    for flag in [
        "--vllm-endpoint",
        "--vllm-connections",
        "--model-path",
        "--model-name",
        "--disaggregation-mode",
    ] {
        assert!(stdout.contains(flag), "missing {flag} in help output");
    }
}
