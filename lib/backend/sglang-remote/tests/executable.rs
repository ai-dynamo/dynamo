// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

#[test]
fn executable_exposes_sglang_managed_contract() {
    let output = Command::new(env!("CARGO_BIN_EXE_dynamo-sglang-remote"))
        .arg("--help")
        .output()
        .expect("run dynamo-sglang-remote --help");

    assert!(
        output.status.success(),
        "--help failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("help output is UTF-8");
    assert!(stdout.contains("--sglang-endpoint"));
    assert!(stdout.contains("SGLANG_GRPC_ENDPOINT"));
}
