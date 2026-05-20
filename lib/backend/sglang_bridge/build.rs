// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

const PROTO_PATH: &str = "proto/sglang/runtime/v1/sglang.proto";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !Path::new(PROTO_PATH).exists() {
        println!(
            "cargo:warning=Proto missing at {PROTO_PATH}; running scripts/sync-proto.sh"
        );
        let status = std::process::Command::new("bash")
            .arg("scripts/sync-proto.sh")
            .status()?;
        if !status.success() {
            return Err(format!(
                "scripts/sync-proto.sh failed (exit {}). \
                 Run it manually or set SGLANG_REPO / pass a ref.",
                status.code().unwrap_or(-1)
            )
            .into());
        }
    }

    tonic_build::configure()
        .build_server(false)
        // protoc < 3.15 (system has 3.12) requires this for proto3 optional.
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(&[PROTO_PATH], &["proto"])?;
    println!("cargo:rerun-if-changed={PROTO_PATH}");
    println!("cargo:rerun-if-changed=scripts/sync-proto.sh");
    Ok(())
}
