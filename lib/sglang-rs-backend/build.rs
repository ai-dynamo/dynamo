// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

const COMMON_PROTO_PATH: &str = "proto/common.proto";
const SGLANG_PROTO_PATH: &str = "proto/sglang_scheduler.proto";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !Path::new(COMMON_PROTO_PATH).exists() || !Path::new(SGLANG_PROTO_PATH).exists() {
        println!("cargo:warning=SMG protos missing; running scripts/sync-proto.sh");
        let status = std::process::Command::new("bash")
            .arg("scripts/sync-proto.sh")
            .status()?;
        if !status.success() {
            return Err(format!(
                "scripts/sync-proto.sh failed (exit {}). \
                 Run it manually or set SMG_REPO / pass a ref.",
                status.code().unwrap_or(-1)
            )
            .into());
        }
    }

    tonic_build::configure()
        .build_server(false)
        // protoc < 3.15 (system has 3.12) requires this for proto3 optional.
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(&[COMMON_PROTO_PATH, SGLANG_PROTO_PATH], &["proto"])?;
    println!("cargo:rerun-if-changed={COMMON_PROTO_PATH}");
    println!("cargo:rerun-if-changed={SGLANG_PROTO_PATH}");
    println!("cargo:rerun-if-changed=scripts/sync-proto.sh");
    Ok(())
}
