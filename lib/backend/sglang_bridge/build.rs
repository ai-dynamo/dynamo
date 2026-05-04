// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(false)
        // protoc < 3.15 (system has 3.12) requires this for proto3 optional.
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(
            &[
                "proto/sglang_scheduler.proto",
                "proto/common.proto",
            ],
            &["proto"],
        )?;
    println!("cargo:rerun-if-changed=proto/sglang_scheduler.proto");
    println!("cargo:rerun-if-changed=proto/common.proto");
    Ok(())
}
