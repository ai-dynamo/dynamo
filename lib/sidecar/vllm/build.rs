// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(
            &["proto/inference.proto", "proto/control.proto"],
            &["proto"],
        )?;
    println!("cargo:rerun-if-changed=proto/inference.proto");
    println!("cargo:rerun-if-changed=proto/control.proto");
    Ok(())
}
