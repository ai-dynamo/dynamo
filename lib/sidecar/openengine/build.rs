// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compiles client stubs from the vendored OpenEngine v0.1.0 contract.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(&["proto/openengine/v1/openengine.proto"], &["proto"])?;
    println!("cargo:rerun-if-changed=proto/openengine/v1");
    Ok(())
}
