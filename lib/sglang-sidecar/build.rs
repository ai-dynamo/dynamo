// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compiles the SGLang native gRPC proto into client + server stubs.
//!
//! The proto is synced from SGLang PR #25185 while its disaggregated serving
//! fields are under review.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // `--experimental_allow_proto3_optional` lets older protoc (the container's
    // apt protoc 3.12) compile the proto3 `optional` fields in sglang.proto
    // (e.g. `routed_dp_rank`). The flag was added in protoc 3.12 for exactly
    // this and is a no-op on 3.15+ where proto3 optional is stable.
    tonic_build::configure()
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(&["proto/sglang.proto"], &["proto"])?;
    println!("cargo:rerun-if-changed=proto/sglang.proto");
    Ok(())
}
