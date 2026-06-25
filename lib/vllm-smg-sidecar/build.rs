// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compiles the vendored SMG vLLM proto into client + server stubs.
//!
//! The sources are copied from `smg-grpc-proto==0.4.11`, which is what upstream
//! vLLM's `vllm[grpc]` path consumes through `smg-grpc-servicer`.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(
            &["proto/common.proto", "proto/vllm_engine.proto"],
            &["proto"],
        )?;
    println!("cargo:rerun-if-changed=proto/common.proto");
    println!("cargo:rerun-if-changed=proto/vllm_engine.proto");
    Ok(())
}
