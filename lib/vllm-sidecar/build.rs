// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compiles vLLM's gRPC schema into client stubs.
//!
//! The vendored schema matches `vllm/rust/proto/vllm_grpc.proto` at the vLLM
//! revision used by this sidecar.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // `--experimental_allow_proto3_optional` lets older protoc (the container's
    // apt protoc 3.12) compile the proto3 `optional` fields in vllm_grpc.proto.
    // The flag was added in protoc 3.12 for exactly
    // this and is a no-op on 3.15+ where proto3 optional is stable.
    tonic_build::configure()
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(&["proto/vllm_grpc.proto"], &["proto"])?;
    println!("cargo:rerun-if-changed=proto/vllm_grpc.proto");
    Ok(())
}
