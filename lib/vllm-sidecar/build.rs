// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compiles vLLM's gRPC schema into client stubs and fails when the vendored
//! file changes without updating its provenance metadata.

use sha2::{Digest, Sha256};

const PROTO_PATH: &str = "proto/vllm_grpc.proto";
const SOURCE_PATH: &str = "proto/VLLM_GRPC_PROTO.source";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    verify_proto_provenance()?;

    // `--experimental_allow_proto3_optional` lets older protoc (the container's
    // apt protoc 3.12) compile the proto3 `optional` fields in vllm_grpc.proto.
    // The flag was added in protoc 3.12 for exactly
    // this and is a no-op on 3.15+ where proto3 optional is stable.
    tonic_build::configure()
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(&[PROTO_PATH], &["proto"])?;
    println!("cargo:rerun-if-changed={PROTO_PATH}");
    println!("cargo:rerun-if-changed={SOURCE_PATH}");
    Ok(())
}

fn verify_proto_provenance() -> Result<(), Box<dyn std::error::Error>> {
    let metadata = std::fs::read_to_string(SOURCE_PATH)?;
    let expected = metadata
        .lines()
        .find_map(|line| line.trim().strip_prefix("sha256 = "))
        .map(|value| value.trim_matches('"'))
        .ok_or("VLLM_GRPC_PROTO.source is missing sha256 metadata")?;
    let actual = format!("{:x}", Sha256::digest(std::fs::read(PROTO_PATH)?));
    if actual != expected {
        return Err(format!(
            "{PROTO_PATH} checksum {actual} does not match {SOURCE_PATH} ({expected}); sync the vLLM proto and update its provenance together"
        )
        .into());
    }
    Ok(())
}
