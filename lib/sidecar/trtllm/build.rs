// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compiles client stubs from the vendored TensorRT-LLM gRPC contract.

use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let proto_dir = manifest_dir.join("proto");
    let proto_path = proto_dir.join("trtllm_service.proto");

    // `--experimental_allow_proto3_optional` lets older protoc compile the
    // proto3 `optional` fields (e.g. `top_k`, `temperature`). The flag was
    // added in protoc 3.12 for exactly this and is a no-op on 3.15+.
    tonic_build::configure()
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(&[proto_path.as_path()], &[proto_dir.as_path()])?;

    println!("cargo:rerun-if-changed={}", proto_path.display());
    Ok(())
}
