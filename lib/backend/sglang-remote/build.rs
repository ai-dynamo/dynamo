// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compiles client stubs from the temporarily vendored SGLang gRPC contract.

use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let proto_dir = manifest_dir.join("proto");
    let proto_path = proto_dir.join("sglang.proto");

    // `--experimental_allow_proto3_optional` lets older protoc (the container's
    // apt protoc 3.12) compile the proto3 `optional` fields in sglang.proto
    // (e.g. `routed_dp_rank`). The flag was added in protoc 3.12 for exactly
    // this and is a no-op on 3.15+ where proto3 optional is stable.
    tonic_build::configure()
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(&[proto_path.as_path()], &[proto_dir.as_path()])?;

    println!("cargo:rerun-if-changed={}", proto_path.display());
    Ok(())
}
