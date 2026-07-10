// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compiles the vendored OpenEngine v1 proto into client + server stubs.
//!
//! The proto is a synced copy of the canonical contract at
//! `openengine/proto/openengine.proto` (see `openengine/gen.sh`). Mirrors the
//! tonic-build setup used by `lib/llm/build.rs`.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // `--experimental_allow_proto3_optional` lets older protoc (the container's
    // apt protoc 3.12) compile the proto3 `optional` fields in openengine.proto
    // (e.g. `data_parallel_rank`). The flag was added in protoc 3.12 for exactly
    // this and is a no-op on 3.15+ where proto3 optional is stable.
    tonic_build::configure()
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(
            &["proto/openengine.proto", "proto/prime_rl.proto"],
            &["proto"],
        )?;
    println!("cargo:rerun-if-changed=proto/openengine.proto");
    println!("cargo:rerun-if-changed=proto/prime_rl.proto");
    Ok(())
}
