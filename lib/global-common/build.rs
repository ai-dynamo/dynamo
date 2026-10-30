// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compiles the replica-sync proto when the `grpc` feature is on. Without
//! it the crate has no generated code and no protoc requirement.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=proto/replica_sync.proto");
    if std::env::var_os("CARGO_FEATURE_GRPC").is_none() {
        return Ok(());
    }
    #[cfg(feature = "grpc")]
    {
        let descriptor_path =
            std::path::PathBuf::from(std::env::var("OUT_DIR")?).join("replica_sync_descriptor.bin");
        tonic_build::configure()
            .build_server(true)
            .build_client(true)
            .file_descriptor_set_path(descriptor_path)
            .compile_protos(&["proto/replica_sync.proto"], &["proto"])?;
    }
    Ok(())
}
