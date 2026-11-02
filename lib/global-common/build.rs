// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=proto/pool_view.proto");
    println!("cargo:rerun-if-changed=proto/replica_sync.proto");
    #[cfg(feature = "grpc")]
    {
        tonic_build::configure()
            .bytes([".dynamo.globalview.v1.PoolState"])
            .compile_protos(&["proto/pool_view.proto"], &["proto"])?;
        let descriptor_path =
            std::path::PathBuf::from(std::env::var("OUT_DIR")?).join("replica_sync_descriptor.bin");
        tonic_build::configure()
            .file_descriptor_set_path(descriptor_path)
            .compile_protos(&["proto/replica_sync.proto"], &["proto"])?;
    }
    Ok(())
}
