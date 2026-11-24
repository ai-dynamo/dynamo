// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=proto/pool_view.proto");
    #[cfg(feature = "grpc")]
    tonic_build::configure()
        .bytes([".dynamo.globalview.v1.PoolState"])
        .compile_protos(&["proto/pool_view.proto"], &["proto"])?;
    Ok(())
}
