// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto = "../kvbm-service/proto/kvbm_service.proto";
    println!("cargo:rerun-if-changed={proto}");
    tonic_build::configure()
        .build_server(false)
        .build_client(true)
        .compile_protos(&[proto], &["../kvbm-service/proto"])?;
    Ok(())
}
