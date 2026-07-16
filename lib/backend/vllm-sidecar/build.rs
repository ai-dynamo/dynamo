// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let descriptor = PathBuf::from(std::env::var("OUT_DIR")?).join("vllm_descriptor.bin");
    tonic_build::configure()
        .file_descriptor_set_path(descriptor)
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(&["proto/vllm_grpc.proto"], &["proto"])?;
    println!("cargo:rerun-if-changed=proto/vllm_grpc.proto");
    Ok(())
}
