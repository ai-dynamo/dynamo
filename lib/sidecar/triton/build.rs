// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compile the KServe protos into a descriptor set so the HTTP transport can
//! translate Triton's JSON model configuration into the protobuf `ModelConfig`
//! that the gRPC transport receives natively.

use std::path::PathBuf;

const PROTO_DIR: &str = "../../llm/src/grpc/protos";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed={PROTO_DIR}");
    let descriptors = protox::Compiler::new([PROTO_DIR])?
        .include_imports(true)
        .open_files(["model_config.proto"])?
        .encode_file_descriptor_set();
    let out = PathBuf::from(std::env::var("OUT_DIR")?).join("model_config_descriptor.bin");
    std::fs::write(out, descriptors)?;
    Ok(())
}
