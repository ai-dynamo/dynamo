// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compiles the vendored SMG SGLang scheduler proto into client + server stubs.
//!
//! The service and field numbers match upstream SGLang's
//! `smg-grpc-client = 1.0.0` scheduler proto. This crate vendors only the
//! messages Dynamo needs; proto3 unknown fields keep it wire-compatible with
//! richer upstream servicers.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    unsafe {
        std::env::set_var("PROTOC", protobuf_src::protoc());
    }
    let proto_include = protobuf_src::include();
    let includes = [std::path::Path::new("proto"), proto_include.as_path()];

    tonic_build::configure()
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(&["proto/sglang_scheduler.proto"], &includes)?;
    println!("cargo:rerun-if-changed=proto/sglang_scheduler.proto");
    Ok(())
}
