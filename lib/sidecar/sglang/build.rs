// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compiles client and server stubs from the vendored SGLang gRPC contract.

use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let proto_dir = manifest_dir.join("proto");
    let proto_path = proto_dir.join("sglang/api/v1/service.proto");

    let encoder = proto_dir.join("sglang/encoder/v1/encoder.proto");

    // `--experimental_allow_proto3_optional` lets older protoc (the container's
    // apt protoc 3.12) compile the proto3 `optional` fields in the contract
    // (e.g. `routed_dp_rank`). The flag was added in protoc 3.12 for exactly
    // this and is a no-op on 3.15+ where proto3 optional is stable.
    let omit_zero = r#"#[serde(default, skip_serializing_if = "crate::proto::is_zero")]"#;
    tonic_build::configure()
        .build_client(true)
        .build_server(true)
        .protoc_arg("--experimental_allow_proto3_optional")
        .type_attribute(
            ".sglang.grpc.encoder",
            "#[derive(serde::Serialize, serde::Deserialize)]",
        )
        .field_attribute(
            ".sglang.grpc.encoder.StagedImageTransfer.videos",
            r#"#[serde(default, skip_serializing_if = "Vec::is_empty")]"#,
        )
        .field_attribute(
            ".sglang.grpc.encoder.StagedImageTransfer.audios",
            r#"#[serde(default, skip_serializing_if = "Vec::is_empty")]"#,
        )
        .field_attribute(
            ".sglang.grpc.encoder.VideoLayout.timestamps",
            "#[serde(default)]",
        )
        .field_attribute(
            ".sglang.grpc.encoder.StagedImageTransfer.native_model_type",
            r#"#[serde(default, skip_serializing_if = "String::is_empty")]"#,
        )
        .field_attribute(
            ".sglang.grpc.encoder.StagedImageTransfer.embedding_width",
            omit_zero,
        )
        .field_attribute(".sglang.grpc.encoder.ImageLayout.embedding_rows", omit_zero)
        .field_attribute(".sglang.grpc.encoder.VideoLayout.embedding_rows", omit_zero)
        .field_attribute(
            ".sglang.grpc.encoder.ImageLayout.original_size",
            r#"#[serde(default, skip_serializing_if = "Vec::is_empty")]"#,
        )
        .compile_protos(
            &[proto_path.as_path(), encoder.as_path()],
            &[proto_dir.as_path()],
        )?;

    // service.proto imports the rest of the contract.
    println!("cargo:rerun-if-changed={}", proto_dir.display());
    Ok(())
}
