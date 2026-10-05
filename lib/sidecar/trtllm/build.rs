// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compiles client stubs from the vendored OpenEngine (`openengine.v1`) contract.

use std::env;
use std::path::PathBuf;

/// The `openengine.v1` files, in `proto/openengine/v1/`. Listed explicitly so a
/// new upstream file has to be vendored deliberately rather than picked up by a
/// glob.
const PROTOS: &[&str] = &[
    "error.proto",
    "generation.proto",
    "kv.proto",
    "lifecycle.proto",
    "lora.proto",
    "model.proto",
    "openengine.proto",
    "server.proto",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?).join("proto");
    let paths: Vec<PathBuf> = PROTOS
        .iter()
        .map(|name| proto_dir.join("openengine/v1").join(name))
        .collect();

    // `--experimental_allow_proto3_optional` lets older protoc compile the
    // proto3 `optional` fields (e.g. `max_context_length`). The flag was added
    // in protoc 3.12 for exactly this and is a no-op on 3.15+.
    tonic_build::configure()
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(&paths, &[proto_dir.as_path()])?;

    for path in &paths {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    Ok(())
}
