// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fetches SGLang's pinned native gRPC contract and compiles client stubs.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let fetch_script = manifest_dir.join("scripts/fetch_sglang_proto.py");
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let proto_path = out_dir.join("sglang.proto");
    let python = env::var_os("PYTHON").unwrap_or_else(|| "python3".into());

    let output = Command::new(&python)
        .arg(&fetch_script)
        .arg(&proto_path)
        .output()
        .map_err(|err| {
            format!(
                "failed to run {} with {:?}: {err}",
                fetch_script.display(),
                python
            )
        })?;

    if !output.status.success() {
        return Err(format!(
            "failed to fetch SGLang gRPC proto (status {}):\n{}{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }

    // `--experimental_allow_proto3_optional` lets older protoc (the container's
    // apt protoc 3.12) compile the proto3 `optional` fields in sglang.proto
    // (e.g. `routed_dp_rank`). The flag was added in protoc 3.12 for exactly
    // this and is a no-op on 3.15+ where proto3 optional is stable.
    tonic_build::configure()
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(&[proto_path.as_path()], &[out_dir.as_path()])?;

    println!("cargo:rerun-if-changed={}", fetch_script.display());
    println!("cargo:rerun-if-env-changed=PYTHON");
    println!("cargo:rerun-if-env-changed=SGLANG_PROTO_PATH");
    println!("cargo:rerun-if-env-changed=SGLANG_SOURCE");
    Ok(())
}
