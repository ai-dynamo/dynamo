// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};
use std::process::Command;

const OPENENGINE_COMMIT: &str = "57cd5033554cd22ab9645ae6c17f34d7fa9f5bb0";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=OPENENGINE_PROTO_ROOT");
    println!("cargo:rustc-env=OPENENGINE_PROTO_COMMIT={OPENENGINE_COMMIT}");

    let source = std::env::var_os("OPENENGINE_PROTO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"))
                .join("../../../../openengine-trtllm")
        });
    let (repository, proto_root) = source_layout(&source);
    verify_source_commit(&repository);

    let entrypoint = proto_root.join("openengine/v1/openengine.proto");
    if !entrypoint.is_file() {
        panic!(
            "OpenEngine schema not found at {}. Set OPENENGINE_PROTO_ROOT to a checkout or `buf export` output for {}",
            entrypoint.display(),
            OPENENGINE_COMMIT
        );
    }
    println!("cargo:rerun-if-changed={}", proto_root.display());
    tonic_prost_build::configure()
        .build_client(true)
        .build_server(true)
        .compile_protos(&[entrypoint], &[proto_root])
        .expect("compile OpenEngine protobuf schema");
}

fn source_layout(source: &Path) -> (PathBuf, PathBuf) {
    if source
        .join("proto/openengine/v1/openengine.proto")
        .is_file()
    {
        return (source.to_path_buf(), source.join("proto"));
    }
    if source.join("openengine/v1/openengine.proto").is_file() {
        return (
            source.parent().unwrap_or(source).to_path_buf(),
            source.to_path_buf(),
        );
    }
    (source.to_path_buf(), source.join("proto"))
}

fn verify_source_commit(repository: &Path) {
    if !repository.join(".git").exists() {
        return;
    }
    let output = Command::new("git")
        .args(["-C"])
        .arg(repository)
        .args(["rev-parse", "HEAD"])
        .output()
        .expect("inspect OpenEngine source commit");
    let revision = String::from_utf8_lossy(&output.stdout);
    if !output.status.success() || revision.trim() != OPENENGINE_COMMIT {
        panic!(
            "OpenEngine checkout {} must be at {}, found {}",
            repository.display(),
            OPENENGINE_COMMIT,
            revision.trim()
        );
    }
}
