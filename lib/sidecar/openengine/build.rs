// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};
use std::process::Command;

const OPENENGINE_COMMIT: &str = "d09a7313b3af2fbcd9b17aa4d31c509207ab51db";
const OPENENGINE_BSR_MODULE_PREFIX: &str = "buf.build/openengine/openengine:";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=OPENENGINE_PROTO_ROOT");
    println!("cargo:rerun-if-env-changed=OPENENGINE_SCHEMA_RELEASE");
    println!("cargo:rerun-if-env-changed=OPENENGINE_BSR_MODULE");

    let (proto_root, schema_release) = if let Ok(module) = std::env::var("OPENENGINE_BSR_MODULE") {
        if std::env::var_os("OPENENGINE_PROTO_ROOT").is_some() {
            panic!("set only one of OPENENGINE_BSR_MODULE and OPENENGINE_PROTO_ROOT");
        }
        export_bsr_module(&module)
    } else {
        local_schema_source()
    };
    println!("cargo:rustc-env=OPENENGINE_SCHEMA_RELEASE={schema_release}");

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

fn local_schema_source() -> (PathBuf, String) {
    let source = std::env::var_os("OPENENGINE_PROTO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"))
                .join("../../../../openengine-trtllm")
        });
    let (repository, proto_root) = source_layout(&source);
    if !proto_root.join("openengine/v1/openengine.proto").is_file() {
        panic!(
            "OpenEngine schema not found under {}. Set OPENENGINE_PROTO_ROOT to a checkout or `buf export` output for {}",
            proto_root.display(),
            OPENENGINE_COMMIT
        );
    }
    let supplied_release = std::env::var("OPENENGINE_SCHEMA_RELEASE").ok();
    let schema_release = match repository {
        Some(repository) => {
            verify_source_commit(&repository);
            verify_clean_proto(&repository);
            match supplied_release {
                Some(release) if release == OPENENGINE_COMMIT => release,
                Some(_) => panic!(
                    "a local Git checkout must advertise its verified source commit {}",
                    OPENENGINE_COMMIT
                ),
                None => OPENENGINE_COMMIT.to_string(),
            }
        }
        None => {
            let release = supplied_release.unwrap_or_else(|| {
                panic!(
                    "metadata-free OpenEngine schema at {} requires OPENENGINE_SCHEMA_RELEASE",
                    proto_root.display()
                )
            });
            validate_schema_release(&release)
        }
    };
    (proto_root, schema_release)
}

fn source_layout(source: &Path) -> (Option<PathBuf>, PathBuf) {
    if source
        .join("proto/openengine/v1/openengine.proto")
        .is_file()
    {
        let repository = source.join(".git").exists().then(|| source.to_path_buf());
        return (repository, source.join("proto"));
    }
    if source.join("openengine/v1/openengine.proto").is_file() {
        let repository = source.parent().and_then(|parent| {
            (source.file_name().is_some_and(|name| name == "proto") && parent.join(".git").exists())
                .then(|| parent.to_path_buf())
        });
        return (repository, source.to_path_buf());
    }
    (None, source.join("proto"))
}

fn verify_source_commit(repository: &Path) {
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

fn verify_clean_proto(repository: &Path) {
    let output = Command::new("git")
        .args(["-C"])
        .arg(repository)
        .args([
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            "proto",
        ])
        .output()
        .expect("inspect OpenEngine proto status");
    if !output.status.success() {
        panic!(
            "could not inspect OpenEngine proto status in {}",
            repository.display()
        );
    }
    if !output.stdout.is_empty() {
        panic!(
            "OpenEngine checkout {} has dirty or untracked proto files",
            repository.display()
        );
    }
}

fn validate_schema_release(release: &str) -> String {
    let valid_source_commit = release == OPENENGINE_COMMIT;
    let valid_bsr_commit = release.len() == 32 && is_lower_hex(release);
    if !valid_source_commit && !valid_bsr_commit {
        panic!(
            "OPENENGINE_SCHEMA_RELEASE must be pinned source commit {} or an immutable 32-character lowercase hexadecimal BSR commit",
            OPENENGINE_COMMIT
        );
    }
    release.to_string()
}

fn export_bsr_module(module: &str) -> (PathBuf, String) {
    let release = module
        .strip_prefix(OPENENGINE_BSR_MODULE_PREFIX)
        .filter(|release| release.len() == 32 && is_lower_hex(release))
        .unwrap_or_else(|| {
            panic!(
                "OPENENGINE_BSR_MODULE must be {}<32-character lowercase hexadecimal commit>; labels are not accepted",
                OPENENGINE_BSR_MODULE_PREFIX
            )
        });
    if let Ok(explicit) = std::env::var("OPENENGINE_SCHEMA_RELEASE")
        && explicit != release
    {
        panic!("OPENENGINE_SCHEMA_RELEASE does not match OPENENGINE_BSR_MODULE");
    }
    let export_root =
        PathBuf::from(std::env::var_os("OUT_DIR").expect("Cargo build output directory"))
            .join("openengine-bsr");
    if export_root.exists() {
        std::fs::remove_dir_all(&export_root).expect("clear prior OpenEngine BSR export");
    }
    let status = Command::new("buf")
        .args(["export", module, "--output"])
        .arg(&export_root)
        .status()
        .unwrap_or_else(|error| panic!("run `buf export` for {module}: {error}"));
    if !status.success() {
        panic!("`buf export {module}` failed");
    }
    (export_root, release.to_string())
}

fn is_lower_hex(value: &str) -> bool {
    value
        .bytes()
        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
