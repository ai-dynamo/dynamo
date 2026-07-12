// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::process::Command;

const OPENENGINE_COMMIT: &str = "cea19cb06acf03c911b84d5c147e519b60dd92a6";

fn main() {
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let repository = manifest.join("../../../openengine-trtllm");
    let package = repository.join("packages/rust/openengine-proto");
    let proto = repository.join("proto");

    println!("cargo:rerun-if-changed={}", package.display());
    println!("cargo:rerun-if-changed={}", proto.display());
    println!("cargo:rustc-env=OPENENGINE_PROTO_COMMIT={OPENENGINE_COMMIT}");

    let output = Command::new("git")
        .args(["-C"])
        .arg(&repository)
        .args(["rev-parse", "HEAD"])
        .output()
        .unwrap_or_else(|error| panic!("failed to inspect local OpenEngine dependency: {error}"));
    assert!(
        output.status.success(),
        "local OpenEngine dependency is not a Git worktree: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    let actual = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        actual.trim(),
        OPENENGINE_COMMIT,
        "local openengine-proto dependency drifted; fetch and checkout the locked OpenEngine commit"
    );

    let status = Command::new("git")
        .args(["-C"])
        .arg(&repository)
        .args(["status", "--porcelain", "--untracked-files=all", "--"])
        .args(["proto", "packages/rust/openengine-proto", "packages/python"])
        .output()
        .unwrap_or_else(|error| panic!("failed to inspect local OpenEngine dependency: {error}"));
    assert!(
        status.status.success(),
        "failed to inspect local OpenEngine dependency status: {}",
        String::from_utf8_lossy(&status.stderr).trim()
    );
    assert!(
        status.stdout.is_empty(),
        "local OpenEngine protocol/package sources are dirty; pinning HEAD would not identify the consumed contract:\n{}",
        String::from_utf8_lossy(&status.stdout)
    );
}
