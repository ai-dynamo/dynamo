// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Build script for dynamo-py3.
//!
//! On macOS, nixl-sys unconditionally links `-lstdc++` which doesn't exist
//! (macOS uses libc++). We create an empty static archive to satisfy the
//! linker since libc++ is already linked.

use std::path::Path;
use std::process::Command;

fn main() {
    emit_git_build_identity();

    #[cfg(target_os = "macos")]
    {
        let out_dir = std::env::var("OUT_DIR").unwrap();
        let lib_path = format!("{}/libstdc++.a", out_dir);

        // Write a minimal valid static archive (just the magic header).
        // macOS `ar` refuses to create an empty archive, so write it directly.
        std::fs::write(&lib_path, b"!<arch>\n").expect("failed to create empty libstdc++.a");

        println!("cargo:rustc-link-search=native={}", out_dir);
    }
}

/// Embed the source checkout identity in the extension. Performance harnesses
/// use this to prove that a release artifact and the recorded source revision
/// came from the same checkout instead of merely sharing a Cargo target path.
fn emit_git_build_identity() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set");
    let source_dir = Path::new(&manifest_dir);
    let revision = git_stdout(source_dir, &["rev-parse", "HEAD"])
        .filter(|value| value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .unwrap_or_else(|| "unknown".to_string());
    let dirty = git_stdout(
        source_dir,
        &["status", "--porcelain=v1", "--untracked-files=normal"],
    )
    .map(|value| !value.is_empty());

    println!("cargo:rustc-env=DYNAMO_BUILD_GIT_REVISION={revision}");
    println!(
        "cargo:rustc-env=DYNAMO_BUILD_GIT_DIRTY={}",
        dirty.map_or("unknown", |value| if value { "true" } else { "false" })
    );

    if let Some(git_head) = git_stdout(source_dir, &["rev-parse", "--git-path", "HEAD"]) {
        println!("cargo:rerun-if-changed={git_head}");
    }
    if let Some(symbolic_ref) = git_stdout(source_dir, &["symbolic-ref", "-q", "HEAD"])
        && let Some(reference_path) =
            git_stdout(source_dir, &["rev-parse", "--git-path", &symbolic_ref])
    {
        println!("cargo:rerun-if-changed={reference_path}");
    }
}

fn git_stdout(source_dir: &Path, arguments: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .args(arguments)
        .current_dir(source_dir)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}
