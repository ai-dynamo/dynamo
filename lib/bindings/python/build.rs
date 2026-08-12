// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Build script for dynamo-py3.
//!
//! On macOS, nixl-sys unconditionally links `-lstdc++` which doesn't exist
//! (macOS uses libc++). We create an empty static archive to satisfy the
//! linker since libc++ is already linked.

fn main() {
    println!("cargo:rerun-if-env-changed=DYNAMO_SOURCE_COMMIT");
    let source_revision = std::env::var("DYNAMO_SOURCE_COMMIT")
        .unwrap_or_else(|_| "unknown".to_string());
    if source_revision != "unknown"
        && (source_revision.len() != 40
            || !source_revision
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)))
    {
        panic!("DYNAMO_SOURCE_COMMIT must be a lowercase 40-character Git SHA");
    }
    println!("cargo:rustc-env=DYNAMO_SOURCE_COMMIT={source_revision}");
    println!(
        "cargo:rustc-env=DYNAMO_BUILD_DEFAULT_FEATURES={}",
        std::env::var_os("CARGO_FEATURE_DEFAULT").is_some()
    );
    println!(
        "cargo:rustc-env=DYNAMO_BUILD_PROFILE={}",
        std::env::var("PROFILE").unwrap_or_else(|_| "unknown".to_string())
    );

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
