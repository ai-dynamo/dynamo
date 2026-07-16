// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Build script for dynamo-memory.
//!
//! On macOS, nixl-sys unconditionally links `-lstdc++` which doesn't exist
//! (macOS uses libc++). We create an empty static archive to satisfy the
//! linker since libc++ is already linked.
//!
//! CUDA 13.2 relocated `CUmemLocation::id` into an anonymous union
//! (`CUmemLocation_st::__bindgen_anon_1.id`). cudarc regenerates its FFI
//! bindings per CUDA toolkit version but does not export the detected version
//! to dependent crates (no `links` key), so we detect it here and expose a
//! `cuda_mem_location_union` cfg for src/pool/cuda.rs.
//!
//! This mirrors cudarc's own `cuda-version-from-build-system` + `fallback-latest`
//! logic so our cfg always agrees with the bindings cudarc actually compiled:
//! shell out to `nvcc` on PATH, and if that fails assume the latest CUDA (which
//! has the union), exactly as cudarc's fallback-latest does.

use std::process::Command;

fn main() {
    #[cfg(target_os = "macos")]
    {
        let out_dir = std::env::var("OUT_DIR").unwrap();
        let lib_path = format!("{}/libstdc++.a", out_dir);

        // Write a minimal valid static archive (just the magic header).
        // macOS `ar` refuses to create an empty archive, so write it directly.
        std::fs::write(&lib_path, b"!<arch>\n").expect("failed to create empty libstdc++.a");

        println!("cargo:rustc-link-search=native={}", out_dir);
    }

    println!("cargo::rustc-check-cfg=cfg(cuda_mem_location_union)");
    let is_union = match cuda_version() {
        Some((major, minor)) => (major, minor) >= (13, 2),
        None => true, // nvcc unavailable -> cudarc's fallback-latest -> newest -> union
    };
    if is_union {
        println!("cargo::rustc-cfg=cuda_mem_location_union");
    }
}

/// Detect the CUDA toolkit `(major, minor)` exactly as cudarc does: run `nvcc
/// --version` from PATH and parse the "release X.Y" token. No env-var lookup, so
/// this can't disagree with the toolkit cudarc itself compiled against.
fn cuda_version() -> Option<(u32, u32)> {
    let output = Command::new("nvcc").arg("--version").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    // "Cuda compilation tools, release 13.2, V13.2.55"
    let release = text.split("release ").nth(1)?;
    let version = release.split(',').next()?.trim();
    let mut parts = version.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    Some((major, minor))
}
