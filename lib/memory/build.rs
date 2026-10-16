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
//! This mirrors cudarc's own detection precedence so our cfg agrees with the
//! bindings cudarc actually compiled: honor the `CUDARC_CUDA_VERSION` env
//! override first, then shell out to `nvcc` on PATH, and if neither resolves a
//! version assume the latest CUDA (which has the union), exactly as cudarc's
//! `fallback-latest` does. A residual mismatch (e.g. cudarc detecting a version
//! through a path this script does not replicate) fails loudly at compile time
//! in src/pool/cuda.rs, not silently at runtime.

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

    // Re-evaluate the cfg when the inputs that drive `cuda_version()` change.
    // cudarc reruns on CUDARC_CUDA_VERSION; PATH covers the `nvcc` lookup.
    println!("cargo::rerun-if-env-changed=CUDARC_CUDA_VERSION");
    println!("cargo::rerun-if-env-changed=PATH");

    println!("cargo::rustc-check-cfg=cfg(cuda_mem_location_union)");
    let is_union = match cuda_version() {
        Some((major, minor)) => (major, minor) >= (13, 2),
        None => true, // nvcc unavailable -> cudarc's fallback-latest -> newest -> union
    };
    if is_union {
        println!("cargo::rustc-cfg=cuda_mem_location_union");
    }
}

/// Detect the CUDA toolkit `(major, minor)` following cudarc's precedence: the
/// `CUDARC_CUDA_VERSION` env override wins, otherwise parse `nvcc --version`.
/// Matching cudarc's order keeps this cfg aligned with the bindings it compiled.
fn cuda_version() -> Option<(u32, u32)> {
    cuda_version_from_env().or_else(cuda_version_from_nvcc)
}

/// Parse `CUDARC_CUDA_VERSION` (encoded `{major}0{minor}0`, e.g. `13020` = 13.2)
/// the same way cudarc's `detect_version_from_env` does.
fn cuda_version_from_env() -> Option<(u32, u32)> {
    let digits: u32 = std::env::var("CUDARC_CUDA_VERSION")
        .ok()?
        .trim()
        .parse()
        .ok()?;
    // Encoding is major * 1000 + minor * 10 (patch digit is always 0).
    Some((digits / 1000, (digits % 1000) / 10))
}

/// Detect the CUDA toolkit `(major, minor)` from `nvcc --version` on PATH,
/// parsing the "release X.Y" token exactly as cudarc does.
fn cuda_version_from_nvcc() -> Option<(u32, u32)> {
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
