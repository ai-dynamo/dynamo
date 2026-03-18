// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Build script for dynamo-py3.
//!
//! On macOS, nixl-sys unconditionally links `-lstdc++` which doesn't exist
//! (macOS uses libc++). We create an empty static archive to satisfy the
//! linker since libc++ is already linked.

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
}
