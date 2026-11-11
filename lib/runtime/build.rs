// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compiles the NVTX v3 C shim (`src/nvtx/shim.c`) against the vendored headers in
//! `third_party/nvtx3` when the `nvtx` feature is on. Nothing is built otherwise.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(feature = "nvtx")]
    {
        println!("cargo:rerun-if-changed=src/nvtx/shim.c");
        println!("cargo:rerun-if-changed=third_party/nvtx3");
        cc::Build::new()
            .file("src/nvtx/shim.c")
            .include("third_party")
            .warnings(true)
            .compile("dynamo_nvtx_shim");
    }
}
