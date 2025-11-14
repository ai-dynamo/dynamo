// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

fn main() {
    println!("cargo:rerun-if-changed=cuda/tensor_kernels.cu");

    let mut build = cc::Build::new();
    build
        .cuda(true)
        .file("cuda/tensor_kernels.cu")
        .flag("-std=c++17")
        .flag("-Xcompiler")
        .flag("-fPIC");

    if let Ok(arch_list) = std::env::var("CUDA_ARCHS") {
        for arch in arch_list.split(',') {
            let arch = arch.trim();
            if arch.is_empty() {
                continue;
            }
            build.flag(format!("-gencode=arch=compute_{arch},code=sm_{arch}"));
        }
    } else {
        // Default to Ampere (SM 80) and Hopper (SM 90) support.
        build.flag("-gencode=arch=compute_80,code=sm_80");
        build.flag("-gencode=arch=compute_90,code=sm_90");
    }

    build.compile("tensor_kernels");

    println!("cargo:rustc-link-lib=cudart");
}
