// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#[cfg(not(feature = "cuda_kv"))]
fn main() {}

#[cfg(feature = "cuda_kv")]
fn main() {
    use std::{path::PathBuf, process::Command};

    println!("cargo:rerun-if-changed=src/kernels/block_copy.cu");

    let cuda_lib = match Command::new("which").arg("nvcc").output() {
        Ok(output) if output.status.success() => {
            let nvcc_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let path = PathBuf::from(nvcc_path);
            if let Some(parent) = path.parent().and_then(|p| p.parent()) {
                parent.to_string_lossy().to_string()
            } else {
                get_cuda_root_or_default()
            }
        }
        _ => {
            println!("cargo:warning=nvcc not found in path");
            get_cuda_root_or_default()
        }
    };

    let cuda_lib_path = PathBuf::from(&cuda_lib).join("lib64");
    println!("cargo:rustc-link-search=native={}", cuda_lib_path.display());
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudadevrt");

    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,{}",
        cuda_lib_path.display()
    );

    let kernel_dir = "src/kernels";
    std::fs::create_dir_all(kernel_dir).expect("Failed to create kernel directory");

    let block_copy_o = format!("{}/libblock_copy.o", kernel_dir);
    let block_copy_a = format!("{}/libblock_copy.a", kernel_dir);

    let compile_status = Command::new("nvcc")
        .args([
            "-O3",
            "--compiler-options",
            "-fPIC",
            "-c",
            "src/kernels/block_copy.cu",
            "-o",
            &block_copy_o,
        ])
        .status()
        .expect("Failed to run nvcc");

    if !compile_status.success() {
        panic!("nvcc failed to compile block_copy.cu");
    }

    let ar_status = Command::new("ar")
        .args(["rcs", &block_copy_a, &block_copy_o])
        .status()
        .expect("Failed to create static library");

    if !ar_status.success() {
        panic!("Failed to create libblock_copy.a");
    }

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let lib_path = PathBuf::from(&manifest_dir).join("src/kernels");

    println!("cargo:rustc-link-search=native={}", lib_path.display());
    println!("cargo:rustc-link-lib=static=block_copy");
}

#[cfg(feature = "cuda_kv")]
fn get_cuda_root_or_default() -> String {
    match std::env::var("CUDA_ROOT") {
        Ok(path) => path,
        Err(_) => {
            // Default locations based on OS
            if cfg!(target_os = "windows") {
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8".to_string()
            } else {
                "/usr/local/cuda".to_string()
            }
        }
    }
}

#[cfg(feature = "trtllm")]
fn main() {
    extern crate bindgen;

    use cmake::Config;
    use std::env;
    use std::path::PathBuf;
    let installed_headers = "/usr/local/include/nvidia/nvllm/nvllm_trt.h";
    let local_headers = "../bindings/cpp/nvllm-trt/include/nvidia/nvllm/nvllm_trt.h";
    let headers_path;

    if PathBuf::from(installed_headers).exists() {
        headers_path = installed_headers;
        println!("cargo:warning=nvllm found. Building with installed version...");
        println!("cargo:rustc-link-search=native=/usr/local/lib");
        println!("cargo:rustc-link-search=native=/opt/tensorrt_llm/lib");
        println!("cargo:rustc-link-lib=dylib=nvllm_trt");
        println!("cargo:rustc-link-lib=dylib=tensorrt_llm");
        println!("cargo:rustc-link-lib=dylib=tensorrt_llm_nvrtc_wrapper");
        println!("cargo:rustc-link-lib=dylib=nvinfer_plugin_tensorrt_llm");
        println!("cargo:rustc-link-lib=dylib=decoder_attention");

        println!("cargo:rerun-if-changed=/usr/local/lib");
    } else if PathBuf::from(local_headers).exists() {
        headers_path = local_headers;
        println!("cargo:warning=nvllm not found. Building stub version...");

        let dst = Config::new("../bindings/cpp/nvllm-trt")
            .define("USE_STUBS", "ON")
            .no_build_target(true)
            .build();

        println!("cargo:warning=building stubs in {}", dst.display());
        let dst = dst.canonicalize().unwrap();

        println!("cargo:rustc-link-search=native={}/build", dst.display());
        println!("cargo:rustc-link-lib=dylib=nvllm_trt");
        println!("cargo:rustc-link-lib=dylib=tensorrt_llm");

        println!("cargo:rerun-if-changed=../bindings/cpp/nvllm-trt");
    } else {
        panic!("nvllm_trt.h not found");
    }

    // generate bindings for the trtllm c api
    let bindings = bindgen::Builder::default()
        .header(headers_path)
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to a file
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Could not write bindings!");

    // // Build protobuf
    // tonic_build::configure()
    //     .build_server(false)
    //     .compile_protos(&["../../proto/trtllm.proto"], &["../../proto"])
    //     .expect("Failed to compile protos");
}
