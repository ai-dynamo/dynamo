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
fn main() {
    println!("cargo:warning=Building CUDA KV");
}

#[cfg(feature = "cuda_kv")]
fn main() {
    use std::{path::PathBuf, process::Command};

    println!("cargo:rerun-if-changed=src/kernels/block_copy.cu");

    // first do a which nvcc, if it is in the path
    // if so, we don't need to set the cuda_lib
    let nvcc = Command::new("which").arg("nvcc").output().unwrap();
    let cuda_lib = if nvcc.status.success() {
        println!("cargo:info=nvcc found in path");
        // Extract the path from nvcc location by removing "bin/nvcc"
        let nvcc_path = String::from_utf8_lossy(&nvcc.stdout).trim().to_string();
        let path = PathBuf::from(nvcc_path);
        if let Some(parent) = path.parent() {
            // Remove "nvcc"
            if let Some(cuda_root) = parent.parent() {
                // Remove "bin"
                cuda_root.to_string_lossy().to_string()
            } else {
                // Fallback to CUDA_ROOT or default if path extraction fails
                get_cuda_root_or_default()
            }
        } else {
            // Fallback to CUDA_ROOT or default if path extraction fails
            get_cuda_root_or_default()
        }
    } else {
        println!("cargo:warning=nvcc not found in path");
        get_cuda_root_or_default()
    };

    println!("cargo:info=Using CUDA installation at: {}", cuda_lib);

    let cuda_lib_path = PathBuf::from(&cuda_lib).join("lib64");
    println!("cargo:info=Using CUDA libs: {}", cuda_lib_path.display());
    println!("cargo:rustc-link-search=native={}", cuda_lib_path.display());

    // Link against multiple CUDA libraries
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudadevrt");

    // Make sure the CUDA libraries are found before other system libraries
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,{}",
        cuda_lib_path.display()
    );

    // Create kernels directory for output if it doesn't exist
    std::fs::create_dir_all("src/kernels").unwrap_or_else(|_| {
        println!("Kernels directory already exists");
    });

    // Compile CUDA code
    let output = Command::new("nvcc")
        .arg("src/kernels/block_copy.cu")
        .arg("-O3")
        .arg("--compiler-options")
        .arg("-fPIC")
        .arg("-o")
        .arg("src/kernels/libblock_copy.o")
        .arg("-c")
        .output()
        .expect("Failed to compile CUDA code");

    if !output.status.success() {
        panic!(
            "Failed to compile CUDA kernel: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    // Create static library
    #[cfg(target_os = "windows")]
    {
        Command::new("lib")
            .arg("/OUT:src/kernels/block_copy.lib")
            .arg("src/kernels/libblock_copy.o")
            .output()
            .expect("Failed to create static library");
        println!("cargo:rustc-link-search=native=src/kernels");
        println!("cargo:rustc-link-lib=static=block_copy");
    }

    #[cfg(not(target_os = "windows"))]
    {
        Command::new("ar")
            .arg("rcs")
            .arg("src/kernels/libblock_copy.a")
            .arg("src/kernels/libblock_copy.o")
            .output()
            .expect("Failed to create static library");
        println!("cargo:rustc-link-search=native=src/kernels");
        println!("cargo:rustc-link-lib=static=block_copy");
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=cuda");
        println!("cargo:rustc-link-lib=dylib=cudadevrt");
    }
}

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
