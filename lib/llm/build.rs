// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::path::PathBuf;

// Environment variable names (build.rs can't import from runtime crate)
const DYNAMO_FATBIN_PATH: &str = "DYNAMO_FATBIN_PATH";
const OUT_DIR: &str = "OUT_DIR";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Declare our custom cfg flag to avoid unexpected_cfgs warnings
    println!("cargo:rustc-check-cfg=cfg(have_vec_copy_fatbin)");

    println!("cargo:warning=Building with CUDA KV off");
    build_protos()?;

    // Get FATBIN path and copy it to OUT_DIR for embedding
    if let Some(fatbin_path) = find_fatbin_file() {
        // Copy FATBIN to OUT_DIR so we can include it with a predictable path
        let out_dir = env::var(OUT_DIR).unwrap();
        let dest_path = PathBuf::from(out_dir).join("vectorized_copy.fatbin");

        if let Err(e) = std::fs::copy(&fatbin_path, &dest_path) {
            println!("cargo:warning=Failed to copy FATBIN to OUT_DIR: {}", e);
        } else {
            // Emit cfg flag for conditional compilation
            println!("cargo:rustc-cfg=have_vec_copy_fatbin");
            println!(
                "cargo:warning=CUDA FATBIN found at: {} - copied to OUT_DIR",
                fatbin_path.display()
            );
        }

        // Tell cargo to rerun if FATBIN file changes
        println!("cargo:rerun-if-changed={}", fatbin_path.display());
    } else {
        println!(
            "cargo:warning=CUDA FATBIN not found - run 'make fatbin' in cuda_kernels directory"
        );
        println!("cargo:warning=Set DYNAMO_FATBIN_PATH env var to specify custom location");
    }

    // Rerun build if environment variable changes
    println!("cargo:rerun-if-env-changed=DYNAMO_FATBIN_PATH");

    Ok(())
}

fn build_protos() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .type_attribute(".", "#[derive(serde::Serialize,serde::Deserialize)]")
        .compile_protos(&["kserve.proto"], &["src/grpc/protos"])?;
    Ok(())
}

fn find_fatbin_file() -> Option<PathBuf> {
    // 1. Check if user specified custom path via environment variable
    if let Ok(custom_path) = env::var(DYNAMO_FATBIN_PATH) {
        let fatbin_file = PathBuf::from(custom_path);
        if fatbin_file.exists() {
            println!(
                "cargo:warning=Using custom FATBIN path: {}",
                fatbin_file.display()
            );
            return Some(fatbin_file);
        } else {
            println!(
                "cargo:warning=Custom FATBIN path does not exist: {}",
                fatbin_file.display()
            );
        }
    }

    // 2. Check standard locations (priority order)
    let default_paths = [
        "./src/block_manager/block/transfer/kernels/vectorized_copy.fatbin", // Primary: Next to transfer module
    ];

    for path in &default_paths {
        let fatbin_file = PathBuf::from(path);
        if fatbin_file.exists() {
            println!(
                "cargo:warning=Found FATBIN at default location: {}",
                fatbin_file.display()
            );
            return Some(fatbin_file);
        }
    }

    None
}

// NOTE: Preserving this build.rs for reference. We may want to re-enable
// custom kernel compilation in the future.

// #[cfg(not(feature = "cuda_kv"))]
// fn main() {}

// #[cfg(feature = "cuda_kv")]
// fn main() {
//     use std::{path::PathBuf, process::Command};

//     println!("cargo:rerun-if-changed=src/kernels/block_copy.cu");

//     // first do a which nvcc, if it is in the path
//     // if so, we don't need to set the cuda_lib
//     let nvcc = Command::new("which").arg("nvcc").output().unwrap();
//     let cuda_lib = if nvcc.status.success() {
//         println!("cargo:info=nvcc found in path");
//         // Extract the path from nvcc location by removing "bin/nvcc"
//         let nvcc_path = String::from_utf8_lossy(&nvcc.stdout).trim().to_string();
//         let path = PathBuf::from(nvcc_path);
//         if let Some(parent) = path.parent() {
//             // Remove "nvcc"
//             if let Some(cuda_root) = parent.parent() {
//                 // Remove "bin"
//                 cuda_root.to_string_lossy().to_string()
//             } else {
//                 // Fallback to CUDA_ROOT or default if path extraction fails
//                 get_cuda_root_or_default()
//             }
//         } else {
//             // Fallback to CUDA_ROOT or default if path extraction fails
//             get_cuda_root_or_default()
//         }
//     } else {
//         println!("cargo:warning=nvcc not found in path");
//         get_cuda_root_or_default()
//     };

//     println!("cargo:info=Using CUDA installation at: {}", cuda_lib);

//     let cuda_lib_path = PathBuf::from(&cuda_lib).join("lib64");
//     println!("cargo:info=Using CUDA libs: {}", cuda_lib_path.display());
//     println!("cargo:rustc-link-search=native={}", cuda_lib_path.display());

//     // Link against multiple CUDA libraries
//     println!("cargo:rustc-link-lib=dylib=cudart");
//     println!("cargo:rustc-link-lib=dylib=cuda");
//     println!("cargo:rustc-link-lib=dylib=cudadevrt");

//     // Make sure the CUDA libraries are found before other system libraries
//     println!(
//         "cargo:rustc-link-arg=-Wl,-rpath,{}",
//         cuda_lib_path.display()
//     );

//     // Create kernels directory for output if it doesn't exist
//     std::fs::create_dir_all("src/kernels").unwrap_or_else(|_| {
//         println!("Kernels directory already exists");
//     });

//     // Compile CUDA code
//     let output = Command::new("nvcc")
//         .arg("src/kernels/block_copy.cu")
//         .arg("-O3")
//         .arg("--compiler-options")
//         .arg("-fPIC")
//         .arg("-o")
//         .arg("src/kernels/libblock_copy.o")
//         .arg("-c")
//         .output()
//         .expect("Failed to compile CUDA code");

//     if !output.status.success() {
//         panic!(
//             "Failed to compile CUDA kernel: {}",
//             String::from_utf8_lossy(&output.stderr)
//         );
//     }

//     // Create static library
//     #[cfg(target_os = "windows")]
//     {
//         Command::new("lib")
//             .arg("/OUT:src/kernels/block_copy.lib")
//             .arg("src/kernels/libblock_copy.o")
//             .output()
//             .expect("Failed to create static library");
//         println!("cargo:rustc-link-search=native=src/kernels");
//         println!("cargo:rustc-link-lib=static=block_copy");
//     }

//     #[cfg(not(target_os = "windows"))]
//     {
//         Command::new("ar")
//             .arg("rcs")
//             .arg("src/kernels/libblock_copy.a")
//             .arg("src/kernels/libblock_copy.o")
//             .output()
//             .expect("Failed to create static library");
//         println!("cargo:rustc-link-search=native=src/kernels");
//         println!("cargo:rustc-link-lib=static=block_copy");
//         println!("cargo:rustc-link-lib=dylib=cudart");
//         println!("cargo:rustc-link-lib=dylib=cuda");
//         println!("cargo:rustc-link-lib=dylib=cudadevrt");
//     }
// }

// #[cfg(feature = "cuda_kv")]
// fn get_cuda_root_or_default() -> String {
//     match std::env::var("CUDA_ROOT") {
//         Ok(path) => path,
//         Err(_) => {
//             // Default locations based on OS
//             if cfg!(target_os = "windows") {
//                 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8".to_string()
//             } else {
//                 "/usr/local/cuda".to_string()
//             }
//         }
//     }
// }
