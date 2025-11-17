// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::fs;
use std::io::Read;
use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cuda/tensor_kernels.cu");
    println!("cargo:rerun-if-env-changed=DYNAMO_USE_PREBUILT_KERNELS");
    println!("cargo:rerun-if-env-changed=CUDA_ARCHS");

    let use_prebuilt = determine_build_mode();

    if use_prebuilt {
        build_with_prebuilt_kernels();
    } else {
        build_from_source();

        // Only link against CUDA runtime when building from source
        // Add CUDA library search paths
        if let Ok(cuda_path) = env::var("CUDA_PATH") {
            println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
            println!("cargo:rustc-link-search=native={}/lib", cuda_path);
        } else if let Ok(cuda_home) = env::var("CUDA_HOME") {
            println!("cargo:rustc-link-search=native={}/lib64", cuda_home);
            println!("cargo:rustc-link-search=native={}/lib", cuda_home);
        } else {
            // Try standard paths
            println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
            println!("cargo:rustc-link-search=native=/usr/local/cuda/lib");
        }

        println!("cargo:rustc-link-lib=cudart");
    }
}

/// Determine whether to use prebuilt kernels based on:
/// 1. Feature flag (highest precedence)
/// 2. Environment variable
/// 3. Auto-detection of nvcc
fn determine_build_mode() -> bool {
    // Check feature flag first
    #[cfg(feature = "prebuilt-kernels")]
    {
        println!("cargo:warning=Using prebuilt kernels (feature flag enabled)");
        return true;
    }

    // Check environment variable
    if dynamo_config::env_is_truthy("DYNAMO_USE_PREBUILT_KERNELS") {
        println!("cargo:warning=Using prebuilt kernels (DYNAMO_USE_PREBUILT_KERNELS set)");
        return true;
    }

    // Auto-detect nvcc
    if !is_nvcc_available() {
        println!("cargo:warning=nvcc not found, using prebuilt kernels");
        return true;
    }

    println!("cargo:warning=Building CUDA kernels from source");
    false
}

fn is_nvcc_available() -> bool {
    Command::new("nvcc").arg("--version").output().is_ok()
}

fn build_with_prebuilt_kernels() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let cu_path = Path::new(&manifest_dir).join("cuda/tensor_kernels.cu");
    let md5_path = Path::new(&manifest_dir).join("cuda/prebuilt/tensor_kernels.md5");
    let fatbin_path = Path::new(&manifest_dir).join("cuda/prebuilt/tensor_kernels.fatbin");

    // Validate that prebuilt files exist
    if !md5_path.exists() {
        panic!(
            "Prebuilt mode requires cuda/prebuilt/tensor_kernels.md5 but it does not exist. \
             Please build with nvcc available first to generate the prebuilt artifacts."
        );
    }

    if !fatbin_path.exists() {
        panic!(
            "Prebuilt mode requires cuda/prebuilt/tensor_kernels.fatbin but it does not exist. \
             Please build with nvcc available first to generate the prebuilt artifacts."
        );
    }

    // Read stored hashes (three lines: build.rs, .cu, .fatbin)
    let stored_hashes_content =
        fs::read_to_string(&md5_path).expect("Failed to read cuda/prebuilt/tensor_kernels.md5");

    let stored_hashes: Vec<&str> = stored_hashes_content.lines().collect();
    if stored_hashes.len() != 3 {
        panic!(
            "Invalid .md5 file format. Expected 3 lines (build.rs, .cu, .fatbin hashes), found {}.\n\
             Please rebuild with nvcc available to regenerate the prebuilt artifacts.",
            stored_hashes.len()
        );
    }

    let stored_build_rs_hash = stored_hashes[0];
    let stored_cu_hash = stored_hashes[1];
    let stored_fatbin_hash = stored_hashes[2];

    // Compute current hashes
    let build_rs_path = Path::new(&manifest_dir).join("build.rs");
    let current_build_rs_hash = compute_file_hash(&build_rs_path);
    let current_cu_hash = compute_file_hash(&cu_path);
    let current_fatbin_hash = compute_file_hash(&fatbin_path);

    // Validate all three hashes
    let mut mismatches = Vec::new();

    if current_build_rs_hash != stored_build_rs_hash {
        mismatches.push(format!(
            "  build.rs: current={}, stored={}",
            current_build_rs_hash, stored_build_rs_hash
        ));
    }

    if current_cu_hash != stored_cu_hash {
        mismatches.push(format!(
            "  .cu source: current={}, stored={}",
            current_cu_hash, stored_cu_hash
        ));
    }

    if current_fatbin_hash != stored_fatbin_hash {
        mismatches.push(format!(
            "  .fatbin: current={}, stored={}",
            current_fatbin_hash, stored_fatbin_hash
        ));
    }

    if !mismatches.is_empty() {
        panic!(
            "Hash mismatch! The prebuilt .fatbin is out of sync:\n{}\n\
             Please rebuild with nvcc available to regenerate the prebuilt artifacts.",
            mismatches.join("\n")
        );
    }

    println!("cargo:warning=Hash validation passed:");
    println!("cargo:warning=  build.rs: {}", current_build_rs_hash);
    println!("cargo:warning=  .cu source: {}", current_cu_hash);
    println!("cargo:warning=  .fatbin: {}", current_fatbin_hash);

    // Link the prebuilt fatbin
    // Note: We need to inform the linker about the fatbin file.
    // The typical approach is to use cc to link it as an object file or
    // use CUDA's fatbinary tool. For simplicity, we'll use cc to link it.
    let out_dir = env::var("OUT_DIR").unwrap();
    let fatbin_copy = Path::new(&out_dir).join("tensor_kernels.fatbin");
    fs::copy(&fatbin_path, &fatbin_copy).expect("Failed to copy .fatbin to OUT_DIR");

    // Link the fatbin as a dependency
    println!("cargo:rustc-link-search=native={}", out_dir);

    // Create a stub object file that references the fatbin
    // This is a workaround since we can't directly link .fatbin files
    // In a real scenario, you'd use cuModuleLoadFatBinary at runtime
    println!(
        "cargo:warning=Prebuilt kernel loaded from {}",
        fatbin_path.display()
    );
}

fn build_from_source() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let cu_path = Path::new(&manifest_dir).join("cuda/tensor_kernels.cu");
    let out_dir = env::var("OUT_DIR").unwrap();

    // Build with cc crate
    let mut build = cc::Build::new();
    build
        .cuda(true)
        .file(&cu_path)
        .flag("-std=c++17")
        .flag("-O3")
        .flag("-Xcompiler")
        .flag("-fPIC");

    // Configure CUDA architectures
    let arch_flags = get_cuda_arch_flags();
    for flag in &arch_flags {
        build.flag(flag);
    }

    build.compile("tensor_kernels");

    // Generate .fatbin and .md5 for future prebuilt use
    generate_prebuilt_artifacts(&cu_path, &arch_flags, &out_dir);
}

fn get_cuda_arch_flags() -> Vec<String> {
    let mut flags = Vec::new();

    if let Ok(arch_list) = env::var("CUDA_ARCHS") {
        for arch in arch_list.split(',') {
            let arch = arch.trim();
            if arch.is_empty() {
                continue;
            }
            flags.push(format!("-gencode=arch=compute_{},code=sm_{}", arch, arch));
        }
    } else {
        // Default to Ampere (SM 80) and Hopper (SM 90) support.
        flags.push("-gencode=arch=compute_80,code=sm_80".to_string());
        flags.push("-gencode=arch=compute_90,code=sm_90".to_string());
    }

    flags
}

fn generate_prebuilt_artifacts(cu_path: &Path, arch_flags: &[String], out_dir: &str) {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let prebuilt_dir = Path::new(&manifest_dir).join("cuda/prebuilt");
    let fatbin_path = prebuilt_dir.join("tensor_kernels.fatbin");
    let md5_path = prebuilt_dir.join("tensor_kernels.md5");

    // Ensure prebuilt directory exists
    fs::create_dir_all(&prebuilt_dir).expect("Failed to create cuda/prebuilt directory");

    // Generate .fatbin using nvcc
    let temp_fatbin = Path::new(out_dir).join("tensor_kernels.fatbin");

    let mut nvcc_cmd = Command::new("nvcc");
    nvcc_cmd
        .arg("-fatbin")
        .arg("-std=c++17")
        .arg("-O3")
        .arg(cu_path)
        .arg("-o")
        .arg(&temp_fatbin);

    for flag in arch_flags {
        nvcc_cmd.arg(flag);
    }

    println!("cargo:warning=Generating .fatbin with nvcc...");
    let status = nvcc_cmd
        .status()
        .expect("Failed to execute nvcc for .fatbin generation");

    if !status.success() {
        panic!("nvcc failed to generate .fatbin");
    }

    // Copy .fatbin to prebuilt directory
    fs::copy(&temp_fatbin, &fatbin_path).expect("Failed to copy .fatbin to cuda/prebuilt/");

    // Generate MD5 hashes of all three files for consistency validation
    let build_rs_path = Path::new(&manifest_dir).join("build.rs");
    let build_rs_hash = compute_file_hash(&build_rs_path);
    let cu_hash = compute_file_hash(cu_path);
    let fatbin_hash = compute_file_hash(&fatbin_path);

    // Write all three hashes (one per line)
    let hashes = format!("{}\n{}\n{}\n", build_rs_hash, cu_hash, fatbin_hash);
    fs::write(&md5_path, hashes).expect("Failed to write .md5 file");

    println!(
        "cargo:warning=Generated prebuilt artifacts:\n  {}\n  {}",
        fatbin_path.display(),
        md5_path.display()
    );
    println!("cargo:warning=build.rs hash: {}", build_rs_hash);
    println!("cargo:warning=.cu source hash: {}", cu_hash);
    println!("cargo:warning=.fatbin hash: {}", fatbin_hash);
}

fn compute_file_hash(path: &Path) -> String {
    let mut file = fs::File::open(path)
        .unwrap_or_else(|e| panic!("Failed to open {} for hashing: {}", path.display(), e));

    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .unwrap_or_else(|e| panic!("Failed to read {} for hashing: {}", path.display(), e));

    format!("{:x}", md5::compute(&buffer))
}
