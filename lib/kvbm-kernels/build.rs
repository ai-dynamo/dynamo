// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::fs;
use std::io::Read;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let cu_files = discover_cuda_files();
    for file in cu_files {
        println!("cargo:rerun-if-changed={}", file.display());
    }
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
    let out_dir = env::var("OUT_DIR").unwrap();
    let cu_files = discover_cuda_files();

    for cu_path in &cu_files {
        let kernel_name = cu_path.file_stem().unwrap().to_str().unwrap();
        let md5_path = Path::new(&manifest_dir).join(format!("cuda/prebuilt/{}.md5", kernel_name));
        let fatbin_path =
            Path::new(&manifest_dir).join(format!("cuda/prebuilt/{}.fatbin", kernel_name));

        // Validate prebuilt files exist
        if !md5_path.exists() {
            panic!(
                "Prebuilt mode requires cuda/prebuilt/{}.md5 but it doesn't exist. \
                 Build with nvcc first.",
                kernel_name
            );
        }
        if !fatbin_path.exists() {
            panic!(
                "Prebuilt mode requires cuda/prebuilt/{}.fatbin but it doesn't exist. \
                 Build with nvcc first.",
                kernel_name
            );
        }

        // Read and validate hashes (only .cu and .fatbin, not build.rs)
        let stored_hashes_content = fs::read_to_string(&md5_path)
            .unwrap_or_else(|_| panic!("Failed to read {}", md5_path.display()));
        let stored_hashes: Vec<&str> = stored_hashes_content.lines().collect();

        if stored_hashes.len() != 2 {
            panic!(
                "Invalid .md5 format for {} (expected 2 lines: .cu hash, .fatbin hash)",
                kernel_name
            );
        }

        let current_cu_hash = compute_file_hash(cu_path);
        let current_fatbin_hash = compute_file_hash(&fatbin_path);

        // Validate hashes
        if current_cu_hash != stored_hashes[0] || current_fatbin_hash != stored_hashes[1] {
            panic!(
                "Hash mismatch for {}! Rebuild with nvcc.\n  .cu: current={}, stored={}\n  .fatbin: current={}, stored={}",
                kernel_name,
                current_cu_hash,
                stored_hashes[0],
                current_fatbin_hash,
                stored_hashes[1]
            );
        }

        // Copy fatbin to OUT_DIR
        let fatbin_copy = Path::new(&out_dir).join(format!("{}.fatbin", kernel_name));
        fs::copy(&fatbin_path, &fatbin_copy).expect("Failed to copy .fatbin");

        println!("cargo:warning=Loaded prebuilt kernel: {}", kernel_name);
    }

    println!("cargo:rustc-link-search=native={}", out_dir);
}

fn build_from_source() {
    let cu_files = discover_cuda_files();
    let out_dir = env::var("OUT_DIR").unwrap();

    // Build with cc crate
    let mut build = cc::Build::new();
    build
        .cuda(true)
        .flag("-std=c++17")
        .flag("-O3")
        .flag("-Xcompiler")
        .flag("-fPIC");

    // Configure CUDA architectures
    let arch_flags = get_cuda_arch_flags();
    for file in &cu_files {
        build.file(file);
        println!("cargo:rerun-if-changed={}", file.display());
    }
    for flag in &arch_flags {
        build.flag(flag);
    }

    build.compile("kvbm_kernels");

    // Generate .fatbin and .md5 for future prebuilt use
    for file in &cu_files {
        generate_prebuilt_artifacts(file, &arch_flags, &out_dir);
    }
}

fn discover_cuda_files() -> Vec<PathBuf> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let cuda_dir = Path::new(&manifest_dir).join("cuda");
    let mut cu_files = Vec::new();

    for entry in fs::read_dir(cuda_dir).expect("Failed to read cuda directory") {
        let entry = entry.expect("Failed to read entry");
        let path = entry.path();
        if path.extension().unwrap_or_default() == "cu" {
            cu_files.push(path);
        }
    }
    cu_files
}

fn get_cuda_arch_flags() -> Vec<String> {
    let mut flags = Vec::new();

    let arch_list = env::var("CUDA_ARCHS").unwrap_or_else(|_| "80,86,89,90,100,120".to_string());

    for arch in arch_list.split(',') {
        let arch = arch.trim();
        if arch.is_empty() {
            continue;
        }
        flags.push(format!("-gencode=arch=compute_{},code=sm_{}", arch, arch));
    }
    // GEnerate PTX for Hopper and Blackwell family
    flags.push("-gencode=arch=compute_90,code=compute_90".to_string());
    flags.push("-gencode=arch=compute_100,code=compute_100".to_string());
    flags.push("-gencode=arch=compute_120,code=compute_120".to_string());

    flags
}

fn generate_prebuilt_artifacts(cu_path: &Path, arch_flags: &[String], out_dir: &str) {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let prebuilt_dir = Path::new(&manifest_dir).join("cuda/prebuilt");
    let kernel_name = cu_path.file_stem().unwrap().to_str().unwrap();
    let fatbin_path = prebuilt_dir.join(format!("{}.fatbin", kernel_name));
    let md5_path = prebuilt_dir.join(format!("{}.md5", kernel_name));
    // Generate .fatbin using nvcc
    let temp_fatbin = Path::new(out_dir).join(format!("{}.fatbin", kernel_name));

    // Ensure prebuilt directory exists
    fs::create_dir_all(&prebuilt_dir).expect("Failed to create cuda/prebuilt directory");

    let mut nvcc_cmd = Command::new("nvcc");
    nvcc_cmd
        .arg("-m64")
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

    // Generate MD5 hashes for consistency validation (only .cu and .fatbin)
    let cu_hash = compute_file_hash(cu_path);
    let fatbin_hash = compute_file_hash(&fatbin_path);

    // Write hashes (one per line: .cu hash, .fatbin hash)
    let hashes = format!("{}\n{}\n", cu_hash, fatbin_hash);
    fs::write(&md5_path, hashes).expect("Failed to write .md5 file");

    println!(
        "cargo:warning=Generated prebuilt artifacts:\n  {}\n  {}",
        fatbin_path.display(),
        md5_path.display()
    );
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
