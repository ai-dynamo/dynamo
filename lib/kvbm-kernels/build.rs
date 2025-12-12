// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Declare the stub_kernels cfg so Rust knows it's a valid cfg option
    println!("cargo::rustc-check-cfg=cfg(stub_kernels)");

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = env::var("OUT_DIR").unwrap();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_else(|_| "unknown".to_string());

    // Track file changes
    let cu_files = discover_cuda_files();
    for file in &cu_files {
        println!("cargo:rerun-if-changed={}", file.display());
    }
    println!(
        "cargo:rerun-if-changed={}",
        Path::new(&manifest_dir).join("cuda/stubs.c").display()
    );
    println!("cargo:rerun-if-env-changed=CUDA_ARCHS");
    println!("cargo:rerun-if-env-changed=KVBM_REQUIRE_CUDA");

    println!("cargo:warning=Target architecture: {}", target_arch);

    // Check if CUDA is required (set by Python bindings build)
    let require_cuda = env::var("KVBM_REQUIRE_CUDA").is_ok();
    let nvcc_available = is_nvcc_available();

    // Fail early if CUDA required but not available
    if require_cuda && !nvcc_available {
        panic!(
            "\n\n\
            ╔════════════════════════════════════════════════════════════════════════╗\n\
            ║  KVBM_REQUIRE_CUDA is set but nvcc is not available!                   ║\n\
            ║                                                                        ║\n\
            ║  Python bindings require real CUDA kernels. Please:                    ║\n\
            ║    1. Install CUDA toolkit with nvcc, or                               ║\n\
            ║    2. Unset KVBM_REQUIRE_CUDA for stub-only build                      ║\n\
            ╚════════════════════════════════════════════════════════════════════════╝\n\
            "
        );
    }

    // Determine build mode
    let build_mode = determine_build_mode(nvcc_available);

    // Check if static linking is requested (only applies to CUDA builds, not stubs)
    #[cfg(feature = "static-kernels")]
    let use_static = true;
    #[cfg(not(feature = "static-kernels"))]
    let use_static = false;

    match build_mode {
        BuildMode::FromSource => {
            if use_static {
                println!("cargo:warning=Building CUDA kernels from source (static linking)");
            } else {
                println!("cargo:warning=Building CUDA kernels from source (dynamic linking)");
            }
            build_cuda_library(&cu_files, &out_dir, &target_arch, use_static);
        }
        BuildMode::Prebuilt => {
            if use_static {
                println!("cargo:warning=Using prebuilt CUDA kernels (static linking)");
            } else {
                println!("cargo:warning=Using prebuilt CUDA kernels (dynamic linking)");
            }
            use_prebuilt_library(&out_dir, &target_arch, use_static);
        }
        BuildMode::Stubs => {
            // Stubs always use dynamic linking regardless of static-kernels feature
            println!("cargo:warning=Building stub kernels (no CUDA available, dynamic linking)");
            build_stub_shared_library(&manifest_dir, &out_dir);
            // Set cfg flag so tests can be skipped
            println!("cargo:rustc-cfg=stub_kernels");
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum BuildMode {
    FromSource,
    Prebuilt,
    Stubs,
}

/// Determine the build mode based on feature flags and nvcc availability.
fn determine_build_mode(nvcc_available: bool) -> BuildMode {
    // Check feature flag first
    #[cfg(feature = "prebuilt-kernels")]
    {
        if has_prebuilt_library() {
            return BuildMode::Prebuilt;
        }
        println!("cargo:warning=prebuilt-kernels feature set but no prebuilt library found");
    }

    if nvcc_available {
        BuildMode::FromSource
    } else if has_prebuilt_library() {
        BuildMode::Prebuilt
    } else {
        BuildMode::Stubs
    }
}

fn is_nvcc_available() -> bool {
    Command::new("nvcc").arg("--version").output().is_ok()
}

/// Check if prebuilt library exists.
/// When `static-kernels` feature is enabled, checks for .a; otherwise checks for .so.
fn has_prebuilt_library() -> bool {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    #[cfg(feature = "static-kernels")]
    {
        let a_path = Path::new(&manifest_dir).join("cuda/prebuilt/libkvbm_kernels.a");
        a_path.exists()
    }

    #[cfg(not(feature = "static-kernels"))]
    {
        let so_path = Path::new(&manifest_dir).join("cuda/prebuilt/libkvbm_kernels.so");
        so_path.exists()
    }
}

/// Build CUDA kernels from source.
/// If `use_static` is true, builds a static archive (.a); otherwise builds a shared library (.so).
fn build_cuda_library(cu_files: &[PathBuf], out_dir: &str, target_arch: &str, use_static: bool) {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let arch_flags = get_cuda_arch_flags();

    // Only build tensor_kernels.cu into the library (it has the extern "C" functions)
    let tensor_kernels_path = cu_files
        .iter()
        .find(|p| p.file_stem().unwrap() == "tensor_kernels")
        .expect("tensor_kernels.cu not found");

    let obj_path = Path::new(out_dir).join("kvbm_kernels.o");

    // Step 1: Compile to object file
    let mut nvcc_cmd = Command::new("nvcc");
    nvcc_cmd
        .arg("-m64")
        .arg("-c")
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-Xcompiler")
        .arg("-fPIC")
        .arg(tensor_kernels_path)
        .arg("-o")
        .arg(&obj_path);

    for flag in &arch_flags {
        nvcc_cmd.arg(flag);
    }

    println!("cargo:warning=Compiling tensor_kernels.cu to object file...");
    let status = nvcc_cmd
        .status()
        .expect("Failed to execute nvcc for object file");

    if !status.success() {
        panic!("nvcc failed to compile tensor_kernels.cu");
    }

    if use_static {
        // Step 2a: Create static archive
        let ar_path = Path::new(out_dir).join("libkvbm_kernels.a");

        let mut ar_cmd = Command::new("ar");
        ar_cmd
            .arg("crus")
            .arg(&ar_path)
            .arg(&obj_path);

        println!("cargo:warning=Creating static archive libkvbm_kernels.a...");
        let status = ar_cmd
            .status()
            .expect("Failed to execute ar for static archive");

        if !status.success() {
            panic!("ar failed to create static archive");
        }

        // Set up static linking
        println!("cargo:rustc-link-search=native={}", out_dir);
        println!("cargo:rustc-link-lib=static=kvbm_kernels");

        // Add CUDA runtime library paths and link cudart dynamically
        add_cuda_library_paths();
        println!("cargo:rustc-link-lib=cudart");
    } else {
        // Step 2b: Link into shared library
        let so_name = "libkvbm_kernels.so";
        let so_path = Path::new(out_dir).join(so_name);

        let mut link_cmd = Command::new("nvcc");
        link_cmd
            .arg("-shared")
            .arg("-o")
            .arg(&so_path)
            .arg(&obj_path)
            .arg("-lcudart");

        println!("cargo:warning=Linking kvbm_kernels into shared library...");
        let status = link_cmd
            .status()
            .expect("Failed to execute nvcc for linking");

        if !status.success() {
            panic!("nvcc failed to link shared library");
        }

        // Generate prebuilt artifacts only when the feature is enabled.
        // The .so alone is sufficient for normal operation - fatbins are only needed
        // for runtime kernel loading which we don't currently use.
        #[cfg(feature = "generate-prebuilt")]
        {
            // Generate fatbin files for all .cu files
            for cu_file in cu_files {
                let kernel_name = cu_file.file_stem().unwrap().to_str().unwrap();
                generate_fatbin(cu_file, &arch_flags, out_dir);

                // Copy fatbin to prebuilt directory for future use
                if target_arch == "x86_64" {
                    copy_to_prebuilt(cu_file, out_dir, &manifest_dir);
                }

                println!("cargo:warning=Generated fatbin for {}", kernel_name);
            }

            // Copy .so to prebuilt directory for future use (x86_64 only)
            if target_arch == "x86_64" {
                let prebuilt_dir = Path::new(&manifest_dir).join("cuda/prebuilt");
                fs::create_dir_all(&prebuilt_dir).expect("Failed to create prebuilt directory");
                let prebuilt_so = prebuilt_dir.join(so_name);
                fs::copy(&so_path, &prebuilt_so).expect("Failed to copy .so to prebuilt");
                println!(
                    "cargo:warning=Copied {} to prebuilt directory",
                    prebuilt_so.display()
                );
            }
        }

        #[cfg(not(feature = "generate-prebuilt"))]
        {
            // Suppress unused variable warnings when feature is disabled
            let _ = cu_files;
            let _ = target_arch;
            let _ = manifest_dir;
        }

        // Set up dynamic linking
        println!("cargo:rustc-link-search=native={}", out_dir);
        println!("cargo:rustc-link-lib=dylib=kvbm_kernels");

        // Add CUDA runtime library paths
        add_cuda_library_paths();
        println!("cargo:rustc-link-lib=cudart");
    }
}

/// Use prebuilt library.
/// If `use_static` is true, looks for .a archive; otherwise uses .so shared library.
fn use_prebuilt_library(out_dir: &str, _target_arch: &str, use_static: bool) {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let prebuilt_dir = Path::new(&manifest_dir).join("cuda/prebuilt");

    if use_static {
        let prebuilt_a = prebuilt_dir.join("libkvbm_kernels.a");

        if !prebuilt_a.exists() {
            panic!(
                "Prebuilt static library not found at {}",
                prebuilt_a.display()
            );
        }

        // Copy to OUT_DIR
        let out_a = Path::new(out_dir).join("libkvbm_kernels.a");
        fs::copy(&prebuilt_a, &out_a).expect("Failed to copy prebuilt .a");

        // Set up static linking
        println!("cargo:rustc-link-search=native={}", out_dir);
        println!("cargo:rustc-link-lib=static=kvbm_kernels");

        // Add CUDA runtime library paths and link cudart dynamically
        add_cuda_library_paths();
        println!("cargo:rustc-link-lib=cudart");
    } else {
        let prebuilt_so = prebuilt_dir.join("libkvbm_kernels.so");

        if !prebuilt_so.exists() {
            panic!(
                "Prebuilt shared library not found at {}",
                prebuilt_so.display()
            );
        }

        // Copy to OUT_DIR
        let out_so = Path::new(out_dir).join("libkvbm_kernels.so");
        fs::copy(&prebuilt_so, &out_so).expect("Failed to copy prebuilt .so");

        // Also copy fatbin files
        for entry in fs::read_dir(&prebuilt_dir).expect("Failed to read prebuilt directory") {
            let entry = entry.expect("Failed to read entry");
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "fatbin") {
                let dest = Path::new(out_dir).join(path.file_name().unwrap());
                fs::copy(&path, &dest).expect("Failed to copy fatbin");
            }
        }

        // Set up dynamic linking
        println!("cargo:rustc-link-search=native={}", out_dir);
        println!("cargo:rustc-link-lib=dylib=kvbm_kernels");

        // Add CUDA runtime library paths
        add_cuda_library_paths();
        println!("cargo:rustc-link-lib=cudart");
    }
}

/// Build stub shared library from stubs.c when CUDA is not available.
fn build_stub_shared_library(manifest_dir: &str, out_dir: &str) {
    let stubs_path = Path::new(manifest_dir).join("cuda/stubs.c");

    if !stubs_path.exists() {
        panic!(
            "Stub source file not found at {}. Cannot build without CUDA.",
            stubs_path.display()
        );
    }

    // Build shared library from stubs.c using the system C compiler
    let so_path = Path::new(out_dir).join("libkvbm_kernels.so");
    let obj_path = Path::new(out_dir).join("stubs.o");

    // Compile to object file
    let mut gcc_compile = Command::new("cc");
    gcc_compile
        .arg("-c")
        .arg("-fPIC")
        .arg("-O2")
        .arg(&stubs_path)
        .arg("-o")
        .arg(&obj_path);

    println!("cargo:warning=Compiling stubs.c...");
    let status = gcc_compile
        .status()
        .expect("Failed to execute cc for stubs");

    if !status.success() {
        panic!("Failed to compile stubs.c");
    }

    // Link into shared library
    let mut gcc_link = Command::new("cc");
    gcc_link
        .arg("-shared")
        .arg("-o")
        .arg(&so_path)
        .arg(&obj_path);

    println!("cargo:warning=Linking stub shared library...");
    let status = gcc_link.status().expect("Failed to link stub library");

    if !status.success() {
        panic!("Failed to link stub shared library");
    }

    // Set up linking
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=dylib=kvbm_kernels");
}

fn discover_cuda_files() -> Vec<PathBuf> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let cuda_dir = Path::new(&manifest_dir).join("cuda");
    let mut cu_files = Vec::new();

    for entry in fs::read_dir(cuda_dir).expect("Failed to read cuda directory") {
        let entry = entry.expect("Failed to read entry");
        let path = entry.path();
        if path.extension().map_or(false, |ext| ext == "cu") {
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
    // Generate PTX for Hopper and Blackwell family
    flags.push("-gencode=arch=compute_90,code=compute_90".to_string());
    flags.push("-gencode=arch=compute_100,code=compute_100".to_string());
    flags.push("-gencode=arch=compute_120,code=compute_120".to_string());

    flags
}

fn add_cuda_library_paths() {
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
}

#[cfg(feature = "generate-prebuilt")]
#[allow(dead_code)]
fn generate_fatbin(cu_path: &Path, arch_flags: &[String], out_dir: &str) {
    let kernel_name = cu_path.file_stem().unwrap().to_str().unwrap();
    let fatbin_path = Path::new(out_dir).join(format!("{}.fatbin", kernel_name));

    let mut nvcc_cmd = Command::new("nvcc");
    nvcc_cmd
        .arg("-m64")
        .arg("-fatbin")
        .arg("-std=c++17")
        .arg("-O3")
        .arg(cu_path)
        .arg("-o")
        .arg(&fatbin_path);

    for flag in arch_flags {
        nvcc_cmd.arg(flag);
    }

    let status = nvcc_cmd
        .status()
        .expect("Failed to execute nvcc for fatbin");

    if !status.success() {
        panic!("nvcc failed to generate fatbin for {}", kernel_name);
    }
}

#[cfg(feature = "generate-prebuilt")]
#[allow(dead_code)]
fn copy_to_prebuilt(cu_path: &Path, out_dir: &str, manifest_dir: &str) {
    let kernel_name = cu_path.file_stem().unwrap().to_str().unwrap();
    let prebuilt_dir = Path::new(manifest_dir).join("cuda/prebuilt");
    fs::create_dir_all(&prebuilt_dir).expect("Failed to create prebuilt directory");

    // Copy fatbin
    let fatbin_src = Path::new(out_dir).join(format!("{}.fatbin", kernel_name));
    let fatbin_dst = prebuilt_dir.join(format!("{}.fatbin", kernel_name));
    if fatbin_src.exists() {
        fs::copy(&fatbin_src, &fatbin_dst).expect("Failed to copy fatbin to prebuilt");
    }

    // Generate and save hash
    let md5_path = prebuilt_dir.join(format!("{}.md5", kernel_name));
    let cu_hash = compute_file_hash(cu_path);
    if fatbin_src.exists() {
        let fatbin_hash = compute_file_hash(&fatbin_src);
        let hashes = format!("{}\n{}\n", cu_hash, fatbin_hash);
        fs::write(&md5_path, hashes).expect("Failed to write md5 file");
    }
}

#[cfg(feature = "generate-prebuilt")]
#[allow(dead_code)]
fn compute_file_hash(path: &Path) -> String {
    use std::io::Read;

    let mut file = fs::File::open(path)
        .unwrap_or_else(|e| panic!("Failed to open {} for hashing: {}", path.display(), e));

    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .unwrap_or_else(|e| panic!("Failed to read {} for hashing: {}", path.display(), e));

    format!("{:x}", md5::compute(&buffer))
}
