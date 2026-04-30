// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic tool for validating NUMA page placement of pinned memory.
//!
//! On a multi-socket machine with multiple GPUs, this binary:
//! 1. Enumerates all visible devices on the chosen backend (CUDA or SYCL/XPU)
//! 2. Maps each device to its expected NUMA node (via PCI BDF → sysfs,
//!    with `nvidia-smi` / `xpu-smi` fallbacks)
//! 3. Allocates pinned host memory via `DeviceContext::allocate_pinned`
//!    (which routes through `NumaWorkerPool` + the backend's
//!    `PinnedAllocator`, and first-touches pages on the pinned worker
//!    thread before returning the pointer)
//! 4. Uses the `move_pages(2)` syscall to query actual page NUMA placement
//! 5. Reports match/mismatch statistics per device
//!
//! # Usage
//!
//! The binary resolves the backend in two steps: **compile time**
//! (what `--features` selects) and **runtime** (what `--backend`
//! selects). Only a backend that was compiled in can be chosen at
//! runtime; the binary refuses otherwise and prints a rebuild hint.
//!
//! ```bash
//! # CUDA host (default features include `cuda`). Auto-detect picks CUDA:
//! cargo run -p kvbm-physical --bin validate_numa_placement
//!
//! # CUDA host, larger allocation, subset of devices:
//! cargo run -p kvbm-physical --bin validate_numa_placement -- \
//!     --size 64 --gpus 0,2
//!
//! # Intel XPU host — requires BOTH the `xpu-sycl` feature AND the
//! # KVBM_ENABLE_XPU_KERNELS env var at build time (the latter triggers
//! # `icpx -fsycl` in `kvbm-kernels` to produce libkvbm_kernels_xpu.so,
//! # which `xpu-sycl` links against):
//! KVBM_ENABLE_XPU_KERNELS=1 \
//!     cargo run -p kvbm-physical \
//!         --no-default-features --features xpu-sycl \
//!         --bin validate_numa_placement -- \
//!         --backend sycl --size 64 --gpus 0,2
//!
//! # Mixed host with both backends compiled in:
//! KVBM_ENABLE_XPU_KERNELS=1 \
//!     cargo run -p kvbm-physical --features cuda,xpu-sycl \
//!         --bin validate_numa_placement -- --backend sycl
//! ```

use std::process;
use std::sync::Arc;

use kvbm_physical::device::{DeviceBackend, DeviceContext};

/// Query the NUMA node of each page in a memory region using `move_pages(2)`.
///
/// `move_pages(pid=0, count, pages, nodes=NULL, status, flags=0)` fills `status`
/// with the current NUMA node of each page without moving anything.
///
/// Returns a Vec of NUMA node IDs (one per page); entries are negative on
/// syscall failure (the kernel uses -errno per page in some edge cases).
fn query_page_nodes(ptr: *const u8, size: usize) -> Vec<i32> {
    let page_size = unsafe {
        let ps = libc::sysconf(libc::_SC_PAGESIZE);
        if ps > 0 { ps as usize } else { 4096 }
    };

    let num_pages = size.div_ceil(page_size);
    if num_pages == 0 {
        return Vec::new();
    }

    // Build array of page-aligned pointers
    let pages: Vec<*const libc::c_void> = (0..num_pages)
        .map(|i| unsafe { ptr.add(i * page_size) as *const libc::c_void })
        .collect();

    let mut status: Vec<i32> = vec![-1; num_pages];

    let ret = unsafe {
        libc::syscall(
            libc::SYS_move_pages,
            0i32,                       // pid = 0 (self)
            num_pages as libc::c_ulong, // count
            pages.as_ptr(),             // pages
            std::ptr::null::<i32>(),    // nodes = NULL (query mode)
            status.as_mut_ptr(),        // status (output)
            0i32,                       // flags
        )
    };

    if ret != 0 {
        let errno = std::io::Error::last_os_error();
        eprintln!("  move_pages syscall failed: {errno}");
        return vec![-1; num_pages];
    }

    status
}

/// Backend-agnostic device info for the "which GPUs to iterate" phase.
struct DeviceEntry {
    /// Backend-local ordinal (what `DeviceContext::new` takes).
    id: u32,
    /// PCI BDF if we can learn it here — otherwise `DeviceContext` will
    /// be asked later; we just use it for pre-phase logging.
    pci_bdf: Option<String>,
}

/// Enumerate every visible device on a given backend.
///
/// Returns `Err` with a human-readable message if the backend is not
/// compiled in, the runtime is missing, or enumeration fails.
fn enumerate_devices(backend: DeviceBackend) -> Result<Vec<DeviceEntry>, String> {
    match backend {
        DeviceBackend::Cuda => enumerate_cuda(),
        DeviceBackend::Sycl => enumerate_sycl(),
    }
}

#[cfg(feature = "cuda")]
fn enumerate_cuda() -> Result<Vec<DeviceEntry>, String> {
    cudarc::driver::result::init().map_err(|e| format!("cudarc init: {e}"))?;
    let count = cudarc::driver::result::device::get_count()
        .map_err(|e| format!("cuDeviceGetCount: {e}"))?;
    if count == 0 {
        return Err("No CUDA devices found".to_string());
    }
    // PCI BDF lookup for CUDA is done inside DeviceContext::pci_bdf_address()
    // during the per-device phase; we return None here to keep this block
    // minimal.
    Ok((0..count as u32)
        .map(|id| DeviceEntry { id, pci_bdf: None })
        .collect())
}

#[cfg(not(feature = "cuda"))]
fn enumerate_cuda() -> Result<Vec<DeviceEntry>, String> {
    Err("this build of validate_numa_placement was compiled without the `cuda` feature".to_string())
}

#[cfg(feature = "xpu-sycl")]
fn enumerate_sycl() -> Result<Vec<DeviceEntry>, String> {
    use oneapi_rs::safe::SyclDevice;

    let count = SyclDevice::count().map_err(|e| format!("SyclDevice::count: {e}"))?;
    if count == 0 {
        return Err("No SYCL/XPU devices found".to_string());
    }
    Ok((0..count as u32)
        .map(|id| {
            let pci_bdf = SyclDevice::by_ordinal(id as usize)
                .ok()
                .and_then(|d| d.info().ok())
                .and_then(|info| info.pci_address);
            DeviceEntry { id, pci_bdf }
        })
        .collect())
}

#[cfg(not(feature = "xpu-sycl"))]
fn enumerate_sycl() -> Result<Vec<DeviceEntry>, String> {
    Err("this build of validate_numa_placement was compiled without the `xpu-sycl` feature"
        .to_string())
}

/// Parse `--backend {cuda,sycl,auto}` → `DeviceBackend` (or error out).
fn parse_backend_arg(s: &str) -> Result<DeviceBackend, String> {
    match s.to_lowercase().as_str() {
        "auto" => DeviceBackend::detect_backend().map_err(|e| format!("auto-detect: {e}")),
        other => other
            .parse::<DeviceBackend>()
            .map_err(|e| format!("unknown backend '{other}': {e}")),
    }
}

fn compiled_features_summary() -> String {
    let mut feats: Vec<&'static str> = Vec::new();
    if cfg!(feature = "cuda") {
        feats.push("cuda");
    }
    if cfg!(feature = "xpu-sycl") {
        feats.push("xpu-sycl");
    }
    if feats.is_empty() {
        "<none — build enables no device backend>".to_string()
    } else {
        feats.join(", ")
    }
}

/// If the current failure is likely due to a missing compile-time feature,
/// print a rebuild hint with the exact `cargo` command the user should run.
///
/// `requested` is the backend the user asked for (via `--backend`, or
/// `"auto"` if they didn't). This helper inspects which features are
/// compiled vs. what was asked and prints a hint when they mismatch.
fn print_rebuild_hint(requested: &str) {
    let has_cuda = cfg!(feature = "cuda");
    let has_sycl = cfg!(feature = "xpu-sycl");
    let req = requested.to_lowercase();

    // Case 1: user asked for a specific backend that isn't compiled in.
    let missing_backend = match req.as_str() {
        "cuda" | "gpu" | "nvidia" if !has_cuda => Some("cuda"),
        "sycl" | "xpu" | "intel" if !has_sycl => Some("xpu-sycl"),
        _ => None,
    };
    if let Some(feat) = missing_backend {
        eprintln!();
        eprintln!(
            "Hint: this build was compiled without the `{feat}` feature."
        );
        eprintln!("      Rebuild with that feature to enable the {} backend:",
            if feat == "cuda" { "CUDA" } else { "SYCL/XPU" });
        if feat == "xpu-sycl" {
            // `xpu-sycl` implies `kvbm-kernels/xpu_permute_kernels`, so
            // `libkvbm_kernels_xpu.so` must exist at link time. That
            // library is produced only when KVBM_ENABLE_XPU_KERNELS=1
            // is set at `cargo build` time (triggers icpx -fsycl in
            // lib/kvbm-kernels/build.rs).
            eprintln!("      KVBM_ENABLE_XPU_KERNELS=1 \\");
            eprintln!(
                "          cargo run -p kvbm-physical --features xpu-sycl \\"
            );
            eprintln!("              --bin validate_numa_placement -- --backend sycl");
        } else {
            eprintln!(
                "      cargo run -p kvbm-physical --features cuda \\"
            );
            eprintln!("          --bin validate_numa_placement -- --backend cuda");
        }
        return;
    }

    // Case 2: user used `--backend auto` (or default) and nothing was
    // detected. The hint tells them how to widen the compiled feature
    // set so auto-detect has more to try.
    if req == "auto" {
        match (has_cuda, has_sycl) {
            (true, false) => {
                eprintln!();
                eprintln!(
                    "Hint: only the `cuda` backend is compiled in, and no CUDA"
                );
                eprintln!("      driver was detected on this host. If this is an Intel");
                eprintln!("      XPU system, rebuild with the SYCL backend enabled.");
                eprintln!("      The `xpu-sycl` feature links against libkvbm_kernels_xpu.so,");
                eprintln!("      which is only produced when KVBM_ENABLE_XPU_KERNELS=1 is set");
                eprintln!("      at build time (triggers icpx -fsycl in kvbm-kernels):");
                eprintln!("      KVBM_ENABLE_XPU_KERNELS=1 \\");
                eprintln!(
                    "          cargo run -p kvbm-physical \\"
                );
                eprintln!(
                    "              --no-default-features --features xpu-sycl \\"
                );
                eprintln!("              --bin validate_numa_placement");
            }
            (false, true) => {
                eprintln!();
                eprintln!(
                    "Hint: only the `xpu-sycl` backend is compiled in, and no"
                );
                eprintln!("      SYCL runtime was detected. Check that `libsycl.so` is on");
                eprintln!("      the loader path and that an Intel GPU is visible to the");
                eprintln!("      runtime (e.g. `sycl-ls` lists at least one device).");
            }
            (false, false) => {
                eprintln!();
                eprintln!(
                    "Hint: no device backend was compiled into this binary."
                );
                eprintln!("      Rebuild with at least one of:");
                eprintln!("        --features cuda");
                eprintln!("        --features xpu-sycl  (requires KVBM_ENABLE_XPU_KERNELS=1 at build time)");
            }
            (true, true) => {
                eprintln!();
                eprintln!(
                    "Hint: both backends are compiled in but neither runtime"
                );
                eprintln!("      was detected on this host. Check driver installation:");
                eprintln!("      - CUDA: `nvidia-smi` should list a device.");
                eprintln!("      - SYCL: `sycl-ls` should list a device.");
            }
        }
    }
}

fn print_help() {
    eprintln!("Usage: validate_numa_placement [OPTIONS]");
    eprintln!();
    eprintln!("Validates that pinned host memory allocated via KVBM's");
    eprintln!("`NumaWorkerPool` path is bound to the NUMA node closest to");
    eprintln!("the chosen device, for all visible devices on that backend.");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --backend BE   Backend: cuda | sycl (a.k.a. xpu | intel) | auto");
    eprintln!("                 (default: auto — CUDA first, then SYCL)");
    eprintln!("  --size MiB     Allocation size per device (default: 16)");
    eprintln!("  --gpus LIST    Comma-separated device ordinals (default: all)");
    eprintln!("  -h, --help     Show this help");
    eprintln!();
    eprintln!("Compiled backends: {}", compiled_features_summary());
}

fn main() {
    // -- Parse args ----------------------------------------------------------
    let args: Vec<String> = std::env::args().collect();
    let mut size_mib: usize = 16;
    let mut gpu_filter: Option<Vec<u32>> = None;
    let mut backend_arg: String = "auto".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--backend" => {
                i += 1;
                backend_arg = args
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| {
                        eprintln!("--backend requires an argument");
                        process::exit(1);
                    });
            }
            "--size" => {
                i += 1;
                size_mib = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| {
                        eprintln!("--size requires a numeric argument (MiB)");
                        process::exit(1);
                    });
            }
            "--gpus" => {
                i += 1;
                let gpus = args.get(i).unwrap_or_else(|| {
                    eprintln!("--gpus requires a comma-separated list (e.g. 0,1,3)");
                    process::exit(1);
                });
                gpu_filter = Some(
                    gpus.split(',')
                        .filter_map(|s| s.trim().parse::<u32>().ok())
                        .collect(),
                );
            }
            "--help" | "-h" => {
                print_help();
                process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}");
                print_help();
                process::exit(1);
            }
        }
        i += 1;
    }

    let backend = parse_backend_arg(&backend_arg).unwrap_or_else(|e| {
        eprintln!("Invalid backend: {e}");
        eprintln!("Compiled backends: {}", compiled_features_summary());
        print_rebuild_hint(&backend_arg);
        process::exit(1);
    });

    let alloc_size = size_mib * 1024 * 1024;

    // -- Enumerate devices ---------------------------------------------------
    let all_devices = enumerate_devices(backend).unwrap_or_else(|e| {
        eprintln!("Failed to enumerate {} devices: {e}", backend.name());
        eprintln!("Compiled backends: {}", compiled_features_summary());
        print_rebuild_hint(&backend_arg);
        process::exit(1);
    });

    let devices: Vec<DeviceEntry> = match gpu_filter {
        Some(ref list) => {
            for &g in list {
                if !all_devices.iter().any(|d| d.id == g) {
                    eprintln!(
                        "{} device ordinal {g} is not visible (have {} devices)",
                        backend.name(),
                        all_devices.len()
                    );
                    process::exit(1);
                }
            }
            all_devices.into_iter().filter(|d| list.contains(&d.id)).collect()
        }
        None => all_devices,
    };

    let ids: Vec<u32> = devices.iter().map(|d| d.id).collect();

    // -- Header --------------------------------------------------------------
    println!("NUMA Placement Validator");
    println!("========================");
    println!("Backend:          {} ({})", backend.name(), backend_arg);
    println!("Compiled feats:   {}", compiled_features_summary());
    println!("Devices:          {ids:?}");
    println!("Alloc size:       {size_mib} MiB ({alloc_size} bytes)");
    println!("NUMA enabled:     {}", dynamo_memory::is_numa_enabled());
    println!();

    // -- Phase 1: device → NUMA mapping --------------------------------------
    println!("--- Device-to-NUMA Topology ---");
    let mut expected_nodes: Vec<Option<u32>> = Vec::with_capacity(devices.len());
    let mut ctx_pci: Vec<Option<String>> = Vec::with_capacity(devices.len());
    for d in &devices {
        // Try PCI BDF via the device context (authoritative); fall back to
        // whatever enumerate_devices learned for us.
        let ctx_bdf = DeviceContext::new(backend, d.id)
            .ok()
            .and_then(|ctx| ctx.pci_bdf_address())
            .or_else(|| d.pci_bdf.clone());
        let numa = ctx_bdf
            .as_deref()
            .and_then(dynamo_memory::numa::get_numa_node_for_pci_address);

        let bdf_str = ctx_bdf.clone().unwrap_or_else(|| "?".to_string());
        let node_str = numa.map(|n| format!("{}", n.0)).unwrap_or_else(|| "UNKNOWN".to_string());
        println!("  {} {} (PCI {bdf_str}) -> NUMA node {node_str}",
                 backend.name(), d.id);

        expected_nodes.push(numa.map(|n| n.0));
        ctx_pci.push(ctx_bdf);
    }
    println!();

    // -- Phase 2: allocate and validate --------------------------------------
    println!("--- Page Placement Validation ---");
    let mut all_ok = true;

    for (idx, d) in devices.iter().enumerate() {
        let expected = expected_nodes[idx];

        print!(
            "  {} {}: allocating {size_mib} MiB via DeviceAllocator... ",
            backend.name(),
            d.id
        );

        let ctx: Arc<dyn dynamo_memory::DeviceAllocator> =
            match DeviceContext::new(backend, d.id) {
                Ok(c) => Arc::new(c),
                Err(e) => {
                    println!("FAILED: DeviceContext::new: {e}");
                    all_ok = false;
                    continue;
                }
            };

        let storage = match dynamo_memory::PinnedStorage::new(alloc_size, ctx) {
            Ok(s) => s,
            Err(e) => {
                println!("FAILED: PinnedStorage::new: {e}");
                all_ok = false;
                continue;
            }
        };

        let ptr = unsafe { storage.as_ptr() };
        println!("OK (ptr={ptr:p})");

        // Query actual page placement
        let page_nodes = query_page_nodes(ptr, alloc_size);
        let total_pages = page_nodes.len();

        if total_pages == 0 {
            println!("    No pages to check");
            continue;
        }

        // Count pages per NUMA node
        let mut node_counts: std::collections::BTreeMap<i32, usize> =
            std::collections::BTreeMap::new();
        for &node in &page_nodes {
            *node_counts.entry(node).or_insert(0) += 1;
        }

        // Report distribution
        print!("    Pages: {total_pages} total -> ");
        let parts: Vec<String> = node_counts
            .iter()
            .map(|(&node, &count)| {
                let pct = (count as f64 / total_pages as f64) * 100.0;
                if node < 0 {
                    format!("ERROR({node}): {count} ({pct:.1}%)")
                } else {
                    format!("node {node}: {count} ({pct:.1}%)")
                }
            })
            .collect();
        println!("{}", parts.join(", "));

        // Validate against expected
        match expected {
            Some(expected_node) => {
                let correct = node_counts.get(&(expected_node as i32)).copied().unwrap_or(0);
                let pct = (correct as f64 / total_pages as f64) * 100.0;

                if correct == total_pages {
                    println!("    PASS: 100% pages on expected NUMA node {expected_node}");
                } else {
                    let misplaced = total_pages - correct;
                    println!(
                        "    FAIL: {misplaced}/{total_pages} pages ({:.1}%) NOT on expected NUMA node {expected_node}",
                        100.0 - pct
                    );
                    all_ok = false;
                }
            }
            None => {
                println!("    SKIP: NUMA node unknown for {} {} (PCI {:?}), cannot validate placement",
                    backend.name(), d.id, ctx_pci[idx]);
            }
        }

        // Storage drops here, freeing the pinned memory
    }

    println!();
    if all_ok {
        println!("Result: ALL PASSED");
    } else {
        println!("Result: SOME FAILED (see above)");
        process::exit(1);
    }
}
