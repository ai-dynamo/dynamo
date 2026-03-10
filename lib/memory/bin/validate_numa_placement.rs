// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic tool for validating NUMA page placement of pinned memory.
//!
//! On a multi-socket machine with multiple GPUs, this binary:
//! 1. Enumerates all visible CUDA devices
//! 2. Maps each GPU to its expected NUMA node (via PCI bus / sysfs)
//! 3. Allocates pinned memory via `PinnedStorage::new_for_device`
//! 4. Uses the `move_pages(2)` syscall to query actual page NUMA placement
//! 5. Reports match/mismatch statistics per GPU
//!
//! # Usage
//! ```bash
//! cargo run -p dynamo-memory --bin validate_numa_placement
//! cargo run -p dynamo-memory --bin validate_numa_placement -- --size 64   # 64 MiB per GPU
//! cargo run -p dynamo-memory --bin validate_numa_placement -- --gpus 0,2  # specific GPUs
//! ```

use std::process;

/// Query the NUMA node of each page in a memory region using `move_pages(2)`.
///
/// `move_pages(pid=0, count, pages, nodes=NULL, status, flags=0)` fills `status`
/// with the current NUMA node of each page without moving anything.
///
/// Returns a Vec of NUMA node IDs (one per page), or negative error codes.
fn query_page_nodes(ptr: *const u8, size: usize) -> Vec<i32> {
    let page_size = unsafe {
        let ps = libc::sysconf(libc::_SC_PAGESIZE);
        if ps > 0 { ps as usize } else { 4096 }
    };

    let num_pages = (size + page_size - 1) / page_size;
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
            0i32,                                      // pid = 0 (self)
            num_pages as libc::c_ulong,                // count
            pages.as_ptr() as *const *const libc::c_void, // pages
            std::ptr::null::<i32>(),                    // nodes = NULL (query mode)
            status.as_mut_ptr(),                        // status (output)
            0i32,                                       // flags
        )
    };

    if ret != 0 {
        let errno = std::io::Error::last_os_error();
        eprintln!("  move_pages syscall failed: {errno}");
        return vec![-1; num_pages];
    }

    status
}

fn main() {
    // Parse args
    let args: Vec<String> = std::env::args().collect();
    let mut size_mib: usize = 16; // default 16 MiB
    let mut gpu_filter: Option<Vec<u32>> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--size" => {
                i += 1;
                size_mib = args.get(i).and_then(|s| s.parse().ok()).unwrap_or_else(|| {
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
                eprintln!("Usage: validate_numa_placement [--size MiB] [--gpus 0,1,...]");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  --size MiB   Allocation size per GPU (default: 16)");
                eprintln!("  --gpus LIST  Comma-separated GPU indices (default: all)");
                process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}");
                process::exit(1);
            }
        }
        i += 1;
    }

    let alloc_size = size_mib * 1024 * 1024;

    // Detect GPU count
    let gpu_count = match cudarc::driver::result::device::get_count() {
        Ok(n) => n,
        Err(e) => {
            eprintln!("Failed to query CUDA device count: {e}");
            process::exit(1);
        }
    };

    if gpu_count == 0 {
        eprintln!("No CUDA devices found");
        process::exit(1);
    }

    let gpus: Vec<u32> = match gpu_filter {
        Some(list) => {
            for &g in &list {
                if g >= gpu_count as u32 {
                    eprintln!("GPU {g} out of range (have {gpu_count} devices)");
                    process::exit(1);
                }
            }
            list
        }
        None => (0..gpu_count as u32).collect(),
    };

    println!("NUMA Placement Validator");
    println!("=======================");
    println!("GPUs:            {gpus:?}");
    println!("Alloc size:      {size_mib} MiB ({alloc_size} bytes)");
    println!("NUMA disabled:   {}", dynamo_memory::is_numa_disabled());
    println!();

    // Phase 1: Show GPU-to-NUMA mapping
    println!("--- GPU-to-NUMA Topology ---");
    let mut expected_nodes: Vec<Option<u32>> = Vec::new();
    for &gpu_id in &gpus {
        let numa_node = dynamo_memory::numa::get_device_numa_node(gpu_id);
        let node_str = match numa_node {
            Some(n) => format!("{}", n.0),
            None => "UNKNOWN".to_string(),
        };
        println!("  GPU {gpu_id} -> NUMA node {node_str}");
        expected_nodes.push(numa_node.map(|n| n.0));
    }
    println!();

    // Phase 2: Allocate and validate
    println!("--- Page Placement Validation ---");
    let mut all_ok = true;

    for (idx, &gpu_id) in gpus.iter().enumerate() {
        let expected = expected_nodes[idx];

        print!("  GPU {gpu_id}: allocating {size_mib} MiB via new_for_device... ");

        let storage = match dynamo_memory::PinnedStorage::new_for_device(alloc_size, Some(gpu_id)) {
            Ok(s) => s,
            Err(e) => {
                println!("FAILED: {e}");
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
                let correct = node_counts
                    .get(&(expected_node as i32))
                    .copied()
                    .unwrap_or(0);
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
                println!("    SKIP: NUMA node unknown for GPU {gpu_id}, cannot validate placement");
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
