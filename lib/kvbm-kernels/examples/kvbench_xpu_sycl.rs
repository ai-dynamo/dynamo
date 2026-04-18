// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV cache transfer benchmark for Intel XPU (SYCL / oneapi-rs).
//!
//! Compares `xpu_vectorized_copy` SYCL kernel against individual
//! `sycl::queue::memcpy` for layerwise vs fully-contiguous block transfers
//! using Llama 3.1 70B KV cache dimensions.
//!
//! # Backends
//!
//! - **vectorized** -- `xpu_vectorized_copy` SYCL C++ kernel dispatched via FFI.
//!   Uploads (src, dst) pointer arrays to device, then one kernel dispatch
//!   copies all chunks in parallel.  This is the same code path used by
//!   production `kvbm-physical`.
//! - **memcpy** -- Individual `sycl::queue::memcpy` (USM async memcpy) calls,
//!   one per chunk.  Direction (H2D, D2H, D2D) is auto-detected by the SYCL
//!   runtime.
//!
//! # Transfer directions
//!
//! - **D2D** -- Device-to-device on the same GPU.
//! - **H2D** -- Host-to-device (upload) via PCIe.
//! - **D2H** -- Device-to-host (download) via PCIe.
//! - **D2Dx** -- Cross-device device-to-device (src on `--device`,
//!   dst on `--dst-device`).  Only `memcpy` backend is supported.
//!
//! # Transfer patterns
//!
//! - **fc_to_fc** -- Fully-contiguous <-> fully-contiguous.
//!   Both sides are allocated as `num_blocks` buffers of
//!   `full_block_size` each. One copy pair per block.
//! - **lw_to_fc** -- Layerwise (scattered) -> fully-contiguous.
//!   Source: `NUM_LAYERS` independent allocations (one per layer),
//!   each holding all blocks' K+V for that layer.
//!   Destination: one contiguous allocation holding all blocks packed
//!   sequentially.
//! - **fc_to_lw** -- Fully-contiguous -> layerwise (scattered).
//!   Reverse of `lw_to_fc`: contiguous source, per-layer destination.
//!
//! # Compatibility matrix
//!
//! | direction | backend      | fc_to_fc | lw_to_fc | fc_to_lw | host mem   | notes                              |
//! |-----------|--------------|----------|----------|----------|------------|------------------------------------|
//! | **D2D**   | vectorized   | OK       | OK       | OK       | n/a        | SYCL kernel via FFI                |
//! | **D2D**   | memcpy       | OK       | OK       | OK       | n/a        | `sycl::queue::memcpy`              |
//! | **H2D**   | vectorized   | OK       | OK       | OK       | pinned     | kernel dereferences USM host ptrs  |
//! | **H2D**   | memcpy       | OK       | OK       | OK       | pinned, system | system uses staging internally |
//! | **D2H**   | vectorized   | OK       | OK       | OK       | pinned     | kernel dereferences USM host ptrs  |
//! | **D2H**   | memcpy       | OK       | OK       | OK       | pinned, system | system uses staging internally |
//! | **D2Dx**  | memcpy       | OK       | OK       | OK       | n/a        | cross-device via PCIe          |
//!
//! **vectorized + system heap**: Skipped. The GPU kernel dereferences raw
//! pointers — system `Vec<u8>` memory is not USM and cannot be accessed
//! by the GPU.  Only pinned (`sycl::malloc_host`) host memory is usable.
//!
//! # Host memory kinds
//!
//! - **pinned** -- USM host memory (`sycl::malloc_host`). Page-locked,
//!   DMA-friendly. This is what the real KV cache offload path uses.
//! - **system** -- Regular heap memory (`Vec<u8>`). Not pinned.
//!   The SYCL runtime stages through an internal buffer, so
//!   H2D/D2H bandwidth is typically lower.
//!
//! # Differences from `kvbench_xpu_ze` (Level-Zero)
//!
//! - Uses a single in-order `sycl::queue` instead of separate BCS/CCS
//!   command lists.
//! - Cross-device D2D (`d2dx`) uses a second queue bound to the
//!   destination device.  Only the `memcpy` backend is supported.
//! - The vectorized kernel is dispatched through C++ FFI
//!   (`xpu_vectorized_copy`) rather than raw SPIR-V + `zeKernelCreate`.
//!
//! Output: CSV on stdout (same schema as `kvbench_xpu`), suitable for
//! direct comparison.
//!
//! # Usage
//!
//! Each command below corresponds to one or more rows in the compatibility
//! matrix.  All commands redirect stderr (progress) to `/dev/null` so only
//! the CSV data reaches stdout.
//!
//! ```sh
//! # --- D2D, vectorized (SYCL kernel) --- fc_to_fc + lw_to_fc + fc_to_lw
//! KVBM_ENABLE_XPU_KERNELS=1 cargo run --example kvbench_xpu_sycl \
//!   --features kvbench-xpu-sycl --release -- \
//!   --direction d2d --backend vectorized \
//!   --pattern fc_to_fc,lw_to_fc,fc_to_lw 2>/dev/null
//!
//! # --- D2D, memcpy (sycl::queue::memcpy) --- fc_to_fc + lw_to_fc + fc_to_lw
//! KVBM_ENABLE_XPU_KERNELS=1 cargo run --example kvbench_xpu_sycl \
//!   --features kvbench-xpu-sycl --release -- \
//!   --direction d2d --backend memcpy \
//!   --pattern fc_to_fc,lw_to_fc,fc_to_lw 2>/dev/null
//!
//! # --- H2D, vectorized --- pinned host mem only
//! KVBM_ENABLE_XPU_KERNELS=1 cargo run --example kvbench_xpu_sycl \
//!   --features kvbench-xpu-sycl --release -- \
//!   --direction h2d --backend vectorized \
//!   --pattern fc_to_fc,lw_to_fc,fc_to_lw --host-mem pinned 2>/dev/null
//!
//! # --- H2D, memcpy --- pinned + system host mem
//! KVBM_ENABLE_XPU_KERNELS=1 cargo run --example kvbench_xpu_sycl \
//!   --features kvbench-xpu-sycl --release -- \
//!   --direction h2d --backend memcpy \
//!   --pattern fc_to_fc,lw_to_fc,fc_to_lw --host-mem pinned,system 2>/dev/null
//!
//! # --- D2H, vectorized --- pinned host mem only
//! KVBM_ENABLE_XPU_KERNELS=1 cargo run --example kvbench_xpu_sycl \
//!   --features kvbench-xpu-sycl --release -- \
//!   --direction d2h --backend vectorized \
//!   --pattern fc_to_fc,lw_to_fc,fc_to_lw --host-mem pinned 2>/dev/null
//!
//! # --- D2H, memcpy --- pinned + system host mem
//! KVBM_ENABLE_XPU_KERNELS=1 cargo run --example kvbench_xpu_sycl \
//!   --features kvbench-xpu-sycl --release -- \
//!   --direction d2h --backend memcpy \
//!   --pattern fc_to_fc,lw_to_fc,fc_to_lw --host-mem pinned,system 2>/dev/null
//!
//! # --- Cross-device D2D, memcpy --- device 0 -> device 1
//! KVBM_ENABLE_XPU_KERNELS=1 cargo run --example kvbench_xpu_sycl \\
//!   --features kvbench-xpu-sycl --release -- \\
//!   --direction d2dx --backend memcpy --device 0 --dst-device 1 \\
//!   --pattern fc_to_fc,lw_to_fc,fc_to_lw 2>/dev/null
//!
//! # --- Full sweep (all 4 same-device rows) ---
//! KVBM_ENABLE_XPU_KERNELS=1 cargo run --example kvbench_xpu_sycl \
//!   --features kvbench-xpu-sycl --release -- \
//!   --direction d2d,h2d,d2h --backend vectorized,memcpy \
//!   --pattern fc_to_fc,lw_to_fc,fc_to_lw --host-mem pinned,system 2>/dev/null
//!
//! # --- Quick smoke test on device 0 ---
//! KVBM_ENABLE_XPU_KERNELS=1 cargo run --example kvbench_xpu_sycl \
//!   --features kvbench-xpu-sycl --release -- \
//!   --device 0 --num-blocks 1,4 --tokens-per-block 16 \
//!   --warmup 3 --iters 10
//! ```

use std::ffi::c_void;
use std::sync::Arc;

use clap::Parser;
use oneapi_rs::sycl::SyclQueue;

use kvbm_kernels::xpu_vectorized_copy;

// ---------------------------------------------------------------------------
// Llama 3.1 70B, bf16 KV cache dimensions
// ---------------------------------------------------------------------------
const NUM_LAYERS: usize = 80;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const ELEM_SIZE: usize = 2; // bf16
const OUTER_DIM: usize = 2; // K and V

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// KV cache transfer benchmark for Intel XPU (SYCL backend, Llama 3.1 70B, bf16).
#[derive(Parser, Debug)]
#[command(name = "kvbench_xpu_sycl", about = "KV cache transfer bandwidth benchmark (XPU / SYCL)")]
struct Cli {
    /// Comma-separated number of blocks to benchmark.
    #[arg(
        long,
        default_value = "1,2,4,8,16,32,64,128,256",
        value_delimiter = ','
    )]
    num_blocks: Vec<usize>,

    /// Comma-separated tokens per block values.
    #[arg(long, default_value = "16,32,64", value_delimiter = ',')]
    tokens_per_block: Vec<usize>,

    /// Comma-separated backends: vectorized, memcpy.
    #[arg(
        long,
        default_value = "vectorized,memcpy",
        value_delimiter = ','
    )]
    backend: Vec<String>,

    /// Comma-separated directions: h2d, d2h, d2d.
    #[arg(long, default_value = "h2d,d2h,d2d", value_delimiter = ',')]
    direction: Vec<String>,

    /// Comma-separated patterns: fc_to_fc, lw_to_fc, fc_to_lw.
    #[arg(long, default_value = "fc_to_fc,lw_to_fc,fc_to_lw", value_delimiter = ',')]
    pattern: Vec<String>,

    /// Comma-separated host memory kinds: pinned, system.
    #[arg(long, default_value = "pinned", value_delimiter = ',')]
    host_mem: Vec<String>,

    /// Number of warmup iterations.
    #[arg(long, default_value = "10")]
    warmup: usize,

    /// Number of timed iterations.
    #[arg(long, default_value = "100")]
    iters: usize,

    /// Device ordinal.
    #[arg(long, default_value = "0")]
    device: usize,

    /// Destination device ordinal (for cross-device D2D).
    #[arg(long, default_value = "1")]
    dst_device: usize,
}

// ---------------------------------------------------------------------------
// Transfer direction
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum Direction {
    H2D,
    D2H,
    D2D,
    /// Cross-device device-to-device (src on --device, dst on --dst-device).
    D2Dx,
}

impl Direction {
    fn label(&self) -> &'static str {
        match self {
            Direction::H2D => "h2d",
            Direction::D2H => "d2h",
            Direction::D2D => "d2d",
            Direction::D2Dx => "d2dx",
        }
    }

    fn from_str(s: &str) -> Option<Self> {
        match s {
            "h2d" => Some(Direction::H2D),
            "d2h" => Some(Direction::D2H),
            "d2d" => Some(Direction::D2D),
            "d2dx" | "cross" => Some(Direction::D2Dx),
            _ => None,
        }
    }

    fn all_labels() -> &'static str {
        "h2d, d2h, d2d, d2dx (cross-device)"
    }
}

// ---------------------------------------------------------------------------
// Transfer pattern
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum Pattern {
    FcToFc,
    LwToFc,
    FcToLw,
}

impl Pattern {
    fn label(&self) -> &'static str {
        match self {
            Pattern::FcToFc => "fc_to_fc",
            Pattern::LwToFc => "lw_to_fc",
            Pattern::FcToLw => "fc_to_lw",
        }
    }

    fn from_str(s: &str) -> Option<Self> {
        match s {
            "fc_to_fc" | "fc" => Some(Pattern::FcToFc),
            "lw_to_fc" | "lw" => Some(Pattern::LwToFc),
            "fc_to_lw" => Some(Pattern::FcToLw),
            _ => None,
        }
    }

    fn all_labels() -> &'static str {
        "fc_to_fc (or fc), lw_to_fc (or lw), fc_to_lw"
    }
}

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum Backend {
    Vectorized,
    Memcpy,
}

impl Backend {
    fn label(&self) -> &'static str {
        match self {
            Backend::Vectorized => "vectorized",
            Backend::Memcpy => "memcpy",
        }
    }

    fn from_str(s: &str) -> Option<Self> {
        match s {
            "vectorized" | "vec" => Some(Backend::Vectorized),
            "memcpy" => Some(Backend::Memcpy),
            _ => None,
        }
    }

    fn all_labels() -> &'static str {
        "vectorized (or vec), memcpy"
    }
}

// ---------------------------------------------------------------------------
// Host memory kind
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum HostMemKind {
    Pinned,
    System,
}

impl HostMemKind {
    fn label(&self) -> &'static str {
        match self {
            HostMemKind::Pinned => "pinned",
            HostMemKind::System => "system",
        }
    }

    fn from_str(s: &str) -> Option<Self> {
        match s {
            "pinned" | "pin" => Some(HostMemKind::Pinned),
            "system" | "sys" | "heap" => Some(HostMemKind::System),
            _ => None,
        }
    }

    fn all_labels() -> &'static str {
        "pinned (or pin), system (or sys/heap)"
    }
}

// ---------------------------------------------------------------------------
// RAII wrappers for raw USM allocations
// ---------------------------------------------------------------------------

/// Device memory allocated via `sycl::malloc_device`.
struct DeviceMem {
    ptr: *mut c_void,
    queue: Arc<SyclQueue>,
}

impl DeviceMem {
    fn new(queue: &Arc<SyclQueue>, bytes: usize) -> Self {
        let ptr = queue.malloc_device(bytes).expect("malloc_device failed");
        Self { ptr, queue: Arc::clone(queue) }
    }
}

impl Drop for DeviceMem {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = self.queue.free_raw(self.ptr);
        }
    }
}

/// Host-pinned (USM host) memory allocated via `sycl::malloc_host`.
struct HostMem {
    ptr: *mut c_void,
    queue: Arc<SyclQueue>,
}

impl HostMem {
    fn new(queue: &Arc<SyclQueue>, bytes: usize) -> Self {
        let ptr = queue.malloc_host(bytes).expect("malloc_host failed");
        // Fill with pattern to avoid zero-page tricks.
        unsafe { std::ptr::write_bytes(ptr as *mut u8, 0xAB, bytes) };
        Self { ptr, queue: Arc::clone(queue) }
    }
}

impl Drop for HostMem {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = self.queue.free_raw(self.ptr);
        }
    }
}

// ---------------------------------------------------------------------------
// Memory pair: holds src + dst buffers for a given direction
// ---------------------------------------------------------------------------

enum SideBuffers {
    /// USM host memory (pinned, DMA-friendly).
    Host(Vec<HostMem>),
    /// Regular heap memory (not pinned).
    System(Vec<Vec<u8>>),
    /// USM device memory.
    Device(Vec<DeviceMem>),
}

impl SideBuffers {
    fn ptr(&self, idx: usize) -> *mut c_void {
        match self {
            SideBuffers::Host(bufs) => bufs[idx].ptr,
            SideBuffers::System(bufs) => bufs[idx].as_ptr() as *mut c_void,
            SideBuffers::Device(bufs) => bufs[idx].ptr,
        }
    }
}

struct MemoryPair {
    src: SideBuffers,
    dst: SideBuffers,
}

fn allocate_side_host(queue: &Arc<SyclQueue>, count: usize, size: usize) -> SideBuffers {
    SideBuffers::Host(
        (0..count)
            .map(|_| HostMem::new(queue, size))
            .collect(),
    )
}

fn allocate_side_system(count: usize, size: usize) -> SideBuffers {
    SideBuffers::System(
        (0..count)
            .map(|_| vec![0xABu8; size])
            .collect(),
    )
}

fn allocate_side_device(queue: &Arc<SyclQueue>, count: usize, size: usize) -> SideBuffers {
    SideBuffers::Device(
        (0..count)
            .map(|_| DeviceMem::new(queue, size))
            .collect(),
    )
}

fn allocate_host_side(
    queue: &Arc<SyclQueue>,
    host_mem: HostMemKind,
    count: usize,
    size: usize,
) -> SideBuffers {
    match host_mem {
        HostMemKind::Pinned => allocate_side_host(queue, count, size),
        HostMemKind::System => allocate_side_system(count, size),
    }
}

fn allocate_memory(
    queue: &Arc<SyclQueue>,
    dst_queue: &Arc<SyclQueue>,
    direction: Direction,
    host_mem: HostMemKind,
    src_count: usize,
    src_size: usize,
    dst_count: usize,
    dst_size: usize,
) -> MemoryPair {
    let (src_host, dst_host) = match direction {
        Direction::H2D => (true, false),
        Direction::D2H => (false, true),
        Direction::D2D | Direction::D2Dx => (false, false),
    };
    // For D2Dx, dst is allocated on the destination device's queue.
    let effective_dst_queue = match direction {
        Direction::D2Dx => dst_queue,
        _ => queue,
    };
    MemoryPair {
        src: if src_host {
            allocate_host_side(queue, host_mem, src_count, src_size)
        } else {
            allocate_side_device(queue, src_count, src_size)
        },
        dst: if dst_host {
            allocate_host_side(queue, host_mem, dst_count, dst_size)
        } else {
            allocate_side_device(effective_dst_queue, dst_count, dst_size)
        },
    }
}

// ---------------------------------------------------------------------------
// Pointer pair lists
// ---------------------------------------------------------------------------

/// FC<->FC: one copy pair per block, each `full_block_size` bytes.
fn build_fc_pairs(mem: &MemoryPair, num_blocks: usize) -> (Vec<u64>, Vec<u64>) {
    let mut src_addrs = Vec::with_capacity(num_blocks);
    let mut dst_addrs = Vec::with_capacity(num_blocks);
    for b in 0..num_blocks {
        src_addrs.push(mem.src.ptr(b) as u64);
        dst_addrs.push(mem.dst.ptr(b) as u64);
    }
    (src_addrs, dst_addrs)
}

/// Build (scattered, contiguous) pointer pairs for LW<->FC patterns.
fn build_lw_fc_pairs(
    scattered: &SideBuffers,
    contiguous: &SideBuffers,
    num_blocks: usize,
    full_block_size: usize,
    inner: usize,
) -> (Vec<u64>, Vec<u64>) {
    let total = num_blocks * NUM_LAYERS * OUTER_DIM;
    let mut sc_addrs = Vec::with_capacity(total);
    let mut fc_addrs = Vec::with_capacity(total);
    let fc_base = contiguous.ptr(0) as u64;
    for b in 0..num_blocks {
        for layer in 0..NUM_LAYERS {
            let sc_base = scattered.ptr(layer) as u64;
            for outer in 0..OUTER_DIM {
                let sc_offset = (b * OUTER_DIM + outer) * inner;
                let fc_offset = b * full_block_size
                    + (layer * OUTER_DIM + outer) * inner;
                sc_addrs.push(sc_base + sc_offset as u64);
                fc_addrs.push(fc_base + fc_offset as u64);
            }
        }
    }
    (sc_addrs, fc_addrs)
}

// ---------------------------------------------------------------------------
// Execute one iteration
// ---------------------------------------------------------------------------

/// Dispatch `xpu_vectorized_copy` SYCL kernel via FFI.
///
/// Pointer arrays are uploaded to device memory, then the kernel is launched
/// through the same FFI used by production code.
fn execute_vectorized(
    queue: &Arc<SyclQueue>,
    src_addrs: &[u64],
    dst_addrs: &[u64],
    src_addrs_dev: &DeviceMem,
    dst_addrs_dev: &DeviceMem,
    ptr_array_bytes: usize,
    copy_size: usize,
    num_copies: usize,
) {
    // Upload pointer arrays (H2D on the in-order queue).
    unsafe {
        queue.memcpy_raw_async(
            src_addrs_dev.ptr,
            src_addrs.as_ptr() as *const c_void,
            ptr_array_bytes,
        ).expect("upload src_addrs");
        queue.memcpy_raw_async(
            dst_addrs_dev.ptr,
            dst_addrs.as_ptr() as *const c_void,
            ptr_array_bytes,
        ).expect("upload dst_addrs");
    }

    // Launch kernel — in-order after the uploads above.
    let status = unsafe {
        xpu_vectorized_copy(
            src_addrs_dev.ptr as *mut *mut c_void,
            dst_addrs_dev.ptr as *mut *mut c_void,
            copy_size,
            num_copies as i32,
            queue.raw_queue_ptr(),
        )
    };
    assert_eq!(status, 0, "xpu_vectorized_copy returned {status}");

    queue.synchronize().expect("sync after vectorized_copy");
}

/// Execute per-chunk memcpy via SYCL queue.
fn execute_memcpy(
    queue: &Arc<SyclQueue>,
    src_addrs: &[u64],
    dst_addrs: &[u64],
    copy_size: usize,
    num_copies: usize,
) {
    for i in 0..num_copies {
        unsafe {
            queue.memcpy_raw_async(
                dst_addrs[i] as *mut c_void,
                src_addrs[i] as *const c_void,
                copy_size,
            ).expect("memcpy_raw_async");
        }
    }
    queue.synchronize().expect("sync after memcpy");
}

// ---------------------------------------------------------------------------
// Run one benchmark configuration
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_benchmark(
    queue: &Arc<SyclQueue>,
    dst_queue: &Arc<SyclQueue>,
    direction: Direction,
    pattern: Pattern,
    backend: Backend,
    host_mem: HostMemKind,
    tokens_per_block: usize,
    num_blocks: usize,
    warmup_iters: usize,
    timed_iters: usize,
) -> (f64, f64) {
    let inner = tokens_per_block * NUM_KV_HEADS * HEAD_DIM * ELEM_SIZE;
    let full_block_size = inner * OUTER_DIM * NUM_LAYERS;
    let total_bytes = full_block_size * num_blocks;

    let (copy_size, num_copies) = match pattern {
        Pattern::FcToFc => (full_block_size, num_blocks),
        Pattern::LwToFc | Pattern::FcToLw => (inner, num_blocks * NUM_LAYERS * OUTER_DIM),
    };

    // Allocate memory.
    let (src_count, src_size, dst_count, dst_size) = match pattern {
        Pattern::FcToFc => (num_blocks, full_block_size, num_blocks, full_block_size),
        Pattern::LwToFc => {
            let layer_buf_size = num_blocks * OUTER_DIM * inner;
            (NUM_LAYERS, layer_buf_size, 1, full_block_size * num_blocks)
        }
        Pattern::FcToLw => {
            let layer_buf_size = num_blocks * OUTER_DIM * inner;
            (1, full_block_size * num_blocks, NUM_LAYERS, layer_buf_size)
        }
    };
    let mem = allocate_memory(
        queue, dst_queue, direction, host_mem,
        src_count, src_size, dst_count, dst_size,
    );

    let (src_addrs, dst_addrs) = match pattern {
        Pattern::FcToFc => build_fc_pairs(&mem, num_blocks),
        Pattern::LwToFc => {
            let (sc, fc) = build_lw_fc_pairs(&mem.src, &mem.dst, num_blocks, full_block_size, inner);
            (sc, fc)
        }
        Pattern::FcToLw => {
            let (sc, fc) = build_lw_fc_pairs(&mem.dst, &mem.src, num_blocks, full_block_size, inner);
            (fc, sc)
        }
    };

    // Device-side pointer arrays for vectorized backend.
    let ptr_array_bytes = num_copies * std::mem::size_of::<u64>();
    let needs_ptr_arrays = matches!(backend, Backend::Vectorized);
    let src_addrs_dev = if needs_ptr_arrays {
        Some(DeviceMem::new(queue, ptr_array_bytes))
    } else {
        None
    };
    let dst_addrs_dev = if needs_ptr_arrays {
        Some(DeviceMem::new(queue, ptr_array_bytes))
    } else {
        None
    };

    // Warmup.
    for _ in 0..warmup_iters {
        match backend {
            Backend::Vectorized => execute_vectorized(
                queue,
                &src_addrs,
                &dst_addrs,
                src_addrs_dev.as_ref().unwrap(),
                dst_addrs_dev.as_ref().unwrap(),
                ptr_array_bytes,
                copy_size,
                num_copies,
            ),
            Backend::Memcpy => {
                execute_memcpy(queue, &src_addrs, &dst_addrs, copy_size, num_copies)
            }
        }
    }

    // Timed iterations (host-side timing; sync after each iteration).
    let mut host_elapsed = Vec::with_capacity(timed_iters);
    for _ in 0..timed_iters {
        let t0 = std::time::Instant::now();
        match backend {
            Backend::Vectorized => execute_vectorized(
                queue,
                &src_addrs,
                &dst_addrs,
                src_addrs_dev.as_ref().unwrap(),
                dst_addrs_dev.as_ref().unwrap(),
                ptr_array_bytes,
                copy_size,
                num_copies,
            ),
            Backend::Memcpy => {
                execute_memcpy(queue, &src_addrs, &dst_addrs, copy_size, num_copies)
            }
        }
        host_elapsed.push(t0.elapsed().as_micros() as f64);
    }

    // Median.
    host_elapsed.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_us = host_elapsed[host_elapsed.len() / 2];
    let median_ms = median_us / 1000.0;
    let bandwidth_gbps = (total_bytes as f64) / (median_ms / 1000.0) / 1e9;

    (median_ms, bandwidth_gbps)
}

// ---------------------------------------------------------------------------
// Parse CLI values
// ---------------------------------------------------------------------------

fn parse_directions(raw: &[String]) -> Vec<Direction> {
    raw.iter()
        .map(|s| {
            Direction::from_str(s).unwrap_or_else(|| {
                panic!("unknown direction '{}', expected: {}", s, Direction::all_labels())
            })
        })
        .collect()
}

fn parse_patterns(raw: &[String]) -> Vec<Pattern> {
    raw.iter()
        .map(|s| {
            Pattern::from_str(s).unwrap_or_else(|| {
                panic!("unknown pattern '{}', expected: {}", s, Pattern::all_labels())
            })
        })
        .collect()
}

fn parse_backends(raw: &[String]) -> Vec<Backend> {
    raw.iter()
        .map(|s| {
            Backend::from_str(s).unwrap_or_else(|| {
                panic!("unknown backend '{}', expected: {}", s, Backend::all_labels())
            })
        })
        .collect()
}

fn parse_host_mem_kinds(raw: &[String]) -> Vec<HostMemKind> {
    raw.iter()
        .map(|s| {
            HostMemKind::from_str(s).unwrap_or_else(|| {
                panic!("unknown host-mem '{}', expected: {}", s, HostMemKind::all_labels())
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();

    // -- Initialize SYCL device -----------------------------------------------
    let device_count = oneapi_rs::sycl::SyclDevice::count()
        .expect("Failed to query SYCL device count");

    if device_count == 0 {
        eprintln!("ERROR: No SYCL devices found.");
        std::process::exit(1);
    }

    eprintln!("SYCL devices:");
    for i in 0..device_count {
        if let Ok(dev) = oneapi_rs::sycl::SyclDevice::by_ordinal(i) {
            if let Ok(info) = dev.info() {
                eprintln!(
                    "  [{i}] {} (max_compute_units={}, global_mem={} GB)",
                    info.name, info.max_compute_units, info.global_mem_size / (1024*1024*1024)
                );
            }
        }
    }
    eprintln!();

    if cli.device >= device_count {
        eprintln!(
            "ERROR: --device {} is out of range. Only {} device(s) available (0..{}).",
            cli.device, device_count, device_count - 1
        );
        std::process::exit(1);
    }
    eprintln!("  Selected device: {}", cli.device);

    let queue = Arc::new(SyclQueue::new_for_device_ordinal(cli.device)
        .expect("Failed to create SYCL queue"));

    // -- Cross-device D2D setup -----------------------------------------------
    let has_d2dx = cli.direction.iter().any(|s| s == "d2dx" || s == "cross");
    let dst_queue = if has_d2dx {
        if device_count < 2 {
            eprintln!("ERROR: Cross-device D2D (d2dx) requires at least 2 devices.");
            std::process::exit(1);
        }
        if cli.dst_device >= device_count {
            eprintln!(
                "ERROR: --dst-device {} is out of range. Only {} device(s) available (0..{}).",
                cli.dst_device, device_count, device_count - 1
            );
            std::process::exit(1);
        }
        if cli.dst_device == cli.device {
            eprintln!(
                "WARNING: --dst-device ({}) == --device ({}); use d2d instead of d2dx for same-device.",
                cli.dst_device, cli.device
            );
        }
        eprintln!("  Destination device: {}", cli.dst_device);
        Arc::new(
            SyclQueue::new_for_device_ordinal(cli.dst_device)
                .expect("Failed to create destination SYCL queue"),
        )
    } else {
        // Placeholder -- unused when d2dx is not requested.
        Arc::clone(&queue)
    };
    eprintln!();

    // -- Parse CLI ------------------------------------------------------------
    let directions = parse_directions(&cli.direction);
    let patterns = parse_patterns(&cli.pattern);
    let backends = parse_backends(&cli.backend);
    let host_mem_kinds = parse_host_mem_kinds(&cli.host_mem);
    let tpb_options = &cli.tokens_per_block;
    let num_blocks_options = &cli.num_blocks;
    let warmup_iters = cli.warmup;
    let timed_iters = cli.iters;

    // Pre-compute the real test count, accounting for skipped combos.
    let total_tests = {
        let base = tpb_options.len() * num_blocks_options.len() * patterns.len();
        let mut count = 0usize;
        for &dir in &directions {
            for &be in &backends {
                for &hm in &host_mem_kinds {
                    // D2D/D2Dx: skip duplicate host-mem variants (keep pinned only).
                    if matches!(dir, Direction::D2D | Direction::D2Dx)
                        && !matches!(hm, HostMemKind::Pinned)
                    {
                        continue;
                    }
                    // D2Dx + vectorized: not supported (device USM is local).
                    if matches!(dir, Direction::D2Dx)
                        && matches!(be, Backend::Vectorized)
                    {
                        continue;
                    }
                    // vectorized + system heap (h2d/d2h): GPU can't access non-USM.
                    if matches!(be, Backend::Vectorized)
                        && matches!(hm, HostMemKind::System)
                        && !matches!(dir, Direction::D2D | Direction::D2Dx)
                    {
                        continue;
                    }
                    count += base;
                }
            }
        }
        count
    };

    // -- Print config ---------------------------------------------------------
    eprintln!("KV Cache Transfer Benchmark (XPU / SYCL)");
    eprintln!("  Device ordinal: {}", cli.device);
    eprintln!("  Model: Llama 3.1 70B (bf16)");
    eprintln!(
        "  Layers: {NUM_LAYERS}, KV heads: {NUM_KV_HEADS}, Head dim: {HEAD_DIM}, Outer dim: {OUTER_DIM}"
    );
    eprintln!("  Warmup: {warmup_iters}, Timed: {timed_iters}");
    eprintln!("  tokens_per_block: {:?}", tpb_options);
    eprintln!("  num_blocks: {:?}", num_blocks_options);
    eprintln!(
        "  directions: [{}]",
        directions.iter().map(|d| d.label()).collect::<Vec<_>>().join(", ")
    );
    eprintln!(
        "  patterns: [{}]",
        patterns.iter().map(|p| p.label()).collect::<Vec<_>>().join(", ")
    );
    eprintln!(
        "  backends: [{}]",
        backends.iter().map(|b| b.label()).collect::<Vec<_>>().join(", ")
    );
    eprintln!(
        "  host_mem: [{}]",
        host_mem_kinds.iter().map(|h| h.label()).collect::<Vec<_>>().join(", ")
    );
    eprintln!("  Total tests: {total_tests}");
    eprintln!();

    // -- CSV header -----------------------------------------------------------
    println!(
        "tokens_per_block,num_blocks,pattern,direction,backend,host_mem,\
         total_bytes,inner_bytes,copy_size,num_copies,median_ms,bandwidth_gbps"
    );

    // -- Benchmark loop -------------------------------------------------------
    let mut test_num = 0;
    for &tpb in tpb_options {
        let inner = tpb * NUM_KV_HEADS * HEAD_DIM * ELEM_SIZE;
        let full_block_size = inner * OUTER_DIM * NUM_LAYERS;

        eprintln!(
            "--- tokens_per_block={tpb}, inner={inner} bytes ({} KB), \
             block={full_block_size} bytes ({:.1} MB) ---",
            inner / 1024,
            full_block_size as f64 / (1024.0 * 1024.0)
        );

        for &num_blocks in num_blocks_options {
            let total_bytes = full_block_size * num_blocks;

            for &direction in &directions {
                for &pattern in &patterns {
                    let (copy_size, num_copies) = match pattern {
                        Pattern::FcToFc => (full_block_size, num_blocks),
                        Pattern::LwToFc | Pattern::FcToLw => (inner, num_blocks * NUM_LAYERS * OUTER_DIM),
                    };

                    for &backend in &backends {
                        for &host_mem in &host_mem_kinds {
                            // D2D/D2Dx don't involve host memory; skip extra variants.
                            if matches!(direction, Direction::D2D | Direction::D2Dx)
                                && !matches!(host_mem, HostMemKind::Pinned)
                            {
                                continue;
                            }

                            // Cross-device D2D: only memcpy is supported (device USM
                            // is local to one GPU; the vectorized kernel cannot access
                            // a remote device's memory).
                            if matches!(direction, Direction::D2Dx)
                                && matches!(backend, Backend::Vectorized)
                            {
                                continue;
                            }

                            // The vectorized kernel dereferences raw USM pointers.
                            // System heap memory is not USM — the GPU cannot access it.
                            if matches!(backend, Backend::Vectorized)
                                && matches!(host_mem, HostMemKind::System)
                                && !matches!(direction, Direction::D2D | Direction::D2Dx)
                            {
                                continue;
                            }

                            test_num += 1;

                            let host_label = if matches!(direction, Direction::D2D | Direction::D2Dx) {
                                "-"
                            } else {
                                host_mem.label()
                            };

                            eprint!(
                                "  [{test_num}/{total_tests}] tpb={tpb} N={num_blocks:>3} \
                                 {:<8} {:<6} {:<12} host={:<7} ... ",
                                pattern.label(),
                                direction.label(),
                                backend.label(),
                                host_label,
                            );

                            let (median_ms, bw) = run_benchmark(
                                &queue,
                                &dst_queue,
                                direction,
                                pattern,
                                backend,
                                host_mem,
                                tpb,
                                num_blocks,
                                warmup_iters,
                                timed_iters,
                            );

                            println!(
                                "{tpb},{num_blocks},{},{},{},{host_label},\
                                 {total_bytes},{inner},{copy_size},{num_copies},\
                                 {median_ms:.4},{bw:.2}",
                                pattern.label(),
                                direction.label(),
                                backend.label(),
                            );
                            eprintln!("{bw:.2} GB/s ({median_ms:.4} ms)");
                        }
                    }
                }
            }
        }
    }

    eprintln!();
    eprintln!("Done. {test_num} tests completed.");
}
