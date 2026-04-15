// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV cache transfer benchmark for Intel XPU (Level Zero).
//!
//! Compares vectorized_copy SPIR-V kernel against individual
//! `append_memcpy` for layerwise vs fully-contiguous block transfers
//! using Llama 3.1 70B KV cache dimensions.
//!
//! # Backends
//!
//! - **vectorized** -- Pre-compiled SPIR-V kernel on the CCS (compute) engine.
//!   Uploads (src, dst) pointer arrays to device, then one kernel dispatch
//!   copies all chunks in parallel.
//! - **memcpy** -- Individual `zeCommandListAppendMemoryCopy` calls
//!   on the BCS (copy / blitter) engine, one per chunk.
//!
//! # Transfer directions
//!
//! - **D2D** -- Device-to-device on the same GPU.
//! - **H2D** -- Host-to-device (upload) via PCIe.
//! - **D2H** -- Device-to-host (download) via PCIe.
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
//! | direction | backend      | fc_to_fc | lw_to_fc | fc_to_lw | host mem   | notes                          |
//! |-----------|--------------|----------|----------|----------|------------|--------------------------------|
//! | **D2D**   | vectorized   | OK       | OK       | OK       | n/a        | CCS kernel, `FLAG_DEVICE`      |
//! | **D2D**   | memcpy       | OK       | OK       | OK       | n/a        | BCS (blitter) engine           |
//! | **H2D**   | vectorized   | OK       | OK       | OK       | pinned     | CCS kernel, `FLAG_HOST|DEVICE` |
//! | **H2D**   | memcpy       | OK       | OK       | OK       | pinned, system | BCS; system uses bounce buf |
//! | **D2H**   | vectorized   | OK       | OK       | OK       | pinned     | CCS kernel, `FLAG_HOST|DEVICE` |
//! | **D2H**   | memcpy       | OK       | OK       | OK       | pinned, system | BCS; system uses bounce buf |
//! | **D2Dx**  | memcpy       | OK       | OK       | OK       | n/a        | BCS; cross-device via PCIe     |
//!
//! **vectorized + system heap**: Skipped. The GPU kernel dereferences raw
//! pointers — system `Vec<u8>` memory is not USM and cannot be accessed
//! by the GPU.  Only pinned (`zeMemAllocHost`) host memory is usable.
//!
//! **Cross-device D2D (d2dx)**: Only the `memcpy` backend is supported
//! (device USM is local to one GPU).  `zeCommandListAppendMemoryCopy`
//! handles the routing.  The benchmark checks P2P access capability
//! via `zeDeviceCanAccessPeer` and exits early if not supported.
//!
//! # Host memory kinds
//!
//! - **pinned** -- USM host memory (`zeMemAllocHost`). Page-locked,
//!   DMA-friendly. This is what the real KV cache offload path uses.
//! - **system** -- Regular heap memory (`Vec<u8>`). Not pinned.
//!   The L0 driver pins on-the-fly or stages through an internal
//!   bounce buffer, so H2D/D2H bandwidth is typically lower.
//!
//! Output: CSV on stdout suitable for piping to a plotting script.
//!
//! # Usage
//!
//! Each command below corresponds to one or more rows in the compatibility
//! matrix.  All commands redirect stderr (progress) to `/dev/null` so only
//! the CSV data reaches stdout.
//!
//! ```sh
//! # --- D2D, vectorized (CCS) --- fc_to_fc + lw_to_fc + fc_to_lw
//! cargo run --example kvbench_xpu_ze --features kvbench-xpu-ze --release -- \
//!   --direction d2d --backend vectorized \
//!   --pattern fc_to_fc,lw_to_fc,fc_to_lw 2>/dev/null
//!
//! # --- D2D, memcpy (BCS) --- fc_to_fc + lw_to_fc + fc_to_lw
//! cargo run --example kvbench_xpu_ze --features kvbench-xpu-ze --release -- \
//!   --direction d2d --backend memcpy \
//!   --pattern fc_to_fc,lw_to_fc,fc_to_lw 2>/dev/null
//!
//! # --- H2D, vectorized (CCS) --- pinned host mem only
//! cargo run --example kvbench_xpu_ze --features kvbench-xpu-ze --release -- \
//!   --direction h2d --backend vectorized \
//!   --pattern fc_to_fc,lw_to_fc,fc_to_lw --host-mem pinned 2>/dev/null
//!
//! # --- H2D, memcpy (BCS) --- pinned + system host mem
//! cargo run --example kvbench_xpu_ze --features kvbench-xpu-ze --release -- \
//!   --direction h2d --backend memcpy \
//!   --pattern fc_to_fc,lw_to_fc,fc_to_lw --host-mem pinned,system 2>/dev/null
//!
//! # --- D2H, vectorized (CCS) --- pinned host mem only
//! cargo run --example kvbench_xpu_ze --features kvbench-xpu-ze --release -- \
//!   --direction d2h --backend vectorized \
//!   --pattern fc_to_fc,lw_to_fc,fc_to_lw --host-mem pinned 2>/dev/null
//!
//! # --- D2H, memcpy (BCS) --- pinned + system host mem
//! cargo run --example kvbench_xpu_ze --features kvbench-xpu-ze --release -- \
//!   --direction d2h --backend memcpy \
//!   --pattern fc_to_fc,lw_to_fc,fc_to_lw --host-mem pinned,system 2>/dev/null
//!
//! # --- Cross-device D2D, memcpy (BCS) --- device 0 -> device 1
//! cargo run --example kvbench_xpu_ze --features kvbench-xpu-ze --release -- \
//!   --direction d2dx --backend memcpy --device 0 --dst-device 1 \
//!   --pattern fc_to_fc,lw_to_fc,fc_to_lw 2>/dev/null
//!
//! # --- Full sweep (all 6 table rows, same-device only) ---
//! cargo run --example kvbench_xpu_ze --features kvbench-xpu-ze --release -- \
//!   --direction d2d,h2d,d2h --backend vectorized,memcpy \
//!   --pattern fc_to_fc,lw_to_fc,fc_to_lw --host-mem pinned,system 2>/dev/null
//!
//! # --- Quick smoke test on device 1 ---
//! cargo run --example kvbench_xpu_ze --features kvbench-xpu-ze --release -- \
//!   --device 1 --num-blocks 1,4 --tokens-per-block 16 \
//!   --warmup 3 --iters 10
//! ```

use std::ffi::c_void;

use clap::Parser;
use level_zero as ze;

// ---------------------------------------------------------------------------
// Embedded SPIR-V for the vectorized_copy kernel.
// Same binary loaded by kvbm-physical ze/mod.rs at runtime.
// ---------------------------------------------------------------------------
static VECTORIZED_COPY_SPIRV: &[u8] =
    include_bytes!("../opencl/vectorized_copy.spv");

const COPY_KERNEL_NAME: &str = "vectorized_copy";
const COPY_WG_SIZE: u32 = 128;
const COPY_MAX_WGS: u32 = 65535;

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

/// KV cache transfer benchmark for Intel XPU (Llama 3.1 70B, bf16).
#[derive(Parser, Debug)]
#[command(name = "kvbench_xpu_ze", about = "KV cache transfer bandwidth benchmark (XPU / Level Zero)")]
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
            "memcpy" | "append_memcpy" => Some(Backend::Memcpy),
            _ => None,
        }
    }

    fn all_labels() -> &'static str {
        "vectorized (or vec), memcpy (or append_memcpy)"
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
// Memory pair: holds src + dst buffers for a given direction
// ---------------------------------------------------------------------------

enum SideBuffers {
    /// USM host memory (page-locked, DMA-friendly).
    Host(Vec<ze::HostBuffer>),
    /// Regular heap memory (not pinned).
    System(Vec<Vec<u8>>),
    /// Device memory.
    Device(Vec<ze::DeviceBuffer>),
}

impl SideBuffers {
    fn ptr(&self, idx: usize) -> *mut c_void {
        match self {
            SideBuffers::Host(bufs) => bufs[idx].as_mut_ptr(),
            SideBuffers::System(bufs) => bufs[idx].as_ptr() as *mut c_void,
            SideBuffers::Device(bufs) => bufs[idx].as_mut_ptr(),
        }
    }
}

struct MemoryPair {
    src: SideBuffers,
    dst: SideBuffers,
}

fn allocate_side_host(ctx: &ze::Context, count: usize, size: usize) -> SideBuffers {
    SideBuffers::Host(
        (0..count)
            .map(|_| {
                let buf = ctx.alloc_host(size, 1).expect("zeMemAllocHost failed");
                // Fill with pattern to avoid zero-page tricks.
                unsafe {
                    std::ptr::write_bytes(buf.as_mut_ptr() as *mut u8, 0xAB, size);
                }
                buf
            })
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

fn allocate_side_device(
    ctx: &ze::Context,
    device: &ze::Device,
    count: usize,
    size: usize,
) -> SideBuffers {
    SideBuffers::Device(
        (0..count)
            .map(|_| {
                ctx.alloc_device(device, size, 1).expect("zeMemAllocDevice failed")
            })
            .collect(),
    )
}

fn allocate_host_side(
    ctx: &ze::Context,
    host_mem: HostMemKind,
    count: usize,
    size: usize,
) -> SideBuffers {
    match host_mem {
        HostMemKind::Pinned => allocate_side_host(ctx, count, size),
        HostMemKind::System => allocate_side_system(count, size),
    }
}

fn allocate_memory(
    ctx: &ze::Context,
    src_device: &ze::Device,
    dst_device: &ze::Device,
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
    MemoryPair {
        src: if src_host {
            allocate_host_side(ctx, host_mem, src_count, src_size)
        } else {
            allocate_side_device(ctx, src_device, src_count, src_size)
        },
        dst: if dst_host {
            allocate_host_side(ctx, host_mem, dst_count, dst_size)
        } else {
            allocate_side_device(ctx, dst_device, dst_count, dst_size)
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
///
/// `scattered`: NUM_LAYERS separate allocations (one per layer).
/// `contiguous`: one allocation holding all blocks packed sequentially.
///
/// Returns `(scattered_addrs, contiguous_addrs)` -- caller swaps to get
/// `(src, dst)` depending on copy direction.
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
                // Scattered (LW): within layer buffer, skip past earlier blocks' K+V
                let sc_offset = (b * OUTER_DIM + outer) * inner;
                // Contiguous (FC): block offset + layer/outer within block
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

/// Launch SPIR-V vectorized_copy kernel via compute command list.
/// Pointer-array uploads and kernel dispatch all go through the same CCS
/// queue, so in-order execution guarantees the kernel sees the arrays
/// without any cross-engine barrier.
fn execute_vectorized(
    cmd_compute: &ze::ImmediateCommandList,
    kernel: &ze::Kernel,
    src_addrs: &[u64],
    dst_addrs: &[u64],
    src_addrs_dev: &ze::DeviceBuffer,
    dst_addrs_dev: &ze::DeviceBuffer,
    ptr_array_bytes: usize,
    num_copies: usize,
) {
    // Upload pointer arrays on the same ImmediateCommandList with CCS
    // (given tiny, few KB, only limited benefit via BCS).
    cmd_compute
        .append_memcpy(
            src_addrs_dev.as_mut_ptr(),
            src_addrs.as_ptr() as *const c_void,
            ptr_array_bytes,
        )
        .expect("upload src_addrs");
    cmd_compute
        .append_memcpy(
            dst_addrs_dev.as_mut_ptr(),
            dst_addrs.as_ptr() as *const c_void,
            ptr_array_bytes,
        )
        .expect("upload dst_addrs");

    // Launch kernel -- in-order after the uploads above.
    let num_groups = std::cmp::min(num_copies as u32, COPY_MAX_WGS);
    cmd_compute
        .append_launch_kernel(
            kernel,
            ze::GroupCount { x: num_groups, y: 1, z: 1 },
        )
        .expect("launch kernel");
    cmd_compute.host_synchronize(u64::MAX).expect("sync compute");
}

/// Execute per-chunk append_memcpy via copy command list.
fn execute_memcpy(
    cmd_copy: &ze::ImmediateCommandList,
    src_addrs: &[u64],
    dst_addrs: &[u64],
    copy_size: usize,
    num_copies: usize,
) {
    for i in 0..num_copies {
        cmd_copy
            .append_memcpy(
                dst_addrs[i] as *mut c_void,
                src_addrs[i] as *const c_void,
                copy_size,
            )
            .expect("memcpy");
    }
    cmd_copy.host_synchronize(u64::MAX).expect("sync");
}

// ---------------------------------------------------------------------------
// Run one benchmark configuration
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_benchmark(
    ctx: &ze::Context,
    device: &ze::Device,
    dst_device: &ze::Device,
    cmd_copy: &ze::ImmediateCommandList,
    cmd_compute: Option<&ze::ImmediateCommandList>,
    kernel: Option<&ze::Kernel>,
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
        ctx, device, dst_device, direction, host_mem,
        src_count, src_size, dst_count, dst_size,
    );

    let (src_addrs, dst_addrs) = match pattern {
        Pattern::FcToFc => build_fc_pairs(&mem, num_blocks),
        Pattern::LwToFc => {
            // src=LW (scattered), dst=FC (contiguous)
            let (sc, fc) = build_lw_fc_pairs(&mem.src, &mem.dst, num_blocks, full_block_size, inner);
            (sc, fc)
        }
        Pattern::FcToLw => {
            // src=FC (contiguous), dst=LW (scattered)
            let (sc, fc) = build_lw_fc_pairs(&mem.dst, &mem.src, num_blocks, full_block_size, inner);
            (fc, sc)
        }
    };

    // Device-side pointer arrays for vectorized backend.
    let ptr_array_bytes = num_copies * std::mem::size_of::<u64>();
    let needs_ptr_arrays = matches!(backend, Backend::Vectorized);
    let src_addrs_dev = if needs_ptr_arrays {
        Some(ctx.alloc_device(device, ptr_array_bytes, 8).expect("alloc src_addrs_dev"))
    } else {
        None
    };
    let dst_addrs_dev = if needs_ptr_arrays {
        Some(ctx.alloc_device(device, ptr_array_bytes, 8).expect("alloc dst_addrs_dev"))
    } else {
        None
    };

    // Set kernel arguments once (they don't change across iterations).
    if matches!(backend, Backend::Vectorized) {
        let k = kernel.expect("vectorized backend requires kernel");
        let src_ptr = src_addrs_dev.as_ref().unwrap().as_mut_ptr() as u64;
        let dst_ptr = dst_addrs_dev.as_ref().unwrap().as_mut_ptr() as u64;
        let copy_sz = copy_size as u64;
        let n_pairs = num_copies as i32;
        k.set_arg(0, &src_ptr).expect("arg0");
        k.set_arg(1, &dst_ptr).expect("arg1");
        k.set_arg(2, &copy_sz).expect("arg2");
        k.set_arg(3, &n_pairs).expect("arg3");
    }

    // Warmup.
    for _ in 0..warmup_iters {
        match backend {
            Backend::Vectorized => execute_vectorized(
                cmd_compute.expect("vectorized needs CCS"),
                kernel.expect("vectorized needs kernel"),
                &src_addrs,
                &dst_addrs,
                src_addrs_dev.as_ref().unwrap(),
                dst_addrs_dev.as_ref().unwrap(),
                ptr_array_bytes,
                num_copies,
            ),
            Backend::Memcpy => {
                execute_memcpy(cmd_copy, &src_addrs, &dst_addrs, copy_size, num_copies)
            }
        }
    }

    // Timed iterations (host-side timing; L0 has no CUDA-like event timing).
    let mut host_elapsed = Vec::with_capacity(timed_iters);
    for _ in 0..timed_iters {
        let t0 = std::time::Instant::now();
        match backend {
            Backend::Vectorized => execute_vectorized(
                cmd_compute.expect("vectorized needs CCS"),
                kernel.expect("vectorized needs kernel"),
                &src_addrs,
                &dst_addrs,
                src_addrs_dev.as_ref().unwrap(),
                dst_addrs_dev.as_ref().unwrap(),
                ptr_array_bytes,
                num_copies,
            ),
            Backend::Memcpy => {
                execute_memcpy(cmd_copy, &src_addrs, &dst_addrs, copy_size, num_copies)
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

    // -- Initialize Level Zero ------------------------------------------------
    ze::init().expect("Level Zero initialization failed");

    let drivers = ze::drivers().expect("Failed to enumerate Level Zero drivers");
    if drivers.is_empty() {
        eprintln!("ERROR: No Level Zero drivers found.");
        std::process::exit(1);
    }

    let driver = &drivers[0];
    let devices = driver.devices().expect("Failed to enumerate devices");
    if devices.is_empty() {
        eprintln!("ERROR: No Level Zero devices found.");
        std::process::exit(1);
    }

    // Enumerate all Level Zero devices across all drivers (GPU, CPU, iGPU
    // may live under different drivers).
    eprintln!("All Level Zero devices:");
    for (drv_idx, drv) in drivers.iter().enumerate() {
        let drv_ctx = ze::Context::create(drv).expect("Failed to create context for discovery");
        let drv_devs = drv.devices().unwrap_or_default();
        for (dev_idx, dev) in drv_devs.iter().enumerate() {
            if let Ok(p) = drv_ctx.device_properties(dev) {
                let type_str = match p.device_type {
                    1 => "GPU",
                    2 => "CPU",
                    _ => continue,
                };
                let total_eus = p.num_slices * p.num_subslices_per_slice * p.num_eus_per_subslice;
                eprintln!("  Driver {drv_idx} / Device {dev_idx}: {}", p.name);
                eprintln!("    Type: {type_str}  Vendor: 0x{:04X}  DeviceID: 0x{:04X}", p.vendor_id, p.device_id);
                eprintln!(
                    "    Slices: {}  SubSlices/Slice: {}  EUs/SubSlice: {}  Total EUs: {}",
                    p.num_slices, p.num_subslices_per_slice, p.num_eus_per_subslice, total_eus,
                );
                eprintln!(
                    "    Threads/EU: {}  SIMD width: {}  Core clock: {} MHz",
                    p.num_threads_per_eu, p.physical_eu_simd_width, p.core_clock_rate,
                );
                eprintln!(
                    "    Max mem alloc: {:.1} GB  Max HW contexts: {}",
                    p.max_mem_alloc_size as f64 / (1024.0 * 1024.0 * 1024.0),
                    p.max_hardware_contexts,
                );
            }
        }
    }
    eprintln!();

    // The benchmark operates under driver 0.
    eprintln!("Benchmark driver: 0  ({} device(s))", devices.len());

    let ctx = ze::Context::create(driver).expect("Failed to create Level Zero context");

    // Print driver-0 device indices for quick reference.
    for (i, dev) in devices.iter().enumerate() {
        if let Ok(p) = ctx.device_properties(dev) {
            let type_str = match p.device_type {
                1 => "GPU",
                2 => "CPU",
                _ => "?",
            };
            eprintln!("  [{i}] {type_str}: {}", p.name);
        }
    }
    eprintln!();

    if cli.device >= devices.len() {
        eprintln!(
            "ERROR: --device {} is out of range. Only {} device(s) available (0..{}).",
            cli.device, devices.len(), devices.len() - 1
        );
        std::process::exit(1);
    }
    eprintln!("  Selected device: {}", cli.device);

    let has_d2dx = cli.direction.iter().any(|s| s == "d2dx" || s == "cross");
    if has_d2dx {
        if devices.len() < 2 {
            eprintln!("ERROR: Cross-device D2D (d2dx) requires at least 2 devices.");
            std::process::exit(1);
        }
        if cli.dst_device >= devices.len() {
            eprintln!(
                "ERROR: --dst-device {} is out of range. Only {} device(s) available (0..{}).",
                cli.dst_device, devices.len(), devices.len() - 1
            );
            std::process::exit(1);
        }
        if cli.dst_device == cli.device {
            eprintln!(
                "WARNING: --dst-device ({}) == --device ({}); use d2d instead of d2dx for same-device.",
                cli.dst_device, cli.device
            );
        }

        // Check P2P access capability between the two devices.
        match ctx.can_access_peer(&devices[cli.device], &devices[cli.dst_device]) {
            Ok(true) => {
                eprintln!("  P2P access: device {} -> device {} = SUPPORTED", cli.device, cli.dst_device);
            }
            Ok(false) => {
                eprintln!(
                    "ERROR: P2P access not supported between device {} and device {}.",
                    cli.device, cli.dst_device
                );
                eprintln!(
                    "  Cross-device D2D (d2dx) requires a platform with P2P support"
                );
                std::process::exit(1);
            }
            Err(e) => {
                eprintln!(
                    "WARNING: Failed to query P2P access (device {} -> device {}): {}",
                    cli.device, cli.dst_device, e
                );
                eprintln!("  Proceeding anyway — the benchmark may hang if P2P is not supported.");
            }
        }

        eprintln!("  Destination device: {}", cli.dst_device);
    }
    eprintln!();

    let device = &devices[cli.device];
    let dst_device_ref: &ze::Device = if has_d2dx {
        &devices[cli.dst_device]
    } else {
        device // placeholder; unused when d2dx is not requested
    };

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
                    // D2Dx + vectorized: not supported.
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

    // -- Create command lists -------------------------------------------------
    // BCS (copy engine) -- always needed.
    let copy_ordinal = ctx.copy_queue_ordinal(device);
    let cmd_copy = match copy_ordinal {
        Some(ord) => ctx
            .create_immediate_command_list_with_ordinal(device, ord)
            .expect("Failed to create BCS command list"),
        None => ctx
            .create_immediate_command_list(device)
            .expect("Failed to create command list (no BCS, using default)"),
    };

    // CCS (compute engine) -- only when vectorized backend is requested.
    let needs_compute = backends.iter().any(|b| matches!(b, Backend::Vectorized));
    let cmd_compute = if needs_compute {
        let compute_ordinal = ctx.compute_queue_ordinal(device)
            .expect("Failed to query compute queue ordinal");
        Some(
            ctx.create_immediate_command_list_with_ordinal(device, compute_ordinal)
                .expect("Failed to create CCS command list"),
        )
    } else {
        None
    };

    // -- Load SPIR-V module + kernel ------------------------------------------
    let (module, kernel) = if needs_compute {
        match ctx.create_module_from_spirv(device, VECTORIZED_COPY_SPIRV, None) {
            Ok(m) => {
                let k = m.create_kernel(COPY_KERNEL_NAME)
                    .expect("Failed to create vectorized_copy kernel");
                k.set_group_size(COPY_WG_SIZE, 1, 1)
                    .expect("Failed to set group size");
                // The kernel dereferences pointers stored in device
                // arrays (src_addrs_dev / dst_addrs_dev).  For D2D the
                // targets are device allocations; for H2D/D2H one side
                // is USM host memory.  Setting both HOST and DEVICE
                // covers all directions.
                k.set_indirect_access(
                    ze::KERNEL_INDIRECT_ACCESS_FLAG_HOST
                    | ze::KERNEL_INDIRECT_ACCESS_FLAG_DEVICE,
                ).expect("Failed to set indirect access flags");
                eprintln!("  SPIR-V module loaded ({} bytes)", VECTORIZED_COPY_SPIRV.len());
                (Some(m), Some(k))
            }
            Err(e) => {
                eprintln!(
                    "WARNING: Failed to load SPIR-V module: {:?}. \
                     Vectorized backend will not be available.",
                    e
                );
                (None, None)
            }
        }
    } else {
        (None, None)
    };
    let _ = &module; // keep module alive (kernel borrows it)

    // -- Print config ---------------------------------------------------------
    eprintln!("KV Cache Transfer Benchmark (XPU / Level Zero)");
    eprintln!("  Device ordinal: {}", cli.device);
    eprintln!("  Model: Llama 3.1 70B (bf16)");
    eprintln!(
        "  Layers: {NUM_LAYERS}, KV heads: {NUM_KV_HEADS}, Head dim: {HEAD_DIM}, Outer dim: {OUTER_DIM}"
    );
    eprintln!("  Warmup: {warmup_iters}, Timed: {timed_iters}");
    eprintln!("  Work-group size: {COPY_WG_SIZE}");
    eprintln!("  BCS ordinal: {:?}", copy_ordinal);
    if needs_compute {
        eprintln!("  CCS ordinal: {}", ctx.compute_queue_ordinal(device).unwrap());
    }
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
                        // Skip vectorized if SPIR-V failed to load.
                        if matches!(backend, Backend::Vectorized) && kernel.is_none() {
                            continue;
                        }

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

                            // The vectorized kernel dereferences raw pointers on the GPU.
                            // System heap memory (Vec<u8>) is not USM — the GPU cannot
                            // access it.  Only pinned (zeMemAllocHost) memory is usable.
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

                            let effective_dst = match direction {
                                Direction::D2Dx => dst_device_ref,
                                _ => device,
                            };

                            let (median_ms, bw) = run_benchmark(
                                &ctx,
                                device,
                                effective_dst,
                                &cmd_copy,
                                cmd_compute.as_ref(),
                                kernel.as_ref(),
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
