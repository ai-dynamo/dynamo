// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV cache transfer benchmark for Intel XPU (Level Zero).
//!
//! Compares the SPIR-V vectorized_copy kernel against individual
//! `append_memcpy` for layerwise vs fully-contiguous block transfers
//! using Llama 3.1 70B KV cache dimensions.
//!
//! # Backends
//!
//! - **vectorized** — SPIR-V kernel on the CCS (compute) engine.
//!   Uploads (src, dst) pointer arrays to device, then one kernel
//!   dispatch copies all chunks in parallel.
//! - **memcpy** — Individual `zeCommandListAppendMemoryCopy` calls
//!   on the BCS (copy / blitter) engine, one per chunk.
//!
//! # Transfer directions
//!
//! - **D2D** — Device-to-device on the same GPU. Both backends work.
//! - **H2D / D2H** — Host-to-device / device-to-host via PCIe.
//!   Only `memcpy` (BCS) works reliably. The `vectorized` (CCS) kernel
//!   dereferences host-pinned pointers from the GPU, which exceeds the
//!   xe driver watchdog timeout (~5 s) on discrete GPUs and triggers
//!   `ZE_RESULT_ERROR_DEVICE_LOST` with a CCS engine reset.
//!
//! # Transfer patterns
//!
//! - **fc_to_fc** — Fully-contiguous ↔ fully-contiguous.
//!   Both sides are allocated as `num_blocks` buffers of
//!   `full_block_size` each. One copy pair per block.
//! - **lw_to_fc** — Layerwise (scattered) → fully-contiguous.
//!   Source: `NUM_LAYERS` independent allocations (one per layer),
//!   each holding all blocks’ K+V for that layer.
//!   Destination: one contiguous allocation holding all blocks packed
//!   sequentially.  This models the real TLB / memory-controller
//!   pressure of a scatter-gather KV cache transfer.
//!
//! # Host memory kinds
//!
//! - **pinned** — USM host memory (`zeMemAllocHost`). Page-locked,
//!   DMA-friendly. This is what the real KV cache offload path uses.
//! - **system** — Regular heap memory (`Vec<u8>`). Not pinned.
//!   The L0 driver pins on-the-fly or stages through an internal
//!   bounce buffer, so H2D/D2H bandwidth is typically lower.
//!
//! # Limitations
//!
//! - The vectorized backend panics with DEVICE_LOST on H2D/D2H for
//!   discrete GPUs (B580, BMG). Use `--backend memcpy` or
//!   `--direction d2d` on these devices.
//! - Only one KV cache shape is modeled (Llama 3.1 70B bf16).
//!   Dimensions are compile-time constants.
//!
//! Output: CSV on stdout suitable for piping to a plotting script.
//!
//! # Usage
//!
//! ```sh
//! # Run ALL safe configurations (skips vectorized H2D/D2H which causes DEVICE_LOST):
//! cargo run --example kvbench_xpu --features kvbench-xpu --release -- \
//!   --direction d2d --backend vectorized,memcpy \
//!   --pattern fc_to_fc,lw_to_fc --host-mem pinned 2>/dev/null && \
//! cargo run --example kvbench_xpu --features kvbench-xpu --release -- \
//!   --direction h2d,d2h --backend memcpy \
//!   --pattern fc_to_fc,lw_to_fc --host-mem pinned,system 2>/dev/null
//!
//! # Run ALL configurations including vectorized H2D/D2H (may DEVICE_LOST):
//! cargo run --example kvbench_xpu --features kvbench-xpu --release -- \
//!   --direction h2d,d2h,d2d --backend vectorized,memcpy \
//!   --pattern fc_to_fc,lw_to_fc --host-mem pinned,system 2>/dev/null
//!
//! # Run all D2D configurations (both backends, both patterns):
//! cargo run --example kvbench_xpu --features kvbench-xpu --release -- \
//!   --direction d2d 2>/dev/null
//!
//! # D2D vectorized, LW→FC pattern only:
//! cargo run --example kvbench_xpu --features kvbench-xpu --release -- \
//!   --direction d2d --backend vectorized --pattern lw_to_fc
//!
//! # H2D/D2H (memcpy only — vectorized causes DEVICE_LOST on discrete GPUs):
//! cargo run --example kvbench_xpu --features kvbench-xpu --release -- \
//!   --direction h2d,d2h --backend memcpy
//!
//! # Compare pinned vs system host memory for H2D:
//! cargo run --example kvbench_xpu --features kvbench-xpu --release -- \
//!   --direction h2d --backend memcpy --host-mem pinned,system
//!
//! # Quick smoke test on device 2:
//! cargo run --example kvbench_xpu --features kvbench-xpu --release -- \
//!   --device 2 --num-blocks 1,4 --tokens-per-block 16 \
//!   --warmup 3 --iters 10
//!
//! # --- SYCL kernel A/B comparison (requires sycl-kernel feature) ---
//!
//! # L0 SPIR-V vs SYCL kernel, D2D:
//! cargo run --example kvbench_xpu --features kvbench-xpu,sycl-kernel --release -- \
//!   --direction d2d --backend vectorized,vectorized_sycl --pattern lw_to_fc \
//!   --num-blocks 32,64,128 2>/dev/null
//!
//! # Just the SYCL kernel:
//! cargo run --example kvbench_xpu --features kvbench-xpu,sycl-kernel --release -- \
//!   --direction d2d --backend sycl
//! ```

use std::ffi::c_void;
use std::sync::Arc;

use clap::Parser;
use syclrc::level_zero::ze::sys;
use syclrc::{ZeDevice, ZeHostSlice, ZeImmediateCmdList, ZeKernel, ZeModule, ZeSlice};

use kvbm_kernels::ze_vectorized_copy as vc;
#[cfg(feature = "sycl-kernel")]
use kvbm_kernels::sycl_vectorized_copy::SyclVectorizedCopy;

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
#[command(name = "kvbench_xpu", about = "KV cache transfer bandwidth benchmark (XPU)")]
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

    /// Comma-separated patterns: fc_to_fc, lw_to_fc.
    #[arg(long, default_value = "fc_to_fc,lw_to_fc", value_delimiter = ',')]
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
}

// ---------------------------------------------------------------------------
// Transfer direction
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum Direction {
    H2D,
    D2H,
    D2D,
}

impl Direction {
    fn label(&self) -> &'static str {
        match self {
            Direction::H2D => "h2d",
            Direction::D2H => "d2h",
            Direction::D2D => "d2d",
        }
    }

    fn from_str(s: &str) -> Option<Self> {
        match s {
            "h2d" => Some(Direction::H2D),
            "d2h" => Some(Direction::D2H),
            "d2d" => Some(Direction::D2D),
            _ => None,
        }
    }

    fn all_labels() -> &'static str {
        "h2d, d2h, d2d"
    }
}

// ---------------------------------------------------------------------------
// Transfer pattern
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum Pattern {
    FcToFc,
    LwToFc,
}

impl Pattern {
    fn label(&self) -> &'static str {
        match self {
            Pattern::FcToFc => "fc_to_fc",
            Pattern::LwToFc => "lw_to_fc",
        }
    }

    fn from_str(s: &str) -> Option<Self> {
        match s {
            "fc_to_fc" | "fc" => Some(Pattern::FcToFc),
            "lw_to_fc" | "lw" => Some(Pattern::LwToFc),
            _ => None,
        }
    }

    fn all_labels() -> &'static str {
        "fc_to_fc (or fc), lw_to_fc (or lw)"
    }
}

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum Backend {
    Vectorized,
    Memcpy,
    #[cfg(feature = "sycl-kernel")]
    VectorizedSycl,
}

impl Backend {
    fn label(&self) -> &'static str {
        match self {
            Backend::Vectorized => "vectorized",
            Backend::Memcpy => "memcpy",
            #[cfg(feature = "sycl-kernel")]
            Backend::VectorizedSycl => "vectorized_sycl",
        }
    }

    fn from_str(s: &str) -> Option<Self> {
        match s {
            "vectorized" | "vec" => Some(Backend::Vectorized),
            "memcpy" | "append_memcpy" => Some(Backend::Memcpy),
            #[cfg(feature = "sycl-kernel")]
            "vectorized_sycl" | "vec_sycl" | "sycl" => Some(Backend::VectorizedSycl),
            _ => None,
        }
    }

    fn all_labels() -> &'static str {
        if cfg!(feature = "sycl-kernel") {
            "vectorized (or vec), memcpy (or append_memcpy), vectorized_sycl (or sycl)"
        } else {
            "vectorized (or vec), memcpy (or append_memcpy)"
        }
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
    Host(Vec<ZeHostSlice<u8>>),
    System(Vec<Vec<u8>>),
    Device(Vec<ZeSlice<u8>>),
}

impl SideBuffers {
    fn ptr(&self, idx: usize) -> *mut c_void {
        match self {
            SideBuffers::Host(bufs) => bufs[idx].as_ptr() as *mut c_void,
            SideBuffers::System(bufs) => bufs[idx].as_ptr() as *mut c_void,
            SideBuffers::Device(bufs) => bufs[idx].as_ptr() as *mut c_void,
        }
    }
}

struct MemoryPair {
    src: SideBuffers,
    dst: SideBuffers,
}

fn allocate_side_host(dev: &Arc<ZeDevice>, count: usize, size: usize) -> SideBuffers {
    SideBuffers::Host(
        (0..count)
            .map(|_| {
                let mut buf = dev
                    .alloc_host::<u8>(size)
                    .expect("zeMemAllocHost failed");
                buf.fill(0xAB); // Avoid benchmarking zero-page tricks.
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

fn allocate_side_device(dev: &Arc<ZeDevice>, count: usize, size: usize) -> SideBuffers {
    let queue = dev.new_queue().expect("queue for alloc");
    SideBuffers::Device(
        (0..count)
            .map(|_| queue.alloc_zeros::<u8>(size).expect("alloc"))
            .collect(),
    )
}

fn allocate_host_side(
    dev: &Arc<ZeDevice>,
    host_mem: HostMemKind,
    count: usize,
    size: usize,
) -> SideBuffers {
    match host_mem {
        HostMemKind::Pinned => allocate_side_host(dev, count, size),
        HostMemKind::System => allocate_side_system(count, size),
    }
}

fn allocate_memory(
    dev: &Arc<ZeDevice>,
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
        Direction::D2D => (false, false),
    };
    MemoryPair {
        src: if src_host {
            allocate_host_side(dev, host_mem, src_count, src_size)
        } else {
            allocate_side_device(dev, src_count, src_size)
        },
        dst: if dst_host {
            allocate_host_side(dev, host_mem, dst_count, dst_size)
        } else {
            allocate_side_device(dev, dst_count, dst_size)
        },
    }
}

// ---------------------------------------------------------------------------
// Pointer pair lists
// ---------------------------------------------------------------------------

/// FC↔FC: one copy pair per block, each `full_block_size` bytes.
/// Both src and dst have `num_blocks` separate allocations.
fn build_fc_pairs(mem: &MemoryPair, num_blocks: usize) -> (Vec<u64>, Vec<u64>) {
    let mut src_addrs = Vec::with_capacity(num_blocks);
    let mut dst_addrs = Vec::with_capacity(num_blocks);
    for b in 0..num_blocks {
        src_addrs.push(mem.src.ptr(b) as u64);
        dst_addrs.push(mem.dst.ptr(b) as u64);
    }
    (src_addrs, dst_addrs)
}

/// LW→FC: src has NUM_LAYERS separate allocations (one per layer, scattered),
/// dst is one contiguous allocation (all blocks packed sequentially).
fn build_lw_pairs(
    mem: &MemoryPair,
    num_blocks: usize,
    full_block_size: usize,
    inner: usize,
) -> (Vec<u64>, Vec<u64>) {
    let total = num_blocks * NUM_LAYERS * OUTER_DIM;
    let mut src_addrs = Vec::with_capacity(total);
    let mut dst_addrs = Vec::with_capacity(total);
    let dst_base = mem.dst.ptr(0) as u64; // single contiguous FC allocation
    for b in 0..num_blocks {
        for layer in 0..NUM_LAYERS {
            let src_base = mem.src.ptr(layer) as u64;
            for outer in 0..OUTER_DIM {
                // Src (LW): within layer buffer, skip past earlier blocks' K+V
                let src_offset = (b * OUTER_DIM + outer) * inner;
                // Dst (FC): contiguous — block offset + layer/outer within block
                let dst_offset = b * full_block_size
                    + (layer * OUTER_DIM + outer) * inner;
                src_addrs.push(src_base + src_offset as u64);
                dst_addrs.push(dst_base + dst_offset as u64);
            }
        }
    }
    (src_addrs, dst_addrs)
}

// ---------------------------------------------------------------------------
// Execute one iteration (shared by warmup and timed loops)
// ---------------------------------------------------------------------------

unsafe fn execute_vectorized(
    cmd_copy: &ZeImmediateCmdList,
    cmd_compute: &ZeImmediateCmdList,
    kernel: &ZeKernel,
    src_addrs: &[u64],
    dst_addrs: &[u64],
    src_addrs_dev: &ZeSlice<u8>,
    dst_addrs_dev: &ZeSlice<u8>,
    ptr_array_bytes: usize,
    num_copies: usize,
) {
    unsafe {
        // Upload pointer arrays.
        cmd_copy
            .append_memcpy(
                src_addrs_dev.as_ptr() as *mut c_void,
                src_addrs.as_ptr() as *const c_void,
                ptr_array_bytes,
                std::ptr::null_mut(),
                &mut [],
            )
            .expect("upload src_addrs");
        cmd_copy
            .append_memcpy(
                dst_addrs_dev.as_ptr() as *mut c_void,
                dst_addrs.as_ptr() as *const c_void,
                ptr_array_bytes,
                std::ptr::null_mut(),
                &mut [],
            )
            .expect("upload dst_addrs");
    }
    cmd_copy.host_synchronize(u64::MAX).expect("sync copy");

    // Kernel arguments are set once by the caller (run_benchmark),
    // not per iteration — they don't change between iterations.

    let num_groups = std::cmp::min(num_copies as u32, vc::MAX_GROUPS);
    let group_count = sys::ze_group_count_t {
        groupCountX: num_groups,
        groupCountY: 1,
        groupCountZ: 1,
    };
    unsafe {
        cmd_compute
            .append_launch_kernel(kernel, &group_count, None, &mut [])
            .expect("launch kernel");
    }
    cmd_compute.host_synchronize(u64::MAX).expect("sync compute");
}

unsafe fn execute_memcpy(
    cmd_copy: &ZeImmediateCmdList,
    src_addrs: &[u64],
    dst_addrs: &[u64],
    copy_size: usize,
    num_copies: usize,
) {
    for i in 0..num_copies {
        unsafe {
            cmd_copy
                .append_memcpy(
                    dst_addrs[i] as *mut c_void,
                    src_addrs[i] as *const c_void,
                    copy_size,
                    std::ptr::null_mut(),
                    &mut [],
                )
                .expect("memcpy");
        }
    }
    cmd_copy.host_synchronize(u64::MAX).expect("sync");
}

#[cfg(feature = "sycl-kernel")]
fn get_sycl_vc(dev: &Arc<ZeDevice>) -> &'static SyclVectorizedCopy {
    use std::sync::OnceLock;
    static SYCL_VC: OnceLock<SyclVectorizedCopy> = OnceLock::new();
    SYCL_VC.get_or_init(|| {
        SyclVectorizedCopy::new(
            dev.ze_context() as *mut c_void,
            dev.ze_device() as *mut c_void,
        )
        .expect("Failed to initialize SyclVectorizedCopy")
    })
}

#[cfg(feature = "sycl-kernel")]
unsafe fn execute_vectorized_sycl(
    cmd_copy: &ZeImmediateCmdList,
    sycl_vc: &SyclVectorizedCopy,
    src_addrs: &[u64],
    dst_addrs: &[u64],
    src_addrs_dev: &ZeSlice<u8>,
    dst_addrs_dev: &ZeSlice<u8>,
    ptr_array_bytes: usize,
    copy_size: usize,
    num_copies: usize,
) {
    // Upload pointer arrays to device (same as L0 path).
    unsafe {
        cmd_copy
            .append_memcpy(
                src_addrs_dev.as_ptr() as *mut c_void,
                src_addrs.as_ptr() as *const c_void,
                ptr_array_bytes,
                std::ptr::null_mut(),
                &mut [],
            )
            .expect("upload src_addrs");
        cmd_copy
            .append_memcpy(
                dst_addrs_dev.as_ptr() as *mut c_void,
                dst_addrs.as_ptr() as *const c_void,
                ptr_array_bytes,
                std::ptr::null_mut(),
                &mut [],
            )
            .expect("upload dst_addrs");
    }
    cmd_copy.host_synchronize(u64::MAX).expect("sync copy");

    // Dispatch via SYCL runtime.
    sycl_vc
        .run(
            src_addrs_dev.as_ptr() as u64,
            dst_addrs_dev.as_ptr() as u64,
            copy_size as u64,
            num_copies as i32,
        )
        .expect("sycl_vc_run failed");
}

// ---------------------------------------------------------------------------
// Run one benchmark configuration
// ---------------------------------------------------------------------------

fn run_benchmark(
    dev: &Arc<ZeDevice>,
    cmd_copy: &ZeImmediateCmdList,
    cmd_compute: &ZeImmediateCmdList,
    kernel: &ZeKernel,
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
        Pattern::LwToFc => (inner, num_blocks * NUM_LAYERS * OUTER_DIM),
    };

    // Allocate memory.
    //   FC↔FC: src = num_blocks × full_block_size,
    //            dst = num_blocks × full_block_size
    //   LW→FC: src = NUM_LAYERS × layer_buf_size  (scattered)
    //            dst = 1 × (full_block_size × num_blocks)  (contiguous)
    let (src_count, src_size, dst_count, dst_size) = match pattern {
        Pattern::FcToFc => (num_blocks, full_block_size, num_blocks, full_block_size),
        Pattern::LwToFc => {
            let layer_buf_size = num_blocks * OUTER_DIM * inner;
            // dst: 1 contiguous FC allocation holding all blocks
            (NUM_LAYERS, layer_buf_size, 1, full_block_size * num_blocks)
        }
    };
    let mem = allocate_memory(dev, direction, host_mem, src_count, src_size, dst_count, dst_size);

    let (src_addrs, dst_addrs) = match pattern {
        Pattern::FcToFc => build_fc_pairs(&mem, num_blocks),
        Pattern::LwToFc => build_lw_pairs(&mem, num_blocks, full_block_size, inner),
    };

    // Scratch for vectorized backend (device-side pointer arrays).
    let queue = dev.new_queue().expect("queue");
    let ptr_array_bytes = num_copies * std::mem::size_of::<u64>();
    let needs_ptr_arrays = match backend {
        Backend::Vectorized => true,
        #[cfg(feature = "sycl-kernel")]
        Backend::VectorizedSycl => true,
        _ => false,
    };
    let src_addrs_dev: Option<ZeSlice<u8>> = if needs_ptr_arrays {
        Some(unsafe { queue.alloc::<u8>(ptr_array_bytes).expect("alloc src_addrs_dev") })
    } else {
        None
    };
    let dst_addrs_dev: Option<ZeSlice<u8>> = if needs_ptr_arrays {
        Some(unsafe { queue.alloc::<u8>(ptr_array_bytes).expect("alloc dst_addrs_dev") })
    } else {
        None
    };

    // Set kernel arguments once (they don't change across iterations).
    if matches!(backend, Backend::Vectorized) {
        let src_ptr = src_addrs_dev.as_ref().unwrap().as_ptr() as u64;
        let dst_ptr = dst_addrs_dev.as_ref().unwrap().as_ptr() as u64;
        let copy_sz = copy_size as u64;
        let n_pairs = num_copies as i32;
        unsafe {
            kernel.set_arg(0, &src_ptr).expect("arg0");
            kernel.set_arg(1, &dst_ptr).expect("arg1");
            kernel.set_arg(2, &copy_sz).expect("arg2");
            kernel.set_arg(3, &n_pairs).expect("arg3");
        }
    }

    // Warmup.
    for _ in 0..warmup_iters {
        unsafe {
            match backend {
                Backend::Vectorized => execute_vectorized(
                    cmd_copy,
                    cmd_compute,
                    kernel,
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
                #[cfg(feature = "sycl-kernel")]
                Backend::VectorizedSycl => execute_vectorized_sycl(
                    cmd_copy,
                    get_sycl_vc(dev),
                    &src_addrs,
                    &dst_addrs,
                    src_addrs_dev.as_ref().unwrap(),
                    dst_addrs_dev.as_ref().unwrap(),
                    ptr_array_bytes,
                    copy_size,
                    num_copies,
                ),
            }
        }
    }

    // Timed iterations.
    let mut host_elapsed = Vec::with_capacity(timed_iters);
    for _ in 0..timed_iters {
        let t0 = std::time::Instant::now();
        unsafe {
            match backend {
                Backend::Vectorized => execute_vectorized(
                    cmd_copy,
                    cmd_compute,
                    kernel,
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
                #[cfg(feature = "sycl-kernel")]
                Backend::VectorizedSycl => execute_vectorized_sycl(
                    cmd_copy,
                    get_sycl_vc(dev),
                    &src_addrs,
                    &dst_addrs,
                    src_addrs_dev.as_ref().unwrap(),
                    dst_addrs_dev.as_ref().unwrap(),
                    ptr_array_bytes,
                    copy_size,
                    num_copies,
                ),
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

    // ── Enumerate all Level Zero devices ─────────────────────────────────
    let device_count = ZeDevice::device_count().expect("Failed to query Level Zero device count");
    if device_count == 0 {
        eprintln!("ERROR: No Level Zero devices found.");
        std::process::exit(1);
    }

    eprintln!("Detected {device_count} Level Zero device(s):");
    for i in 0..device_count {
        let d = ZeDevice::new(i).expect("Failed to open device for enumeration");
        let name = d.name().unwrap_or_else(|_| "unknown".into());
        let mem_props = d.memory_properties().unwrap_or_default();
        let vram_bytes: u64 = mem_props.iter().map(|m| m.totalSize).sum();
        let vram_gib = vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let marker = if i == cli.device { "  <-- selected" } else { "" };
        eprintln!("  [{i}] {name}  ({vram_gib:.1} GiB VRAM){marker}");
    }

    if cli.device >= device_count {
        eprintln!(
            "ERROR: --device {} is out of range. Only {} device(s) available (0..{}).",
            cli.device, device_count, device_count - 1
        );
        std::process::exit(1);
    }
    eprintln!();

    let directions = parse_directions(&cli.direction);
    let patterns = parse_patterns(&cli.pattern);
    let backends = parse_backends(&cli.backend);
    let host_mem_kinds = parse_host_mem_kinds(&cli.host_mem);
    let tpb_options = &cli.tokens_per_block;
    let num_blocks_options = &cli.num_blocks;
    let warmup_iters = cli.warmup;
    let timed_iters = cli.iters;

    let total_tests =
        tpb_options.len() * num_blocks_options.len() * directions.len() * patterns.len() * backends.len() * host_mem_kinds.len();

    // Initialize Level Zero device.
    let dev = ZeDevice::new(cli.device).expect("Failed to create ZeDevice");
    let dev_name = dev.name().unwrap_or_else(|_| "unknown".into());

    // Immediate command lists: copy (BCS) and compute (CCS).
    let cmd_copy =
        ZeImmediateCmdList::new_copy(dev.clone()).expect("Failed to create copy cmd list");
    let cmd_compute =
        ZeImmediateCmdList::new_compute(dev.clone()).expect("Failed to create compute cmd list");

    // Compile SPIR-V module and create kernel.
    let module = Arc::new(
        ZeModule::from_spirv(&dev, vc::SPIRV, None).expect("Failed to compile SPIR-V module"),
    );
    let kernel = ZeKernel::new(&module, vc::KERNEL_NAME).expect("Failed to create kernel");
    kernel
        .set_group_size(vc::WORK_GROUP_SIZE, 1, 1)
        .expect("Failed to set group size");

    // Print config.
    eprintln!("KV Cache Transfer Benchmark (XPU / Level Zero)");
    eprintln!("  Device: {dev_name} (device {})", cli.device);
    eprintln!("  Model: Llama 3.1 70B (bf16)");
    eprintln!(
        "  Layers: {NUM_LAYERS}, KV heads: {NUM_KV_HEADS}, Head dim: {HEAD_DIM}, Outer dim: {OUTER_DIM}"
    );
    eprintln!("  Warmup: {warmup_iters}, Timed: {timed_iters}");
    eprintln!("  SPIR-V module: {} bytes", vc::SPIRV.len());
    eprintln!("  Work-group size: {}", vc::WORK_GROUP_SIZE);
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

    // CSV header.
    println!(
        "tokens_per_block,num_blocks,pattern,direction,backend,host_mem,total_bytes,inner_bytes,copy_size,num_copies,median_ms,bandwidth_gbps"
    );

    let mut test_num = 0;
    for &tpb in tpb_options {
        let inner = tpb * NUM_KV_HEADS * HEAD_DIM * ELEM_SIZE;
        let full_block_size = inner * OUTER_DIM * NUM_LAYERS;

        eprintln!(
            "--- tokens_per_block={tpb}, inner={inner} bytes ({} KB), block={full_block_size} bytes ({:.1} MB) ---",
            inner / 1024,
            full_block_size as f64 / (1024.0 * 1024.0)
        );

        for &num_blocks in num_blocks_options {
            let total_bytes = full_block_size * num_blocks;

            for &direction in &directions {
                for &pattern in &patterns {
                    let (copy_size, num_copies) = match pattern {
                        Pattern::FcToFc => (full_block_size, num_blocks),
                        Pattern::LwToFc => (inner, num_blocks * NUM_LAYERS * OUTER_DIM),
                    };

                    for &backend in &backends {
                        for &host_mem in &host_mem_kinds {
                            // D2D doesn't involve host memory; skip extra host_mem variants.
                            if matches!(direction, Direction::D2D)
                                && !matches!(host_mem, HostMemKind::Pinned)
                            {
                                continue;
                            }

                            test_num += 1;

                            let host_label = if matches!(direction, Direction::D2D) {
                                "-"
                            } else {
                                host_mem.label()
                            };

                            eprint!(
                                "  [{test_num}/{total_tests}] tpb={tpb} N={num_blocks:>3} {:<8} {:<6} {:<12} host={:<7} ... ",
                                pattern.label(),
                                direction.label(),
                                backend.label(),
                                host_label,
                            );

                            let (median_ms, bw) = run_benchmark(
                                &dev,
                                &cmd_copy,
                                &cmd_compute,
                                &kernel,
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
                                "{tpb},{num_blocks},{},{},{},{host_label},{total_bytes},{inner},{copy_size},{num_copies},{median_ms:.4},{bw:.2}",
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
