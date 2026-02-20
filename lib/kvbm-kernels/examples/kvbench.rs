// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV cache transfer benchmark.
//!
//! Compares vectorized copy kernel, cudaMemcpyBatchAsync, and individual
//! cudaMemcpyAsync for layerwise vs fully-contiguous block transfers using
//! Llama 3.1 70B KV cache dimensions.
//!
//! Output: CSV on stdout suitable for piping to a plotting script.
//!
//! Usage:
//!   # Run all configurations:
//!   cargo run --example kvbench --features kvbench 2>/dev/null
//!
//!   # Single test:
//!   cargo run --example kvbench --features kvbench -- \
//!     --num-blocks 16 --tokens-per-block 32 --backend vectorized --direction d2d
//!
//!   # Subset:
//!   cargo run --example kvbench --features kvbench -- \
//!     --num-blocks 1,4,16,64 --backend vectorized,batched --direction d2d --pattern fc_to_fc
//!
//!   # Pipe to plotter:
//!   cargo run --example kvbench --features kvbench 2>/dev/null | python3 scripts/plot_roofline.py

use std::ffi::c_void;

use clap::Parser;
use cudarc::driver::CudaContext;
use cudarc::runtime::sys as cuda_runtime;

use kvbm_kernels::{MemcpyBatchMode, memcpy_batch, vectorized_copy};

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

/// KV cache transfer benchmark (Llama 3.1 70B, bf16).
#[derive(Parser, Debug)]
#[command(name = "kvbench", about = "KV cache transfer bandwidth benchmark")]
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

    /// Comma-separated backends: vectorized, batched, memcpy_async.
    #[arg(
        long,
        default_value = "vectorized,batched,memcpy_async",
        value_delimiter = ','
    )]
    backend: Vec<String>,

    /// Comma-separated directions: h2d, d2h, d2d.
    #[arg(long, default_value = "h2d,d2h,d2d", value_delimiter = ',')]
    direction: Vec<String>,

    /// Comma-separated patterns: fc_to_fc, lw_to_fc.
    #[arg(long, default_value = "fc_to_fc,lw_to_fc", value_delimiter = ',')]
    pattern: Vec<String>,

    /// Number of warmup iterations.
    #[arg(long, default_value = "10")]
    warmup: usize,

    /// Number of timed iterations.
    #[arg(long, default_value = "100")]
    iters: usize,
}

// ---------------------------------------------------------------------------
// Direct FFI for CUDA runtime functions not exposed through cudarc
// ---------------------------------------------------------------------------
unsafe extern "C" {
    fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> u32;
    fn cudaFreeHost(ptr: *mut c_void) -> u32;
    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> u32;
    fn cudaFree(ptr: *mut c_void) -> u32;
    fn cudaEventCreate(event: *mut cuda_runtime::cudaEvent_t) -> u32;
    fn cudaEventDestroy(event: cuda_runtime::cudaEvent_t) -> u32;
    fn cudaEventRecord(event: cuda_runtime::cudaEvent_t, stream: cuda_runtime::cudaStream_t)
    -> u32;
    fn cudaEventSynchronize(event: cuda_runtime::cudaEvent_t) -> u32;
    fn cudaEventElapsedTime(
        ms: *mut f32,
        start: cuda_runtime::cudaEvent_t,
        end: cuda_runtime::cudaEvent_t,
    ) -> u32;
    fn cudaStreamSynchronize(stream: cuda_runtime::cudaStream_t) -> u32;
    fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: u32,
        stream: cuda_runtime::cudaStream_t,
    ) -> u32;
}

const CUDA_MEMCPY_HOST_TO_DEVICE: u32 = 1;

// ---------------------------------------------------------------------------
// Memory management helpers
// ---------------------------------------------------------------------------

/// RAII wrapper for pinned host memory.
struct PinnedBuffer {
    ptr: *mut c_void,
    _len: usize,
}

impl PinnedBuffer {
    fn new(len: usize) -> Self {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let err = unsafe { cudaMallocHost(&mut ptr, len) };
        assert_eq!(err, 0, "cudaMallocHost failed: {err}");
        // Fill with pattern so we're not benchmarking zero-page tricks
        unsafe { std::ptr::write_bytes(ptr as *mut u8, 0xAB, len) };
        Self { ptr, _len: len }
    }

    fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { cudaFreeHost(self.ptr) };
        }
    }
}

/// RAII wrapper for device memory.
struct DeviceBuffer {
    ptr: *mut c_void,
    _len: usize,
}

impl DeviceBuffer {
    fn new(len: usize) -> Self {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let err = unsafe { cudaMalloc(&mut ptr, len) };
        assert_eq!(err, 0, "cudaMalloc failed: {err}");
        Self { ptr, _len: len }
    }

    fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { cudaFree(self.ptr) };
        }
    }
}

/// RAII wrapper for CUDA events.
struct CudaEvent {
    event: cuda_runtime::cudaEvent_t,
}

impl CudaEvent {
    fn new() -> Self {
        let mut event: cuda_runtime::cudaEvent_t = std::ptr::null_mut();
        let err = unsafe { cudaEventCreate(&mut event) };
        assert_eq!(err, 0, "cudaEventCreate failed: {err}");
        Self { event }
    }

    fn record(&self, stream: cuda_runtime::cudaStream_t) {
        let err = unsafe { cudaEventRecord(self.event, stream) };
        assert_eq!(err, 0, "cudaEventRecord failed: {err}");
    }

    fn synchronize(&self) {
        let err = unsafe { cudaEventSynchronize(self.event) };
        assert_eq!(err, 0, "cudaEventSynchronize failed: {err}");
    }

    fn elapsed_ms(&self, start: &CudaEvent) -> f32 {
        let mut ms: f32 = 0.0;
        let err = unsafe { cudaEventElapsedTime(&mut ms, start.event, self.event) };
        assert_eq!(err, 0, "cudaEventElapsedTime failed: {err}");
        ms
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        if !self.event.is_null() {
            unsafe { cudaEventDestroy(self.event) };
        }
    }
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
    Batched,
    MemcpyAsync,
}

impl Backend {
    fn label(&self) -> &'static str {
        match self {
            Backend::Vectorized => "vectorized",
            Backend::Batched => "batched",
            Backend::MemcpyAsync => "memcpy_async",
        }
    }

    fn from_str(s: &str) -> Option<Self> {
        match s {
            "vectorized" | "vec" => Some(Backend::Vectorized),
            "batched" | "batch" => Some(Backend::Batched),
            "memcpy_async" | "async" | "memcpy" => Some(Backend::MemcpyAsync),
            _ => None,
        }
    }

    fn all_labels() -> &'static str {
        "vectorized (or vec), batched (or batch), memcpy_async (or async/memcpy)"
    }
}

// ---------------------------------------------------------------------------
// Allocate src/dst memory pair for a given direction
// ---------------------------------------------------------------------------

struct MemoryPair {
    src_bufs: SideBuffers,
    dst_bufs: SideBuffers,
}

enum SideBuffers {
    Pinned(Vec<PinnedBuffer>),
    Device(Vec<DeviceBuffer>),
}

impl SideBuffers {
    fn block_ptr(&self, block_idx: usize) -> *mut c_void {
        match self {
            SideBuffers::Pinned(bufs) => bufs[block_idx].as_ptr(),
            SideBuffers::Device(bufs) => bufs[block_idx].as_ptr(),
        }
    }
}

fn allocate_memory(direction: Direction, num_blocks: usize, block_size: usize) -> MemoryPair {
    match direction {
        Direction::H2D => MemoryPair {
            src_bufs: SideBuffers::Pinned(
                (0..num_blocks)
                    .map(|_| PinnedBuffer::new(block_size))
                    .collect(),
            ),
            dst_bufs: SideBuffers::Device(
                (0..num_blocks)
                    .map(|_| DeviceBuffer::new(block_size))
                    .collect(),
            ),
        },
        Direction::D2H => MemoryPair {
            src_bufs: SideBuffers::Device(
                (0..num_blocks)
                    .map(|_| DeviceBuffer::new(block_size))
                    .collect(),
            ),
            dst_bufs: SideBuffers::Pinned(
                (0..num_blocks)
                    .map(|_| PinnedBuffer::new(block_size))
                    .collect(),
            ),
        },
        Direction::D2D => MemoryPair {
            src_bufs: SideBuffers::Device(
                (0..num_blocks)
                    .map(|_| DeviceBuffer::new(block_size))
                    .collect(),
            ),
            dst_bufs: SideBuffers::Device(
                (0..num_blocks)
                    .map(|_| DeviceBuffer::new(block_size))
                    .collect(),
            ),
        },
    }
}

// ---------------------------------------------------------------------------
// Build pointer lists
// ---------------------------------------------------------------------------

/// For FC<=>FC: one (src, dst) pair per block, each of full_block_size.
fn build_fc_ptrs(mem: &MemoryPair, num_blocks: usize) -> (Vec<*const c_void>, Vec<*mut c_void>) {
    let mut src_ptrs = Vec::with_capacity(num_blocks);
    let mut dst_ptrs = Vec::with_capacity(num_blocks);
    for b in 0..num_blocks {
        src_ptrs.push(mem.src_bufs.block_ptr(b) as *const c_void);
        dst_ptrs.push(mem.dst_bufs.block_ptr(b));
    }
    (src_ptrs, dst_ptrs)
}

/// For LW<=>FC: loop over blocks, layers, outers.
/// Each entry is `inner` bytes at the appropriate offset into the contiguous block.
fn build_lw_ptrs(
    mem: &MemoryPair,
    num_blocks: usize,
    inner: usize,
) -> (Vec<*const c_void>, Vec<*mut c_void>) {
    let total = num_blocks * NUM_LAYERS * OUTER_DIM;
    let mut src_ptrs = Vec::with_capacity(total);
    let mut dst_ptrs = Vec::with_capacity(total);

    for b in 0..num_blocks {
        let src_base = mem.src_bufs.block_ptr(b) as *const u8;
        let dst_base = mem.dst_bufs.block_ptr(b) as *mut u8;
        for layer in 0..NUM_LAYERS {
            for outer in 0..OUTER_DIM {
                let offset = (layer * OUTER_DIM + outer) * inner;
                unsafe {
                    src_ptrs.push(src_base.add(offset) as *const c_void);
                    dst_ptrs.push(dst_base.add(offset) as *mut c_void);
                }
            }
        }
    }
    (src_ptrs, dst_ptrs)
}

// ---------------------------------------------------------------------------
// Run one benchmark configuration
// ---------------------------------------------------------------------------

fn run_benchmark(
    stream_raw: cuda_runtime::cudaStream_t,
    pattern: Pattern,
    direction: Direction,
    backend: Backend,
    tokens_per_block: usize,
    num_blocks: usize,
    warmup_iters: usize,
    timed_iters: usize,
) -> Option<(f64, f64)> {
    let inner = tokens_per_block * NUM_KV_HEADS * HEAD_DIM * ELEM_SIZE;
    let full_block_size = inner * OUTER_DIM * NUM_LAYERS;
    let total_bytes = full_block_size * num_blocks;

    // Allocate memory
    let mem = allocate_memory(direction, num_blocks, full_block_size);

    // Build pointer lists based on pattern
    let (copy_size, num_copies) = match pattern {
        Pattern::FcToFc => (full_block_size, num_blocks),
        Pattern::LwToFc => (inner, num_blocks * NUM_LAYERS * OUTER_DIM),
    };

    let (src_ptrs, dst_ptrs) = match pattern {
        Pattern::FcToFc => build_fc_ptrs(&mem, num_blocks),
        Pattern::LwToFc => build_lw_ptrs(&mem, num_blocks, inner),
    };

    // For vectorized copy: allocate device pointer arrays
    let (src_ptrs_dev, dst_ptrs_dev) = if matches!(backend, Backend::Vectorized) {
        let ptr_array_bytes = num_copies * std::mem::size_of::<usize>();
        (
            Some(DeviceBuffer::new(ptr_array_bytes)),
            Some(DeviceBuffer::new(ptr_array_bytes)),
        )
    } else {
        (None, None)
    };

    let start_event = CudaEvent::new();
    let end_event = CudaEvent::new();

    let mut elapsed_samples = Vec::with_capacity(timed_iters);

    for iter in 0..(warmup_iters + timed_iters) {
        let is_timed = iter >= warmup_iters;

        if is_timed {
            start_event.record(stream_raw);
        }

        match backend {
            Backend::Vectorized => {
                let src_dev = src_ptrs_dev.as_ref().unwrap();
                let dst_dev = dst_ptrs_dev.as_ref().unwrap();
                let ptr_bytes = num_copies * std::mem::size_of::<usize>();

                // H2D copy of pointer arrays (included in timing)
                unsafe {
                    let err = cudaMemcpyAsync(
                        src_dev.as_ptr(),
                        src_ptrs.as_ptr() as *const c_void,
                        ptr_bytes,
                        CUDA_MEMCPY_HOST_TO_DEVICE,
                        stream_raw,
                    );
                    assert_eq!(err, 0, "H2D ptr copy failed: {err}");

                    let err = cudaMemcpyAsync(
                        dst_dev.as_ptr(),
                        dst_ptrs.as_ptr() as *const c_void,
                        ptr_bytes,
                        CUDA_MEMCPY_HOST_TO_DEVICE,
                        stream_raw,
                    );
                    assert_eq!(err, 0, "H2D ptr copy failed: {err}");
                }

                // Launch vectorized copy kernel
                let status = unsafe {
                    vectorized_copy(
                        src_dev.as_ptr() as *mut *mut c_void,
                        dst_dev.as_ptr() as *mut *mut c_void,
                        copy_size,
                        num_copies as i32,
                        stream_raw,
                    )
                };
                assert_eq!(
                    status,
                    cuda_runtime::cudaError::cudaSuccess,
                    "vectorized_copy failed: {status:?}"
                );
            }
            Backend::Batched => {
                let status = unsafe {
                    memcpy_batch(
                        src_ptrs.as_ptr() as *const *const c_void,
                        dst_ptrs.as_ptr() as *const *mut c_void,
                        copy_size,
                        num_copies,
                        MemcpyBatchMode::BatchedWithFallback,
                        stream_raw,
                    )
                };
                assert_eq!(
                    status,
                    cuda_runtime::cudaError::cudaSuccess,
                    "memcpy_batch (Batched) failed: {status:?}"
                );
            }
            Backend::MemcpyAsync => {
                let status = unsafe {
                    memcpy_batch(
                        src_ptrs.as_ptr() as *const *const c_void,
                        dst_ptrs.as_ptr() as *const *mut c_void,
                        copy_size,
                        num_copies,
                        MemcpyBatchMode::FallbackOnly,
                        stream_raw,
                    )
                };
                assert_eq!(
                    status,
                    cuda_runtime::cudaError::cudaSuccess,
                    "memcpy_batch (FallbackOnly) failed: {status:?}"
                );
            }
        }

        if is_timed {
            end_event.record(stream_raw);
            end_event.synchronize();
            let ms = end_event.elapsed_ms(&start_event);
            elapsed_samples.push(ms);
        }
    }

    // Sync before dropping memory
    unsafe { cudaStreamSynchronize(stream_raw) };

    // Compute median
    elapsed_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_ms = elapsed_samples[elapsed_samples.len() / 2] as f64;
    let bandwidth_gbps = (total_bytes as f64) / (median_ms / 1000.0) / 1e9;

    Some((median_ms, bandwidth_gbps))
}

// ---------------------------------------------------------------------------
// Parse CLI values into typed enums
// ---------------------------------------------------------------------------

fn parse_directions(raw: &[String]) -> Vec<Direction> {
    raw.iter()
        .map(|s| {
            Direction::from_str(s).unwrap_or_else(|| {
                panic!(
                    "unknown direction '{}', expected: {}",
                    s,
                    Direction::all_labels()
                )
            })
        })
        .collect()
}

fn parse_patterns(raw: &[String]) -> Vec<Pattern> {
    raw.iter()
        .map(|s| {
            Pattern::from_str(s).unwrap_or_else(|| {
                panic!(
                    "unknown pattern '{}', expected: {}",
                    s,
                    Pattern::all_labels()
                )
            })
        })
        .collect()
}

fn parse_backends(raw: &[String]) -> Vec<Backend> {
    raw.iter()
        .map(|s| {
            Backend::from_str(s).unwrap_or_else(|| {
                panic!(
                    "unknown backend '{}', expected: {}",
                    s,
                    Backend::all_labels()
                )
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();

    let directions = parse_directions(&cli.direction);
    let patterns = parse_patterns(&cli.pattern);
    let backends = parse_backends(&cli.backend);
    let tpb_options = &cli.tokens_per_block;
    let num_blocks_options = &cli.num_blocks;
    let warmup_iters = cli.warmup;
    let timed_iters = cli.iters;

    let total_tests = tpb_options.len()
        * num_blocks_options.len()
        * directions.len()
        * patterns.len()
        * backends.len();

    // Initialize CUDA context
    let count = CudaContext::device_count().expect("Failed to query CUDA devices");
    assert!(count > 0, "No CUDA devices found");
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.new_stream().expect("Failed to create CUDA stream");
    let stream_raw = stream.cu_stream() as cuda_runtime::cudaStream_t;

    // Print config to stderr
    eprintln!("KV Cache Transfer Benchmark");
    eprintln!("  Model: Llama 3.1 70B (bf16)");
    eprintln!(
        "  Layers: {NUM_LAYERS}, KV heads: {NUM_KV_HEADS}, Head dim: {HEAD_DIM}, Outer dim: {OUTER_DIM}"
    );
    eprintln!("  Warmup: {warmup_iters}, Timed: {timed_iters}");
    eprintln!(
        "  Batch API available: {}",
        kvbm_kernels::is_memcpy_batch_available()
    );
    eprintln!("  tokens_per_block: {:?}", tpb_options);
    eprintln!("  num_blocks: {:?}", num_blocks_options);
    eprintln!(
        "  directions: [{}]",
        directions
            .iter()
            .map(|d| d.label())
            .collect::<Vec<_>>()
            .join(", ")
    );
    eprintln!(
        "  patterns: [{}]",
        patterns
            .iter()
            .map(|p| p.label())
            .collect::<Vec<_>>()
            .join(", ")
    );
    eprintln!(
        "  backends: [{}]",
        backends
            .iter()
            .map(|b| b.label())
            .collect::<Vec<_>>()
            .join(", ")
    );
    eprintln!("  Total tests: {total_tests}");
    eprintln!();

    // CSV header
    println!(
        "tokens_per_block,num_blocks,pattern,direction,backend,total_bytes,inner_bytes,copy_size,num_copies,median_ms,bandwidth_gbps"
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
                        test_num += 1;
                        eprint!(
                            "  [{test_num}/{total_tests}] tpb={tpb} N={num_blocks:>3} {:<8} {:<6} {:<12} ... ",
                            pattern.label(),
                            direction.label(),
                            backend.label(),
                        );

                        match run_benchmark(
                            stream_raw,
                            pattern,
                            direction,
                            backend,
                            tpb,
                            num_blocks,
                            warmup_iters,
                            timed_iters,
                        ) {
                            Some((median_ms, bw)) => {
                                println!(
                                    "{tpb},{num_blocks},{},{},{},{total_bytes},{inner},{copy_size},{num_copies},{median_ms:.4},{bw:.2}",
                                    pattern.label(),
                                    direction.label(),
                                    backend.label(),
                                );
                                eprintln!("{bw:.2} GB/s ({median_ms:.4} ms)");
                            }
                            None => {
                                eprintln!("SKIPPED");
                            }
                        }
                    }
                }
            }
        }
    }

    eprintln!("\nDone.");
}
