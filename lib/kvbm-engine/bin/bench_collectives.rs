// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Collective broadcast bandwidth benchmark for NCCL and oneCCL.
//!
//! Multi-process: rank 0 is the coordinator that spawns child processes
//! (one per rank), distributes the bootstrap token, and collects results.
//! Children perform warmup + measured iterations and report JSONL to stdout.
//!
//! # Backends
//!
//! Both backends are supported in a single build when the matching features
//! are enabled. Select at runtime via `--backend {nccl,oneccl}`.
//!
//! - `oneccl` (Intel XPU, `feature = "oneccl"`) — uses `OneCclBootstrap`
//!   rendezvous, `sycl::queue` operations, and raw `ccl_rs_bcast_inplace` FFI.
//!   Each rank is pinned to an XPU via `ONEAPI_DEVICE_SELECTOR`.
//! - `nccl` (NVIDIA CUDA, `feature = "nccl"`) — uses `NcclBootstrap` + raw
//!   `ncclBcast` FFI on a `CudaStream`. Each rank is pinned to a GPU via
//!   `CUDA_VISIBLE_DEVICES`.
//!
//! Bandwidth is reported in **decimal GB/s** . Two
//! numbers are emitted per result: `alg_bw_gbps` (algorithm bandwidth,
//! `size / time` as seen by this rank) and `bus_bw_gbps` (bus bandwidth).
//! For broadcast, the `nccl-tests` busbw factor is 1, so they are equal;
//! the column exists so the schema extends cleanly if the bench grows
//! support for other collectives (all-reduce uses `2(N-1)/N`, all-gather /
//! reduce-scatter use `(N-1)/N`).
//!
//! # Usage (oneCCL / XPU)
//!
//! `kvbm-physical/xpu-sycl` is required because `bench` implies `testing`,
//! which forwards to `kvbm-physical` and needs a device backend to compile.
//!
//! ```bash
//! cargo run -p kvbm-engine \
//!     --no-default-features --features bench,oneccl,kvbm-physical/xpu-sycl \
//!     --bin bench_collectives -- \
//!     --backend oneccl --world-size 2 \
//!     --sizes 1024,65536,1048576,16777216,67108864 \
//!     --iterations 50 --warmup 5
//! ```
//!
//! # Usage (NCCL / CUDA)
//!
//! ```bash
//! cargo run -p kvbm-engine \
//!     --features bench,nccl \
//!     --bin bench_collectives -- \
//!     --backend nccl --world-size 2 \
//!     --sizes 1024,65536,1048576,16777216,67108864 \
//!     --iterations 50 --warmup 5
//! ```
//!
//! # Usage (combined build — runtime-select backend)
//!
//! ```bash
//! cargo run -p kvbm-engine \
//!     --no-default-features \
//!     --features bench,nccl,oneccl,kvbm-physical/xpu-sycl \
//!     --bin bench_collectives -- \
//!     --backend <nccl|oneccl> ...
//! ```
//!
//! # Grouped broadcast (`--num-regions`)
//!
//! Mirrors the production `broadcast_regions` path: each iteration splits
//! the payload into N equal regions and issues N broadcasts inside one
//! `group_start`/`group_end` batch. In production, `N = num_blocks ×
//! num_layers × outer_dim`; the flag accepts a comma-separated list so a
//! single invocation sweeps the full axis.
//!
//! Reference values (Llama 3.1 70B, bf16, per-block; `outer_dim` is a
//! storage-layout choice per `lib/kvbm-physical/src/layout/config.rs`:
//! 1 = K/V merged, 2 = K/V separated):
//! - `80`  — one block, K/V merged  (outer_dim=1), all 80 layers
//! - `160` — one block, K/V separated (outer_dim=2), all 80 layers
//! - `640` / `1280` — eight blocks, merged / separated
//!
//! ```bash
//! # 64 MiB total per iter, sweep 1 vs. 80 vs. 640 regions.
//! ... -- --backend nccl --sizes 67108864 --num-regions 1,80,640
//! ```
//!
//! # Completion-path overhead (`--no-wait-for-completion`)
//!
//! Run two sweeps (with and without the flag) and diff the per-iteration
//! latency. The difference is the overhead of blocking the host on the
//! collective's completion — relevant to the TODO tracked at
//! `lib/kvbm-engine/src/collectives/oneccl.rs:315`.
//!
//! Completion semantics differ by backend and by grouping:
//! - **oneCCL, single broadcast (`num_regions == 1`)**: wait on the
//!   event returned by `ccl_rs_bcast_inplace`. Every returned event must be
//!   destroyed regardless of whether we wait on it.
//! - **oneCCL, grouped broadcast (`num_regions > 1`)**: `event::wait()`
//!   on an in-group collective's event is not supported by oneCCL
//!   (the runtime warns if you do). Completion is awaited via
//!   `ccl_rs_stream_wait` after `group_end`, which delegates to
//!   `sycl::queue::wait()` on the queue underlying the CCL stream.
//!   `--no-wait-for-completion` skips the stream wait.
//! - **NCCL**: `ncclBcast` has no per-op event; completion is observed by
//!   synchronizing the CUDA stream after `ncclGroupEnd`.
//!   `--no-wait-for-completion` skips that `stream.synchronize()`.

// Several items (OneCclHandles, hex_encode, percentile, Instant, …) are only
// used by one backend path. Silence the warnings in reduced-feature builds
// rather than cfg-gating every one of them.
#![allow(dead_code, unused_imports, unused_variables, unreachable_code)]

use std::ffi::c_void;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use serde::Serialize;

// ─── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "bench_collectives",
    about = "Collective broadcast bandwidth benchmark (NCCL / oneCCL)"
)]
struct Cli {
    /// Backend: nccl or oneccl. Must be compiled in (feature `nccl` or
    /// `oneccl`).
    #[arg(long)]
    backend: String,

    /// Number of ranks (processes) in the collective group
    #[arg(long, default_value_t = 2)]
    world_size: usize,

    /// Total payload per iteration, in bytes, comma-separated to sweep.
    ///
    /// With `--num-regions N > 1`, each value here is split into N equal
    /// regions of `size / N` bytes each; the total stays the same. If the
    /// size is not evenly divisible by N, the actual total is rounded down
    /// to `(size / N) × N` and that's what gets reported in `size_bytes`.
    #[arg(long, value_delimiter = ',', default_values_t = vec![
        1024, 4096, 16384, 65536, 262144, 1048576,
        4194304, 16777216, 67108864,
    ])]
    sizes: Vec<usize>,

    /// Measured iterations per size (must be ≥ 1)
    #[arg(long, default_value_t = 50)]
    iterations: usize,

    /// Warmup iterations (not measured)
    #[arg(long, default_value_t = 5)]
    warmup: usize,

    /// Region counts to sweep, comma-separated. Each value splits the
    /// per-iteration payload across N equal regions, issued inside a single
    /// `group_start` / `group_end` batch.
    ///
    /// - `1` (default) matches the original bench: one full-size broadcast
    ///   per iteration, no group.
    /// - `N > 1` mirrors the production `broadcast_regions` path. In that
    ///   path, `N = num_blocks × num_layers × outer_dim`.
    ///
    /// Handy reference values (Llama 3.1 70B, bf16, per-block; `outer_dim`
    /// is a storage-layout choice: 1 = K/V merged, 2 = K/V separated):
    /// - `80`   — one block,  K/V merged   (outer_dim=1), all 80 layers
    /// - `160`  — one block,  K/V separated (outer_dim=2), all 80 layers
    /// - `640`  — eight blocks, K/V merged
    /// - `1280` — eight blocks, K/V separated
    ///
    /// When N > 1, `size_bytes` is rounded down to a multiple of N so every
    /// region is the same size.
    #[arg(long, value_delimiter = ',', default_values_t = vec![1])]
    num_regions: Vec<usize>,

    /// Measure submit-only latency: skip the per-iteration wait on the
    /// collective's completion event. The collective is still queued (and
    /// must complete before the next iteration's submit can overlap
    /// with prior work), but the host no longer blocks on it.
    ///
    /// Diffing `--no-wait-for-completion` against the default gives the
    /// completion-path overhead. Relevant for the
    /// `OneCclEventRegistrar` gap tracked at
    /// `lib/kvbm-engine/src/collectives/oneccl.rs:315`.
    #[arg(long, default_value_t = false)]
    no_wait_for_completion: bool,

    /// Per-rank device-selector pattern. `{rank}` is substituted with the
    /// rank index.
    ///
    /// Backend-specific env vars (auto-picked from `--backend` if not set):
    /// - oneccl: default `level_zero:{rank}` → `ONEAPI_DEVICE_SELECTOR`
    /// - nccl:   default `{rank}`            → `CUDA_VISIBLE_DEVICES`
    ///
    /// Pass the same literal for every rank (no `{rank}`) if ranks should
    /// share one device.
    #[arg(long)]
    device_selector: Option<String>,

    /// Output directory (default: bench_collectives_<timestamp>)
    #[arg(long)]
    output: Option<String>,
}

// ─── Results ─────────────────────────────────────────────────────────────────

/// Nearest-rank percentile: index = ⌈p × n⌉, clamped to `[0, n-1]`.
fn percentile(sorted: &[Duration], p: f64) -> Duration {
    debug_assert!(!sorted.is_empty());
    debug_assert!((0.0..=1.0).contains(&p));
    let n = sorted.len();
    let idx = ((p * n as f64).ceil() as usize)
        .saturating_sub(1)
        .min(n - 1);
    sorted[idx]
}

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
struct LatencyStats {
    min_us: f64,
    max_us: f64,
    mean_us: f64,
    p50_us: f64,
    p95_us: f64,
    p99_us: f64,
}

impl LatencyStats {
    fn from_durations(mut durations: Vec<Duration>) -> Result<Self> {
        ensure!(!durations.is_empty(), "no latency samples to summarize");
        durations.sort();
        let n = durations.len();
        let sum: Duration = durations.iter().sum();
        Ok(Self {
            min_us: durations[0].as_secs_f64() * 1e6,
            max_us: durations[n - 1].as_secs_f64() * 1e6,
            mean_us: sum.as_secs_f64() * 1e6 / n as f64,
            p50_us: percentile(&durations, 0.50).as_secs_f64() * 1e6,
            p95_us: percentile(&durations, 0.95).as_secs_f64() * 1e6,
            p99_us: percentile(&durations, 0.99).as_secs_f64() * 1e6,
        })
    }
}

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
struct BenchResult {
    backend: String,
    world_size: usize,
    rank: usize,
    /// Total payload per iteration, in bytes. If `num_regions > 1`, this is
    /// the sum of all regions in the group (`num_regions × region_bytes`).
    size_bytes: usize,
    /// Number of regions the payload is split across per iteration (issued
    /// inside one `group_start`/`group_end` batch).
    num_regions: usize,
    /// Whether the measurement includes the per-iteration event wait.
    /// `false` = submit-only latency.
    wait_for_completion: bool,
    iterations: usize,
    latency_us: LatencyStats,
    /// Algorithm bandwidth: `size / mean_time` (decimal GB/s, 1 GB = 1e9 B).
    alg_bw_gbps: f64,
    /// Bus bandwidth (decimal GB/s). For broadcast the `nccl-tests` busbw
    /// factor is 1, so `bus_bw_gbps == alg_bw_gbps`. Kept as a separate
    /// column so the output schema is compatible with future reductions
    /// (e.g. all-reduce would use `alg × 2(N-1)/N`).
    bus_bw_gbps: f64,
}

/// Compute algbw + busbw (decimal GB/s) for a broadcast from latency summary.
///
/// For broadcast the bus-bandwidth factor is 1. The `_world_size` parameter
/// is accepted for API symmetry with other collectives we might benchmark later
/// (all-gather / reduce-scatter use `(N-1)/N`, all-reduce uses `2(N-1)/N`).
fn compute_bandwidth(size_bytes: usize, mean_us: f64, _world_size: usize) -> (f64, f64) {
    // bytes / ns = decimal GB/s (1 GB = 1e9 B). mean_us × 1e3 = mean_ns.
    let alg = size_bytes as f64 / (mean_us * 1e3);
    let bus = alg; // broadcast factor = 1
    (alg, bus)
}

// ─── Multi-process infrastructure ────────────────────────────────────────────

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn hex_decode(s: &str) -> Result<Vec<u8>> {
    ensure!(s.len() % 2 == 0, "hex string has odd length: {}", s.len());
    (0..s.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&s[i..i + 2], 16)
                .with_context(|| format!("invalid hex at offset {i}: {:?}", &s[i..i + 2]))
        })
        .collect()
}

// =============================================================================
// oneCCL worker
// =============================================================================

/// RAII guard for oneCCL stream + comm handles so they get destroyed even if
/// the benchmark loop errors out partway through.
#[cfg(feature = "oneccl")]
struct OneCclHandles {
    stream: *mut oneapi_rs::ccl::sys::ccl_rs_stream_t,
    comm: oneapi_rs::ccl::sys::ccl_rs_comm_t,
}

#[cfg(feature = "oneccl")]
impl Drop for OneCclHandles {
    fn drop(&mut self) {
        unsafe {
            if !self.stream.is_null() {
                oneapi_rs::ccl::sys::ccl_rs_stream_destroy(self.stream);
            }
            if !self.comm.is_null() {
                oneapi_rs::ccl::sys::ccl_rs_comm_destroy(self.comm);
            }
        }
    }
}

#[cfg(feature = "oneccl")]
fn run_oneccl_worker(
    rank: usize,
    world_size: usize,
    bootstrap_bytes: &[u8],
    sizes: &[usize],
    warmup: usize,
    iterations: usize,
    num_regions_list: &[usize],
    wait_for_completion: bool,
) -> Result<Vec<BenchResult>> {
    use kvbm_engine::collectives::OneCclBootstrap;
    use oneapi_rs::ccl::sys;
    use oneapi_rs::sycl::safe::{DevicePtrMut, SyclQueue};
    use std::ptr;

    fn check_ccl(r: sys::ccl_rs_result_t) -> Result<()> {
        if r == sys::ccl_rs_result_t::CCL_RS_RESULT_SUCCESS {
            Ok(())
        } else {
            bail!("oneCCL operation failed: {:?}", r)
        }
    }

    let queue = SyclQueue::new_for_device_ordinal(0)?;
    let bootstrap = OneCclBootstrap::deserialize(bootstrap_bytes)?;

    // stream and comm live in `handles` for the whole sweep — both the
    // size axis and the num_regions axis reuse one communicator (the
    // bootstrap is a one-shot handshake).
    let handles = OneCclHandles {
        stream: unsafe { OneCclBootstrap::create_stream(queue.raw_queue_ptr()) }?,
        comm: bootstrap.init_communicator(rank)?,
    };

    let mut results: Vec<BenchResult> = Vec::with_capacity(sizes.len() * num_regions_list.len());

    for &num_regions in num_regions_list {
        ensure!(num_regions >= 1, "num_regions must be ≥ 1");

        for &total_size in sizes {
            // Round the total down to a multiple of num_regions so every region
            // has the same size. This matches production broadcast_regions
            // where each (ptr, size) pair is independent.
            let region_size = total_size / num_regions;
            ensure!(
                region_size > 0,
                "size {} is too small for num_regions {} (region would be 0 bytes)",
                total_size,
                num_regions
            );
            let effective_total = region_size * num_regions;

            // Allocate one device buffer per region. Rank 0 fills each.
            let mut buffers: Vec<_> = (0..num_regions)
                .map(|_| queue.alloc_zeros::<u8>(region_size))
                .collect::<Result<_, _>>()?;
            if rank == 0 {
                let host_data = vec![0xABu8; region_size];
                for buf in &mut buffers {
                    queue.memcpy_sync(host_data.as_slice(), buf)?;
                }
            }
            queue.synchronize()?;

            let buffer_ptrs: Vec<*mut c_void> = buffers
                .iter_mut()
                .map(|b| b.device_ptr_mut() as *mut c_void)
                .collect();

            // Run one iteration: group_start → N broadcasts → group_end →
            // optional completion wait.
            //
            // `ccl_rs_bcast_inplace` returns an event per call. The API contract
            // requires every returned event to be destroyed (shim heap-
            // allocates unconditionally). We destroy intermediates
            // immediately and only keep the last one alive for the
            // single-region path.
            //
            // For `num_regions > 1` (inside group_start/end), oneCCL warns
            // that `event::wait()` on an in-group collective's event is not
            // supported — the group's collectives are reordered/fused and
            // the per-op event isn't a meaningful completion signal. The
            // spec-correct host-side wait for a group is a queue sync. So:
            //   - num_regions == 1 → wait on the per-op event (no group)
            //   - num_regions  > 1 → destroy per-op event, queue.wait()
            //                         for completion instead.
            let one_iter = |latencies: &mut Option<&mut Vec<Duration>>| -> Result<()> {
                let t0 = Instant::now();
                let grouped = num_regions > 1;
                if grouped {
                    check_ccl(unsafe { sys::ccl_rs_group_start() })?;
                }
                let mut last_event: *mut sys::ccl_rs_event_t = ptr::null_mut();
                for &p in &buffer_ptrs {
                    let mut event: *mut sys::ccl_rs_event_t = ptr::null_mut();
                    check_ccl(unsafe {
                        sys::ccl_rs_bcast_inplace(
                            p,
                            region_size,
                            sys::ccl_rs_data_type_t::CCL_RS_DATA_TYPE_UINT8,
                            0,
                            handles.comm,
                            handles.stream as *mut c_void,
                            &mut event,
                        )
                    })?;
                    if !last_event.is_null() {
                        unsafe { sys::ccl_rs_event_destroy(last_event) };
                    }
                    last_event = event;
                }
                if grouped {
                    check_ccl(unsafe { sys::ccl_rs_group_end() })?;
                }
                if wait_for_completion {
                    if grouped {
                        // Spec-correct group completion: wait on the SYCL
                        // queue underlying the CCL stream via the FFI
                        // wrapper, not on any per-op event. Mirrors the
                        // production `OneCclCollectives::broadcast_regions`
                        // path so the bench measures the same code.
                        check_ccl(unsafe {
                            sys::ccl_rs_stream_wait(handles.stream)
                        })?;
                    } else if !last_event.is_null() {
                        check_ccl(unsafe { sys::ccl_rs_event_wait(last_event) })?;
                    }
                }
                // Always destroy the retained event regardless of whether
                // we waited — the shim allocates it unconditionally.
                if !last_event.is_null() {
                    unsafe { sys::ccl_rs_event_destroy(last_event) };
                }
                if let Some(v) = latencies {
                    v.push(t0.elapsed());
                }
                Ok(())
            };

            // Warmup
            for _ in 0..warmup {
                one_iter(&mut None)?;
            }

            // Measured iterations
            let mut latencies = Vec::with_capacity(iterations);
            for _ in 0..iterations {
                one_iter(&mut Some(&mut latencies))?;
            }

            let stats = LatencyStats::from_durations(latencies)?;
            let (alg_bw_gbps, bus_bw_gbps) =
                compute_bandwidth(effective_total, stats.mean_us, world_size);

            results.push(BenchResult {
                backend: "oneccl".to_string(),
                world_size,
                rank,
                size_bytes: effective_total,
                num_regions,
                wait_for_completion,
                iterations,
                latency_us: stats,
                alg_bw_gbps,
                bus_bw_gbps,
            });
        }
    }

    drop(handles);
    Ok(results)
}

// =============================================================================
// NCCL worker
// =============================================================================

/// RAII guard for the NCCL communicator. The CUDA stream is owned by
/// `CudaStream`/`Arc<CudaStream>` and drops cleanly without manual help.
#[cfg(feature = "nccl")]
struct NcclHandle {
    comm: cudarc::nccl::sys::ncclComm_t,
}

#[cfg(feature = "nccl")]
impl Drop for NcclHandle {
    fn drop(&mut self) {
        if !self.comm.is_null() {
            unsafe {
                let _ = cudarc::nccl::sys::ncclCommDestroy(self.comm);
            }
        }
    }
}

#[cfg(feature = "nccl")]
fn run_nccl_worker(
    rank: usize,
    world_size: usize,
    bootstrap_bytes: &[u8],
    sizes: &[usize],
    warmup: usize,
    iterations: usize,
    num_regions_list: &[usize],
    wait_for_completion: bool,
) -> Result<Vec<BenchResult>> {
    use cudarc::driver::{CudaContext, DevicePtr};
    use cudarc::nccl::sys::{ncclBcast, ncclDataType_t, ncclGroupEnd, ncclGroupStart};
    use kvbm_engine::collectives::NcclBootstrap;

    fn check_nccl(r: cudarc::nccl::sys::ncclResult_t) -> Result<()> {
        if r == cudarc::nccl::sys::ncclResult_t::ncclSuccess {
            Ok(())
        } else {
            bail!("NCCL operation failed: {:?}", r)
        }
    }

    // Device 0 because CUDA_VISIBLE_DEVICES has already restricted this
    // process to exactly one GPU (see `device_selector_env_for` in the
    // coordinator). From inside the child, the assigned GPU always appears
    // as ordinal 0.
    let ctx = CudaContext::new(0).context("CudaContext::new(0) failed")?;
    let stream = ctx.new_stream().context("CudaContext::new_stream failed")?;

    let bootstrap = NcclBootstrap::deserialize(bootstrap_bytes)?;
    let handle = NcclHandle {
        comm: bootstrap.init_communicator(rank, stream.cu_stream())?,
    };

    let mut results: Vec<BenchResult> = Vec::with_capacity(sizes.len() * num_regions_list.len());

    for &num_regions in num_regions_list {
        ensure!(num_regions >= 1, "num_regions must be ≥ 1");

        for &total_size in sizes {
            let region_size = total_size / num_regions;
            ensure!(
                region_size > 0,
                "size {} is too small for num_regions {} (region would be 0 bytes)",
                total_size,
                num_regions
            );
            let effective_total = region_size * num_regions;

            let mut buffers: Vec<cudarc::driver::CudaSlice<u8>> = (0..num_regions)
                .map(|_| stream.alloc_zeros::<u8>(region_size))
                .collect::<Result<_, _>>()?;
            if rank == 0 {
                let host_data = vec![0xABu8; region_size];
                for buf in &mut buffers {
                    stream.memcpy_htod(host_data.as_slice(), buf)?;
                }
            }
            stream.synchronize()?;

            // One iteration: group_start → N bcast → group_end → (optional)
            // stream sync. NCCL has no per-op event type — completion is
            // observed by stream-synchronizing after the group closes. This
            // is the same pattern used by `NcclCollectives::broadcast_regions`
            // in src/collectives/nccl.rs, which returns a single CUDA event
            // recorded on the stream after the group (not N events).
            let one_iter = |latencies: &mut Option<&mut Vec<Duration>>| -> Result<()> {
                let t0 = Instant::now();
                if num_regions > 1 {
                    check_nccl(unsafe { ncclGroupStart() })?;
                }
                for buf in &buffers {
                    let (dev_ptr, _guard) = buf.device_ptr(&stream);
                    check_nccl(unsafe {
                        ncclBcast(
                            dev_ptr as *mut c_void,
                            region_size,
                            ncclDataType_t::ncclChar,
                            0,
                            handle.comm,
                            stream.cu_stream().cast(),
                        )
                    })?;
                }
                if num_regions > 1 {
                    check_nccl(unsafe { ncclGroupEnd() })?;
                }
                if wait_for_completion {
                    stream.synchronize()?;
                }
                if let Some(v) = latencies {
                    v.push(t0.elapsed());
                }
                Ok(())
            };

            // Warmup — always drain the stream after, independent of the
            // measurement flag, to prevent unbounded launch-queue growth.
            for _ in 0..warmup {
                one_iter(&mut None)?;
                if !wait_for_completion {
                    stream.synchronize()?;
                }
            }

            // Measured iterations
            let mut latencies = Vec::with_capacity(iterations);
            for _ in 0..iterations {
                one_iter(&mut Some(&mut latencies))?;
            }
            // Drain the stream after the measurement window so the next
            // (size, num_regions) combination's allocations don't queue
            // behind stale NCCL work.
            if !wait_for_completion {
                stream.synchronize()?;
            }

            let stats = LatencyStats::from_durations(latencies)?;
            let (alg_bw_gbps, bus_bw_gbps) =
                compute_bandwidth(effective_total, stats.mean_us, world_size);

            results.push(BenchResult {
                backend: "nccl".to_string(),
                world_size,
                rank,
                size_bytes: effective_total,
                num_regions,
                wait_for_completion,
                iterations,
                latency_us: stats,
                alg_bw_gbps,
                bus_bw_gbps,
            });
        }
    }

    drop(handle);
    Ok(results)
}

// ─── Child entry point ───────────────────────────────────────────────────────

/// Print this rank's selected device (backend, BDF, name) to stderr.
/// Each child only sees one device because the coordinator has already
/// set `CUDA_VISIBLE_DEVICES` / `ONEAPI_DEVICE_SELECTOR` before fork;
/// we always read the device at restricted-view ordinal 0. Best-effort:
/// any lookup failure prints "unknown" rather than aborting.
fn eprint_rank_device(rank: usize, backend: &str) {
    let (bdf, name) = match backend {
        #[cfg(feature = "oneccl")]
        "oneccl" => {
            use oneapi_rs::sycl::safe::SyclDevice;
            match SyclDevice::by_ordinal(0).and_then(|d| d.info()) {
                Ok(info) => (
                    info.pci_address.unwrap_or_else(|| "unknown".to_string()),
                    info.name,
                ),
                Err(_) => ("unknown".to_string(), "unknown".to_string()),
            }
        }
        #[cfg(feature = "nccl")]
        "nccl" => {
            use cudarc::driver::sys as cu;
            use std::ffi::CStr;
            let mut bdf = String::from("unknown");
            let mut name = String::from("unknown");
            unsafe {
                let mut dev = std::mem::MaybeUninit::uninit();
                if cu::cuInit(0).result().is_ok()
                    && cu::cuDeviceGet(dev.as_mut_ptr(), 0).result().is_ok()
                {
                    let dev = dev.assume_init();
                    let mut domain = 0i32;
                    let mut bus = 0i32;
                    let mut slot_num = 0i32;
                    let dom_ok = cu::cuDeviceGetAttribute(
                        &mut domain,
                        cu::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
                        dev,
                    ).result().is_ok();
                    let bus_ok = cu::cuDeviceGetAttribute(
                        &mut bus,
                        cu::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
                        dev,
                    ).result().is_ok();
                    let slot_ok = cu::cuDeviceGetAttribute(
                        &mut slot_num,
                        cu::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
                        dev,
                    ).result().is_ok();
                    if dom_ok && bus_ok && slot_ok {
                        bdf = format!("{:04x}:{:02x}:{:02x}.0", domain, bus, slot_num);
                    }
                    let mut buf = [0i8; 256];
                    if cu::cuDeviceGetName(buf.as_mut_ptr(), buf.len() as i32, dev).result().is_ok() {
                        name = CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned();
                    }
                }
            }
            (bdf, name)
        }
        _ => ("unknown".to_string(), "unknown".to_string()),
    };
    eprintln!(
        "  rank[{}] backend={}  bdf={}  name=\"{}\"",
        rank, backend, bdf, name,
    );
}

fn run_child() -> Result<Option<()>> {
    let rank_str = match std::env::var("BENCH_CCL_RANK") {
        Ok(r) => r,
        Err(_) => return Ok(None), // Not a child
    };

    let rank: usize = rank_str.parse().context("parsing BENCH_CCL_RANK")?;
    let world_size: usize = std::env::var("BENCH_CCL_WORLD_SIZE")
        .context("BENCH_CCL_WORLD_SIZE not set")?
        .parse()
        .context("parsing BENCH_CCL_WORLD_SIZE")?;
    let bootstrap_hex =
        std::env::var("BENCH_CCL_BOOTSTRAP").context("BENCH_CCL_BOOTSTRAP not set")?;
    let sizes_str = std::env::var("BENCH_CCL_SIZES").context("BENCH_CCL_SIZES not set")?;
    let warmup: usize = std::env::var("BENCH_CCL_WARMUP")
        .context("BENCH_CCL_WARMUP not set")?
        .parse()
        .context("parsing BENCH_CCL_WARMUP")?;
    let iterations: usize = std::env::var("BENCH_CCL_ITERATIONS")
        .context("BENCH_CCL_ITERATIONS not set")?
        .parse()
        .context("parsing BENCH_CCL_ITERATIONS")?;
    let backend = std::env::var("BENCH_CCL_BACKEND").context("BENCH_CCL_BACKEND not set")?;
    let num_regions_str = std::env::var("BENCH_CCL_NUM_REGIONS")
        .context("BENCH_CCL_NUM_REGIONS not set")?;
    let num_regions_list: Vec<usize> = num_regions_str
        .split(',')
        .map(|s| {
            s.parse()
                .with_context(|| format!("parsing num_regions {s:?} in BENCH_CCL_NUM_REGIONS"))
        })
        .collect::<Result<_>>()?;
    let wait_for_completion: bool = std::env::var("BENCH_CCL_WAIT")
        .context("BENCH_CCL_WAIT not set")?
        .parse()
        .context("parsing BENCH_CCL_WAIT")?;

    ensure!(iterations >= 1, "iterations must be ≥ 1");
    ensure!(
        num_regions_list.iter().all(|&n| n >= 1),
        "every num_regions value must be ≥ 1"
    );

    let bootstrap_bytes = hex_decode(&bootstrap_hex).context("decoding BENCH_CCL_BOOTSTRAP")?;
    let sizes: Vec<usize> = sizes_str
        .split(',')
        .map(|s| {
            s.parse()
                .with_context(|| format!("parsing size {s:?} in BENCH_CCL_SIZES"))
        })
        .collect::<Result<_>>()?;

    // Print this rank's device assignment (backend, BDF, name) to stderr
    // before any collective work. Each child only sees one device because
    // CUDA_VISIBLE_DEVICES / ONEAPI_DEVICE_SELECTOR has been pinned by
    // the coordinator; we report it here for log-level traceability.
    eprint_rank_device(rank, &backend);

    // The worker reuses its communicator across the full (size × num_regions)
    // sweep — the collective bootstrap is a one-shot handshake, so we must
    // not re-invoke `init_communicator`.
    let results: Vec<BenchResult> = match backend.as_str() {
        #[cfg(feature = "oneccl")]
        "oneccl" => run_oneccl_worker(
            rank,
            world_size,
            &bootstrap_bytes,
            &sizes,
            warmup,
            iterations,
            &num_regions_list,
            wait_for_completion,
        )?,
        #[cfg(feature = "nccl")]
        "nccl" => run_nccl_worker(
            rank,
            world_size,
            &bootstrap_bytes,
            &sizes,
            warmup,
            iterations,
            &num_regions_list,
            wait_for_completion,
        )?,
        _ => bail!(
            "Unsupported backend '{}' (compile with --features nccl or --features oneccl)",
            backend
        ),
    };

    for r in &results {
        println!("{}", serde_json::to_string(r).unwrap());
    }

    Ok(Some(()))
}

// ─── Coordinator ─────────────────────────────────────────────────────────────

/// Default `(env_var_name, pattern)` for a backend's per-rank device pinning.
fn device_selector_env_for(backend: &str) -> Result<(&'static str, &'static str)> {
    match backend {
        "oneccl" => Ok(("ONEAPI_DEVICE_SELECTOR", "level_zero:{rank}")),
        "nccl" => Ok(("CUDA_VISIBLE_DEVICES", "{rank}")),
        _ => bail!(
            "Unsupported backend '{}' (compile with --features nccl or --features oneccl)",
            backend
        ),
    }
}

/// Generate the bootstrap hex for the selected backend.
fn generate_bootstrap_hex(backend: &str, world_size: usize) -> Result<String> {
    match backend {
        #[cfg(feature = "oneccl")]
        "oneccl" => {
            use kvbm_engine::collectives::OneCclBootstrap;
            let bootstrap = OneCclBootstrap::generate(world_size)?;
            Ok(hex_encode(&bootstrap.serialize()))
        }
        #[cfg(feature = "nccl")]
        "nccl" => {
            use kvbm_engine::collectives::NcclBootstrap;
            let bootstrap = NcclBootstrap::generate(world_size)?;
            Ok(hex_encode(&bootstrap.serialize()))
        }
        _ => bail!(
            "Unsupported backend '{}' (compile with --features nccl or --features oneccl)",
            backend
        ),
    }
}

/// Spawn a thread that drains `reader` to a `Vec<u8>`. Needed so children
/// don't block on pipe-buffer fill while the coordinator is waiting on another
/// rank.
fn spawn_drain<R: std::io::Read + Send + 'static>(
    mut reader: R,
) -> std::thread::JoinHandle<std::io::Result<Vec<u8>>> {
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).map(|_| buf)
    })
}

/// Like `spawn_drain`, but also forwards each line to the coordinator's
/// stderr in real time (prefixed with `[rank N]`) while still buffering
/// the full content for the failure-dump path. Used for child stderr so
/// the user sees device-info lines, oneCCL diagnostics, and progress
/// while the bench runs — instead of only on rank failure at exit.
fn spawn_drain_tee_stderr<R: std::io::Read + Send + 'static>(
    reader: R,
    rank: usize,
) -> std::thread::JoinHandle<std::io::Result<Vec<u8>>> {
    use std::io::{BufRead, BufReader, Write};
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        let mut br = BufReader::new(reader);
        let stderr = std::io::stderr();
        loop {
            let mut line = Vec::new();
            let n = br.read_until(b'\n', &mut line)?;
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&line);
            // Strip trailing newline for the prefix path; re-emit it after.
            let mut handle = stderr.lock();
            let _ = handle.write_all(format!("[rank {rank}] ").as_bytes());
            let _ = handle.write_all(&line);
            // Ensure terminal lines without a trailing \n still flush.
            if !line.ends_with(b"\n") {
                let _ = handle.write_all(b"\n");
            }
        }
        Ok(buf)
    })
}

fn run_coordinator(cli: Cli) -> Result<()> {
    let world_size = cli.world_size;
    ensure!(world_size >= 2, "world_size must be >= 2");
    ensure!(cli.iterations >= 1, "iterations must be >= 1");

    let backend = cli.backend.clone();
    let (selector_env_var, default_pattern) = device_selector_env_for(&backend)?;
    let selector_pattern = cli
        .device_selector
        .clone()
        .unwrap_or_else(|| default_pattern.to_string());

    ensure!(
        !cli.num_regions.is_empty() && cli.num_regions.iter().all(|&n| n >= 1),
        "every --num-regions value must be >= 1"
    );
    let wait_for_completion = !cli.no_wait_for_completion;

    eprintln!(
        "bench_collectives: backend={} world_size={} sizes={:?} warmup={} iterations={} \
         num_regions={:?} wait_for_completion={} selector={}={}",
        backend, world_size, cli.sizes, cli.warmup, cli.iterations,
        cli.num_regions, wait_for_completion, selector_env_var, selector_pattern
    );
    eprintln!("Selected devices ({}, one per rank):", world_size);

    let bootstrap_hex = generate_bootstrap_hex(&backend, world_size)?;

    let sizes_str: String = cli
        .sizes
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(",");

    let binary = std::env::current_exe()?;
    let mut children: Vec<(
        std::process::Child,
        std::thread::JoinHandle<std::io::Result<Vec<u8>>>,
        std::thread::JoinHandle<std::io::Result<Vec<u8>>>,
    )> = Vec::with_capacity(world_size);

    for rank in 0..world_size {
        let selector = selector_pattern.replace("{rank}", &rank.to_string());
        let mut cmd = std::process::Command::new(&binary);
        cmd.env("BENCH_CCL_RANK", rank.to_string())
            .env("BENCH_CCL_WORLD_SIZE", world_size.to_string())
            .env("BENCH_CCL_BOOTSTRAP", &bootstrap_hex)
            .env("BENCH_CCL_SIZES", &sizes_str)
            .env("BENCH_CCL_WARMUP", cli.warmup.to_string())
            .env("BENCH_CCL_ITERATIONS", cli.iterations.to_string())
            .env("BENCH_CCL_BACKEND", &backend)
            .env(
                "BENCH_CCL_NUM_REGIONS",
                cli.num_regions
                    .iter()
                    .map(|n| n.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            )
            .env("BENCH_CCL_WAIT", wait_for_completion.to_string())
            .env(selector_env_var, selector)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        // Suppress oneCCL's stdout "CCL_WARN" noise unless the user has
        // asked for it. oneCCL prints `<timestamp>:(<pid>) |CCL_WARN| …`
        // lines on stdout, which corrupts the JSONL output. `error` only
        // logs genuine errors; set `CCL_LOG_LEVEL=info` in the caller's
        // env to override.
        if backend == "oneccl" && std::env::var_os("CCL_LOG_LEVEL").is_none() {
            cmd.env("CCL_LOG_LEVEL", "error");
        }

        let mut child = cmd.spawn()?;

        // Start draining each child's stdout/stderr immediately. If we waited
        // to drain until after the earlier rank finished, any rank writing
        // more than one pipe-buffer (~64 KB) of output would block on write()
        // and hang the whole benchmark.
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("rank {rank}: no stdout pipe"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| anyhow::anyhow!("rank {rank}: no stderr pipe"))?;
        let stdout_thread = spawn_drain(stdout);
        let stderr_thread = spawn_drain_tee_stderr(stderr, rank);
        children.push((child, stdout_thread, stderr_thread));
    }

    // Wait for all children and collect their output.
    let mut all_results: Vec<BenchResult> = Vec::new();
    let mut total_dropped = 0usize;

    for (rank, (mut child, stdout_t, stderr_t)) in children.into_iter().enumerate() {
        let status = child.wait()?;
        let stdout_bytes = stdout_t
            .join()
            .map_err(|_| anyhow::anyhow!("rank {rank}: stdout reader thread panicked"))??;
        let stderr_bytes = stderr_t
            .join()
            .map_err(|_| anyhow::anyhow!("rank {rank}: stderr reader thread panicked"))??;

        if !status.success() {
            let stderr_str = String::from_utf8_lossy(&stderr_bytes);
            bail!("Rank {} failed (exit={}):\n{}", rank, status, stderr_str);
        }

        let stdout_str = String::from_utf8_lossy(&stdout_bytes);
        let mut dropped = 0usize;
        for line in stdout_str.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            // Only lines that look like JSON objects are candidates; skip
            // noise the underlying runtime might emit on stdout (e.g.
            // oneCCL's `<ts>:(<pid>) |CCL_WARN| …` lines).
            if !trimmed.starts_with('{') {
                continue;
            }
            match serde_json::from_str::<BenchResult>(line) {
                Ok(r) => all_results.push(r),
                Err(e) => {
                    dropped += 1;
                    eprintln!(
                        "warning: rank {rank}: unparseable JSONL line ({e}): {line:?}"
                    );
                }
            }
        }
        if dropped > 0 {
            eprintln!("warning: rank {rank} produced {dropped} unparseable JSONL line(s)");
        }
        total_dropped += dropped;
    }

    // Print summary to stderr
    eprintln!(
        "\n{:<10} {:>4} {:>12} {:>6} {:>4} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "backend", "rank", "size", "nreg", "wait",
        "mean_us", "p50_us", "p99_us", "alg(GB/s)", "bus(GB/s)"
    );
    eprintln!("{}", "-".repeat(96));
    for r in &all_results {
        eprintln!(
            "{:<10} {:>4} {:>12} {:>6} {:>4} {:>10.1} {:>10.1} {:>10.1} {:>10.2} {:>10.2}",
            r.backend,
            r.rank,
            r.size_bytes,
            r.num_regions,
            if r.wait_for_completion { "y" } else { "n" },
            r.latency_us.mean_us,
            r.latency_us.p50_us,
            r.latency_us.p99_us,
            r.alg_bw_gbps,
            r.bus_bw_gbps,
        );
    }

    if total_dropped > 0 {
        eprintln!("\nwarning: {total_dropped} JSONL line(s) dropped across all ranks");
    }

    // Write JSONL output
    let dir_name = cli.output.unwrap_or_else(|| {
        format!(
            "bench_collectives_{}",
            chrono::Local::now().format("%Y%m%d_%H%M%S")
        )
    });
    std::fs::create_dir_all(&dir_name)?;

    let jsonl_path = format!("{}/{}.jsonl", dir_name, dir_name);
    let json_output: String = all_results
        .iter()
        .map(|r| serde_json::to_string(r).unwrap())
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(&jsonl_path, &json_output)?;

    eprintln!("\nResults written to: {}", jsonl_path);
    Ok(())
}

// ─── main ────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    // If we're a child process, run the worker and exit
    if let Some(()) = run_child()? {
        return Ok(());
    }

    // Otherwise we're the coordinator
    let cli = Cli::parse();
    run_coordinator(cli)
}
