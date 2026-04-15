// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! KV-cache transfer benchmark via the `kvbm-physical` TransferManager API.
//!
//! Supports CUDA, XPU Level-Zero, and XPU SYCL backends via `--backend cuda|ze|sycl`.
//!
//! Test matrix:
//!   - tokens_per_block (tpb): 16, 32, 64
//!   - num_blocks (N):         1, 2, 4, 8, 16, 32, 64, 128, 256
//!   - direction:              d2d, h2d, d2h
//!   - pattern:                fc_to_fc, lw_to_fc, fc_to_lw
//!   - host_mem:               pinned, system
//!
//! Model dimensions: Llama 3.1 70B (bf16) -- 80 layers, 8 KV heads, 128 head dim.
//!
//! The TransferManager selects the engine internally (FC-to-FC = BCS/copy memcpy,
//! FC-to/from-LW = CCS/compute vectorized kernel), so the CSV `backend` column is
//! always `transfer_mgr`.
//!
//! Usage:
//! ```bash
//! # XPU (Level-Zero)
//! cargo build --example bench_transfer -p kvbm-physical \
//!   --no-default-features --features xpu-ze --release
//! ./target/release/examples/bench_transfer --backend ze --device 0
//!
//! # XPU (SYCL)
//! KVBM_ENABLE_XPU_KERNELS=1 cargo build --example bench_transfer -p kvbm-physical \
//!   --no-default-features --features xpu-sycl --release
//!   ./target/release/examples/bench_transfer --backend sycl --device 0
//!
//! # CUDA
//! cargo build --example bench_transfer -p kvbm-physical \
//!   --no-default-features --features cuda --release
//! ./target/release/examples/bench_transfer --backend cuda --device 0
//!
//! # Quick smoke test
//! ./target/release/examples/bench_transfer --backend ze \
//!   --num-blocks 1,4 --tokens-per-block 16 --warmup 3 --iters 10
//! ```

use anyhow::Result;
use clap::Parser;
use std::sync::Arc;
use std::time::Instant;

use dynamo_memory::DeviceAllocator;
use kvbm_physical::device::{DeviceBackend, DeviceContext};
use kvbm_physical::layout::{BlockDimension, LayoutConfig, PhysicalLayoutBuilder};
use kvbm_physical::transfer::{NixlAgent, StorageKind};
use kvbm_physical::{BlockId, TransferManager, TransferOptions};

// ---------------------------------------------------------------------------
// Llama 3.1 70B, bf16 KV cache dimensions
// ---------------------------------------------------------------------------
const NUM_LAYERS: usize = 80;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const ELEM_SIZE: usize = 2; // bf16
const OUTER_DIM: usize = 2; // K and V

// ---------------------------------------------------------------------------
// Device info (Level Zero)
// ---------------------------------------------------------------------------

/// Print device info for all Level Zero devices, matching kvbench_xpu_ze.rs.
#[cfg(feature = "xpu-ze")]
fn print_ze_device_info(device_ordinal: u32) {
    use level_zero as ze;

    let drivers = match ze::drivers() {
        Ok(d) if !d.is_empty() => d,
        _ => {
            eprintln!("(No Level Zero drivers found — skipping device info)");
            return;
        }
    };

    eprintln!("All Level Zero devices:");
    for (drv_idx, drv) in drivers.iter().enumerate() {
        let drv_ctx = match ze::Context::create(drv) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let drv_devs = drv.devices().unwrap_or_default();
        for (dev_idx, dev) in drv_devs.iter().enumerate() {
            if let Ok(p) = drv_ctx.device_properties(dev) {
                let type_str = match p.device_type {
                    1 => "GPU",
                    2 => "CPU",
                    _ => continue,
                };
                let total_eus =
                    p.num_slices * p.num_subslices_per_slice * p.num_eus_per_subslice;
                eprintln!("  Driver {drv_idx} / Device {dev_idx}: {}", p.name);
                eprintln!(
                    "    Type: {type_str}  Vendor: 0x{:04X}  DeviceID: 0x{:04X}",
                    p.vendor_id, p.device_id,
                );
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

    // Print driver-0 device list for quick reference.
    let driver = &drivers[0];
    let devices = driver.devices().unwrap_or_default();
    eprintln!("Benchmark driver: 0  ({} device(s))", devices.len());
    if let Ok(ctx) = ze::Context::create(driver) {
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
    }

    if (device_ordinal as usize) >= devices.len() {
        eprintln!(
            "WARNING: --device {} is out of range. Only {} device(s) available (0..{}).",
            device_ordinal, devices.len(), devices.len().saturating_sub(1),
        );
    } else {
        eprintln!("  Selected device: {}", device_ordinal);
    }
    eprintln!();
}

// ---------------------------------------------------------------------------
// Device info (SYCL)
// ---------------------------------------------------------------------------

/// Print device info for all SYCL devices.
#[cfg(feature = "xpu-sycl")]
fn print_sycl_device_info(device_ordinal: u32) {
    use oneapi_rs::safe::{SyclDevice, SyclDeviceInfo};

    let count = match SyclDevice::count() {
        Ok(n) if n > 0 => n,
        _ => {
            eprintln!("(No SYCL devices found — skipping device info)");
            return;
        }
    };

    eprintln!("All SYCL devices:");
    for i in 0..count {
        if let Ok(dev) = SyclDevice::by_ordinal(i) {
            if let Ok(info) = dev.info() {
                eprintln!("  [{}] {}", i, info.name);
                eprintln!("    Vendor: {}  Backend: {}", info.vendor, info.backend);
                eprintln!(
                    "    Max compute units: {}  Global mem: {:.1} GB",
                    info.max_compute_units,
                    info.global_mem_size as f64 / (1024.0 * 1024.0 * 1024.0),
                );
            }
        }
    }

    if (device_ordinal as usize) >= count {
        eprintln!(
            "WARNING: --device {} is out of range. Only {} device(s) available (0..{}).",
            device_ordinal, count, count.saturating_sub(1),
        );
    } else {
        eprintln!("  Selected device: {}", device_ordinal);
    }
    eprintln!();
}

// ---------------------------------------------------------------------------
// Enums
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
}

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
            "fc_to_fc" => Some(Pattern::FcToFc),
            "lw_to_fc" => Some(Pattern::LwToFc),
            "fc_to_lw" => Some(Pattern::FcToLw),
            _ => None,
        }
    }
}

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
            "pinned" => Some(HostMemKind::Pinned),
            "system" => Some(HostMemKind::System),
            _ => None,
        }
    }

    fn to_storage_kind(&self) -> StorageKind {
        match self {
            HostMemKind::Pinned => StorageKind::Pinned,
            HostMemKind::System => StorageKind::System,
        }
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// KV-cache transfer benchmark via TransferManager (CUDA / XPU).
#[derive(Parser, Debug)]
#[command(name = "bench_transfer", about = "KV cache TransferManager benchmark (CUDA/XPU)")]
struct Cli {
    /// Device backend: cuda, ze (Level Zero), or sycl.
    #[arg(long)]
    backend: String,

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

    /// Comma-separated directions: h2d, d2h, d2d.
    #[arg(long, default_value = "h2d,d2h,d2d", value_delimiter = ',')]
    direction: Vec<String>,

    /// Comma-separated patterns: fc_to_fc, lw_to_fc, fc_to_lw.
    #[arg(
        long,
        default_value = "fc_to_fc,lw_to_fc,fc_to_lw",
        value_delimiter = ','
    )]
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
    device: u32,
}

// ---------------------------------------------------------------------------
// Layout builders
// ---------------------------------------------------------------------------

/// Build a fully-contiguous PhysicalLayout for the given storage kind.
fn build_fc_layout(
    agent: &NixlAgent,
    config: &LayoutConfig,
    storage_kind: StorageKind,
    ctx: &Arc<dyn DeviceAllocator>,
) -> Result<kvbm_physical::layout::PhysicalLayout> {
    let builder = PhysicalLayoutBuilder::new(agent.clone())
        .with_config(config.clone())
        .fully_contiguous();

    let layout = match storage_kind {
        StorageKind::System => builder.allocate_system().build()?,
        StorageKind::Pinned => builder.allocate_pinned(ctx.clone()).build()?,
        StorageKind::Device(_) => builder.allocate_device(ctx.clone()).build()?,
        other => anyhow::bail!("Unsupported storage kind: {:?}", other),
    };
    Ok(layout)
}

/// Build a layer-separate PhysicalLayout for the given storage kind.
fn build_lw_layout(
    agent: &NixlAgent,
    config: &LayoutConfig,
    storage_kind: StorageKind,
    ctx: &Arc<dyn DeviceAllocator>,
) -> Result<kvbm_physical::layout::PhysicalLayout> {
    let builder = PhysicalLayoutBuilder::new(agent.clone())
        .with_config(config.clone())
        .layer_separate(BlockDimension::BlockIsFirstDim);

    let layout = match storage_kind {
        StorageKind::System => builder.allocate_system().build()?,
        StorageKind::Pinned => builder.allocate_pinned(ctx.clone()).build()?,
        StorageKind::Device(_) => builder.allocate_device(ctx.clone()).build()?,
        other => anyhow::bail!("Unsupported storage kind: {:?}", other),
    };
    Ok(layout)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Parse backend (DeviceBackend::from_str handles cuda, ze, sycl, xpu, intel, etc.)
    let device_backend: DeviceBackend = cli.backend.parse()?;

    // Parse enums from CLI strings
    let directions: Vec<Direction> = cli
        .direction
        .iter()
        .map(|s| {
            Direction::from_str(s)
                .ok_or_else(|| anyhow::anyhow!("unknown direction: '{}' (valid: h2d, d2h, d2d)", s))
        })
        .collect::<Result<_>>()?;

    let patterns: Vec<Pattern> = cli
        .pattern
        .iter()
        .map(|s| {
            Pattern::from_str(s).ok_or_else(|| {
                anyhow::anyhow!("unknown pattern: '{}' (valid: fc_to_fc, lw_to_fc, fc_to_lw)", s)
            })
        })
        .collect::<Result<_>>()?;

    let host_mem_kinds: Vec<HostMemKind> = cli
        .host_mem
        .iter()
        .map(|s| {
            HostMemKind::from_str(s)
                .ok_or_else(|| anyhow::anyhow!("unknown host_mem: '{}' (valid: pinned, system)", s))
        })
        .collect::<Result<_>>()?;

    let tpb_values = &cli.tokens_per_block;
    let n_values = &cli.num_blocks;
    let warmup = cli.warmup;
    let iters = cli.iters;

    // Pre-compute total test count
    let mut total_tests = 0;
    for _ in tpb_values {
        for _ in n_values {
            for &dir in &directions {
                let hm_count = if matches!(dir, Direction::D2D) {
                    1
                } else {
                    host_mem_kinds.len()
                };
                total_tests += patterns.len() * hm_count;
            }
        }
    }

    // Create device context (shared across all tests)
    let ctx: Arc<dyn DeviceAllocator> =
        Arc::new(DeviceContext::new(device_backend, cli.device)?);

    // -- Print device info ----------------------------------------------------
    #[cfg(feature = "xpu-ze")]
    if matches!(device_backend, DeviceBackend::Ze) {
        print_ze_device_info(cli.device);
    }
    #[cfg(feature = "xpu-sycl")]
    if matches!(device_backend, DeviceBackend::Sycl) {
        print_sycl_device_info(cli.device);
    }

    // -- Print config ---------------------------------------------------------
    eprintln!("KV Cache Transfer Benchmark ({} / TransferManager)", device_backend.name());
    eprintln!("  Device ordinal: {}", cli.device);
    eprintln!("  Model: Llama 3.1 70B (bf16)");
    eprintln!(
        "  Layers: {NUM_LAYERS}, KV heads: {NUM_KV_HEADS}, Head dim: {HEAD_DIM}, \
         Outer dim: {OUTER_DIM}"
    );
    eprintln!("  Warmup: {warmup}, Timed: {iters}");
    eprintln!("  tokens_per_block: {:?}", tpb_values);
    eprintln!("  num_blocks: {:?}", n_values);
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
        "  host_mem: [{}]",
        host_mem_kinds
            .iter()
            .map(|h| h.label())
            .collect::<Vec<_>>()
            .join(", ")
    );
    eprintln!("  Total tests: {total_tests}");
    eprintln!();

    // -- CSV header (matching kvbench_xpu.rs) ---------------------------------
    println!(
        "tokens_per_block,num_blocks,pattern,direction,backend,host_mem,\
         total_bytes,inner_bytes,copy_size,num_copies,median_ms,bandwidth_gbps"
    );

    // -- Benchmark loop -------------------------------------------------------
    let mut test_num = 0;
    for &tpb in tpb_values {
        let inner = tpb * NUM_KV_HEADS * HEAD_DIM * ELEM_SIZE;
        let full_block_size = inner * OUTER_DIM * NUM_LAYERS;

        eprintln!(
            "--- tokens_per_block={tpb}, inner={inner} bytes ({} KB), \
             block={full_block_size} bytes ({:.1} MB) ---",
            inner / 1024,
            full_block_size as f64 / (1024.0 * 1024.0)
        );

        for &n in n_values {
            let total_bytes = full_block_size * n;

            for &direction in &directions {
                for &pattern in &patterns {
                    let (copy_size, num_copies) = match pattern {
                        Pattern::FcToFc => (full_block_size, n),
                        Pattern::LwToFc | Pattern::FcToLw => {
                            (inner, n * NUM_LAYERS * OUTER_DIM)
                        }
                    };

                    // D2D doesn't involve host memory; run once per pattern.
                    let host_mems: Vec<HostMemKind> =
                        if matches!(direction, Direction::D2D) {
                            vec![HostMemKind::Pinned] // sentinel; label will be "-"
                        } else {
                            host_mem_kinds.clone()
                        };

                    for &host_mem in &host_mems {
                        test_num += 1;

                        let host_label =
                            if matches!(direction, Direction::D2D) {
                                "-"
                            } else {
                                host_mem.label()
                            };

                        eprint!(
                            "  [{test_num}/{total_tests}] tpb={tpb} N={n:>3} \
                             {:<8} {:<6} transfer_mgr host={:<7} ... ",
                            pattern.label(),
                            direction.label(),
                            host_label,
                        );

                        if iters == 0 {
                            eprintln!("skipped (0 iters)");
                            continue;
                        }

                        // Determine src/dst layout shapes and storage kinds
                        let src_is_lw = matches!(pattern, Pattern::LwToFc);
                        let dst_is_lw = matches!(pattern, Pattern::FcToLw);

                        let (src_storage, dst_storage) = match direction {
                            Direction::H2D => (
                                host_mem.to_storage_kind(),
                                StorageKind::Device(cli.device),
                            ),
                            Direction::D2H => (
                                StorageKind::Device(cli.device),
                                host_mem.to_storage_kind(),
                            ),
                            Direction::D2D => (
                                StorageKind::Device(cli.device),
                                StorageKind::Device(cli.device),
                            ),
                        };

                        // Fresh TransferManager per test to ensure memory is
                        // freed between tests (no unregister_layout API).
                        let manager = TransferManager::builder()
                            .device_backend(device_backend)
                            .device_id(cli.device as usize)
                            .build()?;
                        let agent = manager.nixl_agent();

                        let config = LayoutConfig::builder()
                            .num_blocks(n)
                            .num_layers(NUM_LAYERS)
                            .outer_dim(OUTER_DIM)
                            .page_size(tpb)
                            .inner_dim(NUM_KV_HEADS * HEAD_DIM)
                            .dtype_width_bytes(ELEM_SIZE)
                            .build()?;

                        // Build source and destination layouts
                        let src_layout = if src_is_lw {
                            build_lw_layout(agent, &config, src_storage, &ctx)?
                        } else {
                            build_fc_layout(agent, &config, src_storage, &ctx)?
                        };
                        let dst_layout = if dst_is_lw {
                            build_lw_layout(agent, &config, dst_storage, &ctx)?
                        } else {
                            build_fc_layout(agent, &config, dst_storage, &ctx)?
                        };

                        let src_h = manager.register_layout(src_layout)?;
                        let dst_h = manager.register_layout(dst_layout)?;

                        let block_ids: Vec<BlockId> = (0..n).collect();
                        let options = TransferOptions::builder().build()?;

                        // Warmup
                        for _ in 0..warmup {
                            let notif = manager.execute_transfer(
                                src_h,
                                &block_ids,
                                dst_h,
                                &block_ids,
                                options.clone(),
                            )?;
                            notif.await?;
                        }

                        // Timed iterations
                        let mut latencies_ms = Vec::with_capacity(iters);
                        for _ in 0..iters {
                            let t0 = Instant::now();
                            let notif = manager.execute_transfer(
                                src_h,
                                &block_ids,
                                dst_h,
                                &block_ids,
                                options.clone(),
                            )?;
                            notif.await?;
                            latencies_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
                        }

                        // Median (matching kvbench_xpu.rs)
                        latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let median_ms = latencies_ms[latencies_ms.len() / 2];
                        let bw = (total_bytes as f64) / (median_ms / 1000.0) / 1e9;

                        println!(
                            "{tpb},{n},{},{},transfer_mgr,{host_label},\
                             {total_bytes},{inner},{copy_size},{num_copies},\
                             {median_ms:.4},{bw:.2}",
                            pattern.label(),
                            direction.label(),
                        );
                        eprintln!("{bw:.2} GB/s ({median_ms:.4} ms)");

                        // manager drops here -> layouts and their memory freed
                    }
                }
            }
        }
    }

    eprintln!();
    eprintln!("Done. {test_num} tests completed.");

    Ok(())
}
