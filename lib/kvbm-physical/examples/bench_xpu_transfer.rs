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

//! XPU integration benchmark using kvbm-physical's TransferManager API.
//!
//! Benchmarks KV cache block transfers between host memory and Intel XPU
//! device memory through the full kvbm-physical stack (TransferManager →
//! executor → ze.rs → vectorized_copy kernel).
//!
//! Covers:
//!   - H2D / D2H with system (unpinned) and ze_host (pinned) memory
//!   - D2D same-device: LW ↔ FC reshuffling (primary production use case)
//!
//! Usage:
//!   # XPU-only (no CUDA dependency):
//!   cargo build --example bench_xpu_transfer -p kvbm-physical \
//!     --features kvbm-physical/bench-xpu,kvbm-physical/no-cuda
//!   LD_LIBRARY_PATH=target/debug/build/kvbm-kernels-<hash>/out:$LD_LIBRARY_PATH \
//!     ./target/debug/examples/bench_xpu_transfer --device 0
//!
//!   # SYCL kernel is the default D2D backend. Build the .so first:
//!   #   make -C lib/kvbm-kernels/sycl
//!   # Then point LD_LIBRARY_PATH to include it.
//!
//!   # With L0 SPIR-V kernel override (requires ocl-kernel feature):
//!   cargo build --example bench_xpu_transfer -p kvbm-physical \
//!     --features kvbm-physical/bench-xpu,kvbm-physical/no-cuda,kvbm-physical/ocl-kernel
//!   KVBM_USE_L0_KERNEL=1 \
//!     LD_LIBRARY_PATH=lib/kvbm-kernels/sycl:target/debug/build/kvbm-kernels-<hash>/out:$LD_LIBRARY_PATH \
//!     ./target/debug/examples/bench_xpu_transfer --device 0
//!
//!  # Note: libkvbm_kernels.so = always needed (CUDA stubs or real) even not used in XPU, statically linked

use anyhow::Result;
use clap::Parser;
use std::time::{Duration, Instant};

use kvbm_physical::layout::{BlockDimension, LayoutConfig, PhysicalLayoutBuilder};
use kvbm_physical::manager::LayoutHandle;
use kvbm_physical::transfer::{NixlAgent, StorageKind};
use kvbm_physical::{BlockId, TransferManager, TransferOptions};

#[derive(Parser)]
#[command(name = "bench_xpu_transfer")]
#[command(about = "XPU integration benchmark via kvbm-physical TransferManager")]
struct Args {
    /// Number of KV-cache layers
    #[arg(long, default_value_t = 80)]
    num_layers: usize,

    /// Number of KV heads
    #[arg(long, default_value_t = 8)]
    num_kv_heads: usize,

    /// Head dimension
    #[arg(long, default_value_t = 128)]
    head_dim: usize,

    /// Outer dimension (K + V)
    #[arg(long, default_value_t = 2)]
    outer_dim: usize,

    /// Tokens per block (page size)
    #[arg(long, default_value_t = 16)]
    page_size: usize,

    /// Number of blocks per layout pool
    #[arg(long, default_value_t = 64)]
    num_blocks: usize,

    /// Blocks transferred per call
    #[arg(long, default_value_t = 16)]
    blocks_per_xfer: usize,

    /// Warmup iterations (untimed)
    #[arg(long, default_value_t = 5)]
    warmup: usize,

    /// Timed iterations
    #[arg(long, default_value_t = 100)]
    iters: usize,

    /// XPU device ordinal
    #[arg(long, default_value_t = 0)]
    device: u32,
}

/// Build a fully-contiguous PhysicalLayout for the given storage kind.
fn build_fc_layout(
    agent: &NixlAgent,
    config: &LayoutConfig,
    storage_kind: StorageKind,
    device_id: u32,
) -> Result<kvbm_physical::layout::PhysicalLayout> {
    let builder = PhysicalLayoutBuilder::new(agent.clone())
        .with_config(config.clone())
        .fully_contiguous();

    let layout = match storage_kind {
        StorageKind::System => builder.allocate_system().build()?,
        StorageKind::Pinned => builder.allocate_ze_host(device_id).build()?,
        StorageKind::XpuDevice(_) => builder.allocate_xpu_device(device_id).build()?,
        other => anyhow::bail!("Unsupported storage kind: {:?}", other),
    };
    Ok(layout)
}

/// Build a layer-separate PhysicalLayout on XPU device memory.
fn build_lw_layout(
    agent: &NixlAgent,
    config: &LayoutConfig,
    device_id: u32,
) -> Result<kvbm_physical::layout::PhysicalLayout> {
    let layout = PhysicalLayoutBuilder::new(agent.clone())
        .with_config(config.clone())
        .layer_separate(BlockDimension::BlockIsFirstDim)
        .allocate_xpu_device(device_id)
        .build()?;
    Ok(layout)
}

fn print_latency_stats(latencies: &[Duration]) {
    let mut sorted_ms: Vec<f64> = latencies.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
    sorted_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mean = sorted_ms.iter().sum::<f64>() / sorted_ms.len() as f64;
    let p50 = sorted_ms[sorted_ms.len() / 2];
    let p99 = sorted_ms[(sorted_ms.len() as f64 * 0.99) as usize];
    let min = sorted_ms[0];
    let max = sorted_ms[sorted_ms.len() - 1];

    println!(
        "    latency (ms): mean={:.3}, p50={:.3}, p99={:.3}, min={:.3}, max={:.3}",
        mean, p50, p99, min, max
    );
}

async fn run_bench(
    manager: &TransferManager,
    src_h: LayoutHandle,
    dst_h: LayoutHandle,
    block_ids: &[BlockId],
    warmup: usize,
    iters: usize,
    label: &str,
    bytes_per_xfer: usize,
) -> Result<()> {
    if iters == 0 && warmup == 0 {
        println!("  {:<26} skipped (0 iters)", label);
        return Ok(());
    }

    let options = TransferOptions::builder().build()?;

    // Warmup
    for _ in 0..warmup {
        let notif = manager.execute_transfer(
            src_h,
            block_ids,
            dst_h,
            block_ids,
            options.clone(),
        )?;
        notif.await?;
    }

    // Timed
    let mut latencies = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        let notif = manager.execute_transfer(
            src_h,
            block_ids,
            dst_h,
            block_ids,
            options.clone(),
        )?;
        notif.await?;
        latencies.push(start.elapsed());
    }

    let mean = latencies.iter().sum::<Duration>() / latencies.len() as u32;
    let bw_gbs = bytes_per_xfer as f64 / mean.as_nanos() as f64;

    println!("  {:<26} {:>8.3} ms   {:>8.3} GB/s",
        label,
        mean.as_secs_f64() * 1000.0,
        bw_gbs,
    );
    print_latency_stats(&latencies);

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let inner_dim = args.num_kv_heads * args.head_dim;
    let bytes_per_xfer = args.blocks_per_xfer
        * args.num_layers
        * args.outer_dim
        * args.page_size
        * inner_dim
        * 2; // bf16

    anyhow::ensure!(
        args.blocks_per_xfer <= args.num_blocks,
        "blocks_per_xfer ({}) must be <= num_blocks ({})",
        args.blocks_per_xfer,
        args.num_blocks,
    );

    println!("=== XPU Integration Benchmark (kvbm-physical) ===");
    println!("  config: num_layers={} page_size={} num_blocks={} blocks_per_xfer={}",
        args.num_layers, args.page_size, args.num_blocks, args.blocks_per_xfer);
    println!("          inner_dim={} ({}×{}) dtype=bf16 outer_dim={}",
        inner_dim, args.num_kv_heads, args.head_dim, args.outer_dim);
    println!("          warmup={} iters={} device={}", args.warmup, args.iters, args.device);
    println!("          bytes_per_xfer={} ({:.2} MiB)",
        bytes_per_xfer, bytes_per_xfer as f64 / (1024.0 * 1024.0));
    println!();

    // --- Build TransferManager (XPU-only, no CUDA, no NIXL backends) ---
    let manager = TransferManager::builder()
        .no_cuda()
        .xpu_device_id(args.device)
        .build()?;

    let agent = manager.nixl_agent();

    let config = LayoutConfig::builder()
        .num_blocks(args.num_blocks)
        .num_layers(args.num_layers)
        .outer_dim(args.outer_dim)
        .page_size(args.page_size)
        .inner_dim(inner_dim)
        .dtype_width_bytes(2_usize)
        .build()?;

    // --- Build and register 4 layouts ---

    // a) System memory (unpinned host)
    let sys = build_fc_layout(agent, &config, StorageKind::System, args.device)?;
    let sys_h = manager.register_layout(sys)?;

    // b) ZE host (pinned host via zeMemAllocHost)
    let pinned = build_fc_layout(agent, &config, StorageKind::Pinned, args.device)?;
    let pin_h = manager.register_layout(pinned)?;

    // c) XPU device — fully contiguous (FC tier)
    let dev_fc = build_fc_layout(agent, &config, StorageKind::XpuDevice(args.device), args.device)?;
    let fc_h = manager.register_layout(dev_fc)?;

    // d) XPU device — layer-wise (LW tier)
    let dev_lw = build_lw_layout(agent, &config, args.device)?;
    let lw_h = manager.register_layout(dev_lw)?;

    let block_ids: Vec<BlockId> = (0..args.blocks_per_xfer).collect();

    // --- Benchmark matrix ---
    println!("  {:<26} {:>11}   {:>11}", "Transfer", "Avg Latency", "Bandwidth");
    println!("  {:-<26} {:-<11}   {:-<11}", "", "", "");

    let pairs: Vec<(LayoutHandle, LayoutHandle, &str)> = vec![
        (sys_h, fc_h,  "system → xpu (H2D)"),
        (fc_h,  sys_h, "xpu → system (D2H)"),
        (pin_h, fc_h,  "pinned → xpu (H2D)"),
        (fc_h,  pin_h, "xpu → pinned (D2H)"),
        (lw_h,  fc_h,  "LW → FC (D2D)"),
        (fc_h,  lw_h,  "FC → LW (D2D)"),
    ];

    for (src_h, dst_h, label) in &pairs {
        run_bench(
            &manager,
            *src_h,
            *dst_h,
            &block_ids,
            args.warmup,
            args.iters,
            label,
            bytes_per_xfer,
        )
        .await?;
    }

    println!();
    println!("Done.");

    Ok(())
}
