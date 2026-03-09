// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

//! XPU local transfer benchmark using Level Zero.
//!
//! Benchmarks KV cache block transfers between host memory and Intel XPU
//! device memory via the Level Zero backend (ze.rs executor).
//!
//! Usage:
//!   cargo run -p dynamo-llm --bin bench_xpu_transfer_v2 \
//!     --features block-manager-bench,level-zero

use anyhow::Result;
use clap::Parser;

use core::time::Duration;
use indicatif::ProgressIterator;
use std::time::Instant;

use dynamo_llm::block_manager::v2::physical::{
    layout::LayoutConfig,
    transfer::{
        NixlAgent, PhysicalLayout, StorageKind, TransferOptions,
        TransportManager, executor::execute_transfer,
    },
};

#[derive(Parser)]
struct Args {
    /// Amount of layers
    #[clap(long, default_value_t = 24)]
    num_layers: usize,

    /// Inner dimension
    #[clap(long, default_value_t = 4096)]
    inner_dim: usize,

    /// Block size (tokens per page)
    #[clap(long, default_value_t = 32)]
    block_size: usize,

    /// Amount of blocks per pool
    #[clap(long, default_value_t = 16)]
    num_blocks: usize,

    /// Amount of blocks per transferred batch
    #[clap(long, default_value_t = 4)]
    blocks_per_batch: usize,

    /// Amount of iterations
    #[clap(long, default_value_t = 100)]
    iterations: usize,

    /// XPU device ordinal
    #[clap(long, default_value_t = 0)]
    xpu_device_id: u32,
}

fn build_layout(
    agent: NixlAgent,
    config: LayoutConfig,
    storage_kind: StorageKind,
    xpu_device_id: u32,
) -> PhysicalLayout {
    let builder = PhysicalLayout::builder(agent)
        .with_config(config)
        .fully_contiguous();

    match storage_kind {
        StorageKind::System => builder.allocate_system().build().unwrap(),
        StorageKind::Pinned => builder.allocate_ze_host(xpu_device_id).build().unwrap(),
        StorageKind::XpuDevice(device_id) => {
            builder.allocate_xpu_device(device_id).build().unwrap()
        }
        other => panic!("Unsupported storage kind for XPU bench: {:?}", other),
    }
}

fn get_bandwidth_gbs(latencies: &[Duration], args: &Args) -> f64 {
    let total_bytes =
        args.num_layers * args.inner_dim * args.block_size * args.blocks_per_batch * 2;
    let mean = latencies.iter().sum::<Duration>() / latencies.len() as u32;

    total_bytes as f64 / mean.as_nanos() as f64
}

fn print_latency_stats(latencies: &[Duration]) {
    let mut sorted: Vec<f64> = latencies.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let p50 = sorted[sorted.len() / 2];
    let p99 = sorted[(sorted.len() as f64 * 0.99) as usize];
    let min = sorted[0];
    let max = sorted[sorted.len() - 1];

    println!(
        "  latency (ms): mean={:.3}, p50={:.3}, p99={:.3}, min={:.3}, max={:.3}",
        mean, p50, p99, min, max
    );
}

#[tokio::main]
pub async fn main() -> Result<()> {
    let args = Args::parse();

    println!("=== XPU Local Transfer Benchmark (Level Zero) ===");
    println!("  layers={}, inner_dim={}, block_size={}", args.num_layers, args.inner_dim, args.block_size);
    println!("  num_blocks={}, blocks_per_batch={}, iterations={}", args.num_blocks, args.blocks_per_batch, args.iterations);
    println!("  xpu_device_id={}", args.xpu_device_id);
    println!();

    benchmark(&args).await?;

    Ok(())
}

async fn benchmark(args: &Args) -> Result<()> {
    // Create NIXL agent with no backends — local Ze transfers don't need NIXL backends
    let agent = NixlAgent::new_with_backends("xpu_bench_agent", &[])?;

    let layout_config = LayoutConfig::builder()
        .num_blocks(args.num_blocks)
        .num_layers(args.num_layers)
        .outer_dim(2)
        .page_size(args.block_size)
        .inner_dim(args.inner_dim)
        .dtype_width_bytes(2)
        .build()?;

    let system_layout = build_layout(
        agent.clone(),
        layout_config.clone(),
        StorageKind::System,
        args.xpu_device_id,
    );
    let pinned_layout = build_layout(
        agent.clone(),
        layout_config.clone(),
        StorageKind::Pinned,
        args.xpu_device_id,
    );
    let xpu_layout = build_layout(
        agent.clone(),
        layout_config.clone(),
        StorageKind::XpuDevice(args.xpu_device_id),
        args.xpu_device_id,
    );

    // Build TransportManager: no CUDA, Ze only
    let ctx = TransportManager::builder()
        .nixl_agent(agent)
        .worker_id(0)
        .no_cuda()
        .xpu_device_id(args.xpu_device_id)
        .build()?;

    let options = TransferOptions::builder().build()?;

    anyhow::ensure!(
        args.blocks_per_batch <= args.num_blocks,
        "blocks_per_batch must be less than or equal to num_blocks"
    );
    let blocks = (0..args.blocks_per_batch).collect::<Vec<_>>();

    // Bench: System → XpuDevice, XpuDevice → System, Pinned → XpuDevice, XpuDevice → Pinned
    let bench_pairs: Vec<(&PhysicalLayout, &PhysicalLayout, &str)> = vec![
        (&system_layout, &xpu_layout, "system_to_xpu (H2D)"),
        (&xpu_layout, &system_layout, "xpu_to_system (D2H)"),
        (&pinned_layout, &xpu_layout, "pinned_to_xpu (H2D pinned)"),
        (&xpu_layout, &pinned_layout, "xpu_to_pinned (D2H pinned)"),
    ];

    for (src, dst, name) in bench_pairs {
        println!("Starting {} benchmark...", name);

        // Warmup: 5 iterations
        for _ in 0..5 {
            let options_clone = options.clone();
            execute_transfer(
                src,
                dst,
                blocks.as_slice(),
                blocks.as_slice(),
                options_clone,
                ctx.context(),
            )?
            .await?;
        }

        let mut latencies = Vec::new();
        for _ in (0..args.iterations).progress() {
            let options_clone = options.clone();
            let start = Instant::now();
            execute_transfer(
                src,
                dst,
                blocks.as_slice(),
                blocks.as_slice(),
                options_clone,
                ctx.context(),
            )?
            .await?;
            let end = Instant::now();
            latencies.push(end.duration_since(start));
        }

        let bw = get_bandwidth_gbs(&latencies, args);
        println!("  {} bandwidth: {:.3} GB/s", name, bw);
        print_latency_stats(&latencies);
        println!();
    }

    Ok(())
}
