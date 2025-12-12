// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

use anyhow::Result;
use clap::{Parser, ValueEnum};

use core::time::Duration;
use indicatif::ProgressIterator;
use std::time::Instant;

use dynamo_kvbm::v2::physical::{
    layout::{BlockDimension, LayoutConfig, PhysicalLayout},
    manager::TransferManager,
    transfer::{BounceBuffer, StorageKind, TransferOptions},
};
use dynamo_memory::nixl::NixlAgent;

use dynamo_kvbm::BlockId;

/// Layout type: fully contiguous (fc) or layer-wise (lw)
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum LayoutType {
    /// Fully contiguous layout
    #[default]
    Fc,
    /// Layer-wise layout
    Lw,
}

#[derive(Parser)]
struct Args {
    /// CUDA device ID
    #[clap(long, default_value_t = 0)]
    device_id: u32,

    /// Amount of layers
    #[clap(long, default_value_t = 24)]
    num_layers: usize,

    /// Inner dimension
    #[clap(long, default_value_t = 4096)]
    inner_dim: usize,

    /// Block size
    #[clap(long, default_value_t = 32)]
    block_size: usize,

    /// Amount of blocks per pool
    #[clap(long, default_value_t = 16)]
    num_blocks: usize,

    /// Amount of blocks per transferred batch
    #[clap(long, default_value_t = 4)]
    blocks_per_batch: usize,

    /// Amount of pinned bounce buffer blocks
    #[clap(long, default_value_t = 2)]
    num_bounce_blocks: usize,

    /// Amount of iterations
    #[clap(long, default_value_t = 100)]
    iterations: usize,

    /// Enable GDS (GPUDirect Storage) backend
    #[clap(long, default_value_t = false)]
    enable_gds: bool,

    /// Source layout type (fc = fully contiguous, lw = layer-wise)
    #[clap(long, value_enum, default_value_t = LayoutType::Fc)]
    src_layout: LayoutType,

    /// Destination layout type (fc = fully contiguous, lw = layer-wise)
    #[clap(long, value_enum, default_value_t = LayoutType::Lw)]
    dst_layout: LayoutType,

    /// Bounce buffer layout type (fc = fully contiguous, lw = layer-wise)
    #[clap(long, value_enum, default_value_t = LayoutType::Fc)]
    bounce_layout: LayoutType,
}

#[tokio::main]
pub async fn main() -> Result<()> {
    let args = Args::parse();

    benchmark(&args).await?;

    Ok(())
}

fn build_layout(
    agent: NixlAgent,
    config: LayoutConfig,
    storage_kind: StorageKind,
    device_id: u32,
    layout_type: LayoutType,
) -> PhysicalLayout {
    let base_builder = PhysicalLayout::builder(agent).with_config(config);

    // Helper to allocate based on storage kind
    macro_rules! allocate {
        ($builder:expr) => {
            match storage_kind {
                StorageKind::System => $builder.allocate_system().build().unwrap(),
                // Pass device_id for NUMA-aligned pinned memory allocation
                StorageKind::Pinned => $builder.allocate_pinned(Some(device_id)).build().unwrap(),
                StorageKind::Device(_) => $builder.allocate_device(device_id).build().unwrap(),
                StorageKind::Disk(_) => $builder.allocate_disk(None).build().unwrap(),
            }
        };
    }

    // Select layout type and allocate
    match layout_type {
        LayoutType::Fc => allocate!(base_builder.fully_contiguous()),
        LayoutType::Lw => {
            allocate!(base_builder.layer_separate(BlockDimension::BlockIsFirstDim))
        }
    }
}

fn get_bandwidth_gbs(latencies: &[Duration], args: &Args) -> f64 {
    let total_bytes =
        args.num_layers * args.inner_dim * args.block_size * args.blocks_per_batch * 2;
    let mean = latencies.iter().sum::<Duration>() / latencies.len() as u32;

    total_bytes as f64 / mean.as_nanos() as f64
}

async fn benchmark(args: &Args) -> Result<()> {
    // Build backend list based on options
    let backends: Vec<&str> = if args.enable_gds {
        vec!["POSIX", "GDS_MT"]
    } else {
        vec!["POSIX"]
    };
    let agent = NixlAgent::with_backends("test_agent", &backends)?;

    println!(
        "Using device_id={}, enable_gds={}, src_layout={:?}, dst_layout={:?}, bounce_layout={:?}",
        args.device_id, args.enable_gds, args.src_layout, args.dst_layout, args.bounce_layout
    );

    let src_dst_config = LayoutConfig::builder()
        .num_blocks(args.num_blocks)
        .num_layers(args.num_layers)
        .outer_dim(2)
        .page_size(args.block_size)
        .inner_dim(args.inner_dim)
        .dtype_width_bytes(2)
        .build()?;

    // Build disk layout (source for disk_to_device, destination for device_to_disk)
    let disk_layout = build_layout(
        agent.clone(),
        src_dst_config.clone(),
        StorageKind::Disk(0),
        args.device_id,
        args.src_layout, // disk is typically the source
    );
    // Build device layout (destination for disk_to_device, source for device_to_disk)
    let device_layout = build_layout(
        agent.clone(),
        src_dst_config.clone(),
        StorageKind::Device(args.device_id),
        args.device_id,
        args.dst_layout, // device is typically the destination
    );

    let bounce_config = LayoutConfig::builder()
        .num_blocks(args.num_bounce_blocks)
        .num_layers(args.num_layers)
        .outer_dim(2)
        .page_size(args.block_size)
        .inner_dim(args.inner_dim)
        .dtype_width_bytes(2)
        .build()?;

    let bounce_layout = build_layout(
        agent.clone(),
        bounce_config.clone(),
        StorageKind::Pinned,
        args.device_id,
        args.bounce_layout,
    );

    // Build a pinned layout for baseline testing (same size as src/dst)
    let pinned_layout = build_layout(
        agent.clone(),
        src_dst_config.clone(),
        StorageKind::Pinned,
        args.device_id,
        args.bounce_layout, // Use bounce layout type for consistency
    );

    // Create transfer manager and register layouts
    let manager = TransferManager::builder()
        .nixl_agent(agent)
        .cuda_device_id(args.device_id as usize)
        .build()?;

    let disk_handle = manager.register_layout(disk_layout)?;
    let device_handle = manager.register_layout(device_layout)?;
    let bounce_handle = manager.register_layout(bounce_layout)?;
    let pinned_handle = manager.register_layout(pinned_layout)?;

    anyhow::ensure!(
        args.blocks_per_batch <= args.num_blocks,
        "blocks_per_batch must be less than or equal to num_blocks"
    );
    let blocks: Vec<BlockId> = (0..args.blocks_per_batch).map(|x| x as BlockId).collect();

    // ============================================================
    // Baseline: Direct pinned <-> device transfers (no bounce buffer)
    // ============================================================
    println!("\n=== Baseline: Direct Pinned <-> Device Transfers ===\n");

    let no_bounce_options = TransferOptions::default();

    for (src_handle, dst_handle, name) in [
        (pinned_handle, device_handle, "pinned_to_device"),
        (device_handle, pinned_handle, "device_to_pinned"),
    ] {
        println!("Starting {} benchmark...", name);

        let mut latencies = Vec::new();
        for _ in (0..args.iterations).progress() {
            let start = Instant::now();
            manager
                .execute_transfer(
                    src_handle,
                    blocks.as_slice(),
                    dst_handle,
                    blocks.as_slice(),
                    no_bounce_options.clone(),
                )?
                .await?;
            let end = Instant::now();
            let duration = end.duration_since(start);
            latencies.push(duration);
        }

        println!(
            "{} bandwidth: {:.3} GB/s\n",
            name,
            get_bandwidth_gbs(&latencies, args)
        );
    }

    // ============================================================
    // Bounce buffer: Disk <-> Device transfers (via pinned bounce buffer)
    // ============================================================
    println!("\n=== Bounce Buffer: Disk <-> Device Transfers ===\n");

    // Create bounce buffer spec using handles
    let bounce_buffer = BounceBuffer::from_handle(
        bounce_handle,
        (0..args.num_bounce_blocks).map(|x| x as BlockId).collect(),
    );

    let bounce_options = TransferOptions::builder()
        .bounce_buffer(bounce_buffer)
        .build()?;

    for (src_handle, dst_handle, name) in [
        (disk_handle, device_handle, "disk_to_device"),
        (device_handle, disk_handle, "device_to_disk"),
    ] {
        println!("Starting {} benchmark...", name);

        let mut latencies = Vec::new();
        for _ in (0..args.iterations).progress() {
            let options_clone = bounce_options.clone();
            let start = Instant::now();
            manager
                .execute_transfer(
                    src_handle,
                    blocks.as_slice(),
                    dst_handle,
                    blocks.as_slice(),
                    options_clone,
                )?
                .await?;
            let end = Instant::now();
            let duration = end.duration_since(start);
            latencies.push(duration);
        }

        println!(
            "{} bandwidth: {:.3} GB/s\n",
            name,
            get_bandwidth_gbs(&latencies, args)
        );
    }

    Ok(())
}
