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
use clap::Parser;
use dynamo_llm::block_manager::{
    block::{BlockExt, MutableBlock},
    storage::{DeviceAllocator, DiskAllocator, PinnedAllocator},
    BasicMetadata, BlockMetadata, BlockPool, KvBlockManager, KvBlockManagerConfig,
    KvManagerLayoutConfig, KvManagerModelConfig, KvManagerRuntimeConfig, Storage,
};

use core::time::Duration;
use indicatif::ProgressIterator;
use std::time::Instant;
use tokio::time::sleep;

#[derive(Parser)]
struct Args {
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

    /// Amount of iterations
    #[clap(long, default_value_t = 100)]
    iterations: usize,
}

fn build_manager(args: &Args) -> Result<KvBlockManager<BasicMetadata>> {
    let runtime_config = KvManagerRuntimeConfig::builder().worker_id(0).build()?;

    let model_config = KvManagerModelConfig::builder()
        .num_layers(args.num_layers)
        .page_size(args.block_size)
        .inner_dim(args.inner_dim)
        .outer_dim(1)
        .build()?;

    let device_layout = KvManagerLayoutConfig::builder()
        .num_blocks(args.num_blocks)
        .allocator(DeviceAllocator::default())
        .build()?;

    let host_layout = KvManagerLayoutConfig::builder()
        .num_blocks(args.num_blocks)
        .allocator(PinnedAllocator::default())
        .build()?;

    let disk_layout = KvManagerLayoutConfig::builder()
        .num_blocks(args.num_blocks)
        .allocator(DiskAllocator)
        .build()?;

    let config = KvBlockManagerConfig::builder()
        .runtime(runtime_config)
        .model(model_config)
        .device_layout(device_layout)
        .host_layout(host_layout)
        .disk_layout(disk_layout)
        .build()?;

    KvBlockManager::<BasicMetadata>::new(config)
}

#[tokio::main]
pub async fn main() -> Result<()> {
    let args = Args::parse();

    let manager = build_manager(&args)?;

    benchmark(manager, args).await?;

    Ok(())
}

/// Create a block in the 'COMPLETED' state.
async fn completed_block<S: Storage, Metadata: BlockMetadata>(
    pool: &BlockPool<S, Metadata>,
    tokens: Vec<u32>,
) -> Result<MutableBlock<S, Metadata>> {
    let mut block = pool.allocate_blocks(1).await?.into_iter().next().unwrap();

    block.init_sequence(42)?;
    for token in tokens {
        block.add_token(token)?;
    }
    block.commit()?;
    Ok(block)
}

fn get_bandwidth_gbs(latencies: Vec<Duration>, args: &Args) -> f64 {
    let total_bytes =
        args.num_layers * args.inner_dim * args.block_size * args.blocks_per_batch * 2;
    let mean = latencies.iter().sum::<Duration>() / latencies.len() as u32;

    total_bytes as f64 / mean.as_nanos() as f64
}

async fn onboard_bandwidth<S: Storage, M: BlockMetadata>(
    manager: &KvBlockManager<M>,
    source: &BlockPool<S, M>,
    args: &Args,
) -> Result<()> {
    let mut latencies = Vec::new();

    for _ in (0..args.iterations).progress() {
        let mut blocks = Vec::new();

        for i in 0..args.blocks_per_batch {
            let source_block = completed_block(source, vec![i as u32; args.block_size]).await?;
            let immutable_source_block = source
                .register_blocks(vec![source_block])
                .await?
                .into_iter()
                .next()
                .unwrap();
            blocks.push(immutable_source_block);
        }

        let start = Instant::now();
        let _ = manager.onboard_blocks(blocks).await?;
        let duration = start.elapsed();
        latencies.push(duration);
    }

    println!(
        "Onboarding latency: {:?}",
        latencies.iter().sum::<Duration>() / latencies.len() as u32
    );
    println!(
        "Onboarding bandwidth: {:?} GB/s",
        get_bandwidth_gbs(latencies, args)
    );

    Ok(())
}

async fn benchmark(manager: KvBlockManager<BasicMetadata>, args: Args) -> Result<()> {
    println!("Warmup...");

    let device = manager.device().unwrap();
    let host = manager.host().unwrap();
    let disk = manager.disk().unwrap();

    for _ in 0..10 {
        let device_block = completed_block(device, vec![0; args.block_size]).await?;

        let immutable_device_block = device
            .register_blocks(vec![device_block])
            .await?
            .into_iter()
            .next()
            .unwrap();

        sleep(Duration::from_millis(100)).await;

        let disk_block = disk
            .match_sequence_hashes(vec![immutable_device_block.sequence_hash()?].as_slice())
            .await?
            .into_iter()
            .next()
            .unwrap();

        let _ = manager.onboard_blocks(vec![disk_block]).await?;
    }

    println!("Starting benchmark...");

    println!("=== Host ===");
    onboard_bandwidth(&manager, host, &args).await?;

    println!("=== Disk ===");
    onboard_bandwidth(&manager, disk, &args).await?;

    Ok(())
}
