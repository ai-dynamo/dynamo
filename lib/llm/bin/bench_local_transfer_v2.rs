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

use anyhow::Result;
use clap::Parser;

use core::time::Duration;
use indicatif::ProgressIterator;
use std::time::Instant;

use dynamo_llm::block_manager::v2::physical::{
    layout::LayoutConfig,
    transfer::{
        BounceBufferSpec, NixlAgent, PhysicalLayout, StorageKind, TransferOptions,
        TransportManager, executor::execute_transfer,
    },
};

use std::sync::Arc;

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

    /// Amount of pinned bounce buffer blocks
    #[clap(long, default_value_t = 2)]
    num_bounce_blocks: usize,

    /// Amount of iterations
    #[clap(long, default_value_t = 100)]
    iterations: usize,
}

struct DummyBounceBufferSpec {
    pub layout: PhysicalLayout,
    pub block_ids: Vec<usize>,
}

impl BounceBufferSpec for DummyBounceBufferSpec {
    fn layout(&self) -> &PhysicalLayout {
        &self.layout
    }
    fn block_ids(&self) -> &[usize] {
        &self.block_ids
    }
}

#[tokio::main]
pub async fn main() -> Result<()> {
    let args = Args::parse();

    // let manager = build_manager(&args).await?;

    benchmark(&args).await?;

    Ok(())
}

fn build_layout(
    agent: NixlAgent,
    config: LayoutConfig,
    storage_kind: StorageKind,
) -> PhysicalLayout {
    let builder = PhysicalLayout::builder(agent)
        .with_config(config)
        .fully_contiguous();

    match storage_kind {
        StorageKind::System => builder.allocate_system().build().unwrap(),
        StorageKind::Pinned => builder.allocate_pinned(false).build().unwrap(),
        StorageKind::Device(device_id) => builder.allocate_device(device_id).build().unwrap(),
        StorageKind::Disk(_) => builder.allocate_disk(None).build().unwrap(),
    }
}

fn get_bandwidth_gbs(latencies: Vec<Duration>, args: &Args) -> f64 {
    let total_bytes =
        args.num_layers * args.inner_dim * args.block_size * args.blocks_per_batch * 2;
    let mean = latencies.iter().sum::<Duration>() / latencies.len() as u32;

    total_bytes as f64 / mean.as_nanos() as f64
}

async fn benchmark(args: &Args) -> Result<()> {
    let agent = NixlAgent::require_backends("test_agent", &["POSIX", "GDS_MT"])?;
    let src_dst_config = LayoutConfig::builder()
        .num_blocks(args.num_blocks)
        .num_layers(args.num_layers)
        .outer_dim(2)
        .page_size(args.block_size)
        .inner_dim(args.inner_dim)
        .dtype_width_bytes(2)
        .build()?;

    let disk_layout = build_layout(agent.clone(), src_dst_config.clone(), StorageKind::Disk(0));
    let device_layout = build_layout(
        agent.clone(),
        src_dst_config.clone(),
        StorageKind::Device(0),
    );

    let bounce_config = LayoutConfig::builder()
        .num_blocks(args.num_bounce_blocks)
        .num_layers(args.num_layers)
        .outer_dim(2)
        .page_size(args.block_size)
        .inner_dim(args.inner_dim)
        .dtype_width_bytes(2)
        .build()?;

    let bounce_layout = build_layout(agent.clone(), bounce_config.clone(), StorageKind::Pinned);

    let ctx = TransportManager::builder()
        .worker_id(0)
        .nixl_agent(agent)
        .cuda_device_id(0)
        .build()?;

    let bounce_buffer_spec: Arc<dyn BounceBufferSpec> = Arc::new(DummyBounceBufferSpec {
        layout: bounce_layout,
        block_ids: (0..args.num_bounce_blocks).collect(),
    });

    let options = TransferOptions::builder()
        .bounce_buffer(bounce_buffer_spec)
        .build()?;

    anyhow::ensure!(
        args.blocks_per_batch <= args.num_blocks,
        "blocks_per_batch must be less than or equal to num_blocks"
    );
    let blocks = (0..args.blocks_per_batch).collect::<Vec<_>>();

    for (src, dst, name) in vec![
        (disk_layout.clone(), device_layout.clone(), "disk_to_device"),
        (device_layout, disk_layout, "device_to_disk"),
    ] {
        println!("Starting {} benchmark...", name);

        let mut latencies = Vec::new();
        for _ in (0..args.iterations).progress() {
            let options_clone = options.clone();
            let start = Instant::now();
            execute_transfer(
                &src,
                &dst,
                blocks.as_slice(),
                blocks.as_slice(),
                options_clone,
                ctx.context(),
            )?
            .await?;
            let end = Instant::now();
            let duration = end.duration_since(start);
            latencies.push(duration);
        }

        println!(
            "{} bandwidth: {:?} GB/s",
            name,
            get_bandwidth_gbs(latencies, args)
        );
    }

    Ok(())
}
