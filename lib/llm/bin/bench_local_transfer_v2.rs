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

use dynamo_llm::block_manager::v2::device::{DeviceBackend, DeviceContext};
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

    /// Device backend to use (cuda, hpu, or ze)
    #[clap(long, default_value = "cuda")]
    backend: String,

    /// Device ID
    #[clap(long, default_value_t = 0)]
    device_id: u32,

    /// NIXL backends (comma-separated, e.g., "POSIX,GDS_MT" or "POSIX")
    #[clap(long, default_value = "POSIX")]
    nixl_backends: String,

    /// Benchmark mode: "disk-device" for disk↔device, "host-device" for host↔device
    #[clap(long, default_value = "disk-device")]
    mode: String,

    /// Transfer size in bytes for memory benchmarks (default: 1MB)
    #[clap(long, default_value_t = 1048576)]
    transfer_size: usize,
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

    match args.mode.to_lowercase().as_str() {
        "disk-device" => benchmark_disk(&args).await?,
        "host-device" => benchmark_memory(&args).await?,
        _ => anyhow::bail!(
            "Invalid mode: '{}'. Use 'disk-device' or 'host-device'",
            args.mode
        ),
    }

    Ok(())
}

fn build_layout(
    agent: NixlAgent,
    config: LayoutConfig,
    storage_kind: StorageKind,
    device_backend: DeviceBackend,
    device_id: u32,
) -> PhysicalLayout {
    let builder = PhysicalLayout::builder(agent, device_backend, device_id)
        .with_config(config)
        .fully_contiguous();

    match storage_kind {
        StorageKind::System => builder.allocate_system().build().unwrap(),
        StorageKind::Pinned => builder.allocate_pinned(None).build().unwrap(),
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

async fn benchmark_disk(args: &Args) -> Result<()> {
    // Parse device backend with runtime availability checks
    let device_backend = match args.backend.to_lowercase().as_str() {
        "cuda" => {
            if DeviceBackend::Cuda.is_available() {
                DeviceBackend::Cuda
            } else {
                anyhow::bail!("CUDA backend not available on this system")
            }
        }
        "hpu" => {
            if DeviceBackend::Hpu.is_available() {
                DeviceBackend::Hpu
            } else {
                anyhow::bail!("HPU (Synapse) backend not available on this system")
            }
        }
        "ze" | "xpu" => {
            if DeviceBackend::Ze.is_available() {
                DeviceBackend::Ze
            } else {
                anyhow::bail!("XPU (Level-Zero) backend not available on this system")
            }
        }
        _ => {
            let available_backends = DeviceBackend::list_available();
            let available_names: Vec<_> = available_backends.iter().map(|b| b.name()).collect();
            if available_names.is_empty() {
                anyhow::bail!(
                    "Invalid backend: '{}'. No device backends are available on this system.",
                    args.backend
                )
            } else {
                anyhow::bail!(
                    "Invalid backend: '{}'. Available backends on this system: {}",
                    args.backend,
                    available_names.join(", ")
                )
            }
        }
    };

    // Parse NIXL backends
    let nixl_backends: Vec<&str> = args.nixl_backends.split(',').map(|s| s.trim()).collect();
    let agent = NixlAgent::require_backends("test_agent", &nixl_backends)?;

    let src_dst_config = LayoutConfig::builder()
        .num_blocks(args.num_blocks)
        .num_layers(args.num_layers)
        .outer_dim(2)
        .page_size(args.block_size)
        .inner_dim(args.inner_dim)
        .dtype_width_bytes(2)
        .build()?;

    let disk_layout = build_layout(
        agent.clone(),
        src_dst_config.clone(),
        StorageKind::Disk(0),
        device_backend,
        args.device_id,
    );
    let device_layout = build_layout(
        agent.clone(),
        src_dst_config.clone(),
        StorageKind::Device(args.device_id),
        device_backend,
        args.device_id,
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
        device_backend,
        args.device_id,
    );

    let ctx = TransportManager::builder()
        .worker_id(0)
        .device_backend(device_backend)
        .device_id(args.device_id)
        .nixl_agent(agent)
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

async fn benchmark_memory(args: &Args) -> Result<()> {
    // Parse device backend with runtime availability checks
    let device_backend = match args.backend.to_lowercase().as_str() {
        "cuda" => {
            if DeviceBackend::Cuda.is_available() {
                DeviceBackend::Cuda
            } else {
                anyhow::bail!("CUDA backend not available on this system")
            }
        }
        "hpu" => {
            if DeviceBackend::Hpu.is_available() {
                DeviceBackend::Hpu
            } else {
                anyhow::bail!("HPU (Synapse) backend not available on this system")
            }
        }
        "ze" | "xpu" => {
            if DeviceBackend::Ze.is_available() {
                DeviceBackend::Ze
            } else {
                anyhow::bail!("XPU (Level-Zero) backend not available on this system")
            }
        }
        _ => {
            let available_backends = DeviceBackend::list_available();
            let available_names: Vec<_> = available_backends.iter().map(|b| b.name()).collect();
            if available_names.is_empty() {
                anyhow::bail!(
                    "Invalid backend: '{}'. No device backends are available on this system.",
                    args.backend
                )
            } else {
                anyhow::bail!(
                    "Invalid backend: '{}'. Available backends on this system: {}",
                    args.backend,
                    available_names.join(", ")
                )
            }
        }
    };

    println!("=== Memory Transfer Benchmark ===");
    println!("Backend: {}", device_backend.name());
    println!("Device ID: {}", args.device_id);
    println!(
        "Transfer size: {} bytes ({:.2} MB)",
        args.transfer_size,
        args.transfer_size as f64 / 1_048_576.0
    );
    println!("Iterations: {}", args.iterations);
    println!();

    // Create device context
    let ctx = DeviceContext::new(device_backend, args.device_id)?;
    let stream = ctx.create_stream()?;

    // Allocate device memory
    let dev_ptr = ctx.allocate_device(args.transfer_size)?;
    println!("✓ Allocated device memory: 0x{:x}", dev_ptr);

    // Allocate pinned host memory (for H2D source)
    let host_src_ptr = ctx.allocate_pinned(args.transfer_size)?;
    println!(
        "✓ Allocated pinned host memory (source): 0x{:x}",
        host_src_ptr
    );

    // Allocate pinned host memory (for D2H destination)
    let host_dst_ptr = ctx.allocate_pinned(args.transfer_size)?;
    println!(
        "✓ Allocated pinned host memory (destination): 0x{:x}",
        host_dst_ptr
    );

    // Initialize source host data with test pattern directly in pinned memory
    let host_src_slice =
        unsafe { std::slice::from_raw_parts_mut(host_src_ptr as *mut u8, args.transfer_size) };
    for i in 0..args.transfer_size {
        host_src_slice[i] = (i % 256) as u8;
    }
    println!("✓ Initialized host source data with test pattern");
    println!();

    // Benchmark H2D (Host-to-Device) - using pinned host buffer
    println!("Starting host_to_device benchmark...");
    let mut h2d_latencies = Vec::new();
    for _ in (0..args.iterations).progress() {
        let start = Instant::now();
        // Create slice view from pinned host memory
        let src_slice =
            unsafe { std::slice::from_raw_parts(host_src_ptr as *const u8, args.transfer_size) };
        stream.copy_h2d(dev_ptr, src_slice)?;
        stream.synchronize()?;
        let end = Instant::now();
        h2d_latencies.push(end.duration_since(start));
    }

    let h2d_bandwidth = get_memory_bandwidth_gbs(&h2d_latencies, args.transfer_size);
    println!("host_to_device bandwidth: {:.2} GB/s", h2d_bandwidth);
    println!(
        "  Mean latency: {:.2} µs",
        get_mean_latency_us(&h2d_latencies)
    );
    println!();

    // Benchmark D2H (Device-to-Host) - using pinned host buffer
    println!("Starting device_to_host benchmark...");
    let mut d2h_latencies = Vec::new();
    for _ in (0..args.iterations).progress() {
        let start = Instant::now();
        // Create mutable slice view from pinned host memory
        let dst_slice =
            unsafe { std::slice::from_raw_parts_mut(host_dst_ptr as *mut u8, args.transfer_size) };
        stream.copy_d2h(dst_slice, dev_ptr)?;
        stream.synchronize()?;
        let end = Instant::now();
        d2h_latencies.push(end.duration_since(start));
    }

    let d2h_bandwidth = get_memory_bandwidth_gbs(&d2h_latencies, args.transfer_size);
    println!("device_to_host bandwidth: {:.2} GB/s", d2h_bandwidth);
    println!(
        "  Mean latency: {:.2} µs",
        get_mean_latency_us(&d2h_latencies)
    );
    println!();

    // Verify data integrity
    println!("Verifying data integrity...");
    let src_slice =
        unsafe { std::slice::from_raw_parts(host_src_ptr as *const u8, args.transfer_size) };
    let dst_slice =
        unsafe { std::slice::from_raw_parts(host_dst_ptr as *const u8, args.transfer_size) };
    if src_slice == dst_slice {
        println!("✓ Data verification PASSED");
    } else {
        let mismatches: usize = src_slice
            .iter()
            .zip(dst_slice.iter())
            .filter(|(a, b)| a != b)
            .count();
        println!(
            "✗ Data verification FAILED: {} / {} bytes mismatch",
            mismatches, args.transfer_size
        );
        anyhow::bail!("Data integrity check failed");
    }

    // Cleanup
    ctx.free_device(dev_ptr)?;
    ctx.free_pinned(host_src_ptr)?;
    ctx.free_pinned(host_dst_ptr)?;
    println!("✓ Memory freed");

    Ok(())
}

fn get_memory_bandwidth_gbs(latencies: &[Duration], transfer_size: usize) -> f64 {
    let mean = latencies.iter().sum::<Duration>() / latencies.len() as u32;
    transfer_size as f64 / mean.as_nanos() as f64
}

fn get_mean_latency_us(latencies: &[Duration]) -> f64 {
    let mean = latencies.iter().sum::<Duration>() / latencies.len() as u32;
    mean.as_micros() as f64
}
