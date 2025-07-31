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
use dynamo_llm::{
    block_manager::{
        block::{ImmutableBlock},
        storage::{DeviceAllocator, DeviceStorage, DiskAllocator, PinnedAllocator, Storage},
        BasicMetadata, BlockMetadata, BlockPool, KvBlockManager, KvBlockManagerConfig,
        KvManagerLayoutConfig, KvManagerModelConfig, KvManagerRuntimeConfig,
        locality
    },
    tokens::{TokenBlockSequence, Tokens},
};
use indicatif::ProgressIterator;
use json::parse;
use ordered_float::NotNan;
use plotters::prelude::*;
use std::collections::HashSet;
use std::fs::File;
use std::future::Future;
use std::io::{BufRead, BufReader};
use std::ops::Range;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio::time::{interval, Duration, Instant};

#[derive(Parser)]
struct Args {
    /// Path to the Mooncake trace.
    mooncake_trace_path: String,

    /// Amount of device blocks
    #[clap(short = 'd', long, default_value_t = 1536)]
    num_device_blocks: usize,

    /// Amount of host blocks
    #[clap(short = 'n', long, default_value_t = 4096)]
    num_host_blocks: usize,

    /// Amount of disk blocks
    #[clap(short = 'D', long, default_value_t = 0)]
    num_disk_blocks: usize,

    /// Amount of layers
    #[clap(long, default_value_t = 24)]
    num_layers: usize,

    /// Inner dimension
    #[clap(long, default_value_t = 512)]
    inner_dim: usize,

    /// Block size. This should match the mooncake trace. This should always be 512.
    #[clap(long, default_value_t = 512)]
    block_size: usize,

    /// Duration of the benchmark in seconds.
    #[clap(short = 't', long, default_value_t = 0)]
    duration_s: usize,

    /// Time scaling factor for the Mooncake trace.
    #[clap(short = 's', long, default_value_t = 1.0)]
    trace_scale_factor: f64,

    /// Simulated ITL (in milliseconds)
    #[clap(short = 'i', long, default_value_t = 32)]
    itl_ms: usize,
}

async fn build_manager(args: &Args) -> Result<KvBlockManager<locality::Local, BasicMetadata>> {
    let runtime_config = KvManagerRuntimeConfig::builder().worker_id(0).build()?;

    let model_config = KvManagerModelConfig::builder()
        .num_layers(args.num_layers)
        .page_size(args.block_size)
        .inner_dim(args.inner_dim)
        .outer_dim(1)
        .build()?;

    let device_layout = KvManagerLayoutConfig::builder()
        .num_blocks(args.num_device_blocks)
        .allocator(DeviceAllocator::default())
        .build()?;

    let mut config_build = KvBlockManagerConfig::builder()
        .runtime(runtime_config)
        .model(model_config)
        .device_layout(device_layout);

    if args.num_host_blocks > 0 {
        config_build = config_build.host_layout(
            KvManagerLayoutConfig::builder()
                .num_blocks(args.num_host_blocks)
                .allocator(PinnedAllocator::default())
                .build()?,
        );
    }

    if args.num_disk_blocks > 0 {
        config_build = config_build.disk_layout(
            KvManagerLayoutConfig::builder()
                .num_blocks(args.num_disk_blocks)
                .allocator(DiskAllocator)
                .build()?,
        );
    }

    let config = config_build.build()?;

    KvBlockManager::<locality::Local, BasicMetadata>::new(config).await
}

#[tokio::main]
pub async fn main() -> Result<()> {
    let args = Args::parse();

    let manager = build_manager(&args).await?;

    println!("Starting benchmark...");
    benchmark(manager, args).await?;

    Ok(())
}

/// A helper function to time the execution of a future.
async fn time<T, E: std::fmt::Debug>(f: impl Future<Output = Result<T, E>>) -> (T, Duration) {
    let start = Instant::now();
    let result = f.await.unwrap();
    let duration = start.elapsed();
    (result, duration)
}

/// A struct for storing timing data for some action.
/// Generally, we'd expect that the time to perform some action on a set of blocks
/// is composed of a fixed latency, as well as some per-block latency, hence the blocks parameter.
struct EventStats {
    time: Duration,
    num_blocks: usize,
}

impl EventStats {
    fn new(time: Duration, num_blocks: usize) -> Self {
        Self { time, num_blocks }
    }
}

struct Stats {
    pub device_match_latency: Arc<Mutex<Vec<EventStats>>>,
    pub host_match_latency: Arc<Mutex<Vec<EventStats>>>,
    pub host_onboard_latency: Arc<Mutex<Vec<EventStats>>>,
    pub disk_match_latency: Arc<Mutex<Vec<EventStats>>>,
    pub disk_onboard_latency: Arc<Mutex<Vec<EventStats>>>,
    pub prefill_allocate_latency: Arc<Mutex<Vec<EventStats>>>,
    pub prefill_register_latency: Arc<Mutex<Vec<EventStats>>>,
    pub decode_allocate_latency: Arc<Mutex<Vec<EventStats>>>,
    pub decode_register_latency: Arc<Mutex<Vec<EventStats>>>,
}

impl Stats {
    fn new() -> Self {
        Self {
            device_match_latency: Arc::new(Mutex::new(Vec::new())),
            host_match_latency: Arc::new(Mutex::new(Vec::new())),
            host_onboard_latency: Arc::new(Mutex::new(Vec::new())),
            disk_match_latency: Arc::new(Mutex::new(Vec::new())),
            disk_onboard_latency: Arc::new(Mutex::new(Vec::new())),
            prefill_allocate_latency: Arc::new(Mutex::new(Vec::new())),
            prefill_register_latency: Arc::new(Mutex::new(Vec::new())),
            decode_allocate_latency: Arc::new(Mutex::new(Vec::new())),
            decode_register_latency: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

fn make_range(seq: Vec<f64>) -> Range<f64> {
    let min = seq
        .iter()
        .map(|x| NotNan::new(*x).unwrap())
        .min()
        .unwrap()
        .into_inner();

    let max = seq
        .iter()
        .map(|x| NotNan::new(*x).unwrap())
        .max()
        .unwrap()
        .into_inner();

    (min - (max - min) * 0.1)..(max + (max - min) * 0.1)
}

// Taken from https://github.com/plotters-rs/plotters/blob/master/plotters/examples/normal-dist.rs
async fn scatter(
    name: &str,
    file_name: &str,
    latencies: Arc<Mutex<Vec<EventStats>>>,
) -> Result<()> {
    if latencies.lock().await.is_empty() {
        println!("Skipping {} because no data was collected.", name);
        return Ok(());
    }

    // TODO: We need to handle outliers better.
    let drawing_area = BitMapBackend::new(file_name, (2560, 1440)).into_drawing_area();
    drawing_area.fill(&WHITE)?;

    // Gather up our data.
    let data = latencies
        .lock()
        .await
        .iter()
        .map(|e| (e.num_blocks as f64, e.time.as_micros() as f64))
        .collect::<Vec<_>>();

    let x = data.iter().map(|(x, _)| *x).collect::<Vec<_>>();
    let y = data.iter().map(|(_, y)| *y).collect::<Vec<_>>();

    // Create a chart builder
    let mut chart = ChartBuilder::on(&drawing_area)
        .caption(name, ("sans-serif", 50))
        .margin(10)
        .x_label_area_size(75)
        .y_label_area_size(75)
        .build_cartesian_2d(make_range(x), make_range(y))?;

    // Configure the chart
    chart
        .configure_mesh()
        .x_desc("Amount of blocks".to_string())
        .y_desc("Latency (us)".to_string())
        .draw()?;

    // Draw the scatter plot
    chart.draw_series(data.iter().map(|point| {
        Circle::new(
            *point,
            4,
            ShapeStyle {
                color: RGBAColor(0, 0, 0, 0.25),
                filled: true,
                stroke_width: 1,
            },
        )
    }))?;

    // Save the result
    drawing_area.present()?;

    Ok(())
}

async fn match_and_onboard<S: Storage, M: BlockMetadata>(
    manager: &KvBlockManager<locality::Local, M>,
    pool: &dyn BlockPool<S, locality::Local, M>,
    sequence_hashes: &[u64],
    match_stats: Arc<Mutex<Vec<EventStats>>>,
    onboard_stats: Arc<Mutex<Vec<EventStats>>>,
) -> Result<Vec<ImmutableBlock<DeviceStorage, locality::Local, M>>> {
    let (blocks, match_latency) = time(pool.match_sequence_hashes(sequence_hashes)).await;

    match_stats
        .lock()
        .await
        .push(EventStats::new(match_latency, blocks.len()));

    // For any host blocks we found, onboard them to the device.
    if !blocks.is_empty() {
        let (onboard_blocks, onboard_latency) = time(manager.onboard_blocks(blocks, None)).await;

        let onboard_blocks = onboard_blocks?;

        onboard_stats
            .lock()
            .await
            .push(EventStats::new(onboard_latency, onboard_blocks.len()));

        Ok(onboard_blocks)
    } else {
        Ok(vec![])
    }
}

struct Request {
    timestamp: u64,
    seq: TokenBlockSequence,
    osl: u64,
}

fn load_trace(args: &Args) -> Result<Vec<Request>> {
    let file = File::open(&args.mooncake_trace_path)?;
    let reader = BufReader::new(file);

    let mut requests = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let data = parse(&line)?;

        let timestamp = data["timestamp"].as_f64().unwrap() / args.trace_scale_factor;

        if args.duration_s > 0 && timestamp > args.duration_s as f64 * 1000.0 {
            break;
        }

        let osl = data["output_length"].as_u64().unwrap();

        let hash_ids = data["hash_ids"]
            .members()
            .map(|h| h.as_u32().unwrap())
            .collect::<Vec<_>>();

        let tokens = hash_ids
            .iter()
            .flat_map(|h| vec![*h; args.block_size])
            .collect::<Vec<_>>();

        let seq = TokenBlockSequence::new(Tokens::from(tokens), args.block_size as u32, None);
        assert_eq!(seq.blocks().len(), hash_ids.len());

        requests.push(Request {
            timestamp: timestamp as u64,
            seq,
            osl,
        });
    }

    let mut cache_hit = 0;
    let mut seen_hashes = HashSet::new();

    for request in &requests {
        for block in request.seq.blocks() {
            if seen_hashes.contains(&block.sequence_hash()) {
                cache_hit += 1;
            }
            seen_hashes.insert(block.sequence_hash());
        }
    }

    println!("Max theoretical cache hits: {}", cache_hit);

    Ok(requests)
}

async fn benchmark(manager: KvBlockManager<locality::Local, BasicMetadata>, args: Args) -> Result<()> {
    let (req_tx, mut req_rx) = mpsc::unbounded_channel();
    let manager = Arc::new(manager);

    println!("Loading trace...");

    // Loading the trace can be quite slow, so we do this ahead of time.
    let inputs = load_trace(&args)?;

    println!("Beginning benchmark...");
    // Enqueue worker.
    let enqueue_worker = tokio::spawn(async move {
        let start = Instant::now();
        for request in inputs.into_iter().progress() {
            let deadline = start + Duration::from_millis(request.timestamp);
            tokio::time::sleep_until(deadline).await;

            if req_tx.send(request).is_err() {
                break;
            }
        }
    });

    // A global object to aggregate timing data.
    let stats = Arc::new(Stats::new());
    let stats_clone = stats.clone();

    // Block manager worker.
    let block_manager_worker = tokio::spawn(async move {
        let mut handles = Vec::new();

        while let Some(mut request) = req_rx.recv().await {
            let manager = manager.clone();
            let stats = stats_clone.clone();

            // We don't necessarily want to finish one request before starting the next.
            // So we spawn a new task for each request.
            // TODO: Could this be a bottleneck for very high request rates?
            handles.push(tokio::spawn(async move {
                let device = manager.device().unwrap();
                let host = manager.host();
                let disk = manager.disk();

                let mut sequence_blocks = Vec::new();

                let sequence_hashes = request
                    .seq
                    .blocks()
                    .iter()
                    .map(|b| b.sequence_hash())
                    .collect::<Vec<_>>();

                // First, check for matching blocks on the device, and log the lookup latency.
                let (device_blocks, device_match_latency) =
                    time(device.match_sequence_hashes(sequence_hashes.as_slice())).await;
                stats
                    .device_match_latency
                    .lock()
                    .await
                    .push(EventStats::new(device_match_latency, device_blocks.len()));

                sequence_blocks.extend(device_blocks);

                // If we weren't able to find all our blocks on the device, check the host.
                if sequence_blocks.len() < sequence_hashes.len() && host.is_some() {
                    sequence_blocks.extend(
                        match_and_onboard(
                            &manager,
                            host.unwrap(),
                            &sequence_hashes[sequence_blocks.len()..],
                            stats.host_match_latency.clone(),
                            stats.host_onboard_latency.clone(),
                        )
                        .await
                        .unwrap(),
                    );
                }

                // If we weren't able to find all our blocks on the device and host, check the disk.
                if sequence_blocks.len() < sequence_hashes.len() && disk.is_some() {
                    sequence_blocks.extend(
                        match_and_onboard(
                            &manager,
                            disk.unwrap(),
                            &sequence_hashes[sequence_blocks.len()..],
                            stats.disk_match_latency.clone(),
                            stats.disk_onboard_latency.clone(),
                        )
                        .await
                        .unwrap(),
                    );
                }

                let remaining = sequence_hashes.len() - sequence_blocks.len();

                // If we still need more blocks, allocate them.
                if remaining > 0 {
                    let (mut allocated_blocks, allocate_latency) =
                        time(device.allocate_blocks(remaining)).await;

                    stats
                        .prefill_allocate_latency
                        .lock()
                        .await
                        .push(EventStats::new(allocate_latency, allocated_blocks.len()));

                    for (block, allocated_block) in request.seq.blocks()[sequence_blocks.len()..]
                        .iter()
                        .zip(allocated_blocks.iter_mut())
                    {
                        allocated_block.apply_token_block(block.clone()).unwrap();
                    }

                    // Register any newly allocated blocks.
                    let (registered_blocks, prefill_register_latency) =
                        time(device.register_blocks(allocated_blocks)).await;

                    stats
                        .prefill_register_latency
                        .lock()
                        .await
                        .push(EventStats::new(
                            prefill_register_latency,
                            registered_blocks.len(),
                        ));

                    sequence_blocks.extend(registered_blocks);
                }

                // Simulate the decode phase.
                let mut itl_interval = interval(Duration::from_millis(args.itl_ms as u64));
                for _ in 0..request.osl {
                    itl_interval.tick().await;
                    if let Ok(Some(_)) = request.seq.append(0) {
                        let block = request.seq.blocks().last().unwrap();

                        let (device_block, allocate_latency) =
                            time(device.allocate_blocks(1)).await;
                        let mut device_block = device_block.into_iter().next().unwrap();

                        stats
                            .decode_allocate_latency
                            .lock()
                            .await
                            .push(EventStats::new(allocate_latency, 1));

                        device_block.apply_token_block(block.clone()).unwrap();
                        let (device_block, decode_register_latency) =
                            time(device.register_blocks(vec![device_block])).await;
                        let device_block = device_block.into_iter().next().unwrap();

                        stats
                            .decode_register_latency
                            .lock()
                            .await
                            .push(EventStats::new(decode_register_latency, 1));

                        sequence_blocks.push(device_block);
                    }
                }
            }));
        }

        futures::future::join_all(handles).await;
    });

    enqueue_worker.await?;
    block_manager_worker.await?;

    println!("Benchmark complete. Generating plots...");
    scatter(
        "Device Match Latency",
        "device_match_latency.png",
        stats.device_match_latency.clone(),
    )
    .await
    .unwrap();

    scatter(
        "Host Match Latency",
        "host_match_latency.png",
        stats.host_match_latency.clone(),
    )
    .await
    .unwrap();

    scatter(
        "Host Onboard Latency",
        "host_onboard_latency.png",
        stats.host_onboard_latency.clone(),
    )
    .await
    .unwrap();

    scatter(
        "Disk Match Latency",
        "disk_match_latency.png",
        stats.disk_match_latency.clone(),
    )
    .await
    .unwrap();

    scatter(
        "Disk Onboard Latency",
        "disk_onboard_latency.png",
        stats.disk_onboard_latency.clone(),
    )
    .await
    .unwrap();

    scatter(
        "Prefill Allocate Latency",
        "prefill_allocate_latency.png",
        stats.prefill_allocate_latency.clone(),
    )
    .await
    .unwrap();

    scatter(
        "Prefill Register Latency",
        "prefill_register_latency.png",
        stats.prefill_register_latency.clone(),
    )
    .await
    .unwrap();

    scatter(
        "Decode Allocate Latency",
        "decode_allocate_latency.png",
        stats.decode_allocate_latency.clone(),
    )
    .await
    .unwrap();

    scatter(
        "Decode Register Latency",
        "decode_register_latency.png",
        stats.decode_register_latency.clone(),
    )
    .await
    .unwrap();

    println!(
        "Device cache hits: {}",
        stats
            .device_match_latency
            .lock()
            .await
            .iter()
            .map(|s| s.num_blocks)
            .sum::<usize>()
    );
    println!(
        "Host cache hits: {}",
        stats
            .host_match_latency
            .lock()
            .await
            .iter()
            .map(|s| s.num_blocks)
            .sum::<usize>()
    );
    println!(
        "Disk cache hits: {}",
        stats
            .disk_match_latency
            .lock()
            .await
            .iter()
            .map(|s| s.num_blocks)
            .sum::<usize>()
    );

    Ok(())
}
