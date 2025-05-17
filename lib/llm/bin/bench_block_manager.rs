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
use clap::{Parser, Subcommand};
use dynamo_llm::{
    block_manager::{
        block::BlockExt,
        storage::cuda::{DeviceAllocator, PinnedAllocator},
        BasicMetadata, KvBlockManager, KvBlockManagerConfig, KvManagerLayoutConfig,
        KvManagerModelConfig, KvManagerRuntimeConfig,
    },
    tokens::{TokenBlockSequence, Tokens},
};
use indicatif::ProgressIterator;
use ordered_float::NotNan;
use plotters::prelude::*;
use rand::{rngs::ThreadRng, Rng};
use std::future::Future;
use std::ops::Range;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio::time::{interval, Duration, Instant};

#[derive(Parser)]
struct Args {
    /// Amount of device blocks
    #[clap(short = 'd', long, default_value_t = 2048)]
    num_device_blocks: usize,

    /// Amount of host blocks
    #[clap(short = 'n', long, default_value_t = 2048)]
    num_host_blocks: usize,

    /// Amount of layers
    #[clap(long, default_value_t = 24)]
    num_layers: usize,

    /// Inner dimension
    #[clap(long, default_value_t = 4096)]
    inner_dim: usize,

    /// Block size
    #[clap(long, default_value_t = 32)]
    block_size: usize,

    /// Duration of the benchmark in seconds.
    #[clap(short = 't', long, default_value_t = 60)]
    duration_s: usize,

    /// Amount of simulated inference requests per second.
    #[clap(short = 'r', long, default_value_t = 10)]
    requests_per_second: usize,

    /// Simulated OSL.
    #[clap(short = 'o', long, default_value_t = 128)]
    osl: usize,

    /// Simulated ITL (in milliseconds)
    #[clap(short = 'i', long, default_value_t = 16)]
    itl_ms: usize,

    /// Token generator to use.
    #[command(subcommand)]
    command: Commands,
}

/// A trait for objects which generate simulated inference requests following some pattern.
trait TokenGenerator: Send + Sync {
    fn next(&self, rng: &mut ThreadRng) -> Result<Vec<u32>>;
}

#[derive(Parser, Clone)]
struct RandomTokenGeneratorArgs {
    /// Number of tokens to generate
    #[clap(short, long, default_value_t = 1536)]
    num_tokens: usize,
}

/// A token generator which generates random tokens.
/// Useful as a sanity test and for testing offloading and allocations.
struct RandomTokenGenerator {
    num_tokens: usize,
}

impl From<RandomTokenGeneratorArgs> for RandomTokenGenerator {
    fn from(args: RandomTokenGeneratorArgs) -> Self {
        RandomTokenGenerator {
            num_tokens: args.num_tokens,
        }
    }
}

impl TokenGenerator for RandomTokenGenerator {
    fn next(&self, rng: &mut ThreadRng) -> Result<Vec<u32>> {
        // Just generate a random vector of specific length.
        Ok((0..self.num_tokens)
            .map(|_| rng.random_range(0..1024))
            .collect())
    }
}

#[derive(Parser, Clone)]
struct HierarchicalTokenGeneratorArgs {
    /// Amount of groups
    #[clap(short, long, default_value_t = 3)]
    num_groups: usize,

    /// Group size
    #[clap(short, long, default_value_t = 512)]
    group_size: usize,

    /// Amount of different sequences per group
    #[clap(short, long, default_value_t = 24)]
    variations_per_group: usize,
}

/// A token generator which generates simulated requests conducive to a lot of reuse.
/// We split the tokens into 'num_groups' groups of 'group_size' tokens each.
/// Each group can take on 'variations_per_group' different token sequences.
/// The variations of the first group will likely stay on device (and be reused),
/// while the later groups may move back and forth between device and host, or even be evicted.
struct HierarchicalTokenGenerator {
    num_groups: usize,
    group_size: usize,
    variations_per_group: usize,
}

impl From<HierarchicalTokenGeneratorArgs> for HierarchicalTokenGenerator {
    fn from(args: HierarchicalTokenGeneratorArgs) -> Self {
        HierarchicalTokenGenerator {
            num_groups: args.num_groups,
            group_size: args.group_size,
            variations_per_group: args.variations_per_group,
        }
    }
}

impl TokenGenerator for HierarchicalTokenGenerator {
    fn next(&self, rng: &mut ThreadRng) -> Result<Vec<u32>> {
        let mut tokens = Vec::new();
        // Generate each group.
        for _ in 0..self.num_groups {
            let variant = rng.random_range(0..self.variations_per_group as u32);
            // Generate each token in the group. For now, just use a fixed token id for all tokens.
            for _ in 0..self.group_size {
                tokens.push(variant);
            }
        }
        Ok(tokens)
    }
}

#[derive(Subcommand)]
enum Commands {
    Random(RandomTokenGeneratorArgs),
    Hierarchical(HierarchicalTokenGeneratorArgs),
}

fn build_manager(args: &Args) -> Result<KvBlockManager<BasicMetadata>> {
    let runtime_config = KvManagerRuntimeConfig::builder().worker_id(0).build()?;

    let model_config = KvManagerModelConfig::builder()
        .num_layers(args.num_layers)
        .page_size(args.block_size)
        .inner_dim(args.inner_dim)
        .build()?;

    let device_layout = KvManagerLayoutConfig::builder()
        .num_blocks(args.num_device_blocks)
        .allocator(DeviceAllocator::default())
        .build()?;

    let host_layout = if args.num_host_blocks > 0 {
        Some(
            KvManagerLayoutConfig::builder()
                .num_blocks(args.num_host_blocks)
                .allocator(PinnedAllocator::default())
                .build()?,
        )
    } else {
        None
    };

    let mut config_build = KvBlockManagerConfig::builder()
        .runtime(runtime_config)
        .model(model_config)
        .device_layout(device_layout);

    if let Some(host_layout) = host_layout {
        config_build = config_build.host_layout(host_layout);
    }

    let config = config_build.build()?;

    KvBlockManager::<BasicMetadata>::new(config)
}

#[tokio::main]
pub async fn main() -> Result<()> {
    let args = Args::parse();

    let manager = build_manager(&args)?;

    let token_generator: Arc<dyn TokenGenerator> = match &args.command {
        Commands::Random(token_gen) => Arc::new(RandomTokenGenerator::from(token_gen.clone())),
        Commands::Hierarchical(token_gen) => {
            Arc::new(HierarchicalTokenGenerator::from(token_gen.clone()))
        }
    };

    benchmark(manager, token_generator, args).await?;

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
    pub onboard_latency: Arc<Mutex<Vec<EventStats>>>,
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
            onboard_latency: Arc::new(Mutex::new(Vec::new())),
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
    // TODO: We need to handle outliers better.
    let drawing_area = BitMapBackend::new(file_name, (1920, 1080)).into_drawing_area();
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

async fn benchmark(
    manager: KvBlockManager<BasicMetadata>,
    token_generator: Arc<dyn TokenGenerator>,
    args: Args,
) -> Result<()> {
    let (req_tx, mut req_rx) = mpsc::unbounded_channel();
    let manager = Arc::new(manager);

    let mut rng = rand::rng();

    println!("Generating inputs...");

    // Generating our inputs can be quite slow, so we do this ahead of time.
    let block_size = args.block_size;
    let inputs: Vec<TokenBlockSequence> = (0..args.requests_per_second * args.duration_s)
        .progress()
        .map(|_| {
            let input_tokens = token_generator.next(&mut rng).unwrap();
            let tokens = Tokens::from(input_tokens);

            // Break up the sequence into blocks and get the sequence hashes.

            tokens.into_sequence(block_size, Some(0))
        })
        .collect();

    println!("Beginning benchmark...");
    // Enqueue worker.
    let enqueue_worker = tokio::spawn(async move {
        // Send a request every 1 / requests_per_second seconds.
        let mut interval = interval(Duration::from_micros(
            1000000 / args.requests_per_second as u64,
        ));

        for sequence_hashes in inputs.into_iter().progress() {
            req_tx.send(sequence_hashes).unwrap();

            tokio::select! {
                _ = interval.tick() => {}
                // Wait for signal to stop.
                _ = tokio::signal::ctrl_c() => {
                    break;
                },
            }
        }
    });

    // A global object to aggregate timing data.
    let stats = Arc::new(Stats::new());
    let stats_clone = stats.clone();

    // Block manager worker.
    let block_manager_worker = tokio::spawn(async move {
        while let Some(mut sequence) = req_rx.recv().await {
            let manager = manager.clone();
            let stats = stats_clone.clone();

            // We don't necessarily want to finish one request before starting the next.
            // So we spawn a new task for each request.
            // TODO: Could this be a bottleneck for very high request rates?
            tokio::spawn(async move {
                let device = manager.device().unwrap();
                let host = manager.host().unwrap();

                let mut sequence_blocks = Vec::new();

                let sequence_hashes = sequence
                    .blocks()
                    .iter()
                    .map(|block| block.sequence_hash())
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
                if sequence_blocks.len() < sequence_hashes.len() {
                    let (host_blocks, host_match_latency) =
                        time(host.match_sequence_hashes(&sequence_hashes[sequence_blocks.len()..]))
                            .await;

                    stats
                        .host_match_latency
                        .lock()
                        .await
                        .push(EventStats::new(host_match_latency, host_blocks.len()));

                    // For any host blocks we found, onboard them to the device.
                    if !host_blocks.is_empty() {
                        let (onboard_blocks, onboard_latency) =
                            time(manager.onboard_blocks(host_blocks)).await;

                        stats
                            .onboard_latency
                            .lock()
                            .await
                            .push(EventStats::new(onboard_latency, onboard_blocks.len()));

                        sequence_blocks.extend(onboard_blocks);
                    }
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

                    for (block, allocated_block) in sequence.blocks()[sequence_blocks.len()..]
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
                for _ in 0..args.osl {
                    itl_interval.tick().await;
                    if let Ok(Some(_)) = sequence.append(0) {
                        let block = sequence.blocks().last().unwrap();

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
            });
        }
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
        "Onboard Latency",
        "onboard_latency.png",
        stats.onboard_latency.clone(),
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

    Ok(())
}
