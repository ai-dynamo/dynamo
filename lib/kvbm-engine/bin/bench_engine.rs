// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KVBM transfer bandwidth benchmark.
//!
//! Measures effective bandwidth of KV cache block transfers across the KVBM tiered
//! storage hierarchy: G1 (GPU HBM), G2 (pinned DRAM), G3 (NVMe).
//!
//! # Usage
//! ```bash
//! cargo run -p kvbm-engine --features bench --bin bench_engine -- \
//!     --devices 0 --page-sizes 32,64,128 --concurrency 1,2,4 --iterations 50
//! ```

use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{Result, ensure};
use clap::Parser;
use figment::Figment;
use figment::providers::{Env, Format, Serialized, Toml};
use serde::{Deserialize, Serialize};

use kvbm_physical::layout::{LayoutConfig, PhysicalLayout};
use kvbm_physical::manager::LayoutHandle;
use kvbm_physical::transfer::{
    BounceBuffer, NixlAgent, TransferCompleteNotification, TransferManager, TransferOptions,
};

// ─── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "bench_engine", about = "KVBM transfer bandwidth benchmark")]
struct Cli {
    /// GPU device IDs (comma-separated)
    #[arg(long, value_delimiter = ',', default_value = "0")]
    devices: Vec<u32>,

    /// Tokens-per-block values to sweep
    #[arg(long, value_delimiter = ',', default_values_t = vec![32, 64, 128, 256])]
    page_sizes: Vec<usize>,

    /// Concurrency levels to sweep
    #[arg(long, value_delimiter = ',', default_values_t = vec![1, 2, 4, 8])]
    concurrency: Vec<usize>,

    /// Blocks per transfer batch
    #[arg(long, default_value_t = 8)]
    blocks_per_batch: usize,

    /// Total blocks per pool (must be >= max_concurrency * blocks_per_batch * 2)
    #[arg(long, default_value_t = 128)]
    num_blocks: usize,

    /// Number of KV-cache layers
    #[arg(long, default_value_t = 24)]
    num_layers: usize,

    /// Inner dimension (hidden_dim / tp_size)
    #[arg(long, default_value_t = 4096)]
    inner_dim: usize,

    /// Bounce buffer block counts to sweep (tail blocks of G2 used as bounce for staged G1↔G3)
    #[arg(long, value_delimiter = ',', default_values_t = vec![2, 4, 8])]
    bounce_blocks: Vec<usize>,

    /// Warmup iterations
    #[arg(long, default_value_t = 5)]
    warmup: usize,

    /// Measurement iterations per test
    #[arg(long, default_value_t = 50)]
    iterations: usize,

    /// Disk path for G3 layouts (default: tempdir)
    #[arg(long)]
    disk_path: Option<PathBuf>,

    /// Skip G3/disk tests
    #[arg(long)]
    skip_disk: bool,

    /// Skip GDS tests
    #[arg(long)]
    skip_gds: bool,

    /// Run only isolated (phase 1) tests
    #[arg(long)]
    isolated_only: bool,

    /// Run only bidirectional (phase 2) tests
    #[arg(long)]
    bidir_only: bool,

    /// Base directory for output (default: current directory)
    #[arg(long, short)]
    output: Option<PathBuf>,

    /// Optional TOML config file (overridden by CLI args)
    #[arg(long)]
    config: Option<PathBuf>,
}

// ─── Config ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchConfig {
    devices: Vec<u32>,
    page_sizes: Vec<usize>,
    concurrency: Vec<usize>,
    blocks_per_batch: usize,
    num_blocks: usize,
    num_layers: usize,
    inner_dim: usize,
    bounce_blocks: Vec<usize>,
    warmup: usize,
    iterations: usize,
    disk_path: Option<PathBuf>,
    skip_disk: bool,
    skip_gds: bool,
    isolated_only: bool,
    bidir_only: bool,
    output: Option<PathBuf>,
}

impl From<Cli> for BenchConfig {
    fn from(cli: Cli) -> Self {
        Self {
            devices: cli.devices,
            page_sizes: cli.page_sizes,
            concurrency: cli.concurrency,
            blocks_per_batch: cli.blocks_per_batch,
            num_blocks: cli.num_blocks,
            num_layers: cli.num_layers,
            inner_dim: cli.inner_dim,
            bounce_blocks: cli.bounce_blocks,
            warmup: cli.warmup,
            iterations: cli.iterations,
            disk_path: cli.disk_path,
            skip_disk: cli.skip_disk,
            skip_gds: cli.skip_gds,
            isolated_only: cli.isolated_only,
            bidir_only: cli.bidir_only,
            output: cli.output,
        }
    }
}

fn build_config(cli: Cli) -> Result<BenchConfig> {
    let cli_config = BenchConfig::from(cli);

    // Check for TOML config file from environment
    let config_path: Option<PathBuf> = std::env::var("KVBM_BENCH_CONFIG").ok().map(PathBuf::from);

    let mut figment = Figment::new().merge(Serialized::defaults(&cli_config));

    if let Some(path) = config_path {
        figment = figment.merge(Toml::file(path));
    }

    figment = figment
        .merge(Env::prefixed("KVBM_BENCH_"))
        .merge(Serialized::defaults(&cli_config)); // CLI wins

    Ok(figment.extract()?)
}

// ─── Results ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
struct LatencyStats {
    min_us: f64,
    max_us: f64,
    mean_us: f64,
    p50_us: f64,
    p95_us: f64,
    p99_us: f64,
}

impl LatencyStats {
    fn from_durations(mut durations: Vec<Duration>) -> Self {
        durations.sort();
        let n = durations.len();
        let sum: Duration = durations.iter().sum();

        Self {
            min_us: durations[0].as_secs_f64() * 1e6,
            max_us: durations[n - 1].as_secs_f64() * 1e6,
            mean_us: sum.as_secs_f64() * 1e6 / n as f64,
            p50_us: durations[n / 2].as_secs_f64() * 1e6,
            p95_us: durations[(n as f64 * 0.95) as usize].as_secs_f64() * 1e6,
            p99_us: durations[(n as f64 * 0.99) as usize].as_secs_f64() * 1e6,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct BenchResult {
    test: String,
    device_id: u32,
    page_size: usize,
    blocks_per_batch: usize,
    concurrency: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    bounce_blocks: Option<usize>,
    bytes_per_iter: usize,
    iterations: usize,
    latency_us: LatencyStats,
    bandwidth_gbs: f64,
    aggregate_bandwidth_gbs: f64,
}

// ─── Layout Helpers ────────────────────────────────────────────────────────────

struct TierHandles {
    g1: LayoutHandle,
    g2: LayoutHandle,
    g3: Option<LayoutHandle>,
}

fn create_layouts(
    manager: &TransferManager,
    agent: &NixlAgent,
    device_id: u32,
    page_size: usize,
    config: &BenchConfig,
) -> Result<TierHandles> {
    let layout_config = LayoutConfig::builder()
        .num_blocks(config.num_blocks)
        .num_layers(config.num_layers)
        .outer_dim(2) // K + V
        .page_size(page_size)
        .inner_dim(config.inner_dim)
        .dtype_width_bytes(2) // fp16
        .build()?;

    let g1 = PhysicalLayout::builder(agent.clone())
        .with_config(layout_config.clone())
        .fully_contiguous()
        .allocate_device(device_id)
        .build()?;
    let g1_handle = manager.register_layout(g1)?;

    let g2 = PhysicalLayout::builder(agent.clone())
        .with_config(layout_config.clone())
        .fully_contiguous()
        .allocate_pinned(Some(device_id))
        .build()?;
    let g2_handle = manager.register_layout(g2)?;

    let g3_handle = if !config.skip_disk {
        let g3 = PhysicalLayout::builder(agent.clone())
            .with_config(layout_config)
            .fully_contiguous()
            .allocate_disk(config.disk_path.clone())
            .build()?;
        Some(manager.register_layout(g3)?)
    } else {
        None
    };

    Ok(TierHandles {
        g1: g1_handle,
        g2: g2_handle,
        g3: g3_handle,
    })
}

// ─── Transfer Execution ────────────────────────────────────────────────────────

async fn run_concurrent_batch(
    manager: &TransferManager,
    src: LayoutHandle,
    dst: LayoutHandle,
    blocks_per_batch: usize,
    concurrency: usize,
    options_fn: &dyn Fn() -> TransferOptions,
) -> Result<()> {
    let mut notifications: Vec<TransferCompleteNotification> = Vec::with_capacity(concurrency);
    for slot in 0..concurrency {
        let offset = slot * blocks_per_batch;
        let block_ids: Vec<usize> = (offset..offset + blocks_per_batch).collect();
        let notif = manager.execute_transfer(src, &block_ids, dst, &block_ids, options_fn())?;
        notifications.push(notif);
    }
    for notif in notifications {
        notif.await?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn bench_transfers(
    manager: &TransferManager,
    src: LayoutHandle,
    dst: LayoutHandle,
    blocks_per_batch: usize,
    concurrency: usize,
    warmup: usize,
    iterations: usize,
    options_fn: impl Fn() -> TransferOptions,
) -> Result<Vec<Duration>> {
    // Warmup
    for _ in 0..warmup {
        run_concurrent_batch(
            manager,
            src,
            dst,
            blocks_per_batch,
            concurrency,
            &options_fn,
        )
        .await?;
    }

    // Measure
    let mut latencies = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        run_concurrent_batch(
            manager,
            src,
            dst,
            blocks_per_batch,
            concurrency,
            &options_fn,
        )
        .await?;
        latencies.push(start.elapsed());
    }
    Ok(latencies)
}

fn compute_bytes_per_block(config: &BenchConfig, page_size: usize) -> usize {
    config.num_layers * 2 * page_size * config.inner_dim * 2
}

fn make_result(
    test: &str,
    device_id: u32,
    page_size: usize,
    concurrency: usize,
    bounce_blocks: Option<usize>,
    config: &BenchConfig,
    latencies: Vec<Duration>,
) -> BenchResult {
    let bytes_per_block = compute_bytes_per_block(config, page_size);
    let bytes_per_iter = bytes_per_block * config.blocks_per_batch * concurrency;
    let stats = LatencyStats::from_durations(latencies);
    let bandwidth_gbs = bytes_per_iter as f64 / (stats.mean_us * 1e3); // bytes / ns = GB/s
    let aggregate_bandwidth_gbs = bandwidth_gbs;

    BenchResult {
        test: test.to_string(),
        device_id,
        page_size,
        blocks_per_batch: config.blocks_per_batch,
        concurrency,
        bounce_blocks,
        bytes_per_iter,
        iterations: config.iterations,
        latency_us: stats,
        bandwidth_gbs,
        aggregate_bandwidth_gbs,
    }
}

fn print_result_stderr(r: &BenchResult) {
    eprintln!(
        "[GPU {}] {} | page={} conc={}{} | {:.1} GB/s (agg) | p50={:.0}us p99={:.0}us",
        r.device_id,
        r.test,
        r.page_size,
        r.concurrency,
        r.bounce_blocks
            .map(|b| format!(" bounce={b}"))
            .unwrap_or_default(),
        r.aggregate_bandwidth_gbs,
        r.latency_us.p50_us,
        r.latency_us.p99_us,
    );
}

// ─── Phase 1: Isolated Transfers ───────────────────────────────────────────────

async fn run_isolated_benchmarks(
    device_id: u32,
    manager: &TransferManager,
    agent: &NixlAgent,
    config: &BenchConfig,
) -> Result<Vec<BenchResult>> {
    let mut results = Vec::new();

    for &page_size in &config.page_sizes {
        eprintln!("[GPU {device_id}] Allocating layouts for page_size={page_size}...");
        let handles = create_layouts(manager, agent, device_id, page_size, config)?;

        for &conc in &config.concurrency {
            // G1→G2 (D2H offload)
            let latencies = bench_transfers(
                manager,
                handles.g1,
                handles.g2,
                config.blocks_per_batch,
                conc,
                config.warmup,
                config.iterations,
                TransferOptions::default,
            )
            .await?;
            let r = make_result(
                "g1_to_g2", device_id, page_size, conc, None, config, latencies,
            );
            print_result_stderr(&r);
            results.push(r);

            // G2→G1 (H2D onboard)
            let latencies = bench_transfers(
                manager,
                handles.g2,
                handles.g1,
                config.blocks_per_batch,
                conc,
                config.warmup,
                config.iterations,
                TransferOptions::default,
            )
            .await?;
            let r = make_result(
                "g2_to_g1", device_id, page_size, conc, None, config, latencies,
            );
            print_result_stderr(&r);
            results.push(r);

            // G2↔G3 tests (if disk enabled)
            if let Some(g3) = handles.g3 {
                // G2→G3 (host→disk)
                let latencies = bench_transfers(
                    manager,
                    handles.g2,
                    g3,
                    config.blocks_per_batch,
                    conc,
                    config.warmup,
                    config.iterations,
                    TransferOptions::default,
                )
                .await?;
                let r = make_result(
                    "g2_to_g3", device_id, page_size, conc, None, config, latencies,
                );
                print_result_stderr(&r);
                results.push(r);

                // G3→G2 (disk→host)
                let latencies = bench_transfers(
                    manager,
                    g3,
                    handles.g2,
                    config.blocks_per_batch,
                    conc,
                    config.warmup,
                    config.iterations,
                    TransferOptions::default,
                )
                .await?;
                let r = make_result(
                    "g3_to_g2", device_id, page_size, conc, None, config, latencies,
                );
                print_result_stderr(&r);
                results.push(r);
            }
        }

        // Bounce buffer sweep for staged G1↔G3
        // G2 IS the bounce buffer — use tail blocks of the G2 layout so they
        // don't overlap with the transfer block range [0..max_conc*bpb).
        if let Some(g3) = handles.g3 {
            for &num_bounce in &config.bounce_blocks {
                eprintln!(
                    "[GPU {device_id}] Bounce buffer sweep: page_size={page_size} bounce_blocks={num_bounce}"
                );
                let bounce_start = config.num_blocks - num_bounce;
                let bounce_block_ids: Vec<usize> = (bounce_start..config.num_blocks).collect();

                for &conc in &config.concurrency {
                    let bounce = BounceBuffer::from_handle(handles.g2, bounce_block_ids.clone());

                    // G1→G3 staged (via bounce)
                    let latencies = bench_transfers(
                        manager,
                        handles.g1,
                        g3,
                        config.blocks_per_batch,
                        conc,
                        config.warmup,
                        config.iterations,
                        || {
                            TransferOptions::builder()
                                .bounce_buffer(bounce.clone())
                                .build()
                                .unwrap()
                        },
                    )
                    .await?;
                    let r = make_result(
                        "g1_to_g3_staged",
                        device_id,
                        page_size,
                        conc,
                        Some(num_bounce),
                        config,
                        latencies,
                    );
                    print_result_stderr(&r);
                    results.push(r);

                    // G3→G1 staged (via bounce)
                    let bounce = BounceBuffer::from_handle(handles.g2, bounce_block_ids.clone());
                    let latencies = bench_transfers(
                        manager,
                        g3,
                        handles.g1,
                        config.blocks_per_batch,
                        conc,
                        config.warmup,
                        config.iterations,
                        || {
                            TransferOptions::builder()
                                .bounce_buffer(bounce.clone())
                                .build()
                                .unwrap()
                        },
                    )
                    .await?;
                    let r = make_result(
                        "g3_to_g1_staged",
                        device_id,
                        page_size,
                        conc,
                        Some(num_bounce),
                        config,
                        latencies,
                    );
                    print_result_stderr(&r);
                    results.push(r);
                }
            }

            // GDS direct tests (G1↔G3 without bounce)
            if !config.skip_gds {
                for &conc in &config.concurrency {
                    // G1→G3 direct (GDS)
                    match bench_transfers(
                        manager,
                        handles.g1,
                        g3,
                        config.blocks_per_batch,
                        conc,
                        config.warmup,
                        config.iterations,
                        TransferOptions::default,
                    )
                    .await
                    {
                        Ok(latencies) => {
                            let r = make_result(
                                "g1_to_g3_gds",
                                device_id,
                                page_size,
                                conc,
                                None,
                                config,
                                latencies,
                            );
                            print_result_stderr(&r);
                            results.push(r);
                        }
                        Err(e) => {
                            eprintln!(
                                "[GPU {device_id}] GDS g1_to_g3 failed (GDS may not be available): {e}"
                            );
                        }
                    }

                    // G3→G1 direct (GDS)
                    match bench_transfers(
                        manager,
                        g3,
                        handles.g1,
                        config.blocks_per_batch,
                        conc,
                        config.warmup,
                        config.iterations,
                        TransferOptions::default,
                    )
                    .await
                    {
                        Ok(latencies) => {
                            let r = make_result(
                                "g3_to_g1_gds",
                                device_id,
                                page_size,
                                conc,
                                None,
                                config,
                                latencies,
                            );
                            print_result_stderr(&r);
                            results.push(r);
                        }
                        Err(e) => {
                            eprintln!(
                                "[GPU {device_id}] GDS g3_to_g1 failed (GDS may not be available): {e}"
                            );
                        }
                    }
                }
            }
        }
    }

    Ok(results)
}

// ─── Phase 2: Bidirectional Contention ─────────────────────────────────────────

async fn run_bidir_benchmarks(
    device_id: u32,
    manager: &TransferManager,
    agent: &NixlAgent,
    config: &BenchConfig,
) -> Result<Vec<BenchResult>> {
    let mut results = Vec::new();

    // Use limited concurrency for bidir to keep block counts reasonable
    let bidir_concurrencies: Vec<usize> = config
        .concurrency
        .iter()
        .copied()
        .filter(|&c| c <= 4)
        .collect();

    for &page_size in &config.page_sizes {
        eprintln!("[GPU {device_id}] Bidirectional tests for page_size={page_size}...");
        let handles = create_layouts(manager, agent, device_id, page_size, config)?;

        for &conc in &bidir_concurrencies {
            // Need separate block ranges for D2H and H2D to avoid contention
            let d2h_blocks_per_slot = config.blocks_per_batch;
            let h2d_blocks_per_slot = config.blocks_per_batch;
            let total_blocks_needed = (d2h_blocks_per_slot + h2d_blocks_per_slot) * conc;

            if total_blocks_needed > config.num_blocks {
                eprintln!(
                    "[GPU {device_id}] Skipping bidir page_size={page_size} conc={conc}: \
                     need {total_blocks_needed} blocks but only have {}",
                    config.num_blocks
                );
                continue;
            }

            // Warmup
            for _ in 0..config.warmup {
                run_bidir_batch(manager, &handles, config, conc).await?;
            }

            // Measure
            let mut d2h_latencies = Vec::with_capacity(config.iterations);
            let mut h2d_latencies = Vec::with_capacity(config.iterations);

            for _ in 0..config.iterations {
                let (d2h_dur, h2d_dur) = run_bidir_batch(manager, &handles, config, conc).await?;
                d2h_latencies.push(d2h_dur);
                h2d_latencies.push(h2d_dur);
            }

            // D2H direction result
            let r = make_result(
                "bidir_g1_to_g2",
                device_id,
                page_size,
                conc,
                None,
                config,
                d2h_latencies,
            );
            print_result_stderr(&r);
            results.push(r);

            // H2D direction result
            let r = make_result(
                "bidir_g2_to_g1",
                device_id,
                page_size,
                conc,
                None,
                config,
                h2d_latencies,
            );
            print_result_stderr(&r);
            results.push(r);
        }
    }

    Ok(results)
}

async fn run_bidir_batch(
    manager: &TransferManager,
    handles: &TierHandles,
    config: &BenchConfig,
    concurrency: usize,
) -> Result<(Duration, Duration)> {
    let bpb = config.blocks_per_batch;

    // D2H uses blocks [0..conc*bpb), H2D uses blocks [conc*bpb..2*conc*bpb)
    let d2h_start = Instant::now();
    let h2d_start = Instant::now();

    // Launch all D2H and H2D transfers concurrently
    let mut notifications = Vec::new();
    // D2H: G1→G2
    for slot in 0..concurrency {
        let offset = slot * bpb;
        let block_ids: Vec<usize> = (offset..offset + bpb).collect();
        let notif = manager.execute_transfer(
            handles.g1,
            &block_ids,
            handles.g2,
            &block_ids,
            TransferOptions::default(),
        )?;
        notifications.push(("d2h", notif));
    }
    // H2D: G2→G1 (different block range)
    for slot in 0..concurrency {
        let offset = (concurrency + slot) * bpb;
        let block_ids: Vec<usize> = (offset..offset + bpb).collect();
        let notif = manager.execute_transfer(
            handles.g2,
            &block_ids,
            handles.g1,
            &block_ids,
            TransferOptions::default(),
        )?;
        notifications.push(("h2d", notif));
    }

    let mut d2h_done = false;
    let mut h2d_done = false;
    let mut d2h_elapsed = Duration::ZERO;
    let mut h2d_elapsed = Duration::ZERO;

    for (direction, notif) in notifications {
        notif.await?;
        match direction {
            "d2h" if !d2h_done => {
                // Only record when last D2H completes - but we don't track per-notification.
                // Simplification: record after all d2h notifications resolve
            }
            "h2d" if !h2d_done => {}
            _ => {}
        }
        // Track last completion per direction
        if direction == "d2h" {
            d2h_elapsed = d2h_start.elapsed();
            d2h_done = true;
        } else {
            h2d_elapsed = h2d_start.elapsed();
            h2d_done = true;
        }
    }

    Ok((d2h_elapsed, h2d_elapsed))
}

// ─── Per-Device Worker ─────────────────────────────────────────────────────────

async fn run_device_benchmarks(device_id: u32, config: &BenchConfig) -> Result<Vec<BenchResult>> {
    eprintln!("[GPU {device_id}] Initializing NixlAgent and TransferManager...");

    let agent_name = format!("bench-gpu-{device_id}");

    // Create NixlAgent with available backends
    let mut agent = NixlAgent::new(&agent_name)?;
    // Try POSIX for disk transfers
    if !config.skip_disk && agent.add_backend("POSIX").is_err() {
        eprintln!("[GPU {device_id}] POSIX backend unavailable");
    }
    // Try GDS for direct GPU↔disk
    if !config.skip_gds && !config.skip_disk && agent.add_backend("GDS_MT").is_err() {
        eprintln!("[GPU {device_id}] GDS_MT backend unavailable");
    }

    let manager = TransferManager::builder()
        .nixl_agent(agent.clone())
        .cuda_device_id(device_id as usize)
        .build()?;

    let mut all_results = Vec::new();

    if !config.bidir_only {
        eprintln!("[GPU {device_id}] === Phase 1: Isolated Transfers ===");
        let isolated = run_isolated_benchmarks(device_id, &manager, &agent, config).await?;
        all_results.extend(isolated);
    }

    if !config.isolated_only {
        eprintln!("[GPU {device_id}] === Phase 2: Bidirectional Contention ===");
        let bidir = run_bidir_benchmarks(device_id, &manager, &agent, config).await?;
        all_results.extend(bidir);
    }

    Ok(all_results)
}

fn spawn_device_worker(device_id: u32, config: BenchConfig, tx: mpsc::Sender<Vec<BenchResult>>) {
    std::thread::Builder::new()
        .name(format!("bench-gpu-{device_id}"))
        .spawn(move || {
            // Try to pin to the device's NUMA-subdivided CPU set
            if let Some(cpus) = dynamo_memory::numa::get_device_cpu_set(device_id) {
                eprintln!(
                    "[GPU {device_id}] Pinning to CPUs: {}",
                    format_cpu_set(&cpus)
                );
                pin_thread_to_cpus(&cpus);
            } else {
                // Fallback: pin to device's NUMA node
                let node = dynamo_memory::numa::get_device_numa_node(device_id);
                if !node.is_unknown() {
                    eprintln!("[GPU {device_id}] Pinning to NUMA node {node}");
                    let _ = dynamo_memory::numa::pin_thread_to_numa_node(node);
                } else {
                    eprintln!("[GPU {device_id}] No NUMA pinning (node unknown)");
                }
            }

            // Build a tokio runtime for this device
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .thread_name(format!("bench-gpu-{device_id}-tokio"))
                .build()
                .expect("failed to build tokio runtime");

            let results = rt.block_on(async { run_device_benchmarks(device_id, &config).await });

            match results {
                Ok(r) => {
                    tx.send(r).ok();
                }
                Err(e) => {
                    eprintln!("[GPU {device_id}] ERROR: {e:#}");
                    tx.send(Vec::new()).ok();
                }
            }
        })
        .expect("failed to spawn device worker thread");
}

fn pin_thread_to_cpus(cpus: &[usize]) {
    unsafe {
        let mut cpu_set: libc::cpu_set_t = std::mem::zeroed();
        for &cpu in cpus {
            libc::CPU_SET(cpu, &mut cpu_set);
        }
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &cpu_set);
    }
}

fn format_cpu_set(cpus: &[usize]) -> String {
    if cpus.is_empty() {
        return String::new();
    }
    // Compress into ranges: [0,1,2,3,8,9,10] -> "0-3,8-10"
    let mut parts = Vec::new();
    let mut start = cpus[0];
    let mut end = cpus[0];

    for &cpu in &cpus[1..] {
        if cpu == end + 1 {
            end = cpu;
        } else {
            if start == end {
                parts.push(format!("{start}"));
            } else {
                parts.push(format!("{start}-{end}"));
            }
            start = cpu;
            end = cpu;
        }
    }
    if start == end {
        parts.push(format!("{start}"));
    } else {
        parts.push(format!("{start}-{end}"));
    }
    parts.join(",")
}

// ─── Validation ────────────────────────────────────────────────────────────────

fn validate_config(config: &BenchConfig) -> Result<()> {
    let max_conc = config.concurrency.iter().max().copied().unwrap_or(1);
    let max_bounce = config.bounce_blocks.iter().max().copied().unwrap_or(0);

    // For bidir tests we need 2x the blocks (separate ranges for each direction)
    let multiplier = if config.isolated_only { 1 } else { 2 };
    let transfer_blocks = max_conc * config.blocks_per_batch * multiplier;

    // Bounce blocks come from the tail of G2, so they must not overlap with
    // the transfer block range [0..transfer_blocks).
    let min_blocks = transfer_blocks + max_bounce;

    ensure!(
        config.num_blocks >= min_blocks,
        "num_blocks ({}) must be >= max_concurrency ({}) * blocks_per_batch ({}) * {} + max_bounce ({}) = {}",
        config.num_blocks,
        max_conc,
        config.blocks_per_batch,
        multiplier,
        max_bounce,
        min_blocks,
    );

    ensure!(
        !config.devices.is_empty(),
        "must specify at least one device"
    );
    ensure!(
        !config.page_sizes.is_empty(),
        "must specify at least one page_size"
    );
    ensure!(
        !config.concurrency.is_empty(),
        "must specify at least one concurrency level"
    );
    ensure!(config.iterations > 0, "iterations must be > 0");

    // Validate disk path if G3 tests enabled
    if let Some(ref path) = config.disk_path
        && !config.skip_disk
    {
        ensure!(
            path.exists() || path.parent().is_some_and(|p| p.exists()),
            "disk path {} does not exist",
            path.display()
        );
    }

    Ok(())
}

// ─── Main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    // Initialize tracing for debug output
    tracing_subscriber_init();

    let cli = Cli::parse();
    let config = build_config(cli)?;
    validate_config(&config)?;

    eprintln!("KVBM Engine Benchmark");
    eprintln!("  Devices: {:?}", config.devices);
    eprintln!("  Page sizes: {:?}", config.page_sizes);
    eprintln!("  Concurrency: {:?}", config.concurrency);
    eprintln!("  Blocks per batch: {}", config.blocks_per_batch);
    eprintln!("  Total blocks per pool: {}", config.num_blocks);
    eprintln!(
        "  Layers: {}, Inner dim: {}",
        config.num_layers, config.inner_dim
    );
    eprintln!(
        "  Warmup: {}, Iterations: {}",
        config.warmup, config.iterations
    );
    eprintln!(
        "  Disk: {}",
        if config.skip_disk {
            "disabled"
        } else {
            "enabled"
        }
    );
    eprintln!(
        "  GDS: {}",
        if config.skip_gds {
            "disabled"
        } else {
            "enabled"
        }
    );
    eprintln!();

    let (tx, rx) = mpsc::channel();

    // Spawn a worker thread per device
    for &device_id in &config.devices {
        spawn_device_worker(device_id, config.clone(), tx.clone());
    }
    drop(tx); // Close sender so rx iterator ends when all workers finish

    // Collect results from all devices
    let mut all_results: Vec<BenchResult> = Vec::new();
    for device_results in rx {
        all_results.extend(device_results);
    }

    // Build timestamped output directory: <cwd>/YYMMDD-HH:MM:SS-bench-engine/
    let now = chrono::Local::now();
    let dir_name = now.format("%y%m%d-%H:%M:%S-bench-engine").to_string();
    let out_dir = if let Some(ref base) = config.output {
        base.join(&dir_name)
    } else {
        PathBuf::from(&dir_name)
    };
    std::fs::create_dir_all(&out_dir)?;

    // Write JSON Lines results
    let json_output: String = all_results
        .iter()
        .map(|r| serde_json::to_string(r).unwrap())
        .collect::<Vec<_>>()
        .join("\n");

    let jsonl_path = out_dir.join(format!("{dir_name}.jsonl"));
    std::fs::write(&jsonl_path, &json_output)?;

    // Copy the viewer HTML into the output directory
    let viewer_html = include_str!("../scripts/bench_viewer.html");
    let viewer_path = out_dir.join(format!("{dir_name}.html"));
    std::fs::write(&viewer_path, viewer_html)?;

    eprintln!(
        "\nBenchmark complete. {} results collected.",
        all_results.len()
    );
    eprintln!("Results directory: {}", out_dir.display());
    eprintln!("  {}", jsonl_path.display());
    eprintln!("  {}", viewer_path.display());
    Ok(())
}

fn tracing_subscriber_init() {
    use std::env;
    if env::var("RUST_LOG").is_err() {
        // SAFETY: Called at program start before any threads are spawned.
        unsafe { env::set_var("RUST_LOG", "error") };
    }
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
}
