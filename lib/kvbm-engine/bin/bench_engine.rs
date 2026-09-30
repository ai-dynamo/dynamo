// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KVBM transfer bandwidth benchmark with full Leader+Worker architecture.
//!
//! Uses production-fidelity InstanceLeader, VeloWorkerService/Client, SpmdParallelWorkers,
//! and optionally OffloadEngine pipelines. Each worker runs on a NUMA-pinned thread with
//! its own tokio runtime and NixlAgent.
//!
//! # Usage
//! ```bash
//! # Direct transfer benchmark:
//! cargo run -p kvbm-engine --features bench --bin bench_engine -- \
//!     --devices 0 --page-sizes 32,64 --concurrency 1,2 --iterations 10 --skip-disk --skip-gds
//!
//! # With offload pipeline:
//! cargo run -p kvbm-engine --features bench --bin bench_engine -- \
//!     --devices 0 --page-sizes 64 --concurrency 1 --iterations 10 --skip-disk --skip-gds \
//!     --offload --offload-batch-sizes 8,32 --offload-concurrency 1,2
//!
//! # Multi-GPU:
//! cargo run -p kvbm-engine --features bench --bin bench_engine -- \
//!     --devices 0,1 --page-sizes 128 --concurrency 1,2,4 --iterations 50
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Result, ensure};
use clap::Parser;
use figment::Figment;
use figment::providers::{Env, Format, Serialized, Toml};
use serde::{Deserialize, Serialize};

use kvbm_engine::{
    BlockId, G1, G2, G3, LogicalLayoutHandle,
    leader::InstanceLeader,
    offload::{ExternalBlock, OffloadEngine, PipelineBuilder, PresenceFilter, SourceBlocks},
    testing::{
        TestManagerBuilder, TestRegistryBuilder, create_messenger_tcp,
        managers::populate_manager_with_blocks, token_blocks,
    },
    worker::{DirectWorker, Worker, WorkerTransfers},
};
use kvbm_logical::blocks::BlockRegistry;
use kvbm_logical::manager::BlockManager;
use kvbm_physical::layout::{LayoutConfig, PhysicalLayout};
use kvbm_physical::transfer::{NixlAgent, TransferManager, TransferOptions};

// ─── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "bench_engine",
    about = "KVBM transfer bandwidth benchmark (leader+worker architecture)"
)]
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

    /// Enable offload pipeline benchmarks (phase 3)
    #[arg(long)]
    offload: bool,

    /// Offload pipeline batch sizes to sweep
    #[arg(long, value_delimiter = ',', default_values_t = vec![8, 16, 32, 64])]
    offload_batch_sizes: Vec<usize>,

    /// Max concurrent transfers for offload pipeline
    #[arg(long, value_delimiter = ',', default_values_t = vec![1, 2, 4])]
    offload_concurrency: Vec<usize>,

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
    offload: bool,
    offload_batch_sizes: Vec<usize>,
    offload_concurrency: Vec<usize>,
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
            offload: cli.offload,
            offload_batch_sizes: cli.offload_batch_sizes,
            offload_concurrency: cli.offload_concurrency,
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
    let num_devices = config.devices.len();
    let aggregate_bandwidth_gbs = bandwidth_gbs * num_devices as f64;

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
        "[GPU {}] {} | page={} conc={}{} | {:.1} GB/s (per-dev) {:.1} GB/s (agg) | p50={:.0}us p99={:.0}us",
        r.device_id,
        r.test,
        r.page_size,
        r.concurrency,
        r.bounce_blocks
            .map(|b| format!(" bounce={b}"))
            .unwrap_or_default(),
        r.bandwidth_gbs,
        r.aggregate_bandwidth_gbs,
        r.latency_us.p50_us,
        r.latency_us.p99_us,
    );
}

// ─── Worker Thread Infrastructure ──────────────────────────────────────────────

struct WorkerHandle {
    worker: Arc<DirectWorker>,
    join_handle: std::thread::JoinHandle<()>,
    shutdown_tx: tokio::sync::oneshot::Sender<()>,
}

/// Spawn a long-lived worker thread for a single GPU device.
///
/// The worker thread:
/// 1. Pins to the device's NUMA node
/// 2. Creates its own tokio runtime (2 worker threads)
/// 3. Creates NixlAgent, TransferManager
/// 4. Creates G1/G2/G3 PhysicalLayouts and registers them (NUMA-local allocations)
/// 5. Builds DirectWorker and sends Arc back to leader
/// 6. Waits on shutdown signal (keeps runtime alive for transfers)
fn spawn_worker_thread(
    device_id: u32,
    page_size: usize,
    config: &BenchConfig,
) -> Result<WorkerHandle> {
    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();

    let num_blocks = config.num_blocks;
    let num_layers = config.num_layers;
    let inner_dim = config.inner_dim;
    let skip_disk = config.skip_disk;
    let skip_gds = config.skip_gds;
    let disk_path = config.disk_path.clone();

    let join_handle = std::thread::Builder::new()
        .name(format!("bench-gpu-{device_id}"))
        .spawn(move || {
            // Pin to device's NUMA node
            if let Some(cpus) = dynamo_memory::numa::get_device_cpu_set(device_id) {
                eprintln!(
                    "[GPU {device_id}] Worker pinned to CPUs: {}",
                    format_cpu_set(&cpus)
                );
                pin_thread_to_cpus(&cpus);
            } else {
                if let Some(node) = dynamo_memory::numa::get_device_numa_node(device_id) {
                    eprintln!("[GPU {device_id}] Worker pinned to NUMA node {node}");
                    let _ = dynamo_memory::numa::pin_thread_to_numa_node(node);
                } else {
                    eprintln!("[GPU {device_id}] No NUMA pinning (node unknown)");
                }
            }

            // Build tokio runtime on this NUMA-pinned thread
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .thread_name(format!("bench-gpu-{device_id}-tokio"))
                .build()
                .expect("failed to build tokio runtime");

            let result = rt.block_on(async {
                // Create a local EventManager for this worker's transfer notifications
                let event_system = Arc::new(velo::EventManager::local());

                // Create NixlAgent with available backends
                let agent_name = format!("bench-gpu-{device_id}");
                let mut agent = NixlAgent::new(&agent_name)?;
                if !skip_disk && agent.add_backend("POSIX").is_err() {
                    eprintln!("[GPU {device_id}] POSIX backend unavailable");
                }
                if !skip_gds && !skip_disk && agent.add_backend("GDS_MT").is_err() {
                    eprintln!("[GPU {device_id}] GDS_MT backend unavailable");
                }

                // Create TransferManager
                let manager = TransferManager::builder()
                    .event_system(event_system)
                    .nixl_agent(agent.clone())
                    .cuda_device_id(device_id as usize)
                    .build()?;

                // Build layout config
                let layout_config = LayoutConfig::builder()
                    .num_blocks(num_blocks)
                    .num_layers(num_layers)
                    .outer_dim(2) // K + V
                    .page_size(page_size)
                    .inner_dim(inner_dim)
                    .dtype_width_bytes(2) // fp16
                    .build()?;

                // Allocate G1 (GPU device memory) — NUMA-local allocation
                let g1 = PhysicalLayout::builder(agent.clone())
                    .with_config(layout_config.clone())
                    .fully_contiguous()
                    .allocate_device(device_id)
                    .build()?;
                let g1_handle = manager.register_layout(g1)?;

                // Allocate G2 (pinned host memory) — NUMA-local allocation
                let g2 = PhysicalLayout::builder(agent.clone())
                    .with_config(layout_config.clone())
                    .fully_contiguous()
                    .allocate_pinned(Some(device_id))
                    .build()?;
                let g2_handle = manager.register_layout(g2)?;

                // Allocate G3 (disk) if enabled
                let g3_handle = if !skip_disk {
                    let g3 = PhysicalLayout::builder(agent.clone())
                        .with_config(layout_config)
                        .fully_contiguous()
                        .allocate_disk(disk_path)
                        .build()?;
                    Some(manager.register_layout(g3)?)
                } else {
                    None
                };

                // Build DirectWorker (PhysicalWorker)
                let mut worker_builder = DirectWorker::builder()
                    .manager(manager)
                    .g1_handle(g1_handle)
                    .g2_handle(g2_handle);
                if let Some(g3) = g3_handle {
                    worker_builder = worker_builder.g3_handle(g3);
                }
                let worker = Arc::new(worker_builder.build()?);

                ready_tx.send(Ok(worker.clone())).ok();

                // Keep runtime alive so TransferManager notification threads stay running
                let _ = shutdown_rx.await;

                Ok::<(), anyhow::Error>(())
            });

            if let Err(e) = result {
                ready_tx.send(Err(e)).ok();
            }
        })
        .expect("failed to spawn worker thread");

    // Wait for worker to be ready
    let worker = ready_rx
        .recv()
        .map_err(|_| anyhow::anyhow!("Worker thread died before sending ready signal"))??;

    Ok(WorkerHandle {
        worker,
        join_handle,
        shutdown_tx,
    })
}

// ─── BenchInstance: Full Leader+Worker Setup ───────────────────────────────────

struct BenchInstance {
    leader: InstanceLeader,
    #[allow(dead_code)]
    registry: BlockRegistry,
    g2_manager: Arc<BlockManager<G2>>,
    #[allow(dead_code)]
    g3_manager: Option<Arc<BlockManager<G3>>>,
    offload_engine: Option<OffloadEngine>,
    worker_handles: Vec<WorkerHandle>,
    config: BenchConfig,
    page_size: usize,
}

impl BenchInstance {
    /// Create a full leader+worker bench instance for a given page_size.
    ///
    /// One leader with SpmdParallelWorkers, N DirectWorkers (one per GPU device),
    /// each on a NUMA-pinned thread with its own tokio runtime and TransferManager.
    async fn new(config: BenchConfig, page_size: usize) -> Result<Self> {
        let num_devices = config.devices.len();
        eprintln!(
            "Setting up BenchInstance: page_size={page_size}, {} device(s)",
            num_devices
        );

        // Spawn worker threads (one per device)
        let mut worker_handles = Vec::with_capacity(num_devices);
        for &device_id in &config.devices {
            let handle = spawn_worker_thread(device_id, page_size, &config)?;
            eprintln!("[GPU {device_id}] Worker ready");
            worker_handles.push(handle);
        }

        // Collect DirectWorker references for the leader
        let worker_refs: Vec<Arc<dyn Worker>> = worker_handles
            .iter()
            .map(|wh| wh.worker.clone() as Arc<dyn Worker>)
            .collect();

        // Create leader Messenger (needed by InstanceLeader for event system)
        let leader_messenger = create_messenger_tcp().await?;

        // Build BlockRegistry and BlockManagers
        let registry = TestRegistryBuilder::new().build();
        let g2_manager = Arc::new(
            TestManagerBuilder::<G2>::new()
                .block_count(config.num_blocks)
                .block_size(page_size)
                .registry(registry.clone())
                .build(),
        );
        let g3_manager = if !config.skip_disk {
            Some(Arc::new(
                TestManagerBuilder::<G3>::new()
                    .block_count(config.num_blocks)
                    .block_size(page_size)
                    .registry(registry.clone())
                    .build(),
            ))
        } else {
            None
        };

        // Build InstanceLeader with direct worker references
        let mut leader_builder = InstanceLeader::builder()
            .messenger(leader_messenger)
            .registry(registry.clone())
            .g2_manager(g2_manager.clone())
            .workers(worker_refs);

        if let Some(ref g3m) = g3_manager {
            leader_builder = leader_builder.g3_manager(g3m.clone());
        }

        let leader = leader_builder.build()?;

        // Build OffloadEngine if requested
        let offload_engine = if config.offload {
            let mut engine_builder = OffloadEngine::builder(Arc::new(leader.clone()))
                .with_registry(Arc::new(registry.clone()))
                .with_g2_manager(g2_manager.clone())
                .with_runtime(tokio::runtime::Handle::current());

            if let Some(ref g3m) = g3_manager {
                engine_builder = engine_builder.with_g3_manager(g3m.clone());
            }

            // Configure G1→G2 pipeline with a pass-through presence filter
            let g1_to_g2_config = PipelineBuilder::<G1, G2>::new()
                .policy(Arc::new(PresenceFilter::<G1, G2>::new(Arc::new(
                    registry.clone(),
                ))))
                .batch_size(64)
                .max_concurrent_transfers(4)
                .build();
            engine_builder = engine_builder.with_g1_to_g2_pipeline(g1_to_g2_config);

            // Configure G2→G3 pipeline if disk enabled
            if g3_manager.is_some() {
                let g2_to_g3_config = PipelineBuilder::<G2, G3>::new()
                    .policy(Arc::new(PresenceFilter::<G2, G3>::new(Arc::new(
                        registry.clone(),
                    ))))
                    .batch_size(64)
                    .max_concurrent_transfers(4)
                    .build();
                engine_builder = engine_builder.with_g2_to_g3_pipeline(g2_to_g3_config);
            }

            Some(engine_builder.build()?)
        } else {
            None
        };

        Ok(Self {
            leader,
            registry,
            g2_manager,
            g3_manager,
            offload_engine,
            worker_handles,
            config,
            page_size,
        })
    }

    /// Run all benchmark phases and return results.
    async fn run_benchmarks(&self) -> Result<Vec<BenchResult>> {
        let mut results = Vec::new();

        if !self.config.bidir_only {
            eprintln!(
                "=== Phase 1: Isolated Transfers (page_size={}) ===",
                self.page_size
            );
            results.extend(self.bench_isolated_transfers().await?);
        }

        if !self.config.isolated_only {
            eprintln!(
                "=== Phase 2: Bidirectional Contention (page_size={}) ===",
                self.page_size
            );
            results.extend(self.bench_bidir_transfers().await?);
        }

        if self.config.offload && self.offload_engine.is_some() {
            eprintln!(
                "=== Phase 3: Offload Pipeline (page_size={}) ===",
                self.page_size
            );
            results.extend(self.bench_offload_pipeline().await?);
        }

        Ok(results)
    }

    // ─── Phase 1: Isolated Transfers ───────────────────────────────────────

    async fn bench_isolated_transfers(&self) -> Result<Vec<BenchResult>> {
        let mut results = Vec::new();
        let device_id = self.config.devices[0]; // Report results under first device
        let parallel_worker = self
            .leader
            .parallel_worker()
            .ok_or_else(|| anyhow::anyhow!("No parallel worker available"))?;

        for &conc in &self.config.concurrency {
            let bpb = self.config.blocks_per_batch;
            let block_ids: Arc<[BlockId]> =
                Arc::from((0..conc * bpb).map(|i| i as BlockId).collect::<Vec<_>>());

            // G1→G2 (D2H offload)
            let latencies = self
                .bench_transfer(
                    &*parallel_worker,
                    LogicalLayoutHandle::G1,
                    LogicalLayoutHandle::G2,
                    block_ids.clone(),
                    block_ids.clone(),
                )
                .await?;
            let r = make_result(
                "g1_to_g2",
                device_id,
                self.page_size,
                conc,
                None,
                &self.config,
                latencies,
            );
            print_result_stderr(&r);
            results.push(r);

            // G2→G1 (H2D onboard)
            let latencies = self
                .bench_transfer(
                    &*parallel_worker,
                    LogicalLayoutHandle::G2,
                    LogicalLayoutHandle::G1,
                    block_ids.clone(),
                    block_ids.clone(),
                )
                .await?;
            let r = make_result(
                "g2_to_g1",
                device_id,
                self.page_size,
                conc,
                None,
                &self.config,
                latencies,
            );
            print_result_stderr(&r);
            results.push(r);

            // G2↔G3 tests (if disk enabled)
            if !self.config.skip_disk {
                // G2→G3
                let latencies = self
                    .bench_transfer(
                        &*parallel_worker,
                        LogicalLayoutHandle::G2,
                        LogicalLayoutHandle::G3,
                        block_ids.clone(),
                        block_ids.clone(),
                    )
                    .await?;
                let r = make_result(
                    "g2_to_g3",
                    device_id,
                    self.page_size,
                    conc,
                    None,
                    &self.config,
                    latencies,
                );
                print_result_stderr(&r);
                results.push(r);

                // G3→G2
                let latencies = self
                    .bench_transfer(
                        &*parallel_worker,
                        LogicalLayoutHandle::G3,
                        LogicalLayoutHandle::G2,
                        block_ids.clone(),
                        block_ids.clone(),
                    )
                    .await?;
                let r = make_result(
                    "g3_to_g2",
                    device_id,
                    self.page_size,
                    conc,
                    None,
                    &self.config,
                    latencies,
                );
                print_result_stderr(&r);
                results.push(r);
            }
        }

        // G1↔G3 direct tests (GDS or bounce-buffer-free path)
        if !self.config.skip_disk {
            // GDS direct tests (G1↔G3 without bounce)
            if !self.config.skip_gds {
                for &conc in &self.config.concurrency {
                    let bpb = self.config.blocks_per_batch;
                    let block_ids: Arc<[BlockId]> =
                        Arc::from((0..conc * bpb).map(|i| i as BlockId).collect::<Vec<_>>());

                    // G1→G3 direct (GDS)
                    match self
                        .bench_transfer(
                            &*parallel_worker,
                            LogicalLayoutHandle::G1,
                            LogicalLayoutHandle::G3,
                            block_ids.clone(),
                            block_ids.clone(),
                        )
                        .await
                    {
                        Ok(latencies) => {
                            let r = make_result(
                                "g1_to_g3_gds",
                                device_id,
                                self.page_size,
                                conc,
                                None,
                                &self.config,
                                latencies,
                            );
                            print_result_stderr(&r);
                            results.push(r);
                        }
                        Err(e) => {
                            eprintln!("GDS g1_to_g3 failed (GDS may not be available): {e}");
                        }
                    }

                    // G3→G1 direct (GDS)
                    match self
                        .bench_transfer(
                            &*parallel_worker,
                            LogicalLayoutHandle::G3,
                            LogicalLayoutHandle::G1,
                            block_ids.clone(),
                            block_ids.clone(),
                        )
                        .await
                    {
                        Ok(latencies) => {
                            let r = make_result(
                                "g3_to_g1_gds",
                                device_id,
                                self.page_size,
                                conc,
                                None,
                                &self.config,
                                latencies,
                            );
                            print_result_stderr(&r);
                            results.push(r);
                        }
                        Err(e) => {
                            eprintln!("GDS g3_to_g1 failed (GDS may not be available): {e}");
                        }
                    }
                }
            }
        }

        Ok(results)
    }

    // ─── Phase 2: Bidirectional Contention ─────────────────────────────────

    async fn bench_bidir_transfers(&self) -> Result<Vec<BenchResult>> {
        let mut results = Vec::new();
        let device_id = self.config.devices[0];
        let parallel_worker = self
            .leader
            .parallel_worker()
            .ok_or_else(|| anyhow::anyhow!("No parallel worker available"))?;

        let bidir_concurrencies: Vec<usize> = self
            .config
            .concurrency
            .iter()
            .copied()
            .filter(|&c| c <= 4)
            .collect();

        for &conc in &bidir_concurrencies {
            let bpb = self.config.blocks_per_batch;
            let total_blocks_needed = 2 * conc * bpb;

            if total_blocks_needed > self.config.num_blocks {
                eprintln!(
                    "Skipping bidir page_size={} conc={conc}: need {total_blocks_needed} blocks but only have {}",
                    self.page_size, self.config.num_blocks
                );
                continue;
            }

            // D2H block range: [0..conc*bpb)
            let d2h_ids: Arc<[BlockId]> =
                Arc::from((0..conc * bpb).map(|i| i as BlockId).collect::<Vec<_>>());
            // H2D block range: [conc*bpb..2*conc*bpb)
            let h2d_ids: Arc<[BlockId]> = Arc::from(
                (conc * bpb..2 * conc * bpb)
                    .map(|i| i as BlockId)
                    .collect::<Vec<_>>(),
            );

            // Warmup
            for _ in 0..self.config.warmup {
                let d2h_notif = parallel_worker.execute_local_transfer(
                    LogicalLayoutHandle::G1,
                    LogicalLayoutHandle::G2,
                    d2h_ids.clone(),
                    d2h_ids.clone(),
                    TransferOptions::default(),
                )?;
                let h2d_notif = parallel_worker.execute_local_transfer(
                    LogicalLayoutHandle::G2,
                    LogicalLayoutHandle::G1,
                    h2d_ids.clone(),
                    h2d_ids.clone(),
                    TransferOptions::default(),
                )?;
                d2h_notif.await?;
                h2d_notif.await?;
            }

            // Measure
            let mut d2h_latencies = Vec::with_capacity(self.config.iterations);
            let mut h2d_latencies = Vec::with_capacity(self.config.iterations);

            for _ in 0..self.config.iterations {
                let start = Instant::now();

                let d2h_notif = parallel_worker.execute_local_transfer(
                    LogicalLayoutHandle::G1,
                    LogicalLayoutHandle::G2,
                    d2h_ids.clone(),
                    d2h_ids.clone(),
                    TransferOptions::default(),
                )?;
                let h2d_notif = parallel_worker.execute_local_transfer(
                    LogicalLayoutHandle::G2,
                    LogicalLayoutHandle::G1,
                    h2d_ids.clone(),
                    h2d_ids.clone(),
                    TransferOptions::default(),
                )?;

                d2h_notif.await?;
                let d2h_elapsed = start.elapsed();
                h2d_notif.await?;
                let h2d_elapsed = start.elapsed();

                d2h_latencies.push(d2h_elapsed);
                h2d_latencies.push(h2d_elapsed);
            }

            let r = make_result(
                "bidir_g1_to_g2",
                device_id,
                self.page_size,
                conc,
                None,
                &self.config,
                d2h_latencies,
            );
            print_result_stderr(&r);
            results.push(r);

            let r = make_result(
                "bidir_g2_to_g1",
                device_id,
                self.page_size,
                conc,
                None,
                &self.config,
                h2d_latencies,
            );
            print_result_stderr(&r);
            results.push(r);
        }

        Ok(results)
    }

    // ─── Phase 3: Offload Pipeline ─────────────────────────────────────────

    async fn bench_offload_pipeline(&self) -> Result<Vec<BenchResult>> {
        let mut results = Vec::new();
        let device_id = self.config.devices[0];
        let engine = self
            .offload_engine
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("OffloadEngine not configured"))?;

        // Populate G2 manager with test blocks so the registry has entries
        let token_seq =
            token_blocks::create_token_sequence(self.config.num_blocks, self.page_size, 0);
        let seq_hashes = populate_manager_with_blocks(&self.g2_manager, token_seq.blocks())?;

        for &batch_size in &self.config.offload_batch_sizes {
            if batch_size > self.config.num_blocks {
                eprintln!(
                    "Skipping offload batch_size={batch_size}: exceeds num_blocks={}",
                    self.config.num_blocks
                );
                continue;
            }

            for &conc in &self.config.offload_concurrency {
                eprintln!("Offload G1→G2 pipeline: batch_size={batch_size} concurrency={conc}");

                // Warmup
                for _ in 0..self.config.warmup {
                    let blocks: Vec<ExternalBlock<G1>> = (0..batch_size)
                        .map(|i| ExternalBlock::new(i as BlockId, seq_hashes[i]))
                        .collect();
                    let mut handle = engine.enqueue_g1_to_g2(SourceBlocks::External(blocks))?;
                    handle.wait().await?;
                }

                // Measure
                let mut latencies = Vec::with_capacity(self.config.iterations);
                for _ in 0..self.config.iterations {
                    let blocks: Vec<ExternalBlock<G1>> = (0..batch_size)
                        .map(|i| ExternalBlock::new(i as BlockId, seq_hashes[i]))
                        .collect();

                    let start = Instant::now();
                    let mut handle = engine.enqueue_g1_to_g2(SourceBlocks::External(blocks))?;
                    handle.wait().await?;
                    latencies.push(start.elapsed());
                }

                let bytes_per_block = compute_bytes_per_block(&self.config, self.page_size);
                let bytes_per_iter = bytes_per_block * batch_size;
                let stats = LatencyStats::from_durations(latencies);
                let bandwidth_gbs = bytes_per_iter as f64 / (stats.mean_us * 1e3);
                let num_devices = self.config.devices.len();

                let r = BenchResult {
                    test: "offload_g1_to_g2_pipeline".to_string(),
                    device_id,
                    page_size: self.page_size,
                    blocks_per_batch: batch_size,
                    concurrency: conc,
                    bounce_blocks: None,
                    bytes_per_iter,
                    iterations: self.config.iterations,
                    latency_us: stats,
                    bandwidth_gbs,
                    aggregate_bandwidth_gbs: bandwidth_gbs * num_devices as f64,
                };
                print_result_stderr(&r);
                results.push(r);
            }
        }

        // G2→G3 pipeline if disk enabled
        if !self.config.skip_disk && engine.has_g2_to_g3() {
            for &batch_size in &self.config.offload_batch_sizes {
                if batch_size > self.config.num_blocks {
                    continue;
                }

                for &conc in &self.config.offload_concurrency {
                    eprintln!("Offload G2→G3 pipeline: batch_size={batch_size} concurrency={conc}");

                    // Get immutable blocks from g2_manager for SourceBlocks::Strong
                    let matched = self.g2_manager.match_blocks(&seq_hashes[..batch_size]);

                    // Warmup
                    for _ in 0..self.config.warmup {
                        let mut handle =
                            engine.enqueue_g2_to_g3(SourceBlocks::Strong(matched.clone()))?;
                        handle.wait().await?;
                    }

                    // Measure
                    let mut latencies = Vec::with_capacity(self.config.iterations);
                    for _ in 0..self.config.iterations {
                        let start = Instant::now();
                        let mut handle =
                            engine.enqueue_g2_to_g3(SourceBlocks::Strong(matched.clone()))?;
                        handle.wait().await?;
                        latencies.push(start.elapsed());
                    }

                    let bytes_per_block = compute_bytes_per_block(&self.config, self.page_size);
                    let bytes_per_iter = bytes_per_block * batch_size;
                    let stats = LatencyStats::from_durations(latencies);
                    let bandwidth_gbs = bytes_per_iter as f64 / (stats.mean_us * 1e3);
                    let num_devices = self.config.devices.len();

                    let r = BenchResult {
                        test: "offload_g2_to_g3_pipeline".to_string(),
                        device_id,
                        page_size: self.page_size,
                        blocks_per_batch: batch_size,
                        concurrency: conc,
                        bounce_blocks: None,
                        bytes_per_iter,
                        iterations: self.config.iterations,
                        latency_us: stats,
                        bandwidth_gbs,
                        aggregate_bandwidth_gbs: bandwidth_gbs * num_devices as f64,
                    };
                    print_result_stderr(&r);
                    results.push(r);
                }
            }
        }

        Ok(results)
    }

    // ─── Transfer Helpers ──────────────────────────────────────────────────

    /// Benchmark a single transfer direction via the parallel worker (SPMD).
    async fn bench_transfer(
        &self,
        parallel_worker: &dyn WorkerTransfers,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_block_ids: Arc<[BlockId]>,
        dst_block_ids: Arc<[BlockId]>,
    ) -> Result<Vec<Duration>> {
        self.bench_transfer_with_options(
            parallel_worker,
            src,
            dst,
            src_block_ids,
            dst_block_ids,
            TransferOptions::default(),
        )
        .await
    }

    /// Benchmark a transfer with custom TransferOptions (e.g., bounce buffer).
    async fn bench_transfer_with_options(
        &self,
        parallel_worker: &dyn WorkerTransfers,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_block_ids: Arc<[BlockId]>,
        dst_block_ids: Arc<[BlockId]>,
        options: TransferOptions,
    ) -> Result<Vec<Duration>> {
        // Warmup
        for _ in 0..self.config.warmup {
            let notif = parallel_worker.execute_local_transfer(
                src,
                dst,
                src_block_ids.clone(),
                dst_block_ids.clone(),
                options.clone(),
            )?;
            notif.await?;
        }

        // Measure
        let mut latencies = Vec::with_capacity(self.config.iterations);
        for _ in 0..self.config.iterations {
            let start = Instant::now();
            let notif = parallel_worker.execute_local_transfer(
                src,
                dst,
                src_block_ids.clone(),
                dst_block_ids.clone(),
                options.clone(),
            )?;
            notif.await?;
            latencies.push(start.elapsed());
        }

        Ok(latencies)
    }

    /// Shutdown all workers.
    fn shutdown(self) {
        for handle in self.worker_handles {
            handle.shutdown_tx.send(()).ok();
            handle.join_handle.join().ok();
        }
    }
}

// ─── NUMA Pinning Helpers ──────────────────────────────────────────────────────

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

    // Validate offload config
    if config.offload {
        ensure!(
            !config.offload_batch_sizes.is_empty(),
            "offload enabled but no batch sizes specified"
        );
        ensure!(
            !config.offload_concurrency.is_empty(),
            "offload enabled but no concurrency levels specified"
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

    eprintln!("KVBM Engine Benchmark (Leader+Worker Architecture)");
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
    if config.offload {
        eprintln!("  Offload: enabled");
        eprintln!("    Batch sizes: {:?}", config.offload_batch_sizes);
        eprintln!("    Concurrency: {:?}", config.offload_concurrency);
    }
    eprintln!();

    // Build a main-thread tokio runtime for the leader
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .thread_name("bench-leader-tokio")
        .build()?;

    let all_results = rt.block_on(async {
        let mut all_results: Vec<BenchResult> = Vec::new();

        // Page-size sweep: rebuild full worker stack per page_size
        // (mirrors production where model config determines page_size at startup)
        for &page_size in &config.page_sizes {
            eprintln!("\n{}", "=".repeat(72));
            eprintln!("Page size: {page_size}");
            eprintln!("{}", "=".repeat(72));

            let instance = BenchInstance::new(config.clone(), page_size).await?;
            let results = instance.run_benchmarks().await?;
            all_results.extend(results);
            instance.shutdown();
        }

        Ok::<_, anyhow::Error>(all_results)
    })?;

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
