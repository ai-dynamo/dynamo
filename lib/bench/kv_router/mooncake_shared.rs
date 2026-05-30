// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use dynamo_kv_router::LocalBlockHash;
use dynamo_kv_router::indexer::pruning::PruneConfig;
use dynamo_kv_router::indexer::{KvIndexer, KvIndexerInterface, KvIndexerMetrics};
use dynamo_kv_router::protocols::{
    KvCacheEvent, KvCacheEventData, RouterEvent, StorageTier, TokensWithHashes, WorkerWithDpRank,
};
#[cfg(feature = "uds-raw-bench")]
use dynamo_kv_router::shard_router::RawUdsShardClient;
use dynamo_kv_router::{
    BranchShardedIndexer, ConcurrentRadixTree, ConcurrentRadixTreeCompressed, PositionalIndexer,
    ThreadPoolIndexer,
};

/// Keeps alive the spawned `kv_shard_server` child processes for the duration
/// of a `BranchShardedUds` benchmark run.
///
/// Each child was spawned with `kill_on_drop(true)`.  Dropping this guard
/// sends SIGKILL to every child and removes the socket temp directory.
#[cfg(feature = "uds-raw-bench")]
pub struct UdsShardGuards {
    _tmp: tempfile::TempDir,
    /// Processes spawned with `kill_on_drop(true)`; killed when this is dropped.
    _children: Vec<tokio::process::Child>,
}
use dynamo_mocker::loadgen::Trace;
use tokio::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

use dynamo_bench::kv_router_common::replay::WorkerReplayArtifacts;
use dynamo_bench::kv_router_common::results::{BenchmarkRun, compute_benchmark_run};
use dynamo_bench::kv_router_common::trace_gen::{OrderedMerge, ReplayStartGate, WorkerTimelines};

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MooncakeIndexerKind {
    RadixTree,
    NestedMap,
    ConcurrentRadixTree,
    ConcurrentRadixTreeCompressed,
    BranchShardedCrtc,
    /// N independent `kv_shard_server` OS processes communicating over UDS.
    /// Each shard lives in a separate address space with no shared Tokio
    /// scheduler.  Requires the `kv_shard_server` binary to be built and its
    /// path passed via `--shard-server-path`.
    #[cfg(feature = "uds-raw-bench")]
    BranchShardedUds,
}

#[derive(Clone, Debug)]
pub struct MooncakeIndexerConfig {
    pub kind: MooncakeIndexerKind,
    pub jump_size: usize,
    pub num_event_workers: usize,
    pub num_shards: usize,
    pub num_event_workers_per_shard: usize,
    pub prefix_depth: usize,
    /// Path to the `kv_shard_server` binary.  Required for `BranchShardedUds`;
    /// ignored for all other variants.
    #[cfg(feature = "uds-raw-bench")]
    pub shard_server_path: std::path::PathBuf,
}

#[allow(dead_code)]
impl MooncakeIndexerConfig {
    pub fn radix_tree() -> Self {
        Self {
            kind: MooncakeIndexerKind::RadixTree,
            jump_size: 8,
            num_event_workers: 16,
            num_shards: 2,
            num_event_workers_per_shard: 4,
            prefix_depth: 2,
            #[cfg(feature = "uds-raw-bench")]
            shard_server_path: std::path::PathBuf::from("kv_shard_server"),
        }
    }

    pub fn nested_map(jump_size: usize, num_event_workers: usize) -> Self {
        Self {
            kind: MooncakeIndexerKind::NestedMap,
            jump_size,
            num_event_workers,
            ..Self::radix_tree()
        }
    }

    pub fn concurrent_radix_tree(num_event_workers: usize) -> Self {
        Self {
            kind: MooncakeIndexerKind::ConcurrentRadixTree,
            num_event_workers,
            ..Self::radix_tree()
        }
    }

    pub fn concurrent_radix_tree_compressed(num_event_workers: usize) -> Self {
        Self {
            kind: MooncakeIndexerKind::ConcurrentRadixTreeCompressed,
            num_event_workers,
            ..Self::radix_tree()
        }
    }

    pub fn branch_sharded_crtc(
        num_shards: usize,
        num_event_workers_per_shard: usize,
        prefix_depth: usize,
    ) -> Self {
        Self {
            kind: MooncakeIndexerKind::BranchShardedCrtc,
            num_shards,
            num_event_workers_per_shard,
            prefix_depth,
            ..Self::radix_tree()
        }
    }

    #[cfg(feature = "uds-raw-bench")]
    pub fn branch_sharded_uds(
        num_shards: usize,
        num_event_workers_per_shard: usize,
        prefix_depth: usize,
        shard_server_path: std::path::PathBuf,
    ) -> Self {
        Self {
            kind: MooncakeIndexerKind::BranchShardedUds,
            num_shards,
            num_event_workers_per_shard,
            prefix_depth,
            shard_server_path,
            ..Self::radix_tree()
        }
    }

    pub fn short_name(&self) -> &'static str {
        match self.kind {
            MooncakeIndexerKind::RadixTree => "radix-tree",
            MooncakeIndexerKind::NestedMap => "nested-map",
            MooncakeIndexerKind::ConcurrentRadixTree => "concurrent-radix-tree",
            MooncakeIndexerKind::ConcurrentRadixTreeCompressed => {
                "concurrent-radix-tree-compressed"
            }
            MooncakeIndexerKind::BranchShardedCrtc => "branch-sharded-crtc",
            #[cfg(feature = "uds-raw-bench")]
            MooncakeIndexerKind::BranchShardedUds => "branch-sharded-uds",
        }
    }

    pub fn is_multi_threaded(&self) -> bool {
        #[cfg(feature = "uds-raw-bench")]
        if matches!(self.kind, MooncakeIndexerKind::BranchShardedUds) {
            return true;
        }
        matches!(
            self.kind,
            MooncakeIndexerKind::NestedMap
                | MooncakeIndexerKind::ConcurrentRadixTree
                | MooncakeIndexerKind::ConcurrentRadixTreeCompressed
                | MooncakeIndexerKind::BranchShardedCrtc
        )
    }

    pub fn supports_remove(&self) -> bool {
        true
    }

    pub fn supports_approximate(&self) -> bool {
        match self.kind {
            MooncakeIndexerKind::BranchShardedCrtc => false,
            #[cfg(feature = "uds-raw-bench")]
            MooncakeIndexerKind::BranchShardedUds => false,
            _ => true,
        }
    }

    pub fn from_short_name(name: &str, num_event_workers: usize) -> anyhow::Result<Self> {
        let config = match name {
            "radix-tree" => Self::radix_tree(),
            "nested-map" => Self::nested_map(8, num_event_workers),
            "concurrent-radix-tree" => Self::concurrent_radix_tree(num_event_workers),
            "concurrent-radix-tree-compressed" => {
                Self::concurrent_radix_tree_compressed(num_event_workers)
            }
            "branch-sharded-crtc" => Self::branch_sharded_crtc(2, num_event_workers, 2),
            #[cfg(feature = "uds-raw-bench")]
            "branch-sharded-uds" => {
                anyhow::bail!(
                    "branch-sharded-uds requires --shard-server-path; \
                     use the BranchShardedUds subcommand instead of --compare"
                )
            }
            _ => anyhow::bail!(
                "Unknown indexer '{}'. Valid names: radix-tree, nested-map, concurrent-radix-tree, \
                 concurrent-radix-tree-compressed, branch-sharded-crtc{}",
                name,
                if cfg!(feature = "uds-raw-bench") {
                    ", branch-sharded-uds"
                } else {
                    ""
                },
            ),
        };
        Ok(config)
    }

    pub fn build(
        &self,
        block_size: u32,
        metrics: Arc<KvIndexerMetrics>,
    ) -> Arc<dyn KvIndexerInterface + Send + Sync> {
        match self.kind {
            MooncakeIndexerKind::RadixTree => Arc::new(KvIndexer::new(
                CancellationToken::new(),
                block_size,
                metrics,
            )),
            MooncakeIndexerKind::NestedMap => Arc::new(ThreadPoolIndexer::new_with_metrics(
                PositionalIndexer::new(self.jump_size),
                self.num_event_workers,
                block_size,
                Some(metrics),
            )),
            MooncakeIndexerKind::ConcurrentRadixTree => {
                Arc::new(ThreadPoolIndexer::new_with_metrics(
                    ConcurrentRadixTree::new(),
                    self.num_event_workers,
                    block_size,
                    Some(metrics),
                ))
            }
            MooncakeIndexerKind::ConcurrentRadixTreeCompressed => {
                Arc::new(ThreadPoolIndexer::new_with_metrics(
                    ConcurrentRadixTreeCompressed::new(),
                    self.num_event_workers,
                    block_size,
                    Some(metrics),
                ))
            }
            MooncakeIndexerKind::BranchShardedCrtc => {
                let shards = (0..self.num_shards)
                    .map(|_| {
                        ThreadPoolIndexer::new_with_metrics(
                            ConcurrentRadixTreeCompressed::new(),
                            self.num_event_workers_per_shard,
                            block_size,
                            Some(Arc::clone(&metrics)),
                        )
                    })
                    .collect();
                Arc::new(BranchShardedIndexer::new_with_options(
                    shards,
                    self.prefix_depth,
                    block_size,
                ))
            }
            #[cfg(feature = "uds-raw-bench")]
            MooncakeIndexerKind::BranchShardedUds => {
                panic!(
                    "BranchShardedUds requires async construction; \
                     call build_uds() instead of build()"
                )
            }
        }
    }

    /// Cross-process UDS: spawns `num_shards` independent `kv_shard_server`
    /// OS processes and connects a client to each.
    ///
    /// Each child is spawned with `kill_on_drop(true)` so the guard's `Drop`
    /// sends SIGKILL automatically.  The bench waits up to 10 s per socket
    /// for the child process to bind and start accepting connections.
    #[cfg(feature = "uds-raw-bench")]
    pub async fn build_uds(
        &self,
        block_size: u32,
    ) -> anyhow::Result<(Arc<dyn KvIndexerInterface + Send + Sync>, UdsShardGuards)> {
        use std::time::Duration;

        let tmp = tempfile::TempDir::new()?;
        let mut children = Vec::with_capacity(self.num_shards);
        let mut socket_paths = Vec::with_capacity(self.num_shards);

        // Spawn all child processes first so they can bind concurrently.
        for shard_idx in 0..self.num_shards {
            let socket_path = tmp.path().join(format!("shard_{shard_idx}.sock"));
            let child = tokio::process::Command::new(&self.shard_server_path)
                .arg("--socket-path")
                .arg(&socket_path)
                .arg("--num-threads")
                .arg(self.num_event_workers_per_shard.to_string())
                .arg("--block-size")
                .arg(block_size.to_string())
                .kill_on_drop(true)
                .spawn()
                .map_err(|e| {
                    anyhow::anyhow!(
                        "failed to spawn kv_shard_server at '{}': {e}\n\
                         Build it first: cargo build -p dynamo-kv-router \
                         --features uds-raw-bench --bin kv_shard_server --release",
                        self.shard_server_path.display()
                    )
                })?;
            children.push(child);
            socket_paths.push(socket_path);
        }

        // Wait for each socket to become connectable.
        let mut clients = Vec::with_capacity(self.num_shards);
        for (shard_idx, socket_path) in socket_paths.iter().enumerate() {
            let client = wait_and_connect(socket_path, shard_idx, Duration::from_secs(10)).await?;
            clients.push(client);
        }

        let indexer: Arc<dyn KvIndexerInterface + Send + Sync> = Arc::new(
            BranchShardedIndexer::new_with_options(clients, self.prefix_depth, block_size),
        );
        Ok((
            indexer,
            UdsShardGuards {
                _tmp: tmp,
                _children: children,
            },
        ))
    }

    pub fn build_approximate(
        &self,
        block_size: u32,
        metrics: Arc<KvIndexerMetrics>,
    ) -> anyhow::Result<Arc<dyn KvIndexerInterface + Send + Sync>> {
        self.build_approximate_with_prune_config(block_size, metrics, PruneConfig::default())
    }

    pub fn build_approximate_with_prune_config(
        &self,
        block_size: u32,
        metrics: Arc<KvIndexerMetrics>,
        prune_config: PruneConfig,
    ) -> anyhow::Result<Arc<dyn KvIndexerInterface + Send + Sync>> {
        let indexer: Arc<dyn KvIndexerInterface + Send + Sync> = match self.kind {
            MooncakeIndexerKind::RadixTree => Arc::new(KvIndexer::new_with_frequency(
                CancellationToken::new(),
                None,
                block_size,
                metrics,
                Some(prune_config),
            )),
            MooncakeIndexerKind::NestedMap => {
                Arc::new(ThreadPoolIndexer::new_with_metrics_and_pruning(
                    PositionalIndexer::new(self.jump_size),
                    self.num_event_workers,
                    block_size,
                    Some(metrics),
                    Some(prune_config),
                ))
            }
            MooncakeIndexerKind::ConcurrentRadixTree => {
                Arc::new(ThreadPoolIndexer::new_with_metrics_and_pruning(
                    ConcurrentRadixTree::new(),
                    self.num_event_workers,
                    block_size,
                    Some(metrics),
                    Some(prune_config),
                ))
            }
            MooncakeIndexerKind::ConcurrentRadixTreeCompressed => {
                Arc::new(ThreadPoolIndexer::new_with_metrics_and_pruning(
                    ConcurrentRadixTreeCompressed::new(),
                    self.num_event_workers,
                    block_size,
                    Some(metrics),
                    Some(prune_config),
                ))
            }
            MooncakeIndexerKind::BranchShardedCrtc => {
                anyhow::bail!("branch-sharded-crtc does not support approximate pruning")
            }
            #[cfg(feature = "uds-raw-bench")]
            MooncakeIndexerKind::BranchShardedUds => {
                anyhow::bail!("branch-sharded-uds does not support approximate pruning")
            }
        };
        Ok(indexer)
    }
}

/// Retries `UnixStream::connect` until the socket file exists and accepts
/// a connection, or until `timeout` elapses.
///
/// Child processes need a small startup window before they bind the socket.
/// 10 ms polling interval is fast enough to not slow down bench setup while
/// avoiding a tight spin.
#[cfg(feature = "uds-raw-bench")]
async fn wait_and_connect(
    socket_path: &std::path::Path,
    shard_idx: usize,
    timeout: std::time::Duration,
) -> anyhow::Result<RawUdsShardClient> {
    use tokio::time::{Duration, sleep};
    let deadline = std::time::Instant::now() + timeout;
    loop {
        match RawUdsShardClient::connect(socket_path).await {
            Ok(client) => {
                tracing::debug!(shard_idx, path = %socket_path.display(), "shard server connected");
                return Ok(client);
            }
            Err(_) if std::time::Instant::now() < deadline => {
                sleep(Duration::from_millis(10)).await;
            }
            Err(e) => {
                anyhow::bail!(
                    "timed out ({timeout:?}) waiting for shard {shard_idx} at '{}': {e}",
                    socket_path.display()
                );
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MooncakeBenchmarkConfig {
    pub benchmark_duration_ms: u64,
    pub inference_worker_duplication_factor: usize,
    pub count_events: bool,
    pub find_matches_concurrency: usize,
    pub block_size: u32,
}

/// A single entry in a worker's merged benchmark timeline.
#[derive(Clone)]
enum WorkerTraceEntry {
    Request(Vec<LocalBlockHash>),
    Event {
        event: KvCacheEvent,
        storage_tier: StorageTier,
    },
    ApproxWrite {
        tokens: Vec<u32>,
        num_blocks: usize,
    },
}

/// A timestamped entry in a worker's benchmark trace, used to replay requests
/// and events at the correct relative timing.
#[derive(Clone)]
struct WorkerTrace {
    entry: WorkerTraceEntry,
    timestamp_us: u64,
}

#[derive(Clone)]
pub(crate) struct MergedMooncakeBenchmark {
    worker_traces: WorkerTimelines<WorkerTrace>,
    block_size: u32,
}

#[derive(Clone, Copy)]
struct PreparedTraceTotals {
    requests: usize,
    events: usize,
    approx_writes: usize,
    request_blocks: usize,
    event_blocks: usize,
}

pub(crate) struct PreparedMooncakeBenchmark {
    worker_traces: WorkerTimelines<WorkerTrace>,
    seq_pool: Arc<Vec<Vec<LocalBlockHash>>>,
    totals: PreparedTraceTotals,
    benchmark_duration_ms: u64,
    block_size: u32,
}

#[derive(Clone)]
pub enum MooncakeBenchmarkInput {
    KvEvents(Vec<WorkerReplayArtifacts>),
    Approx(Vec<Trace>),
}

fn merge_event_worker_trace(
    worker_idx: usize,
    artifact: &WorkerReplayArtifacts,
) -> anyhow::Result<Vec<WorkerTrace>> {
    OrderedMerge::left_first(
        worker_idx,
        "requests",
        &artifact.requests,
        "kv_events",
        &artifact.kv_events,
    )
    .merge(
        |request| request.timestamp_us,
        |event| event.timestamp_us,
        |request| WorkerTrace {
            timestamp_us: request.timestamp_us,
            entry: WorkerTraceEntry::Request(request.replay_hashes.local_block_hashes.clone()),
        },
        |event| WorkerTrace {
            timestamp_us: event.timestamp_us,
            entry: WorkerTraceEntry::Event {
                event: event.event.clone(),
                storage_tier: event.storage_tier,
            },
        },
    )
}

fn merge_event_worker_traces(
    artifacts: &[WorkerReplayArtifacts],
) -> anyhow::Result<WorkerTimelines<WorkerTrace>> {
    let worker_traces = artifacts
        .iter()
        .enumerate()
        .map(|(worker_idx, artifact)| merge_event_worker_trace(worker_idx, artifact))
        .collect::<anyhow::Result<Vec<_>>>()?;

    Ok(WorkerTimelines::new(worker_traces))
}

fn merge_approx_worker_traces(
    traces: &[Trace],
    block_size: u32,
) -> anyhow::Result<WorkerTimelines<WorkerTrace>> {
    let mut worker_traces = Vec::with_capacity(traces.len());
    for trace in traces {
        let trace_block_size = trace.block_size;
        let mut entries = Vec::new();
        for session in &trace.sessions {
            let mut timestamp_ms = session.first_arrival_timestamp_ms.unwrap_or(0.0);
            for (turn_idx, turn) in session.turns.iter().enumerate() {
                if turn_idx > 0 {
                    timestamp_ms += turn.delay_after_previous_ms;
                }
                let replay_hashes = turn.to_replay_hashes(trace_block_size, block_size as usize)?;
                let tokens = turn.synthesize_tokens(trace_block_size)?;
                let timestamp_us = (timestamp_ms.max(0.0) * 1000.0) as u64;
                entries.push(WorkerTrace {
                    timestamp_us,
                    entry: WorkerTraceEntry::Request(replay_hashes.local_block_hashes),
                });
                entries.push(WorkerTrace {
                    timestamp_us,
                    entry: WorkerTraceEntry::ApproxWrite {
                        tokens,
                        num_blocks: replay_hashes.sequence_hashes.len(),
                    },
                });
            }
        }
        entries.sort_by_key(|entry| entry.timestamp_us);
        worker_traces.push(entries);
    }

    Ok(WorkerTimelines::new(worker_traces))
}

pub(crate) fn merge_worker_traces(
    input: &MooncakeBenchmarkInput,
    block_size: u32,
) -> anyhow::Result<MergedMooncakeBenchmark> {
    let worker_traces = match input {
        MooncakeBenchmarkInput::KvEvents(artifacts) => merge_event_worker_traces(artifacts)?,
        MooncakeBenchmarkInput::Approx(traces) => merge_approx_worker_traces(traces, block_size)?,
    };

    Ok(MergedMooncakeBenchmark {
        worker_traces,
        block_size,
    })
}

pub(crate) fn prepare_scaled_benchmark(
    merged: &MergedMooncakeBenchmark,
    config: MooncakeBenchmarkConfig,
) -> PreparedMooncakeBenchmark {
    let worker_traces = merged.worker_traces.rescale(
        config.benchmark_duration_ms,
        |entry| entry.timestamp_us,
        |entry, timestamp_us| WorkerTrace {
            entry: entry.entry.clone(),
            timestamp_us,
        },
    );

    let mut seq_pool = Vec::new();
    let mut totals = PreparedTraceTotals {
        requests: 0,
        events: 0,
        approx_writes: 0,
        request_blocks: 0,
        event_blocks: 0,
    };

    for entry in worker_traces.iter().flatten() {
        match &entry.entry {
            WorkerTraceEntry::Request(hashes) => {
                totals.requests += 1;
                totals.request_blocks += hashes.len();
                seq_pool.push(hashes.clone());
            }
            WorkerTraceEntry::Event { event, .. } => {
                totals.events += 1;
                totals.event_blocks += match &event.data {
                    KvCacheEventData::Stored(store) => store.blocks.len(),
                    _ => 0,
                };
            }
            WorkerTraceEntry::ApproxWrite { num_blocks, .. } => {
                totals.approx_writes += 1;
                totals.event_blocks += *num_blocks;
            }
        }
    }

    PreparedMooncakeBenchmark {
        worker_traces,
        seq_pool: Arc::new(seq_pool),
        totals,
        benchmark_duration_ms: config.benchmark_duration_ms,
        block_size: merged.block_size,
    }
}

#[allow(dead_code)]
pub async fn run_benchmark(
    indexer: Arc<dyn KvIndexerInterface + Send + Sync>,
    input: &MooncakeBenchmarkInput,
    config: MooncakeBenchmarkConfig,
) -> anyhow::Result<BenchmarkRun> {
    let merged = merge_worker_traces(input, config.block_size)?;
    let prepared = prepare_scaled_benchmark(&merged, config);
    run_prepared_benchmark(indexer, &prepared, config).await
}

pub(crate) async fn run_prepared_benchmark(
    indexer: Arc<dyn KvIndexerInterface + Send + Sync>,
    prepared: &PreparedMooncakeBenchmark,
    config: MooncakeBenchmarkConfig,
) -> anyhow::Result<BenchmarkRun> {
    if prepared.benchmark_duration_ms != config.benchmark_duration_ms {
        anyhow::bail!(
            "prepared benchmark duration mismatch: prepared={}ms, requested={}ms",
            prepared.benchmark_duration_ms,
            config.benchmark_duration_ms
        );
    }
    if prepared.block_size != config.block_size {
        anyhow::bail!(
            "prepared benchmark block size mismatch: prepared={}, requested={}",
            prepared.block_size,
            config.block_size
        );
    }

    let num_trace_workers = prepared.worker_traces.len();
    let mut task_inputs =
        Vec::with_capacity(num_trace_workers * config.inference_worker_duplication_factor);
    for replica in 0..config.inference_worker_duplication_factor {
        for (worker_id, worker_trace) in prepared.worker_traces.iter().enumerate() {
            let worker_id = worker_id + replica * num_trace_workers;
            task_inputs.push((worker_id, worker_trace.clone()));
        }
    }

    let num_find_match_tasks =
        if config.find_matches_concurrency > 0 && !prepared.seq_pool.is_empty() {
            config.find_matches_concurrency
        } else {
            0
        };
    let num_timed_tasks = task_inputs.len() + num_find_match_tasks;
    let start_gate = ReplayStartGate::new(num_timed_tasks);

    let mut tasks = Vec::new();
    for (worker_id, trace) in task_inputs {
        let indexer = Arc::clone(&indexer);
        let start_gate = start_gate.clone();
        tasks.push(tokio::spawn(async move {
            let mut request_latencies = Vec::with_capacity(trace.len());

            start_gate.wait_for_start().await;

            let submit = |entry: WorkerTrace| async {
                match entry.entry {
                    WorkerTraceEntry::Request(request) => {
                        let start = minstant::Instant::now();
                        indexer.find_matches(request).await?;
                        Ok::<Option<u64>, anyhow::Error>(Some(start.elapsed().as_nanos() as u64))
                    }
                    WorkerTraceEntry::Event {
                        event,
                        storage_tier,
                    } => {
                        if storage_tier.is_gpu() {
                            indexer
                                .apply_event(RouterEvent::with_storage_tier(
                                    worker_id as u64,
                                    event,
                                    storage_tier,
                                ))
                                .await;
                        }
                        Ok(None)
                    }
                    WorkerTraceEntry::ApproxWrite { tokens, .. } => {
                        let mut tokens_with_hashes =
                            TokensWithHashes::new(tokens, config.block_size);
                        indexer
                            .process_routing_decision_for_request(
                                &mut tokens_with_hashes,
                                WorkerWithDpRank::from_worker_id(worker_id as u64),
                            )
                            .await?;
                        Ok(None)
                    }
                }
            };

            let mut target = Instant::now();
            let mut trace = trace.into_iter().peekable();

            while let Some(entry) = trace.next() {
                let entry_timestamp_us = entry.timestamp_us;

                if let Some(latency) = submit(entry).await? {
                    request_latencies.push(latency);
                }

                while let Some(next) = trace.peek() {
                    if next.timestamp_us == entry_timestamp_us {
                        if let Some(latency) = submit(trace.next().unwrap()).await? {
                            request_latencies.push(latency);
                        }
                    } else {
                        break;
                    }
                }

                if let Some(next) = trace.peek() {
                    target += Duration::from_micros(next.timestamp_us - entry_timestamp_us);
                }

                if target > Instant::now() {
                    tokio::time::sleep_until(target).await;
                }
            }

            Ok::<_, anyhow::Error>(request_latencies)
        }));
    }

    let fm_stop = Arc::new(AtomicBool::new(false));
    let mut fm_tasks = Vec::new();
    if num_find_match_tasks > 0 {
        for task_id in 0..num_find_match_tasks {
            let indexer = Arc::clone(&indexer);
            let pool = Arc::clone(&prepared.seq_pool);
            let stop = Arc::clone(&fm_stop);
            let start_gate = start_gate.clone();
            fm_tasks.push(tokio::spawn(async move {
                let mut latencies = Vec::new();
                let mut idx = task_id % pool.len();

                start_gate.wait_for_start().await;

                while !stop.load(Ordering::Relaxed) {
                    let seq = pool[idx].clone();
                    let start = minstant::Instant::now();
                    let _ = indexer.find_matches(seq).await;
                    latencies.push(start.elapsed().as_nanos() as u64);
                    idx = (idx + 1) % pool.len();
                }
                latencies
            }));
        }
    }

    let started_at = start_gate.start().await;

    let mut latencies = Vec::new();
    for task in tasks {
        latencies.extend(task.await??);
    }

    // Flush the indexer before stopping the clock.  For cross-process UDS
    // variants, apply_event only enqueues a frame locally; the frame may not
    // have been drained to the shard server process yet.  flush() waits until
    // all previously enqueued writes have been processed, giving trustworthy
    // "submitted + drained" throughput numbers.
    indexer.flush().await;

    let total_duration = started_at.elapsed();

    fm_stop.store(true, Ordering::Relaxed);
    for task in fm_tasks {
        if let Ok(fm_latencies) = task.await {
            latencies.extend(fm_latencies);
        }
    }

    let replay_factor = config.inference_worker_duplication_factor;
    let total_requests = prepared.totals.requests * replay_factor;
    let total_writes = (prepared.totals.events + prepared.totals.approx_writes) * replay_factor;
    let total_request_blocks = prepared.totals.request_blocks * replay_factor;
    let total_event_blocks = prepared.totals.event_blocks * replay_factor;

    let counted_events = if config.count_events { total_writes } else { 0 };
    let counted_event_blocks = if config.count_events {
        total_event_blocks
    } else {
        0
    };

    let run = compute_benchmark_run(
        total_requests + counted_events,
        total_request_blocks + counted_event_blocks,
        config.benchmark_duration_ms,
        total_duration,
        latencies,
    );

    println!(
        "Offered Ops Throughput: {} ops/s | Achieved: {} ops/s (requests + events)",
        run.results.offered_ops_throughput as u64, run.results.ops_throughput as u64,
    );
    println!(
        "Offered Block Throughput: {} block ops/s | Achieved: {} block ops/s",
        run.results.offered_block_throughput as u64, run.results.block_throughput as u64,
    );
    println!("Latency p99: {}us", run.results.latency_p99_us);

    Ok(run)
}
