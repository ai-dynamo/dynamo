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
    KvCacheEvent, KvCacheEventData, RouterEvent, TokensWithHashes, WorkerWithDpRank,
};
use dynamo_kv_router::{
    BranchShardedIndexer, ConcurrentRadixTree, ConcurrentRadixTreeCompressed, PositionalIndexer,
    ThreadPoolIndexer,
};
use dynamo_mocker::loadgen::Trace;
use tokio::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

use crate::common::{
    BenchmarkRun, WorkerReplayArtifacts, compute_benchmark_run, make_progress_bar,
    rescale_trace_timestamps,
};

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MooncakeIndexerKind {
    RadixTree,
    NestedMap,
    ConcurrentRadixTree,
    ConcurrentRadixTreeCompressed,
    BranchShardedCrtc,
}

#[derive(Clone, Debug)]
pub struct MooncakeIndexerConfig {
    pub kind: MooncakeIndexerKind,
    pub jump_size: usize,
    pub num_event_workers: usize,
    pub num_shards: usize,
    pub num_event_workers_per_shard: usize,
    pub prefix_depth: usize,
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

    pub fn short_name(&self) -> &'static str {
        match self.kind {
            MooncakeIndexerKind::RadixTree => "radix-tree",
            MooncakeIndexerKind::NestedMap => "nested-map",
            MooncakeIndexerKind::ConcurrentRadixTree => "concurrent-radix-tree",
            MooncakeIndexerKind::ConcurrentRadixTreeCompressed => {
                "concurrent-radix-tree-compressed"
            }
            MooncakeIndexerKind::BranchShardedCrtc => "branch-sharded-crtc",
        }
    }

    pub fn is_multi_threaded(&self) -> bool {
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

    pub fn from_short_name(name: &str, num_event_workers: usize) -> anyhow::Result<Self> {
        let config = match name {
            "radix-tree" => Self::radix_tree(),
            "nested-map" => Self::nested_map(8, num_event_workers),
            "concurrent-radix-tree" => Self::concurrent_radix_tree(num_event_workers),
            "concurrent-radix-tree-compressed" => {
                Self::concurrent_radix_tree_compressed(num_event_workers)
            }
            "branch-sharded-crtc" => Self::branch_sharded_crtc(2, num_event_workers, 2),
            _ => anyhow::bail!(
                "Unknown indexer '{}'. Valid names: radix-tree, nested-map, concurrent-radix-tree, concurrent-radix-tree-compressed, branch-sharded-crtc",
                name
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
        }
    }

    pub fn build_approximate(
        &self,
        block_size: u32,
        metrics: Arc<KvIndexerMetrics>,
    ) -> anyhow::Result<Arc<dyn KvIndexerInterface + Send + Sync>> {
        let prune_config = PruneConfig::default();
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
        };
        Ok(indexer)
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
    Event(KvCacheEvent),
    ApproxWrite { tokens: Vec<u32>, num_blocks: usize },
}

/// A timestamped entry in a worker's benchmark trace, used to replay requests
/// and events at the correct relative timing.
#[derive(Clone)]
struct WorkerTrace {
    entry: WorkerTraceEntry,
    timestamp_us: u64,
}

#[derive(Clone)]
pub enum MooncakeBenchmarkInput {
    KvEvents(Vec<WorkerReplayArtifacts>),
    Approx(Vec<Trace>),
}

fn prepare_event_worker_traces(
    artifacts: Vec<WorkerReplayArtifacts>,
    benchmark_duration_ms: u64,
) -> Vec<Vec<WorkerTrace>> {
    let traces = artifacts
        .into_iter()
        .map(|artifact| {
            let mut merged = artifact
                .requests
                .into_iter()
                .map(|request| WorkerTrace {
                    timestamp_us: request.timestamp_us,
                    entry: WorkerTraceEntry::Request(request.replay_hashes.local_block_hashes),
                })
                .chain(artifact.kv_events.into_iter().map(|event| WorkerTrace {
                    timestamp_us: event.timestamp_us,
                    entry: WorkerTraceEntry::Event(event.event),
                }))
                .collect::<Vec<_>>();
            merged.sort_by_key(|entry| entry.timestamp_us);
            merged
        })
        .collect::<Vec<_>>();

    rescale_trace_timestamps(
        &traces,
        benchmark_duration_ms,
        |entry| entry.timestamp_us,
        |entry, timestamp_us| WorkerTrace {
            entry: entry.entry.clone(),
            timestamp_us,
        },
    )
}

fn synthesized_tokens(
    turn: &dynamo_mocker::loadgen::TurnTrace,
    trace_block_size: usize,
) -> Vec<u32> {
    let mut tokens = Vec::with_capacity(turn.input_length);
    for &hash_id in &turn.hash_ids {
        tokens.extend((0..trace_block_size).map(|_| hash_id as u32));
        if tokens.len() >= turn.input_length {
            tokens.truncate(turn.input_length);
            break;
        }
    }
    tokens
}

fn prepare_approx_worker_traces(
    traces: Vec<Trace>,
    block_size: u32,
    benchmark_duration_ms: u64,
) -> anyhow::Result<Vec<Vec<WorkerTrace>>> {
    let mut worker_traces = Vec::with_capacity(traces.len());
    for trace in traces {
        let trace_block_size = trace.block_size;
        let mut entries = Vec::new();
        for session in trace.sessions {
            let mut timestamp_ms = session.first_arrival_timestamp_ms.unwrap_or(0.0);
            for (turn_idx, turn) in session.turns.into_iter().enumerate() {
                if turn_idx > 0 {
                    timestamp_ms += turn.delay_after_previous_ms;
                }
                let replay_hashes = turn.to_replay_hashes(trace_block_size, block_size as usize)?;
                let tokens = synthesized_tokens(&turn, trace_block_size);
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
    Ok(rescale_trace_timestamps(
        &worker_traces,
        benchmark_duration_ms,
        |entry| entry.timestamp_us,
        |entry, timestamp_us| WorkerTrace {
            entry: entry.entry.clone(),
            timestamp_us,
        },
    ))
}

fn prepare_worker_traces(
    input: MooncakeBenchmarkInput,
    config: MooncakeBenchmarkConfig,
) -> anyhow::Result<Vec<Vec<WorkerTrace>>> {
    match input {
        MooncakeBenchmarkInput::KvEvents(artifacts) => Ok(prepare_event_worker_traces(
            artifacts,
            config.benchmark_duration_ms,
        )),
        MooncakeBenchmarkInput::Approx(traces) => {
            prepare_approx_worker_traces(traces, config.block_size, config.benchmark_duration_ms)
        }
    }
}

pub async fn run_benchmark(
    indexer: Arc<dyn KvIndexerInterface + Send + Sync>,
    input: MooncakeBenchmarkInput,
    config: MooncakeBenchmarkConfig,
) -> anyhow::Result<BenchmarkRun> {
    let worker_traces = prepare_worker_traces(input, config)?;
    let worker_traces = worker_traces.into_iter().map(Arc::new).collect::<Vec<_>>();

    let progress = make_progress_bar(Some(
        worker_traces
            .iter()
            .map(|trace| trace.len() as u64)
            .sum::<u64>()
            * config.inference_worker_duplication_factor as u64,
    ));

    let mut tasks = Vec::new();
    for replica in 0..config.inference_worker_duplication_factor {
        for (worker_id, worker_trace) in worker_traces.iter().enumerate() {
            let indexer = Arc::clone(&indexer);
            let trace = Arc::clone(worker_trace);
            let progress = progress.clone();
            let worker_id = worker_id + replica * worker_traces.len();
            tasks.push(tokio::spawn(async move {
                let mut request_latencies = Vec::with_capacity(trace.len());

                let submit = |entry: WorkerTrace| async {
                    match entry.entry {
                        WorkerTraceEntry::Request(request) => {
                            let start = minstant::Instant::now();
                            indexer.find_matches(request).await?;
                            Ok::<Option<u64>, anyhow::Error>(
                                Some(start.elapsed().as_nanos() as u64),
                            )
                        }
                        WorkerTraceEntry::Event(event) => {
                            indexer
                                .apply_event(RouterEvent::new(worker_id as u64, event))
                                .await;
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
                let mut trace = trace.iter().peekable();
                let mut local_count = 0;

                while let Some(entry) = trace.next() {
                    let mut processed = 1;
                    let entry_timestamp_us = entry.timestamp_us;

                    if let Some(latency) = submit(entry.clone()).await? {
                        request_latencies.push(latency);
                    }

                    while let Some(next) = trace.peek() {
                        if next.timestamp_us == entry_timestamp_us {
                            if let Some(latency) = submit(trace.next().unwrap().clone()).await? {
                                request_latencies.push(latency);
                            }
                            processed += 1;
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

                    local_count += processed;

                    if local_count > 100 {
                        progress.inc(local_count);
                        local_count = 0;
                    }
                }

                progress.inc(local_count);

                Ok::<_, anyhow::Error>(request_latencies)
            }));
        }
    }

    let fm_stop = Arc::new(AtomicBool::new(false));
    let mut fm_tasks = Vec::new();
    if config.find_matches_concurrency > 0 {
        let seq_pool: Arc<Vec<Vec<LocalBlockHash>>> = Arc::new(
            worker_traces
                .iter()
                .flat_map(|trace| trace.iter())
                .filter_map(|entry| match &entry.entry {
                    WorkerTraceEntry::Request(hashes) => Some(hashes.clone()),
                    WorkerTraceEntry::Event(_) | WorkerTraceEntry::ApproxWrite { .. } => None,
                })
                .collect(),
        );

        if !seq_pool.is_empty() {
            for task_id in 0..config.find_matches_concurrency {
                let indexer = Arc::clone(&indexer);
                let pool = Arc::clone(&seq_pool);
                let stop = Arc::clone(&fm_stop);
                fm_tasks.push(tokio::spawn(async move {
                    let mut latencies = Vec::new();
                    let mut idx = task_id % pool.len();
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
    }

    let mut latencies = Vec::new();
    for task in tasks {
        latencies.extend(task.await??);
    }

    fm_stop.store(true, Ordering::Relaxed);
    for task in fm_tasks {
        if let Ok(fm_latencies) = task.await {
            latencies.extend(fm_latencies);
        }
    }

    let total_duration = progress.elapsed();
    let total_events = worker_traces
        .iter()
        .map(|trace| {
            trace
                .iter()
                .filter(|entry| matches!(entry.entry, WorkerTraceEntry::Event(_)))
                .count()
        })
        .sum::<usize>()
        * config.inference_worker_duplication_factor;
    let total_approx_writes = worker_traces
        .iter()
        .map(|trace| {
            trace
                .iter()
                .filter(|entry| matches!(entry.entry, WorkerTraceEntry::ApproxWrite { .. }))
                .count()
        })
        .sum::<usize>()
        * config.inference_worker_duplication_factor;

    let total_requests = worker_traces.iter().map(|trace| trace.len()).sum::<usize>()
        * config.inference_worker_duplication_factor
        - total_events
        - total_approx_writes;

    let total_request_blocks = worker_traces
        .iter()
        .flat_map(|trace| trace.iter())
        .filter_map(|entry| match &entry.entry {
            WorkerTraceEntry::Request(hashes) => Some(hashes.len()),
            WorkerTraceEntry::Event(_) | WorkerTraceEntry::ApproxWrite { .. } => None,
        })
        .sum::<usize>()
        * config.inference_worker_duplication_factor;

    let total_event_blocks = worker_traces
        .iter()
        .flat_map(|trace| trace.iter())
        .filter_map(|entry| match &entry.entry {
            WorkerTraceEntry::Event(event) => match &event.data {
                KvCacheEventData::Stored(store) => Some(store.blocks.len()),
                _ => Some(0),
            },
            WorkerTraceEntry::ApproxWrite { num_blocks, .. } => Some(*num_blocks),
            WorkerTraceEntry::Request(_) => None,
        })
        .sum::<usize>()
        * config.inference_worker_duplication_factor;

    let total_writes = total_events + total_approx_writes;
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
