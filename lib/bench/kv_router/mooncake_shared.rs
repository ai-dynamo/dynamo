// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_kv_router::LocalBlockHash;
use dynamo_kv_router::indexer::cuckoo::{CkfConfig, CkfMatchMode, EventTransposedCkfIndexer};
use dynamo_kv_router::indexer::pruning::PruneConfig;
use dynamo_kv_router::indexer::{KvIndexer, KvIndexerInterface, KvIndexerMetrics};
use dynamo_kv_router::protocols::{KvCacheEvent, KvCacheEventData, StorageTier, WorkerWithDpRank};
use dynamo_kv_router::{
    BranchShardedIndexer, ConcurrentRadixTree, ConcurrentRadixTreeCompressed, PositionalIndexer,
    ThreadPoolIndexer,
};
use tokio_util::sync::CancellationToken;

use dynamo_bench::kv_router_common::replay::WorkerReplayArtifacts;
use dynamo_bench::kv_router_common::trace_gen::WorkerTimelines;

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MooncakeIndexerKind {
    RadixTree,
    NestedMap,
    ConcurrentRadixTree,
    ConcurrentRadixTreeCompressed,
    TransposedCkf,
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
    pub expected_blocks_per_dc: usize,
    pub publish_every_n_events: usize,
    pub ckf_match_mode: CkfMatchMode,
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
            expected_blocks_per_dc: 16_384,
            publish_every_n_events: 1,
            ckf_match_mode: CkfMatchMode::FullMap,
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

    pub fn transposed_ckf(num_event_workers: usize, expected_blocks_per_dc: usize) -> Self {
        Self {
            kind: MooncakeIndexerKind::TransposedCkf,
            num_event_workers,
            expected_blocks_per_dc,
            ..Self::radix_tree()
        }
    }

    pub fn with_publish_every_n_events(mut self, publish_every_n_events: usize) -> Self {
        self.publish_every_n_events = publish_every_n_events;
        self
    }

    pub fn with_ckf_match_mode(mut self, ckf_match_mode: CkfMatchMode) -> Self {
        self.ckf_match_mode = ckf_match_mode;
        self
    }

    pub(crate) fn build_transposed_ckf_backend(&self) -> anyhow::Result<EventTransposedCkfIndexer> {
        let workers = std::array::from_fn(|lane| WorkerWithDpRank::new(lane as u64, 0));
        let mut config = CkfConfig::new(self.expected_blocks_per_dc);
        config.publish_every_n_events = self.publish_every_n_events;
        Ok(EventTransposedCkfIndexer::new_with_match_mode(
            workers,
            config,
            self.ckf_match_mode,
        )?)
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
            MooncakeIndexerKind::TransposedCkf => match self.ckf_match_mode {
                CkfMatchMode::FullMap => "transposed-ckf",
                CkfMatchMode::MaxDepthMatches => "transposed-ckf-max-depth",
            },
            MooncakeIndexerKind::BranchShardedCrtc => "branch-sharded-crtc",
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
            "transposed-ckf" => Self::transposed_ckf(num_event_workers, 16_384),
            "branch-sharded-crtc" => Self::branch_sharded_crtc(2, num_event_workers, 2),
            _ => anyhow::bail!(
                "Unknown indexer '{}'. Valid names: radix-tree, nested-map, concurrent-radix-tree, concurrent-radix-tree-compressed, transposed-ckf, branch-sharded-crtc",
                name
            ),
        };
        Ok(config)
    }

    pub fn build(
        &self,
        block_size: u32,
        metrics: Arc<KvIndexerMetrics>,
    ) -> anyhow::Result<Arc<dyn KvIndexerInterface + Send + Sync>> {
        let indexer: Arc<dyn KvIndexerInterface + Send + Sync> = match self.kind {
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
            MooncakeIndexerKind::TransposedCkf => {
                let backend = self.build_transposed_ckf_backend()?;
                Arc::new(ThreadPoolIndexer::new_with_metrics(
                    backend,
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
        };
        Ok(indexer)
    }

    pub fn build_approximate_with_prune_config(
        &self,
        block_size: u32,
        metrics: Arc<KvIndexerMetrics>,
        prune_config: PruneConfig,
    ) -> anyhow::Result<Arc<dyn KvIndexerInterface + Send + Sync>> {
        let indexer: Arc<dyn KvIndexerInterface + Send + Sync> = match self.kind {
            MooncakeIndexerKind::RadixTree => Arc::new(KvIndexer::new_with_pruning(
                CancellationToken::new(),
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
            MooncakeIndexerKind::TransposedCkf => {
                anyhow::bail!("transposed-ckf does not support approximate pruning")
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
}

/// A single entry in a worker's merged benchmark timeline.
#[derive(Clone)]
pub(crate) enum WorkerTraceEntry {
    Request(Vec<LocalBlockHash>),
    Event {
        event: KvCacheEvent,
        storage_tier: StorageTier,
    },
}

/// A timestamped entry in a worker's benchmark trace, used to replay requests
/// and events at the correct relative timing.
#[derive(Clone)]
pub(crate) struct WorkerTrace {
    pub(crate) entry: WorkerTraceEntry,
    pub(crate) timestamp_us: u64,
}

pub(crate) struct MergedMooncakeBenchmark {
    worker_traces: WorkerTimelines<WorkerTrace>,
    block_size: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct MooncakeTraceTotals {
    pub(crate) requests: usize,
    pub(crate) stored_events: usize,
    pub(crate) removed_events: usize,
    pub(crate) cleared_events: usize,
    pub(crate) request_blocks: usize,
    pub(crate) stored_blocks: usize,
    pub(crate) removed_blocks: usize,
}

impl MooncakeTraceTotals {
    pub(crate) fn events(self) -> usize {
        self.stored_events + self.removed_events + self.cleared_events
    }

    pub(crate) fn event_blocks(self) -> usize {
        self.stored_blocks + self.removed_blocks
    }

    pub(crate) fn write_blocks(self) -> usize {
        self.event_blocks()
    }

    pub(crate) fn total_block_ops(self) -> usize {
        self.request_blocks + self.write_blocks()
    }

    pub(crate) fn expanded(self, factor: usize) -> Self {
        Self {
            requests: self.requests * factor,
            stored_events: self.stored_events * factor,
            removed_events: self.removed_events * factor,
            cleared_events: self.cleared_events * factor,
            request_blocks: self.request_blocks * factor,
            stored_blocks: self.stored_blocks * factor,
            removed_blocks: self.removed_blocks * factor,
        }
    }
}

#[derive(Clone)]
pub(crate) struct PreparedMooncakeBenchmark {
    pub(crate) worker_traces: WorkerTimelines<WorkerTrace>,
    pub(crate) totals: MooncakeTraceTotals,
    pub(crate) benchmark_duration_ms: u64,
    pub(crate) block_size: u32,
}

fn merge_event_worker_trace(
    worker_idx: usize,
    artifact: WorkerReplayArtifacts,
) -> anyhow::Result<Vec<WorkerTrace>> {
    let WorkerReplayArtifacts {
        requests,
        output_signals: _,
        kv_events,
    } = artifact;
    validate_timestamps(worker_idx, "requests", &requests, |request| {
        request.timestamp_us
    })?;
    validate_timestamps(worker_idx, "kv_events", &kv_events, |event| {
        event.timestamp_us
    })?;

    let mut requests = requests.into_iter().peekable();
    let mut events = kv_events.into_iter().peekable();
    let mut merged = Vec::with_capacity(requests.len() + events.len());
    while requests.peek().is_some() || events.peek().is_some() {
        let take_request = match (requests.peek(), events.peek()) {
            (Some(request), Some(event)) => request.timestamp_us <= event.timestamp_us,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => break,
        };
        if take_request {
            let request = requests.next().expect("peeked request must exist");
            merged.push(WorkerTrace {
                timestamp_us: request.timestamp_us,
                entry: WorkerTraceEntry::Request(request.replay_hashes.local_block_hashes),
            });
        } else {
            let event = events.next().expect("peeked event must exist");
            merged.push(WorkerTrace {
                timestamp_us: event.timestamp_us,
                entry: WorkerTraceEntry::Event {
                    event: event.event,
                    storage_tier: event.storage_tier,
                },
            });
        }
    }
    Ok(merged)
}

fn merge_event_worker_traces(
    artifacts: Vec<WorkerReplayArtifacts>,
) -> anyhow::Result<WorkerTimelines<WorkerTrace>> {
    let worker_traces = artifacts
        .into_iter()
        .enumerate()
        .map(|(worker_idx, artifact)| merge_event_worker_trace(worker_idx, artifact))
        .collect::<anyhow::Result<Vec<_>>>()?;

    Ok(WorkerTimelines::new(worker_traces))
}

fn validate_timestamps<T>(
    worker_idx: usize,
    source: &'static str,
    entries: &[T],
    timestamp_of: impl Fn(&T) -> u64,
) -> anyhow::Result<()> {
    for idx in 1..entries.len() {
        let previous = timestamp_of(&entries[idx - 1]);
        let current = timestamp_of(&entries[idx]);
        if current < previous {
            anyhow::bail!(
                "worker {worker_idx} {source} timestamps are not ordered at index {idx}: prev={previous}, curr={current}"
            );
        }
    }
    Ok(())
}

pub(crate) fn merge_worker_traces(
    artifacts: Vec<WorkerReplayArtifacts>,
    block_size: u32,
) -> anyhow::Result<MergedMooncakeBenchmark> {
    Ok(MergedMooncakeBenchmark {
        worker_traces: merge_event_worker_traces(artifacts)?,
        block_size,
    })
}

pub(crate) fn prepare_scaled_benchmark(
    merged: MergedMooncakeBenchmark,
    benchmark_duration_ms: u64,
) -> PreparedMooncakeBenchmark {
    let worker_traces = merged.worker_traces.into_rescaled_from_first(
        benchmark_duration_ms,
        |entry| entry.timestamp_us,
        |entry, timestamp_us| WorkerTrace {
            entry: entry.entry,
            timestamp_us,
        },
    );

    let mut totals = MooncakeTraceTotals::default();

    for entry in worker_traces.iter().flatten() {
        match &entry.entry {
            WorkerTraceEntry::Request(hashes) => {
                totals.requests += 1;
                totals.request_blocks += hashes.len();
            }
            WorkerTraceEntry::Event { event, .. } => match &event.data {
                KvCacheEventData::Stored(store) => {
                    totals.stored_events += 1;
                    totals.stored_blocks += store.blocks.len();
                }
                KvCacheEventData::Removed(remove) => {
                    totals.removed_events += 1;
                    totals.removed_blocks += remove.block_hashes.len();
                }
                KvCacheEventData::Cleared => totals.cleared_events += 1,
            },
        }
    }

    PreparedMooncakeBenchmark {
        worker_traces,
        totals,
        benchmark_duration_ms,
        block_size: merged.block_size,
    }
}
