// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Agentic workload preparation for `mooncake_bench`: load the row pool, capture every
//! worker's closed-loop replay in virtual time, then merge and rescale into the same
//! prepared structure the Mooncake open-loop replay consumes.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use dynamo_bench::kv_router_common::agentic::{
    AgenticCorpusConfig, AgenticPool, AgenticPrepReport, generate_agentic_artifacts,
};
use dynamo_bench::kv_router_common::replay::mock_engine_args_with_speedup;
use dynamo_kv_router::protocols::KvCacheEventData;
use dynamo_mocker::common::protocols::{EngineType, MockEngineArgs, SglangArgs};

use super::mooncake_shared::{
    PreparedMooncakeBenchmark, WarmupEvent, WorkerTraceEntry, merge_worker_traces,
    prepare_scaled_benchmark_global,
};
use super::scaling_diag::PrepTimings;

/// Engine settings for the closed-loop capture.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AgenticEngine {
    pub(crate) num_gpu_blocks: usize,
    pub(crate) block_size: u32,
    pub(crate) speedup_ratio: f64,
    /// `vllm` keeps the original capture; `sglang` uses the radix-cache engine with
    /// `page_size = block_size`.
    pub(crate) sglang: bool,
}

impl AgenticEngine {
    fn mock_engine_args(self) -> anyhow::Result<MockEngineArgs> {
        if !self.sglang {
            return mock_engine_args_with_speedup(
                self.num_gpu_blocks,
                self.block_size as usize,
                self.speedup_ratio,
            );
        }
        Ok(MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .sglang(Some(SglangArgs {
                page_size: Some(self.block_size as usize),
                ..SglangArgs::default()
            }))
            .num_gpu_blocks(self.num_gpu_blocks)
            .block_size(self.block_size as usize)
            .speedup_ratio(self.speedup_ratio)
            .enable_prefix_caching(true)
            .max_num_batched_tokens(None)
            .max_num_seqs(None)
            .build()?)
    }
}

/// Load the pool (failing closed on an expected SHA-256), capture, merge, and rescale.
pub(crate) async fn prepare_agentic_benchmark(
    pool_path: &Path,
    expected_pool_sha256: Option<&str>,
    config: &AgenticCorpusConfig,
    engine: AgenticEngine,
    benchmark_duration_ms: u64,
) -> anyhow::Result<(PreparedMooncakeBenchmark, PrepTimings, AgenticPrepReport)> {
    let mut timings = PrepTimings::default();
    let started = Instant::now();
    let (pool, pool_sha256) = AgenticPool::read(pool_path)?;
    if let Some(expected) = expected_pool_sha256
        && !expected.eq_ignore_ascii_case(&pool_sha256)
    {
        anyhow::bail!("agentic pool sha256 {pool_sha256} != expected {expected}");
    }
    timings.trace_load_ms = started.elapsed().as_secs_f64() * 1e3;
    let mut report = AgenticPrepReport {
        workload: "agentic",
        pool_path: pool_path.display().to_string(),
        pool_sha256,
        pool_source_sha256: pool.source_sha256.clone(),
        pool_plays: pool.plays.len(),
        pool_requests: pool.request_count(),
        pool_block_size: pool.header.block_size,
        pool_source_digest: pool.header.source.digest.clone(),
        nested_timestamp_basis: pool.nested_timestamp_basis.clone(),
        ..AgenticPrepReport::default()
    };

    let started = Instant::now();
    let engine_args = engine.mock_engine_args()?;
    report.engine_type = if engine.sglang { "sglang" } else { "vllm" }.to_string();
    let (artifacts, warmups) =
        generate_agentic_artifacts(Arc::new(pool), config, engine_args, &mut report).await?;
    timings.simulation_ms = started.elapsed().as_secs_f64() * 1e3;

    let started = Instant::now();
    let warmup_events = merge_warmup_events(warmups);
    let merged = merge_worker_traces(artifacts, engine.block_size)?;
    let mut prepared = prepare_scaled_benchmark_global(merged, benchmark_duration_ms);
    prepared.warmup_events = warmup_events;
    check_worker_order(&prepared)?;
    report.merged_corpus_digest = prepared_corpus_digest(&prepared);
    timings.merge_and_rescale_ms = started.elapsed().as_secs_f64() * 1e3;
    Ok((prepared, timings, report))
}

/// Merge per-worker warm-up events into one list ordered by (virtual time, worker, source
/// order); per-worker order is what the indexer needs, the global order mimics arrival.
fn merge_warmup_events(
    warmups: Vec<Vec<dynamo_mocker::replay::ReplayTimedKvEvent>>,
) -> Vec<WarmupEvent> {
    let mut keyed = Vec::with_capacity(warmups.iter().map(Vec::len).sum());
    for (worker, events) in warmups.into_iter().enumerate() {
        for (ordinal, event) in events.into_iter().enumerate() {
            keyed.push((event.timestamp_us, worker, ordinal, event));
        }
    }
    keyed.sort_unstable_by_key(|(timestamp_us, worker, ordinal, _)| {
        (*timestamp_us, *worker, *ordinal)
    });
    keyed
        .into_iter()
        .map(|(_, worker, _, event)| WarmupEvent {
            worker,
            event: event.event,
            storage_tier: event.storage_tier,
        })
        .collect()
}

/// Fail closed unless every worker's merged timeline is time-ordered.
pub(crate) fn check_worker_order(prepared: &PreparedMooncakeBenchmark) -> anyhow::Result<()> {
    for (worker, trace) in prepared.worker_traces.iter().enumerate() {
        if let Some(index) = trace
            .windows(2)
            .position(|pair| pair[1].timestamp_us < pair[0].timestamp_us)
        {
            anyhow::bail!(
                "worker {worker} merged timeline is out of order at entry {}",
                index + 1
            );
        }
    }
    Ok(())
}

/// xxh3 digest of a prepared corpus: every worker's timestamps, query hashes, and events.
/// Two preparations with equal digests replay identical operations.
pub(crate) fn prepared_corpus_digest(prepared: &PreparedMooncakeBenchmark) -> String {
    let mut hasher = xxhash_rust::xxh3::Xxh3::new();
    let mut put = |value: u64| hasher.update(&value.to_le_bytes());
    put(prepared.benchmark_duration_ms);
    put(u64::from(prepared.block_size));
    for (worker, trace) in prepared.worker_traces.iter().enumerate() {
        put(worker as u64);
        put(trace.len() as u64);
        for entry in trace {
            put(entry.timestamp_us);
            match &entry.entry {
                WorkerTraceEntry::Request(hashes) => {
                    put(0);
                    put(hashes.len() as u64);
                    hashes.iter().for_each(|hash| put(hash.0));
                }
                WorkerTraceEntry::Event {
                    event,
                    storage_tier,
                } => {
                    put(1);
                    put(event.event_id);
                    put(u64::from(event.dp_rank));
                    put(u64::from(storage_tier.is_gpu()));
                    match &event.data {
                        KvCacheEventData::Stored(store) => {
                            put(2);
                            put(store.parent_hash.map_or(u64::MAX, |hash| hash.0));
                            put(store.start_position.map_or(u64::MAX, u64::from));
                            put(store.blocks.len() as u64);
                            for block in &store.blocks {
                                put(block.block_hash.0);
                                put(block.tokens_hash.0);
                            }
                        }
                        KvCacheEventData::Removed(remove) => {
                            put(3);
                            put(remove.block_hashes.len() as u64);
                            remove.block_hashes.iter().for_each(|hash| put(hash.0));
                        }
                        KvCacheEventData::Cleared => put(4),
                    }
                }
            }
        }
    }
    put(prepared.warmup_events.len() as u64);
    for warmup in &prepared.warmup_events {
        put(warmup.worker as u64);
        put(warmup.event.event_id);
        match &warmup.event.data {
            KvCacheEventData::Stored(store) => {
                put(2);
                put(store.parent_hash.map_or(u64::MAX, |hash| hash.0));
                put(store.blocks.len() as u64);
                for block in &store.blocks {
                    put(block.block_hash.0);
                    put(block.tokens_hash.0);
                }
            }
            KvCacheEventData::Removed(remove) => {
                put(3);
                put(remove.block_hashes.len() as u64);
                remove.block_hashes.iter().for_each(|hash| put(hash.0));
            }
            KvCacheEventData::Cleared => put(4),
        }
    }
    format!("{:016x}", hasher.digest())
}
