// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use anyhow::Context;
use dynamo_kv_router::ConcurrentRadixTreeCompressed;
use dynamo_kv_router::LocalBlockHash;
use dynamo_kv_router::ThreadPoolIndexer;
use dynamo_kv_router::indexer::concurrent_radix_tree_compressed::CrtcStructureStats;
use dynamo_kv_router::indexer::{
    EventEnqueueCompletion, KvIndexerInterface, KvRouterError, SyncIndexer,
};
use dynamo_kv_router::protocols::{KvCacheEventData, OverlapScores, RouterEvent};

use crate::corpus::{ballast_family_count, ballast_store_event};
use crate::types::{CorpusHeader, PipelineErrors, PreparedCorpus, QueueMetrics};

pub struct CrtcStats {
    pub raw_events: AtomicU64,
    pub raw_blocks: AtomicU64,
    pub maximum_queue_depth: AtomicU64,
    completion_sender: flume::Sender<EventEnqueueCompletion>,
    completion_receiver: flume::Receiver<EventEnqueueCompletion>,
}

impl Default for CrtcStats {
    fn default() -> Self {
        let (completion_sender, completion_receiver) = flume::unbounded();
        Self {
            raw_events: AtomicU64::new(0),
            raw_blocks: AtomicU64::new(0),
            maximum_queue_depth: AtomicU64::new(0),
            completion_sender,
            completion_receiver,
        }
    }
}

pub struct CrtcBackend {
    indexer: Arc<ThreadPoolIndexer<ConcurrentRadixTreeCompressed>>,
    pub stats: Arc<CrtcStats>,
}

pub struct CrtcCompletionMetrics {
    pub drain_completed_at: Instant,
    pub scheduled_to_enqueue_ns: Vec<u64>,
    pub enqueue_to_applied_ns: Vec<u64>,
    pub scheduled_to_applied_ns: Vec<u64>,
    pub errors: PipelineErrors,
}

impl CrtcBackend {
    pub async fn from_corpus(corpus: &PreparedCorpus) -> anyhow::Result<Arc<Self>> {
        Self::from_header(&corpus.header).await
    }

    pub async fn from_header(header: &CorpusHeader) -> anyhow::Result<Arc<Self>> {
        let indexer = Arc::new(ThreadPoolIndexer::new(
            ConcurrentRadixTreeCompressed::new(),
            header.event_threads,
            header.block_size,
        ));
        indexer.pre_register_bench_workers((0..header.num_dcs).map(|dc| dc as u64));
        let mut event_id = 1u64 << 63;
        for dc in 0..header.num_dcs {
            for family in 0..ballast_family_count(&header.ballast) {
                indexer
                    .apply_event(ballast_store_event(dc, &header.ballast, family, event_id))
                    .await;
                event_id += 1;
            }
        }
        <ThreadPoolIndexer<ConcurrentRadixTreeCompressed> as KvIndexerInterface>::flush(&*indexer)
            .await;
        Ok(Arc::new(Self {
            indexer,
            stats: Arc::new(CrtcStats::default()),
        }))
    }

    pub fn submit(
        &self,
        event: &RouterEvent,
        logical_event_id: u64,
        scheduled_at: Instant,
    ) -> anyhow::Result<()> {
        let (_, depth) = self
            .indexer
            .enqueue_event_with_completion(
                event.clone(),
                logical_event_id,
                scheduled_at,
                self.stats.completion_sender.clone(),
            )
            .map_err(kv_error)?;
        self.stats.raw_events.fetch_add(1, Ordering::Relaxed);
        self.stats.raw_blocks.fetch_add(
            event_block_count(&event.event.data) as u64,
            Ordering::Relaxed,
        );
        update_max(&self.stats.maximum_queue_depth, depth);
        Ok(())
    }

    pub fn reset_stats(&self) {
        self.stats.raw_events.store(0, Ordering::Relaxed);
        self.stats.raw_blocks.store(0, Ordering::Relaxed);
        self.stats.maximum_queue_depth.store(0, Ordering::Relaxed);
        while self.stats.completion_receiver.try_recv().is_ok() {}
    }

    pub fn lookup(&self, local_hashes: &[LocalBlockHash]) -> OverlapScores {
        self.indexer.backend().find_matches(local_hashes, false)
    }

    pub fn touch_for_benchmark(&self) {
        self.indexer.backend().touch_for_benchmark();
    }

    pub fn structure_stats(&self) -> CrtcStructureStats {
        self.indexer.backend().structure_stats()
    }

    pub async fn flush(&self) -> anyhow::Result<(QueueMetrics, CrtcCompletionMetrics)> {
        let at_stop = self.indexer.bench_queue_depth() as u64;
        let started = Instant::now();
        <ThreadPoolIndexer<ConcurrentRadixTreeCompressed> as KvIndexerInterface>::flush(
            &*self.indexer,
        )
        .await;
        let drain_completed_at = Instant::now();
        let drain_ms = started.elapsed().as_secs_f64() * 1000.0;
        let expected = self.stats.raw_events.load(Ordering::Relaxed) as usize;
        let mut scheduled_to_enqueue_ns = Vec::with_capacity(expected);
        let mut enqueue_to_applied_ns = Vec::with_capacity(expected);
        let mut scheduled_to_applied_ns = Vec::with_capacity(expected);
        let mut errors = PipelineErrors::default();
        for _ in 0..expected {
            let completion = self
                .stats
                .completion_receiver
                .recv_timeout(std::time::Duration::from_secs(5))
                .context("missing CRTC event completion")?;
            scheduled_to_enqueue_ns.push(
                completion
                    .metadata
                    .enqueued_at
                    .saturating_duration_since(completion.metadata.scheduled_at)
                    .as_nanos() as u64,
            );
            enqueue_to_applied_ns.push(
                completion
                    .applied_at
                    .saturating_duration_since(completion.metadata.enqueued_at)
                    .as_nanos() as u64,
            );
            scheduled_to_applied_ns.push(
                completion
                    .applied_at
                    .saturating_duration_since(completion.metadata.scheduled_at)
                    .as_nanos() as u64,
            );
            if !completion.applied {
                errors.application += 1;
            }
        }
        Ok((
            QueueMetrics {
                at_stop,
                maximum_depth: self.stats.maximum_queue_depth.load(Ordering::Relaxed),
                drain_ms,
            },
            CrtcCompletionMetrics {
                drain_completed_at,
                scheduled_to_enqueue_ns,
                enqueue_to_applied_ns,
                scheduled_to_applied_ns,
                errors,
            },
        ))
    }

    pub fn shutdown(&self) {
        self.indexer.shutdown();
    }
}

fn kv_error(error: KvRouterError) -> anyhow::Error {
    anyhow::anyhow!(error.to_string())
}

fn event_block_count(data: &KvCacheEventData) -> usize {
    match data {
        KvCacheEventData::Stored(store) => store.blocks.len(),
        KvCacheEventData::Removed(remove) => remove.block_hashes.len(),
        KvCacheEventData::Cleared => 0,
    }
}

fn update_max(counter: &AtomicU64, value: usize) {
    let value = value as u64;
    let mut current = counter.load(Ordering::Relaxed);
    while current < value {
        match counter.compare_exchange_weak(current, value, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(actual) => current = actual,
        }
    }
}
