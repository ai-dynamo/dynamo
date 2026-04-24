// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use dynamo_kv_router::RouterEventSink;
use dynamo_kv_router::indexer::LocalKvIndexer;
use dynamo_kv_router::protocols::*;
use dynamo_runtime::transports::event_plane::EventPublisher;
use dynamo_runtime::transports::nats::NatsQueue;

use crate::kv_router::KV_EVENT_SUBJECT;

use super::{DEFAULT_MAX_BATCH_BLOCKS, kv_publisher_metrics};

/// Reference-counting filter that deduplicates KV cache events.
///
/// vLLM can emit multiple store/remove events for the same block hash.
/// Refcounts are tracked **per DP rank** because identical block hashes
/// on different ranks represent independent blocks.
///
/// - **Store**: always passes through; increments refcount for the rank.
/// - **Remove**: only passes through when refcount decrements to 0.
/// - **Cleared**: resets refcounts for all ranks.
pub(super) struct EventDedupFilter {
    /// Per-(dp_rank, storage_tier) refcounts.
    per_rank_tier: HashMap<(u32, StorageTier), HashMap<ExternalSequenceBlockHash, usize>>,
}

impl EventDedupFilter {
    pub(super) fn new() -> Self {
        Self {
            per_rank_tier: HashMap::new(),
        }
    }

    /// Track a store event. Increments refcount for each block hash on the
    /// given (DP rank, storage tier). Stores always pass through — this only
    /// updates bookkeeping.
    pub(super) fn track_store(
        &mut self,
        dp_rank: u32,
        storage_tier: StorageTier,
        data: &KvCacheStoreData,
    ) {
        let refcounts = self
            .per_rank_tier
            .entry((dp_rank, storage_tier))
            .or_default();
        for block in &data.blocks {
            *refcounts.entry(block.block_hash).or_insert(0) += 1;
        }
    }

    /// Filter a remove event. Retains only block hashes whose refcount on the
    /// given (DP rank, storage tier) decrements to 0 (removing them from the
    /// map). Returns `None` if no hashes survive filtering.
    pub(super) fn filter_remove(
        &mut self,
        dp_rank: u32,
        storage_tier: StorageTier,
        mut data: KvCacheRemoveData,
    ) -> Option<KvCacheRemoveData> {
        let refcounts = self
            .per_rank_tier
            .entry((dp_rank, storage_tier))
            .or_default();
        data.block_hashes.retain(|hash| {
            match refcounts.entry(*hash) {
                Entry::Occupied(mut entry) => {
                    *entry.get_mut() -= 1;
                    if *entry.get() == 0 {
                        entry.remove();
                        true // refcount hit 0 → pass through
                    } else {
                        false // still has references → filter out
                    }
                }
                Entry::Vacant(_) => {
                    true // not tracked → pass through defensively
                }
            }
        });
        if data.block_hashes.is_empty() {
            None
        } else {
            Some(data)
        }
    }

    /// Clear refcounts for all DP ranks and tiers. A `Cleared` event from any
    /// rank causes the indexer to wipe all blocks for the entire worker, so we
    /// must reset all refcounts to stay consistent.
    pub(super) fn clear(&mut self) {
        self.per_rank_tier.clear();
    }
}

/// Accumulator for in-flight KV cache events that will be merged into a single
/// [`RouterEvent`] before being forwarded to the event sink.
#[derive(Debug)]
pub(super) struct BatchingState {
    pub(super) pending_removed: Option<KvCacheRemoveData>,
    pub(super) pending_stored: Option<KvCacheStoreData>,
    pub(super) next_publish_id: u64,
    pub(super) last_dp_rank: u32,
    pub(super) last_storage_tier: StorageTier,
    pub(super) last_flush_time: Instant,
}

impl BatchingState {
    pub(super) fn new() -> Self {
        Self {
            pending_removed: None,
            pending_stored: None,
            next_publish_id: 1,
            last_dp_rank: 0,
            last_storage_tier: StorageTier::Device,
            last_flush_time: Instant::now(),
        }
    }

    pub(super) fn has_pending(&self) -> bool {
        self.pending_removed.is_some() || self.pending_stored.is_some()
    }

    pub(super) fn pending_block_count(&self) -> usize {
        self.pending_removed
            .as_ref()
            .map(|r| r.block_hashes.len())
            .unwrap_or(0)
            + self
                .pending_stored
                .as_ref()
                .map(|s| s.blocks.len())
                .unwrap_or(0)
    }

    pub(super) fn record_flush_time(&mut self) {
        self.last_flush_time = Instant::now();
    }

    pub(super) fn remaining_timeout(&self, timeout_ms: u64) -> Duration {
        let timeout = Duration::from_millis(timeout_ms);
        let elapsed = self.last_flush_time.elapsed();
        if elapsed >= timeout {
            Duration::ZERO
        } else {
            timeout - elapsed
        }
    }

    pub(super) fn is_timeout_elapsed(&self, timeout_ms: u64) -> bool {
        self.remaining_timeout(timeout_ms) == Duration::ZERO
    }

    async fn flush<P: RouterEventSink + Send + Sync + 'static>(
        &mut self,
        publisher: &P,
        local_indexer: &Option<Arc<LocalKvIndexer>>,
        worker_id: u64,
        dedup: &mut EventDedupFilter,
    ) {
        if !self.has_pending() {
            return;
        }
        let dp_rank = self.last_dp_rank;
        let mut emitted = false;
        if let Some(data) = self.pending_removed.take()
            && let Some(filtered) = dedup.filter_remove(dp_rank, self.last_storage_tier, data)
        {
            emit(
                publisher,
                local_indexer,
                worker_id,
                self.last_storage_tier,
                KvCacheEvent {
                    event_id: self.next_publish_id,
                    data: KvCacheEventData::Removed(filtered),
                    dp_rank,
                },
            )
            .await;
            emitted = true;
        }
        if let Some(data) = self.pending_stored.take() {
            dedup.track_store(dp_rank, self.last_storage_tier, &data);
            emit(
                publisher,
                local_indexer,
                worker_id,
                self.last_storage_tier,
                KvCacheEvent {
                    event_id: self.next_publish_id,
                    data: KvCacheEventData::Stored(data),
                    dp_rank,
                },
            )
            .await;
            emitted = true;
        }
        if emitted {
            self.next_publish_id += 1;
        }
        self.record_flush_time();
    }
}

pub(super) struct EventPlanePublisher(pub(super) EventPublisher);

impl RouterEventSink for EventPlanePublisher {
    fn publish_event(&self, event: &RouterEvent) -> impl Future<Output = Result<()>> + Send {
        self.0.publish(event)
    }
}

pub(super) struct JetStreamPublisher(pub(super) NatsQueue);

impl RouterEventSink for JetStreamPublisher {
    fn publish_event(&self, event: &RouterEvent) -> impl Future<Output = Result<()>> + Send {
        NatsQueue::publish_event(&self.0, KV_EVENT_SUBJECT, event)
    }
}

async fn emit<P: RouterEventSink>(
    publisher: &P,
    local_indexer: &Option<Arc<LocalKvIndexer>>,
    worker_id: u64,
    storage_tier: StorageTier,
    event: KvCacheEvent,
) {
    let router_event = RouterEvent::with_storage_tier(worker_id, event, storage_tier);
    if let Some(indexer) = local_indexer
        && let Err(e) = indexer.apply_event_with_buffer(router_event.clone()).await
    {
        tracing::warn!(worker_id, error = %e, "Failed to apply event to local indexer");
    }
    if let Err(e) = publisher.publish_event(&router_event).await {
        tracing::error!(worker_id, error = %e, "Failed to publish event");
    }
}

pub(super) async fn run_event_processor_loop<P: RouterEventSink + Send + Sync + 'static>(
    publisher: P,
    worker_id: u64,
    cancellation_token: CancellationToken,
    mut rx: mpsc::UnboundedReceiver<PlacementEvent>,
    local_indexer: Option<Arc<LocalKvIndexer>>,
    timeout_ms: Option<u64>,
    max_batch_blocks: usize,
) {
    let mut batching_state = BatchingState::new();
    let mut dedup = EventDedupFilter::new();
    let mut last_raw_input_id: Option<u64> = None;

    loop {
        tokio::select! {
            _ = cancellation_token.cancelled() => {
                tracing::info!("KV Event source received cancellation signal");
                batching_state.flush(&publisher, &local_indexer, worker_id, &mut dedup).await;
                break;
            }
            event = rx.recv() => {
                let Some(placement_event) = event else {
                    tracing::debug!("Event processor channel closed.");
                    batching_state.flush(&publisher, &local_indexer, worker_id, &mut dedup).await;
                    break;
                };

                let raw_event_id = placement_event.event.event_id;
                if let Some(last_id) = last_raw_input_id
                    && raw_event_id > last_id + 1
                {
                    let gap = raw_event_id - last_id - 1;
                    tracing::warn!(
                        worker_id,
                        last_raw_input_id = last_id,
                        raw_event_id,
                        gap,
                        "Input event gap detected: raw events dropped before batching"
                    );
                    if let Some(metrics) = kv_publisher_metrics() {
                        metrics.increment_engines_dropped_events(gap);
                    } else {
                        tracing::warn!(
                            worker_id,
                            gap,
                            "Failed to record dropped events metric: metrics not initialized"
                        );
                    }
                }
                last_raw_input_id = Some(raw_event_id);

                let storage_tier = placement_event.placement.tier;
                let event = placement_event.event;
                tracing::trace!(
                    "Event processor for worker_id {} processing event: {:?}",
                    worker_id,
                    event.data
                );

                let dp_rank_changed =
                    batching_state.has_pending() && event.dp_rank != batching_state.last_dp_rank;
                let storage_tier_changed =
                    batching_state.has_pending() && storage_tier != batching_state.last_storage_tier;

                match event.data {
                    KvCacheEventData::Removed(data) => {
                        if batching_state.pending_stored.is_some()
                            || dp_rank_changed
                            || storage_tier_changed
                        {
                            batching_state.flush(&publisher, &local_indexer, worker_id, &mut dedup).await;
                        }
                        match &mut batching_state.pending_removed {
                            Some(pending) => pending.block_hashes.extend(data.block_hashes),
                            None => {
                                batching_state.pending_removed = Some(data);
                            }
                        }
                    }
                    KvCacheEventData::Stored(data) => {
                        let should_flush = dp_rank_changed
                            || storage_tier_changed
                            || batching_state.pending_removed.is_some()
                            || batching_state.pending_stored.as_ref().is_some_and(|p| {
                                data.parent_hash != p.blocks.last().map(|b| b.block_hash)
                            });
                        if should_flush {
                            batching_state.flush(&publisher, &local_indexer, worker_id, &mut dedup).await;
                        }
                        match &mut batching_state.pending_stored {
                            Some(pending) => pending.blocks.extend(data.blocks),
                            None => {
                                batching_state.pending_stored = Some(data);
                            }
                        }
                    }
                    KvCacheEventData::Cleared => {
                        batching_state.flush(&publisher, &local_indexer, worker_id, &mut dedup).await;
                        dedup.clear();
                        emit(
                            &publisher,
                            &local_indexer,
                            worker_id,
                            storage_tier,
                            KvCacheEvent {
                                event_id: batching_state.next_publish_id,
                                data: KvCacheEventData::Cleared,
                                dp_rank: event.dp_rank,
                            },
                        )
                        .await;
                        batching_state.next_publish_id += 1;
                    }
                }

                batching_state.last_dp_rank = event.dp_rank;
                batching_state.last_storage_tier = storage_tier;

                if batching_state.has_pending()
                    && (timeout_ms.is_none_or(|ms| batching_state.is_timeout_elapsed(ms))
                        || batching_state.pending_block_count() > max_batch_blocks)
                {
                    batching_state.flush(&publisher, &local_indexer, worker_id, &mut dedup).await;
                }
            }
            _ = tokio::time::sleep(
                timeout_ms
                    .map(|ms| batching_state.remaining_timeout(ms))
                    .unwrap_or(Duration::from_secs(3600))
            ), if timeout_ms.is_some() && batching_state.has_pending() => {
                batching_state.flush(&publisher, &local_indexer, worker_id, &mut dedup).await;
            }
        }
    }
}

pub(super) async fn start_event_processor<P: RouterEventSink + Send + Sync + 'static>(
    publisher: P,
    worker_id: u64,
    cancellation_token: CancellationToken,
    rx: mpsc::UnboundedReceiver<PlacementEvent>,
    local_indexer: Option<Arc<LocalKvIndexer>>,
    batching_timeout_ms: Option<u64>,
) {
    run_event_processor_loop(
        publisher,
        worker_id,
        cancellation_token,
        rx,
        local_indexer,
        batching_timeout_ms,
        DEFAULT_MAX_BATCH_BLOCKS,
    )
    .await
}

pub(super) async fn start_event_processor_jetstream(
    publisher: NatsQueue,
    worker_id: u64,
    cancellation_token: CancellationToken,
    rx: mpsc::UnboundedReceiver<PlacementEvent>,
    local_indexer: Option<Arc<LocalKvIndexer>>,
    batching_timeout_ms: Option<u64>,
) {
    run_event_processor_loop(
        JetStreamPublisher(publisher),
        worker_id,
        cancellation_token,
        rx,
        local_indexer,
        batching_timeout_ms,
        DEFAULT_MAX_BATCH_BLOCKS,
    )
    .await
}
