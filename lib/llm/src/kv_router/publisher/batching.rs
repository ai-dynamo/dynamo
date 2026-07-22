// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::{Duration, Instant};

use dynamo_kv_router::RouterEventSink;
use dynamo_kv_router::indexer::LocalKvIndexer;
use dynamo_kv_router::protocols::{
    KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData, StorageTier,
};

use super::dedup::EventDedupFilter;
use super::sinks::emit;

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

    pub(super) async fn flush<P: RouterEventSink + Send + Sync + 'static>(
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
