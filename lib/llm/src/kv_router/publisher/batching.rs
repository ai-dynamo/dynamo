// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::{Duration, Instant};

use dynamo_kv_router::indexer::LocalKvIndexer;
use dynamo_kv_router::protocols::{KvCacheEventData, Placement, PlacementEvent, RouterEvent};

use super::dedup::{EventDedupFilter, EventDedupPolicy};
use super::sinks::emit;

/// Pure accumulator for adjacent compatible placement mutations.
///
/// The caller-provided key carries the exact logical owner/source and physical
/// tier. The coalescer adds DP-rank, operation, and Store-chain boundaries. It
/// deliberately owns no event IDs, timers, indexers, publishers, or error
/// policy, so both publisher lifecycles can share it without sharing commits.
#[derive(Debug)]
pub(super) struct PlacementEventCoalescer<K> {
    pending: Option<(K, PlacementEvent)>,
    max_batch_blocks: usize,
}

impl<K: Eq> PlacementEventCoalescer<K> {
    pub(super) fn new(max_batch_blocks: usize) -> Self {
        Self {
            pending: None,
            max_batch_blocks,
        }
    }

    pub(super) fn has_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Push one input and return up to two ready outputs in stream order.
    ///
    /// Two outputs are possible when an incompatible or oversized mutation
    /// follows a pending mutation, or when `Cleared` flushes the pending value
    /// and then passes through as its own barrier.
    pub(super) fn push(&mut self, key: K, event: PlacementEvent) -> [Option<PlacementEvent>; 2] {
        if matches!(&event.event.data, KvCacheEventData::Cleared) {
            return [self.flush(), Some(event)];
        }

        let can_merge = self.pending.as_ref().is_some_and(|(pending_key, pending)| {
            pending_key == &key
                && pending.event.dp_rank == event.event.dp_rank
                && compatible_data(&pending.event.data, &event.event.data)
        });
        if can_merge {
            let pending = &mut self
                .pending
                .as_mut()
                .expect("merge compatibility requires one pending event")
                .1;
            merge_data(&mut pending.event.data, event.event.data);
            if event_block_count(pending) >= self.max_batch_blocks {
                return [self.flush(), None];
            }
            return [None, None];
        }

        let flushed = self.flush();
        if event_block_count(&event) >= self.max_batch_blocks {
            return [flushed, Some(event)];
        }
        self.pending = Some((key, event));
        [flushed, None]
    }

    pub(super) fn flush(&mut self) -> Option<PlacementEvent> {
        self.pending.take().map(|(_, event)| event)
    }
}

fn merge_data(pending: &mut KvCacheEventData, next: KvCacheEventData) {
    match (pending, next) {
        (KvCacheEventData::Stored(pending), KvCacheEventData::Stored(next)) => {
            pending.blocks.extend(next.blocks);
        }
        (KvCacheEventData::Removed(pending), KvCacheEventData::Removed(next)) => {
            pending.block_hashes.extend(next.block_hashes);
        }
        _ => unreachable!("merge compatibility requires matching mutation kinds"),
    }
}

fn compatible_data(pending: &KvCacheEventData, next: &KvCacheEventData) -> bool {
    match (pending, next) {
        (KvCacheEventData::Stored(pending), KvCacheEventData::Stored(next)) => {
            next.parent_hash == pending.blocks.last().map(|block| block.block_hash)
        }
        (KvCacheEventData::Removed(_), KvCacheEventData::Removed(_)) => true,
        _ => false,
    }
}

fn event_block_count(event: &PlacementEvent) -> usize {
    match &event.event.data {
        KvCacheEventData::Stored(data) => data.blocks.len(),
        KvCacheEventData::Removed(data) => data.block_hashes.len(),
        KvCacheEventData::Cleared => 0,
    }
}

/// Accumulator for in-flight KV cache events that will be merged into a single
/// [`RouterEvent`] before being forwarded to the event sink.
#[derive(Debug)]
pub(super) struct BatchingState {
    coalescer: PlacementEventCoalescer<Placement>,
    pub(super) next_publish_id: u64,
    pub(super) last_flush_time: Instant,
}

impl BatchingState {
    pub(super) fn new(max_batch_blocks: usize) -> Self {
        Self {
            coalescer: PlacementEventCoalescer::new(max_batch_blocks),
            next_publish_id: 1,
            last_flush_time: Instant::now(),
        }
    }

    pub(super) fn has_pending(&self) -> bool {
        self.coalescer.has_pending()
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

    pub(super) async fn flush(
        &mut self,
        local_indexer: &Option<Arc<LocalKvIndexer>>,
        worker_id: u64,
        dedup: &mut EventDedupFilter,
        output: &mut Vec<RouterEvent>,
    ) {
        if let Some(event) = self.coalescer.flush() {
            self.emit_ready(event, local_indexer, worker_id, dedup, output)
                .await;
        }
        self.record_flush_time();
    }

    pub(super) async fn push(
        &mut self,
        event: PlacementEvent,
        local_indexer: &Option<Arc<LocalKvIndexer>>,
        worker_id: u64,
        dedup: &mut EventDedupFilter,
        output: &mut Vec<RouterEvent>,
    ) {
        let key = event.placement.clone();
        let ready = self.coalescer.push(key, event);
        let flushed = ready.iter().any(Option::is_some);
        for ready in ready.into_iter().flatten() {
            self.emit_ready(ready, local_indexer, worker_id, dedup, output)
                .await;
        }
        if flushed {
            self.record_flush_time();
        }
    }

    async fn emit_ready(
        &mut self,
        placement_event: PlacementEvent,
        local_indexer: &Option<Arc<LocalKvIndexer>>,
        worker_id: u64,
        dedup: &mut EventDedupFilter,
        output: &mut Vec<RouterEvent>,
    ) {
        let tier = placement_event.placement.tier;
        let domain = placement_event.placement.residency_domain;
        let mut event = placement_event.event;
        event.data = match event.data {
            KvCacheEventData::Removed(data) => {
                let Some(filtered) = dedup.filter_remove_in_domain(
                    event.dp_rank,
                    tier,
                    domain,
                    EventDedupPolicy::RefCounted,
                    data,
                ) else {
                    return;
                };
                KvCacheEventData::Removed(filtered)
            }
            KvCacheEventData::Stored(data) => {
                dedup.track_store_in_domain(
                    event.dp_rank,
                    tier,
                    domain,
                    EventDedupPolicy::RefCounted,
                    &data,
                );
                KvCacheEventData::Stored(data)
            }
            KvCacheEventData::Cleared => {
                unreachable!("Cleared is handled by the publisher's barrier policy")
            }
        };
        event.event_id = self.next_publish_id;
        let _ = emit(local_indexer, worker_id, tier, domain, event, output).await;
        self.next_publish_id = self
            .next_publish_id
            .checked_add(1)
            .expect("KV event publisher outbound cursor exhausted");
    }
}

#[cfg(test)]
mod tests {
    use dynamo_kv_router::protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheRemoveData, KvCacheStoreData,
        KvCacheStoredBlockData, LocalBlockHash, Placement, ResidencyDomain, StorageTier,
    };

    use super::*;

    fn event(data: KvCacheEventData) -> PlacementEvent {
        PlacementEvent::new(
            Placement::local_worker(7, 0, StorageTier::HostPinned),
            KvCacheEvent {
                event_id: 0,
                data,
                dp_rank: 0,
            },
        )
    }

    fn stored(parent: Option<u64>, block: u64) -> PlacementEvent {
        event(KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash: parent.map(ExternalSequenceBlockHash),
            start_position: None,
            blocks: vec![KvCacheStoredBlockData {
                block_hash: ExternalSequenceBlockHash(block),
                tokens_hash: LocalBlockHash(block),
                mm_extra_info: None,
            }],
        }))
    }

    fn removed(block: u64) -> PlacementEvent {
        event(KvCacheEventData::Removed(KvCacheRemoveData {
            block_hashes: vec![ExternalSequenceBlockHash(block)],
        }))
    }

    #[test]
    fn coalesces_legacy_mutations_and_keeps_clear_as_a_boundary() {
        let mut coalescer = PlacementEventCoalescer::new(128);
        let mut output = Vec::new();
        for input in [
            stored(None, 1),
            stored(Some(1), 2),
            removed(1),
            removed(2),
            event(KvCacheEventData::Cleared),
        ] {
            let key = input.placement.clone();
            output.extend(coalescer.push(key, input).into_iter().flatten());
        }
        output.extend(coalescer.flush());

        assert_eq!(output.len(), 3);
        assert!(matches!(
            &output[0].event.data,
            KvCacheEventData::Stored(data) if data.blocks.len() == 2
        ));
        assert!(matches!(
            &output[1].event.data,
            KvCacheEventData::Removed(data) if data.block_hashes.len() == 2
        ));
        assert!(matches!(output[2].event.data, KvCacheEventData::Cleared));
    }

    #[test]
    fn exact_source_key_prevents_cross_owner_coalescing() {
        let mut coalescer = PlacementEventCoalescer::new(128);
        let first = stored(None, 1);
        assert!(
            coalescer
                .push(ResidencyDomain::Worker, first)
                .into_iter()
                .flatten()
                .next()
                .is_none()
        );
        let second = stored(Some(1), 2);
        let output = coalescer
            .push(ResidencyDomain::CacheOwner, second)
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        assert_eq!(output.len(), 1);
        assert!(matches!(
            &output[0].event.data,
            KvCacheEventData::Stored(data) if data.blocks.len() == 1
        ));
    }
}
