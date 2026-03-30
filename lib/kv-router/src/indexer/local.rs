// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::{
    DumpRequest, GetWorkersRequest, KvIndexer, KvIndexerInterface, KvIndexerMetrics, KvRouterError,
    LowerTierIndexer, WorkerKvQueryResponse,
};
use crate::protocols::*;

// -------------------------------------------------
// Decentralized router: LocalKvIndexer for workers
// -------------------------------------------------

/// A thin wrapper around KvIndexer that buffers recent events
/// (e.g. which may be queued by router upon startup)
///
pub struct LocalKvIndexer {
    /// The underlying indexer
    indexer: KvIndexer,
    /// Lazily-created exact lower-tier indexes partitioned by storage tier.
    lower_tier_indexers: Arc<Mutex<HashMap<StorageTier, Arc<LowerTierIndexer>>>>,
    /// Circular buffer of recent events
    pub(super) event_buffer: Mutex<VecDeque<RouterEvent>>,
    /// Maximum number of events to keep in buffer
    max_buffer_size: usize, // Router sets this to WORKER_KV_INDEXER_BUFFER_SIZE
}

fn map_lower_tier_error(error: KvCacheEventError) -> KvRouterError {
    match error {
        KvCacheEventError::BlockNotFound => KvRouterError::BlockNotFound,
        KvCacheEventError::ParentBlockNotFound | KvCacheEventError::InvalidBlockSequence => {
            KvRouterError::IndexerDroppedRequest
        }
    }
}

impl LocalKvIndexer {
    /// create a new LocalKvIndexer pointing to a KvIndexer.
    pub fn new(
        token: CancellationToken,
        kv_block_size: u32,
        metrics: Arc<KvIndexerMetrics>,
        max_buffer_size: usize,
    ) -> Self {
        Self {
            indexer: KvIndexer::new(token, kv_block_size, metrics),
            lower_tier_indexers: Arc::new(Mutex::new(HashMap::new())),
            event_buffer: Mutex::new(VecDeque::with_capacity(max_buffer_size)),
            max_buffer_size,
        }
    }

    /// Get all buffered events (oldest first).
    pub fn get_all_events_in_buffer(&self) -> Vec<RouterEvent> {
        let buffer = self.event_buffer.lock().unwrap();
        buffer.iter().cloned().collect()
    }

    /// Build a tree dump response with the given `last_event_id`.
    async fn tree_dump_response(&self, last_event_id: u64) -> WorkerKvQueryResponse {
        let events = self.dump_events().await.unwrap_or_default();
        WorkerKvQueryResponse::TreeDump {
            events,
            last_event_id,
        }
    }

    /// Query events by ID range, returning events in `[start_id, end_id]` (both inclusive).
    ///
    /// ### Arguments
    ///
    /// * `start_id` - Starting event ID (inclusive). If `None`, dumps entire tree.
    /// * `end_id` - Ending event ID (inclusive). If `None`, returns up to newest available.
    ///
    /// ### Returns
    ///
    /// - `Events`: Buffered events with original IDs (when range is within buffer)
    /// - `TreeDump`: Full tree dump with synthetic IDs and the worker's latest real event ID (when range is too old or unspecified)
    /// - `TooNew`: Error when requested range is newer than available data
    /// - `InvalidRange`: Error when end_id < start_id
    pub async fn get_events_in_id_range(
        &self,
        start_id: Option<u64>,
        end_id: Option<u64>,
    ) -> WorkerKvQueryResponse {
        // Validate range if both specified
        if let (Some(s), Some(e)) = (start_id, end_id)
            && e < s
        {
            tracing::warn!(start_id = s, end_id = e, "Invalid range: end_id < start_id");
            return WorkerKvQueryResponse::InvalidRange {
                start_id: s,
                end_id: e,
            };
        }

        // Get buffer state
        let (first_id, last_id) = {
            let buffer = self.event_buffer.lock().unwrap();
            if buffer.is_empty() {
                (None, None)
            } else {
                (
                    Some(buffer.front().unwrap().event.event_id),
                    Some(buffer.back().unwrap().event.event_id),
                )
            }
        };

        // If no start_id specified, dump entire tree
        if start_id.is_none() {
            tracing::debug!("No start_id specified, dumping entire tree");
            return self.tree_dump_response(last_id.unwrap_or(0)).await;
        }

        let start_id = start_id.unwrap();
        let end_id = end_id.unwrap_or_else(|| last_id.unwrap_or(start_id));

        // Check for empty buffer
        let Some(first_buffered) = first_id else {
            tracing::debug!("Buffer empty, dumping entire tree");
            return self.tree_dump_response(0).await;
        };
        let last_buffered = last_id.unwrap();

        // Check if request is too new
        if start_id > last_buffered {
            tracing::warn!(
                start_id,
                last_buffered,
                "Requested start_id is newer than buffer"
            );
            return WorkerKvQueryResponse::TooNew {
                requested_start: Some(start_id),
                requested_end: Some(end_id),
                newest_available: last_buffered,
            };
        }

        // Check if start_id is too old (before buffer) -> tree dump
        if start_id < first_buffered {
            tracing::info!(
                start_id,
                first_buffered,
                "Requested start_id is older than buffer, dumping entire tree"
            );
            return self.tree_dump_response(last_buffered).await;
        }

        // Serve from buffer
        let buffer = self.event_buffer.lock().unwrap();

        let start_idx = match buffer.binary_search_by_key(&start_id, |e| e.event.event_id) {
            Ok(idx) => idx,
            Err(insertion_point) => insertion_point,
        };

        // Clamp end_id to buffer bounds
        let clamped_end_id = end_id.min(last_buffered);
        let end_idx = match buffer.binary_search_by_key(&clamped_end_id, |e| e.event.event_id) {
            Ok(idx) => idx + 1, // Include the matched element
            Err(insertion_point) => insertion_point,
        };

        let events: Vec<RouterEvent> = buffer
            .iter()
            .skip(start_idx)
            .take(end_idx.saturating_sub(start_idx))
            .cloned()
            .collect();

        WorkerKvQueryResponse::Events(events)
    }

    /// Record an event in the buffer
    fn record_event(&self, event: RouterEvent) {
        let mut buffer = self.event_buffer.lock().unwrap();

        // Check that event id is consecutive to last one
        if let Some(last_event) = buffer.back()
            && event.event.event_id != last_event.event.event_id + 1
        {
            let expected = last_event.event.event_id + 1;
            tracing::error!(
                worker_id = event.worker_id,
                expected,
                got = event.event.event_id,
                "Non-consecutive KV event id; buffer may have gaps"
            );
        }
        tracing::debug!(
            "Recorded event {:?} in buffer, now size is {}",
            event,
            buffer.len()
        );

        // Add to back
        buffer.push_back(event);

        // Remove from front if over capacity (circular buffer behavior)
        while buffer.len() > self.max_buffer_size {
            buffer.pop_front();
        }
    }

    /// Apply event with buffering.
    ///
    /// This forwards the event to the underlying indexer and records it on success.
    pub async fn apply_event_with_buffer(&self, event: RouterEvent) -> Result<(), KvRouterError> {
        let result = self.apply_event_by_tier(&event).await;
        if result.is_ok() {
            self.record_event(event);
        }

        result
    }

    /// Clear the event buffer.
    pub fn clear_buffer(&self) {
        let mut buffer = self.event_buffer.lock().unwrap();
        buffer.clear();
    }

    /// Get the current buffer size.
    pub fn buffer_len(&self) -> usize {
        let buffer = self.event_buffer.lock().unwrap();
        buffer.len()
    }

    // Delegation methods to underlying KvIndexer
    /// Get a sender for `RouterEvent`s.
    pub fn event_sender(&self) -> mpsc::Sender<RouterEvent> {
        self.indexer.event_sender()
    }

    /// Get a sender for dump requests (snapshot events).
    pub fn snapshot_event_sender(&self) -> mpsc::Sender<DumpRequest> {
        self.indexer.snapshot_event_sender()
    }

    /// Get a sender for worker removal requests.
    pub fn remove_worker_sender(&self) -> mpsc::Sender<WorkerId> {
        self.indexer.remove_worker_sender()
    }

    /// Get a sender for get workers requests.
    pub fn get_workers_sender(&self) -> mpsc::Sender<GetWorkersRequest> {
        self.indexer.get_workers_sender()
    }

    /// Get the KV block size.
    pub fn block_size(&self) -> u32 {
        self.indexer.block_size()
    }

    async fn apply_event_to_primary(&self, event: RouterEvent) -> Result<(), KvRouterError> {
        self.indexer
            .event_sender()
            .send(event)
            .await
            .map_err(|_| KvRouterError::IndexerOffline)
    }

    fn apply_event_to_lower_tier(&self, event: RouterEvent) -> Result<(), KvRouterError> {
        self.get_or_create_lower_tier_indexer(event.storage_tier)
            .apply_event(event)
            .map_err(map_lower_tier_error)
    }

    async fn apply_event_by_tier(&self, event: &RouterEvent) -> Result<(), KvRouterError> {
        match &event.event.data {
            KvCacheEventData::Cleared => {
                self.apply_event_to_primary(event.clone()).await?;
                for indexer in self.all_lower_tier_indexers() {
                    indexer
                        .apply_event(event.clone())
                        .map_err(map_lower_tier_error)?;
                }
                Ok(())
            }
            _ if event.storage_tier.is_gpu() => self.apply_event_to_primary(event.clone()).await,
            _ => self.apply_event_to_lower_tier(event.clone()),
        }
    }

    fn get_or_create_lower_tier_indexer(&self, storage_tier: StorageTier) -> Arc<LowerTierIndexer> {
        debug_assert!(!storage_tier.is_gpu());
        let mut indexers = self.lower_tier_indexers.lock().unwrap();
        indexers
            .entry(storage_tier)
            .or_insert_with(|| Arc::new(LowerTierIndexer::new()))
            .clone()
    }

    fn all_lower_tier_indexers(&self) -> Vec<Arc<LowerTierIndexer>> {
        let indexers = self.lower_tier_indexers.lock().unwrap();
        indexers.values().cloned().collect()
    }
}

// Implement KvIndexerInterface by delegating to the underlying indexer
#[async_trait]
impl KvIndexerInterface for LocalKvIndexer {
    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        self.indexer.find_matches(sequence).await
    }

    async fn find_matches_for_request(
        &self,
        tokens: &[u32],
        lora_name: Option<&str>,
        is_eagle: Option<bool>,
    ) -> Result<OverlapScores, KvRouterError> {
        self.indexer
            .find_matches_for_request(tokens, lora_name, is_eagle)
            .await
    }

    async fn apply_event(&self, event: RouterEvent) {
        // Use the buffering version
        let _ = self.apply_event_with_buffer(event).await;
    }

    async fn remove_worker(&self, worker: WorkerId) {
        for indexer in self.all_lower_tier_indexers() {
            indexer.remove_worker(worker);
        }
        let _ = self.indexer.remove_worker_sender().send(worker).await;
    }

    async fn remove_worker_dp_rank(&self, worker: WorkerId, dp_rank: DpRank) {
        for indexer in self.all_lower_tier_indexers() {
            indexer.remove_worker_dp_rank(worker, dp_rank);
        }
        let _ = self.indexer.remove_worker_dp_rank(worker, dp_rank).await;
    }

    fn shutdown(&self) {
        self.indexer.shutdown();
    }

    async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError> {
        self.indexer.dump_events().await
    }

    async fn process_routing_decision_for_request(
        &self,
        tokens_with_hashes: &mut TokensWithHashes,
        worker: WorkerWithDpRank,
    ) -> Result<(), KvRouterError> {
        // TODO I guess the local kvindexers have little use for this method?
        // Keeping it here now to implement the trait fully
        self.indexer
            .process_routing_decision_for_request(tokens_with_hashes, worker)
            .await
    }

    async fn flush(&self) -> usize {
        self.indexer.flush().await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rustc_hash::FxHashMap;
    use tokio_util::sync::CancellationToken;

    use super::LocalKvIndexer;
    use crate::indexer::{KvIndexerInterface, KvIndexerMetrics, LowerTierContinuation};
    use crate::protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
        KvCacheStoredBlockData, LocalBlockHash, RouterEvent, StorageTier, WorkerWithDpRank,
    };

    fn lower_tier_store_event(
        worker_id: u64,
        dp_rank: u32,
        event_id: u64,
        parent_hash: u64,
        tokens_hash: u64,
        block_hash: u64,
        storage_tier: StorageTier,
    ) -> RouterEvent {
        RouterEvent::with_storage_tier(
            worker_id,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: Some(ExternalSequenceBlockHash(parent_hash)),
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(block_hash),
                        tokens_hash: LocalBlockHash(tokens_hash),
                        mm_extra_info: None,
                    }],
                }),
                dp_rank,
            },
            storage_tier,
        )
    }

    fn lower_tier_hits(
        indexer: &LocalKvIndexer,
        storage_tier: StorageTier,
        worker_id: u64,
        dp_rank: u32,
        parent_hash: u64,
        tokens_hash: u64,
    ) -> usize {
        let lower_tier_indexer = {
            let indexers = indexer.lower_tier_indexers.lock().unwrap();
            indexers.get(&storage_tier).cloned()
        };

        let Some(lower_tier_indexer) = lower_tier_indexer else {
            return 0;
        };

        let mut continuations = FxHashMap::default();
        continuations.insert(
            WorkerWithDpRank::new(worker_id, dp_rank),
            LowerTierContinuation::new(0, ExternalSequenceBlockHash(parent_hash)),
        );

        lower_tier_indexer
            .query_contiguous_hits(&[LocalBlockHash(tokens_hash)], &continuations)
            .get(&WorkerWithDpRank::new(worker_id, dp_rank))
            .copied()
            .unwrap_or(0)
    }

    #[tokio::test]
    async fn lower_tier_events_are_buffered_without_touching_primary_index() {
        let indexer = LocalKvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            16,
        );
        let event = lower_tier_store_event(7, 0, 1, 900, 11, 101, StorageTier::HostPinned);

        indexer
            .apply_event_with_buffer(event.clone())
            .await
            .unwrap();

        assert_eq!(indexer.get_all_events_in_buffer(), vec![event]);
        assert_eq!(indexer.lower_tier_indexers.lock().unwrap().len(), 1);
        assert_eq!(
            lower_tier_hits(&indexer, StorageTier::HostPinned, 7, 0, 900, 11),
            1
        );

        let overlap = indexer
            .find_matches(vec![LocalBlockHash(11)])
            .await
            .unwrap();
        assert!(overlap.scores.is_empty());
    }

    #[tokio::test]
    async fn lower_tier_events_are_partitioned_by_storage_tier() {
        let indexer = LocalKvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            16,
        );

        assert_eq!(indexer.lower_tier_indexers.lock().unwrap().len(), 0);

        indexer
            .apply_event_with_buffer(lower_tier_store_event(
                19,
                0,
                1,
                1000,
                31,
                301,
                StorageTier::HostPinned,
            ))
            .await
            .unwrap();
        assert_eq!(indexer.lower_tier_indexers.lock().unwrap().len(), 1);

        indexer
            .apply_event_with_buffer(lower_tier_store_event(
                19,
                0,
                2,
                2000,
                31,
                302,
                StorageTier::Disk,
            ))
            .await
            .unwrap();
        assert_eq!(indexer.lower_tier_indexers.lock().unwrap().len(), 2);

        assert_eq!(
            lower_tier_hits(&indexer, StorageTier::HostPinned, 19, 0, 1000, 31),
            1
        );
        assert_eq!(
            lower_tier_hits(&indexer, StorageTier::Disk, 19, 0, 2000, 31),
            1
        );
        assert_eq!(
            lower_tier_hits(&indexer, StorageTier::HostPinned, 19, 0, 2000, 31),
            0
        );
        assert_eq!(
            lower_tier_hits(&indexer, StorageTier::Disk, 19, 0, 1000, 31),
            0
        );
    }

    #[tokio::test]
    async fn cleared_event_clears_all_lower_tier_dp_ranks_for_worker() {
        let indexer = LocalKvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            16,
        );

        indexer
            .apply_event_with_buffer(lower_tier_store_event(
                11,
                0,
                1,
                1000,
                21,
                201,
                StorageTier::HostPinned,
            ))
            .await
            .unwrap();
        indexer
            .apply_event_with_buffer(lower_tier_store_event(
                11,
                1,
                2,
                2000,
                22,
                202,
                StorageTier::HostPinned,
            ))
            .await
            .unwrap();

        indexer
            .apply_event_with_buffer(RouterEvent::with_storage_tier(
                11,
                KvCacheEvent {
                    event_id: 3,
                    data: KvCacheEventData::Cleared,
                    dp_rank: 0,
                },
                StorageTier::HostPinned,
            ))
            .await
            .unwrap();

        assert_eq!(
            lower_tier_hits(&indexer, StorageTier::HostPinned, 11, 0, 1000, 21),
            0
        );
        assert_eq!(
            lower_tier_hits(&indexer, StorageTier::HostPinned, 11, 1, 2000, 22),
            0
        );
    }
}
