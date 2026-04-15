// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use tokio::sync::futures::OwnedNotified;
use tokio::sync::{Mutex as AsyncMutex, Notify, mpsc};
use tokio_util::sync::CancellationToken;

use super::{
    GetWorkersRequest, KvIndexer, KvIndexerInterface, KvIndexerMetrics, KvRouterError,
    LowerTierIndexer, ThreadPoolIndexer, WorkerKvQueryResponse,
};
use crate::protocols::*;

#[cfg(test)]
use super::DumpRequest;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(test)]
use tokio::time::Duration;

// -------------------------------------------------
// Decentralized router: LocalKvIndexer for workers
// -------------------------------------------------

#[derive(Clone)]
struct CachedRecoverySnapshot {
    events: Arc<Vec<RouterEvent>>,
    base_last_event_id: u64,
    last_event_id: u64,
}

impl CachedRecoverySnapshot {
    fn into_response(self) -> WorkerKvQueryResponse {
        WorkerKvQueryResponse::TreeDump {
            events: self.events.as_ref().clone(),
            last_event_id: self.last_event_id,
        }
    }
}

#[derive(Clone)]
struct InFlightRecoveryBuild {
    generation: u64,
    notify: Arc<Notify>,
}

#[derive(Default)]
struct RecoveryCacheState {
    generation: u64,
    cached: Option<CachedRecoverySnapshot>,
    building: Option<InFlightRecoveryBuild>,
}

struct RecoverySnapshotCache {
    state: AsyncMutex<RecoveryCacheState>,
}

enum DumpPlan {
    Immediate(WorkerKvQueryResponse),
    RequiresDump { last_event_id: u64 },
}

enum CacheReuseDecision {
    ReturnExact(CachedRecoverySnapshot),
    ReturnExtended(WorkerKvQueryResponse),
    WaitForBuilder(OwnedNotified),
    BuildFresh {
        build: InFlightRecoveryBuild,
        last_event_id: u64,
    },
}

enum TailAppendSafety {
    ExactHit,
    Safe {
        last_event_id: u64,
        tail: Vec<RouterEvent>,
    },
    Invalidate,
}

enum BuildTaskResult {
    Response(WorkerKvQueryResponse),
    StaleGeneration,
}

struct FreshDumpOutput {
    response: WorkerKvQueryResponse,
    snapshot: Option<CachedRecoverySnapshot>,
}

impl RecoverySnapshotCache {
    fn new() -> Self {
        Self {
            state: AsyncMutex::new(RecoveryCacheState::default()),
        }
    }

    async fn decide_reuse_or_build<F>(
        &self,
        fallback_last_event_id: u64,
        current_last_event_id: Option<u64>,
        assess_tail_append_safety: F,
    ) -> CacheReuseDecision
    where
        F: FnOnce(&CachedRecoverySnapshot) -> TailAppendSafety,
    {
        let mut cache_state = self.state.lock().await;

        if let Some(cached) = cache_state.cached.clone() {
            match assess_tail_append_safety(&cached) {
                TailAppendSafety::ExactHit => return CacheReuseDecision::ReturnExact(cached),
                TailAppendSafety::Safe {
                    last_event_id,
                    tail,
                } => {
                    let mut events = cached.events.as_ref().clone();
                    events.extend(tail);
                    let shared_events = Arc::new(events);
                    cache_state.cached = Some(CachedRecoverySnapshot {
                        events: shared_events.clone(),
                        base_last_event_id: cached.base_last_event_id,
                        last_event_id,
                    });
                    return CacheReuseDecision::ReturnExtended(WorkerKvQueryResponse::TreeDump {
                        events: shared_events.as_ref().clone(),
                        last_event_id,
                    });
                }
                TailAppendSafety::Invalidate => {
                    cache_state.cached = None;
                }
            }
        }

        if let Some(build) = cache_state.building.clone() {
            return CacheReuseDecision::WaitForBuilder(build.notify.notified_owned());
        }

        let build = InFlightRecoveryBuild {
            generation: cache_state.generation,
            notify: Arc::new(Notify::new()),
        };
        let last_event_id = current_last_event_id.unwrap_or(fallback_last_event_id);
        cache_state.building = Some(build.clone());
        CacheReuseDecision::BuildFresh {
            build,
            last_event_id,
        }
    }

    async fn finish_build(
        &self,
        build: &InFlightRecoveryBuild,
        build_output: FreshDumpOutput,
    ) -> BuildTaskResult {
        let mut cache_state = self.state.lock().await;
        let is_current_build = cache_state
            .building
            .as_ref()
            .is_some_and(|inflight| inflight.generation == build.generation);
        let generation_matches = cache_state.generation == build.generation;

        if is_current_build {
            cache_state.building = None;
        }

        if !is_current_build || !generation_matches {
            return BuildTaskResult::StaleGeneration;
        }

        if let Some(snapshot) = build_output.snapshot {
            cache_state.cached = Some(snapshot);
        }
        BuildTaskResult::Response(build_output.response)
    }

    async fn clear_build_if_current(&self, generation: u64) {
        let mut cache_state = self.state.lock().await;
        if cache_state
            .building
            .as_ref()
            .is_some_and(|inflight| inflight.generation == generation)
        {
            cache_state.building = None;
        }
    }

    async fn invalidate(&self) {
        let mut cache_state = self.state.lock().await;
        cache_state.generation = cache_state.generation.saturating_add(1);
        cache_state.cached = None;
    }
}

/// A thin wrapper around KvIndexer that buffers recent events
/// (e.g. which may be queued by router upon startup)
///
pub struct LocalKvIndexer {
    /// The underlying indexer
    indexer: KvIndexer,
    /// Lazily-created exact lower-tier indexes partitioned by storage tier.
    lower_tier_indexers: Arc<Mutex<HashMap<StorageTier, Arc<ThreadPoolIndexer<LowerTierIndexer>>>>>,
    /// Circular buffer of recent events
    pub(super) event_buffer: Mutex<VecDeque<RouterEvent>>,
    /// Coordinates single-flight tree dumps and the cached recovery snapshot.
    /// This stays separate from `event_buffer` so dump wait/build state can be
    /// managed on the async path without holding the buffer lock across `.await`.
    recovery_cache: Arc<RecoverySnapshotCache>,
    /// Maximum number of events to keep in buffer
    max_buffer_size: usize, // Router sets this to WORKER_KV_INDEXER_BUFFER_SIZE
    #[cfg(test)]
    dump_build_count: AtomicUsize,
    #[cfg(test)]
    dump_build_delay: Mutex<Option<Duration>>,
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
            recovery_cache: Arc::new(RecoverySnapshotCache::new()),
            max_buffer_size,
            #[cfg(test)]
            dump_build_count: AtomicUsize::new(0),
            #[cfg(test)]
            dump_build_delay: Mutex::new(None),
        }
    }

    #[cfg(test)]
    pub fn get_all_events_in_buffer(&self) -> Vec<RouterEvent> {
        let buffer = self.event_buffer.lock().unwrap();
        buffer.iter().cloned().collect()
    }

    /// Query events by ID range, returning events in `[start_id, end_id]` (both inclusive).
    ///
    /// ### Arguments
    ///
    /// * `start_id` - Starting event ID (inclusive). If `None`, dumps entire tree.
    /// * `end_id` - Ending event ID (inclusive). Used for validation and
    ///   `TooNew` responses; successful buffer-backed responses may still
    ///   return through the newest buffered event.
    ///
    /// ### Returns
    ///
    /// - `Events`: Buffered events with original IDs from `start_id` through the
    ///   current buffered tail, plus the buffered `last_event_id`
    /// - `TreeDump`: Full tree dump with synthetic IDs and the worker's latest real event ID (when range is too old or unspecified)
    /// - `TooNew`: Error when requested range is newer than available data
    /// - `InvalidRange`: Error when end_id < start_id
    pub async fn get_events_in_id_range(
        &self,
        start_id: Option<u64>,
        end_id: Option<u64>,
    ) -> WorkerKvQueryResponse {
        match self.classify_query(start_id, end_id) {
            DumpPlan::Immediate(response) => response,
            DumpPlan::RequiresDump { last_event_id } => {
                self.get_cached_or_fresh_dump(last_event_id).await
            }
        }
    }

    /// Check if a query can likely be served from the buffer (fast path).
    /// Returns true if:
    /// - start_id is Some (not a full dump request)
    /// - buffer is not empty
    /// - start_id is within or after the buffer range
    ///
    /// Note: This is a heuristic - the buffer state may change between this check
    /// and the actual query, so a tree dump may still occur even if this returns true.
    pub fn likely_served_from_buffer(&self, start_id: Option<u64>) -> bool {
        if start_id.is_none() {
            return false;
        }

        let buffer = self.event_buffer.lock().unwrap();
        if buffer.is_empty() {
            return false;
        }

        let first_buffered = buffer.front().unwrap().event.event_id;
        start_id.unwrap() >= first_buffered
    }

    /// Record an event in the buffer
    fn record_event(&self, event: RouterEvent) -> bool {
        let mut buffer = self.event_buffer.lock().unwrap();
        let mut detected_gap = false;

        // Check that event id is consecutive to last one
        if let Some(last_event) = buffer.back()
            && event.event.event_id != last_event.event.event_id + 1
        {
            detected_gap = true;
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

        detected_gap
    }

    /// Apply event with buffering.
    ///
    /// This forwards the event to the underlying indexer and records it on success.
    pub async fn apply_event_with_buffer(&self, event: RouterEvent) -> Result<(), KvRouterError> {
        let result = self.apply_event_by_tier(&event).await;
        if result.is_ok() {
            let should_invalidate = matches!(event.event.data, KvCacheEventData::Cleared);
            let detected_gap = self.record_event(event);
            if should_invalidate || detected_gap {
                self.recovery_cache.invalidate().await;
            }
        }

        result
    }

    #[cfg(test)]
    pub fn buffer_len(&self) -> usize {
        let buffer = self.event_buffer.lock().unwrap();
        buffer.len()
    }

    fn classify_query(&self, start_id: Option<u64>, end_id: Option<u64>) -> DumpPlan {
        if let (Some(s), Some(e)) = (start_id, end_id)
            && e < s
        {
            tracing::warn!(start_id = s, end_id = e, "Invalid range: end_id < start_id");
            return DumpPlan::Immediate(WorkerKvQueryResponse::InvalidRange {
                start_id: s,
                end_id: e,
            });
        }

        let buffer = self.event_buffer.lock().unwrap();
        let (first_id, last_id) = if buffer.is_empty() {
            (None, None)
        } else {
            (
                Some(buffer.front().unwrap().event.event_id),
                Some(buffer.back().unwrap().event.event_id),
            )
        };

        if start_id.is_none() {
            tracing::debug!("No start_id specified, dumping entire tree");
            return DumpPlan::RequiresDump {
                last_event_id: last_id.unwrap_or(0),
            };
        }

        let start_id = start_id.expect("checked above");
        let end_id = end_id.unwrap_or_else(|| last_id.unwrap_or(start_id));

        let Some(first_buffered) = first_id else {
            tracing::debug!("Buffer empty, dumping entire tree");
            return DumpPlan::RequiresDump { last_event_id: 0 };
        };
        let last_buffered = last_id.expect("buffer non-empty");

        if start_id > last_buffered {
            tracing::warn!(
                start_id,
                last_buffered,
                "Requested start_id is newer than buffer"
            );
            return DumpPlan::Immediate(WorkerKvQueryResponse::TooNew {
                requested_start: Some(start_id),
                requested_end: Some(end_id),
                newest_available: last_buffered,
            });
        }

        if start_id < first_buffered {
            tracing::info!(
                start_id,
                first_buffered,
                "Requested start_id is older than buffer, dumping entire tree"
            );
            return DumpPlan::RequiresDump {
                last_event_id: last_buffered,
            };
        }

        let start_idx = match buffer.binary_search_by_key(&start_id, |event| event.event.event_id) {
            Ok(idx) => idx,
            Err(insertion_point) => insertion_point,
        };
        let events = buffer.iter().skip(start_idx).cloned().collect();

        DumpPlan::Immediate(WorkerKvQueryResponse::Events {
            events,
            last_event_id: last_buffered,
        })
    }

    async fn get_cached_or_fresh_dump(&self, fallback_last_event_id: u64) -> WorkerKvQueryResponse {
        loop {
            let decision = self
                .recovery_cache
                .decide_reuse_or_build(
                    fallback_last_event_id,
                    self.current_buffer_last_event_id(),
                    |cached| self.assess_tail_append_safety(cached),
                )
                .await;

            match decision {
                CacheReuseDecision::ReturnExact(snapshot) => return snapshot.into_response(),
                CacheReuseDecision::ReturnExtended(response) => return response,
                CacheReuseDecision::WaitForBuilder(waiter) => waiter.await,
                CacheReuseDecision::BuildFresh {
                    build,
                    last_event_id,
                } => {
                    let notify = build.notify.clone();
                    let generation = build.generation;
                    let build_handle = self.spawn_dump_build(build, last_event_id);
                    match build_handle.await {
                        Ok(BuildTaskResult::Response(response)) => return response,
                        Ok(BuildTaskResult::StaleGeneration) => continue,
                        Err(error) => {
                            tracing::warn!("Recovery cache build task failed: {error}");
                            self.recovery_cache.clear_build_if_current(generation).await;
                            notify.notify_waiters();
                            return WorkerKvQueryResponse::TreeDump {
                                events: Vec::new(),
                                last_event_id,
                            };
                        }
                    }
                }
            }
        }
    }

    fn assess_tail_append_safety(&self, cached: &CachedRecoverySnapshot) -> TailAppendSafety {
        let append_budget = self.recovery_cache_append_budget();
        let buffer = self.event_buffer.lock().unwrap();
        let Some(first_buffered) = buffer.front().map(|event| event.event.event_id) else {
            return if cached.last_event_id == 0 {
                TailAppendSafety::ExactHit
            } else {
                TailAppendSafety::Invalidate
            };
        };
        let last_buffered = buffer.back().unwrap().event.event_id;

        if last_buffered <= cached.last_event_id {
            return TailAppendSafety::ExactHit;
        }

        let appended_since_base = last_buffered.saturating_sub(cached.base_last_event_id);
        if appended_since_base > append_budget {
            return TailAppendSafety::Invalidate;
        }

        let next_event_id = cached.last_event_id.saturating_add(1);
        if next_event_id < first_buffered {
            return TailAppendSafety::Invalidate;
        }

        let start_idx =
            match buffer.binary_search_by_key(&next_event_id, |event| event.event.event_id) {
                Ok(idx) => idx,
                Err(insertion_point) => insertion_point,
            };

        let mut tail = Vec::with_capacity(buffer.len().saturating_sub(start_idx));
        for event in buffer.iter().skip(start_idx) {
            match event.event.data {
                KvCacheEventData::Stored(_) | KvCacheEventData::Removed(_) => {
                    tail.push(event.clone());
                }
                _ => {
                    return TailAppendSafety::Invalidate;
                }
            }
        }

        TailAppendSafety::Safe {
            last_event_id: last_buffered,
            tail,
        }
    }

    fn recovery_cache_append_budget(&self) -> u64 {
        (self.max_buffer_size / 2) as u64
    }

    fn current_buffer_last_event_id(&self) -> Option<u64> {
        self.event_buffer
            .lock()
            .unwrap()
            .back()
            .map(|event| event.event.event_id)
    }

    fn spawn_dump_build(
        &self,
        build: InFlightRecoveryBuild,
        last_event_id: u64,
    ) -> tokio::task::JoinHandle<BuildTaskResult> {
        let indexer = self.indexer.clone();
        let recovery_cache = self.recovery_cache.clone();
        #[cfg(test)]
        let build_delay = *self.dump_build_delay.lock().unwrap();
        #[cfg(test)]
        self.dump_build_count.fetch_add(1, Ordering::Relaxed);

        tokio::spawn(async move {
            #[cfg(test)]
            if let Some(delay) = build_delay {
                tokio::time::sleep(delay).await;
            }

            let build_output = Self::build_fresh_dump(indexer, last_event_id).await;
            let notify = build.notify.clone();
            let result = recovery_cache.finish_build(&build, build_output).await;

            notify.notify_waiters();
            result
        })
    }

    async fn build_fresh_dump(indexer: KvIndexer, last_event_id: u64) -> FreshDumpOutput {
        match indexer.dump_events().await {
            Ok(events) => FreshDumpOutput {
                response: WorkerKvQueryResponse::TreeDump {
                    events: events.clone(),
                    last_event_id,
                },
                snapshot: Some(CachedRecoverySnapshot {
                    events: Arc::new(events),
                    base_last_event_id: last_event_id,
                    last_event_id,
                }),
            },
            Err(error) => {
                tracing::warn!("Failed to build recovery dump: {error}");
                FreshDumpOutput {
                    response: WorkerKvQueryResponse::TreeDump {
                        events: Vec::new(),
                        last_event_id,
                    },
                    snapshot: None,
                }
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn dump_build_count(&self) -> usize {
        self.dump_build_count.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn set_dump_build_delay(&self, delay: Option<Duration>) {
        *self.dump_build_delay.lock().unwrap() = delay;
    }

    // Delegation methods to underlying KvIndexer
    /// Get a sender for `RouterEvent`s.
    pub fn event_sender(&self) -> mpsc::Sender<RouterEvent> {
        self.indexer.event_sender()
    }

    #[cfg(test)]
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

    async fn apply_event_to_lower_tier(&self, event: RouterEvent) -> Result<(), KvRouterError> {
        self.get_or_create_lower_tier_indexer(event.storage_tier)
            .apply_event(event)
            .await;
        Ok(())
    }

    async fn apply_event_by_tier(&self, event: &RouterEvent) -> Result<(), KvRouterError> {
        match &event.event.data {
            KvCacheEventData::Cleared => {
                self.apply_event_to_primary(event.clone()).await?;
                for indexer in self.all_lower_tier_indexers() {
                    indexer.apply_event(event.clone()).await;
                }
                Ok(())
            }
            _ if event.storage_tier.is_gpu() => self.apply_event_to_primary(event.clone()).await,
            _ => self.apply_event_to_lower_tier(event.clone()).await,
        }
    }

    fn get_or_create_lower_tier_indexer(
        &self,
        storage_tier: StorageTier,
    ) -> Arc<ThreadPoolIndexer<LowerTierIndexer>> {
        debug_assert!(!storage_tier.is_gpu());
        let mut indexers = self.lower_tier_indexers.lock().unwrap();
        indexers
            .entry(storage_tier)
            .or_insert_with(|| {
                Arc::new(ThreadPoolIndexer::new(
                    LowerTierIndexer::new(),
                    1,
                    self.block_size(),
                ))
            })
            .clone()
    }

    fn all_lower_tier_indexers(&self) -> Vec<Arc<ThreadPoolIndexer<LowerTierIndexer>>> {
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
            indexer.remove_worker(worker).await;
        }
        let _ = self.indexer.remove_worker_sender().send(worker).await;
    }

    async fn remove_worker_dp_rank(&self, worker: WorkerId, dp_rank: DpRank) {
        for indexer in self.all_lower_tier_indexers() {
            KvIndexerInterface::remove_worker_dp_rank(&*indexer, worker, dp_rank).await;
        }
        KvIndexerInterface::remove_worker_dp_rank(&self.indexer, worker, dp_rank).await;
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
        let queued = self.indexer.flush().await;
        for indexer in self.all_lower_tier_indexers() {
            let _ = indexer.dump_events().await;
        }
        queued
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
            .backend()
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
        let _ = indexer.flush().await;

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
        let _ = indexer.flush().await;
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
        let _ = indexer.flush().await;
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
        let _ = indexer.flush().await;

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
