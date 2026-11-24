// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::hash_map::Entry;
use std::sync::{Arc, OnceLock};

use dynamo_kv_ledger::{Applied, Effect, Ledger, Replay};
use rustc_hash::FxHashMap;

use super::{WorkerKvQueryResponse, record_unsupported_residency_event};
use crate::prelude::*;

// -------------------------------------------------
// Decentralized router: LocalKvIndexer for workers
// -------------------------------------------------

/// The blocks each residency owner holds on each storage tier. A tier keeps its own parent
/// and token hash for a block, because an engine can describe one block differently per tier.
type Residency = FxHashMap<(ResidencyOwner, StorageTier), Ledger<u64>>;

struct Inner {
    replay: Replay<RouterEvent, Residency>,
    /// The tree dump of the replay's open cut, set when its build finishes. Requesters that
    /// arrive while the cut is open clone the one dump.
    dump: Option<Arc<OnceLock<Vec<RouterEvent>>>>,
}

/// One rank publisher's recent events and current residency, kept on the worker so a router
/// that missed events can catch up.
///
/// It answers a recovery query with the buffered events from the requested id, or, when the
/// buffer no longer reaches back that far, with a tree dump: the residency as `Stored` events,
/// followed by the buffered events applied since the dump was cut.
///
/// NOTE: One `LocalKvIndexer` belongs to one rank publisher, so its sequence and any latest
/// `Cleared` event are rank-local. Do not merge independent rank streams into it.
pub struct LocalKvIndexer {
    /// Never held across `.await`.
    inner: parking_lot::Mutex<Inner>,
    /// Held while a dump is cut and built, so one build and one copy of the residency exist
    /// at a time.
    dump_gate: Arc<tokio::sync::Mutex<()>>,
    #[cfg(test)]
    dump_builds: std::sync::atomic::AtomicUsize,
    #[cfg(test)]
    dump_build_delay: parking_lot::Mutex<Option<std::time::Duration>>,
}

impl LocalKvIndexer {
    /// `max_buffer_size` is how many recent events are kept for replay
    /// (`WORKER_KV_INDEXER_BUFFER_SIZE` on a worker).
    pub fn new(max_buffer_size: usize) -> Self {
        Self {
            inner: parking_lot::Mutex::new(Inner {
                replay: Replay::new(Residency::default(), max_buffer_size),
                dump: None,
            }),
            dump_gate: Arc::default(),
            #[cfg(test)]
            dump_builds: Default::default(),
            #[cfg(test)]
            dump_build_delay: Default::default(),
        }
    }

    /// Query events by ID range, returning a recovery-equivalent suffix for `[start_id, end_id]`.
    ///
    /// ### Arguments
    ///
    /// * `start_id` - Starting event ID (inclusive). If `None`, dumps the residency.
    /// * `end_id` - Ending event ID (inclusive). Used for validation and
    ///   `TooNew` responses; successful buffer-backed responses may still
    ///   return through the newest buffered event.
    ///
    /// ### Returns
    ///
    /// - `Events`: Buffered events with original IDs through the current buffered tail,
    ///   plus the buffered `last_event_id`. If the buffered suffix contains an all-domain
    ///   `Cleared` event, events before the last such clear may be omitted because it supersedes
    ///   every residency owned by this rank publisher.
    /// - `TreeDump`: The residency as `Stored` events with synthetic IDs, then the buffered
    ///   events applied since it was cut, and the worker's latest real event ID (when the range
    ///   is too old or unspecified)
    /// - `TooNew`: Error when requested range is newer than available data
    /// - `InvalidRange`: Error when end_id < start_id
    pub async fn get_events_in_id_range(
        &self,
        start_id: Option<u64>,
        end_id: Option<u64>,
    ) -> WorkerKvQueryResponse {
        if let (Some(s), Some(e)) = (start_id, end_id)
            && e < s
        {
            tracing::warn!(start_id = s, end_id = e, "Invalid range: end_id < start_id");
            return WorkerKvQueryResponse::InvalidRange {
                start_id: s,
                end_id: e,
            };
        }

        // NOTE: KV RECOVERY CONTRACT: Decide Events versus TreeDump here, against history
        // retained when the query is handled. A client's ordinary range request does not
        // guarantee buffered replay: expired/unavailable history falls back to a dump.
        // See test_local_indexer_get_events_in_id_range_all_cases and
        // test_local_indexer_buffer_response_starts_at_last_all_domain_clear.
        if let Some(start_id) = start_id {
            let inner = self.inner.lock();
            let last_event_id = inner.replay.last_seq().unwrap_or(0);
            if inner.replay.oldest_seq().is_some() {
                if start_id > last_event_id {
                    tracing::warn!(
                        start_id,
                        last_buffered = last_event_id,
                        "Requested start_id is newer than buffer"
                    );
                    return WorkerKvQueryResponse::TooNew {
                        requested_start: Some(start_id),
                        requested_end: Some(end_id.unwrap_or(last_event_id)),
                        newest_available: last_event_id,
                    };
                }
                if let Some(buffered) = inner.replay.since(start_id) {
                    return WorkerKvQueryResponse::Events {
                        events: from_last_all_domain_clear(buffered),
                        last_event_id,
                    };
                }
                tracing::info!(start_id, "Requested start_id is older than buffer; dumping");
            }
        }

        // A request that arrives during a build waits here, then shares that build's dump if
        // its cut is still open. The gate goes into the build task, so a requester that goes
        // away does not let a second build start beside the one it left running.
        let one_build = self.dump_gate.clone().lock_owned().await;
        let (dump, copy, tail, last_event_id) = {
            let mut inner = self.inner.lock();
            let shared = match (&inner.dump, inner.replay.tail()) {
                // The gate is held, so no build is running: an unset dump is one whose build
                // did not finish, and it is replaced.
                (Some(dump), Some(tail)) if dump.get().is_some() => {
                    Some((dump.clone(), tail.cloned().collect()))
                }
                _ => None,
            };
            let (dump, copy, tail) = match shared {
                Some((dump, tail)) => (dump, None, tail),
                None => {
                    // The residency is copied here, under the lock, so the dump is an exact
                    // cut at its last event id. Turning it into events is left to the
                    // blocking pool.
                    #[cfg(test)]
                    self.dump_builds
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let copy = inner.replay.cut().clone();
                    let dump = Arc::new(OnceLock::new());
                    inner.dump = Some(dump.clone());
                    (dump, Some(copy), Vec::new())
                }
            };
            (dump, copy, tail, inner.replay.last_seq().unwrap_or(0))
        };
        #[cfg(test)]
        {
            let delay = *self.dump_build_delay.lock();
            if let Some(delay) = delay {
                tokio::time::sleep(delay).await;
            }
        }

        let built = tokio::task::spawn_blocking(move || {
            let dumped = match copy {
                Some(residency) => {
                    let events = dump_events(&residency);
                    tracing::info!(
                        event_count = events.len(),
                        last_event_id,
                        "Built recovery dump"
                    );
                    // Unset until here: the gate admits one build, and this dump is new.
                    dump.get_or_init(|| events)
                }
                None => dump.get().expect("a shared dump was set when it was found"),
            };
            // The copy of the residency is freed. Requesters that share this dump from here
            // on copy it side by side.
            drop(one_build);
            let mut events = Vec::with_capacity(dumped.len() + tail.len());
            events.extend_from_slice(dumped);
            events.extend(tail);
            events
        })
        .await;
        match built {
            Ok(events) => WorkerKvQueryResponse::TreeDump {
                events,
                last_event_id,
                reset_scope: ResetScope::All,
            },
            Err(error) => {
                tracing::warn!("Recovery dump build task failed: {error}");
                WorkerKvQueryResponse::TreeDumpFailed {
                    last_event_id,
                    message: format!("recovery dump task failed: {error}"),
                }
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
        let Some(start_id) = start_id else {
            return false;
        };
        self.inner
            .lock()
            .replay
            .oldest_seq()
            .is_some_and(|first| start_id >= first)
    }

    /// Newest locally applied outbound event cursor.
    pub fn current_event_id(&self) -> u64 {
        self.inner.lock().replay.last_seq().unwrap_or(0)
    }

    /// Applies an event to the residency and records it for replay. When this returns, a
    /// recovery query sees the event.
    pub fn apply_event_with_buffer(&self, event: RouterEvent) {
        let (worker_id, event_id) = (event.worker_id, event.event.event_id);
        let (applied, retired) = {
            let mut inner = self.inner.lock();
            let applied = inner.replay.apply(event_id, event, fold);
            // A dump whose cut has closed can never be served again. Take it out so it is
            // freed after the lock is released.
            let retired = if inner.replay.tail().is_none() {
                inner.dump.take()
            } else {
                None
            };
            (applied, retired)
        };
        drop(retired);

        if let Applied::OutOfOrder { expected } = applied {
            tracing::error!(
                worker_id,
                expected,
                got = event_id,
                "Non-consecutive KV event id; replay restarts at this event"
            );
        }
    }

    #[cfg(test)]
    pub fn get_all_events_in_buffer(&self) -> Vec<RouterEvent> {
        let inner = self.inner.lock();
        let Some(oldest) = inner.replay.oldest_seq() else {
            return Vec::new();
        };
        inner
            .replay
            .since(oldest)
            .into_iter()
            .flatten()
            .cloned()
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn has_cached_recovery_snapshot(&self) -> bool {
        self.inner.lock().dump.is_some()
    }

    /// How many times the residency was copied for a dump.
    #[cfg(test)]
    pub(crate) fn dump_build_count(&self) -> usize {
        self.dump_builds.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Holds each dump build open for `delay` before it converts its copy.
    #[cfg(test)]
    pub(crate) fn set_dump_build_delay(&self, delay: Option<std::time::Duration>) {
        *self.dump_build_delay.lock() = delay;
    }
}

/// The buffered events a recovery needs: all of them, or from the last all-domain `Cleared`
/// on, which supersedes everything before it.
fn from_last_all_domain_clear<'a>(
    buffered: impl Iterator<Item = &'a RouterEvent>,
) -> Vec<RouterEvent> {
    let buffered = buffered.collect::<Vec<_>>();
    let start = buffered
        .iter()
        .rposition(|event| matches!(event.reset_scope(), Ok(Some(ResetScope::All))))
        .unwrap_or(0);
    buffered[start..].iter().copied().cloned().collect()
}

/// Applies one event to the residency.
///
/// An event whose residency cannot be resolved changes nothing and is logged; it is still
/// recorded for replay, so event ids stay consecutive.
fn fold(event: &RouterEvent, residency: &mut Residency) -> Effect {
    let placement = || match event.residency_owner() {
        Ok(owner) => Some((owner, event.storage_tier)),
        Err(_) => {
            record_unsupported_residency_event(None, event);
            None
        }
    };
    match &event.event.data {
        KvCacheEventData::Stored(store) => {
            if let Some(placement) = placement() {
                let ledger = residency.entry(placement).or_default();
                let mut parent = store.parent_hash.map(|hash| hash.0);
                for block in &store.blocks {
                    ledger.store(block.block_hash.0, block.tokens_hash.0, parent);
                    parent = Some(block.block_hash.0);
                }
            }
            Effect::Delta
        }
        KvCacheEventData::Removed(remove) => {
            if let Some(placement) = placement()
                && let Entry::Occupied(mut ledger) = residency.entry(placement)
            {
                for hash in &remove.block_hashes {
                    ledger.get_mut().remove(hash.0);
                }
                if ledger.get().is_empty() {
                    ledger.remove();
                }
            }
            Effect::Delta
        }
        KvCacheEventData::Cleared => {
            // NOTE: A clear names owners, not tiers. It drops the emitting rank's worker-owned
            // blocks, its cache owner's blocks, or both, on every tier. The router's
            // lower-tier index scopes a clear the same way; keep the two in step.
            let worker =
                ResidencyOwner::worker(WorkerWithDpRank::new(event.worker_id, event.event.dp_rank));
            let cache_owner = event.state_source.map(ResidencyOwner::cache_owner);
            let cleared = match event.reset_scope() {
                Ok(Some(ResetScope::All)) => [Some(worker), cache_owner],
                Ok(Some(ResetScope::Domain(ResidencyDomain::Worker))) => [Some(worker), None],
                Ok(Some(ResetScope::Domain(ResidencyDomain::CacheOwner)))
                    if cache_owner.is_some() =>
                {
                    [cache_owner, None]
                }
                _ => {
                    record_unsupported_residency_event(None, event);
                    [None, None]
                }
            };
            residency.retain(|(owner, _), _| !cleared.contains(&Some(*owner)));
            Effect::Reset
        }
    }
}

/// The residency as the `Stored` events of a tree dump, with synthetic event ids.
///
/// Walks every block: run it off the thread that applies events.
fn dump_events(residency: &Residency) -> Vec<RouterEvent> {
    let stored_block = |hash: u64, tokens_hash: u64| KvCacheStoredBlockData {
        block_hash: ExternalSequenceBlockHash(hash),
        tokens_hash: LocalBlockHash(tokens_hash),
        mm_extra_info: None,
    };
    let mut events = Vec::new();
    for (&(owner, storage_tier), ledger) in residency {
        if storage_tier.is_gpu() {
            // The device tier feeds a prefix tree, which takes a block only after its parent.
            // So these are chains, parents first, and a block that does not chain back to the
            // start of a sequence on the device is left out: the tree could not hold it.
            for chain in ledger.chains() {
                let blocks = chain
                    .blocks
                    .iter()
                    .map(|&(hash, tokens_hash)| stored_block(hash, tokens_hash))
                    .collect();
                events.push(stored_event(
                    owner,
                    storage_tier,
                    events.len() as u64,
                    chain.parent,
                    blocks,
                ));
            }
        } else {
            // A lower-tier index keys a block by its parent hash and does not need the parent
            // present, and it stops a multi-block store at the first block it cannot place.
            // So every block is dumped, one per event, in any order.
            //
            // NOTE: That index holds one block per (parent, token hash) and refuses a second.
            // The ledger keeps the second, so once the first is removed a dump still carries
            // it and a recovering router takes it. Two engine hashes for one (parent, token
            // hash) is a collision we do not expect.
            for (hash, block) in ledger.iter() {
                events.push(stored_event(
                    owner,
                    storage_tier,
                    events.len() as u64,
                    block.parent,
                    vec![stored_block(hash, block.content)],
                ));
            }
        }
    }
    events
}

fn stored_event(
    owner: ResidencyOwner,
    storage_tier: StorageTier,
    event_id: u64,
    parent: Option<u64>,
    blocks: Vec<KvCacheStoredBlockData>,
) -> RouterEvent {
    // A cache owner's blocks belong to no worker rank.
    let (worker_id, dp_rank, state_source) = match owner {
        ResidencyOwner::Worker(worker) => (worker.worker_id, worker.dp_rank, None),
        ResidencyOwner::CacheOwner(cache_owner) => (0, 0, Some(cache_owner)),
    };
    let mut event = RouterEvent::with_residency_domain(
        worker_id,
        KvCacheEvent {
            event_id,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: parent.map(ExternalSequenceBlockHash),
                start_position: None,
                blocks,
            }),
            dp_rank,
        },
        storage_tier,
        owner.domain(),
    );
    event.state_source = state_source;
    event
}

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashSet;

    use super::LocalKvIndexer;
    use crate::WorkerKvQueryResponse;
    use crate::prelude::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
        KvCacheStoredBlockData, LocalBlockHash, ResetScope, ResidencyDomain, RouterEvent,
        StorageTier,
    };
    use dynamo_router_protocol::identity::{
        CacheOwnerId, CacheSemanticsId, ClusterId, IdentitySource, IndexerDomainId, PoolId,
        RoutingScopeId, StableDpSlotId,
    };

    fn cache_owner_id() -> CacheOwnerId {
        CacheOwnerId::new(
            PoolId::new(
                IndexerDomainId::new(
                    CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
                    RoutingScopeId::new([2; 16], IdentitySource::Explicit),
                ),
                ClusterId::new(3),
            ),
            StableDpSlotId::new([4; 16], IdentitySource::Explicit),
        )
    }

    fn make_indexer() -> LocalKvIndexer {
        LocalKvIndexer::new(16)
    }

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
                    start_position: None,
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

    fn lower_tier_store_event_in_domain(
        worker_id: u64,
        event_id: u64,
        parent_hash: u64,
        tokens_hash: u64,
        block_hash: u64,
        residency_domain: ResidencyDomain,
    ) -> RouterEvent {
        let event = KvCacheEvent {
            event_id,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: Some(ExternalSequenceBlockHash(parent_hash)),
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(block_hash),
                    tokens_hash: LocalBlockHash(tokens_hash),
                    mm_extra_info: None,
                }],
            }),
            dp_rank: 0,
        };
        match residency_domain {
            ResidencyDomain::Worker => RouterEvent::with_residency_domain(
                worker_id,
                event,
                StorageTier::HostPinned,
                ResidencyDomain::Worker,
            ),
            ResidencyDomain::CacheOwner => RouterEvent::with_cache_owner(
                worker_id,
                event,
                StorageTier::HostPinned,
                cache_owner_id(),
            ),
        }
    }

    /// One block of a tree dump: `(tier, worker_id, dp_rank, parent, block, tokens)`.
    type DumpedBlock = (StorageTier, u64, u32, Option<u64>, u64, u64);

    async fn tree_dump(indexer: &LocalKvIndexer) -> (Vec<RouterEvent>, u64) {
        match indexer.get_events_in_id_range(None, None).await {
            WorkerKvQueryResponse::TreeDump {
                events,
                last_event_id,
                reset_scope,
            } => {
                assert_eq!(reset_scope, ResetScope::All);
                (events, last_event_id)
            }
            other => panic!("full recovery should return a tree dump, got: {other:?}"),
        }
    }

    /// The blocks a full recovery of `indexer` would install.
    async fn dumped_blocks(indexer: &LocalKvIndexer) -> FxHashSet<DumpedBlock> {
        let mut blocks = FxHashSet::default();
        for event in tree_dump(indexer).await.0 {
            let KvCacheEventData::Stored(store) = &event.event.data else {
                panic!("a tree dump holds only stores, got: {event:?}");
            };
            let mut parent = store.parent_hash.map(|hash| hash.0);
            for block in &store.blocks {
                blocks.insert((
                    event.storage_tier,
                    event.worker_id,
                    event.event.dp_rank,
                    parent,
                    block.block_hash.0,
                    block.tokens_hash.0,
                ));
                parent = Some(block.block_hash.0);
            }
        }
        blocks
    }

    #[tokio::test]
    async fn lower_tier_events_are_buffered_and_dumped_on_their_own_tier() {
        let indexer = make_indexer();
        let event = lower_tier_store_event(7, 0, 1, 900, 11, 101, StorageTier::HostPinned);

        indexer.apply_event_with_buffer(event.clone());

        assert_eq!(indexer.get_all_events_in_buffer(), vec![event]);
        // On the host tier only: nothing is dumped for the device tier.
        assert_eq!(
            dumped_blocks(&indexer).await,
            FxHashSet::from_iter([(StorageTier::HostPinned, 7, 0, Some(900), 101, 11)])
        );
    }

    #[tokio::test]
    async fn lower_tier_events_are_partitioned_by_storage_tier() {
        let indexer = make_indexer();

        indexer.apply_event_with_buffer(lower_tier_store_event(
            19,
            0,
            1,
            1000,
            31,
            301,
            StorageTier::HostPinned,
        ));
        indexer.apply_event_with_buffer(lower_tier_store_event(
            19,
            0,
            2,
            2000,
            31,
            302,
            StorageTier::Disk,
        ));

        assert_eq!(
            dumped_blocks(&indexer).await,
            FxHashSet::from_iter([
                (StorageTier::HostPinned, 19, 0, Some(1000), 301, 31),
                (StorageTier::Disk, 19, 0, Some(2000), 302, 31),
            ])
        );
    }

    #[tokio::test]
    async fn scoped_clear_history_and_all_tier_snapshot_preserve_cache_owner() {
        let indexer = make_indexer();

        indexer.apply_event_with_buffer(lower_tier_store_event_in_domain(
            19,
            1,
            900,
            11,
            101,
            ResidencyDomain::CacheOwner,
        ));
        indexer.apply_event_with_buffer(lower_tier_store_event_in_domain(
            19,
            2,
            900,
            11,
            101,
            ResidencyDomain::Worker,
        ));
        indexer.apply_event_with_buffer(RouterEvent::with_residency_domain(
            19,
            KvCacheEvent {
                event_id: 3,
                data: KvCacheEventData::Cleared,
                dp_rank: 0,
            },
            StorageTier::Device,
            ResidencyDomain::Worker,
        ));
        indexer.apply_event_with_buffer(lower_tier_store_event_in_domain(
            19,
            4,
            101,
            12,
            102,
            ResidencyDomain::CacheOwner,
        ));

        let WorkerKvQueryResponse::Events { events, .. } =
            indexer.get_events_in_id_range(Some(1), Some(4)).await
        else {
            panic!("history should be served from the recovery buffer");
        };
        assert_eq!(
            events
                .iter()
                .map(|event| event.event.event_id)
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );

        // The worker-domain clear took the worker's copy and left the cache owner's.
        let (events, last_event_id) = tree_dump(&indexer).await;
        assert_eq!(last_event_id, 4);
        assert_eq!(events.len(), 2);
        assert!(events.iter().all(|event| {
            event.storage_tier == StorageTier::HostPinned
                && event.resolved_residency_domain() == Ok(ResidencyDomain::CacheOwner)
                && event.state_source == Some(cache_owner_id())
        }));

        // An indexer fed the dump holds what the first one held.
        let restored = make_indexer();
        for event in events {
            restored.apply_event_with_buffer(event);
        }
        let expected = FxHashSet::from_iter([
            (StorageTier::HostPinned, 0, 0, Some(900), 101, 11),
            (StorageTier::HostPinned, 0, 0, Some(101), 102, 12),
        ]);
        assert_eq!(dumped_blocks(&indexer).await, expected);
        assert_eq!(dumped_blocks(&restored).await, expected);
    }

    #[tokio::test]
    async fn cleared_event_only_clears_target_rank_across_lower_tiers() {
        let indexer = make_indexer();

        indexer.apply_event_with_buffer(lower_tier_store_event(
            11,
            0,
            1,
            1000,
            21,
            201,
            StorageTier::HostPinned,
        ));
        indexer.apply_event_with_buffer(lower_tier_store_event(
            11,
            1,
            2,
            2000,
            22,
            202,
            StorageTier::HostPinned,
        ));

        indexer.apply_event_with_buffer(RouterEvent::with_storage_tier(
            11,
            KvCacheEvent {
                event_id: 3,
                data: KvCacheEventData::Cleared,
                dp_rank: 0,
            },
            StorageTier::HostPinned,
        ));

        assert_eq!(
            dumped_blocks(&indexer).await,
            FxHashSet::from_iter([(StorageTier::HostPinned, 11, 1, Some(2000), 202, 22)])
        );
    }

    /// A device block below an evicted block is left out of a dump, and is back in a dump cut
    /// after the evicted block is stored again: the engine never removed it. A dump cut
    /// before that cannot bring it back through its tail, which only misses a reuse.
    #[tokio::test]
    async fn device_blocks_below_an_evicted_block_return_when_it_is_stored_again() {
        use crate::testing::{create_remove_event, create_store_event};

        let device = |blocks: &[u64]| {
            let mut parent = None;
            FxHashSet::from_iter(blocks.iter().map(|&block| {
                let dumped = (StorageTier::Device, 7, 0, parent, block * 100, block);
                parent = Some(block * 100);
                dumped
            }))
        };
        let evict = [
            create_store_event(7, 1, vec![1, 2, 3], None),
            create_remove_event(7, 2, vec![2]),
        ];
        let store_again = create_store_event(7, 3, vec![2], Some(ExternalSequenceBlockHash(100)));

        let cut_before = make_indexer();
        let cut_after = make_indexer();
        for event in evict {
            cut_before.apply_event_with_buffer(event.clone());
            cut_after.apply_event_with_buffer(event);
        }
        assert_eq!(dumped_blocks(&cut_before).await, device(&[1]));

        for indexer in [&cut_before, &cut_after] {
            indexer.apply_event_with_buffer(store_again.clone());
        }
        assert_eq!(dumped_blocks(&cut_before).await, device(&[1, 2]));
        assert_eq!(dumped_blocks(&cut_after).await, device(&[1, 2, 3]));
    }

    /// A lower-tier index restored from a tree dump holds what one that followed the same
    /// events live holds, tier by tier. The disk event gives block 101 a different token hash
    /// than the host event did, as a store that omits a salt does.
    #[tokio::test]
    async fn tree_dump_restores_each_lower_tier_as_live_ingestion_built_it() {
        use crate::{KvIndexerInterface, LowerTierIndexers};

        async fn held(
            indexers: &LowerTierIndexers,
            tier: StorageTier,
        ) -> FxHashSet<(Option<u64>, u64, u64)> {
            let events = indexers.get_or_create(tier).dump_events().await.unwrap();
            let blocks = events.iter().flat_map(|event| match &event.event.data {
                KvCacheEventData::Stored(store) => store
                    .blocks
                    .iter()
                    .map(|block| {
                        (
                            store.parent_hash.map(|hash| hash.0),
                            block.block_hash.0,
                            block.tokens_hash.0,
                        )
                    })
                    .collect::<Vec<_>>(),
                _ => Vec::new(),
            });
            blocks.collect()
        }

        let events = [
            lower_tier_store_event(7, 0, 1, 900, 11, 101, StorageTier::HostPinned),
            lower_tier_store_event(7, 0, 2, 900, 12, 101, StorageTier::Disk),
            lower_tier_store_event(7, 0, 3, 101, 13, 102, StorageTier::Disk),
        ];
        let indexer = make_indexer();
        let live = LowerTierIndexers::new(1, 4);
        for event in events {
            live.get_or_create(event.storage_tier)
                .apply_event_and_wait(event.clone())
                .await
                .unwrap();
            indexer.apply_event_with_buffer(event);
        }

        let restored = LowerTierIndexers::new(1, 4);
        for event in tree_dump(&indexer).await.0 {
            restored
                .get_or_create(event.storage_tier)
                .apply_event_and_wait(event)
                .await
                .unwrap();
        }

        assert_eq!(
            held(&live, StorageTier::Disk).await,
            FxHashSet::from_iter([(Some(900), 101, 12), (Some(101), 102, 13)])
        );
        for tier in [StorageTier::HostPinned, StorageTier::Disk] {
            assert_eq!(held(&restored, tier).await, held(&live, tier).await);
        }
    }
}
