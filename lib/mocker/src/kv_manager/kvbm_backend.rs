// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # KV Manager (kvbm-logical G1 backend)
//!
//! Synchronous vLLM-flavour G1 block manager built on `kvbm-logical::BlockManager<G1>`.
//! Translates the mocker's `MoveBlock` protocol into the RAII lifecycle
//! (allocate → stage → register → drop) exposed by kvbm-logical.
//!
//! ## MoveBlock semantics
//!
//! - **Use**: prepare all active/inactive hits and fresh slots as one
//!   transaction, then commit the whole request atomically. Capacity exhaustion
//!   leaves ownership and sequence state unchanged so the scheduler can decide
//!   whether to preempt a running request.
//! - **Deref**: release one logical request owner. For `PartialBlock` this
//!   drops the unique `MutableBlock` and returns it to the reset pool. For
//!   `FullBlock` this decrements an explicit logical refcount; the final
//!   release drops the canonical `ImmutableBlock` and transitions the block to
//!   kvbm-logical's inactive pool (RAII return).
//! - **Promote**: PartialBlock (`MutableBlock`) → FullBlock (`ImmutableBlock`).
//!   Collapses onto an existing registered handle if the PLH / SequenceHash is
//!   already present; otherwise stages + registers a new block.
//!
//! ## Eviction backends
//!
//! Three backends are exposed via [`MockerEvictionBackend`]:
//! - `Lineage` (default) — parent-chain aware, evicts leaves first. Subsumes
//!   the `push_front` preemption-priority behaviour of the old `LRUEvictor`.
//! - `Lru` — simple recency-based LRU.
//! - `MultiLru` — 4-tier frequency-aware LRU (requires TinyLFU tracker).

use std::collections::hash_map::Entry;
use std::sync::Arc;
#[cfg(feature = "kvbm-offload")]
use std::sync::Mutex;

use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash, StorageTier,
};
use dynamo_tokens::blocks::UniqueBlock;
use dynamo_tokens::{BlockHash, PositionalLineageHash, SequenceHash};
use kvbm_logical::blocks::BlockDuplicationPolicy;
use kvbm_logical::registry::BlockRegistry;
use kvbm_logical::tinylfu::TinyLFUTracker;
use kvbm_logical::{BlockManager, ImmutableBlock, MutableBlock};
use rustc_hash::FxHashMap;
use uuid::Uuid;

use crate::common::kv_cache_trace;
use crate::common::protocols::{
    G1, KvEventPublishers, MockerEvictionBackend, MoveBlock, PrefillCost,
};
use crate::common::sequence::ActiveSequence;
use crate::kv_manager::{G1Acquire, OffloadDependency};
#[cfg(feature = "kvbm-offload")]
use crate::kvbm_offload::{
    G1EvictionOutcome, G2BlockEventMetadata, G2OffloadBlock, G2RouterEvent, MockOffloadEngine,
    OffloadId, SwapInHandle,
};

/// Outcome of [`KvManager::try_batch_swap_in`]. The caller uses this to
/// decide whether to park the request on a pending-swap-in queue or to
/// fall through to normal G1 allocation.
#[cfg(feature = "kvbm-offload")]
pub enum BatchSwapInOutcome {
    /// No G2 hits (or no offload engine attached). Caller must allocate
    /// fresh G1 blocks.
    NoHits,
    /// Swap-in reservation accepted. Caller parks the request with this
    /// handle and polls `SwapInHandle::is_complete()` on subsequent
    /// scheduler passes. The coordinator retains matched lower-tier blocks,
    /// G1 destination slots, and any pinned cached prefix for the transfer.
    Scheduled { handle: SwapInHandle },
    /// G2 had a match, but reserving destination G1 slots first had to
    /// trigger a G1→G2 eviction. Caller should retry after offload advances.
    BlockedOnG1Offload(OffloadDependency),
}

#[cfg(feature = "kvbm-offload")]
pub struct SwapInRegistrationOutcome {
    pub consumed_entries: usize,
}

#[cfg(feature = "kvbm-offload")]
pub(crate) struct SwapInRegistrationBlock {
    pub(crate) seq_hash: SequenceHash,
    pub(crate) plh: PositionalLineageHash,
    pub(crate) local_hash: Option<BlockHash>,
    pub(crate) token_ids: Option<Vec<u32>>,
}

#[cfg(feature = "kvbm-offload")]
enum SwapInSlotReservation {
    Reserved(Vec<MutableBlock<G1>>),
    BlockedOnG1Offload(OffloadDependency),
    NoCapacity,
}

/// Classification for each block processed inside `Use`.
///
/// - `ActiveHit`: block is already pinned in `active_full` / `active_partial`;
///   commit bumps its explicit logical refcount without cloning a handle.
/// - `InactiveHit`: block was in kvbm-logical's inactive pool and was
///   reactivated by the aligned scattered batch lookup.
/// - `NewStore`: block was freshly allocated, staged, and registered.
///
/// The router radix tree already knows about `ActiveHit` and `InactiveHit`
/// (it only forgets on explicit `Removed`), so only `NewStore` should emit a
/// `Stored` KV event. Both hit outcomes still advance the parent cursor so
/// subsequent `NewStore` batches anchor to the last reused full block.
#[cfg(not(feature = "kvbm-offload"))]
enum G1EvictionOutcome {}

enum PreparedUseBlock {
    /// Already represented in `active_full`; no temporary RAII clone is
    /// needed while the transaction reserves its fresh suffix.
    ExistingActiveFull {
        seq_hash: SequenceHash,
    },
    /// Resurrected from KVBM's inactive pool (or otherwise matched outside the
    /// mocker's active map). The handle pins it until commit or rollback.
    ExistingMatchedFull {
        seq_hash: SequenceHash,
        handle: ImmutableBlock<G1>,
    },
    /// Not present in `active_full`; resolved by the single aligned scattered
    /// lookup after the initial classification pass.
    PendingNonLocalFull {
        seq_hash: SequenceHash,
        full_idx: usize,
    },
    ExistingPartial,
    FreshFull {
        seq_hash: SequenceHash,
        full_idx: usize,
        mutable: Option<MutableBlock<G1>>,
    },
    FreshPartial {
        uuid: Uuid,
        mutable: Option<MutableBlock<G1>>,
    },
}

struct UseSignalRef<'a> {
    local_hashes: &'a [BlockHash],
    plhs: &'a [PositionalLineageHash],
    token_ids: Option<&'a [Vec<u32>]>,
    parent: Option<&'a UniqueBlock>,
}

struct UseTransaction<'a> {
    signal: UseSignalRef<'a>,
    prepared: Vec<PreparedUseBlock>,
    fresh_full_blocks: usize,
    evicted_plhs: Vec<PositionalLineageHash>,
}

struct G1SlotReservation {
    blocks: Vec<MutableBlock<G1>>,
    evicted_plhs: Vec<PositionalLineageHash>,
}

pub struct DecodeBlockReservation {
    blocks: Vec<MutableBlock<G1>>,
}

pub struct VllmDestinationReservation {
    cached_prefix: Vec<(SequenceHash, ImmutableBlock<G1>)>,
    unpublished_blocks: Vec<MutableBlock<G1>>,
    layout: Option<MoveBlock>,
}

impl VllmDestinationReservation {
    pub(crate) fn transferable_prompt_tokens(&self, block_size: usize) -> usize {
        self.unpublished_blocks.len().saturating_mul(block_size)
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.cached_prefix.len() + self.unpublished_blocks.len()
    }

    #[cfg(test)]
    pub(crate) fn block_ids(&self) -> Vec<usize> {
        self.cached_prefix
            .iter()
            .map(|(_, block)| block.block_id())
            .chain(self.unpublished_blocks.iter().map(MutableBlock::block_id))
            .collect()
    }
}

impl DecodeBlockReservation {
    fn take(&mut self) -> Option<MutableBlock<G1>> {
        self.blocks.pop()
    }

    pub(crate) fn len(&self) -> usize {
        self.blocks.len()
    }
}

#[derive(Clone)]
struct RegisteredBlockInfo {
    seq_hash: SequenceHash,
    #[cfg_attr(not(feature = "kvbm-offload"), allow(dead_code))]
    block_id: usize,
    #[cfg_attr(not(feature = "kvbm-offload"), allow(dead_code))]
    parent_hash: Option<SequenceHash>,
    #[cfg_attr(not(feature = "kvbm-offload"), allow(dead_code))]
    local_hash: Option<BlockHash>,
    #[cfg_attr(not(feature = "kvbm-offload"), allow(dead_code))]
    token_ids: Option<Vec<u32>>,
}

struct FullBlockMetadata {
    seq_hash: SequenceHash,
    plh: PositionalLineageHash,
    parent_hash: Option<SequenceHash>,
    local_hash: Option<BlockHash>,
    token_ids: Option<Vec<u32>>,
}

/// One physical full-block pin plus the number of logical request owners.
///
/// `ImmutableBlock` clones are physical-lifetime guards, not request block
/// tables. Keeping a clone per logical owner needlessly makes KVBM's handle
/// count, allocator traffic, and Arc traffic scale with prefix sharing. The
/// canonical handle pins the physical block while `logical_refs` tracks the
/// ownership semantics the mocker needs for Deref.
struct ActiveFullBlock {
    handle: ImmutableBlock<G1>,
    logical_refs: usize,
}

impl ActiveFullBlock {
    fn new(handle: ImmutableBlock<G1>) -> Self {
        Self {
            handle,
            logical_refs: 1,
        }
    }

    fn retain(&mut self) {
        self.logical_refs = self
            .logical_refs
            .checked_add(1)
            .expect("active full-block logical reference count overflowed");
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FullBlockCommit {
    Reused,
    Stored,
}

/// Synchronous G1 KV block manager backed by `kvbm-logical::BlockManager<G1>`.
pub struct KvManager {
    block_manager: BlockManager<G1>,
    max_capacity: usize,
    block_size: usize,
    kv_event_publishers: KvEventPublishers,
    dp_rank: u32,
    next_event_id: u64,

    /// PartialBlocks (still filling tokens) held as `MutableBlock`.
    /// Dropped blocks return to kvbm-logical's reset pool.
    active_partial: FxHashMap<Uuid, MutableBlock<G1>>,

    /// FullBlocks held as one canonical `ImmutableBlock` per physical block,
    /// keyed by `SequenceHash`, plus the number of logical request owners.
    /// The final logical `Deref` drops the canonical handle and transitions the
    /// block to kvbm-logical's inactive pool.
    active_full: FxHashMap<SequenceHash, ActiveFullBlock>,

    /// Shadow registry for every block registered in kvbm-logical. The logical
    /// registry is keyed by `PositionalLineageHash`, while the router's radix
    /// tree is keyed by the mocker's u64 `SequenceHash`; the physical G1 block
    /// id is kept so offload simulation can enqueue the actual block shape when
    /// kvbm-logical later evicts it from the inactive pool.
    registered_blocks: FxHashMap<PositionalLineageHash, RegisteredBlockInfo>,

    /// Handle to the G1↔G2 offload engine. `None` until
    /// [`attach_new_offload_engine`](Self::attach_new_offload_engine) wires
    /// one in after construction (the engine is built async and cannot be
    /// created inside `new_*`).
    ///
    /// Mocker source-lifetime note: G1 eviction hands kvbm-engine
    /// `SourceBlocks::External(block_id, plh)` without a strong immutable G1
    /// block ref. A real byte copy still needs the source HBM slot to stay
    /// unavailable until DMA completes, so the mocker holds the reset
    /// `MutableBlock<G1>` capacity token inside the offload engine until the
    /// simulated transfer completes. The worker never reads source bytes;
    /// destination presence is registered by `plh`.
    #[cfg(feature = "kvbm-offload")]
    offload_engine: Option<Arc<Mutex<MockOffloadEngine>>>,

    /// Changes whenever modeled G1 allocability increases. Immediate retry
    /// witnesses must name the current generation and a positive slot delta.
    capacity_generation: u64,
}

impl KvManager {
    pub fn new_with_event_sink(
        max_capacity: usize,
        block_size: usize,
        kv_event_publishers: KvEventPublishers,
        dp_rank: u32,
    ) -> Self {
        Self::new_with_eviction_backend(
            max_capacity,
            block_size,
            kv_event_publishers,
            dp_rank,
            MockerEvictionBackend::default(),
        )
    }

    pub fn new_with_eviction_backend(
        max_capacity: usize,
        block_size: usize,
        kv_event_publishers: KvEventPublishers,
        dp_rank: u32,
        eviction_backend: MockerEvictionBackend,
    ) -> Self {
        debug_assert!(max_capacity > 0, "max_capacity must be > 0");

        let mut registry_builder = BlockRegistry::builder();
        if matches!(eviction_backend, MockerEvictionBackend::MultiLru) {
            let tracker = Arc::new(TinyLFUTracker::new(max_capacity));
            registry_builder = registry_builder.frequency_tracker(tracker);
        }
        let registry = registry_builder.build();

        let mut mgr_builder = BlockManager::builder()
            .block_count(max_capacity)
            .block_size(block_size)
            .registry(registry)
            .duplication_policy(BlockDuplicationPolicy::Reject);

        // Intentional vLLM drift: upstream permits duplicate physical blocks
        // for append-only request block tables. The mocker has one canonical
        // registered_blocks entry per PLH and no request-owned physical block
        // tables or duplicate-reset events, so Allow would make offload metadata
        // and KV-event cleanup unsafe. Reject may discard a reserved block,
        // adopt an existing block ID, reduce occupancy, and omit duplicate Stored.
        // Revisit Allow only with per-request physical ownership, duplicate-aware
        // offload metadata, and balanced duplicate Stored/Removed lifecycle events.
        mgr_builder = match eviction_backend {
            MockerEvictionBackend::Lineage => mgr_builder.with_lineage_backend(),
            MockerEvictionBackend::Lru => mgr_builder.with_lru_backend(),
            MockerEvictionBackend::MultiLru => mgr_builder.with_multi_lru_backend(),
        };
        let block_manager = mgr_builder.build().expect("BlockManager build failed");

        if !kv_event_publishers.is_empty() {
            tracing::info!(
                "KvManager initialized with event sink for DP rank {dp_rank} with block_size {block_size}, eviction={eviction_backend:?}"
            );
        }

        Self {
            block_manager,
            max_capacity,
            block_size,
            kv_event_publishers,
            dp_rank,
            next_event_id: 0,
            active_partial: FxHashMap::default(),
            active_full: FxHashMap::default(),
            registered_blocks: FxHashMap::default(),
            #[cfg(feature = "kvbm-offload")]
            offload_engine: None,
            capacity_generation: 0,
        }
    }

    /// Install a newly acquired physical handle or merge it into an entry that
    /// became active earlier in the same serial commit.
    fn insert_or_retain_active_full(&mut self, seq_hash: SequenceHash, handle: ImmutableBlock<G1>) {
        match self.active_full.entry(seq_hash) {
            Entry::Vacant(entry) => {
                entry.insert(ActiveFullBlock::new(handle));
            }
            Entry::Occupied(mut entry) => {
                assert_eq!(
                    entry.get().handle.block_id(),
                    handle.block_id(),
                    "active full-block hash resolved to a different physical block"
                );
                entry.get_mut().retain();
                // `handle` is a redundant physical pin. Dropping it leaves the
                // canonical entry alive while logical ownership is tracked by
                // `logical_refs`.
                drop(handle);
            }
        }
    }

    /// Add one logical owner for a block known to be present in the active map.
    /// This is called only after every fallible reservation for the surrounding
    /// Use transaction has succeeded.
    fn retain_active_full(&mut self, seq_hash: SequenceHash) {
        self.active_full
            .get_mut(&seq_hash)
            .unwrap_or_else(|| panic!("active full block {seq_hash:?} disappeared before commit"))
            .retain();
    }

    /// Release one logical owner. Removing the final entry drops the sole
    /// physical handle and lets KVBM transition the block to inactive.
    fn release_active_full(&mut self, seq_hash: SequenceHash) {
        let Entry::Occupied(mut entry) = self.active_full.entry(seq_hash) else {
            panic!("Deref: full block not in active pool");
        };
        assert!(
            entry.get().logical_refs > 0,
            "active full block must retain at least one logical owner"
        );
        if entry.get().logical_refs == 1 {
            entry.remove();
        } else {
            entry.get_mut().logical_refs -= 1;
        }
    }

    /// Wrap `engine` in `Arc<Mutex<_>>`, install it onto this
    /// `KvManager`, and return a clone of the Arc to the caller.
    /// Called once after construction by the scheduler's init helper;
    /// a second call replaces the previous engine (primarily for tests).
    #[cfg(feature = "kvbm-offload")]
    pub fn attach_new_offload_engine(
        &mut self,
        engine: MockOffloadEngine,
    ) -> Arc<Mutex<MockOffloadEngine>> {
        let shared = Arc::new(Mutex::new(engine));
        self.offload_engine = Some(shared.clone());
        shared
    }

    /// `true` once an offload engine has been attached.
    #[cfg(feature = "kvbm-offload")]
    pub fn has_offload_engine(&self) -> bool {
        self.offload_engine.is_some()
    }

    /// Advance the offload engine's PS models and fire any
    /// completion sinks for drained transfers. Scheduler calls this at
    /// the top of every pass so swap-in statuses publish before the
    /// promote-completed loop runs, and offload awaiters fire before
    /// the next enqueue measures the active-set size. No-op when no
    /// engine is attached.
    #[cfg(feature = "kvbm-offload")]
    pub fn tick_offload_engine(&mut self, now_ms: f64) {
        let Some(engine_arc) = self.offload_engine.clone() else {
            return;
        };
        let prepared = {
            let engine = engine_arc.lock().expect("offload engine mutex poisoned");
            engine.prepare_tick_for_kv_manager(now_ms)
        };
        self.publish_g2_router_events(prepared.router_events);
        let released_g1_slots = engine_arc
            .lock()
            .expect("offload engine mutex poisoned")
            .acknowledge_tick_for_kv_manager(prepared.acknowledgement)
            .expect("freshly prepared offload advance must acknowledge");
        if released_g1_slots > 0 {
            self.bump_capacity_generation(released_g1_slots);
        }
    }

    /// Earliest pending completion time across offload + onboard links,
    /// or `None` when both are idle or no engine is attached. Scheduler
    /// uses this to drive stall-advance in virtual-time replay.
    #[cfg(feature = "kvbm-offload")]
    pub fn earliest_offload_deadline(&self) -> Option<f64> {
        let engine_arc = self.offload_engine.as_ref()?;
        let engine = engine_arc.lock().expect("offload engine mutex poisoned");
        engine.earliest_pending_deadline()
    }

    #[cfg(feature = "kvbm-offload")]
    pub(crate) fn refresh_offload_dependency(
        &self,
        dependency: OffloadDependency,
    ) -> Option<OffloadDependency> {
        let engine_arc = self.offload_engine.as_ref()?;
        let engine = engine_arc.lock().expect("offload engine mutex poisoned");
        engine
            .g1_offload_dependency(dependency.offload_id)
            .map(|(offload_id, deadline_ms)| OffloadDependency {
                offload_id,
                deadline_ms,
            })
    }

    #[cfg(not(feature = "kvbm-offload"))]
    pub(crate) fn refresh_offload_dependency(
        &self,
        _dependency: OffloadDependency,
    ) -> Option<OffloadDependency> {
        None
    }

    /// Hand blocks that were actually evicted from G1 inactive to the
    /// offload engine as mock `ExternalBlock`s (no strong immutable ref; see
    /// `offload_engine` field docs). When capacity pressure tried to reuse
    /// the same G1 slots, `source_slots` carries reset `MutableBlock` tokens
    /// that must remain unavailable until the simulated source copy finishes.
    #[cfg(feature = "kvbm-offload")]
    fn enqueue_evictions_to_g2(
        &mut self,
        evicted: &[G2OffloadBlock],
        source_slots: Vec<MutableBlock<G1>>,
        now_ms: Option<f64>,
    ) -> (Vec<G2RouterEvent>, Option<G1EvictionOutcome>) {
        let Some(engine_arc) = self.offload_engine.as_ref() else {
            drop(source_slots);
            return (Vec::new(), None);
        };
        if evicted.is_empty() {
            drop(source_slots);
            return (Vec::new(), None);
        }
        let mut engine = engine_arc.lock().expect("offload engine mutex poisoned");
        let outcome = engine.enqueue_g1_evictions_with_metadata(evicted, source_slots, now_ms);
        (engine.drain_g2_router_events(), outcome)
    }

    /// Register a batch of completed G2-swapped-in blocks into the G1
    /// inactive pool. `destination_slots` were reserved before the G2→G1
    /// transfer started and are consumed here as DMA write targets.
    ///
    /// Entries already cached in G1 (active or inactive) are skipped, but still
    /// advance the parent cursor so later fresh suffix stores publish the same
    /// router tree shape as `process_use`.
    #[cfg(feature = "kvbm-offload")]
    pub(crate) fn register_swapped_in_blocks(
        &mut self,
        entries: Vec<SwapInRegistrationBlock>,
        initial_parent_hash: Option<SequenceHash>,
        destination_slots: Vec<MutableBlock<G1>>,
    ) -> SwapInRegistrationOutcome {
        let total_entries = entries.len();
        let mut stored_seq_hashes = Vec::with_capacity(total_entries);
        let mut stored_local_hashes = Vec::with_capacity(total_entries);
        let mut stored_token_ids = Vec::with_capacity(total_entries);
        let mut stored_parent_hash = initial_parent_hash;
        let mut metadata_parent_hash = initial_parent_hash;
        let mut consumed_entries = 0usize;
        let mut destination_slots = destination_slots.into_iter();

        for entry in entries {
            let Some(mutable) = destination_slots.next() else {
                tracing::warn!(
                    consumed_entries,
                    entries = total_entries,
                    "kvbm-offload: swap-in registration ran out of reserved G1 slots"
                );
                break;
            };
            if self.active_full.contains_key(&entry.seq_hash) {
                drop(mutable);
                if !stored_seq_hashes.is_empty() {
                    self.publish_swap_in_stored_batch(
                        &mut stored_seq_hashes,
                        &mut stored_local_hashes,
                        &mut stored_token_ids,
                        stored_parent_hash,
                    );
                }
                stored_parent_hash = Some(entry.seq_hash);
                metadata_parent_hash = Some(entry.seq_hash);
                consumed_entries += 1;
                continue;
            }
            let presence = self
                .block_manager
                .block_registry()
                .check_presence::<G1>(&[entry.plh]);
            if presence.first().is_some_and(|(_, p)| *p) {
                drop(mutable);
                if !stored_seq_hashes.is_empty() {
                    self.publish_swap_in_stored_batch(
                        &mut stored_seq_hashes,
                        &mut stored_local_hashes,
                        &mut stored_token_ids,
                        stored_parent_hash,
                    );
                }
                stored_parent_hash = Some(entry.seq_hash);
                metadata_parent_hash = Some(entry.seq_hash);
                consumed_entries += 1;
                continue;
            }
            let complete = mutable
                .stage(entry.plh, self.block_size)
                .expect("stage failed during swap-in registration");
            let immutable = self.block_manager.register_block(complete);
            let block_id = immutable.block_id();
            // Drop ImmutableBlock → block lands in kvbm-logical's
            // inactive pool, where `process_use`'s `match_blocks`
            // later reactivates it.
            drop(immutable);
            // Clone token_ids only when downstream still needs both copies
            // (registry + publish batch). The publish batch takes ownership.
            let registry_token_ids = entry.token_ids.clone();
            if let Some(token_ids) = entry.token_ids {
                stored_token_ids.push(token_ids);
            }
            self.registered_blocks.insert(
                entry.plh,
                RegisteredBlockInfo {
                    seq_hash: entry.seq_hash,
                    block_id,
                    parent_hash: metadata_parent_hash,
                    local_hash: entry.local_hash,
                    token_ids: registry_token_ids,
                },
            );
            stored_seq_hashes.push(entry.seq_hash);
            if let Some(local_hash) = entry.local_hash {
                stored_local_hashes.push(local_hash);
            }
            metadata_parent_hash = Some(entry.seq_hash);
            consumed_entries += 1;
        }

        if !stored_seq_hashes.is_empty() {
            self.publish_swap_in_stored_batch(
                &mut stored_seq_hashes,
                &mut stored_local_hashes,
                &mut stored_token_ids,
                stored_parent_hash,
            );
        }

        SwapInRegistrationOutcome { consumed_entries }
    }

    #[cfg(feature = "kvbm-offload")]
    fn publish_swap_in_stored_batch(
        &mut self,
        stored_seq_hashes: &mut Vec<SequenceHash>,
        stored_local_hashes: &mut Vec<BlockHash>,
        stored_token_ids: &mut Vec<Vec<u32>>,
        parent_hash: Option<SequenceHash>,
    ) {
        if stored_seq_hashes.is_empty() {
            return;
        }

        let full_blocks = std::mem::take(stored_seq_hashes);
        let local_hashes = if stored_local_hashes.len() == full_blocks.len() {
            std::mem::take(stored_local_hashes)
        } else {
            stored_local_hashes.clear();
            Vec::new()
        };
        let token_ids = if stored_token_ids.len() == full_blocks.len() {
            Some(std::mem::take(stored_token_ids))
        } else {
            stored_token_ids.clear();
            None
        };

        self.publish_kv_event(full_blocks, &local_hashes, parent_hash, true, token_ids);
    }

    /// Try to satisfy a request's remaining prefix via a G2→G1 swap-in.
    ///
    /// Admission path stays linear: `active → inactive → (this) →
    /// allocate fresh`. Returns [`BatchSwapInOutcome::NoHits`] when no
    /// engine is attached or when no configured lower tier holds
    /// `remaining_plhs`.
    ///
    /// Lower tiers are keyed by `PositionalLineageHash` (kvbm-engine's
    /// native identity), not the router-facing `u64` SequenceHash — the
    /// caller already holds these on the admission path. We first prepare the
    /// lower-tier match, then reserve destination G1 slots, and only then
    /// reserve onboard bandwidth. That prevents swap-in from borrowing
    /// imaginary HBM capacity while the transfer is in flight.
    #[cfg(feature = "kvbm-offload")]
    pub fn try_batch_swap_in(
        &mut self,
        remaining_plhs: &[PositionalLineageHash],
        prefix_pins: Vec<ImmutableBlock<G1>>,
        now_ms: Option<f64>,
    ) -> BatchSwapInOutcome {
        let Some(engine_arc) = self.offload_engine.clone() else {
            return BatchSwapInOutcome::NoHits;
        };
        let Some(prepared) = ({
            let mut engine = engine_arc.lock().expect("offload engine mutex poisoned");
            engine.prepare_onboard_prefix(remaining_plhs)
        }) else {
            return BatchSwapInOutcome::NoHits;
        };
        let block_count = prepared.block_count();
        // Do not hold the offload-engine mutex while reserving G1 slots:
        // allocation may evict G1 blocks and enqueue G1→G2 work back into
        // the same engine. `PreparedSwapIn` pins ready G2 blocks, and for
        // deferred G3 staging it holds only the G2 staging capacity so a failed
        // admission probe does not start a G3→G2 copy.
        let destination_slots = match self.reserve_swap_in_destination_slots(block_count) {
            SwapInSlotReservation::Reserved(slots) => slots,
            SwapInSlotReservation::BlockedOnG1Offload(dependency) => {
                return BatchSwapInOutcome::BlockedOnG1Offload(dependency);
            }
            SwapInSlotReservation::NoCapacity => return BatchSwapInOutcome::NoHits,
        };
        let handle = {
            let mut engine = engine_arc.lock().expect("offload engine mutex poisoned");
            engine.start_onboard_prefix(prepared, destination_slots, prefix_pins, now_ms)
        };
        BatchSwapInOutcome::Scheduled { handle }
    }

    #[cfg(feature = "kvbm-offload")]
    pub(crate) fn cancel_swap_in(&mut self, id: OffloadId) -> bool {
        self.offload_engine.as_ref().is_some_and(|engine| {
            engine
                .lock()
                .expect("offload engine mutex poisoned")
                .cancel_swap_in(id)
        })
    }

    #[cfg(feature = "kvbm-offload")]
    pub(crate) fn register_completed_swap_in(
        &mut self,
        id: OffloadId,
        entries: Vec<SwapInRegistrationBlock>,
        parent_hash: Option<SequenceHash>,
    ) -> SwapInRegistrationOutcome {
        let (destination_slots, prefix_pins) = self
            .offload_engine
            .as_ref()
            .and_then(|engine| {
                engine
                    .lock()
                    .expect("offload engine mutex poisoned")
                    .take_completed_swap_in(id)
            })
            .expect("completed swap-in lease must retain its G1 resources");
        let outcome = self.register_swapped_in_blocks(entries, parent_hash, destination_slots);
        drop(prefix_pins);
        outcome
    }

    /// Hold the G1 prefix that admission used when deciding to swap in only a
    /// G2 suffix. The returned guards keep those blocks out of the inactive
    /// eviction pool until the pending swap-in publishes its Device events.
    #[cfg(feature = "kvbm-offload")]
    pub(crate) fn try_pin_g1_prefix(
        &mut self,
        prefix_plhs: &[PositionalLineageHash],
    ) -> Option<Vec<ImmutableBlock<G1>>> {
        if prefix_plhs.is_empty() {
            return Some(Vec::new());
        }

        let pins = self.block_manager.match_blocks(prefix_plhs);
        if pins.len() == prefix_plhs.len() {
            Some(pins)
        } else {
            None
        }
    }

    /// Emit a `Stored` or `Removed` KV event to the router.
    /// Ported verbatim from the old `vllm_backend::publish_kv_event` to
    /// preserve KV-aware routing semantics (parent_hash chaining, token_ids).
    fn publish_kv_event(
        &mut self,
        full_blocks: Vec<SequenceHash>,
        local_hashes: &[BlockHash],
        parent_hash: Option<u64>,
        is_store: bool,
        token_ids: Option<Vec<Vec<u32>>>,
    ) {
        self.publish_kv_event_for_tier(
            full_blocks,
            local_hashes,
            parent_hash,
            is_store,
            token_ids,
            StorageTier::Device,
        );
    }

    fn publish_kv_event_for_tier(
        &mut self,
        full_blocks: Vec<SequenceHash>,
        local_hashes: &[BlockHash],
        parent_hash: Option<u64>,
        is_store: bool,
        token_ids: Option<Vec<Vec<u32>>>,
        storage_tier: StorageTier,
    ) {
        if full_blocks.is_empty() {
            return;
        }

        kv_cache_trace::log_vllm_trace(
            if is_store { "allocation" } else { "eviction" },
            self.dp_rank,
            self.block_size,
            self.num_active_blocks(),
            self.num_inactive_blocks(),
            self.max_capacity,
        );

        if self.kv_event_publishers.is_empty() {
            return;
        }

        let event_data = if is_store {
            // `local_hashes` is either empty (caller has no token-derived
            // hashes to publish) or 1:1 with `full_blocks`. Match the
            // front-door contract in `process_use`.
            debug_assert!(
                local_hashes.is_empty() || local_hashes.len() == full_blocks.len(),
                "publish_kv_event: local_hashes must be empty or 1:1 with full_blocks ({} vs {})",
                local_hashes.len(),
                full_blocks.len(),
            );

            KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: parent_hash.map(ExternalSequenceBlockHash),
                start_position: None,
                blocks: full_blocks
                    .into_iter()
                    .enumerate()
                    .map(|(i, global_hash)| KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(global_hash),
                        tokens_hash: LocalBlockHash(
                            local_hashes.get(i).copied().unwrap_or_default(),
                        ),
                        mm_extra_info: None,
                    })
                    .collect(),
            })
        } else {
            KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: full_blocks
                    .into_iter()
                    .map(ExternalSequenceBlockHash)
                    .collect(),
            })
        };

        let event_id = self.next_event_id;
        self.next_event_id += 1;

        let event = KvCacheEvent {
            event_id,
            data: event_data,
            dp_rank: self.dp_rank,
        };

        if let Err(e) = self.kv_event_publishers.publish_with_storage_tier(
            event,
            token_ids.as_deref(),
            storage_tier,
        ) {
            tracing::warn!("Failed to publish KV event: {e}");
        }
    }

    #[cfg(feature = "kvbm-offload")]
    fn publish_g2_router_events(&mut self, events: Vec<G2RouterEvent>) {
        for event in events {
            match event {
                G2RouterEvent::Stored(meta) => {
                    let local_hashes = meta.local_hash.into_iter().collect::<Vec<_>>();
                    self.publish_kv_event_for_tier(
                        vec![meta.seq_hash],
                        &local_hashes,
                        meta.parent_hash,
                        true,
                        meta.token_ids.map(|ids| vec![ids]),
                        StorageTier::HostPinned,
                    );
                }
                G2RouterEvent::Removed { seq_hash } => {
                    self.publish_kv_event_for_tier(
                        vec![seq_hash],
                        &[],
                        None,
                        false,
                        None,
                        StorageTier::HostPinned,
                    );
                }
            }
        }
    }

    /// Process a `MoveBlock` instruction synchronously.
    ///
    /// `Use` is atomic: every block commits or none do. Capacity and offload
    /// waits are returned explicitly so callers cannot mistake a dependency for
    /// partial success.
    #[cfg_attr(feature = "profile", inline(never))]
    pub(crate) fn process(&mut self, event: &MoveBlock) -> G1Acquire<usize> {
        match event {
            MoveBlock::Use(blocks, local_hashes, plhs, token_ids, parent) => self.process_use(
                blocks,
                local_hashes,
                plhs,
                token_ids.as_deref(),
                parent.as_ref(),
                None,
            ),
            MoveBlock::Deref(hashes) => {
                self.process_deref(hashes);
                G1Acquire::Ready(1)
            }
            MoveBlock::Promote(uuid, seq_hash, parent_hash, local_hash, plh, token_ids) => {
                self.process_promote(
                    *uuid,
                    *seq_hash,
                    *parent_hash,
                    *local_hash,
                    *plh,
                    token_ids.clone(),
                );
                G1Acquire::Ready(1)
            }
        }
    }

    pub(crate) fn reserve_decode_blocks(
        &mut self,
        count: usize,
    ) -> G1Acquire<DecodeBlockReservation> {
        let mut attempted_generation = self.capacity_generation;
        let mut retried = false;
        loop {
            match self.allocate_use_slots(count, None) {
                G1Acquire::Ready(blocks) => {
                    return G1Acquire::Ready(DecodeBlockReservation { blocks });
                }
                G1Acquire::CapacityExhausted => return G1Acquire::CapacityExhausted,
                G1Acquire::BlockedOnOffload {
                    offload_id,
                    deadline_ms,
                } => {
                    return G1Acquire::BlockedOnOffload {
                        offload_id,
                        deadline_ms,
                    };
                }
                G1Acquire::RetryNow {
                    capacity_generation,
                    released_slots,
                } => {
                    self.validate_retry_witness(
                        attempted_generation,
                        retried,
                        capacity_generation,
                        released_slots,
                    );
                    attempted_generation = capacity_generation;
                    retried = true;
                }
            }
        }
    }

    fn bump_capacity_generation(&mut self, released_slots: usize) -> u64 {
        assert!(released_slots > 0, "capacity increase must release slots");
        let released_slots =
            u64::try_from(released_slots).expect("released G1 slot count does not fit in u64");
        self.capacity_generation = self
            .capacity_generation
            .checked_add(released_slots)
            .expect("G1 capacity generation exhausted");
        self.capacity_generation
    }

    fn allocate_use_slots(
        &mut self,
        count: usize,
        eviction_now_ms: Option<f64>,
    ) -> G1Acquire<Vec<MutableBlock<G1>>> {
        match self.reserve_g1_slots(count, eviction_now_ms) {
            G1Acquire::Ready(reservation) => {
                self.handle_evictions(reservation.evicted_plhs);
                G1Acquire::Ready(reservation.blocks)
            }
            G1Acquire::CapacityExhausted => G1Acquire::CapacityExhausted,
            G1Acquire::BlockedOnOffload {
                offload_id,
                deadline_ms,
            } => G1Acquire::BlockedOnOffload {
                offload_id,
                deadline_ms,
            },
            G1Acquire::RetryNow {
                capacity_generation,
                released_slots,
            } => G1Acquire::RetryNow {
                capacity_generation,
                released_slots,
            },
        }
    }

    fn reserve_g1_slots(
        &mut self,
        count: usize,
        eviction_now_ms: Option<f64>,
    ) -> G1Acquire<G1SlotReservation> {
        #[cfg(not(feature = "kvbm-offload"))]
        let _ = eviction_now_ms;
        if count == 0 {
            return G1Acquire::Ready(G1SlotReservation {
                blocks: Vec::new(),
                evicted_plhs: Vec::new(),
            });
        }
        let Some((blocks, evicted_plhs)) = self.block_manager.allocate_blocks_with_evictions(count)
        else {
            return G1Acquire::CapacityExhausted;
        };
        if !self.should_block_on_g1_offload(&evicted_plhs) {
            return G1Acquire::Ready(G1SlotReservation {
                blocks,
                evicted_plhs,
            });
        }

        #[cfg(feature = "kvbm-offload")]
        {
            let outcome = self
                .handle_evictions_with_source_slots_at(evicted_plhs, blocks, eviction_now_ms)
                .expect("G1 offload-enabled eviction must return a dependency outcome");
            match outcome {
                G1EvictionOutcome::BlockedOnOffload {
                    offload_id,
                    deadline_ms,
                } => G1Acquire::BlockedOnOffload {
                    offload_id,
                    deadline_ms,
                },
                G1EvictionOutcome::RetryNow { released_slots } => {
                    let capacity_generation = self.bump_capacity_generation(released_slots);
                    G1Acquire::RetryNow {
                        capacity_generation,
                        released_slots,
                    }
                }
            }
        }

        #[cfg(not(feature = "kvbm-offload"))]
        unreachable!("G1 offload blocking is disabled without kvbm-offload")
    }

    fn allocate_unpublished_blocks(
        &mut self,
        count: usize,
        eviction_now_ms: Option<f64>,
    ) -> G1Acquire<Vec<MutableBlock<G1>>> {
        self.allocate_use_slots(count, eviction_now_ms)
    }

    fn acquire_existing_full(
        &mut self,
        seq_hash: SequenceHash,
        plh: PositionalLineageHash,
    ) -> Option<ImmutableBlock<G1>> {
        if let Some(active) = self.active_full.get(&seq_hash) {
            return Some(active.handle.clone());
        }
        self.block_manager.match_blocks(&[plh]).into_iter().next()
    }

    fn commit_active_full(
        &mut self,
        candidate: MutableBlock<G1>,
        metadata: FullBlockMetadata,
    ) -> FullBlockCommit {
        let FullBlockMetadata {
            seq_hash,
            plh,
            parent_hash,
            local_hash,
            token_ids,
        } = metadata;

        if let Some(canonical) = self.acquire_existing_full(seq_hash, plh) {
            drop(candidate);
            self.insert_or_retain_active_full(seq_hash, canonical);
            return FullBlockCommit::Reused;
        }

        let candidate_block_id = candidate.block_id();
        let complete = candidate
            .stage(plh, self.block_size)
            .expect("full block stage failed");
        let canonical = self.block_manager.register_block(complete);
        let canonical_block_id = canonical.block_id();
        self.insert_or_retain_active_full(seq_hash, canonical);

        if canonical_block_id != candidate_block_id {
            return FullBlockCommit::Reused;
        }

        let previous = self.registered_blocks.insert(
            plh,
            RegisteredBlockInfo {
                seq_hash,
                block_id: canonical_block_id,
                parent_hash,
                local_hash,
                token_ids,
            },
        );
        debug_assert!(previous.is_none());
        FullBlockCommit::Stored
    }

    pub(crate) fn reserve_destination_at(
        &mut self,
        sequence: &ActiveSequence,
        eviction_now_ms: Option<f64>,
    ) -> G1Acquire<VllmDestinationReservation> {
        let layout = sequence.prepare_allocation(sequence.num_input_tokens());
        let Some(MoveBlock::Use(blocks, _, plhs, _, _)) = layout.as_ref() else {
            return G1Acquire::Ready(VllmDestinationReservation {
                cached_prefix: Vec::new(),
                unpublished_blocks: Vec::new(),
                layout,
            });
        };

        let mut cached_prefix = Vec::new();
        for (plh_idx, block) in blocks.iter().enumerate() {
            let UniqueBlock::FullBlock(seq_hash) = block else {
                break;
            };
            let plh = plhs[plh_idx];
            let Some(handle) = self.acquire_existing_full(*seq_hash, plh) else {
                break;
            };
            cached_prefix.push((*seq_hash, handle));
        }

        let count = blocks.len() - cached_prefix.len();
        let mut attempted_generation = self.capacity_generation;
        let mut retried = false;
        loop {
            match self.allocate_unpublished_blocks(count, eviction_now_ms) {
                G1Acquire::Ready(unpublished_blocks) => {
                    return G1Acquire::Ready(VllmDestinationReservation {
                        cached_prefix,
                        unpublished_blocks,
                        layout,
                    });
                }
                G1Acquire::CapacityExhausted => return G1Acquire::CapacityExhausted,
                G1Acquire::BlockedOnOffload {
                    offload_id,
                    deadline_ms,
                } => {
                    return G1Acquire::BlockedOnOffload {
                        offload_id,
                        deadline_ms,
                    };
                }
                G1Acquire::RetryNow {
                    capacity_generation,
                    released_slots,
                } => {
                    self.validate_retry_witness(
                        attempted_generation,
                        retried,
                        capacity_generation,
                        released_slots,
                    );
                    attempted_generation = capacity_generation;
                    retried = true;
                }
            }
        }
    }

    /// Publish transferred destination blocks while reconciling any cache entry
    /// that appeared after reservation. Unlike upstream vLLM, a collision may
    /// replace the reserved block ID and reduce occupancy during activation;
    /// the mocker acquires the canonical handle instead of strictly moving the
    /// originally reserved handle.
    pub(crate) fn activate_destination(&mut self, reservation: VllmDestinationReservation) {
        let VllmDestinationReservation {
            cached_prefix,
            unpublished_blocks,
            layout,
        } = reservation;
        let Some(MoveBlock::Use(blocks, local_hashes, plhs, token_ids, parent)) = layout else {
            debug_assert!(cached_prefix.is_empty());
            debug_assert!(unpublished_blocks.is_empty());
            return;
        };

        let prefix_len = cached_prefix.len();
        let full_blocks = blocks
            .iter()
            .filter(|block| matches!(block, UniqueBlock::FullBlock(_)))
            .count();
        assert_eq!(
            plhs.len(),
            full_blocks,
            "destination PLH count must match full block count"
        );
        assert!(
            local_hashes.is_empty() || local_hashes.len() == full_blocks,
            "destination local hash count must be empty or match full block count"
        );
        assert!(
            token_ids
                .as_ref()
                .is_none_or(|all_ids| all_ids.len() == full_blocks),
            "destination token metadata count must match full block count"
        );
        assert!(
            prefix_len <= blocks.len(),
            "destination cached prefix exceeds block layout"
        );
        assert_eq!(
            unpublished_blocks.len(),
            blocks.len() - prefix_len,
            "destination unpublished block count must cover the uncached layout"
        );
        for (block, (reserved_hash, _)) in blocks.iter().zip(&cached_prefix) {
            let UniqueBlock::FullBlock(layout_hash) = block else {
                panic!("destination cached prefix cannot contain a partial block");
            };
            assert_eq!(
                layout_hash, reserved_hash,
                "destination cached prefix hash must match block layout"
            );
        }

        let mut cached_prefix = cached_prefix.into_iter();
        let mut unpublished_blocks = unpublished_blocks.into_iter();
        let mut stored_seq_hashes = Vec::new();
        let mut stored_local_hashes = Vec::new();
        let mut stored_token_ids = token_ids.as_ref().map(|_| Vec::new());
        let mut first_store_parent = None;
        let mut metadata_parent_hash = match parent {
            Some(UniqueBlock::FullBlock(hash)) => Some(hash),
            Some(UniqueBlock::PartialBlock(_)) => panic!("parent block cannot be partial"),
            None => None,
        };
        let mut plh_idx = 0usize;

        for (block_idx, block) in blocks.into_iter().enumerate() {
            match block {
                UniqueBlock::FullBlock(seq_hash) => {
                    let full_idx = plh_idx;
                    let plh = plhs[plh_idx];
                    plh_idx += 1;
                    if block_idx < prefix_len {
                        let (_, handle) = cached_prefix
                            .next()
                            .expect("reserved prefix handle must exist");
                        self.insert_or_retain_active_full(seq_hash, handle);
                        metadata_parent_hash = Some(seq_hash);
                        continue;
                    }

                    let mutable = unpublished_blocks
                        .next()
                        .expect("reserved destination block must exist");
                    let local_hash = local_hashes.get(full_idx).copied();
                    let block_token_ids = token_ids
                        .as_ref()
                        .and_then(|all_ids| all_ids.get(full_idx).cloned());
                    let commit = self.commit_active_full(
                        mutable,
                        FullBlockMetadata {
                            seq_hash,
                            plh,
                            parent_hash: metadata_parent_hash,
                            local_hash,
                            token_ids: block_token_ids.clone(),
                        },
                    );
                    if commit == FullBlockCommit::Reused {
                        if !stored_seq_hashes.is_empty() {
                            let local_hashes = std::mem::take(&mut stored_local_hashes);
                            self.publish_kv_event(
                                std::mem::take(&mut stored_seq_hashes),
                                &local_hashes,
                                first_store_parent,
                                true,
                                stored_token_ids.take(),
                            );
                            first_store_parent = None;
                            stored_token_ids = token_ids.as_ref().map(|_| Vec::new());
                        }
                        metadata_parent_hash = Some(seq_hash);
                        continue;
                    }
                    if stored_seq_hashes.is_empty() {
                        first_store_parent = metadata_parent_hash;
                    }
                    stored_seq_hashes.push(seq_hash);
                    if let Some(local_hash) = local_hash {
                        stored_local_hashes.push(local_hash);
                    }
                    if let (Some(stored), Some(block_token_ids)) =
                        (stored_token_ids.as_mut(), block_token_ids)
                    {
                        stored.push(block_token_ids);
                    }
                    metadata_parent_hash = Some(seq_hash);
                }
                UniqueBlock::PartialBlock(uuid) => {
                    let mutable = unpublished_blocks
                        .next()
                        .expect("reserved destination partial block must exist");
                    let previous = self.active_partial.insert(uuid, mutable);
                    debug_assert!(previous.is_none());
                }
            }
        }

        self.publish_kv_event(
            stored_seq_hashes,
            &stored_local_hashes,
            first_store_parent,
            true,
            stored_token_ids,
        );
    }

    pub fn process_decode_signal(
        &mut self,
        event: &MoveBlock,
        reservation: &mut DecodeBlockReservation,
    ) {
        match event {
            MoveBlock::Use(blocks, local_hashes, plhs, token_ids, parent) => {
                let outcome = self.process_use(
                    blocks,
                    local_hashes,
                    plhs,
                    token_ids.as_deref(),
                    parent.as_ref(),
                    Some(reservation),
                );
                match outcome {
                    G1Acquire::Ready(allocated) => assert_eq!(
                        allocated,
                        blocks.len(),
                        "reserved decode allocation must commit every block"
                    ),
                    G1Acquire::CapacityExhausted
                    | G1Acquire::BlockedOnOffload { .. }
                    | G1Acquire::RetryNow { .. } => {
                        panic!("reserved decode allocation must be infallible")
                    }
                }
            }
            _ => {
                assert!(
                    matches!(self.process(event), G1Acquire::Ready(_)),
                    "non-Use decode signal must be infallible"
                );
            }
        }
    }

    #[cfg(feature = "kvbm-offload")]
    fn reserve_swap_in_destination_slots(&mut self, count: usize) -> SwapInSlotReservation {
        let mut attempted_generation = self.capacity_generation;
        let mut retried = false;
        loop {
            match self.allocate_use_slots(count, None) {
                G1Acquire::Ready(slots) => return SwapInSlotReservation::Reserved(slots),
                G1Acquire::CapacityExhausted => return SwapInSlotReservation::NoCapacity,
                G1Acquire::BlockedOnOffload {
                    offload_id,
                    deadline_ms,
                } => {
                    return SwapInSlotReservation::BlockedOnG1Offload(OffloadDependency {
                        offload_id,
                        deadline_ms,
                    });
                }
                G1Acquire::RetryNow {
                    capacity_generation,
                    released_slots,
                } => {
                    self.validate_retry_witness(
                        attempted_generation,
                        retried,
                        capacity_generation,
                        released_slots,
                    );
                    attempted_generation = capacity_generation;
                    retried = true;
                }
            }
        }
    }

    #[cfg(feature = "kvbm-offload")]
    fn should_block_on_g1_offload(&self, evicted_plhs: &[PositionalLineageHash]) -> bool {
        self.offload_engine.is_some()
            && evicted_plhs
                .iter()
                .any(|plh| self.registered_blocks.contains_key(plh))
    }

    #[cfg(not(feature = "kvbm-offload"))]
    fn should_block_on_g1_offload(&self, _evicted_plhs: &[PositionalLineageHash]) -> bool {
        false
    }

    fn process_use(
        &mut self,
        blocks: &[UniqueBlock],
        local_hashes: &[BlockHash],
        plhs: &[PositionalLineageHash],
        token_ids: Option<&[Vec<u32>]>,
        parent: Option<&UniqueBlock>,
        mut reservation: Option<&mut DecodeBlockReservation>,
    ) -> G1Acquire<usize> {
        let mut attempted_generation = self.capacity_generation;
        let mut retried = false;

        loop {
            let outcome = self.prepare_use(
                blocks,
                local_hashes,
                plhs,
                token_ids,
                parent,
                reservation.as_deref_mut(),
            );
            match outcome {
                G1Acquire::Ready(transaction) => {
                    self.commit_use(transaction);
                    return G1Acquire::Ready(blocks.len());
                }
                G1Acquire::CapacityExhausted => return G1Acquire::CapacityExhausted,
                G1Acquire::BlockedOnOffload {
                    offload_id,
                    deadline_ms,
                } => {
                    return G1Acquire::BlockedOnOffload {
                        offload_id,
                        deadline_ms,
                    };
                }
                G1Acquire::RetryNow {
                    capacity_generation,
                    released_slots,
                } => {
                    self.validate_retry_witness(
                        attempted_generation,
                        retried,
                        capacity_generation,
                        released_slots,
                    );
                    attempted_generation = capacity_generation;
                    retried = true;
                }
            }
        }
    }

    fn validate_retry_witness(
        &self,
        attempted_generation: u64,
        retried: bool,
        witness_generation: u64,
        released_slots: usize,
    ) {
        assert!(!retried, "one atomic G1 reservation retried more than once");
        assert!(released_slots > 0, "RetryNow released zero G1 slots");
        assert!(
            witness_generation > attempted_generation,
            "RetryNow generation {witness_generation} is not newer than attempted generation {attempted_generation}"
        );
        assert_eq!(
            witness_generation, self.capacity_generation,
            "RetryNow generation does not match current G1 capacity generation"
        );
    }

    fn prepare_use<'a>(
        &mut self,
        blocks: &[UniqueBlock],
        local_hashes: &'a [BlockHash],
        plhs: &'a [PositionalLineageHash],
        token_ids: Option<&'a [Vec<u32>]>,
        parent: Option<&'a UniqueBlock>,
        mut reservation: Option<&mut DecodeBlockReservation>,
    ) -> G1Acquire<UseTransaction<'a>> {
        let expected_full_blocks = blocks
            .iter()
            .filter(|block| matches!(block, UniqueBlock::FullBlock(_)))
            .count();
        assert_eq!(
            plhs.len(),
            expected_full_blocks,
            "Use: plhs.len() must match FullBlock count in blocks"
        );
        assert!(
            local_hashes.is_empty() || local_hashes.len() == expected_full_blocks,
            "Use: local_hashes must be empty or match FullBlock count ({} vs {})",
            local_hashes.len(),
            expected_full_blocks,
        );
        assert!(
            token_ids.is_none_or(|ids| ids.len() == expected_full_blocks),
            "Use: token_ids must be absent or match FullBlock count ({} vs {})",
            token_ids.map_or(0, |ids| ids.len()),
            expected_full_blocks,
        );

        // Classify locally active blocks once, and preserve the existing
        // per-block scattered reuse semantics while collapsing all non-local
        // lookups into one store-lock acquisition. `match_blocks` cannot be
        // used here because it stops at the first miss, whereas the existing
        // singleton loop can still reuse a later registered block.
        //
        // Start empty rather than reserving for every full block: an all-active
        // request never needs storage for non-local lookup inputs.
        let mut prepared = Vec::with_capacity(blocks.len());
        let mut nonlocal_plhs = Vec::new();
        let mut fresh_blocks = 0usize;
        let mut fresh_full_blocks = 0usize;
        let mut full_idx = 0usize;
        for block in blocks {
            match block {
                UniqueBlock::FullBlock(seq_hash) => {
                    if self.active_full.contains_key(seq_hash) {
                        prepared.push(PreparedUseBlock::ExistingActiveFull {
                            seq_hash: *seq_hash,
                        });
                    } else {
                        prepared.push(PreparedUseBlock::PendingNonLocalFull {
                            seq_hash: *seq_hash,
                            full_idx,
                        });
                        // Allocate once, but only when the request actually
                        // contains a non-local full block. Every remaining
                        // full block is the largest possible suffix here.
                        if nonlocal_plhs.is_empty() {
                            nonlocal_plhs.reserve_exact(expected_full_blocks - full_idx);
                        }
                        nonlocal_plhs.push(plhs[full_idx]);
                    }
                    full_idx += 1;
                }
                UniqueBlock::PartialBlock(uuid) => {
                    if self.active_partial.contains_key(uuid) {
                        prepared.push(PreparedUseBlock::ExistingPartial);
                    } else {
                        fresh_blocks += 1;
                        prepared.push(PreparedUseBlock::FreshPartial {
                            uuid: *uuid,
                            mutable: None,
                        });
                    }
                }
            }
        }

        if !nonlocal_plhs.is_empty() {
            let mut nonlocal_matches = self
                .block_manager
                .match_blocks_scattered(&nonlocal_plhs)
                .into_iter();
            for entry in &mut prepared {
                let PreparedUseBlock::PendingNonLocalFull { seq_hash, full_idx } = entry else {
                    continue;
                };
                let seq_hash = *seq_hash;
                let full_idx = *full_idx;
                *entry = if let Some(handle) = nonlocal_matches
                    .next()
                    .expect("scattered match result must align with non-local full blocks")
                {
                    PreparedUseBlock::ExistingMatchedFull { seq_hash, handle }
                } else {
                    fresh_blocks += 1;
                    fresh_full_blocks += 1;
                    PreparedUseBlock::FreshFull {
                        seq_hash,
                        full_idx,
                        mutable: None,
                    }
                };
            }
            assert!(
                nonlocal_matches.next().is_none(),
                "scattered match returned more entries than non-local full blocks"
            );
        }

        let mut evicted_plhs = Vec::new();

        if let Some(reservation) = reservation.as_mut() {
            if reservation.len() < fresh_blocks {
                return G1Acquire::CapacityExhausted;
            }
            for entry in &mut prepared {
                match entry {
                    PreparedUseBlock::FreshFull { mutable, .. }
                    | PreparedUseBlock::FreshPartial { mutable, .. } => {
                        *mutable = Some(
                            reservation
                                .take()
                                .expect("prechecked decode reservation must contain a slot"),
                        );
                    }
                    PreparedUseBlock::ExistingActiveFull { .. }
                    | PreparedUseBlock::ExistingMatchedFull { .. }
                    | PreparedUseBlock::ExistingPartial => {}
                    PreparedUseBlock::PendingNonLocalFull { .. } => {
                        unreachable!("non-local full block must be resolved before reservation")
                    }
                }
            }
        } else {
            let reservation = match self.reserve_g1_slots(fresh_blocks, None) {
                G1Acquire::Ready(reservation) => reservation,
                G1Acquire::CapacityExhausted => return G1Acquire::CapacityExhausted,
                G1Acquire::BlockedOnOffload {
                    offload_id,
                    deadline_ms,
                } => {
                    return G1Acquire::BlockedOnOffload {
                        offload_id,
                        deadline_ms,
                    };
                }
                G1Acquire::RetryNow {
                    capacity_generation,
                    released_slots,
                } => {
                    return G1Acquire::RetryNow {
                        capacity_generation,
                        released_slots,
                    };
                }
            };
            evicted_plhs = reservation.evicted_plhs;
            let mut slots = reservation.blocks.into_iter();
            for entry in &mut prepared {
                match entry {
                    PreparedUseBlock::FreshFull { mutable, .. }
                    | PreparedUseBlock::FreshPartial { mutable, .. } => {
                        *mutable = Some(
                            slots
                                .next()
                                .expect("atomic Use reservation returned too few slots"),
                        );
                    }
                    PreparedUseBlock::ExistingActiveFull { .. }
                    | PreparedUseBlock::ExistingMatchedFull { .. }
                    | PreparedUseBlock::ExistingPartial => {}
                    PreparedUseBlock::PendingNonLocalFull { .. } => {
                        unreachable!("non-local full block must be resolved before reservation")
                    }
                }
            }
            assert!(
                slots.next().is_none(),
                "atomic Use reservation returned too many slots"
            );
        }

        G1Acquire::Ready(UseTransaction {
            signal: UseSignalRef {
                local_hashes,
                plhs,
                token_ids,
                parent,
            },
            prepared,
            fresh_full_blocks,
            evicted_plhs,
        })
    }

    fn commit_use(&mut self, transaction: UseTransaction<'_>) {
        let UseTransaction {
            signal,
            mut prepared,
            fresh_full_blocks,
            evicted_plhs,
        } = transaction;

        // Complete every fresh full block first, then register the whole set
        // under one BlockStore lock. Registration results preserve input order,
        // so the second pass can consume them alongside the fresh prepared
        // entries while preserving router-event segmentation and metadata.
        let mut completed_blocks = Vec::with_capacity(fresh_full_blocks);
        let mut candidate_block_ids = Vec::with_capacity(fresh_full_blocks);
        for entry in &mut prepared {
            if let PreparedUseBlock::FreshFull {
                full_idx, mutable, ..
            } = entry
            {
                let mutable = mutable
                    .take()
                    .expect("committing Use must own every fresh full slot");
                candidate_block_ids.push(mutable.block_id());
                let complete = mutable
                    .stage(signal.plhs[*full_idx], self.block_size)
                    .expect("Use full block stage failed");
                completed_blocks.push(complete);
            }
        }
        let registered_blocks = self.block_manager.register_blocks(completed_blocks);
        assert_eq!(
            candidate_block_ids.len(),
            fresh_full_blocks,
            "prepared fresh full count must match staged candidate IDs"
        );
        assert_eq!(
            candidate_block_ids.len(),
            registered_blocks.len(),
            "fresh candidate IDs must align with batch registration results"
        );
        let mut fresh_registrations = candidate_block_ids.into_iter().zip(registered_blocks);

        let mut metadata_parent_hash = match signal.parent {
            None => None,
            Some(UniqueBlock::FullBlock(seq_hash)) => Some(*seq_hash),
            Some(UniqueBlock::PartialBlock(_)) => panic!("parent block cannot be partial"),
        };
        let mut first_store_parent = metadata_parent_hash;
        let mut blocks_stored = Vec::<SequenceHash>::new();
        let mut stored_local_hashes = Vec::<BlockHash>::new();
        let mut stored_token_ids = signal.token_ids.map(|_| Vec::<Vec<u32>>::new());

        for entry in prepared {
            match entry {
                PreparedUseBlock::ExistingActiveFull { seq_hash } => {
                    if !blocks_stored.is_empty() {
                        let hashes = std::mem::take(&mut blocks_stored);
                        let local_hashes = std::mem::take(&mut stored_local_hashes);
                        let token_ids = stored_token_ids.as_mut().map(std::mem::take);
                        self.publish_kv_event(
                            hashes,
                            &local_hashes,
                            first_store_parent,
                            true,
                            token_ids,
                        );
                    }
                    self.retain_active_full(seq_hash);
                    metadata_parent_hash = Some(seq_hash);
                    first_store_parent = metadata_parent_hash;
                }
                PreparedUseBlock::ExistingMatchedFull { seq_hash, handle } => {
                    if !blocks_stored.is_empty() {
                        let hashes = std::mem::take(&mut blocks_stored);
                        let local_hashes = std::mem::take(&mut stored_local_hashes);
                        let token_ids = stored_token_ids.as_mut().map(std::mem::take);
                        self.publish_kv_event(
                            hashes,
                            &local_hashes,
                            first_store_parent,
                            true,
                            token_ids,
                        );
                    }
                    self.insert_or_retain_active_full(seq_hash, handle);
                    metadata_parent_hash = Some(seq_hash);
                    first_store_parent = metadata_parent_hash;
                }
                PreparedUseBlock::PendingNonLocalFull { .. } => {
                    unreachable!("non-local full block must be resolved before commit")
                }
                PreparedUseBlock::ExistingPartial => {}
                PreparedUseBlock::FreshFull {
                    seq_hash,
                    full_idx,
                    mutable,
                } => {
                    if blocks_stored.is_empty() {
                        first_store_parent = metadata_parent_hash;
                    }
                    let plh = signal.plhs[full_idx];
                    assert!(
                        mutable.is_none(),
                        "fresh full slot must be consumed by batch staging"
                    );
                    let (candidate_block_id, immutable) = fresh_registrations
                        .next()
                        .expect("fresh full block must have a registration result");
                    if immutable.block_id() != candidate_block_id {
                        // Reject deduplication can resolve two fresh entries in
                        // this same batch to one canonical block. Finish the
                        // preceding Stored group, retain the returned handle as
                        // another logical owner, and advance the lineage cursor
                        // without replacing canonical shadow metadata or
                        // publishing a duplicate Stored event.
                        if !blocks_stored.is_empty() {
                            let hashes = std::mem::take(&mut blocks_stored);
                            let local_hashes = std::mem::take(&mut stored_local_hashes);
                            let token_ids = stored_token_ids.as_mut().map(std::mem::take);
                            self.publish_kv_event(
                                hashes,
                                &local_hashes,
                                first_store_parent,
                                true,
                                token_ids,
                            );
                        }
                        self.insert_or_retain_active_full(seq_hash, immutable);
                        metadata_parent_hash = Some(seq_hash);
                        first_store_parent = metadata_parent_hash;
                        continue;
                    }
                    self.insert_or_retain_active_full(seq_hash, immutable);

                    let local_hash = signal.local_hashes.get(full_idx).copied();
                    let registry_token_ids = signal
                        .token_ids
                        .and_then(|token_ids| token_ids.get(full_idx).cloned());
                    let previous = self.registered_blocks.insert(
                        plh,
                        RegisteredBlockInfo {
                            seq_hash,
                            block_id: candidate_block_id,
                            parent_hash: metadata_parent_hash,
                            local_hash,
                            token_ids: registry_token_ids,
                        },
                    );
                    assert!(
                        previous.is_none(),
                        "fresh Use replaced registered block {plh:?}"
                    );
                    blocks_stored.push(seq_hash);
                    if let Some(local_hash) = local_hash {
                        stored_local_hashes.push(local_hash);
                    }
                    if let (Some(stored), Some(token_ids)) =
                        (stored_token_ids.as_mut(), signal.token_ids)
                    {
                        stored.push(token_ids[full_idx].clone());
                    }
                    metadata_parent_hash = Some(seq_hash);
                }
                PreparedUseBlock::FreshPartial { uuid, mutable } => {
                    let mutable =
                        mutable.expect("committing Use must own every fresh partial slot");
                    assert!(
                        self.active_partial.insert(uuid, mutable).is_none(),
                        "fresh Use replaced active partial block {uuid}"
                    );
                }
            }
        }
        assert!(
            fresh_registrations.next().is_none(),
            "unused fresh full registration result"
        );

        if !blocks_stored.is_empty() {
            self.publish_kv_event(
                blocks_stored,
                &stored_local_hashes,
                first_store_parent,
                true,
                stored_token_ids,
            );
        }
        self.handle_evictions(evicted_plhs);
    }

    /// Translate PLHs that kvbm-logical evicted from its inactive pool
    /// (during an `allocate_blocks_with_evictions` call) into offload
    /// enqueues plus router `Removed` events. No-op when the input is empty
    /// or none of the PLHs are in our shadow registry.
    fn handle_evictions(
        &mut self,
        evicted_plhs: Vec<PositionalLineageHash>,
    ) -> Option<G1EvictionOutcome> {
        self.handle_evictions_with_source_slots(evicted_plhs, Vec::new())
    }

    /// Same as [`handle_evictions`](Self::handle_evictions), but also hands
    /// reset source slots to the offload engine so G1 capacity remains pinned
    /// until the simulated G1→G2 transfer completes.
    fn handle_evictions_with_source_slots(
        &mut self,
        evicted_plhs: Vec<PositionalLineageHash>,
        source_slots: Vec<MutableBlock<G1>>,
    ) -> Option<G1EvictionOutcome> {
        self.handle_evictions_with_source_slots_at(evicted_plhs, source_slots, None)
    }

    fn handle_evictions_with_source_slots_at(
        &mut self,
        evicted_plhs: Vec<PositionalLineageHash>,
        source_slots: Vec<MutableBlock<G1>>,
        eviction_now_ms: Option<f64>,
    ) -> Option<G1EvictionOutcome> {
        #[cfg(not(feature = "kvbm-offload"))]
        let _ = eviction_now_ms;
        if evicted_plhs.is_empty() {
            drop(source_slots);
            return None;
        }
        let mut evicted_seq_hashes = Vec::with_capacity(evicted_plhs.len());
        #[cfg(feature = "kvbm-offload")]
        let mut offload_blocks = Vec::with_capacity(evicted_plhs.len());

        for plh in evicted_plhs {
            let Some(info) = self.registered_blocks.remove(&plh) else {
                continue;
            };
            evicted_seq_hashes.push(info.seq_hash);
            #[cfg(feature = "kvbm-offload")]
            offload_blocks.push(G2OffloadBlock {
                block_id: info.block_id,
                plh,
                metadata: G2BlockEventMetadata {
                    seq_hash: info.seq_hash,
                    parent_hash: info.parent_hash,
                    local_hash: info.local_hash,
                    token_ids: info.token_ids,
                },
            });
        }

        #[cfg(feature = "kvbm-offload")]
        let (g2_events, offload_outcome) = {
            let offload_source_slots = if source_slots.is_empty() {
                Vec::new()
            } else {
                let mut source_slots_by_id: FxHashMap<_, _> = source_slots
                    .into_iter()
                    .map(|slot| (slot.block_id(), slot))
                    .collect();
                let mut matching_slots = Vec::with_capacity(offload_blocks.len());
                for block in &offload_blocks {
                    let source_slot =
                        source_slots_by_id
                            .remove(&block.block_id)
                            .unwrap_or_else(|| {
                                panic!(
                                    "G1 offload block {} has no matching source slot",
                                    block.block_id
                                )
                            });
                    matching_slots.push(source_slot);
                }
                drop(source_slots_by_id);
                matching_slots
            };
            self.enqueue_evictions_to_g2(&offload_blocks, offload_source_slots, eviction_now_ms)
        };
        #[cfg(not(feature = "kvbm-offload"))]
        drop(source_slots);

        if !evicted_seq_hashes.is_empty() {
            self.publish_kv_event(evicted_seq_hashes, &[], None, false, None);
        }

        #[cfg(feature = "kvbm-offload")]
        self.publish_g2_router_events(g2_events);

        #[cfg(feature = "kvbm-offload")]
        return offload_outcome;

        #[cfg(not(feature = "kvbm-offload"))]
        None
    }

    fn process_deref(&mut self, blocks: &[UniqueBlock]) {
        let available_before = self.block_manager.available_blocks();
        for block in blocks {
            match block {
                UniqueBlock::PartialBlock(uuid) => {
                    self.active_partial
                        .remove(uuid)
                        .expect("Deref: partial block not in active pool");
                }
                UniqueBlock::FullBlock(seq_hash) => {
                    self.release_active_full(*seq_hash);
                }
            }
        }
        let released_slots = self
            .block_manager
            .available_blocks()
            .saturating_sub(available_before);
        if released_slots > 0 {
            self.bump_capacity_generation(released_slots);
        }
    }

    fn process_promote(
        &mut self,
        uuid: Uuid,
        seq_hash: SequenceHash,
        parent_hash: Option<u64>,
        local_hash: Option<BlockHash>,
        plh: PositionalLineageHash,
        token_ids: Option<Vec<u32>>,
    ) {
        let mutable = self
            .active_partial
            .remove(&uuid)
            .expect("Promote: partial block not found");

        let commit = self.commit_active_full(
            mutable,
            FullBlockMetadata {
                seq_hash,
                plh,
                parent_hash,
                local_hash,
                token_ids: token_ids.clone(),
            },
        );

        if commit == FullBlockCommit::Stored {
            let local_hashes = local_hash.into_iter().collect::<Vec<_>>();
            self.publish_kv_event(
                vec![seq_hash],
                &local_hashes,
                parent_hash,
                true,
                token_ids.map(|t| vec![t]),
            );
        }
    }

    /// Number of **distinct** physically-resident KV blocks currently pinned
    /// by mocker (not available for eviction).
    pub fn num_active_blocks(&self) -> usize {
        // kvbm-logical partitions physical blocks into three pools:
        //   total = reset + inactive + active
        // where `available = reset + inactive`. So `total - available`
        // includes request-owned Mutable/Immutable blocks plus any reset
        // source slots quarantined behind in-flight G1→G2 offloads.
        self.block_manager.total_blocks() - self.block_manager.available_blocks()
    }

    /// Total number of logical block owners: one per held `MutableBlock` plus
    /// the explicit logical reference count of every full block. This remains
    /// a request-ownership metric even though KVBM's `inflight_immutable`
    /// metric now counts only the canonical physical handles retained here.
    pub fn num_active_block_refs(&self) -> usize {
        self.active_partial.len()
            + self
                .active_full
                .values()
                .map(|active| active.logical_refs)
                .sum::<usize>()
    }

    #[cfg(test)]
    pub(crate) fn active_block_ids(&self, sequence: &ActiveSequence) -> Vec<usize> {
        sequence
            .unique_blocks()
            .iter()
            .filter_map(|block| match block {
                UniqueBlock::FullBlock(hash) => self
                    .active_full
                    .get(hash)
                    .map(|active| active.handle.block_id()),
                UniqueBlock::PartialBlock(uuid) => {
                    self.active_partial.get(uuid).map(MutableBlock::block_id)
                }
            })
            .collect()
    }

    pub fn get_active_perc(&self) -> f64 {
        self.num_active_blocks() as f64 / self.max_capacity as f64
    }

    pub fn num_inactive_blocks(&self) -> usize {
        self.block_manager.metrics().snapshot().inactive_pool_size as usize
    }

    pub fn max_capacity(&self) -> usize {
        self.max_capacity
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn dp_rank(&self) -> u32 {
        self.dp_rank
    }

    /// Calculate the prefill cost for a sequence by scanning `unique_blocks` in
    /// order and counting the longest prefix that is cached (active or
    /// inactive). Stops at first cache miss — KV states are computed
    /// sequentially, so anything after a miss must be recomputed.
    pub fn get_prefill_cost(&self, sequence: &ActiveSequence) -> PrefillCost {
        let seq_blocks = sequence.unique_blocks();

        // Without prefix caching, each `UniqueBlock::FullBlock` carries a
        // randomised hash that can't possibly be in the cache across requests
        // — skip the PLH lookup (PLH is deterministic from tokens) to stay
        // consistent with that no-reuse contract.
        // overlap = all reusable prefix blocks (compute); active_overlap = only
        // those backed by an active block (capacity — inactive reuse is re-consumed).
        let (overlap_blocks, active_overlap_blocks) = if sequence.enable_prefix_caching() {
            let plhs = sequence.positional_lineage_hashes();
            let mut overlap = 0;
            let mut active_overlap = 0;
            for (i, block) in seq_blocks.iter().enumerate() {
                match block {
                    UniqueBlock::FullBlock(seq_hash) => {
                        if self.active_full.contains_key(seq_hash) {
                            overlap += 1;
                            active_overlap += 1;
                            continue;
                        }
                        let Some(plh) = plhs.get(i) else {
                            break;
                        };
                        if self.registered_blocks.contains_key(plh) {
                            overlap += 1;
                        } else {
                            break;
                        }
                    }
                    UniqueBlock::PartialBlock(_) => break,
                }
            }
            (overlap, active_overlap)
        } else {
            (0, 0)
        };

        let new_blocks = seq_blocks.len() - overlap_blocks;
        // Preemption resets scheduler progress but retains generated tokens in
        // the logical request. vLLM performs prefix lookup over that complete
        // known context (`request.num_tokens`), not only the original prompt.
        let cached_tokens = (overlap_blocks * self.block_size).min(sequence.len());
        let active_cached_tokens = (active_overlap_blocks * self.block_size).min(sequence.len());
        let new_tokens = sequence.len() - cached_tokens;

        PrefillCost {
            new_blocks,
            new_tokens,
            cached_tokens,
            active_cached_tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::common::protocols::{KvCacheEventSink, RawKvEvent, RawKvEventSink};

    /// Capturing event sink for router-publication assertions.
    #[derive(Default)]
    struct CapturingSink {
        events: Mutex<Vec<KvCacheEvent>>,
    }
    impl KvCacheEventSink for CapturingSink {
        fn publish(&self, event: KvCacheEvent) -> anyhow::Result<()> {
            self.events.lock().unwrap().push(event);
            Ok(())
        }
    }

    #[derive(Default)]
    struct CapturingRawSink {
        events: Mutex<Vec<RawKvEvent>>,
    }

    impl RawKvEventSink for CapturingRawSink {
        fn publish(&self, event: RawKvEvent) -> anyhow::Result<()> {
            self.events.lock().unwrap().push(event);
            Ok(())
        }
    }

    fn make_mgr(capacity: usize, block_size: usize) -> KvManager {
        KvManager::new_with_event_sink(capacity, block_size, KvEventPublishers::default(), 0)
    }

    fn expect_ready<T>(outcome: G1Acquire<T>) -> T {
        match outcome {
            G1Acquire::Ready(value) => value,
            G1Acquire::CapacityExhausted => panic!("expected Ready, got CapacityExhausted"),
            G1Acquire::BlockedOnOffload { .. } => {
                panic!("expected Ready, got BlockedOnOffload")
            }
            G1Acquire::RetryNow { .. } => panic!("expected Ready, got RetryNow"),
        }
    }

    fn make_mgr_capturing(capacity: usize, block_size: usize) -> (KvManager, Arc<CapturingSink>) {
        let sink = Arc::new(CapturingSink::default());
        let publishers = KvEventPublishers::new(Some(sink.clone() as _), None);
        (
            KvManager::new_with_event_sink(capacity, block_size, publishers, 0),
            sink,
        )
    }

    fn make_mgr_capturing_with_raw(
        capacity: usize,
        block_size: usize,
    ) -> (KvManager, Arc<CapturingSink>, Arc<CapturingRawSink>) {
        let sink = Arc::new(CapturingSink::default());
        let raw_sink = Arc::new(CapturingRawSink::default());
        let publishers =
            KvEventPublishers::new(Some(sink.clone() as _), Some(raw_sink.clone() as _));
        (
            KvManager::new_with_event_sink(capacity, block_size, publishers, 0),
            sink,
            raw_sink,
        )
    }

    fn make_mgr_capturing_with_backend(
        capacity: usize,
        block_size: usize,
        backend: MockerEvictionBackend,
    ) -> (KvManager, Arc<CapturingSink>) {
        let sink = Arc::new(CapturingSink::default());
        let publishers = KvEventPublishers::new(Some(sink.clone() as _), None);
        (
            KvManager::new_with_eviction_backend(capacity, block_size, publishers, 0, backend),
            sink,
        )
    }

    #[test]
    #[should_panic(expected = "not newer than attempted generation")]
    fn retry_witness_rejects_same_generation() {
        make_mgr(1, 4).validate_retry_witness(0, false, 0, 1);
    }

    #[test]
    #[should_panic(expected = "released zero G1 slots")]
    fn retry_witness_rejects_zero_released_slots() {
        let mut mgr = make_mgr(1, 4);
        mgr.capacity_generation = 1;
        mgr.validate_retry_witness(0, false, 1, 0);
    }

    #[test]
    #[should_panic(expected = "retried more than once")]
    fn retry_witness_rejects_second_retry() {
        let mut mgr = make_mgr(1, 4);
        mgr.capacity_generation = 1;
        mgr.validate_retry_witness(0, true, 1, 1);
    }

    fn plh(v: u64) -> PositionalLineageHash {
        PositionalLineageHash::new(v, None, 0)
    }

    fn lineage_plh(id: u64) -> PositionalLineageHash {
        match id {
            0 => PositionalLineageHash::new(0, None, 0),
            1 => PositionalLineageHash::new(1, Some(0), 1),
            2 => PositionalLineageHash::new(2, Some(1), 2),
            3 => PositionalLineageHash::new(3, Some(2), 3),
            4 => PositionalLineageHash::new(4, Some(3), 4),
            5 => PositionalLineageHash::new(5, Some(1), 2),
            6 => PositionalLineageHash::new(6, Some(5), 3),
            7 => PositionalLineageHash::new(7, Some(2), 3),
            8 => PositionalLineageHash::new(8, Some(7), 4),
            9 => PositionalLineageHash::new(9, Some(8), 5),
            10 => PositionalLineageHash::new(10, None, 0),
            11 => PositionalLineageHash::new(11, Some(10), 1),
            12 => PositionalLineageHash::new(12, Some(11), 2),
            13 => PositionalLineageHash::new(13, None, 0),
            _ => plh(id),
        }
    }

    fn use_full(mgr: &mut KvManager, seq_hash: u64, p: PositionalLineageHash) -> usize {
        expect_ready(mgr.process(&MoveBlock::Use(
            vec![UniqueBlock::FullBlock(seq_hash)],
            vec![],
            vec![p],
            None,
            None,
        )))
    }

    fn use_partial(mgr: &mut KvManager, uuid: Uuid) -> usize {
        expect_ready(mgr.process(&MoveBlock::Use(
            vec![UniqueBlock::PartialBlock(uuid)],
            vec![],
            vec![],
            None,
            None,
        )))
    }

    fn deref_full(mgr: &mut KvManager, seq_hash: u64) {
        mgr.process(&MoveBlock::Deref(vec![UniqueBlock::FullBlock(seq_hash)]));
    }

    fn deref_partial(mgr: &mut KvManager, uuid: Uuid) {
        mgr.process(&MoveBlock::Deref(vec![UniqueBlock::PartialBlock(uuid)]));
    }

    #[test]
    fn test_use_single_full_block() {
        let mut mgr = make_mgr(10, 16);
        assert_eq!(use_full(&mut mgr, 1, plh(100)), 1);
        assert_eq!(mgr.num_active_blocks(), 1);
    }

    /// `get_prefill_cost` must report an inactive cached prefix as reusable for
    /// compute (`cached_tokens`) but NOT for no-evict capacity reservation
    /// (`active_cached_tokens`), since reactivation re-consumes the block.
    #[test]
    fn prefill_cost_splits_active_and_inactive_cached_reuse() {
        let mut mgr = make_mgr(10, 4);
        // 2 full blocks (8 tokens, block_size 4), prefix caching on.
        let seq = ActiveSequence::new((0u32..8).collect(), 4, Some(4), true, false);
        let blocks = seq.unique_blocks();
        let plhs = seq.positional_lineage_hashes();
        let h0 = match &blocks[0] {
            UniqueBlock::FullBlock(h) => *h,
            other => panic!("expected a full block, got {other:?}"),
        };
        // Register block 0, then deref so it falls inactive (still registered;
        // only eviction prunes registered_blocks).
        use_full(&mut mgr, h0, plhs[0]);
        deref_full(&mut mgr, h0);

        let cost = mgr.get_prefill_cost(&seq);
        assert!(
            cost.cached_tokens >= 4,
            "inactive prefix should count for compute reuse: {cost:?}"
        );
        assert_eq!(
            cost.active_cached_tokens, 0,
            "inactive reuse must not be discounted for capacity: {cost:?}"
        );
    }

    #[test]
    fn use_rejects_short_token_ids_before_mutating_state() {
        let (mut mgr, sink) = make_mgr_capturing(10, 4);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            mgr.process(&MoveBlock::Use(
                vec![UniqueBlock::FullBlock(1), UniqueBlock::FullBlock(2)],
                vec![101, 102],
                vec![plh(100), plh(200)],
                Some(vec![vec![1, 2, 3, 4]]),
                None,
            ));
        }));

        assert!(result.is_err());
        assert_eq!(mgr.num_active_blocks(), 0);
        assert!(mgr.active_full.is_empty());
        assert!(sink.events.lock().unwrap().is_empty());
    }

    #[test]
    fn test_duplicate_use_bumps_refcount() {
        let mut mgr = make_mgr(10, 16);
        use_full(&mut mgr, 1, plh(100));
        use_full(&mut mgr, 1, plh(100));
        // Same seq_hash used twice: only one distinct physical block is
        // resident and pinned by one canonical RAII handle, while the mocker
        // tracks two logical request owners.
        assert_eq!(mgr.num_active_blocks(), 1);
        assert_eq!(mgr.num_active_block_refs(), 2);
        assert_eq!(mgr.block_manager.metrics().snapshot().inflight_immutable, 1);

        deref_full(&mut mgr, 1);
        assert_eq!(mgr.num_active_blocks(), 1);
        assert_eq!(mgr.num_active_block_refs(), 1);
        assert_eq!(mgr.block_manager.metrics().snapshot().inflight_immutable, 1);

        deref_full(&mut mgr, 1);
        assert_eq!(mgr.num_active_blocks(), 0);
        assert_eq!(mgr.num_active_block_refs(), 0);
        assert_eq!(mgr.block_manager.metrics().snapshot().inflight_immutable, 0);
    }

    #[test]
    fn all_active_multi_block_use_only_retains_logical_owners() {
        let (mut mgr, sink) = make_mgr_capturing(4, 4);
        let blocks = vec![UniqueBlock::FullBlock(10), UniqueBlock::FullBlock(20)];
        let plhs = vec![plh(100), plh(200)];

        assert_eq!(
            expect_ready(mgr.process(&MoveBlock::Use(
                blocks.clone(),
                vec![101, 201],
                plhs.clone(),
                None,
                None,
            ))),
            2
        );
        let available_before = mgr.block_manager.available_blocks();
        let first_block_id = mgr.active_full[&10].handle.block_id();
        let second_block_id = mgr.active_full[&20].handle.block_id();
        sink.events.lock().unwrap().clear();

        assert_eq!(
            expect_ready(mgr.process(&MoveBlock::Use(blocks, vec![101, 201], plhs, None, None,))),
            2
        );

        assert_eq!(mgr.block_manager.available_blocks(), available_before);
        assert_eq!(mgr.num_active_blocks(), 2);
        assert_eq!(mgr.num_active_block_refs(), 4);
        assert_eq!(mgr.active_full[&10].handle.block_id(), first_block_id);
        assert_eq!(mgr.active_full[&20].handle.block_id(), second_block_id);
        assert!(
            sink.events.lock().unwrap().is_empty(),
            "retaining active blocks must not publish another Stored event"
        );

        for _ in 0..2 {
            deref_full(&mut mgr, 10);
            deref_full(&mut mgr, 20);
        }
        assert_eq!(mgr.num_active_blocks(), 0);
        assert_eq!(mgr.num_active_block_refs(), 0);
    }

    #[test]
    fn capacity_exhaustion_returns_without_partial_commit() {
        let mut mgr = make_mgr(4, 16);
        for i in 0..4 {
            assert_eq!(use_full(&mut mgr, i, plh(i + 100)), 1);
        }
        let refs_before = mgr.num_active_block_refs();
        assert!(matches!(
            mgr.process(&MoveBlock::Use(
                vec![UniqueBlock::FullBlock(4)],
                vec![],
                vec![plh(500)],
                None,
                None,
            )),
            G1Acquire::CapacityExhausted
        ));
        assert_eq!(mgr.num_active_block_refs(), refs_before);
    }

    #[test]
    fn failed_mixed_use_does_not_retain_existing_active_blocks() {
        let mut mgr = make_mgr(1, 16);
        use_full(&mut mgr, 1, plh(100));

        assert!(matches!(
            mgr.process(&MoveBlock::Use(
                vec![UniqueBlock::FullBlock(1), UniqueBlock::FullBlock(2)],
                vec![],
                vec![plh(100), plh(200)],
                None,
                None,
            )),
            G1Acquire::CapacityExhausted
        ));
        assert_eq!(mgr.num_active_blocks(), 1);
        assert_eq!(mgr.num_active_block_refs(), 1);
        assert_eq!(mgr.active_full[&1].logical_refs, 1);
        assert!(!mgr.active_full.contains_key(&2));
    }

    #[test]
    fn test_deref_returns_to_inactive() {
        let mut mgr = make_mgr(4, 16);
        use_full(&mut mgr, 1, plh(100));
        deref_full(&mut mgr, 1);
        assert_eq!(mgr.num_active_blocks(), 0);
    }

    #[test]
    fn test_inactive_reuse_via_match_blocks() {
        let mut mgr = make_mgr(10, 16);
        let p = plh(100);
        use_full(&mut mgr, 1, p);
        deref_full(&mut mgr, 1);
        // Use with same PLH reuses the inactive block.
        assert_eq!(use_full(&mut mgr, 2, p), 1);
    }

    #[test]
    fn scattered_use_reuses_later_hit_after_a_miss() {
        let (mut mgr, sink) = make_mgr_capturing(10, 16);
        let first_plh = plh(100);
        let missing_plh = plh(200);
        let later_plh = plh(300);

        use_full(&mut mgr, 10, first_plh);
        use_full(&mut mgr, 30, later_plh);
        let later_block_id = mgr.active_full[&30].handle.block_id();
        deref_full(&mut mgr, 10);
        deref_full(&mut mgr, 30);
        sink.events.lock().unwrap().clear();

        assert_eq!(
            expect_ready(mgr.process(&MoveBlock::Use(
                vec![
                    UniqueBlock::FullBlock(10),
                    UniqueBlock::FullBlock(20),
                    UniqueBlock::FullBlock(30),
                ],
                vec![],
                vec![first_plh, missing_plh, later_plh],
                None,
                None,
            ))),
            3
        );

        assert_eq!(mgr.num_active_blocks(), 3);
        assert_eq!(mgr.num_active_block_refs(), 3);
        assert_eq!(
            mgr.active_full[&30].handle.block_id(),
            later_block_id,
            "the registered block after a miss must still be reused"
        );

        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 1, "only the missing middle block is stored");
        let KvCacheEventData::Stored(stored) = &events[0].data else {
            panic!("expected Stored event, got {:?}", events[0].data);
        };
        assert_eq!(stored.parent_hash.map(|hash| hash.0), Some(10));
        assert_eq!(stored.blocks.len(), 1);
        assert_eq!(stored.blocks[0].block_hash.0, 20);
    }

    #[test]
    fn mixed_use_keeps_fresh_registration_and_event_order() {
        let (mut mgr, sink) = make_mgr_capturing(8, 4);
        let reused_plh = plh(200);

        use_full(&mut mgr, 20, reused_plh);
        sink.events.lock().unwrap().clear();

        let seq_hashes = [10, 11, 20, 30, 31];
        let plhs = [plh(100), plh(110), reused_plh, plh(300), plh(310)];
        let local_hashes = vec![1010, 1011, 1020, 1030, 1031];
        let token_ids = vec![
            vec![10, 10, 10, 10],
            vec![11, 11, 11, 11],
            vec![20, 20, 20, 20],
            vec![30, 30, 30, 30],
            vec![31, 31, 31, 31],
        ];
        let blocks = seq_hashes.into_iter().map(UniqueBlock::FullBlock).collect();

        assert_eq!(
            expect_ready(mgr.process(&MoveBlock::Use(
                blocks,
                local_hashes.clone(),
                plhs.to_vec(),
                Some(token_ids.clone()),
                Some(UniqueBlock::FullBlock(5)),
            ))),
            seq_hashes.len()
        );

        for (idx, (seq_hash, plh)) in [(10, plhs[0]), (11, plhs[1]), (30, plhs[3]), (31, plhs[4])]
            .into_iter()
            .enumerate()
        {
            let signal_idx = [0, 1, 3, 4][idx];
            let info = mgr
                .registered_blocks
                .get(&plh)
                .expect("fresh block must retain registration metadata");
            assert_eq!(info.seq_hash, seq_hash);
            assert_eq!(info.block_id, mgr.active_full[&seq_hash].handle.block_id());
            assert_eq!(info.local_hash, Some(local_hashes[signal_idx]));
            assert_eq!(info.token_ids.as_ref(), Some(&token_ids[signal_idx]));
        }
        assert_eq!(mgr.registered_blocks[&plhs[0]].parent_hash, Some(5));
        assert_eq!(mgr.registered_blocks[&plhs[1]].parent_hash, Some(10));
        assert_eq!(mgr.registered_blocks[&plhs[3]].parent_hash, Some(20));
        assert_eq!(mgr.registered_blocks[&plhs[4]].parent_hash, Some(30));

        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 2, "the reused middle block splits stores");
        for (event, expected_hashes, expected_local_hashes, expected_parent) in [
            (&events[0], &[10, 11][..], &[1010, 1011][..], Some(5)),
            (&events[1], &[30, 31][..], &[1030, 1031][..], Some(20)),
        ] {
            let KvCacheEventData::Stored(stored) = &event.data else {
                panic!("expected Stored event, got {:?}", event.data);
            };
            assert_eq!(stored.parent_hash.map(|hash| hash.0), expected_parent);
            assert_eq!(
                stored
                    .blocks
                    .iter()
                    .map(|block| block.block_hash.0)
                    .collect::<Vec<_>>(),
                expected_hashes
            );
            assert_eq!(
                stored
                    .blocks
                    .iter()
                    .map(|block| block.tokens_hash.0)
                    .collect::<Vec<_>>(),
                expected_local_hashes
            );
        }
    }

    #[test]
    fn duplicate_fresh_registration_reuses_canonical_without_duplicate_event() {
        const CAPACITY: usize = 6;
        let (mut mgr, sink, raw_sink) = make_mgr_capturing_with_raw(CAPACITY, 4);
        let a_plh = plh(100);
        let b_plh = plh(200);
        let local_hashes = vec![101, 102, 201];
        let token_ids = vec![vec![1; 4], vec![2; 4], vec![3; 4]];
        let dedup_before = mgr.block_manager.metrics().snapshot().registration_dedup;

        assert_eq!(
            expect_ready(mgr.process(&MoveBlock::Use(
                vec![
                    UniqueBlock::FullBlock(10),
                    UniqueBlock::FullBlock(10),
                    UniqueBlock::FullBlock(20),
                ],
                local_hashes.clone(),
                vec![a_plh, a_plh, b_plh],
                Some(token_ids.clone()),
                Some(UniqueBlock::FullBlock(5)),
            ))),
            3
        );

        assert_eq!(mgr.num_active_blocks(), 2);
        assert_eq!(mgr.num_active_block_refs(), 3);
        assert_eq!(mgr.block_manager.available_blocks(), CAPACITY - 2);
        assert_eq!(
            mgr.block_manager.metrics().snapshot().registration_dedup,
            dedup_before + 1
        );
        assert_eq!(mgr.registered_blocks.len(), 2);

        let a_info = &mgr.registered_blocks[&a_plh];
        assert_eq!(a_info.seq_hash, 10);
        assert_eq!(a_info.block_id, mgr.active_full[&10].handle.block_id());
        assert_eq!(a_info.parent_hash, Some(5));
        assert_eq!(a_info.local_hash, Some(local_hashes[0]));
        assert_eq!(a_info.token_ids.as_ref(), Some(&token_ids[0]));

        let b_info = &mgr.registered_blocks[&b_plh];
        assert_eq!(b_info.seq_hash, 20);
        assert_eq!(b_info.block_id, mgr.active_full[&20].handle.block_id());
        assert_eq!(b_info.parent_hash, Some(10));
        assert_eq!(b_info.local_hash, Some(local_hashes[2]));
        assert_eq!(b_info.token_ids.as_ref(), Some(&token_ids[2]));

        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 2, "the duplicate must split Stored groups");
        for (event, expected_hash, expected_local_hash, expected_parent) in [
            (&events[0], 10, local_hashes[0], Some(5)),
            (&events[1], 20, local_hashes[2], Some(10)),
        ] {
            let KvCacheEventData::Stored(stored) = &event.data else {
                panic!("expected Stored event, got {:?}", event.data);
            };
            assert_eq!(stored.parent_hash.map(|hash| hash.0), expected_parent);
            assert_eq!(stored.blocks.len(), 1);
            assert_eq!(stored.blocks[0].block_hash.0, expected_hash);
            assert_eq!(stored.blocks[0].tokens_hash.0, expected_local_hash);
        }
        drop(events);

        let raw_events = raw_sink.events.lock().unwrap();
        assert_eq!(
            raw_events.len(),
            2,
            "the duplicate must split raw Stored groups"
        );
        assert_eq!(
            raw_events[0].block_token_ids.as_deref(),
            Some(std::slice::from_ref(&token_ids[0]))
        );
        assert_eq!(
            raw_events[1].block_token_ids.as_deref(),
            Some(std::slice::from_ref(&token_ids[2]))
        );
        drop(raw_events);

        deref_full(&mut mgr, 10);
        deref_full(&mut mgr, 10);
        deref_full(&mut mgr, 20);
        assert_eq!(mgr.num_active_blocks(), 0);
        assert_eq!(mgr.num_active_block_refs(), 0);
        assert!(mgr.active_full.is_empty());
        assert_eq!(mgr.block_manager.available_blocks(), CAPACITY);
    }

    #[test]
    fn failed_decode_reservation_preserves_inactive_cache() {
        let (mut mgr, sink) = make_mgr_capturing(2, 16);
        let first = plh(100);
        let second = plh(200);
        use_full(&mut mgr, 1, first);
        use_full(&mut mgr, 2, second);
        deref_full(&mut mgr, 1);
        deref_full(&mut mgr, 2);
        assert_eq!(mgr.num_inactive_blocks(), 2);
        sink.events.lock().unwrap().clear();

        assert!(matches!(
            mgr.reserve_decode_blocks(3),
            G1Acquire::CapacityExhausted
        ));
        assert_eq!(mgr.num_inactive_blocks(), 2);
        assert!(sink.events.lock().unwrap().is_empty());
        assert_eq!(use_full(&mut mgr, 3, first), 1);
        assert_eq!(use_full(&mut mgr, 4, second), 1);
    }

    #[test]
    fn test_eviction_frees_inactive_for_new_allocation() {
        let mut mgr = make_mgr(4, 16);
        for i in 0..4 {
            use_full(&mut mgr, i, plh(i + 100));
        }
        for i in 0..4 {
            deref_full(&mut mgr, i);
        }
        for i in 10..14 {
            assert_eq!(use_full(&mut mgr, i, plh(i + 1000)), 1);
        }
        assert_eq!(mgr.num_active_blocks(), 4);
    }

    #[test]
    fn test_promote_basic() {
        let mut mgr = make_mgr(10, 16);
        let uuid = Uuid::new_v4();
        use_partial(&mut mgr, uuid);
        mgr.process(&MoveBlock::Promote(uuid, 42, None, Some(0), plh(500), None));
        assert_eq!(mgr.num_active_blocks(), 1);
        assert!(mgr.active_partial.is_empty());
        assert!(mgr.active_full.contains_key(&42));
    }

    #[test]
    #[should_panic(expected = "Promote: partial block not found")]
    fn test_promote_nonexistent_panics() {
        let mut mgr = make_mgr(10, 16);
        mgr.process(&MoveBlock::Promote(
            Uuid::new_v4(),
            42,
            None,
            Some(0),
            plh(500),
            None,
        ));
    }

    #[test]
    fn test_deref_partial_returns_to_reset() {
        let mut mgr = make_mgr(10, 16);
        let uuid = Uuid::new_v4();
        use_partial(&mut mgr, uuid);
        assert_eq!(mgr.active_partial.len(), 1);
        deref_partial(&mut mgr, uuid);
        assert!(mgr.active_partial.is_empty());
        assert_eq!(mgr.num_active_block_refs(), 0);
    }

    #[test]
    fn test_prefill_cost_no_overlap() {
        let mgr = make_mgr(10, 16);
        let tokens: Vec<u32> = (0..35).collect();
        let seq = ActiveSequence::new(tokens, 10, Some(16), true, false);
        let cost = mgr.get_prefill_cost(&seq);
        assert_eq!(cost.new_blocks, seq.unique_blocks().len());
        assert_eq!(cost.new_tokens, 35);
    }

    #[test]
    fn test_eviction_backend_lru_and_multi_lru() {
        for backend in [MockerEvictionBackend::Lru, MockerEvictionBackend::MultiLru] {
            let mut mgr = KvManager::new_with_eviction_backend(
                4,
                16,
                KvEventPublishers::default(),
                0,
                backend,
            );
            for i in 0..4u64 {
                assert_eq!(use_full(&mut mgr, i, plh(i + 100)), 1);
            }
            for i in 0..4u64 {
                deref_full(&mut mgr, i);
            }
            for i in 10..14u64 {
                assert_eq!(
                    use_full(&mut mgr, i, plh(i + 1000)),
                    1,
                    "backend={backend:?}"
                );
            }
            assert_eq!(mgr.num_active_blocks(), 4);
        }
    }

    #[test]
    fn test_failure_on_max_capacity() {
        fn use_batch(mgr: &mut KvManager, ids: &[u64]) -> G1Acquire<usize> {
            let blocks: Vec<_> = ids.iter().map(|&id| UniqueBlock::FullBlock(id)).collect();
            let plhs: Vec<_> = ids.iter().map(|&id| plh(id)).collect();
            mgr.process(&MoveBlock::Use(blocks, vec![], plhs, None, None))
        }

        let mut mgr = make_mgr(10, 16);

        // Fill capacity in a single Use batch.
        let ids: Vec<u64> = (0..10).collect();
        assert!(matches!(use_batch(&mut mgr, &ids), G1Acquire::Ready(10)));
        assert_eq!(mgr.num_active_blocks(), 10);

        assert!(matches!(
            use_batch(&mut mgr, &[10]),
            G1Acquire::CapacityExhausted
        ));
    }

    #[test]
    fn test_block_lifecycle_stringent() {
        fn use_blocks(mgr: &mut KvManager, ids: &[u64]) -> usize {
            let blocks: Vec<_> = ids.iter().map(|&id| UniqueBlock::FullBlock(id)).collect();
            let plhs: Vec<_> = ids.iter().map(|&id| lineage_plh(id)).collect();
            expect_ready(mgr.process(&MoveBlock::Use(blocks, vec![], plhs, None, None)))
        }
        fn deref_blocks(mgr: &mut KvManager, ids: &[u64]) {
            let blocks = ids.iter().map(|&id| UniqueBlock::FullBlock(id)).collect();
            mgr.process(&MoveBlock::Deref(blocks));
        }
        fn refcount(mgr: &KvManager, id: u64) -> usize {
            mgr.active_full
                .get(&id)
                .map(|active| active.logical_refs)
                .unwrap_or(0)
        }
        fn assert_active(mgr: &KvManager, expected: &[(u64, usize)]) {
            let distinct = expected.len();
            let total_refs: usize = expected.iter().map(|&(_, r)| r).sum();
            assert_eq!(
                mgr.num_active_blocks(),
                distinct,
                "distinct active-block count mismatch; expected={expected:?}"
            );
            assert_eq!(
                mgr.num_active_block_refs(),
                total_refs,
                "active handle-refcount mismatch; expected={expected:?}"
            );
            for &(id, r) in expected {
                assert_eq!(refcount(mgr, id), r, "block {id} refcount mismatch");
            }
        }
        // Inactive membership helper. Uses `check_presence::<G1>` (non-mutating)
        // against a snapshot of PLHs to confirm each expected id is present in
        // kvbm-logical AND absent from `active_full`. Also checks total count
        // matches so we catch stray inactive entries too.
        //
        // NOTE: under kvbm-logical, once the last `ImmutableBlock` handle is
        // dropped, the block returns to the inactive pool and remains matchable
        // until eviction.
        fn assert_inactive_blocks(mgr: &KvManager, expected_ids: &[u64]) {
            assert_eq!(
                mgr.num_inactive_blocks(),
                expected_ids.len(),
                "inactive count mismatch; expected={expected_ids:?}"
            );
            let plhs: Vec<_> = expected_ids.iter().map(|&id| lineage_plh(id)).collect();
            let presence = mgr
                .block_manager
                .block_registry()
                .check_presence::<G1>(&plhs);
            for ((_, present), &id) in presence.iter().zip(expected_ids.iter()) {
                assert!(
                    *present,
                    "block {id} expected in inactive pool, not found in registry"
                );
                assert!(
                    !mgr.active_full.contains_key(&id),
                    "block {id} expected inactive but is in active pool"
                );
            }
        }
        fn drain_events(sink: &Arc<CapturingSink>) -> Vec<KvCacheEvent> {
            std::mem::take(&mut *sink.events.lock().unwrap())
        }
        fn assert_stored_event(
            event: &KvCacheEvent,
            expected_blocks: &[u64],
            expected_parent: Option<u64>,
        ) {
            let KvCacheEventData::Stored(data) = &event.data else {
                panic!("expected Stored event, got {:?}", event.data);
            };
            let actual_blocks: Vec<u64> =
                data.blocks.iter().map(|block| block.block_hash.0).collect();
            assert_eq!(actual_blocks, expected_blocks, "stored blocks mismatch");
            assert_eq!(
                data.parent_hash.map(|hash| hash.0),
                expected_parent,
                "stored parent_hash mismatch"
            );
        }
        fn assert_removed_event(event: &KvCacheEvent, expected_blocks: &[u64]) {
            let KvCacheEventData::Removed(data) = &event.data else {
                panic!("expected Removed event, got {:?}", event.data);
            };
            let actual_blocks: Vec<u64> = data.block_hashes.iter().map(|hash| hash.0).collect();
            assert_eq!(actual_blocks, expected_blocks, "removed blocks mismatch");
        }

        let (mut mgr, sink) =
            make_mgr_capturing_with_backend(10, 16, MockerEvictionBackend::Lineage);

        // Use blocks 0..=4, then 0, 1, 5, 6 — 0 and 1 bump refcount to 2.
        assert_eq!(use_blocks(&mut mgr, &[0, 1, 2, 3, 4]), 5);
        let events = drain_events(&sink);
        assert_eq!(events.len(), 1, "expected one Stored event for [0..=4]");
        assert_stored_event(&events[0], &[0, 1, 2, 3, 4], None);

        assert_eq!(use_blocks(&mut mgr, &[0, 1, 5, 6]), 4);
        let events = drain_events(&sink);
        assert_eq!(events.len(), 1, "expected one Stored event for [5, 6]");
        assert_stored_event(&events[0], &[5, 6], Some(1));
        assert_active(
            &mgr,
            &[(0, 2), (1, 2), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)],
        );

        // Leaf-to-root release order is what makes the resulting inactive set
        // deterministic under the Lineage backend.
        deref_blocks(&mut mgr, &[4, 3, 2, 1, 0]);
        let events = drain_events(&sink);
        assert!(events.is_empty(), "Deref should not emit KV events");
        assert_active(&mgr, &[(0, 1), (1, 1), (5, 1), (6, 1)]);
        assert_inactive_blocks(&mgr, &[2, 3, 4]);

        // Release the second branch leaf-to-root too. Active drains; inactive = {0..=6}.
        deref_blocks(&mut mgr, &[6, 5, 1, 0]);
        let events = drain_events(&sink);
        assert!(events.is_empty(), "Deref should not emit KV events");
        assert_active(&mgr, &[]);
        assert_inactive_blocks(&mgr, &[0, 1, 2, 3, 4, 5, 6]);

        // Re-use 0, 1, 2 (reactivates from inactive) + 7, 8, 9 (new, 3 free
        // slots). No eviction needed — inactive shrinks to {3, 4, 5, 6}.
        assert_eq!(use_blocks(&mut mgr, &[0, 1, 2, 7, 8, 9]), 6);
        let events = drain_events(&sink);
        assert_eq!(events.len(), 1, "expected one Stored event for [7, 8, 9]");
        assert_stored_event(&events[0], &[7, 8, 9], Some(2));
        assert_active(&mgr, &[(0, 1), (1, 1), (2, 1), (7, 1), (8, 1), (9, 1)]);
        assert_inactive_blocks(&mgr, &[3, 4, 5, 6]);

        // Capacity pressure now forces exact leaf-first evictions: 4, then 3,
        // then 6. The sole inactive survivor is 5.
        assert_eq!(use_blocks(&mut mgr, &[10, 11, 12]), 3);
        let events = drain_events(&sink);
        assert_eq!(
            events.len(),
            2,
            "expected Stored + Removed for [10, 11, 12]"
        );
        assert_stored_event(&events[0], &[10, 11, 12], None);
        assert_removed_event(&events[1], &[4, 3, 6]);
        assert_active(
            &mgr,
            &[
                (0, 1),
                (1, 1),
                (2, 1),
                (7, 1),
                (8, 1),
                (9, 1),
                (10, 1),
                (11, 1),
                (12, 1),
            ],
        );
        assert_inactive_blocks(&mgr, &[5]);

        assert_eq!(use_blocks(&mut mgr, &[13]), 1);
        let events = drain_events(&sink);
        assert_eq!(events.len(), 2, "expected Stored + Removed for [13]");
        assert_stored_event(&events[0], &[13], None);
        assert_removed_event(&events[1], &[5]);
        assert_active(
            &mgr,
            &[
                (0, 1),
                (1, 1),
                (2, 1),
                (7, 1),
                (8, 1),
                (9, 1),
                (10, 1),
                (11, 1),
                (12, 1),
                (13, 1),
            ],
        );
        assert_eq!(mgr.num_inactive_blocks(), 0);
    }

    #[test]
    fn test_chunked_prefill_parent_hash() {
        let block_size = 64;
        let tokens: Vec<u32> = (0..512).collect(); // 8 full blocks
        let mut seq = ActiveSequence::new(tokens, 100, Some(block_size), true, false);

        let (mut mgr, sink) = make_mgr_capturing(256, block_size);

        // Chunk 1: blocks 0..=3 (cumulative 256 tokens).
        let signal = seq.prepare_allocation(256).unwrap();
        mgr.process(&signal);
        seq.commit_allocation(256);

        // Chunk 2: blocks 4..=7 (cumulative 512 tokens).
        let signal = seq.prepare_allocation(512).unwrap();
        mgr.process(&signal);
        seq.commit_allocation(512);

        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 2, "expected two Stored events");

        let KvCacheEventData::Stored(ref store1) = events[0].data else {
            panic!("expected Stored event");
        };
        assert!(
            store1.parent_hash.is_none(),
            "first chunk should have no parent_hash"
        );

        let KvCacheEventData::Stored(ref store2) = events[1].data else {
            panic!("expected Stored event");
        };
        let UniqueBlock::FullBlock(expected_hash) = seq.unique_blocks()[3].clone() else {
            panic!("expected FullBlock at index 3");
        };
        assert_eq!(
            store2.parent_hash,
            Some(ExternalSequenceBlockHash(expected_hash)),
            "second chunk's parent_hash should be block 3's seq_hash"
        );
    }

    #[test]
    fn test_repreempt_after_partial_recompute_only_frees_reallocated_blocks() {
        let mut seq = ActiveSequence::new((0..6).collect(), 16, Some(4), true, false);
        let mut mgr = make_mgr(16, 4);

        let signal = seq.take_creation_signal().unwrap();
        assert_eq!(expect_ready(mgr.process(&signal)), 2);

        for _ in 0..3 {
            let signals = seq.generate();
            for signal in &signals {
                mgr.process(signal);
            }
            if seq.generated_tokens() < seq.max_output_tokens() {
                seq.commit_allocation(seq.len());
            }
        }
        assert_eq!(mgr.num_active_blocks(), 3);

        let first_reset = seq.reset_with_signal();
        for signal in &first_reset {
            mgr.process(signal);
        }
        assert_eq!(mgr.num_active_blocks(), 0);

        let prompt_only = seq.prepare_allocation(seq.num_input_tokens()).unwrap();
        assert_eq!(expect_ready(mgr.process(&prompt_only)), 2);
        seq.commit_allocation(seq.num_input_tokens());
        assert_eq!(mgr.num_active_blocks(), 2);

        let second_reset = seq.reset_with_signal();
        for signal in &second_reset {
            mgr.process(signal);
        }
        assert_eq!(mgr.num_active_blocks(), 0);
    }

    /// When a FullBlock is used, deref'd (becomes inactive in kvbm-logical),
    /// then used again, the router already knows about it — reactivation must
    /// NOT emit a second `Stored` event.
    #[test]
    fn test_inactive_hit_does_not_republish_stored() {
        let (mut mgr, sink) = make_mgr_capturing(4, 16);

        // First Use: fresh registration → 1 Stored.
        use_full(&mut mgr, 1, plh(100));
        // Deref → block transitions to inactive pool. No Removed (we don't
        // emit one on Deref).
        deref_full(&mut mgr, 1);
        // Second Use: match_blocks reactivates from inactive → InactiveHit.
        // No new Stored should fire.
        use_full(&mut mgr, 1, plh(100));

        let events = sink.events.lock().unwrap();
        let stored_count = events
            .iter()
            .filter(|e| matches!(e.data, KvCacheEventData::Stored(_)))
            .count();
        let removed_count = events
            .iter()
            .filter(|e| matches!(e.data, KvCacheEventData::Removed(_)))
            .count();
        assert_eq!(stored_count, 1, "reactivation must not re-emit Stored");
        assert_eq!(removed_count, 0, "Deref must not emit Removed");
    }

    #[test]
    fn destination_activation_collision_reuses_canonical_block_and_offload_metadata() {
        let (mut mgr, sink) = make_mgr_capturing(2, 4);
        assert_eq!(
            mgr.block_manager.duplication_policy(),
            &BlockDuplicationPolicy::Reject
        );
        let sequence = ActiveSequence::new(vec![1, 2, 3, 4], 1, Some(4), true, true);
        let reservation = expect_ready(mgr.reserve_destination_at(&sequence, None));
        let reserved_block_id = reservation.block_ids()[0];
        assert_eq!(mgr.num_active_blocks(), 1);

        let signal = sequence
            .prepare_allocation(sequence.num_input_tokens())
            .expect("full prompt should require allocation");
        let MoveBlock::Use(blocks, local_hashes, plhs, token_ids, _) = &signal else {
            panic!("expected full prompt allocation");
        };
        let UniqueBlock::FullBlock(seq_hash) = blocks[0] else {
            panic!("expected a full prompt block");
        };
        let plh = plhs[0];
        let local_hash = local_hashes[0];
        let token_ids = token_ids.as_ref().expect("token metadata enabled")[0].clone();

        assert_eq!(expect_ready(mgr.process(&signal)), 1);
        let canonical_block_id = mgr.active_block_ids(&sequence)[0];
        assert_ne!(reserved_block_id, canonical_block_id);
        assert_eq!(mgr.num_active_blocks(), 2);
        sink.events.lock().unwrap().clear();

        mgr.activate_destination(reservation);

        assert_eq!(mgr.num_active_blocks(), 1);
        assert_eq!(mgr.active_block_ids(&sequence), vec![canonical_block_id]);
        assert!(
            sink.events.lock().unwrap().is_empty(),
            "collision must not emit another Stored"
        );

        // handle_evictions consumes this entry when it later creates offload metadata.
        let metadata = mgr
            .registered_blocks
            .get(&plh)
            .expect("canonical registry metadata must remain available for offload");
        assert_eq!(metadata.seq_hash, seq_hash);
        assert_eq!(metadata.block_id, canonical_block_id);
        assert_eq!(metadata.parent_hash, None);
        assert_eq!(metadata.local_hash, Some(local_hash));
        assert_eq!(metadata.token_ids.as_deref(), Some(token_ids.as_slice()));
    }

    #[test]
    fn destination_transfer_footprint_uses_missing_physical_blocks() {
        let (mut mgr, _) = make_mgr_capturing(16, 4);

        let cold = ActiveSequence::new((0..10).collect(), 1, Some(4), true, true);
        let cold_reservation = expect_ready(mgr.reserve_destination_at(&cold, None));
        assert_eq!(cold_reservation.transferable_prompt_tokens(4), 12);
        drop(cold_reservation);

        let mut prefix = ActiveSequence::new((0..4).collect(), 1, Some(4), true, true);
        let prefix_allocation = prefix
            .prepare_allocation(prefix.num_input_tokens())
            .expect("prefix should require one full block");
        assert_eq!(expect_ready(mgr.process(&prefix_allocation)), 1);
        prefix.commit_allocation(prefix.num_input_tokens());

        let partial = expect_ready(mgr.reserve_destination_at(&cold, None));
        assert_eq!(partial.transferable_prompt_tokens(4), 8);
        drop(partial);

        let mut aligned = ActiveSequence::new((20..28).collect(), 1, Some(4), true, true);
        let aligned_allocation = aligned
            .prepare_allocation(aligned.num_input_tokens())
            .expect("aligned prompt should require two full blocks");
        assert_eq!(expect_ready(mgr.process(&aligned_allocation)), 2);
        aligned.commit_allocation(aligned.num_input_tokens());
        let full_hit = expect_ready(mgr.reserve_destination_at(&aligned, None));
        assert_eq!(full_hit.transferable_prompt_tokens(4), 0);
    }

    #[test]
    fn destination_activation_splits_stores_across_reused_middle_block() {
        let (mut mgr, sink) = make_mgr_capturing(8, 4);
        let sequence = ActiveSequence::new((0..12).collect(), 1, Some(4), true, true);
        let reservation = expect_ready(mgr.reserve_destination_at(&sequence, None));
        let signal = sequence
            .prepare_allocation(sequence.num_input_tokens())
            .expect("full prompt should require allocation");
        let MoveBlock::Use(blocks, _, plhs, _, _) = &signal else {
            panic!("expected full prompt allocation");
        };
        let [
            UniqueBlock::FullBlock(first),
            UniqueBlock::FullBlock(middle),
            UniqueBlock::FullBlock(last),
        ] = blocks.as_slice()
        else {
            panic!("expected three full prompt blocks");
        };

        use_full(&mut mgr, *middle, plhs[1]);
        sink.events.lock().unwrap().clear();

        mgr.activate_destination(reservation);

        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 2);
        let KvCacheEventData::Stored(first_store) = &events[0].data else {
            panic!("expected first Stored event");
        };
        assert_eq!(first_store.blocks.len(), 1);
        assert_eq!(
            first_store.blocks[0].block_hash,
            ExternalSequenceBlockHash(*first)
        );
        let KvCacheEventData::Stored(last_store) = &events[1].data else {
            panic!("expected second Stored event");
        };
        assert_eq!(last_store.blocks.len(), 1);
        assert_eq!(
            last_store.blocks[0].block_hash,
            ExternalSequenceBlockHash(*last)
        );
        assert_eq!(
            last_store.parent_hash,
            Some(ExternalSequenceBlockHash(*middle))
        );
    }

    #[test]
    fn destination_activation_validates_layout_before_committing_blocks() {
        let mut mgr = make_mgr(4, 4);
        let unpublished_blocks = expect_ready(mgr.allocate_unpublished_blocks(2, None));
        let reservation = VllmDestinationReservation {
            cached_prefix: Vec::new(),
            unpublished_blocks,
            layout: Some(MoveBlock::Use(
                vec![UniqueBlock::FullBlock(1), UniqueBlock::FullBlock(2)],
                Vec::new(),
                vec![plh(1)],
                None,
                None,
            )),
        };

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            mgr.activate_destination(reservation);
        }));

        assert!(result.is_err());
        assert_eq!(mgr.num_active_blocks(), 0);
        assert!(mgr.active_full.is_empty());
        assert!(mgr.registered_blocks.is_empty());
    }

    /// After reusing a prefix [A, B] and storing a new suffix [C], the
    /// `Stored` event for C must anchor `parent_hash` to B (the last reused
    /// full block), not to whatever parent the caller originally passed.
    #[test]
    fn test_stored_suffix_anchors_to_last_reused_block() {
        let (mut mgr, sink) = make_mgr_capturing(8, 16);

        // Prime the cache with [A=10, B=11].
        mgr.process(&MoveBlock::Use(
            vec![UniqueBlock::FullBlock(10), UniqueBlock::FullBlock(11)],
            vec![],
            vec![plh(10), plh(11)],
            None,
            None,
        ));
        // Drop both to inactive.
        deref_full(&mut mgr, 10);
        deref_full(&mut mgr, 11);

        // Clear captured events from priming.
        sink.events.lock().unwrap().clear();

        // New request reuses [A, B] and stores a new block C=12.
        mgr.process(&MoveBlock::Use(
            vec![
                UniqueBlock::FullBlock(10),
                UniqueBlock::FullBlock(11),
                UniqueBlock::FullBlock(12),
            ],
            vec![],
            vec![plh(10), plh(11), plh(12)],
            None,
            None, // no explicit parent → scheduler would pass None for a head-chunk
        ));

        let events = sink.events.lock().unwrap();
        // Only one Stored (for C); no Stored for reused A or B.
        assert_eq!(events.len(), 1, "only new suffix should fire a Stored");
        let KvCacheEventData::Stored(ref data) = events[0].data else {
            panic!("expected Stored");
        };
        assert_eq!(data.blocks.len(), 1, "Stored must only include C");
        assert_eq!(data.blocks[0].block_hash, ExternalSequenceBlockHash(12));
        assert_eq!(
            data.parent_hash,
            Some(ExternalSequenceBlockHash(11)),
            "parent_hash must anchor to last reused full block (B=11)"
        );
    }

    #[cfg(feature = "kvbm-offload")]
    #[test]
    fn test_swap_in_registration_anchors_suffix_to_reused_prefix_parent() {
        let (mut mgr, sink) = make_mgr_capturing(8, 16);
        let slots = match mgr.reserve_swap_in_destination_slots(2) {
            SwapInSlotReservation::Reserved(slots) => slots,
            SwapInSlotReservation::BlockedOnG1Offload(_) => {
                panic!("fresh manager should not need G1 offload")
            }
            SwapInSlotReservation::NoCapacity => panic!("fresh manager should have capacity"),
        };

        let entries = vec![
            SwapInRegistrationBlock {
                seq_hash: 12,
                plh: plh(12),
                local_hash: Some(120),
                token_ids: Some(vec![1; 16]),
            },
            SwapInRegistrationBlock {
                seq_hash: 13,
                plh: plh(13),
                local_hash: Some(130),
                token_ids: Some(vec![2; 16]),
            },
        ];
        let entries_len = entries.len();
        let outcome = mgr.register_swapped_in_blocks(entries, Some(11), slots);
        assert_eq!(
            outcome.consumed_entries, entries_len,
            "all reserved swap-in slots should be consumed"
        );

        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 1, "swap-in suffix should publish one Stored");
        let KvCacheEventData::Stored(ref data) = events[0].data else {
            panic!("expected Stored");
        };
        assert_eq!(
            data.parent_hash,
            Some(ExternalSequenceBlockHash(11)),
            "swapped-in suffix must anchor to the last reused prefix block"
        );
        let blocks: Vec<u64> = data.blocks.iter().map(|block| block.block_hash.0).collect();
        assert_eq!(blocks, vec![12, 13]);
        let local_hashes: Vec<u64> = data
            .blocks
            .iter()
            .map(|block| block.tokens_hash.0)
            .collect();
        assert_eq!(local_hashes, vec![120, 130]);
        assert_eq!(
            mgr.num_inactive_blocks(),
            2,
            "registered swap-in blocks should land in inactive G1"
        );
    }

    /// Two requests sharing a prefix must not inflate scheduler-visible
    /// occupancy. The distinct count reflects physically-resident blocks; the
    /// refcount metric reflects held handles.
    #[test]
    fn test_shared_prefix_distinct_vs_refcount() {
        let mut mgr = make_mgr(8, 16);

        // Request A uses [10, 11, 12].
        mgr.process(&MoveBlock::Use(
            vec![
                UniqueBlock::FullBlock(10),
                UniqueBlock::FullBlock(11),
                UniqueBlock::FullBlock(12),
            ],
            vec![],
            vec![plh(10), plh(11), plh(12)],
            None,
            None,
        ));
        assert_eq!(mgr.num_active_blocks(), 3);
        assert_eq!(mgr.num_active_block_refs(), 3);

        // Request B reuses prefix [10, 11] and adds its own block [13].
        mgr.process(&MoveBlock::Use(
            vec![
                UniqueBlock::FullBlock(10),
                UniqueBlock::FullBlock(11),
                UniqueBlock::FullBlock(13),
            ],
            vec![],
            vec![plh(10), plh(11), plh(13)],
            None,
            None,
        ));

        // Distinct resident blocks: {10, 11, 12, 13} = 4 (scheduler view).
        assert_eq!(
            mgr.num_active_blocks(),
            4,
            "shared prefix must not inflate distinct count"
        );
        // Handle count: 10 and 11 each held twice, 12 once, 13 once → 6.
        assert_eq!(
            mgr.num_active_block_refs(),
            6,
            "handle count should reflect per-request refcount"
        );
    }

    /// With `enable_prefix_caching=false`, each sequence should still be able
    /// to reactivate its OWN inactive blocks after preemption and re-admit.
    #[test]
    fn test_random_plh_stable_across_preempt_retry() {
        // 4 blocks of size 16 → 64 tokens of prompt.
        let block_size = 16;
        let tokens: Vec<u32> = (0..64).collect();
        let mut seq = ActiveSequence::new(tokens, 100, Some(block_size), false, false);

        let (mut mgr, sink) = make_mgr_capturing(8, block_size);

        // Admit: allocate prompt blocks.
        let signal = seq.take_creation_signal().unwrap();
        assert_eq!(expect_ready(mgr.process(&signal)), 4);
        assert_eq!(mgr.num_active_blocks(), 4);

        // Preempt: reset_with_signal frees all active blocks (Deref) →
        // kvbm-logical keeps them in the inactive pool (no Removed events).
        let reset_signals = seq.reset_with_signal();
        for signal in &reset_signals {
            mgr.process(signal);
        }
        assert_eq!(mgr.num_active_blocks(), 0);
        assert_eq!(mgr.num_inactive_blocks(), 4);

        // Re-admit: prompt blocks must reactivate via InactiveHit, NOT allocate
        // fresh. The cached per-sequence PLHs are what make this work.
        let signal = seq.take_creation_signal().unwrap();
        assert_eq!(expect_ready(mgr.process(&signal)), 4);
        assert_eq!(mgr.num_active_blocks(), 4);
        assert_eq!(mgr.num_inactive_blocks(), 0);

        // Router-event witness: only ONE `Stored` (from the original admit).
        let events = sink.events.lock().unwrap();
        let stored_count = events
            .iter()
            .filter(|e| matches!(e.data, KvCacheEventData::Stored(_)))
            .count();
        assert_eq!(
            stored_count, 1,
            "preempted request should self-match on re-admit (no duplicate Stored)"
        );
    }

    #[test]
    fn test_eviction_emits_exact_removed_event() {
        // Capacity = 2. Use three blocks (10, 11, 12); deref 10, 11 to push
        // them into the inactive pool; then use a third distinct block (12)
        // that isn't already in the active or inactive pool — this forces
        // allocation → inactive-pool eviction.
        let (mut mgr, sink) = make_mgr_capturing(2, 16);

        // Seed 10 and 11 in the inactive pool.
        mgr.process(&MoveBlock::Use(
            vec![UniqueBlock::FullBlock(10), UniqueBlock::FullBlock(11)],
            vec![],
            vec![plh(10), plh(11)],
            None,
            None,
        ));
        deref_full(&mut mgr, 10);
        deref_full(&mut mgr, 11);
        assert_eq!(mgr.num_active_blocks(), 0);
        assert_eq!(mgr.num_inactive_blocks(), 2);

        sink.events.lock().unwrap().clear();

        // Introduce block 12 → must evict exactly one of {10, 11}.
        use_full(&mut mgr, 12, plh(12));

        let events = sink.events.lock().unwrap();
        let removed: Vec<u64> = events
            .iter()
            .filter_map(|e| match &e.data {
                KvCacheEventData::Removed(data) => Some(
                    data.block_hashes
                        .iter()
                        .map(|ExternalSequenceBlockHash(h)| *h)
                        .collect::<Vec<_>>(),
                ),
                _ => None,
            })
            .flatten()
            .collect();
        let stored_count = events
            .iter()
            .filter(|e| matches!(e.data, KvCacheEventData::Stored(_)))
            .count();

        assert_eq!(
            removed.len(),
            1,
            "exactly one block should be reported as evicted"
        );
        assert!(
            removed[0] == 10 || removed[0] == 11,
            "evicted hash must be one we seeded ({}), got {}",
            "10 or 11",
            removed[0]
        );
        assert_eq!(stored_count, 1, "one Stored event for the fresh block 12");
    }

    #[cfg(feature = "kvbm-offload")]
    mod offload {
        use super::*;
        use crate::common::protocols::{RawKvEvent, RawKvEventSink};
        use crate::kvbm_offload::{KvbmOffloadConfig, MockOffloadEngine};
        use std::sync::{Arc, Mutex};

        #[derive(Default)]
        struct TierCapturingSink {
            events: Mutex<Vec<(StorageTier, KvCacheEvent)>>,
        }

        impl TierCapturingSink {
            fn clear(&self) {
                self.events.lock().unwrap().clear();
            }

            fn take(&self) -> Vec<(StorageTier, KvCacheEvent)> {
                std::mem::take(&mut *self.events.lock().unwrap())
            }
        }

        impl KvCacheEventSink for TierCapturingSink {
            fn publish(&self, event: KvCacheEvent) -> anyhow::Result<()> {
                self.publish_with_storage_tier(event, StorageTier::Device)
            }

            fn publish_with_storage_tier(
                &self,
                event: KvCacheEvent,
                storage_tier: StorageTier,
            ) -> anyhow::Result<()> {
                self.events.lock().unwrap().push((storage_tier, event));
                Ok(())
            }
        }

        #[derive(Default)]
        struct RawCapturingSink {
            events: Mutex<Vec<RawKvEvent>>,
        }

        impl RawCapturingSink {
            fn clear(&self) {
                self.events.lock().unwrap().clear();
            }

            fn take(&self) -> Vec<RawKvEvent> {
                std::mem::take(&mut *self.events.lock().unwrap())
            }
        }

        impl RawKvEventSink for RawCapturingSink {
            fn publish(&self, event: RawKvEvent) -> anyhow::Result<()> {
                self.events.lock().unwrap().push(event);
                Ok(())
            }
        }

        fn make_mgr_tier_capturing(
            capacity: usize,
            block_size: usize,
        ) -> (KvManager, Arc<TierCapturingSink>) {
            let sink = Arc::new(TierCapturingSink::default());
            let publishers = KvEventPublishers::new(Some(sink.clone() as _), None);
            (
                KvManager::new_with_event_sink(capacity, block_size, publishers, 0),
                sink,
            )
        }

        fn make_mgr_raw_capturing(
            capacity: usize,
            block_size: usize,
        ) -> (KvManager, Arc<RawCapturingSink>) {
            let sink = Arc::new(RawCapturingSink::default());
            let publishers = KvEventPublishers::new(None, Some(sink.clone() as _));
            (
                KvManager::new_with_event_sink(capacity, block_size, publishers, 0),
                sink,
            )
        }

        fn attach_test_offload_engine(
            mgr: &mut KvManager,
            num_g2_blocks: usize,
            block_size_tokens: usize,
        ) {
            let config = KvbmOffloadConfig {
                num_g2_blocks,
                block_size_tokens,
                block_size_bytes: Some(1_000_000),
                bandwidth_g1_to_g2_gbps: 1.0,
                ..Default::default()
            };
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .enable_all()
                .build()
                .unwrap();
            let mut engine = rt
                .block_on(MockOffloadEngine::new(config))
                .expect("engine build");
            engine.attach_runtime(rt);
            mgr.attach_new_offload_engine(engine);
        }

        fn seed_g2_block(mgr: &KvManager, p: PositionalLineageHash) {
            let engine = mgr
                .offload_engine
                .as_ref()
                .expect("offload engine attached")
                .lock()
                .expect("offload engine mutex poisoned");
            let g2 = engine.g2_manager();
            let (mut slots, _evicted) = g2
                .allocate_blocks_with_evictions(1)
                .expect("G2 test seed should fit");
            let mutable = slots.pop().expect("one G2 test slot");
            let complete = mutable
                .stage(p, g2.block_size())
                .expect("G2 test seed stage");
            drop(g2.register_block(complete));
        }

        fn use_full_with_hash(
            mgr: &mut KvManager,
            seq_hash: u64,
            p: PositionalLineageHash,
            local_hash: BlockHash,
            token_ids: Vec<u32>,
        ) -> G1Acquire<usize> {
            mgr.process(&MoveBlock::Use(
                vec![UniqueBlock::FullBlock(seq_hash)],
                vec![local_hash],
                vec![p],
                Some(vec![token_ids]),
                None,
            ))
        }

        fn has_removed(
            events: &[(StorageTier, KvCacheEvent)],
            storage_tier: StorageTier,
            seq_hash: u64,
        ) -> bool {
            events.iter().any(|(tier, event)| {
                *tier == storage_tier
                    && matches!(
                        &event.data,
                        KvCacheEventData::Removed(data)
                            if data.block_hashes.contains(&ExternalSequenceBlockHash(seq_hash))
                    )
            })
        }

        fn stored_block(
            events: &[(StorageTier, KvCacheEvent)],
            storage_tier: StorageTier,
            seq_hash: u64,
        ) -> Option<KvCacheStoredBlockData> {
            events.iter().find_map(|(tier, event)| {
                if *tier != storage_tier {
                    return None;
                }
                let KvCacheEventData::Stored(data) = &event.data else {
                    return None;
                };
                data.blocks
                    .iter()
                    .find(|block| block.block_hash == ExternalSequenceBlockHash(seq_hash))
                    .cloned()
            })
        }

        fn raw_stored_with_token_ids(
            events: &[RawKvEvent],
            storage_tier: StorageTier,
            seq_hash: u64,
            token_ids: &[u32],
        ) -> bool {
            let expected_token_ids = vec![token_ids.to_vec()];
            events.iter().any(|event| {
                event.storage_tier == storage_tier
                    && event.block_token_ids.as_ref() == Some(&expected_token_ids)
                    && matches!(
                        &event.event.data,
                        KvCacheEventData::Stored(data)
                            if data.blocks.iter().any(|block| {
                                block.block_hash == ExternalSequenceBlockHash(seq_hash)
                            })
                    )
            })
        }

        #[test]
        fn unregistered_evictions_commit_without_offload_wait() {
            const SLOTS: usize = 2;

            let (mut mgr, sink) = make_mgr_tier_capturing(SLOTS, 4);
            attach_test_offload_engine(&mut mgr, SLOTS + 1, 4);

            let source_plhs: Vec<_> = (0..SLOTS).map(|index| plh(50_000 + index as u64)).collect();
            for (index, source_plh) in source_plhs.iter().copied().enumerate() {
                assert_eq!(use_full(&mut mgr, 60_000 + index as u64, source_plh), 1);
                deref_full(&mut mgr, 60_000 + index as u64);
            }
            for source_plh in &source_plhs {
                assert!(mgr.registered_blocks.remove(source_plh).is_some());
            }
            assert_eq!(mgr.block_manager.available_blocks(), SLOTS);
            sink.clear();

            let mut sequence = ActiveSequence::new((0..8).collect(), 1, Some(4), true, true);
            let signal = sequence
                .prepare_allocation(sequence.num_input_tokens())
                .expect("two-block prompt must allocate");
            let MoveBlock::Use(blocks, _, target_plhs, _, _) = &signal else {
                panic!("creation signal must be Use");
            };
            let target_hashes: Vec<_> = blocks
                .iter()
                .map(|block| match block {
                    UniqueBlock::FullBlock(seq_hash) => *seq_hash,
                    UniqueBlock::PartialBlock(_) => panic!("exact prompt blocks must be full"),
                })
                .collect();
            assert_eq!(target_hashes.len(), SLOTS);
            assert!(
                target_plhs
                    .iter()
                    .all(|target| !mgr.registered_blocks.contains_key(target))
            );
            let generation_before = mgr.capacity_generation;
            let allocated_before = sequence.num_allocated_tokens();

            assert_eq!(expect_ready(mgr.process(&signal)), SLOTS);

            assert_eq!(mgr.capacity_generation, generation_before);
            assert!(mgr.earliest_offload_deadline().is_none());
            let (source_slot_ids, offload_block_ids) = mgr
                .offload_engine
                .as_ref()
                .expect("offload engine attached")
                .lock()
                .expect("offload engine mutex poisoned")
                .pending_g1_transfer_ownership();
            assert!(source_slot_ids.is_empty());
            assert!(offload_block_ids.is_empty());
            assert_eq!(mgr.num_active_block_refs(), SLOTS);
            assert_eq!(sequence.num_allocated_tokens(), allocated_before);
            assert!(
                target_plhs
                    .iter()
                    .all(|target| mgr.registered_blocks.contains_key(target))
            );
            assert!(target_hashes.iter().all(|target| {
                mgr.active_full
                    .get(target)
                    .is_some_and(|active| active.logical_refs == 1)
            }));

            let committed = sink.take();
            assert!(target_hashes.iter().all(|target| {
                stored_block(&committed, StorageTier::Device, *target).is_some()
            }));
            sequence.commit_allocation(sequence.num_input_tokens());
            assert_eq!(sequence.num_allocated_tokens(), sequence.num_input_tokens());
        }

        #[test]
        fn blocked_fresh_use_is_invisible_until_commit() {
            let (mut mgr, sink) = make_mgr_tier_capturing(2, 4);
            attach_test_offload_engine(&mut mgr, 4, 4);

            assert_eq!(use_full(&mut mgr, 99, plh(99)), 1);
            deref_full(&mut mgr, 99);
            sink.clear();

            let mut sequence = ActiveSequence::new((0..8).collect(), 1, Some(4), true, true);
            let signal = sequence
                .prepare_allocation(sequence.num_input_tokens())
                .expect("two-block prompt must allocate");
            let MoveBlock::Use(blocks, _, target_plhs, _, _) = &signal else {
                panic!("creation signal must be Use");
            };
            let target_hashes: Vec<_> = blocks
                .iter()
                .map(|block| match block {
                    UniqueBlock::FullBlock(seq_hash) => *seq_hash,
                    UniqueBlock::PartialBlock(_) => panic!("exact prompt blocks must be full"),
                })
                .collect();
            assert_eq!(target_hashes.len(), 2);
            let allocated_before = sequence.num_allocated_tokens();
            assert_eq!(allocated_before, 0);
            let cost_before = mgr.get_prefill_cost(&sequence);
            let cost_before = (
                cost_before.new_blocks,
                cost_before.new_tokens,
                cost_before.cached_tokens,
                cost_before.active_cached_tokens,
            );
            let refs_before = mgr.num_active_block_refs();

            let outcome = mgr.process(&signal);
            assert!(matches!(outcome, G1Acquire::BlockedOnOffload { .. }));
            assert_eq!(mgr.num_active_block_refs(), refs_before);
            assert_eq!(sequence.num_allocated_tokens(), allocated_before);
            let cost_after = mgr.get_prefill_cost(&sequence);
            assert_eq!(
                (
                    cost_after.new_blocks,
                    cost_after.new_tokens,
                    cost_after.cached_tokens,
                    cost_after.active_cached_tokens,
                ),
                cost_before
            );
            assert!(
                target_plhs
                    .iter()
                    .all(|target| !mgr.registered_blocks.contains_key(target)),
                "failed Use must not register a fresh block"
            );
            let immediate = sink.take();
            assert!(target_hashes.iter().all(|target| {
                stored_block(&immediate, StorageTier::Device, *target).is_none()
            }));

            let deadline = mgr
                .earliest_offload_deadline()
                .expect("real G1 eviction must expose its active transfer deadline");
            mgr.tick_offload_engine(deadline);
            assert_eq!(expect_ready(mgr.process(&signal)), blocks.len());
            assert_eq!(sequence.num_allocated_tokens(), allocated_before);
            sequence.commit_allocation(sequence.num_input_tokens());
            assert_eq!(sequence.num_allocated_tokens(), sequence.num_input_tokens());

            let committed = sink.take();
            assert!(target_hashes.iter().all(|target| {
                stored_block(&committed, StorageTier::Device, *target).is_some()
            }));
        }

        #[test]
        fn blocked_use_holds_only_actual_offload_sources() {
            const REQUESTED_SLOTS: usize = 40;
            const OFFLOADED_SLOTS: usize = 5;

            let (mut mgr, sink) = make_mgr_tier_capturing(REQUESTED_SLOTS, 4);
            attach_test_offload_engine(&mut mgr, OFFLOADED_SLOTS + 1, 4);

            let source_plhs: Vec<_> = (0..OFFLOADED_SLOTS)
                .map(|index| plh(10_000 + index as u64))
                .collect();
            for (index, source_plh) in source_plhs.iter().copied().enumerate() {
                assert_eq!(use_full(&mut mgr, 20_000 + index as u64, source_plh), 1);
            }
            let mut expected_source_ids: Vec<_> = source_plhs
                .iter()
                .map(|source_plh| mgr.registered_blocks[source_plh].block_id)
                .collect();
            expected_source_ids.sort_unstable();
            for index in 0..OFFLOADED_SLOTS {
                deref_full(&mut mgr, 20_000 + index as u64);
            }
            assert_eq!(mgr.block_manager.available_blocks(), REQUESTED_SLOTS);
            sink.clear();

            let mut sequence = ActiveSequence::new(
                (0..(REQUESTED_SLOTS * 4) as u32).collect(),
                1,
                Some(4),
                true,
                true,
            );
            let signal = sequence
                .prepare_allocation(sequence.num_input_tokens())
                .expect("forty-block prompt must allocate");
            let MoveBlock::Use(blocks, _, target_plhs, _, _) = &signal else {
                panic!("creation signal must be Use");
            };
            assert_eq!(blocks.len(), REQUESTED_SLOTS);
            let target_hashes: Vec<_> = blocks
                .iter()
                .map(|block| match block {
                    UniqueBlock::FullBlock(seq_hash) => *seq_hash,
                    UniqueBlock::PartialBlock(_) => panic!("exact prompt blocks must be full"),
                })
                .collect();
            let allocated_before = sequence.num_allocated_tokens();
            let refs_before = mgr.num_active_block_refs();
            let generation_before = mgr.capacity_generation;
            let cost_before = mgr.get_prefill_cost(&sequence);
            let cost_before = (
                cost_before.new_blocks,
                cost_before.new_tokens,
                cost_before.cached_tokens,
                cost_before.active_cached_tokens,
            );

            assert!(matches!(
                mgr.process(&signal),
                G1Acquire::BlockedOnOffload { .. }
            ));

            assert_eq!(mgr.num_active_block_refs(), refs_before);
            assert_eq!(sequence.num_allocated_tokens(), allocated_before);
            assert_eq!(mgr.capacity_generation, generation_before);
            assert_eq!(
                mgr.block_manager.available_blocks(),
                REQUESTED_SLOTS - OFFLOADED_SLOTS
            );
            assert_eq!(mgr.num_active_blocks(), OFFLOADED_SLOTS);
            let (source_slot_ids, offload_block_ids) = mgr
                .offload_engine
                .as_ref()
                .expect("offload engine attached")
                .lock()
                .expect("offload engine mutex poisoned")
                .pending_g1_transfer_ownership();
            assert_eq!(source_slot_ids, expected_source_ids);
            assert_eq!(offload_block_ids, expected_source_ids);
            let cost_after = mgr.get_prefill_cost(&sequence);
            assert_eq!(
                (
                    cost_after.new_blocks,
                    cost_after.new_tokens,
                    cost_after.cached_tokens,
                    cost_after.active_cached_tokens,
                ),
                cost_before
            );
            assert!(
                target_plhs
                    .iter()
                    .all(|target| !mgr.registered_blocks.contains_key(target))
            );
            let blocked_events = sink.take();
            assert!(target_hashes.iter().all(|target| {
                stored_block(&blocked_events, StorageTier::Device, *target).is_none()
            }));

            let deadline = mgr
                .earliest_offload_deadline()
                .expect("real G1 eviction must expose its active transfer deadline");
            mgr.tick_offload_engine(deadline);
            assert_eq!(mgr.block_manager.available_blocks(), REQUESTED_SLOTS);
            assert_eq!(
                mgr.capacity_generation,
                generation_before + OFFLOADED_SLOTS as u64
            );

            assert_eq!(expect_ready(mgr.process(&signal)), REQUESTED_SLOTS);
            assert_eq!(sequence.num_allocated_tokens(), allocated_before);
            sequence.commit_allocation(sequence.num_input_tokens());
            assert_eq!(sequence.num_allocated_tokens(), sequence.num_input_tokens());
            assert_eq!(mgr.num_active_block_refs(), REQUESTED_SLOTS);
            let committed_events = sink.take();
            assert!(target_hashes.iter().all(|target| {
                stored_block(&committed_events, StorageTier::Device, *target).is_some()
            }));
        }

        #[test]
        fn presence_filtered_eviction_retries_once() {
            const RELEASED_SLOTS: usize = 5;

            let (mut mgr, sink) = make_mgr_tier_capturing(RELEASED_SLOTS, 4);
            attach_test_offload_engine(&mut mgr, RELEASED_SLOTS + 1, 4);

            for index in 0..RELEASED_SLOTS {
                let source_plh = plh(30_000 + index as u64);
                assert_eq!(use_full(&mut mgr, 40_000 + index as u64, source_plh), 1);
                seed_g2_block(&mgr, source_plh);
            }
            for index in 0..RELEASED_SLOTS {
                deref_full(&mut mgr, 40_000 + index as u64);
            }
            sink.clear();
            let generation_before = mgr.capacity_generation;

            let sequence = ActiveSequence::new(
                (100..100 + (RELEASED_SLOTS * 4) as u32).collect(),
                1,
                Some(4),
                true,
                true,
            );
            let signal = sequence
                .prepare_allocation(sequence.num_input_tokens())
                .expect("five-block prompt must allocate");
            assert_eq!(expect_ready(mgr.process(&signal)), RELEASED_SLOTS);
            assert_eq!(
                mgr.capacity_generation,
                generation_before + RELEASED_SLOTS as u64
            );
            assert!(mgr.earliest_offload_deadline().is_none());
            assert_eq!(mgr.num_active_block_refs(), RELEASED_SLOTS);

            let events = sink.take();
            let MoveBlock::Use(blocks, _, _, _, _) = signal else {
                panic!("creation signal must be Use");
            };
            assert!(blocks.iter().all(|block| {
                let UniqueBlock::FullBlock(seq_hash) = block else {
                    return false;
                };
                stored_block(&events, StorageTier::Device, *seq_hash).is_some()
            }));
        }

        #[test]
        fn queued_use_tracks_exact_transfer_dependency() {
            let mut mgr = make_mgr(2, 4);
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .enable_all()
                .build()
                .unwrap();
            let mut engine = runtime
                .block_on(MockOffloadEngine::new(KvbmOffloadConfig {
                    block_size_tokens: 4,
                    block_size_bytes: Some(1_000_000),
                    bandwidth_g1_to_g2_gbps: 1.0,
                    offload_batch_size: 1,
                    ..Default::default()
                }))
                .expect("engine build");
            engine.attach_runtime(runtime);
            mgr.attach_new_offload_engine(engine);

            assert_eq!(use_full(&mut mgr, 1, plh(1)), 1);
            assert_eq!(use_full(&mut mgr, 2, plh(2)), 1);
            deref_full(&mut mgr, 1);
            deref_full(&mut mgr, 2);

            let first_dependency =
                match use_full_with_hash(&mut mgr, 3, plh(3), 303, vec![9, 10, 11, 12]) {
                    G1Acquire::BlockedOnOffload {
                        offload_id,
                        deadline_ms,
                    } => OffloadDependency {
                        offload_id,
                        deadline_ms,
                    },
                    _ => panic!("first eviction must start an offload dependency"),
                };
            let first_id = first_dependency.offload_id;
            assert!(first_dependency.deadline_ms.is_some());

            let queued_dependency =
                match use_full_with_hash(&mut mgr, 4, plh(4), 404, vec![13, 14, 15, 16]) {
                    G1Acquire::BlockedOnOffload {
                        offload_id,
                        deadline_ms,
                    } => OffloadDependency {
                        offload_id,
                        deadline_ms,
                    },
                    _ => panic!("second eviction must queue behind the active offload"),
                };
            assert_ne!(queued_dependency.offload_id, first_id);
            assert_eq!(queued_dependency.deadline_ms, first_dependency.deadline_ms);
            assert_eq!(
                mgr.refresh_offload_dependency(queued_dependency),
                Some(queued_dependency),
                "the queued request must stay protected while its exact lease is active"
            );
            mgr.tick_offload_engine(
                first_dependency
                    .deadline_ms
                    .expect("first offload dependency deadline"),
            );
            assert_eq!(
                mgr.refresh_offload_dependency(first_dependency),
                None,
                "a completed lease must not retarget to an unrelated live offload"
            );
            assert_eq!(
                mgr.refresh_offload_dependency(queued_dependency)
                    .map(|dependency| dependency.offload_id),
                Some(queued_dependency.offload_id),
                "the queued lease must remain independently live"
            );
            assert_eq!(mgr.num_active_block_refs(), 0);
        }

        #[test]
        fn fresh_manager_has_no_offload_engine() {
            let mgr = make_mgr(8, 4);
            assert!(!mgr.has_offload_engine());
        }

        #[tokio::test]
        async fn attach_new_offload_engine_wires_in_after_construction() {
            let mut mgr = make_mgr(16, 4);
            assert!(!mgr.has_offload_engine());

            let engine = MockOffloadEngine::new(KvbmOffloadConfig::default())
                .await
                .expect("engine build");
            mgr.attach_new_offload_engine(engine);
            assert!(mgr.has_offload_engine());
        }

        #[test]
        fn g2_completion_publishes_host_pinned_stored_event() {
            let (mut mgr, sink) = make_mgr_tier_capturing(1, 4);
            attach_test_offload_engine(&mut mgr, 1, 4);

            assert_eq!(
                expect_ready(use_full_with_hash(
                    &mut mgr,
                    1,
                    plh(1),
                    101,
                    vec![1, 2, 3, 4],
                )),
                1
            );
            deref_full(&mut mgr, 1);
            sink.clear();

            // Capacity pressure evicts block 1 from G1 and starts G1→G2.
            assert!(matches!(
                use_full_with_hash(&mut mgr, 2, plh(2), 202, vec![5, 6, 7, 8]),
                G1Acquire::BlockedOnOffload { .. }
            ));
            let immediate = sink.take();
            assert!(
                has_removed(&immediate, StorageTier::Device, 1),
                "G1 eviction should publish a Device-tier Removed event"
            );
            assert!(
                stored_block(&immediate, StorageTier::HostPinned, 1).is_none(),
                "G2 Stored must not publish before the transfer completes"
            );

            let deadline = mgr
                .earliest_offload_deadline()
                .expect("G1→G2 offload should expose a completion deadline");
            mgr.tick_offload_engine(deadline);

            let completed = sink.take();
            let stored = stored_block(&completed, StorageTier::HostPinned, 1)
                .expect("G2 completion should publish HostPinned Stored");
            assert_eq!(stored.tokens_hash, LocalBlockHash(101));
        }

        #[test]
        fn g2_eviction_publishes_host_pinned_removed_event() {
            let (mut mgr, sink) = make_mgr_tier_capturing(1, 4);
            attach_test_offload_engine(&mut mgr, 1, 4);

            assert_eq!(
                expect_ready(use_full_with_hash(
                    &mut mgr,
                    1,
                    plh(1),
                    101,
                    vec![1, 2, 3, 4],
                )),
                1
            );
            deref_full(&mut mgr, 1);
            assert!(matches!(
                use_full_with_hash(&mut mgr, 2, plh(2), 202, vec![5, 6, 7, 8]),
                G1Acquire::BlockedOnOffload { .. }
            ));
            let deadline = mgr
                .earliest_offload_deadline()
                .expect("first G1→G2 offload should expose a deadline");
            mgr.tick_offload_engine(deadline);

            // Now block 1 is resident in G2. Admit block 2 into G1, then evict
            // it to the one-block G2 tier; this must evict block 1 from G2.
            assert_eq!(
                expect_ready(use_full_with_hash(
                    &mut mgr,
                    2,
                    plh(2),
                    202,
                    vec![5, 6, 7, 8],
                )),
                1
            );
            deref_full(&mut mgr, 2);
            sink.clear();

            assert!(matches!(
                use_full_with_hash(&mut mgr, 3, plh(3), 303, vec![9, 10, 11, 12]),
                G1Acquire::BlockedOnOffload { .. }
            ));
            let deadline = mgr
                .earliest_offload_deadline()
                .expect("second G1→G2 offload should expose a deadline");
            mgr.tick_offload_engine(deadline);

            let events = sink.take();
            assert!(
                has_removed(&events, StorageTier::HostPinned, 1),
                "G2 capacity eviction should publish HostPinned Removed"
            );
            assert!(
                stored_block(&events, StorageTier::HostPinned, 2).is_some(),
                "second G2 completion should publish HostPinned Stored for block 2"
            );
        }

        #[test]
        fn reoffloaded_swapped_in_block_keeps_token_ids_for_g2_raw_event() {
            let (mut mgr, sink) = make_mgr_raw_capturing(1, 4);
            attach_test_offload_engine(&mut mgr, 1, 4);

            let slots = match mgr.reserve_swap_in_destination_slots(1) {
                SwapInSlotReservation::Reserved(slots) => slots,
                SwapInSlotReservation::BlockedOnG1Offload(_) => {
                    panic!("fresh manager should not need G1 offload")
                }
                SwapInSlotReservation::NoCapacity => panic!("fresh manager should have capacity"),
            };
            let token_ids = vec![1, 2, 3, 4];
            let entries = vec![SwapInRegistrationBlock {
                seq_hash: 1,
                plh: plh(1),
                local_hash: Some(101),
                token_ids: Some(token_ids.clone()),
            }];
            let outcome = mgr.register_swapped_in_blocks(entries, None, slots);
            assert_eq!(outcome.consumed_entries, 1);
            sink.clear();

            assert!(matches!(
                use_full_with_hash(&mut mgr, 2, plh(2), 202, vec![5, 6, 7, 8]),
                G1Acquire::BlockedOnOffload { .. }
            ));
            let deadline = mgr
                .earliest_offload_deadline()
                .expect("G1→G2 offload should expose a deadline");
            mgr.tick_offload_engine(deadline);

            let events = sink.take();
            assert!(
                raw_stored_with_token_ids(&events, StorageTier::HostPinned, 1, &token_ids),
                "re-offloaded swapped-in block should preserve token ids for HostPinned raw Stored"
            );
        }

        #[test]
        fn g1_eviction_offload_holds_source_slot_until_complete() {
            let mut mgr = make_mgr(1, 4);
            let config = KvbmOffloadConfig {
                block_size_tokens: 4,
                block_size_bytes: Some(1_000_000),
                bandwidth_g1_to_g2_gbps: 1.0,
                ..Default::default()
            };
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .enable_all()
                .build()
                .unwrap();
            let mut engine = rt
                .block_on(MockOffloadEngine::new(config))
                .expect("engine build");
            engine.attach_runtime(rt);
            mgr.attach_new_offload_engine(engine);

            assert_eq!(use_full(&mut mgr, 1, plh(1)), 1);
            deref_full(&mut mgr, 1);
            assert_eq!(mgr.num_active_blocks(), 0);
            assert_eq!(mgr.num_inactive_blocks(), 1);

            // Capacity pressure evicts block 1 and starts G1→G2. The returned
            // reset slot is held as the source-capacity token, so block 2
            // cannot be allocated until the simulated transfer completes.
            assert!(matches!(
                mgr.process(&MoveBlock::Use(
                    vec![UniqueBlock::FullBlock(2)],
                    vec![],
                    vec![plh(2)],
                    None,
                    None,
                )),
                G1Acquire::BlockedOnOffload { .. }
            ));
            assert_eq!(
                mgr.num_active_blocks(),
                1,
                "quarantined source slot must count against G1 capacity"
            );
            let deadline = mgr
                .earliest_offload_deadline()
                .expect("G1→G2 offload should expose a stall-advance deadline");

            mgr.tick_offload_engine(deadline);
            assert_eq!(
                mgr.num_active_blocks(),
                0,
                "source slot should release after transfer completion"
            );
            assert_eq!(use_full(&mut mgr, 2, plh(2)), 1);
            assert_eq!(mgr.num_active_blocks(), 1);
        }

        #[test]
        fn try_batch_swap_in_returns_no_hits_without_engine() {
            let mut mgr = make_mgr(8, 4);
            let plhs = [plh(1), plh(2), plh(3)];
            let outcome = mgr.try_batch_swap_in(&plhs, Vec::new(), None);
            assert!(matches!(outcome, BatchSwapInOutcome::NoHits));
        }
    }
}
