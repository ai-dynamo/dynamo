// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Single-mutex block store: unified bookkeeping for reset and inactive pools.
//!
//! `BlockStore<T>` owns the entire block bookkeeping for a single metadata
//! tier. A contiguous `Vec<BlockSlot>` indexed by `BlockId` tracks the
//! current state of every slot; a free list (`VecDeque<BlockId>`) implements
//! the reset pool; a pluggable [`InactiveIndex`] backend implements the
//! inactive pool's eviction order. All transitions happen under one
//! `parking_lot::Mutex`, eliminating the cross-mutex race that the
//! dual-weak-ref `WeakBlockEntry` resurrection scheme was working around.
//!
//! # Lock ordering
//!
//! `BlockRegistrationHandle.attachments` (Mutex inside the registry) →
//! `BlockStore.inner` (Mutex). Never the reverse.
//!
//! # Transitional layer
//!
//! While the public guard types still wrap `Option<Block<T, _>>` + return-fn
//! closures, the store carries a side-table `HashMap<BlockId, Block<T,
//! Registered>>` for the inactive payloads. The side-table goes away in
//! commit 4 once guards talk to the store directly.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::BlockId;
use crate::blocks::{
    Block, BlockMetadata, MutableBlock, PrimaryBlock, RegisteredBlock, RegisteredReturnFn,
    ResetReturnFn, SequenceHash,
    state::{Registered, Reset},
};
use crate::metrics::BlockPoolMetrics;

/// Index trait for inactive-pool eviction backends.
///
/// `T`-free replacement for the legacy `InactivePoolBackend<T>` — backends
/// only need `(SequenceHash, BlockId)` pairs; the slot's typed payload lives
/// in the `BlockStore`.
#[allow(dead_code)]
pub(crate) trait InactiveIndex: Send + Sync {
    /// Find blocks for the given hashes in order, stopping on first miss.
    /// Removes matched entries from the index.
    fn find_matches(
        &mut self,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> Vec<(SequenceHash, BlockId)>;

    /// Like `find_matches` but does not stop on miss.
    fn scan_matches(
        &mut self,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> Vec<(SequenceHash, BlockId)>;

    /// Pull `count` blocks for eviction in policy order.
    fn allocate(&mut self, count: usize) -> Vec<(SequenceHash, BlockId)>;

    /// Make `block_id` evictable under `seq_hash`.
    fn insert(&mut self, seq_hash: SequenceHash, block_id: BlockId);

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn has(&self, seq_hash: SequenceHash) -> bool;

    /// Remove a specific `block_id`/`seq_hash` pair if present.
    #[allow(dead_code)]
    fn take(&mut self, seq_hash: SequenceHash, block_id: BlockId) -> bool;

    /// Drain the entire index.
    fn allocate_all(&mut self) -> Vec<(SequenceHash, BlockId)> {
        let n = self.len();
        self.allocate(n)
    }
}

/// State of an individual slot. Tracked under the unified store mutex.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum SlotState {
    /// Slot index is in `free`; nothing else holds it.
    Reset,
    /// A guard owns this slot.
    CheckedOut,
    /// Slot index is in the inactive index, available for eviction or resurrection.
    Inactive,
}

#[derive(Debug)]
pub(crate) struct BlockSlot {
    #[allow(dead_code)]
    pub(crate) block_size: usize,
    pub(crate) state: SlotState,
}

impl BlockSlot {
    fn new_reset(block_size: usize) -> Self {
        Self {
            block_size,
            state: SlotState::Reset,
        }
    }
}

/// Inner state of a `BlockStore` — protected by a single mutex.
pub(crate) struct BlockStoreInner<T: BlockMetadata> {
    /// `slots[block_id]` — created at construction, never grows.
    slots: Vec<BlockSlot>,
    /// Free list (reset pool). FIFO via `pop_front`/`push_back`.
    free: VecDeque<BlockId>,
    /// Inactive eviction index (T-free).
    inactive: Box<dyn InactiveIndex>,
    /// Side-table of `Block<T, Registered>` payloads, keyed by BlockId.
    /// Always synced with the inactive index. Goes away in commit 4.
    inactive_blocks: HashMap<BlockId, Block<T, Registered>>,
}

/// Single-mutex bookkeeping store for the reset and inactive pools.
pub(crate) struct BlockStore<T: BlockMetadata> {
    inner: Arc<Mutex<BlockStoreInner<T>>>,
    block_size: usize,
    total_blocks: usize,
    metrics: Option<Arc<BlockPoolMetrics>>,
    reset_return_fn: ResetReturnFn<T>,
    inactive_return_fn: RegisteredReturnFn<T>,
}

#[allow(dead_code)]
impl<T: BlockMetadata + Sync> BlockStore<T> {
    pub(crate) fn new(
        blocks: Vec<Block<T, Reset>>,
        block_size: usize,
        inactive: Box<dyn InactiveIndex>,
        metrics: Option<Arc<BlockPoolMetrics>>,
    ) -> Self {
        for (i, block) in blocks.iter().enumerate() {
            assert_eq!(
                block.block_id(),
                i,
                "Block ids must be monotonically increasing starting at 0"
            );
        }

        let total_blocks = blocks.len();
        let mut slots = Vec::with_capacity(total_blocks);
        let mut free = VecDeque::with_capacity(total_blocks);
        for block in &blocks {
            slots.push(BlockSlot::new_reset(block.block_size()));
            free.push_back(block.block_id());
        }
        // Discard the input blocks — the slot tracks all the state we need;
        // the synthesized `Block<T, Reset>` is bookkeeping-only. The
        // reset_pool_size gauge is set by the caller (matches the original
        // ResetPool construction contract).
        let _ = blocks;

        let inner = Arc::new(Mutex::new(BlockStoreInner {
            slots,
            free,
            inactive,
            inactive_blocks: HashMap::new(),
        }));

        let inner_for_reset = inner.clone();
        let metrics_for_reset = metrics.clone();
        let reset_return_fn: ResetReturnFn<T> = Arc::new(move |block: Block<T, Reset>| {
            // The synthesized Block<T, Reset> carries no payload; consume it.
            let block_id = block.block_id();
            let _ = block;
            let mut inner = inner_for_reset.lock();
            // In the transitional layer, tests sometimes synthesize blocks
            // outside the store; only track slot state for in-bounds ids.
            if block_id < inner.slots.len() {
                inner.slots[block_id].state = SlotState::Reset;
            }
            inner.free.push_back(block_id);
            if let Some(ref m) = metrics_for_reset {
                m.inc_reset_pool_size();
            }
        });

        let inner_for_inactive = inner.clone();
        let metrics_for_inactive = metrics.clone();
        let inactive_return_fn: RegisteredReturnFn<T> =
            Arc::new(move |arc_block: Arc<Block<T, Registered>>| {
                let seq_hash = arc_block.sequence_hash();
                match Arc::try_unwrap(arc_block) {
                    Ok(block) => {
                        let block_id = block.block_id();
                        let mut inner = inner_for_inactive.lock();
                        inner.inactive.insert(seq_hash, block_id);
                        inner.inactive_blocks.insert(block_id, block);
                        if block_id < inner.slots.len() {
                            inner.slots[block_id].state = SlotState::Inactive;
                        }
                        if let Some(ref m) = metrics_for_inactive {
                            m.inc_inactive_pool_size();
                        }
                        tracing::trace!(?seq_hash, block_id, "Block stored in inactive pool");
                    }
                    Err(block) => {
                        let block_id = block.block_id();
                        let weak = Arc::downgrade(&block);
                        drop(block);
                        if weak.strong_count() == 0 {
                            tracing::warn!(?seq_hash, block_id, "Possible KV Block leak detected");
                        }
                    }
                }
            });

        Self {
            inner,
            block_size,
            total_blocks,
            metrics,
            reset_return_fn,
            inactive_return_fn,
        }
    }

    pub(crate) fn block_size(&self) -> usize {
        self.block_size
    }

    pub(crate) fn total_blocks(&self) -> usize {
        self.total_blocks
    }

    pub(crate) fn metrics(&self) -> Option<&Arc<BlockPoolMetrics>> {
        self.metrics.as_ref()
    }

    pub(crate) fn reset_return_fn(&self) -> ResetReturnFn<T> {
        self.reset_return_fn.clone()
    }

    pub(crate) fn inactive_return_fn(&self) -> RegisteredReturnFn<T> {
        self.inactive_return_fn.clone()
    }

    pub(crate) fn reset_len(&self) -> usize {
        self.inner.lock().free.len()
    }

    pub(crate) fn inactive_len(&self) -> usize {
        self.inner.lock().inactive.len()
    }

    pub(crate) fn has_inactive(&self, seq_hash: SequenceHash) -> bool {
        self.inner.lock().inactive.has(seq_hash)
    }

    /// Allocate up to `count` mutable blocks from the reset pool.
    pub(crate) fn allocate_reset_blocks(&self, count: usize) -> Vec<MutableBlock<T>> {
        let mut inner = self.inner.lock();
        let take = std::cmp::min(count, inner.free.len());
        let mut out = Vec::with_capacity(take);
        for _ in 0..take {
            let id = inner.free.pop_front().unwrap();
            inner.slots[id].state = SlotState::CheckedOut;
            if let Some(ref m) = self.metrics {
                m.dec_reset_pool_size();
            }
            // Synthesize a fresh Block<T, Reset> for the guard to wrap. The
            // Block struct is bookkeeping-only — slot state is the source of
            // truth; this synthesized value is discarded on guard drop.
            let synth = Block::<T, Reset>::new(id, self.block_size);
            out.push(MutableBlock::new(
                synth,
                self.reset_return_fn.clone(),
                self.metrics.clone(),
            ));
        }
        out
    }

    /// Evict up to `count` blocks from the inactive pool, returning them as
    /// fresh MutableBlocks. Returns `None` if the inactive pool has fewer
    /// than `count` blocks. Reports the evicted sequence hashes for upstream
    /// cache-invalidation tracking.
    pub(crate) fn evict_to_reset(
        &self,
        count: usize,
    ) -> Option<(Vec<MutableBlock<T>>, Vec<SequenceHash>)> {
        if count == 0 {
            return Some((Vec::new(), Vec::new()));
        }
        let mut inner = self.inner.lock();
        if inner.inactive.len() < count {
            return None;
        }
        let evicted_pairs = inner.inactive.allocate(count);
        if evicted_pairs.len() != count {
            for (h, id) in evicted_pairs {
                inner.inactive.insert(h, id);
            }
            return None;
        }
        if let Some(ref m) = self.metrics {
            for _ in 0..count {
                m.dec_inactive_pool_size();
            }
        }

        let mut mutable_blocks = Vec::with_capacity(count);
        let mut evicted = Vec::with_capacity(count);
        for (seq_hash, block_id) in evicted_pairs {
            let registered = inner
                .inactive_blocks
                .remove(&block_id)
                .expect("inactive index/side-table out of sync");
            if block_id < inner.slots.len() {
                inner.slots[block_id].state = SlotState::CheckedOut;
            }
            evicted.push(seq_hash);
            // `reset()` calls mark_absent::<T> on the registration handle.
            let reset_block = registered.reset();
            mutable_blocks.push(MutableBlock::new(
                reset_block,
                self.reset_return_fn.clone(),
                self.metrics.clone(),
            ));
        }
        Some((mutable_blocks, evicted))
    }

    /// Drain the entire inactive pool into MutableBlocks. The blocks return
    /// to the reset pool when dropped.
    pub(crate) fn drain_inactive_to_reset(&self) -> Vec<MutableBlock<T>> {
        let mut inner = self.inner.lock();
        let drained = inner.inactive.allocate_all();
        let count = drained.len();
        if let Some(ref m) = self.metrics {
            for _ in 0..count {
                m.dec_inactive_pool_size();
            }
        }
        drained
            .into_iter()
            .map(|(_seq_hash, block_id)| {
                let registered = inner
                    .inactive_blocks
                    .remove(&block_id)
                    .expect("inactive index/side-table out of sync");
                if block_id < inner.slots.len() {
                    inner.slots[block_id].state = SlotState::CheckedOut;
                }
                let reset_block = registered.reset();
                MutableBlock::new(
                    reset_block,
                    self.reset_return_fn.clone(),
                    self.metrics.clone(),
                )
            })
            .collect()
    }

    /// Find blocks by hash, stopping on first miss. Each returned block is
    /// removed from the inactive pool and wrapped in a `PrimaryBlock` guard.
    pub(crate) fn find_inactive_blocks(
        &self,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> Vec<Arc<dyn RegisteredBlock<T>>> {
        let mut inner = self.inner.lock();
        let matched = inner.inactive.find_matches(hashes, touch);
        let count = matched.len();
        if let Some(ref m) = self.metrics {
            for _ in 0..count {
                m.dec_inactive_pool_size();
            }
        }
        matched
            .into_iter()
            .map(|(_seq_hash, block_id)| {
                let registered = inner
                    .inactive_blocks
                    .remove(&block_id)
                    .expect("inactive index/side-table out of sync");
                if block_id < inner.slots.len() {
                    inner.slots[block_id].state = SlotState::CheckedOut;
                }
                PrimaryBlock::new_attached(Arc::new(registered), self.inactive_return_fn.clone())
                    as Arc<dyn RegisteredBlock<T>>
            })
            .collect()
    }

    /// Scan-style find — does not stop on miss. Found blocks are removed
    /// from the inactive pool.
    pub(crate) fn scan_inactive_blocks(
        &self,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> Vec<(SequenceHash, Arc<dyn RegisteredBlock<T>>)> {
        let mut inner = self.inner.lock();
        let found = inner.inactive.scan_matches(hashes, touch);
        let count = found.len();
        if let Some(ref m) = self.metrics {
            for _ in 0..count {
                m.dec_inactive_pool_size();
            }
        }
        found
            .into_iter()
            .map(|(seq_hash, block_id)| {
                let registered = inner
                    .inactive_blocks
                    .remove(&block_id)
                    .expect("inactive index/side-table out of sync");
                if block_id < inner.slots.len() {
                    inner.slots[block_id].state = SlotState::CheckedOut;
                }
                let primary = PrimaryBlock::new_attached(
                    Arc::new(registered),
                    self.inactive_return_fn.clone(),
                ) as Arc<dyn RegisteredBlock<T>>;
                (seq_hash, primary)
            })
            .collect()
    }

    /// Resurrect a single inactive block by hash, returning a `PrimaryBlock`.
    ///
    /// Uses `new_unattached` because the registry's `try_find_existing_block`
    /// holds the attachments lock while calling this. The caller MUST call
    /// `PrimaryBlock::store_weak_refs` once it drops the attachments lock.
    pub(crate) fn find_inactive_as_primary(
        &self,
        hash: SequenceHash,
        touch: bool,
    ) -> Option<Arc<PrimaryBlock<T>>> {
        let mut inner = self.inner.lock();
        let mut matched = inner.inactive.find_matches(&[hash], touch);
        let (_, block_id) = matched.pop()?;
        let registered = inner
            .inactive_blocks
            .remove(&block_id)
            .expect("inactive index/side-table out of sync");
        if block_id < inner.slots.len() {
            inner.slots[block_id].state = SlotState::CheckedOut;
        }
        if let Some(ref m) = self.metrics {
            m.dec_inactive_pool_size();
        }
        Some(PrimaryBlock::new_unattached(
            Arc::new(registered),
            self.inactive_return_fn.clone(),
        ))
    }
}

impl<T: BlockMetadata> std::fmt::Debug for BlockStore<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockStore")
            .field("block_size", &self.block_size)
            .field("total_blocks", &self.total_blocks)
            .finish()
    }
}
