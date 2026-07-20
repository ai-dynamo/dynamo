// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Framework-neutral G2 residency and transfer simulation.
//!
//! Framework adapters decide when blocks are eligible for store or load. This
//! manager owns the shared mechanics: G2 capacity, pluggable eviction,
//! asynchronous transfer timing, source fences, and normalized trace events.

use std::collections::VecDeque;
use std::sync::Arc;

use anyhow::{Result, bail};
use rustc_hash::{FxHashMap, FxHashSet};

use super::eviction::{G2EvictionStrategy, build_g2_eviction_strategy};
use super::{
    CapacityHandling, G1Location, HostBlockKey, HostOffloadEvent, HostOffloadEventSink,
    PostLoadResidency, ResolvedHostOffloadPolicy, SourceFenceReason, TransferId,
};

/// Physical parameters for one G2 simulation.
///
/// The initial vLLM adapter is restricted to one worker/rank. Multi-rank
/// integration must resolve vLLM's engine-global logical capacity before it
/// can be reported as framework-native parity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HostOffloadConfig {
    pub capacity_blocks: usize,
    pub block_bytes: usize,
    /// `0.0` means zero simulated transfer latency.
    pub d2h_bandwidth_gbps: f64,
    /// `0.0` means zero simulated transfer latency.
    pub h2d_bandwidth_gbps: f64,
}

/// One logical block in a framework-selected store job.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StoreBlock {
    pub key: HostBlockKey,
    pub source: G1Location,
}

/// One logical block in a framework-selected load job.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoadBlock {
    pub key: HostBlockKey,
    pub destination: G1Location,
}

impl HostOffloadConfig {
    pub fn validate(self) -> Result<Self> {
        if self.capacity_blocks == 0 {
            bail!("host-offload G2 capacity must be greater than zero");
        }
        if self.block_bytes == 0 {
            bail!("host-offload block size must be greater than zero bytes");
        }
        let max_transfer_bytes = self
            .capacity_blocks
            .checked_mul(self.block_bytes)
            .ok_or_else(|| anyhow::anyhow!("host-offload G2 capacity in bytes overflowed"))?;
        validate_bandwidth("D2H", self.d2h_bandwidth_gbps, max_transfer_bytes)?;
        validate_bandwidth("H2D", self.h2d_bandwidth_gbps, max_transfer_bytes)?;
        Ok(self)
    }
}

/// Result of atomically preparing a framework-selected store batch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PrepareStoreOutcome {
    Prepared {
        transfer_id: TransferId,
        stored_blocks: usize,
    },
    AlreadyPresent,
    RetryCapacity,
}

/// One prepared store submitted to the D2H transfer lane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubmittedStore {
    pub transfer_id: TransferId,
    pub completes_at_ms: f64,
}

/// Visibility of one G2 key during framework prefix lookup.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum G2Lookup {
    Miss,
    Pending { transfer_id: TransferId },
    Hit,
}

/// Result of loading one visible G2 block into a reserved G1 destination.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LoadScheduleOutcome {
    Queued {
        transfer_id: TransferId,
        completes_at_ms: f64,
        loaded_blocks: usize,
    },
    Miss,
}

/// Pending stores that constrain source reuse or request preemption.
#[derive(Clone, Debug, PartialEq)]
pub struct SourceFence {
    pub until_ms: f64,
    pub transfer_ids: Vec<TransferId>,
}

/// Whether the supplied G1 sources are safe to reuse at the current time.
#[derive(Clone, Debug, PartialEq)]
pub enum SourceFenceOutcome {
    /// No matching store is prepared or in flight.
    Ready,
    /// Matching stores have not entered the D2H lane. The framework must
    /// submit prepared stores and query the fence again.
    NeedsSubmission { transfer_ids: Vec<TransferId> },
    /// Matching stores are in flight until the reported deadline.
    Pending(SourceFence),
}

/// Transfer completion returned to the framework adapter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompletedTransfer {
    Store {
        transfer_id: TransferId,
        blocks: Vec<StoreBlock>,
    },
    Load {
        transfer_id: TransferId,
        blocks: Vec<LoadBlock>,
    },
}

/// State changes produced by advancing the simulation clock.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HostOffloadEffects {
    pub completed: Vec<CompletedTransfer>,
}

/// G2 residency, transfer, and capacity core shared by framework profiles.
pub struct HostOffloadManager {
    policy: ResolvedHostOffloadPolicy,
    config: HostOffloadConfig,
    entries: FxHashMap<HostBlockKey, G2Entry>,
    eviction: Box<dyn G2EvictionStrategy>,
    free_slots: Vec<usize>,
    transfers: FxHashMap<TransferId, PendingTransfer>,
    prepared_stores: VecDeque<TransferId>,
    stores_by_source: FxHashMap<G1Location, Vec<TransferId>>,
    d2h: FifoTransferLane,
    h2d: FifoTransferLane,
    next_transfer_id: u64,
    current_time_ms: f64,
    event_sink: Option<Arc<dyn HostOffloadEventSink>>,
}

#[derive(Clone, Copy, Debug)]
struct G2Entry {
    slot: usize,
    state: G2EntryState,
    pins: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum G2EntryState {
    PendingStore {
        transfer_id: TransferId,
        source: G1Location,
    },
    Resident,
}

#[derive(Clone, Debug)]
struct PendingTransfer {
    phase: TransferPhase,
    kind: PendingTransferKind,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum TransferPhase {
    Prepared,
    Submitted { completes_at_ms: f64 },
}

#[derive(Clone, Debug)]
enum PendingTransferKind {
    Store { blocks: Vec<StoreBlock> },
    Load { blocks: Vec<LoadBlock> },
}

#[derive(Clone, Copy, Debug)]
struct FifoTransferLane {
    bytes_per_ms: f64,
    tail_ms: f64,
}

impl FifoTransferLane {
    fn new(gbps: f64) -> Self {
        let bytes_per_ms = if gbps > 0.0 {
            gbps * 1_000_000.0
        } else {
            f64::INFINITY
        };
        Self {
            bytes_per_ms,
            tail_ms: 0.0,
        }
    }

    fn submit(&mut self, now_ms: f64, bytes: usize) -> f64 {
        let start_ms = now_ms.max(self.tail_ms);
        let duration_ms = if self.bytes_per_ms.is_finite() {
            bytes as f64 / self.bytes_per_ms
        } else {
            0.0
        };
        self.tail_ms = start_ms + duration_ms;
        self.tail_ms
    }
}

impl HostOffloadManager {
    pub fn new(config: HostOffloadConfig, policy: ResolvedHostOffloadPolicy) -> Result<Self> {
        Self::build(config, policy, None)
    }

    pub fn with_event_sink(
        config: HostOffloadConfig,
        policy: ResolvedHostOffloadPolicy,
        event_sink: Arc<dyn HostOffloadEventSink>,
    ) -> Result<Self> {
        Self::build(config, policy, Some(event_sink))
    }

    fn build(
        config: HostOffloadConfig,
        policy: ResolvedHostOffloadPolicy,
        event_sink: Option<Arc<dyn HostOffloadEventSink>>,
    ) -> Result<Self> {
        let config = config.validate()?;
        validate_supported_policy(policy)?;
        let eviction = build_g2_eviction_strategy(policy.g2_eviction())?;
        let free_slots = (0..config.capacity_blocks).rev().collect();
        Ok(Self {
            policy,
            config,
            entries: FxHashMap::default(),
            eviction,
            free_slots,
            transfers: FxHashMap::default(),
            prepared_stores: VecDeque::new(),
            stores_by_source: FxHashMap::default(),
            d2h: FifoTransferLane::new(config.d2h_bandwidth_gbps),
            h2d: FifoTransferLane::new(config.h2d_bandwidth_gbps),
            next_transfer_id: 0,
            current_time_ms: 0.0,
            event_sink,
        })
    }

    pub fn policy(&self) -> ResolvedHostOffloadPolicy {
        self.policy
    }

    pub fn config(&self) -> HostOffloadConfig {
        self.config
    }

    /// Atomically reserve G2 capacity and prepare one D2H store job for a
    /// framework-selected block batch.
    ///
    /// The adapter must call [`Self::tick`] at the start of each scheduler
    /// pass before scheduling work at that pass's time.
    pub fn prepare_store(&mut self, blocks: &[StoreBlock], now_ms: f64) -> PrepareStoreOutcome {
        let now_ms = self.prepare_schedule(now_ms);
        let protected: Vec<_> = blocks.iter().map(|block| block.key).collect();
        let mut seen = FxHashSet::default();
        let blocks: Vec<_> = blocks
            .iter()
            .copied()
            .filter(|block| seen.insert(block.key) && !self.entries.contains_key(&block.key))
            .collect();
        if blocks.is_empty() {
            return PrepareStoreOutcome::AlreadyPresent;
        }

        let Some(slots) = self.reserve_g2_slots(blocks.len(), &protected, now_ms) else {
            for block in &blocks {
                self.record(HostOffloadEvent::CapacityRetry {
                    at_ms: now_ms,
                    key: block.key,
                });
            }
            return PrepareStoreOutcome::RetryCapacity;
        };

        let transfer_id = self.next_transfer_id();
        for (block, slot) in blocks.iter().zip(slots) {
            self.entries.insert(
                block.key,
                G2Entry {
                    slot,
                    state: G2EntryState::PendingStore {
                        transfer_id,
                        source: block.source,
                    },
                    pins: 0,
                },
            );
            let source_jobs = self.stores_by_source.entry(block.source).or_default();
            if !source_jobs.contains(&transfer_id) {
                source_jobs.push(transfer_id);
            }
            self.record(HostOffloadEvent::StorePrepared {
                at_ms: now_ms,
                transfer_id,
                key: block.key,
                source: block.source,
                bytes: self.config.block_bytes,
            });
        }
        self.transfers.insert(
            transfer_id,
            PendingTransfer {
                phase: TransferPhase::Prepared,
                kind: PendingTransferKind::Store {
                    blocks: blocks.clone(),
                },
            },
        );
        self.prepared_stores.push_back(transfer_id);
        PrepareStoreOutcome::Prepared {
            transfer_id,
            stored_blocks: blocks.len(),
        }
    }

    /// Submit every prepared store at the beginning of an engine step.
    ///
    /// Stores share one FIFO D2H lane and are submitted in transfer-id order,
    /// which is their framework preparation order.
    pub fn submit_prepared_stores(&mut self, now_ms: f64) -> Vec<SubmittedStore> {
        let now_ms = self.prepare_schedule(now_ms);
        let mut submitted = Vec::with_capacity(self.prepared_stores.len());
        while let Some(transfer_id) = self.prepared_stores.pop_front() {
            let blocks = match &self
                .transfers
                .get(&transfer_id)
                .expect("prepared store must remain registered")
                .kind
            {
                PendingTransferKind::Store { blocks } => blocks.clone(),
                PendingTransferKind::Load { .. } => {
                    unreachable!("prepared queue can contain only store jobs")
                }
            };
            let completes_at_ms = self.d2h.submit(now_ms, self.transfer_bytes(blocks.len()));
            let transfer = self
                .transfers
                .get_mut(&transfer_id)
                .expect("prepared store must remain registered");
            debug_assert_eq!(transfer.phase, TransferPhase::Prepared);
            transfer.phase = TransferPhase::Submitted { completes_at_ms };
            for block in &blocks {
                self.record(HostOffloadEvent::StoreQueued {
                    at_ms: now_ms,
                    transfer_id,
                    key: block.key,
                    source: block.source,
                    bytes: self.config.block_bytes,
                });
            }
            submitted.push(SubmittedStore {
                transfer_id,
                completes_at_ms,
            });
        }
        submitted
    }

    /// Query one host key. Framework adapters own prefix scanning because
    /// pending results have framework-specific deferral behavior.
    pub fn lookup(&self, key: HostBlockKey) -> G2Lookup {
        match self.entries.get(&key).map(|entry| entry.state) {
            None => G2Lookup::Miss,
            Some(G2EntryState::PendingStore { transfer_id, .. }) => {
                G2Lookup::Pending { transfer_id }
            }
            Some(G2EntryState::Resident) => G2Lookup::Hit,
        }
    }

    /// Mark one visible G2 block most-recently used.
    ///
    /// Framework adapters own batch ordering. For example, vLLM touches a
    /// logical prefix in reverse order so its earliest block becomes MRU.
    pub fn touch(&mut self, key: HostBlockKey) {
        if self
            .entries
            .get(&key)
            .is_some_and(|entry| entry.state == G2EntryState::Resident && entry.pins == 0)
        {
            self.touch_resident(key);
        }
    }

    /// Enqueue one load job for a visible G2 block batch.
    pub fn schedule_load(&mut self, blocks: &[LoadBlock], now_ms: f64) -> LoadScheduleOutcome {
        self.schedule_load_not_before(blocks, now_ms, now_ms)
    }

    /// Enqueue one load job after an external source-reuse fence.
    ///
    /// `now_ms` is the scheduler decision time, while `not_before_ms` is the
    /// earliest instant at which the worker may start H2D. This models vLLM's
    /// allocation-before-worker-flush ordering: the destination may be
    /// reserved immediately, but cannot be overwritten until a pending D2H
    /// reading that physical block completes.
    pub fn schedule_load_not_before(
        &mut self,
        blocks: &[LoadBlock],
        now_ms: f64,
        not_before_ms: f64,
    ) -> LoadScheduleOutcome {
        let now_ms = self.prepare_schedule(now_ms);
        assert_valid_time("host-offload load not-before time", not_before_ms);
        let not_before_ms = not_before_ms.max(now_ms);
        if blocks.is_empty()
            || blocks.len() > self.config.capacity_blocks
            || blocks.iter().any(|block| {
                !self
                    .entries
                    .get(&block.key)
                    .is_some_and(|entry| entry.state == G2EntryState::Resident)
            })
        {
            return LoadScheduleOutcome::Miss;
        }
        for block in blocks {
            let became_pinned = {
                let entry = self
                    .entries
                    .get_mut(&block.key)
                    .expect("validated G2 load block must remain resident");
                let became_pinned = entry.pins == 0;
                entry.pins = entry
                    .pins
                    .checked_add(1)
                    .expect("G2 load pin count exhausted");
                became_pinned
            };
            if became_pinned {
                self.eviction.on_pin(block.key);
            }
        }

        let transfer_id = self.next_transfer_id();
        let bytes = self.transfer_bytes(blocks.len());
        let completes_at_ms = self.h2d.submit(not_before_ms, bytes);
        self.transfers.insert(
            transfer_id,
            PendingTransfer {
                phase: TransferPhase::Submitted { completes_at_ms },
                kind: PendingTransferKind::Load {
                    blocks: blocks.to_vec(),
                },
            },
        );
        for block in blocks {
            self.record(HostOffloadEvent::LoadQueued {
                at_ms: now_ms,
                transfer_id,
                key: block.key,
                destination: block.destination,
                bytes: self.config.block_bytes,
            });
        }
        LoadScheduleOutcome::Queued {
            transfer_id,
            completes_at_ms,
            loaded_blocks: blocks.len(),
        }
    }

    /// Cancel an in-flight load and release its G2 pins. The H2D lane tail is
    /// intentionally left unchanged so loads already queued behind it keep a
    /// conservative completion time.
    pub fn cancel_load(&mut self, transfer_id: TransferId) -> bool {
        let Some(transfer) = self.transfers.remove(&transfer_id) else {
            return false;
        };
        let PendingTransferKind::Load { blocks } = transfer.kind else {
            self.transfers.insert(transfer_id, transfer);
            return false;
        };
        for block in blocks {
            let became_evictable = {
                let entry = self
                    .entries
                    .get_mut(&block.key)
                    .expect("cancelled load must retain its G2 entry");
                entry.pins = entry
                    .pins
                    .checked_sub(1)
                    .expect("cancelled G2 load must hold a pin");
                entry.pins == 0
            };
            if became_evictable {
                self.eviction.on_unpin(block.key);
            }
        }
        true
    }

    /// Report what must happen before the supplied G1 sources can be reused or
    /// released by preemption.
    ///
    /// When this returns [`SourceFenceOutcome::NeedsSubmission`], framework
    /// adapters that flush as part of their fence semantics must call
    /// [`Self::submit_prepared_stores`] and query again.
    pub fn fence_sources(
        &mut self,
        sources: &[G1Location],
        reason: SourceFenceReason,
        now_ms: f64,
    ) -> SourceFenceOutcome {
        let now_ms = self.prepare_schedule(now_ms);
        let mut needs_submission = Vec::new();
        let mut pending = Vec::new();
        let mut fenced_blocks = Vec::new();
        let mut until_ms = now_ms;
        for source in sources {
            let Some(ids) = self.stores_by_source.get(source) else {
                continue;
            };
            for transfer_id in ids {
                let Some(transfer) = self.transfers.get(transfer_id) else {
                    continue;
                };
                let PendingTransferKind::Store { blocks } = &transfer.kind else {
                    continue;
                };
                match transfer.phase {
                    TransferPhase::Prepared => needs_submission.push(*transfer_id),
                    TransferPhase::Submitted { completes_at_ms } => {
                        until_ms = until_ms.max(completes_at_ms);
                        pending.push(*transfer_id);
                        fenced_blocks.extend(
                            blocks
                                .iter()
                                .filter(|block| block.source == *source)
                                .map(|block| (*transfer_id, *block)),
                        );
                    }
                }
            }
        }

        if !needs_submission.is_empty() {
            needs_submission.sort_unstable();
            needs_submission.dedup();
            return SourceFenceOutcome::NeedsSubmission {
                transfer_ids: needs_submission,
            };
        }
        if pending.is_empty() {
            return SourceFenceOutcome::Ready;
        }

        pending.sort_unstable();
        pending.dedup();
        fenced_blocks
            .sort_unstable_by_key(|(transfer_id, block)| (*transfer_id, block.source, block.key));
        fenced_blocks.dedup();
        for (transfer_id, block) in fenced_blocks {
            self.record(HostOffloadEvent::SourceFenced {
                at_ms: now_ms,
                transfer_id,
                key: block.key,
                source: block.source,
                reason,
            });
        }
        SourceFenceOutcome::Pending(SourceFence {
            until_ms,
            transfer_ids: pending,
        })
    }

    /// Advance transfer state and make completed stores visible in G2.
    pub fn tick(&mut self, now_ms: f64) -> HostOffloadEffects {
        assert_valid_time("host-offload tick", now_ms);
        if now_ms < self.current_time_ms {
            return HostOffloadEffects::default();
        }
        self.current_time_ms = now_ms;
        let mut ready: Vec<_> = self
            .transfers
            .iter()
            .filter_map(|(id, transfer)| {
                let TransferPhase::Submitted { completes_at_ms } = transfer.phase else {
                    return None;
                };
                (completes_at_ms <= now_ms).then_some((completes_at_ms, *id))
            })
            .collect();
        ready.sort_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        });

        let mut effects = HostOffloadEffects::default();
        for (completes_at_ms, transfer_id) in ready {
            let transfer = self
                .transfers
                .remove(&transfer_id)
                .expect("ready transfer must remain registered");
            match transfer.kind {
                PendingTransferKind::Store { blocks } => {
                    // The framework does not expose a semantic recency order
                    // among blocks completing in one job. Keep adapter order
                    // deterministic; parity workloads must touch blocks before
                    // asserting an LRU victim from such a tie.
                    for block in &blocks {
                        let entry = self
                            .entries
                            .get_mut(&block.key)
                            .expect("pending store must reserve a G2 entry");
                        debug_assert_eq!(
                            entry.state,
                            G2EntryState::PendingStore {
                                transfer_id,
                                source: block.source
                            }
                        );
                        entry.state = G2EntryState::Resident;
                        self.remove_source_transfer(block.source, transfer_id);
                        self.eviction.on_resident(block.key);
                        self.record(HostOffloadEvent::StoreCompleted {
                            at_ms: completes_at_ms,
                            transfer_id,
                            key: block.key,
                        });
                    }
                    effects.completed.push(CompletedTransfer::Store {
                        transfer_id,
                        blocks,
                    });
                }
                PendingTransferKind::Load { blocks } => {
                    for block in &blocks {
                        let became_evictable = {
                            let entry = self
                                .entries
                                .get_mut(&block.key)
                                .expect("in-flight load must pin a G2 entry");
                            entry.pins = entry
                                .pins
                                .checked_sub(1)
                                .expect("completed G2 load must hold a pin");
                            entry.pins == 0
                        };
                        if became_evictable {
                            self.eviction.on_unpin(block.key);
                        }
                        self.record(HostOffloadEvent::LoadCompleted {
                            at_ms: completes_at_ms,
                            transfer_id,
                            key: block.key,
                        });
                    }
                    effects.completed.push(CompletedTransfer::Load {
                        transfer_id,
                        blocks,
                    });
                }
            }
        }
        effects
    }

    pub fn next_deadline(&self) -> Option<f64> {
        self.transfers
            .values()
            .filter_map(|transfer| match transfer.phase {
                TransferPhase::Prepared => None,
                TransferPhase::Submitted { completes_at_ms } => Some(completes_at_ms),
            })
            .min_by(f64::total_cmp)
    }

    pub fn has_pending_work(&self) -> bool {
        !self.transfers.is_empty()
    }

    /// Prepared stores require another engine step even though they do not
    /// have a transfer deadline yet.
    pub fn needs_engine_step(&self) -> bool {
        !self.prepared_stores.is_empty()
    }

    /// Blocks consuming G2 capacity, including pending stores.
    pub fn used_blocks(&self) -> usize {
        self.entries.len()
    }

    /// Visible, fully stored blocks.
    pub fn resident_blocks(&self) -> usize {
        self.entries
            .values()
            .filter(|entry| entry.state == G2EntryState::Resident)
            .count()
    }

    pub fn is_resident(&self, key: HostBlockKey) -> bool {
        self.entries
            .get(&key)
            .is_some_and(|entry| entry.state == G2EntryState::Resident)
    }

    pub fn resident_snapshot(&self) -> Vec<HostBlockKey> {
        let mut keys: Vec<_> = self
            .entries
            .iter()
            .filter_map(|(key, entry)| (entry.state == G2EntryState::Resident).then_some(*key))
            .collect();
        keys.sort_unstable_by(|left, right| {
            left.group_index
                .cmp(&right.group_index)
                .then_with(|| left.parent.cmp(&right.parent))
                .then_with(|| left.token_digest.cmp(&right.token_digest))
        });
        keys
    }

    fn reserve_g2_slots(
        &mut self,
        needed: usize,
        protected: &[HostBlockKey],
        now_ms: f64,
    ) -> Option<Vec<usize>> {
        debug_assert!(needed > 0);
        let to_evict = needed.saturating_sub(self.free_slots.len());
        let protected: FxHashSet<_> = protected.iter().copied().collect();
        let entries = &self.entries;
        let victims = self.eviction.select_victims(to_evict, &mut |key| {
            !protected.contains(&key)
                && entries
                    .get(&key)
                    .is_some_and(|entry| entry.state == G2EntryState::Resident && entry.pins == 0)
        });
        if victims.len() < to_evict {
            return None;
        }

        for victim in victims {
            let entry = self
                .entries
                .remove(&victim)
                .expect("selected G2 victim must remain resident");
            debug_assert_eq!(entry.state, G2EntryState::Resident);
            debug_assert_eq!(entry.pins, 0);
            self.eviction.on_remove(victim);
            self.free_slots.push(entry.slot);
            self.record(HostOffloadEvent::G2Evicted {
                at_ms: now_ms,
                key: victim,
            });
        }

        Some(
            (0..needed)
                .map(|_| {
                    self.free_slots
                        .pop()
                        .expect("validated G2 capacity must provide enough slots")
                })
                .collect(),
        )
    }

    fn prepare_schedule(&mut self, now_ms: f64) -> f64 {
        assert_valid_time("host-offload schedule time", now_ms);
        let now_ms = now_ms.max(self.current_time_ms);
        assert!(
            self.next_deadline()
                .is_none_or(|deadline| deadline >= now_ms),
            "tick must process host-offload completions through {now_ms} ms before scheduling"
        );
        self.current_time_ms = now_ms;
        now_ms
    }

    fn touch_resident(&mut self, key: HostBlockKey) {
        let entry = self.entries.get(&key).expect("touched G2 block must exist");
        debug_assert_eq!(entry.state, G2EntryState::Resident);
        debug_assert_eq!(entry.pins, 0);
        self.eviction.on_access(key);
    }

    fn next_transfer_id(&mut self) -> TransferId {
        let id = TransferId::new(self.next_transfer_id);
        self.next_transfer_id = self
            .next_transfer_id
            .checked_add(1)
            .expect("host-offload transfer id counter exhausted");
        id
    }

    fn transfer_bytes(&self, blocks: usize) -> usize {
        self.config
            .block_bytes
            .checked_mul(blocks)
            .expect("host-offload transfer byte count overflowed")
    }

    fn remove_source_transfer(&mut self, source: G1Location, transfer_id: TransferId) {
        let remove_entry = if let Some(ids) = self.stores_by_source.get_mut(&source) {
            ids.retain(|id| *id != transfer_id);
            ids.is_empty()
        } else {
            false
        };
        if remove_entry {
            self.stores_by_source.remove(&source);
        }
    }

    fn record(&self, event: HostOffloadEvent) {
        if let Some(event_sink) = &self.event_sink {
            event_sink.record(&event);
        }
    }
}

fn validate_bandwidth(direction: &str, gbps: f64, max_transfer_bytes: usize) -> Result<()> {
    if !gbps.is_finite() || gbps < 0.0 {
        bail!("host-offload {direction} bandwidth must be finite and non-negative, got {gbps}");
    }
    let bytes_per_ms = gbps * 1_000_000.0;
    if !bytes_per_ms.is_finite() {
        bail!("host-offload {direction} bandwidth conversion overflowed");
    }
    if gbps > 0.0 {
        let max_duration_ms = max_transfer_bytes as f64 / bytes_per_ms;
        if !max_duration_ms.is_finite() {
            bail!("host-offload {direction} worst-case transfer duration overflowed");
        }
    }
    Ok(())
}

fn validate_supported_policy(policy: ResolvedHostOffloadPolicy) -> Result<()> {
    // The neutral manager validates only policy dimensions that alter its G2
    // mechanics. Every framework adapter must exhaustively validate the
    // remaining scheduler-owned dimensions before constructing this manager.
    if policy.capacity_handling() != CapacityHandling::EvictOrRetry {
        bail!("the host-offload PoC supports only evict-or-retry capacity handling");
    }
    if policy.post_load_residency() != PostLoadResidency::RetainG2Copy {
        bail!("the host-offload PoC supports only retaining the G2 copy after load");
    }
    Ok(())
}

fn assert_valid_time(label: &str, time_ms: f64) {
    assert!(
        time_ms.is_finite() && time_ms >= 0.0,
        "{label} must be finite and non-negative, got {time_ms}"
    );
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    #[derive(Default)]
    struct CapturingSink(Mutex<Vec<HostOffloadEvent>>);

    impl HostOffloadEventSink for CapturingSink {
        fn record(&self, event: &HostOffloadEvent) {
            self.0.lock().unwrap().push(event.clone());
        }
    }

    fn key(value: u8) -> HostBlockKey {
        HostBlockKey::new(0, None, [value; 32])
    }

    fn store(key_value: u8, source: u64) -> StoreBlock {
        StoreBlock {
            key: key(key_value),
            source: G1Location::new(source),
        }
    }

    fn load(key_value: u8, destination: u64) -> LoadBlock {
        LoadBlock {
            key: key(key_value),
            destination: G1Location::new(destination),
        }
    }

    fn manager(capacity_blocks: usize) -> HostOffloadManager {
        HostOffloadManager::new(
            HostOffloadConfig {
                capacity_blocks,
                block_bytes: 1_000_000,
                d2h_bandwidth_gbps: 1.0,
                h2d_bandwidth_gbps: 1.0,
            },
            ResolvedHostOffloadPolicy::vllm_offloading_connector_defaults(),
        )
        .unwrap()
    }

    #[test]
    fn store_is_prepared_then_submitted_on_the_next_step() {
        let mut manager = manager(1);
        let outcome = manager.prepare_store(&[store(1, 10)], 0.0);
        assert_eq!(
            outcome,
            PrepareStoreOutcome::Prepared {
                transfer_id: TransferId::new(0),
                stored_blocks: 1,
            }
        );
        assert_eq!(manager.used_blocks(), 1);
        assert_eq!(manager.resident_blocks(), 0);
        assert_eq!(
            manager.lookup(key(1)),
            G2Lookup::Pending {
                transfer_id: TransferId::new(0)
            }
        );
        assert!(manager.needs_engine_step());
        assert_eq!(manager.next_deadline(), None);
        assert!(manager.tick(0.5).completed.is_empty());

        assert_eq!(
            manager.submit_prepared_stores(1.0),
            vec![SubmittedStore {
                transfer_id: TransferId::new(0),
                completes_at_ms: 2.0,
            }]
        );
        assert!(!manager.needs_engine_step());
        assert_eq!(manager.next_deadline(), Some(2.0));

        assert!(manager.tick(1.5).completed.is_empty());
        let effects = manager.tick(2.0);
        assert_eq!(manager.resident_blocks(), 1);
        assert_eq!(manager.lookup(key(1)), G2Lookup::Hit);
        assert_eq!(
            effects.completed,
            vec![CompletedTransfer::Store {
                transfer_id: TransferId::new(0),
                blocks: vec![store(1, 10)],
            }]
        );
    }

    #[test]
    fn batch_store_becomes_visible_atomically() {
        let mut manager = manager(2);
        assert_eq!(
            manager.prepare_store(&[store(1, 1), store(2, 2)], 0.0),
            PrepareStoreOutcome::Prepared {
                transfer_id: TransferId::new(0),
                stored_blocks: 2,
            }
        );
        assert_eq!(
            manager.submit_prepared_stores(1.0),
            vec![SubmittedStore {
                transfer_id: TransferId::new(0),
                completes_at_ms: 3.0,
            }]
        );

        manager.tick(2.0);
        assert_eq!(
            manager.lookup(key(1)),
            G2Lookup::Pending {
                transfer_id: TransferId::new(0)
            }
        );
        assert_eq!(
            manager.lookup(key(2)),
            G2Lookup::Pending {
                transfer_id: TransferId::new(0)
            }
        );
        let effects = manager.tick(3.0);
        assert_eq!(manager.resident_snapshot(), vec![key(1), key(2)]);
        assert_eq!(
            effects.completed,
            vec![CompletedTransfer::Store {
                transfer_id: TransferId::new(0),
                blocks: vec![store(1, 1), store(2, 2)],
            }]
        );
    }

    #[test]
    fn presence_filter_deduplicates_pending_and_resident_stores() {
        let mut manager = manager(1);
        manager.prepare_store(&[store(1, 10)], 0.0);
        assert_eq!(
            manager.prepare_store(&[store(1, 11)], 0.0),
            PrepareStoreOutcome::AlreadyPresent
        );
        manager.submit_prepared_stores(0.0);
        manager.tick(1.0);
        assert_eq!(
            manager.prepare_store(&[store(1, 11)], 1.0),
            PrepareStoreOutcome::AlreadyPresent
        );
    }

    #[test]
    fn store_batch_protects_existing_input_keys_from_eviction() {
        let mut manager = manager(1);
        manager.prepare_store(&[store(1, 1)], 0.0);
        manager.submit_prepared_stores(0.0);
        manager.tick(1.0);

        assert_eq!(
            manager.prepare_store(&[store(1, 1), store(2, 2)], 1.0),
            PrepareStoreOutcome::RetryCapacity
        );
        assert_eq!(manager.lookup(key(1)), G2Lookup::Hit);
        assert_eq!(manager.lookup(key(2)), G2Lookup::Miss);

        // A rejected atomic admission only previews victims. The next
        // unprotected admission must still be able to select the same LRU.
        assert!(matches!(
            manager.prepare_store(&[store(3, 3)], 1.0),
            PrepareStoreOutcome::Prepared { .. }
        ));
        assert_eq!(manager.lookup(key(1)), G2Lookup::Miss);
    }

    #[test]
    fn vllm_touch_order_controls_lru_victim() {
        let mut manager = manager(2);
        manager.prepare_store(&[store(1, 1), store(2, 2)], 0.0);
        manager.submit_prepared_stores(0.0);
        manager.tick(2.0);
        // vLLM reverses the logical prefix when applying LRU touches.
        manager.touch(key(2));
        manager.touch(key(1));

        manager.prepare_store(&[store(3, 3)], 2.0);
        assert!(manager.is_resident(key(1)));
        assert!(!manager.is_resident(key(2)));
        assert!(!manager.is_resident(key(3)));
    }

    #[test]
    fn pinned_load_causes_capacity_retry_not_drop() {
        let mut manager = manager(1);
        manager.prepare_store(&[store(1, 1)], 0.0);
        manager.submit_prepared_stores(0.0);
        manager.tick(1.0);
        manager.schedule_load(&[load(1, 9)], 1.0);

        assert_eq!(
            manager.prepare_store(&[store(2, 2)], 1.0),
            PrepareStoreOutcome::RetryCapacity
        );
        assert!(manager.is_resident(key(1)));

        manager.tick(2.0);
        assert!(matches!(
            manager.prepare_store(&[store(2, 2)], 2.0),
            PrepareStoreOutcome::Prepared { .. }
        ));
        assert!(!manager.is_resident(key(1)));
    }

    #[test]
    fn cancelled_load_returns_g2_block_to_evictable_lru() {
        let mut manager = manager(1);
        manager.prepare_store(&[store(1, 1)], 0.0);
        manager.submit_prepared_stores(0.0);
        manager.tick(1.0);
        let transfer_id = match manager.schedule_load(&[load(1, 9)], 1.0) {
            LoadScheduleOutcome::Queued { transfer_id, .. } => transfer_id,
            outcome => panic!("resident block must queue a load, got {outcome:?}"),
        };

        assert_eq!(
            manager.prepare_store(&[store(2, 2)], 1.0),
            PrepareStoreOutcome::RetryCapacity,
            "in-flight load must pin its G2 source"
        );
        assert!(manager.cancel_load(transfer_id));
        assert!(matches!(
            manager.prepare_store(&[store(2, 2)], 1.0),
            PrepareStoreOutcome::Prepared { .. }
        ));
        assert!(!manager.is_resident(key(1)));
    }

    #[test]
    fn manager_eligibility_excludes_pinned_and_cohort_protected_blocks() {
        let mut manager = manager(3);
        manager.prepare_store(&[store(1, 1), store(2, 2), store(3, 3)], 0.0);
        manager.submit_prepared_stores(0.0);
        manager.tick(3.0);

        manager.schedule_load(&[load(1, 10)], 3.0);
        assert!(matches!(
            manager.prepare_store(&[store(2, 2), store(4, 4)], 3.0),
            PrepareStoreOutcome::Prepared { .. }
        ));

        assert!(manager.is_resident(key(1)), "pinned block was evicted");
        assert!(manager.is_resident(key(2)), "protected block was evicted");
        assert!(!manager.is_resident(key(3)), "eligible LRU was retained");
        assert!(matches!(manager.lookup(key(4)), G2Lookup::Pending { .. }));
    }

    #[test]
    fn load_completion_reinserts_block_at_mru() {
        let mut manager = manager(2);
        manager.prepare_store(&[store(1, 1), store(2, 2)], 0.0);
        manager.submit_prepared_stores(0.0);
        manager.tick(2.0);

        manager.schedule_load(&[load(1, 10)], 2.0);
        // Active blocks are outside the evictable LRU; framework touches
        // during the load must not reinsert them early.
        manager.touch(key(1));
        manager.tick(3.0);
        manager.prepare_store(&[store(3, 3)], 3.0);

        assert!(manager.is_resident(key(1)));
        assert!(!manager.is_resident(key(2)));
    }

    #[test]
    fn directional_lanes_overlap_and_each_lane_is_fifo() {
        let mut manager = manager(3);
        manager.prepare_store(&[store(1, 1)], 0.0);
        assert_eq!(manager.submit_prepared_stores(0.0)[0].completes_at_ms, 1.0);
        manager.tick(1.0);
        manager.prepare_store(&[store(2, 2)], 1.0);
        manager.prepare_store(&[store(3, 3)], 1.0);
        let submitted = manager.submit_prepared_stores(1.0);
        assert_eq!(submitted[0].completes_at_ms, 2.0);
        assert_eq!(submitted[1].completes_at_ms, 3.0);
        assert!(matches!(
            manager.schedule_load(&[load(1, 10)], 1.0),
            LoadScheduleOutcome::Queued {
                completes_at_ms: 2.0,
                loaded_blocks: 1,
                ..
            }
        ));
    }

    #[test]
    fn source_fence_reports_pending_store_deadline() {
        let sink = Arc::new(CapturingSink::default());
        let mut manager = HostOffloadManager::with_event_sink(
            HostOffloadConfig {
                capacity_blocks: 1,
                block_bytes: 1_000_000,
                d2h_bandwidth_gbps: 1.0,
                h2d_bandwidth_gbps: 1.0,
            },
            ResolvedHostOffloadPolicy::vllm_offloading_connector_defaults(),
            sink.clone(),
        )
        .unwrap();
        manager.prepare_store(&[store(1, 7)], 0.0);

        assert_eq!(
            manager.fence_sources(&[G1Location::new(7)], SourceFenceReason::SourceReuse, 0.25),
            SourceFenceOutcome::NeedsSubmission {
                transfer_ids: vec![TransferId::new(0)]
            }
        );
        manager.submit_prepared_stores(0.25);

        let SourceFenceOutcome::Pending(fence) =
            manager.fence_sources(&[G1Location::new(7)], SourceFenceReason::SourceReuse, 0.25)
        else {
            panic!("submitted store must produce a pending source fence");
        };
        assert_eq!(fence.until_ms, 1.25);
        assert_eq!(fence.transfer_ids, vec![TransferId::new(0)]);
        assert_eq!(
            manager.fence_sources(&[G1Location::new(99)], SourceFenceReason::SourceReuse, 0.25),
            SourceFenceOutcome::Ready
        );
        let events = sink.0.lock().unwrap();
        assert!(matches!(events[0], HostOffloadEvent::StorePrepared { .. }));
        assert!(matches!(events[1], HostOffloadEvent::StoreQueued { .. }));
        assert!(matches!(
            events[2],
            HostOffloadEvent::SourceFenced {
                reason: SourceFenceReason::SourceReuse,
                ..
            }
        ));
    }

    #[test]
    fn invalid_capacity_and_bandwidth_are_rejected() {
        let policy = ResolvedHostOffloadPolicy::vllm_offloading_connector_defaults();
        let invalid_capacity = HostOffloadConfig {
            capacity_blocks: 0,
            block_bytes: 1,
            d2h_bandwidth_gbps: 1.0,
            h2d_bandwidth_gbps: 1.0,
        };
        assert!(HostOffloadManager::new(invalid_capacity, policy).is_err());

        let invalid_bandwidth = HostOffloadConfig {
            capacity_blocks: 1,
            block_bytes: 1,
            d2h_bandwidth_gbps: f64::NAN,
            h2d_bandwidth_gbps: 1.0,
        };
        assert!(HostOffloadManager::new(invalid_bandwidth, policy).is_err());

        let negative_bandwidth = HostOffloadConfig {
            capacity_blocks: 1,
            block_bytes: 1,
            d2h_bandwidth_gbps: 1.0,
            h2d_bandwidth_gbps: -1.0,
        };
        assert!(HostOffloadManager::new(negative_bandwidth, policy).is_err());

        let overflowing_capacity = HostOffloadConfig {
            capacity_blocks: usize::MAX,
            block_bytes: 2,
            d2h_bandwidth_gbps: 1.0,
            h2d_bandwidth_gbps: 1.0,
        };
        assert!(HostOffloadManager::new(overflowing_capacity, policy).is_err());

        let overflowing_duration = HostOffloadConfig {
            capacity_blocks: 1,
            block_bytes: usize::MAX,
            d2h_bandwidth_gbps: f64::MIN_POSITIVE,
            h2d_bandwidth_gbps: 1.0,
        };
        assert!(HostOffloadManager::new(overflowing_duration, policy).is_err());
    }

    #[test]
    fn oversized_load_batch_is_rejected_before_transfer_sizing() {
        let mut manager = manager(1);
        assert!(matches!(
            manager.prepare_store(&[store(1, 10)], 0.0),
            PrepareStoreOutcome::Prepared { .. }
        ));
        manager.submit_prepared_stores(0.0);
        manager.tick(1.0);

        assert_eq!(
            manager.schedule_load(&[load(1, 20), load(1, 21)], 1.0),
            LoadScheduleOutcome::Miss
        );
    }
}
