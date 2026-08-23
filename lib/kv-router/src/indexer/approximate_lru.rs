// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Router-owned physical-capacity model for local approximate indexing.
//!
//! This intentionally mirrors the useful behavioral contract of the simulator
//! without depending on `aisimulate-core`, whose API is not yet stable enough
//! for a production dependency.
//!
//! NOTE/TODO: This shared lifecycle and capacity model may eventually motivate
//! a unified slot-tracker and cache-evictor substrate, similar in spirit to a
//! HiCache-style unified residency/load model. Keep this implementation narrowly
//! scoped and compositional while `aisimulate-core` remains experimental and fluid.
//!
//! NOTE: LRU bookkeeping is intentionally not rolled back if a synthetic radix
//! event fails to apply. The request release still makes its copies inactive,
//! and later capacity pressure reconciles them through ordinary eviction.

use std::{
    cmp::Reverse,
    collections::BTreeSet,
    sync::{Arc, atomic::Ordering},
    time::Instant,
};

use rustc_hash::FxHashMap;
use tokio::sync::oneshot;

use super::{KvRouterError, pruning::PruneConfig};
use crate::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash, RouterEvent, WorkerWithDpRank,
};
use crate::scheduling::AttemptId;
use dynamo_tokens::SequenceHash;

pub type ApproximateLruIncarnation = u64;
type BlockCopyId = u64;

#[derive(Debug, Clone)]
pub enum ApproximateRetentionConfig {
    Ttl(PruneConfig),
    Lru { fallback_ttl: PruneConfig },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApproximateAcquireMode {
    Lru,
    TtlFallback,
    Ignored,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ApproximateLruBlock {
    pub local_hash: LocalBlockHash,
    pub sequence_hash: SequenceHash,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ApproximateLruStats {
    pub ranks: usize,
    pub fallback_ranks: usize,
    pub resident_blocks: usize,
    pub active_blocks: usize,
    pub inactive_blocks: usize,
    pub private_blocks: usize,
    pub leases: usize,
    pub overcapacity_blocks: usize,
    pub requests: u64,
    pub request_messages: u64,
    pub output_batches: u64,
    pub fallback_activations: u64,
    pub eviction_batches: u64,
    pub evicted_blocks: u64,
    pub mutation_queue_depth: usize,
    pub mutation_wait_ns: u64,
    pub mutation_wait_samples: u64,
}

pub(crate) enum ApproximateLruCommand {
    SetCapacity {
        worker: WorkerWithDpRank,
        incarnation: ApproximateLruIncarnation,
        capacity: Option<usize>,
    },
    ResetRank {
        worker: WorkerWithDpRank,
    },
    Acquire {
        worker: WorkerWithDpRank,
        incarnation: ApproximateLruIncarnation,
        attempt_id: AttemptId,
        blocks: Vec<ApproximateLruBlock>,
        private_blocks: usize,
    },
    Materialize {
        worker: WorkerWithDpRank,
        incarnation: ApproximateLruIncarnation,
        attempt_id: AttemptId,
        parent_hash: Option<SequenceHash>,
        blocks: Vec<ApproximateLruBlock>,
        start_position: usize,
        private_blocks: usize,
    },
    Release {
        worker: WorkerWithDpRank,
        incarnation: ApproximateLruIncarnation,
        attempt_id: AttemptId,
    },
    Stats,
}

pub(crate) enum ApproximateLruReply {
    Applied,
    Acquired(ApproximateAcquireMode),
    Stats(ApproximateLruStats),
}

pub struct ApproximateLruTask {
    pub(crate) command: ApproximateLruCommand,
    pub(crate) response: Option<oneshot::Sender<Result<ApproximateLruReply, KvRouterError>>>,
    pub(crate) enqueued_at: Instant,
    pub(crate) queue_depth_at_enqueue: usize,
}

impl ApproximateLruTask {
    fn acknowledged(
        command: ApproximateLruCommand,
    ) -> (
        Self,
        oneshot::Receiver<Result<ApproximateLruReply, KvRouterError>>,
    ) {
        let (response, receiver) = oneshot::channel();
        (
            Self {
                command,
                response: Some(response),
                enqueued_at: Instant::now(),
                queue_depth_at_enqueue: 0,
            },
            receiver,
        )
    }

    fn unacknowledged(command: ApproximateLruCommand) -> Self {
        Self {
            command,
            response: None,
            enqueued_at: Instant::now(),
            queue_depth_at_enqueue: 0,
        }
    }

    pub(crate) fn complete(self, result: Result<ApproximateLruReply, KvRouterError>) {
        if let Some(response) = self.response {
            let _ = response.send(result);
        }
    }

    pub(crate) fn observe_enqueue_depth(&mut self, queue_depth: usize) {
        self.queue_depth_at_enqueue = queue_depth;
    }
}

pub(crate) trait ApproximateLruCommandSink: Send + Sync {
    fn send(&self, task: ApproximateLruTask) -> Result<(), KvRouterError>;
}

#[derive(Clone)]
pub struct ApproximateLruClient {
    sink: Arc<dyn ApproximateLruCommandSink>,
}

impl ApproximateLruClient {
    pub(crate) fn new(sink: Arc<dyn ApproximateLruCommandSink>) -> Self {
        Self { sink }
    }

    pub fn begin_request(
        &self,
        worker: WorkerWithDpRank,
        incarnation: ApproximateLruIncarnation,
        attempt_id: AttemptId,
    ) -> ApproximateLruLease {
        ApproximateLruLease::new(Arc::clone(&self.sink), worker, incarnation, attempt_id)
    }

    pub async fn set_capacity(
        &self,
        worker: WorkerWithDpRank,
        incarnation: ApproximateLruIncarnation,
        capacity: Option<usize>,
    ) -> Result<(), KvRouterError> {
        let reply = send_acknowledged(
            &self.sink,
            ApproximateLruCommand::SetCapacity {
                worker,
                incarnation,
                capacity,
            },
        )
        .await?;
        expect_applied(reply)
    }

    /// Enqueue rank registration without waiting for acknowledgement.
    ///
    /// Acquires use the same FIFO and never create rank state themselves, so a
    /// successfully enqueued capacity command is the durable incarnation fence.
    pub fn set_capacity_now(
        &self,
        worker: WorkerWithDpRank,
        incarnation: ApproximateLruIncarnation,
        capacity: Option<usize>,
    ) -> Result<(), KvRouterError> {
        self.sink.send(ApproximateLruTask::unacknowledged(
            ApproximateLruCommand::SetCapacity {
                worker,
                incarnation,
                capacity,
            },
        ))
    }

    pub async fn reset_rank(&self, worker: WorkerWithDpRank) -> Result<(), KvRouterError> {
        let reply =
            send_acknowledged(&self.sink, ApproximateLruCommand::ResetRank { worker }).await?;
        expect_applied(reply)
    }

    pub async fn stats(&self) -> Result<ApproximateLruStats, KvRouterError> {
        match send_acknowledged(&self.sink, ApproximateLruCommand::Stats).await? {
            ApproximateLruReply::Stats(stats) => Ok(stats),
            _ => Err(KvRouterError::IndexerDroppedRequest),
        }
    }
}

struct ApproximateLruLeaseInner {
    sink: Arc<dyn ApproximateLruCommandSink>,
    worker: WorkerWithDpRank,
    incarnation: ApproximateLruIncarnation,
    attempt_id: AttemptId,
    released: std::sync::atomic::AtomicBool,
}

#[derive(Clone)]
pub struct ApproximateLruLease {
    inner: Arc<ApproximateLruLeaseInner>,
}

pub struct ApproximateLruReleaseAck {
    response: oneshot::Receiver<Result<ApproximateLruReply, KvRouterError>>,
}

impl ApproximateLruReleaseAck {
    pub async fn wait(self) -> Result<(), KvRouterError> {
        expect_applied(
            self.response
                .await
                .map_err(|_| KvRouterError::IndexerDroppedRequest)??,
        )
    }
}

impl ApproximateLruLease {
    pub(crate) fn new(
        sink: Arc<dyn ApproximateLruCommandSink>,
        worker: WorkerWithDpRank,
        incarnation: ApproximateLruIncarnation,
        attempt_id: AttemptId,
    ) -> Self {
        Self {
            inner: Arc::new(ApproximateLruLeaseInner {
                sink,
                worker,
                incarnation,
                attempt_id,
                released: std::sync::atomic::AtomicBool::new(false),
            }),
        }
    }

    pub async fn acquire(
        &self,
        blocks: Vec<ApproximateLruBlock>,
        private_blocks: usize,
    ) -> Result<ApproximateAcquireMode, KvRouterError> {
        if self.inner.released.load(Ordering::Acquire) {
            return Err(KvRouterError::Unsupported(
                "approximate LRU lease is already complete".to_string(),
            ));
        }
        match send_acknowledged(
            &self.inner.sink,
            ApproximateLruCommand::Acquire {
                worker: self.inner.worker,
                incarnation: self.inner.incarnation,
                attempt_id: self.inner.attempt_id,
                blocks,
                private_blocks,
            },
        )
        .await?
        {
            ApproximateLruReply::Acquired(mode) => Ok(mode),
            _ => Err(KvRouterError::IndexerDroppedRequest),
        }
    }

    pub fn materialize(
        &self,
        parent_hash: Option<SequenceHash>,
        blocks: Vec<ApproximateLruBlock>,
        start_position: usize,
        private_blocks: usize,
    ) -> Result<(), KvRouterError> {
        if self.inner.released.load(Ordering::Acquire) {
            return Ok(());
        }
        self.inner.sink.send(ApproximateLruTask::unacknowledged(
            ApproximateLruCommand::Materialize {
                worker: self.inner.worker,
                incarnation: self.inner.incarnation,
                attempt_id: self.inner.attempt_id,
                parent_hash,
                blocks,
                start_position,
                private_blocks,
            },
        ))
    }

    pub fn begin_finish(&self) -> Result<Option<ApproximateLruReleaseAck>, KvRouterError> {
        if self.inner.released.swap(true, Ordering::AcqRel) {
            return Ok(None);
        }
        let (task, response) = ApproximateLruTask::acknowledged(ApproximateLruCommand::Release {
            worker: self.inner.worker,
            incarnation: self.inner.incarnation,
            attempt_id: self.inner.attempt_id,
        });
        self.inner.sink.send(task)?;
        Ok(Some(ApproximateLruReleaseAck { response }))
    }

    pub async fn finish(&self) -> Result<(), KvRouterError> {
        let Some(ack) = self.begin_finish()? else {
            return Ok(());
        };
        ack.wait().await
    }

    /// Synchronously enqueue an idempotent release without waiting for acknowledgement.
    pub fn release_now(&self) {
        if self.inner.released.swap(true, Ordering::AcqRel) {
            return;
        }
        let _ = self.inner.sink.send(ApproximateLruTask::unacknowledged(
            ApproximateLruCommand::Release {
                worker: self.inner.worker,
                incarnation: self.inner.incarnation,
                attempt_id: self.inner.attempt_id,
            },
        ));
    }
}

impl Drop for ApproximateLruLeaseInner {
    fn drop(&mut self) {
        if self.released.swap(true, Ordering::AcqRel) {
            return;
        }
        let _ = self.sink.send(ApproximateLruTask::unacknowledged(
            ApproximateLruCommand::Release {
                worker: self.worker,
                incarnation: self.incarnation,
                attempt_id: self.attempt_id,
            },
        ));
    }
}

async fn send_acknowledged(
    sink: &Arc<dyn ApproximateLruCommandSink>,
    command: ApproximateLruCommand,
) -> Result<ApproximateLruReply, KvRouterError> {
    let (task, response) = ApproximateLruTask::acknowledged(command);
    sink.send(task)?;
    response
        .await
        .map_err(|_| KvRouterError::IndexerDroppedRequest)?
}

fn expect_applied(reply: ApproximateLruReply) -> Result<(), KvRouterError> {
    match reply {
        ApproximateLruReply::Applied => Ok(()),
        _ => Err(KvRouterError::IndexerDroppedRequest),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct InactiveKey {
    last_access_epoch: u64,
    reverse_position: Reverse<usize>,
    copy_id: BlockCopyId,
}

struct BlockCopy {
    sequence_hash: SequenceHash,
    refs: usize,
    last_access_epoch: u64,
    sequence_position: usize,
}

#[derive(Default)]
struct LeaseState {
    copies: Vec<BlockCopyId>,
    private_blocks: usize,
}

struct RankLruState {
    capacity: usize,
    epoch: u64,
    next_copy_id: BlockCopyId,
    copies: FxHashMap<BlockCopyId, BlockCopy>,
    by_hash: FxHashMap<SequenceHash, Vec<BlockCopyId>>,
    inactive: BTreeSet<InactiveKey>,
    leases: FxHashMap<AttemptId, LeaseState>,
    private_blocks: usize,
    evicted_blocks: u64,
}

impl RankLruState {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            epoch: 0,
            next_copy_id: 1,
            copies: FxHashMap::default(),
            by_hash: FxHashMap::default(),
            inactive: BTreeSet::new(),
            leases: FxHashMap::default(),
            private_blocks: 0,
            evicted_blocks: 0,
        }
    }

    fn next_epoch(&mut self) -> u64 {
        self.epoch = self.epoch.wrapping_add(1).max(1);
        self.epoch
    }

    fn inactive_key(copy_id: BlockCopyId, copy: &BlockCopy) -> InactiveKey {
        InactiveKey {
            last_access_epoch: copy.last_access_epoch,
            reverse_position: Reverse(copy.sequence_position),
            copy_id,
        }
    }

    fn acquire(
        &mut self,
        attempt_id: AttemptId,
        blocks: &[ApproximateLruBlock],
        private_blocks: usize,
    ) -> Result<Vec<SequenceHash>, KvRouterError> {
        if self.leases.contains_key(&attempt_id) {
            return Err(KvRouterError::Unsupported(format!(
                "duplicate approximate LRU attempt {attempt_id}"
            )));
        }

        let epoch = self.next_epoch();
        let mut lease = LeaseState {
            copies: Vec::with_capacity(blocks.len()),
            private_blocks,
        };
        let mut prefix_hit = true;

        for (position, block) in blocks.iter().enumerate() {
            let copy_id = if prefix_hit {
                self.by_hash
                    .get(&block.sequence_hash)
                    .and_then(|copies| copies.first())
                    .copied()
            } else {
                None
            };

            let copy_id = if let Some(copy_id) = copy_id {
                let copy = self
                    .copies
                    .get_mut(&copy_id)
                    .ok_or(KvRouterError::IndexerDroppedRequest)?;
                if copy.refs == 0 {
                    let key = Self::inactive_key(copy_id, copy);
                    self.inactive.remove(&key);
                }
                copy.refs += 1;
                copy.last_access_epoch = copy.last_access_epoch.max(epoch);
                copy.sequence_position = position;
                copy_id
            } else {
                prefix_hit = false;
                let copy_id = self.next_copy_id;
                self.next_copy_id = self.next_copy_id.wrapping_add(1).max(1);
                self.copies.insert(
                    copy_id,
                    BlockCopy {
                        sequence_hash: block.sequence_hash,
                        refs: 1,
                        last_access_epoch: epoch,
                        sequence_position: position,
                    },
                );
                self.by_hash
                    .entry(block.sequence_hash)
                    .or_default()
                    .push(copy_id);
                copy_id
            };
            lease.copies.push(copy_id);
        }

        self.private_blocks += private_blocks;
        self.leases.insert(attempt_id, lease);
        Ok(self.reconcile())
    }

    fn materialize(
        &mut self,
        attempt_id: AttemptId,
        blocks: &[ApproximateLruBlock],
        start_position: usize,
        private_blocks: usize,
    ) -> Option<Vec<SequenceHash>> {
        let mut lease = self.leases.remove(&attempt_id)?;
        let epoch = self.next_epoch();
        for (offset, block) in blocks.iter().enumerate() {
            let copy_id = self.next_copy_id;
            self.next_copy_id = self.next_copy_id.wrapping_add(1).max(1);
            self.copies.insert(
                copy_id,
                BlockCopy {
                    sequence_hash: block.sequence_hash,
                    refs: 1,
                    last_access_epoch: epoch,
                    sequence_position: start_position + offset,
                },
            );
            self.by_hash
                .entry(block.sequence_hash)
                .or_default()
                .push(copy_id);
            lease.copies.push(copy_id);
        }
        if private_blocks > lease.private_blocks {
            self.private_blocks += private_blocks - lease.private_blocks;
        } else {
            self.private_blocks -= lease.private_blocks - private_blocks;
        }
        lease.private_blocks = private_blocks;
        self.leases.insert(attempt_id, lease);
        Some(self.reconcile())
    }

    fn release(&mut self, attempt_id: AttemptId) -> Vec<SequenceHash> {
        let Some(lease) = self.leases.remove(&attempt_id) else {
            return Vec::new();
        };
        self.private_blocks = self.private_blocks.saturating_sub(lease.private_blocks);
        for copy_id in lease.copies {
            let Some(copy) = self.copies.get_mut(&copy_id) else {
                continue;
            };
            if copy.refs == 0 {
                continue;
            }
            copy.refs -= 1;
            if copy.refs == 0 {
                self.inactive.insert(Self::inactive_key(copy_id, copy));
            }
        }
        self.reconcile()
    }

    fn reconcile(&mut self) -> Vec<SequenceHash> {
        let mut removed_hashes = Vec::new();
        let mut evicted_blocks = 0_u64;
        while self.resident_blocks() > self.capacity {
            let Some(key) = self.inactive.pop_first() else {
                break;
            };
            let Some(copy) = self.copies.remove(&key.copy_id) else {
                continue;
            };
            evicted_blocks = evicted_blocks.saturating_add(1);
            debug_assert_eq!(copy.refs, 0);
            let mut remove_hash = false;
            if let Some(copies) = self.by_hash.get_mut(&copy.sequence_hash) {
                if let Some(position) = copies.iter().position(|id| *id == key.copy_id) {
                    copies.swap_remove(position);
                }
                remove_hash = copies.is_empty();
            }
            if remove_hash {
                self.by_hash.remove(&copy.sequence_hash);
                removed_hashes.push(copy.sequence_hash);
            }
        }
        self.evicted_blocks = self.evicted_blocks.saturating_add(evicted_blocks);
        removed_hashes
    }

    fn resident_blocks(&self) -> usize {
        self.copies.len() + self.private_blocks
    }

    fn active_blocks(&self) -> usize {
        self.copies.values().filter(|copy| copy.refs > 0).count() + self.private_blocks
    }

    fn stats(&self) -> ApproximateLruStats {
        ApproximateLruStats {
            ranks: 1,
            fallback_ranks: 0,
            resident_blocks: self.resident_blocks(),
            active_blocks: self.active_blocks(),
            inactive_blocks: self.inactive.len(),
            private_blocks: self.private_blocks,
            leases: self.leases.len(),
            overcapacity_blocks: self.resident_blocks().saturating_sub(self.capacity),
            ..Default::default()
        }
    }

    #[cfg(test)]
    fn assert_invariants(&self) {
        assert_eq!(
            self.inactive.len(),
            self.copies.values().filter(|copy| copy.refs == 0).count()
        );
        assert_eq!(
            self.private_blocks,
            self.leases
                .values()
                .map(|lease| lease.private_blocks)
                .sum::<usize>()
        );
        for (copy_id, copy) in &self.copies {
            assert!(
                self.by_hash
                    .get(&copy.sequence_hash)
                    .is_some_and(|copies| copies.contains(copy_id))
            );
            let inactive_key = Self::inactive_key(*copy_id, copy);
            assert_eq!(copy.refs == 0, self.inactive.contains(&inactive_key));
        }
        for (hash, copy_ids) in &self.by_hash {
            assert!(!copy_ids.is_empty());
            for copy_id in copy_ids {
                assert_eq!(
                    self.copies.get(copy_id).map(|copy| copy.sequence_hash),
                    Some(*hash)
                );
            }
        }
        for lease in self.leases.values() {
            for copy_id in &lease.copies {
                assert!(self.copies.contains_key(copy_id));
            }
        }
    }
}

enum WorkerRetentionState {
    Lru {
        incarnation: ApproximateLruIncarnation,
        state: RankLruState,
    },
    TtlFallback {
        incarnation: ApproximateLruIncarnation,
    },
}

impl WorkerRetentionState {
    fn incarnation(&self) -> ApproximateLruIncarnation {
        match self {
            Self::Lru { incarnation, .. } | Self::TtlFallback { incarnation } => *incarnation,
        }
    }
}

#[derive(Default)]
pub(crate) struct ApproximateLruLane {
    ranks: FxHashMap<WorkerWithDpRank, WorkerRetentionState>,
    next_event_id: u64,
    requests: u64,
    request_messages: u64,
    output_batches: u64,
    fallback_activations: u64,
    eviction_batches: u64,
    evicted_blocks: u64,
    mutation_queue_depth: usize,
    mutation_wait_ns: u64,
    mutation_wait_samples: u64,
}

pub(crate) struct ApproximateLruApplyOutput {
    pub events: Vec<RouterEvent>,
    pub reply: ApproximateLruReply,
}

impl ApproximateLruLane {
    pub(crate) fn observe_task(&mut self, task: &ApproximateLruTask) {
        self.mutation_queue_depth = task.queue_depth_at_enqueue;
        self.mutation_wait_ns = self.mutation_wait_ns.saturating_add(
            task.enqueued_at
                .elapsed()
                .as_nanos()
                .min(u128::from(u64::MAX)) as u64,
        );
        self.mutation_wait_samples = self.mutation_wait_samples.saturating_add(1);
    }

    pub(crate) fn forget_worker(&mut self, worker_id: u64) {
        self.ranks.retain(|worker, _| worker.worker_id != worker_id);
    }

    pub(crate) fn forget_rank(&mut self, worker: WorkerWithDpRank) {
        self.ranks.remove(&worker);
    }

    pub fn apply(
        &mut self,
        command: ApproximateLruCommand,
    ) -> Result<ApproximateLruApplyOutput, KvRouterError> {
        let mut events = Vec::new();
        let reply = match command {
            ApproximateLruCommand::SetCapacity {
                worker,
                incarnation,
                capacity,
            } => {
                if self
                    .ranks
                    .get(&worker)
                    .is_some_and(|state| state.incarnation() != incarnation)
                {
                    self.push_clear_event(&mut events, worker);
                    self.ranks.remove(&worker);
                }
                if let Some(capacity) = capacity.filter(|capacity| *capacity > 0) {
                    let (removed, evicted) = match self.ranks.get_mut(&worker) {
                        Some(WorkerRetentionState::Lru { state, .. }) => {
                            let before = state.evicted_blocks;
                            state.capacity = capacity;
                            let removed = state.reconcile();
                            (removed, state.evicted_blocks.saturating_sub(before))
                        }
                        Some(WorkerRetentionState::TtlFallback { .. }) => (Vec::new(), 0),
                        None => {
                            self.ranks.insert(
                                worker,
                                WorkerRetentionState::Lru {
                                    incarnation,
                                    state: RankLruState::new(capacity),
                                },
                            );
                            (Vec::new(), 0)
                        }
                    };
                    self.record_evictions(evicted);
                    self.push_remove_event(&mut events, worker, removed);
                } else {
                    let was_lru = matches!(
                        self.ranks.get(&worker),
                        Some(WorkerRetentionState::Lru { .. })
                    );
                    let was_fallback = matches!(
                        self.ranks.get(&worker),
                        Some(WorkerRetentionState::TtlFallback { .. })
                    );
                    if was_lru {
                        self.push_clear_event(&mut events, worker);
                    }
                    if !was_fallback {
                        self.fallback_activations = self.fallback_activations.saturating_add(1);
                    }
                    self.ranks
                        .insert(worker, WorkerRetentionState::TtlFallback { incarnation });
                }
                ApproximateLruReply::Applied
            }
            ApproximateLruCommand::ResetRank { worker } => {
                self.ranks.remove(&worker);
                self.push_clear_event(&mut events, worker);
                ApproximateLruReply::Applied
            }
            ApproximateLruCommand::Acquire {
                worker,
                incarnation,
                attempt_id,
                blocks,
                private_blocks,
            } => {
                self.requests = self.requests.saturating_add(1);
                self.request_messages = self.request_messages.saturating_add(1);
                let Some(state) = self.ranks.get_mut(&worker) else {
                    return Ok(ApproximateLruApplyOutput {
                        events,
                        reply: ApproximateLruReply::Acquired(ApproximateAcquireMode::Ignored),
                    });
                };
                if state.incarnation() != incarnation {
                    return Ok(ApproximateLruApplyOutput {
                        events,
                        reply: ApproximateLruReply::Acquired(ApproximateAcquireMode::Ignored),
                    });
                }
                let WorkerRetentionState::Lru { state, .. } = state else {
                    return Ok(ApproximateLruApplyOutput {
                        events,
                        reply: ApproximateLruReply::Acquired(ApproximateAcquireMode::TtlFallback),
                    });
                };
                let before = state.evicted_blocks;
                let removed = state.acquire(attempt_id, &blocks, private_blocks)?;
                let evicted = state.evicted_blocks.saturating_sub(before);
                self.record_evictions(evicted);
                self.push_store_event(&mut events, worker, None, blocks);
                self.push_remove_event(&mut events, worker, removed);
                ApproximateLruReply::Acquired(ApproximateAcquireMode::Lru)
            }
            ApproximateLruCommand::Materialize {
                worker,
                incarnation,
                attempt_id,
                parent_hash,
                blocks,
                start_position,
                private_blocks,
            } => {
                self.request_messages = self.request_messages.saturating_add(1);
                if !blocks.is_empty() {
                    self.output_batches = self.output_batches.saturating_add(1);
                }
                if let Some(WorkerRetentionState::Lru {
                    incarnation: current,
                    state,
                }) = self.ranks.get_mut(&worker)
                    && *current == incarnation
                {
                    let before = state.evicted_blocks;
                    if let Some(removed) =
                        state.materialize(attempt_id, &blocks, start_position, private_blocks)
                    {
                        let evicted = state.evicted_blocks.saturating_sub(before);
                        self.record_evictions(evicted);
                        self.push_store_event(&mut events, worker, parent_hash, blocks);
                        self.push_remove_event(&mut events, worker, removed);
                    }
                }
                ApproximateLruReply::Applied
            }
            ApproximateLruCommand::Release {
                worker,
                incarnation,
                attempt_id,
            } => {
                self.request_messages = self.request_messages.saturating_add(1);
                if let Some(WorkerRetentionState::Lru {
                    incarnation: current,
                    state,
                }) = self.ranks.get_mut(&worker)
                    && *current == incarnation
                {
                    let before = state.evicted_blocks;
                    let removed = state.release(attempt_id);
                    let evicted = state.evicted_blocks.saturating_sub(before);
                    self.record_evictions(evicted);
                    self.push_remove_event(&mut events, worker, removed);
                }
                ApproximateLruReply::Applied
            }
            ApproximateLruCommand::Stats => ApproximateLruReply::Stats(self.stats()),
        };
        Ok(ApproximateLruApplyOutput { events, reply })
    }

    fn next_event_id(&mut self) -> u64 {
        self.next_event_id = self.next_event_id.wrapping_add(1).max(1);
        self.next_event_id
    }

    fn record_evictions(&mut self, evicted: u64) {
        if evicted == 0 {
            return;
        }
        self.eviction_batches = self.eviction_batches.saturating_add(1);
        self.evicted_blocks = self.evicted_blocks.saturating_add(evicted);
    }

    fn push_store_event(
        &mut self,
        events: &mut Vec<RouterEvent>,
        worker: WorkerWithDpRank,
        parent_hash: Option<SequenceHash>,
        blocks: Vec<ApproximateLruBlock>,
    ) {
        if blocks.is_empty() {
            return;
        }
        let stored = blocks
            .into_iter()
            .map(|block| KvCacheStoredBlockData {
                tokens_hash: block.local_hash,
                block_hash: ExternalSequenceBlockHash(block.sequence_hash),
                mm_extra_info: None,
            })
            .collect();
        events.push(RouterEvent::new(
            worker.worker_id,
            KvCacheEvent {
                event_id: self.next_event_id(),
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: parent_hash.map(ExternalSequenceBlockHash),
                    start_position: None,
                    blocks: stored,
                }),
                dp_rank: worker.dp_rank,
            },
        ));
    }

    fn push_remove_event(
        &mut self,
        events: &mut Vec<RouterEvent>,
        worker: WorkerWithDpRank,
        hashes: Vec<SequenceHash>,
    ) {
        if hashes.is_empty() {
            return;
        }
        events.push(RouterEvent::new(
            worker.worker_id,
            KvCacheEvent {
                event_id: self.next_event_id(),
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: hashes.into_iter().map(ExternalSequenceBlockHash).collect(),
                }),
                dp_rank: worker.dp_rank,
            },
        ));
    }

    fn push_clear_event(&mut self, events: &mut Vec<RouterEvent>, worker: WorkerWithDpRank) {
        events.push(RouterEvent::new(
            worker.worker_id,
            KvCacheEvent {
                event_id: self.next_event_id(),
                data: KvCacheEventData::Cleared,
                dp_rank: worker.dp_rank,
            },
        ));
    }

    fn stats(&self) -> ApproximateLruStats {
        let mut total = ApproximateLruStats::default();
        for state in self.ranks.values() {
            match state {
                WorkerRetentionState::Lru { state, .. } => {
                    let stats = state.stats();
                    total.ranks += 1;
                    total.resident_blocks += stats.resident_blocks;
                    total.active_blocks += stats.active_blocks;
                    total.inactive_blocks += stats.inactive_blocks;
                    total.private_blocks += stats.private_blocks;
                    total.leases += stats.leases;
                    total.overcapacity_blocks += stats.overcapacity_blocks;
                }
                WorkerRetentionState::TtlFallback { .. } => {
                    total.fallback_ranks += 1;
                }
            }
        }
        total.requests = self.requests;
        total.request_messages = self.request_messages;
        total.output_batches = self.output_batches;
        total.fallback_activations = self.fallback_activations;
        total.eviction_batches = self.eviction_batches;
        total.evicted_blocks = self.evicted_blocks;
        total.mutation_queue_depth = self.mutation_queue_depth;
        total.mutation_wait_ns = self.mutation_wait_ns;
        total.mutation_wait_samples = self.mutation_wait_samples;
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::compute_seq_hash_for_block;
    use crate::{
        ConcurrentRadixTreeCompressed,
        indexer::{KvIndexerInterface, ThreadPoolIndexer},
    };
    use std::time::Duration;

    fn worker() -> WorkerWithDpRank {
        WorkerWithDpRank::new(7, 0)
    }

    fn attempt(value: u64) -> AttemptId {
        AttemptId::new(value)
    }

    fn block(value: u64) -> ApproximateLruBlock {
        ApproximateLruBlock {
            local_hash: LocalBlockHash(value),
            sequence_hash: value,
        }
    }

    fn apply(lane: &mut ApproximateLruLane, command: ApproximateLruCommand) {
        lane.apply(command).unwrap();
        for state in lane.ranks.values() {
            if let WorkerRetentionState::Lru { state, .. } = state {
                state.assert_invariants();
            }
        }
    }

    #[test]
    fn equal_epoch_evicts_suffix_before_prefix() {
        let mut lane = ApproximateLruLane::default();
        apply(
            &mut lane,
            ApproximateLruCommand::SetCapacity {
                worker: worker(),
                incarnation: 1,
                capacity: Some(3),
            },
        );
        apply(
            &mut lane,
            ApproximateLruCommand::Acquire {
                worker: worker(),
                incarnation: 1,
                attempt_id: attempt(1),
                blocks: vec![block(1), block(2), block(3)],
                private_blocks: 0,
            },
        );
        apply(
            &mut lane,
            ApproximateLruCommand::Release {
                worker: worker(),
                incarnation: 1,
                attempt_id: attempt(1),
            },
        );
        let output = lane
            .apply(ApproximateLruCommand::SetCapacity {
                worker: worker(),
                incarnation: 1,
                capacity: Some(2),
            })
            .unwrap();
        let KvCacheEventData::Removed(removed) = &output.events[0].event.data else {
            panic!("expected remove event");
        };
        assert_eq!(removed.block_hashes, vec![ExternalSequenceBlockHash(3)]);
    }

    #[test]
    fn active_overcapacity_reconciles_on_release() {
        let mut lane = ApproximateLruLane::default();
        apply(
            &mut lane,
            ApproximateLruCommand::SetCapacity {
                worker: worker(),
                incarnation: 1,
                capacity: Some(1),
            },
        );
        apply(
            &mut lane,
            ApproximateLruCommand::Acquire {
                worker: worker(),
                incarnation: 1,
                attempt_id: attempt(1),
                blocks: vec![block(1), block(2)],
                private_blocks: 0,
            },
        );
        assert_eq!(lane.stats().overcapacity_blocks, 1);
        apply(
            &mut lane,
            ApproximateLruCommand::Release {
                worker: worker(),
                incarnation: 1,
                attempt_id: attempt(1),
            },
        );
        let stats = lane.stats();
        assert_eq!(stats.resident_blocks, 1);
        assert_eq!(stats.inactive_blocks, 1);
        assert_eq!(stats.leases, 0);
    }

    #[test]
    fn missing_capacity_pins_rank_to_ttl_until_reset() {
        let mut lane = ApproximateLruLane::default();
        apply(
            &mut lane,
            ApproximateLruCommand::SetCapacity {
                worker: worker(),
                incarnation: 1,
                capacity: None,
            },
        );
        let output = lane
            .apply(ApproximateLruCommand::Acquire {
                worker: worker(),
                incarnation: 1,
                attempt_id: attempt(1),
                blocks: vec![block(1)],
                private_blocks: 0,
            })
            .unwrap();
        assert!(matches!(
            output.reply,
            ApproximateLruReply::Acquired(ApproximateAcquireMode::TtlFallback)
        ));
        apply(
            &mut lane,
            ApproximateLruCommand::SetCapacity {
                worker: worker(),
                incarnation: 1,
                capacity: Some(4),
            },
        );
        assert_eq!(lane.stats().fallback_ranks, 1);
        apply(
            &mut lane,
            ApproximateLruCommand::ResetRank { worker: worker() },
        );
        apply(
            &mut lane,
            ApproximateLruCommand::SetCapacity {
                worker: worker(),
                incarnation: 2,
                capacity: Some(4),
            },
        );
        assert_eq!(lane.stats().ranks, 1);
    }

    #[test]
    fn reset_rejects_late_acquire_without_rank_registration() {
        let mut lane = ApproximateLruLane::default();
        apply(
            &mut lane,
            ApproximateLruCommand::SetCapacity {
                worker: worker(),
                incarnation: 1,
                capacity: Some(4),
            },
        );
        apply(
            &mut lane,
            ApproximateLruCommand::ResetRank { worker: worker() },
        );

        let output = lane
            .apply(ApproximateLruCommand::Acquire {
                worker: worker(),
                incarnation: 1,
                attempt_id: attempt(1),
                blocks: vec![block(1)],
                private_blocks: 0,
            })
            .unwrap();

        assert!(output.events.is_empty());
        assert!(matches!(
            output.reply,
            ApproximateLruReply::Acquired(ApproximateAcquireMode::Ignored)
        ));
        assert_eq!(lane.stats().ranks, 0);
    }

    #[test]
    fn cached_prefix_is_referenced_and_only_missing_suffix_allocates() {
        let mut lane = ApproximateLruLane::default();
        apply(
            &mut lane,
            ApproximateLruCommand::SetCapacity {
                worker: worker(),
                incarnation: 1,
                capacity: Some(4),
            },
        );
        apply(
            &mut lane,
            ApproximateLruCommand::Acquire {
                worker: worker(),
                incarnation: 1,
                attempt_id: attempt(1),
                blocks: vec![block(1), block(2)],
                private_blocks: 0,
            },
        );
        apply(
            &mut lane,
            ApproximateLruCommand::Release {
                worker: worker(),
                incarnation: 1,
                attempt_id: attempt(1),
            },
        );
        apply(
            &mut lane,
            ApproximateLruCommand::Acquire {
                worker: worker(),
                incarnation: 1,
                attempt_id: attempt(2),
                blocks: vec![block(1), block(2), block(3)],
                private_blocks: 0,
            },
        );

        let stats = lane.stats();
        assert_eq!(stats.resident_blocks, 3);
        assert_eq!(stats.active_blocks, 3);
        assert_eq!(stats.inactive_blocks, 0);
        apply(
            &mut lane,
            ApproximateLruCommand::Release {
                worker: worker(),
                incarnation: 1,
                attempt_id: attempt(2),
            },
        );
    }

    #[test]
    fn duplicate_physical_hash_is_removed_from_radix_only_after_final_copy() {
        let mut lane = ApproximateLruLane::default();
        apply(
            &mut lane,
            ApproximateLruCommand::SetCapacity {
                worker: worker(),
                incarnation: 1,
                capacity: Some(2),
            },
        );
        apply(
            &mut lane,
            ApproximateLruCommand::Acquire {
                worker: worker(),
                incarnation: 1,
                attempt_id: attempt(1),
                blocks: vec![block(1)],
                private_blocks: 0,
            },
        );
        apply(
            &mut lane,
            ApproximateLruCommand::Release {
                worker: worker(),
                incarnation: 1,
                attempt_id: attempt(1),
            },
        );
        apply(
            &mut lane,
            ApproximateLruCommand::Acquire {
                worker: worker(),
                incarnation: 1,
                attempt_id: attempt(2),
                blocks: Vec::new(),
                private_blocks: 0,
            },
        );
        apply(
            &mut lane,
            ApproximateLruCommand::Materialize {
                worker: worker(),
                incarnation: 1,
                attempt_id: attempt(2),
                parent_hash: None,
                blocks: vec![block(1)],
                start_position: 0,
                private_blocks: 0,
            },
        );
        apply(
            &mut lane,
            ApproximateLruCommand::Release {
                worker: worker(),
                incarnation: 1,
                attempt_id: attempt(2),
            },
        );

        let first_eviction = lane
            .apply(ApproximateLruCommand::SetCapacity {
                worker: worker(),
                incarnation: 1,
                capacity: Some(1),
            })
            .unwrap();
        assert!(first_eviction.events.is_empty());
        assert_eq!(lane.stats().resident_blocks, 1);

        let final_eviction = lane
            .apply(ApproximateLruCommand::Acquire {
                worker: worker(),
                incarnation: 1,
                attempt_id: attempt(3),
                blocks: vec![block(2)],
                private_blocks: 0,
            })
            .unwrap();
        let removed = final_eviction
            .events
            .iter()
            .find_map(|event| match &event.event.data {
                KvCacheEventData::Removed(removed) => Some(&removed.block_hashes),
                _ => None,
            })
            .expect("final copy eviction must remove radix membership");
        assert_eq!(removed, &vec![ExternalSequenceBlockHash(1)]);
        apply(
            &mut lane,
            ApproximateLruCommand::Release {
                worker: worker(),
                incarnation: 1,
                attempt_id: attempt(3),
            },
        );
    }

    #[test]
    fn reset_fences_late_release_and_output_from_old_lease() {
        let mut lane = ApproximateLruLane::default();
        apply(
            &mut lane,
            ApproximateLruCommand::SetCapacity {
                worker: worker(),
                incarnation: 1,
                capacity: Some(2),
            },
        );
        apply(
            &mut lane,
            ApproximateLruCommand::Acquire {
                worker: worker(),
                incarnation: 1,
                attempt_id: attempt(1),
                blocks: vec![block(1)],
                private_blocks: 1,
            },
        );
        apply(
            &mut lane,
            ApproximateLruCommand::ResetRank { worker: worker() },
        );
        apply(
            &mut lane,
            ApproximateLruCommand::SetCapacity {
                worker: worker(),
                incarnation: 2,
                capacity: Some(2),
            },
        );
        // Reuse the same attempt ID in the replacement incarnation to prove
        // that neither half of the fence is sufficient on its own.
        apply(
            &mut lane,
            ApproximateLruCommand::Acquire {
                worker: worker(),
                incarnation: 2,
                attempt_id: attempt(1),
                blocks: vec![block(9)],
                private_blocks: 1,
            },
        );
        let stale_output = lane
            .apply(ApproximateLruCommand::Materialize {
                worker: worker(),
                incarnation: 1,
                attempt_id: attempt(1),
                parent_hash: Some(1),
                blocks: vec![block(2)],
                start_position: 1,
                private_blocks: 0,
            })
            .unwrap();
        assert!(stale_output.events.is_empty());
        apply(
            &mut lane,
            ApproximateLruCommand::Release {
                worker: worker(),
                incarnation: 1,
                attempt_id: attempt(1),
            },
        );
        let replacement = lane.stats();
        assert_eq!(replacement.resident_blocks, 2);
        assert_eq!(replacement.private_blocks, 1);
        assert_eq!(replacement.leases, 1);
        apply(
            &mut lane,
            ApproximateLruCommand::Release {
                worker: worker(),
                incarnation: 2,
                attempt_id: attempt(1),
            },
        );
        assert_eq!(lane.stats().leases, 0);
    }

    #[tokio::test]
    async fn completed_output_block_is_visible_to_a_later_lookup() {
        let indexer = ThreadPoolIndexer::new_with_metrics_and_approximate_retention(
            ConcurrentRadixTreeCompressed::new(),
            2,
            4,
            None,
            Some(ApproximateRetentionConfig::Lru {
                fallback_ttl: PruneConfig {
                    ttl: Duration::from_secs(120),
                },
            }),
        );
        let worker = worker();

        let local_hashes = vec![LocalBlockHash(11), LocalBlockHash(22)];
        let sequence_hashes = compute_seq_hash_for_block(&local_hashes);
        indexer
            .set_approximate_lru_capacity(worker, 1, Some(4))
            .await
            .unwrap();
        let lease = indexer
            .begin_approximate_lru_request(worker, 1, attempt(1))
            .unwrap();
        lease
            .acquire(
                vec![ApproximateLruBlock {
                    local_hash: local_hashes[0],
                    sequence_hash: sequence_hashes[0],
                }],
                0,
            )
            .await
            .unwrap();
        lease
            .materialize(
                Some(sequence_hashes[0]),
                vec![ApproximateLruBlock {
                    local_hash: local_hashes[1],
                    sequence_hash: sequence_hashes[1],
                }],
                1,
                0,
            )
            .unwrap();
        lease.finish().await.unwrap();

        let scores = indexer.find_matches(local_hashes).await.unwrap();
        assert_eq!(scores.scores.get(&worker), Some(&2));
        let stats = indexer.approximate_lru_stats().await.unwrap();
        assert_eq!(stats.leases, 0);
        assert_eq!(stats.inactive_blocks, 2);
    }
}
