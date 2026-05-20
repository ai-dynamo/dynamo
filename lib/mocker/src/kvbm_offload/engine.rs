// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-process wrapper over kvbm-engine's `OffloadEngine` + `InstanceLeader`
//! backed by [`MockWorker`](super::worker::MockWorker).
//!
//! Construction is `async` (velo's `Messenger` needs a short TCP warmup).
//! The hot-path methods ([`tick`](MockOffloadEngine::tick),
//! [`prepare_onboard_prefix`](MockOffloadEngine::prepare_onboard_prefix),
//! [`start_onboard_prefix`](MockOffloadEngine::start_onboard_prefix),
//! [`earliest_pending_deadline`](MockOffloadEngine::earliest_pending_deadline))
//! are synchronous. `tick` drives PS completion using `now_ms` supplied by
//! the caller — live replay feeds wall-clock time, offline replay feeds
//! virtual time.

use std::future::Future;
use std::net::TcpListener;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use anyhow::Result;
use dynamo_tokens::{BlockHash, SequenceHash as RouterSequenceHash};
use futures::Stream;
use futures::stream::{FuturesUnordered, StreamExt};
use futures::task::noop_waker_ref;

use kvbm_engine::leader::{
    FindMatchesOptions, FindMatchesResult, InstanceLeader, Leader, OnboardingStatus, StagingMode,
};
use kvbm_engine::offload::{
    ExternalBlock, OffloadEngine, PendingTracker, PipelineBuilder, PresenceFilter, SourceBlocks,
    TransferHandle, TransferStatus,
};
use kvbm_engine::worker::Worker;
use kvbm_engine::{BlockId, G1 as EngineG1, G2, G3, SequenceHash};
use kvbm_logical::blocks::{BlockMetadata, ImmutableBlock, MutableBlock};
use kvbm_logical::events::{EventsManager, KvCacheEvent as LogicalKvCacheEvent};
use kvbm_logical::manager::{BlockManager, FrequencyTrackingCapacity};
use kvbm_logical::pools::BlockDuplicationPolicy;
use kvbm_logical::registry::BlockRegistry;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::common::protocols::G1 as MockerG1;

use super::capacity_reservation::{
    CapacityReservationGuard, CapacityReservationPolicy, CapacityReservations,
};
use super::config::KvbmOffloadConfig;
use super::shared_g3::SharedG3Pool;
use super::worker::MockWorker;

// Successful offline barriers wake via kvbm-engine watch channels or the
// mock worker's Notify. The timeout is only a hang guard for pipeline bugs.
const PIPELINE_BARRIER_TIMEOUT: Duration = Duration::from_secs(1);

#[derive(Clone, Copy)]
enum ReservationBlocker {
    LocalOffload,
    SharedG3Offload,
}

#[derive(Clone, Copy)]
enum TransferLane {
    G1ToG2,
    G2ToG3,
}

/// Handle returned by [`MockOffloadEngine::start_onboard_prefix`]. Scheduler
/// parks one per deferred request and polls
/// [`is_complete`](Self::is_complete) each pass; the bit is flipped by
/// [`MockOffloadEngine::tick`] when the underlying transfer drains from
/// the onboard model.
///
/// The handle holds strong [`ImmutableBlock<G2>`] references for the
/// duration of the swap-in. kvbm-logical's inactive pool refuses to
/// reclaim a G2 block while any strong ref is live — so a concurrent
/// offload cannot race in and reassign the slots we're about to
/// onboard. Dropping the handle (after the scheduler promotes or
/// abandons the swap-in) releases the blocks back.
pub struct SwapInHandle {
    complete: Arc<AtomicBool>,
    /// Number of G2 blocks this swap-in delivers.
    block_count: Arc<AtomicUsize>,
    /// Strong refs pinning the G2 blocks for the transfer's lifetime.
    /// Not accessed directly — presence alone upholds the RAII contract.
    _g2_blocks: Option<Vec<ImmutableBlock<G2>>>,
}

impl SwapInHandle {
    pub fn is_complete(&self) -> bool {
        self.complete.load(std::sync::atomic::Ordering::Acquire)
    }

    pub fn block_count(&self) -> usize {
        self.block_count.load(Ordering::Acquire)
    }
}

/// Lower-tier lookup prepared while the caller reserves destination G1 slots.
/// G2-ready matches hold source blocks immediately; G3 matches are kept as a
/// staging plan so a failed admission probe does not spend G3 bandwidth.
pub(crate) enum PreparedSwapIn {
    Ready {
        requested_blocks: usize,
        g2_blocks: Vec<ImmutableBlock<G2>>,
    },
    Staging {
        requested_blocks: usize,
        reservation_blocks: usize,
        g2_capacity_reservation: Option<CapacityReservationGuard>,
        lookup_plhs: Vec<SequenceHash>,
    },
}

impl PreparedSwapIn {
    fn from_g2_blocks(requested_blocks: usize, g2_blocks: Vec<ImmutableBlock<G2>>) -> Self {
        Self::Ready {
            requested_blocks,
            g2_blocks,
        }
    }

    fn from_staging_plan(
        requested_blocks: usize,
        reservation_blocks: usize,
        g2_capacity_reservation: Option<CapacityReservationGuard>,
        lookup_plhs: Vec<SequenceHash>,
    ) -> Self {
        Self::Staging {
            requested_blocks,
            reservation_blocks,
            g2_capacity_reservation,
            lookup_plhs,
        }
    }

    pub(crate) fn reservation_block_count(&self) -> usize {
        match self {
            Self::Ready { g2_blocks, .. } => g2_blocks.len(),
            Self::Staging {
                reservation_blocks, ..
            } => *reservation_blocks,
        }
    }

    pub(crate) fn block_count(&self) -> usize {
        self.reservation_block_count()
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct LowerTierLookupPlan {
    g2_prefix_blocks: usize,
    g3_stage_blocks: usize,
}

impl LowerTierLookupPlan {
    fn reservation_blocks(self) -> usize {
        self.g2_prefix_blocks.saturating_add(self.g3_stage_blocks)
    }
}

struct PendingStagedSwapIn {
    result: FindMatchesResult,
    reservation_blocks: usize,
    complete: Arc<AtomicBool>,
    block_count: Arc<AtomicUsize>,
    g2_capacity_reservation: Option<CapacityReservationGuard>,
    g2_blocks: Option<Vec<ImmutableBlock<G2>>>,
    g2_to_g1_started: bool,
}

impl PendingStagedSwapIn {
    fn is_done(&self) -> bool {
        self.g2_to_g1_started && self.complete.load(Ordering::Acquire)
    }
}

/// Router-facing metadata for a block that may become resident in G2.
///
/// kvbm-engine indexes G2 by [`SequenceHash`] (a positional lineage hash),
/// while the router protocol needs the external sequence hash plus the local
/// token hash edge that reconstructs lower-tier continuations.
#[derive(Clone, Debug)]
pub(crate) struct G2BlockEventMetadata {
    pub(crate) seq_hash: RouterSequenceHash,
    pub(crate) parent_hash: Option<RouterSequenceHash>,
    pub(crate) local_hash: BlockHash,
    pub(crate) token_ids: Option<Vec<u32>>,
}

#[derive(Clone, Debug)]
pub(crate) struct G2OffloadBlock {
    pub(crate) block_id: BlockId,
    pub(crate) plh: SequenceHash,
    pub(crate) metadata: G2BlockEventMetadata,
}

#[derive(Clone, Debug)]
pub(crate) enum G2RouterEvent {
    Stored(G2BlockEventMetadata),
    Removed { seq_hash: RouterSequenceHash },
}

struct PendingG1ToG2 {
    handle: TransferHandle,
    g2_to_g3_chain_blocks: FxHashMap<BlockId, SequenceHash>,
    /// Reset G1 slots held until the simulated source copy completes.
    ///
    /// These tokens do not preserve old bytes — `MockWorker` never reads
    /// source contents — but they do preserve G1 capacity accounting. A real
    /// DMA source slot cannot be reassigned while the copy is in flight.
    source_slots: FxHashMap<BlockId, MutableBlock<MockerG1>>,
}

impl PendingG1ToG2 {
    fn source_slots_releasable(&self) -> bool {
        if self.source_slots.is_empty() && self.g2_to_g3_chain_blocks.is_empty() {
            return true;
        }
        if !self.g2_to_g3_chain_blocks.is_empty() {
            return false;
        }
        if self.handle.is_complete() {
            return true;
        }
        let passed = self.handle.passed_blocks().len();
        if passed == 0 {
            return !matches!(self.handle.status(), TransferStatus::Evaluating);
        }
        self.handle.completed_blocks().len() + self.handle.failed_blocks().len() >= passed
    }

    fn release_completed_source_slots(&mut self) -> usize {
        let mut released = 0usize;
        for block_id in self
            .handle
            .completed_blocks()
            .into_iter()
            .chain(self.handle.failed_blocks())
        {
            if self.source_slots.remove(&block_id).is_some() {
                released += 1;
            }
        }
        released
    }

    fn collect_completed_chain_blocks(&mut self) -> Vec<SequenceHash> {
        if self.g2_to_g3_chain_blocks.is_empty() {
            return Vec::new();
        }

        let mut chain_blocks = Vec::new();
        for block_id in self.handle.completed_blocks() {
            if let Some(seq_hash) = self.g2_to_g3_chain_blocks.remove(&block_id) {
                chain_blocks.push(seq_hash);
            }
        }

        for block_id in self.handle.failed_blocks() {
            self.g2_to_g3_chain_blocks.remove(&block_id);
        }

        if !matches!(self.handle.status(), TransferStatus::Evaluating) {
            let passed: FxHashSet<BlockId> = self.handle.passed_blocks().into_iter().collect();
            self.g2_to_g3_chain_blocks
                .retain(|block_id, _seq_hash| passed.contains(block_id));
        }

        chain_blocks
    }
}

struct PendingG2ToG3 {
    handle: TransferHandle,
    released_failed_reservations: usize,
}

impl PendingG2ToG3 {
    fn take_unreleased_failed_reservations(&mut self) -> usize {
        let failed = self.handle.failed_blocks().len();
        let unreleased = failed.saturating_sub(self.released_failed_reservations);
        self.released_failed_reservations =
            self.released_failed_reservations.saturating_add(unreleased);
        unreleased
    }

    fn is_complete(&self) -> bool {
        self.handle.is_complete()
    }
}

/// In-process offload engine driving a G1→G2 pipeline over [`MockWorker`].
///
/// G1 blocks are handed to kvbm-engine as
/// [`kvbm_engine::offload::SourceBlocks::External`] — `(block_id, plh)`
/// pairs with no strong ref.
///
/// This is a mocker-only data shortcut. A real G1→G2 byte copy would need
/// the source HBM slot to remain stable until DMA completes. Here,
/// [`MockWorker`] never reads the source block contents; it uses only the
/// block count for timing and the PLH for destination registration. To keep
/// G1 capacity accurate, callers can pass reset source-slot tokens alongside
/// the external refs; the engine holds those tokens until transfer completion.
#[allow(dead_code)]
pub struct MockOffloadEngine {
    config: KvbmOffloadConfig,

    engine: OffloadEngine,
    leader: Arc<InstanceLeader>,
    worker: Arc<MockWorker>,
    registry: Arc<BlockRegistry>,
    g2_manager: Arc<BlockManager<G2>>,
    g2_destination_reservations: Arc<CapacityReservations>,
    shared_g3: Option<Arc<SharedG3Pool>>,
    g3_manager: Option<Arc<BlockManager<G3>>>,
    pending_g1_to_g2: Mutex<Vec<PendingG1ToG2>>,
    pending_g2_to_g3: Mutex<Vec<PendingG2ToG3>>,
    pending_staged_swap_ins: Mutex<Vec<PendingStagedSwapIn>>,
    g2_event_stream: Mutex<Pin<Box<dyn Stream<Item = LogicalKvCacheEvent> + Send>>>,
    g2_event_metadata: Mutex<FxHashMap<SequenceHash, G2BlockEventMetadata>>,

    /// Runtime the engine owns for kvbm-engine background pipeline /
    /// session-receiver tasks. Keeping this runtime on the engine lets both
    /// live and offline synchronous scheduler passes explicitly pump transfer
    /// publication after PS completions drain.
    _runtime: Option<tokio::runtime::Runtime>,
}

impl MockOffloadEngine {
    /// Build the engine end-to-end against a fresh `InstanceLeader` and
    /// `MockWorker`. Caller must be inside a tokio runtime; in offline
    /// mode, `init_kvbm_offline` constructs a one-worker multi-thread
    /// runtime and calls this via `block_on`.
    pub async fn new(config: KvbmOffloadConfig) -> Result<Self> {
        let messenger = create_local_messenger().await?;
        let g2_events_manager = Arc::new(EventsManager::builder().build());
        let g2_event_stream = Box::pin(g2_events_manager.subscribe());
        let registry = Arc::new(build_registry(g2_events_manager));
        let g2_manager = Arc::new(build_g2_block_manager(
            config.num_g2_blocks,
            config.block_size_tokens,
            &registry,
        ));
        let shared_g3 = SharedG3Pool::get_or_create(&config)?;
        let g3_manager = shared_g3.as_ref().map(|pool| pool.manager());
        tracing::debug!(
            num_g2_blocks = config.num_g2_blocks,
            num_g3_blocks = config.num_g3_blocks,
            g3_enabled = g3_manager.is_some(),
            "kvbm-offload: building mock offload engine"
        );

        let worker = Arc::new(MockWorker::new(
            config.block_size_bytes.unwrap_or(0),
            config.bandwidth_g1_to_g2_gbps,
            config.bandwidth_g2_to_g1_gbps,
            None,
            None,
            shared_g3.clone(),
        ));
        let worker_for_leader: Arc<dyn Worker> = worker.clone();

        // `InstanceLeader::build` calls `Handle::current()` internally,
        // hence the `async fn` and the in-runtime caller requirement.
        let mut leader_builder = InstanceLeader::builder()
            .messenger(messenger)
            .registry((*registry).clone())
            .g2_manager(g2_manager.clone())
            .worker(worker_for_leader);
        if let Some(g3_manager) = &g3_manager {
            leader_builder = leader_builder.g3_manager(g3_manager.clone());
        }
        let leader = Arc::new(leader_builder.build()?);

        let g2_destination_reservations = Arc::new(CapacityReservations::default());
        let g1_to_g2_pending = Arc::new(PendingTracker::new());
        let g1_to_g2_presence = PresenceFilter::<EngineG1, G2>::new(registry.clone())
            .with_pending_tracker(g1_to_g2_pending.clone());
        let g1_to_g2_capacity = CapacityReservationPolicy::<EngineG1, G2>::new(
            g2_manager.clone(),
            g2_destination_reservations.clone(),
        );
        let g1_to_g2_pipeline = PipelineBuilder::<EngineG1, G2>::new()
            .policy(Arc::new(g1_to_g2_presence))
            .policy(Arc::new(g1_to_g2_capacity))
            .pending_tracker(g1_to_g2_pending)
            .batch_size(config.offload_batch_size)
            .max_concurrent_transfers(config.offload_batch_size)
            .build();

        let mut engine_builder = OffloadEngine::builder(leader.clone())
            .with_registry(registry.clone())
            .with_g2_manager(g2_manager.clone())
            .with_runtime(tokio::runtime::Handle::current())
            .with_g1_to_g2_pipeline(g1_to_g2_pipeline);

        if let (Some(g3_manager), Some(shared_g3)) = (&g3_manager, &shared_g3) {
            let g2_to_g3_pending = shared_g3.pending_tracker();
            let g3_registry = Arc::new(g3_manager.block_registry().clone());
            let g2_to_g3_presence = PresenceFilter::<G2, G3>::new(g3_registry)
                .with_pending_tracker(g2_to_g3_pending.clone());
            let g2_to_g3_capacity = CapacityReservationPolicy::<G2, G3>::new(
                g3_manager.clone(),
                shared_g3.capacity_reservations(),
            );
            let g2_to_g3_pipeline = PipelineBuilder::<G2, G3>::new()
                .policy(Arc::new(g2_to_g3_presence))
                .policy(Arc::new(g2_to_g3_capacity))
                .pending_tracker(g2_to_g3_pending)
                .batch_size(config.offload_batch_size)
                // G2→G3 is a background write-through copy over one shared
                // lower-tier link. Multiple concurrent batches do not add
                // bandwidth in the mock PS model, but they pin many G2 source
                // blocks and can starve foreground G1→G2 offloads.
                .max_concurrent_transfers(1)
                .build();
            engine_builder = engine_builder
                .with_g3_manager(g3_manager.clone())
                .with_g2_to_g3_pipeline(g2_to_g3_pipeline);
        }

        let engine = engine_builder.build()?;

        Ok(Self {
            config,
            engine,
            leader,
            worker,
            registry,
            g2_manager,
            g2_destination_reservations,
            shared_g3,
            g3_manager,
            pending_g1_to_g2: Mutex::new(Vec::new()),
            pending_g2_to_g3: Mutex::new(Vec::new()),
            pending_staged_swap_ins: Mutex::new(Vec::new()),
            g2_event_stream: Mutex::new(g2_event_stream),
            g2_event_metadata: Mutex::new(FxHashMap::default()),
            _runtime: None,
        })
    }

    /// Hand ownership of a tokio runtime to the engine so its worker thread
    /// outlives the `block_on` that constructed us.
    ///
    /// ```ignore
    /// let rt = tokio::runtime::Builder::new_multi_thread()
    ///     .worker_threads(1).enable_all().build()?;
    /// let mut engine = rt.block_on(MockOffloadEngine::new(cfg))?;
    /// engine.attach_runtime(rt);
    /// ```
    pub fn attach_runtime(&mut self, rt: tokio::runtime::Runtime) {
        self._runtime = Some(rt);
    }

    fn remember_g2_event_metadata(&self, blocks: &[G2OffloadBlock]) {
        let mut metadata = self
            .g2_event_metadata
            .lock()
            .expect("G2 event metadata mutex poisoned");
        for block in blocks {
            metadata.insert(block.plh, block.metadata.clone());
        }
    }

    fn drain_g2_lifecycle_events(&self) -> Vec<LogicalKvCacheEvent> {
        let mut stream = self
            .g2_event_stream
            .lock()
            .expect("G2 event stream mutex poisoned");
        let mut events = Vec::new();
        let mut cx = Context::from_waker(noop_waker_ref());
        while let Poll::Ready(Some(event)) = stream.as_mut().poll_next(&mut cx) {
            events.push(event);
        }
        events
    }

    /// Drain kvbm-logical G2 lifecycle notifications and translate them into
    /// router-tier events. The caller owns event IDs and publishing.
    pub(crate) fn drain_g2_router_events(&self) -> Vec<G2RouterEvent> {
        let lifecycle_events = self.drain_g2_lifecycle_events();
        if lifecycle_events.is_empty() {
            return Vec::new();
        }

        let mut metadata = self
            .g2_event_metadata
            .lock()
            .expect("G2 event metadata mutex poisoned");
        let mut router_events = Vec::new();
        for event in lifecycle_events {
            match event {
                LogicalKvCacheEvent::Create(plh) => {
                    if let Some(meta) = metadata.get(&plh).cloned() {
                        router_events.push(G2RouterEvent::Stored(meta));
                    }
                }
                LogicalKvCacheEvent::Remove(plh) => {
                    if let Some(meta) = metadata.remove(&plh) {
                        router_events.push(G2RouterEvent::Removed {
                            seq_hash: meta.seq_hash,
                        });
                    }
                }
            }
        }
        router_events
    }

    async fn with_barrier_timeout<F>(wait: F) -> bool
    where
        F: Future<Output = bool>,
    {
        tokio::time::timeout(PIPELINE_BARRIER_TIMEOUT, wait)
            .await
            .unwrap_or_default()
    }

    fn wait_on_attached_runtime<F>(&self, wait: F) -> bool
    where
        F: Future<Output = bool>,
    {
        let Some(rt) = self._runtime.as_ref() else {
            return true;
        };
        let current = tokio::runtime::Handle::try_current().ok();
        match current.as_ref().map(tokio::runtime::Handle::runtime_flavor) {
            Some(tokio::runtime::RuntimeFlavor::MultiThread) => {
                tokio::task::block_in_place(|| rt.block_on(Self::with_barrier_timeout(wait)))
            }
            // Starting a runtime from inside a current-thread runtime would
            // panic. Tests in that shape can still make progress on the next
            // explicit tick.
            Some(_) => true,
            None => rt.block_on(Self::with_barrier_timeout(wait)),
        }
    }

    fn wait_for_policy_evaluation(&self, handle: &TransferHandle) -> bool {
        let mut status = handle.subscribe_status();
        self.wait_on_attached_runtime(async move {
            loop {
                if !matches!(*status.borrow(), TransferStatus::Evaluating) {
                    return true;
                }
                if status.changed().await.is_err() {
                    return false;
                }
            }
        })
    }

    fn wait_for_reservations_or_completion(
        &self,
        handle: &TransferHandle,
        target_reservation_count: u64,
        blocker: ReservationBlocker,
    ) -> bool {
        let reservation_notify = self.worker.reservation_notifier();
        let mut status = handle.subscribe_status();
        self.wait_on_attached_runtime(async move {
            loop {
                if handle.is_complete()
                    || self.worker.reservation_count() >= target_reservation_count
                {
                    return true;
                }
                // Active transfers mean this enqueue may be backpressured
                // behind the pipeline executor. Offline replay should advance
                // virtual time to that deadline, not spend wall time waiting.
                let blocked_by_active_transfer = match blocker {
                    ReservationBlocker::LocalOffload => {
                        self.worker.earliest_local_offload_finish().is_some()
                    }
                    ReservationBlocker::SharedG3Offload => {
                        self.worker.earliest_shared_g3_offload_finish().is_some()
                    }
                };
                if blocked_by_active_transfer {
                    return false;
                }
                tokio::select! {
                    _ = reservation_notify.notified() => {}
                    changed = status.changed() => {
                        if changed.is_err() {
                            return false;
                        }
                    }
                }
            }
        })
    }

    fn tier_registrations<T: BlockMetadata>(manager: &BlockManager<T>) -> u64 {
        manager.metrics().snapshot().registrations
    }

    fn wait_for_tier_registrations<T: BlockMetadata>(
        &self,
        manager: Arc<BlockManager<T>>,
        expected_registrations: u64,
    ) -> bool {
        self.wait_on_attached_runtime(async move {
            loop {
                let registrations = Self::tier_registrations(&manager);
                if registrations >= expected_registrations {
                    return true;
                }
                tokio::task::yield_now().await;
            }
        })
    }

    fn wait_for_tier_publish_blocks<T: BlockMetadata>(
        &self,
        manager: Arc<BlockManager<T>>,
        registrations_before: u64,
        drained_blocks: usize,
    ) -> (bool, u64) {
        if drained_blocks == 0 {
            return (true, Self::tier_registrations(&manager));
        }

        let expected =
            registrations_before.saturating_add(u64::try_from(drained_blocks).unwrap_or(u64::MAX));
        let published = self.wait_for_tier_registrations(manager.clone(), expected);
        (published, Self::tier_registrations(&manager))
    }

    fn pending_handles(&self, lane: TransferLane) -> Vec<TransferHandle> {
        match lane {
            TransferLane::G1ToG2 => {
                let pending = self
                    .pending_g1_to_g2
                    .lock()
                    .expect("pending G1→G2 handles mutex poisoned");
                pending
                    .iter()
                    .map(|pending| pending.handle.clone())
                    .collect()
            }
            TransferLane::G2ToG3 => {
                let pending = self
                    .pending_g2_to_g3
                    .lock()
                    .expect("pending G2→G3 handles mutex poisoned");
                pending
                    .iter()
                    .map(|pending| pending.handle.clone())
                    .collect()
            }
        }
    }

    fn settled_blocks(&self, lane: TransferLane) -> usize {
        self.pending_handles(lane)
            .iter()
            .map(|handle| handle.completed_blocks().len() + handle.failed_blocks().len())
            .sum()
    }

    fn wait_for_pending_settled_blocks(
        &self,
        lane: TransferLane,
        expected_settled_blocks: usize,
    ) -> bool {
        let handles = self.pending_handles(lane);
        self.wait_on_attached_runtime(async move {
            let mut completed: Vec<_> = handles
                .iter()
                .map(TransferHandle::subscribe_completed)
                .collect();
            let mut failed: Vec<_> = handles
                .iter()
                .map(TransferHandle::subscribe_failed)
                .collect();

            loop {
                let settled_blocks: usize = handles
                    .iter()
                    .map(|handle| handle.completed_blocks().len() + handle.failed_blocks().len())
                    .sum();
                if settled_blocks >= expected_settled_blocks {
                    return true;
                }
                if handles.is_empty() || handles.iter().all(TransferHandle::is_complete) {
                    return false;
                }

                let mut changes = FuturesUnordered::new();
                for rx in completed.iter_mut() {
                    changes.push(rx.changed());
                }
                for rx in failed.iter_mut() {
                    changes.push(rx.changed());
                }

                let mut observed_change = false;
                while let Some(changed) = changes.next().await {
                    if changed.is_ok() {
                        observed_change = true;
                        break;
                    }
                }
                if !observed_change {
                    return false;
                }
            }
        })
    }

    fn wait_for_find_result_completion(&self, result: &FindMatchesResult) -> bool {
        let wait = result.wait_for_completion();
        self.wait_on_attached_runtime(async move { wait.await.is_ok() })
    }

    fn completed_match_count(result: &FindMatchesResult) -> Option<usize> {
        match result.as_async()?.status() {
            OnboardingStatus::Complete { matched_blocks } => Some(matched_blocks),
            _ => None,
        }
    }

    fn wait_for_g3_staging_reservation_or_completion(
        &self,
        result: &FindMatchesResult,
        reservation_count_before: u64,
    ) -> bool {
        let reservation_notify = self.worker.reservation_notifier();
        let wait = result.wait_for_completion();
        self.wait_on_attached_runtime(async move {
            tokio::select! {
                reserved = async {
                    loop {
                        if self.worker.reservation_count() > reservation_count_before {
                            return true;
                        }
                        reservation_notify.notified().await;
                    }
                } => reserved,
                completed = wait => completed.is_ok(),
            }
        })
    }

    fn build_lower_tier_lookup_plan(
        &self,
        plhs: &[SequenceHash],
        g3_manager: &BlockManager<G3>,
    ) -> LowerTierLookupPlan {
        // This is a planning pass only. `check_presence` does not acquire
        // blocks, so the later G2-ready path must still call `match_blocks`
        // to pin the source blocks it will onboard.
        let g2_presence = self.g2_manager.block_registry().check_presence::<G2>(plhs);
        let g3_presence = g3_manager.block_registry().check_presence::<G3>(plhs);
        let g2_prefix_blocks = g2_presence.iter().take_while(|(_, in_g2)| *in_g2).count();
        let g3_stage_blocks = g3_presence
            .iter()
            .skip(g2_prefix_blocks)
            .take_while(|(_, in_g3)| *in_g3)
            .count();
        LowerTierLookupPlan {
            g2_prefix_blocks,
            g3_stage_blocks,
        }
    }

    fn cleanup_g2_to_g3_pending_handles(&self) {
        let Some(shared_g3) = self.shared_g3.as_ref() else {
            return;
        };

        // Successful G2->G3 completions release their shared reservation when
        // SharedG3Pool drains the global DES queue, regardless of which worker
        // advanced time. This owner-local pass prunes terminal handles and only
        // releases reservations for failed blocks that never produced a shared
        // completion.
        let mut pending = self
            .pending_g2_to_g3
            .lock()
            .expect("pending G2→G3 handles mutex poisoned");
        let mut failed_reservations = 0usize;
        for pending in pending.iter_mut() {
            failed_reservations += pending.take_unreleased_failed_reservations();
        }
        pending.retain(|pending| !pending.is_complete());
        drop(pending);

        if failed_reservations > 0 {
            shared_g3.release_capacity_reservations(failed_reservations);
        }
    }

    fn pump_pending_staged_swap_ins(&self, now_ms: f64) {
        // Waiting for a session while foreground transfers are active can stall
        // the virtual-time loop: the session may itself be waiting for a
        // G3→G2 transfer deadline. When no foreground transfer is active, a
        // bounded wait lets same-timestamp G3 staging publish G2 blocks before
        // the scheduler immediately retries admission.
        let should_wait_for_sessions = self.worker.earliest_foreground_finish().is_none();
        let pending = {
            let mut pending = self
                .pending_staged_swap_ins
                .lock()
                .expect("pending staged swap-ins mutex poisoned");
            pending.drain(..).collect::<Vec<_>>()
        };
        let mut keep = Vec::with_capacity(pending.len());

        for mut staged in pending {
            if !staged.g2_to_g1_started {
                let session_finished = should_wait_for_sessions
                    && self.wait_for_find_result_completion(&staged.result);
                let maybe_g2_blocks = staged.result.take_g2_blocks();
                if let Some(g2_blocks) = maybe_g2_blocks {
                    drop(staged.g2_capacity_reservation.take());
                    let block_count = g2_blocks.len();
                    staged.block_count.store(block_count, Ordering::Release);
                    tracing::trace!(
                        now_ms,
                        block_count,
                        "kvbm-offload: G3→G2 staging produced G2 blocks"
                    );
                    if block_count == 0 {
                        staged.complete.store(true, Ordering::Release);
                        staged.g2_to_g1_started = true;
                    } else {
                        tracing::trace!(
                            now_ms,
                            block_count,
                            "kvbm-offload: starting staged G2→G1 swap-in"
                        );
                        self.worker
                            .reserve_swap_in(now_ms, block_count, staged.complete.clone());
                        staged.g2_blocks = Some(g2_blocks);
                        staged.g2_to_g1_started = true;
                    }
                } else if session_finished {
                    let matched_blocks = Self::completed_match_count(&staged.result);
                    tracing::debug!(
                        now_ms,
                        reservation_blocks = staged.reservation_blocks,
                        matched_blocks,
                        status = ?staged.result.as_async().map(|session| session.status()),
                        "kvbm-offload: G3 staging session completed without available G2 blocks"
                    );
                    if matched_blocks.unwrap_or_default() > 0 {
                        tracing::debug!(
                            now_ms,
                            reservation_blocks = staged.reservation_blocks,
                            matched_blocks,
                            "kvbm-offload: G3 staging completed with matches but no G2 blocks; treating as 0-block swap-in"
                        );
                    }
                    drop(staged.g2_capacity_reservation.take());
                    staged.block_count.store(0, Ordering::Release);
                    staged.complete.store(true, Ordering::Release);
                    staged.g2_to_g1_started = true;
                }
            }

            if !staged.is_done() {
                keep.push(staged);
            }
        }

        let mut pending = self
            .pending_staged_swap_ins
            .lock()
            .expect("pending staged swap-ins mutex poisoned");
        pending.extend(keep);
    }

    /// Number of G1→G2 transfer batches that an idle pipeline can reserve
    /// immediately for this enqueue.
    ///
    /// Offline replay needs those immediate reservations to observe the
    /// enqueue's current virtual `now_ms`; otherwise the kvbm-engine task may
    /// first run after the scheduler advances time and stamp the transfer too
    /// late. We still do *not* wait for the whole burst: any batches beyond
    /// the pipeline's transfer slots are real queueing work and should start
    /// only after virtual time advances to an active-transfer deadline.
    fn initial_runnable_transfer_batches(&self, passed_blocks: usize) -> usize {
        if passed_blocks == 0 {
            return 0;
        }
        let transfer_batch_size = self.config.offload_batch_size.max(1);
        // The pipeline builder wires max_concurrent_transfers to the same
        // config knob as batch_size for this mocker-only G1→G2 pipeline.
        let max_concurrent_transfer_batches = self.config.offload_batch_size.max(1);
        passed_blocks
            .div_ceil(transfer_batch_size)
            .min(max_concurrent_transfer_batches)
    }

    /// Drop pending entries whose source slots are safe to release; the
    /// `Vec<MutableBlock<MockerG1>>` Drop returns the G1 slots to the pool.
    fn prune_releasable_g1_to_g2_sources(&self) {
        let mut pending = self
            .pending_g1_to_g2
            .lock()
            .expect("pending G1→G2 handles mutex poisoned");
        pending.retain(|pending| !pending.source_slots_releasable());
    }

    fn release_completed_g1_to_g2_sources(&self) -> usize {
        let mut released = 0usize;
        let mut pending = self
            .pending_g1_to_g2
            .lock()
            .expect("pending G1→G2 handles mutex poisoned");
        for pending in pending.iter_mut() {
            released += pending.release_completed_source_slots();
        }
        pending.retain(|pending| !pending.source_slots_releasable());
        released
    }

    fn collect_g2_to_g3_chain_blocks(&self) -> Vec<SequenceHash> {
        let mut chain_blocks = Vec::new();
        let mut pending = self
            .pending_g1_to_g2
            .lock()
            .expect("pending G1→G2 handles mutex poisoned");
        for pending in pending.iter_mut() {
            chain_blocks.extend(pending.collect_completed_chain_blocks());
        }
        pending.retain(|pending| !pending.source_slots_releasable());
        chain_blocks
    }

    fn enqueue_g2_to_g3_background(&self, hashes: Vec<SequenceHash>) {
        if hashes.is_empty() || self.g3_manager.is_none() {
            return;
        }

        let mut matches = self.g2_manager.scan_matches(&hashes, false);
        let blocks: Vec<_> = hashes
            .into_iter()
            .filter_map(|seq_hash| matches.remove(&seq_hash))
            .collect();
        if blocks.is_empty() {
            return;
        }

        // Pin the G2 source while the G2→G3 write-through is in flight. Even
        // though MockWorker only models timing, a real copy cannot let the
        // source block reset/reuse before G3 has consumed its bytes.
        let source: SourceBlocks<G2> = SourceBlocks::Strong(blocks);
        let reservation_count_before = self.worker.reservation_count();
        let Ok(handle) = self.engine.enqueue_g2_to_g3(source) else {
            return;
        };
        self.wait_for_policy_evaluation(&handle);
        if !handle.passed_blocks().is_empty() {
            let mut pending = self
                .pending_g2_to_g3
                .lock()
                .expect("pending G2→G3 handles mutex poisoned");
            pending.push(PendingG2ToG3 {
                handle: handle.clone(),
                released_failed_reservations: 0,
            });
            drop(pending);
            self.wait_for_reservations_or_completion(
                &handle,
                reservation_count_before + 1,
                ReservationBlocker::SharedG3Offload,
            );
        }
    }

    /// Advance PS state to `now_ms` and fire awaiters for any transfers
    /// that completed in the interval. Caller picks `now_ms`: live mode
    /// passes wall-clock (`Instant::elapsed`-derived); offline replay
    /// passes the runtime's virtual time. Hot-path logic is identical
    /// in both — only the source of `now_ms` differs.
    ///
    /// # Per-worker virtual-time containment
    ///
    /// This `tick` only advances PS models owned by *this*
    /// engine's `MockWorker`. With one engine per scheduler worker and
    /// G1↔G2 transfers physically scoped to that worker's CPU/host
    /// memory, the per-worker PS queue is the correct contention unit:
    /// concurrent offloads on different workers do not contend for the
    /// same DRAM bandwidth, and concurrent offloads on the same worker
    /// fair-share via `BandwidthSharingModel`'s PS math.
    ///
    /// G3 is modeled by the process-local shared pool hanging off the
    /// worker, so G2↔G3 transfers contend globally while G1↔G2 remains
    /// worker-local. Future shared tiers should follow the same pattern
    /// instead of adding another worker-local PS queue.
    pub fn tick(&self, now_ms: f64) {
        self.worker.set_now_ms(now_ms);
        let g2_registrations_before = Self::tier_registrations(&self.g2_manager);
        let g3_registrations_before = self
            .g3_manager
            .as_ref()
            .map(|manager| Self::tier_registrations(manager))
            .unwrap_or_default();
        let g1_to_g2_settled_before = self.settled_blocks(TransferLane::G1ToG2);
        let g2_to_g3_settled_before = self.settled_blocks(TransferLane::G2ToG3);
        let drained = self.worker.drain_completions_summary(now_ms);
        let offload_drained = drained.local.offload_transfers;
        let offload_drained_blocks = drained.local.offload_blocks;
        let shared_g3 = drained.shared_g3.counts;
        let current_shared_onboard_blocks = shared_g3
            .onboard_blocks
            .saturating_sub(drained.shared_g3.deferred_onboard_blocks);
        let g2_publish_blocks =
            offload_drained_blocks.saturating_add(current_shared_onboard_blocks);

        // Offline replay owns a private runtime for the kvbm-engine
        // pipeline. Once drain fires transfer awaiters, give those
        // pipeline tasks a chance to register completed destination blocks
        // before the scheduler immediately queries G2 in the same pass.
        if offload_drained > 0 {
            tracing::debug!(
                now_ms,
                transfers = offload_drained,
                blocks = offload_drained_blocks,
                "kvbm-offload: G1→G2 drained mock transfers"
            );
        }

        if offload_drained_blocks > 0 {
            self.g2_destination_reservations
                .release(offload_drained_blocks);
            let released_source_slots = self.release_completed_g1_to_g2_sources();
            tracing::debug!(
                now_ms,
                offload_drained_blocks,
                released_source_slots,
                "kvbm-offload: released G1 source slots for drained G1→G2 transfers"
            );
        }

        if g2_publish_blocks > 0 {
            let (published, registrations_after) = self.wait_for_tier_publish_blocks(
                self.g2_manager.clone(),
                g2_registrations_before,
                g2_publish_blocks,
            );
            if !published {
                tracing::warn!(
                    now_ms,
                    offload_drained,
                    offload_drained_blocks,
                    g3_to_g2_drained_blocks = current_shared_onboard_blocks,
                    deferred_g3_to_g2_drained_blocks = drained.shared_g3.deferred_onboard_blocks,
                    registrations_before = g2_registrations_before,
                    registrations_after,
                    "kvbm-offload: G2 registration barrier did not observe drained transfers"
                );
            }
        }
        if offload_drained_blocks > 0 {
            let expected_settled = g1_to_g2_settled_before.saturating_add(offload_drained_blocks);
            if !self.wait_for_pending_settled_blocks(TransferLane::G1ToG2, expected_settled) {
                tracing::debug!(
                    now_ms,
                    offload_drained_blocks,
                    expected_settled,
                    "kvbm-offload: G1→G2 handle progress not yet visible for G2→G3 chaining"
                );
            }
        }
        let g2_to_g3_chain_blocks = self.collect_g2_to_g3_chain_blocks();

        if !g2_to_g3_chain_blocks.is_empty() && self.g3_manager.is_some() {
            tracing::trace!(
                now_ms,
                blocks = g2_to_g3_chain_blocks.len(),
                "kvbm-offload: enqueue G2→G3 background copies"
            );
            self.enqueue_g2_to_g3_background(g2_to_g3_chain_blocks);
        }

        if let (Some(g3_manager), blocks @ 1..) =
            (self.g3_manager.as_ref(), shared_g3.offload_blocks)
        {
            let registration_baseline = drained
                .shared_g3
                .offload_registration_baseline
                .unwrap_or(g3_registrations_before);
            let (published, registrations_after) = self.wait_for_tier_publish_blocks(
                g3_manager.clone(),
                registration_baseline,
                blocks,
            );
            if !published {
                tracing::warn!(
                    now_ms,
                    drained_blocks = blocks,
                    registration_baseline,
                    registrations_before = g3_registrations_before,
                    registrations_after,
                    "kvbm-offload: G2→G3 pipeline did not publish drained transfers"
                );
            }
            let expected_settled = if registration_baseline < g3_registrations_before {
                self.settled_blocks(TransferLane::G2ToG3)
            } else {
                g2_to_g3_settled_before.saturating_add(blocks)
            };
            if !self.wait_for_pending_settled_blocks(TransferLane::G2ToG3, expected_settled) {
                tracing::debug!(
                    now_ms,
                    drained_blocks = blocks,
                    expected_settled,
                    "kvbm-offload: G2→G3 handle progress not yet visible after G3 registration"
                );
            }
        }
        self.cleanup_g2_to_g3_pending_handles();
        self.pump_pending_staged_swap_ins(now_ms);
    }

    /// Earliest transfer completion that can change offload-visible state.
    ///
    /// G2→G3 write-through copies are background from the scheduler's matching
    /// perspective, but they still pin G2 source blocks and hold shared G3
    /// reservations. Offline replay must therefore drain those completions at
    /// their DES timestamp, not at an arbitrary later arrival, otherwise G3 can
    /// artificially reduce effective G2 capacity even when the G2→G3 link is
    /// configured as instant.
    pub fn earliest_pending_deadline(&self) -> Option<f64> {
        self.worker.earliest_finish()
    }

    /// Enqueue a burst of G1→G2 evictions with router metadata that will be
    /// used to publish HostPinned-tier events when G2 lifecycle notifications
    /// arrive.
    pub(crate) fn enqueue_g1_evictions_with_metadata(
        &mut self,
        evicted: &[G2OffloadBlock],
        source_slots: Vec<MutableBlock<MockerG1>>,
        now_ms: Option<f64>,
    ) {
        if evicted.is_empty() {
            drop(source_slots);
            return;
        }
        self.remember_g2_event_metadata(evicted);
        let engine_blocks: Vec<_> = evicted
            .iter()
            .map(|block| (block.block_id, block.plh))
            .collect();
        self.enqueue_g1_evictions_holding_sources(&engine_blocks, source_slots, now_ms);
    }

    /// Enqueue a burst of G1→G2 evictions and hold the reset source slots
    /// until the simulated transfer reaches a terminal state.
    fn enqueue_g1_evictions_holding_sources(
        &mut self,
        evicted: &[(BlockId, SequenceHash)],
        source_slots: Vec<MutableBlock<MockerG1>>,
        now_ms: Option<f64>,
    ) {
        if evicted.is_empty() {
            return;
        }
        if let Some(ms) = now_ms {
            self.worker.set_now_ms(ms);
        }
        tracing::debug!(
            now_ms = self.worker.now_ms(),
            blocks = evicted.len(),
            "kvbm-offload: G1→G2 enqueue evictions"
        );
        let reservation_count_before = self.worker.reservation_count();
        let source: SourceBlocks<EngineG1> = SourceBlocks::External(
            evicted
                .iter()
                .map(|(block_id, seq_hash)| ExternalBlock::<EngineG1>::new(*block_id, *seq_hash))
                .collect(),
        );
        let handle = self
            .engine
            .enqueue_g1_to_g2(source)
            .expect("G1→G2 pipeline must be configured at engine construction");
        {
            let mut pending = self
                .pending_g1_to_g2
                .lock()
                .expect("pending G1→G2 handles mutex poisoned");
            pending.push(PendingG1ToG2 {
                handle: handle.clone(),
                g2_to_g3_chain_blocks: evicted.iter().copied().collect(),
                source_slots: source_slots
                    .into_iter()
                    .map(|slot| (slot.block_id(), slot))
                    .collect(),
            });
        }

        // Sync pump so policy and the first wave of batch reservations both
        // run on the current virtual `now_ms`, not a later scheduler tick.
        self.wait_for_policy_evaluation(&handle);
        let target_reservation_count = reservation_count_before
            + self.initial_runnable_transfer_batches(handle.passed_blocks().len()) as u64;
        if target_reservation_count > reservation_count_before {
            self.wait_for_reservations_or_completion(
                &handle,
                target_reservation_count,
                ReservationBlocker::LocalOffload,
            );
        }
        if handle.is_complete() {
            let g2_to_g3_chain_blocks = self.collect_g2_to_g3_chain_blocks();
            if !g2_to_g3_chain_blocks.is_empty() && self.g3_manager.is_some() {
                self.enqueue_g2_to_g3_background(g2_to_g3_chain_blocks);
            }
            self.prune_releasable_g1_to_g2_sources();
        }
    }

    /// Prepare the longest lower-tier prefix without reserving G2→G1 bandwidth.
    ///
    /// With G3 disabled this pins the currently available G2 prefix. With G3
    /// enabled it first does a presence-only G2/G3 planning pass. If the G3
    /// suffix can be staged into G2, it returns a deferred staging plan; if not,
    /// it falls back to pinning only the G2 prefix that is available right now.
    /// The caller must reserve destination G1 slots before passing the prepared
    /// lookup to [`start_onboard_prefix`](Self::start_onboard_prefix).
    pub(crate) fn prepare_onboard_prefix(
        &mut self,
        plhs: &[SequenceHash],
    ) -> Option<PreparedSwapIn> {
        if plhs.is_empty() {
            return None;
        }

        let Some(g3_manager) = self.g3_manager.as_ref() else {
            let g2_blocks = self.g2_manager.match_blocks(plhs);
            if g2_blocks.is_empty() {
                return None;
            }
            return Some(PreparedSwapIn::from_g2_blocks(plhs.len(), g2_blocks));
        };

        let lower_tier_plan = self.build_lower_tier_lookup_plan(plhs, g3_manager);
        if lower_tier_plan.g3_stage_blocks > 0 {
            let available_g2 = self.g2_manager.available_blocks();
            let required_g2 = lower_tier_plan.g3_stage_blocks;
            if self
                .g2_destination_reservations
                .try_reserve(available_g2, required_g2)
            {
                let g2_capacity_reservation = CapacityReservationGuard::new(
                    self.g2_destination_reservations.clone(),
                    required_g2,
                );
                return Some(PreparedSwapIn::from_staging_plan(
                    plhs.len(),
                    lower_tier_plan.reservation_blocks(),
                    Some(g2_capacity_reservation),
                    plhs[..lower_tier_plan.reservation_blocks()].to_vec(),
                ));
            }

            tracing::debug!(
                plhs_len = plhs.len(),
                g2_prefix_blocks = lower_tier_plan.g2_prefix_blocks,
                stage_blocks = lower_tier_plan.g3_stage_blocks,
                reservation_blocks = lower_tier_plan.reservation_blocks(),
                available_g2,
                required_g2,
                reserved_g2 = self.g2_destination_reservations.reserved_blocks(),
                "kvbm-offload: skipping G3 staging; insufficient G2 capacity"
            );
        }

        if lower_tier_plan.g2_prefix_blocks == 0 {
            tracing::debug!(
                plhs_len = plhs.len(),
                g3_stage_blocks = lower_tier_plan.g3_stage_blocks,
                "kvbm-offload: lower-tier lookup MISS"
            );
            return None;
        }

        let lookup_plhs = &plhs[..lower_tier_plan.g2_prefix_blocks];
        let g2_blocks = self.g2_manager.match_blocks(lookup_plhs);
        if g2_blocks.is_empty() {
            tracing::debug!(
                plhs_len = plhs.len(),
                lookup_blocks = lookup_plhs.len(),
                "kvbm-offload: lower-tier lookup MISS"
            );
            return None;
        }
        Some(PreparedSwapIn::from_g2_blocks(plhs.len(), g2_blocks))
    }

    /// Reserve G2→G1 bandwidth for a prepared lookup after the caller has
    /// already reserved destination G1 slots.
    pub(crate) fn start_onboard_prefix(
        &mut self,
        prepared: PreparedSwapIn,
        now_ms: Option<f64>,
    ) -> SwapInHandle {
        let now_ms = now_ms.unwrap_or_else(|| self.worker.now_ms());
        self.worker.set_now_ms(now_ms);
        match prepared {
            PreparedSwapIn::Ready {
                requested_blocks,
                g2_blocks,
            } => {
                let block_count = g2_blocks.len();
                tracing::debug!(
                    now_ms,
                    plhs_len = requested_blocks,
                    block_count,
                    "kvbm-offload: G2→G1 swap-in HIT"
                );
                let complete = Arc::new(AtomicBool::new(false));
                let block_count_cell = Arc::new(AtomicUsize::new(block_count));
                self.worker
                    .reserve_swap_in(now_ms, block_count, complete.clone());
                SwapInHandle {
                    complete,
                    block_count: block_count_cell,
                    _g2_blocks: Some(g2_blocks),
                }
            }
            PreparedSwapIn::Staging {
                requested_blocks,
                reservation_blocks,
                g2_capacity_reservation,
                lookup_plhs,
            } => {
                tracing::debug!(
                    now_ms,
                    plhs_len = requested_blocks,
                    reservation_blocks,
                    "kvbm-offload: G3→G2 staging swap-in HIT"
                );
                let complete = Arc::new(AtomicBool::new(false));
                let block_count = Arc::new(AtomicUsize::new(0));
                let reservation_count_before = self.worker.reservation_count();
                let result = self
                    .leader
                    .find_matches_with_options(
                        &lookup_plhs,
                        FindMatchesOptions {
                            search_remote: true,
                            staging_mode: StagingMode::Full,
                        },
                    )
                    .expect("find_matches_with_options must not fail for mocker G3 staging");
                let reserved_or_done = self.wait_for_g3_staging_reservation_or_completion(
                    &result,
                    reservation_count_before,
                );
                if !reserved_or_done {
                    tracing::debug!(
                        now_ms,
                        plhs_len = requested_blocks,
                        reservation_blocks,
                        "kvbm-offload: G3 staging session has not reserved transfer yet"
                    );
                }
                let mut pending = self
                    .pending_staged_swap_ins
                    .lock()
                    .expect("pending staged swap-ins mutex poisoned");
                pending.push(PendingStagedSwapIn {
                    result,
                    reservation_blocks,
                    complete: complete.clone(),
                    block_count: block_count.clone(),
                    g2_capacity_reservation,
                    g2_blocks: None,
                    g2_to_g1_started: false,
                });
                drop(pending);
                self.pump_pending_staged_swap_ins(now_ms);
                SwapInHandle {
                    complete,
                    block_count,
                    _g2_blocks: None,
                }
            }
        }
    }

    /// Test-only accessor: integration tests outside this module
    /// register synthetic G2 blocks (allocate → stage → register)
    /// through it so [`prepare_onboard_prefix`](Self::prepare_onboard_prefix)
    /// has something to match.
    #[cfg(test)]
    pub(crate) fn g2_manager(&self) -> &Arc<BlockManager<G2>> {
        &self.g2_manager
    }

    #[cfg(test)]
    pub(crate) fn g3_manager(&self) -> Option<&Arc<BlockManager<G3>>> {
        self.g3_manager.as_ref()
    }
}

impl Drop for MockOffloadEngine {
    fn drop(&mut self) {
        let Some(rt) = self._runtime.take() else {
            return;
        };
        if tokio::runtime::Handle::try_current().is_ok() {
            let _ = std::thread::spawn(move || drop(rt)).join();
        } else {
            drop(rt);
        }
    }
}

/// Local velo `Messenger` on a random TCP port. Avoids pulling in
/// kvbm-engine's `testing` feature for a ~20 line helper.
async fn create_local_messenger() -> Result<Arc<velo::Messenger>> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let transport: Arc<dyn velo::backend::Transport> = Arc::new(
        velo::backend::tcp::TcpTransportBuilder::new()
            .from_listener(listener)?
            .build()?,
    );
    let messenger = velo::Messenger::builder()
        .add_transport(transport)
        .build()
        .await?;
    // Velo's TCP accept loop needs a moment to reach Ready before it will
    // route the first message.
    tokio::time::sleep(Duration::from_millis(100)).await;
    Ok(messenger)
}

fn build_registry(events_manager: Arc<EventsManager>) -> BlockRegistry {
    BlockRegistry::builder()
        .frequency_tracker(FrequencyTrackingCapacity::Medium.create_tracker())
        .event_manager(events_manager)
        .build()
}

fn build_g2_block_manager(
    block_count: usize,
    block_size_tokens: usize,
    registry: &BlockRegistry,
) -> BlockManager<G2> {
    BlockManager::<G2>::builder()
        .block_count(block_count)
        .block_size(block_size_tokens)
        .registry(registry.clone())
        .duplication_policy(BlockDuplicationPolicy::Reject)
        .with_lineage_backend()
        .build()
        .expect("BlockManager<G2> should build with valid config")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kvbm_offload::shared_g3::{shared_g3_test_guard, shared_g3_test_guard_blocking};

    fn g3_config() -> KvbmOffloadConfig {
        KvbmOffloadConfig {
            num_g3_blocks: Some(1_024),
            block_size_bytes: Some(1_000_000),
            offload_batch_size: 4,
            bandwidth_g2_to_g1_gbps: 1.0,
            bandwidth_g2_to_g3_gbps: 1.0,
            bandwidth_g3_to_g2_gbps: 1.0,
            ..Default::default()
        }
    }

    fn single_thread_runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap()
    }

    fn register_test_block<T: BlockMetadata>(manager: &BlockManager<T>, plh: SequenceHash) {
        let (mut alloc, _evicted) = manager
            .allocate_blocks_with_evictions(1)
            .expect("allocate test block");
        let mutable = alloc.pop().unwrap();
        let complete = mutable
            .stage(plh, manager.block_size())
            .expect("stage test block");
        drop(manager.register_block(complete));
    }

    #[tokio::test]
    async fn mock_offload_engine_new_builds_end_to_end() {
        let config = KvbmOffloadConfig::default();
        let engine = MockOffloadEngine::new(config)
            .await
            .expect("construction should succeed");

        assert!(engine.engine.has_g1_to_g2());
        assert!(!engine.engine.has_g2_to_g3());
        assert!(!engine.engine.has_g2_to_g4());
        assert_eq!(engine.earliest_pending_deadline(), None);
    }

    #[tokio::test]
    async fn mock_offload_engine_new_builds_g3_pipeline_when_enabled() {
        let _guard = shared_g3_test_guard().await;
        let engine = MockOffloadEngine::new(g3_config())
            .await
            .expect("construction should succeed");

        assert!(engine.engine.has_g1_to_g2());
        assert!(engine.engine.has_g2_to_g3());
        assert!(engine.g3_manager().is_some());
        assert_eq!(engine.earliest_pending_deadline(), None);
    }

    #[tokio::test]
    async fn shared_g3_pool_is_visible_across_engines() {
        use dynamo_tokens::PositionalLineageHash;
        use kvbm_logical::MutableBlock;

        let _guard = shared_g3_test_guard().await;
        let config = g3_config();
        let engine_a = MockOffloadEngine::new(config.clone())
            .await
            .expect("engine A build");
        let mut engine_b = MockOffloadEngine::new(config)
            .await
            .expect("engine B build");

        let plh = PositionalLineageHash::new(126, None, 0);
        let g3_manager = engine_a.g3_manager().expect("G3 enabled").clone();
        let (mut alloc, _evicted) = g3_manager
            .allocate_blocks_with_evictions(1)
            .expect("G3 allocate");
        let mutable: MutableBlock<G3> = alloc.pop().unwrap();
        let complete = mutable
            .stage(plh, g3_manager.block_size())
            .expect("G3 stage");
        drop(g3_manager.register_block(complete));

        let prepared = engine_b
            .prepare_onboard_prefix(&[plh])
            .expect("worker B should see worker A's shared G3 block");
        assert_eq!(prepared.reservation_block_count(), 1);
    }

    #[test]
    fn shared_g3_capacity_reservations_span_engines() {
        use dynamo_tokens::PositionalLineageHash;

        let _guard = shared_g3_test_guard_blocking();
        let rt_a = single_thread_runtime();
        let rt_b = single_thread_runtime();
        let config = KvbmOffloadConfig {
            num_g3_blocks: Some(1),
            ..g3_config()
        };
        let mut engine_a = rt_a
            .block_on(MockOffloadEngine::new(config.clone()))
            .expect("engine A build");
        let mut engine_b = rt_b
            .block_on(MockOffloadEngine::new(config))
            .expect("engine B build");
        engine_a.attach_runtime(rt_a);
        engine_b.attach_runtime(rt_b);
        engine_a.tick(0.0);
        engine_b.tick(0.0);

        let plh_a = PositionalLineageHash::new(5000, None, 0);
        let plh_b = PositionalLineageHash::new(5001, None, 0);
        engine_a.enqueue_g1_evictions_holding_sources(&[(0, plh_a)], Vec::new(), Some(0.0));
        engine_b.enqueue_g1_evictions_holding_sources(&[(0, plh_b)], Vec::new(), Some(0.0));

        let first_deadline = engine_a
            .earliest_pending_deadline()
            .expect("engine A G1→G2 should reserve bandwidth");
        engine_a.tick(first_deadline);
        engine_b.tick(first_deadline);

        let g3_deadline = engine_a
            .worker
            .earliest_finish()
            .expect("engine A should own the one shared G3 transfer");
        engine_a.tick(g3_deadline);
        engine_b.tick(g3_deadline);

        let g3_manager = engine_a.g3_manager().expect("G3 enabled").clone();
        let presence = g3_manager
            .block_registry()
            .check_presence::<G3>(&[plh_a, plh_b]);
        assert!(presence[0].1, "first engine should fill the only G3 slot");
        assert!(
            !presence[1].1,
            "second engine must not over-admit into reserved shared G3 capacity"
        );
    }

    #[test]
    fn shared_g3_release_is_visible_to_same_time_non_owner_admission() {
        use dynamo_tokens::PositionalLineageHash;

        let _guard = shared_g3_test_guard_blocking();
        let rt_a = single_thread_runtime();
        let rt_b = single_thread_runtime();
        let config = KvbmOffloadConfig {
            // Two blocks lets the test isolate the reservation bug. After A's
            // copy completes, A legitimately occupies one real G3 block; B
            // should be able to reserve the other block once A's transient
            // reservation is released by the shared completion event.
            num_g3_blocks: Some(2),
            offload_batch_size: 1,
            ..g3_config()
        };
        let mut engine_a = rt_a
            .block_on(MockOffloadEngine::new(config.clone()))
            .expect("engine A build");
        let mut engine_b = rt_b
            .block_on(MockOffloadEngine::new(config))
            .expect("engine B build");
        engine_a.attach_runtime(rt_a);
        engine_b.attach_runtime(rt_b);
        engine_a.tick(0.0);
        engine_b.tick(0.0);

        let plh_a = PositionalLineageHash::new(5100, None, 0);
        let plh_b = PositionalLineageHash::new(5101, None, 0);
        register_test_block(engine_a.g2_manager(), plh_a);
        register_test_block(engine_b.g2_manager(), plh_b);

        let shared_reservations = engine_a
            .shared_g3
            .as_ref()
            .expect("G3 enabled")
            .capacity_reservations();

        // A starts a G2->G3 copy and reserves one shared G3 capacity slot.
        engine_a.enqueue_g2_to_g3_background(vec![plh_a]);
        assert_eq!(
            engine_a.pending_g2_to_g3.lock().unwrap().len(),
            1,
            "engine A should have one owner-local G2->G3 handle"
        );
        assert_eq!(
            shared_reservations.reserved_blocks(),
            1,
            "A's in-flight copy should hold one shared G3 reservation"
        );

        // B, not A, advances the shared G3 queue to A's completion time.
        let g3_deadline = engine_a
            .worker
            .earliest_finish()
            .expect("engine A should own the shared G3 transfer");
        engine_b.tick(g3_deadline);
        assert_eq!(
            engine_a.pending_g2_to_g3.lock().unwrap().len(),
            1,
            "worker B must not perform engine A's owner-local cleanup"
        );
        assert_eq!(
            shared_reservations.reserved_blocks(),
            0,
            "B's shared drain should release A's completed G2->G3 reservation"
        );

        // At the same timestamp, B should see the released reservation and
        // admit its own G2->G3 copy without waiting for A to tick.
        let pending_before = engine_b.pending_g2_to_g3.lock().unwrap().len();
        engine_b.enqueue_g2_to_g3_background(vec![plh_b]);
        let pending_after = engine_b.pending_g2_to_g3.lock().unwrap().len();
        assert_eq!(
            pending_after,
            pending_before + 1,
            "same-time G2->G3 admission should see capacity released by the shared drain"
        );
        assert_eq!(
            shared_reservations.reserved_blocks(),
            1,
            "B's same-time G2->G3 copy should reserve the remaining G3 block"
        );

        let second_deadline = engine_b
            .worker
            .earliest_finish()
            .expect("engine B's same-time G2->G3 transfer should reserve bandwidth");
        engine_b.tick(second_deadline);
        assert_eq!(
            shared_reservations.reserved_blocks(),
            0,
            "B's G2->G3 reservation should release when its shared completion drains"
        );

        let g3_manager = engine_a.g3_manager().expect("G3 enabled").clone();
        let presence = g3_manager.block_registry().check_presence::<G3>(&[plh_b]);
        assert!(
            presence[0].1,
            "engine B's same-time G2->G3 transfer should publish to shared G3"
        );
    }

    #[test]
    fn g1_to_g2_completion_feeds_g3_with_presence_policy() {
        use dynamo_tokens::PositionalLineageHash;

        let _guard = shared_g3_test_guard_blocking();
        let rt = single_thread_runtime();
        let mut engine = rt
            .block_on(MockOffloadEngine::new(g3_config()))
            .expect("engine build");
        engine.attach_runtime(rt);
        engine.tick(0.0);

        let plh = PositionalLineageHash::new(999, None, 0);
        engine.enqueue_g1_evictions_holding_sources(&[(0, plh)], Vec::new(), Some(0.0));

        let first_deadline = engine
            .earliest_pending_deadline()
            .expect("G1→G2 should reserve bandwidth");
        engine.tick(first_deadline);

        let second_deadline = engine
            .earliest_pending_deadline()
            .expect("G2→G3 background copy should create a capacity-release deadline");
        engine.tick(second_deadline);

        let g3_manager = engine.g3_manager().expect("G3 enabled").clone();
        let g3_presence = g3_manager.block_registry().check_presence::<G3>(&[plh]);
        assert!(g3_presence[0].1, "G2→G3 should register the block in G3");
        let matched = g3_manager.match_blocks(&[plh]);
        assert_eq!(
            matched.len(),
            1,
            "presence-only G2→G3 policy should offload first-seen G2 block"
        );
    }

    #[test]
    fn g2_to_g3_background_copy_pins_g2_source_until_transfer_completes() {
        use dynamo_tokens::PositionalLineageHash;

        let _guard = shared_g3_test_guard_blocking();
        let rt = single_thread_runtime();
        let config = KvbmOffloadConfig {
            num_g2_blocks: 1,
            bandwidth_g1_to_g2_gbps: 1.0,
            ..g3_config()
        };
        let mut engine = rt
            .block_on(MockOffloadEngine::new(config))
            .expect("engine build");
        engine.attach_runtime(rt);
        engine.tick(0.0);

        let plh = PositionalLineageHash::new(1_234, None, 0);
        engine.enqueue_g1_evictions_holding_sources(&[(0, plh)], Vec::new(), Some(0.0));

        let first_deadline = engine
            .earliest_pending_deadline()
            .expect("G1→G2 should reserve bandwidth");
        engine.tick(first_deadline);
        assert!(
            engine.worker.earliest_finish().is_some(),
            "G2→G3 background copy should be in flight"
        );
        assert!(
            engine.g2_manager.allocate_blocks(1).is_none(),
            "in-flight G2→G3 copy must pin the G2 source block"
        );
    }

    #[test]
    fn g3_staging_reserves_only_contiguous_lower_tier_prefix() {
        use dynamo_tokens::PositionalLineageHash;

        let _guard = shared_g3_test_guard_blocking();
        let rt = single_thread_runtime();
        let mut engine = rt
            .block_on(MockOffloadEngine::new(g3_config()))
            .expect("engine build");
        engine.attach_runtime(rt);
        engine.tick(0.0);

        let plhs: Vec<_> = (0..5)
            .map(|i| PositionalLineageHash::new(2_000 + i, None, i))
            .collect();
        register_test_block(engine.g2_manager(), plhs[0]);
        let g3_manager = engine.g3_manager().expect("G3 enabled").clone();
        register_test_block(&g3_manager, plhs[1]);
        register_test_block(&g3_manager, plhs[2]);
        // Leave plhs[3] absent: plhs[4] must not count past the first hole.
        register_test_block(&g3_manager, plhs[4]);

        let prepared = engine
            .prepare_onboard_prefix(&plhs)
            .expect("lower-tier prefix should match");
        assert_eq!(
            prepared.reservation_block_count(),
            3,
            "reserve only G2 prefix + contiguous G3 continuation"
        );
    }

    #[test]
    fn g3_staging_allows_full_contiguous_lower_tier_prefix() {
        use dynamo_tokens::PositionalLineageHash;

        let _guard = shared_g3_test_guard_blocking();
        let rt = single_thread_runtime();
        let config = KvbmOffloadConfig {
            offload_batch_size: 2,
            ..g3_config()
        };
        let mut engine = rt
            .block_on(MockOffloadEngine::new(config))
            .expect("engine build");
        engine.attach_runtime(rt);
        engine.tick(0.0);

        let plhs: Vec<_> = (0..5)
            .map(|i| PositionalLineageHash::new(4_000 + i, None, i))
            .collect();
        let g3_manager = engine.g3_manager().expect("G3 enabled").clone();
        for plh in &plhs {
            register_test_block(&g3_manager, *plh);
        }

        let prepared = engine
            .prepare_onboard_prefix(&plhs)
            .expect("full contiguous G3 prefix should be staged");
        assert_eq!(
            prepared.reservation_block_count(),
            plhs.len(),
            "foreground G3 staging should not be capped by offload batch size"
        );
        assert_eq!(
            engine.earliest_pending_deadline(),
            None,
            "G3 staging must wait until start_onboard_prefix reserves G1 destinations"
        );
    }

    #[test]
    fn g3_capacity_skip_returns_g2_prefix_without_local_g3_session() {
        use dynamo_tokens::PositionalLineageHash;

        let _guard = shared_g3_test_guard_blocking();
        let rt = single_thread_runtime();
        let config = KvbmOffloadConfig {
            num_g2_blocks: 1,
            ..g3_config()
        };
        let mut engine = rt
            .block_on(MockOffloadEngine::new(config))
            .expect("engine build");
        engine.attach_runtime(rt);
        engine.tick(0.0);

        let plhs: Vec<_> = (0..3)
            .map(|i| PositionalLineageHash::new(3_000 + i, None, i))
            .collect();
        register_test_block(engine.g2_manager(), plhs[0]);
        let g3_manager = engine.g3_manager().expect("G3 enabled").clone();
        register_test_block(&g3_manager, plhs[1]);
        register_test_block(&g3_manager, plhs[2]);

        let prepared = engine
            .prepare_onboard_prefix(&plhs)
            .expect("G2 prefix should still be usable when G3 staging lacks capacity");
        assert_eq!(
            prepared.reservation_block_count(),
            1,
            "G3 capacity preflight should not invoke local-only G3 session fallback"
        );
        assert!(
            matches!(prepared, PreparedSwapIn::Ready { .. }),
            "capacity-disabled G3 lookup should degrade to G2-only"
        );
    }

    #[tokio::test]
    async fn tick_is_noop_when_idle() {
        let engine = MockOffloadEngine::new(KvbmOffloadConfig::default())
            .await
            .unwrap();
        engine.tick(100.0);
        engine.tick(1_000_000.0);
        assert_eq!(engine.earliest_pending_deadline(), None);
    }

    #[test]
    fn offline_runtime_attach_pattern() {
        let rt = single_thread_runtime();
        let mut engine = rt
            .block_on(MockOffloadEngine::new(KvbmOffloadConfig::default()))
            .expect("offline construction succeeds");
        engine.attach_runtime(rt);
        assert_eq!(engine.earliest_pending_deadline(), None);
    }

    #[tokio::test]
    async fn prepare_onboard_prefix_empty_input_returns_none() {
        let mut engine = MockOffloadEngine::new(KvbmOffloadConfig::default())
            .await
            .unwrap();
        assert!(engine.prepare_onboard_prefix(&[]).is_none());
    }

    #[tokio::test]
    async fn prepare_onboard_prefix_returns_none_when_g2_empty() {
        use dynamo_tokens::PositionalLineageHash;
        let mut engine = MockOffloadEngine::new(KvbmOffloadConfig::default())
            .await
            .unwrap();
        let hashes: Vec<SequenceHash> = (0..5)
            .map(|i| PositionalLineageHash::new(i as u64, None, 0))
            .collect();
        assert!(engine.prepare_onboard_prefix(&hashes).is_none());
    }

    /// End-to-end: register a G2 block directly in the engine's
    /// `g2_manager`, call the prepare/start swap-in path, and verify (a) the
    /// handle is produced, (b) the reservation is reflected in
    /// `earliest_pending_deadline`, (c) `tick` past the finish time
    /// flips the completion bit, and (d) the handle pins the G2 block
    /// via RAII.
    #[tokio::test]
    async fn start_onboard_prefix_pins_g2_blocks_until_handle_drops() {
        use dynamo_tokens::PositionalLineageHash;
        use kvbm_logical::MutableBlock;
        let config = KvbmOffloadConfig {
            block_size_bytes: Some(1_000_000),
            bandwidth_g2_to_g1_gbps: 1.0,
            ..Default::default()
        };
        let mut engine = MockOffloadEngine::new(config).await.unwrap();
        engine.tick(0.0);

        // Register a G2 block. Allocate, stage with a PLH, register; let
        // the returned ImmutableBlock drop so the block lands in the
        // inactive pool (still matchable via find_matches).
        let plh = PositionalLineageHash::new(42, None, 0);
        let (mut alloc, _evicted) = engine
            .g2_manager
            .allocate_blocks_with_evictions(1)
            .expect("G2 allocate");
        let mutable: MutableBlock<G2> = alloc.pop().unwrap();
        let complete = mutable
            .stage(plh, engine.g2_manager.block_size())
            .expect("G2 stage");
        drop(engine.g2_manager.register_block(complete));

        let prepared = engine
            .prepare_onboard_prefix(&[plh])
            .expect("G2 prefix match must produce a prepared swap-in");
        let handle = engine.start_onboard_prefix(prepared, Some(0.0));
        assert_eq!(handle.block_count(), 1);
        assert!(!handle.is_complete());

        let deadline = engine
            .earliest_pending_deadline()
            .expect("swap-in must appear in earliest_finish");
        assert!(
            (deadline - 1.0).abs() < 1e-6,
            "1 MB / 1 GB/s = 1.0 ms, got {deadline}"
        );

        engine.tick(0.5);
        assert!(
            !handle.is_complete(),
            "swap-in must remain pending before finish time"
        );
        engine.tick(1.0);
        assert!(
            handle.is_complete(),
            "swap-in bit must flip once tick advances past finish"
        );
    }

    #[test]
    fn staged_g3_swap_in_runs_g3_to_g2_before_g2_to_g1() {
        use dynamo_tokens::PositionalLineageHash;
        use kvbm_logical::MutableBlock;

        let _guard = shared_g3_test_guard_blocking();
        let rt = single_thread_runtime();
        let mut engine = rt
            .block_on(MockOffloadEngine::new(g3_config()))
            .expect("engine build");
        engine.attach_runtime(rt);
        engine.tick(0.0);

        let plh = PositionalLineageHash::new(84, None, 0);
        let g3_manager = engine.g3_manager().expect("G3 enabled").clone();
        let (mut alloc, _evicted) = g3_manager
            .allocate_blocks_with_evictions(1)
            .expect("G3 allocate");
        let mutable: MutableBlock<G3> = alloc.pop().unwrap();
        let complete = mutable
            .stage(plh, g3_manager.block_size())
            .expect("G3 stage");
        drop(g3_manager.register_block(complete));

        let prepared = engine
            .prepare_onboard_prefix(&[plh])
            .expect("G3 prefix match must produce a staged swap-in");
        assert_eq!(prepared.reservation_block_count(), 1);
        let handle = engine.start_onboard_prefix(prepared, Some(0.0));
        assert!(!handle.is_complete());

        let first_deadline = engine
            .earliest_pending_deadline()
            .expect("G3→G2 staging should reserve shared bandwidth");
        assert!(
            (first_deadline - 1.0).abs() < 1e-6,
            "1 MB / 1 GB/s G3→G2 should finish at 1 ms, got {first_deadline}"
        );

        engine.tick(first_deadline);
        assert!(
            !handle.is_complete(),
            "G2→G1 transfer should start after G3 staging, not complete immediately"
        );
        assert_eq!(handle.block_count(), 1);
        let second_deadline = engine
            .earliest_pending_deadline()
            .expect("G2→G1 onboard should reserve bandwidth after staging");
        assert!(
            (second_deadline - 2.0).abs() < 1e-6,
            "second 1 MB / 1 GB/s hop should finish at 2 ms, got {second_deadline}"
        );

        engine.tick(second_deadline);
        assert!(handle.is_complete());
    }
}
