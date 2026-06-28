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
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use anyhow::Result;
use dynamo_tokens::{BlockHash, SequenceHash as RouterSequenceHash};
use futures::Stream;
use futures::task::noop_waker_ref;

use kvbm_engine::leader::{
    FindMatchesOptions, FindMatchesResult, InstanceLeader, Leader, OnboardingStatus, StagingMode,
};
use kvbm_engine::object::ObjectBlockOps;
use kvbm_engine::offload::{
    ExternalBlock, ObjectPipelineBuilder, ObjectPresenceFilter, OffloadEngine, PendingTracker,
    PipelineBuilder, PipelineLane, PresenceFilter, S3PresenceChecker, SettlementTarget,
    SettlementToken, SourceBlocks, TransferHandle, TransferStatus,
};
use kvbm_engine::worker::Worker;
use kvbm_engine::{BlockId, G1 as EngineG1, G2, G3, SequenceHash};
use kvbm_logical::blocks::{BlockMetadata, ImmutableBlock, MutableBlock};
use kvbm_logical::events::{EventsManager, KvCacheEvent as LogicalKvCacheEvent};
use kvbm_logical::manager::{BlockManager, FrequencyTrackingCapacity};
use kvbm_logical::pools::BlockDuplicationPolicy;
use kvbm_logical::registry::BlockRegistry;
use rustc_hash::FxHashMap;
use tokio::sync::watch;

use crate::common::protocols::G1 as MockerG1;

use super::capacity_reservation::{
    CapacityReservationGuard, CapacityReservationPolicy, CapacityReservations,
};
use super::config::KvbmOffloadConfig;
use super::coordinator::{
    DirectSwapInLease, G1ToG2Lease, G2ToG3Lease, G2ToG4Lease, LeaseState, OffloadCoordinator,
    OffloadId, StagedSwapInLease, SwapInHandle, SwapInResources, SwapInStatus, SwapInTerminal,
};
use super::shared_g3::SharedG3Pool;
use super::shared_g4::SharedG4Store;
use super::worker::MockWorker;

// Successful offline barriers wake via watch channels. The timeout is only a
// hang guard for pipeline bugs.
const PIPELINE_BARRIER_TIMEOUT: Duration = Duration::from_secs(1);

#[derive(Clone, Copy)]
enum ReservationBlocker {
    LocalOffload,
    SharedG3Offload,
    SharedG4Offload,
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
    g4_stage_blocks: usize,
}

impl LowerTierLookupPlan {
    fn reservation_blocks(self) -> usize {
        self.g2_prefix_blocks
            .saturating_add(self.g3_stage_blocks)
            .saturating_add(self.g4_stage_blocks)
    }

    fn stage_blocks(self) -> usize {
        self.g3_stage_blocks.saturating_add(self.g4_stage_blocks)
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
    pub(crate) local_hash: Option<BlockHash>,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AdvanceAcknowledgement(u64);

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum AdvanceAcknowledgementError {
    NoPreparedAdvance,
    Stale {
        expected: AdvanceAcknowledgement,
        actual: AdvanceAcknowledgement,
    },
}

#[derive(Clone)]
pub(crate) struct PreparedAdvance {
    pub(crate) router_events: Vec<G2RouterEvent>,
    pub(crate) acknowledgement: AdvanceAcknowledgement,
}

#[derive(Default)]
struct AdvanceState {
    next_id: u64,
    pending: Option<PreparedAdvance>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum G1EvictionOutcome {
    BlockedOnOffload {
        offload_id: OffloadId,
        deadline_ms: Option<f64>,
    },
    RetryNow {
        released_slots: usize,
    },
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
    shared_g4: Option<Arc<SharedG4Store>>,
    coordinator: OffloadCoordinator,
    g2_event_stream: Mutex<Pin<Box<dyn Stream<Item = LogicalKvCacheEvent> + Send>>>,
    g2_event_metadata: Mutex<FxHashMap<SequenceHash, G2BlockEventMetadata>>,
    pending_router_events: Mutex<Vec<G2RouterEvent>>,
    advance_state: Mutex<AdvanceState>,

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
        let shared_g4 = SharedG4Store::get_or_create(&config)?;
        tracing::debug!(
            num_g2_blocks = config.num_g2_blocks,
            num_g3_blocks = config.num_g3_blocks,
            g3_enabled = g3_manager.is_some(),
            g4_enabled = shared_g4.is_some(),
            "kvbm-offload: building mock offload engine"
        );

        let worker = Arc::new(MockWorker::new(
            config.block_size_bytes.unwrap_or(0),
            config.bandwidth_g1_to_g2_gbps,
            config.bandwidth_g2_to_g1_gbps,
            None,
            None,
            shared_g3.clone(),
            shared_g4.clone(),
        ));
        let worker_for_leader: Arc<dyn Worker> = worker.clone();
        let object_ops: Option<Arc<dyn ObjectBlockOps>> = shared_g4
            .as_ref()
            .map(|_| worker.clone() as Arc<dyn ObjectBlockOps>);

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
        if let Some(object_ops) = &object_ops {
            leader_builder = leader_builder.object_client(object_ops.clone());
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

        if let (Some(shared_g4), Some(object_ops)) = (&shared_g4, &object_ops) {
            let g2_to_g4_pending = shared_g4.pending_tracker();
            let g2_to_g4_presence = ObjectPresenceFilter::<G2>::new(Arc::new(
                S3PresenceChecker::new(object_ops.clone()),
            ))
            .with_pending_tracker(g2_to_g4_pending.clone());
            let g2_to_g4_pipeline = ObjectPipelineBuilder::<G2>::new()
                .policy(Arc::new(g2_to_g4_presence))
                .pending_tracker(g2_to_g4_pending)
                .batch_size(config.offload_batch_size)
                // Let object writes overlap; the shared G4 PS queue is the
                // throughput limiter. The same cap as batch size bounds
                // source pinning and spawned transfer tasks.
                .max_concurrent_transfers(config.offload_batch_size)
                .build();
            engine_builder = engine_builder
                .with_object_ops(object_ops.clone())
                .with_g2_to_g4_pipeline(g2_to_g4_pipeline);
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
            shared_g4,
            coordinator: OffloadCoordinator::new(),
            g2_event_stream: Mutex::new(g2_event_stream),
            g2_event_metadata: Mutex::new(FxHashMap::default()),
            pending_router_events: Mutex::new(Vec::new()),
            advance_state: Mutex::new(AdvanceState::default()),
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
    fn collect_g2_router_events(&self) -> Vec<G2RouterEvent> {
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

    pub(crate) fn drain_g2_router_events(&self) -> Vec<G2RouterEvent> {
        std::mem::take(
            &mut *self
                .pending_router_events
                .lock()
                .expect("pending router events mutex poisoned"),
        )
    }

    async fn with_barrier_timeout<F>(wait: F) -> bool
    where
        F: Future<Output = bool>,
    {
        tokio::time::timeout(PIPELINE_BARRIER_TIMEOUT, wait)
            .await
            .unwrap_or_default()
    }

    async fn with_barrier_timeout_value<F, T>(wait: F) -> Option<T>
    where
        F: Future<Output = T>,
    {
        tokio::time::timeout(PIPELINE_BARRIER_TIMEOUT, wait)
            .await
            .ok()
    }

    fn wait_on_attached_runtime<F>(&self, wait: F) -> bool
    where
        F: Future<Output = bool>,
    {
        let current = tokio::runtime::Handle::try_current().ok();
        let Some(runtime) = self
            ._runtime
            .as_ref()
            .map(tokio::runtime::Runtime::handle)
            .or(current.as_ref())
        else {
            return false;
        };
        match current.as_ref().map(tokio::runtime::Handle::runtime_flavor) {
            Some(tokio::runtime::RuntimeFlavor::MultiThread) => {
                tokio::task::block_in_place(|| runtime.block_on(Self::with_barrier_timeout(wait)))
            }
            // Starting a runtime from inside a current-thread runtime would
            // panic. Tests in that shape can still make progress on the next
            // explicit tick.
            Some(tokio::runtime::RuntimeFlavor::CurrentThread) => false,
            None => runtime.block_on(Self::with_barrier_timeout(wait)),
            _ => unreachable!("Tokio runtime flavor is non-exhaustive"),
        }
    }

    fn wait_on_attached_runtime_value<F, T>(&self, wait: F) -> Option<T>
    where
        F: Future<Output = T>,
    {
        let current = tokio::runtime::Handle::try_current().ok();
        let runtime = self
            ._runtime
            .as_ref()
            .map(tokio::runtime::Runtime::handle)
            .or(current.as_ref())?;
        match current.as_ref().map(tokio::runtime::Handle::runtime_flavor) {
            Some(tokio::runtime::RuntimeFlavor::MultiThread) => tokio::task::block_in_place(|| {
                runtime.block_on(Self::with_barrier_timeout_value(wait))
            }),
            Some(tokio::runtime::RuntimeFlavor::CurrentThread) => None,
            None => runtime.block_on(Self::with_barrier_timeout_value(wait)),
            _ => unreachable!("Tokio runtime flavor is non-exhaustive"),
        }
    }

    async fn with_settlement_timeout<F>(wait: F) -> Result<(), String>
    where
        F: Future<Output = Result<(), kvbm_engine::offload::SettlementError>>,
    {
        match tokio::time::timeout(PIPELINE_BARRIER_TIMEOUT, wait).await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(error)) => Err(error.to_string()),
            Err(_) => Err(format!(
                "pipeline settlement timed out after {:?}",
                PIPELINE_BARRIER_TIMEOUT
            )),
        }
    }

    fn settle_or_panic(&self, token: SettlementToken, target: SettlementTarget) {
        let current = tokio::runtime::Handle::try_current().ok();
        let runtime = self
            ._runtime
            .as_ref()
            .map(tokio::runtime::Runtime::handle)
            .or(current.as_ref())
            .expect("pipeline settlement requires a Tokio runtime");
        let wait = Self::with_settlement_timeout(self.engine.settle_after(token, target));
        let result = match current.as_ref().map(tokio::runtime::Handle::runtime_flavor) {
            Some(tokio::runtime::RuntimeFlavor::MultiThread) => {
                tokio::task::block_in_place(|| runtime.block_on(wait))
            }
            Some(tokio::runtime::RuntimeFlavor::CurrentThread) => {
                panic!("pipeline settlement cannot block a current-thread Tokio runtime")
            }
            None => runtime.block_on(wait),
            _ => unreachable!("Tokio runtime flavor is non-exhaustive"),
        };
        if let Err(error) = result {
            panic!("KVBM pipeline settlement failed: {error}");
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
        let mut reservation_count = self.worker.subscribe_reservation_count();
        let mut status = handle.subscribe_status();
        self.wait_on_attached_runtime(async move {
            loop {
                if handle.is_complete() || *reservation_count.borrow() >= target_reservation_count {
                    return true;
                }
                // Active transfers mean this enqueue may be backpressured
                // behind the pipeline executor. Offline replay should advance
                // virtual time to that deadline, not spend wall time waiting.
                let blocked_by_active_transfer = match blocker {
                    ReservationBlocker::LocalOffload => {
                        self.worker.local_offload_active_count()
                            >= self.config.offload_batch_size.max(1)
                    }
                    ReservationBlocker::SharedG3Offload => {
                        self.worker.earliest_shared_g3_offload_finish().is_some()
                    }
                    ReservationBlocker::SharedG4Offload => {
                        self.worker.earliest_shared_g4_offload_finish().is_some()
                    }
                };
                if blocked_by_active_transfer {
                    return false;
                }
                tokio::select! {
                    changed = reservation_count.changed() => {
                        if changed.is_err() {
                            return false;
                        }
                    }
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

    fn completed_match_count(result: &FindMatchesResult) -> Option<usize> {
        match result.as_async()?.status() {
            OnboardingStatus::Complete { matched_blocks } => Some(matched_blocks),
            _ => None,
        }
    }

    fn wait_for_staging_reservation_or_completion(
        &self,
        result: &FindMatchesResult,
        reservation_count_before: u64,
    ) -> bool {
        let mut reservation_count = self.worker.subscribe_reservation_count();
        let wait = result.wait_for_completion();
        self.wait_on_attached_runtime(async move {
            tokio::select! {
                reserved = async {
                    loop {
                        if *reservation_count.borrow() > reservation_count_before {
                            return true;
                        }
                        if reservation_count.changed().await.is_err() {
                            return false;
                        }
                    }
                } => reserved,
                completed = wait => completed.is_ok(),
            }
        })
    }

    fn build_lower_tier_lookup_plan(&self, plhs: &[SequenceHash]) -> LowerTierLookupPlan {
        // This is a planning pass only. `check_presence` does not acquire
        // blocks, so the later G2-ready path must still call `match_blocks`
        // to pin the source blocks it will onboard.
        // This local mocker planner is tier-prioritized: contiguous G2
        // blocks, then one contiguous G3 run, then one contiguous G4 run. It
        // does not interleave lower tiers when, for example, a G4 hit appears
        // before a later G3 hit.
        let g2_presence = self.g2_manager.block_registry().check_presence::<G2>(plhs);
        let g2_prefix_blocks = g2_presence.iter().take_while(|(_, in_g2)| *in_g2).count();

        let mut offset = g2_prefix_blocks;
        let g3_stage_blocks = self
            .g3_manager
            .as_ref()
            .map(|g3_manager| {
                let g3_presence = g3_manager.block_registry().check_presence::<G3>(plhs);
                g3_presence
                    .iter()
                    .skip(offset)
                    .take_while(|(_, in_g3)| *in_g3)
                    .count()
            })
            .unwrap_or_default();
        offset = offset.saturating_add(g3_stage_blocks);

        let g4_stage_blocks = self
            .shared_g4
            .as_ref()
            .map(|shared_g4| {
                plhs.iter()
                    .skip(offset)
                    .take_while(|hash| shared_g4.has_object(hash).is_some())
                    .count()
            })
            .unwrap_or_default();
        LowerTierLookupPlan {
            g2_prefix_blocks,
            g3_stage_blocks,
            g4_stage_blocks,
        }
    }

    fn pump_pending_staged_swap_ins(&self, now_ms: f64) {
        // Waiting for a session while foreground transfers are active can stall
        // the virtual-time loop: the session may itself be waiting for a
        // lower-tier transfer deadline. When no foreground transfer is active, a
        // bounded wait lets same-timestamp staging publish G2 blocks before
        // the scheduler immediately retries admission.
        let should_wait_for_sessions = self.worker.earliest_foreground_finish().is_none();
        for id in self.coordinator.staged_swap_in_ids() {
            #[cfg(test)]
            if let Some(context) = self.worker.take_injected_staging_failure() {
                self.coordinator.fail_swap_in(id, context);
                continue;
            }
            let wait = self
                .coordinator
                .with_staged_swap_in_mut(id, |lease, _| {
                    (!lease.g2_to_g1_started).then(|| lease.result.wait_for_completion())
                })
                .flatten();
            let session_result =
                should_wait_for_sessions
                    .then(|| wait)
                    .flatten()
                    .and_then(|wait| {
                        self.wait_on_attached_runtime_value(async move {
                            wait.await
                                .map_err(|error| Arc::<str>::from(error.to_string()))
                        })
                    });
            if let Some(Err(context)) = session_result {
                self.coordinator.fail_swap_in(id, context);
                continue;
            }
            let session_finished = matches!(session_result, Some(Ok(())));

            let onboard = self
                .coordinator
                .with_staged_swap_in_mut(id, |staged, lease_state| {
                    if staged.g2_to_g1_started {
                        return None;
                    }
                    let maybe_g2_blocks = staged.result.take_g2_blocks();
                    if let Some(g2_blocks) = maybe_g2_blocks {
                        drop(staged.g2_capacity_reservation.take());
                        let block_count = g2_blocks.len();
                        tracing::trace!(
                            now_ms,
                            block_count,
                            "kvbm-offload: lower-tier staging produced G2 blocks"
                        );
                        staged.resources.g2_blocks = Some(g2_blocks);
                        staged.g2_to_g1_started = true;
                        if lease_state == LeaseState::CancelRequested {
                            staged.resources.status_tx.send_replace(SwapInStatus {
                                terminal: SwapInTerminal::Cancelled,
                                block_count: 0,
                            });
                            return None;
                        }
                        if block_count == 0 {
                            staged
                                .resources
                                .status_tx
                                .send_replace(SwapInStatus::completed(0));
                            return None;
                        }
                        tracing::trace!(
                            now_ms,
                            block_count,
                            "kvbm-offload: starting staged G2→G1 swap-in"
                        );
                        staged
                            .resources
                            .status_tx
                            .send_replace(SwapInStatus::pending(block_count));
                        return Some((block_count, staged.resources.status_tx.clone()));
                    }
                    if session_finished {
                        let matched_blocks = Self::completed_match_count(&staged.result);
                        tracing::debug!(
                            now_ms,
                            reservation_blocks = staged.reservation_blocks,
                            matched_blocks,
                            status = ?staged.result.as_async().map(|session| session.status()),
                            "kvbm-offload: lower-tier staging session completed without available G2 blocks"
                        );
                        drop(staged.g2_capacity_reservation.take());
                        staged.g2_to_g1_started = true;
                        staged.resources.status_tx.send_replace(
                            if lease_state == LeaseState::CancelRequested {
                                SwapInStatus {
                                    terminal: SwapInTerminal::Cancelled,
                                    block_count: 0,
                                }
                            } else {
                                SwapInStatus::completed(0)
                            },
                        );
                    }
                    None
                })
                .flatten();
            if let Some((block_count, status_tx)) = onboard {
                self.worker.reserve_swap_in(now_ms, block_count, status_tx);
            }
        }
    }

    /// Number of G1→G2 transfer batches that an idle pipeline can reserve
    /// immediately for this enqueue.
    ///
    /// Offline replay needs those immediate reservations to observe the
    /// enqueue's current virtual `now_ms`; otherwise the kvbm-engine task may
    /// first run after the scheduler advances time and stamp the transfer too
    /// late. Subsequent permit handoffs are covered by pipeline settlement.
    fn initial_runnable_transfer_batches(&self, passed_blocks: usize) -> usize {
        if passed_blocks == 0 {
            return 0;
        }
        let transfer_batch_size = self.config.offload_batch_size.max(1);
        // The pipeline builder wires max_concurrent_transfers to the same
        // config knob as batch_size for this mocker-only G1→G2 pipeline.
        let max_concurrent_transfer_batches = self
            .config
            .offload_batch_size
            .max(1)
            .saturating_sub(self.worker.local_offload_active_count());
        passed_blocks
            .div_ceil(transfer_batch_size)
            .min(max_concurrent_transfer_batches)
    }

    fn external_g2_source_for_hashes(&self, hashes: Vec<SequenceHash>) -> Option<SourceBlocks<G2>> {
        let mut matches = self.g2_manager.scan_matches(&hashes, false);
        let blocks: Vec<_> = hashes
            .into_iter()
            .filter_map(|seq_hash| matches.remove(&seq_hash))
            .collect();
        if blocks.is_empty() {
            return None;
        }
        Some(SourceBlocks::Strong(blocks))
    }

    fn enqueue_g2_to_g3_background(&self, hashes: Vec<SequenceHash>) -> Result<()> {
        if hashes.is_empty() || self.g3_manager.is_none() {
            return Ok(());
        }

        let Some(source) = self.external_g2_source_for_hashes(hashes) else {
            anyhow::bail!("G2→G3 chain source disappeared before child registration");
        };
        self.enqueue_g2_to_g3_background_source(source)
    }

    fn enqueue_g2_to_g3_background_source(&self, source: SourceBlocks<G2>) -> Result<()> {
        let reservation_count_before = self.worker.reservation_count();
        self.coordinator
            .begin_shared_lane(PipelineLane::G2ToG3, self.engine.settlement_token());
        let handle = self.engine.enqueue_g2_to_g3(source)?;
        self.coordinator
            .insert_g2_to_g3(G2ToG3Lease::new(handle.clone()));
        self.wait_for_policy_evaluation(&handle);
        if !handle.is_complete() {
            self.wait_for_reservations_or_completion(
                &handle,
                reservation_count_before + 1,
                ReservationBlocker::SharedG3Offload,
            );
        }
        Ok(())
    }

    fn enqueue_g2_to_g4_background(&self, hashes: Vec<SequenceHash>) -> Result<()> {
        if hashes.is_empty() || self.shared_g4.is_none() {
            return Ok(());
        }

        let Some(source) = self.external_g2_source_for_hashes(hashes) else {
            anyhow::bail!("G2→G4 chain source disappeared before child registration");
        };
        self.enqueue_g2_to_g4_background_source(source)
    }

    fn enqueue_g2_to_g4_background_source(&self, source: SourceBlocks<G2>) -> Result<()> {
        let reservation_count_before = self.worker.reservation_count();
        self.coordinator
            .begin_shared_lane(PipelineLane::G2ToG4, self.engine.settlement_token());
        let handle = self.engine.enqueue_g2_to_g4(source)?;
        self.coordinator
            .insert_g2_to_g4(G2ToG4Lease::new(handle.clone()));
        self.wait_for_policy_evaluation(&handle);
        if !handle.is_complete() {
            self.wait_for_reservations_or_completion(
                &handle,
                reservation_count_before + 1,
                ReservationBlocker::SharedG4Offload,
            );
        }
        Ok(())
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
    /// G3/G4 are modeled by process-local shared resources hanging off the
    /// worker, so lower-tier transfers contend globally while G1↔G2 remains
    /// worker-local.
    fn prepare_tick(&self, now_ms: f64) -> PreparedAdvance {
        if let Some(pending) = self
            .advance_state
            .lock()
            .expect("offload advance mutex poisoned")
            .pending
            .clone()
        {
            return pending;
        }
        self.worker.set_now_ms(now_ms);
        let g2_registrations_before = Self::tier_registrations(&self.g2_manager);
        let local_token = self.engine.settlement_token();
        let drained = self.worker.drain_completions_summary(now_ms);
        let offload_drained = drained.local.offload_transfers;
        let offload_drained_blocks = drained.local.offload_blocks;
        let shared_g3 = drained.shared_g3.counts;
        let shared_g4 = drained.shared_g4.counts;
        if offload_drained > 0 {
            let mut target = SettlementTarget::new();
            target
                .add_completed_batches(
                    PipelineLane::G1ToG2,
                    u64::try_from(offload_drained).expect("G1→G2 completion count exceeds u64"),
                )
                .expect("G1→G2 settlement target overflow");
            self.settle_or_panic(local_token, target);
        }
        if let Some(settlement) = self
            .coordinator
            .shared_settlement(PipelineLane::G2ToG3, shared_g3.offload_transfers)
        {
            self.settle_or_panic(settlement.token, settlement.target);
        }
        if let Some(settlement) = self
            .coordinator
            .shared_settlement(PipelineLane::G2ToG4, shared_g4.offload_transfers)
        {
            self.settle_or_panic(settlement.token, settlement.target);
        }

        let current_shared_g3_onboard_blocks = shared_g3
            .onboard_blocks
            .saturating_sub(drained.shared_g3.deferred_onboard_blocks);
        let current_shared_g4_onboard_blocks = shared_g4
            .onboard_blocks
            .saturating_sub(drained.shared_g4.deferred_onboard_blocks);
        let g2_publish_blocks =
            current_shared_g3_onboard_blocks.saturating_add(current_shared_g4_onboard_blocks);

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
                    g3_to_g2_drained_blocks = current_shared_g3_onboard_blocks,
                    deferred_g3_to_g2_drained_blocks = drained.shared_g3.deferred_onboard_blocks,
                    g4_to_g2_drained_blocks = current_shared_g4_onboard_blocks,
                    deferred_g4_to_g2_drained_blocks = drained.shared_g4.deferred_onboard_blocks,
                    registrations_before = g2_registrations_before,
                    registrations_after,
                    "kvbm-offload: G2 registration barrier did not observe drained transfers"
                );
            }
        }
        let prepared = self.coordinator.prepare_progress(
            offload_drained > 0,
            shared_g3.offload_transfers > 0,
            shared_g4.offload_transfers > 0,
        );
        if let Some(failure) = prepared.failures.first() {
            panic!(
                "offload lease {:?} failed while preparing finalization: {}",
                failure.id, failure.message
            );
        }
        for chain in &prepared.chains {
            tracing::trace!(
                now_ms,
                parent = ?chain.parent,
                lane = ?chain.lane,
                blocks = chain.hashes.len(),
                "kvbm-offload: enqueue lower-tier background copies"
            );
            match chain.lane {
                PipelineLane::G2ToG3 => self.enqueue_g2_to_g3_background(chain.hashes.clone()),
                PipelineLane::G2ToG4 => self.enqueue_g2_to_g4_background(chain.hashes.clone()),
                PipelineLane::G1ToG2 => unreachable!("G1→G2 cannot be a lower-tier child"),
            }
            .expect("prepared lower-tier chain must register every child lease");
            self.coordinator
                .mark_chain_registered(chain.parent, chain.lane, chain.hashes.len());
        }

        self.pump_pending_staged_swap_ins(now_ms);
        let router_events = self.collect_g2_router_events();
        let mut state = self
            .advance_state
            .lock()
            .expect("offload advance mutex poisoned");
        let acknowledgement = AdvanceAcknowledgement(state.next_id);
        state.next_id = state
            .next_id
            .checked_add(1)
            .expect("offload advance acknowledgement overflow");
        let prepared = PreparedAdvance {
            router_events,
            acknowledgement,
        };
        state.pending = Some(prepared.clone());
        prepared
    }

    fn acknowledge_tick(
        &self,
        acknowledgement: AdvanceAcknowledgement,
    ) -> std::result::Result<usize, AdvanceAcknowledgementError> {
        {
            let state = self
                .advance_state
                .lock()
                .expect("offload advance mutex poisoned");
            let Some(pending) = state.pending.as_ref() else {
                return Err(AdvanceAcknowledgementError::NoPreparedAdvance);
            };
            if pending.acknowledgement != acknowledgement {
                return Err(AdvanceAcknowledgementError::Stale {
                    expected: pending.acknowledgement,
                    actual: acknowledgement,
                });
            }
        }
        let acknowledged = self.coordinator.acknowledge_prepared();
        if !acknowledged.abandoned_visibility.is_empty() {
            let mut metadata = self
                .g2_event_metadata
                .lock()
                .expect("G2 event metadata mutex poisoned");
            for hash in &acknowledged.abandoned_visibility {
                metadata.remove(hash);
            }
        }
        self.g2_destination_reservations
            .release(acknowledged.released_g2_reservations);
        if acknowledged.released_g3_reservations > 0 {
            self.shared_g3
                .as_ref()
                .expect("G2→G3 lease requires shared G3")
                .release_capacity_reservations(acknowledged.released_g3_reservations);
        }
        let released = acknowledged
            .released_g1_slots
            .saturating_add(self.coordinator.reap_terminal_swap_ins());
        let removed = self
            .advance_state
            .lock()
            .expect("offload advance mutex poisoned")
            .pending
            .take()
            .expect("prepared offload advance disappeared during acknowledgement");
        assert_eq!(removed.acknowledgement, acknowledgement);
        Ok(released)
    }

    pub(crate) fn prepare_tick_for_kv_manager(&self, now_ms: f64) -> PreparedAdvance {
        self.prepare_tick(now_ms)
    }

    pub(crate) fn acknowledge_tick_for_kv_manager(
        &self,
        acknowledgement: AdvanceAcknowledgement,
    ) -> std::result::Result<usize, AdvanceAcknowledgementError> {
        self.acknowledge_tick(acknowledgement)
    }

    pub fn tick(&self, now_ms: f64) -> usize {
        let prepared = self.prepare_tick(now_ms);
        self.pending_router_events
            .lock()
            .expect("pending router events mutex poisoned")
            .extend(prepared.router_events);
        self.acknowledge_tick(prepared.acknowledgement)
            .expect("freshly prepared offload advance must acknowledge")
    }

    /// Earliest transfer completion that can change offload-visible state.
    ///
    /// G2→G3/G2→G4 write-through copies are background from the scheduler's
    /// matching perspective, but they still pin G2 source blocks and may hold
    /// shared lower-tier reservations. Offline replay must therefore drain
    /// those completions at their DES timestamp, not at an arbitrary later
    /// arrival.
    pub fn earliest_pending_deadline(&self) -> Option<f64> {
        self.worker.earliest_finish()
    }

    pub(crate) fn g1_offload_dependency(&self, id: OffloadId) -> Option<(OffloadId, Option<f64>)> {
        if !self.coordinator.has_live_g1(id) {
            return None;
        }
        Some((
            id,
            self.worker
                .earliest_local_offload_finish_with_id()
                .map(|(_, d)| d),
        ))
    }

    #[cfg(test)]
    pub(crate) fn pending_g1_transfer_ownership(&self) -> (Vec<BlockId>, Vec<BlockId>) {
        self.coordinator.pending_g1_ownership()
    }

    /// Enqueue a burst of G1→G2 evictions with router metadata that will be
    /// used to publish HostPinned-tier events when G2 lifecycle notifications
    /// arrive.
    pub(crate) fn enqueue_g1_evictions_with_metadata(
        &mut self,
        evicted: &[G2OffloadBlock],
        source_slots: Vec<MutableBlock<MockerG1>>,
        now_ms: Option<f64>,
    ) -> Option<G1EvictionOutcome> {
        if evicted.is_empty() {
            drop(source_slots);
            return None;
        }
        self.remember_g2_event_metadata(evicted);
        let engine_blocks: Vec<_> = evicted
            .iter()
            .map(|block| (block.block_id, block.plh))
            .collect();
        self.enqueue_g1_evictions_holding_sources(&engine_blocks, source_slots, now_ms)
    }

    /// Enqueue a burst of G1→G2 evictions and hold the reset source slots
    /// until the simulated transfer reaches a terminal state.
    fn enqueue_g1_evictions_holding_sources(
        &mut self,
        evicted: &[(BlockId, SequenceHash)],
        source_slots: Vec<MutableBlock<MockerG1>>,
        now_ms: Option<f64>,
    ) -> Option<G1EvictionOutcome> {
        if evicted.is_empty() {
            drop(source_slots);
            return None;
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
        let settlement_token = self.engine.settlement_token();
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
        let visibility: FxHashMap<_, _> = evicted.iter().copied().collect();
        let lower_chain = if self.g3_manager.is_some() || self.shared_g4.is_some() {
            visibility.clone()
        } else {
            FxHashMap::default()
        };
        let offload_id = self.coordinator.insert_g1_to_g2(G1ToG2Lease::new(
            handle.clone(),
            source_slots,
            lower_chain,
            visibility,
            self.g3_manager.is_some(),
            self.shared_g4.is_some(),
        ));

        // Sync pump so policy and the first wave of batch reservations both
        // run on the current virtual `now_ms`, not a later scheduler tick.
        self.wait_for_policy_evaluation(&handle);
        let expected_reservations = self
            .initial_runnable_transfer_batches(handle.progress_counts().passed)
            .max(1);
        if !handle.is_complete() {
            let target_reservation_count = reservation_count_before
                .checked_add(expected_reservations as u64)
                .expect("mock transfer reservation count overflow");
            self.wait_for_reservations_or_completion(
                &handle,
                target_reservation_count,
                ReservationBlocker::LocalOffload,
            );
        }
        if handle.is_complete() {
            let released_slots = self.tick(self.worker.now_ms());
            return Some(G1EvictionOutcome::RetryNow { released_slots });
        }

        let dependency = self.worker.earliest_local_offload_finish_with_id();
        let can_block_for_settlement = match tokio::runtime::Handle::try_current()
            .ok()
            .as_ref()
            .map(tokio::runtime::Handle::runtime_flavor)
        {
            Some(tokio::runtime::RuntimeFlavor::MultiThread) => true,
            Some(tokio::runtime::RuntimeFlavor::CurrentThread) => false,
            None => self._runtime.is_some(),
            _ => unreachable!("Tokio runtime flavor is non-exhaustive"),
        };
        if dependency.is_none() && can_block_for_settlement {
            let expected_batches = self
                .initial_runnable_transfer_batches(handle.progress_counts().passed)
                .max(1);
            let mut target = SettlementTarget::new();
            target
                .add_completed_batches(
                    PipelineLane::G1ToG2,
                    u64::try_from(expected_batches).expect("G1→G2 batch count exceeds u64"),
                )
                .expect("G1→G2 settlement target overflow");
            self.settle_or_panic(settlement_token, target);
            let released_slots = self.tick(self.worker.now_ms());
            return Some(G1EvictionOutcome::RetryNow { released_slots });
        }

        Some(G1EvictionOutcome::BlockedOnOffload {
            offload_id,
            deadline_ms: dependency.map(|(_transfer_id, deadline_ms)| deadline_ms),
        })
    }

    /// Prepare the longest lower-tier prefix without reserving G2→G1 bandwidth.
    ///
    /// With only G2 configured this pins the currently available G2 prefix.
    /// With G3/G4 configured it first does a presence-only lower-tier planning
    /// pass. If the suffix can be staged into G2, it returns a deferred staging
    /// plan; if not, it falls back to pinning only the G2 prefix that is
    /// available right now.
    /// The caller must reserve destination G1 slots before passing the prepared
    /// lookup to [`start_onboard_prefix`](Self::start_onboard_prefix).
    pub(crate) fn prepare_onboard_prefix(
        &mut self,
        plhs: &[SequenceHash],
    ) -> Option<PreparedSwapIn> {
        if plhs.is_empty() {
            return None;
        }

        if self.g3_manager.is_none() && self.shared_g4.is_none() {
            let g2_blocks = self.g2_manager.match_blocks(plhs);
            if g2_blocks.is_empty() {
                return None;
            }
            return Some(PreparedSwapIn::from_g2_blocks(plhs.len(), g2_blocks));
        }

        let lower_tier_plan = self.build_lower_tier_lookup_plan(plhs);
        if lower_tier_plan.stage_blocks() > 0 {
            let available_g2 = self.g2_manager.available_blocks();
            let required_g2 = lower_tier_plan.stage_blocks();
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
                g3_stage_blocks = lower_tier_plan.g3_stage_blocks,
                g4_stage_blocks = lower_tier_plan.g4_stage_blocks,
                stage_blocks = lower_tier_plan.stage_blocks(),
                reservation_blocks = lower_tier_plan.reservation_blocks(),
                available_g2,
                required_g2,
                reserved_g2 = self.g2_destination_reservations.reserved_blocks(),
                "kvbm-offload: skipping lower-tier staging; insufficient G2 capacity"
            );
        }

        if lower_tier_plan.g2_prefix_blocks == 0 {
            tracing::debug!(
                plhs_len = plhs.len(),
                g3_stage_blocks = lower_tier_plan.g3_stage_blocks,
                g4_stage_blocks = lower_tier_plan.g4_stage_blocks,
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
        destination_slots: Vec<MutableBlock<MockerG1>>,
        prefix_pins: Vec<ImmutableBlock<MockerG1>>,
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
                let (status_tx, _) = watch::channel(SwapInStatus::pending(block_count));
                self.worker
                    .reserve_swap_in(now_ms, block_count, status_tx.clone());
                self.coordinator
                    .insert_direct_swap_in(DirectSwapInLease::new(SwapInResources {
                        status_tx,
                        g2_blocks: Some(g2_blocks),
                        destination_slots: Some(destination_slots),
                        prefix_pins: Some(prefix_pins),
                    }))
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
                    "kvbm-offload: lower-tier staging swap-in HIT"
                );
                let (status_tx, _) = watch::channel(SwapInStatus::pending(0));
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
                    .expect(
                        "find_matches_with_options must not fail for mocker lower-tier staging",
                    );
                let reserved_or_done = self
                    .wait_for_staging_reservation_or_completion(&result, reservation_count_before);
                if !reserved_or_done {
                    tracing::debug!(
                        now_ms,
                        plhs_len = requested_blocks,
                        reservation_blocks,
                        "kvbm-offload: lower-tier staging session has not reserved transfer yet"
                    );
                }
                let handle = self.coordinator.insert_staged_swap_in(StagedSwapInLease {
                    result,
                    reservation_blocks,
                    g2_capacity_reservation,
                    resources: SwapInResources {
                        status_tx,
                        g2_blocks: None,
                        destination_slots: Some(destination_slots),
                        prefix_pins: Some(prefix_pins),
                    },
                    g2_to_g1_started: false,
                });
                self.pump_pending_staged_swap_ins(now_ms);
                handle
            }
        }
    }

    pub(crate) fn cancel_swap_in(&self, id: OffloadId) -> bool {
        self.coordinator.cancel_swap_in(id)
    }

    pub(crate) fn take_completed_swap_in(
        &self,
        id: OffloadId,
    ) -> Option<(Vec<MutableBlock<MockerG1>>, Vec<ImmutableBlock<MockerG1>>)> {
        self.coordinator.take_completed_swap_in(id)
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

    #[cfg(test)]
    pub(crate) fn shared_g4(&self) -> Option<&Arc<SharedG4Store>> {
        self.shared_g4.as_ref()
    }
}

impl Drop for MockOffloadEngine {
    fn drop(&mut self) {
        let live_leases = self.coordinator.live_lease_count();
        let prepared_advance = self
            .advance_state
            .get_mut()
            .expect("offload advance mutex poisoned")
            .pending
            .is_some();
        if live_leases > 0 || prepared_advance {
            tracing::warn!(
                live_leases,
                prepared_advance,
                "offload coordinator dropped with live or prepared leases"
            );
        }
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
    use crate::kvbm_offload::shared_g4::{shared_g4_test_guard, shared_g4_test_guard_blocking};
    use crate::kvbm_offload::worker::WorkerFault;

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

    fn g4_config() -> KvbmOffloadConfig {
        KvbmOffloadConfig {
            enable_g4_storage: true,
            block_size_bytes: Some(1_000_000),
            offload_batch_size: 4,
            bandwidth_g2_to_g1_gbps: 1.0,
            bandwidth_g2_to_g4_gbps: 1.0,
            bandwidth_g4_to_g2_gbps: 1.0,
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

    fn allocate_g1_slots(count: usize) -> (BlockManager<MockerG1>, Vec<MutableBlock<MockerG1>>) {
        let registry = build_registry(Arc::new(EventsManager::builder().build()));
        let manager = BlockManager::<MockerG1>::builder()
            .block_count(count)
            .block_size(4)
            .registry(registry)
            .build()
            .expect("source manager build");
        let (slots, evicted) = manager
            .allocate_blocks_with_evictions(count)
            .expect("allocate G1 slots");
        assert!(evicted.is_empty());
        (manager, slots)
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

    #[test]
    fn g2_only_drained_batches_release_sources_without_chain_tracking() {
        const SOURCE_BLOCKS: usize = 5;

        let rt = single_thread_runtime();
        let config = KvbmOffloadConfig {
            num_g2_blocks: SOURCE_BLOCKS + 1,
            block_size_tokens: 4,
            offload_batch_size: 2,
            block_size_bytes: Some(1_000_000),
            bandwidth_g1_to_g2_gbps: 1.0,
            ..Default::default()
        };
        let mut engine = rt
            .block_on(MockOffloadEngine::new(config))
            .expect("engine build");
        engine.attach_runtime(rt);

        let source_registry = build_registry(Arc::new(EventsManager::builder().build()));
        let source_manager = BlockManager::<MockerG1>::builder()
            .block_count(SOURCE_BLOCKS)
            .block_size(4)
            .registry(source_registry)
            .build()
            .expect("source manager build");
        let (source_slots, evicted_sources) = source_manager
            .allocate_blocks_with_evictions(SOURCE_BLOCKS)
            .expect("allocate source slots");
        assert!(evicted_sources.is_empty());
        let evicted: Vec<_> = source_slots
            .iter()
            .enumerate()
            .map(|(index, slot)| {
                (
                    slot.block_id(),
                    SequenceHash::new(42 + index as u64, None, index as u64),
                )
            })
            .collect();
        assert!(matches!(
            engine.enqueue_g1_evictions_holding_sources(&evicted, source_slots, Some(0.0)),
            Some(G1EvictionOutcome::BlockedOnOffload { .. })
        ));

        assert_eq!(engine.coordinator.lane_lease_count(PipelineLane::G1ToG2), 1);
        assert_eq!(
            engine.pending_g1_transfer_ownership().0.len(),
            SOURCE_BLOCKS
        );

        let deadline = engine
            .earliest_pending_deadline()
            .expect("first transfer wave must have a deadline");
        let first_wave_blocks = engine.tick(deadline);
        assert!(first_wave_blocks > 0);
        assert!(first_wave_blocks < SOURCE_BLOCKS);
        assert_eq!(source_manager.available_blocks(), first_wave_blocks);

        assert_eq!(
            engine.pending_g1_transfer_ownership().0.len(),
            SOURCE_BLOCKS - first_wave_blocks
        );

        let final_deadline = engine
            .earliest_pending_deadline()
            .expect("the queued successor must start before first-wave settlement returns");
        assert!(final_deadline >= deadline);
        let mut released_blocks = first_wave_blocks + engine.tick(final_deadline);
        while released_blocks < SOURCE_BLOCKS {
            let deadline = engine
                .earliest_pending_deadline()
                .expect("every settled wave must hand its permit to the queued successor");
            let released = engine.tick(deadline);
            assert!(released > 0);
            released_blocks += released;
        }
        assert_eq!(released_blocks, SOURCE_BLOCKS);
        assert_eq!(source_manager.available_blocks(), SOURCE_BLOCKS);
        assert_eq!(engine.coordinator.lane_lease_count(PipelineLane::G1ToG2), 0);
    }

    #[test]
    fn prepared_advance_registers_children_before_parent_capacity_release() {
        use dynamo_tokens::PositionalLineageHash;

        let _guard = shared_g3_test_guard_blocking();
        let rt = single_thread_runtime();
        let mut engine = rt
            .block_on(MockOffloadEngine::new(g3_config()))
            .expect("engine build");
        engine.attach_runtime(rt);
        engine.tick(0.0);

        let (source_manager, source_slots) = allocate_g1_slots(1);
        let plh = PositionalLineageHash::new(8_001, None, 0);
        let source_id = source_slots[0].block_id();
        engine.enqueue_g1_evictions_holding_sources(&[(source_id, plh)], source_slots, Some(0.0));
        let deadline = engine
            .earliest_pending_deadline()
            .expect("G1→G2 completion deadline");

        let prepared = engine.prepare_tick_for_kv_manager(deadline);
        assert_eq!(
            source_manager.available_blocks(),
            0,
            "prepared parent resources must remain retained"
        );
        assert_eq!(
            engine.coordinator.lane_lease_count(PipelineLane::G2ToG3),
            1,
            "child lease must be coordinator-visible before acknowledgement"
        );

        let retried = engine.prepare_tick_for_kv_manager(deadline);
        assert_eq!(retried.acknowledgement, prepared.acknowledgement);
        assert_eq!(
            engine.coordinator.lane_lease_count(PipelineLane::G2ToG3),
            1,
            "retrying an unacknowledged advance must not duplicate child leases"
        );
        assert_eq!(
            engine
                .acknowledge_tick_for_kv_manager(prepared.acknowledgement)
                .expect("prepared advance acknowledgement"),
            1
        );
        assert_eq!(source_manager.available_blocks(), 1);

        let child_deadline = engine
            .earliest_pending_deadline()
            .expect("G2→G3 child deadline");
        engine.tick(child_deadline);
    }

    #[test]
    fn injected_executor_terminal_paths_release_source_ownership() {
        use dynamo_tokens::PositionalLineageHash;

        for (index, fault) in [
            WorkerFault::ExecutorError,
            WorkerFault::ExecutorPanic,
            WorkerFault::ChannelClosure,
        ]
        .into_iter()
        .enumerate()
        {
            let rt = single_thread_runtime();
            let mut engine = rt
                .block_on(MockOffloadEngine::new(KvbmOffloadConfig::default()))
                .expect("engine build");
            engine.attach_runtime(rt);
            engine.tick(0.0);
            let (source_manager, source_slots) = allocate_g1_slots(1);
            let source_id = source_slots[0].block_id();
            let plh = PositionalLineageHash::new(8_100 + index as u64, None, 0);
            engine.worker.inject_fault(fault);

            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                engine.enqueue_g1_evictions_holding_sources(
                    &[(source_id, plh)],
                    source_slots,
                    Some(0.0),
                )
            }));
            if matches!(fault, WorkerFault::ExecutorPanic) {
                assert!(
                    outcome.is_err(),
                    "executor task panic must surface as a fatal structured settlement failure"
                );
                drop(engine);
                assert_eq!(source_manager.available_blocks(), 1);
                continue;
            }
            let outcome = outcome.expect("non-panic executor fault must not unwind");
            assert!(
                matches!(outcome, Some(G1EvictionOutcome::RetryNow { .. })),
                "terminal executor fault {fault:?} must complete lease cleanup; got {outcome:?}"
            );
            assert_eq!(source_manager.available_blocks(), 1);
            assert_eq!(engine.coordinator.lane_lease_count(PipelineLane::G1ToG2), 0);
        }
    }

    #[tokio::test]
    async fn advance_acknowledgements_reject_foreign_and_stale_tokens() {
        let engine = MockOffloadEngine::new(KvbmOffloadConfig::default())
            .await
            .expect("engine build");
        let prepared = engine.prepare_tick_for_kv_manager(0.0);
        let foreign = AdvanceAcknowledgement(prepared.acknowledgement.0 + 1);
        assert!(matches!(
            engine.acknowledge_tick_for_kv_manager(foreign),
            Err(AdvanceAcknowledgementError::Stale { .. })
        ));
        engine
            .acknowledge_tick_for_kv_manager(prepared.acknowledgement)
            .expect("matching acknowledgement");
        assert_eq!(
            engine.acknowledge_tick_for_kv_manager(prepared.acknowledgement),
            Err(AdvanceAcknowledgementError::NoPreparedAdvance)
        );
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
    async fn mock_offload_engine_new_builds_g4_pipeline_when_enabled() {
        let _guard = shared_g4_test_guard().await;
        let engine = MockOffloadEngine::new(g4_config())
            .await
            .expect("construction should succeed");

        assert!(engine.engine.has_g1_to_g2());
        assert!(!engine.engine.has_g2_to_g3());
        assert!(engine.engine.has_g2_to_g4());
        assert!(engine.shared_g4().is_some());
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
        engine_a
            .enqueue_g2_to_g3_background(vec![plh_a])
            .expect("engine A G2→G3 enqueue");
        assert_eq!(
            engine_a.coordinator.lane_lease_count(PipelineLane::G2ToG3),
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
            engine_a.coordinator.lane_lease_count(PipelineLane::G2ToG3),
            1,
            "worker B must not perform engine A's owner-local cleanup"
        );
        assert_eq!(
            shared_reservations.reserved_blocks(),
            1,
            "non-owner drain must retain A's reservation until A settles"
        );

        assert_eq!(
            engine_a.earliest_pending_deadline(),
            Some(g3_deadline),
            "deferred owner completion must expose an immediate wake"
        );
        engine_a.tick(g3_deadline);
        assert_eq!(
            shared_reservations.reserved_blocks(),
            0,
            "owner settlement should release the shared reservation"
        );

        engine_b
            .enqueue_g2_to_g3_background(vec![plh_b])
            .expect("same-time admission after owner settlement");
        assert_eq!(
            engine_b.coordinator.lane_lease_count(PipelineLane::G2ToG3),
            1,
            "B should register a child lease at the same timestamp"
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
    fn shared_g3_deferred_waves_settle_against_one_pre_drain_checkpoint() {
        use dynamo_tokens::PositionalLineageHash;

        let _guard = shared_g3_test_guard_blocking();
        let rt_a = single_thread_runtime();
        let rt_b = single_thread_runtime();
        let config = KvbmOffloadConfig {
            num_g3_blocks: Some(4),
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

        let first = PositionalLineageHash::new(5_200, None, 0);
        let second = PositionalLineageHash::new(5_201, None, 0);
        register_test_block(engine_a.g2_manager(), first);
        register_test_block(engine_a.g2_manager(), second);
        engine_a
            .enqueue_g2_to_g3_background(vec![first])
            .expect("first G2→G3 enqueue");
        engine_a
            .enqueue_g2_to_g3_background(vec![second])
            .expect("queued second G2→G3 enqueue");
        assert_eq!(
            engine_a.coordinator.lane_lease_count(PipelineLane::G2ToG3),
            2
        );

        let first_deadline = engine_a
            .earliest_pending_deadline()
            .expect("first shared completion");
        engine_b.tick(first_deadline);
        engine_a.tick(first_deadline);
        assert_eq!(
            engine_a.coordinator.lane_lease_count(PipelineLane::G2ToG3),
            1,
            "first settlement must retain the checkpoint for the queued lease"
        );

        let second_deadline = engine_a
            .earliest_pending_deadline()
            .expect("permit handoff must start the second shared transfer");
        assert!(second_deadline >= first_deadline);
        engine_b.tick(second_deadline);
        engine_a.tick(second_deadline);
        assert_eq!(
            engine_a.coordinator.lane_lease_count(PipelineLane::G2ToG3),
            0,
            "second deferred wave must settle cumulatively from the original token"
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
    fn g1_to_g2_completion_feeds_g4_with_object_presence_policy() {
        use dynamo_tokens::PositionalLineageHash;

        let _guard = shared_g4_test_guard_blocking();
        let rt = single_thread_runtime();
        let mut engine = rt
            .block_on(MockOffloadEngine::new(g4_config()))
            .expect("engine build");
        engine.attach_runtime(rt);
        engine.tick(0.0);

        let plh = PositionalLineageHash::new(6_500, None, 0);
        engine.enqueue_g1_evictions_holding_sources(&[(0, plh)], Vec::new(), Some(0.0));

        let first_deadline = engine
            .earliest_pending_deadline()
            .expect("G1→G2 should reserve bandwidth");
        engine.tick(first_deadline);

        let second_deadline = engine
            .earliest_pending_deadline()
            .expect("G2→G4 background copy should create a completion deadline");
        engine.tick(second_deadline);

        let shared_g4 = engine.shared_g4().expect("G4 enabled");
        assert_eq!(
            shared_g4.has_object(&plh),
            Some(1_000_000),
            "G2→G4 should store the block in shared object storage"
        );
        assert_eq!(shared_g4.object_count(), 1);
    }

    #[test]
    fn shared_g4_cross_owner_deferred_waves_settle_cumulatively() {
        use dynamo_tokens::PositionalLineageHash;

        let _guard = shared_g4_test_guard_blocking();
        let rt_a = single_thread_runtime();
        let rt_b = single_thread_runtime();
        let config = KvbmOffloadConfig {
            offload_batch_size: 1,
            ..g4_config()
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

        let first = PositionalLineageHash::new(6_700, None, 0);
        let second = PositionalLineageHash::new(6_701, None, 0);
        register_test_block(engine_a.g2_manager(), first);
        register_test_block(engine_a.g2_manager(), second);
        engine_a
            .enqueue_g2_to_g4_background(vec![first])
            .expect("first G2→G4 enqueue");
        engine_a
            .enqueue_g2_to_g4_background(vec![second])
            .expect("queued second G2→G4 enqueue");

        let first_deadline = engine_a
            .earliest_pending_deadline()
            .expect("first shared object completion");
        engine_b.tick(first_deadline);
        engine_a.tick(first_deadline);
        assert_eq!(
            engine_a.coordinator.lane_lease_count(PipelineLane::G2ToG4),
            1
        );

        let second_deadline = engine_a
            .earliest_pending_deadline()
            .expect("object permit handoff must start the second transfer");
        engine_b.tick(second_deadline);
        engine_a.tick(second_deadline);
        assert_eq!(
            engine_a.coordinator.lane_lease_count(PipelineLane::G2ToG4),
            0
        );
        let store = engine_a.shared_g4().expect("G4 enabled");
        assert!(store.has_object(&first).is_some());
        assert!(store.has_object(&second).is_some());
    }

    #[test]
    fn g2_to_g3_background_copy_strong_pins_g2_source() {
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
            "in-flight G2→G3 mock copy should strong-pin the G2 source block"
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
        let handle = engine.start_onboard_prefix(prepared, Vec::new(), Vec::new(), Some(0.0));
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
            "swap-in status must complete once tick advances past finish"
        );
        engine
            .take_completed_swap_in(handle.id())
            .expect("completed direct swap-in resources");
    }

    #[tokio::test]
    async fn direct_swap_in_cancellation_releases_only_after_modeled_completion() {
        use dynamo_tokens::PositionalLineageHash;

        let config = KvbmOffloadConfig {
            block_size_bytes: Some(1_000_000),
            bandwidth_g2_to_g1_gbps: 1.0,
            ..Default::default()
        };
        let mut engine = MockOffloadEngine::new(config).await.expect("engine build");
        engine.tick(0.0);
        let plh = PositionalLineageHash::new(9_001, None, 0);
        register_test_block(engine.g2_manager(), plh);
        let (g1_manager, destination_slots) = allocate_g1_slots(1);
        let prepared = engine
            .prepare_onboard_prefix(&[plh])
            .expect("G2 prefix match");
        let handle =
            engine.start_onboard_prefix(prepared, destination_slots, Vec::new(), Some(0.0));
        assert!(engine.cancel_swap_in(handle.id()));

        engine.tick(0.5);
        assert_eq!(handle.terminal(), SwapInTerminal::Pending);
        assert_eq!(g1_manager.available_blocks(), 0);

        engine.tick(1.0);
        assert_eq!(handle.terminal(), SwapInTerminal::Cancelled);
        assert_eq!(g1_manager.available_blocks(), 1);
        assert!(engine.take_completed_swap_in(handle.id()).is_none());
    }

    #[test]
    fn staged_swap_in_cancellation_drains_active_phase_without_successor() {
        use dynamo_tokens::PositionalLineageHash;

        let _guard = shared_g3_test_guard_blocking();
        let rt = single_thread_runtime();
        let mut engine = rt
            .block_on(MockOffloadEngine::new(g3_config()))
            .expect("engine build");
        engine.attach_runtime(rt);
        engine.tick(0.0);

        let plh = PositionalLineageHash::new(9_002, None, 0);
        register_test_block(engine.g3_manager().expect("G3 enabled"), plh);
        let (g1_manager, destination_slots) = allocate_g1_slots(1);
        let prepared = engine
            .prepare_onboard_prefix(&[plh])
            .expect("G3 prefix match");
        let handle =
            engine.start_onboard_prefix(prepared, destination_slots, Vec::new(), Some(0.0));
        assert!(engine.cancel_swap_in(handle.id()));
        assert_eq!(g1_manager.available_blocks(), 0);

        let staging_deadline = engine
            .earliest_pending_deadline()
            .expect("active G3→G2 phase");
        engine.tick(staging_deadline);
        assert_eq!(handle.terminal(), SwapInTerminal::Cancelled);
        assert_eq!(g1_manager.available_blocks(), 1);
        assert_eq!(
            engine.earliest_pending_deadline(),
            None,
            "cancellation must not start a G2→G1 successor"
        );
    }

    #[test]
    fn staged_swap_in_cancellation_during_successor_retains_destination() {
        use dynamo_tokens::PositionalLineageHash;

        let _guard = shared_g3_test_guard_blocking();
        let rt = single_thread_runtime();
        let mut engine = rt
            .block_on(MockOffloadEngine::new(g3_config()))
            .expect("engine build");
        engine.attach_runtime(rt);
        engine.tick(0.0);

        let plh = PositionalLineageHash::new(9_003, None, 0);
        register_test_block(engine.g3_manager().expect("G3 enabled"), plh);
        let (g1_manager, destination_slots) = allocate_g1_slots(1);
        let prepared = engine
            .prepare_onboard_prefix(&[plh])
            .expect("G3 prefix match");
        let handle =
            engine.start_onboard_prefix(prepared, destination_slots, Vec::new(), Some(0.0));
        let staging_deadline = engine
            .earliest_pending_deadline()
            .expect("G3→G2 staging deadline");
        engine.tick(staging_deadline);
        let onboard_deadline = engine
            .earliest_pending_deadline()
            .expect("G2→G1 successor deadline");
        assert!(engine.cancel_swap_in(handle.id()));

        engine.tick((staging_deadline + onboard_deadline) / 2.0);
        assert_eq!(handle.terminal(), SwapInTerminal::Pending);
        assert_eq!(g1_manager.available_blocks(), 0);

        engine.tick(onboard_deadline);
        assert_eq!(handle.terminal(), SwapInTerminal::Cancelled);
        assert_eq!(g1_manager.available_blocks(), 1);
    }

    #[test]
    fn staged_channel_closure_publishes_failure_before_releasing_resources() {
        use dynamo_tokens::PositionalLineageHash;

        let _guard = shared_g3_test_guard_blocking();
        let rt = single_thread_runtime();
        let mut engine = rt
            .block_on(MockOffloadEngine::new(g3_config()))
            .expect("engine build");
        engine.attach_runtime(rt);
        engine.tick(0.0);

        let plh = PositionalLineageHash::new(9_004, None, 0);
        register_test_block(engine.g3_manager().expect("G3 enabled"), plh);
        let (g1_manager, destination_slots) = allocate_g1_slots(1);
        engine.worker.inject_fault(WorkerFault::ChannelClosure);
        let prepared = engine
            .prepare_onboard_prefix(&[plh])
            .expect("G3 prefix match");
        let handle =
            engine.start_onboard_prefix(prepared, destination_slots, Vec::new(), Some(0.0));

        assert!(
            matches!(handle.terminal(), SwapInTerminal::Failed(_)),
            "staging channel closure must publish failure; got {:?}",
            handle.terminal()
        );
        assert_eq!(
            g1_manager.available_blocks(),
            0,
            "failure publication must precede terminal resource cleanup"
        );
        engine.tick(0.0);
        assert_eq!(g1_manager.available_blocks(), 1);
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
        let handle = engine.start_onboard_prefix(prepared, Vec::new(), Vec::new(), Some(0.0));
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
        engine
            .take_completed_swap_in(handle.id())
            .expect("completed staged G3 swap-in resources");
    }

    #[test]
    fn staged_g4_swap_in_runs_g4_to_g2_before_g2_to_g1() {
        use dynamo_tokens::PositionalLineageHash;

        let _guard = shared_g4_test_guard_blocking();
        let rt = single_thread_runtime();
        let mut engine = rt
            .block_on(MockOffloadEngine::new(g4_config()))
            .expect("engine build");
        engine.attach_runtime(rt);
        engine.tick(0.0);

        let plh = PositionalLineageHash::new(6_600, None, 0);
        engine
            .shared_g4()
            .expect("G4 enabled")
            .insert_object(plh, 1_000_000);

        let prepared = engine
            .prepare_onboard_prefix(&[plh])
            .expect("G4 prefix match must produce a staged swap-in");
        assert_eq!(prepared.reservation_block_count(), 1);
        let handle = engine.start_onboard_prefix(prepared, Vec::new(), Vec::new(), Some(0.0));
        assert!(!handle.is_complete());

        let first_deadline = engine
            .earliest_pending_deadline()
            .expect("G4→G2 staging should reserve shared bandwidth");
        assert!(
            (first_deadline - 1.0).abs() < 1e-6,
            "1 MB / 1 GB/s G4→G2 should finish at 1 ms, got {first_deadline}"
        );

        engine.tick(first_deadline);
        assert!(
            !handle.is_complete(),
            "G2→G1 transfer should start after G4 staging, not complete immediately"
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
        engine
            .take_completed_swap_in(handle.id())
            .expect("completed staged G4 swap-in resources");
    }
}
