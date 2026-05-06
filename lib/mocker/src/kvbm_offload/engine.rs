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
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use anyhow::Result;
use dynamo_tokens::{BlockHash, SequenceHash as RouterSequenceHash};
use futures::Stream;
use futures::stream::{FuturesUnordered, StreamExt};
use futures::task::noop_waker_ref;

use kvbm_engine::leader::{FindMatchesOptions, InstanceLeader, Leader, StagingMode};
use kvbm_engine::offload::{
    ExternalBlock, OffloadEngine, PendingTracker, PipelineBuilder, PresenceFilter, SourceBlocks,
    TransferHandle, TransferStatus,
};
use kvbm_engine::worker::Worker;
use kvbm_engine::{BlockId, G1 as EngineG1, G2, SequenceHash};
use kvbm_logical::blocks::{ImmutableBlock, MutableBlock};
use kvbm_logical::events::{EventsManager, KvCacheEvent as LogicalKvCacheEvent};
use kvbm_logical::manager::{BlockManager, FrequencyTrackingCapacity};
use kvbm_logical::pools::BlockDuplicationPolicy;
use kvbm_logical::registry::BlockRegistry;
use rustc_hash::FxHashMap;

use crate::common::protocols::G1 as MockerG1;

use super::config::KvbmOffloadConfig;
use super::worker::MockWorker;

// Successful offline barriers wake via kvbm-engine watch channels or the
// mock worker's Notify. The timeout is only a hang guard for pipeline bugs.
const PIPELINE_BARRIER_TIMEOUT: Duration = Duration::from_secs(1);

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
    block_count: usize,
    /// Strong refs pinning the G2 blocks for the transfer's lifetime.
    /// Not accessed directly — presence alone upholds the RAII contract.
    _g2_blocks: Vec<ImmutableBlock<G2>>,
}

impl SwapInHandle {
    pub fn is_complete(&self) -> bool {
        self.complete.load(std::sync::atomic::Ordering::Acquire)
    }

    pub fn block_count(&self) -> usize {
        self.block_count
    }
}

/// G2 lookup result pinned while the caller reserves destination G1 slots.
pub struct PreparedSwapIn {
    requested_blocks: usize,
    g2_blocks: Vec<ImmutableBlock<G2>>,
}

impl PreparedSwapIn {
    pub fn block_count(&self) -> usize {
        self.g2_blocks.len()
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
    /// Reset G1 slots held until the simulated source copy completes.
    ///
    /// These tokens do not preserve old bytes — `MockWorker` never reads
    /// source contents — but they do preserve G1 capacity accounting. A real
    /// DMA source slot cannot be reassigned while the copy is in flight.
    _source_slots: Vec<MutableBlock<MockerG1>>,
}

impl PendingG1ToG2 {
    fn source_slots_releasable(&self) -> bool {
        if self.handle.is_complete() {
            return true;
        }
        let passed = self.handle.passed_blocks().len();
        if passed == 0 {
            return !matches!(self.handle.status(), TransferStatus::Evaluating);
        }
        self.handle.completed_blocks().len() + self.handle.failed_blocks().len() >= passed
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
    pending_g1_to_g2: Mutex<Vec<PendingG1ToG2>>,
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

        let worker = Arc::new(MockWorker::new(
            config.block_size_bytes.unwrap_or(0),
            config.bandwidth_g1_to_g2_gbps,
            config.bandwidth_g2_to_g1_gbps,
            None,
            None,
        ));
        let worker_for_leader: Arc<dyn Worker> = worker.clone();

        // `InstanceLeader::build` calls `Handle::current()` internally,
        // hence the `async fn` and the in-runtime caller requirement.
        let leader = Arc::new(
            InstanceLeader::builder()
                .messenger(messenger)
                .registry((*registry).clone())
                .g2_manager(g2_manager.clone())
                .worker(worker_for_leader)
                .build()?,
        );

        let g1_to_g2_pending = Arc::new(PendingTracker::new());
        let g1_to_g2_presence = PresenceFilter::<EngineG1, G2>::new(registry.clone())
            .with_pending_tracker(g1_to_g2_pending.clone());
        let g1_to_g2_pipeline = PipelineBuilder::<EngineG1, G2>::new()
            .policy(Arc::new(g1_to_g2_presence))
            .pending_tracker(g1_to_g2_pending)
            .batch_size(config.offload_batch_size)
            .max_concurrent_transfers(config.offload_batch_size)
            .build();

        let engine = OffloadEngine::builder(leader.clone())
            .with_registry(registry.clone())
            .with_g2_manager(g2_manager.clone())
            .with_g1_to_g2_pipeline(g1_to_g2_pipeline)
            .build()?;

        Ok(Self {
            config,
            engine,
            leader,
            worker,
            registry,
            g2_manager,
            pending_g1_to_g2: Mutex::new(Vec::new()),
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
                if self.worker.earliest_finish().is_some() {
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

    fn wait_for_settled_g1_to_g2_blocks(&self, expected_settled_blocks: usize) -> bool {
        let pending = self
            .pending_g1_to_g2
            .lock()
            .expect("pending G1→G2 handles mutex poisoned");
        let handles: Vec<TransferHandle> = pending
            .iter()
            .map(|pending| pending.handle.clone())
            .collect();
        drop(pending);

        let mut completed: Vec<_> = handles
            .iter()
            .map(TransferHandle::subscribe_completed)
            .collect();
        let mut failed: Vec<_> = handles
            .iter()
            .map(TransferHandle::subscribe_failed)
            .collect();

        self.wait_on_attached_runtime(async move {
            loop {
                let settled_blocks: usize = handles
                    .iter()
                    .map(|handle| handle.completed_blocks().len() + handle.failed_blocks().len())
                    .sum();
                if settled_blocks >= expected_settled_blocks {
                    return true;
                }
                if handles.is_empty() {
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

    /// Total blocks across all pending handles that have reached a terminal
    /// per-block state (completed or failed); used as the pump target so
    /// `tick` can confirm drained transfers have published before returning.
    fn pending_g1_to_g2_settled_blocks(&self) -> usize {
        let pending = self
            .pending_g1_to_g2
            .lock()
            .expect("pending G1→G2 handles mutex poisoned");
        pending
            .iter()
            .map(|pending| {
                pending.handle.completed_blocks().len() + pending.handle.failed_blocks().len()
            })
            .sum()
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
    /// **Assumption that breaks for shared tiers (G3 NVMe, G4 object
    /// storage):** when multiple workers can drive transfers into the
    /// same physical resource, per-worker PS undercounts contention —
    /// two workers each see N=1 on their local model while the
    /// underlying device sees N=2. A harness-global PS queue is
    /// required before extending this simulation past G2.
    pub fn tick(&self, now_ms: f64) {
        self.worker.set_now_ms(now_ms);
        let settled_before = self.pending_g1_to_g2_settled_blocks();
        let (offload_drained, _, offload_drained_blocks, _) = self.worker.drain_completions(now_ms);

        // Offline replay owns a private runtime for the kvbm-engine
        // pipeline. Once drain fires transfer awaiters, give those
        // pipeline tasks a chance to register completed destination blocks
        // before the scheduler immediately queries G2 in the same pass.
        if offload_drained > 0 {
            if offload_drained_blocks > 0 {
                let expected_settled_blocks = settled_before + offload_drained_blocks;
                let published = self.wait_for_settled_g1_to_g2_blocks(expected_settled_blocks);
                if !published {
                    tracing::warn!(
                        now_ms,
                        offload_drained,
                        offload_drained_blocks,
                        settled_before,
                        settled_after = self.pending_g1_to_g2_settled_blocks(),
                        "kvbm-offload: pipeline did not publish drained transfers"
                    );
                }
            }
            self.prune_releasable_g1_to_g2_sources();
        } else {
            self.prune_releasable_g1_to_g2_sources();
        }
    }

    /// Earliest pending completion across both PS links, or `None` when
    /// both are idle.
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
                _source_slots: source_slots,
            });
        }

        // Sync pump so policy and the first wave of batch reservations both
        // run on the current virtual `now_ms`, not a later scheduler tick.
        self.wait_for_policy_evaluation(&handle);
        let target_reservation_count = reservation_count_before
            + self.initial_runnable_transfer_batches(handle.passed_blocks().len()) as u64;
        if target_reservation_count > reservation_count_before {
            self.wait_for_reservations_or_completion(&handle, target_reservation_count);
        }
        if handle.is_complete() {
            self.prune_releasable_g1_to_g2_sources();
        }
    }

    /// Find and pin the longest G2-resident prefix without reserving G2→G1
    /// bandwidth yet. The caller must reserve destination G1 slots before
    /// passing the prepared lookup to [`start_onboard_prefix`](Self::start_onboard_prefix).
    pub fn prepare_onboard_prefix(&mut self, plhs: &[SequenceHash]) -> Option<PreparedSwapIn> {
        if plhs.is_empty() {
            return None;
        }
        let options = FindMatchesOptions {
            search_remote: false,
            staging_mode: StagingMode::Hold,
        };
        let mut result = self
            .leader
            .find_matches_with_options(plhs, options)
            .expect("find_matches_with_options must not fail on local-only Hold lookup");
        let g2_blocks = result
            .take_g2_blocks()
            .expect("Ready variant must yield G2 blocks on Hold + !search_remote path");
        if g2_blocks.is_empty() {
            tracing::debug!(
                plhs_len = plhs.len(),
                "kvbm-offload: G2→G1 lookup MISS (0 blocks in G2)"
            );
            return None;
        }
        Some(PreparedSwapIn {
            requested_blocks: plhs.len(),
            g2_blocks,
        })
    }

    /// Reserve G2→G1 bandwidth for a prepared lookup after the caller has
    /// already reserved destination G1 slots.
    pub fn start_onboard_prefix(
        &mut self,
        prepared: PreparedSwapIn,
        now_ms: Option<f64>,
    ) -> SwapInHandle {
        let block_count = prepared.g2_blocks.len();
        let now_ms = now_ms.unwrap_or_else(|| self.worker.now_ms());
        self.worker.set_now_ms(now_ms);
        tracing::debug!(
            now_ms,
            plhs_len = prepared.requested_blocks,
            block_count,
            "kvbm-offload: G2→G1 swap-in HIT"
        );
        let complete = Arc::new(AtomicBool::new(false));
        self.worker
            .reserve_swap_in(now_ms, block_count, complete.clone());
        SwapInHandle {
            complete,
            block_count,
            _g2_blocks: prepared.g2_blocks,
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
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap();
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
}
