// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Endpoint-pool CKF actor, publisher, and rank-recovery target.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

#[cfg(feature = "ckf-diagnostics")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "ckf-diagnostics")]
use std::time::Instant;

#[cfg(test)]
use dynamo_kv_router::indexer::cuckoo::CkfConfig;
#[cfg(any(test, feature = "ckf-diagnostics"))]
use dynamo_kv_router::indexer::cuckoo::DcCkfStats;
#[cfg(feature = "ckf-diagnostics")]
use dynamo_kv_router::indexer::cuckoo::PublisherEmitOutcome;
use dynamo_kv_router::indexer::cuckoo::{
    CkfFailureAction, CkfFailureDisposition, CkfFailurePoint, DcCkfDelta, DcCkfDeltaSink,
    DcCkfPublisher, DcCkfRankReplacement, DcCkfSnapshot, DcCkfState, LaneLease, ProducerIdentity,
};
#[cfg(test)]
use dynamo_kv_router::protocols::ExternalSequenceBlockHash;
use dynamo_kv_router::protocols::{
    DpRank, KvCacheEventData, KvCacheEventError, RouterEvent, WorkerId, WorkerWithDpRank,
};
#[cfg(feature = "ckf-diagnostics")]
use parking_lot::Mutex;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, broadcast, mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::discovery::PublisherId;
use crate::kv_router::indexer::{RecoveryResetReason, RecoveryTarget};

use super::host::KvDcRelayError;

const DEFAULT_MAILBOX_CAPACITY: usize = 256;
const DEFAULT_PENDING_BLOCK_PERMITS: usize = 65_536;
const DEFAULT_PUBLICATION_CAPACITY: usize = 64;
pub(super) const DEFAULT_FAULT_CAPACITY: usize = 16;
#[cfg(test)]
const DEFAULT_PUBLICATION_DELAY: Duration = Duration::from_millis(1);
const RECOVERY_REBUILD_BATCH_WINDOW: Duration = Duration::from_millis(5);

#[derive(Debug)]
// NOTE: This is the intentional producer snapshot/delta seam for the future non-local adapter.
// It remains crate-private until that transport has delivery cursors and recovery semantics.
#[allow(dead_code)]
pub(crate) struct DcCkfSubscription {
    pub(crate) snapshot: DcCkfSnapshot,
    pub(crate) deltas: broadcast::Receiver<DcCkfDelta>,
}

// NOTE: `dynamo-llm` enables the router's general metrics feature in production. Keep these
// pull-only diagnostics on a separate feature so ordinary commands do not acquire an activity
// mutex, read the clock, or update mailbox/publication atomics.
#[cfg(feature = "ckf-diagnostics")]
#[derive(Debug, Default)]
pub(super) struct ActorCounters {
    pub(super) mailbox_wait_ns: AtomicU64,
    pub(super) mailbox_max_wait_ns: AtomicU64,
    pub(super) degraded_resets: AtomicU64,
    pub(super) publications: AtomicU64,
    pub(super) unchanged_publications: AtomicU64,
    pub(super) rebuild_count: AtomicU64,
    pub(super) rebuild_ns: AtomicU64,
    pub(super) rebuild_max_ns: AtomicU64,
}

#[cfg(feature = "ckf-diagnostics")]
#[derive(Debug, Default)]
pub(super) struct ActorActivity {
    pub(super) active_command: Option<&'static str>,
    pub(super) active_since: Option<Instant>,
    pub(super) shutting_down: bool,
    pub(super) last_error: Option<String>,
}

#[cfg(feature = "ckf-diagnostics")]
#[derive(Debug, Default)]
pub(super) struct ActorDiagnostics {
    pub(super) counters: ActorCounters,
    pub(super) activity: Mutex<ActorActivity>,
}

#[cfg(feature = "ckf-diagnostics")]
#[derive(Debug, Clone, Default)]
pub(super) struct ActorDiagnosticsHandle(pub(super) Arc<ActorDiagnostics>);

#[cfg(not(feature = "ckf-diagnostics"))]
#[derive(Debug, Clone, Default)]
pub(super) struct ActorDiagnosticsHandle;

#[cfg(feature = "ckf-diagnostics")]
impl ActorDiagnosticsHandle {
    fn new() -> Self {
        Self::default()
    }

    fn start_command(&self, command: &ActorCommand) {
        let mut activity = self.0.activity.lock();
        activity.active_command = Some(command.kind());
        activity.active_since = Some(Instant::now());
    }

    fn finish_command(&self) {
        let mut activity = self.0.activity.lock();
        activity.active_command = None;
        activity.active_since = None;
    }

    fn record_error(&self, error: &impl std::fmt::Display) {
        self.0.activity.lock().last_error = Some(error.to_string());
    }

    fn record_shutdown(&self) {
        self.0.activity.lock().shutting_down = true;
    }

    fn record_mailbox_wait(&self, started: Instant) {
        let waited = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        self.0
            .counters
            .mailbox_wait_ns
            .fetch_add(waited, Ordering::Relaxed);
        self.0
            .counters
            .mailbox_max_wait_ns
            .fetch_max(waited, Ordering::Relaxed);
    }

    fn record_publish_outcome(&self, outcome: &PublisherEmitOutcome) {
        match outcome {
            PublisherEmitOutcome::Published { .. } => {
                self.0.counters.publications.fetch_add(1, Ordering::Relaxed);
            }
            PublisherEmitOutcome::NoSubscriber { .. } => {
                self.record_no_publication();
            }
        }
    }

    fn record_no_publication(&self) {
        self.0
            .counters
            .unchanged_publications
            .fetch_add(1, Ordering::Relaxed);
    }

    fn record_degraded_reset(&self) {
        self.0
            .counters
            .degraded_resets
            .fetch_add(1, Ordering::Relaxed);
    }

    fn record_rebuild(&self, started: Instant) {
        let elapsed = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        self.0
            .counters
            .rebuild_count
            .fetch_add(1, Ordering::Relaxed);
        self.0
            .counters
            .rebuild_ns
            .fetch_add(elapsed, Ordering::Relaxed);
        self.0
            .counters
            .rebuild_max_ns
            .fetch_max(elapsed, Ordering::Relaxed);
    }
}

#[cfg(not(feature = "ckf-diagnostics"))]
impl ActorDiagnosticsHandle {
    fn new() -> Self {
        Self
    }

    #[inline(always)]
    fn start_command(&self, _command: &ActorCommand) {}

    #[inline(always)]
    fn finish_command(&self) {}

    #[inline(always)]
    fn record_error(&self, _error: &impl std::fmt::Display) {}

    #[inline(always)]
    fn record_shutdown(&self) {}

    #[inline(always)]
    fn record_publish_outcome<T>(&self, _outcome: &T) {}

    #[inline(always)]
    fn record_no_publication(&self) {}

    #[inline(always)]
    fn record_degraded_reset(&self) {}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ActorFaultCategory {
    Resource,
    SourceProtocol,
    ProducerInvariant,
}

#[derive(Debug)]
pub(super) struct ActorFault {
    pub(super) worker_id: WorkerId,
    pub(super) dp_rank: DpRank,
    pub(super) publisher_id: PublisherId,
    pub(super) event_id: Option<u64>,
    pub(super) category: ActorFaultCategory,
    pub(super) disposition: CkfFailureDisposition,
    pub(super) message: String,
}

struct CancelOnDrop(CancellationToken);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

// NOTE: Recovery is selected from the state whose commit became uncertain—not merely from the
// error's name.
fn event_failure_point(error: KvCacheEventError) -> CkfFailurePoint {
    match error {
        KvCacheEventError::CapacityExhausted => CkfFailurePoint::BoundedRelocationFailure,
        KvCacheEventError::AllocationFailed => CkfFailurePoint::PrecommitAllocationFailure,
        KvCacheEventError::OwnershipDegreeOverflow
        | KvCacheEventError::ParentBlockNotFound
        | KvCacheEventError::BlockNotFound
        | KvCacheEventError::InvalidBlockSequence => CkfFailurePoint::SourceProtocolFailure,
        KvCacheEventError::IndexerInvariantViolation => CkfFailurePoint::PrewriteInvariantMismatch,
        _ => CkfFailurePoint::SourceProtocolFailure,
    }
}

fn actor_fault_category(disposition: CkfFailureDisposition) -> ActorFaultCategory {
    match disposition.action {
        CkfFailureAction::ReportResourceFailure => ActorFaultCategory::Resource,
        CkfFailureAction::RejectSource => ActorFaultCategory::SourceProtocol,
        CkfFailureAction::FenceAndRebuildProducer | CkfFailureAction::ContinueCapacityOmission => {
            ActorFaultCategory::ProducerInvariant
        }
        CkfFailureAction::DeactivateAndSnapshot | CkfFailureAction::RetrySnapshot => {
            unreachable!("consumer-lane disposition cannot originate from a producer event")
        }
    }
}

async fn send_actor_fault(
    sender: &mpsc::Sender<ActorFault>,
    fence: &CancellationToken,
    fault: ActorFault,
) -> bool {
    tokio::select! {
        biased;
        _ = fence.cancelled() => false,
        result = sender.send(fault) => result.is_ok(),
    }
}

#[derive(Debug, Clone)]
pub(super) struct StreamScope {
    pub(super) relay_incarnation: u64,
    pub(super) layout_generation: u64,
    pub(super) pool_id: dynamo_kv_router::identity::PoolId,
}

#[derive(Debug, Clone)]
struct BroadcastDeltaSink {
    sender: broadcast::Sender<DcCkfDelta>,
}

impl DcCkfDeltaSink for BroadcastDeltaSink {
    type Error = broadcast::error::SendError<DcCkfDelta>;

    fn enqueue(&mut self, delta: DcCkfDelta) -> Result<(), Self::Error> {
        self.sender.send(delta).map(|_| ())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct KvDcRelayHandle {
    sender: mpsc::Sender<ActorCommand>,
    identity: ProducerIdentity,
    payload_permits: Arc<Semaphore>,
    fence: CancellationToken,
    stopped: CancellationToken,
    #[cfg(feature = "ckf-diagnostics")]
    pub(super) diagnostics: ActorDiagnosticsHandle,
}

impl KvDcRelayHandle {
    #[cfg(test)]
    fn spawn(
        config: CkfConfig,
        scope: StreamScope,
    ) -> Result<(Self, mpsc::Receiver<ActorFault>), KvDcRelayError> {
        Self::spawn_with_capacity_and_delay(
            config,
            scope,
            DEFAULT_MAILBOX_CAPACITY,
            DEFAULT_PUBLICATION_DELAY,
        )
    }

    #[cfg(test)]
    pub(super) fn spawn_with_publication_delay(
        config: CkfConfig,
        scope: StreamScope,
        publication_delay: Duration,
    ) -> Result<(Self, mpsc::Receiver<ActorFault>), KvDcRelayError> {
        Self::spawn_with_capacity_and_delay(
            config,
            scope,
            DEFAULT_MAILBOX_CAPACITY,
            publication_delay,
        )
    }

    pub(super) fn spawn_with_state_and_publication_delay(
        state: DcCkfState,
        scope: StreamScope,
        publication_delay: Duration,
    ) -> (Self, mpsc::Receiver<ActorFault>) {
        Self::spawn_with_state_capacity_and_delay(
            state,
            scope,
            DEFAULT_MAILBOX_CAPACITY,
            publication_delay,
        )
    }

    #[cfg(test)]
    fn spawn_with_capacity(
        config: CkfConfig,
        scope: StreamScope,
        capacity: usize,
    ) -> Result<(Self, mpsc::Receiver<ActorFault>), KvDcRelayError> {
        Self::spawn_with_capacity_and_delay(config, scope, capacity, DEFAULT_PUBLICATION_DELAY)
    }

    #[cfg(test)]
    fn spawn_with_capacity_and_delay(
        config: CkfConfig,
        scope: StreamScope,
        capacity: usize,
        publication_delay: Duration,
    ) -> Result<(Self, mpsc::Receiver<ActorFault>), KvDcRelayError> {
        let state = DcCkfState::new(config)?;
        Ok(Self::spawn_with_state_capacity_and_delay(
            state,
            scope,
            capacity,
            publication_delay,
        ))
    }

    fn spawn_with_state_capacity_and_delay(
        state: DcCkfState,
        scope: StreamScope,
        capacity: usize,
        publication_delay: Duration,
    ) -> (Self, mpsc::Receiver<ActorFault>) {
        let (sender, receiver) = mpsc::channel(capacity);
        let (publication_tx, _) = broadcast::channel(DEFAULT_PUBLICATION_CAPACITY);
        let identity = ProducerIdentity::new(
            scope.pool_id,
            scope.relay_incarnation,
            scope.layout_generation,
            state.format(),
        );
        let publisher = DcCkfPublisher::new(
            identity,
            0,
            BroadcastDeltaSink {
                sender: publication_tx.clone(),
            },
        );
        let (fault_tx, fault_rx) = mpsc::channel(DEFAULT_FAULT_CAPACITY);
        let diagnostics = ActorDiagnosticsHandle::new();
        let fence = CancellationToken::new();
        let stopped = CancellationToken::new();
        tokio::spawn(run_actor(
            state,
            publisher,
            receiver,
            publication_delay,
            fault_tx,
            diagnostics.clone(),
            fence.clone(),
            stopped.clone(),
        ));
        (
            Self {
                sender,
                identity,
                payload_permits: Arc::new(Semaphore::new(DEFAULT_PENDING_BLOCK_PERMITS)),
                fence,
                stopped,
                #[cfg(feature = "ckf-diagnostics")]
                diagnostics,
            },
            fault_rx,
        )
    }

    pub(super) const fn identity(&self) -> ProducerIdentity {
        self.identity
    }

    async fn submit<T>(
        &self,
        make_command: impl FnOnce(oneshot::Sender<Result<T, KvDcRelayError>>) -> ActorCommand,
    ) -> Result<T, KvDcRelayError> {
        let (response_tx, response_rx) = oneshot::channel();
        #[cfg(feature = "ckf-diagnostics")]
        let wait_started = Instant::now();
        self.sender
            .send(make_command(response_tx))
            .await
            .map_err(|_| KvDcRelayError::ShuttingDown)?;
        #[cfg(feature = "ckf-diagnostics")]
        self.diagnostics.record_mailbox_wait(wait_started);
        response_rx
            .await
            .map_err(|_| KvDcRelayError::ActorStopped)?
    }

    pub(crate) async fn admit_event(
        &self,
        publisher_id: PublisherId,
        event: RouterEvent,
    ) -> Result<(), KvDcRelayError> {
        let weight = event_payload_weight(&event).min(DEFAULT_PENDING_BLOCK_PERMITS) as u32;
        #[cfg(feature = "ckf-diagnostics")]
        let wait_started = Instant::now();
        let permit = self
            .payload_permits
            .clone()
            .acquire_many_owned(weight.max(1))
            .await
            .map_err(|_| KvDcRelayError::ShuttingDown)?;
        self.sender
            .send(ActorCommand::Apply {
                publisher_id,
                event,
                _payload_permit: permit,
            })
            .await
            .map_err(|_| KvDcRelayError::ShuttingDown)?;
        #[cfg(feature = "ckf-diagnostics")]
        self.diagnostics.record_mailbox_wait(wait_started);
        Ok(())
    }

    async fn replace_ranks(
        &self,
        replacements: Vec<RankReplacement>,
    ) -> Result<(), KvDcRelayError> {
        let weight = replacements
            .iter()
            .flat_map(|replacement| &replacement.events)
            .map(event_payload_weight)
            .fold(0usize, usize::saturating_add)
            .min(DEFAULT_PENDING_BLOCK_PERMITS) as u32;
        let permit = self
            .payload_permits
            .clone()
            .acquire_many_owned(weight.max(1))
            .await
            .map_err(|_| KvDcRelayError::ShuttingDown)?;
        self.submit(|response| ActorCommand::ReplaceRanks {
            replacements,
            _payload_permit: permit,
            response,
        })
        .await
    }

    #[cfg(test)]
    async fn replace_rank(
        &self,
        publisher_id: PublisherId,
        worker_id: WorkerId,
        dp_rank: DpRank,
        events: Vec<RouterEvent>,
    ) -> Result<(), KvDcRelayError> {
        self.replace_ranks(vec![RankReplacement {
            publisher_id,
            worker_id,
            dp_rank,
            events,
        }])
        .await
    }

    async fn reset_rank(
        &self,
        publisher_id: PublisherId,
        worker_id: WorkerId,
        dp_rank: DpRank,
        degraded: bool,
    ) -> Result<(), KvDcRelayError> {
        self.submit(|response| ActorCommand::ResetRank {
            publisher_id,
            worker_id,
            dp_rank,
            degraded,
            response,
        })
        .await
    }

    pub(super) async fn flush(&self) -> Result<(), KvDcRelayError> {
        self.submit(|response| ActorCommand::Flush { response })
            .await
    }

    #[cfg(feature = "ckf-diagnostics")]
    pub(super) async fn snapshot(&self) -> Result<ActorSnapshot, KvDcRelayError> {
        self.submit(|response| ActorCommand::Snapshot { response })
            .await
    }

    #[cfg(any(test, feature = "ckf-diagnostics"))]
    pub(super) async fn state_stats(
        &self,
    ) -> Result<(DcCkfStats, u64, Vec<(WorkerWithDpRank, usize)>), KvDcRelayError> {
        self.submit(|response| ActorCommand::Stats { response })
            .await
    }

    // Kept with `DcCkfSubscription` as the crate-private producer boundary described above.
    #[allow(dead_code)]
    pub(crate) async fn subscribe(
        &self,
        lease: LaneLease,
    ) -> Result<DcCkfSubscription, KvDcRelayError> {
        let subscription = self
            .submit(|response| ActorCommand::Subscribe { lease, response })
            .await?;
        Ok(DcCkfSubscription {
            snapshot: subscription.snapshot,
            deltas: subscription.deltas,
        })
    }

    pub(super) async fn shutdown(&self) -> Result<(), KvDcRelayError> {
        self.submit(|response| ActorCommand::Shutdown { response })
            .await
    }

    pub(super) async fn fence(&self) -> Result<(), KvDcRelayError> {
        self.fence.cancel();
        self.stopped.cancelled().await;
        Ok(())
    }

    #[cfg(any(test, feature = "ckf-diagnostics"))]
    pub(super) fn mailbox_depth(&self) -> usize {
        self.sender
            .max_capacity()
            .saturating_sub(self.sender.capacity())
    }

    #[cfg(feature = "ckf-diagnostics")]
    pub(super) fn mailbox_capacity(&self) -> usize {
        self.sender.max_capacity()
    }
}

#[derive(Debug)]
struct RankReplacement {
    publisher_id: PublisherId,
    worker_id: WorkerId,
    dp_rank: DpRank,
    events: Vec<RouterEvent>,
}

struct PendingRankReplacement {
    replacement: RankReplacement,
    response: oneshot::Sender<Result<(), String>>,
}

struct RankReplacementBatcher {
    state: tokio::sync::Mutex<RankReplacementBatchState>,
    initial_deadline: Duration,
}

#[derive(Default)]
struct RankReplacementBatchState {
    pending: Vec<PendingRankReplacement>,
    flush_scheduled: bool,
    initial_timer_scheduled: bool,
    initial_expected: Option<HashSet<WorkerWithDpRank>>,
    initial_completed: HashSet<WorkerWithDpRank>,
}

#[derive(Clone)]
pub(super) struct KvDcRelayRecoveryTarget {
    handle: KvDcRelayHandle,
    rebuild_permit: Arc<Semaphore>,
    replacement_batcher: Arc<RankReplacementBatcher>,
}

impl KvDcRelayRecoveryTarget {
    pub(super) fn new(
        handle: KvDcRelayHandle,
        rebuild_permit: Arc<Semaphore>,
        expected: HashSet<WorkerWithDpRank>,
        initial_deadline: Duration,
    ) -> Self {
        Self {
            handle,
            rebuild_permit,
            replacement_batcher: Self::new_replacement_batcher(expected, initial_deadline),
        }
    }

    async fn flush_replacement_batch(self, wait_for_quiet: bool) {
        if wait_for_quiet {
            let mut observed = 0usize;
            loop {
                tokio::time::sleep(RECOVERY_REBUILD_BATCH_WINDOW).await;
                let current = self.replacement_batcher.state.lock().await.pending.len();
                if current == observed {
                    break;
                }
                observed = current;
            }
        }
        let pending = {
            let mut state = self.replacement_batcher.state.lock().await;
            state.flush_scheduled = false;
            std::mem::take(&mut state.pending)
        };
        let (replacements, responses): (Vec<_>, Vec<_>) = pending
            .into_iter()
            .map(|pending| (pending.replacement, pending.response))
            .unzip();
        let batch_result = match self.rebuild_permit.acquire().await {
            Ok(_permit) => self
                .handle
                .replace_ranks(replacements)
                .await
                .map_err(|error| error.to_string()),
            Err(_) => Err(KvDcRelayError::ShuttingDown.to_string()),
        };
        for response_tx in responses {
            let response = match &batch_result {
                Ok(()) => Ok(()),
                Err(error) => Err(error.clone()),
            };
            let _ = response_tx.send(response);
        }
    }

    async fn expire_initial_recovery_batch(self) {
        tokio::time::sleep(self.replacement_batcher.initial_deadline).await;
        let schedule_flush = {
            let mut state = self.replacement_batcher.state.lock().await;
            let initial_open = state.initial_expected.take().is_some();
            if initial_open {
                state.initial_completed.clear();
            }
            if !initial_open || state.pending.is_empty() || state.flush_scheduled {
                false
            } else {
                state.flush_scheduled = true;
                true
            }
        };
        if schedule_flush {
            self.flush_replacement_batch(false).await;
        }
    }

    fn new_replacement_batcher(
        expected: HashSet<WorkerWithDpRank>,
        initial_deadline: Duration,
    ) -> Arc<RankReplacementBatcher> {
        Arc::new(RankReplacementBatcher {
            state: tokio::sync::Mutex::new(RankReplacementBatchState {
                initial_expected: (!expected.is_empty()).then_some(expected),
                ..RankReplacementBatchState::default()
            }),
            initial_deadline,
        })
    }

    async fn mark_initial_complete(&self, member: WorkerWithDpRank) {
        let schedule_flush = {
            let mut state = self.replacement_batcher.state.lock().await;
            let Some(expected) = state.initial_expected.as_ref() else {
                return;
            };
            if !expected.contains(&member) {
                return;
            }
            state.initial_completed.insert(member);
            let complete = state
                .initial_expected
                .as_ref()
                .is_some_and(|expected| expected.is_subset(&state.initial_completed));
            if !complete {
                return;
            }
            state.initial_expected = None;
            state.initial_completed.clear();
            if state.pending.is_empty() || state.flush_scheduled {
                false
            } else {
                state.flush_scheduled = true;
                true
            }
        };
        if schedule_flush {
            tokio::spawn(self.clone().flush_replacement_batch(false));
        }
    }
}

impl RecoveryTarget for KvDcRelayRecoveryTarget {
    async fn admit_event(
        &self,
        publisher_id: PublisherId,
        event: RouterEvent,
    ) -> anyhow::Result<()> {
        self.handle
            .admit_event(publisher_id, event)
            .await
            .map_err(Into::into)
    }

    async fn replace_rank(
        &self,
        publisher_id: PublisherId,
        worker_id: WorkerId,
        dp_rank: DpRank,
        events: Vec<RouterEvent>,
    ) -> anyhow::Result<()> {
        let (response, result) = oneshot::channel();
        let member = WorkerWithDpRank::new(worker_id, dp_rank);
        let (schedule_flush, schedule_deadline, wait_for_quiet) = {
            let mut state = self.replacement_batcher.state.lock().await;
            state.pending.push(PendingRankReplacement {
                replacement: RankReplacement {
                    publisher_id,
                    worker_id,
                    dp_rank,
                    events,
                },
                response,
            });
            let initial = state
                .initial_expected
                .as_ref()
                .is_some_and(|expected| expected.contains(&member));
            if initial {
                state.initial_completed.insert(member);
            }
            let initial_complete = initial
                && state
                    .initial_expected
                    .as_ref()
                    .is_some_and(|expected| expected.is_subset(&state.initial_completed));
            if initial_complete {
                state.initial_expected = None;
                state.initial_completed.clear();
            }
            let initial_wave_open = state.initial_expected.is_some();
            let schedule_deadline = initial
                && !initial_complete
                && !std::mem::replace(&mut state.initial_timer_scheduled, true);
            if state.flush_scheduled || initial_wave_open {
                (false, schedule_deadline, false)
            } else {
                state.flush_scheduled = true;
                (true, schedule_deadline, !initial)
            }
        };
        if schedule_flush {
            tokio::spawn(self.clone().flush_replacement_batch(wait_for_quiet));
        }
        if schedule_deadline {
            tokio::spawn(self.clone().expire_initial_recovery_batch());
        }
        result
            .await
            .map_err(|_| anyhow::anyhow!("rank replacement batch coordinator stopped"))?
            .map_err(anyhow::Error::msg)
    }

    async fn complete_initial_recovery(&self, worker_id: WorkerId, dp_rank: DpRank) {
        self.mark_initial_complete(WorkerWithDpRank::new(worker_id, dp_rank))
            .await;
    }

    async fn reset_rank(
        &self,
        publisher_id: PublisherId,
        worker_id: WorkerId,
        dp_rank: DpRank,
        reason: RecoveryResetReason,
    ) -> anyhow::Result<()> {
        self.handle
            .reset_rank(
                publisher_id,
                worker_id,
                dp_rank,
                reason == RecoveryResetReason::TreeDumpFailed,
            )
            .await
            .map_err(Into::into)
    }
}

#[cfg(any(test, feature = "ckf-diagnostics"))]
type ActorStatsResult = Result<(DcCkfStats, u64, Vec<(WorkerWithDpRank, usize)>), KvDcRelayError>;

enum ActorCommand {
    Apply {
        publisher_id: PublisherId,
        event: RouterEvent,
        _payload_permit: OwnedSemaphorePermit,
    },
    ReplaceRanks {
        replacements: Vec<RankReplacement>,
        _payload_permit: OwnedSemaphorePermit,
        response: oneshot::Sender<Result<(), KvDcRelayError>>,
    },
    ResetRank {
        publisher_id: PublisherId,
        worker_id: WorkerId,
        dp_rank: DpRank,
        degraded: bool,
        response: oneshot::Sender<Result<(), KvDcRelayError>>,
    },
    Flush {
        response: oneshot::Sender<Result<(), KvDcRelayError>>,
    },
    #[cfg(feature = "ckf-diagnostics")]
    Snapshot {
        response: oneshot::Sender<Result<ActorSnapshot, KvDcRelayError>>,
    },
    Subscribe {
        lease: LaneLease,
        response: oneshot::Sender<Result<ActorSubscription, KvDcRelayError>>,
    },
    #[cfg(any(test, feature = "ckf-diagnostics"))]
    Stats {
        response: oneshot::Sender<ActorStatsResult>,
    },
    Shutdown {
        response: oneshot::Sender<Result<(), KvDcRelayError>>,
    },
    #[cfg(test)]
    Pause {
        entered: oneshot::Sender<()>,
        release: oneshot::Receiver<()>,
    },
    #[cfg(test)]
    InjectFault { fault: ActorFault },
}

#[cfg(feature = "ckf-diagnostics")]
pub(super) struct ActorSnapshot {
    pub(super) identity: ProducerIdentity,
    pub(super) sequence: u64,
    pub(super) buckets: Box<[u64]>,
    pub(super) stats: DcCkfStats,
}

struct ActorSubscription {
    snapshot: DcCkfSnapshot,
    deltas: broadcast::Receiver<DcCkfDelta>,
}

impl ActorCommand {
    #[cfg(feature = "ckf-diagnostics")]
    fn kind(&self) -> &'static str {
        match self {
            Self::Apply { .. } => "apply_event",
            Self::ReplaceRanks { .. } => "replace_ranks",
            Self::ResetRank { .. } => "reset_rank",
            Self::Flush { .. } => "flush",
            Self::Snapshot { .. } => "snapshot",
            Self::Subscribe { .. } => "subscribe",
            #[cfg(any(test, feature = "ckf-diagnostics"))]
            Self::Stats { .. } => "stats",
            Self::Shutdown { .. } => "shutdown",
            #[cfg(test)]
            Self::Pause { .. } => "test_pause",
            #[cfg(test)]
            Self::InjectFault { .. } => "test_fault",
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_actor(
    mut state: DcCkfState,
    mut publisher: DcCkfPublisher<BroadcastDeltaSink>,
    mut receiver: mpsc::Receiver<ActorCommand>,
    publication_delay: Duration,
    fault_tx: mpsc::Sender<ActorFault>,
    diagnostics: ActorDiagnosticsHandle,
    fence: CancellationToken,
    stopped: CancellationToken,
) {
    let _stopped_guard = CancelOnDrop(stopped);
    let mut unknown_removal_events = 0u64;
    let mut capacity_omission_events = 0u64;
    let mut shutdown_response = None;
    let mut discard_tail = false;
    let publication_timer = tokio::time::sleep(Duration::ZERO);
    tokio::pin!(publication_timer);
    let mut publication_timer_armed = false;
    loop {
        let command = tokio::select! {
            biased;
            _ = fence.cancelled() => {
                discard_tail = true;
                break;
            }
            command = receiver.recv() => command,
            _ = &mut publication_timer, if publication_timer_armed => {
                publication_timer_armed = false;
                if state.has_pending_publication()
                    && let Err(error) = publish_pending(&mut state, &mut publisher, &diagnostics)
                {
                    diagnostics.record_error(&error);
                }
                continue;
            }
        };
        let Some(command) = command else {
            break;
        };
        diagnostics.start_command(&command);
        match command {
            ActorCommand::Apply {
                publisher_id,
                event,
                ..
            } => {
                let worker_id = event.worker_id;
                let dp_rank = event.event.dp_rank;
                let event_id = event.event.event_id;
                // NOTE: Subscriber-free Relay operation is transitional or diagnostic, not an
                // optimized steady-state mode. Keep one publication path until an unsubscribed
                // deployment is measured; a useful Relay is expected to have a consumer.
                let outcome = state.apply_event(event);
                let first_error = outcome.first_error().copied();
                let publication_boundary = outcome.publication_boundary();
                if outcome.unknown_removals() != 0 {
                    unknown_removal_events = unknown_removal_events.saturating_add(1);
                    if unknown_removal_events == 1 {
                        tracing::warn!(
                            worker_id,
                            dp_rank,
                            event_id,
                            unknown_removals = outcome.unknown_removals(),
                            unknown_removal_events,
                            "Ignoring KV DC Relay removals not owned by this worker/rank"
                        );
                    } else if unknown_removal_events.is_power_of_two() {
                        tracing::debug!(
                            worker_id,
                            dp_rank,
                            event_id,
                            unknown_removals = outcome.unknown_removals(),
                            unknown_removal_events,
                            "KV DC Relay continues after repeated unknown removals"
                        );
                    }
                }
                if let Some(batch) = outcome.into_publication() {
                    if let Err(error) = publish_batch(batch, &mut publisher, &diagnostics) {
                        diagnostics.record_error(&error);
                    }
                } else if publication_boundary {
                    diagnostics.record_no_publication();
                }
                if publication_boundary {
                    publication_timer_armed = false;
                } else if state.pending_event_count() == 1 && !publication_timer_armed {
                    publication_timer
                        .as_mut()
                        .reset(tokio::time::Instant::now() + publication_delay);
                    publication_timer_armed = true;
                }
                if let Some(error) = first_error {
                    let disposition = event_failure_point(error).disposition();
                    if disposition.action == CkfFailureAction::ContinueCapacityOmission {
                        // NOTE: A bounded relocation miss is a deterministic physical-index
                        // omission. Exact source lineage and ownership remain committed, while the
                        // failed fingerprint is non-resident. Do not turn it into a lifecycle
                        // fault: successful siblings remain committed, a new owner may retry
                        // admission, and removal still resolves through source lineage.
                        capacity_omission_events = capacity_omission_events.saturating_add(1);
                        if capacity_omission_events == 1 {
                            tracing::warn!(
                                worker_id,
                                dp_rank,
                                event_id,
                                capacity_omission_events,
                                "KV DC Relay omitted a capacity-exhausted mutation; service continues"
                            );
                        } else if capacity_omission_events.is_power_of_two() {
                            tracing::debug!(
                                worker_id,
                                dp_rank,
                                event_id,
                                capacity_omission_events,
                                "KV DC Relay continues after repeated capacity omissions"
                            );
                        }
                        diagnostics.finish_command();
                        continue;
                    }
                    let message = error.to_string();
                    let category = actor_fault_category(disposition);
                    diagnostics.record_error(&message);
                    if !send_actor_fault(
                        &fault_tx,
                        &fence,
                        ActorFault {
                            worker_id,
                            dp_rank,
                            publisher_id,
                            event_id: Some(event_id),
                            category,
                            disposition,
                            message,
                        },
                    )
                    .await
                    {
                        discard_tail = fence.is_cancelled();
                        break;
                    }
                }
            }
            ActorCommand::ReplaceRanks {
                replacements,
                _payload_permit: _,
                response,
            } => {
                #[cfg(feature = "ckf-diagnostics")]
                let rebuild_started = Instant::now();
                let result = replacement_batch_states(replacements)
                    .and_then(|states| state.replace_ranks(states).map_err(Into::into))
                    .and_then(|publication| {
                        if let Some(batch) = publication {
                            publish_batch(batch, &mut publisher, &diagnostics)?;
                        } else {
                            diagnostics.record_no_publication();
                        }
                        Ok(())
                    });
                #[cfg(feature = "ckf-diagnostics")]
                diagnostics.record_rebuild(rebuild_started);
                // The whole cold-start batch is built off-side. A pre-swap failure leaves every
                // prior rank unchanged; the strong responses all observe the same atomic result.
                let _ = response.send(result);
            }
            ActorCommand::ResetRank {
                publisher_id,
                worker_id,
                dp_rank,
                degraded,
                response,
            } => {
                let key = WorkerWithDpRank::new(worker_id, dp_rank);
                let mut removal = state.remove_rank(key);
                if let Err(error) = removal {
                    // Clear may have committed earlier hashes while remaining exact. Retry the
                    // still-tracked suffix once; the strong acknowledgement reports failure if
                    // progress cannot be completed.
                    tracing::warn!(
                        worker_id,
                        dp_rank,
                        publisher_id,
                        %error,
                        "Retrying the remaining tracked hashes after a partial rank reset"
                    );
                    removal = state.remove_rank(key);
                }
                let result = removal
                    .map_err(KvDcRelayError::from)
                    .and_then(|publication| {
                        if degraded {
                            diagnostics.record_degraded_reset();
                        }
                        if let Some(batch) = publication {
                            publish_batch(batch, &mut publisher, &diagnostics)?;
                        } else {
                            diagnostics.record_no_publication();
                        }
                        Ok(())
                    });
                let _ = response.send(result);
            }
            ActorCommand::Flush { response } => {
                let result = publish_pending(&mut state, &mut publisher, &diagnostics);
                let _ = response.send(result);
            }
            #[cfg(feature = "ckf-diagnostics")]
            ActorCommand::Snapshot { response } => {
                let result = diagnostic_barrier_snapshot(&mut state, &mut publisher, &diagnostics);
                let _ = response.send(result);
            }
            ActorCommand::Subscribe { lease, response } => {
                let result = publisher
                    .snapshot_after_barrier(&mut state, lease)
                    .map_err(|error| KvDcRelayError::Publisher(format!("{error:?}")))
                    .map(|snapshot| {
                        // The actor cannot process a continuation mutation until this command
                        // returns. Subscribe after any old-lease tail so the new receiver starts
                        // exactly after snapshot sequence N.
                        let deltas = publisher.sink().sender.subscribe();
                        ActorSubscription { snapshot, deltas }
                    });
                let _ = response.send(result);
            }
            #[cfg(any(test, feature = "ckf-diagnostics"))]
            ActorCommand::Stats { response } => {
                let _ = response.send(Ok((
                    state.stats(),
                    publisher.last_sequence(),
                    state.member_counts(),
                )));
            }
            ActorCommand::Shutdown { response } => {
                if shutdown_response.is_some() {
                    let _ = response.send(Err(KvDcRelayError::ShuttingDown));
                } else {
                    receiver.close();
                    diagnostics.record_shutdown();
                    shutdown_response = Some(response);
                }
            }
            #[cfg(test)]
            ActorCommand::Pause { entered, release } => {
                let _ = entered.send(());
                let _ = release.await;
            }
            #[cfg(test)]
            ActorCommand::InjectFault { fault } => {
                if !send_actor_fault(&fault_tx, &fence, fault).await {
                    discard_tail = fence.is_cancelled();
                    break;
                }
            }
        }
        if !state.has_pending_publication() {
            publication_timer_armed = false;
        }
        diagnostics.finish_command();
    }

    if !discard_tail && let Err(error) = publish_pending(&mut state, &mut publisher, &diagnostics) {
        diagnostics.record_error(&error);
    }
    publisher.retire_lease();
    drop(fault_tx);
    if let Some(response) = shutdown_response {
        let _ = response.send(Ok(()));
    }
}

fn replacement_state(
    worker_id: WorkerId,
    dp_rank: DpRank,
    events: Vec<RouterEvent>,
) -> Result<DcCkfRankReplacement, KvDcRelayError> {
    let mut replacement = DcCkfRankReplacement::new();
    for event in events {
        if event.worker_id != worker_id || event.event.dp_rank != dp_rank {
            return Err(KvDcRelayError::InvalidTreeDump {
                worker_id,
                dp_rank,
                message: "event identity does not match replacement rank".to_string(),
            });
        }
        replacement.push_event(event).map_err(|error| match error {
            KvCacheEventError::AllocationFailed => KvDcRelayError::Build(
                dynamo_kv_router::indexer::cuckoo::CkfBuildError::AllocationFailed,
            ),
            _ => KvDcRelayError::InvalidTreeDump {
                worker_id,
                dp_rank,
                message: format!("tree dump is not canonical replay order: {error}"),
            },
        })?;
    }
    Ok(replacement)
}

fn replacement_batch_states(
    replacements: Vec<RankReplacement>,
) -> Result<HashMap<WorkerWithDpRank, DcCkfRankReplacement>, KvDcRelayError> {
    let mut states_by_rank = HashMap::new();
    states_by_rank
        .try_reserve(replacements.len())
        .map_err(|_| {
            KvDcRelayError::Build(
                dynamo_kv_router::indexer::cuckoo::CkfBuildError::AllocationFailed,
            )
        })?;
    for replacement in replacements {
        let member = WorkerWithDpRank::new(replacement.worker_id, replacement.dp_rank);
        if states_by_rank.contains_key(&member) {
            return Err(KvDcRelayError::InvalidTreeDump {
                worker_id: replacement.worker_id,
                dp_rank: replacement.dp_rank,
                message: format!(
                    "replacement batch contains the same rank more than once (publisher {})",
                    replacement.publisher_id
                ),
            });
        }
        let state = replacement_state(
            replacement.worker_id,
            replacement.dp_rank,
            replacement.events,
        )?;
        states_by_rank.insert(member, state);
    }
    Ok(states_by_rank)
}

fn publish_batch(
    batch: dynamo_kv_router::indexer::cuckoo::DcCkfPublicationBatch,
    publisher: &mut DcCkfPublisher<BroadcastDeltaSink>,
    diagnostics: &ActorDiagnosticsHandle,
) -> Result<(), KvDcRelayError> {
    let outcome = publisher
        .publish(batch)
        .map_err(|error| KvDcRelayError::Publisher(format!("{error:?}")))?;
    diagnostics.record_publish_outcome(&outcome);
    Ok(())
}

fn publish_pending(
    state: &mut DcCkfState,
    publisher: &mut DcCkfPublisher<BroadcastDeltaSink>,
    diagnostics: &ActorDiagnosticsHandle,
) -> Result<(), KvDcRelayError> {
    let Some(batch) = state.flush() else {
        return Ok(());
    };
    publish_batch(batch, publisher, diagnostics)
}

#[cfg(feature = "ckf-diagnostics")]
fn diagnostic_barrier_snapshot(
    state: &mut DcCkfState,
    publisher: &mut DcCkfPublisher<BroadcastDeltaSink>,
    diagnostics: &ActorDiagnosticsHandle,
) -> Result<ActorSnapshot, KvDcRelayError> {
    let (publication, buckets) = state.barrier_snapshot()?;
    if let Some(batch) = publication {
        publish_batch(batch, publisher, diagnostics)?;
    }
    Ok(ActorSnapshot {
        identity: publisher.identity(),
        sequence: publisher.last_sequence(),
        buckets,
        stats: state.stats(),
    })
}

fn event_payload_weight(event: &RouterEvent) -> usize {
    match &event.event.data {
        KvCacheEventData::Stored(store) => store.blocks.len().max(1),
        KvCacheEventData::Removed(remove) => remove.block_hashes.len().max(1),
        KvCacheEventData::Cleared => 1,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::time::Duration;

    use anyhow::Result;
    use async_trait::async_trait;
    use dynamo_kv_router::identity::{
        CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, PoolId, RoutingScopeId,
    };
    use dynamo_kv_router::indexer::{
        WorkerKvQueryResponse,
        cuckoo::{CkfCommitState, CkfFailureDomain, ConsumerInstanceId},
    };
    use dynamo_kv_router::protocols::{
        KvCacheEvent, KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash,
    };
    use dynamo_runtime::{component::Instance, protocols::EndpointId};
    use tokio::sync::watch;

    use super::*;
    use crate::discovery::{
        KvEventSource, KvSourceMembershipView, KvSourceStatus, KvStateEndpointResolution,
    };
    use crate::kv_router::indexer::{WorkerQueryClient, WorkerQueryTransport};

    struct UnusedWorkerQueryTransport;

    #[async_trait]
    impl WorkerQueryTransport for UnusedWorkerQueryTransport {
        async fn query_worker(
            &self,
            _worker_id: WorkerId,
            _dp_rank: DpRank,
            _target: Instance,
            _start_event_id: Option<u64>,
            _end_event_id: Option<u64>,
        ) -> Result<WorkerKvQueryResponse> {
            panic!("live-only source must not start worker recovery")
        }
    }

    const EXTERNAL_MASK: u64 = 0xBADC_0FFE_E0DD_F00D;

    fn scope(_name: &str) -> StreamScope {
        let dc_id = DcId::new(2);
        let domain = IndexerDomainId::new(
            CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
            RoutingScopeId::new([3; 16], IdentitySource::Explicit),
        );
        StreamScope {
            relay_incarnation: 1,
            layout_generation: 1,
            pool_id: PoolId::new(domain, dc_id),
        }
    }

    fn lease(epoch: u64) -> LaneLease {
        LaneLease::new(ConsumerInstanceId::new(4), 0, epoch)
    }

    fn stored(worker: WorkerWithDpRank, event_id: u64, hashes: &[u64]) -> RouterEvent {
        RouterEvent::new(
            worker.worker_id,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: hashes
                        .iter()
                        .copied()
                        .map(|hash| KvCacheStoredBlockData {
                            block_hash: ExternalSequenceBlockHash(hash ^ EXTERNAL_MASK),
                            tokens_hash: LocalBlockHash(hash),
                            mm_extra_info: None,
                        })
                        .collect(),
                }),
                dp_rank: worker.dp_rank,
            },
        )
    }

    async fn pause_actor(handle: &KvDcRelayHandle) -> oneshot::Sender<()> {
        let (entered_tx, entered_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        handle
            .sender
            .send(ActorCommand::Pause {
                entered: entered_tx,
                release: release_rx,
            })
            .await
            .unwrap();
        entered_rx.await.unwrap();
        release_tx
    }

    #[cfg(feature = "ckf-diagnostics")]
    #[tokio::test]
    async fn diagnostic_feature_exposes_rich_actor_and_snapshot_state() {
        let worker = WorkerWithDpRank::new(1, 0);
        let (handle, _faults) =
            KvDcRelayHandle::spawn(CkfConfig::new(32), scope("diagnostics")).unwrap();

        handle
            .admit_event(0, stored(worker, 1, &[1, 2]))
            .await
            .unwrap();
        handle.flush().await.unwrap();
        let snapshot = handle.snapshot().await.unwrap();

        assert_eq!(snapshot.stats.aggregation().unique_block_count(), 2);
        assert_eq!(
            snapshot.buckets.len(),
            snapshot.identity.format().bucket_count()
        );
        assert_eq!(handle.mailbox_capacity(), DEFAULT_MAILBOX_CAPACITY);
        assert!(
            handle
                .diagnostics
                .0
                .counters
                .unchanged_publications
                .load(Ordering::Relaxed)
                > 0
        );
    }

    #[tokio::test]
    async fn admission_completes_before_a_paused_actor_applies_the_event() {
        let worker = WorkerWithDpRank::new(1, 0);
        let (handle, _faults) =
            KvDcRelayHandle::spawn_with_capacity(CkfConfig::new(32), scope("admit"), 4).unwrap();
        let release = pause_actor(&handle).await;

        tokio::time::timeout(
            Duration::from_millis(50),
            handle.admit_event(0, stored(worker, 1, &[1])),
        )
        .await
        .expect("queue admission should not await CKF mutation")
        .unwrap();
        assert_eq!(handle.mailbox_depth(), 1);

        release.send(()).unwrap();
        handle.flush().await.unwrap();
        let (stats, _, _) = handle.state_stats().await.unwrap();
        assert_eq!(stats.aggregation().unique_block_count(), 1);
    }

    #[tokio::test]
    async fn bounded_mailbox_backpressures_before_admission_without_dropping_commands() {
        let worker = WorkerWithDpRank::new(1, 0);
        let (handle, _faults) =
            KvDcRelayHandle::spawn_with_capacity(CkfConfig::new(32), scope("backpressure"), 1)
                .unwrap();
        let release = pause_actor(&handle).await;
        handle
            .admit_event(0, stored(worker, 1, &[1]))
            .await
            .unwrap();

        let second_handle = handle.clone();
        let mut second =
            tokio::spawn(
                async move { second_handle.admit_event(0, stored(worker, 2, &[2])).await },
            );
        assert!(
            tokio::time::timeout(Duration::from_millis(20), &mut second)
                .await
                .is_err(),
            "the second command should wait for bounded mailbox capacity"
        );

        release.send(()).unwrap();
        second.await.unwrap().unwrap();
        handle.flush().await.unwrap();
        let (stats, _, _) = handle.state_stats().await.unwrap();
        assert_eq!(stats.aggregation().unique_block_count(), 2);
    }

    #[tokio::test]
    async fn subscriber_gets_snapshot_then_one_atomic_replacement_delta() {
        let worker = WorkerWithDpRank::new(1, 0);
        let (handle, _faults) =
            KvDcRelayHandle::spawn(CkfConfig::new(32), scope("replace")).unwrap();
        handle
            .admit_event(0, stored(worker, 1, &[1, 2]))
            .await
            .unwrap();
        handle.flush().await.unwrap();
        let mut subscription = handle.subscribe(lease(1)).await.unwrap();
        let base_sequence = subscription.snapshot.sequence();

        handle
            .replace_rank(
                0,
                worker.worker_id,
                worker.dp_rank,
                vec![stored(worker, 0, &[3, 4])],
            )
            .await
            .unwrap();
        let delta = subscription.deltas.recv().await.unwrap();
        assert_eq!(delta.base_sequence(), base_sequence);
        assert_eq!(delta.sequence(), base_sequence + 1);
    }

    #[tokio::test]
    async fn initial_rank_recoveries_share_one_transactional_pool_rebuild() {
        let first = WorkerWithDpRank::new(1, 0);
        let second = WorkerWithDpRank::new(2, 0);
        let expected = [first, second].into_iter().collect();
        let (handle, _faults) =
            KvDcRelayHandle::spawn(CkfConfig::new(32), scope("batch-replace")).unwrap();
        let target = KvDcRelayRecoveryTarget {
            handle: handle.clone(),
            rebuild_permit: Arc::new(Semaphore::new(1)),
            replacement_batcher: KvDcRelayRecoveryTarget::new_replacement_batcher(
                expected,
                Duration::from_millis(100),
            ),
        };

        let first_target = target.clone();
        let mut first_replacement = tokio::spawn(async move {
            first_target
                .replace_rank(
                    1,
                    first.worker_id,
                    first.dp_rank,
                    vec![stored(first, 0, &[1, 2])],
                )
                .await
        });
        assert!(
            tokio::time::timeout(Duration::from_millis(20), &mut first_replacement)
                .await
                .is_err(),
            "the first cold-start rank must wait for the recovery wave"
        );
        target
            .replace_rank(
                1,
                second.worker_id,
                second.dp_rank,
                vec![stored(second, 0, &[3])],
            )
            .await
            .unwrap();
        first_replacement.await.unwrap().unwrap();

        let (stats, _, members) = handle.state_stats().await.unwrap();
        assert_eq!(stats.aggregation().unique_block_count(), 3);
        assert_eq!(members.len(), 2);
    }

    #[tokio::test]
    async fn shutdown_drains_admitted_events_and_rejects_new_admission() {
        let worker = WorkerWithDpRank::new(1, 0);
        let (handle, _faults) =
            KvDcRelayHandle::spawn_with_capacity(CkfConfig::new(32), scope("shutdown"), 4).unwrap();
        let release = pause_actor(&handle).await;
        handle
            .admit_event(0, stored(worker, 1, &[1]))
            .await
            .unwrap();
        let shutdown_handle = handle.clone();
        let shutdown = tokio::spawn(async move { shutdown_handle.shutdown().await });
        release.send(()).unwrap();

        shutdown.await.unwrap().unwrap();
        assert!(matches!(
            handle.admit_event(0, stored(worker, 2, &[2])).await,
            Err(KvDcRelayError::ShuttingDown)
        ));
    }

    #[tokio::test]
    async fn producer_fence_retires_stream_without_publishing_uncertain_tail() {
        let worker = WorkerWithDpRank::new(1, 0);
        let mut config = CkfConfig::new(32);
        config.publish_every_n_events = 16;
        let (handle, _faults) = KvDcRelayHandle::spawn_with_publication_delay(
            config,
            scope("fence"),
            Duration::from_secs(10),
        )
        .unwrap();
        let mut subscription = handle.subscribe(lease(1)).await.unwrap();
        handle
            .admit_event(0, stored(worker, 1, &[1]))
            .await
            .unwrap();

        handle.fence().await.unwrap();
        assert!(matches!(
            subscription.deltas.recv().await,
            Err(broadcast::error::RecvError::Closed)
        ));
        assert!(matches!(
            handle.admit_event(0, stored(worker, 2, &[2])).await,
            Err(KvDcRelayError::ShuttingDown)
        ));
    }

    #[tokio::test]
    async fn producer_fence_interrupts_a_full_fault_channel() {
        let worker = WorkerWithDpRank::new(1, 0);
        let (handle, faults) =
            KvDcRelayHandle::spawn(CkfConfig::new(32), scope("fault-backpressure")).unwrap();
        let disposition = event_failure_point(KvCacheEventError::ParentBlockNotFound).disposition();

        for event_id in 1..=(DEFAULT_FAULT_CAPACITY as u64 + 1) {
            handle
                .sender
                .send(ActorCommand::InjectFault {
                    fault: ActorFault {
                        worker_id: worker.worker_id,
                        dp_rank: worker.dp_rank,
                        publisher_id: 100,
                        event_id: Some(event_id),
                        category: ActorFaultCategory::SourceProtocol,
                        disposition,
                        message: format!("fault {event_id}"),
                    },
                })
                .await
                .unwrap();
        }
        tokio::time::timeout(Duration::from_secs(1), async {
            while faults.len() != DEFAULT_FAULT_CAPACITY || handle.mailbox_depth() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("actor must block after filling the fault channel");

        tokio::time::timeout(Duration::from_secs(1), handle.fence())
            .await
            .expect("fence must interrupt a blocked fault send")
            .unwrap();
    }

    #[tokio::test]
    async fn cadence_advances_on_duplicate_events_without_acknowledging_mutation() {
        let worker = WorkerWithDpRank::new(1, 0);
        let mut config = CkfConfig::new(32);
        config.publish_every_n_events = 16;
        let (handle, _faults) = KvDcRelayHandle::spawn(config, scope("cadence")).unwrap();
        let mut subscription = handle.subscribe(lease(1)).await.unwrap();

        for event_id in 1..=15 {
            handle
                .admit_event(0, stored(worker, event_id, &[7]))
                .await
                .unwrap();
        }
        let (stats, sequence, _) = handle.state_stats().await.unwrap();
        assert_eq!(sequence, 0);
        assert_eq!(stats.publication().pending_events(), 15);
        assert!(subscription.deltas.try_recv().is_err());

        handle
            .admit_event(0, stored(worker, 16, &[7]))
            .await
            .unwrap();
        let delta = subscription.deltas.recv().await.unwrap();
        assert_eq!(delta.base_sequence(), 0);
        assert_eq!(delta.sequence(), 1);
        let (stats, _, _) = handle.state_stats().await.unwrap();
        assert_eq!(stats.publication().pending_events(), 0);
        assert_eq!(stats.aggregation().unique_block_count(), 1);
    }

    #[tokio::test]
    async fn publication_timer_emits_a_sparse_tail_without_flush() {
        let worker = WorkerWithDpRank::new(1, 0);
        let mut config = CkfConfig::new(32);
        config.publish_every_n_events = 16;
        let (handle, _faults) = KvDcRelayHandle::spawn_with_publication_delay(
            config,
            scope("timer"),
            Duration::from_millis(1),
        )
        .unwrap();
        let mut subscription = handle.subscribe(lease(1)).await.unwrap();

        handle
            .admit_event(0, stored(worker, 1, &[7]))
            .await
            .unwrap();
        let delta = tokio::time::timeout(Duration::from_millis(100), subscription.deltas.recv())
            .await
            .expect("the 1 ms timer must publish a sparse dirty tail")
            .unwrap();

        assert_eq!((delta.base_sequence(), delta.sequence()), (0, 1));
        assert_eq!(handle.state_stats().await.unwrap().1, 1);
    }

    #[tokio::test]
    async fn replacement_subscription_starts_after_the_old_lease_tail() {
        let worker = WorkerWithDpRank::new(1, 0);
        let mut config = CkfConfig::new(32);
        config.publish_every_n_events = 16;
        let (handle, _faults) = KvDcRelayHandle::spawn_with_publication_delay(
            config,
            scope("subscription-tail"),
            Duration::from_secs(10),
        )
        .unwrap();
        let mut old = handle.subscribe(lease(1)).await.unwrap();
        handle
            .admit_event(0, stored(worker, 1, &[7]))
            .await
            .unwrap();

        let mut replacement = handle.subscribe(lease(2)).await.unwrap();
        let old_tail = old.deltas.recv().await.unwrap();
        assert_eq!(old_tail.lease(), lease(1));
        assert_eq!(replacement.snapshot.sequence(), old_tail.sequence());
        assert!(replacement.deltas.try_recv().is_err());

        handle
            .admit_event(0, stored(worker, 2, &[8]))
            .await
            .unwrap();
        handle.flush().await.unwrap();
        let continuation = replacement.deltas.recv().await.unwrap();
        assert_eq!(continuation.lease(), lease(2));
        assert_eq!(
            continuation.base_sequence(),
            replacement.snapshot.sequence()
        );
        assert_eq!(continuation.sequence(), replacement.snapshot.sequence() + 1);
    }

    #[tokio::test]
    async fn capacity_omission_is_observable_without_a_lifecycle_fault() {
        let worker = WorkerWithDpRank::new(1, 0);
        let mut config = CkfConfig::new(1);
        config.max_kicks = 1;
        let (handle, mut faults) = KvDcRelayHandle::spawn(config, scope("fault")).unwrap();
        let hashes: Vec<_> = (1..=32).collect();

        handle
            .admit_event(0, stored(worker, 1, &hashes))
            .await
            .unwrap();
        let (stats, _, _) = handle.state_stats().await.unwrap();
        assert!(stats.aggregation().unique_block_count() > 0);
        assert!(stats.aggregation().capacity_failures() > 0);

        assert!(
            tokio::time::timeout(Duration::from_millis(20), faults.recv())
                .await
                .is_err(),
            "a pre-commit capacity omission must not enter lifecycle fault handling"
        );

        handle
            .admit_event(0, stored(worker, 2, &[1]))
            .await
            .unwrap();
        handle.flush().await.unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(20), faults.recv())
                .await
                .is_err(),
            "later work must remain live without delayed capacity lifecycle faults"
        );
    }

    #[tokio::test]
    async fn failed_replacement_returns_barrier_error_without_replaying_or_faulting() {
        let worker = WorkerWithDpRank::new(1, 0);
        let foreign = WorkerWithDpRank::new(2, 0);
        let (handle, mut faults) =
            KvDcRelayHandle::spawn(CkfConfig::new(32), scope("barrier-fault")).unwrap();

        assert!(
            handle
                .replace_rank(
                    0,
                    worker.worker_id,
                    worker.dp_rank,
                    vec![stored(foreign, 1, &[1])],
                )
                .await
                .is_err()
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(20), faults.recv())
                .await
                .is_err(),
            "a replacement build failure before swap leaves the old generation unchanged"
        );
    }

    #[test]
    fn event_failures_keep_commit_domains_distinct() {
        let capacity = event_failure_point(KvCacheEventError::CapacityExhausted).disposition();
        assert_eq!(capacity.action, CkfFailureAction::ContinueCapacityOmission);
        assert_eq!(capacity.domain, CkfFailureDomain::ProducerCore);
        assert_eq!(capacity.commit, CkfCommitState::KnownUnchanged);
        assert_eq!(capacity.recovery_domain, None);

        let allocation = event_failure_point(KvCacheEventError::AllocationFailed).disposition();
        assert_eq!(allocation.action, CkfFailureAction::ReportResourceFailure);
        assert_eq!(allocation.commit, CkfCommitState::KnownUnchanged);

        let source = event_failure_point(KvCacheEventError::OwnershipDegreeOverflow).disposition();
        assert_eq!(source.action, CkfFailureAction::RejectSource);
        assert_eq!(source.commit, CkfCommitState::KnownUnchanged);

        let missing_parent =
            event_failure_point(KvCacheEventError::ParentBlockNotFound).disposition();
        assert_eq!(missing_parent.action, CkfFailureAction::RejectSource);
        assert_eq!(missing_parent.domain, CkfFailureDomain::ProducerCore);
        assert_eq!(missing_parent.commit, CkfCommitState::KnownUnchanged);

        let invariant =
            event_failure_point(KvCacheEventError::IndexerInvariantViolation).disposition();
        assert_eq!(invariant.action, CkfFailureAction::FenceAndRebuildProducer);
        assert_eq!(invariant.commit, CkfCommitState::KnownUnchanged);
        assert_eq!(
            invariant.recovery_domain,
            Some(CkfFailureDomain::ProducerCore)
        );
    }

    #[tokio::test]
    async fn worker_query_replacement_orders_old_apply_before_reset_and_fences_old_source() {
        let worker = WorkerWithDpRank::new(1, 0);
        let serving_endpoint = EndpointId::from("ns.worker.ordering");
        let kv_state_endpoint = EndpointId::from("ns.worker.kv");
        let source = |publisher_id| KvEventSource {
            kv_state_endpoint: kv_state_endpoint.clone(),
            worker,
            publisher_id,
            recovery_target: None,
        };
        let view = |source| KvSourceMembershipView {
            serving_endpoint: serving_endpoint.clone(),
            endpoint_resolution: KvStateEndpointResolution::Resolved(kv_state_endpoint.clone()),
            sources: HashMap::from([(worker, KvSourceStatus::ActiveLiveOnly(source))]),
            kv_event_publishing_enabled: HashMap::new(),
            kv_event_source_mode: HashMap::new(),
            recovery_expected: HashMap::new(),
        };

        let mut config = CkfConfig::new(32);
        config.publish_every_n_events = 1;
        let (handle, mut faults) = KvDcRelayHandle::spawn_with_publication_delay(
            config,
            scope("worker-query-ordering"),
            Duration::from_secs(10),
        )
        .unwrap();
        let mut subscription = handle.subscribe(lease(1)).await.unwrap();
        let target = KvDcRelayRecoveryTarget::new(
            handle.clone(),
            Arc::new(Semaphore::new(1)),
            HashSet::new(),
            Duration::from_secs(1),
        );
        let source_a = source(100);
        let source_b = source(200);
        let (membership_tx, membership_rx) = watch::channel(view(source_a));
        let client = WorkerQueryClient::new_target_for_test(
            target,
            membership_rx,
            Arc::new(UnusedWorkerQueryTransport),
        );
        client.sync_membership().await;

        let release = pause_actor(&handle).await;
        client
            .handle_live_batch(100, vec![stored(worker, 1, &[1])])
            .await;
        membership_tx.send(view(source_b)).unwrap();
        let sync_client = client.clone();
        let replacement = tokio::spawn(async move { sync_client.sync_membership().await });
        tokio::time::timeout(Duration::from_secs(1), async {
            while handle.mailbox_depth() < 2 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("A apply and A reset were not both accepted by the actor");
        assert!(!replacement.is_finished());

        release.send(()).unwrap();
        replacement.await.unwrap();
        let applied = subscription.deltas.recv().await.unwrap();
        let reset = subscription.deltas.recv().await.unwrap();
        assert_eq!((applied.base_sequence(), applied.sequence()), (0, 1));
        assert_eq!((reset.base_sequence(), reset.sequence()), (1, 2));

        client
            .handle_live_batch(100, vec![stored(worker, 2, &[2])])
            .await;
        client
            .handle_live_batch(200, vec![stored(worker, 1, &[3])])
            .await;
        handle.flush().await.unwrap();
        let (stats, _, _) = handle.state_stats().await.unwrap();
        assert_eq!(stats.aggregation().unique_block_count(), 1);

        assert!(
            tokio::time::timeout(Duration::from_millis(20), faults.recv())
                .await
                .is_err(),
            "stale traffic must not fault the replacement source"
        );
    }
}
