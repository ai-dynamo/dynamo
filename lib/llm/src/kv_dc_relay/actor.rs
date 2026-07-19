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

#[cfg(any(test, feature = "ckf-diagnostics"))]
use dynamo_kv_router::indexer::cuckoo::DcCkfStats;
#[cfg(feature = "ckf-diagnostics")]
use dynamo_kv_router::indexer::cuckoo::PublisherEmitOutcome;
use dynamo_kv_router::indexer::cuckoo::{
    CkfConfig, CkfFailureAction, CkfFailureDisposition, CkfFailurePoint, DcCkfDelta,
    DcCkfDeltaSink, DcCkfPublisher, DcCkfSnapshot, DcCkfState, LaneLease, ProducerIdentity,
};
use dynamo_kv_router::protocols::{
    DpRank, ExternalSequenceBlockHash, KvCacheEventData, KvCacheEventError, RouterEvent,
    StorageTier, WorkerId, WorkerWithDpRank,
};
#[cfg(feature = "ckf-diagnostics")]
use parking_lot::Mutex;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, broadcast, mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::kv_router::indexer::{RecoveryResetReason, RecoveryTarget, SourceEpoch};

use super::host::KvDcRelayError;
use super::resolution::PoolBinding;

const DEFAULT_MAILBOX_CAPACITY: usize = 256;
const DEFAULT_PENDING_BLOCK_PERMITS: usize = 65_536;
const DEFAULT_PUBLICATION_CAPACITY: usize = 64;
const DEFAULT_FAULT_CAPACITY: usize = 16;
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
    pub(super) source_epoch: SourceEpoch,
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

#[derive(Debug, Clone)]
pub(super) struct StreamScope {
    pub(super) process_incarnation: u64,
    pub(super) layout_generation: u64,
    pub(super) pool_binding: PoolBinding,
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
    payload_permits: Arc<Semaphore>,
    fence: CancellationToken,
    stopped: CancellationToken,
    #[cfg(feature = "ckf-diagnostics")]
    pub(super) diagnostics: ActorDiagnosticsHandle,
    pub(super) scope: StreamScope,
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

    #[cfg(test)]
    fn spawn_with_capacity(
        config: CkfConfig,
        scope: StreamScope,
        capacity: usize,
    ) -> Result<(Self, mpsc::Receiver<ActorFault>), KvDcRelayError> {
        Self::spawn_with_capacity_and_delay(config, scope, capacity, DEFAULT_PUBLICATION_DELAY)
    }

    fn spawn_with_capacity_and_delay(
        config: CkfConfig,
        scope: StreamScope,
        capacity: usize,
        publication_delay: Duration,
    ) -> Result<(Self, mpsc::Receiver<ActorFault>), KvDcRelayError> {
        let state = DcCkfState::new(config)?;
        let (sender, receiver) = mpsc::channel(capacity);
        let (publication_tx, _) = broadcast::channel(DEFAULT_PUBLICATION_CAPACITY);
        let identity = ProducerIdentity::new(
            scope.pool_binding.pool_id(),
            scope.process_incarnation,
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
        Ok((
            Self {
                sender,
                payload_permits: Arc::new(Semaphore::new(DEFAULT_PENDING_BLOCK_PERMITS)),
                fence,
                stopped,
                #[cfg(feature = "ckf-diagnostics")]
                diagnostics,
                scope,
            },
            fault_rx,
        ))
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
        source_epoch: SourceEpoch,
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
                source_epoch,
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
        source_epoch: SourceEpoch,
        worker_id: WorkerId,
        dp_rank: DpRank,
        events: Vec<RouterEvent>,
    ) -> Result<(), KvDcRelayError> {
        self.replace_ranks(vec![RankReplacement {
            source_epoch,
            worker_id,
            dp_rank,
            events,
        }])
        .await
    }

    async fn reset_rank(
        &self,
        source_epoch: SourceEpoch,
        worker_id: WorkerId,
        dp_rank: DpRank,
        degraded: bool,
    ) -> Result<(), KvDcRelayError> {
        self.submit(|response| ActorCommand::ResetRank {
            source_epoch,
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
    source_epoch: SourceEpoch,
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
        source_epoch: SourceEpoch,
        event: RouterEvent,
    ) -> anyhow::Result<()> {
        self.handle
            .admit_event(source_epoch, event)
            .await
            .map_err(Into::into)
    }

    async fn replace_rank(
        &self,
        source_epoch: SourceEpoch,
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
                    source_epoch,
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
        source_epoch: SourceEpoch,
        worker_id: WorkerId,
        dp_rank: DpRank,
        reason: RecoveryResetReason,
    ) -> anyhow::Result<()> {
        self.handle
            .reset_rank(
                source_epoch,
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
        source_epoch: SourceEpoch,
        event: RouterEvent,
        _payload_permit: OwnedSemaphorePermit,
    },
    ReplaceRanks {
        replacements: Vec<RankReplacement>,
        _payload_permit: OwnedSemaphorePermit,
        response: oneshot::Sender<Result<(), KvDcRelayError>>,
    },
    ResetRank {
        source_epoch: SourceEpoch,
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
    let mut source_epochs = HashMap::<WorkerWithDpRank, SourceEpoch>::new();
    let mut capacity_omission_events = 0u64;
    let mut shutdown_response = None;
    let mut discard_tail = false;
    let mut publication_deadline = None;
    loop {
        let command = if let Some(deadline) = publication_deadline {
            tokio::select! {
                biased;
                _ = fence.cancelled() => {
                    discard_tail = true;
                    break;
                }
                command = receiver.recv() => command,
                _ = tokio::time::sleep_until(deadline) => {
                    publication_deadline = None;
                    if state.has_pending_publication()
                        && let Err(error) = publish_pending(&mut state, &mut publisher, &diagnostics)
                    {
                        diagnostics.record_error(&error);
                    }
                    continue;
                }
            }
        } else {
            tokio::select! {
                biased;
                _ = fence.cancelled() => {
                    discard_tail = true;
                    break;
                }
                command = receiver.recv() => command,
            }
        };
        let Some(command) = command else {
            break;
        };
        diagnostics.start_command(&command);
        match command {
            ActorCommand::Apply {
                source_epoch,
                event,
                ..
            } => {
                let worker_id = event.worker_id;
                let dp_rank = event.event.dp_rank;
                let event_id = event.event.event_id;
                let key = WorkerWithDpRank::new(worker_id, dp_rank);
                let current_epoch = source_epochs.get(&key).copied();
                if current_epoch.is_some_and(|current| source_epoch < current) {
                    tracing::debug!(
                        worker_id,
                        dp_rank,
                        event_id,
                        source_epoch = source_epoch.get(),
                        current_epoch = current_epoch.expect("guarded current epoch").get(),
                        "Dropping an admitted KV mutation from a superseded source epoch"
                    );
                    diagnostics.finish_command();
                    continue;
                }
                if let Some(current_epoch) = current_epoch
                    && source_epoch > current_epoch
                {
                    let disposition = CkfFailurePoint::SourceProtocolFailure.disposition();
                    let message = format!(
                        "source epoch advanced from {} to {} without a reset or replacement barrier",
                        current_epoch.get(),
                        source_epoch.get()
                    );
                    diagnostics.record_error(&message);
                    if fault_tx
                        .send(ActorFault {
                            worker_id,
                            dp_rank,
                            source_epoch,
                            event_id: Some(event_id),
                            category: actor_fault_category(disposition),
                            disposition,
                            message,
                        })
                        .await
                        .is_err()
                    {
                        break;
                    }
                    diagnostics.finish_command();
                    continue;
                }
                source_epochs.entry(key).or_insert(source_epoch);
                let outcome = state.apply_event(event);
                let first_error = outcome.first_error().copied();
                let publication_boundary = outcome.publication_boundary();
                if outcome.unknown_removals() != 0 {
                    tracing::warn!(
                        worker_id,
                        dp_rank,
                        event_id,
                        unknown_removals = outcome.unknown_removals(),
                        "Ignoring KV DC Relay removals not owned by this worker/rank"
                    );
                }
                if let Some(batch) = outcome.into_publication() {
                    if let Err(error) = publish_batch(batch, &mut publisher, &diagnostics) {
                        diagnostics.record_error(&error);
                    }
                } else if publication_boundary {
                    diagnostics.record_no_publication();
                }
                if publication_boundary {
                    publication_deadline = None;
                } else if state.pending_event_count() == 1 && publication_deadline.is_none() {
                    publication_deadline = Some(tokio::time::Instant::now() + publication_delay);
                }
                if let Some(error) = first_error {
                    let disposition = event_failure_point(error).disposition();
                    if disposition.action == CkfFailureAction::ContinueCapacityOmission {
                        // NOTE: A bounded relocation miss is a pre-commit lossy-index omission.
                        // Do not turn it into a lifecycle fault: the affected block is unchanged,
                        // successful sibling blocks remain committed, a later Store may retry, and
                        // an omitted hash's Remove remains a safe no-op.
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
                    if fault_tx
                        .send(ActorFault {
                            worker_id,
                            dp_rank,
                            source_epoch,
                            event_id: Some(event_id),
                            category,
                            disposition,
                            message,
                        })
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
            }
            ActorCommand::ReplaceRanks {
                replacements,
                _payload_permit: _,
                response,
            } => {
                let stale = replacements.iter().find_map(|replacement| {
                    let key = WorkerWithDpRank::new(replacement.worker_id, replacement.dp_rank);
                    let current = source_epochs.get(&key).copied()?;
                    (replacement.source_epoch < current).then_some((replacement, current))
                });
                if let Some((replacement, current)) = stale {
                    let _ = response.send(Err(KvDcRelayError::StaleSourceEpoch {
                        worker_id: replacement.worker_id,
                        dp_rank: replacement.dp_rank,
                        current: current.get(),
                        received: replacement.source_epoch.get(),
                    }));
                    diagnostics.finish_command();
                    continue;
                }
                #[cfg(feature = "ckf-diagnostics")]
                let rebuild_started = Instant::now();
                let mut committed_epochs = Vec::with_capacity(replacements.len());
                let result = replacement_batch_hashes(replacements, &mut committed_epochs)
                    .and_then(|hashes| state.replace_ranks(hashes).map_err(Into::into))
                    .and_then(|publication| {
                        if let Some(batch) = publication {
                            publish_batch(batch, &mut publisher, &diagnostics)?;
                        } else {
                            diagnostics.record_no_publication();
                        }
                        Ok(())
                    });
                if result.is_ok() {
                    source_epochs.extend(committed_epochs);
                }
                #[cfg(feature = "ckf-diagnostics")]
                diagnostics.record_rebuild(rebuild_started);
                // The whole cold-start batch is built off-side. A pre-swap failure leaves every
                // prior rank unchanged; the strong responses all observe the same atomic result.
                let _ = response.send(result);
            }
            ActorCommand::ResetRank {
                source_epoch,
                worker_id,
                dp_rank,
                degraded,
                response,
            } => {
                let key = WorkerWithDpRank::new(worker_id, dp_rank);
                if let Some(current) = source_epochs.get(&key).copied()
                    && source_epoch < current
                {
                    let _ = response.send(Err(KvDcRelayError::StaleSourceEpoch {
                        worker_id,
                        dp_rank,
                        current: current.get(),
                        received: source_epoch.get(),
                    }));
                    diagnostics.finish_command();
                    continue;
                }
                let mut removal = state.remove_rank(key);
                if let Err(error) = removal {
                    // Clear may have committed earlier hashes while remaining exact. Retry the
                    // still-tracked suffix once; the strong acknowledgement reports failure if
                    // progress cannot be completed.
                    tracing::warn!(
                        worker_id,
                        dp_rank,
                        source_epoch = source_epoch.get(),
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
                if result.is_ok() {
                    source_epochs.insert(key, source_epoch);
                }
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
        }
        if !state.has_pending_publication() {
            publication_deadline = None;
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

fn replacement_hashes(
    worker_id: WorkerId,
    dp_rank: DpRank,
    events: Vec<RouterEvent>,
) -> Result<HashSet<ExternalSequenceBlockHash>, KvDcRelayError> {
    let mut hashes = HashSet::new();
    for event in events {
        if event.worker_id != worker_id || event.event.dp_rank != dp_rank {
            return Err(KvDcRelayError::InvalidTreeDump {
                worker_id,
                dp_rank,
                message: "event identity does not match replacement rank".to_string(),
            });
        }
        if event.storage_tier != StorageTier::Device {
            continue;
        }
        let KvCacheEventData::Stored(store) = event.event.data else {
            return Err(KvDcRelayError::InvalidTreeDump {
                worker_id,
                dp_rank,
                message: "tree dump contains a non-Stored event".to_string(),
            });
        };
        hashes.try_reserve(store.blocks.len()).map_err(|_| {
            KvDcRelayError::Build(
                dynamo_kv_router::indexer::cuckoo::CkfBuildError::AllocationFailed,
            )
        })?;
        hashes.extend(store.blocks.into_iter().map(|block| block.block_hash));
    }
    Ok(hashes)
}

fn replacement_batch_hashes(
    replacements: Vec<RankReplacement>,
    committed_epochs: &mut Vec<(WorkerWithDpRank, SourceEpoch)>,
) -> Result<HashMap<WorkerWithDpRank, HashSet<ExternalSequenceBlockHash>>, KvDcRelayError> {
    let mut hashes_by_rank = HashMap::new();
    for replacement in replacements {
        let member = WorkerWithDpRank::new(replacement.worker_id, replacement.dp_rank);
        if hashes_by_rank.contains_key(&member) {
            return Err(KvDcRelayError::InvalidTreeDump {
                worker_id: replacement.worker_id,
                dp_rank: replacement.dp_rank,
                message: "replacement batch contains the same rank more than once".to_string(),
            });
        }
        let hashes = replacement_hashes(
            replacement.worker_id,
            replacement.dp_rank,
            replacement.events,
        )?;
        hashes_by_rank.insert(member, hashes);
        committed_epochs.push((member, replacement.source_epoch));
    }
    Ok(hashes_by_rank)
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
    use std::time::Duration;

    use dynamo_kv_router::identity::{
        CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, PoolId, RoutingScopeId,
    };
    use dynamo_kv_router::indexer::cuckoo::{CkfCommitState, CkfFailureDomain, ConsumerInstanceId};
    use dynamo_kv_router::protocols::{
        KvCacheEvent, KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash,
    };
    use dynamo_runtime::protocols::EndpointId;

    use super::*;
    use crate::kv_dc_relay::resolution::EndpointLocator;

    fn scope(name: &str) -> StreamScope {
        let endpoint = format!("ns.worker.{name}");
        let endpoint_id = EndpointId::from(endpoint.as_str());
        let dc_id = DcId::new(2);
        let domain = IndexerDomainId::new(
            CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
            RoutingScopeId::new([3; 16], IdentitySource::Explicit),
        );
        StreamScope {
            process_incarnation: 1,
            layout_generation: 1,
            pool_binding: PoolBinding::new(
                PoolId::new(domain, dc_id),
                EndpointLocator::new(dc_id, endpoint_id),
                None,
            ),
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
                            block_hash: ExternalSequenceBlockHash(hash),
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
            .admit_event(SourceEpoch::new(0), stored(worker, 1, &[1, 2]))
            .await
            .unwrap();
        handle.flush().await.unwrap();
        let snapshot = handle.snapshot().await.unwrap();

        assert_eq!(snapshot.stats.aggregation().unique_block_count(), 2);
        assert_eq!(
            snapshot.buckets.len(),
            snapshot.identity.format().bucket_count()
        );
        assert_eq!(
            actor_health(&handle).mailbox_capacity,
            DEFAULT_MAILBOX_CAPACITY
        );
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
            handle.admit_event(SourceEpoch::new(0), stored(worker, 1, &[1])),
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
            .admit_event(SourceEpoch::new(0), stored(worker, 1, &[1]))
            .await
            .unwrap();

        let second_handle = handle.clone();
        let mut second = tokio::spawn(async move {
            second_handle
                .admit_event(SourceEpoch::new(0), stored(worker, 2, &[2]))
                .await
        });
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
            .admit_event(SourceEpoch::new(0), stored(worker, 1, &[1, 2]))
            .await
            .unwrap();
        handle.flush().await.unwrap();
        let mut subscription = handle.subscribe(lease(1)).await.unwrap();
        let base_sequence = subscription.snapshot.sequence();

        handle
            .replace_rank(
                SourceEpoch::new(0),
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
                    SourceEpoch::new(1),
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
                SourceEpoch::new(1),
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
            .admit_event(SourceEpoch::new(0), stored(worker, 1, &[1]))
            .await
            .unwrap();
        let shutdown_handle = handle.clone();
        let shutdown = tokio::spawn(async move { shutdown_handle.shutdown().await });
        release.send(()).unwrap();

        shutdown.await.unwrap().unwrap();
        assert!(matches!(
            handle
                .admit_event(SourceEpoch::new(0), stored(worker, 2, &[2]))
                .await,
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
            .admit_event(SourceEpoch::new(0), stored(worker, 1, &[1]))
            .await
            .unwrap();

        handle.fence().await.unwrap();
        assert!(matches!(
            subscription.deltas.recv().await,
            Err(broadcast::error::RecvError::Closed)
        ));
        assert!(matches!(
            handle
                .admit_event(SourceEpoch::new(0), stored(worker, 2, &[2]))
                .await,
            Err(KvDcRelayError::ShuttingDown)
        ));
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
                .admit_event(SourceEpoch::new(0), stored(worker, event_id, &[7]))
                .await
                .unwrap();
        }
        let (stats, sequence, _) = handle.state_stats().await.unwrap();
        assert_eq!(sequence, 0);
        assert_eq!(stats.publication().pending_events(), 15);
        assert!(subscription.deltas.try_recv().is_err());

        handle
            .admit_event(SourceEpoch::new(0), stored(worker, 16, &[7]))
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
            .admit_event(SourceEpoch::new(0), stored(worker, 1, &[7]))
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
            .admit_event(SourceEpoch::new(0), stored(worker, 1, &[7]))
            .await
            .unwrap();

        let mut replacement = handle.subscribe(lease(2)).await.unwrap();
        let old_tail = old.deltas.recv().await.unwrap();
        assert_eq!(old_tail.lease(), lease(1));
        assert_eq!(replacement.snapshot.sequence(), old_tail.sequence());
        assert!(replacement.deltas.try_recv().is_err());

        handle
            .admit_event(SourceEpoch::new(0), stored(worker, 2, &[8]))
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
            .admit_event(SourceEpoch::new(0), stored(worker, 1, &hashes))
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
            .admit_event(SourceEpoch::new(0), stored(worker, 2, &[1]))
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
                    SourceEpoch::new(0),
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
    async fn stale_source_epoch_cannot_mutate_or_fault_a_replacement_rank() {
        let worker = WorkerWithDpRank::new(1, 0);
        let (handle, mut faults) =
            KvDcRelayHandle::spawn(CkfConfig::new(32), scope("stale-epoch")).unwrap();

        handle
            .admit_event(SourceEpoch::new(1), stored(worker, 1, &[1]))
            .await
            .unwrap();
        handle
            .reset_rank(SourceEpoch::new(2), worker.worker_id, worker.dp_rank, false)
            .await
            .unwrap();

        handle
            .admit_event(SourceEpoch::new(1), stored(worker, 2, &[2]))
            .await
            .unwrap();
        handle.flush().await.unwrap();
        let (stats, _, _) = handle.state_stats().await.unwrap();
        assert_eq!(stats.aggregation().unique_block_count(), 0);

        assert!(matches!(
            handle
                .reset_rank(SourceEpoch::new(1), worker.worker_id, worker.dp_rank, false)
                .await,
            Err(KvDcRelayError::StaleSourceEpoch {
                current: 2,
                received: 1,
                ..
            })
        ));
        assert!(matches!(
            handle
                .replace_rank(
                    SourceEpoch::new(1),
                    worker.worker_id,
                    worker.dp_rank,
                    vec![stored(worker, 3, &[3])],
                )
                .await,
            Err(KvDcRelayError::StaleSourceEpoch {
                current: 2,
                received: 1,
                ..
            })
        ));
        assert!(
            tokio::time::timeout(Duration::from_millis(20), faults.recv())
                .await
                .is_err(),
            "stale traffic must not fault the replacement epoch"
        );
    }
}
