// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::sync::Arc;

use futures::future::select_all;
use parking_lot::Mutex;
use tokio::sync::{OwnedSemaphorePermit, watch};
use uuid::Uuid;

/// An offload pipeline whose executor can participate in causal settlement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineLane {
    G1ToG2,
    G2ToG3,
    G2ToG4,
}

impl PipelineLane {
    pub(crate) const ALL: [Self; 3] = [Self::G1ToG2, Self::G2ToG3, Self::G2ToG4];

    pub(crate) const fn index(self) -> usize {
        match self {
            Self::G1ToG2 => 0,
            Self::G2ToG3 => 1,
            Self::G2ToG4 => 2,
        }
    }
}

impl fmt::Display for PipelineLane {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::G1ToG2 => f.write_str("G1→G2"),
            Self::G2ToG3 => f.write_str("G2→G3"),
            Self::G2ToG4 => f.write_str("G2→G4"),
        }
    }
}

/// Terminal executor failure observed while waiting for causal settlement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineFailure {
    kind: PipelineFailureKind,
    message: String,
}

impl PipelineFailure {
    pub(crate) fn new(kind: PipelineFailureKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// Failure category.
    pub fn kind(&self) -> PipelineFailureKind {
        self.kind
    }

    /// Diagnostic detail supplied by the failing executor.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for PipelineFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)
    }
}

/// Stable failure categories for pipeline settlement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineFailureKind {
    TaskAborted,
    TaskPanicked,
    Executor,
    Shutdown,
    CounterOverflow,
    Invariant,
}

impl fmt::Display for PipelineFailureKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TaskAborted => f.write_str("task aborted"),
            Self::TaskPanicked => f.write_str("task panicked"),
            Self::Executor => f.write_str("executor failure"),
            Self::Shutdown => f.write_str("pipeline shutdown"),
            Self::CounterOverflow => f.write_str("counter overflow"),
            Self::Invariant => f.write_str("pipeline invariant violation"),
        }
    }
}

/// Opaque checkpoint captured before an external completion source is fired.
#[derive(Debug, Clone)]
pub struct SettlementToken {
    pub(crate) engine_id: Uuid,
    pub(crate) checkpoints: [Option<PipelineCheckpoint>; 3],
}

impl SettlementToken {
    pub(crate) fn validate_engine(&self, engine_id: Uuid) -> Result<(), SettlementError> {
        if self.engine_id == engine_id {
            Ok(())
        } else {
            Err(SettlementError::ForeignToken)
        }
    }

    pub(crate) fn checkpoint(
        &self,
        lane: PipelineLane,
    ) -> Result<PipelineCheckpoint, SettlementError> {
        self.checkpoints[lane.index()].ok_or(SettlementError::LaneUnavailable { lane })
    }
}

/// Expected completed transfer batches for each pipeline lane.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SettlementTarget {
    completed_batches: [u64; 3],
}

impl SettlementTarget {
    /// Create an empty target. Empty targets settle immediately.
    pub fn new() -> Self {
        Self::default()
    }

    /// Require `count` completed batches on `lane` after the checkpoint.
    pub fn with_completed_batches(mut self, lane: PipelineLane, count: u64) -> Self {
        self.completed_batches[lane.index()] = count;
        self
    }

    /// Add completed batches to a lane's requirement.
    pub fn add_completed_batches(
        &mut self,
        lane: PipelineLane,
        count: u64,
    ) -> Result<(), SettlementError> {
        let value = &mut self.completed_batches[lane.index()];
        *value = value
            .checked_add(count)
            .ok_or(SettlementError::TargetOverflow { lane })?;
        Ok(())
    }

    pub(crate) fn completed_batches(self, lane: PipelineLane) -> u64 {
        self.completed_batches[lane.index()]
    }

    pub(crate) fn is_empty(self) -> bool {
        self.completed_batches.iter().all(|count| *count == 0)
    }
}

/// Error returned when a causal settlement boundary cannot be established.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SettlementError {
    ForeignToken,
    LaneUnavailable {
        lane: PipelineLane,
    },
    UnsupportedAutoChain {
        lane: PipelineLane,
    },
    StaleToken {
        lane: PipelineLane,
    },
    TargetOverflow {
        lane: PipelineLane,
    },
    ProgressClosed {
        lane: PipelineLane,
    },
    PipelineFailed {
        lane: PipelineLane,
        failure: PipelineFailure,
    },
}

impl fmt::Display for SettlementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ForeignToken => f.write_str("settlement token belongs to a different engine"),
            Self::LaneUnavailable { lane } => write!(f, "pipeline lane {lane} is not configured"),
            Self::UnsupportedAutoChain { lane } => {
                write!(f, "pipeline lane {lane} uses unsupported auto-chaining")
            }
            Self::StaleToken { lane } => {
                write!(f, "settlement token is stale for pipeline lane {lane}")
            }
            Self::TargetOverflow { lane } => {
                write!(f, "settlement target overflowed for lane {lane}")
            }
            Self::ProgressClosed { lane } => {
                write!(f, "pipeline progress channel closed for lane {lane}")
            }
            Self::PipelineFailed { lane, failure } => {
                write!(f, "pipeline lane {lane} failed: {failure}")
            }
        }
    }
}

impl std::error::Error for SettlementError {}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PipelineCheckpoint {
    pub(crate) epoch: u64,
    pub(crate) completed_batches: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct PipelineSnapshot {
    pub(crate) epoch: u64,
    pub(crate) queued: usize,
    pub(crate) starting: usize,
    pub(crate) in_flight: usize,
    pub(crate) settling: usize,
    pub(crate) completed_batches: u64,
    pub(crate) failure: Option<PipelineFailure>,
}

impl PipelineSnapshot {
    pub(crate) fn checkpoint(&self) -> PipelineCheckpoint {
        PipelineCheckpoint {
            epoch: self.epoch,
            completed_batches: self.completed_batches,
        }
    }

    pub(crate) fn is_stable(&self, max_concurrent_transfers: usize) -> bool {
        if self.starting > 0 || self.settling > 0 {
            return false;
        }
        self.queued == 0 || self.in_flight >= max_concurrent_transfers
    }
}

pub(crate) struct SettlementWaiter {
    lane: PipelineLane,
    checkpoint: PipelineCheckpoint,
    expected_completed: u64,
    max_concurrent_transfers: usize,
    progress: watch::Receiver<PipelineSnapshot>,
}

impl SettlementWaiter {
    pub(crate) fn new(
        lane: PipelineLane,
        tracker: &PipelineSettlementTracker,
        checkpoint: PipelineCheckpoint,
        delta: u64,
    ) -> Result<Self, SettlementError> {
        let expected_completed = checkpoint
            .completed_batches
            .checked_add(delta)
            .ok_or(SettlementError::TargetOverflow { lane })?;
        Ok(Self {
            lane,
            checkpoint,
            expected_completed,
            max_concurrent_transfers: tracker.max_concurrent_transfers(),
            progress: tracker.subscribe(),
        })
    }
}

pub(crate) async fn wait_for_settlement(
    mut waiters: Vec<SettlementWaiter>,
) -> Result<(), SettlementError> {
    loop {
        let snapshots: Vec<PipelineSnapshot> = waiters
            .iter_mut()
            .map(|waiter| waiter.progress.borrow_and_update().clone())
            .collect();

        let mut all_settled = true;
        for (waiter, snapshot) in waiters.iter().zip(&snapshots) {
            if snapshot.epoch < waiter.checkpoint.epoch
                || snapshot.completed_batches < waiter.checkpoint.completed_batches
            {
                return Err(SettlementError::StaleToken { lane: waiter.lane });
            }
            if let Some(failure) = &snapshot.failure {
                return Err(SettlementError::PipelineFailed {
                    lane: waiter.lane,
                    failure: failure.clone(),
                });
            }
            all_settled &= snapshot.completed_batches >= waiter.expected_completed
                && snapshot.is_stable(waiter.max_concurrent_transfers);
        }

        let epoch_changed = waiters
            .iter()
            .zip(&snapshots)
            .any(|(waiter, snapshot)| waiter.progress.borrow().epoch != snapshot.epoch);
        if epoch_changed {
            continue;
        }
        if all_settled {
            return Ok(());
        }

        let changes = waiters
            .iter_mut()
            .map(|waiter| Box::pin(waiter.progress.changed()))
            .collect::<Vec<_>>();
        let (result, changed_index, _) = select_all(changes).await;
        if result.is_err() {
            return Err(SettlementError::ProgressClosed {
                lane: waiters[changed_index].lane,
            });
        }
    }
}

struct TrackerInner {
    snapshot: Mutex<PipelineSnapshot>,
    progress_tx: watch::Sender<PipelineSnapshot>,
    max_concurrent_transfers: usize,
}

/// One linearization point for all executor state used by settlement.
#[derive(Clone)]
pub(crate) struct PipelineSettlementTracker {
    inner: Arc<TrackerInner>,
}

impl PipelineSettlementTracker {
    pub(crate) fn new(max_concurrent_transfers: usize) -> Self {
        let snapshot = PipelineSnapshot::default();
        let (progress_tx, _progress_rx) = watch::channel(snapshot.clone());
        Self {
            inner: Arc::new(TrackerInner {
                snapshot: Mutex::new(snapshot),
                progress_tx,
                max_concurrent_transfers: max_concurrent_transfers.max(1),
            }),
        }
    }

    pub(crate) fn snapshot(&self) -> PipelineSnapshot {
        self.inner.snapshot.lock().clone()
    }

    pub(crate) fn subscribe(&self) -> watch::Receiver<PipelineSnapshot> {
        self.inner.progress_tx.subscribe()
    }

    pub(crate) fn max_concurrent_transfers(&self) -> usize {
        self.inner.max_concurrent_transfers
    }

    pub(crate) fn queue_batch(&self) {
        self.update(|snapshot| {
            increment_usize(&mut snapshot.queued, "queued")?;
            Ok(())
        });
    }

    pub(crate) fn discard_queued(&self) {
        self.update(|snapshot| decrement_usize(&mut snapshot.queued, "queued"));
    }

    pub(crate) fn fail_queued(&self, failure: PipelineFailure) {
        self.update(|snapshot| {
            decrement_usize(&mut snapshot.queued, "queued")?;
            record_failure(snapshot, failure);
            Ok(())
        });
    }

    pub(crate) fn admit(&self) {
        self.update(|snapshot| {
            decrement_usize(&mut snapshot.queued, "queued")?;
            increment_usize(&mut snapshot.starting, "starting")?;
            Ok(())
        });
    }

    fn transition(&self, from: BatchPhase, to: BatchPhase) {
        self.update(|snapshot| {
            decrement_phase(snapshot, from)?;
            increment_phase(snapshot, to)?;
            Ok(())
        });
    }

    fn complete(&self, phase: BatchPhase) {
        self.update(|snapshot| {
            decrement_phase(snapshot, phase)?;
            snapshot.completed_batches =
                snapshot.completed_batches.checked_add(1).ok_or_else(|| {
                    PipelineFailure::new(
                        PipelineFailureKind::CounterOverflow,
                        "completed_batches overflowed",
                    )
                })?;
            Ok(())
        });
    }

    fn fail_phase(&self, phase: BatchPhase, failure: PipelineFailure) {
        self.update(|snapshot| {
            decrement_phase(snapshot, phase)?;
            record_failure(snapshot, failure);
            Ok(())
        });
    }

    pub(crate) fn fail(&self, failure: PipelineFailure) {
        self.update(|snapshot| {
            record_failure(snapshot, failure);
            Ok(())
        });
    }

    fn update(&self, update: impl FnOnce(&mut PipelineSnapshot) -> Result<(), PipelineFailure>) {
        let mut snapshot = self.inner.snapshot.lock();
        if let Err(failure) = update(&mut snapshot) {
            record_failure(&mut snapshot, failure);
        }
        if let Some(next_epoch) = snapshot.epoch.checked_add(1) {
            snapshot.epoch = next_epoch;
        } else if snapshot.failure.is_none() {
            snapshot.failure = Some(PipelineFailure::new(
                PipelineFailureKind::CounterOverflow,
                "pipeline epoch overflowed",
            ));
        }
        self.inner.progress_tx.send_replace(snapshot.clone());
    }
}

pub(crate) struct QueuedBatchGuard {
    tracker: PipelineSettlementTracker,
    queued: bool,
}

impl QueuedBatchGuard {
    pub(crate) fn new(tracker: PipelineSettlementTracker) -> Self {
        tracker.queue_batch();
        Self {
            tracker,
            queued: true,
        }
    }

    pub(crate) fn sent(mut self) {
        self.queued = false;
    }

    pub(crate) fn finish_failure(mut self, failure: PipelineFailure) {
        self.queued = false;
        self.tracker.fail_queued(failure);
    }
}

impl Drop for QueuedBatchGuard {
    fn drop(&mut self) {
        if !self.queued {
            return;
        }
        let (kind, message) = if std::thread::panicking() {
            (
                PipelineFailureKind::TaskPanicked,
                "precondition task panicked while forwarding a queued batch",
            )
        } else {
            (
                PipelineFailureKind::TaskAborted,
                "precondition task ended while forwarding a queued batch",
            )
        };
        self.tracker
            .fail_queued(PipelineFailure::new(kind, message));
    }
}

pub(crate) struct PipelineRunGuard {
    tracker: PipelineSettlementTracker,
    running: bool,
    pipeline_name: &'static str,
}

impl PipelineRunGuard {
    pub(crate) fn new(tracker: PipelineSettlementTracker, pipeline_name: &'static str) -> Self {
        Self {
            tracker,
            running: true,
            pipeline_name,
        }
    }

    pub(crate) fn finish_shutdown(mut self) {
        self.running = false;
        self.tracker.fail(PipelineFailure::new(
            PipelineFailureKind::Shutdown,
            format!("{} stopped", self.pipeline_name),
        ));
    }
}

impl Drop for PipelineRunGuard {
    fn drop(&mut self) {
        if !self.running {
            return;
        }
        let (kind, reason) = if std::thread::panicking() {
            (PipelineFailureKind::TaskPanicked, "panicked")
        } else {
            (PipelineFailureKind::TaskAborted, "was aborted")
        };
        self.tracker.fail(PipelineFailure::new(
            kind,
            format!("{} {reason}", self.pipeline_name),
        ));
    }
}

#[derive(Debug, Clone, Copy)]
enum BatchPhase {
    Starting,
    InFlight,
    Settling,
}

pub(crate) struct BatchPhaseGuard {
    tracker: PipelineSettlementTracker,
    phase: Option<BatchPhase>,
    permit: Option<OwnedSemaphorePermit>,
}

impl BatchPhaseGuard {
    pub(crate) fn starting(
        tracker: PipelineSettlementTracker,
        permit: OwnedSemaphorePermit,
    ) -> Self {
        tracker.admit();
        Self {
            tracker,
            phase: Some(BatchPhase::Starting),
            permit: Some(permit),
        }
    }

    pub(crate) fn mark_in_flight(&mut self) {
        self.transition(BatchPhase::InFlight);
    }

    pub(crate) fn mark_settling(&mut self) {
        self.transition(BatchPhase::Settling);
    }

    pub(crate) fn finish_success(mut self) {
        let phase = self
            .phase
            .take()
            .expect("batch phase guard already finished");
        drop(self.permit.take());
        self.tracker.complete(phase);
    }

    pub(crate) fn finish_failure(mut self, failure: PipelineFailure) {
        let phase = self
            .phase
            .take()
            .expect("batch phase guard already finished");
        drop(self.permit.take());
        self.tracker.fail_phase(phase, failure);
    }

    fn transition(&mut self, next: BatchPhase) {
        let current = self.phase.expect("batch phase guard already finished");
        self.tracker.transition(current, next);
        self.phase = Some(next);
    }
}

impl Drop for BatchPhaseGuard {
    fn drop(&mut self) {
        let Some(phase) = self.phase.take() else {
            return;
        };
        drop(self.permit.take());
        let (kind, message) = if std::thread::panicking() {
            (
                PipelineFailureKind::TaskPanicked,
                "transfer task panicked before publishing settlement",
            )
        } else {
            (
                PipelineFailureKind::TaskAborted,
                "transfer task ended before publishing settlement",
            )
        };
        self.tracker
            .fail_phase(phase, PipelineFailure::new(kind, message));
    }
}

fn increment_phase(
    snapshot: &mut PipelineSnapshot,
    phase: BatchPhase,
) -> Result<(), PipelineFailure> {
    match phase {
        BatchPhase::Starting => increment_usize(&mut snapshot.starting, "starting"),
        BatchPhase::InFlight => increment_usize(&mut snapshot.in_flight, "in_flight"),
        BatchPhase::Settling => increment_usize(&mut snapshot.settling, "settling"),
    }
}

fn decrement_phase(
    snapshot: &mut PipelineSnapshot,
    phase: BatchPhase,
) -> Result<(), PipelineFailure> {
    match phase {
        BatchPhase::Starting => decrement_usize(&mut snapshot.starting, "starting"),
        BatchPhase::InFlight => decrement_usize(&mut snapshot.in_flight, "in_flight"),
        BatchPhase::Settling => decrement_usize(&mut snapshot.settling, "settling"),
    }
}

fn increment_usize(value: &mut usize, name: &str) -> Result<(), PipelineFailure> {
    *value = value.checked_add(1).ok_or_else(|| {
        PipelineFailure::new(
            PipelineFailureKind::CounterOverflow,
            format!("{name} counter overflowed"),
        )
    })?;
    Ok(())
}

fn decrement_usize(value: &mut usize, name: &str) -> Result<(), PipelineFailure> {
    *value = value.checked_sub(1).ok_or_else(|| {
        PipelineFailure::new(
            PipelineFailureKind::Invariant,
            format!("{name} counter underflowed"),
        )
    })?;
    Ok(())
}

fn record_failure(snapshot: &mut PipelineSnapshot, failure: PipelineFailure) {
    if snapshot.failure.is_none() {
        snapshot.failure = Some(failure);
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use futures::{FutureExt, poll};
    use tokio::sync::{Semaphore, oneshot};

    use super::*;

    async fn starting_phase(
        tracker: &PipelineSettlementTracker,
        semaphore: &Arc<Semaphore>,
    ) -> BatchPhaseGuard {
        let permit = semaphore
            .clone()
            .acquire_owned()
            .await
            .expect("test semaphore should remain open");
        BatchPhaseGuard::starting(tracker.clone(), permit)
    }

    fn waiter(
        tracker: &PipelineSettlementTracker,
        checkpoint: PipelineCheckpoint,
        delta: u64,
    ) -> SettlementWaiter {
        SettlementWaiter::new(PipelineLane::G1ToG2, tracker, checkpoint, delta)
            .expect("test settlement target should be valid")
    }

    #[tokio::test]
    async fn concurrency_one_waits_for_queued_successor_to_invoke_worker() {
        let tracker = PipelineSettlementTracker::new(1);
        let semaphore = Arc::new(Semaphore::new(1));
        let checkpoint = tracker.snapshot().checkpoint();
        tracker.queue_batch();
        tracker.queue_batch();

        let mut first = starting_phase(&tracker, &semaphore).await;
        first.mark_in_flight();
        let mut settlement = wait_for_settlement(vec![waiter(&tracker, checkpoint, 1)]).boxed();

        first.mark_settling();
        first.finish_success();
        assert!(poll!(&mut settlement).is_pending());

        let mut successor = starting_phase(&tracker, &semaphore).await;
        assert!(poll!(&mut settlement).is_pending());
        successor.mark_in_flight();
        assert_eq!(settlement.await, Ok(()));

        successor.mark_settling();
        successor.finish_success();
    }

    #[tokio::test]
    async fn simultaneous_completions_fill_every_available_successor_slot() {
        let tracker = PipelineSettlementTracker::new(2);
        let semaphore = Arc::new(Semaphore::new(2));
        let checkpoint = tracker.snapshot().checkpoint();
        for _ in 0..4 {
            tracker.queue_batch();
        }

        let mut first = starting_phase(&tracker, &semaphore).await;
        let mut second = starting_phase(&tracker, &semaphore).await;
        first.mark_in_flight();
        second.mark_in_flight();
        let mut settlement = wait_for_settlement(vec![waiter(&tracker, checkpoint, 2)]).boxed();

        first.mark_settling();
        second.mark_settling();
        first.finish_success();
        second.finish_success();
        assert!(poll!(&mut settlement).is_pending());

        let mut third = starting_phase(&tracker, &semaphore).await;
        let mut fourth = starting_phase(&tracker, &semaphore).await;
        third.mark_in_flight();
        assert!(poll!(&mut settlement).is_pending());
        fourth.mark_in_flight();
        assert_eq!(settlement.await, Ok(()));

        third.mark_settling();
        fourth.mark_settling();
        third.finish_success();
        fourth.finish_success();
    }

    #[tokio::test]
    async fn completion_without_successor_settles() {
        let tracker = PipelineSettlementTracker::new(1);
        let semaphore = Arc::new(Semaphore::new(1));
        let checkpoint = tracker.snapshot().checkpoint();
        tracker.queue_batch();
        let mut phase = starting_phase(&tracker, &semaphore).await;
        phase.mark_in_flight();
        let settlement = wait_for_settlement(vec![waiter(&tracker, checkpoint, 1)]);

        phase.mark_settling();
        phase.finish_success();
        assert_eq!(settlement.await, Ok(()));
    }

    #[tokio::test]
    async fn progress_between_subscription_and_initial_check_is_observed() {
        let tracker = PipelineSettlementTracker::new(1);
        let semaphore = Arc::new(Semaphore::new(1));
        let checkpoint = tracker.snapshot().checkpoint();
        tracker.queue_batch();
        let mut phase = starting_phase(&tracker, &semaphore).await;
        phase.mark_in_flight();
        let subscribed = waiter(&tracker, checkpoint, 1);

        phase.mark_settling();
        phase.finish_success();
        assert_eq!(wait_for_settlement(vec![subscribed]).await, Ok(()));
    }

    #[tokio::test]
    async fn task_abort_removes_phase_and_publishes_terminal_failure() {
        let tracker = PipelineSettlementTracker::new(1);
        tracker.queue_batch();
        let task_tracker = tracker.clone();
        let semaphore = Arc::new(Semaphore::new(1));
        let task_semaphore = semaphore.clone();
        let (started_tx, started_rx) = oneshot::channel();
        let task = tokio::spawn(async move {
            let mut phase = starting_phase(&task_tracker, &task_semaphore).await;
            phase.mark_in_flight();
            started_tx.send(()).expect("test should observe task start");
            std::future::pending::<()>().await;
            phase.mark_settling();
        });
        started_rx.await.expect("transfer task should start");

        task.abort();
        assert!(
            task.await
                .expect_err("task should be cancelled")
                .is_cancelled()
        );
        let snapshot = tracker.snapshot();
        assert_eq!(snapshot.in_flight, 0);
        assert_eq!(
            snapshot.failure.expect("abort must publish failure").kind(),
            PipelineFailureKind::TaskAborted
        );
    }

    #[tokio::test]
    async fn cancellation_panic_executor_error_and_shutdown_are_structured() {
        let queued_tracker = PipelineSettlementTracker::new(1);
        drop(QueuedBatchGuard::new(queued_tracker.clone()));
        assert_eq!(
            queued_tracker
                .snapshot()
                .failure
                .expect("queued cancellation must fail")
                .kind(),
            PipelineFailureKind::TaskAborted
        );

        let panic_tracker = PipelineSettlementTracker::new(1);
        panic_tracker.queue_batch();
        let panic_semaphore = Arc::new(Semaphore::new(1));
        let phase = starting_phase(&panic_tracker, &panic_semaphore).await;
        assert!(
            catch_unwind(AssertUnwindSafe(move || {
                let _phase = phase;
                panic!("intentional transfer panic");
            }))
            .is_err()
        );
        assert_eq!(
            panic_tracker
                .snapshot()
                .failure
                .expect("panic must publish failure")
                .kind(),
            PipelineFailureKind::TaskPanicked
        );

        let error_tracker = PipelineSettlementTracker::new(1);
        error_tracker.queue_batch();
        let error_semaphore = Arc::new(Semaphore::new(1));
        let phase = starting_phase(&error_tracker, &error_semaphore).await;
        phase.finish_failure(PipelineFailure::new(
            PipelineFailureKind::Executor,
            "injected executor error",
        ));
        assert_eq!(
            error_tracker
                .snapshot()
                .failure
                .expect("executor error must publish failure")
                .kind(),
            PipelineFailureKind::Executor
        );

        let shutdown_tracker = PipelineSettlementTracker::new(1);
        PipelineRunGuard::new(shutdown_tracker.clone(), "test pipeline").finish_shutdown();
        assert_eq!(
            shutdown_tracker
                .snapshot()
                .failure
                .expect("shutdown must publish failure")
                .kind(),
            PipelineFailureKind::Shutdown
        );
    }

    #[tokio::test]
    async fn closed_progress_and_stale_checkpoint_return_errors() {
        let tracker = PipelineSettlementTracker::new(1);
        let checkpoint = tracker.snapshot().checkpoint();
        let closed_waiter = waiter(&tracker, checkpoint, 1);
        drop(tracker);
        assert!(matches!(
            wait_for_settlement(vec![closed_waiter]).await,
            Err(SettlementError::ProgressClosed {
                lane: PipelineLane::G1ToG2
            })
        ));

        let tracker = PipelineSettlementTracker::new(1);
        let stale = PipelineCheckpoint {
            epoch: tracker.snapshot().epoch + 1,
            completed_batches: 0,
        };
        assert!(matches!(
            wait_for_settlement(vec![waiter(&tracker, stale, 0)]).await,
            Err(SettlementError::StaleToken {
                lane: PipelineLane::G1ToG2
            })
        ));
    }

    #[test]
    fn foreign_unavailable_and_overflow_targets_return_errors() {
        let engine_id = Uuid::new_v4();
        let token = SettlementToken {
            engine_id,
            checkpoints: [None; 3],
        };
        assert_eq!(
            token.validate_engine(Uuid::new_v4()),
            Err(SettlementError::ForeignToken)
        );
        assert!(matches!(
            token.checkpoint(PipelineLane::G2ToG3),
            Err(SettlementError::LaneUnavailable {
                lane: PipelineLane::G2ToG3
            })
        ));

        let mut target = SettlementTarget::new();
        target
            .add_completed_batches(PipelineLane::G2ToG4, u64::MAX)
            .expect("first target addition should fit");
        assert!(matches!(
            target.add_completed_batches(PipelineLane::G2ToG4, 1),
            Err(SettlementError::TargetOverflow {
                lane: PipelineLane::G2ToG4
            })
        ));

        let tracker = PipelineSettlementTracker::new(1);
        let checkpoint = PipelineCheckpoint {
            epoch: 0,
            completed_batches: u64::MAX,
        };
        assert!(matches!(
            SettlementWaiter::new(PipelineLane::G1ToG2, &tracker, checkpoint, 1),
            Err(SettlementError::TargetOverflow {
                lane: PipelineLane::G1ToG2
            })
        ));
    }
}
