// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded lane-sticky dispatch for indexer-domain-scoped global CKF ingestion.
//!
//! A physical lane has exactly one ingestion-worker owner. Deltas are admitted asynchronously,
//! while assignment, snapshot installation, and exact-drain markers await the same FIFO. Queue
//! placement preserves lane order only; query threads continue to read just the atomic ready mask
//! and packed buckets owned by [`GlobalCkfIndexer`].

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use super::failure::{CkfFailureAction, CkfFailureDisposition, CkfFailurePoint};
use super::{
    ConsumerDrainMarker, GlobalCkfAssignmentError, GlobalCkfBuildError, GlobalCkfDelta,
    GlobalCkfIndexer, GlobalCkfIngestOutcome, GlobalCkfLaneFault, GlobalCkfLaneIngestor,
    GlobalCkfSnapshot, LaneLease, ProducerIdentity,
};

pub const DEFAULT_GLOBAL_INGESTION_WORKERS: usize = 4;
pub const DEFAULT_GLOBAL_INGESTION_QUEUE_CAPACITY: usize = 256;
pub const DEFAULT_GLOBAL_INGESTION_CONTROL_TIMEOUT: Duration = Duration::from_secs(10);
pub const DEFAULT_GLOBAL_MAX_DIRTY_TO_APPLIED_AGE: Duration = Duration::from_millis(10);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlobalCkfIngestionPoolConfig {
    pub worker_count: usize,
    pub queue_capacity: usize,
    pub control_timeout: Duration,
    /// `None` uses one complete physical lane, matching the production safety bound.
    pub max_outstanding_images_per_lane: Option<usize>,
    pub max_dirty_to_applied_age: Duration,
}

impl Default for GlobalCkfIngestionPoolConfig {
    fn default() -> Self {
        Self {
            worker_count: DEFAULT_GLOBAL_INGESTION_WORKERS,
            queue_capacity: DEFAULT_GLOBAL_INGESTION_QUEUE_CAPACITY,
            control_timeout: DEFAULT_GLOBAL_INGESTION_CONTROL_TIMEOUT,
            max_outstanding_images_per_lane: None,
            max_dirty_to_applied_age: DEFAULT_GLOBAL_MAX_DIRTY_TO_APPLIED_AGE,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GlobalCkfIngestionPoolBuildError {
    #[error("global CKF ingestion worker count must be greater than zero")]
    InvalidWorkerCount,

    #[error("global CKF ingestion queue capacity must be greater than zero")]
    InvalidQueueCapacity,

    #[error("global CKF ingestion control timeout must be greater than zero")]
    InvalidControlTimeout,

    #[error("global CKF outstanding-image bound must be greater than zero")]
    InvalidOutstandingImageBound,

    #[error("global CKF dirty-to-applied age bound must be greater than zero")]
    InvalidDirtyToAppliedAge,

    #[error(transparent)]
    GlobalCkf(#[from] GlobalCkfBuildError),

    #[error("failed to spawn global CKF ingestion worker {worker}: {source}")]
    SpawnWorker {
        worker: usize,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to spawn global CKF lag watchdog: {0}")]
    SpawnLagWatchdog(#[source] std::io::Error),
}

#[derive(Debug, thiserror::Error, Clone, Copy, PartialEq, Eq)]
pub enum GlobalCkfIngestionError {
    #[error("global CKF lane {lane} is not configured")]
    UnconfiguredLane { lane: usize },

    #[error("global CKF lane {lane} has no current assignment")]
    LaneUnassigned { lane: usize },

    #[error("global CKF lane {lane} is retired and requires a new lease and snapshot")]
    LaneRetired { lane: usize },

    #[error("global CKF lane {lane} is awaiting a snapshot")]
    AwaitingSnapshot { lane: usize },

    #[error("global CKF lane {lane} ingestion queue saturated; its lease was retired")]
    Saturated { lane: usize },

    #[error(
        "global CKF lane {lane} exceeded its outstanding-image bound ({requested} requested, {limit} allowed); its lease was retired"
    )]
    OutstandingImageLimit {
        lane: usize,
        requested: usize,
        limit: usize,
    },

    #[error("global CKF lane {lane} exceeded its dirty-to-applied age; its lease was retired")]
    DirtyToAppliedAgeExceeded { lane: usize },

    #[error("global CKF lane {lane} control operation timed out; its lease was retired")]
    ControlTimeout { lane: usize },

    #[error("global CKF ingestion worker {worker} failed")]
    WorkerFailed { worker: usize },

    #[error("global CKF assignment failed: {0}")]
    Assignment(#[from] GlobalCkfAssignmentError),

    #[error("global CKF lane admission epoch exhausted for lane {lane}")]
    AdmissionEpochExhausted { lane: usize },
}

impl GlobalCkfIngestionError {
    /// Recovery disposition for errors that retire consumer state.
    ///
    /// Admission/assignment rejections return `None`: they made no consumer write and stale or
    /// superseded traffic must not fence the current assignment.
    pub const fn failure_disposition(self) -> Option<CkfFailureDisposition> {
        match self {
            Self::Saturated { .. } | Self::OutstandingImageLimit { .. } => {
                Some(CkfFailurePoint::ConsumerGapOrMalformedBeforeWrite.disposition())
            }
            Self::DirtyToAppliedAgeExceeded { .. }
            | Self::ControlTimeout { .. }
            | Self::WorkerFailed { .. } => {
                Some(CkfFailurePoint::ConsumerWorkerFailureMidApply.disposition())
            }
            Self::UnconfiguredLane { .. }
            | Self::LaneUnassigned { .. }
            | Self::LaneRetired { .. }
            | Self::AwaitingSnapshot { .. }
            | Self::Assignment(_)
            | Self::AdmissionEpochExhausted { .. } => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlobalCkfIngestionFault {
    Saturated {
        lane: usize,
        lease: Option<LaneLease>,
    },
    OutstandingImageLimit {
        lane: usize,
        lease: Option<LaneLease>,
        requested: usize,
        limit: usize,
    },
    DirtyToAppliedAgeExceeded {
        lane: usize,
        lease: Option<LaneLease>,
    },
    LaneDeactivated {
        lane: usize,
        lease: Option<LaneLease>,
        fault: GlobalCkfLaneFault,
        disposition: CkfFailureDisposition,
    },
    WorkerFailed {
        worker: usize,
    },
}

impl GlobalCkfIngestionFault {
    /// Classify recovery from the consumer state whose commit is known or uncertain.
    pub const fn failure_disposition(self) -> CkfFailureDisposition {
        match self {
            Self::LaneDeactivated { disposition, .. } => disposition,
            Self::WorkerFailed { .. } | Self::DirtyToAppliedAgeExceeded { .. } => {
                CkfFailurePoint::ConsumerWorkerFailureMidApply.disposition()
            }
            Self::Saturated { .. } | Self::OutstandingImageLimit { .. } => {
                CkfFailurePoint::ConsumerGapOrMalformedBeforeWrite.disposition()
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LaneAdmissionState {
    Unassigned,
    Transitioning,
    AwaitingSnapshot,
    Active,
    Retired,
}

#[derive(Debug)]
struct LaneAdmission {
    epoch: u64,
    state: LaneAdmissionState,
    lease: Option<LaneLease>,
    last_assignment_epoch: Option<u64>,
    next_delta_id: u64,
    outstanding_images: usize,
    outstanding_deltas: VecDeque<OutstandingDelta>,
}

#[derive(Debug, Clone, Copy)]
struct OutstandingDelta {
    id: u64,
    images: usize,
    submitted_at: Instant,
}

#[derive(Debug)]
struct LaneControl {
    published_epoch: AtomicU64,
    permanently_fenced: AtomicBool,
    admission: Mutex<LaneAdmission>,
    transition: Mutex<()>,
    max_outstanding_images: usize,
    max_dirty_to_applied_age: Duration,
    #[cfg(test)]
    snapshot_activation_pause: Mutex<Option<(flume::Sender<()>, flume::Receiver<()>)>>,
}

impl LaneControl {
    fn new(
        max_outstanding_images: usize,
        max_dirty_to_applied_age: Duration,
        queue_capacity: usize,
    ) -> Self {
        let mut outstanding_deltas = VecDeque::new();
        outstanding_deltas.reserve_exact(queue_capacity.saturating_add(1));
        Self {
            published_epoch: AtomicU64::new(0),
            permanently_fenced: AtomicBool::new(false),
            admission: Mutex::new(LaneAdmission {
                epoch: 0,
                state: LaneAdmissionState::Unassigned,
                lease: None,
                last_assignment_epoch: None,
                next_delta_id: 0,
                outstanding_images: 0,
                outstanding_deltas,
            }),
            transition: Mutex::new(()),
            max_outstanding_images,
            max_dirty_to_applied_age,
            #[cfg(test)]
            snapshot_activation_pause: Mutex::new(None),
        }
    }

    #[cfg(test)]
    fn pause_before_snapshot_activation(&self) {
        let pause = self.snapshot_activation_pause.lock().take();
        let Some((entered, release)) = pause else {
            return;
        };
        let _ = entered.send(());
        let _ = release.recv();
    }
}

#[derive(Debug)]
struct LaneRoute {
    worker: usize,
    control: Arc<LaneControl>,
}

#[derive(Debug)]
struct WorkerHealth {
    failed_workers: Box<[AtomicBool]>,
}

impl WorkerHealth {
    fn new(worker_count: usize) -> Self {
        Self {
            failed_workers: (0..worker_count).map(|_| AtomicBool::new(false)).collect(),
        }
    }

    fn record_failure(&self, worker: usize) {
        self.failed_workers[worker].store(true, Ordering::Release);
    }

    fn is_failed(&self, worker: usize) -> bool {
        self.failed_workers[worker].load(Ordering::Acquire)
    }

    fn failed_worker(&self) -> Option<usize> {
        self.failed_workers
            .iter()
            .position(|failed| failed.load(Ordering::Acquire))
    }
}

struct DeltaPermit {
    control: Arc<LaneControl>,
    id: u64,
}

impl Drop for DeltaPermit {
    fn drop(&mut self) {
        let mut admission = self.control.admission.lock();
        let outstanding = if admission
            .outstanding_deltas
            .front()
            .is_some_and(|delta| delta.id == self.id)
        {
            admission.outstanding_deltas.pop_front()
        } else if admission
            .outstanding_deltas
            .back()
            .is_some_and(|delta| delta.id == self.id)
        {
            // A failed try_send drops the just-admitted delta before older queued work.
            admission.outstanding_deltas.pop_back()
        } else {
            // Exceptional worker/channel teardown need not preserve permit-drop order. Keep that
            // cold path correct without making ordinary FIFO completion scan the backlog.
            admission
                .outstanding_deltas
                .iter()
                .position(|delta| delta.id == self.id)
                .and_then(|position| admission.outstanding_deltas.remove(position))
        };
        let Some(outstanding) = outstanding else {
            return;
        };
        admission.outstanding_images = admission
            .outstanding_images
            .checked_sub(outstanding.images)
            .expect("delta permit accounting must not underflow");
    }
}

enum IngestionCommand {
    Assign {
        lane: usize,
        epoch: u64,
        identity: ProducerIdentity,
        lease: LaneLease,
        response: flume::Sender<Result<(), GlobalCkfAssignmentError>>,
    },
    Snapshot {
        lane: usize,
        epoch: u64,
        snapshot: GlobalCkfSnapshot,
        response: flume::Sender<Result<ProcessedIngestOutcome, GlobalCkfIngestionError>>,
    },
    Delta {
        lane: usize,
        epoch: u64,
        delta: GlobalCkfDelta,
        _permit: DeltaPermit,
    },
    Drain {
        lane: usize,
        epoch: u64,
        marker: ConsumerDrainMarker,
        response: flume::Sender<Result<GlobalCkfIngestOutcome, GlobalCkfIngestionError>>,
    },
    Shutdown,
    #[cfg(test)]
    Pause {
        entered: flume::Sender<()>,
        release: flume::Receiver<()>,
    },
    #[cfg(test)]
    Crash,
}

#[derive(Debug, Clone, Copy)]
struct ProcessedIngestOutcome {
    outcome: GlobalCkfIngestOutcome,
    failure_disposition: Option<CkfFailureDisposition>,
}

impl IngestionCommand {
    fn reject_stale(self, lane: usize) {
        let error = Err(GlobalCkfIngestionError::LaneRetired { lane });
        match self {
            Self::Assign { response, .. } => {
                // A superseded assignment is semantically stale, not an assignment-validation
                // error. Dropping this response makes its waiter observe retirement.
                drop(response);
            }
            Self::Snapshot { response, .. } => {
                let _ = response.send(Err(GlobalCkfIngestionError::LaneRetired { lane }));
            }
            Self::Drain { response, .. } => {
                let _ = response.send(error);
            }
            Self::Delta { .. } | Self::Shutdown => {}
            #[cfg(test)]
            Self::Pause { .. } | Self::Crash => {}
        }
    }
}

struct WorkerLane {
    ingestor: GlobalCkfLaneIngestor,
    control: Arc<LaneControl>,
    observed_epoch: u64,
}

struct IngestionWorker {
    receiver: flume::Receiver<IngestionCommand>,
    lanes: [Option<WorkerLane>; super::DC_COUNT],
    indexer: GlobalCkfIndexer,
    fault_tx: flume::Sender<GlobalCkfIngestionFault>,
}

impl IngestionWorker {
    fn retire_all(&mut self) {
        for lane in self.lanes.iter_mut().flatten() {
            lane.ingestor.retire();
        }
    }

    fn run(&mut self) {
        while let Ok(command) = self.receiver.recv() {
            match command {
                IngestionCommand::Shutdown => {
                    self.retire_all();
                    return;
                }
                #[cfg(test)]
                IngestionCommand::Pause { entered, release } => {
                    let _ = entered.send(());
                    let _ = release.recv();
                }
                #[cfg(test)]
                IngestionCommand::Crash => panic!("injected global CKF ingestion worker failure"),
                command => self.process(command),
            }
        }
        self.retire_all();
    }

    fn process(&mut self, command: IngestionCommand) {
        let (lane, epoch) = match &command {
            IngestionCommand::Assign { lane, epoch, .. }
            | IngestionCommand::Snapshot { lane, epoch, .. }
            | IngestionCommand::Delta { lane, epoch, .. }
            | IngestionCommand::Drain { lane, epoch, .. } => (*lane, *epoch),
            IngestionCommand::Shutdown => return,
            #[cfg(test)]
            IngestionCommand::Pause { .. } | IngestionCommand::Crash => return,
        };

        let control = Arc::clone(
            &self.lanes[lane]
                .as_ref()
                .expect("lane-sticky dispatch must target the lane's owning worker")
                .control,
        );
        let worker_lane = self.lanes[lane]
            .as_mut()
            .expect("lane-sticky dispatch must target the lane's owning worker");
        if control.permanently_fenced.load(Ordering::Acquire) {
            worker_lane.ingestor.retire();
            command.reject_stale(lane);
            return;
        }
        let current_epoch = worker_lane.control.published_epoch.load(Ordering::Acquire);
        if worker_lane.observed_epoch != current_epoch {
            worker_lane.ingestor.retire();
            worker_lane.observed_epoch = current_epoch;
        }
        if epoch != current_epoch {
            command.reject_stale(lane);
            return;
        }

        match command {
            IngestionCommand::Assign {
                identity,
                lease,
                response,
                ..
            } => {
                let _ = response.send(worker_lane.ingestor.assign(identity, lease));
            }
            IngestionCommand::Snapshot {
                snapshot, response, ..
            } => {
                let outcome = worker_lane.ingestor.install_snapshot_guarded(
                    &snapshot,
                    |ingestor, sequence| {
                        #[cfg(test)]
                        control.pause_before_snapshot_activation();
                        let admission = control.admission.lock();
                        if control.permanently_fenced.load(Ordering::Acquire)
                            || admission.epoch != epoch
                            || admission.lease != Some(snapshot.lease())
                            || admission.state != LaneAdmissionState::AwaitingSnapshot
                        {
                            return false;
                        }
                        ingestor.activate_snapshot(sequence);
                        true
                    },
                );
                let failure_disposition = worker_lane.ingestor.last_failure_disposition();
                observe_outcome(
                    &self.indexer,
                    &self.fault_tx,
                    &control,
                    LaneMessageScope::new(lane, epoch, snapshot.lease()),
                    outcome,
                    failure_disposition,
                );
                let _ = response.send(Ok(ProcessedIngestOutcome {
                    outcome,
                    failure_disposition,
                }));
            }
            IngestionCommand::Delta {
                delta,
                _permit: _permit_guard,
                ..
            } => {
                let lease = delta.lease();
                let outcome = worker_lane.ingestor.apply_delta(&delta);
                let failure_disposition = worker_lane.ingestor.last_failure_disposition();
                observe_outcome(
                    &self.indexer,
                    &self.fault_tx,
                    &control,
                    LaneMessageScope::new(lane, epoch, lease),
                    outcome,
                    failure_disposition,
                );
            }
            IngestionCommand::Drain {
                marker, response, ..
            } => {
                let lease = marker.lease();
                let outcome = worker_lane.ingestor.complete_drain(marker);
                let failure_disposition = worker_lane.ingestor.last_failure_disposition();
                observe_outcome(
                    &self.indexer,
                    &self.fault_tx,
                    &control,
                    LaneMessageScope::new(lane, epoch, lease),
                    outcome,
                    failure_disposition,
                );
                let _ = response.send(Ok(outcome));
            }
            IngestionCommand::Shutdown => {}
            #[cfg(test)]
            IngestionCommand::Pause { .. } | IngestionCommand::Crash => {}
        }

        // Retirement can race a command already being applied. Recheck the admission epoch after
        // application so such a command cannot reactivate a lane after saturation or timeout.
        let current_epoch = control.published_epoch.load(Ordering::Acquire);
        if control.permanently_fenced.load(Ordering::Acquire) || current_epoch != epoch {
            self.lanes[lane]
                .as_mut()
                .expect("lane-sticky dispatch must retain its lane")
                .ingestor
                .retire();
        }
    }
}

struct IngestionWorkerHandle {
    sender: flume::Sender<IngestionCommand>,
    join: Option<JoinHandle<()>>,
}

/// Shared bounded ingestion pool for one indexer-domain-scoped [`GlobalCkfIndexer`].
pub struct GlobalCkfIngestionPool {
    indexer: GlobalCkfIndexer,
    workers: Vec<IngestionWorkerHandle>,
    routes: [Option<LaneRoute>; super::DC_COUNT],
    control_timeout: Duration,
    health: Arc<WorkerHealth>,
    fault_tx: flume::Sender<GlobalCkfIngestionFault>,
    fault_rx: flume::Receiver<GlobalCkfIngestionFault>,
    lag_watchdog_stop: Arc<AtomicBool>,
    lag_watchdog: Option<JoinHandle<()>>,
}

impl GlobalCkfIngestionPool {
    pub fn new(
        indexer: GlobalCkfIndexer,
        config: GlobalCkfIngestionPoolConfig,
    ) -> Result<Self, GlobalCkfIngestionPoolBuildError> {
        if config.worker_count == 0 {
            return Err(GlobalCkfIngestionPoolBuildError::InvalidWorkerCount);
        }
        if config.queue_capacity == 0 {
            return Err(GlobalCkfIngestionPoolBuildError::InvalidQueueCapacity);
        }
        if config.control_timeout.is_zero() {
            return Err(GlobalCkfIngestionPoolBuildError::InvalidControlTimeout);
        }
        if config.max_outstanding_images_per_lane == Some(0) {
            return Err(GlobalCkfIngestionPoolBuildError::InvalidOutstandingImageBound);
        }
        if config.max_dirty_to_applied_age.is_zero() {
            return Err(GlobalCkfIngestionPoolBuildError::InvalidDirtyToAppliedAge);
        }

        let max_outstanding_images = config
            .max_outstanding_images_per_lane
            .unwrap_or_else(|| indexer.manifest().format().bucket_count());

        let mut worker_lanes: Vec<[Option<WorkerLane>; super::DC_COUNT]> = (0..config.worker_count)
            .map(|_| std::array::from_fn(|_| None))
            .collect();
        let mut routes: [Option<LaneRoute>; super::DC_COUNT] = std::array::from_fn(|_| None);
        let consumer = indexer.manifest().consumer_instance();
        for lane in 0..super::DC_COUNT {
            if indexer.manifest().pool_id(lane).is_none() {
                continue;
            }
            let worker = sticky_worker(consumer.get(), lane as u8, config.worker_count);
            let control = Arc::new(LaneControl::new(
                max_outstanding_images,
                config.max_dirty_to_applied_age,
                config.queue_capacity,
            ));
            let ingestor = indexer.claim_lane(lane)?;
            worker_lanes[worker][lane] = Some(WorkerLane {
                ingestor,
                control: Arc::clone(&control),
                observed_epoch: 0,
            });
            routes[lane] = Some(LaneRoute { worker, control });
        }

        let health = Arc::new(WorkerHealth::new(config.worker_count));
        // Readiness is the authoritative failure signal. Keep notifications bounded and
        // best-effort so an adapter that does not consume diagnostics cannot retain an
        // unbounded stream of repeated faults for an already-retired lane.
        let fault_capacity = super::DC_COUNT
            .saturating_mul(2)
            .saturating_add(config.worker_count);
        let (fault_tx, fault_rx) = flume::bounded(fault_capacity);
        // NOTE: One fixed ingestion pool and watchdog per consumer is the deliberately simple
        // initial topology. Revisit shared execution before supporting high consumer/domain
        // cardinality; do not complicate lane ordering speculatively.
        let mut workers: Vec<IngestionWorkerHandle> = Vec::with_capacity(config.worker_count);
        for (worker_index, lanes) in worker_lanes.into_iter().enumerate() {
            let (sender, receiver) = flume::bounded(config.queue_capacity);
            let worker_health = Arc::clone(&health);
            let worker_fault_tx = fault_tx.clone();
            let worker_indexer = indexer.clone();
            let spawn = thread::Builder::new()
                .name(format!("global-ckf-ingest-{worker_index}"))
                .spawn(move || {
                    let mut worker = IngestionWorker {
                        receiver,
                        lanes,
                        indexer: worker_indexer,
                        fault_tx: worker_fault_tx.clone(),
                    };
                    if catch_unwind(AssertUnwindSafe(|| worker.run())).is_err() {
                        // Dropping the worker would eventually clear readiness, but retire before
                        // publishing the fault so observers cannot see failure with live lanes.
                        worker.retire_all();
                        worker_health.record_failure(worker_index);
                        report_fault(
                            &worker_fault_tx,
                            GlobalCkfIngestionFault::WorkerFailed {
                                worker: worker_index,
                            },
                        );
                    }
                });
            let join = match spawn {
                Ok(join) => join,
                Err(source) => {
                    for worker in &workers {
                        let _ = worker.sender.send(IngestionCommand::Shutdown);
                    }
                    for worker in &mut workers {
                        if let Some(join) = worker.join.take() {
                            let _ = join.join();
                        }
                    }
                    return Err(GlobalCkfIngestionPoolBuildError::SpawnWorker {
                        worker: worker_index,
                        source,
                    });
                }
            };
            workers.push(IngestionWorkerHandle {
                sender,
                join: Some(join),
            });
        }

        let lag_watchdog_stop = Arc::new(AtomicBool::new(false));
        let watchdog_stop = Arc::clone(&lag_watchdog_stop);
        let watchdog_indexer = indexer.clone();
        let watchdog_fault_tx = fault_tx.clone();
        let watchdog_lanes: Vec<_> = routes
            .iter()
            .enumerate()
            .filter_map(|(lane, route)| {
                route
                    .as_ref()
                    .map(|route| (lane, Arc::clone(&route.control)))
            })
            .collect();
        let poll_interval = config
            .max_dirty_to_applied_age
            .checked_div(4)
            .unwrap_or(config.max_dirty_to_applied_age)
            .clamp(Duration::from_micros(100), Duration::from_millis(10));
        let lag_watchdog = match thread::Builder::new()
            .name("global-ckf-lag-watchdog".to_string())
            .spawn(move || {
                while !watchdog_stop.load(Ordering::Acquire) {
                    thread::park_timeout(poll_interval);
                    if watchdog_stop.load(Ordering::Acquire) {
                        break;
                    }
                    fence_expired_lanes(
                        &watchdog_indexer,
                        &watchdog_fault_tx,
                        &watchdog_lanes,
                        Instant::now(),
                    );
                }
            }) {
            Ok(watchdog) => watchdog,
            Err(source) => {
                for worker in &workers {
                    let _ = worker.sender.send(IngestionCommand::Shutdown);
                }
                for worker in &mut workers {
                    if let Some(join) = worker.join.take() {
                        let _ = join.join();
                    }
                }
                return Err(GlobalCkfIngestionPoolBuildError::SpawnLagWatchdog(source));
            }
        };

        Ok(Self {
            indexer,
            workers,
            routes,
            control_timeout: config.control_timeout,
            health,
            fault_tx,
            fault_rx,
            lag_watchdog_stop,
            lag_watchdog: Some(lag_watchdog),
        })
    }

    pub fn indexer(&self) -> &GlobalCkfIndexer {
        &self.indexer
    }

    pub fn worker_for_lane(&self, lane: usize) -> Option<usize> {
        self.routes.get(lane)?.as_ref().map(|route| route.worker)
    }

    pub fn failed_worker(&self) -> Option<usize> {
        self.health.failed_worker()
    }

    pub fn try_recv_fault(&self) -> Option<GlobalCkfIngestionFault> {
        self.fault_rx.try_recv().ok()
    }

    pub fn assign(
        &self,
        identity: ProducerIdentity,
        lease: LaneLease,
    ) -> Result<(), GlobalCkfIngestionError> {
        let lane = usize::from(lease.physical_lane());
        let route = self.route(lane)?;
        let _transition = route.control.transition.lock();
        self.ensure_worker(route.worker)?;
        let epoch = {
            let mut admission = route.control.admission.lock();
            if route.control.permanently_fenced.load(Ordering::Acquire) {
                return Err(GlobalCkfIngestionError::AdmissionEpochExhausted { lane });
            }
            self.indexer.validate_assignment(
                lane,
                identity,
                lease,
                admission.last_assignment_epoch,
            )?;
            let Some(epoch) = admission.epoch.checked_add(1) else {
                permanently_fence_admission_locked(
                    &self.indexer,
                    &route.control,
                    lane,
                    &mut admission,
                );
                return Err(GlobalCkfIngestionError::AdmissionEpochExhausted { lane });
            };
            // A rejected stale/foreign assignment never reaches this point, so it cannot clear
            // the currently valid lane. Once transition begins, readiness is retired before any
            // lease state changes.
            self.indexer.retire_lane_readiness(lane);
            admission.epoch = epoch;
            admission.state = LaneAdmissionState::Transitioning;
            admission.lease = Some(lease);
            route
                .control
                .published_epoch
                .store(epoch, Ordering::Release);
            epoch
        };

        let (response_tx, response_rx) = flume::bounded(1);
        let command = IngestionCommand::Assign {
            lane,
            epoch,
            identity,
            lease,
            response: response_tx,
        };
        self.send_control(route, lane, epoch, command)?;
        let result = self.await_response(route, lane, epoch, response_rx)?;
        let mut admission = route.control.admission.lock();
        if admission.epoch != epoch {
            return Err(GlobalCkfIngestionError::LaneRetired { lane });
        }
        match result {
            Ok(()) => {
                admission.state = LaneAdmissionState::AwaitingSnapshot;
                admission.last_assignment_epoch = Some(lease.assignment_epoch());
                Ok(())
            }
            Err(error) => {
                admission.state = LaneAdmissionState::Retired;
                admission.lease = None;
                Err(error.into())
            }
        }
    }

    pub fn install_snapshot(
        &self,
        snapshot: GlobalCkfSnapshot,
    ) -> Result<GlobalCkfIngestOutcome, GlobalCkfIngestionError> {
        let lane = usize::from(snapshot.lease().physical_lane());
        let route = self.route(lane)?;
        let _transition = route.control.transition.lock();
        self.ensure_worker(route.worker)?;
        let epoch = {
            let admission = route.control.admission.lock();
            match admission.state {
                LaneAdmissionState::AwaitingSnapshot | LaneAdmissionState::Active => {}
                LaneAdmissionState::Unassigned => {
                    return Err(GlobalCkfIngestionError::LaneUnassigned { lane });
                }
                LaneAdmissionState::Transitioning | LaneAdmissionState::Retired => {
                    return Err(GlobalCkfIngestionError::LaneRetired { lane });
                }
            }
            if admission.lease != Some(snapshot.lease()) {
                return Err(GlobalCkfIngestionError::LaneRetired { lane });
            }
            admission.epoch
        };

        let (response_tx, response_rx) = flume::bounded(1);
        let command = IngestionCommand::Snapshot {
            lane,
            epoch,
            snapshot,
            response: response_tx,
        };
        self.send_control(route, lane, epoch, command)?;
        let processed = self.await_response(route, lane, epoch, response_rx)??;
        let outcome = processed.outcome;
        let mut admission = route.control.admission.lock();
        if admission.epoch != epoch {
            if matches!(outcome, GlobalCkfIngestOutcome::LaneDeactivated { .. }) {
                return Ok(outcome);
            }
            return Err(GlobalCkfIngestionError::LaneRetired { lane });
        }
        if matches!(outcome, GlobalCkfIngestOutcome::SnapshotInstalled { .. }) {
            admission.state = LaneAdmissionState::Active;
        } else if matches!(outcome, GlobalCkfIngestOutcome::LaneDeactivated { .. }) {
            // An inactive snapshot validates before any bucket write. Keep the assignment in
            // AwaitingSnapshot so the adapter can retry without recovering the producer.
            admission.state = match processed.failure_disposition.map(|value| value.action) {
                Some(CkfFailureAction::RetrySnapshot) => LaneAdmissionState::AwaitingSnapshot,
                _ => LaneAdmissionState::Retired,
            };
        }
        Ok(outcome)
    }

    /// Admit one delta without waiting for application.
    pub fn submit_delta(&self, delta: GlobalCkfDelta) -> Result<(), GlobalCkfIngestionError> {
        let lane = usize::from(delta.lease().physical_lane());
        let route = self.route(lane)?;
        self.ensure_worker(route.worker)?;
        let mut admission = route.control.admission.lock();
        match admission.state {
            LaneAdmissionState::Active => {}
            LaneAdmissionState::AwaitingSnapshot => {
                return Err(GlobalCkfIngestionError::AwaitingSnapshot { lane });
            }
            LaneAdmissionState::Unassigned => {
                return Err(GlobalCkfIngestionError::LaneUnassigned { lane });
            }
            LaneAdmissionState::Transitioning | LaneAdmissionState::Retired => {
                return Err(GlobalCkfIngestionError::LaneRetired { lane });
            }
        }
        if admission.lease != Some(delta.lease()) {
            return Err(GlobalCkfIngestionError::LaneRetired { lane });
        }
        let epoch = admission.epoch;
        let image_count = delta.images().len();
        let submitted_at = Instant::now();
        if oldest_outstanding_age(&admission, submitted_at)
            .is_some_and(|age| age >= route.control.max_dirty_to_applied_age)
        {
            let lease = admission.lease;
            let _ = retire_admission_locked(&self.indexer, &route.control, lane, &mut admission);
            report_fault(
                &self.fault_tx,
                GlobalCkfIngestionFault::DirtyToAppliedAgeExceeded { lane, lease },
            );
            return Err(GlobalCkfIngestionError::DirtyToAppliedAgeExceeded { lane });
        }
        let requested = admission.outstanding_images.saturating_add(image_count);
        if requested > route.control.max_outstanding_images {
            let lease = admission.lease;
            let _ = retire_admission_locked(&self.indexer, &route.control, lane, &mut admission);
            report_fault(
                &self.fault_tx,
                GlobalCkfIngestionFault::OutstandingImageLimit {
                    lane,
                    lease,
                    requested,
                    limit: route.control.max_outstanding_images,
                },
            );
            return Err(GlobalCkfIngestionError::OutstandingImageLimit {
                lane,
                requested,
                limit: route.control.max_outstanding_images,
            });
        }
        let Some(delta_id) = admission.next_delta_id.checked_add(1) else {
            permanently_fence_admission_locked(&self.indexer, &route.control, lane, &mut admission);
            return Err(GlobalCkfIngestionError::AdmissionEpochExhausted { lane });
        };
        admission.next_delta_id = delta_id;
        admission.outstanding_images = requested;
        admission.outstanding_deltas.push_back(OutstandingDelta {
            id: delta_id,
            images: image_count,
            submitted_at,
        });
        let permit = DeltaPermit {
            control: Arc::clone(&route.control),
            id: delta_id,
        };
        match self.workers[route.worker]
            .sender
            .try_send(IngestionCommand::Delta {
                lane,
                epoch,
                delta,
                _permit: permit,
            }) {
            Ok(()) => Ok(()),
            Err(flume::TrySendError::Full(command)) => {
                // Image permits are lane-attributable, but the physical Flume queue is shared by
                // every lane sticky to this worker. Exact queue-slot fairness would require
                // per-lane subqueues or a scheduler; this failure retires only the submitting lane.
                let lease = admission.lease;
                let retirement =
                    retire_admission_locked(&self.indexer, &route.control, lane, &mut admission);
                drop(admission);
                drop(command);
                retirement?;
                report_fault(
                    &self.fault_tx,
                    GlobalCkfIngestionFault::Saturated { lane, lease },
                );
                Err(GlobalCkfIngestionError::Saturated { lane })
            }
            Err(flume::TrySendError::Disconnected(command)) => {
                drop(admission);
                drop(command);
                self.mark_worker_failed(route.worker);
                Err(GlobalCkfIngestionError::WorkerFailed {
                    worker: route.worker,
                })
            }
        }
    }

    pub fn complete_drain(
        &self,
        marker: ConsumerDrainMarker,
    ) -> Result<GlobalCkfIngestOutcome, GlobalCkfIngestionError> {
        let lane = usize::from(marker.lease().physical_lane());
        let route = self.route(lane)?;
        let _transition = route.control.transition.lock();
        self.ensure_worker(route.worker)?;
        let epoch = {
            let admission = route.control.admission.lock();
            if admission.state != LaneAdmissionState::Active
                || admission.lease != Some(marker.lease())
            {
                return Err(GlobalCkfIngestionError::LaneRetired { lane });
            }
            admission.epoch
        };
        let (response_tx, response_rx) = flume::bounded(1);
        let command = IngestionCommand::Drain {
            lane,
            epoch,
            marker,
            response: response_tx,
        };
        self.send_control(route, lane, epoch, command)?;
        self.await_response(route, lane, epoch, response_rx)?
    }

    fn route(&self, lane: usize) -> Result<&LaneRoute, GlobalCkfIngestionError> {
        self.routes
            .get(lane)
            .and_then(Option::as_ref)
            .ok_or(GlobalCkfIngestionError::UnconfiguredLane { lane })
    }

    fn ensure_worker(&self, worker: usize) -> Result<(), GlobalCkfIngestionError> {
        if self.health.is_failed(worker) || self.workers[worker].sender.is_disconnected() {
            return Err(GlobalCkfIngestionError::WorkerFailed { worker });
        }
        Ok(())
    }

    fn send_control(
        &self,
        route: &LaneRoute,
        lane: usize,
        epoch: u64,
        command: IngestionCommand,
    ) -> Result<(), GlobalCkfIngestionError> {
        match self.workers[route.worker]
            .sender
            .send_timeout(command, self.control_timeout)
        {
            Ok(()) => Ok(()),
            Err(flume::SendTimeoutError::Timeout(_)) => {
                self.retire_on_timeout(route, lane, epoch)?;
                Err(GlobalCkfIngestionError::ControlTimeout { lane })
            }
            Err(flume::SendTimeoutError::Disconnected(_)) => {
                self.mark_worker_failed(route.worker);
                Err(GlobalCkfIngestionError::WorkerFailed {
                    worker: route.worker,
                })
            }
        }
    }

    fn await_response<T>(
        &self,
        route: &LaneRoute,
        lane: usize,
        epoch: u64,
        response: flume::Receiver<T>,
    ) -> Result<T, GlobalCkfIngestionError> {
        match response.recv_timeout(self.control_timeout) {
            Ok(value) => Ok(value),
            Err(flume::RecvTimeoutError::Timeout) => {
                self.retire_on_timeout(route, lane, epoch)?;
                Err(GlobalCkfIngestionError::ControlTimeout { lane })
            }
            Err(flume::RecvTimeoutError::Disconnected) => {
                if self.health.is_failed(route.worker)
                    || self.workers[route.worker].sender.is_disconnected()
                {
                    return Err(GlobalCkfIngestionError::WorkerFailed {
                        worker: route.worker,
                    });
                }
                Err(GlobalCkfIngestionError::LaneRetired { lane })
            }
        }
    }

    fn retire_on_timeout(
        &self,
        route: &LaneRoute,
        lane: usize,
        epoch: u64,
    ) -> Result<(), GlobalCkfIngestionError> {
        let mut admission = route.control.admission.lock();
        if admission.epoch != epoch {
            return Ok(());
        }
        retire_admission_locked(&self.indexer, &route.control, lane, &mut admission)
    }

    fn mark_worker_failed(&self, worker: usize) {
        if self.health.is_failed(worker) {
            return;
        }
        self.health.record_failure(worker);
        for (lane, route) in self.routes.iter().enumerate() {
            if route.as_ref().is_some_and(|route| route.worker == worker) {
                self.indexer.retire_lane_readiness(lane);
            }
        }
        report_fault(
            &self.fault_tx,
            GlobalCkfIngestionFault::WorkerFailed { worker },
        );
    }

    #[cfg(test)]
    fn pause_lane_worker(&self, lane: usize) -> (flume::Receiver<()>, flume::Sender<()>) {
        let worker = self.worker_for_lane(lane).unwrap();
        let (entered_tx, entered_rx) = flume::bounded(1);
        let (release_tx, release_rx) = flume::bounded(1);
        self.workers[worker]
            .sender
            .send(IngestionCommand::Pause {
                entered: entered_tx,
                release: release_rx,
            })
            .unwrap();
        (entered_rx, release_tx)
    }

    #[cfg(test)]
    fn crash_lane_worker(&self, lane: usize) {
        let worker = self.worker_for_lane(lane).unwrap();
        self.workers[worker]
            .sender
            .send(IngestionCommand::Crash)
            .unwrap();
    }

    #[cfg(test)]
    fn pause_next_snapshot_activation(
        &self,
        lane: usize,
    ) -> (flume::Receiver<()>, flume::Sender<()>) {
        let route = self.route(lane).unwrap();
        let (entered_tx, entered_rx) = flume::bounded(1);
        let (release_tx, release_rx) = flume::bounded(1);
        *route.control.snapshot_activation_pause.lock() = Some((entered_tx, release_rx));
        (entered_rx, release_tx)
    }

    #[cfg(test)]
    fn retire_lane_for_test(&self, lane: usize) -> Result<(), GlobalCkfIngestionError> {
        let route = self.route(lane)?;
        let mut admission = route.control.admission.lock();
        retire_admission_locked(&self.indexer, &route.control, lane, &mut admission)
    }

    #[cfg(test)]
    fn exhaust_lane_epoch_for_test(&self, lane: usize) {
        let route = self.route(lane).unwrap();
        let mut admission = route.control.admission.lock();
        admission.epoch = u64::MAX;
        route
            .control
            .published_epoch
            .store(u64::MAX, Ordering::Release);
    }
}

impl Drop for GlobalCkfIngestionPool {
    fn drop(&mut self) {
        self.lag_watchdog_stop.store(true, Ordering::Release);
        if let Some(watchdog) = self.lag_watchdog.take() {
            watchdog.thread().unpark();
            let _ = watchdog.join();
        }

        // Fail closed before waiting for thread exit. A command already in application rechecks
        // this epoch before returning, and queued tail commands become no-ops.
        for (lane, route) in self.routes.iter().enumerate() {
            let Some(route) = route else {
                continue;
            };
            let mut admission = route.control.admission.lock();
            let _ = retire_admission_locked(&self.indexer, &route.control, lane, &mut admission);
        }

        for worker in &mut self.workers {
            // A full queue is allowed to drain after its last sender is dropped. Do not make pool
            // destruction wait forever merely to place a shutdown message behind a wedged lane.
            let should_join = !matches!(
                worker.sender.try_send(IngestionCommand::Shutdown),
                Err(flume::TrySendError::Full(_))
            );
            if should_join && let Some(join) = worker.join.take() {
                let _ = join.join();
            }
        }
    }
}

fn sticky_worker(consumer: u64, lane: u8, worker_count: usize) -> usize {
    let mut value = consumer ^ u64::from(lane).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    (value as usize) % worker_count
}

#[derive(Clone, Copy)]
struct LaneMessageScope {
    lane: usize,
    epoch: u64,
    lease: LaneLease,
}

impl LaneMessageScope {
    const fn new(lane: usize, epoch: u64, lease: LaneLease) -> Self {
        Self { lane, epoch, lease }
    }
}

fn observe_outcome(
    indexer: &GlobalCkfIndexer,
    fault_tx: &flume::Sender<GlobalCkfIngestionFault>,
    control: &LaneControl,
    scope: LaneMessageScope,
    outcome: GlobalCkfIngestOutcome,
    failure_disposition: Option<CkfFailureDisposition>,
) {
    let GlobalCkfIngestOutcome::LaneDeactivated { fault } = outcome else {
        debug_assert!(failure_disposition.is_none());
        return;
    };
    let disposition = failure_disposition
        .expect("a current-lane deactivation must carry a commit-domain disposition");
    match disposition.action {
        CkfFailureAction::RetrySnapshot => {
            // Snapshot validation failed while the lane was already inactive and before writes.
            // Preserve the current assignment so the same lease can install a corrected image.
        }
        CkfFailureAction::DeactivateAndSnapshot => {
            retire_generation(indexer, control, scope.lane, scope.epoch);
        }
        unexpected @ (CkfFailureAction::ContinueCapacityOmission
        | CkfFailureAction::ReportResourceFailure
        | CkfFailureAction::RejectSource
        | CkfFailureAction::FenceAndRebuildProducer) => {
            panic!("unexpected consumer-lane failure action {unexpected:?}")
        }
    }
    report_fault(
        fault_tx,
        GlobalCkfIngestionFault::LaneDeactivated {
            lane: scope.lane,
            lease: Some(scope.lease),
            fault,
            disposition,
        },
    );
}

fn retire_generation(indexer: &GlobalCkfIndexer, control: &LaneControl, lane: usize, epoch: u64) {
    let mut admission = control.admission.lock();
    if admission.epoch != epoch {
        return;
    }
    let _ = retire_admission_locked(indexer, control, lane, &mut admission);
}

fn retire_admission_locked(
    indexer: &GlobalCkfIndexer,
    control: &LaneControl,
    lane: usize,
    admission: &mut LaneAdmission,
) -> Result<(), GlobalCkfIngestionError> {
    let Some(epoch) = admission.epoch.checked_add(1) else {
        permanently_fence_admission_locked(indexer, control, lane, admission);
        return Err(GlobalCkfIngestionError::AdmissionEpochExhausted { lane });
    };
    // Clear readiness while the admission guard excludes guarded snapshot activation. A query
    // that captured the old bit may finish, but no later activation from this epoch can win a
    // race and republish it.
    indexer.retire_lane_readiness(lane);
    admission.epoch = epoch;
    admission.state = LaneAdmissionState::Retired;
    admission.lease = None;
    control.published_epoch.store(epoch, Ordering::Release);
    Ok(())
}

fn permanently_fence_admission_locked(
    indexer: &GlobalCkfIndexer,
    control: &LaneControl,
    lane: usize,
    admission: &mut LaneAdmission,
) {
    indexer.retire_lane_readiness(lane);
    admission.state = LaneAdmissionState::Retired;
    admission.lease = None;
    control.permanently_fenced.store(true, Ordering::Release);
}

fn oldest_outstanding_age(admission: &LaneAdmission, now: Instant) -> Option<Duration> {
    admission
        .outstanding_deltas
        .front()
        .map(|delta| now.saturating_duration_since(delta.submitted_at))
}

fn fence_expired_lanes(
    indexer: &GlobalCkfIndexer,
    fault_tx: &flume::Sender<GlobalCkfIngestionFault>,
    lanes: &[(usize, Arc<LaneControl>)],
    now: Instant,
) {
    for (lane, control) in lanes {
        let mut admission = control.admission.lock();
        if admission.state != LaneAdmissionState::Active
            || oldest_outstanding_age(&admission, now)
                .is_none_or(|age| age < control.max_dirty_to_applied_age)
        {
            continue;
        }
        let lease = admission.lease;
        let _ = retire_admission_locked(indexer, control, *lane, &mut admission);
        drop(admission);
        report_fault(
            fault_tx,
            GlobalCkfIngestionFault::DirtyToAppliedAgeExceeded { lane: *lane, lease },
        );
    }
}

fn report_fault(fault_tx: &flume::Sender<GlobalCkfIngestionFault>, fault: GlobalCkfIngestionFault) {
    // Lane retirement/readiness is authoritative. Repeated notification traffic is deliberately
    // coalesced when the bounded diagnostic channel is full.
    let _ = fault_tx.try_send(fault);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{
        CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, PoolId, RoutingScopeId,
    };
    use crate::indexer::cuckoo::{
        CkfCommitState, CkfConfig, CkfFailureAction, CkfFailureDomain, ConsumerInstanceId,
        DcCkfState, GlobalCkfBucketImage, GlobalCkfManifest, PrefixSearchConfig,
    };

    struct Fixture {
        pool: GlobalCkfIngestionPool,
        identity: ProducerIdentity,
        lease: LaneLease,
        bucket_count: usize,
        lane: usize,
    }

    impl Fixture {
        fn new(queue_capacity: usize) -> Self {
            Self::new_with_limits(queue_capacity, None, Duration::from_secs(10))
        }

        fn new_with_limits(
            queue_capacity: usize,
            max_outstanding_images_per_lane: Option<usize>,
            max_dirty_to_applied_age: Duration,
        ) -> Self {
            let producer = DcCkfState::new(CkfConfig::new(64)).unwrap();
            let format = producer.format();
            let domain = IndexerDomainId::new(
                CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
                RoutingScopeId::new([9; 16], IdentitySource::Explicit),
            );
            let dc = DcId::new(7);
            let pool_id = PoolId::new(domain, dc);
            let consumer = ConsumerInstanceId::new(11);
            let lane = 3;
            let mut lanes = [None; super::super::DC_COUNT];
            lanes[lane] = Some(pool_id);
            let manifest = GlobalCkfManifest::new(consumer, domain, format, lanes).unwrap();
            let indexer = GlobalCkfIndexer::new(manifest, PrefixSearchConfig::default()).unwrap();
            let pool = GlobalCkfIngestionPool::new(
                indexer,
                GlobalCkfIngestionPoolConfig {
                    worker_count: 1,
                    queue_capacity,
                    control_timeout: Duration::from_secs(1),
                    max_outstanding_images_per_lane,
                    max_dirty_to_applied_age,
                },
            )
            .unwrap();
            let identity = ProducerIdentity::new(pool_id, 13, 1, format);
            let lease = LaneLease::new(consumer, lane as u8, 1);
            Self {
                pool,
                identity,
                lease,
                bucket_count: format.bucket_count(),
                lane,
            }
        }

        fn assign_and_snapshot(&self, sequence: u64) {
            self.pool.assign(self.identity, self.lease).unwrap();
            let snapshot = GlobalCkfSnapshot::new(
                self.identity,
                self.lease,
                sequence,
                vec![0; self.bucket_count].into_boxed_slice(),
            );
            assert_eq!(
                self.pool.install_snapshot(snapshot).unwrap(),
                GlobalCkfIngestOutcome::SnapshotInstalled { sequence }
            );
        }
    }

    #[test]
    fn async_delta_precedes_awaited_drain_on_sticky_lane() {
        let fixture = Fixture::new(8);
        fixture.assign_and_snapshot(4);
        let delta = GlobalCkfDelta::new(
            fixture.identity,
            fixture.lease,
            4,
            5,
            vec![GlobalCkfBucketImage::new(0, 17)],
        );
        fixture.pool.submit_delta(delta).unwrap();

        assert_eq!(
            fixture
                .pool
                .complete_drain(ConsumerDrainMarker::new(fixture.lease, 5))
                .unwrap(),
            GlobalCkfIngestOutcome::DrainAcknowledged {
                installed_sequence: 5
            }
        );
        assert_eq!(fixture.pool.indexer().ready_lanes(), 1u16 << fixture.lane);
    }

    #[test]
    fn current_stream_gap_is_known_unchanged_and_deactivates_only_consumer_lane() {
        let fixture = Fixture::new(8);
        fixture.assign_and_snapshot(4);
        let gap = GlobalCkfDelta::new(
            fixture.identity,
            fixture.lease,
            5,
            6,
            vec![GlobalCkfBucketImage::new(0, 17)],
        );
        fixture.pool.submit_delta(gap).unwrap();

        let deadline = Instant::now() + Duration::from_secs(1);
        let fault = loop {
            if let Some(fault) = fixture.pool.try_recv_fault() {
                break fault;
            }
            assert!(Instant::now() < deadline, "gap fault was not reported");
            thread::yield_now();
        };
        assert!(matches!(
            fault,
            GlobalCkfIngestionFault::LaneDeactivated {
                fault: GlobalCkfLaneFault::BaseSequenceMismatch { .. },
                ..
            }
        ));
        let disposition = fault.failure_disposition();
        assert_eq!(disposition.domain, CkfFailureDomain::ConsumerLane);
        assert_eq!(disposition.commit, CkfCommitState::KnownUnchanged);
        assert_eq!(disposition.action, CkfFailureAction::DeactivateAndSnapshot);
        assert_eq!(fixture.pool.indexer().ready_lanes(), 0);
    }

    #[test]
    fn stale_assignment_cannot_retire_ready_replacement() {
        let fixture = Fixture::new(8);
        fixture.assign_and_snapshot(4);
        let replacement = LaneLease::new(
            fixture.lease.consumer_instance(),
            fixture.lane as u8,
            fixture.lease.assignment_epoch() + 1,
        );
        fixture.pool.assign(fixture.identity, replacement).unwrap();
        let replacement_snapshot = GlobalCkfSnapshot::new(
            fixture.identity,
            replacement,
            10,
            vec![0; fixture.bucket_count].into_boxed_slice(),
        );
        assert!(matches!(
            fixture.pool.install_snapshot(replacement_snapshot),
            Ok(GlobalCkfIngestOutcome::SnapshotInstalled { sequence: 10 })
        ));

        assert_eq!(
            fixture.pool.assign(fixture.identity, fixture.lease),
            Err(GlobalCkfIngestionError::Assignment(
                GlobalCkfAssignmentError::StaleAssignmentEpoch {
                    minimum_exclusive: replacement.assignment_epoch(),
                    actual: fixture.lease.assignment_epoch(),
                }
            ))
        );
        assert_eq!(fixture.pool.indexer().ready_lanes(), 1u16 << fixture.lane);
        assert_eq!(
            fixture
                .pool
                .complete_drain(ConsumerDrainMarker::new(replacement, 10))
                .unwrap(),
            GlobalCkfIngestOutcome::DrainAcknowledged {
                installed_sequence: 10
            }
        );
    }

    #[test]
    fn foreign_assignment_validation_preserves_ready_lane() {
        let fixture = Fixture::new(8);
        fixture.assign_and_snapshot(4);
        let foreign_domain = IndexerDomainId::new(
            CacheSemanticsId::new([99; 16], IdentitySource::Explicit),
            fixture.identity.indexer_domain().routing_scope(),
        );
        let foreign_identity = ProducerIdentity::new(
            PoolId::new(foreign_domain, fixture.identity.dc_id()),
            fixture.identity.producer_incarnation(),
            fixture.identity.layout_generation(),
            fixture.identity.format(),
        );
        let foreign_lease = LaneLease::new(
            fixture.lease.consumer_instance(),
            fixture.lane as u8,
            fixture.lease.assignment_epoch() + 1,
        );

        assert!(matches!(
            fixture.pool.assign(foreign_identity, foreign_lease),
            Err(GlobalCkfIngestionError::Assignment(
                GlobalCkfAssignmentError::WrongLaneOwner { .. }
            ))
        ));
        assert_eq!(fixture.pool.indexer().ready_lanes(), 1u16 << fixture.lane);
        assert_eq!(
            fixture
                .pool
                .complete_drain(ConsumerDrainMarker::new(fixture.lease, 4))
                .unwrap(),
            GlobalCkfIngestOutcome::DrainAcknowledged {
                installed_sequence: 4
            }
        );
    }

    #[test]
    fn stale_active_snapshot_is_ignored_without_rollback() {
        let fixture = Fixture::new(8);
        fixture.assign_and_snapshot(4);
        let stale = GlobalCkfSnapshot::new(
            fixture.identity,
            fixture.lease,
            3,
            vec![u64::MAX; fixture.bucket_count].into_boxed_slice(),
        );

        assert_eq!(
            fixture.pool.install_snapshot(stale).unwrap(),
            GlobalCkfIngestOutcome::IgnoredStaleOrDuplicate {
                installed_sequence: 4
            }
        );
        assert_eq!(fixture.pool.indexer().ready_lanes(), 1u16 << fixture.lane);
        assert!(matches!(
            fixture
                .pool
                .complete_drain(ConsumerDrainMarker::new(fixture.lease, 4)),
            Ok(GlobalCkfIngestOutcome::DrainAcknowledged { .. })
        ));
    }

    #[test]
    fn inactive_snapshot_failure_keeps_lease_retryable_without_producer_recovery() {
        let fixture = Fixture::new(8);
        fixture
            .pool
            .assign(fixture.identity, fixture.lease)
            .unwrap();
        let malformed = GlobalCkfSnapshot::new(
            fixture.identity,
            fixture.lease,
            4,
            vec![0; fixture.bucket_count - 1].into_boxed_slice(),
        );

        assert!(matches!(
            fixture.pool.install_snapshot(malformed),
            Ok(GlobalCkfIngestOutcome::LaneDeactivated {
                fault: GlobalCkfLaneFault::InvalidSnapshotBucketCount { .. }
            })
        ));
        let fault = fixture.pool.try_recv_fault().unwrap();
        let disposition = fault.failure_disposition();
        assert_eq!(disposition.domain, CkfFailureDomain::ConsumerLane);
        assert_eq!(disposition.commit, CkfCommitState::KnownUnchanged);
        assert_eq!(disposition.action, CkfFailureAction::RetrySnapshot);

        let corrected = GlobalCkfSnapshot::new(
            fixture.identity,
            fixture.lease,
            4,
            vec![0; fixture.bucket_count].into_boxed_slice(),
        );
        assert_eq!(
            fixture.pool.install_snapshot(corrected).unwrap(),
            GlobalCkfIngestOutcome::SnapshotInstalled { sequence: 4 }
        );
        assert_eq!(fixture.pool.indexer().ready_lanes(), 1u16 << fixture.lane);
    }

    #[test]
    fn retirement_cannot_race_snapshot_activation_back_to_ready() {
        let fixture = Fixture::new(8);
        fixture
            .pool
            .assign(fixture.identity, fixture.lease)
            .unwrap();
        let snapshot = GlobalCkfSnapshot::new(
            fixture.identity,
            fixture.lease,
            4,
            vec![0; fixture.bucket_count].into_boxed_slice(),
        );
        let (entered, release) = fixture.pool.pause_next_snapshot_activation(fixture.lane);

        thread::scope(|scope| {
            let install = scope.spawn(|| fixture.pool.install_snapshot(snapshot));
            entered.recv_timeout(Duration::from_secs(1)).unwrap();
            fixture.pool.retire_lane_for_test(fixture.lane).unwrap();
            assert_eq!(fixture.pool.indexer().ready_lanes(), 0);
            release.send(()).unwrap();
            assert_eq!(
                install.join().unwrap(),
                Err(GlobalCkfIngestionError::LaneRetired { lane: fixture.lane })
            );
        });
        assert_eq!(fixture.pool.indexer().ready_lanes(), 0);
    }

    #[test]
    fn admission_epoch_exhaustion_fences_lane_fail_closed() {
        let fixture = Fixture::new(8);
        fixture.assign_and_snapshot(4);
        fixture.pool.exhaust_lane_epoch_for_test(fixture.lane);

        assert_eq!(
            fixture.pool.retire_lane_for_test(fixture.lane),
            Err(GlobalCkfIngestionError::AdmissionEpochExhausted { lane: fixture.lane })
        );
        assert_eq!(fixture.pool.indexer().ready_lanes(), 0);
        let replacement = LaneLease::new(
            fixture.lease.consumer_instance(),
            fixture.lane as u8,
            fixture.lease.assignment_epoch() + 1,
        );
        assert_eq!(
            fixture.pool.assign(fixture.identity, replacement),
            Err(GlobalCkfIngestionError::AdmissionEpochExhausted { lane: fixture.lane })
        );
    }

    #[test]
    fn outstanding_image_limit_is_lane_attributable_and_fail_closed() {
        let fixture = Fixture::new_with_limits(8, Some(1), Duration::from_secs(10));
        fixture.assign_and_snapshot(0);
        let delta = GlobalCkfDelta::new(
            fixture.identity,
            fixture.lease,
            0,
            1,
            vec![
                GlobalCkfBucketImage::new(0, 1),
                GlobalCkfBucketImage::new(1, 2),
            ],
        );

        assert_eq!(
            fixture.pool.submit_delta(delta),
            Err(GlobalCkfIngestionError::OutstandingImageLimit {
                lane: fixture.lane,
                requested: 2,
                limit: 1,
            })
        );
        assert_eq!(fixture.pool.indexer().ready_lanes(), 0);
        assert!(matches!(
            fixture.pool.try_recv_fault(),
            Some(GlobalCkfIngestionFault::OutstandingImageLimit { lane, .. })
                if lane == fixture.lane
        ));
    }

    #[test]
    fn stalled_delta_retires_lane_at_dirty_to_applied_deadline() {
        let fixture = Fixture::new_with_limits(8, None, Duration::from_millis(20));
        fixture.assign_and_snapshot(0);
        let (entered, release) = fixture.pool.pause_lane_worker(fixture.lane);
        entered.recv_timeout(Duration::from_secs(1)).unwrap();
        let delta = GlobalCkfDelta::new(
            fixture.identity,
            fixture.lease,
            0,
            1,
            vec![GlobalCkfBucketImage::new(0, 1)],
        );
        fixture.pool.submit_delta(delta).unwrap();

        let deadline = Instant::now() + Duration::from_secs(1);
        while fixture.pool.indexer().ready_lanes() != 0 && Instant::now() < deadline {
            thread::yield_now();
        }
        let mut fault = None;
        while fault.is_none() && Instant::now() < deadline {
            fault = fixture.pool.try_recv_fault();
            thread::yield_now();
        }
        release.send(()).unwrap();
        assert_eq!(fixture.pool.indexer().ready_lanes(), 0);
        assert!(matches!(
            fault,
            Some(GlobalCkfIngestionFault::DirtyToAppliedAgeExceeded { lane, .. })
                if lane == fixture.lane
        ));
    }

    #[test]
    fn worker_health_tracks_each_failed_worker() {
        let health = WorkerHealth::new(3);
        health.record_failure(0);
        health.record_failure(2);
        assert!(health.is_failed(0));
        assert!(!health.is_failed(1));
        assert!(health.is_failed(2));
        assert_eq!(health.failed_worker(), Some(0));
    }

    #[test]
    fn saturation_retires_readiness_and_discards_queued_tail() {
        let fixture = Fixture::new(1);
        fixture.assign_and_snapshot(0);
        let (entered, release) = fixture.pool.pause_lane_worker(fixture.lane);
        entered.recv_timeout(Duration::from_secs(1)).unwrap();

        let first = GlobalCkfDelta::new(
            fixture.identity,
            fixture.lease,
            0,
            1,
            vec![GlobalCkfBucketImage::new(0, 1)],
        );
        fixture.pool.submit_delta(first).unwrap();
        let second = GlobalCkfDelta::new(
            fixture.identity,
            fixture.lease,
            1,
            2,
            vec![GlobalCkfBucketImage::new(1, 2)],
        );
        assert_eq!(
            fixture.pool.submit_delta(second),
            Err(GlobalCkfIngestionError::Saturated { lane: fixture.lane })
        );
        assert_eq!(fixture.pool.indexer().ready_lanes(), 0);
        release.send(()).unwrap();
        assert_eq!(
            fixture
                .pool
                .complete_drain(ConsumerDrainMarker::new(fixture.lease, 1)),
            Err(GlobalCkfIngestionError::LaneRetired { lane: fixture.lane })
        );
        assert!(matches!(
            fixture.pool.try_recv_fault(),
            Some(GlobalCkfIngestionFault::Saturated { lane, .. }) if lane == fixture.lane
        ));
    }

    #[test]
    fn worker_failure_is_observable_and_clears_readiness() {
        let fixture = Fixture::new(4);
        fixture.assign_and_snapshot(0);
        let worker = fixture.pool.worker_for_lane(fixture.lane).unwrap();
        fixture.pool.crash_lane_worker(fixture.lane);

        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        while fixture.pool.failed_worker().is_none() && std::time::Instant::now() < deadline {
            thread::yield_now();
        }
        assert_eq!(fixture.pool.failed_worker(), Some(worker));
        assert_eq!(fixture.pool.indexer().ready_lanes(), 0);
        let deadline = Instant::now() + Duration::from_secs(1);
        let fault = loop {
            if let Some(fault) = fixture.pool.try_recv_fault() {
                break fault;
            }
            assert!(Instant::now() < deadline, "worker fault was not reported");
            thread::yield_now();
        };
        let disposition = fault.failure_disposition();
        assert_eq!(disposition.domain, CkfFailureDomain::ConsumerLane);
        assert_eq!(disposition.commit, CkfCommitState::Uncertain);
        assert_eq!(disposition.action, CkfFailureAction::DeactivateAndSnapshot);
        let delta = GlobalCkfDelta::new(
            fixture.identity,
            fixture.lease,
            0,
            1,
            vec![GlobalCkfBucketImage::new(0, 1)],
        );
        assert_eq!(
            fixture.pool.submit_delta(delta),
            Err(GlobalCkfIngestionError::WorkerFailed { worker })
        );
    }

    #[test]
    fn sticky_assignment_is_stable_for_consumer_and_lane() {
        let fixture = Fixture::new(4);
        let worker = fixture.pool.worker_for_lane(fixture.lane).unwrap();
        assert_eq!(fixture.pool.worker_for_lane(fixture.lane), Some(worker));
        assert_eq!(
            worker,
            sticky_worker(
                fixture.lease.consumer_instance().get(),
                fixture.lane as u8,
                1
            )
        );
    }
}
