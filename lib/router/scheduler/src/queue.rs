// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeSet, HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use arc_swap::ArcSwapOption;
use crossbeam_queue::SegQueue;
use parking_lot::Mutex;
use rustc_hash::{FxHashMap, FxHashSet};
use tokio::sync::{mpsc, oneshot, watch};
use tokio::time::Instant;

use self::capacity::CapacityUpdates;
use super::filter::{RoutingEligibility, worker_config_for_rank};
use super::overlap::SelectedWorkerTierSnapshot;
use super::overlap_refresh::{
    NoopOverlapScoresRefresh, OVERLAP_REFRESH_TIMEOUT, OverlapScoresRefresh, RefreshedOverlap,
    read_overlap_refresh_after, refresh_overlap, should_refresh_overlap,
};
use super::prefill_load::{PrefillLoadEstimator, effective_prefill_tokens};
use super::selector::{
    CandidatePreparer, CandidatePreparerExt, PreparedCandidates, WorkerInputs, WorkerLoadInputExt,
    WorkerSelectionInput, WorkerSelector, worker_input_rows,
};
use super::types::{
    AdvisorySchedulingResponse, AdvisoryWorkerLoad, AttemptId, AvailableWorkerIds,
    KvSchedulerError, NonMaxOverlapSelection, NonMaxOverlapSelectionObserver,
    OverloadedWorkerProvider, SchedulingContext, SchedulingRequest, SchedulingResponse,
    WorkerAvailabilityProvider, WorkerIdSet,
};
use crate::protocols::{WorkerConfigLike, WorkerSelectionResult};
use crate::selector::WorkerCapacityExt;
use crate::types::RejectThresholdsProvider;
use dynamo_admission::classifier::{ClassificationOverrides, ClassifyRequest};
use dynamo_admission::placement::WorkerPlacement;
use dynamo_admission::queue::{PolicyQueue, QueueMetadata, QueueSnapshot};
use dynamo_admission::{PolicyClassConfig, PolicyProfile};
use dynamo_kv_hashing::LocalBlockHash;
use dynamo_kv_indexer::ResidentBlockCountsHandle;
use dynamo_router_protocol::worker::{WorkerId, WorkerWithDpRank};
use dynamo_slot_tracker::busy::{BusyThresholds, FullRankCounts};
use dynamo_slot_tracker::wire::PrefillLoadHint;
use dynamo_slot_tracker::{
    ActiveSequencesMultiWorker, SequenceError, SequencePublisher, SequenceRequest,
    WorkerLoadProjection, WorkerLoadReader,
};
use dynamo_worker_selection::{RuntimePolicyConfig, RuntimePolicyError, RuntimePolicyUpdate};
use dynamo_worker_selection::{SelectionTrace, SelectionTraceSpec};
use dynamo_worker_selection::{WorkerCapacity, WorkerInputRows, WorkerLoadInput};

// The fallback lives beside `WorkerCapacity`; this path stays for existing importers.
pub use dynamo_router_protocol::worker::DEFAULT_MAX_BATCHED_TOKENS;

mod capacity;

/// Callers build an Enqueue's prepared rows only once they hold a slot, so this also
/// bounds the rows in flight: ranks x ~120 B each, about 60 MiB at 2048 ranks. A
/// backlog past it waits, in order, for a slot.
const ADMISSION_CHANNEL_CAPACITY: usize = 256;

/// Commands the actor takes from the channel in one receive. A receive spends one unit of
/// the task's cooperative budget whatever it returns, so a backlog costs the actor a yield
/// to the runtime every 128 batches, not every 128 commands. The size also sets how long the
/// actor holds its thread under a backlog: at 32 its longest polls reached 0.33 to 0.66 s.
/// A taken batch has freed its channel slots, so its rows add to those the capacity bounds.
const ADMISSION_BATCH: usize = 8;

struct ClassQueueCounters {
    pending_count: AtomicUsize,
    pending_isl_tokens: AtomicUsize,
    pending_cached_tokens: AtomicUsize,
    received_total: AtomicU64,
    rejected_due_time_passed_total: AtomicU64,
    rejected_queue_size_exceeded_total: AtomicU64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClassQueueStats {
    pub pending_count: usize,
    pub pending_isl_tokens: usize,
    pub pending_cached_tokens: usize,
    /// Attempts received by the admission actor; includes retries, excludes advisory probes.
    pub received_total: u64,
    /// Attempts rejected because their due time passed, counted once at rejection. An attempt
    /// whose response receiver was already closed at the rejection check is not counted.
    pub rejected_due_time_passed_total: u64,
    /// Attempts rejected by a class queue request or token limit, counted once at rejection. An
    /// attempt whose response receiver was already closed at the rejection check is not counted.
    pub rejected_queue_size_exceeded_total: u64,
}

struct QueuedRequest {
    request: SchedulingRequest,
    attempt_tx: Option<oneshot::Sender<AttemptId>>,
    lifecycle_transfer: Option<Arc<AdmissionLifecycleTransfer>>,
    enqueue_at: Instant,
    /// When the request expires: its explicit deadline, else its class due time when the
    /// class expires them. It stays fixed through refresh and admission.
    due_at: Option<Instant>,
    block_hashes: Option<Vec<LocalBlockHash>>,
    /// Actor turn that received the request. Its prepared rows last only that turn.
    enqueue_turn: u64,
}

/// Candidate rows built on the caller task, before the request reaches the actor.
///
/// The actor selects from them only when it would build the same rows: the worker
/// configuration, availability, and overload it reads at selection match the ones
/// the rows were built from. Load inputs and the ranks past the host's reject
/// thresholds depend on the actor's earlier bookings, so it fills the loads and drops
/// the rows of workers whose every rank is full at selection.
struct PreparedAdmission {
    candidates: PreparedCandidates,
    /// Configuration generation, loaded before the configuration was read.
    workers_generation: u64,
    available_worker_ids: Option<Arc<WorkerIdSet>>,
    overloaded_worker_ids: Option<Arc<HashSet<WorkerId>>>,
}

impl PreparedAdmission {
    /// The rows, when built from the inputs the actor selects with.
    fn into_current(
        self,
        workers_generation: u64,
        available_worker_ids: Option<&Arc<WorkerIdSet>>,
        overloaded_worker_ids: Option<&Arc<HashSet<WorkerId>>>,
    ) -> Option<PreparedCandidates> {
        // Availability and overload are each one shared snapshot until they change,
        // so these usually compare pointers.
        (self.workers_generation == workers_generation
            && self.available_worker_ids.as_ref() == available_worker_ids
            && self.overloaded_worker_ids.as_ref() == overloaded_worker_ids)
            .then_some(self.candidates)
    }
}

/// Shared by the handle and the actor, so prepared rows apply the constraints the
/// actor selects with. Only the actor has full ranks: its projection finds them.
fn selection_eligibility<'a, C: WorkerConfigLike>(
    request: &'a SchedulingRequest,
    workers: &HashMap<WorkerId, C>,
    overloaded_worker_ids: Option<&'a HashSet<WorkerId>>,
    available_worker_ids: Option<&'a WorkerIdSet>,
    full_ranks: Option<&'a FullRankCounts>,
    exclusive_affinity: bool,
) -> RoutingEligibility<'a> {
    let mut eligibility = request
        .eligibility_with_overloaded(overloaded_worker_ids)
        .with_available_workers(available_worker_ids);
    if let Some(full_ranks) = full_ranks {
        // Workers whose every rank the projection pass found past the host's reject
        // thresholds are skipped like overloaded ones; nothing is scanned here.
        eligibility = eligibility.with_full_worker_ranks(full_ranks);
    }
    match request.affinity_target {
        Some(target)
            if exclusive_affinity && eligibility.affinity_target_is_eligible(workers, target) =>
        {
            eligibility.with_affinity_target(target)
        }
        _ => eligibility,
    }
}

struct SelectedWorkerForRequest {
    selection: WorkerSelectionResult,
    selection_trace: Option<SelectionTrace>,
    selected_worker_tiers: SelectedWorkerTierSnapshot,
    selected_worker_load: AdvisoryWorkerLoad,
    non_max_overlap_selection: Option<NonMaxOverlapSelection>,
}

/// One expiry timer per class queue, armed at the earliest due time at which an entry of that
/// class expires. A single ordered index over the armed timers lets the actor wait for the
/// earliest without scanning classes.
struct ClassTimers {
    armed: Vec<Option<Instant>>,
    by_deadline: BTreeSet<(Instant, usize)>,
}

impl ClassTimers {
    fn new(class_count: usize) -> Self {
        Self {
            armed: vec![None; class_count],
            by_deadline: BTreeSet::new(),
        }
    }

    /// Arms the class timer at `due_at`, replacing its previous deadline; `None` disarms it.
    fn arm(&mut self, class_index: usize, due_at: Option<Instant>) {
        let armed = &mut self.armed[class_index];
        if *armed == due_at {
            return;
        }
        if let Some(previous) = armed.take() {
            self.by_deadline.remove(&(previous, class_index));
        }
        if let Some(due_at) = due_at {
            self.by_deadline.insert((due_at, class_index));
        }
        *armed = due_at;
    }

    /// Moves the class timer to `due_at` if that is earlier: an arrival can only advance it.
    fn arm_if_earlier(&mut self, class_index: usize, due_at: Instant) {
        if self.armed[class_index].is_none_or(|armed| due_at < armed) {
            self.arm(class_index, Some(due_at));
        }
    }

    fn next_deadline(&self) -> Option<Instant> {
        self.by_deadline.first().map(|&(due_at, _)| due_at)
    }

    /// Disarms and returns a class whose timer is due at `now`, earliest deadline first.
    fn pop_due(&mut self, now: Instant) -> Option<usize> {
        let &(due_at, class_index) = self.by_deadline.first()?;
        if due_at > now {
            return None;
        }
        self.by_deadline.pop_first();
        self.armed[class_index] = None;
        Some(class_index)
    }
}

/// Scheduler work that arrives during an optional overlap refresh.
#[allow(clippy::large_enum_variant)]
enum RefreshEvent {
    ClassTimer,
    Capacity,
    Command(AdmissionCommand),
}

/// How the optional overlap refresh of an extracted candidate ended. Every ending but `Finished`
/// drops the refresh unfinished, and a candidate still dispatched keeps its queued scores. A
/// transient return value, never stored, so its refreshed scores stay unboxed.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
enum RefreshEnd {
    /// The scores to use, or `None` to keep the queued ones.
    Finished(Option<RefreshedOverlap>),
    /// The candidate's due time passed first.
    DueTimePassed,
    /// The candidate's response closed first.
    ResponseClosed,
    /// The caller released the candidate's lifecycle first. The candidate leaves unanswered,
    /// as a queued entry that cleanup removes does.
    Released,
    /// Every command sender is gone. The actor stops once the candidate is settled.
    Shutdown,
    /// The refresh outlasted [`OVERLAP_REFRESH_TIMEOUT`].
    TimedOut,
}

/// What admission needs besides the request, for the direct and queued paths alike.
struct AdmissionContext {
    class_index: usize,
    /// When the request expires: checked again before selection and before booking or
    /// delivery.
    due_at: Option<Instant>,
    attempt_tx: Option<oneshot::Sender<AttemptId>>,
    lifecycle_transfer: Option<Arc<AdmissionLifecycleTransfer>>,
}

#[cfg(test)]
impl AdmissionContext {
    /// A request of `class_index` with no attempt sender or lifecycle transfer.
    fn untracked(class_index: usize, due_at: Option<Instant>) -> Self {
        Self {
            class_index,
            due_at,
            attempt_tx: None,
            lifecycle_transfer: None,
        }
    }
}

/// How an arrival's attempt to enter its class queue ended.
enum Insertion {
    /// The arrival entered the queue.
    Queued,
    /// The arrival was answered with a rejection, counted once, or dropped uncounted because
    /// its receiver was already closed. `exposed` reports whether expiry during the attempt
    /// removed an entry from a class that kept others, which can expose a head that runs.
    Refused { exposed: bool },
}

/// A command's reply, sent when the drain that follows the command ends or first awaits a
/// refresh, or at once for a command applied during a refresh.
#[allow(clippy::large_enum_variant)]
enum CommandReply {
    Enqueue {
        ack_tx: oneshot::Sender<Option<Box<RequestLifecycleLease>>>,
        lease: Option<Box<RequestLifecycleLease>>,
    },
    SelectWithoutAdmission {
        resp_tx: oneshot::Sender<Result<AdvisorySchedulingResponse, KvSchedulerError>>,
        result: Result<AdvisorySchedulingResponse, KvSchedulerError>,
    },
    Update {
        ack_tx: oneshot::Sender<()>,
    },
}

impl CommandReply {
    fn send(self) {
        match self {
            Self::Enqueue { ack_tx, lease } => {
                let _ = ack_tx.send(lease);
            }
            Self::SelectWithoutAdmission { resp_tx, result } => {
                let _ = resp_tx.send(result);
            }
            Self::Update { ack_tx } => {
                let _ = ack_tx.send(());
            }
        }
    }
}

/// How [`SchedulerQueueActor::admit_one`] ended. Only an admitted request spends its DRR
/// charge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Admission {
    /// Booked, and the response was delivered.
    Booked,
    /// An untracked request's selection was delivered.
    Selected,
    /// Not admitted: the due time passed, the response was abandoned, or selection,
    /// booking, or delivery failed.
    Refused,
}

/// The drain a command calls for, and how many blocked lanes it rechecks first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Drain {
    /// Nothing made work ready.
    None,
    /// An arrival entered the queue or was booked. Only lanes of workers with recorded
    /// capacity changes are rechecked, so an older blocked pinned lane competes with later
    /// shared arrivals.
    Enqueued,
    /// Every blocked lane is rechecked.
    Full,
}

impl Drain {
    fn full_if(ready: bool) -> Self {
        if ready { Self::Full } else { Self::None }
    }
}

fn non_max_overlap_selection<C: WorkerConfigLike>(
    workers: &HashMap<WorkerId, C>,
    request: &SchedulingRequest,
    eligibility: RoutingEligibility<'_>,
    selected_worker: WorkerWithDpRank,
    selected_overlap_blocks: f64,
) -> Option<NonMaxOverlapSelection> {
    if eligibility.pinned_worker().is_some() {
        return None;
    }

    let mut highest_overlap = None;
    for (&worker, &overlap_blocks) in &request.overlap.effective_overlap_blocks {
        if overlap_blocks <= selected_overlap_blocks
            || eligibility.validate_worker_rank(workers, worker).is_err()
        {
            continue;
        }
        let is_better = highest_overlap.is_none_or(
            |(current_worker, current_overlap): (WorkerWithDpRank, f64)| {
                overlap_blocks > current_overlap
                    || (overlap_blocks == current_overlap && worker < current_worker)
            },
        );
        if is_better {
            highest_overlap = Some((worker, overlap_blocks));
        }
    }

    let (highest_overlap_worker, highest_overlap_blocks) = highest_overlap?;
    (highest_overlap_blocks > selected_overlap_blocks).then_some(NonMaxOverlapSelection {
        selected_worker,
        highest_overlap_worker,
        highest_overlap_blocks,
        selected_overlap_blocks,
    })
}

/// Highest raw cached-prefix tokens over workers eligible under `eligibility`, starting
/// from the selector's own maximum. Workers with no overlap contribute zero, so only the
/// workers present in the tier overlap maps are visited.
fn best_eligible_raw_cached_tokens<C: WorkerConfigLike>(
    workers: &HashMap<WorkerId, C>,
    request: &SchedulingRequest,
    eligibility: RoutingEligibility<'_>,
    selector_max: usize,
    block_size: u32,
) -> usize {
    let tiers = &request.overlap.tier_overlap_blocks;
    tiers
        .device
        .keys()
        .chain(tiers.host_pinned.keys())
        .chain(tiers.disk.keys())
        .filter(|&&worker| eligibility.validate_worker_rank(workers, worker).is_ok())
        .map(|&worker| request.raw_cached_tokens_for(worker, block_size))
        .fold(selector_max, usize::max)
}

#[allow(clippy::large_enum_variant)]
enum AdmissionCommand {
    PolicyConfig {
        update: Option<RuntimePolicyUpdate>,
        response: oneshot::Sender<Result<RuntimePolicyConfig, RuntimePolicyError>>,
    },
    Enqueue {
        request: SchedulingRequest,
        attempt_tx: Option<oneshot::Sender<AttemptId>>,
        block_hashes: Option<Vec<LocalBlockHash>>,
        prepared: Option<Box<PreparedAdmission>>,
        queue_metadata: QueueMetadata,
        lease: Option<Box<RequestLifecycleLease>>,
        ack_tx: oneshot::Sender<Option<Box<RequestLifecycleLease>>>,
    },
    SelectWithoutAdmission {
        request: SchedulingRequest,
        resp_tx: oneshot::Sender<Result<AdvisorySchedulingResponse, KvSchedulerError>>,
    },
    Update {
        ack_tx: oneshot::Sender<()>,
    },
    Cleanup,
}

/// Control handle for the policy owned by one scheduler actor. Updates do not replace the
/// scheduler, its queued requests, or active bookings. A stopped actor rejects further operations.
pub struct RuntimePolicyControl {
    tx: mpsc::WeakSender<AdmissionCommand>,
}

impl RuntimePolicyControl {
    pub async fn get(&self) -> Result<RuntimePolicyConfig, RuntimePolicyError> {
        self.request(None).await
    }

    pub async fn update(
        &self,
        update: RuntimePolicyUpdate,
    ) -> Result<RuntimePolicyConfig, RuntimePolicyError> {
        self.request(Some(update)).await
    }

    async fn request(
        &self,
        update: Option<RuntimePolicyUpdate>,
    ) -> Result<RuntimePolicyConfig, RuntimePolicyError> {
        let (response, receiver) = oneshot::channel();
        let sender = self.tx.upgrade().ok_or(RuntimePolicyError::Unavailable)?;
        sender
            .send(AdmissionCommand::PolicyConfig { update, response })
            .await
            .map_err(|_| RuntimePolicyError::Unavailable)?;
        drop(sender);
        receiver
            .await
            .map_err(|_| RuntimePolicyError::Unavailable)?
    }
}

pub use dynamo_slot_tracker::booking::SchedulerBookingDescriptor;

#[derive(Debug)]
enum SchedulerCleanupTarget {
    AdmissionLifecycle(Arc<AdmissionLifecycleTransfer>),
    Booking(SchedulerBookingDescriptor),
    ExpiredBooking(SchedulerBookingDescriptor),
}

#[derive(Debug)]
enum AdmissionLifecycleState {
    Unarmed { request_id: String },
    Pending { request_id: String },
    Booking(SchedulerBookingDescriptor),
    Disarmed,
}

#[derive(Debug)]
struct AdmissionLifecycleTransfer {
    state: Mutex<AdmissionLifecycleState>,
}

enum AdmissionLifecycleCleanup {
    Pending { request_id: String },
    Booking(SchedulerBookingDescriptor),
}

impl AdmissionLifecycleTransfer {
    fn new(request_id: String) -> Self {
        Self {
            state: Mutex::new(AdmissionLifecycleState::Unarmed { request_id }),
        }
    }

    fn arm_pending(&self) {
        let mut state = self.state.lock();
        let AdmissionLifecycleState::Unarmed { request_id } = &*state else {
            return;
        };
        *state = AdmissionLifecycleState::Pending {
            request_id: request_id.clone(),
        };
    }

    fn arm_booking(&self, booking: SchedulerBookingDescriptor) {
        let mut state = self.state.lock();
        if matches!(
            *state,
            AdmissionLifecycleState::Unarmed { .. } | AdmissionLifecycleState::Pending { .. }
        ) {
            *state = AdmissionLifecycleState::Booking(booking);
        }
    }

    fn disarm(&self) {
        *self.state.lock() = AdmissionLifecycleState::Disarmed;
    }

    fn cleanup_target(&self) -> Option<AdmissionLifecycleCleanup> {
        let mut state = self.state.lock();
        let cleanup = match &*state {
            AdmissionLifecycleState::Unarmed { .. } | AdmissionLifecycleState::Disarmed => None,
            AdmissionLifecycleState::Pending { request_id } => {
                Some(AdmissionLifecycleCleanup::Pending {
                    request_id: request_id.clone(),
                })
            }
            AdmissionLifecycleState::Booking(booking) => {
                Some(AdmissionLifecycleCleanup::Booking(booking.clone()))
            }
        };
        *state = AdmissionLifecycleState::Disarmed;
        cleanup
    }

    fn is_armed(&self) -> bool {
        matches!(
            *self.state.lock(),
            AdmissionLifecycleState::Pending { .. } | AdmissionLifecycleState::Booking(_)
        )
    }
}

struct AdmissionCleanupEntry {
    target: SchedulerCleanupTarget,
}

#[derive(Default)]
struct AdmissionCleanup {
    dirty: SegQueue<AdmissionCleanupEntry>,
    pending: AtomicBool,
}

impl AdmissionCleanup {
    fn enqueue(&self, cleanup: AdmissionCleanupEntry) -> bool {
        self.dirty.push(cleanup);
        !self.pending.swap(true, AtomicOrdering::AcqRel)
    }

    fn drain(&self) -> Vec<AdmissionCleanupEntry> {
        if !self.pending.load(AtomicOrdering::Acquire) {
            return Vec::new();
        }

        // Drain to a quiescent point to preserve the coalesced-wake handoff. A large burst of
        // active lease drops can delay actor commands; any bounded or interleaved drain
        // must preserve wake correctness and be benchmarked.
        let mut dirty = Vec::new();
        loop {
            while let Some(cleanup) = self.dirty.pop() {
                dirty.push(cleanup);
            }
            self.pending.store(false, AtomicOrdering::Release);
            if self.dirty.is_empty() {
                return dirty;
            }
            self.pending.store(true, AtomicOrdering::Release);
        }
    }
}

fn enqueue_cleanup_entry(
    cleanup: &AdmissionCleanup,
    actor_tx: &mpsc::Sender<AdmissionCommand>,
    entry: AdmissionCleanupEntry,
) {
    if !cleanup.enqueue(entry) {
        return;
    }
    match actor_tx.try_send(AdmissionCommand::Cleanup) {
        Ok(()) | Err(mpsc::error::TrySendError::Full(_)) => {}
        Err(mpsc::error::TrySendError::Closed(_)) => {
            // No actor remains to consume destructor-only cleanup.
            cleanup.drain();
        }
    }
}

/// Cloneable non-blocking ingress for attempt-scoped scheduler cleanup.
#[derive(Clone)]
pub struct SchedulerBookingCleanup {
    cleanup: Arc<AdmissionCleanup>,
    actor_tx: mpsc::Sender<AdmissionCommand>,
    direct_release: DirectBookingRelease,
}

type DirectBookingRelease =
    Arc<dyn Fn(&SchedulerBookingDescriptor) -> Result<(), SequenceError> + Send + Sync>;

/// The scheduler's acknowledgement of an enqueued booking release; `wait` resolves with
/// the release outcome.
pub struct SchedulerCleanupAck {
    response: Result<(), SequenceError>,
}

#[cfg(any(test, feature = "testing"))]
impl SchedulerCleanupAck {
    /// The cleanup outcome, for tests that assert on acknowledgement.
    pub fn response(&self) -> &Result<(), SequenceError> {
        &self.response
    }
}

impl SchedulerCleanupAck {
    pub async fn wait(self) -> Result<(), SequenceError> {
        self.response
    }
}

impl SchedulerBookingCleanup {
    fn enqueue_entry(&self, entry: AdmissionCleanupEntry) {
        enqueue_cleanup_entry(&self.cleanup, &self.actor_tx, entry);
    }

    pub fn enqueue(&self, booking: SchedulerBookingDescriptor) {
        self.enqueue_entry(AdmissionCleanupEntry {
            target: SchedulerCleanupTarget::Booking(booking),
        });
    }

    pub fn enqueue_expired(&self, booking: SchedulerBookingDescriptor) {
        self.enqueue_entry(AdmissionCleanupEntry {
            target: SchedulerCleanupTarget::ExpiredBooking(booking),
        });
    }

    /// Apply explicit release now. The acknowledgement covers the mutation and
    /// capacity notification; pending admission runs asynchronously.
    pub fn enqueue_acknowledged(&self, booking: SchedulerBookingDescriptor) -> SchedulerCleanupAck {
        SchedulerCleanupAck {
            response: (self.direct_release)(&booking),
        }
    }

    pub async fn enqueue_and_wait(
        &self,
        booking: SchedulerBookingDescriptor,
    ) -> Result<(), SequenceError> {
        self.enqueue_acknowledged(booking).wait().await
    }
}

/// A booking whose release this handle owns until `commit` hands it over.
/// Dropping an armed handle frees the booking through the scheduler's cleanup queue.
#[must_use]
pub struct BookingHandle {
    booking: SchedulerBookingDescriptor,
    cleanup: SchedulerBookingCleanup,
    armed: bool,
}

impl std::fmt::Debug for BookingHandle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BookingHandle")
            .field("booking", &self.booking)
            .field("armed", &self.armed)
            .finish()
    }
}

impl BookingHandle {
    /// Hand the booking to a longer-lived owner; the handle stops guarding it.
    #[must_use]
    pub fn commit(mut self) -> SchedulerBookingDescriptor {
        self.armed = false;
        SchedulerBookingDescriptor {
            request_id: std::mem::take(&mut self.booking.request_id),
            worker: self.booking.worker,
            attempt_id: self.booking.attempt_id,
        }
    }

    /// Free the booking directly and record a capacity notification.
    /// This does not wait for the actor to recheck pending admissions.
    pub async fn release(mut self) -> Result<(), SequenceError> {
        self.armed = false;
        self.cleanup
            .enqueue_acknowledged(self.booking.clone())
            .wait()
            .await
    }
}

impl Drop for BookingHandle {
    fn drop(&mut self) {
        if self.armed {
            self.cleanup.enqueue(self.booking.clone());
        }
    }
}

/// Single-owner cleanup lease for one scheduler-tracked request.
pub struct RequestLifecycleLease {
    cleanup: Arc<AdmissionCleanup>,
    actor_tx: mpsc::Sender<AdmissionCommand>,
    transfer: Option<Arc<AdmissionLifecycleTransfer>>,
}

impl std::fmt::Debug for RequestLifecycleLease {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RequestLifecycleLease")
            .field("transfer", &self.transfer)
            .finish_non_exhaustive()
    }
}

impl RequestLifecycleLease {
    pub(crate) fn disarm(&mut self) {
        if let Some(transfer) = self.transfer.take() {
            transfer.disarm();
        }
    }

    /// Hand the booking to its long-term owner: the lease stops guarding it.
    #[must_use]
    pub fn commit(mut self) -> Option<SchedulerBookingDescriptor> {
        let booking = match self.transfer.as_ref().map(|transfer| transfer.state.lock()) {
            Some(state) => match &*state {
                AdmissionLifecycleState::Booking(booking) => Some(booking.clone()),
                _ => None,
            },
            None => None,
        };
        self.disarm();
        booking
    }
}

impl Drop for RequestLifecycleLease {
    fn drop(&mut self) {
        let Some(transfer) = self.transfer.take() else {
            return;
        };
        if !transfer.is_armed() {
            return;
        }
        enqueue_cleanup_entry(
            &self.cleanup,
            &self.actor_tx,
            AdmissionCleanupEntry {
                target: SchedulerCleanupTarget::AdmissionLifecycle(transfer),
            },
        );
    }
}

/// The actor's end of the admission channel. Commands leave the channel a batch at a time
/// and are handed out in arrival order.
struct CommandInbox {
    rx: mpsc::Receiver<AdmissionCommand>,
    /// Taken from `rx` and not yet handed out, newest first.
    batch: Vec<AdmissionCommand>,
}

impl CommandInbox {
    fn new(rx: mpsc::Receiver<AdmissionCommand>) -> Self {
        Self {
            rx,
            batch: Vec::with_capacity(ADMISSION_BATCH),
        }
    }

    /// `None` once the channel is closed and drained. Cancel safe: a batch is only taken by
    /// a receive that completes.
    async fn next(&mut self) -> Option<AdmissionCommand> {
        if self.batch.is_empty() {
            self.rx.recv_many(&mut self.batch, ADMISSION_BATCH).await;
            self.batch.reverse();
        }
        self.batch.pop()
    }

    /// No command is waiting, taken or still in the channel.
    fn is_empty(&self) -> bool {
        self.batch.is_empty() && self.rx.is_empty()
    }
}

struct SchedulerQueueActor<
    P: SequencePublisher,
    C: WorkerConfigLike,
    Sel: WorkerSelector<C>,
    RF: OverlapScoresRefresh,
> {
    commands: CommandInbox,
    pending: PolicyQueue<QueuedRequest>,
    class_timers: ClassTimers,
    cleanup: Arc<AdmissionCleanup>,
    capacity_updates: Option<Arc<CapacityUpdates>>,
    capacity_workers: FxHashSet<WorkerWithDpRank>,
    profile: Arc<PolicyProfile>,
    pending_count: Arc<AtomicUsize>,
    pending_isl_tokens: Arc<AtomicUsize>,
    class_counters: Arc<Vec<ClassQueueCounters>>,
    slots: Arc<ActiveSequencesMultiWorker<P>>,
    workers_with_configs: watch::Receiver<HashMap<WorkerId, C>>,
    /// Advanced when the actor observes a configuration change; see `PreparedAdmission`.
    workers_generation: Arc<AtomicU64>,
    /// Current preparer, republished after a runtime policy update.
    candidate_preparer: Arc<ArcSwapOption<CandidatePreparer>>,
    /// One per command or wakeup. A request enqueued in an earlier turn rebuilds its rows.
    turn: u64,
    /// Rows prepared for the request enqueued this turn, dropped when the turn ends.
    turn_prepared: Option<Box<PreparedAdmission>>,
    projected_loads: FxHashMap<WorkerWithDpRank, WorkerLoadProjection>,
    /// Ranks the current request's load projection found past the host's reject thresholds,
    /// counted per worker. Refilled by `with_projected_loads`; empty while nothing is full.
    full_ranks: FullRankCounts,
    block_size: u32,
    selector: Sel,
    prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
    overlap_scores_refresh: Option<Arc<RF>>,
    overlap_refresh_after: Option<Duration>,
    overloaded_worker_provider: Option<OverloadedWorkerProvider>,
    available_worker_ids: Option<Arc<AvailableWorkerIds>>,
    non_max_overlap_selection_observer: Arc<OnceLock<NonMaxOverlapSelectionObserver>>,
    /// Per-candidate cost capture for admitted selections, installed once by the host.
    selection_trace: Arc<OnceLock<SelectionTraceSpec>>,
    /// The partition's live resident block counts, installed once by the host for a policy
    /// that declared RESIDENT_BLOCKS.
    resident_blocks: Arc<OnceLock<ResidentBlockCountsHandle>>,
    reject_thresholds: Option<RejectThresholdsProvider>,
}

/// Queue that gates scheduling requests behind a capacity check.
/// When all workers exceed `threshold_frac` utilisation the request is parked in `pending`.
/// When capacity frees up (`update()`), pending requests are scheduled in priority order.
/// If queueing is disabled (threshold_frac is None), requests are scheduled immediately.
pub struct SchedulerQueue<
    P: SequencePublisher,
    C: WorkerConfigLike,
    Sel: WorkerSelector<C> = super::selector::WorkerSelectionPolicy,
    RF: OverlapScoresRefresh = NoopOverlapScoresRefresh,
> {
    admission_tx: mpsc::Sender<AdmissionCommand>,
    runtime_policy: Arc<RuntimePolicyControl>,
    cleanup: Arc<AdmissionCleanup>,
    capacity_updates: Option<Arc<CapacityUpdates>>,
    direct_release: DirectBookingRelease,
    /// Number of requests currently parked in the pending queue.
    /// Incremented after push, decremented after pop. Lock-free reads via `Relaxed` load.
    pending_count: Arc<AtomicUsize>,
    /// Sum of `isl_tokens` for requests currently parked in the pending queue.
    /// Incremented after push, decremented after pop. Lock-free reads via `Relaxed` load.
    pending_isl_tokens: Arc<AtomicUsize>,
    class_counters: Arc<Vec<ClassQueueCounters>>,
    workers_with_configs: watch::Receiver<HashMap<WorkerId, C>>,
    workers_generation: Arc<AtomicU64>,
    candidate_preparer: Arc<ArcSwapOption<CandidatePreparer>>,
    exclusive_affinity: bool,
    block_size: u32,
    profile: Arc<PolicyProfile>,
    start_time: Instant,
    available_worker_ids: Option<Arc<AvailableWorkerIds>>,
    overloaded_worker_provider: Option<OverloadedWorkerProvider>,
    supports_overlap_refresh: bool,
    non_max_overlap_selection_observer: Arc<OnceLock<NonMaxOverlapSelectionObserver>>,
    /// Per-candidate cost capture for admitted selections, installed once by the host.
    selection_trace: Arc<OnceLock<SelectionTraceSpec>>,
    /// The partition's live resident block counts, installed once by the host for a policy
    /// that declared RESIDENT_BLOCKS.
    resident_blocks: Arc<OnceLock<ResidentBlockCountsHandle>>,
    reject_thresholds: Option<RejectThresholdsProvider>,
    #[allow(clippy::type_complexity)]
    // Covariant type markers, without ownership or auto-trait bounds.
    _marker: PhantomData<fn() -> (P, C, Sel, RF)>,
}

impl<
    P: SequencePublisher + 'static,
    C: WorkerConfigLike + Send + Sync + 'static,
    Sel: WorkerSelector<C> + Send + 'static,
    RF: OverlapScoresRefresh + Send + Sync + 'static,
> SchedulerQueue<P, C, Sel, RF>
{
    /// The policy profile this queue admits under, for tests.
    #[cfg(any(test, feature = "testing"))]
    pub fn profile(&self) -> &PolicyProfile {
        &self.profile
    }

    /// Hold every admission permit until the returned guard drops, for tests that prove
    /// capacity progress needs no space in the command channel.
    #[cfg(any(test, feature = "testing"))]
    pub async fn hold_admission_permits(&self) -> impl Drop + '_ {
        self.admission_tx
            .reserve_many(self.admission_tx.max_capacity())
            .await
            .expect("admission channel open")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        slots: Arc<ActiveSequencesMultiWorker<P>>,
        workers_with_configs: watch::Receiver<HashMap<WorkerId, C>>,
        profile: PolicyProfile,
        block_size: u32,
        selector: Sel,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        overlap_scores_refresh: Option<Arc<RF>>,
        overloaded_worker_provider: Option<OverloadedWorkerProvider>,
        available_worker_provider: Option<WorkerAvailabilityProvider>,
        reject_thresholds: Option<RejectThresholdsProvider>,
    ) -> Self {
        Self::new_with_capacity(
            slots,
            workers_with_configs,
            profile,
            block_size,
            selector,
            prefill_load_estimator,
            overlap_scores_refresh,
            overloaded_worker_provider,
            available_worker_provider,
            reject_thresholds,
            ADMISSION_CHANNEL_CAPACITY,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_capacity(
        slots: Arc<ActiveSequencesMultiWorker<P>>,
        workers_with_configs: watch::Receiver<HashMap<WorkerId, C>>,
        profile: PolicyProfile,
        block_size: u32,
        selector: Sel,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        overlap_scores_refresh: Option<Arc<RF>>,
        overloaded_worker_provider: Option<OverloadedWorkerProvider>,
        available_worker_provider: Option<WorkerAvailabilityProvider>,
        reject_thresholds: Option<RejectThresholdsProvider>,
        admission_channel_capacity: usize,
    ) -> Self {
        let profile = Arc::new(profile);
        let pending = PolicyQueue::with_class_budget_expiry(&profile);
        let queueing_enabled = profile
            .classes()
            .iter()
            .any(PolicyClassConfig::queueing_enabled);
        for class in profile.classes() {
            tracing::info!(
                policy_class = class.name,
                queue_policy = %class.queue_policy,
                quantum = class.quantum,
                prefill_busy_threshold = ?class.busy_thresholds.prefill_busy_threshold,
                prefill_busy_threshold_frac = ?class.busy_thresholds.prefill_busy_threshold_frac,
                "Router policy class configured"
            );
        }
        let overlap_refresh_after = if overlap_scores_refresh.is_some() {
            let configured = read_overlap_refresh_after();
            match configured {
                Some(d) => tracing::info!(
                    "Router queue overlap-score refresh enabled after {:.1}s wait",
                    d.as_secs_f64()
                ),
                None => tracing::info!(
                    "Router queue overlap-score refresh disabled via DYN_ROUTER_OVERLAP_REFRESH_AFTER_SECS"
                ),
            }
            configured
        } else {
            None
        };
        let pending_count = Arc::new(AtomicUsize::new(0));
        let pending_isl_tokens = Arc::new(AtomicUsize::new(0));
        let class_counters = Arc::new(
            profile
                .classes()
                .iter()
                .map(|_| ClassQueueCounters {
                    pending_count: AtomicUsize::new(0),
                    pending_isl_tokens: AtomicUsize::new(0),
                    pending_cached_tokens: AtomicUsize::new(0),
                    received_total: AtomicU64::new(0),
                    rejected_due_time_passed_total: AtomicU64::new(0),
                    rejected_queue_size_exceeded_total: AtomicU64::new(0),
                })
                .collect(),
        );
        let (admission_tx, admission_rx) = mpsc::channel(admission_channel_capacity);
        let runtime_policy = Arc::new(RuntimePolicyControl {
            tx: admission_tx.downgrade(),
        });
        let cleanup = Arc::new(AdmissionCleanup::default());
        let non_max_overlap_selection_observer = Arc::new(OnceLock::new());
        let selection_trace = Arc::new(OnceLock::new());
        let resident_blocks = Arc::new(OnceLock::new());
        let start_time = Instant::now();
        let capacity_updates = queueing_enabled.then(|| Arc::new(CapacityUpdates::new()));
        let direct_release: DirectBookingRelease = {
            let slots = Arc::clone(&slots);
            let capacity_updates = capacity_updates.clone();
            Arc::new(move |booking| {
                let outcome = slots.free_if_booking(
                    &booking.request_id,
                    booking.worker,
                    booking.attempt_id,
                    Instant::now(),
                )?;
                if outcome.is_applied()
                    && let Some(updates) = &capacity_updates
                {
                    updates.record(Some(booking.worker));
                }
                Ok(())
            })
        };
        let workers_generation = Arc::new(AtomicU64::new(0));
        let available_worker_ids =
            available_worker_provider.map(|provider| Arc::new(AvailableWorkerIds::new(provider)));
        let candidate_preparer = Arc::new(ArcSwapOption::from(
            selector.candidate_preparer().map(Arc::new),
        ));
        let exclusive_affinity = selector.uses_exclusive_affinity_target();
        let actor = SchedulerQueueActor {
            commands: CommandInbox::new(admission_rx),
            pending,
            class_timers: ClassTimers::new(profile.classes().len()),
            cleanup: Arc::clone(&cleanup),
            capacity_updates: capacity_updates.clone(),
            capacity_workers: FxHashSet::default(),
            profile: Arc::clone(&profile),
            pending_count: Arc::clone(&pending_count),
            pending_isl_tokens: Arc::clone(&pending_isl_tokens),
            class_counters: Arc::clone(&class_counters),
            slots,
            workers_with_configs: workers_with_configs.clone(),
            workers_generation: Arc::clone(&workers_generation),
            candidate_preparer: Arc::clone(&candidate_preparer),
            turn: 0,
            turn_prepared: None,
            projected_loads: FxHashMap::default(),
            full_ranks: FullRankCounts::default(),
            block_size,
            selector,
            prefill_load_estimator,
            overlap_scores_refresh,
            overlap_refresh_after,
            overloaded_worker_provider: overloaded_worker_provider.clone(),
            available_worker_ids: available_worker_ids.clone(),
            non_max_overlap_selection_observer: Arc::clone(&non_max_overlap_selection_observer),
            selection_trace: Arc::clone(&selection_trace),
            resident_blocks: Arc::clone(&resident_blocks),
            reject_thresholds: reject_thresholds.clone(),
        };
        tokio::spawn(actor.run());
        Self {
            admission_tx,
            runtime_policy,
            cleanup,
            capacity_updates,
            direct_release,
            pending_count,
            pending_isl_tokens,
            class_counters,
            workers_with_configs,
            workers_generation,
            candidate_preparer,
            exclusive_affinity,
            block_size,
            profile,
            start_time,
            available_worker_ids,
            overloaded_worker_provider,
            supports_overlap_refresh: overlap_refresh_after.is_some(),
            non_max_overlap_selection_observer,
            selection_trace,
            resident_blocks,
            reject_thresholds,
            _marker: PhantomData,
        }
    }
}

impl<
    P: SequencePublisher + 'static,
    C: WorkerConfigLike + Send + Sync + 'static,
    Sel: WorkerSelector<C> + Send + 'static,
    RF: OverlapScoresRefresh + Send + Sync + 'static,
> SchedulerQueue<P, C, Sel, RF>
{
    /// Install the observer for admitted selections that sacrifice KV overlap.
    ///
    /// Returns `false` when an observer is already installed.
    pub fn set_non_max_overlap_selection_observer(
        &self,
        observer: NonMaxOverlapSelectionObserver,
    ) -> bool {
        self.non_max_overlap_selection_observer
            .set(observer)
            .is_ok()
    }

    /// Install per-candidate cost capture for admitted selections.
    ///
    /// Returns `false` when a spec is already installed.
    pub fn set_selection_trace(&self, spec: SelectionTraceSpec) -> bool {
        self.selection_trace.set(spec).is_ok()
    }

    /// Lend the partition's live resident block counts to every selection, for a policy that
    /// declared `WorkerInputs::RESIDENT_BLOCKS`.
    ///
    /// Returns `false` when counts are already installed.
    pub fn set_resident_block_counts(&self, counts: ResidentBlockCountsHandle) -> bool {
        self.resident_blocks.set(counts).is_ok()
    }

    /// Enqueue a new request through the router-owned policy queue.
    ///
    /// Every request enters the policy queue and the actor drains the ready
    /// backlog in the same turn, so an uncontended request is admitted before
    /// the enqueue acknowledgement returns. Two direct-admission carve-outs
    /// bypass the queue: classes with any zero per-worker limit (which can
    /// never hold a queued entry) and windows with zero discovered workers
    /// (so selection reports `NoEndpoints` instead of a scaled-to-zero
    /// queue-limit rejection).
    pub async fn enqueue(&self, request: SchedulingRequest) {
        self.enqueue_with_block_hashes(request, None).await;
    }

    pub async fn enqueue_with_block_hashes(
        &self,
        request: SchedulingRequest,
        block_hashes: Option<Vec<LocalBlockHash>>,
    ) {
        let _ = self
            .enqueue_with_block_hashes_and_lease(request, block_hashes, None)
            .await;
    }

    pub(crate) async fn enqueue_with_block_hashes_and_lease(
        &self,
        request: SchedulingRequest,
        block_hashes: Option<Vec<LocalBlockHash>>,
        lease: Option<Box<RequestLifecycleLease>>,
    ) -> Option<Box<RequestLifecycleLease>> {
        self.enqueue_admitted_with_block_hashes_and_lease(
            request,
            block_hashes,
            lease,
            None,
            None,
            Instant::now(),
        )
        .await
    }

    pub async fn enqueue_admitted_with_block_hashes_and_lease(
        &self,
        mut request: SchedulingRequest,
        block_hashes: Option<Vec<LocalBlockHash>>,
        lease: Option<Box<RequestLifecycleLease>>,
        attempt_tx: Option<oneshot::Sender<AttemptId>>,
        classified_request: Option<ClassifyRequest>,
        ingress_at: Instant,
    ) -> Option<Box<RequestLifecycleLease>> {
        if self.capacity_updates.is_some()
            && lease.is_none()
            && request.mode.lifecycle_request_id().is_some()
        {
            request.respond(Err(KvSchedulerError::BookingFailed(
                "admission-managed requests must be scheduled through LocalScheduler".to_string(),
            )));
            return None;
        }

        let queue_metadata = match classified_request {
            Some(classified) => self.validate_classification(&mut request, classified, ingress_at),
            None => self.default_queue_metadata(&request, ingress_at),
        };
        let queue_metadata = match queue_metadata {
            Ok(metadata) => metadata,
            Err(error) => {
                request.respond(Err(error));
                return None;
            }
        };

        let eligibility = request.eligibility();

        if let Err(error) = eligibility.validate_pinned_worker_allowed() {
            request.respond(Err(error));
            return None;
        }

        let Ok(slot) = self.admission_tx.reserve().await else {
            request.respond(Err(KvSchedulerError::SubscriberShutdown));
            return None;
        };
        let prepared = self.prepare_candidates(&request);
        let (ack_tx, ack_rx) = oneshot::channel();
        slot.send(AdmissionCommand::Enqueue {
            request,
            attempt_tx,
            block_hashes: self.prepare_block_hashes_for_refresh(block_hashes),
            prepared,
            queue_metadata,
            lease,
            ack_tx,
        });

        match ack_rx.await {
            Ok(lease) => lease,
            Err(_) => {
                tracing::warn!("scheduler queue actor dropped enqueue acknowledgement");
                None
            }
        }
    }

    /// Build the classifier's input and the rows for its declared worker inputs. Uses
    /// `request.isl_tokens` — the routing-token basis the queue's bucketing, limits, and DRR
    /// cost already use — so a pass-through classifier is behavior-identical to no
    /// classifier.
    pub fn build_classify_request(
        &self,
        request: &SchedulingRequest,
        ingress_at: Instant,
        worker_inputs: WorkerInputs,
    ) -> (ClassifyRequest, WorkerInputRows) {
        let available = self
            .available_worker_ids
            .as_ref()
            .and_then(|ids| ids.get(request));
        let workers = self.workers_with_configs.borrow();
        let cached_tokens = SchedulingContext::new(request, &workers)
            .with_available_workers(available.as_deref())
            .best_cached_tokens();
        let reject_thresholds = self
            .reject_thresholds
            .as_ref()
            .map(|provider| provider())
            .unwrap_or_default();
        let mut classification =
            ClassifyRequest::with_timing(request.isl_tokens, cached_tokens, ingress_at)
                .with_pinned_worker(request.pinned_worker)
                .with_sequence_hashes(request.token_seq.as_deref())
                .with_busy_thresholds(Arc::clone(&self.profile), reject_thresholds);
        if let Some(request_id) = request.mode.request_id() {
            classification = classification.with_request_id(request_id);
        }
        if let Some(policy_class) = request.policy_class.as_deref() {
            classification = classification.with_initial_policy_class(policy_class);
        }
        if let Some(session_context) = request.session_context.clone() {
            classification = classification.with_session_context(session_context);
        }
        let rows = if worker_inputs == WorkerInputs::NONE {
            WorkerInputRows::empty()
        } else {
            let eligibility = request
                .eligibility()
                .with_available_workers(available.as_deref());
            worker_input_rows(&workers, request, eligibility, worker_inputs)
        };
        (classification, rows)
    }

    /// Enqueue a request with `due_at` for actor-expiry tests.
    #[cfg(test)]
    pub(crate) async fn enqueue_with_due_at_for_test(
        &self,
        request: SchedulingRequest,
        due_at: Instant,
    ) {
        let mut queue_metadata = self
            .default_queue_metadata(&request, Instant::now())
            .expect("test request names a configured policy class");
        queue_metadata.due_at = Some(due_at);
        let (ack_tx, ack_rx) = oneshot::channel();
        let command = AdmissionCommand::Enqueue {
            request,
            attempt_tx: None,
            block_hashes: None,
            prepared: None,
            queue_metadata,
            lease: None,
            ack_tx,
        };
        self.admission_tx
            .send(command)
            .await
            .expect("scheduler queue actor gone");
        let _ = ack_rx.await;
    }

    fn default_queue_metadata(
        &self,
        request: &SchedulingRequest,
        ingress_at: Instant,
    ) -> Result<QueueMetadata, KvSchedulerError> {
        let class_index = self
            .profile
            .resolve_class_index(request.policy_class.as_deref())?;
        Ok(self.queue_metadata(request, class_index, ingress_at))
    }

    fn queue_metadata(
        &self,
        request: &SchedulingRequest,
        class_index: usize,
        ingress_at: Instant,
    ) -> QueueMetadata {
        let available = self
            .available_worker_ids
            .as_ref()
            .and_then(|ids| ids.get(request));
        let workers = self.workers_with_configs.borrow();
        let snapshot = QueueSnapshot::new(
            request.isl_tokens,
            SchedulingContext::new(request, &workers)
                .with_available_workers(available.as_deref())
                .best_cached_tokens(),
        );
        QueueMetadata {
            class_index,
            snapshot,
            due_at: None,
            // The final class's budget counts from the original router ingress, so classifier
            // and channel waits spend it; an explicit deadline set later takes precedence.
            class_due_at: self.profile.class(class_index).slo_ttft_due_at(ingress_at),
            // "Arrival" is the router ingress time, stamped before the
            // admission channel — one basis for classified and unclassified
            // requests alike, so time spent in a classifier does not reorder
            // FCFS/LCFS. Two racing callers can still enqueue in the opposite
            // order of their offsets; `enqueue_seq` stays the authoritative
            // FCFS tiebreak for same-score entries.
            arrival_offset_secs: ingress_at
                .saturating_duration_since(self.start_time)
                .as_secs_f64(),
        }
    }

    pub fn validate_classification(
        &self,
        request: &mut SchedulingRequest,
        classified_request: ClassifyRequest,
        ingress_at: Instant,
    ) -> Result<QueueMetadata, KvSchedulerError> {
        let ClassificationOverrides {
            policy_class,
            due_at,
            scheduling_cost_tokens,
            worker_selection_target,
        } = classified_request.into_queue_inputs();

        if scheduling_cost_tokens == Some(0) {
            return Err(KvSchedulerError::InvalidClassificationMetadata(
                "scheduling cost must be greater than zero".to_string(),
            ));
        }
        // An already-expired `due_at` is not validated here: the actor is the
        // single deadline authority and rejects it at enqueue on its own clock.

        // Hard pins must not acquire a conflicting soft target. Apply a permitted
        // override before recomputing cache eligibility for the resulting target.
        if request.pinned_worker.is_none()
            && let Some(target) = worker_selection_target
        {
            request.affinity_target = target;
        }

        // Queue inputs are recomputed from the current workers, exactly as the
        // default path computes them: worker state may have changed while the
        // classification was pending, and only explicit overrides survive. The
        // class entering the queue is the classifier's override, if it set one,
        // otherwise the request's own class.
        let class_index = match policy_class {
            Some(policy_class) => self
                .profile
                .resolve_class_index_strict(&policy_class)
                .ok_or_else(|| {
                    KvSchedulerError::InvalidClassificationMetadata(format!(
                        "unknown policy class {policy_class:?}"
                    ))
                })?,
            None => self
                .profile
                .resolve_class_index(request.policy_class.as_deref())?,
        };
        let mut metadata = self.queue_metadata(request, class_index, ingress_at);
        if let Some(cost) = scheduling_cost_tokens {
            metadata.snapshot.scheduling_cost_tokens = cost;
        }
        metadata.due_at = due_at;
        Ok(metadata)
    }

    pub(crate) fn new_request_lifecycle_lease(
        &self,
        request_id: Option<&str>,
    ) -> Option<Box<RequestLifecycleLease>> {
        let request_id = request_id?;
        Some(Box::new(RequestLifecycleLease {
            cleanup: Arc::clone(&self.cleanup),
            actor_tx: self.admission_tx.clone(),
            transfer: Some(Arc::new(AdmissionLifecycleTransfer::new(
                request_id.to_string(),
            ))),
        }))
    }

    pub fn booking_cleanup(&self) -> SchedulerBookingCleanup {
        SchedulerBookingCleanup {
            cleanup: Arc::clone(&self.cleanup),
            actor_tx: self.admission_tx.clone(),
            direct_release: Arc::clone(&self.direct_release),
        }
    }

    /// An armed handle for an existing booking.
    pub(crate) fn booking_handle(&self, booking: SchedulerBookingDescriptor) -> BookingHandle {
        BookingHandle {
            booking,
            cleanup: self.booking_cleanup(),
            armed: true,
        }
    }

    /// Select a worker from current scheduler state without entering admission.
    ///
    /// This is for advisory policy probes that must not wait in the router
    /// queue and must not book active scheduler state.
    pub async fn select_without_admission(
        &self,
        request: SchedulingRequest,
    ) -> Result<AdvisorySchedulingResponse, KvSchedulerError> {
        request.eligibility().validate_pinned_worker_allowed()?;

        let (resp_tx, resp_rx) = oneshot::channel();
        let command = AdmissionCommand::SelectWithoutAdmission { request, resp_tx };
        if self.admission_tx.send(command).await.is_err() {
            return Err(KvSchedulerError::SubscriberShutdown);
        }

        resp_rx
            .await
            .map_err(|_| KvSchedulerError::SubscriberShutdown)?
    }

    /// Wait for a full admission recheck when queueing is enabled and the actor is running.
    /// With queueing disabled, return immediately without an admission barrier.
    pub async fn update(&self) {
        if self.capacity_updates.is_none() {
            return;
        }

        let _ = self.admission_barrier().await;
    }

    /// Process previously submitted admission commands and recheck ready work.
    ///
    /// Unlike [`Self::update`], this waits for the actor even when queueing is
    /// disabled. It does not wait for capacity-blocked requests to be admitted,
    /// for ready work behind an overlap refresh that is still running, or
    /// submit scheduling futures that the caller has not polled yet.
    /// Returns an error if the actor stops before acknowledging the barrier.
    pub async fn admission_barrier(&self) -> Result<(), KvSchedulerError> {
        let (ack_tx, ack_rx) = oneshot::channel();
        self.admission_tx
            .send(AdmissionCommand::Update { ack_tx })
            .await
            .map_err(|_| KvSchedulerError::SubscriberShutdown)?;
        ack_rx
            .await
            .map_err(|_| KvSchedulerError::SubscriberShutdown)
    }

    pub fn policy_control(&self) -> &Arc<RuntimePolicyControl> {
        &self.runtime_policy
    }

    /// Notify pending admission after the authoritative mutation is complete.
    pub fn capacity_changed(&self, worker: Option<WorkerWithDpRank>) {
        if let Some(updates) = &self.capacity_updates {
            updates.record(worker);
        }
    }

    pub(crate) fn ensure_running(&self) -> Result<(), KvSchedulerError> {
        if self.admission_tx.is_closed() {
            Err(KvSchedulerError::SubscriberShutdown)
        } else {
            Ok(())
        }
    }

    /// Number of requests currently parked in the pending queue (lock-free).
    pub fn pending_count(&self) -> usize {
        self.pending_count.load(AtomicOrdering::Relaxed)
    }

    /// Sum of `isl_tokens` for requests currently parked in the pending queue (lock-free).
    pub fn pending_isl_tokens(&self) -> usize {
        self.pending_isl_tokens.load(AtomicOrdering::Relaxed)
    }

    pub fn class_queue_stats(&self, class_index: usize) -> Option<ClassQueueStats> {
        let counters = self.class_counters.get(class_index)?;
        Some(ClassQueueStats {
            pending_count: counters.pending_count.load(AtomicOrdering::Relaxed),
            pending_isl_tokens: counters.pending_isl_tokens.load(AtomicOrdering::Relaxed),
            pending_cached_tokens: counters.pending_cached_tokens.load(AtomicOrdering::Relaxed),
            received_total: counters.received_total.load(AtomicOrdering::Relaxed),
            rejected_due_time_passed_total: counters
                .rejected_due_time_passed_total
                .load(AtomicOrdering::Relaxed),
            rejected_queue_size_exceeded_total: counters
                .rejected_queue_size_exceeded_total
                .load(AtomicOrdering::Relaxed),
        })
    }

    /// Build candidate rows on the caller task, leaving the actor only the
    /// load inputs that depend on its bookings.
    fn prepare_candidates(&self, request: &SchedulingRequest) -> Option<Box<PreparedAdmission>> {
        if request.pinned_worker.is_some() {
            return None;
        }
        let preparer = self.candidate_preparer.load();
        let preparer = preparer.as_deref()?;
        // Load before reading the configuration: a change after this load
        // advances the generation before the actor selects with these rows.
        let workers_generation = self.workers_generation.load(AtomicOrdering::Acquire);
        let available_worker_ids = self
            .available_worker_ids
            .as_ref()
            .and_then(|ids| ids.get(request));
        let overloaded_worker_ids = self
            .overloaded_worker_provider
            .as_ref()
            .and_then(|provider| provider());
        let workers = self.workers_with_configs.borrow();
        let eligibility = selection_eligibility(
            request,
            &workers,
            overloaded_worker_ids.as_deref(),
            available_worker_ids.as_deref(),
            None,
            self.exclusive_affinity,
        );
        let candidates = preparer.prepare(WorkerSelectionInput::configured(
            &workers,
            request,
            eligibility,
            self.block_size,
        ))?;
        Some(Box::new(PreparedAdmission {
            candidates,
            workers_generation,
            available_worker_ids,
            overloaded_worker_ids,
        }))
    }

    fn prepare_block_hashes_for_refresh(
        &self,
        block_hashes: Option<Vec<LocalBlockHash>>,
    ) -> Option<Vec<LocalBlockHash>> {
        if !self.supports_overlap_refresh {
            return None;
        }
        block_hashes.filter(|hashes| !hashes.is_empty())
    }
}

impl<
    P: SequencePublisher + 'static,
    C: WorkerConfigLike + Send + Sync + 'static,
    Sel: WorkerSelector<C> + Send + 'static,
    RF: OverlapScoresRefresh + Send + Sync + 'static,
> SchedulerQueueActor<P, C, Sel, RF>
{
    /// Runs one turn per command, capacity wake, or class timer until every sender is gone.
    async fn run(mut self) {
        let mut commands_since_cleanup = 0usize;
        let capacity_updates = self.capacity_updates.clone();
        loop {
            self.begin_turn();
            let due_at = self.class_timers.next_deadline();
            let command = if capacity_updates.is_none() && due_at.is_none() {
                self.commands.next().await
            } else {
                tokio::select! {
                    command = self.commands.next() => command,
                    _ = async {
                        if let Some(updates) = &capacity_updates {
                            updates.notified().await;
                        }
                    }, if capacity_updates.is_some() => {
                        let drain_cleanup =
                            self.commands.is_empty() || commands_since_cleanup == 256;
                        if drain_cleanup {
                            commands_since_cleanup = 0;
                        }
                        self.handle_capacity_updates(drain_cleanup).await;
                        continue;
                    }
                    _ = async {
                        if let Some(due_at) = due_at {
                            tokio::time::sleep_until(due_at).await;
                        }
                    }, if due_at.is_some() => {
                        self.handle_update(None).await;
                        continue;
                    }
                }
            };
            let Some(command) = command else { break };
            commands_since_cleanup += 1;
            let drain_cleanup = self.commands.is_empty() || commands_since_cleanup == 256;
            if drain_cleanup {
                commands_since_cleanup = 0;
            }
            let (drain, reply) = self.apply_command(command, drain_cleanup);
            self.drain_after_command(drain, reply).await;
        }
        self.drain_cleanup();

        let class_counters = Arc::clone(&self.class_counters);
        for entry in self.pending.drain() {
            let class_index = entry.class_index();
            let snapshot = entry.snapshot();
            self.pending_count.fetch_sub(1, AtomicOrdering::Relaxed);
            self.pending_isl_tokens
                .fetch_sub(snapshot.raw_isl_tokens, AtomicOrdering::Relaxed);
            let counters = &class_counters[class_index];
            counters.pending_count.fetch_sub(1, AtomicOrdering::Relaxed);
            counters
                .pending_isl_tokens
                .fetch_sub(snapshot.raw_isl_tokens, AtomicOrdering::Relaxed);
            counters
                .pending_cached_tokens
                .fetch_sub(snapshot.cached_tokens, AtomicOrdering::Relaxed);

            let mut request = entry.into_payload().request;
            request.respond(Err(KvSchedulerError::SubscriberShutdown));
        }
    }

    /// One turn per command or wakeup, including one handled during an overlap refresh. Rows
    /// prepared in the previous turn are dropped; a request still queued rebuilds them when
    /// admitted.
    fn begin_turn(&mut self) {
        self.turn = self.turn.wrapping_add(1);
        self.turn_prepared = None;
    }

    async fn drain_after_command(&mut self, drain: Drain, reply: Option<CommandReply>) {
        match drain {
            Drain::None => {
                if let Some(reply) = reply {
                    reply.send();
                }
            }
            Drain::Enqueued => self.handle_enqueued(reply).await,
            Drain::Full => self.handle_update(reply).await,
        }
    }

    /// Applies one command without draining: returns the drain it calls for and its reply.
    fn apply_command(
        &mut self,
        command: AdmissionCommand,
        drain_cleanup: bool,
    ) -> (Drain, Option<CommandReply>) {
        match command {
            AdmissionCommand::Enqueue {
                request,
                attempt_tx,
                block_hashes,
                prepared,
                queue_metadata,
                lease,
                ack_tx,
            } => {
                let lifecycle_transfer = lease
                    .as_ref()
                    .and_then(|lease| lease.transfer.as_ref().map(Arc::clone));
                // Released entries leave before the arrival's class limits count them.
                let cleanup_ready = drain_cleanup && self.drain_cleanup();
                let enqueue_ready = self.handle_enqueue(
                    request,
                    attempt_tx,
                    lifecycle_transfer,
                    block_hashes,
                    prepared,
                    queue_metadata,
                );
                let drain = if cleanup_ready {
                    Drain::Full
                } else if enqueue_ready {
                    Drain::Enqueued
                } else {
                    Drain::None
                };
                (drain, Some(CommandReply::Enqueue { ack_tx, lease }))
            }
            AdmissionCommand::SelectWithoutAdmission { request, resp_tx } => {
                let result = self.select_without_admission_inner(request, Instant::now());
                (
                    Drain::full_if(drain_cleanup && self.drain_cleanup()),
                    Some(CommandReply::SelectWithoutAdmission { resp_tx, result }),
                )
            }
            AdmissionCommand::PolicyConfig { update, response } => {
                // A cancelled or timed-out caller must not leave a queued mutation behind.
                if !response.is_closed() {
                    let result = match update {
                        Some(update) => {
                            let result = self.selector.update_runtime_config(update);
                            if result.is_ok() {
                                // A replacement may need fewer inputs; prepare what it reads.
                                self.candidate_preparer
                                    .store(self.selector.candidate_preparer().map(Arc::new));
                            }
                            result
                        }
                        None => self.selector.runtime_config(),
                    };
                    let _ = response.send(result);
                }
                (Drain::full_if(drain_cleanup && self.drain_cleanup()), None)
            }
            AdmissionCommand::Update { ack_tx } => {
                if drain_cleanup {
                    self.drain_cleanup();
                }
                (Drain::Full, Some(CommandReply::Update { ack_tx }))
            }
            AdmissionCommand::Cleanup => (Drain::full_if(self.drain_cleanup()), None),
        }
    }

    /// Returns whether ready work may have changed: the request entered the queue or was booked,
    /// or the expiry it ran exposed a class head.
    fn handle_enqueue(
        &mut self,
        mut request: SchedulingRequest,
        attempt_tx: Option<oneshot::Sender<AttemptId>>,
        lifecycle_transfer: Option<Arc<AdmissionLifecycleTransfer>>,
        block_hashes: Option<Vec<LocalBlockHash>>,
        prepared: Option<Box<PreparedAdmission>>,
        queue_metadata: QueueMetadata,
    ) -> bool {
        let class_index = queue_metadata.class_index;
        self.class_counters[class_index]
            .received_total
            .fetch_add(1, AtomicOrdering::Relaxed);
        let decay_now = Instant::now();
        let due_at = self.pending.enforced_due_at(&queue_metadata);
        if due_at.is_some_and(|due_at| due_at <= decay_now) {
            self.reject_due_time_passed(class_index, &mut request);
            return false;
        }
        let context = AdmissionContext {
            class_index,
            due_at,
            attempt_tx,
            lifecycle_transfer,
        };
        let snapshot = queue_metadata.snapshot;
        let class = self.profile.class(class_index);
        let mut limits_now = decay_now;
        // With zero discovered workers every per-worker limit scales to zero and
        // `queue_rejection` would reject with `limit: 0`. Admit directly instead
        // so selection reports the real condition, `NoEndpoints`.
        if self.workers_with_configs.borrow().is_empty() {
            return self.admit_one(context, request, decay_now, prepared) == Admission::Booked;
        }
        // A class with any zero per-worker limit never queues: keep the direct
        // admission semantics by admitting while a worker has prefill capacity,
        // and otherwise fall through so `queue_rejection` produces the same
        // limit-zero rejection the queued path always has.
        if class.never_queues() {
            let available = self
                .available_worker_ids
                .as_ref()
                .and_then(|ids| ids.get(&request));
            let should_queue = class.queueing_enabled()
                && (self.pending.has_backlog(class_index) || {
                    let configs = self.workers_with_configs.borrow();
                    Self::all_eligible_workers_full(
                        &self.slots.worker_load_reader(decay_now),
                        &configs,
                        class.busy_thresholds,
                        request
                            .eligibility()
                            .with_available_workers(available.as_deref()),
                    )
                });
            if !should_queue {
                return self.admit_one(context, request, decay_now, prepared) == Admission::Booked;
            }
            // The availability check and load projection can take time. Judge this request's
            // due time, and the queued entries' below, at a fresh instant before the limits.
            limits_now = Instant::now();
            if due_at.is_some_and(|due_at| due_at <= limits_now) {
                self.reject_due_time_passed(class_index, &mut request);
                return false;
            }
        }
        tracing::trace!(
            policy_class = self.profile.class(class_index).name,
            "ordering request"
        );
        let AdmissionContext {
            attempt_tx,
            lifecycle_transfer,
            ..
        } = context;
        let queued = QueuedRequest {
            request,
            attempt_tx,
            lifecycle_transfer: lifecycle_transfer.clone(),
            enqueue_at: decay_now,
            due_at,
            block_hashes,
            enqueue_turn: self.turn,
        };
        self.turn_prepared = prepared;
        if let Insertion::Refused { exposed } =
            self.insert_arrival(queue_metadata, queued, limits_now)
        {
            return exposed;
        }
        if let Some(due_at) = due_at {
            self.class_timers.arm_if_earlier(class_index, due_at);
        }
        if let Some(transfer) = lifecycle_transfer {
            transfer.arm_pending();
        }
        self.pending_count.fetch_add(1, AtomicOrdering::Relaxed);
        self.pending_isl_tokens
            .fetch_add(snapshot.raw_isl_tokens, AtomicOrdering::Relaxed);
        self.add_class_counters(class_index, snapshot);
        true
    }

    /// Inserts an arrival once the entries due by `now` have left, so the class limits do not
    /// count them. The arrival's own due time is judged at a fresh instant right before each
    /// insertion attempt and, ahead of its class's, after a limit refusal. A refusal stands only
    /// once no entry of the class was due; each retry removes an existing entry and nothing joins
    /// the class meanwhile, so this ends.
    ///
    /// A refused arrival is answered and counted here, or dropped uncounted when its receiver is
    /// already closed. The caller counts a queued arrival as pending and arms its class timer.
    fn insert_arrival(
        &mut self,
        queue_metadata: QueueMetadata,
        mut queued: QueuedRequest,
        now: Instant,
    ) -> Insertion {
        let class_index = queue_metadata.class_index;
        let due_at = queued.due_at;
        // A head that expiry exposes may be able to run, so every refusal below still reports it.
        let mut exposed = self.expire_due_classes(now);
        let priority_jump = queued.request.priority_jump;
        let strict_priority = queued.request.strict_priority;
        let placement = queued
            .request
            .pinned_worker
            .map_or(WorkerPlacement::Any, WorkerPlacement::Exact);
        let worker_count = self.workers_with_configs.borrow().len();
        loop {
            if due_at.is_some_and(|due_at| due_at <= Instant::now()) {
                self.reject_due_time_passed(class_index, &mut queued.request);
                return Insertion::Refused { exposed };
            }
            let (rejection, rejected) = match self.pending.enqueue_with_due_at(
                queue_metadata,
                worker_count,
                priority_jump,
                strict_priority,
                placement,
                queued,
            ) {
                Ok(()) => return Insertion::Queued,
                Err(rejected) => rejected,
            };
            let mut request = rejected.request;
            let now = Instant::now();
            if due_at.is_some_and(|due_at| due_at <= now) {
                self.reject_due_time_passed(class_index, &mut request);
                return Insertion::Refused { exposed };
            }
            // As with a due time, a receiver already closed at this check is not counted.
            if request.response_is_closed() {
                return Insertion::Refused { exposed };
            }
            if self
                .pending
                .next_due_at(class_index)
                .is_some_and(|due_at| due_at <= now)
            {
                exposed |= self.expire_class(class_index, now);
                queued = QueuedRequest {
                    request,
                    ..rejected
                };
                continue;
            }
            self.class_counters[class_index]
                .rejected_queue_size_exceeded_total
                .fetch_add(1, AtomicOrdering::Relaxed);
            request.respond(Err(KvSchedulerError::QueueRejected(rejection)));
            return Insertion::Refused { exposed };
        }
    }

    /// Rejects a request whose due time passed and counts it against its class. A request whose
    /// response receiver is already closed at this check is dropped uncounted, as booking drops
    /// it; a receiver closed after the check can still be counted.
    fn reject_due_time_passed(&self, class_index: usize, request: &mut SchedulingRequest) {
        if request.response_is_closed() {
            return;
        }
        self.class_counters[class_index]
            .rejected_due_time_passed_total
            .fetch_add(1, AtomicOrdering::Relaxed);
        request.respond(Err(KvSchedulerError::DeadlineExceeded));
    }

    /// Rejects the queued requests of every class whose timer is due at `now`. Each such class
    /// loses its overdue entries, without a DRR charge, and rearms at its next due time. Returns
    /// whether a class kept entries after losing one, as [`Self::expire_class`] does.
    fn expire_due_classes(&mut self, now: Instant) -> bool {
        let mut exposed = false;
        while let Some(class_index) = self.class_timers.pop_due(now) {
            exposed |= self.expire_class(class_index, now);
        }
        exposed
    }

    /// Rejects the class's entries that expire by `now`, without a DRR charge, and moves its
    /// timer to its next due time. Returns whether it removed an entry and the class kept others:
    /// a lane root may then have changed to one that can run.
    fn expire_class(&mut self, class_index: usize, now: Instant) -> bool {
        let expired = self.pending.take_expired(class_index, now);
        let removed = !expired.is_empty();
        for entry in expired {
            self.subtract_pending_counters(class_index, entry.snapshot());
            let mut request = entry.into_payload().request;
            self.reject_due_time_passed(class_index, &mut request);
        }
        let due_at = self.pending.next_due_at(class_index);
        self.class_timers.arm(class_index, due_at);
        removed && self.pending.has_backlog(class_index)
    }

    /// Follows a removal from the class: an unarmed timer means the class had no entry that
    /// expires, and removing one cannot change that.
    fn rearm_class_timer(&mut self, class_index: usize) {
        if self.class_timers.armed[class_index].is_some() {
            let due_at = self.pending.next_due_at(class_index);
            self.class_timers.arm(class_index, due_at);
        }
    }

    fn drain_cleanup(&mut self) -> bool {
        let dirty = self.cleanup.drain();
        if dirty.is_empty() {
            return false;
        }

        let mut made_ready = false;
        let mut removed_ready_head = false;
        let mut unmanaged_request_ids = HashSet::new();
        for cleanup in dirty {
            match cleanup.target {
                SchedulerCleanupTarget::AdmissionLifecycle(transfer) => {
                    let result = match transfer.cleanup_target() {
                        Some(AdmissionLifecycleCleanup::Pending { request_id }) => {
                            unmanaged_request_ids.insert(request_id);
                            Ok(())
                        }
                        Some(AdmissionLifecycleCleanup::Booking(booking)) => self
                            .slots
                            .free_if_booking(
                                &booking.request_id,
                                booking.worker,
                                booking.attempt_id,
                                Instant::now(),
                            )
                            .map(|outcome| {
                                made_ready |= outcome.is_applied();
                            }),
                        None => Ok(()),
                    };
                    if let Err(error) = result {
                        tracing::error!(%error, "Failed to release scheduler admission lifecycle");
                    }
                }
                SchedulerCleanupTarget::Booking(booking) => {
                    let result = self
                        .slots
                        .free_if_booking(
                            &booking.request_id,
                            booking.worker,
                            booking.attempt_id,
                            Instant::now(),
                        )
                        .map(|outcome| {
                            made_ready |= outcome.is_applied();
                        });
                    if let Err(error) = result {
                        tracing::error!(
                            request_id = %booking.request_id,
                            worker = ?booking.worker,
                            attempt_id = %booking.attempt_id,
                            %error,
                            "Failed to release request-attempt scheduler booking"
                        );
                    }
                }
                SchedulerCleanupTarget::ExpiredBooking(booking) => {
                    let result = self
                        .slots
                        .expire_if_booking(
                            &booking.request_id,
                            booking.worker,
                            booking.attempt_id,
                            Instant::now(),
                        )
                        .map(|outcome| {
                            made_ready |= outcome.is_applied();
                        });
                    if let Err(error) = result {
                        tracing::error!(
                            request_id = %booking.request_id,
                            worker = ?booking.worker,
                            attempt_id = %booking.attempt_id,
                            %error,
                            "Failed to expire request-attempt scheduler booking"
                        );
                    }
                }
            }
        }
        if !unmanaged_request_ids.is_empty() {
            for class_index in 0..self.profile.classes().len() {
                let (removed, class_head_removed) =
                    self.pending.take_if_in_class(class_index, |queued| {
                        queued
                            .request
                            .mode
                            .tracked_request_id()
                            .is_some_and(|request_id| unmanaged_request_ids.contains(request_id))
                    });
                removed_ready_head |= class_head_removed;
                let lost_entries = !removed.is_empty();
                for entry in removed {
                    self.subtract_pending_counters(class_index, entry.snapshot());
                }
                if lost_entries {
                    self.rearm_class_timer(class_index);
                }
            }
        }
        made_ready || (removed_ready_head && self.has_dispatchable_ready_head())
    }

    fn has_dispatchable_ready_head(&self) -> bool {
        let decay_now = Instant::now();
        let slots = &self.slots;
        let available_ids = self.available_worker_ids.as_ref();
        let workers = &self.workers_with_configs;
        self.pending.any_ready_head(|_, class, queued| {
            if !class.queueing_enabled() {
                return true;
            }
            let available = available_ids.and_then(|ids| ids.get(&queued.request));
            let configs = workers.borrow();
            !Self::all_eligible_workers_full(
                &slots.worker_load_reader(decay_now),
                &configs,
                class.busy_thresholds,
                queued
                    .request
                    .eligibility()
                    .with_available_workers(available.as_deref()),
            )
        })
    }

    fn subtract_pending_counters(&self, class_index: usize, snapshot: QueueSnapshot) {
        self.pending_count.fetch_sub(1, AtomicOrdering::Relaxed);
        self.pending_isl_tokens
            .fetch_sub(snapshot.raw_isl_tokens, AtomicOrdering::Relaxed);
        self.subtract_class_counters(class_index, snapshot);
    }

    /// Consume recorded capacity changes before choosing among shared and pinned lanes.
    fn recheck_capacity_updates(&mut self, mut recheck_all: bool) {
        if let Some(updates) = &self.capacity_updates {
            recheck_all |= updates.drain_into(&mut self.capacity_workers);
        }
        // Consume hints even when there is no pending work. Blocked pinned lanes
        // count as ready here and still need their capacity rechecked.
        if !self.pending.has_ready() {
            return;
        }
        if !recheck_all {
            let configs = self.workers_with_configs.borrow();
            recheck_all = self
                .capacity_workers
                .iter()
                .any(|&worker| worker_config_for_rank(&configs, worker).is_err());
        }
        if recheck_all {
            self.pending.recheck_all_workers();
        } else {
            for &worker in &self.capacity_workers {
                self.pending.recheck_worker(worker);
            }
        }
    }

    async fn handle_capacity_updates(&mut self, drain_cleanup: bool) {
        self.expire_due_classes(Instant::now());
        let cleanup_ready = drain_cleanup && self.drain_cleanup();
        self.recheck_capacity_updates(cleanup_ready);
        if !self.pending.has_ready() {
            return;
        }
        self.drain_ready(None).await;
    }

    /// Rechecks every blocked lane, then drains as [`Self::drain_ready`] does.
    async fn handle_update(&mut self, reply: Option<CommandReply>) {
        self.expire_due_classes(Instant::now());
        self.recheck_capacity_updates(true);
        self.drain_ready(reply).await;
    }

    /// Recheck recorded capacity changes before an enqueue drains the queue, so
    /// an older blocked pinned lane competes with later shared arrivals.
    async fn handle_enqueued(&mut self, reply: Option<CommandReply>) {
        self.expire_due_classes(Instant::now());
        self.recheck_capacity_updates(false);
        self.drain_ready(reply).await;
    }

    /// Admits ready work until no lane head can dispatch. `reply`, from the command that started
    /// the drain, is sent before the first refresh await or when the drain ends, never held across
    /// that await.
    async fn drain_ready(&mut self, mut reply: Option<CommandReply>) {
        // Continuation draining stays actor-local; never self-send through the
        // bounded command channel while processing an update.
        loop {
            if !self.pending.has_ready() {
                break;
            }
            let decay_now = Instant::now();
            // An expired head is rejected here rather than dispatched.
            self.expire_due_classes(decay_now);
            // Load is read after the previous admission booked, so this one sees that
            // reservation. Frees and peer bookings from other tasks can land between
            // two reads, as they can right after the gate.
            let extracted = {
                let slots = &self.slots;
                let available_ids = self.available_worker_ids.as_ref();
                let workers = &self.workers_with_configs;
                self.pending.extract_next(|_, class, queued| {
                    if !class.queueing_enabled() {
                        return true;
                    }
                    let available = available_ids.and_then(|ids| ids.get(&queued.request));
                    let configs = workers.borrow();
                    // TODO: This preserves head-of-line blocking within each policy
                    // class. A blocked constrained head can stall later entries in
                    // that class until a bounded non-HOL policy is introduced.
                    !Self::all_eligible_workers_full(
                        &slots.worker_load_reader(decay_now),
                        &configs,
                        class.busy_thresholds,
                        queued
                            .request
                            .eligibility()
                            .with_available_workers(available.as_deref()),
                    )
                })
            };
            let Some((mut candidate, charge)) = extracted else {
                break;
            };
            let snapshot = candidate.snapshot();
            let current_pending_count = self.pending_count.load(AtomicOrdering::Relaxed);
            debug_assert!(
                current_pending_count > 0,
                "pending_count underflow on queue drain"
            );
            self.pending_count.fetch_sub(1, AtomicOrdering::Relaxed);
            let current_pending_isl_tokens = self.pending_isl_tokens.load(AtomicOrdering::Relaxed);
            debug_assert!(
                current_pending_isl_tokens >= snapshot.raw_isl_tokens,
                "pending_isl_tokens underflow: pending={} request_isl_tokens={}",
                current_pending_isl_tokens,
                snapshot.raw_isl_tokens
            );
            self.pending_isl_tokens
                .fetch_sub(snapshot.raw_isl_tokens, AtomicOrdering::Relaxed);
            let class_index = candidate.class_index();
            self.subtract_class_counters(class_index, snapshot);
            let queued = candidate.payload_mut();
            let mut refreshed = false;
            let mut released = false;
            let mut stopping = false;
            let refreshing = should_refresh_overlap(
                self.overlap_scores_refresh.is_some(),
                self.overlap_refresh_after,
                queued.block_hashes.as_deref(),
                queued.enqueue_at,
                decay_now,
            );
            if refreshing {
                if let Some(reply) = reply.take() {
                    reply.send();
                }
                // NOTE: Overlap refresh is bounded by `OVERLAP_REFRESH_TIMEOUT`. We intentionally
                // accept load crossing the class threshold during this await: busy
                // thresholds guide admission, not reservation. The candidate's DRR
                // charge stays unpaid until its admission outcome is known.
                match self.refresh_candidate(queued, decay_now).await {
                    RefreshEnd::Finished(Some(snapshot)) => {
                        refreshed = true;
                        tracing::debug!(
                            request_id = queued.request.mode.request_id().unwrap_or("unknown"),
                            wait_ms = queued.enqueue_at.elapsed().as_millis() as u64,
                            "refreshed overlap scores after long queue wait"
                        );
                        queued.request.overlap = snapshot.overlap;
                        queued.request.kv_transfer_candidates =
                            if queued.request.retain_kv_transfer_chain {
                                snapshot.kv_transfer_candidates
                            } else {
                                None
                            };
                    }
                    RefreshEnd::Finished(None) => {}
                    end => {
                        tracing::debug!(
                            request_id = queued.request.mode.request_id().unwrap_or("unknown"),
                            wait_ms = queued.enqueue_at.elapsed().as_millis() as u64,
                            outcome = ?end,
                            "overlap refresh ended unfinished"
                        );
                        released = matches!(end, RefreshEnd::Released);
                        stopping = matches!(end, RefreshEnd::Shutdown);
                    }
                }
            }
            let queued = candidate.into_payload();
            // Only this turn's request has rows, and a refresh replaces the overlap
            // they were built from.
            let prepared = if queued.enqueue_turn == self.turn && !refreshed {
                self.turn_prepared.take()
            } else {
                None
            };
            // The due time can pass during the refresh. Admission checks it again before
            // selection and before booking or delivery.
            let admission = if released {
                Admission::Refused
            } else if refreshing && queued.due_at.is_some_and(|due_at| due_at <= Instant::now()) {
                let mut request = queued.request;
                self.reject_due_time_passed(class_index, &mut request);
                Admission::Refused
            } else {
                tracing::trace!(
                    policy_class = self.profile.class(class_index).name,
                    "scheduling request from pending queue"
                );
                let context = AdmissionContext {
                    class_index,
                    due_at: queued.due_at,
                    attempt_tx: queued.attempt_tx,
                    lifecycle_transfer: queued.lifecycle_transfer,
                };
                self.admit_one(context, queued.request, Instant::now(), prepared)
            };
            self.pending.settle(charge, admission != Admission::Refused);
            self.rearm_class_timer(class_index);
            // Every sender is gone, so the actor stops next.
            if stopping {
                break;
            }
        }
        if let Some(reply) = reply {
            reply.send();
        }
    }

    /// Runs the optional overlap refresh of an extracted candidate while class timers, capacity
    /// wakes, and commands are handled, each in a new actor turn. A command applied here is
    /// answered at once; the drain it calls for is the one that resumes. The candidate has left
    /// its class queue, so class expiry no longer covers it: the refresh races its own due time.
    /// That due time, its response closing, its lifecycle cleanup, shutdown, or the refresh
    /// timeout (see [`OverlapScoresRefresh`]) ends the refresh early.
    async fn refresh_candidate(
        &mut self,
        queued: &mut QueuedRequest,
        decay_now: Instant,
    ) -> RefreshEnd {
        let refresher = self.overlap_scores_refresh.clone();
        // One timeout for the whole refresh: the events handled below never restart it.
        let refresh = tokio::time::timeout(
            OVERLAP_REFRESH_TIMEOUT,
            refresh_overlap(
                refresher.as_deref(),
                self.overlap_refresh_after,
                queued.block_hashes.as_deref(),
                queued.request.retain_kv_transfer_chain,
                queued.enqueue_at,
                decay_now,
            ),
        );
        tokio::pin!(refresh);
        let capacity_updates = self.capacity_updates.clone();
        let due_at = queued.due_at;
        loop {
            let timer = self.class_timers.next_deadline();
            let response = &mut queued.request.resp_tx;
            let event = tokio::select! {
                biased;
                refreshed = &mut refresh => {
                    return refreshed.map_or(RefreshEnd::TimedOut, RefreshEnd::Finished);
                }
                _ = async {
                    if let Some(response) = response {
                        response.closed().await;
                    }
                } => return RefreshEnd::ResponseClosed,
                _ = async {
                    if let Some(due_at) = due_at {
                        tokio::time::sleep_until(due_at).await;
                    }
                }, if due_at.is_some() => return RefreshEnd::DueTimePassed,
                _ = async {
                    if let Some(timer) = timer {
                        tokio::time::sleep_until(timer).await;
                    }
                }, if timer.is_some() => RefreshEvent::ClassTimer,
                _ = async {
                    if let Some(updates) = &capacity_updates {
                        updates.notified().await;
                    }
                }, if capacity_updates.is_some() => RefreshEvent::Capacity,
                command = self.commands.next() => match command {
                    Some(command) => RefreshEvent::Command(command),
                    None => return RefreshEnd::Shutdown,
                },
            };
            self.begin_turn();
            match event {
                RefreshEvent::ClassTimer => {
                    self.expire_due_classes(Instant::now());
                }
                RefreshEvent::Capacity => {
                    let cleanup_ready = self.drain_cleanup();
                    self.recheck_capacity_updates(cleanup_ready);
                }
                RefreshEvent::Command(command) => {
                    let (drain, reply) = self.apply_command(command, true);
                    match drain {
                        Drain::None => {}
                        Drain::Enqueued => self.recheck_capacity_updates(false),
                        Drain::Full => self.recheck_capacity_updates(true),
                    }
                    if let Some(reply) = reply {
                        reply.send();
                    }
                }
            }
            // Cleanup may have released the candidate's own lifecycle. Its caller is gone, and a
            // booking made now would have no owner to free it.
            if queued
                .lifecycle_transfer
                .as_ref()
                .is_some_and(|transfer| !transfer.is_armed())
            {
                return RefreshEnd::Released;
            }
        }
    }

    /// Whether a load projection counts each rank's overlap with the request's active prompts.
    /// Finding it walks the prompt trie, and only `LOAD` rows read it; without them, only the
    /// selected rank's is computed.
    fn projects_overlap(&self) -> bool {
        self.selector
            .required_worker_inputs()
            .contains(WorkerInputs::LOAD)
    }

    /// Refresh derived load for every request, retaining the allocation through
    /// selection, booking, and response-field capture, including error paths.
    fn with_projected_loads<R>(
        &mut self,
        mut request: SchedulingRequest,
        decay_now: Instant,
        handle: impl FnOnce(&Self, &mut SchedulingRequest) -> R,
    ) -> R {
        request.worker_loads = std::mem::take(&mut self.projected_loads);
        let project_overlap = self.projects_overlap();
        // The host's reject thresholds are judged in the same pass that projects each rank's
        // load, so selection only consults the ranks found full: no second read per rank.
        let full_ranks = match self
            .reject_thresholds
            .as_ref()
            .map(|provider| provider())
            .filter(BusyThresholds::is_configured)
        {
            Some(limits) => {
                self.full_ranks.reset(limits);
                Some(&mut self.full_ranks)
            }
            None => {
                self.full_ranks.clear();
                None
            }
        };
        self.slots.project_worker_loads_into(
            request.token_seq.as_deref().filter(|_| project_overlap),
            decay_now,
            &mut request.worker_loads,
            full_ranks,
        );
        let result = handle(self, &mut request);
        self.projected_loads = std::mem::take(&mut request.worker_loads);
        result
    }

    /// `admitted` selections may carry the installed selection trace; advisory
    /// probes never do, so they do not pay for capture.
    fn select_worker_for_request(
        &self,
        request: &SchedulingRequest,
        admitted: bool,
        prepared: Option<Box<PreparedAdmission>>,
    ) -> Result<SelectedWorkerForRequest, KvSchedulerError> {
        let available_worker_ids = self
            .available_worker_ids
            .as_ref()
            .and_then(|ids| ids.get(request));

        {
            let workers = self.workers_with_configs.borrow();
            let overloaded_worker_ids = self
                .overloaded_worker_provider
                .as_ref()
                .and_then(|provider| provider());
            let exclusive_affinity = self.selector.uses_exclusive_affinity_target();
            let eligibility = selection_eligibility(
                request,
                &workers,
                overloaded_worker_ids.as_deref(),
                available_worker_ids.as_deref(),
                Some(&self.full_ranks),
                exclusive_affinity,
            );
            // Selection drops full workers' rows from prepared rows. Preparation did not know
            // which ranks are full, so an exclusive affinity target it found eligible may be
            // full, and fetch sources span every row: those requests rebuild.
            let prepared_rows_usable = self.full_ranks.is_empty()
                || (!(exclusive_affinity && request.affinity_target.is_some())
                    && request.kv_transfer_candidates.is_none());
            // Checked under the borrow selection uses: a change since the actor
            // last advanced the generation is not counted in it yet.
            let prepared = prepared
                .filter(|_| !workers.has_changed() && prepared_rows_usable)
                .and_then(|prepared| {
                    prepared.into_current(
                        self.workers_generation.load(AtomicOrdering::Relaxed),
                        available_worker_ids.as_ref(),
                        overloaded_worker_ids.as_ref(),
                    )
                });
            let trace = if admitted {
                self.selection_trace.get().copied()
            } else {
                None
            };
            let progress = |worker| self.slots.decode_progress_for(worker);
            // Held for this selection, so the snapshot the policy reads borrows live counters.
            let resident_blocks = self
                .resident_blocks
                .get()
                .map(ResidentBlockCountsHandle::load);
            self.selector
                .select_worker_with_decode_progress(
                    WorkerSelectionInput::configured(
                        &workers,
                        request,
                        eligibility,
                        self.block_size,
                    )
                    .with_resident_blocks(resident_blocks.as_deref().map(|counts| &**counts)),
                    prepared,
                    trace,
                    Some(&progress),
                )
                .map(|(mut selection, selection_trace)| {
                    // A projection without overlap counts none of the request's blocks as
                    // new on any rank. Count them for the selected rank alone.
                    if !self.projects_overlap()
                        && let Some(token_seq) = request.token_seq.as_deref()
                        && request.worker_loads.contains_key(&selection.worker)
                    {
                        selection.potential_decode_blocks += self
                            .slots
                            .additional_active_blocks(token_seq, selection.worker);
                    }
                    // Measure affinity selections against every eligible worker so cache
                    // metrics still mean "best eligible worker". Explicit pins are not
                    // widened, nor are selectors whose own filters may reject workers.
                    let affinity_selected = request
                        .affinity_target
                        .is_some_and(|target| target.matches(selection.worker));
                    let metrics_eligibility = if affinity_selected
                        && !self.selector.applies_worker_filters()
                    {
                        let unconstrained = RoutingEligibility::new(
                            request.allowed_worker_ids.as_ref(),
                            overloaded_worker_ids.as_deref(),
                            None,
                            &request.routing_constraints,
                        )
                        .with_available_workers(available_worker_ids.as_deref())
                        .with_full_worker_ranks(&self.full_ranks);
                        if let Some(max_raw_cached_tokens) = &mut selection.max_raw_cached_tokens {
                            *max_raw_cached_tokens = best_eligible_raw_cached_tokens(
                                &workers,
                                request,
                                unconstrained,
                                *max_raw_cached_tokens,
                                self.block_size,
                            );
                        }
                        unconstrained
                    } else {
                        eligibility
                    };
                    let non_max_overlap_selection = if request.mode.is_tracked()
                        && self.non_max_overlap_selection_observer.get().is_some()
                    {
                        non_max_overlap_selection(
                            &workers,
                            request,
                            metrics_eligibility,
                            selection.worker,
                            selection.effective_overlap_blocks,
                        )
                    } else {
                        None
                    };
                    let config = workers
                        .get(&selection.worker.worker_id)
                        .expect("selected worker config must exist");
                    let selected_worker_tiers = request
                        .overlap
                        .selected_worker_tiers(selection.worker, config);
                    let worker_load = request.worker_load_for(selection.worker);
                    let capacity = WorkerCapacity::of(config);
                    let selected_worker_load = AdvisoryWorkerLoad {
                        active_prefill_tokens: worker_load.active_prefill_tokens,
                        prefill_token_capacity: capacity.prefill_budget() as usize,
                        // TODO(rank-aware-kv-capacity): resolve the selected DP rank and preserve
                        // capacity quality. Estimated fallbacks must not authorize load-based
                        // admission/bypass decisions that require an exact or conservative bound.
                        total_kv_blocks: capacity.total_kv_blocks().map(|blocks| blocks as usize),
                    };
                    SelectedWorkerForRequest {
                        selection,
                        selection_trace,
                        selected_worker_tiers,
                        selected_worker_load,
                        non_max_overlap_selection,
                    }
                })
        }
    }

    fn select_without_admission_inner(
        &mut self,
        request: SchedulingRequest,
        decay_now: Instant,
    ) -> Result<AdvisorySchedulingResponse, KvSchedulerError> {
        self.with_projected_loads(request, decay_now, |actor, request| {
            let selected = actor.select_worker_for_request(request, false, None)?;

            Ok(AdvisorySchedulingResponse {
                selected_worker_load: selected.selected_worker_load,
                response: SchedulingResponse {
                    best_worker: selected.selection.worker,
                    effective_overlap_blocks: selected.selection.effective_overlap_blocks,
                    cached_tokens: selected.selection.cached_tokens,
                    max_raw_cached_tokens: selected.selection.max_raw_cached_tokens,
                    selected_raw_cached_tokens: selected.selection.selected_raw_cached_tokens,
                    selected_worker_tiers: selected.selected_worker_tiers,
                    kv_copy_source: request.kv_copy_source(
                        selected.selection.worker,
                        selected.selection.kv_transfer_source,
                    ),
                    kv_transfer_candidates: request.kv_transfer_candidates.take(),
                    potential_decode_blocks: selected.selection.potential_decode_blocks,
                    selection_trace: selected.selection_trace,
                    log_fields: selected.selection.log_fields,
                },
            })
        })
    }

    /// Run the full scheduling pipeline for a single request of the context's class:
    /// compute projected load -> select worker -> book tracked state -> respond.
    /// A request whose due time passes before selection, booking, or delivery is rejected
    /// instead.
    fn admit_one(
        &mut self,
        context: AdmissionContext,
        request: SchedulingRequest,
        decay_now: Instant,
        prepared: Option<Box<PreparedAdmission>>,
    ) -> Admission {
        if prepared.is_some() {
            self.refresh_workers_generation();
        }
        self.with_projected_loads(request, decay_now, |actor, request| {
            actor.admit_one_projected(context, request, prepared)
        })
    }

    /// Advance the generation when the configuration changed since the actor
    /// last looked, so rows prepared against an earlier one are rebuilt.
    fn refresh_workers_generation(&mut self) {
        if self.workers_with_configs.borrow_and_update().has_changed() {
            self.workers_generation
                .fetch_add(1, AtomicOrdering::Release);
        }
    }

    fn admit_one_projected(
        &self,
        context: AdmissionContext,
        request: &mut SchedulingRequest,
        prepared: Option<Box<PreparedAdmission>>,
    ) -> Admission {
        // The dispatch gate and load projection ran after the drain read the clock. Check the
        // due time again so a failing selection cannot hide an expiry.
        if context
            .due_at
            .is_some_and(|due_at| due_at <= Instant::now())
        {
            self.reject_due_time_passed(context.class_index, request);
            return Admission::Refused;
        }
        let selected = match self.select_worker_for_request(request, true, prepared) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("scheduling failed: {e}");
                request.respond(Err(e));
                return Admission::Refused;
            }
        };

        let response = SchedulingResponse {
            best_worker: selected.selection.worker,
            effective_overlap_blocks: selected.selection.effective_overlap_blocks,
            cached_tokens: selected.selection.cached_tokens,
            max_raw_cached_tokens: selected.selection.max_raw_cached_tokens,
            selected_raw_cached_tokens: selected.selection.selected_raw_cached_tokens,
            selected_worker_tiers: selected.selected_worker_tiers,
            kv_copy_source: request.kv_copy_source(
                selected.selection.worker,
                selected.selection.kv_transfer_source,
            ),
            kv_transfer_candidates: request.kv_transfer_candidates.take(),
            potential_decode_blocks: selected.selection.potential_decode_blocks,
            selection_trace: selected.selection_trace,
            log_fields: selected.selection.log_fields,
        };
        let non_max_overlap_selection = selected.non_max_overlap_selection;

        if !request.mode.is_tracked() {
            if context
                .due_at
                .is_some_and(|due_at| due_at <= Instant::now())
            {
                self.reject_due_time_passed(context.class_index, request);
                return Admission::Refused;
            }
            return if request.respond(Ok(response)) {
                Admission::Selected
            } else {
                Admission::Refused
            };
        }

        let request_id = request
            .mode
            .tracked_request_id()
            .expect("tracked mode always has a request ID")
            .to_string();

        let prefill_load_hint = self.prefill_load_hint_for(
            request.isl_tokens,
            selected.selection.cached_tokens,
            request.track_prefill_tokens,
        );

        let sequence_request = SequenceRequest {
            request_id,
            token_sequence: request.token_seq.take(),
            track_prefill_tokens: request.track_prefill_tokens,
            expected_output_tokens: request.expected_output_tokens,
            prefill_load_hint,
            worker: selected.selection.worker,
            lora_name: request
                .accounting_attribution
                .accounting_name(request.lora_name.take()),
        };
        self.book_and_respond(
            context,
            request,
            sequence_request,
            response,
            non_max_overlap_selection,
        )
    }

    /// A closed receiver means the actor-owned request was abandoned before
    /// booking, so there is nothing to install. A request whose `due_at` has passed is
    /// rejected unbooked, judged on the same instant that would stamp its booking. Otherwise
    /// booking precedes the response: once delivery succeeds, the response channel no longer
    /// tracks request lifetime and the caller must install its RAII cleanup owner. If
    /// delivery loses that race, roll back the booking here.
    fn book_and_respond(
        &self,
        context: AdmissionContext,
        request: &mut SchedulingRequest,
        sequence_request: SequenceRequest,
        response: SchedulingResponse,
        non_max_overlap_selection: Option<NonMaxOverlapSelection>,
    ) -> Admission {
        let AdmissionContext {
            class_index,
            due_at,
            attempt_tx,
            lifecycle_transfer,
        } = context;
        if request.response_is_closed() {
            tracing::debug!(
                request_id = %sequence_request.request_id,
                "Skipping scheduler booking for cancelled request"
            );
            return Admission::Refused;
        }

        let admitted_at = Instant::now();
        if due_at.is_some_and(|due_at| due_at <= admitted_at) {
            self.reject_due_time_passed(class_index, request);
            return Admission::Refused;
        }
        let request_id = sequence_request.request_id.clone();
        let worker = sequence_request.worker;
        let attempt_id = match self
            .slots
            .add_request_admitted(sequence_request, admitted_at)
        {
            Ok(attempt_id) => attempt_id,
            Err(error) => {
                tracing::warn!(%request_id, %error, "Failed to book scheduler state");
                request.respond(Err(KvSchedulerError::BookingFailed(error.to_string())));
                return Admission::Refused;
            }
        };
        if let Some(transfer) = lifecycle_transfer {
            transfer.arm_booking(SchedulerBookingDescriptor {
                request_id: request_id.clone(),
                worker,
                attempt_id,
            });
        }
        if let Some(attempt_tx) = attempt_tx {
            let _ = attempt_tx.send(attempt_id);
        }
        if request.respond(Ok(response)) {
            if let Some(selection) = non_max_overlap_selection {
                self.dispatch_non_max_overlap_selection(request_id, selection);
            }
            return Admission::Booked;
        }

        tracing::debug!(%request_id, "Rolling back undelivered scheduler booking");
        if let Err(error) =
            self.slots
                .free_if_booking(&request_id, worker, attempt_id, Instant::now())
        {
            tracing::error!(%request_id, %error, "Failed to roll back scheduler booking");
        }
        Admission::Refused
    }

    fn dispatch_non_max_overlap_selection(
        &self,
        request_id: String,
        selection: NonMaxOverlapSelection,
    ) {
        let Some(observer) = self.non_max_overlap_selection_observer.get() else {
            return;
        };
        let observer = Arc::clone(observer);
        let _observer_task = tokio::task::spawn_blocking(move || {
            observer(&request_id, selection);
        });
    }

    fn prefill_load_hint_for(
        &self,
        isl_tokens: usize,
        cached_tokens: usize,
        track_prefill_tokens: bool,
    ) -> Option<PrefillLoadHint> {
        if !track_prefill_tokens {
            return None;
        }

        let effective_isl = effective_prefill_tokens(isl_tokens, cached_tokens);
        if effective_isl == 0 {
            return None;
        }
        let prefix = isl_tokens - effective_isl;

        let expected_prefill_duration = match &self.prefill_load_estimator {
            Some(estimator) => match estimator.predict_prefill_duration(1, effective_isl, prefix) {
                Ok(expected_prefill_duration) => Some(expected_prefill_duration),
                Err(error) => {
                    tracing::warn!(
                        effective_isl,
                        prefix,
                        "failed to predict prefill duration for active load tracking: {error}"
                    );
                    None
                }
            },
            None => None,
        };

        Some(PrefillLoadHint {
            initial_effective_prefill_tokens: effective_isl,
            expected_prefill_duration,
        })
    }

    /// Whether every eligible rank is past `limits`, judged on the same `LOAD` row worker
    /// selection reads.
    fn all_eligible_workers_full(
        worker_loads: &WorkerLoadReader<'_>,
        configs: &HashMap<WorkerId, C>,
        limits: BusyThresholds,
        eligibility: RoutingEligibility<'_>,
    ) -> bool {
        let has_room = |worker: WorkerWithDpRank, config: &C| {
            let load = WorkerLoadInput::from_observed(
                worker_loads.get(worker),
                WorkerLoadProjection::default,
            );
            limits.has_room(&load, WorkerCapacity::of(config))
        };

        if let Some(worker) = eligibility.pinned_worker() {
            let Ok(config) = eligibility.validate_worker_rank(configs, worker) else {
                return false;
            };
            return !has_room(worker, config);
        }

        let mut checked_any = false;
        let has_available = eligibility.any_eligible_worker_rank(configs, |worker, config| {
            checked_any = true;
            has_room(worker, config)
        });

        checked_any && !has_available
    }

    fn add_class_counters(&self, class_index: usize, snapshot: QueueSnapshot) {
        let counters = &self.class_counters[class_index];
        counters.pending_count.fetch_add(1, AtomicOrdering::Relaxed);
        counters
            .pending_isl_tokens
            .fetch_add(snapshot.raw_isl_tokens, AtomicOrdering::Relaxed);
        counters
            .pending_cached_tokens
            .fetch_add(snapshot.cached_tokens, AtomicOrdering::Relaxed);
    }

    fn subtract_class_counters(&self, class_index: usize, snapshot: QueueSnapshot) {
        let counters = &self.class_counters[class_index];
        counters.pending_count.fetch_sub(1, AtomicOrdering::Relaxed);
        counters
            .pending_isl_tokens
            .fetch_sub(snapshot.raw_isl_tokens, AtomicOrdering::Relaxed);
        counters
            .pending_cached_tokens
            .fetch_sub(snapshot.cached_tokens, AtomicOrdering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use std::sync::{Arc, Condvar, Mutex as StdMutex};

    use std::time::Duration;

    use async_trait::async_trait;

    use rustc_hash::FxHashMap;

    use tokio::sync::{Barrier, watch};

    use super::*;

    use crate::OverlapSignals;

    use crate::classifier_rows::ClassifierRows;

    use crate::kv_hints::KvTransferCandidates;

    use crate::protocols::WorkerSelectionResult;
    use dynamo_kv_hashing::ExternalSequenceBlockHash;
    use dynamo_router_protocol::worker::WorkerWithDpRank;
    use dynamo_slot_tracker::wire::ActiveSequenceEvent;

    use crate::selector::{DefaultWorkerSelector, WorkerInputs, WorkerSelector};

    use crate::test_utils::{NoopSequencePublisher, SimpleWorkerConfig};

    use crate::types::{KvSchedulerError, ScheduleMode};

    use crate::RefreshedOverlap;

    use dynamo_admission::classifier::WorkerView;
    use dynamo_admission::{PolicyClassConfig, RouterQueuePolicy};
    use dynamo_slot_tracker::busy::BusyThresholds;
    use std::num::NonZeroU32;

    use dynamo_slot_tracker::topology::WorkerDpRange;

    use dynamo_slot_tracker::{ActiveSequencesMultiWorker, SequencePublisher};

    fn decay_now() -> Instant {
        Instant::now()
    }

    /// A configured class as the router's policy file resolves it: EDF order, no budget, no
    /// limits. Tests set the fields the file would.
    fn class(name: &str, quantum: usize) -> PolicyClassConfig {
        PolicyClassConfig {
            name: name.to_string(),
            queue_policy: RouterQueuePolicy::Edf,
            quantum,
            slo_ttft_ms: None,
            controlled_delay: false,
            busy_thresholds: BusyThresholds::default(),
            request_queue_limit_per_worker: None,
            raw_isl_token_queue_limit_per_worker: None,
            cached_token_queue_limit_per_worker: None,
        }
    }

    /// A profile of configured classes, as the composed router's YAML resolver hands it to the
    /// kernel; the file format itself is tested beside that resolver.
    fn policy_profile(
        default_policy_class: Option<&str>,
        classes: Vec<PolicyClassConfig>,
    ) -> PolicyProfile {
        PolicyProfile::from_classes(classes, default_policy_class).unwrap()
    }

    /// Moves the paused test clock forward by `by` from synchronous code, as slow synchronous
    /// admission work would. `advance` moves the clock on its first poll and then only yields,
    /// so dropping it keeps the move and runs no timer.
    fn advance_paused_clock(by: Duration) {
        let before = Instant::now();
        let _ = futures_util::FutureExt::now_or_never(tokio::time::advance(by));
        assert_eq!(Instant::now(), before + by);
    }

    /// Moves the paused clock by `by` the first time it is woken.
    struct AdvanceClockOnWake {
        by: Duration,
        fired: AtomicBool,
    }

    impl std::task::Wake for AdvanceClockOnWake {
        fn wake(self: Arc<Self>) {
            self.wake_by_ref();
        }

        fn wake_by_ref(self: &Arc<Self>) {
            if !self.fired.swap(true, Ordering::Relaxed) {
                advance_paused_clock(self.by);
            }
        }
    }

    /// Sending this receiver's response wakes it, and that moves the paused clock by `by`: the
    /// actor work that responds then takes that long.
    fn advance_clock_on_response(response_rx: &mut SchedulingResponseReceiver, by: Duration) {
        let waker = std::task::Waker::from(Arc::new(AdvanceClockOnWake {
            by,
            fired: AtomicBool::new(false),
        }));
        let mut cx = std::task::Context::from_waker(&waker);
        assert!(std::future::Future::poll(std::pin::Pin::new(response_rx), &mut cx).is_pending());
    }

    struct FixedPrefillLoadEstimator {
        duration: Duration,
    }

    impl PrefillLoadEstimator for FixedPrefillLoadEstimator {
        fn predict_prefill_duration(
            &self,
            _batch_size: usize,
            _effective_isl: usize,
            _prefix: usize,
        ) -> anyhow::Result<Duration> {
            Ok(self.duration)
        }
    }

    type SchedulingResponseReceiver =
        tokio::sync::oneshot::Receiver<Result<SchedulingResponse, KvSchedulerError>>;

    fn book_directly(
        slots: &ActiveSequencesMultiWorker<NoopSequencePublisher>,
        request_id: &str,
    ) -> SchedulerBookingDescriptor {
        let worker = WorkerWithDpRank::new(0, 0);
        let attempt_id = slots
            .add_request_admitted(
                dynamo_slot_tracker::SequenceRequest {
                    request_id: request_id.to_string(),
                    token_sequence: None,
                    track_prefill_tokens: true,
                    expected_output_tokens: None,
                    prefill_load_hint: None,
                    worker,
                    lora_name: None,
                },
                Instant::now(),
            )
            .expect("worker 0 is registered");
        SchedulerBookingDescriptor {
            request_id: request_id.to_string(),
            worker,
            attempt_id,
        }
    }

    async fn wait_freed(
        slots: &ActiveSequencesMultiWorker<NoopSequencePublisher>,
        booking: &SchedulerBookingDescriptor,
    ) {
        tokio::time::timeout(Duration::from_secs(2), async {
            while slots.has_booking(booking) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("booking was freed");
    }

    struct DropResponseOnLoadPublisher {
        response_rx: Arc<StdMutex<Option<SchedulingResponseReceiver>>>,
    }

    impl SequencePublisher for DropResponseOnLoadPublisher {
        fn enqueue_event(&self, _event: ActiveSequenceEvent) -> anyhow::Result<()> {
            Ok(())
        }

        fn publish_scheduler_load(&self, _load: dynamo_slot_tracker::SchedulerLoadSnapshot) {
            self.response_rx.lock().unwrap().take();
        }

        fn observe_load(
            &self,
            _: &WorkerWithDpRank,
            _: &str,
            _: dynamo_slot_tracker::LocalWorkerLoad,
        ) {
        }
    }

    #[derive(Default)]
    struct SelectorRendezvous {
        arrivals: StdMutex<usize>,
        cv: Condvar,
    }

    impl SelectorRendezvous {
        fn wait_for_peer(&self) {
            let mut arrivals = self.arrivals.lock().unwrap();
            *arrivals += 1;

            if *arrivals == 1 {
                let _ = self
                    .cv
                    .wait_timeout(arrivals, Duration::from_millis(100))
                    .unwrap();
                return;
            }

            self.cv.notify_all();
        }
    }

    #[derive(Clone)]
    struct MinDecodeSelector {
        rendezvous: Option<Arc<SelectorRendezvous>>,
    }

    impl WorkerSelector<SimpleWorkerConfig> for MinDecodeSelector {
        fn required_worker_inputs(&self) -> WorkerInputs {
            WorkerInputs::CACHE | WorkerInputs::LOAD
        }

        fn select_worker(
            &self,
            input: WorkerSelectionInput<'_, SimpleWorkerConfig>,
        ) -> Result<WorkerSelectionResult, KvSchedulerError> {
            let (workers, request, eligibility, block_size) = input.into_configured();
            if let Some(rendezvous) = &self.rendezvous {
                rendezvous.wait_for_peer();
            }

            let mut best_worker = None;
            eligibility.for_each_eligible_worker_rank(workers, |worker, _| {
                let load = request.worker_load_for(worker);
                let potential_prefill_tokens = if request.track_prefill_tokens {
                    load.active_prefill_tokens
                        .saturating_add(effective_prefill_tokens(
                            request.isl_tokens,
                            request.effective_cached_tokens_for(worker),
                        ))
                } else {
                    0
                };
                let potential_decode_blocks = load.potential_decode_blocks();
                let key = (
                    potential_prefill_tokens,
                    potential_decode_blocks,
                    worker.worker_id,
                    worker.dp_rank,
                );
                if best_worker.is_none_or(|(_, best_key)| key < best_key) {
                    best_worker = Some((worker, key));
                }
            });

            let Some((worker, _)) = best_worker else {
                return Err(KvSchedulerError::NoEndpoints);
            };

            Ok(WorkerSelectionResult {
                worker,
                required_blocks: request.request_blocks(block_size),
                effective_overlap_blocks: request.effective_overlap_blocks_for(worker),
                cached_tokens: request.effective_cached_tokens_for(worker),
                max_raw_cached_tokens: None,
                selected_raw_cached_tokens: None,
                potential_decode_blocks: request
                    .potential_decode_blocks_after_admission(worker, block_size),
                kv_transfer_source: None,
                log_fields: None,
            })
        }
    }

    /// The default selector behind a hook that first sees the ID of each request it selects.
    /// The hook can fail the selection, or take time as a slow selection would.
    struct HookedSelector<F> {
        inner: DefaultWorkerSelector,
        hook: F,
    }

    impl<F> HookedSelector<F> {
        fn new(hook: F) -> Self {
            Self {
                inner: DefaultWorkerSelector::new(None, "test"),
                hook,
            }
        }
    }

    impl<F> WorkerSelector<SimpleWorkerConfig> for HookedSelector<F>
    where
        F: Fn(Option<&str>) -> Result<(), KvSchedulerError>,
    {
        fn required_worker_inputs(&self) -> WorkerInputs {
            WorkerSelector::<SimpleWorkerConfig>::required_worker_inputs(&self.inner)
        }

        fn uses_exclusive_affinity_target(&self) -> bool {
            WorkerSelector::<SimpleWorkerConfig>::uses_exclusive_affinity_target(&self.inner)
        }

        fn select_worker(
            &self,
            input: WorkerSelectionInput<'_, SimpleWorkerConfig>,
        ) -> Result<WorkerSelectionResult, KvSchedulerError> {
            let WorkerSelectionInput::Configured { request, .. } = &input;
            (self.hook)(request.mode.request_id())?;
            self.inner.select_worker(input)
        }
    }

    #[allow(clippy::type_complexity)]
    fn make_queue(
        num_workers: usize,
        block_size: u32,
        isl: usize,
        threshold_frac: Option<f64>,
    ) -> (
        Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, DefaultWorkerSelector>>,
        Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
    ) {
        let (queue, slots, _tx) =
            make_queue_with_sender(num_workers, block_size, isl, threshold_frac, None);
        (queue, slots)
    }

    #[allow(clippy::type_complexity)]
    fn make_queue_with_custom_selector<Sel: WorkerSelector<SimpleWorkerConfig> + Send + 'static>(
        num_workers: usize,
        block_size: u32,
        isl: usize,
        threshold_frac: Option<f64>,
        selector: Sel,
    ) -> (
        Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, Sel>>,
        Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
    ) {
        let dp_range: HashMap<u64, (u32, u32)> =
            (0..num_workers as u64).map(|id| (id, (0, 1))).collect();
        let slots = Arc::new(ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            block_size as usize,
            dp_range,
            false,
            0,
            "test",
        ));

        let mut configs: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        for id in 0..num_workers as u64 {
            configs.insert(
                id,
                SimpleWorkerConfig {
                    max_num_batched_tokens: Some(isl as u64),
                    ..Default::default()
                },
            );
        }
        let (_cfg_tx, cfg_rx) = watch::channel(configs);

        let queue = Arc::new(SchedulerQueue::new(
            Arc::clone(&slots),
            cfg_rx,
            PolicyProfile::synthetic(threshold_frac, RouterQueuePolicy::Fcfs),
            block_size,
            selector,
            None,
            None,
            None,
            None,
            None,
        ));

        (queue, slots)
    }

    #[allow(clippy::type_complexity)]
    fn make_queue_with_sender(
        num_workers: usize,
        block_size: u32,
        isl: usize,
        threshold_frac: Option<f64>,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
    ) -> (
        Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, DefaultWorkerSelector>>,
        Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
        watch::Sender<HashMap<u64, SimpleWorkerConfig>>,
    ) {
        let dp_range: HashMap<u64, (u32, u32)> =
            (0..num_workers as u64).map(|id| (id, (0, 1))).collect();
        let slots = Arc::new(ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            block_size as usize,
            dp_range,
            false,
            0,
            "test",
        ));

        let mut configs: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        for id in 0..num_workers as u64 {
            configs.insert(
                id,
                SimpleWorkerConfig {
                    max_num_batched_tokens: Some(isl as u64),
                    ..Default::default()
                },
            );
        }
        let (cfg_tx, cfg_rx) = watch::channel(configs);

        let selector = DefaultWorkerSelector::new(None, "test");
        let queue = Arc::new(SchedulerQueue::new(
            Arc::clone(&slots),
            cfg_rx,
            PolicyProfile::synthetic(threshold_frac, RouterQueuePolicy::Fcfs),
            block_size,
            selector,
            prefill_load_estimator,
            None,
            None,
            None,
            None,
        ));

        (queue, slots, cfg_tx)
    }

    #[allow(clippy::type_complexity)]
    fn make_queue_with_profile(
        num_workers: usize,
        block_size: u32,
        max_num_batched_tokens: usize,
        profile: PolicyProfile,
    ) -> (
        Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, DefaultWorkerSelector>>,
        Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
    ) {
        let (queue, slots, _cfg_tx) = make_queue_with_profile_and_sender(
            num_workers,
            block_size,
            max_num_batched_tokens,
            profile,
        );
        (queue, slots)
    }

    #[allow(clippy::type_complexity)]
    fn make_queue_with_profile_and_sender(
        num_workers: usize,
        block_size: u32,
        max_num_batched_tokens: usize,
        profile: PolicyProfile,
    ) -> (
        Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, DefaultWorkerSelector>>,
        Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
        watch::Sender<HashMap<u64, SimpleWorkerConfig>>,
    ) {
        make_queue_with_profile_sender_and_reject(
            num_workers,
            block_size,
            max_num_batched_tokens,
            profile,
            None,
        )
    }

    #[allow(clippy::type_complexity)]
    fn make_queue_with_profile_sender_and_reject(
        num_workers: usize,
        block_size: u32,
        max_num_batched_tokens: usize,
        profile: PolicyProfile,
        reject_thresholds: Option<RejectThresholdsProvider>,
    ) -> (
        Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, DefaultWorkerSelector>>,
        Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
        watch::Sender<HashMap<u64, SimpleWorkerConfig>>,
    ) {
        let dp_range: HashMap<u64, (u32, u32)> =
            (0..num_workers as u64).map(|id| (id, (0, 1))).collect();
        let slots = Arc::new(ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            block_size as usize,
            dp_range,
            false,
            0,
            "test",
        ));
        let configs = (0..num_workers as u64)
            .map(|id| {
                (
                    id,
                    SimpleWorkerConfig {
                        max_num_batched_tokens: Some(max_num_batched_tokens as u64),
                        ..Default::default()
                    },
                )
            })
            .collect();
        let (cfg_tx, cfg_rx) = watch::channel(configs);
        set_worker_capacities_from_configs(&slots, &cfg_rx.borrow());
        let queue = Arc::new(SchedulerQueue::new(
            Arc::clone(&slots),
            cfg_rx,
            profile,
            block_size,
            DefaultWorkerSelector::new(None, "test"),
            None,
            None,
            None,
            None,
            reject_thresholds,
        ));
        (queue, slots, cfg_tx)
    }

    /// What `LocalScheduler::reconcile_worker_configs` does after a config publish: the
    /// registry learns each rank's reported capacity so the projection pass can judge it.
    fn set_worker_capacities_from_configs(
        slots: &ActiveSequencesMultiWorker<NoopSequencePublisher>,
        configs: &HashMap<u64, SimpleWorkerConfig>,
    ) {
        slots.set_worker_capacities(configs.iter().map(|(&id, cfg)| {
            (
                dynamo_slot_tracker::topology::WorkerDpRange::new(
                    id,
                    cfg.data_parallel_start_rank(),
                    cfg.data_parallel_size(),
                ),
                WorkerCapacity::of(cfg),
            )
        }));
    }

    #[allow(clippy::type_complexity)]
    fn make_queue_with_providers(
        num_workers: usize,
        block_size: u32,
        isl: usize,
        overloaded_worker_provider: Option<OverloadedWorkerProvider>,
        available_worker_provider: Option<WorkerAvailabilityProvider>,
    ) -> (
        Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, DefaultWorkerSelector>>,
        Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
    ) {
        let dp_range: HashMap<u64, (u32, u32)> =
            (0..num_workers as u64).map(|id| (id, (0, 1))).collect();
        let slots = Arc::new(ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            block_size as usize,
            dp_range,
            false,
            0,
            "test",
        ));

        let mut configs: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        for id in 0..num_workers as u64 {
            configs.insert(
                id,
                SimpleWorkerConfig {
                    max_num_batched_tokens: Some(isl as u64),
                    ..Default::default()
                },
            );
        }
        let (_cfg_tx, cfg_rx) = watch::channel(configs);

        let selector = DefaultWorkerSelector::new(None, "test");
        let queue = Arc::new(SchedulerQueue::new(
            Arc::clone(&slots),
            cfg_rx,
            PolicyProfile::synthetic(None, RouterQueuePolicy::Fcfs),
            block_size,
            selector,
            None,
            None,
            overloaded_worker_provider,
            available_worker_provider,
            None,
        ));

        (queue, slots)
    }

    /// One worker that holds one 64-token request, scheduled under `profile` by `selector`.
    /// The actor asks `available_workers` for each request's available workers.
    #[allow(clippy::type_complexity)]
    fn make_queue_with_selector_and_availability<Sel>(
        profile: PolicyProfile,
        selector: Sel,
        available_workers: WorkerAvailabilityProvider,
    ) -> (
        Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, Sel>>,
        Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
    )
    where
        Sel: WorkerSelector<SimpleWorkerConfig> + Send + 'static,
    {
        let slots = Arc::new(ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            16,
            HashMap::from([(0, (0, 1))]),
            false,
            0,
            "test",
        ));
        let configs = HashMap::from([(
            0,
            SimpleWorkerConfig {
                max_num_batched_tokens: Some(64),
                ..Default::default()
            },
        )]);
        let (_cfg_tx, cfg_rx) = watch::channel(configs);
        set_worker_capacities_from_configs(&slots, &cfg_rx.borrow());
        let queue = Arc::new(SchedulerQueue::new(
            Arc::clone(&slots),
            cfg_rx,
            profile,
            16,
            selector,
            None,
            None,
            None,
            Some(available_workers),
            None,
        ));
        (queue, slots)
    }

    struct CountingRefresher {
        calls: AtomicUsize,
        last_retain_kv_transfer_chain: AtomicBool,
        response: RefreshedOverlap,
    }

    #[async_trait]
    impl OverlapScoresRefresh for CountingRefresher {
        async fn refresh(
            &self,
            _block_hashes: &[LocalBlockHash],
            retain_kv_transfer_chain: bool,
        ) -> Option<RefreshedOverlap> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.last_retain_kv_transfer_chain
                .store(retain_kv_transfer_chain, Ordering::Relaxed);
            Some(self.response.clone())
        }
    }

    struct BlockingRefresher {
        calls: AtomicUsize,
        started: tokio::sync::Notify,
        release: tokio::sync::Notify,
        response: RefreshedOverlap,
    }

    impl BlockingRefresher {
        fn new(response: RefreshedOverlap) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                started: tokio::sync::Notify::new(),
                release: tokio::sync::Notify::new(),
                response,
            }
        }

        async fn wait_for_calls(&self, target: usize) {
            while self.calls.load(Ordering::Relaxed) < target {
                self.started.notified().await;
            }
        }

        fn release_one(&self) {
            self.release.notify_one();
        }
    }

    #[async_trait]
    impl OverlapScoresRefresh for BlockingRefresher {
        async fn refresh(
            &self,
            _block_hashes: &[LocalBlockHash],
            _retain_kv_transfer_chain: bool,
        ) -> Option<RefreshedOverlap> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.started.notify_one();
            self.release.notified().await;
            Some(self.response.clone())
        }
    }

    #[allow(clippy::type_complexity)]
    fn make_queue_with_refresher(
        num_workers: usize,
        block_size: u32,
        isl: usize,
        threshold_frac: Option<f64>,
        refresher: Arc<CountingRefresher>,
    ) -> (
        Arc<
            SchedulerQueue<
                NoopSequencePublisher,
                SimpleWorkerConfig,
                DefaultWorkerSelector,
                CountingRefresher,
            >,
        >,
        Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
    ) {
        let dp_range: HashMap<u64, (u32, u32)> =
            (0..num_workers as u64).map(|id| (id, (0, 1))).collect();
        let slots = Arc::new(ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            block_size as usize,
            dp_range,
            false,
            0,
            "test",
        ));

        let mut configs: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        for id in 0..num_workers as u64 {
            configs.insert(
                id,
                SimpleWorkerConfig {
                    max_num_batched_tokens: Some(isl as u64),
                    ..Default::default()
                },
            );
        }
        let (_cfg_tx, cfg_rx) = watch::channel(configs);

        let queue = Arc::new(SchedulerQueue::new(
            Arc::clone(&slots),
            cfg_rx,
            PolicyProfile::synthetic(threshold_frac, RouterQueuePolicy::Fcfs),
            block_size,
            DefaultWorkerSelector::new(None, "test"),
            None,
            Some(refresher),
            None,
            None,
            None,
        ));

        (queue, slots)
    }

    #[allow(clippy::type_complexity)]
    fn make_queue_with_blocking_refresher(
        num_workers: usize,
        block_size: u32,
        isl: usize,
        threshold_frac: Option<f64>,
        refresher: Arc<BlockingRefresher>,
        admission_channel_capacity: usize,
    ) -> (
        Arc<
            SchedulerQueue<
                NoopSequencePublisher,
                SimpleWorkerConfig,
                DefaultWorkerSelector,
                BlockingRefresher,
            >,
        >,
        Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
    ) {
        make_queue_with_profile_and_blocking_refresher(
            num_workers,
            block_size,
            isl,
            PolicyProfile::synthetic(threshold_frac, RouterQueuePolicy::Fcfs),
            refresher,
            admission_channel_capacity,
        )
    }

    #[allow(clippy::type_complexity)]
    fn make_queue_with_profile_and_blocking_refresher(
        num_workers: usize,
        block_size: u32,
        isl: usize,
        profile: PolicyProfile,
        refresher: Arc<BlockingRefresher>,
        admission_channel_capacity: usize,
    ) -> (
        Arc<
            SchedulerQueue<
                NoopSequencePublisher,
                SimpleWorkerConfig,
                DefaultWorkerSelector,
                BlockingRefresher,
            >,
        >,
        Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
    ) {
        let dp_range: HashMap<u64, (u32, u32)> =
            (0..num_workers as u64).map(|id| (id, (0, 1))).collect();
        let slots = Arc::new(ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            block_size as usize,
            dp_range,
            false,
            0,
            "test",
        ));

        let mut configs: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        for id in 0..num_workers as u64 {
            configs.insert(
                id,
                SimpleWorkerConfig {
                    max_num_batched_tokens: Some(isl as u64),
                    ..Default::default()
                },
            );
        }
        let (_cfg_tx, cfg_rx) = watch::channel(configs);

        let queue = Arc::new(SchedulerQueue::new_with_capacity(
            Arc::clone(&slots),
            cfg_rx,
            profile,
            block_size,
            DefaultWorkerSelector::new(None, "test"),
            None,
            Some(refresher),
            None,
            None,
            None,
            admission_channel_capacity,
        ));

        (queue, slots)
    }

    fn make_request(
        request_id: &str,
        isl_tokens: usize,
    ) -> (
        SchedulingRequest,
        tokio::sync::oneshot::Receiver<Result<SchedulingResponse, crate::types::KvSchedulerError>>,
    ) {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let req = SchedulingRequest {
            accounting_attribution: Default::default(),
            mode: ScheduleMode::Tracked {
                request_id: request_id.to_string(),
            },
            token_seq: None,
            isl_tokens,
            overlap: OverlapSignals::default(),
            kv_transfer_candidates: None,
            retain_kv_transfer_chain: false,
            worker_loads: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            expected_output_tokens: None,
            affinity_target: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints::default(),
            shared_cache_hits: None,
            resp_tx: Some(tx),
        };
        (req, rx)
    }

    struct CountingPreparedPolicy {
        inner: dynamo_worker_selection::WorkerSelectionPolicy,
        prepared_selections: Arc<AtomicUsize>,
    }

    impl WorkerSelector<SimpleWorkerConfig> for CountingPreparedPolicy {
        fn required_worker_inputs(&self) -> WorkerInputs {
            WorkerSelector::<SimpleWorkerConfig>::required_worker_inputs(&self.inner)
        }

        fn uses_exclusive_affinity_target(&self) -> bool {
            WorkerSelector::<SimpleWorkerConfig>::uses_exclusive_affinity_target(&self.inner)
        }

        fn select_worker(
            &self,
            input: WorkerSelectionInput<'_, SimpleWorkerConfig>,
        ) -> Result<WorkerSelectionResult, KvSchedulerError> {
            self.inner.select_worker(input)
        }

        fn candidate_preparer(&self) -> Option<CandidatePreparer> {
            WorkerSelector::<SimpleWorkerConfig>::candidate_preparer(&self.inner)
        }

        fn select_worker_prepared(
            &self,
            input: WorkerSelectionInput<'_, SimpleWorkerConfig>,
            prepared: Option<PreparedCandidates>,
            trace: Option<SelectionTraceSpec>,
        ) -> Result<(WorkerSelectionResult, Option<SelectionTrace>), KvSchedulerError> {
            if prepared.is_some() {
                self.prepared_selections.fetch_add(1, Ordering::Relaxed);
            }
            self.inner.select_worker_prepared(input, prepared, trace)
        }
        fn select_worker_with_decode_progress(
            &self,
            input: WorkerSelectionInput<'_, SimpleWorkerConfig>,
            prepared: Option<PreparedCandidates>,
            trace: Option<SelectionTraceSpec>,
            progress: Option<&dynamo_slot_tracker::host::DecodeProgressProvider<'_>>,
        ) -> Result<(WorkerSelectionResult, Option<SelectionTrace>), KvSchedulerError> {
            if prepared.is_some() {
                self.prepared_selections.fetch_add(1, Ordering::Relaxed);
            }
            self.inner
                .select_worker_with_decode_progress(input, prepared, trace, progress)
        }
    }

    /// Picks the least-loaded worker; ties go to the lowest worker.
    struct LeastLoaded;

    impl dynamo_worker_selection::WorkerPicker for LeastLoaded {
        fn required_worker_inputs(&self) -> WorkerInputs {
            WorkerInputs::LOAD
        }

        fn pick(
            &self,
            _context: &dynamo_worker_selection::WorkerSelectionContext<'_>,
            input: dynamo_worker_selection::WorkerInputView<'_>,
        ) -> Result<usize, dynamo_worker_selection::WorkerSelectionPolicyError> {
            let (candidates, loads) = (input.candidates(), input.load().unwrap());
            Ok((0..candidates.len())
                .min_by_key(|&i| {
                    let load = loads[i].active_requests() + loads[i].active_prefill_tokens();
                    (load, candidates[i].worker())
                })
                .unwrap())
        }
    }

    struct PickWorker {
        inputs: WorkerInputs,
        target: WorkerWithDpRank,
    }

    impl dynamo_worker_selection::WorkerPicker for PickWorker {
        fn required_worker_inputs(&self) -> WorkerInputs {
            self.inputs
        }

        fn pick(
            &self,
            _context: &dynamo_worker_selection::WorkerSelectionContext<'_>,
            input: dynamo_worker_selection::WorkerInputView<'_>,
        ) -> Result<usize, dynamo_worker_selection::WorkerSelectionPolicyError> {
            Ok(input
                .candidates()
                .iter()
                .position(|candidate| candidate.worker() == self.target)
                .unwrap())
        }
    }

    #[tokio::test]
    async fn selected_worker_decode_blocks_count_active_overlap_without_load_rows() {
        let block_size = 16;
        let target = WorkerWithDpRank::new(0, 0);
        let mut decode_blocks = Vec::new();
        for inputs in [WorkerInputs::LOAD, WorkerInputs::ACTIVE_REQUESTS] {
            let policy = dynamo_worker_selection::WorkerSelectionPolicy::new(
                dynamo_worker_selection::SelectionConfig::default(),
                "test",
                vec![],
                Box::new(PickWorker { inputs, target }),
            );
            let (queue, slots) = make_queue_with_custom_selector(2, block_size, 64, None, policy);
            // Four active blocks on the target, the first three shared with the probe.
            slots
                .add_request(
                    dynamo_slot_tracker::SequenceRequest {
                        request_id: "active".to_string(),
                        token_sequence: Some(vec![1, 2, 3, 9]),
                        track_prefill_tokens: true,
                        expected_output_tokens: None,
                        prefill_load_hint: None,
                        worker: target,
                        lora_name: None,
                    },
                    decay_now(),
                )
                .unwrap();
            let (mut probe, _rx) = make_request("probe", 4 * block_size as usize);
            probe.token_seq = Some(vec![1, 2, 3, 4]);
            let response = queue
                .select_without_admission(probe)
                .await
                .unwrap()
                .response;
            assert_eq!(response.best_worker, target);
            decode_blocks.push(response.potential_decode_blocks);
        }
        // Four active blocks plus the probe's one block not active on the target.
        assert_eq!(decode_blocks, [5, 5]);
    }

    fn counting_policy(
        prepared_selections: &Arc<AtomicUsize>,
        exclusive_affinity: bool,
    ) -> CountingPreparedPolicy {
        CountingPreparedPolicy {
            inner: dynamo_worker_selection::WorkerSelectionPolicy::new(
                dynamo_worker_selection::SelectionConfig::default(),
                "test",
                vec![],
                Box::new(LeastLoaded),
            )
            .with_exclusive_affinity(exclusive_affinity),
            prepared_selections: Arc::clone(prepared_selections),
        }
    }

    /// Two workers behind a counting policy, with availability, overload, and reject
    /// thresholds the test can change.
    struct PreparedQueue {
        queue:
            Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, CountingPreparedPolicy>>,
        slots: Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
        cfg_tx: watch::Sender<HashMap<u64, SimpleWorkerConfig>>,
        available: Arc<StdMutex<Option<Arc<HashSet<WorkerId>>>>>,
        overloaded: Arc<StdMutex<Option<Arc<HashSet<WorkerId>>>>>,
        reject: Arc<StdMutex<BusyThresholds>>,
        prepared_selections: Arc<AtomicUsize>,
    }

    impl PreparedQueue {
        fn prepared_selections(&self) -> usize {
            self.prepared_selections.load(Ordering::Relaxed)
        }
    }

    fn make_prepared_queue() -> PreparedQueue {
        make_prepared_queue_with(1, false)
    }

    /// Two workers of `dp_size` ranks, each rank with a 64-token prefill budget.
    fn make_prepared_queue_with(dp_size: u32, exclusive_affinity: bool) -> PreparedQueue {
        let slots = Arc::new(ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            16,
            (0..2).map(|id| (id, (0, dp_size))).collect(),
            false,
            0,
            "test",
        ));
        let configs = (0..2)
            .map(|id| {
                let config = SimpleWorkerConfig {
                    max_num_batched_tokens: Some(64),
                    data_parallel_size: dp_size,
                    ..Default::default()
                };
                (id, config)
            })
            .collect();
        let (cfg_tx, cfg_rx) = watch::channel(configs);
        set_worker_capacities_from_configs(&slots, &cfg_rx.borrow());
        let available = Arc::new(StdMutex::new(None));
        let available_provider: WorkerAvailabilityProvider = {
            let available = Arc::clone(&available);
            Arc::new(move |_| available.lock().unwrap().clone())
        };
        let overloaded = Arc::new(StdMutex::new(None));
        let overloaded_provider: OverloadedWorkerProvider = {
            let overloaded = Arc::clone(&overloaded);
            Arc::new(move || overloaded.lock().unwrap().clone())
        };
        let reject = Arc::new(StdMutex::new(BusyThresholds::default()));
        let reject_provider: RejectThresholdsProvider = {
            let reject = Arc::clone(&reject);
            Arc::new(move || *reject.lock().unwrap())
        };
        let prepared_selections = Arc::new(AtomicUsize::new(0));
        let queue = Arc::new(SchedulerQueue::new(
            Arc::clone(&slots),
            cfg_rx,
            PolicyProfile::synthetic(None, RouterQueuePolicy::Fcfs),
            16,
            counting_policy(&prepared_selections, exclusive_affinity),
            None,
            None,
            Some(overloaded_provider),
            Some(available_provider),
            Some(reject_provider),
        ));
        PreparedQueue {
            queue,
            slots,
            cfg_tx,
            available,
            overloaded,
            reject,
            prepared_selections,
        }
    }

    #[test]
    fn admission_lifecycle_cleanup_upgrades_to_the_exact_booking() {
        let transfer = AdmissionLifecycleTransfer::new("retryable-request".to_string());
        transfer.arm_pending();
        let booking = SchedulerBookingDescriptor {
            request_id: "retryable-request".to_string(),
            worker: WorkerWithDpRank::new(7, 3),
            attempt_id: AttemptId::for_testing(41),
        };
        transfer.arm_booking(booking.clone());

        let Some(AdmissionLifecycleCleanup::Booking(cleanup)) = transfer.cleanup_target() else {
            panic!("admitted cleanup must carry the exact booking");
        };
        assert_eq!(cleanup, booking);
    }

    /// The handle frees its booking exactly when it is dropped armed; `commit`
    /// hands the booking over untouched and `release` frees it with an ack.
    #[tokio::test(flavor = "multi_thread")]
    async fn booking_handle_frees_only_while_armed() {
        let (queue, slots) = make_queue(1, 16, 512, Some(0.0));

        let armed = book_directly(&slots, "armed");
        drop(queue.booking_handle(armed.clone()));
        wait_freed(&slots, &armed).await;

        let committed = book_directly(&slots, "committed");
        let handle = queue.booking_handle(committed.clone());
        let request_id_ptr = handle.booking.request_id.as_ptr();
        let descriptor = handle.commit();
        assert_eq!(descriptor, committed);
        assert_eq!(descriptor.request_id.as_ptr(), request_id_ptr);

        // Explicit release is immediate. The following actor barrier also
        // proves commit did not enqueue deferred cleanup for its booking.
        let released = book_directly(&slots, "released");
        queue
            .booking_handle(released.clone())
            .release()
            .await
            .expect("acknowledged release");
        queue.update().await;
        assert!(
            !slots.has_booking(&released),
            "release frees before it returns"
        );
        assert!(
            slots.has_booking(&committed),
            "a committed handle leaves the booking to its new owner"
        );
        slots.free(&committed.request_id, decay_now()).unwrap();
        slots.assert_completely_drained(decay_now());
    }

    #[tokio::test]
    async fn classifier_sequence_hashes_preserve_absence_empty_and_order() {
        use dynamo_admission::classifier::{ClassifyFuture, RequestClassifier};

        struct InspectHashes(Option<Vec<u64>>, WorkerWithDpRank);
        impl RequestClassifier for InspectHashes {
            fn classify(
                &mut self,
                request: ClassifyRequest,
                _workers: &dyn WorkerView,
            ) -> ClassifyFuture {
                let expected = self.0.clone();
                let pin = self.1;
                Box::pin(async move {
                    tokio::task::yield_now().await;
                    assert_eq!(request.sequence_hashes(), expected.as_deref());
                    assert_eq!(request.pinned_worker(), Some(pin));
                    Ok(request)
                })
            }
        }

        let profile = PolicyProfile::synthetic(None, dynamo_admission::RouterQueuePolicy::Fcfs);
        let (queue, _slots) = make_queue_with_profile(1, 64, 256, profile);
        let pin = WorkerWithDpRank::new(0, 0);
        for hashes in [None, Some(vec![]), Some(vec![11, 7, 11])] {
            let (mut request, _rx) = make_request("prefix-input", 193);
            request.token_seq = hashes.clone();
            request.pinned_worker = Some(pin);
            let (classified, workers) =
                queue.build_classify_request(&request, Instant::now(), WorkerInputs::NONE);
            assert_eq!(request.token_seq, hashes);
            let mut classifier = InspectHashes(hashes, pin);
            let pending = classifier.classify(classified, &ClassifierRows(workers));
            // The independently pollable future must not borrow actor-owned data.
            request.token_seq = Some(vec![99]);
            request.pinned_worker = None;
            drop(request);
            pending.await.unwrap();
        }
    }

    /// Measures real queue projection with snapshots held in pending classifier futures.
    /// Run in release mode with CLASSIFIER_BENCH_HASHES ("none" or a count) and
    /// CLASSIFIER_BENCH_PENDING set; source setup and cancellation are not timed.
    #[tokio::test]
    #[ignore = "CPU microbenchmark; run explicitly with --release --ignored --nocapture"]
    async fn classifier_input_projection_benchmark() {
        use dynamo_admission::classifier::{ClassifyFuture, RequestClassifier};
        use std::hint::black_box;
        use std::task::Context;
        use std::time::Instant as WallClock;

        struct Park;
        impl RequestClassifier for Park {
            fn classify(
                &mut self,
                request: ClassifyRequest,
                _workers: &dyn WorkerView,
            ) -> ClassifyFuture {
                Box::pin(async move {
                    std::future::pending::<()>().await;
                    Ok(black_box(request))
                })
            }
        }

        let hash_count = std::env::var("CLASSIFIER_BENCH_HASHES").unwrap();
        let hashes = if hash_count == "none" {
            None
        } else {
            Some(hash_count.parse::<usize>().unwrap())
        };
        let retained: usize = std::env::var("CLASSIFIER_BENCH_PENDING")
            .unwrap()
            .parse()
            .unwrap();
        assert!(retained > 0 && retained <= 1024);
        assert!(hashes.unwrap_or(0) <= 62_500);
        let profile = PolicyProfile::synthetic(None, dynamo_admission::RouterQueuePolicy::Fcfs);
        let (queue, _slots) = make_queue_with_profile(1, 16, 1_000_000, profile);
        let sources: Vec<_> = (0..retained)
            .map(|i| {
                let (mut request, rx) = make_request(
                    &format!("copy-bench-{i}"),
                    hashes.unwrap_or(0).saturating_mul(16).max(1),
                );
                request.token_seq = hashes.map(|n| (0..n as u64).collect());
                (request, rx)
            })
            .collect();
        let mut classifier = Park;
        let mut futures = Vec::with_capacity(retained);
        let mut cx = Context::from_waker(futures_util::task::noop_waker_ref());
        let ingress_at = Instant::now();
        let repeats = (1024 / retained).max(1);
        let mut samples = Vec::with_capacity(40);
        for sample in 0..45 {
            let mut elapsed = std::time::Duration::ZERO;
            for _ in 0..repeats {
                let start = WallClock::now();
                for (request, _) in &sources {
                    let (input, workers) = queue.build_classify_request(
                        black_box(request),
                        ingress_at,
                        WorkerInputs::NONE,
                    );
                    let mut future = classifier.classify(input, &ClassifierRows(workers));
                    assert!(future.as_mut().poll(&mut cx).is_pending());
                    futures.push(future);
                }
                elapsed += start.elapsed();
                // Keep every snapshot live until the batch ends; cancellation is excluded.
                black_box(&futures);
                futures.clear();
            }
            if sample >= 5 {
                samples.push(elapsed.as_secs_f64() * 1e9 / (retained * repeats) as f64);
            }
        }
        println!(
            "CLASSIFIER_BENCH {}",
            serde_json::json!({
                "hashes": hashes,
                "pending": retained,
                "repeats_per_sample": repeats,
                "classify_request_bytes": std::mem::size_of::<ClassifyRequest>(),
                "ns_per_request_samples": samples,
            })
        );
    }

    #[tokio::test]
    async fn classifier_targets_preserve_pins_and_caller_eligibility() {
        let (queue, _slots) = make_queue(2, 16, 64, None);
        let (mut request, _rx) = make_request("targeted", 48);
        let worker = WorkerWithDpRank::new(1, 0);
        let pin = WorkerWithDpRank::new(0, 0);
        let now = Instant::now();

        let mut classified = queue
            .build_classify_request(&request, now, WorkerInputs::NONE)
            .0;
        classified.set_worker_selection_target(worker);
        queue
            .validate_classification(&mut request, classified, now)
            .unwrap();
        assert_eq!(request.affinity_target, Some(worker.into()));

        let mut classified = queue
            .build_classify_request(&request, now, WorkerInputs::NONE)
            .0;
        classified.clear_worker_selection_target();
        queue
            .validate_classification(&mut request, classified, now)
            .unwrap();
        assert!(request.affinity_target.is_none());

        request.pinned_worker = Some(pin);
        let mut classified = queue
            .build_classify_request(&request, now, WorkerInputs::NONE)
            .0;
        assert_eq!(classified.pinned_worker(), Some(pin));
        classified.set_worker_selection_target(worker);
        queue
            .validate_classification(&mut request, classified, now)
            .unwrap();
        assert_eq!(request.pinned_worker, Some(pin));
        assert!(request.affinity_target.is_none());
        request.pinned_worker = None;
        request.allowed_worker_ids = Some(HashSet::from([0]));
        let mut classified = queue
            .build_classify_request(&request, now, WorkerInputs::NONE)
            .0;
        classified.set_worker_selection_target(worker);
        queue
            .validate_classification(&mut request, classified, now)
            .unwrap();
        assert_eq!(request.allowed_worker_ids, Some(HashSet::from([0])));
        // The selector still applies the allowlist after a classifier supplies a target.
        let result = queue.select_without_admission(request).await.unwrap();
        assert_eq!(result.response.best_worker, pin);
    }

    #[tokio::test]
    async fn expired_deadline_is_rejected_at_enqueue() {
        let (queue, _slots) = make_queue(1, 16, 64, Some(0.0));
        let (probe, _probe_rx) = make_request("advisory", 64);
        queue.select_without_admission(probe).await.unwrap();
        let (request, response_rx) = make_request("expired-on-arrival", 64);
        queue
            .enqueue_with_due_at_for_test(request, Instant::now())
            .await;
        assert!(matches!(
            response_rx.await.unwrap(),
            Err(KvSchedulerError::DeadlineExceeded)
        ));
        assert_eq!(queue.pending_count(), 0);
        queue.update().await;
        let stats = queue.class_queue_stats(0).unwrap();
        assert_eq!(stats.received_total, 1);
        assert_eq!(stats.rejected_due_time_passed_total, 1);
    }

    #[tokio::test(start_paused = true)]
    async fn parked_request_expires_via_actor_timer() {
        let (queue, _slots) = make_queue(1, 16, 64, Some(0.0));
        let (active, active_rx) = make_request("active", 64);
        queue.enqueue(active).await;
        active_rx.await.unwrap().unwrap();

        // The worker is busy, so the deadline-bearing request parks; the actor
        // timer must fire at due_at and reject it without any further command.
        let (parked, parked_rx) = make_request("parked-deadline", 64);
        queue
            .enqueue_with_due_at_for_test(parked, Instant::now() + Duration::from_secs(5))
            .await;
        assert_eq!(queue.pending_count(), 1);

        assert!(matches!(
            parked_rx.await.unwrap(),
            Err(KvSchedulerError::DeadlineExceeded)
        ));
        assert_eq!(queue.pending_count(), 0);
        queue.update().await;
        let stats = queue.class_queue_stats(0).unwrap();
        assert_eq!(stats.received_total, 2);
        assert_eq!(stats.rejected_due_time_passed_total, 1);
    }

    #[tokio::test(start_paused = true)]
    async fn deadline_expiry_during_refresh_is_counted_once() {
        let refresher = Arc::new(BlockingRefresher::new(RefreshedOverlap::default()));
        let (queue, slots) = make_queue_with_blocking_refresher(
            1,
            16,
            64,
            Some(0.0),
            Arc::clone(&refresher),
            ADMISSION_CHANNEL_CAPACITY,
        );
        let (active, active_rx) = make_request("active", 64);
        queue.enqueue(active).await;
        active_rx.await.unwrap().unwrap();

        let (request, response_rx) = make_request("expires-during-refresh", 64);
        let mut queue_metadata = queue
            .default_queue_metadata(&request, Instant::now())
            .unwrap();
        queue_metadata.due_at = Some(Instant::now() + Duration::from_secs(30));
        let (ack_tx, ack_rx) = oneshot::channel();
        queue
            .admission_tx
            .send(AdmissionCommand::Enqueue {
                request,
                attempt_tx: None,
                block_hashes: Some(vec![LocalBlockHash(42)]),
                prepared: None,
                queue_metadata,
                lease: None,
                ack_tx,
            })
            .await
            .unwrap();
        ack_rx.await.unwrap();
        slots
            .mark_prefill_completed(&"active".to_string(), decay_now())
            .unwrap();
        slots.free(&"active".to_string(), decay_now()).unwrap();
        tokio::time::advance(Duration::from_secs(11)).await;
        let updating_queue = Arc::clone(&queue);
        let update = tokio::spawn(async move { updating_queue.update().await });
        refresher.wait_for_calls(1).await;
        tokio::time::advance(Duration::from_secs(20)).await;
        refresher.release_one();
        update.await.unwrap();
        assert!(matches!(
            response_rx.await.unwrap(),
            Err(KvSchedulerError::DeadlineExceeded)
        ));
        queue.update().await;
        assert_eq!(queue.pending_count(), 0);
        let stats = queue.class_queue_stats(0).unwrap();
        assert_eq!(stats.received_total, 2);
        assert_eq!(stats.rejected_due_time_passed_total, 1);
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn stalled_refresh_sweeps_expired_entries_without_refreshing_them() {
        let block_size = 16;
        let isl = 64;
        let refresher = Arc::new(BlockingRefresher::new(RefreshedOverlap::default()));
        let (queue, slots) = make_queue_with_blocking_refresher(
            1,
            block_size,
            isl,
            Some(0.0),
            Arc::clone(&refresher),
            ADMISSION_CHANNEL_CAPACITY,
        );

        let (active, active_rx) = make_request("active", isl);
        queue.enqueue(active).await;
        active_rx.await.unwrap().unwrap();

        // Two deadline-bearing entries with the same due time: FCFS pops the
        // first into a refresh that stalls past both deadlines.
        let ingress_at = Instant::now();
        let due_at = ingress_at + Duration::from_secs(12);
        let mut receivers = Vec::new();
        for (name, hash) in [("stalled", 41), ("swept", 42)] {
            let (queued, queued_rx) = make_request(name, isl);
            let mut classified = queue
                .build_classify_request(&queued, ingress_at, WorkerInputs::NONE)
                .0;
            classified.set_due_at(due_at);
            queue
                .enqueue_admitted_with_block_hashes_and_lease(
                    queued,
                    Some(vec![LocalBlockHash(hash)]),
                    None,
                    None,
                    Some(classified),
                    ingress_at,
                )
                .await;
            receivers.push(queued_rx);
        }
        assert_eq!(queue.pending_count(), 2);

        slots
            .mark_prefill_completed(&"active".to_string(), decay_now())
            .unwrap();
        slots.free(&"active".to_string(), decay_now()).unwrap();
        tokio::time::advance(Duration::from_secs(11)).await;

        let update = {
            let queue = Arc::clone(&queue);
            tokio::spawn(async move { queue.update().await })
        };
        refresher.wait_for_calls(1).await;
        tokio::time::advance(Duration::from_secs(2)).await;
        refresher.release_one();
        // Pre-arm a second release so a regression (popping the expired entry
        // into its own refresh) fails the call-count assertion below instead
        // of hanging the test.
        refresher.release_one();
        update.await.unwrap();

        for queued_rx in receivers {
            assert!(matches!(
                queued_rx.await.unwrap(),
                Err(KvSchedulerError::DeadlineExceeded)
            ));
        }
        assert_eq!(queue.pending_count(), 0);
        assert_eq!(refresher.calls.load(Ordering::Relaxed), 1);
        slots.assert_completely_drained(decay_now());
    }

    // The caller drives this actor directly. Its hints are separate from the
    // idle actor spawned by make_queue, so tests control notification ordering.
    fn actor_for_test(
        queue: &SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, DefaultWorkerSelector>,
        slots: &Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
    ) -> SchedulerQueueActor<
        NoopSequencePublisher,
        SimpleWorkerConfig,
        DefaultWorkerSelector,
        NoopOverlapScoresRefresh,
    > {
        actor_with_selector_for_test(queue, slots, DefaultWorkerSelector::new(None, "test"))
    }

    fn actor_with_selector_for_test<
        QueueSel: WorkerSelector<SimpleWorkerConfig>,
        Sel: WorkerSelector<SimpleWorkerConfig>,
    >(
        queue: &SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, QueueSel>,
        slots: &Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
        selector: Sel,
    ) -> SchedulerQueueActor<NoopSequencePublisher, SimpleWorkerConfig, Sel, NoopOverlapScoresRefresh>
    {
        SchedulerQueueActor {
            // Commands go to the queue's own actor; this one sees a closed channel.
            commands: CommandInbox::new(mpsc::channel(1).1),
            pending: PolicyQueue::with_class_budget_expiry(&queue.profile),
            class_timers: ClassTimers::new(queue.profile.classes().len()),
            cleanup: Arc::clone(&queue.cleanup),
            capacity_updates: None,
            capacity_workers: FxHashSet::default(),
            profile: Arc::clone(&queue.profile),
            pending_count: Arc::clone(&queue.pending_count),
            pending_isl_tokens: Arc::clone(&queue.pending_isl_tokens),
            class_counters: Arc::clone(&queue.class_counters),
            slots: Arc::clone(slots),
            workers_with_configs: queue.workers_with_configs.clone(),
            workers_generation: Arc::clone(&queue.workers_generation),
            candidate_preparer: Arc::clone(&queue.candidate_preparer),
            turn: 0,
            turn_prepared: None,
            projected_loads: FxHashMap::with_capacity_and_hasher(8, Default::default()),
            full_ranks: FullRankCounts::default(),
            block_size: 16,
            selector,
            prefill_load_estimator: None,
            overlap_scores_refresh: None::<Arc<NoopOverlapScoresRefresh>>,
            overlap_refresh_after: None,
            overloaded_worker_provider: None,
            available_worker_ids: None,
            non_max_overlap_selection_observer: Arc::clone(
                &queue.non_max_overlap_selection_observer,
            ),
            selection_trace: Arc::clone(&queue.selection_trace),
            resident_blocks: Arc::clone(&queue.resident_blocks),
            reject_thresholds: queue.reject_thresholds.clone(),
        }
    }

    async fn enqueue_prepared(
        queue: &SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, CountingPreparedPolicy>,
        request: SchedulingRequest,
        prepared: Option<Box<PreparedAdmission>>,
    ) {
        let queue_metadata = queue
            .default_queue_metadata(&request, Instant::now())
            .unwrap();
        let (ack_tx, ack_rx) = oneshot::channel();
        queue
            .admission_tx
            .send(AdmissionCommand::Enqueue {
                request,
                attempt_tx: None,
                block_hashes: None,
                prepared,
                queue_metadata,
                lease: None,
                ack_tx,
            })
            .await
            .unwrap();
        ack_rx.await.unwrap();
    }

    #[tokio::test]
    async fn prepared_rows_see_bookings_made_after_preparation() {
        let prepared_queue = make_prepared_queue();
        let queue = &prepared_queue.queue;
        let (later, later_rx) = make_request("later", 64);
        let prepared = queue.prepare_candidates(&later);
        assert!(prepared.is_some());

        let (earlier, earlier_rx) = make_request("earlier", 64);
        queue.enqueue(earlier).await;
        let earlier = earlier_rx.await.unwrap().unwrap().best_worker;
        assert_eq!(earlier, WorkerWithDpRank::new(0, 0));

        // Load inputs are read at selection, so the earlier booking moves this request.
        enqueue_prepared(queue, later, prepared).await;
        let later = later_rx.await.unwrap().unwrap().best_worker;
        assert_eq!(later, WorkerWithDpRank::new(1, 0));
        assert_eq!(prepared_queue.prepared_selections(), 2);
    }

    #[tokio::test]
    async fn prepared_rows_read_decode_progress_at_actor_selection() {
        use dynamo_slot_tracker::host::DecodeProgressUpdate;
        use dynamo_worker_selection::{
            WorkerInputView, WorkerPicker, WorkerSelectionContext, WorkerSelectionPolicy,
        };

        struct ProgressPicker;
        impl WorkerPicker for ProgressPicker {
            fn required_worker_inputs(&self) -> WorkerInputs {
                WorkerInputs::DECODE_PROGRESS
            }

            fn pick(
                &self,
                context: &WorkerSelectionContext<'_>,
                input: WorkerInputView<'_>,
            ) -> Result<usize, crate::WorkerSelectionPolicyError> {
                let progress = &context.decode_progress().unwrap().entries()[0];
                assert_eq!(progress.remaining_token_budget(), 48);
                Ok(input
                    .candidates()
                    .iter()
                    .position(|row| row.worker().worker_id == 1)
                    .unwrap())
            }
        }
        let prepared_selections = Arc::new(AtomicUsize::new(0));
        let policy = || CountingPreparedPolicy {
            inner: WorkerSelectionPolicy::new(
                dynamo_worker_selection::SelectionConfig::default(),
                "test",
                vec![],
                Box::new(ProgressPicker),
            ),
            prepared_selections: Arc::clone(&prepared_selections),
        };
        let (queue, slots) = make_queue_with_custom_selector(2, 16, 64, None, policy());
        let mut actor = actor_with_selector_for_test(&queue, &slots, policy());
        let request_id = "background".to_string();
        let worker = WorkerWithDpRank::new(0, 0);
        let attempt = slots
            .add_request_if_registered_admitted(
                dynamo_slot_tracker::host::SequenceRequest {
                    request_id: request_id.clone(),
                    token_sequence: None,
                    expected_output_tokens: None,
                    track_prefill_tokens: false,
                    prefill_load_hint: None,
                    worker,
                    lora_name: None,
                },
                Instant::now(),
            )
            .unwrap();
        slots
            .mark_prefill_completed(&request_id, Instant::now())
            .unwrap();
        slots
            .update_output_if_booking(
                &request_id,
                worker,
                attempt,
                0,
                None,
                Some(DecodeProgressUpdate::observed(128, 8)),
            )
            .unwrap();
        let (request, mut rx) = make_request("prepared-progress", 16);
        let prepared = queue.prepare_candidates(&request);
        assert!(prepared.is_some());
        slots
            .update_output_if_booking(
                &request_id,
                worker,
                attempt,
                0,
                None,
                Some(DecodeProgressUpdate::observed(128, 80)),
            )
            .unwrap();
        actor.refresh_workers_generation();
        assert_eq!(
            actor.with_projected_loads(request, Instant::now(), |actor, request| {
                actor.admit_one_projected(AdmissionContext::untracked(0, None), request, prepared)
            }),
            Admission::Booked
        );
        assert_eq!(
            rx.try_recv().unwrap().unwrap().best_worker,
            WorkerWithDpRank::new(1, 0)
        );
        assert_eq!(prepared_selections.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn prepared_rows_are_rebuilt_when_selection_inputs_change() {
        for change in ["configuration", "availability", "overload"] {
            let prepared_queue = make_prepared_queue();
            let (request, rx) = make_request("request", 64);
            let prepared = prepared_queue.queue.prepare_candidates(&request);
            // Each change rules worker 0 out after the rows were prepared.
            match change {
                "configuration" => prepared_queue.cfg_tx.send_modify(|configs| {
                    configs.remove(&0);
                }),
                "availability" => {
                    *prepared_queue.available.lock().unwrap() = Some(Arc::new(HashSet::from([1])))
                }
                _ => {
                    *prepared_queue.overloaded.lock().unwrap() = Some(Arc::new(HashSet::from([0])))
                }
            }

            enqueue_prepared(&prepared_queue.queue, request, prepared).await;
            let selected = rx.await.unwrap().unwrap().best_worker;
            assert_eq!(selected, WorkerWithDpRank::new(1, 0), "{change}");
            assert_eq!(prepared_queue.prepared_selections(), 0, "{change}");
        }
    }

    /// Reject thresholds under which a rank past 32 tokens, half its prefill budget, is full.
    fn half_budget_reject() -> BusyThresholds {
        BusyThresholds {
            prefill_busy_threshold_frac: Some(0.5),
            ..Default::default()
        }
    }

    /// Book `tokens` of prefill on `worker`, as a pinned request.
    async fn hold_prefill(
        queue: &SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, CountingPreparedPolicy>,
        request_id: &str,
        worker: WorkerWithDpRank,
        tokens: usize,
    ) {
        let (mut held, held_rx) = make_request(request_id, tokens);
        held.pinned_worker = Some(worker);
        queue.enqueue(held).await;
        held_rx.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn prepared_rows_drop_only_workers_whose_every_rank_is_full() {
        let prepared_queue = make_prepared_queue_with(2, false);
        let queue = &prepared_queue.queue;
        *prepared_queue.reject.lock().unwrap() = half_budget_reject();
        let select = |request_id: &'static str| async move {
            let (request, rx) = make_request(request_id, 16);
            let prepared = queue.prepare_candidates(&request);
            enqueue_prepared(queue, request, prepared).await;
            rx.await.unwrap()
        };

        // One full rank leaves worker 0 eligible: its other rank is the least loaded.
        hold_prefill(queue, "held-0-0", WorkerWithDpRank::new(0, 0), 48).await;
        let selected = select("one-rank-full").await.unwrap().best_worker;
        assert_eq!(selected, WorkerWithDpRank::new(0, 1));

        // Both ranks full: worker 0's rows are dropped.
        hold_prefill(queue, "held-0-1", WorkerWithDpRank::new(0, 1), 48).await;
        let selected = select("worker-full").await.unwrap().best_worker;
        assert_eq!(selected, WorkerWithDpRank::new(1, 0));

        hold_prefill(queue, "held-1-0", WorkerWithDpRank::new(1, 0), 48).await;
        hold_prefill(queue, "held-1-1", WorkerWithDpRank::new(1, 1), 48).await;
        assert!(matches!(
            select("all-full").await,
            Err(KvSchedulerError::AllEligibleWorkersOverloaded)
        ));
        assert_eq!(prepared_queue.prepared_selections(), 3);
    }

    #[tokio::test]
    async fn prepared_rows_are_rebuilt_for_an_exclusive_affinity_target_once_a_rank_is_full() {
        let prepared_queue = make_prepared_queue_with(1, true);
        let queue = &prepared_queue.queue;
        *prepared_queue.reject.lock().unwrap() = half_budget_reject();
        hold_prefill(queue, "held", WorkerWithDpRank::new(0, 0), 48).await;

        // Preparation held only the full target's rows; selection, which sees the target
        // full, routes like any request.
        let (mut request, rx) = make_request("request", 16);
        request.affinity_target = Some(dynamo_session_affinity::WorkerAffinityTarget::new(0, None));
        let prepared = queue.prepare_candidates(&request);
        assert!(prepared.is_some());
        enqueue_prepared(queue, request, prepared).await;
        assert_eq!(
            rx.await.unwrap().unwrap().best_worker,
            WorkerWithDpRank::new(1, 0)
        );
        assert_eq!(prepared_queue.prepared_selections(), 0);
    }

    #[tokio::test]
    async fn prepared_rows_recount_the_best_cached_prefix_without_full_workers() {
        let prepared_queue = make_prepared_queue();
        let queue = &prepared_queue.queue;
        *prepared_queue.reject.lock().unwrap() = half_budget_reject();
        hold_prefill(queue, "held", WorkerWithDpRank::new(0, 0), 48).await;

        // Worker 0 caches the longest prefix but is full, so only worker 1's prefix counts.
        let (mut request, rx) = make_request("request", 64);
        request.overlap.tier_overlap_blocks.device = FxHashMap::from_iter([
            (WorkerWithDpRank::new(0, 0), 3),
            (WorkerWithDpRank::new(1, 0), 1),
        ]);
        let prepared = queue.prepare_candidates(&request);
        enqueue_prepared(queue, request, prepared).await;
        let response = rx.await.unwrap().unwrap();
        assert_eq!(response.best_worker, WorkerWithDpRank::new(1, 0));
        assert_eq!(response.max_raw_cached_tokens, Some(16));
        assert_eq!(prepared_queue.prepared_selections(), 1);
    }

    #[tokio::test]
    async fn prepared_rows_recheck_config_under_selection_borrow() {
        let prepared_queue = make_prepared_queue();
        let mut actor = actor_with_selector_for_test(
            &prepared_queue.queue,
            &prepared_queue.slots,
            counting_policy(&prepared_queue.prepared_selections, false),
        );
        let (request, mut rx) = make_request("request", 64);
        let prepared = prepared_queue.queue.prepare_candidates(&request);
        assert!(prepared.is_some());

        // Worker 0 leaves after the actor last advanced the generation.
        actor.refresh_workers_generation();
        prepared_queue.cfg_tx.send_modify(|configs| {
            configs.remove(&0);
        });
        assert!(
            actor.with_projected_loads(request, Instant::now(), |actor, request| {
                actor.admit_one_projected(AdmissionContext::untracked(0, None), request, prepared)
                    == Admission::Booked
            })
        );
        let selected = rx.try_recv().unwrap().unwrap().best_worker;
        assert_eq!(selected, WorkerWithDpRank::new(1, 0));
        assert_eq!(prepared_queue.prepared_selections(), 0);
    }

    #[tokio::test]
    async fn prepared_rows_are_built_after_a_channel_slot() {
        let prepared_queue = make_prepared_queue();
        let queue = &prepared_queue.queue;
        // Hold every slot, so the enqueue below waits before preparing.
        let held = queue
            .admission_tx
            .reserve_many(queue.admission_tx.max_capacity())
            .await
            .unwrap();
        let (request, rx) = make_request("request", 64);
        let enqueue = tokio::spawn({
            let queue = Arc::clone(queue);
            async move { queue.enqueue(request).await }
        });
        tokio::task::yield_now().await;

        // Worker 0 leaves while the caller waits; rows built after the wait see it.
        *prepared_queue.available.lock().unwrap() = Some(Arc::new(HashSet::from([1])));
        drop(held);
        enqueue.await.unwrap();
        let selected = rx.await.unwrap().unwrap().best_worker;
        assert_eq!(selected, WorkerWithDpRank::new(1, 0));
        assert_eq!(prepared_queue.prepared_selections(), 1);
    }

    #[tokio::test]
    async fn enqueue_rechecks_capacity_before_later_shared_admission() {
        for overflow in [false, true] {
            let (queue, slots) = make_queue(1, 16, 64, Some(0.0));
            let mut actor = actor_for_test(&queue, &slots);
            let updates = Arc::new(CapacityUpdates::new());
            actor.capacity_updates = Some(Arc::clone(&updates));
            let worker = WorkerWithDpRank::new(0, 0);

            let (active, mut active_rx) = make_request("active", 64);
            let metadata = queue
                .default_queue_metadata(&active, Instant::now())
                .unwrap();
            assert!(actor.handle_enqueue(active, None, None, None, None, metadata));
            actor.handle_enqueued(None).await;
            assert_eq!(active_rx.try_recv().unwrap().unwrap().best_worker, worker);

            let (mut older, mut older_rx) = make_request("older-pinned", 64);
            older.pinned_worker = Some(worker);
            let metadata = queue
                .default_queue_metadata(&older, Instant::now())
                .unwrap();
            assert!(actor.handle_enqueue(older, None, None, None, None, metadata));
            actor.handle_enqueued(None).await;
            assert!(older_rx.try_recv().is_err());

            slots.free(&"active".to_owned(), Instant::now()).unwrap();
            for _ in 0..if overflow { 257 } else { 1 } {
                updates.record(Some(worker));
            }
            let (later, mut later_rx) = make_request("later-shared", 64);
            let metadata = queue
                .default_queue_metadata(&later, Instant::now())
                .unwrap();
            assert!(actor.handle_enqueue(later, None, None, None, None, metadata));
            // Force enqueue processing before the capacity select branch.
            actor.handle_enqueued(None).await;
            assert_eq!(older_rx.try_recv().unwrap().unwrap().best_worker, worker);
            assert!(later_rx.try_recv().is_err());
            assert_eq!(queue.pending_count(), 1);

            slots
                .free(&"older-pinned".to_owned(), Instant::now())
                .unwrap();
            updates.record(Some(worker));
            actor.handle_capacity_updates(false).await;
            assert_eq!(later_rx.try_recv().unwrap().unwrap().best_worker, worker);
            slots
                .free(&"later-shared".to_owned(), Instant::now())
                .unwrap();
            slots.assert_completely_drained(Instant::now());
        }
    }

    #[tokio::test]
    async fn cleanup_progresses_with_admission_commands_and_capacity_wakes() {
        let (queue, slots) = make_queue(1, 16, 64, Some(0.0));
        let booking = book_directly(&slots, "abandoned");
        let mut permits = queue
            .admission_tx
            .reserve_many(queue.admission_tx.max_capacity())
            .await
            .unwrap();
        // No permit is available for the destructor's explicit Cleanup command.
        drop(queue.booking_handle(booking.clone()));
        assert!(slots.has_booking(&booking));
        let mut responses = Vec::new();
        for _ in 0..ADMISSION_CHANNEL_CAPACITY {
            let (request, _rx) = make_request("probe", 64);
            let (resp_tx, resp_rx) = oneshot::channel();
            permits
                .next()
                .unwrap()
                .send(AdmissionCommand::SelectWithoutAdmission { request, resp_tx });
            responses.push(resp_rx);
            queue.capacity_changed(Some(booking.worker));
        }
        tokio::time::timeout(Duration::from_secs(2), async {
            for response in responses {
                response.await.unwrap().unwrap();
            }
        })
        .await
        .expect("admission commands must progress with capacity wakes");
        assert!(!slots.has_booking(&booking));
        drop(permits);
        slots.assert_completely_drained(Instant::now());
    }

    #[tokio::test]
    async fn runtime_control_drains_cleanup_when_the_command_channel_was_full() {
        let (queue, slots) = make_queue(1, 16, 64, Some(0.0));
        let booking = book_directly(&slots, "abandoned");
        let mut permits = queue
            .admission_tx
            .reserve_many(queue.admission_tx.max_capacity())
            .await
            .unwrap();
        drop(queue.booking_handle(booking.clone()));
        assert!(slots.has_booking(&booking));
        let (response, receiver) = oneshot::channel();
        permits
            .next()
            .unwrap()
            .send(AdmissionCommand::PolicyConfig {
                update: None,
                response,
            });
        let result = tokio::time::timeout(Duration::from_secs(2), receiver)
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(result, Err(RuntimePolicyError::Unsupported)));
        assert!(!slots.has_booking(&booking));
        drop(permits);
        slots.assert_completely_drained(Instant::now());
    }

    #[tokio::test]
    async fn projection_storage_survives_admission_and_errors_without_stale_entries() {
        let (queue, slots) = make_queue(1, 16, 64, None);
        let mut actor = actor_for_test(&queue, &slots);
        let worker = WorkerWithDpRank::new(0, 0);
        let stale = WorkerWithDpRank::new(999, 0);
        let capacity = actor.projected_loads.capacity();
        // The second admission refreshes load after the first booking. The
        // duplicate then exercises booking failure with the same allocation.
        for (id, succeeds) in [("first", true), ("second", true), ("first", false)] {
            actor
                .projected_loads
                .insert(stale, WorkerLoadProjection::default());
            let now = Instant::now();
            let expected = slots.project_worker_loads(None, now);
            let (request, mut rx) = make_request(id, 64);
            let storage = actor.with_projected_loads(request, now, |actor, request| {
                assert_eq!(request.worker_loads, expected);
                assert!(!request.worker_loads.contains_key(&stale));
                let storage = request.worker_loads.get(&worker).unwrap() as *const _;
                assert_eq!(
                    actor.admit_one_projected(AdmissionContext::untracked(0, None), request, None)
                        == Admission::Booked,
                    succeeds
                );
                assert_eq!(rx.try_recv().unwrap().is_ok(), succeeds);
                assert_eq!(request.worker_loads, expected);
                storage
            });
            assert_eq!(actor.projected_loads.capacity(), capacity);
            assert_eq!(
                actor.projected_loads.get(&worker).unwrap() as *const _,
                storage
            );
        }
        let (mut request, _rx) = make_request("missing-worker", 64);
        request.pinned_worker = Some(stale);
        assert!(
            actor
                .select_without_admission_inner(request, Instant::now())
                .is_err()
        );
        assert_eq!(actor.projected_loads.capacity(), capacity);
        for id in ["first", "second"] {
            slots.free(&id.to_owned(), Instant::now()).unwrap();
        }
        slots.assert_completely_drained(Instant::now());
    }

    #[test]
    fn non_max_overlap_selection_ignores_ties_pins_and_ineligible_workers() {
        let workers = HashMap::from([
            (0, SimpleWorkerConfig::default()),
            (1, SimpleWorkerConfig::default()),
        ]);
        let worker0 = WorkerWithDpRank::new(0, 0);
        let worker1 = WorkerWithDpRank::new(1, 0);
        let (mut request, _rx) = make_request("locality-exclusions", 64);
        request
            .overlap
            .effective_overlap_blocks
            .extend([(worker0, 8.0), (worker1, 8.0)]);
        assert!(
            non_max_overlap_selection(&workers, &request, request.eligibility(), worker1, 8.0)
                .is_none()
        );

        request
            .overlap
            .effective_overlap_blocks
            .insert(worker1, 2.0);
        request.pinned_worker = Some(worker1);
        assert!(
            non_max_overlap_selection(&workers, &request, request.eligibility(), worker1, 2.0)
                .is_none()
        );

        request.pinned_worker = None;
        request.allowed_worker_ids = Some(HashSet::from([worker1.worker_id]));
        assert!(
            non_max_overlap_selection(&workers, &request, request.eligibility(), worker1, 2.0)
                .is_none()
        );
    }

    #[tokio::test]
    async fn scheduler_observer_receives_non_max_overlap_selection() {
        let (queue, _slots) = make_queue_with_custom_selector(
            2,
            16,
            64,
            None,
            MinDecodeSelector { rendezvous: None },
        );
        let worker0 = WorkerWithDpRank::new(0, 0);
        let worker1 = WorkerWithDpRank::new(1, 0);
        let (observer_tx, mut observer_rx) = tokio::sync::mpsc::unbounded_channel();
        assert!(queue.set_non_max_overlap_selection_observer(Arc::new(
            move |request_id, event| {
                observer_tx
                    .send((request_id.to_string(), event))
                    .expect("observer receiver should remain open");
            }
        )));
        let (mut request, response_rx) = make_request("locality-response", 64);
        request
            .overlap
            .effective_overlap_blocks
            .extend([(worker0, 1.0), (worker1, 4.0)]);

        queue.enqueue(request).await;
        let response = response_rx.await.unwrap().unwrap();

        assert_eq!(response.best_worker, worker0);
        let event = tokio::time::timeout(Duration::from_secs(1), observer_rx.recv())
            .await
            .expect("observer did not run")
            .expect("observer channel closed");
        assert_eq!(event.1.overlap_blocks_lost(), 3.0);
        assert_eq!(
            event,
            (
                "locality-response".to_string(),
                NonMaxOverlapSelection {
                    selected_worker: worker0,
                    highest_overlap_worker: worker1,
                    highest_overlap_blocks: 4.0,
                    selected_overlap_blocks: 1.0,
                },
            )
        );
    }

    #[tokio::test]
    async fn affinity_pinned_selection_reports_non_max_overlap() {
        let (queue, _slots) = make_queue(2, 16, 64, None);
        let worker0 = WorkerWithDpRank::new(0, 0);
        let worker1 = WorkerWithDpRank::new(1, 0);
        let (observer_tx, mut observer_rx) = tokio::sync::mpsc::unbounded_channel();
        assert!(queue.set_non_max_overlap_selection_observer(Arc::new(
            move |request_id, event| {
                observer_tx
                    .send((request_id.to_string(), event))
                    .expect("observer receiver should remain open");
            }
        )));
        let pinned_request = |request_id: &str, affinity: bool| {
            let (mut request, response_rx) = make_request(request_id, 64);
            request
                .overlap
                .effective_overlap_blocks
                .extend([(worker0, 8.0), (worker1, 2.0)]);
            request.pinned_worker = Some(worker1);
            request.affinity_target = affinity.then(|| worker1.into());
            (request, response_rx)
        };

        // An explicit pin is the caller's choice, not a router decision: no event.
        let (request, response_rx) = pinned_request("explicit-pin", false);
        queue.enqueue(request).await;
        assert_eq!(response_rx.await.unwrap().unwrap().best_worker, worker1);

        let (request, response_rx) = pinned_request("affinity-pin", true);
        queue.enqueue(request).await;
        assert_eq!(response_rx.await.unwrap().unwrap().best_worker, worker1);

        let event = tokio::time::timeout(Duration::from_secs(1), observer_rx.recv())
            .await
            .expect("observer did not run")
            .expect("observer channel closed");
        assert_eq!(event.1.overlap_blocks_lost(), 6.0);
        assert_eq!(
            event,
            (
                "affinity-pin".to_string(),
                NonMaxOverlapSelection {
                    selected_worker: worker1,
                    highest_overlap_worker: worker0,
                    highest_overlap_blocks: 8.0,
                    selected_overlap_blocks: 2.0,
                },
            )
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(200), observer_rx.recv())
                .await
                .is_err(),
            "explicit pin must not report a non-max-overlap selection"
        );
    }

    #[tokio::test]
    async fn affinity_pinned_selection_reports_best_eligible_raw_cache() {
        let (queue, _slots) = make_queue(2, 16, 64, None);
        let worker0 = WorkerWithDpRank::new(0, 0);
        let worker1 = WorkerWithDpRank::new(1, 0);
        let select = |request_id: &str, affinity: bool, allowed: Option<HashSet<WorkerId>>| {
            let (mut request, response_rx) = make_request(request_id, 64);
            request
                .overlap
                .tier_overlap_blocks
                .device
                .extend([(worker0, 4), (worker1, 1)]);
            request.pinned_worker = Some(worker1);
            request.affinity_target = affinity.then(|| worker1.into());
            request.allowed_worker_ids = allowed;
            let queue = Arc::clone(&queue);
            async move {
                queue.enqueue(request).await;
                response_rx.await.unwrap().unwrap()
            }
        };

        let response = select("affinity-pin", true, None).await;
        assert_eq!(response.best_worker, worker1);
        assert_eq!(response.selected_raw_cached_tokens, Some(16));
        assert_eq!(response.max_raw_cached_tokens, Some(64));

        // Explicit pins keep measuring against the pinned worker only.
        let response = select("explicit-pin", false, None).await;
        assert_eq!(response.selected_raw_cached_tokens, Some(16));
        assert_eq!(response.max_raw_cached_tokens, Some(16));

        // Routing eligibility still applies: a disallowed worker is not "best eligible".
        let response = select(
            "affinity-pin-disallowed-best",
            true,
            Some(HashSet::from([worker1.worker_id])),
        )
        .await;
        assert_eq!(response.selected_raw_cached_tokens, Some(16));
        assert_eq!(response.max_raw_cached_tokens, Some(16));
    }

    #[tokio::test]
    async fn affinity_selection_is_not_widened_for_filtering_selectors() {
        // A custom selector may reject workers through its own filters, so the host must not
        // count a worker it never saw as the best eligible alternative.
        let (queue, _slots) = make_queue_with_custom_selector(
            2,
            16,
            64,
            None,
            MinDecodeSelector { rendezvous: None },
        );
        let worker0 = WorkerWithDpRank::new(0, 0);
        let worker1 = WorkerWithDpRank::new(1, 0);
        let (observer_tx, mut observer_rx) = tokio::sync::mpsc::unbounded_channel();
        assert!(queue.set_non_max_overlap_selection_observer(Arc::new(
            move |request_id, event| {
                observer_tx
                    .send((request_id.to_string(), event))
                    .expect("observer receiver should remain open");
            }
        )));
        let (mut request, response_rx) = make_request("affinity-pin-custom", 64);
        request
            .overlap
            .effective_overlap_blocks
            .extend([(worker0, 8.0), (worker1, 2.0)]);
        request.pinned_worker = Some(worker1);
        request.affinity_target = Some(worker1.into());

        queue.enqueue(request).await;
        assert_eq!(response_rx.await.unwrap().unwrap().best_worker, worker1);
        assert!(
            tokio::time::timeout(Duration::from_millis(200), observer_rx.recv())
                .await
                .is_err(),
            "a filtering selector's affinity pin must not report a non-max-overlap selection"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn scheduler_observer_does_not_block_actor() {
        let (queue, _slots) = make_queue_with_custom_selector(
            2,
            16,
            64,
            None,
            MinDecodeSelector { rendezvous: None },
        );
        let observer_gate = Arc::new((StdMutex::new(false), Condvar::new()));
        let callback_gate = Arc::clone(&observer_gate);
        let (started_tx, mut started_rx) = tokio::sync::mpsc::unbounded_channel();
        let (finished_tx, mut finished_rx) = tokio::sync::mpsc::unbounded_channel();
        assert!(queue.set_non_max_overlap_selection_observer(Arc::new(
            move |_request_id, _event| {
                started_tx.send(()).unwrap();
                let (released, wake) = &*callback_gate;
                let mut released = released.lock().unwrap();
                while !*released {
                    released = wake.wait(released).unwrap();
                }
                finished_tx.send(()).unwrap();
            }
        )));

        let worker0 = WorkerWithDpRank::new(0, 0);
        let worker1 = WorkerWithDpRank::new(1, 0);
        let (mut first, first_rx) = make_request("blocking-observer", 64);
        first
            .overlap
            .effective_overlap_blocks
            .extend([(worker0, 1.0), (worker1, 4.0)]);
        let first_queue = Arc::clone(&queue);
        let first_enqueue = tokio::spawn(async move {
            first_queue.enqueue(first).await;
        });

        tokio::time::timeout(Duration::from_secs(1), started_rx.recv())
            .await
            .expect("observer did not start")
            .expect("observer start channel closed");

        let (second, second_rx) = make_request("actor-remains-responsive", 64);
        tokio::time::timeout(Duration::from_secs(1), queue.enqueue(second))
            .await
            .expect("observer blocked the scheduler actor");

        let (released, wake) = &*observer_gate;
        *released.lock().unwrap() = true;
        wake.notify_one();

        first_enqueue.await.unwrap();
        first_rx.await.unwrap().unwrap();
        second_rx.await.unwrap().unwrap();
        tokio::time::timeout(Duration::from_secs(1), finished_rx.recv())
            .await
            .expect("observer did not finish")
            .expect("observer finish channel closed");
    }

    #[tokio::test]
    async fn scheduler_observer_ignores_advisory_and_abandoned_requests() {
        let (queue, _slots) = make_queue_with_custom_selector(
            2,
            16,
            64,
            None,
            MinDecodeSelector { rendezvous: None },
        );
        let worker0 = WorkerWithDpRank::new(0, 0);
        let worker1 = WorkerWithDpRank::new(1, 0);
        let observed = Arc::new(StdMutex::new(Vec::new()));
        let observer_events = Arc::clone(&observed);
        assert!(queue.set_non_max_overlap_selection_observer(Arc::new(
            move |request_id, event| {
                observer_events
                    .lock()
                    .unwrap()
                    .push((request_id.to_string(), event));
            }
        )));

        let (mut advisory, _advisory_rx) = make_request("locality-advisory", 64);
        advisory
            .overlap
            .effective_overlap_blocks
            .extend([(worker0, 1.0), (worker1, 4.0)]);
        let response = queue.select_without_admission(advisory).await.unwrap();
        assert_eq!(response.response.best_worker, worker0);

        let (mut abandoned, abandoned_rx) = make_request("locality-abandoned", 64);
        abandoned
            .overlap
            .effective_overlap_blocks
            .extend([(worker0, 1.0), (worker1, 4.0)]);
        drop(abandoned_rx);
        queue.enqueue(abandoned).await;

        assert!(observed.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn disabled_queueing_lifecycle_lease_releases_an_admitted_booking_on_cancellation() {
        let isl = 512;
        let (queue, slots) = make_queue(1, 16, isl, None);
        let request_id = "default-path-cancelled";
        let (mut request, response_rx) = make_request(request_id, isl);
        request.mode = ScheduleMode::TrackedWithLifecycle {
            request_id: request_id.to_string(),
        };

        let lease = queue
            .new_request_lifecycle_lease(Some(request_id))
            .expect("lifecycle-tracked request must receive a cancellation lease");
        let lease = queue
            .enqueue_with_block_hashes_and_lease(request, None, Some(lease))
            .await
            .expect("scheduler must return the admitted request lease");
        response_rx
            .await
            .expect("scheduler response sender dropped")
            .expect("request should be admitted before cancellation");

        drop(lease);
        tokio::time::timeout(Duration::from_secs(1), async {
            while slots.any_worker_matches_active_tokens(Instant::now(), |_, tokens| tokens != 0) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("cancellation lease cleanup did not release the admitted booking");
        slots.assert_completely_drained(decay_now());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_cancelled_pending_request_is_not_booked() {
        let isl = 512;
        let (queue, slots) = make_queue(1, 16, isl, Some(0.0));

        let (first, first_rx) = make_request("first", isl);
        queue.enqueue(first).await;
        first_rx
            .await
            .expect("first response sender dropped")
            .expect("first request should be scheduled");

        let (cancelled, cancelled_rx) = make_request("cancelled", isl);
        queue.enqueue(cancelled).await;
        assert_eq!(queue.pending_count(), 1);
        drop(cancelled_rx);

        slots.free(&"first".to_string(), decay_now()).unwrap();
        queue.update().await;

        assert_eq!(queue.pending_count(), 0);
        slots.assert_completely_drained(decay_now());
    }

    #[tokio::test]
    async fn dropped_legacy_lease_retracts_pending_request_immediately() {
        let isl = 512;
        let (queue, slots) = make_queue(1, 16, isl, Some(0.0));

        let (first, first_rx) = make_request("legacy-first", isl);
        queue.enqueue(first).await;
        first_rx.await.unwrap().unwrap();

        let (cancelled, cancelled_rx) = make_request("legacy-cancelled", isl);
        let lease = queue
            .new_request_lifecycle_lease(Some("legacy-cancelled"))
            .unwrap();
        let lease = queue
            .enqueue_with_block_hashes_and_lease(cancelled, None, Some(lease))
            .await
            .unwrap();
        assert_eq!(queue.pending_count(), 1);

        drop(cancelled_rx);
        drop(lease);
        tokio::time::timeout(Duration::from_secs(1), async {
            while queue.pending_count() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("legacy lease did not retract pending request");

        slots.free(&"legacy-first".to_owned(), decay_now()).unwrap();
        slots.assert_completely_drained(decay_now());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_strict_priority_drains_before_policy_score() {
        let isl = 512;
        let (queue, slots) = make_queue(1, 16, isl, Some(0.0));

        let (first, first_rx) = make_request("first", isl);
        queue.enqueue(first).await;
        first_rx.await.unwrap().unwrap();

        let (mut low, mut low_rx) = make_request("low", isl);
        low.priority_jump = 10_000.0;
        queue.enqueue(low).await;

        let (mut high, high_rx) = make_request("high", isl);
        high.strict_priority = 1;
        queue.enqueue(high).await;
        assert_eq!(queue.pending_count(), 2);

        slots.free(&"first".to_string(), decay_now()).unwrap();
        queue.update().await;

        let high_response = high_rx.await.unwrap().unwrap();
        assert_eq!(high_response.best_worker, WorkerWithDpRank::new(0, 0));
        assert!(
            low_rx.try_recv().is_err(),
            "lower strict priority should remain queued"
        );

        slots.free(&"high".to_string(), decay_now()).unwrap();
        queue.update().await;
        low_rx.await.unwrap().unwrap();
        assert_eq!(queue.pending_count(), 0);

        slots.free(&"low".to_string(), decay_now()).unwrap();
        slots.assert_completely_drained(decay_now());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_failed_response_delivery_rolls_back_booking() {
        let isl = 512;
        let response_rx = Arc::new(StdMutex::new(None));
        let publisher = DropResponseOnLoadPublisher {
            response_rx: Arc::clone(&response_rx),
        };
        let slots = Arc::new(ActiveSequencesMultiWorker::new(
            publisher,
            16,
            HashMap::from([(0, (0, 1))]),
            false,
            0,
            "test",
        ));
        let (_cfg_tx, cfg_rx) = watch::channel(HashMap::from([(
            0,
            SimpleWorkerConfig {
                max_num_batched_tokens: Some(isl as u64),
                ..Default::default()
            },
        )]));
        let queue = SchedulerQueue::new(
            Arc::clone(&slots),
            cfg_rx,
            PolicyProfile::synthetic(None, RouterQueuePolicy::Fcfs),
            16,
            DefaultWorkerSelector::new(None, "test"),
            None,
            None::<Arc<NoopOverlapScoresRefresh>>,
            None,
            None,
            None,
        );

        let (request, receiver) = make_request("delivery-race", isl);
        *response_rx.lock().unwrap() = Some(receiver);
        queue.enqueue(request).await;

        assert!(response_rx.lock().unwrap().is_none());
        slots.assert_completely_drained(decay_now());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_concurrent_flood() {
        let block_size = 16;
        let isl = 512;
        let num_workers = 4;
        let num_tasks = 25;

        let (queue, slots) = make_queue(num_workers, block_size, isl, None);

        let mut handles = Vec::new();
        for i in 0..num_tasks {
            let queue = Arc::clone(&queue);
            let slots = Arc::clone(&slots);
            handles.push(tokio::spawn(async move {
                let req_id = format!("req-{i}");
                let (req, rx) = make_request(&req_id, isl);
                queue.enqueue(req).await;
                let resp = rx.await.expect("oneshot dropped");
                let resp = resp.expect("scheduling failed");
                assert!(resp.best_worker.worker_id < num_workers as u64);

                slots.mark_prefill_completed(&req_id, decay_now()).unwrap();
                slots.free(&req_id, decay_now()).unwrap();
                queue.update().await;
            }));
        }

        for h in handles {
            h.await.expect("task panicked");
        }

        let active = slots.active_tokens(decay_now());
        for (worker, tokens) in &active {
            assert_eq!(
                *tokens, 0,
                "worker {worker:?} still has {tokens} active tokens"
            );
        }
    }

    /// A backlog comes out in arrival order within one poll. A receive spends one unit of the
    /// task's cooperative budget, which runs out after 128: one command per receive would yield.
    #[tokio::test]
    async fn inbox_hands_out_a_backlog_in_order_without_yielding() {
        const BACKLOG: usize = 200;
        let (tx, rx) = mpsc::channel(BACKLOG);
        let mut acks = Vec::new();
        for _ in 0..BACKLOG {
            let (ack_tx, ack_rx) = oneshot::channel();
            tx.try_send(AdmissionCommand::Update { ack_tx }).unwrap();
            acks.push(ack_rx);
        }
        let mut inbox = CommandInbox::new(rx);
        let drain = std::pin::pin!(async {
            for ack in &mut acks {
                let Some(AdmissionCommand::Update { ack_tx }) = inbox.next().await else {
                    panic!("expected a queued update");
                };
                ack_tx.send(()).unwrap();
                ack.try_recv().unwrap();
            }
        });
        assert!(futures_util::poll!(drain).is_ready());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_concurrent_immediate_admissions_see_prior_booking() {
        let selector = MinDecodeSelector {
            rendezvous: Some(Arc::new(SelectorRendezvous::default())),
        };
        let (queue, slots) = make_queue_with_custom_selector(2, 16, 512, None, selector);
        let barrier = Arc::new(Barrier::new(3));

        let (req1, rx1) = make_request("req-1", 512);
        let queue1 = Arc::clone(&queue);
        let barrier1 = Arc::clone(&barrier);
        let handle1 = tokio::spawn(async move {
            barrier1.wait().await;
            queue1.enqueue(req1).await;
        });

        let (req2, rx2) = make_request("req-2", 512);
        let queue2 = Arc::clone(&queue);
        let barrier2 = Arc::clone(&barrier);
        let handle2 = tokio::spawn(async move {
            barrier2.wait().await;
            queue2.enqueue(req2).await;
        });

        barrier.wait().await;
        handle1.await.unwrap();
        handle2.await.unwrap();

        let resp1 = rx1.await.unwrap().unwrap();
        let resp2 = rx2.await.unwrap().unwrap();
        assert_ne!(
            resp1.best_worker, resp2.best_worker,
            "second admission should see the first booking and choose the other idle worker"
        );

        for request_id in ["req-1", "req-2"] {
            slots
                .mark_prefill_completed(&request_id.to_string(), decay_now())
                .unwrap();
            slots.free(&request_id.to_string(), decay_now()).unwrap();
        }
    }

    #[tokio::test]
    async fn reject_thresholds_shed_full_workers_and_reject_when_none_remain() {
        let profile = PolicyProfile::synthetic(None, RouterQueuePolicy::Fcfs);
        let (queue, slots, cfg_tx) = make_queue_with_profile_sender_and_reject(
            2,
            16,
            1_000_000,
            profile,
            Some(Arc::new(|| dynamo_slot_tracker::busy::BusyThresholds {
                decode_busy_threshold_frac: Some(0.5),
                ..Default::default()
            })),
        );
        let worker_config = SimpleWorkerConfig {
            max_num_batched_tokens: Some(1_000_000),
            total_kv_blocks: Some(8),
            ..Default::default()
        };
        cfg_tx
            .send(HashMap::from([
                (0, worker_config.clone()),
                (1, worker_config),
            ]))
            .unwrap();
        set_worker_capacities_from_configs(&slots, &cfg_tx.borrow());
        // Hold 6 of 8 KV blocks on `worker` with a decoding request.
        let hold = |request_id: &'static str, worker: WorkerWithDpRank| {
            let (mut held, held_rx) = make_request(request_id, 96);
            held.token_seq = Some((0..6).collect());
            held.pinned_worker = Some(worker);
            (held, held_rx)
        };
        let first = WorkerWithDpRank::new(0, 0);
        let second = WorkerWithDpRank::new(1, 0);
        let (held, held_rx) = hold("held-0", first);
        queue.enqueue(held).await;
        held_rx.await.unwrap().unwrap();
        slots
            .mark_prefill_completed(&"held-0".to_string(), decay_now())
            .unwrap();

        let (next, next_rx) = make_request("next", 16);
        queue.enqueue(next).await;
        assert_eq!(
            next_rx.await.unwrap().unwrap().best_worker,
            second,
            "a worker past the reject thresholds must be skipped"
        );
        slots.free(&"next".to_string(), decay_now()).unwrap();

        // The scan, not the picker's preference, keeps `first` out: pinning to it fails.
        let (mut pinned, pinned_rx) = make_request("pinned-full", 16);
        pinned.pinned_worker = Some(first);
        queue.enqueue(pinned).await;
        assert!(matches!(
            pinned_rx.await.unwrap(),
            Err(KvSchedulerError::PinnedWorkerOverloaded { worker_id: 0 })
        ));

        let (held, held_rx) = hold("held-1", second);
        queue.enqueue(held).await;
        held_rx.await.unwrap().unwrap();
        slots
            .mark_prefill_completed(&"held-1".to_string(), decay_now())
            .unwrap();

        let (rejected, rejected_rx) = make_request("rejected", 16);
        queue.enqueue(rejected).await;
        assert!(matches!(
            rejected_rx.await.unwrap(),
            Err(KvSchedulerError::AllEligibleWorkersOverloaded)
        ));
    }

    #[tokio::test]
    async fn reject_thresholds_skip_a_worker_only_when_every_rank_is_full() {
        let profile = PolicyProfile::synthetic(None, RouterQueuePolicy::Fcfs);
        let (queue, slots, cfg_tx) = make_queue_with_profile_sender_and_reject(
            1,
            16,
            100,
            profile,
            Some(Arc::new(|| dynamo_slot_tracker::busy::BusyThresholds {
                prefill_busy_threshold_frac: Some(0.5),
                ..Default::default()
            })),
        );
        // Worker 0 grows to two ranks with a 100-token prefill budget each.
        let worker_config = SimpleWorkerConfig {
            max_num_batched_tokens: Some(100),
            data_parallel_size: 2,
            ..Default::default()
        };
        cfg_tx.send(HashMap::from([(0, worker_config)])).unwrap();
        slots
            .reconcile_workers([dynamo_slot_tracker::topology::WorkerDpRange::new(0, 0, 2)])
            .unwrap();
        set_worker_capacities_from_configs(&slots, &cfg_tx.borrow());
        let rank0 = WorkerWithDpRank::new(0, 0);
        let rank1 = WorkerWithDpRank::new(0, 1);

        let (mut held, held_rx) = make_request("held-rank0", 96);
        held.pinned_worker = Some(rank0);
        queue.enqueue(held).await;
        held_rx.await.unwrap().unwrap();

        // One full rank: the worker stays eligible and the request lands on the other rank.
        let (next, next_rx) = make_request("next", 16);
        queue.enqueue(next).await;
        assert_eq!(next_rx.await.unwrap().unwrap().best_worker, rank1);
        slots.free(&"next".to_string(), decay_now()).unwrap();

        let (mut held, held_rx) = make_request("held-rank1", 96);
        held.pinned_worker = Some(rank1);
        queue.enqueue(held).await;
        held_rx.await.unwrap().unwrap();

        // Both ranks full: the worker is skipped and nothing remains.
        let (rejected, rejected_rx) = make_request("rejected", 16);
        queue.enqueue(rejected).await;
        assert!(matches!(
            rejected_rx.await.unwrap(),
            Err(KvSchedulerError::AllEligibleWorkersOverloaded)
        ));
    }

    #[tokio::test]
    async fn reject_thresholds_follow_the_live_provider_on_prefill_tokens() {
        let live = Arc::new(Mutex::new(dynamo_slot_tracker::busy::BusyThresholds {
            prefill_busy_threshold_frac: Some(0.5),
            ..Default::default()
        }));
        let provider = Arc::clone(&live);
        let profile = PolicyProfile::synthetic(None, RouterQueuePolicy::Fcfs);
        let (queue, slots, _cfg_tx) = make_queue_with_profile_sender_and_reject(
            2,
            16,
            100,
            profile,
            Some(Arc::new(move || *provider.lock())),
        );
        let first = WorkerWithDpRank::new(0, 0);
        let second = WorkerWithDpRank::new(1, 0);

        // 96 booked prefill tokens on `first` is past half of its 100-token budget.
        let (mut held, held_rx) = make_request("held", 96);
        held.pinned_worker = Some(first);
        queue.enqueue(held).await;
        held_rx.await.unwrap().unwrap();

        let (mut next, next_rx) = make_request("next", 16);
        next.pinned_worker = Some(first);
        queue.enqueue(next).await;
        assert!(matches!(
            next_rx.await.unwrap(),
            Err(KvSchedulerError::PinnedWorkerOverloaded { worker_id: 0 })
        ));
        let (free, free_rx) = make_request("free", 16);
        queue.enqueue(free).await;
        assert_eq!(
            free_rx.await.unwrap().unwrap().best_worker,
            second,
            "an unpinned request goes to the worker with room"
        );
        slots.free(&"free".to_string(), decay_now()).unwrap();

        // Lifting the thresholds through the live provider takes effect at the next selection.
        *live.lock() = dynamo_slot_tracker::busy::BusyThresholds::default();
        let (mut again, again_rx) = make_request("again", 16);
        again.pinned_worker = Some(first);
        queue.enqueue(again).await;
        assert_eq!(again_rx.await.unwrap().unwrap().best_worker, first);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_queueing_under_pressure() {
        let block_size = 16;
        let isl = 512;
        let num_workers = 2;
        let num_requests = 10;

        let (queue, slots) = make_queue(num_workers, block_size, isl, Some(0.0));

        let mut receivers = Vec::new();
        let mut req_ids = Vec::new();

        for i in 0..num_requests {
            let req_id = format!("pressure-{i}");
            let (req, rx) = make_request(&req_id, isl);
            queue.enqueue(req).await;
            receivers.push(rx);
            req_ids.push(req_id);
        }

        // Drain pending by cycling mark_prefill_completed + free + update
        // on already-scheduled requests until all receivers have a response.
        for _ in 0..num_requests {
            queue.update().await;
            for rid in &req_ids {
                let _ = slots.mark_prefill_completed(rid, decay_now());
                let _ = slots.free(rid, decay_now());
            }
        }
        queue.update().await;

        let mut ok_count = 0;
        for mut rx in receivers {
            if let Ok(result) = rx.try_recv() {
                result.expect("scheduling returned error");
                ok_count += 1;
            }
        }
        assert_eq!(ok_count, num_requests, "not all requests were scheduled");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_pending_requests_receive_shutdown_on_queue_drop() {
        let block_size = 16;
        let isl = 512;
        let (queue, _slots) = make_queue(1, block_size, isl, Some(0.0));

        let (req1, rx1) = make_request("req-1", isl);
        queue.enqueue(req1).await;
        rx1.await
            .expect("first response sender dropped")
            .expect("first request should be scheduled");

        let (req2, rx2) = make_request("req-2", isl);
        queue.enqueue(req2).await;
        assert_eq!(queue.pending_count(), 1);

        drop(queue);

        let response = tokio::time::timeout(Duration::from_secs(1), rx2)
            .await
            .expect("shutdown response timed out")
            .expect("pending response sender dropped");
        assert!(matches!(
            response,
            Err(KvSchedulerError::SubscriberShutdown)
        ));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_pending_count() {
        let block_size = 16;
        let isl = 512;
        let num_workers = 1;

        // threshold_frac=0.0 means any active tokens trigger queueing
        let (queue, slots) = make_queue(num_workers, block_size, isl, Some(0.0));
        assert_eq!(queue.pending_count(), 0);

        // First request goes through (worker is idle)
        let (req1, rx1) = make_request("req-1", isl);
        queue.enqueue(req1).await;
        let _resp1 = rx1.await.unwrap().unwrap();
        assert_eq!(queue.pending_count(), 0); // scheduled immediately

        // Second and third requests should be queued (worker is now prefill-busy)
        let (req2, _rx2) = make_request("req-2", isl);
        queue.enqueue(req2).await;
        assert_eq!(queue.pending_count(), 1);

        let (req3, _rx3) = make_request("req-3", isl);
        queue.enqueue(req3).await;
        assert_eq!(queue.pending_count(), 2);

        // Free the first request and update — should drain one from pending
        slots
            .mark_prefill_completed(&"req-1".to_string(), decay_now())
            .unwrap();
        slots.free(&"req-1".to_string(), decay_now()).unwrap();
        queue.update().await;

        // After update, one pending request should have been scheduled
        assert!(
            queue.pending_count() < 2,
            "pending_count should decrease after free+update, got {}",
            queue.pending_count()
        );

        // Free req-2 and update to drain remaining
        let _ = slots.mark_prefill_completed(&"req-2".to_string(), decay_now());
        let _ = slots.free(&"req-2".to_string(), decay_now());
        queue.update().await;
        let _ = slots.mark_prefill_completed(&"req-3".to_string(), decay_now());
        let _ = slots.free(&"req-3".to_string(), decay_now());
        queue.update().await;

        assert_eq!(queue.pending_count(), 0, "all requests should be drained");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn class_names_select_queues_independent_of_cache_state() {
        let profile = policy_profile(
            Some("standard"),
            vec![
                PolicyClassConfig {
                    busy_thresholds: BusyThresholds {
                        prefill_busy_threshold: Some(0),
                        ..Default::default()
                    },
                    ..class("standard", 1)
                },
                PolicyClassConfig {
                    busy_thresholds: BusyThresholds {
                        prefill_busy_threshold: Some(0),
                        ..Default::default()
                    },
                    ..class("latency", 1)
                },
                PolicyClassConfig {
                    busy_thresholds: BusyThresholds {
                        prefill_busy_threshold: Some(0),
                        ..Default::default()
                    },
                    ..class("custom_priority", 1)
                },
            ],
        );
        let (queue, _slots) = make_queue_with_profile(1, 16, 64, profile);
        let worker = WorkerWithDpRank::new(0, 0);

        // Cache state changes a request's token snapshot and cost, never its class.
        let (mut probe, _probe_rx) = make_request("probe", 64);
        probe.policy_class = Some("latency".to_string());
        let uncached = queue
            .default_queue_metadata(&probe, Instant::now())
            .unwrap();
        probe.overlap.effective_cached_tokens.insert(worker, 48);
        let cached = queue
            .default_queue_metadata(&probe, Instant::now())
            .unwrap();
        assert_eq!((uncached.class_index, cached.class_index), (1, 1));
        assert_eq!(uncached.snapshot, QueueSnapshot::new(64, 0));
        assert_eq!(cached.snapshot, QueueSnapshot::new(64, 48));

        let (active, active_rx) = make_request("active", 64);
        queue.enqueue(active).await;
        active_rx.await.unwrap().unwrap();

        let (mut latency_cached, _latency_cached_rx) = make_request("latency-cached", 64);
        latency_cached.policy_class = Some("latency".to_string());
        latency_cached
            .overlap
            .effective_cached_tokens
            .insert(worker, 64);
        queue.enqueue(latency_cached).await;

        let (mut latency_uncached, _latency_uncached_rx) = make_request("latency-uncached", 64);
        latency_uncached.policy_class = Some("latency".to_string());
        queue.enqueue(latency_uncached).await;

        // An unknown name fails before it is queued, cached or not.
        let (mut unknown_cached, unknown_cached_rx) = make_request("unknown-cached", 64);
        unknown_cached.policy_class = Some("unknown".to_string());
        unknown_cached
            .overlap
            .effective_cached_tokens
            .insert(worker, 64);
        queue.enqueue(unknown_cached).await;
        assert!(matches!(
            unknown_cached_rx.await.unwrap(),
            Err(KvSchedulerError::UnknownPolicyClass(name)) if name == "unknown"
        ));

        // Missing names use the default class, cached or not.
        let (mut missing_cached, _missing_cached_rx) = make_request("missing-cached", 64);
        missing_cached
            .overlap
            .effective_cached_tokens
            .insert(worker, 64);
        queue.enqueue(missing_cached).await;

        let (missing, _missing_rx) = make_request("missing", 64);
        queue.enqueue(missing).await;

        let (mut custom, _custom_rx) = make_request("custom", 64);
        custom.policy_class = Some("custom_priority".to_string());
        queue.enqueue(custom).await;

        assert_eq!(
            queue.class_queue_stats(0),
            Some(ClassQueueStats {
                received_total: 3,
                rejected_due_time_passed_total: 0,
                rejected_queue_size_exceeded_total: 0,
                pending_count: 2,
                pending_isl_tokens: 128,
                pending_cached_tokens: 64,
            })
        );
        assert_eq!(
            queue.class_queue_stats(1),
            Some(ClassQueueStats {
                received_total: 2,
                rejected_due_time_passed_total: 0,
                rejected_queue_size_exceeded_total: 0,
                pending_count: 2,
                pending_isl_tokens: 128,
                pending_cached_tokens: 64,
            })
        );
        assert_eq!(
            queue.class_queue_stats(2),
            Some(ClassQueueStats {
                received_total: 1,
                rejected_due_time_passed_total: 0,
                rejected_queue_size_exceeded_total: 0,
                pending_count: 1,
                pending_isl_tokens: 64,
                pending_cached_tokens: 0,
            })
        );
    }

    #[tokio::test(start_paused = true)]
    async fn wspt_class_budget_neither_reorders_nor_expires_queued_requests() {
        let profile = policy_profile(
            None,
            vec![PolicyClassConfig {
                queue_policy: RouterQueuePolicy::Wspt,
                slo_ttft_ms: NonZeroU32::new(1),
                busy_thresholds: BusyThresholds {
                    prefill_busy_threshold_frac: Some(0.0),
                    ..Default::default()
                },
                ..class("standard", 1)
            }],
        );
        let (queue, slots) = make_queue_with_profile(1, 16, 64, profile);
        let (active, active_rx) = make_request("active", 64);
        queue.enqueue(active).await;
        active_rx.await.unwrap().unwrap();

        let (long, mut long_rx) = make_request("long", 64);
        let ingress_at = Instant::now();
        let metadata = queue.default_queue_metadata(&long, ingress_at).unwrap();
        // The class due time is resolved for every class, but only EDF orders by it.
        assert_eq!(metadata.due_at, None);
        assert_eq!(
            metadata.class_due_at,
            Some(ingress_at + Duration::from_millis(1))
        );
        queue.enqueue(long).await;
        let (short, mut short_rx) = make_request("short", 16);
        queue.enqueue(short).await;
        let (explicit, explicit_rx) = make_request("explicit", 64);
        queue
            .enqueue_with_due_at_for_test(explicit, Instant::now() + Duration::from_secs(30))
            .await;
        assert_eq!(queue.pending_count(), 3);

        // Far past the 1 ms class budget, only the explicit deadline expires, at its own time.
        tokio::time::advance(Duration::from_secs(29)).await;
        queue.update().await;
        assert_eq!(queue.pending_count(), 3);
        tokio::time::advance(Duration::from_secs(2)).await;
        assert!(matches!(
            explicit_rx.await.unwrap(),
            Err(KvSchedulerError::DeadlineExceeded)
        ));
        assert_eq!(queue.pending_count(), 2);

        // WSPT still orders the class, not arrival plus the budget: the later, shorter
        // request is admitted first.
        slots
            .mark_prefill_completed(&"active".to_string(), decay_now())
            .unwrap();
        slots.free(&"active".to_string(), decay_now()).unwrap();
        queue.update().await;
        short_rx.try_recv().unwrap().unwrap();
        assert!(matches!(
            long_rx.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));

        slots
            .mark_prefill_completed(&"short".to_string(), decay_now())
            .unwrap();
        slots.free(&"short".to_string(), decay_now()).unwrap();
        queue.update().await;
        long_rx.try_recv().unwrap().unwrap();
        assert_eq!(queue.pending_count(), 0);
        let stats = queue.class_queue_stats(0).unwrap();
        assert_eq!(stats.received_total, 4);
        assert_eq!(stats.rejected_due_time_passed_total, 1);
    }

    #[tokio::test(start_paused = true)]
    async fn class_due_time_counts_from_original_ingress_for_the_final_class() {
        let profile = policy_profile(
            Some("slow"),
            vec![
                PolicyClassConfig {
                    slo_ttft_ms: NonZeroU32::new(100),
                    ..class("fast", 1)
                },
                PolicyClassConfig {
                    slo_ttft_ms: NonZeroU32::new(10000),
                    ..class("slow", 1)
                },
                class("unbudgeted", 1),
            ],
        );
        let (queue, _slots) = make_queue_with_profile(1, 16, 64, profile);
        let ingress_at = Instant::now();
        // A slow classifier and channel wait spend the budget instead of restarting it.
        tokio::time::advance(Duration::from_secs(5)).await;

        let (mut request, _rx) = make_request("timed", 16);
        let class_due_at = |request: &SchedulingRequest| {
            queue
                .default_queue_metadata(request, ingress_at)
                .unwrap()
                .class_due_at
        };
        assert_eq!(
            class_due_at(&request),
            Some(ingress_at + Duration::from_secs(10))
        );
        request.policy_class = Some("fast".to_string());
        assert_eq!(
            class_due_at(&request),
            Some(ingress_at + Duration::from_millis(100))
        );

        // A classifier's class replaces the requested one before the budget applies.
        for (selected, expected) in [
            ("slow", Some(ingress_at + Duration::from_secs(10))),
            ("unbudgeted", None),
        ] {
            let mut classified = queue
                .build_classify_request(&request, ingress_at, WorkerInputs::NONE)
                .0;
            classified.set_policy_class(selected);
            let metadata = queue
                .validate_classification(&mut request, classified, ingress_at)
                .unwrap();
            assert_eq!(metadata.class_due_at, expected, "{selected}");
        }

        // An explicit deadline stays separate; it takes precedence when the queue orders.
        let due_at = ingress_at + Duration::from_secs(60);
        let mut classified = queue
            .build_classify_request(&request, ingress_at, WorkerInputs::NONE)
            .0;
        classified.set_due_at(due_at);
        let metadata = queue
            .validate_classification(&mut request, classified, ingress_at)
            .unwrap();
        assert_eq!(metadata.due_at, Some(due_at));
        assert_eq!(
            metadata.class_due_at,
            Some(ingress_at + Duration::from_millis(100))
        );
    }

    #[tokio::test(start_paused = true)]
    async fn edf_class_budget_orders_by_original_ingress_and_explicit_deadlines_override() {
        let profile = policy_profile(
            None,
            vec![PolicyClassConfig {
                slo_ttft_ms: NonZeroU32::new(1000),
                busy_thresholds: BusyThresholds {
                    prefill_busy_threshold_frac: Some(0.0),
                    ..Default::default()
                },
                ..class("standard", 1)
            }],
        );
        let (queue, slots) = make_queue_with_profile(1, 16, 64, profile);
        assert_eq!(queue.profile.class(0).queue_policy, RouterQueuePolicy::Edf);
        let start = Instant::now();
        let (active, active_rx) = make_request("active", 64);
        queue.enqueue(active).await;
        active_rx.await.unwrap().unwrap();

        // Enqueue order differs from ingress order, as after slow classification. Strict
        // priority and priority jumps do not reorder EDF.
        tokio::time::advance(Duration::from_millis(30)).await;
        let mut receivers = Vec::new();
        for (request_id, ingress_ms, explicit_due_ms, strict_priority) in [
            ("late-ingress", 20, None, 9),
            ("explicit-late", 0, Some(30_000), 0),
            ("early-ingress", 10, None, 0),
            ("explicit-early", 25, Some(900), 0),
        ] {
            let (mut request, response_rx) = make_request(request_id, 64);
            request.strict_priority = strict_priority;
            request.priority_jump = f64::from(strict_priority) * 100.0;
            let ingress_at = start + Duration::from_millis(ingress_ms);
            let classified = explicit_due_ms.map(|due_ms| {
                let mut classified = queue
                    .build_classify_request(&request, ingress_at, WorkerInputs::NONE)
                    .0;
                classified.set_due_at(start + Duration::from_millis(due_ms));
                classified
            });
            queue
                .enqueue_admitted_with_block_hashes_and_lease(
                    request, None, None, None, classified, ingress_at,
                )
                .await;
            receivers.push((request_id, response_rx));
        }
        assert_eq!(queue.pending_count(), 4);

        let mut admitted = Vec::new();
        let mut running = "active".to_string();
        while !receivers.is_empty() {
            slots.mark_prefill_completed(&running, decay_now()).unwrap();
            slots.free(&running, decay_now()).unwrap();
            queue.update().await;
            let ready = receivers
                .iter_mut()
                .position(|(_, response_rx)| response_rx.try_recv().is_ok())
                .expect("freed capacity admits the next request");
            running = receivers.remove(ready).0.to_string();
            admitted.push(running.clone());
        }
        assert_eq!(
            admitted,
            [
                "explicit-early",
                "early-ingress",
                "late-ingress",
                "explicit-late"
            ]
        );
        assert_eq!(
            queue
                .class_queue_stats(0)
                .unwrap()
                .rejected_due_time_passed_total,
            0
        );
    }

    #[tokio::test(start_paused = true)]
    async fn edf_class_budget_passing_neither_rejects_nor_blocks_admission() {
        let profile = policy_profile(
            None,
            vec![PolicyClassConfig {
                slo_ttft_ms: NonZeroU32::new(1),
                busy_thresholds: BusyThresholds {
                    prefill_busy_threshold_frac: Some(0.0),
                    ..Default::default()
                },
                ..class("standard", 1)
            }],
        );
        let (queue, slots) = make_queue_with_profile(1, 16, 64, profile);
        let (active, active_rx) = make_request("active", 64);
        queue.enqueue(active).await;
        active_rx.await.unwrap().unwrap();
        let (first, mut first_rx) = make_request("first", 64);
        queue.enqueue(first).await;
        let (second, mut second_rx) = make_request("second", 64);
        queue.enqueue(second).await;

        // No timer, enqueue check, or refresh check rejects a passed class budget.
        tokio::time::advance(Duration::from_secs(10)).await;
        queue.update().await;
        assert_eq!(queue.pending_count(), 2);

        for (finished, admitted_rx) in [("active", &mut first_rx), ("first", &mut second_rx)] {
            slots
                .mark_prefill_completed(&finished.to_string(), decay_now())
                .unwrap();
            slots.free(&finished.to_string(), decay_now()).unwrap();
            queue.update().await;
            admitted_rx.try_recv().unwrap().unwrap();
        }
        assert_eq!(queue.pending_count(), 0);
        let stats = queue.class_queue_stats(0).unwrap();
        assert_eq!(stats.received_total, 3);
        assert_eq!(stats.rejected_due_time_passed_total, 0);
    }

    #[tokio::test(start_paused = true)]
    async fn edf_class_budget_expires_a_queued_request_on_its_own_timer() {
        let profile = policy_profile(
            None,
            vec![PolicyClassConfig {
                slo_ttft_ms: NonZeroU32::new(100),
                controlled_delay: true,
                busy_thresholds: BusyThresholds {
                    prefill_busy_threshold_frac: Some(0.0),
                    ..Default::default()
                },
                ..class("standard", 1)
            }],
        );
        let (queue, slots) = make_queue_with_profile(1, 16, 64, profile);
        let start = Instant::now();
        // Under capacity, an arrival is admitted without any load notification.
        let (active, active_rx) = make_request("active", 64);
        queue.enqueue(active).await;
        active_rx.await.unwrap().unwrap();

        // The worker stays busy until 300 ms. The request arriving at 10 ms is rejected at
        // its 110 ms due time by its class timer: nothing else wakes the actor.
        tokio::time::advance(Duration::from_millis(10)).await;
        let (expires, expires_rx) = make_request("expires", 64);
        queue.enqueue(expires).await;
        assert!(matches!(
            expires_rx.await.unwrap(),
            Err(KvSchedulerError::DeadlineExceeded)
        ));
        assert_eq!(Instant::now() - start, Duration::from_millis(110));
        assert_eq!(queue.pending_count(), 0);

        // A fresh request arriving at 250 ms is admitted at 300 ms, before its 350 ms due.
        tokio::time::advance(Duration::from_millis(140)).await;
        let (fresh, mut fresh_rx) = make_request("fresh", 64);
        queue.enqueue(fresh).await;
        tokio::time::advance(Duration::from_millis(50)).await;
        assert!(fresh_rx.try_recv().is_err());
        slots
            .mark_prefill_completed(&"active".to_string(), decay_now())
            .unwrap();
        slots.free(&"active".to_string(), decay_now()).unwrap();
        queue.update().await;
        assert_eq!(Instant::now() - start, Duration::from_millis(300));
        fresh_rx.try_recv().unwrap().unwrap();

        let stats = queue.class_queue_stats(0).unwrap();
        assert_eq!(stats.received_total, 3);
        assert_eq!(stats.rejected_due_time_passed_total, 1);
        assert_eq!(stats.pending_count, 0);
        slots.free(&"fresh".to_string(), decay_now()).unwrap();
        slots.assert_completely_drained(decay_now());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn edf_blocked_pinned_lane_does_not_block_other_worker() {
        let profile = policy_profile(
            None,
            vec![PolicyClassConfig {
                slo_ttft_ms: NonZeroU32::new(1000),
                busy_thresholds: BusyThresholds {
                    prefill_busy_threshold_frac: Some(0.0),
                    ..Default::default()
                },
                ..class("standard", 1)
            }],
        );
        let (queue, slots) = make_queue_with_profile(2, 16, 256, profile);
        assert_eq!(queue.profile.class(0).queue_policy, RouterQueuePolicy::Edf);

        let (mut first, first_rx) = make_request("pinned-1", 256);
        first.pinned_worker = Some(WorkerWithDpRank::new(1, 0));
        queue.enqueue(first).await;
        let first_resp = first_rx.await.unwrap().unwrap();
        assert_eq!(first_resp.best_worker, WorkerWithDpRank::new(1, 0));

        // The earliest-due request waits for its full pinned worker.
        let (mut second, mut second_rx) = make_request("pinned-2", 256);
        second.pinned_worker = Some(WorkerWithDpRank::new(1, 0));
        queue.enqueue(second).await;
        assert_eq!(queue.pending_count(), 1);
        assert!(
            second_rx.try_recv().is_err(),
            "request should remain queued"
        );

        // A later request pinned to the idle worker does not wait behind it.
        let (mut other_worker, mut other_worker_rx) = make_request("pinned-0", 256);
        other_worker.pinned_worker = Some(WorkerWithDpRank::new(0, 0));
        queue.enqueue(other_worker).await;
        assert_eq!(queue.pending_count(), 1);
        let other_worker_resp = other_worker_rx
            .try_recv()
            .expect("other worker request should have been scheduled")
            .expect("scheduling returned error");
        assert_eq!(other_worker_resp.best_worker, WorkerWithDpRank::new(0, 0));
        assert!(
            second_rx.try_recv().is_err(),
            "pinned request should still be queued"
        );

        slots
            .mark_prefill_completed(&"pinned-1".to_string(), decay_now())
            .unwrap();
        slots.free(&"pinned-1".to_string(), decay_now()).unwrap();
        queue.capacity_changed(Some(WorkerWithDpRank::new(1, 0)));

        let second_resp = tokio::time::timeout(Duration::from_secs(1), second_rx)
            .await
            .expect("capacity notification should schedule the pinned request")
            .expect("response channel should remain open");
        let second_resp = second_resp.expect("scheduling returned error");
        assert_eq!(second_resp.best_worker, WorkerWithDpRank::new(1, 0));
        assert_eq!(queue.pending_count(), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn test_queue_update_uses_decayed_oldest_prefill_load() {
        let estimator: Arc<dyn PrefillLoadEstimator> = Arc::new(FixedPrefillLoadEstimator {
            duration: Duration::from_secs(10),
        });
        let (queue, _slots, _cfg_tx) =
            make_queue_with_sender(1, 16, 100, Some(0.5), Some(estimator));

        let (req1, rx1) = make_request("req-1", 100);
        queue.enqueue(req1).await;
        let _ = rx1.await.unwrap().unwrap();

        let (req2, mut rx2) = make_request("req-2", 100);
        queue.enqueue(req2).await;
        assert_eq!(queue.pending_count(), 1);

        tokio::time::advance(Duration::from_secs(6)).await;
        queue.update().await;

        let scheduled = rx2
            .try_recv()
            .expect("queued request should have been scheduled");
        let response = scheduled.expect("scheduling returned error");
        assert_eq!(response.best_worker.worker_id, 0);
        assert_eq!(queue.pending_count(), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_overloaded_provider_filters_at_admission() {
        let overloaded_worker_provider: OverloadedWorkerProvider =
            Arc::new(|| Some(Arc::new(HashSet::from([0]))));
        let (queue, _slots) =
            make_queue_with_providers(1, 16, 256, Some(overloaded_worker_provider), None);

        let (req, rx) = make_request("overloaded", 256);
        queue.enqueue(req).await;

        let resp = rx.await.expect("oneshot dropped");
        assert!(matches!(
            resp,
            Err(KvSchedulerError::AllEligibleWorkersOverloaded)
        ));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn hard_availability_provider_filters_unpinned_and_pinned_selection() {
        let available_worker_provider: WorkerAvailabilityProvider =
            Arc::new(|_| Some(Arc::new(HashSet::from([1]))));
        let (queue, slots) =
            make_queue_with_providers(2, 16, 256, None, Some(available_worker_provider));

        let request_id = "available".to_string();
        let (request, response_rx) = make_request(&request_id, 256);
        queue.enqueue(request).await;
        let response = response_rx.await.unwrap().unwrap();
        assert_eq!(response.best_worker.worker_id, 1);
        slots
            .mark_prefill_completed(&request_id, decay_now())
            .unwrap();
        slots.free(&request_id, decay_now()).unwrap();

        let (mut pinned, pinned_rx) = make_request("unavailable-pin", 256);
        pinned.pinned_worker = Some(WorkerWithDpRank::from_worker_id(0));
        queue.enqueue(pinned).await;
        assert!(matches!(
            pinned_rx.await.unwrap(),
            Err(KvSchedulerError::NoEndpoints)
        ));
    }

    #[tokio::test]
    async fn queue_snapshot_excludes_workers_unavailable_to_the_request() {
        let provider: WorkerAvailabilityProvider = Arc::new(|request| {
            assert_eq!(request.lora_name.as_deref(), Some("adapter"));
            Some(Arc::new(HashSet::from([1])))
        });
        let (queue, _slots) = make_queue_with_providers(2, 16, 256, None, Some(provider));
        let (mut request, _response) = make_request("adapter-request", 256);
        request.lora_name = Some("adapter".to_string());
        request.overlap.effective_cached_tokens = FxHashMap::from_iter([
            (WorkerWithDpRank::from_worker_id(0), 224),
            (WorkerWithDpRank::from_worker_id(1), 32),
        ]);
        assert_eq!(
            queue
                .default_queue_metadata(&request, Instant::now())
                .unwrap()
                .snapshot
                .cached_tokens,
            32
        );
        assert_eq!(
            queue
                .build_classify_request(&request, Instant::now(), WorkerInputs::NONE)
                .0
                .scheduling_cost_tokens(),
            224
        );

        request.pinned_worker = Some(WorkerWithDpRank::from_worker_id(0));
        assert_eq!(
            queue
                .default_queue_metadata(&request, Instant::now())
                .unwrap()
                .snapshot
                .cached_tokens,
            0
        );
        assert_eq!(
            queue
                .build_classify_request(&request, Instant::now(), WorkerInputs::NONE)
                .0
                .scheduling_cost_tokens(),
            256
        );
    }

    /// A queue starting with zero workers can route after slots and configs arrive.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_worker_updates_after_empty_start() {
        let block_size = 16;
        let isl = 512;

        // Start with zero workers (mimics skip_initial_worker_wait=true)
        let (queue, slots, cfg_tx) = make_queue_with_sender(0, block_size, isl, None, None);

        // Routing with no workers must fail
        let (req_fail, rx_fail) = make_request("before-register", isl);
        queue.enqueue(req_fail).await;
        let resp = rx_fail.await.expect("oneshot dropped");
        assert!(
            matches!(resp, Err(crate::types::KvSchedulerError::NoEndpoints)),
            "expected NoEndpoints before worker updates, got {resp:?}"
        );

        // Add two workers to the slot tracker, then publish their configs.
        slots.upsert_worker(WorkerDpRange::new(100, 0, 1)).unwrap();
        slots.upsert_worker(WorkerDpRange::new(200, 0, 1)).unwrap();

        // Also update the config watch so the selector can see these workers
        let mut configs = HashMap::new();
        for &id in &[100_u64, 200_u64] {
            configs.insert(
                id,
                SimpleWorkerConfig {
                    max_num_batched_tokens: Some(isl as u64),
                    ..Default::default()
                },
            );
        }
        cfg_tx.send(configs).unwrap();

        // Routing after registration must succeed and pick one of the registered workers
        let (req_ok, rx_ok) = make_request("after-register", isl);
        queue.enqueue(req_ok).await;
        let resp = rx_ok
            .await
            .expect("oneshot dropped")
            .expect("scheduling failed");
        assert!(
            resp.best_worker.worker_id == 100 || resp.best_worker.worker_id == 200,
            "expected worker 100 or 200, got {}",
            resp.best_worker.worker_id
        );

        // Clean up
        slots
            .mark_prefill_completed(&"after-register".to_string(), decay_now())
            .unwrap();
        slots
            .free(&"after-register".to_string(), decay_now())
            .unwrap();
    }

    /// Upserting a new worker preserves existing workers.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_worker_updates_are_additive() {
        let block_size = 16;
        let isl = 256;

        let (queue, slots, cfg_tx) = make_queue_with_sender(0, block_size, isl, None, None);

        // Register worker 10 in slots and config
        slots.upsert_worker(WorkerDpRange::new(10, 0, 1)).unwrap();

        let mut configs = HashMap::new();
        configs.insert(
            10_u64,
            SimpleWorkerConfig {
                max_num_batched_tokens: Some(isl as u64),
                ..Default::default()
            },
        );
        cfg_tx.send(configs.clone()).unwrap();

        // Register worker 20 (worker 10 must NOT be evicted)
        slots.upsert_worker(WorkerDpRange::new(20, 0, 1)).unwrap();

        configs.insert(
            20_u64,
            SimpleWorkerConfig {
                max_num_batched_tokens: Some(isl as u64),
                ..Default::default()
            },
        );
        cfg_tx.send(configs).unwrap();

        // Send enough requests to statistically prove both workers are available
        let mut seen = std::collections::HashSet::new();
        for i in 0..20 {
            let req_id = format!("add-{i}");
            let (req, rx) = make_request(&req_id, isl);
            queue.enqueue(req).await;
            let resp = rx
                .await
                .expect("oneshot dropped")
                .expect("scheduling failed");
            seen.insert(resp.best_worker.worker_id);
            slots.mark_prefill_completed(&req_id, decay_now()).unwrap();
            slots.free(&req_id, decay_now()).unwrap();
        }

        assert!(
            seen.contains(&10) && seen.contains(&20),
            "both workers should be reachable after additive registration, saw: {seen:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn allowed_worker_request_joins_backlog_and_dispatches_within_allow_list() {
        let block_size = 16;
        let isl = 256;
        let (queue, slots) = make_queue(2, block_size, isl, Some(0.0));

        let (active_a, active_a_rx) = make_request("active-a", isl);
        queue.enqueue(active_a).await;
        let active_a_worker = active_a_rx.await.unwrap().unwrap().best_worker.worker_id;

        let (active_b, active_b_rx) = make_request("active-b", isl);
        queue.enqueue(active_b).await;
        active_b_rx.await.unwrap().unwrap();

        let (backlog_head, mut backlog_head_rx) = make_request("backlog-head", isl);
        queue.enqueue(backlog_head).await;
        assert_eq!(queue.pending_count(), 1);

        slots
            .mark_prefill_completed(&"active-a".to_string(), decay_now())
            .unwrap();
        slots.free(&"active-a".to_string(), decay_now()).unwrap();

        let (mut allowed, mut allowed_rx) = make_request("allowed", isl);
        allowed.allowed_worker_ids = Some(HashSet::from([active_a_worker]));
        queue.enqueue(allowed).await;
        assert_eq!(
            queue.pending_count(),
            1,
            "enqueue should dispatch the existing head before the allow-list request"
        );
        assert!(allowed_rx.try_recv().is_err());

        let backlog_head_worker = backlog_head_rx
            .try_recv()
            .expect("existing backlog head should be admitted")
            .expect("backlog head failed")
            .best_worker
            .worker_id;
        assert!(allowed_rx.try_recv().is_err());

        slots
            .mark_prefill_completed(&"backlog-head".to_string(), decay_now())
            .unwrap();
        slots
            .free(&"backlog-head".to_string(), decay_now())
            .unwrap();
        queue.update().await;

        let allowed_worker = allowed_rx.await.unwrap().unwrap().best_worker.worker_id;
        assert_eq!(allowed_worker, active_a_worker);

        for request_id in ["active-b", "allowed"] {
            slots
                .mark_prefill_completed(&request_id.to_string(), decay_now())
                .unwrap();
            slots.free(&request_id.to_string(), decay_now()).unwrap();
        }
        assert_eq!(backlog_head_worker, active_a_worker);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_pinned_worker_conflict_with_allowed_ids_fails_early() {
        let (queue, _slots) = make_queue(1, 16, 256, Some(0.0));
        let (mut req, rx) = make_request("conflict", 256);
        req.pinned_worker = Some(WorkerWithDpRank::new(0, 0));
        req.allowed_worker_ids = Some(HashSet::from([1]));

        queue.enqueue(req).await;

        let resp = rx.await.expect("oneshot dropped");
        assert!(matches!(
            resp,
            Err(KvSchedulerError::PinnedWorkerNotAllowed { worker_id: 0 })
        ));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_disallowed_worker_ids_fail_without_queueing() {
        let (queue, _slots) = make_queue(1, 16, 256, Some(0.0));
        let (mut req, rx) = make_request("disallowed", 256);
        req.allowed_worker_ids = Some(HashSet::from([999]));

        queue.enqueue(req).await;

        let resp = rx.await.expect("oneshot dropped");
        assert!(matches!(resp, Err(KvSchedulerError::NoEndpoints)));
        assert_eq!(queue.pending_count(), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_incompatible_required_taints_fail_without_queueing() {
        let (queue, _slots, cfg_tx) = make_queue_with_sender(1, 16, 256, Some(0.0), None);
        let mut configs = HashMap::new();
        configs.insert(
            0_u64,
            SimpleWorkerConfig {
                max_num_batched_tokens: Some(256),
                taints: HashSet::from(["mdc-a".to_string()]),
                ..Default::default()
            },
        );
        cfg_tx.send(configs).unwrap();

        let (mut req, rx) = make_request("tainted", 256);
        req.routing_constraints = crate::protocols::RoutingConstraints {
            required_taints: HashSet::from(["mdc-b".to_string()]),
            preferred_taints: HashMap::new(),
        };

        queue.enqueue(req).await;

        let resp = rx.await.expect("oneshot dropped");
        assert!(matches!(resp, Err(KvSchedulerError::NoEndpoints)));
        assert_eq!(queue.pending_count(), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_blocked_pinned_lane_does_not_block_other_worker() {
        let (queue, slots) = make_queue(2, 16, 256, Some(0.0));

        let (mut first, first_rx) = make_request("pinned-1", 256);
        first.pinned_worker = Some(WorkerWithDpRank::new(1, 0));
        queue.enqueue(first).await;
        let first_resp = first_rx.await.unwrap().unwrap();
        assert_eq!(first_resp.best_worker, WorkerWithDpRank::new(1, 0));

        let (mut second, mut second_rx) = make_request("pinned-2", 256);
        second.pinned_worker = Some(WorkerWithDpRank::new(1, 0));
        queue.enqueue(second).await;
        assert_eq!(queue.pending_count(), 1);
        assert!(
            second_rx.try_recv().is_err(),
            "request should remain queued"
        );

        let (mut other_worker, mut other_worker_rx) = make_request("pinned-0", 256);
        other_worker.pinned_worker = Some(WorkerWithDpRank::new(0, 0));
        queue.enqueue(other_worker).await;
        assert_eq!(queue.pending_count(), 1);
        let other_worker_resp = other_worker_rx
            .try_recv()
            .expect("other worker request should have been scheduled")
            .expect("scheduling returned error");
        assert_eq!(other_worker_resp.best_worker, WorkerWithDpRank::new(0, 0));
        assert!(
            second_rx.try_recv().is_err(),
            "pinned request should still be queued"
        );

        slots
            .mark_prefill_completed(&"pinned-1".to_string(), decay_now())
            .unwrap();
        slots.free(&"pinned-1".to_string(), decay_now()).unwrap();
        queue.capacity_changed(Some(WorkerWithDpRank::new(1, 0)));

        let second_resp = tokio::time::timeout(Duration::from_secs(1), second_rx)
            .await
            .expect("capacity notification should schedule the pinned request")
            .expect("response channel should remain open");
        let second_resp = second_resp.expect("scheduling returned error");
        assert_eq!(second_resp.best_worker, WorkerWithDpRank::new(1, 0));
        assert_eq!(queue.pending_count(), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_queue_prefill_busy_check_ignores_untracked_prefill_tokens() {
        let (queue, slots) = make_queue(1, 16, 256, Some(0.0));

        let (mut req1, rx1) = make_request("req-1", 256);
        req1.track_prefill_tokens = false;
        queue.enqueue(req1).await;
        let _resp1 = rx1.await.unwrap().unwrap();
        assert_eq!(
            slots
                .active_tokens(decay_now())
                .get(&WorkerWithDpRank::new(0, 0))
                .copied(),
            Some(0)
        );

        let (req2, rx2) = make_request("req-2", 256);
        queue.enqueue(req2).await;
        let _resp2 = rx2.await.unwrap().unwrap();
        assert_eq!(queue.pending_count(), 0);

        let _ = slots.mark_prefill_completed(&"req-1".to_string(), decay_now());
        let _ = slots.free(&"req-1".to_string(), decay_now());
        let _ = slots.mark_prefill_completed(&"req-2".to_string(), decay_now());
        let _ = slots.free(&"req-2".to_string(), decay_now());
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn update_refresh_can_change_selected_worker_after_queue_wait() {
        let block_size = 16u32;
        let isl = 64usize;
        let refresher = Arc::new(CountingRefresher {
            calls: AtomicUsize::new(0),
            last_retain_kv_transfer_chain: AtomicBool::new(false),
            response: RefreshedOverlap {
                kv_transfer_candidates: Some(KvTransferCandidates {
                    block_hashes: vec![
                        ExternalSequenceBlockHash(101),
                        ExternalSequenceBlockHash(102),
                    ],
                    owner_prefix_blocks: vec![(WorkerWithDpRank::new(1, 0).into(), 2)],
                    source_intervals: Vec::new(),
                    routing_snapshot: None,
                }),
                overlap: OverlapSignals {
                    tier_overlap_blocks: Default::default(),
                    effective_overlap_blocks: FxHashMap::from_iter([
                        (WorkerWithDpRank::new(0, 0), 1.0),
                        (WorkerWithDpRank::new(1, 0), 9.0),
                    ]),
                    effective_cached_tokens: FxHashMap::from_iter([
                        (WorkerWithDpRank::new(0, 0), 16),
                        (WorkerWithDpRank::new(1, 0), 144),
                    ]),
                },
            },
        });
        let (queue, slots) =
            make_queue_with_refresher(2, block_size, isl, Some(0.0), refresher.clone());

        let (mut req1, rx1) = make_request("req-1", isl);
        req1.overlap
            .effective_overlap_blocks
            .insert(WorkerWithDpRank::new(0, 0), 3.0);
        req1.overlap
            .effective_cached_tokens
            .insert(WorkerWithDpRank::new(0, 0), 48);
        queue.enqueue(req1).await;
        let resp1 = rx1.await.expect("rx1 dropped").expect("req-1 failed");
        assert_eq!(resp1.best_worker, WorkerWithDpRank::new(0, 0));

        let (mut req2, rx2) = make_request("req-2", isl);
        req2.overlap
            .effective_overlap_blocks
            .insert(WorkerWithDpRank::new(1, 0), 3.0);
        req2.overlap
            .effective_cached_tokens
            .insert(WorkerWithDpRank::new(1, 0), 48);
        queue.enqueue(req2).await;
        let resp2 = rx2.await.expect("rx2 dropped").expect("req-2 failed");
        assert_eq!(resp2.best_worker, WorkerWithDpRank::new(1, 0));

        let (mut req3, rx3) = make_request("req-3", isl);
        req3.retain_kv_transfer_chain = true;
        req3.overlap
            .effective_overlap_blocks
            .insert(WorkerWithDpRank::new(0, 0), 8.0);
        req3.overlap
            .effective_overlap_blocks
            .insert(WorkerWithDpRank::new(1, 0), 2.0);
        req3.overlap
            .effective_cached_tokens
            .insert(WorkerWithDpRank::new(0, 0), 128);
        req3.overlap
            .effective_cached_tokens
            .insert(WorkerWithDpRank::new(1, 0), 32);
        queue
            .enqueue_with_block_hashes(req3, Some(vec![LocalBlockHash(42)]))
            .await;
        assert_eq!(queue.pending_count(), 1);
        assert_eq!(refresher.calls.load(Ordering::Relaxed), 0);

        tokio::time::advance(Duration::from_secs(11)).await;

        slots.free(&"req-1".to_string(), decay_now()).unwrap();
        slots.free(&"req-2".to_string(), decay_now()).unwrap();
        queue.update().await;

        let resp3 = rx3.await.expect("rx3 dropped").expect("req-3 failed");
        assert_eq!(refresher.calls.load(Ordering::Relaxed), 1);
        assert!(
            refresher
                .last_retain_kv_transfer_chain
                .load(Ordering::Relaxed)
        );
        assert_eq!(resp3.best_worker, WorkerWithDpRank::new(1, 0));
        assert_eq!(resp3.effective_overlap_blocks, 9.0);
        assert_eq!(resp3.cached_tokens, 144);
        assert_eq!(
            resp3
                .kv_transfer_candidates
                .as_ref()
                .map(|candidates| candidates.owner_prefix_blocks.as_slice()),
            Some(&[(WorkerWithDpRank::new(1, 0).into(), 2)][..])
        );
        assert_eq!(queue.pending_count(), 0);
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn update_refresh_drops_kv_transfer_candidates_when_retention_disabled() {
        let block_size = 16u32;
        let isl = 64usize;
        let worker = WorkerWithDpRank::new(0, 0);
        let refresher = Arc::new(CountingRefresher {
            calls: AtomicUsize::new(0),
            last_retain_kv_transfer_chain: AtomicBool::new(true),
            response: RefreshedOverlap {
                kv_transfer_candidates: Some(KvTransferCandidates {
                    block_hashes: vec![ExternalSequenceBlockHash(101)],
                    owner_prefix_blocks: vec![(worker.into(), 1)],
                    source_intervals: Vec::new(),
                    routing_snapshot: None,
                }),
                overlap: OverlapSignals {
                    tier_overlap_blocks: Default::default(),
                    effective_overlap_blocks: FxHashMap::from_iter([(worker, 5.0)]),
                    effective_cached_tokens: FxHashMap::from_iter([(worker, 80)]),
                },
            },
        });
        let (queue, slots) =
            make_queue_with_refresher(1, block_size, isl, Some(0.0), refresher.clone());

        let (req1, rx1) = make_request("req-1", isl);
        queue.enqueue(req1).await;
        let _ = rx1.await.expect("rx1 dropped").expect("req-1 failed");

        let (req2, rx2) = make_request("req-2", isl);
        queue
            .enqueue_with_block_hashes(req2, Some(vec![LocalBlockHash(42)]))
            .await;
        assert_eq!(queue.pending_count(), 1);

        tokio::time::advance(Duration::from_secs(11)).await;
        slots.free(&"req-1".to_string(), decay_now()).unwrap();
        queue.update().await;

        let resp2 = rx2.await.expect("rx2 dropped").expect("req-2 failed");
        assert_eq!(refresher.calls.load(Ordering::Relaxed), 1);
        assert!(
            !refresher
                .last_retain_kv_transfer_chain
                .load(Ordering::Relaxed)
        );
        assert_eq!(resp2.best_worker, worker);
        assert_eq!(resp2.effective_overlap_blocks, 5.0);
        assert_eq!(resp2.cached_tokens, 80);
        assert!(resp2.kv_transfer_candidates.is_none());
        assert_eq!(queue.pending_count(), 0);
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn selected_request_dispatches_after_refresh_if_worker_becomes_busy() {
        let block_size = 16u32;
        let isl = 64usize;
        let worker = WorkerWithDpRank::new(0, 0);
        let refresher = Arc::new(BlockingRefresher::new(RefreshedOverlap::from_overlap(
            OverlapSignals {
                tier_overlap_blocks: Default::default(),
                effective_overlap_blocks: FxHashMap::from_iter([(worker, 7.0)]),
                effective_cached_tokens: FxHashMap::from_iter([(worker, 56)]),
            },
        )));
        let (queue, slots) = make_queue_with_blocking_refresher(
            1,
            block_size,
            isl,
            Some(0.0),
            refresher.clone(),
            ADMISSION_CHANNEL_CAPACITY,
        );

        let (req1, rx1) = make_request("req-1", isl);
        queue.enqueue(req1).await;
        let _ = rx1.await.expect("rx1 dropped").expect("req-1 failed");

        let (mut req2, rx2) = make_request("req-2", isl);
        req2.overlap
            .effective_overlap_blocks
            .insert(WorkerWithDpRank::new(0, 0), 4.0);
        req2.overlap
            .effective_cached_tokens
            .insert(WorkerWithDpRank::new(0, 0), 64);
        queue
            .enqueue_with_block_hashes(req2, Some(vec![LocalBlockHash(42)]))
            .await;
        assert_eq!(queue.pending_count(), 1);
        assert_eq!(
            queue.class_queue_stats(0).unwrap().pending_cached_tokens,
            64
        );

        slots
            .mark_prefill_completed(&"req-1".to_string(), decay_now())
            .unwrap();
        slots.free(&"req-1".to_string(), decay_now()).unwrap();

        tokio::time::advance(Duration::from_secs(11)).await;

        let update = {
            let queue = Arc::clone(&queue);
            tokio::spawn(async move {
                queue.update().await;
            })
        };
        refresher.wait_for_calls(1).await;
        assert_eq!(
            queue.pending_count(),
            0,
            "DRR-selected request must be removed before refresh"
        );
        assert_eq!(
            queue.class_queue_stats(0).unwrap().pending_cached_tokens,
            0,
            "queue counters must reflect the irrevocable dequeue"
        );

        slots
            .add_request(
                SequenceRequest {
                    request_id: "occupy-during-refresh".to_string(),
                    token_sequence: None,
                    track_prefill_tokens: true,
                    expected_output_tokens: None,
                    prefill_load_hint: Some(PrefillLoadHint {
                        initial_effective_prefill_tokens: isl,
                        expected_prefill_duration: None,
                    }),
                    worker,
                    lora_name: None,
                },
                decay_now(),
            )
            .unwrap();

        refresher.release_one();
        update.await.unwrap();

        let resp2 = rx2.await.expect("rx2 dropped").expect("req-2 failed");
        assert_eq!(refresher.calls.load(Ordering::Relaxed), 1);
        assert_eq!(resp2.best_worker, worker);
        assert_eq!(resp2.effective_overlap_blocks, 7.0);
        assert_eq!(resp2.cached_tokens, 56);
        assert_eq!(queue.pending_count(), 0);

        for request_id in ["occupy-during-refresh", "req-2"] {
            slots
                .mark_prefill_completed(&request_id.to_string(), decay_now())
                .unwrap();
            slots.free(&request_id.to_string(), decay_now()).unwrap();
        }
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn cancelled_enqueue_wait_keeps_cleanup_behind_command() {
        let block_size = 16u32;
        let isl = 64usize;
        let refresher = Arc::new(BlockingRefresher::new(RefreshedOverlap::default()));
        let (queue, slots) =
            make_queue_with_blocking_refresher(1, block_size, isl, Some(0.0), refresher.clone(), 1);

        let (active, active_rx) = make_request("active", isl);
        queue.enqueue(active).await;
        active_rx.await.unwrap().unwrap();

        let (queued, queued_rx) = make_request("queued", isl);
        queue
            .enqueue_with_block_hashes(queued, Some(vec![LocalBlockHash(42)]))
            .await;
        slots.free(&"active".to_owned(), decay_now()).unwrap();
        tokio::time::advance(Duration::from_secs(11)).await;

        let update = {
            let queue = Arc::clone(&queue);
            tokio::spawn(async move { queue.update().await })
        };
        refresher.wait_for_calls(1).await;

        // The waiter is cancelled as soon as its command is accepted: the lease travels
        // inside the command until the actor establishes request ownership.
        let (cancelled, cancelled_rx) = make_request("cancelled", isl);
        let lease = queue.new_request_lifecycle_lease(Some("cancelled"));
        let queue_metadata = queue
            .default_queue_metadata(&cancelled, Instant::now())
            .unwrap();
        let (ack_tx, ack_rx) = oneshot::channel();
        queue
            .admission_tx
            .send(AdmissionCommand::Enqueue {
                request: cancelled,
                attempt_tx: None,
                block_hashes: None,
                prepared: None,
                queue_metadata,
                lease,
                ack_tx,
            })
            .await
            .unwrap();
        drop(ack_rx);
        drop(cancelled_rx);

        // The command is applied during the refresh, which keeps running. The failed
        // acknowledgement returns the lease, whose cleanup follows the command: the waiter
        // entered the queue and leaves it before the update behind it is answered.
        update.await.unwrap();
        queue.update().await;
        assert_eq!(queue.pending_count(), 0);
        assert_eq!(class_outcomes(&queue, 0), (3, 0, 0));
        assert_eq!(refresher.calls.load(Ordering::Relaxed), 1);

        // Released, the refresh admits its candidate; the cleaned-up waiter never books.
        refresher.release_one();
        queued_rx.await.unwrap().unwrap();
        queue.update().await;
        assert_eq!(queue.pending_count(), 0);
        assert_eq!(refresher.calls.load(Ordering::Relaxed), 1);

        slots.free(&"queued".to_owned(), decay_now()).unwrap();
        slots.assert_completely_drained(decay_now());
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn continuation_drain_does_not_self_send_into_saturated_actor_channel() {
        let block_size = 16u32;
        let isl = 64usize;
        let refresher = Arc::new(BlockingRefresher::new(RefreshedOverlap::default()));
        let (queue, slots) =
            make_queue_with_blocking_refresher(2, block_size, isl, Some(0.0), refresher.clone(), 1);

        for id in ["active-0", "active-1"] {
            let (active, active_rx) = make_request(id, isl);
            queue.enqueue(active).await;
            active_rx.await.unwrap().unwrap();
        }
        let (queued, queued_rx) = make_request("queued", isl);
        queue
            .enqueue_with_block_hashes(queued, Some(vec![LocalBlockHash(42)]))
            .await;
        let (following, following_rx) = make_request("following", isl);
        queue.enqueue(following).await;
        for id in ["active-0", "active-1"] {
            slots
                .mark_prefill_completed(&id.to_string(), decay_now())
                .unwrap();
            slots.free(&id.to_string(), decay_now()).unwrap();
        }
        tokio::time::advance(Duration::from_secs(11)).await;

        let update = {
            let queue = Arc::clone(&queue);
            tokio::spawn(async move { queue.update().await })
        };
        refresher.wait_for_calls(1).await;
        // Hold every command permit while the drain is suspended in the refresh.
        let permits = queue
            .admission_tx
            .reserve_many(queue.admission_tx.max_capacity())
            .await
            .unwrap();
        assert_eq!(
            queue.admission_tx.capacity(),
            0,
            "test must saturate the actor command channel"
        );

        refresher.release_one();
        for response_rx in [queued_rx, following_rx] {
            tokio::time::timeout(Duration::from_secs(1), response_rx)
                .await
                .expect("the drain deadlocked with a full actor command channel")
                .unwrap()
                .unwrap();
        }
        update.await.unwrap();
        drop(permits);

        for id in ["queued", "following"] {
            slots
                .mark_prefill_completed(&id.to_string(), decay_now())
                .unwrap();
            slots.free(&id.to_string(), decay_now()).unwrap();
        }
        slots.assert_completely_drained(decay_now());
    }

    /// Enqueues request `id` into `policy_class` through the classifier path, with an
    /// explicit deadline when `due_at` is set, and router ingress now.
    async fn enqueue_to_class<Sel, RF>(
        queue: &SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, Sel, RF>,
        id: &str,
        policy_class: &str,
        due_at: Option<Instant>,
        block_hashes: Option<Vec<LocalBlockHash>>,
    ) -> SchedulingResponseReceiver
    where
        Sel: WorkerSelector<SimpleWorkerConfig> + Send + 'static,
        RF: OverlapScoresRefresh + Send + Sync + 'static,
    {
        let (mut request, response_rx) = make_request(id, 64);
        request.policy_class = Some(policy_class.to_string());
        let ingress_at = Instant::now();
        let classified = due_at.map(|due_at| {
            let mut classified = queue
                .build_classify_request(&request, ingress_at, WorkerInputs::NONE)
                .0;
            classified.set_due_at(due_at);
            classified
        });
        queue
            .enqueue_admitted_with_block_hashes_and_lease(
                request,
                block_hashes,
                None,
                None,
                classified,
                ingress_at,
            )
            .await;
        response_rx
    }

    /// Waits for a deadline rejection and returns its arrival time, counted from `start`.
    async fn rejected_after(response_rx: SchedulingResponseReceiver, start: Instant) -> Duration {
        assert!(matches!(
            response_rx.await.unwrap(),
            Err(KvSchedulerError::DeadlineExceeded)
        ));
        Instant::now() - start
    }

    /// The class's received total and its two rejection totals: due time passed, then queue
    /// size exceeded.
    fn class_outcomes<Sel, RF>(
        queue: &SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, Sel, RF>,
        class_index: usize,
    ) -> (u64, u64, u64)
    where
        Sel: WorkerSelector<SimpleWorkerConfig> + Send + 'static,
        RF: OverlapScoresRefresh + Send + Sync + 'static,
    {
        let stats = queue.class_queue_stats(class_index).unwrap();
        (
            stats.received_total,
            stats.rejected_due_time_passed_total,
            stats.rejected_queue_size_exceeded_total,
        )
    }

    fn free_requests(slots: &ActiveSequencesMultiWorker<NoopSequencePublisher>, ids: &[&str]) {
        for id in ids {
            slots.free(&id.to_string(), decay_now()).unwrap();
        }
    }

    /// Enqueues request `id` straight into a test actor, with an explicit deadline when
    /// `due_at` is set. Returns what `handle_enqueue` reports: whether ready work may have
    /// changed, as when the request entered the queue.
    fn enqueue_into_actor(
        queue: &SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, DefaultWorkerSelector>,
        actor: &mut SchedulerQueueActor<
            NoopSequencePublisher,
            SimpleWorkerConfig,
            DefaultWorkerSelector,
            NoopOverlapScoresRefresh,
        >,
        id: &str,
        due_at: Option<Instant>,
    ) -> (bool, SchedulingResponseReceiver) {
        let (request, response_rx) = make_request(id, 64);
        let queued = enqueue_request_into_actor(queue, actor, request, None, due_at);
        (queued, response_rx)
    }

    /// Enqueues `request` straight into a test actor, with an optional lifecycle transfer and
    /// explicit deadline. Returns what `handle_enqueue` reports, as [`enqueue_into_actor`] does.
    fn enqueue_request_into_actor(
        queue: &SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, DefaultWorkerSelector>,
        actor: &mut SchedulerQueueActor<
            NoopSequencePublisher,
            SimpleWorkerConfig,
            DefaultWorkerSelector,
            NoopOverlapScoresRefresh,
        >,
        request: SchedulingRequest,
        lifecycle_transfer: Option<Arc<AdmissionLifecycleTransfer>>,
        due_at: Option<Instant>,
    ) -> bool {
        let now = Instant::now();
        let mut metadata = queue.default_queue_metadata(&request, now).unwrap();
        metadata.due_at = due_at;
        actor.handle_enqueue(request, None, lifecycle_transfer, None, None, metadata)
    }

    /// Applies an enqueue of `request` as the actor loop applies a command, with an optional
    /// explicit deadline. Returns the drain the command calls for and its reply.
    fn apply_enqueue_command(
        queue: &SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, DefaultWorkerSelector>,
        actor: &mut SchedulerQueueActor<
            NoopSequencePublisher,
            SimpleWorkerConfig,
            DefaultWorkerSelector,
            NoopOverlapScoresRefresh,
        >,
        request: SchedulingRequest,
        due_at: Option<Instant>,
    ) -> (Drain, Option<CommandReply>) {
        let mut queue_metadata = queue
            .default_queue_metadata(&request, Instant::now())
            .unwrap();
        queue_metadata.due_at = due_at;
        let (ack_tx, _) = oneshot::channel();
        actor.apply_command(
            AdmissionCommand::Enqueue {
                request,
                attempt_tx: None,
                block_hashes: None,
                prepared: None,
                queue_metadata,
                lease: None,
                ack_tx,
            },
            true,
        )
    }

    struct BlockedSharedHead {
        queue:
            Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, DefaultWorkerSelector>>,
        slots: Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
        actor: SchedulerQueueActor<
            NoopSequencePublisher,
            SimpleWorkerConfig,
            DefaultWorkerSelector,
            NoopOverlapScoresRefresh,
        >,
        start: Instant,
        head_rx: SchedulingResponseReceiver,
        successor_rx: SchedulingResponseReceiver,
        overdue_rx: Option<SchedulingResponseReceiver>,
    }

    /// Two workers. Busy worker 0 blocks the head of class `a`'s shared queue, which may run only
    /// there and is due at `head_due_ms`. Behind it waits a `successor_isl`-token successor that
    /// idle worker 1 could serve: a shared queue does not search past its blocked head. Rejecting
    /// the head takes `head_rejection` of the paused clock. With `overdue`, class `b` holds an
    /// entry for worker 0, due at 99 ms, whose rejection takes 2 ms. Returns at 100 ms.
    async fn blocked_shared_head(
        profile: PolicyProfile,
        head_due_ms: u64,
        head_rejection: Duration,
        successor_isl: usize,
        overdue: bool,
    ) -> BlockedSharedHead {
        let (queue, slots) = make_queue_with_profile(2, 16, 64, profile);
        let mut actor = actor_for_test(&queue, &slots);
        let start = Instant::now();
        let at = |millis| Some(start + Duration::from_millis(millis));
        let pinned = |id: &str, class: &str| {
            let (mut request, response_rx) = make_request(id, 16);
            request.pinned_worker = Some(WorkerWithDpRank::new(0, 0));
            request.policy_class = Some(class.to_string());
            (request, response_rx)
        };
        let (active, mut active_rx) = pinned("active", "a");
        assert!(enqueue_request_into_actor(
            &queue, &mut actor, active, None, None
        ));
        actor.handle_update(None).await;
        active_rx.try_recv().unwrap().unwrap();

        let (mut head, mut head_rx) = make_request("head", 16);
        head.allowed_worker_ids = Some(HashSet::from([0]));
        head.policy_class = Some("a".to_string());
        assert!(enqueue_request_into_actor(
            &queue,
            &mut actor,
            head,
            None,
            at(head_due_ms)
        ));
        if !head_rejection.is_zero() {
            advance_clock_on_response(&mut head_rx, head_rejection);
        }
        let (mut successor, mut successor_rx) = make_request("successor", successor_isl);
        successor.policy_class = Some("a".to_string());
        assert!(enqueue_request_into_actor(
            &queue, &mut actor, successor, None, None
        ));
        let mut overdue_rx = None;
        if overdue {
            let (request, mut response_rx) = pinned("overdue", "b");
            assert!(enqueue_request_into_actor(
                &queue,
                &mut actor,
                request,
                None,
                at(99)
            ));
            advance_clock_on_response(&mut response_rx, Duration::from_millis(2));
            overdue_rx = Some(response_rx);
        }
        actor.handle_update(None).await;
        assert!(matches!(
            successor_rx.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));
        tokio::time::advance(Duration::from_millis(100)).await;
        BlockedSharedHead {
            queue,
            slots,
            actor,
            start,
            head_rx,
            successor_rx,
            overdue_rx,
        }
    }

    /// Classes `first` and `second`, each with a 100-token quantum and no queueing threshold.
    fn two_class_profile() -> PolicyProfile {
        policy_profile(
            Some("first"),
            vec![
                PolicyClassConfig {
                    busy_thresholds: BusyThresholds {
                        prefill_busy_threshold_frac: Some(0.0),
                        ..Default::default()
                    },
                    ..class("first", 100)
                },
                PolicyClassConfig {
                    busy_thresholds: BusyThresholds {
                        prefill_busy_threshold_frac: Some(0.0),
                        ..Default::default()
                    },
                    ..class("second", 100)
                },
            ],
        )
    }

    #[tokio::test(start_paused = true)]
    async fn class_queues_expire_on_independent_timers() {
        let profile = policy_profile(
            Some("fast"),
            vec![
                PolicyClassConfig {
                    slo_ttft_ms: NonZeroU32::new(100),
                    controlled_delay: true,
                    busy_thresholds: BusyThresholds {
                        prefill_busy_threshold_frac: Some(0.0),
                        ..Default::default()
                    },
                    ..class("fast", 1)
                },
                PolicyClassConfig {
                    slo_ttft_ms: NonZeroU32::new(200),
                    controlled_delay: true,
                    busy_thresholds: BusyThresholds {
                        prefill_busy_threshold_frac: Some(0.0),
                        ..Default::default()
                    },
                    ..class("medium", 1)
                },
                PolicyClassConfig {
                    slo_ttft_ms: NonZeroU32::new(300),
                    controlled_delay: true,
                    busy_thresholds: BusyThresholds {
                        prefill_busy_threshold_frac: Some(0.0),
                        ..Default::default()
                    },
                    ..class("slow", 1)
                },
            ],
        );
        let (queue, slots) = make_queue_with_profile(1, 16, 64, profile);
        let start = Instant::now();
        let millis = Duration::from_millis;
        let active = enqueue_to_class(&queue, "active", "fast", None, None).await;
        active.await.unwrap().unwrap();
        // The worker stays busy, so only the class timers wake the actor.
        let fast = enqueue_to_class(&queue, "fast", "fast", None, None).await;
        let medium = enqueue_to_class(&queue, "medium", "medium", None, None).await;
        let slow = enqueue_to_class(&queue, "slow", "slow", None, None).await;
        tokio::time::advance(millis(10)).await;
        // An earlier deadline moves only the slow queue's timer.
        let slow_explicit = enqueue_to_class(
            &queue,
            "slow-explicit",
            "slow",
            Some(start + millis(150)),
            None,
        )
        .await;
        tokio::time::advance(millis(10)).await;
        let medium_later = enqueue_to_class(&queue, "medium-later", "medium", None, None).await;

        assert_eq!(rejected_after(fast, start).await, millis(100));
        tokio::time::advance(millis(20)).await;
        let fast_later = enqueue_to_class(&queue, "fast-later", "fast", None, None).await;
        assert_eq!(rejected_after(slow_explicit, start).await, millis(150));
        assert_eq!(rejected_after(medium, start).await, millis(200));
        // Two queues due at the same instant both expire on that wake.
        assert_eq!(rejected_after(medium_later, start).await, millis(220));
        assert_eq!(rejected_after(fast_later, start).await, millis(220));
        assert_eq!(rejected_after(slow, start).await, millis(300));
        // An emptied queue's timer rearms when an arrival refills it.
        let refill = enqueue_to_class(&queue, "refill", "fast", None, None).await;
        assert_eq!(rejected_after(refill, start).await, millis(400));

        assert_eq!(queue.pending_count(), 0);
        for (class_index, outcomes) in [(0, (4, 3, 0)), (1, (2, 2, 0)), (2, (2, 2, 0))] {
            assert_eq!(class_outcomes(&queue, class_index), outcomes);
        }
        free_requests(&slots, &["active"]);
        slots.assert_completely_drained(decay_now());
    }

    #[tokio::test(start_paused = true)]
    async fn class_timer_follows_its_queue_head_through_cancellation_and_refill() {
        let profile = policy_profile(
            None,
            vec![PolicyClassConfig {
                slo_ttft_ms: NonZeroU32::new(100),
                controlled_delay: true,
                busy_thresholds: BusyThresholds {
                    prefill_busy_threshold_frac: Some(0.0),
                    ..Default::default()
                },
                ..class("standard", 1)
            }],
        );
        let (queue, slots) = make_queue_with_profile(1, 16, 64, profile);
        let mut actor = actor_for_test(&queue, &slots);
        let start = Instant::now();
        let at = |millis| Some(start + Duration::from_millis(millis));
        let metadata = |request: &SchedulingRequest| {
            queue
                .default_queue_metadata(request, Instant::now())
                .unwrap()
        };

        let (active, mut active_rx) = make_request("active", 64);
        let active_metadata = metadata(&active);
        assert!(actor.handle_enqueue(active, None, None, None, None, active_metadata));
        actor.handle_update(None).await;
        active_rx.try_recv().unwrap().unwrap();
        assert_eq!(actor.class_timers.armed[0], None);

        let head_transfer = Arc::new(AdmissionLifecycleTransfer::new("head".to_string()));
        let (head, mut head_rx) = make_request("head", 64);
        let head_metadata = metadata(&head);
        assert!(actor.handle_enqueue(
            head,
            None,
            Some(Arc::clone(&head_transfer)),
            None,
            None,
            head_metadata
        ));
        assert_eq!(actor.class_timers.armed[0], at(100));
        tokio::time::advance(Duration::from_millis(50)).await;
        let (later, mut later_rx) = make_request("later", 64);
        let later_metadata = metadata(&later);
        assert!(actor.handle_enqueue(later, None, None, None, None, later_metadata));
        assert_eq!(actor.class_timers.armed[0], at(100));

        // Cancelling the head moves the timer to its successor's due time.
        actor.cleanup.enqueue(AdmissionCleanupEntry {
            target: SchedulerCleanupTarget::AdmissionLifecycle(head_transfer),
        });
        actor.drain_cleanup();
        assert!(matches!(
            head_rx.try_recv(),
            Err(oneshot::error::TryRecvError::Closed)
        ));
        assert_eq!(actor.class_timers.armed[0], at(150));
        // The head's old due time passes without a rejection.
        tokio::time::advance(Duration::from_millis(50)).await;
        actor.handle_update(None).await;
        assert!(matches!(
            later_rx.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));
        tokio::time::advance(Duration::from_millis(50)).await;
        actor.handle_update(None).await;
        assert!(matches!(
            later_rx.try_recv(),
            Ok(Err(KvSchedulerError::DeadlineExceeded))
        ));
        assert_eq!(actor.class_timers.armed[0], None);
        assert!(actor.class_timers.by_deadline.is_empty());

        let (refill, _refill_rx) = make_request("refill", 64);
        let refill_metadata = metadata(&refill);
        assert!(actor.handle_enqueue(refill, None, None, None, None, refill_metadata));
        assert_eq!(actor.class_timers.armed[0], at(250));
        // The cancelled head was never counted as a rejection.
        assert_eq!(class_outcomes(&queue, 0), (4, 1, 0));
        slots.free(&"active".to_string(), decay_now()).unwrap();
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn stalled_refresh_expires_its_candidate_without_release_or_drr_credit() {
        let profile = two_class_profile();
        let refresher = Arc::new(BlockingRefresher::new(RefreshedOverlap::default()));
        let (queue, slots) = make_queue_with_profile_and_blocking_refresher(
            1,
            16,
            64,
            profile,
            Arc::clone(&refresher),
            ADMISSION_CHANNEL_CAPACITY,
        );
        let start = Instant::now();
        // Admitting the busy request from the second queue starts the next DRR ring at the
        // first.
        let active = enqueue_to_class(&queue, "active", "second", None, None).await;
        active.await.unwrap().unwrap();
        let expiring = enqueue_to_class(
            &queue,
            "expiring",
            "first",
            Some(start + Duration::from_millis(11_100)),
            Some(vec![LocalBlockHash(41)]),
        )
        .await;
        let mut successor = enqueue_to_class(&queue, "successor", "first", None, None).await;
        let mut other = enqueue_to_class(&queue, "other", "second", None, None).await;
        tokio::time::advance(Duration::from_secs(11)).await;
        slots
            .mark_prefill_completed(&"active".to_string(), decay_now())
            .unwrap();
        slots.free(&"active".to_string(), decay_now()).unwrap();
        // The update is acknowledged before the drain awaits the refresh.
        queue.update().await;
        assert_eq!(refresher.calls.load(Ordering::Relaxed), 1);

        // The refresh is never released. Its candidate is rejected at its due time, which comes
        // before the refresh timeout.
        assert_eq!(
            rejected_after(expiring, start).await,
            Duration::from_millis(11_100)
        );
        queue.update().await;
        // The rejected candidate spent none of its class's credit, so the same class keeps
        // the turn: its successor takes the worker ahead of the other class.
        successor.try_recv().unwrap().unwrap();
        assert!(matches!(
            other.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));
        assert_eq!(refresher.calls.load(Ordering::Relaxed), 1);
        assert_eq!(class_outcomes(&queue, 0), (2, 1, 0));
        assert_eq!(class_outcomes(&queue, 1), (2, 0, 0));

        slots.free(&"successor".to_string(), decay_now()).unwrap();
        queue.update().await;
        other.try_recv().unwrap().unwrap();
        free_requests(&slots, &["other"]);
        slots.assert_completely_drained(decay_now());
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn another_class_expires_while_a_candidate_refresh_stalls() {
        let profile = policy_profile(
            Some("refreshing"),
            vec![
                PolicyClassConfig {
                    busy_thresholds: BusyThresholds {
                        prefill_busy_threshold_frac: Some(0.0),
                        ..Default::default()
                    },
                    ..class("refreshing", 100)
                },
                PolicyClassConfig {
                    slo_ttft_ms: NonZeroU32::new(11100),
                    controlled_delay: true,
                    busy_thresholds: BusyThresholds {
                        prefill_busy_threshold_frac: Some(0.0),
                        ..Default::default()
                    },
                    ..class("budgeted", 100)
                },
            ],
        );
        let worker = WorkerWithDpRank::new(0, 0);
        let refreshed = RefreshedOverlap::from_overlap(OverlapSignals {
            tier_overlap_blocks: Default::default(),
            effective_overlap_blocks: FxHashMap::from_iter([(worker, 3.0)]),
            effective_cached_tokens: FxHashMap::from_iter([(worker, 48)]),
        });
        let refresher = Arc::new(BlockingRefresher::new(refreshed));
        let (queue, slots) = make_queue_with_profile_and_blocking_refresher(
            1,
            16,
            64,
            profile,
            Arc::clone(&refresher),
            ADMISSION_CHANNEL_CAPACITY,
        );
        let start = Instant::now();
        let active = enqueue_to_class(&queue, "active", "budgeted", None, None).await;
        active.await.unwrap().unwrap();
        let mut candidate = enqueue_to_class(
            &queue,
            "candidate",
            "refreshing",
            None,
            Some(vec![LocalBlockHash(42)]),
        )
        .await;
        let budgeted = enqueue_to_class(&queue, "budgeted", "budgeted", None, None).await;
        tokio::time::advance(Duration::from_secs(11)).await;
        slots
            .mark_prefill_completed(&"active".to_string(), decay_now())
            .unwrap();
        slots.free(&"active".to_string(), decay_now()).unwrap();
        queue.update().await;
        assert_eq!(refresher.calls.load(Ordering::Relaxed), 1);

        // The other queue's timer fires during the stalled refresh, before the refresh timeout.
        // That queue rejects its own overdue request on time while the refresh keeps running, and
        // the candidate, which has no due time, is neither expired nor admitted.
        assert_eq!(
            rejected_after(budgeted, start).await,
            Duration::from_millis(11_100)
        );
        assert!(matches!(
            candidate.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));
        // The same refresh, once released, admits the candidate with the refreshed scores.
        refresher.release_one();
        assert_eq!(candidate.await.unwrap().unwrap().cached_tokens, 48);
        assert_eq!(refresher.calls.load(Ordering::Relaxed), 1);
        assert_eq!(class_outcomes(&queue, 0), (1, 0, 0));
        assert_eq!(class_outcomes(&queue, 1), (2, 1, 0));
        free_requests(&slots, &["candidate"]);
        slots.assert_completely_drained(decay_now());
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn commands_during_a_refresh_are_answered_without_abandoning_it() {
        let refresher = Arc::new(BlockingRefresher::new(RefreshedOverlap::default()));
        let (queue, slots) = make_queue_with_blocking_refresher(
            3,
            16,
            64,
            Some(0.0),
            Arc::clone(&refresher),
            ADMISSION_CHANNEL_CAPACITY,
        );
        let busy = ["active-0", "active-1", "active-2"];
        for id in busy {
            let (active, active_rx) = make_request(id, 64);
            queue.enqueue(active).await;
            active_rx.await.unwrap().unwrap();
        }
        let mut receivers = Vec::new();
        for (id, hash) in [("first", 1), ("second", 2)] {
            let (request, response_rx) = make_request(id, 64);
            queue
                .enqueue_with_block_hashes(request, Some(vec![LocalBlockHash(hash)]))
                .await;
            receivers.push(response_rx);
        }
        tokio::time::advance(Duration::from_secs(11)).await;
        free_requests(&slots, &busy);
        tokio::time::timeout(Duration::from_secs(1), queue.update())
            .await
            .expect("the update reply waited for a stalled refresh");
        assert_eq!(refresher.calls.load(Ordering::Relaxed), 1);

        // A lifecycle-tracked arrival is applied during the first refresh and answered at once,
        // and the refresh keeps running.
        let (arrival, arrival_rx) = make_request("arrival", 64);
        let lease = queue.new_request_lifecycle_lease(Some("arrival"));
        let lease = tokio::time::timeout(
            Duration::from_secs(1),
            queue.enqueue_with_block_hashes_and_lease(arrival, None, lease),
        )
        .await
        .expect("the arrival's reply waited for the refresh")
        .expect("the actor returns the arrival's lease");
        assert_eq!(queue.pending_count(), 2);

        // Cancelling the arrival is handled during the same refresh: its cleanup removes it from
        // the queue before the update behind it is answered.
        drop(arrival_rx);
        drop(lease);
        tokio::time::timeout(Duration::from_secs(1), queue.update())
            .await
            .expect("the update reply waited for the refresh");
        assert_eq!(queue.pending_count(), 1);
        assert_eq!(refresher.calls.load(Ordering::Relaxed), 1);
        let mut receivers = receivers.into_iter();
        let mut first_rx = receivers.next().unwrap();
        assert!(matches!(
            first_rx.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));

        // Released, the first refresh admits its candidate, and the drain refreshes the second.
        refresher.release_one();
        first_rx.await.unwrap().unwrap();
        refresher.wait_for_calls(2).await;
        refresher.release_one();
        receivers.next().unwrap().await.unwrap().unwrap();
        queue.update().await;
        assert_eq!(queue.pending_count(), 0);
        assert_eq!(refresher.calls.load(Ordering::Relaxed), 2);
        assert_eq!(class_outcomes(&queue, 0), (6, 0, 0));
        free_requests(&slots, &["first", "second"]);
        slots.assert_completely_drained(decay_now());
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn a_pending_refresh_completes_under_ordinary_command_traffic() {
        let worker = WorkerWithDpRank::new(0, 0);
        let refreshed = RefreshedOverlap::from_overlap(OverlapSignals {
            tier_overlap_blocks: Default::default(),
            effective_overlap_blocks: FxHashMap::from_iter([(worker, 3.0)]),
            effective_cached_tokens: FxHashMap::from_iter([(worker, 48)]),
        });
        let refresher = Arc::new(BlockingRefresher::new(refreshed));
        let (queue, slots) = make_queue_with_blocking_refresher(
            1,
            16,
            64,
            Some(0.0),
            Arc::clone(&refresher),
            ADMISSION_CHANNEL_CAPACITY,
        );
        let (active, active_rx) = make_request("active", 64);
        queue.enqueue(active).await;
        active_rx.await.unwrap().unwrap();
        let (candidate, mut candidate_rx) = make_request("candidate", 64);
        queue
            .enqueue_with_block_hashes(candidate, Some(vec![LocalBlockHash(42)]))
            .await;
        tokio::time::advance(Duration::from_secs(11)).await;
        free_requests(&slots, &["active"]);
        queue.update().await;
        assert_eq!(refresher.calls.load(Ordering::Relaxed), 1);

        // While the lookup is pending, every kind of ordinary command and a capacity wake arrive
        // in each round. Each command is answered while the refresh runs; none abandons it.
        let mut arrivals = Vec::new();
        for round in 0..8 {
            let id = format!("arrival-{round}");
            let (arrival, arrival_rx) = make_request(&id, 64);
            tokio::time::timeout(Duration::from_secs(1), queue.enqueue(arrival))
                .await
                .expect("an arrival waited for the refresh");
            arrivals.push((id, arrival_rx));
            let (advisory, _advisory_rx) = make_request(&format!("advisory-{round}"), 64);
            tokio::time::timeout(
                Duration::from_secs(1),
                queue.select_without_admission(advisory),
            )
            .await
            .expect("a selection without admission waited for the refresh")
            .unwrap();
            // The default selector has no runtime policy: the read is answered as unsupported.
            assert!(
                tokio::time::timeout(Duration::from_secs(1), queue.policy_control().get())
                    .await
                    .expect("a policy read waited for the refresh")
                    .is_err()
            );
            queue.capacity_changed(Some(worker));
            tokio::time::timeout(Duration::from_secs(1), queue.update())
                .await
                .expect("an update waited for the refresh");
        }
        assert_eq!(refresher.calls.load(Ordering::Relaxed), 1);
        assert!(matches!(
            candidate_rx.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));
        assert_eq!(queue.pending_count(), 8);

        // Released, the refresh that ran through all of that traffic admits the candidate with
        // the refreshed scores rather than the queued ones.
        refresher.release_one();
        assert_eq!(candidate_rx.await.unwrap().unwrap().cached_tokens, 48);
        assert_eq!(refresher.calls.load(Ordering::Relaxed), 1);

        // The arrivals queued behind the candidate run in order as the worker frees.
        let mut previous = "candidate".to_string();
        for (id, arrival_rx) in arrivals {
            free_requests(&slots, &[&previous]);
            queue.update().await;
            arrival_rx.await.unwrap().unwrap();
            previous = id;
        }
        free_requests(&slots, &[&previous]);
        assert_eq!(queue.pending_count(), 0);
        assert_eq!(refresher.calls.load(Ordering::Relaxed), 1);
        assert_eq!(class_outcomes(&queue, 0), (10, 0, 0));
        slots.assert_completely_drained(decay_now());
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn a_refresh_timeout_runs_from_the_refresh_start_through_scheduler_traffic() {
        let worker = WorkerWithDpRank::new(0, 0);
        let refreshed = RefreshedOverlap::from_overlap(OverlapSignals {
            tier_overlap_blocks: Default::default(),
            effective_overlap_blocks: FxHashMap::from_iter([(worker, 3.0)]),
            effective_cached_tokens: FxHashMap::from_iter([(worker, 48)]),
        });
        // A lookup released before the timeout supplies its scores. One never released is
        // dropped at the timeout, counted from the refresh start, and the candidate keeps its
        // queued scores.
        for (release, admitted_after, cached_tokens) in [
            (true, Duration::from_millis(200), 48),
            (false, OVERLAP_REFRESH_TIMEOUT, 0),
        ] {
            let profile = two_class_profile();
            let refresher = Arc::new(BlockingRefresher::new(refreshed.clone()));
            let (queue, slots) = make_queue_with_profile_and_blocking_refresher(
                2,
                16,
                64,
                profile,
                Arc::clone(&refresher),
                ADMISSION_CHANNEL_CAPACITY,
            );
            // Admitting the busy requests from the second queue starts the next DRR ring at the
            // first.
            for id in ["active-0", "active-1"] {
                let active = enqueue_to_class(&queue, id, "second", None, None).await;
                active.await.unwrap().unwrap();
            }
            // Neither queued request has a due time, so only the timeout bounds the refresh.
            let mut candidate = enqueue_to_class(
                &queue,
                "candidate",
                "first",
                None,
                Some(vec![LocalBlockHash(42)]),
            )
            .await;
            let mut other = enqueue_to_class(&queue, "other", "second", None, None).await;
            tokio::time::advance(Duration::from_secs(11)).await;
            free_requests(&slots, &["active-0"]);
            queue.update().await;
            let refresh_start = Instant::now();
            assert_eq!(refresher.calls.load(Ordering::Relaxed), 1);

            // The other worker frees, and an update arrives every 100 ms. Each is answered while
            // the refresh holds back both queued requests, the other class's included.
            free_requests(&slots, &["active-1"]);
            for _ in 0..2 {
                tokio::time::sleep(Duration::from_millis(100)).await;
                queue.update().await;
                assert!(matches!(
                    candidate.try_recv(),
                    Err(oneshot::error::TryRecvError::Empty)
                ));
                assert!(matches!(
                    other.try_recv(),
                    Err(oneshot::error::TryRecvError::Empty)
                ));
            }
            if release {
                refresher.release_one();
            }
            // Dispatch held past the timeout fails here rather than waiting on.
            let bound = refresh_start + OVERLAP_REFRESH_TIMEOUT + Duration::from_millis(1);
            let admitted = tokio::time::timeout_at(bound, candidate)
                .await
                .expect("the candidate was not dispatched by the refresh timeout");
            assert_eq!(admitted.unwrap().unwrap().cached_tokens, cached_tokens);
            assert_eq!(Instant::now() - refresh_start, admitted_after);
            // The drain goes on to the other class at once.
            tokio::time::timeout_at(bound, other)
                .await
                .expect("the other class was not dispatched by the refresh timeout")
                .unwrap()
                .unwrap();
            assert_eq!(Instant::now() - refresh_start, admitted_after);
            assert_eq!(refresher.calls.load(Ordering::Relaxed), 1);
            free_requests(&slots, &["candidate", "other"]);
            assert_eq!(queue.pending_count(), 0);
            slots.assert_completely_drained(decay_now());
        }
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn a_command_during_a_refresh_starts_a_new_turn_for_prepared_rows() {
        let slots = Arc::new(ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            16,
            (0..3).map(|id| (id, (0, 1))).collect(),
            false,
            0,
            "test",
        ));
        let configs = (0..3)
            .map(|id| {
                let config = SimpleWorkerConfig {
                    max_num_batched_tokens: Some(64),
                    ..Default::default()
                };
                (id, config)
            })
            .collect();
        let (_cfg_tx, cfg_rx) = watch::channel(configs);
        set_worker_capacities_from_configs(&slots, &cfg_rx.borrow());
        let refresher = Arc::new(BlockingRefresher::new(RefreshedOverlap::default()));
        let prepared_selections = Arc::new(AtomicUsize::new(0));
        let queue = SchedulerQueue::new(
            Arc::clone(&slots),
            cfg_rx,
            PolicyProfile::synthetic(Some(0.0), RouterQueuePolicy::Fcfs),
            16,
            counting_policy(&prepared_selections, false),
            None,
            Some(Arc::clone(&refresher)),
            None,
            None,
            None,
        );
        let busy = ["active-0", "active-1", "active-2"];
        for id in busy {
            let (active, active_rx) = make_request(id, 64);
            queue.enqueue(active).await;
            active_rx.await.unwrap().unwrap();
        }
        // Each was admitted in the turn that received it, with its own rows.
        assert_eq!(prepared_selections.load(Ordering::Relaxed), 3);
        let (stale, stale_rx) = make_request("stale", 64);
        queue
            .enqueue_with_block_hashes(stale, Some(vec![LocalBlockHash(1)]))
            .await;
        tokio::time::advance(Duration::from_secs(11)).await;
        free_requests(&slots, &busy[..2]);

        // `first` may run only on worker 1, `second` only on worker 2. Enqueuing `first`
        // drains `stale` into an overlap refresh that stalls.
        let (mut first, first_rx) = make_request("first", 64);
        first.allowed_worker_ids = Some(HashSet::from([1]));
        queue.enqueue(first).await;
        refresher.wait_for_calls(1).await;
        // `second` arrives during the refresh and starts a new turn, so `first`, from the earlier
        // turn, selects with neither its dropped rows nor the rows prepared for `second`.
        let (mut second, mut second_rx) = make_request("second", 64);
        second.allowed_worker_ids = Some(HashSet::from([2]));
        queue.enqueue(second).await;
        refresher.release_one();
        assert_eq!(
            stale_rx.await.unwrap().unwrap().best_worker,
            WorkerWithDpRank::new(0, 0)
        );
        assert_eq!(
            first_rx.await.unwrap().unwrap().best_worker,
            WorkerWithDpRank::new(1, 0)
        );
        assert!(second_rx.try_recv().is_err());

        // `second` stayed queued past its turn, so it rebuilds its rows too.
        free_requests(&slots, &["active-2"]);
        queue.update().await;
        assert_eq!(
            second_rx.await.unwrap().unwrap().best_worker,
            WorkerWithDpRank::new(2, 0)
        );
        assert_eq!(prepared_selections.load(Ordering::Relaxed), 3);
        free_requests(&slots, &["stale", "first", "second"]);
        slots.assert_completely_drained(decay_now());
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn capacity_wake_during_a_refresh_is_consumed_once_without_abandoning_it() {
        let refresher = Arc::new(BlockingRefresher::new(RefreshedOverlap::default()));
        let (queue, slots) = make_queue_with_blocking_refresher(
            1,
            16,
            64,
            Some(0.0),
            Arc::clone(&refresher),
            ADMISSION_CHANNEL_CAPACITY,
        );
        let (active, active_rx) = make_request("active", 64);
        queue.enqueue(active).await;
        active_rx.await.unwrap().unwrap();
        let (queued, mut queued_rx) = make_request("queued", 64);
        queue
            .enqueue_with_block_hashes(queued, Some(vec![LocalBlockHash(42)]))
            .await;
        tokio::time::advance(Duration::from_secs(11)).await;
        free_requests(&slots, &["active"]);
        queue.update().await;
        assert_eq!(refresher.calls.load(Ordering::Relaxed), 1);

        // The wake is handled during the refresh, ahead of the update behind it, and consumes its
        // hint: none is left for a second wake. The refresh keeps running.
        let worker = WorkerWithDpRank::new(0, 0);
        queue.capacity_changed(Some(worker));
        queue.update().await;
        let mut workers = FxHashSet::default();
        assert!(
            !queue
                .capacity_updates
                .as_ref()
                .unwrap()
                .drain_into(&mut workers)
        );
        assert!(workers.is_empty());
        assert!(matches!(
            queued_rx.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));

        refresher.release_one();
        queued_rx.await.unwrap().unwrap();
        queue.update().await;
        assert_eq!(refresher.calls.load(Ordering::Relaxed), 1);
        free_requests(&slots, &["queued"]);
        slots.assert_completely_drained(decay_now());
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn closed_response_abandons_a_stalled_refresh_without_booking() {
        let refresher = Arc::new(BlockingRefresher::new(RefreshedOverlap::default()));
        let (queue, slots) = make_queue_with_blocking_refresher(
            1,
            16,
            64,
            Some(0.0),
            Arc::clone(&refresher),
            ADMISSION_CHANNEL_CAPACITY,
        );
        let (active, active_rx) = make_request("active", 64);
        queue.enqueue(active).await;
        active_rx.await.unwrap().unwrap();
        let (abandoned, abandoned_rx) = make_request("abandoned", 64);
        queue
            .enqueue_with_block_hashes(abandoned, Some(vec![LocalBlockHash(42)]))
            .await;
        let (next, next_rx) = make_request("next", 64);
        queue.enqueue(next).await;
        tokio::time::advance(Duration::from_secs(11)).await;
        free_requests(&slots, &["active"]);
        queue.update().await;
        assert_eq!(refresher.calls.load(Ordering::Relaxed), 1);

        drop(abandoned_rx);
        next_rx.await.unwrap().unwrap();
        assert_eq!(queue.pending_count(), 0);
        assert_eq!(refresher.calls.load(Ordering::Relaxed), 1);
        free_requests(&slots, &["next"]);
        slots.assert_completely_drained(decay_now());
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn released_lifecycle_ends_a_refresh_without_booking_or_rejection() {
        let refresher = Arc::new(BlockingRefresher::new(RefreshedOverlap::default()));
        let (queue, slots) = make_queue_with_blocking_refresher(
            1,
            16,
            64,
            Some(0.0),
            Arc::clone(&refresher),
            ADMISSION_CHANNEL_CAPACITY,
        );
        let (active, active_rx) = make_request("active", 64);
        queue.enqueue(active).await;
        active_rx.await.unwrap().unwrap();
        let (candidate, candidate_rx) = make_request("candidate", 64);
        let lease = queue.new_request_lifecycle_lease(Some("candidate"));
        let lease = queue
            .enqueue_with_block_hashes_and_lease(candidate, Some(vec![LocalBlockHash(42)]), lease)
            .await
            .expect("the actor returns the candidate's lease");
        let (next, next_rx) = make_request("next", 64);
        queue.enqueue(next).await;
        tokio::time::advance(Duration::from_secs(11)).await;
        free_requests(&slots, &["active"]);
        queue.update().await;
        assert_eq!(refresher.calls.load(Ordering::Relaxed), 1);

        // The caller releases the candidate's lifecycle but keeps its receiver open. The cleanup
        // runs during the refresh and ends it: a booking made now would have no owner. The
        // candidate leaves unanswered and uncounted, and the free worker goes to `next`.
        drop(lease);
        next_rx.await.unwrap().unwrap();
        assert!(candidate_rx.await.is_err());
        assert_eq!(queue.pending_count(), 0);
        assert_eq!(refresher.calls.load(Ordering::Relaxed), 1);
        assert_eq!(class_outcomes(&queue, 0), (3, 0, 0));
        free_requests(&slots, &["next"]);
        slots.assert_completely_drained(decay_now());
    }

    #[tokio::test(start_paused = true)]
    async fn expiry_precedes_queue_limits_and_each_rejection_counts_once() {
        let profile = policy_profile(
            None,
            vec![PolicyClassConfig {
                slo_ttft_ms: NonZeroU32::new(100),
                controlled_delay: true,
                busy_thresholds: BusyThresholds {
                    prefill_busy_threshold_frac: Some(0.0),
                    ..Default::default()
                },
                request_queue_limit_per_worker: Some(1),
                ..class("capped", 1)
            }],
        );
        let (queue, slots) = make_queue_with_profile(1, 16, 64, profile);
        let mut actor = actor_for_test(&queue, &slots);
        let start = Instant::now();
        let (_, mut active_rx) = enqueue_into_actor(&queue, &mut actor, "active", None);
        actor.handle_update(None).await;
        active_rx.try_recv().unwrap().unwrap();
        let (queued, _expires_rx) = enqueue_into_actor(&queue, &mut actor, "expires", None);
        assert!(queued);

        tokio::time::advance(Duration::from_millis(10)).await;
        let (queued, mut limited_rx) = enqueue_into_actor(&queue, &mut actor, "limited", None);
        assert!(!queued);
        let Ok(Err(KvSchedulerError::QueueRejected(rejection))) = limited_rx.try_recv() else {
            panic!("a full class queue rejects with its limit");
        };
        assert_eq!(
            rejection.limit_kind,
            dynamo_admission::QueueLimitKind::Requests
        );
        // An expired arrival is a deadline rejection even though the queue is full.
        let (queued, mut expired_rx) =
            enqueue_into_actor(&queue, &mut actor, "expired", Some(start));
        assert!(!queued);
        assert!(matches!(
            expired_rx.try_recv(),
            Ok(Err(KvSchedulerError::DeadlineExceeded))
        ));

        // At 100 ms the queued request expires before the arrival's limit check, so the
        // arrival takes its place instead of being rejected for it.
        tokio::time::advance(Duration::from_millis(90)).await;
        let (queued, _accepted_rx) = enqueue_into_actor(&queue, &mut actor, "accepted", None);
        assert!(queued);
        assert_eq!(queue.pending_count(), 1);
        assert_eq!(class_outcomes(&queue, 0), (5, 2, 1));
        slots.free(&"active".to_string(), decay_now()).unwrap();
    }

    #[tokio::test]
    async fn every_queue_limit_kind_counts_as_queue_size_exceeded() {
        let profile = policy_profile(
            Some("requests"),
            vec![
                PolicyClassConfig {
                    busy_thresholds: BusyThresholds {
                        prefill_busy_threshold_frac: Some(0.0),
                        ..Default::default()
                    },
                    request_queue_limit_per_worker: Some(1),
                    ..class("requests", 1)
                },
                PolicyClassConfig {
                    busy_thresholds: BusyThresholds {
                        prefill_busy_threshold_frac: Some(0.0),
                        ..Default::default()
                    },
                    raw_isl_token_queue_limit_per_worker: Some(1),
                    ..class("raw_tokens", 1)
                },
                PolicyClassConfig {
                    busy_thresholds: BusyThresholds {
                        prefill_busy_threshold_frac: Some(0.0),
                        ..Default::default()
                    },
                    cached_token_queue_limit_per_worker: Some(1),
                    ..class("cached_tokens", 1)
                },
            ],
        );
        let (queue, slots) = make_queue_with_profile(1, 16, 64, profile);
        let worker = WorkerWithDpRank::new(0, 0);
        let (active, active_rx) = make_request("active", 64);
        queue.enqueue(active).await;
        active_rx.await.unwrap().unwrap();
        for (class_index, class, limit_kind) in [
            (0, "requests", dynamo_admission::QueueLimitKind::Requests),
            (
                1,
                "raw_tokens",
                dynamo_admission::QueueLimitKind::RawIslTokens,
            ),
            (
                2,
                "cached_tokens",
                dynamo_admission::QueueLimitKind::CachedTokens,
            ),
        ] {
            let mut receivers = Vec::new();
            for attempt in ["queued", "rejected"] {
                let (mut request, response_rx) = make_request(&format!("{class}-{attempt}"), 64);
                request.policy_class = Some(class.to_string());
                request.overlap.effective_cached_tokens.insert(worker, 32);
                queue.enqueue(request).await;
                receivers.push(response_rx);
            }
            let Ok(Err(KvSchedulerError::QueueRejected(rejection))) = receivers[1].try_recv()
            else {
                panic!("{class} rejects its second queued request");
            };
            assert_eq!(rejection.limit_kind, limit_kind);
            // The busy request also passed through the first class.
            let received = if class_index == 0 { 3 } else { 2 };
            assert_eq!(
                class_outcomes(&queue, class_index),
                (received, 0, 1),
                "{class}"
            );
        }
        slots.free(&"active".to_string(), decay_now()).unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn deadline_rejections_count_against_the_final_class() {
        let profile = policy_profile(
            Some("standard"),
            vec![
                PolicyClassConfig {
                    slo_ttft_ms: NonZeroU32::new(300),
                    controlled_delay: true,
                    busy_thresholds: BusyThresholds {
                        prefill_busy_threshold_frac: Some(0.0),
                        ..Default::default()
                    },
                    ..class("standard", 1)
                },
                PolicyClassConfig {
                    slo_ttft_ms: NonZeroU32::new(50),
                    controlled_delay: true,
                    busy_thresholds: BusyThresholds {
                        prefill_busy_threshold_frac: Some(0.0),
                        ..Default::default()
                    },
                    ..class("latency", 1)
                },
            ],
        );
        let (queue, slots) = make_queue_with_profile(1, 16, 64, profile);
        let start = Instant::now();
        let active = enqueue_to_class(&queue, "active", "standard", None, None).await;
        active.await.unwrap().unwrap();

        // The request asks for the default class, and the classifier moves it to latency.
        let (mut overridden, overridden_rx) = make_request("overridden", 64);
        overridden.policy_class = Some("standard".to_string());
        let mut classified = queue
            .build_classify_request(&overridden, start, WorkerInputs::NONE)
            .0;
        classified.set_policy_class("latency");
        queue
            .enqueue_admitted_with_block_hashes_and_lease(
                overridden,
                None,
                None,
                None,
                Some(classified),
                start,
            )
            .await;
        // An omitted class uses the default class. A name that matches no class is refused
        // before the queue, so no class counts it.
        let (omitted, omitted_rx) = make_request("omitted", 64);
        queue.enqueue(omitted).await;
        let unknown = enqueue_to_class(&queue, "unknown", "missing", None, None).await;
        assert!(matches!(
            unknown.await.unwrap(),
            Err(KvSchedulerError::UnknownPolicyClass(name)) if name == "missing"
        ));

        assert_eq!(
            rejected_after(overridden_rx, start).await,
            Duration::from_millis(50)
        );
        assert_eq!(
            rejected_after(omitted_rx, start).await,
            Duration::from_millis(300)
        );
        assert_eq!(class_outcomes(&queue, 0), (2, 1, 0));
        assert_eq!(class_outcomes(&queue, 1), (1, 1, 0));
        free_requests(&slots, &["active"]);
        slots.assert_completely_drained(decay_now());
    }

    #[tokio::test(start_paused = true)]
    async fn controlled_delay_applies_per_class_within_one_profile() {
        let profile = policy_profile(
            Some("omitted"),
            vec![
                PolicyClassConfig {
                    slo_ttft_ms: NonZeroU32::new(100),
                    controlled_delay: true,
                    busy_thresholds: BusyThresholds {
                        prefill_busy_threshold_frac: Some(0.0),
                        ..Default::default()
                    },
                    ..class("enabled", 1)
                },
                PolicyClassConfig {
                    slo_ttft_ms: NonZeroU32::new(100),
                    busy_thresholds: BusyThresholds {
                        prefill_busy_threshold_frac: Some(0.0),
                        ..Default::default()
                    },
                    ..class("omitted", 1)
                },
                PolicyClassConfig {
                    slo_ttft_ms: NonZeroU32::new(100),
                    controlled_delay: false,
                    busy_thresholds: BusyThresholds {
                        prefill_busy_threshold_frac: Some(0.0),
                        ..Default::default()
                    },
                    ..class("disabled", 1)
                },
            ],
        );
        let (queue, slots) = make_queue_with_profile(1, 16, 64, profile);
        let mut actor = actor_for_test(&queue, &slots);
        let start = Instant::now();
        let at = |millis| Some(start + Duration::from_millis(millis));
        let request = |id: &str, class: &str| {
            let (mut request, response_rx) = make_request(id, 64);
            request.policy_class = Some(class.to_string());
            (request, response_rx)
        };
        let (_, mut active_rx) = enqueue_into_actor(&queue, &mut actor, "active", None);
        actor.handle_update(None).await;
        active_rx.try_recv().unwrap().unwrap();

        // Each class queues a head with only a class due time. A later explicit deadline
        // waits behind the disabled class's head, and the omitted class holds one whose
        // caller leaves first.
        let mut heads = Vec::new();
        for class in ["enabled", "omitted", "disabled"] {
            let (head, head_rx) = request(&format!("{class}-head"), class);
            assert!(enqueue_request_into_actor(
                &queue, &mut actor, head, None, None
            ));
            heads.push(head_rx);
        }
        let (hidden, mut hidden_rx) = request("hidden", "disabled");
        assert!(enqueue_request_into_actor(
            &queue,
            &mut actor,
            hidden,
            None,
            at(200)
        ));
        let leaving_transfer = Arc::new(AdmissionLifecycleTransfer::new("leaving".to_string()));
        let (leaving, mut leaving_rx) = request("leaving", "omitted");
        assert!(enqueue_request_into_actor(
            &queue,
            &mut actor,
            leaving,
            Some(Arc::clone(&leaving_transfer)),
            at(150)
        ));
        // A class timer arms only at a due time that its class enforces.
        assert_eq!(actor.class_timers.armed, [at(100), at(150), at(200)]);

        actor.cleanup.enqueue(AdmissionCleanupEntry {
            target: SchedulerCleanupTarget::AdmissionLifecycle(leaving_transfer),
        });
        actor.drain_cleanup();
        assert!(matches!(
            leaving_rx.try_recv(),
            Err(oneshot::error::TryRecvError::Closed)
        ));
        assert_eq!(actor.class_timers.armed, [at(100), None, at(200)]);

        // At the class budget, only the opted-in class rejects its waiting head.
        tokio::time::advance(Duration::from_millis(100)).await;
        actor.handle_update(None).await;
        assert!(matches!(
            heads[0].try_recv(),
            Ok(Err(KvSchedulerError::DeadlineExceeded))
        ));
        for head in &mut heads[1..] {
            assert!(matches!(
                head.try_recv(),
                Err(oneshot::error::TryRecvError::Empty)
            ));
        }
        // The explicit deadline still expires behind the disabled class's ordering-only head.
        tokio::time::advance(Duration::from_millis(100)).await;
        actor.handle_update(None).await;
        assert!(matches!(
            hidden_rx.try_recv(),
            Ok(Err(KvSchedulerError::DeadlineExceeded))
        ));
        assert_eq!(actor.class_timers.armed, [None, None, None]);
        assert_eq!(actor.pending.pending_count(), 2);
        // Each class counts its own rejections, and none for the caller that left.
        assert_eq!(class_outcomes(&queue, 0), (1, 1, 0));
        assert_eq!(class_outcomes(&queue, 1), (3, 0, 0));
        assert_eq!(class_outcomes(&queue, 2), (2, 1, 0));

        // Freed capacity admits both heads whose class budgets only ordered them.
        free_requests(&slots, &["active"]);
        for _ in 0..2 {
            actor.handle_update(None).await;
            let admitted = match heads[1].try_recv() {
                Ok(response) => {
                    response.unwrap();
                    "omitted-head"
                }
                Err(_) => {
                    heads[2].try_recv().unwrap().unwrap();
                    "disabled-head"
                }
            };
            free_requests(&slots, &[admitted]);
        }
        assert_eq!(actor.pending.pending_count(), 0);
        slots.assert_completely_drained(decay_now());
    }

    #[tokio::test(start_paused = true)]
    async fn the_final_class_decides_controlled_delay() {
        let profile = policy_profile(
            Some("ordered"),
            vec![
                PolicyClassConfig {
                    slo_ttft_ms: NonZeroU32::new(100),
                    controlled_delay: true,
                    busy_thresholds: BusyThresholds {
                        prefill_busy_threshold_frac: Some(0.0),
                        ..Default::default()
                    },
                    ..class("enforced", 1)
                },
                PolicyClassConfig {
                    slo_ttft_ms: NonZeroU32::new(100),
                    busy_thresholds: BusyThresholds {
                        prefill_busy_threshold_frac: Some(0.0),
                        ..Default::default()
                    },
                    ..class("ordered", 1)
                },
            ],
        );
        let (queue, slots) = make_queue_with_profile(1, 16, 64, profile);
        let start = Instant::now();
        let active = enqueue_to_class(&queue, "active", "ordered", None, None).await;
        active.await.unwrap().unwrap();

        // The classifier moves each request to the other class.
        let mut moved = Vec::new();
        for (id, requested, selected) in [
            ("to-enforced", "ordered", "enforced"),
            ("to-ordered", "enforced", "ordered"),
        ] {
            let (mut request, response_rx) = make_request(id, 64);
            request.policy_class = Some(requested.to_string());
            let mut classified = queue
                .build_classify_request(&request, start, WorkerInputs::NONE)
                .0;
            classified.set_policy_class(selected);
            queue
                .enqueue_admitted_with_block_hashes_and_lease(
                    request,
                    None,
                    None,
                    None,
                    Some(classified),
                    start,
                )
                .await;
            moved.push(response_rx);
        }
        let mut to_ordered = moved.pop().unwrap();
        let to_enforced = moved.pop().unwrap();
        assert_eq!(
            rejected_after(to_enforced, start).await,
            Duration::from_millis(100)
        );
        // Long past its budget, the request moved to the ordering-only class still waits.
        tokio::time::advance(Duration::from_secs(1)).await;
        assert!(matches!(
            to_ordered.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));
        assert_eq!(class_outcomes(&queue, 0), (1, 1, 0));
        assert_eq!(class_outcomes(&queue, 1), (2, 0, 0));

        free_requests(&slots, &["active"]);
        queue.update().await;
        to_ordered.await.unwrap().unwrap();
        free_requests(&slots, &["to-ordered"]);
        slots.assert_completely_drained(decay_now());
    }

    #[tokio::test(start_paused = true)]
    async fn class_entries_due_during_the_limit_sweep_leave_before_its_limit_rejects() {
        // "index": the held entry's explicit deadline, found through the class's deadline index.
        // "head": the class enables controlled delay, and the held entry's class budget is found
        // at the class head.
        for (mode, head_budget) in [("index", false), ("head", true)] {
            let profile = policy_profile(
                Some("limited"),
                vec![
                    PolicyClassConfig {
                        slo_ttft_ms: head_budget.then(|| NonZeroU32::new(100).unwrap()),
                        controlled_delay: head_budget,
                        busy_thresholds: BusyThresholds {
                            prefill_busy_threshold_frac: Some(0.0),
                            ..Default::default()
                        },
                        request_queue_limit_per_worker: Some(1),
                        ..class("limited", 1)
                    },
                    PolicyClassConfig {
                        busy_thresholds: BusyThresholds {
                            prefill_busy_threshold_frac: Some(0.0),
                            ..Default::default()
                        },
                        ..class("other", 1)
                    },
                ],
            );
            for (case, arrival_due_ms, caller_left) in [
                ("takes-the-freed-slot", 199, false),
                ("own-deadline-first", 100, false),
                ("caller-left", 199, true),
            ] {
                let (queue, slots) = make_queue_with_profile(1, 16, 64, profile.clone());
                let mut actor = actor_for_test(&queue, &slots);
                let start = Instant::now();
                let at = |millis| Some(start + Duration::from_millis(millis));
                let request = |id: &str, class: &str| {
                    let (mut request, response_rx) = make_request(id, 64);
                    request.policy_class = Some(class.to_string());
                    (request, response_rx)
                };
                let (_, mut active_rx) = enqueue_into_actor(&queue, &mut actor, "active", None);
                actor.handle_update(None).await;
                active_rx.try_recv().unwrap().unwrap();

                // The limited class is full with an entry due at 100 ms. The other class's entry is
                // due at 99 ms, and rejecting it takes 2 ms.
                let held_deadline = if head_budget { None } else { at(100) };
                let (held, mut held_rx) = request("held", "limited");
                assert!(enqueue_request_into_actor(
                    &queue,
                    &mut actor,
                    held,
                    None,
                    held_deadline
                ));
                let (overdue, mut overdue_rx) = request("overdue", "other");
                assert!(enqueue_request_into_actor(
                    &queue,
                    &mut actor,
                    overdue,
                    None,
                    at(99)
                ));
                advance_clock_on_response(&mut overdue_rx, Duration::from_millis(2));
                tokio::time::advance(Duration::from_millis(99)).await;

                let (arrival, arrival_rx) = request("arrival", "limited");
                let mut arrival_rx = (!caller_left).then_some(arrival_rx);
                let queued = enqueue_request_into_actor(
                    &queue,
                    &mut actor,
                    arrival,
                    None,
                    at(arrival_due_ms),
                );
                // The sweep at 99 ms rejected the other class's entry and crossed the held entry's
                // due time.
                assert_eq!(
                    Instant::now(),
                    start + Duration::from_millis(101),
                    "{mode} {case}"
                );
                assert!(
                    matches!(
                        overdue_rx.try_recv(),
                        Ok(Err(KvSchedulerError::DeadlineExceeded))
                    ),
                    "{mode} {case}"
                );
                assert_eq!(class_outcomes(&queue, 1), (1, 1, 0), "{mode} {case}");
                match case {
                    "takes-the-freed-slot" => {
                        // The held entry leaves first, and the arrival takes its slot.
                        assert!(queued);
                        assert!(matches!(
                            held_rx.try_recv(),
                            Ok(Err(KvSchedulerError::DeadlineExceeded))
                        ));
                        assert!(matches!(
                            arrival_rx.as_mut().unwrap().try_recv(),
                            Err(oneshot::error::TryRecvError::Empty)
                        ));
                        assert_eq!(actor.class_timers.armed, [at(199), None]);
                        assert_eq!(class_outcomes(&queue, 0), (3, 1, 0));
                    }
                    "own-deadline-first" => {
                        // The arrival's own due time passed too: a deadline rejection, before any
                        // class cleanup.
                        assert!(!queued);
                        assert!(matches!(
                            arrival_rx.as_mut().unwrap().try_recv(),
                            Ok(Err(KvSchedulerError::DeadlineExceeded))
                        ));
                        assert!(matches!(
                            held_rx.try_recv(),
                            Err(oneshot::error::TryRecvError::Empty)
                        ));
                        assert_eq!(class_outcomes(&queue, 0), (3, 1, 0));
                    }
                    _ => {
                        // A caller that left is dropped uncounted, without class cleanup.
                        assert!(!queued);
                        assert!(matches!(
                            held_rx.try_recv(),
                            Err(oneshot::error::TryRecvError::Empty)
                        ));
                        assert_eq!(class_outcomes(&queue, 0), (3, 0, 0));
                    }
                }
                if case != "takes-the-freed-slot" {
                    // Its class timer still rejects the held entry.
                    assert_eq!(actor.class_timers.armed, [at(100), None]);
                    actor.handle_update(None).await;
                    assert!(matches!(
                        held_rx.try_recv(),
                        Ok(Err(KvSchedulerError::DeadlineExceeded))
                    ));
                    assert_eq!(actor.class_timers.armed, [None, None]);
                }
            }
        }
    }

    #[tokio::test(start_paused = true)]
    async fn limit_cleanup_repeats_while_class_entries_become_due() {
        // A token limit: removing the small held entry alone leaves the class over it.
        let profile = policy_profile(
            Some("limited"),
            vec![
                PolicyClassConfig {
                    busy_thresholds: BusyThresholds {
                        prefill_busy_threshold_frac: Some(0.0),
                        ..Default::default()
                    },
                    raw_isl_token_queue_limit_per_worker: Some(100),
                    ..class("limited", 1)
                },
                PolicyClassConfig {
                    busy_thresholds: BusyThresholds {
                        prefill_busy_threshold_frac: Some(0.0),
                        ..Default::default()
                    },
                    ..class("other", 1)
                },
            ],
        );
        for (large_due_ms, arrival_queued) in [(102, true), (500, false)] {
            let (queue, slots) = make_queue_with_profile(1, 16, 64, profile.clone());
            let mut actor = actor_for_test(&queue, &slots);
            let start = Instant::now();
            let at = |millis| Some(start + Duration::from_millis(millis));
            let request = |id: &str, class: &str, isl_tokens| {
                let (mut request, response_rx) = make_request(id, isl_tokens);
                request.policy_class = Some(class.to_string());
                (request, response_rx)
            };
            let (_, mut active_rx) = enqueue_into_actor(&queue, &mut actor, "active", None);
            actor.handle_update(None).await;
            active_rx.try_recv().unwrap().unwrap();

            // 16 + 112 tokens fill the 100-token limit. Rejecting the other class's entry at
            // 99 ms takes 2 ms; rejecting the small entry, due at 100 ms, takes 2 more.
            let (small, mut small_rx) = request("small", "limited", 16);
            assert!(enqueue_request_into_actor(
                &queue,
                &mut actor,
                small,
                None,
                at(100)
            ));
            let (large, mut large_rx) = request("large", "limited", 112);
            assert!(enqueue_request_into_actor(
                &queue,
                &mut actor,
                large,
                None,
                at(large_due_ms)
            ));
            let (overdue, mut overdue_rx) = request("overdue", "other", 16);
            assert!(enqueue_request_into_actor(
                &queue,
                &mut actor,
                overdue,
                None,
                at(99)
            ));
            advance_clock_on_response(&mut overdue_rx, Duration::from_millis(2));
            advance_clock_on_response(&mut small_rx, Duration::from_millis(2));
            tokio::time::advance(Duration::from_millis(99)).await;

            let (arrival, mut arrival_rx) = request("arrival", "limited", 64);
            // Ready work changed either way: the arrival entered the queue, or the cleanup left
            // the large entry at the class head.
            assert!(enqueue_request_into_actor(
                &queue,
                &mut actor,
                arrival,
                None,
                at(1000)
            ));
            assert_eq!(Instant::now(), start + Duration::from_millis(103));
            assert!(matches!(
                small_rx.try_recv(),
                Ok(Err(KvSchedulerError::DeadlineExceeded))
            ));
            if arrival_queued {
                // The large entry became due during the first cleanup; the next pass removed it.
                assert!(matches!(
                    large_rx.try_recv(),
                    Ok(Err(KvSchedulerError::DeadlineExceeded))
                ));
                assert!(matches!(
                    arrival_rx.try_recv(),
                    Err(oneshot::error::TryRecvError::Empty)
                ));
                assert_eq!(actor.class_timers.armed, [at(1000), None]);
                assert_eq!(class_outcomes(&queue, 0), (4, 2, 0));
            } else {
                // Nothing else of the class was due when the limit judged again: it stands.
                assert!(matches!(
                    large_rx.try_recv(),
                    Err(oneshot::error::TryRecvError::Empty)
                ));
                assert!(matches!(
                    arrival_rx.try_recv(),
                    Ok(Err(KvSchedulerError::QueueRejected(_)))
                ));
                assert_eq!(actor.class_timers.armed, [at(500), None]);
                assert_eq!(class_outcomes(&queue, 0), (4, 1, 1));
            }
        }
    }

    #[tokio::test(start_paused = true)]
    async fn a_refused_arrival_drains_the_head_its_expiry_exposed() {
        // Class `a` has no class budget or controlled delay; its two workers give it a 64-token
        // limit, which the 64-token successor alone fills once the head leaves.
        let profile = policy_profile(
            Some("a"),
            vec![
                PolicyClassConfig {
                    busy_thresholds: BusyThresholds {
                        prefill_busy_threshold_frac: Some(0.0),
                        ..Default::default()
                    },
                    raw_isl_token_queue_limit_per_worker: Some(32),
                    ..class("a", 1)
                },
                PolicyClassConfig {
                    busy_thresholds: BusyThresholds {
                        prefill_busy_threshold_frac: Some(0.0),
                        ..Default::default()
                    },
                    ..class("b", 1)
                },
            ],
        );
        // Each arrival is refused after expiry removed the blocked head of its class:
        // "sweep": the expiry sweep at 100 ms removes the head, and the limit still refuses.
        // "retry": the head is due at 101 ms. The sweep only empties class `b`, and rejecting its
        // entry takes 2 ms, so the limit's cleanup removes the head; the limit still refuses.
        // "arrival-expired": rejecting the head takes 2 ms, past the arrival's own deadline.
        // "arrival-cancelled": the arrival's caller has left.
        for case in ["sweep", "retry", "arrival-expired", "arrival-cancelled"] {
            let head_rejection = if case == "arrival-expired" {
                Duration::from_millis(2)
            } else {
                Duration::ZERO
            };
            let BlockedSharedHead {
                queue,
                slots,
                mut actor,
                start,
                mut head_rx,
                mut successor_rx,
                overdue_rx,
            } = blocked_shared_head(
                profile.clone(),
                if case == "retry" { 101 } else { 100 },
                head_rejection,
                64,
                case == "retry",
            )
            .await;
            let at = |millis| Some(start + Duration::from_millis(millis));
            let (arrival, arrival_rx) = make_request("arrival", 16);
            let mut arrival_rx = (case != "arrival-cancelled").then_some(arrival_rx);
            let arrival_due = if case == "arrival-expired" {
                at(101)
            } else {
                None
            };
            let (ready, reply) = apply_enqueue_command(&queue, &mut actor, arrival, arrival_due);
            let clock_ms = if matches!(case, "retry" | "arrival-expired") {
                102
            } else {
                100
            };
            assert_eq!(Instant::now(), start + Duration::from_millis(clock_ms));
            assert!(
                matches!(
                    head_rx.try_recv(),
                    Ok(Err(KvSchedulerError::DeadlineExceeded))
                ),
                "{case}"
            );
            if let Some(arrival_rx) = arrival_rx.as_mut() {
                let response = arrival_rx.try_recv().unwrap();
                if case == "arrival-expired" {
                    assert!(matches!(response, Err(KvSchedulerError::DeadlineExceeded)));
                } else {
                    assert!(matches!(response, Err(KvSchedulerError::QueueRejected(_))));
                }
            }
            // Refused, the arrival still reports the ready work its expiry exposed. The drain the
            // actor loop runs for that report, with no capacity event or update, admits the
            // successor.
            assert_eq!(ready, Drain::Enqueued, "{case}");
            actor.drain_after_command(ready, reply).await;
            assert_eq!(
                successor_rx.try_recv().unwrap().unwrap().best_worker,
                WorkerWithDpRank::new(1, 0),
                "{case}"
            );
            assert_eq!((queue.pending_count(), queue.pending_isl_tokens()), (0, 0));
            assert_eq!(actor.class_timers.armed, [None, None], "{case}");
            let outcomes = match case {
                "arrival-expired" => (4, 2, 0),
                "arrival-cancelled" => (4, 1, 0),
                _ => (4, 1, 1),
            };
            assert_eq!(class_outcomes(&queue, 0), outcomes, "{case}");
            if let Some(mut overdue_rx) = overdue_rx {
                assert!(matches!(
                    overdue_rx.try_recv(),
                    Ok(Err(KvSchedulerError::DeadlineExceeded))
                ));
                assert_eq!(class_outcomes(&queue, 1), (1, 1, 0));
            }
            free_requests(&slots, &["active", "successor"]);
            slots.assert_completely_drained(decay_now());
        }
    }

    /// An arrival whose enqueue expires the root of a blocked exact lane drains that lane's next
    /// head, though no capacity change was recorded for the lane's worker and the expiry left no
    /// class timer armed. Otherwise the successor would wait for a capacity change on its worker
    /// or for a full recheck, which the scheduler's periodic queue update runs.
    #[tokio::test(start_paused = true)]
    async fn an_arrival_drains_the_exact_lane_its_expiry_unblocked() {
        // One worker: a 64-token limit refuses the arrival once the successor alone fills it, and
        // a 128-token limit admits it to the lane behind the successor.
        for (case, queue_limit) in [("refused", 64), ("accepted", 128)] {
            let arrival_queued = case == "accepted";
            let profile = policy_profile(
                None,
                vec![PolicyClassConfig {
                    queue_policy: RouterQueuePolicy::Fcfs,
                    busy_thresholds: BusyThresholds {
                        prefill_busy_threshold_frac: Some(0.0),
                        ..Default::default()
                    },
                    raw_isl_token_queue_limit_per_worker: Some(queue_limit),
                    ..class("a", 1)
                }],
            );
            let (queue, slots) = make_queue_with_profile(1, 16, 64, profile);
            let mut actor = actor_for_test(&queue, &slots);
            let start = Instant::now();
            let worker = WorkerWithDpRank::new(0, 0);
            let pinned = |id: &str, isl_tokens| {
                let (mut request, response_rx) = make_request(id, isl_tokens);
                request.pinned_worker = Some(worker);
                (request, response_rx)
            };
            let (active, mut active_rx) = pinned("active", 16);
            assert!(enqueue_request_into_actor(
                &queue, &mut actor, active, None, None
            ));
            actor.handle_update(None).await;
            active_rx.try_recv().unwrap().unwrap();

            // The busy worker blocks its exact lane at the head, due at 100 ms. The successor,
            // with no due time, waits behind it.
            let (head, mut head_rx) = pinned("head", 16);
            let head_due = start + Duration::from_millis(100);
            assert!(enqueue_request_into_actor(
                &queue,
                &mut actor,
                head,
                None,
                Some(head_due)
            ));
            let (successor, mut successor_rx) = pinned("successor", 64);
            assert!(enqueue_request_into_actor(
                &queue, &mut actor, successor, None, None
            ));
            actor.handle_update(None).await;
            assert!(matches!(
                successor_rx.try_recv(),
                Err(oneshot::error::TryRecvError::Empty)
            ));

            // The worker gains room with no recorded capacity change, as decaying prefill load
            // gives it, and the head's due time passes before the actor's timer branch runs.
            slots.free(&"active".to_string(), decay_now()).unwrap();
            tokio::time::advance(Duration::from_millis(100)).await;

            // Applied as the actor loop applies a command, the arrival's expiry sweep removes the
            // head and leaves the class with no due time to arm.
            let (arrival, mut arrival_rx) = if arrival_queued {
                pinned("arrival", 16)
            } else {
                make_request("arrival", 16)
            };
            let (ready, reply) = apply_enqueue_command(&queue, &mut actor, arrival, None);
            assert!(
                matches!(
                    head_rx.try_recv(),
                    Ok(Err(KvSchedulerError::DeadlineExceeded))
                ),
                "{case}"
            );
            assert_eq!(actor.class_timers.armed, [None], "{case}");
            assert_eq!(ready, Drain::Enqueued, "{case}");

            // The drain the actor loop runs for the arrival, with no capacity event or update,
            // dispatches the successor to the worker that gained room.
            actor.drain_after_command(ready, reply).await;
            assert_eq!(
                successor_rx.try_recv().unwrap().unwrap().best_worker,
                worker,
                "{case}"
            );
            if arrival_queued {
                // The successor's booking fills the worker, so the arrival waits on the lane.
                assert!(matches!(
                    arrival_rx.try_recv(),
                    Err(oneshot::error::TryRecvError::Empty)
                ));
                assert_eq!(queue.pending_count(), 1);
                free_requests(&slots, &["successor"]);
                actor.handle_update(None).await;
                assert_eq!(arrival_rx.try_recv().unwrap().unwrap().best_worker, worker);
                free_requests(&slots, &["arrival"]);
            } else {
                assert!(matches!(
                    arrival_rx.try_recv(),
                    Ok(Err(KvSchedulerError::QueueRejected(_)))
                ));
                assert_eq!(queue.pending_count(), 0);
                free_requests(&slots, &["successor"]);
            }
            slots.assert_completely_drained(decay_now());
        }
    }

    #[tokio::test(start_paused = true)]
    async fn an_arrival_due_by_its_insertion_attempt_never_enters_the_queue() {
        // Class `a` has no class budget or controlled delay; its two workers give it a 2-request
        // limit, which the head and the successor fill.
        let profile = policy_profile(
            Some("a"),
            vec![
                PolicyClassConfig {
                    busy_thresholds: BusyThresholds {
                        prefill_busy_threshold_frac: Some(0.0),
                        ..Default::default()
                    },
                    request_queue_limit_per_worker: Some(1),
                    ..class("a", 1)
                },
                PolicyClassConfig {
                    busy_thresholds: BusyThresholds {
                        prefill_busy_threshold_frac: Some(0.0),
                        ..Default::default()
                    },
                    ..class("b", 1)
                },
            ],
        );
        // Expiry removes the blocked head of class `a`, which makes room for the arrival, and takes
        // 2 ms, past the arrival's deadline:
        // "sweep": the sweep at 100 ms removes the head; the arrival is due at 101 ms.
        // "retry": the head is due at 101 ms and the arrival at 103 ms. The sweep at 100 ms only
        // empties class `b`, which takes 2 ms; the limit refuses the arrival, and its cleanup
        // removes the head.
        // "caller-left": as "sweep", but the arrival's caller has left.
        for case in ["sweep", "retry", "caller-left"] {
            let BlockedSharedHead {
                queue,
                slots,
                mut actor,
                start,
                mut head_rx,
                mut successor_rx,
                overdue_rx,
            } = blocked_shared_head(
                profile.clone(),
                if case == "retry" { 101 } else { 100 },
                Duration::from_millis(2),
                16,
                case == "retry",
            )
            .await;
            let (arrival, arrival_rx) = make_request("arrival", 16);
            let mut arrival_rx = (case != "caller-left").then_some(arrival_rx);
            let arrival_due_ms = if case == "retry" { 103 } else { 101 };
            let (ready, reply) = apply_enqueue_command(
                &queue,
                &mut actor,
                arrival,
                Some(start + Duration::from_millis(arrival_due_ms)),
            );
            let clock_ms = if case == "retry" { 104 } else { 102 };
            assert_eq!(
                Instant::now(),
                start + Duration::from_millis(clock_ms),
                "{case}"
            );
            assert!(
                matches!(
                    head_rx.try_recv(),
                    Ok(Err(KvSchedulerError::DeadlineExceeded))
                ),
                "{case}"
            );
            // Before any drain, the arrival is rejected (uncounted when its caller left) and never
            // entered the queue, its pending counts or its class timer: only the successor waits.
            if let Some(arrival_rx) = arrival_rx.as_mut() {
                assert!(
                    matches!(
                        arrival_rx.try_recv(),
                        Ok(Err(KvSchedulerError::DeadlineExceeded))
                    ),
                    "{case}"
                );
            }
            assert_eq!(
                (queue.pending_count(), queue.pending_isl_tokens()),
                (1, 16),
                "{case}"
            );
            let stats = queue.class_queue_stats(0).unwrap();
            assert_eq!(
                (stats.pending_count, stats.pending_isl_tokens),
                (1, 16),
                "{case}"
            );
            assert_eq!(actor.class_timers.armed, [None, None], "{case}");
            let due_time_passed = if case == "caller-left" { 1 } else { 2 };
            assert_eq!(class_outcomes(&queue, 0), (4, due_time_passed, 0), "{case}");
            // The rejection still reports the head that expiry exposed, and the drain the actor
            // loop runs for that report admits the successor.
            assert_eq!(ready, Drain::Enqueued, "{case}");
            actor.drain_after_command(ready, reply).await;
            assert_eq!(
                successor_rx.try_recv().unwrap().unwrap().best_worker,
                WorkerWithDpRank::new(1, 0),
                "{case}"
            );
            assert_eq!((queue.pending_count(), queue.pending_isl_tokens()), (0, 0));
            if let Some(mut overdue_rx) = overdue_rx {
                assert!(matches!(
                    overdue_rx.try_recv(),
                    Ok(Err(KvSchedulerError::DeadlineExceeded))
                ));
                assert_eq!(class_outcomes(&queue, 1), (1, 1, 0));
            }
            free_requests(&slots, &["active", "successor"]);
            slots.assert_completely_drained(decay_now());
        }
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn due_time_passing_before_selection_is_rejected_without_selecting() {
        let profile = two_class_profile();
        // The dispatch gate asks for the available workers after the drain reads the clock.
        // Once armed, that call for "crossing" takes 10 ms of the clock, as slow gate work or
        // preemption would. Selecting "crossing" would fail.
        let slow_gate = Arc::new(AtomicBool::new(false));
        let available_workers: WorkerAvailabilityProvider = {
            let slow_gate = Arc::clone(&slow_gate);
            Arc::new(move |request| {
                if request.mode.request_id() == Some("crossing")
                    && slow_gate.swap(false, Ordering::Relaxed)
                {
                    advance_paused_clock(Duration::from_millis(10));
                }
                None
            })
        };
        let crossing_selections = Arc::new(AtomicUsize::new(0));
        let selector = HookedSelector::new({
            let crossing_selections = Arc::clone(&crossing_selections);
            move |request_id: Option<&str>| {
                if request_id == Some("crossing") {
                    crossing_selections.fetch_add(1, Ordering::Relaxed);
                    return Err(KvSchedulerError::NoEndpoints);
                }
                Ok(())
            }
        });
        let (queue, slots) =
            make_queue_with_selector_and_availability(profile, selector, available_workers);
        let start = Instant::now();
        // Admitting the busy request from the second queue starts the next DRR ring at the
        // first.
        let active = enqueue_to_class(&queue, "active", "second", None, None).await;
        active.await.unwrap().unwrap();
        let crossing = enqueue_to_class(
            &queue,
            "crossing",
            "first",
            Some(start + Duration::from_secs(12)),
            None,
        )
        .await;
        let mut successor = enqueue_to_class(&queue, "successor", "first", None, None).await;
        let mut other = enqueue_to_class(&queue, "other", "second", None, None).await;

        // The worker frees 5 ms before the due time, and the gate passes it.
        tokio::time::advance(Duration::from_secs(12) - Duration::from_millis(5)).await;
        slow_gate.store(true, Ordering::Relaxed);
        slots
            .mark_prefill_completed(&"active".to_string(), decay_now())
            .unwrap();
        slots.free(&"active".to_string(), decay_now()).unwrap();
        queue.update().await;
        assert!(!slow_gate.load(Ordering::Relaxed));
        assert!(matches!(
            crossing.await.unwrap(),
            Err(KvSchedulerError::DeadlineExceeded)
        ));
        assert_eq!(crossing_selections.load(Ordering::Relaxed), 0);
        // The rejected request reserved nothing and spent none of its class's credit, so its
        // successor takes the worker ahead of the other class.
        successor.try_recv().unwrap().unwrap();
        assert!(matches!(
            other.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));
        assert_eq!(class_outcomes(&queue, 0), (2, 1, 0));
        assert_eq!(class_outcomes(&queue, 1), (2, 0, 0));

        slots.free(&"successor".to_string(), decay_now()).unwrap();
        queue.update().await;
        other.try_recv().unwrap().unwrap();
        free_requests(&slots, &["other"]);
        slots.assert_completely_drained(decay_now());
    }

    #[tokio::test(start_paused = true)]
    async fn final_admission_check_rejects_a_due_time_passed_during_selection() {
        let (queue, slots) = make_queue(1, 16, 64, None);
        // Each selection takes 10 ms of the clock, as a slow selector would.
        let selections = Arc::new(AtomicUsize::new(0));
        let selector = HookedSelector::new({
            let selections = Arc::clone(&selections);
            move |_: Option<&str>| {
                selections.fetch_add(1, Ordering::Relaxed);
                advance_paused_clock(Duration::from_millis(10));
                Ok(())
            }
        });
        let mut actor = actor_with_selector_for_test(&queue, &slots, selector);
        // The due time is still ahead at the check before selection and has passed at the
        // check right before booking or delivery, which rejects.
        for mode in [
            ScheduleMode::Tracked {
                request_id: "tracked".to_string(),
            },
            ScheduleMode::QueryOnly { request_id: None },
        ] {
            let (mut request, mut response_rx) = make_request("expired", 64);
            request.mode = mode;
            let now = Instant::now();
            let due_at = now + Duration::from_millis(5);
            assert_eq!(
                actor.admit_one(
                    AdmissionContext::untracked(0, Some(due_at)),
                    request,
                    now,
                    None
                ),
                Admission::Refused
            );
            assert!(matches!(
                response_rx.try_recv(),
                Ok(Err(KvSchedulerError::DeadlineExceeded))
            ));
        }
        assert_eq!(selections.load(Ordering::Relaxed), 2);
        slots.assert_completely_drained(decay_now());

        // A due time that outlasts selection admits.
        let (request, mut response_rx) = make_request("in-time", 64);
        let now = Instant::now();
        assert_eq!(
            actor.admit_one(
                AdmissionContext::untracked(0, Some(now + Duration::from_secs(60))),
                request,
                now,
                None
            ),
            Admission::Booked
        );
        response_rx.try_recv().unwrap().unwrap();
        assert_eq!(selections.load(Ordering::Relaxed), 3);
        assert_eq!(class_outcomes(&queue, 0), (0, 2, 0));
        free_requests(&slots, &["in-time"]);
        slots.assert_completely_drained(decay_now());
    }

    #[tokio::test(start_paused = true)]
    async fn due_times_passing_during_a_blocked_direct_enqueue_are_judged_before_its_limit() {
        let profile = policy_profile(
            Some("direct"),
            vec![
                PolicyClassConfig {
                    busy_thresholds: BusyThresholds {
                        prefill_busy_threshold_frac: Some(0.0),
                        ..Default::default()
                    },
                    request_queue_limit_per_worker: Some(0),
                    ..class("direct", 1)
                },
                PolicyClassConfig {
                    slo_ttft_ms: NonZeroU32::new(100),
                    controlled_delay: true,
                    busy_thresholds: BusyThresholds {
                        prefill_busy_threshold_frac: Some(0.0),
                        ..Default::default()
                    },
                    ..class("queued", 1)
                },
            ],
        );
        let (queue, slots) = make_queue_with_profile(1, 16, 64, profile);
        let mut actor = actor_for_test(&queue, &slots);
        // The direct class never queues, so a free worker admits directly.
        let (admitted, mut active_rx) = enqueue_into_actor(&queue, &mut actor, "active", None);
        assert!(admitted);
        active_rx.try_recv().unwrap().unwrap();
        // The worker is now full, so the direct path falls through to the zero limit. Once
        // armed, the availability check takes 200 ms of the clock.
        let slow_check = Arc::new(AtomicBool::new(false));
        let available_workers: WorkerAvailabilityProvider = {
            let slow_check = Arc::clone(&slow_check);
            Arc::new(move |_| {
                if slow_check.swap(false, Ordering::Relaxed) {
                    advance_paused_clock(Duration::from_millis(200));
                }
                None
            })
        };
        actor.available_worker_ids = Some(Arc::new(AvailableWorkerIds::new(available_workers)));

        // A request still in time is rejected by the zero limit.
        let in_time_due = Instant::now() + Duration::from_millis(300);
        let (queued, mut in_time_rx) =
            enqueue_into_actor(&queue, &mut actor, "in-time", Some(in_time_due));
        assert!(!queued);
        assert!(matches!(
            in_time_rx.try_recv(),
            Ok(Err(KvSchedulerError::QueueRejected(_)))
        ));
        assert_eq!(class_outcomes(&queue, 0), (2, 0, 1));

        // The arrival's deadline passes during the availability check: a deadline rejection.
        slow_check.store(true, Ordering::Relaxed);
        let crossing_due = Instant::now() + Duration::from_millis(100);
        let (queued, mut crossing_rx) =
            enqueue_into_actor(&queue, &mut actor, "crossing", Some(crossing_due));
        assert!(!queued);
        assert!(!slow_check.load(Ordering::Relaxed));
        assert!(matches!(
            crossing_rx.try_recv(),
            Ok(Err(KvSchedulerError::DeadlineExceeded))
        ));
        assert_eq!(class_outcomes(&queue, 0), (3, 1, 1));

        // A queued request of the other class reaches its due time during the check. It
        // leaves, rejected, before the zero limit rejects the arrival.
        let (mut waiting, mut waiting_rx) = make_request("waiting", 64);
        waiting.policy_class = Some("queued".to_string());
        assert!(enqueue_request_into_actor(
            &queue, &mut actor, waiting, None, None
        ));
        slow_check.store(true, Ordering::Relaxed);
        let (queued, mut limited_rx) = enqueue_into_actor(&queue, &mut actor, "limited", None);
        assert!(!queued);
        assert!(matches!(
            waiting_rx.try_recv(),
            Ok(Err(KvSchedulerError::DeadlineExceeded))
        ));
        assert!(matches!(
            limited_rx.try_recv(),
            Ok(Err(KvSchedulerError::QueueRejected(_)))
        ));
        assert_eq!(queue.pending_count(), 0);
        assert_eq!(class_outcomes(&queue, 0), (4, 1, 2));
        assert_eq!(class_outcomes(&queue, 1), (1, 1, 0));

        // A caller that already left is counted under neither reason.
        slow_check.store(true, Ordering::Relaxed);
        let (request, response_rx) = make_request("left", 64);
        drop(response_rx);
        let left_due = Instant::now() + Duration::from_millis(100);
        assert!(!enqueue_request_into_actor(
            &queue,
            &mut actor,
            request,
            None,
            Some(left_due)
        ));
        assert!(!slow_check.load(Ordering::Relaxed));
        assert_eq!(class_outcomes(&queue, 0), (5, 1, 2));

        free_requests(&slots, &["active"]);
        slots.assert_completely_drained(decay_now());
    }

    #[tokio::test(start_paused = true)]
    async fn rejections_skip_callers_that_already_left() {
        let profile = policy_profile(
            None,
            vec![PolicyClassConfig {
                slo_ttft_ms: NonZeroU32::new(100),
                controlled_delay: true,
                busy_thresholds: BusyThresholds {
                    prefill_busy_threshold_frac: Some(0.0),
                    ..Default::default()
                },
                request_queue_limit_per_worker: Some(2),
                ..class("capped", 1)
            }],
        );
        let (queue, slots) = make_queue_with_profile(1, 16, 64, profile);
        let mut actor = actor_for_test(&queue, &slots);
        let start = Instant::now();
        let left_caller = |id| {
            let (request, response_rx) = make_request(id, 64);
            drop(response_rx);
            request
        };
        let (_, mut active_rx) = enqueue_into_actor(&queue, &mut actor, "active", None);
        actor.handle_update(None).await;
        active_rx.try_recv().unwrap().unwrap();

        // Two queued requests fill the limit. One receiver is closed, and its lease cleanup is
        // still waiting when the due time passes; the other receiver stays open.
        let transfer = Arc::new(AdmissionLifecycleTransfer::new("left-queued".to_string()));
        let left = left_caller("left-queued");
        assert!(enqueue_request_into_actor(
            &queue,
            &mut actor,
            left,
            Some(Arc::clone(&transfer)),
            None
        ));
        actor.cleanup.enqueue(AdmissionCleanupEntry {
            target: SchedulerCleanupTarget::AdmissionLifecycle(transfer),
        });
        let (queued, mut live_queued_rx) =
            enqueue_into_actor(&queue, &mut actor, "live-queued", None);
        assert!(queued);

        // A limit rejection and an already-expired arrival are counted only when the receiver is
        // still open at the rejection check.
        let left = left_caller("left-limited");
        assert!(!enqueue_request_into_actor(
            &queue, &mut actor, left, None, None
        ));
        let (queued, mut live_limited_rx) =
            enqueue_into_actor(&queue, &mut actor, "live-limited", None);
        assert!(!queued);
        assert!(matches!(
            live_limited_rx.try_recv(),
            Ok(Err(KvSchedulerError::QueueRejected(_)))
        ));
        let left = left_caller("left-expired");
        assert!(!enqueue_request_into_actor(
            &queue,
            &mut actor,
            left,
            None,
            Some(start)
        ));
        let (queued, mut live_expired_rx) =
            enqueue_into_actor(&queue, &mut actor, "live-expired", Some(start));
        assert!(!queued);
        assert!(matches!(
            live_expired_rx.try_recv(),
            Ok(Err(KvSchedulerError::DeadlineExceeded))
        ));
        assert_eq!(class_outcomes(&queue, 0), (7, 1, 1));

        // At the due time the sweep removes both queued requests and counts only the one whose
        // receiver is open. The waiting cleanup then finds nothing left to remove.
        tokio::time::advance(Duration::from_millis(100)).await;
        actor.handle_update(None).await;
        assert!(matches!(
            live_queued_rx.try_recv(),
            Ok(Err(KvSchedulerError::DeadlineExceeded))
        ));
        assert_eq!(queue.pending_count(), 0);
        assert_eq!(class_outcomes(&queue, 0), (7, 2, 1));
        actor.drain_cleanup();
        assert_eq!(queue.pending_count(), 0);
        assert_eq!(class_outcomes(&queue, 0), (7, 2, 1));
        slots.free(&"active".to_string(), decay_now()).unwrap();
        slots.assert_completely_drained(decay_now());
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn caller_leaving_as_the_due_time_passes_during_a_refresh_is_not_a_rejection() {
        let refresher = Arc::new(BlockingRefresher::new(RefreshedOverlap::default()));
        let (queue, slots) = make_queue_with_blocking_refresher(
            1,
            16,
            64,
            Some(0.0),
            Arc::clone(&refresher),
            ADMISSION_CHANNEL_CAPACITY,
        );
        let start = Instant::now();
        let (active, active_rx) = make_request("active", 64);
        queue.enqueue(active).await;
        active_rx.await.unwrap().unwrap();
        let candidate_rx = enqueue_to_class(
            &queue,
            "candidate",
            "default",
            Some(start + Duration::from_secs(12)),
            Some(vec![LocalBlockHash(42)]),
        )
        .await;
        tokio::time::advance(Duration::from_secs(11)).await;
        slots
            .mark_prefill_completed(&"active".to_string(), decay_now())
            .unwrap();
        slots.free(&"active".to_string(), decay_now()).unwrap();
        queue.update().await;
        assert_eq!(refresher.calls.load(Ordering::Relaxed), 1);

        // The receiver closes and the due time passes before the actor runs again, so the response
        // closure and the candidate's own due time end the never-released refresh together. The
        // receiver is already closed at the due-time check, so the rejection is not counted.
        drop(candidate_rx);
        tokio::time::advance(Duration::from_secs(2)).await;
        queue.update().await;
        assert_eq!(queue.pending_count(), 0);
        assert_eq!(class_outcomes(&queue, 0), (2, 0, 0));
        assert_eq!(refresher.calls.load(Ordering::Relaxed), 1);
        slots.assert_completely_drained(decay_now());
    }
}
