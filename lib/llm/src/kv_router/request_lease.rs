// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::HashMap,
    sync::{
        Arc, OnceLock, Weak,
        atomic::{AtomicU8, Ordering},
    },
    time::Duration,
};

use dynamo_router_protocol::attempt::AttemptId;
use dynamo_scheduler::host::SchedulerBookingCleanup;
use dynamo_slot_tracker::host::{
    ReplicaRequestLeaseObserver, SchedulerBookingDescriptor, active_request_expiry_duration,
};
use parking_lot::Mutex;
use tokio_util::sync::CancellationToken;

use super::indexer::ApproximateRequestLease;

const LEASE_QUIET: u8 = 0;
const LEASE_TOUCHED: u8 = 1;
const LEASE_CLAIMED: u8 = 2;

struct LeaseClock(AtomicU8);

impl LeaseClock {
    fn new() -> Self {
        Self(AtomicU8::new(LEASE_TOUCHED))
    }

    fn touch(&self) {
        let _ = self.0.compare_exchange(
            LEASE_QUIET,
            LEASE_TOUCHED,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }

    fn is_active(&self) -> bool {
        self.0.load(Ordering::Acquire) != LEASE_CLAIMED
    }

    fn claim_now(&self) -> bool {
        self.0.swap(LEASE_CLAIMED, Ordering::AcqRel) != LEASE_CLAIMED
    }

    fn reap(&self) -> bool {
        match self.0.load(Ordering::Acquire) {
            LEASE_TOUCHED => {
                let _ = self.0.compare_exchange(
                    LEASE_TOUCHED,
                    LEASE_QUIET,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );
                false
            }
            LEASE_QUIET => self
                .0
                .compare_exchange(
                    LEASE_QUIET,
                    LEASE_CLAIMED,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok(),
            LEASE_CLAIMED => false,
            state => unreachable!("invalid request lease CLOCK state {state}"),
        }
    }
}

struct RequestLeaseRecord {
    clock: LeaseClock,
    booking: SchedulerBookingDescriptor,
    approximate_lru: Option<ApproximateRequestLease>,
    /// Whether this router owns the booking; `false` for a copy mirrored from a peer.
    local: bool,
}

impl RequestLeaseRecord {
    fn new(
        booking: SchedulerBookingDescriptor,
        approximate_lru: Option<ApproximateRequestLease>,
        local: bool,
    ) -> Self {
        Self {
            clock: LeaseClock::new(),
            booking,
            approximate_lru,
            local,
        }
    }
}

struct RequestLeaseManagerInner {
    active: Mutex<ActiveRequestLeases>,
    /// Set once the scheduler exists (`set_scheduler`); the manager is built
    /// first because the scheduler takes it as its lease observer.
    scheduler: OnceLock<SchedulerBookingCleanup>,
}

#[derive(Default)]
struct ActiveRequestLeases {
    by_attempt: HashMap<AttemptId, Arc<RequestLeaseRecord>>,
    current_by_request: HashMap<String, AttemptId>,
}

impl RequestLeaseManagerInner {
    fn insert(&self, record: Arc<RequestLeaseRecord>, track_request_id: bool) {
        let attempt_id = record.booking.attempt_id;
        let mut active = self.active.lock();
        if let Some(existing) = active.by_attempt.get(&attempt_id) {
            if existing.booking == record.booking {
                existing.clock.touch();
                return;
            }
            tracing::error!(
                attempt_id = %attempt_id,
                existing_request_id = %existing.booking.request_id,
                replacement_request_id = %record.booking.request_id,
                "Duplicate request lease attempt ID; preserving the existing lease"
            );
            return;
        }
        if track_request_id {
            active
                .current_by_request
                .insert(record.booking.request_id.clone(), attempt_id);
        }
        active.by_attempt.insert(attempt_id, record);
    }

    fn matching_record(
        &self,
        booking: &SchedulerBookingDescriptor,
    ) -> Option<Arc<RequestLeaseRecord>> {
        self.active
            .lock()
            .by_attempt
            .get(&booking.attempt_id)
            .filter(|record| record.booking == *booking)
            .cloned()
    }

    fn current_record(&self, request_id: &str) -> Option<Arc<RequestLeaseRecord>> {
        let active = self.active.lock();
        let attempt_id = active.current_by_request.get(request_id)?;
        active.by_attempt.get(attempt_id).cloned()
    }

    fn remove(&self, record: &Arc<RequestLeaseRecord>) {
        let attempt_id = record.booking.attempt_id;
        let mut active = self.active.lock();
        if active
            .by_attempt
            .get(&attempt_id)
            .is_some_and(|current| Arc::ptr_eq(current, record))
        {
            active.by_attempt.remove(&attempt_id);
            if active
                .current_by_request
                .get(&record.booking.request_id)
                .is_some_and(|current| *current == attempt_id)
            {
                active.current_by_request.remove(&record.booking.request_id);
            }
        }
    }

    /// The scheduler cleanup, or `None` (logged) for an event that arrives
    /// before `set_scheduler`; that is a wiring bug and the booking is not
    /// released here.
    fn scheduler(
        &self,
        event: &str,
        booking: &SchedulerBookingDescriptor,
    ) -> Option<&SchedulerBookingCleanup> {
        let scheduler = self.scheduler.get();
        if scheduler.is_none() {
            tracing::error!(
                request_id = %booking.request_id,
                event,
                "request lifecycle event before the lease manager's scheduler was set; the booking will not be released"
            );
        }
        scheduler
    }

    fn enqueue_completion(&self, record: &RequestLeaseRecord) {
        if let Some(scheduler) = self.scheduler("completed", &record.booking) {
            scheduler.enqueue(record.booking.clone());
        }
        if let Some(approximate_lru) = &record.approximate_lru {
            approximate_lru.release_now();
        }
    }

    fn enqueue_expiry(&self, record: &RequestLeaseRecord) {
        // NOTE: Request-liveness expiry is deliberately isolated to this router.
        // It publishes nothing: only an explicit lifecycle completion sends `Free`
        // to peer routers. A peer's mirrored copy of a request expired here is
        // reclaimed by that peer's own reaper once this router's keepalives stop.
        if let Some(scheduler) = self.scheduler("expired", &record.booking) {
            scheduler.enqueue_expired(record.booking.clone());
        }
        if let Some(approximate_lru) = &record.approximate_lru {
            approximate_lru.release_now();
        }
    }

    fn reap(&self) {
        let records = self
            .active
            .lock()
            .by_attempt
            .values()
            .cloned()
            .collect::<Vec<_>>();
        for record in records {
            if !record.clock.reap() {
                continue;
            }
            self.remove(&record);
            self.enqueue_expiry(&record);
        }
    }

    /// Re-send the prefill-completed event for every local lease past prefill, so
    /// peer routers keep their mirrored copies alive. Returns the number sent.
    fn publish_keepalives(&self) -> usize {
        let Some(scheduler) = self.scheduler.get() else {
            return 0;
        };
        let records = self
            .active
            .lock()
            .by_attempt
            .values()
            .filter(|record| record.local && record.clock.is_active())
            .cloned()
            .collect::<Vec<_>>();
        records
            .iter()
            .filter(|record| scheduler.publish_prefill_keepalive(&record.booking))
            .count()
    }
}

/// One request-liveness coordinator and periodic reaper per `KvRouter`.
#[derive(Clone)]
pub(crate) struct RequestLeaseManager {
    inner: Arc<RequestLeaseManagerInner>,
}

impl RequestLeaseManager {
    pub(crate) fn new(cancellation: CancellationToken) -> Self {
        let inner = Arc::new(RequestLeaseManagerInner {
            active: Mutex::new(ActiveRequestLeases::default()),
            scheduler: OnceLock::new(),
        });
        start_reaper(
            Arc::downgrade(&inner),
            active_request_expiry_duration(),
            cancellation,
        );
        Self { inner }
    }

    /// Install the scheduler cleanup. Must run before any lifecycle event can
    /// reach this manager (the replica ingress starts after it).
    pub(crate) fn set_scheduler(&self, scheduler: SchedulerBookingCleanup) {
        let installed = self.inner.scheduler.set(scheduler).is_ok();
        debug_assert!(installed, "request lease manager scheduler set twice");
    }

    pub(crate) fn replica_observer(&self) -> Arc<dyn ReplicaRequestLeaseObserver> {
        // The scheduler actor owns slots and their observer. A strong manager reference
        // here would keep the actor's own command sender alive through booking cleanup.
        Arc::new(WeakRequestLeaseObserver {
            manager: Arc::downgrade(&self.inner),
        })
    }

    pub(crate) fn register_local(
        &self,
        booking: SchedulerBookingDescriptor,
        approximate_lru: Option<ApproximateRequestLease>,
    ) -> RequestAttemptLease {
        let record = Arc::new(RequestLeaseRecord::new(booking, approximate_lru, true));
        self.inner.insert(Arc::clone(&record), false);
        RequestAttemptLease {
            manager: self.clone(),
            record,
        }
    }

    pub(crate) fn register_detached(
        &self,
        booking: SchedulerBookingDescriptor,
        approximate_lru: Option<ApproximateRequestLease>,
    ) -> DetachedRequestLeaseEnrollment {
        let record = Arc::new(RequestLeaseRecord::new(booking, approximate_lru, true));
        self.inner.insert(Arc::clone(&record), true);
        DetachedRequestLeaseEnrollment {
            manager: self.clone(),
            record,
            armed: true,
        }
    }

    pub(crate) fn touch_request(&self, request_id: &str) {
        if let Some(record) = self.inner.current_record(request_id) {
            record.clock.touch();
        }
    }

    pub(crate) async fn finish_request(&self, request_id: &str) -> bool {
        let Some(record) = self.inner.current_record(request_id) else {
            return false;
        };
        self.finish(&record).await;
        true
    }

    fn register_remote(&self, booking: SchedulerBookingDescriptor) {
        self.inner.insert(
            Arc::new(RequestLeaseRecord::new(booking, None, false)),
            false,
        );
    }

    fn touch_booking(&self, booking: &SchedulerBookingDescriptor) {
        if let Some(record) = self.inner.matching_record(booking) {
            record.clock.touch();
        }
    }

    fn complete_remote(&self, booking: &SchedulerBookingDescriptor) {
        let Some(record) = self.inner.matching_record(booking) else {
            return;
        };
        if record.clock.claim_now() {
            self.inner.remove(&record);
        }
    }

    fn enqueue_completion(&self, record: &Arc<RequestLeaseRecord>) {
        if !record.clock.claim_now() {
            return;
        }
        self.inner.remove(record);
        self.inner.enqueue_completion(record);
    }

    /// Returns `false` when another lifecycle path already claimed the record.
    async fn finish(&self, record: &Arc<RequestLeaseRecord>) -> bool {
        if !record.clock.claim_now() {
            return false;
        }
        self.inner.remove(record);

        // Release the scheduler booking and start LRU cleanup before the first
        // await. Cancellation of this future cannot strand either cleanup.
        let scheduler_ack = self
            .inner
            .scheduler("finished", &record.booking)
            .map(|scheduler| scheduler.enqueue_acknowledged(record.booking.clone()));
        let lru_ack = record
            .approximate_lru
            .as_ref()
            .map(ApproximateRequestLease::begin_finish)
            .transpose();

        if let Some(scheduler_ack) = scheduler_ack
            && let Err(error) = scheduler_ack.wait().await
        {
            tracing::warn!(
                request_id = %record.booking.request_id,
                worker = ?record.booking.worker,
                attempt_id = %record.booking.attempt_id,
                %error,
                "Failed to release scheduler booking"
            );
        }
        match lru_ack {
            Ok(Some(Some(ack))) => {
                if let Err(error) = ack.wait().await {
                    tracing::warn!(
                        request_id = %record.booking.request_id,
                        worker = ?record.booking.worker,
                        attempt_id = %record.booking.attempt_id,
                        %error,
                        "Failed to release approximate LRU request lease"
                    );
                }
            }
            Ok(Some(None)) | Ok(None) => {}
            Err(error) => tracing::warn!(
                request_id = %record.booking.request_id,
                worker = ?record.booking.worker,
                attempt_id = %record.booking.attempt_id,
                %error,
                "Failed to enqueue approximate LRU request release"
            ),
        }
        true
    }

    /// Release the scheduler booking now, even if another path owns the
    /// record's cleanup and has not applied it yet. Attempt-scoped, so it is a
    /// no-op once that cleanup lands and cannot free a later attempt.
    async fn release_booking(&self, booking: &SchedulerBookingDescriptor) {
        let Some(scheduler) = self.inner.scheduler("released", booking) else {
            return;
        };
        if let Err(error) = scheduler.enqueue_and_wait(booking.clone()).await {
            tracing::warn!(
                request_id = %booking.request_id,
                worker = ?booking.worker,
                attempt_id = %booking.attempt_id,
                %error,
                "Failed to release scheduler booking"
            );
        }
    }
}

struct WeakRequestLeaseObserver {
    manager: Weak<RequestLeaseManagerInner>,
}

impl ReplicaRequestLeaseObserver for WeakRequestLeaseObserver {
    fn admitted(&self, booking: SchedulerBookingDescriptor) {
        if let Some(inner) = self.manager.upgrade() {
            RequestLeaseManager { inner }.register_remote(booking);
        }
    }

    fn progressed(&self, booking: &SchedulerBookingDescriptor) {
        if let Some(inner) = self.manager.upgrade() {
            RequestLeaseManager { inner }.touch_booking(booking);
        }
    }

    fn completed(&self, booking: &SchedulerBookingDescriptor) {
        if let Some(inner) = self.manager.upgrade() {
            RequestLeaseManager { inner }.complete_remote(booking);
        }
    }
}

pub(crate) struct RequestAttemptLease {
    manager: RequestLeaseManager,
    record: Arc<RequestLeaseRecord>,
}

impl RequestAttemptLease {
    pub(crate) fn booking(&self) -> &SchedulerBookingDescriptor {
        &self.record.booking
    }

    pub(crate) fn touch(&self) {
        self.record.clock.touch();
    }

    pub(crate) fn is_active(&self) -> bool {
        self.record.clock.is_active()
    }

    pub(crate) async fn finish(&self) {
        self.manager.finish(&self.record).await;
    }

    pub(crate) fn release_handle(&self) -> RequestAttemptRelease {
        RequestAttemptRelease {
            manager: self.manager.clone(),
            record: Arc::clone(&self.record),
        }
    }
}

/// Releases one attempt's booking before the lease that owns it completes.
/// Release is idempotent and scoped to this attempt, so the owner's eventual
/// completion is a no-op and cannot free a later attempt of the same request.
/// The booking is gone when `release` returns, so a migration retry can
/// re-book the same request ID.
pub(crate) struct RequestAttemptRelease {
    manager: RequestLeaseManager,
    record: Arc<RequestLeaseRecord>,
}

impl RequestAttemptRelease {
    pub(crate) async fn release(&self) {
        if !self.manager.finish(&self.record).await {
            self.manager.release_booking(&self.record.booking).await;
        }
    }
}

impl std::fmt::Debug for RequestAttemptRelease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RequestAttemptRelease")
            .field("booking", &self.record.booking)
            .finish()
    }
}

impl Drop for RequestAttemptLease {
    fn drop(&mut self) {
        self.manager.enqueue_completion(&self.record);
    }
}

/// Cancellation owner while a public admission installs its detached lifecycle.
/// Once committed, the manager retains the record until explicit completion or expiry.
#[must_use = "detached request enrollment must be committed or cleaned up"]
pub(crate) struct DetachedRequestLeaseEnrollment {
    manager: RequestLeaseManager,
    record: Arc<RequestLeaseRecord>,
    armed: bool,
}

impl DetachedRequestLeaseEnrollment {
    pub(crate) fn commit(mut self) {
        self.armed = false;
    }

    pub(crate) async fn finish(mut self) {
        self.manager.finish(&self.record).await;
        self.armed = false;
    }
}

impl Drop for DetachedRequestLeaseEnrollment {
    fn drop(&mut self) {
        if self.armed {
            self.manager.enqueue_completion(&self.record);
        }
    }
}

fn start_reaper(
    manager: Weak<RequestLeaseManagerInner>,
    scan_interval: Duration,
    cancellation: CancellationToken,
) {
    tokio::spawn(async move {
        // Ticks run at half the scan interval. Keepalives go out on every tick, so
        // a peer's mirrored lease is touched inside each of its own scan windows
        // whatever the phase between the two reapers. Scans keep the full interval.
        let mut interval = tokio::time::interval(scan_interval / 2);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut scan_tick = true;
        loop {
            tokio::select! {
                _ = cancellation.cancelled() => break,
                _ = interval.tick() => {
                    let Some(manager) = manager.upgrade() else {
                        break;
                    };
                    if scan_tick {
                        // NOTE: This is deliberately a two-scan, second-chance (2S)
                        // approximation. A touched lease becomes quiet on one scan and
                        // is eligible for cleanup on the next. Cache-retention TTL is
                        // a separate policy and never enters this manager.
                        manager.reap();
                    }
                    scan_tick = !scan_tick;
                    manager.publish_keepalives();
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn replica_observer_forwards_live_callbacks_without_retaining_manager() {
        use dynamo_router_protocol::worker::WorkerWithDpRank;

        use crate::kv_router::{sequence::SequenceRequest, tests::tracked_router};

        let router = tracked_router("replica-lease-observer").await;
        let request_id = "replica-lease-callbacks".to_string();
        let worker = WorkerWithDpRank::new(0, 0);
        let attempt_id = router
            .selection
            .scheduler()
            .add_request_admitted(SequenceRequest {
                request_id: request_id.clone(),
                token_sequence: None,
                track_prefill_tokens: false,
                expected_output_tokens: None,
                prefill_load_hint: None,
                worker,
                lora_name: None,
            })
            .await
            .unwrap();
        let booking = SchedulerBookingDescriptor {
            request_id: request_id.clone(),
            worker,
            attempt_id,
        };
        let manager = Arc::downgrade(&router.request_leases.inner);
        let observer = router.request_leases.replica_observer();
        assert!(
            router
                .request_leases
                .inner
                .matching_record(&booking)
                .is_none()
        );
        observer.admitted(booking.clone());
        let record = router
            .request_leases
            .inner
            .matching_record(&booking)
            .unwrap();
        assert!(!record.clock.reap());
        observer.progressed(&booking);
        assert!(!record.clock.reap());
        assert!(record.clock.is_active());
        observer.completed(&booking);
        assert!(!record.clock.is_active());
        assert!(
            router
                .request_leases
                .inner
                .matching_record(&booking)
                .is_none()
        );
        router
            .selection
            .scheduler()
            .free(&request_id)
            .await
            .unwrap();
        router.cancellation_token.cancel();
        drop(router);
        assert!(manager.upgrade().is_none());
        observer.admitted(booking.clone());
        observer.progressed(&booking);
        observer.completed(&booking);
        assert!(manager.upgrade().is_none());
    }

    #[tokio::test]
    async fn attempt_release_frees_booking_when_another_path_owns_cleanup() {
        use dynamo_router_protocol::worker::WorkerWithDpRank;

        use crate::kv_router::{sequence::SequenceRequest, tests::tracked_router};

        let router = tracked_router("attempt-release-lost-claim").await;
        let request_id = "attempt-release-lost-claim".to_string();
        let worker = WorkerWithDpRank::new(0, 0);
        let sequence_request = || SequenceRequest {
            request_id: request_id.clone(),
            token_sequence: None,
            track_prefill_tokens: false,
            expected_output_tokens: None,
            prefill_load_hint: None,
            worker,
            lora_name: None,
        };
        let scheduler = router.selection.scheduler();
        let attempt_id = scheduler
            .add_request_admitted(sequence_request())
            .await
            .unwrap();
        let booking = SchedulerBookingDescriptor {
            request_id: request_id.clone(),
            worker,
            attempt_id,
        };
        let lease = router.request_leases.register_local(booking.clone(), None);
        let release = lease.release_handle();

        // Another lifecycle path has claimed the attempt but not yet released it.
        assert!(lease.record.clock.claim_now());
        assert!(scheduler.has_booking(&booking));

        release.release().await;
        assert!(!scheduler.has_booking(&booking));

        let retry_attempt_id = scheduler
            .add_request_admitted(sequence_request())
            .await
            .unwrap();
        let retry_booking = SchedulerBookingDescriptor {
            request_id: request_id.clone(),
            worker,
            attempt_id: retry_attempt_id,
        };
        release.release().await;
        drop(lease);
        assert!(scheduler.has_booking(&retry_booking));

        scheduler.free(&request_id).await.unwrap();
        router.cancellation_token.cancel();
    }

    #[tokio::test]
    async fn keepalive_covers_local_leases_past_prefill_only() {
        use dynamo_router_protocol::worker::WorkerWithDpRank;

        use crate::kv_router::{
            sequence::SequenceRequest, tests::tracked_router_with_replica_sync,
        };

        let router = tracked_router_with_replica_sync("lease-keepalive", true).await;
        let scheduler = router.selection.scheduler();
        let worker = WorkerWithDpRank::new(0, 0);
        let book = |request_id: &'static str, past_prefill: bool| async move {
            let attempt_id = scheduler
                .add_request_admitted(SequenceRequest {
                    request_id: request_id.to_string(),
                    token_sequence: None,
                    track_prefill_tokens: false,
                    expected_output_tokens: None,
                    prefill_load_hint: None,
                    worker,
                    lora_name: None,
                })
                .await
                .unwrap();
            let booking = SchedulerBookingDescriptor {
                request_id: request_id.to_string(),
                worker,
                attempt_id,
            };
            if past_prefill {
                scheduler
                    .mark_prefill_completed_if_booking(&booking)
                    .await
                    .unwrap();
            }
            booking
        };
        let leases = &router.request_leases;
        let decoding = leases.register_local(book("decoding", true).await, None);
        let _prefilling = leases.register_local(book("prefilling", false).await, None);
        leases.register_remote(book("mirrored", true).await);

        assert_eq!(leases.inner.publish_keepalives(), 1);
        decoding.finish().await;
        assert_eq!(leases.inner.publish_keepalives(), 0);
        router.cancellation_token.cancel();
    }

    #[tokio::test(start_paused = true)]
    async fn reaper_half_ticks_keep_the_two_scan_lifetime() {
        use dynamo_router_protocol::{attempt::AttemptIdMinter, worker::WorkerWithDpRank};

        let period = Duration::from_secs(60);
        let inner = Arc::new(RequestLeaseManagerInner {
            active: Mutex::new(ActiveRequestLeases::default()),
            scheduler: OnceLock::new(),
        });
        let cancel = CancellationToken::new();
        start_reaper(Arc::downgrade(&inner), period, cancel.clone());
        // The reaper's first tick is immediate; register after it.
        tokio::task::yield_now().await;
        let lease = RequestLeaseManager { inner }.register_local(
            SchedulerBookingDescriptor {
                request_id: "untouched".to_string(),
                worker: WorkerWithDpRank::new(0, 0),
                attempt_id: AttemptIdMinter::new().mint(),
            },
            None,
        );

        let just_after = Duration::from_millis(1);
        tokio::time::sleep(period * 3 / 2 + just_after).await;
        assert!(lease.is_active());
        tokio::time::sleep(period / 2).await;
        assert!(!lease.is_active());
        cancel.cancel();
    }

    #[test]
    fn clock_coalesces_progress_and_cannot_resurrect_a_claimed_lease() {
        let clock = LeaseClock::new();

        assert!(!clock.reap());
        clock.touch();
        clock.touch();
        assert!(!clock.reap());
        assert!(clock.reap());
        assert!(!clock.is_active());

        clock.touch();
        assert!(!clock.is_active());
        assert!(!clock.reap());
    }
}
