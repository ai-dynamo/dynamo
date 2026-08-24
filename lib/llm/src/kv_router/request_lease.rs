// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::HashMap,
    sync::{
        Arc, Weak,
        atomic::{AtomicU8, Ordering},
    },
    time::Duration,
};

use dynamo_kv_router::{
    multi_worker_sequence::{ReplicaRequestLeaseObserver, active_request_expiry_duration},
    scheduling::AttemptId,
};
use parking_lot::Mutex;
use tokio_util::sync::CancellationToken;

use super::{
    indexer::ApproximateRequestLease,
    scheduler::{SchedulerBookingCleanup, SchedulerBookingDescriptor},
};

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
}

impl RequestLeaseRecord {
    fn new(
        booking: SchedulerBookingDescriptor,
        approximate_lru: Option<ApproximateRequestLease>,
    ) -> Self {
        Self {
            clock: LeaseClock::new(),
            booking,
            approximate_lru,
        }
    }
}

struct RequestLeaseManagerInner {
    active: Mutex<ActiveRequestLeases>,
    scheduler: SchedulerBookingCleanup,
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

    fn enqueue_cleanup(&self, record: &RequestLeaseRecord) {
        self.scheduler.enqueue(record.booking.clone());
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
            self.enqueue_cleanup(&record);
        }
    }
}

/// One request-liveness coordinator and periodic reaper per `KvRouter`.
#[derive(Clone)]
pub(crate) struct RequestLeaseManager {
    inner: Arc<RequestLeaseManagerInner>,
}

impl RequestLeaseManager {
    pub(crate) fn new(scheduler: SchedulerBookingCleanup, cancellation: CancellationToken) -> Self {
        let inner = Arc::new(RequestLeaseManagerInner {
            active: Mutex::new(ActiveRequestLeases::default()),
            scheduler,
        });
        start_reaper(
            Arc::downgrade(&inner),
            active_request_expiry_duration(),
            cancellation,
        );
        Self { inner }
    }

    pub(crate) fn register_local(
        &self,
        booking: SchedulerBookingDescriptor,
        approximate_lru: Option<ApproximateRequestLease>,
    ) -> RequestAttemptLease {
        let record = Arc::new(RequestLeaseRecord::new(booking, approximate_lru));
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
    ) {
        self.inner.insert(
            Arc::new(RequestLeaseRecord::new(booking, approximate_lru)),
            true,
        );
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

    pub(crate) async fn finish_booking(&self, booking: &SchedulerBookingDescriptor) -> bool {
        let Some(record) = self.inner.matching_record(booking) else {
            return false;
        };
        self.finish(&record).await;
        true
    }

    fn register_remote(&self, booking: SchedulerBookingDescriptor) {
        self.inner
            .insert(Arc::new(RequestLeaseRecord::new(booking, None)), false);
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
        self.inner.enqueue_cleanup(record);
    }

    async fn finish(&self, record: &Arc<RequestLeaseRecord>) {
        if !record.clock.claim_now() {
            return;
        }
        self.inner.remove(record);

        // Enqueue both subsystem commands before the first await. Cancellation of
        // the finishing future therefore cannot strand either cleanup.
        let scheduler_ack = self
            .inner
            .scheduler
            .enqueue_acknowledged(record.booking.clone());
        let lru_ack = record
            .approximate_lru
            .as_ref()
            .map(ApproximateRequestLease::begin_finish)
            .transpose();

        if let Err(error) = scheduler_ack.wait().await {
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
    }
}

impl ReplicaRequestLeaseObserver for RequestLeaseManager {
    fn admitted(&self, booking: SchedulerBookingDescriptor) {
        self.register_remote(booking);
    }

    fn progressed(&self, booking: &SchedulerBookingDescriptor) {
        self.touch_booking(booking);
    }

    fn completed(&self, booking: &SchedulerBookingDescriptor) {
        self.complete_remote(booking);
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
}

impl Drop for RequestAttemptLease {
    fn drop(&mut self) {
        self.manager.enqueue_completion(&self.record);
    }
}

fn start_reaper(
    manager: Weak<RequestLeaseManagerInner>,
    scan_interval: Duration,
    cancellation: CancellationToken,
) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(scan_interval);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                _ = cancellation.cancelled() => break,
                _ = interval.tick() => {
                    let Some(manager) = manager.upgrade() else {
                        break;
                    };
                    // NOTE: This is deliberately a two-scan, second-chance (2S)
                    // approximation. A touched lease becomes quiet on one scan and
                    // is eligible for cleanup on the next. Cache-retention TTL is
                    // a separate policy and never enters this manager.
                    manager.reap();
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

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
