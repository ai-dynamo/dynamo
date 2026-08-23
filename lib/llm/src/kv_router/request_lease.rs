// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, hash_map::Entry},
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
    active: Mutex<HashMap<AttemptId, Arc<RequestLeaseRecord>>>,
    scheduler: SchedulerBookingCleanup,
}

impl RequestLeaseManagerInner {
    fn insert(&self, record: Arc<RequestLeaseRecord>) {
        let attempt_id = record.booking.attempt_id;
        match self.active.lock().entry(attempt_id) {
            Entry::Vacant(entry) => {
                entry.insert(record);
            }
            Entry::Occupied(entry) => {
                if entry.get().booking == record.booking {
                    entry.get().clock.touch();
                    return;
                }
                tracing::error!(
                    attempt_id = %attempt_id,
                    existing_request_id = %entry.get().booking.request_id,
                    replacement_request_id = %record.booking.request_id,
                    "Duplicate request lease attempt ID; preserving the existing lease"
                );
            }
        }
    }

    fn matching_record(
        &self,
        booking: &SchedulerBookingDescriptor,
    ) -> Option<Arc<RequestLeaseRecord>> {
        self.active
            .lock()
            .get(&booking.attempt_id)
            .filter(|record| record.booking == *booking)
            .cloned()
    }

    fn remove(&self, record: &Arc<RequestLeaseRecord>) {
        let attempt_id = record.booking.attempt_id;
        let mut active = self.active.lock();
        if active
            .get(&attempt_id)
            .is_some_and(|current| Arc::ptr_eq(current, record))
        {
            active.remove(&attempt_id);
        }
    }

    fn enqueue_cleanup(&self, record: &RequestLeaseRecord) {
        self.scheduler.enqueue(record.booking.clone());
        if let Some(approximate_lru) = &record.approximate_lru {
            approximate_lru.release_now();
        }
    }

    fn reap(&self) {
        let records = self.active.lock().values().cloned().collect::<Vec<_>>();
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
            active: Mutex::new(HashMap::new()),
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
        self.inner.insert(Arc::clone(&record));
        RequestAttemptLease {
            manager: self.clone(),
            record,
        }
    }

    fn register_remote(&self, booking: SchedulerBookingDescriptor) {
        self.inner
            .insert(Arc::new(RequestLeaseRecord::new(booking, None)));
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
