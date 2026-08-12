// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, hash_map::Entry},
    sync::{Arc, Weak},
    time::Duration,
};

use anyhow::Result;
use dynamo_kv_router::protocols::{BlockExtraInfo, RoutingConstraints, WorkerId};
use futures::{StreamExt, stream};
use parking_lot::Mutex;
use tokio::time::{Instant, MissedTickBehavior};
use tokio_util::sync::CancellationToken;

use super::{
    InnerPrefillRouter, PrefillError, PrefillLifecycleState, PrefillQueryOutcome, PrefillRouter,
};
use crate::kv_router::{KvRouter, sequence::SequenceError};

const PREFILL_SCHEDULER_ID_PREFIX: &str = "epp-prefill/";
const ACTIVE_REQUEST_EXPIRY_DURATION: Duration = Duration::from_secs(300);
const RESERVATION_EXPIRY_GRACE: Duration = Duration::from_secs(60);
const CANCELLED_RESERVATION_RETENTION: Duration = Duration::from_secs(60);
const RESERVATION_REAPER_INTERVAL: Duration = Duration::from_secs(30);
const RESERVATION_RELEASE_TIMEOUT: Duration = Duration::from_secs(5);
// Four timeout waves cap each batch at 20 seconds, below the 30-second interval.
const RESERVATION_REAPER_BATCH_SIZE: usize = 128;
const RESERVATION_REAPER_CONCURRENCY: usize = 32;

struct ActivePrefillReservation {
    chooser: Arc<KvRouter>,
    scheduler_id: String,
    created_at: Instant,
    reap_attempts: u64,
}

impl Clone for ActivePrefillReservation {
    fn clone(&self) -> Self {
        Self {
            chooser: self.chooser.clone(),
            scheduler_id: self.scheduler_id.clone(),
            created_at: self.created_at,
            reap_attempts: self.reap_attempts,
        }
    }
}

enum PrefillReservationEntry {
    /// Cancelling this token drops the selection future. Release/1.3.0 keeps a
    /// queued entry until its next dequeue, where the closed response receiver
    /// prevents scheduler booking.
    Pending {
        cancellation: CancellationToken,
        created_at: Instant,
        claimed: bool,
    },
    Active(ActivePrefillReservation),
    /// Preserve a cancellation that reaches Rust before the blocking reserve
    /// call creates the pending entry.
    Cancelled {
        created_at: Instant,
    },
}

enum BeginReservation {
    Pending(CancellationToken),
    Cancelled,
    AlreadyExists,
}

enum Activation {
    Active,
    Cancelled,
    Lost,
}

pub struct EppReservationManager {
    entries: Mutex<HashMap<String, PrefillReservationEntry>>,
}

impl Default for EppReservationManager {
    fn default() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }
}

impl EppReservationManager {
    fn begin_entry(&self, reservation_id: &str) -> BeginReservation {
        let mut entries = self.entries.lock();
        match entries.entry(reservation_id.to_string()) {
            Entry::Vacant(entry) => {
                let cancellation = CancellationToken::new();
                entry.insert(PrefillReservationEntry::Pending {
                    cancellation: cancellation.clone(),
                    created_at: Instant::now(),
                    claimed: false,
                });
                BeginReservation::Pending(cancellation)
            }
            Entry::Occupied(entry) => match entry.get() {
                PrefillReservationEntry::Cancelled { .. } => {
                    entry.remove();
                    BeginReservation::Cancelled
                }
                PrefillReservationEntry::Pending { cancellation, .. } => {
                    BeginReservation::Pending(cancellation.clone())
                }
                PrefillReservationEntry::Active(_) => BeginReservation::AlreadyExists,
            },
        }
    }

    // Claim the one scheduler-admission attempt for this reservation. `begin` remains
    // idempotent while preprocessing runs, but only one `reserve` may turn its Pending
    // state into an admission future.
    fn claim_pending(&self, reservation_id: &str) -> BeginReservation {
        let mut entries = self.entries.lock();
        match entries.entry(reservation_id.to_string()) {
            Entry::Vacant(entry) => {
                let cancellation = CancellationToken::new();
                entry.insert(PrefillReservationEntry::Pending {
                    cancellation: cancellation.clone(),
                    created_at: Instant::now(),
                    claimed: true,
                });
                BeginReservation::Pending(cancellation)
            }
            Entry::Occupied(mut entry) => {
                if matches!(entry.get(), PrefillReservationEntry::Cancelled { .. }) {
                    entry.remove();
                    return BeginReservation::Cancelled;
                }

                match entry.get_mut() {
                    PrefillReservationEntry::Pending {
                        cancellation,
                        claimed,
                        ..
                    } => {
                        if *claimed {
                            BeginReservation::AlreadyExists
                        } else {
                            *claimed = true;
                            BeginReservation::Pending(cancellation.clone())
                        }
                    }
                    PrefillReservationEntry::Active(_) => BeginReservation::AlreadyExists,
                    PrefillReservationEntry::Cancelled { .. } => unreachable!("handled above"),
                }
            }
        }
    }

    fn activate(&self, reservation_id: &str, reservation: ActivePrefillReservation) -> Activation {
        let mut entries = self.entries.lock();
        let Some(entry) = entries.get(reservation_id) else {
            return Activation::Lost;
        };

        let cancelled = match entry {
            PrefillReservationEntry::Pending {
                cancellation,
                claimed: true,
                ..
            } => cancellation.is_cancelled(),
            // A reaped pending entry retains a cancellation tombstone. If its admission
            // won just before cancellation, retain the active booking so release/reaper
            // can retry cleanup rather than leaking it until scheduler expiry.
            PrefillReservationEntry::Cancelled { .. } => true,
            PrefillReservationEntry::Pending { .. } | PrefillReservationEntry::Active(_) => {
                return Activation::Lost;
            }
        };

        entries.insert(
            reservation_id.to_string(),
            PrefillReservationEntry::Active(reservation),
        );
        if cancelled {
            Activation::Cancelled
        } else {
            Activation::Active
        }
    }

    fn remove_pending(&self, reservation_id: &str) {
        let mut entries = self.entries.lock();
        if matches!(
            entries.get(reservation_id),
            Some(PrefillReservationEntry::Pending { claimed: true, .. })
        ) {
            entries.remove(reservation_id);
        }
    }

    // Preprocessing happens before reserve claims the entry. Never let a duplicate
    // caller's preprocessing failure remove an admission another caller already owns.
    fn abort_unclaimed(&self, reservation_id: &str) {
        let mut entries = self.entries.lock();
        let state = match entries.get(reservation_id) {
            Some(PrefillReservationEntry::Pending {
                cancellation,
                claimed: false,
                ..
            }) if cancellation.is_cancelled() => Some(true),
            Some(PrefillReservationEntry::Pending { claimed: false, .. }) => Some(false),
            _ => None,
        };
        match state {
            Some(true) => {
                entries.insert(
                    reservation_id.to_string(),
                    PrefillReservationEntry::Cancelled {
                        created_at: Instant::now(),
                    },
                );
            }
            Some(false) => {
                entries.remove(reservation_id);
            }
            None => {}
        }
    }

    fn cancel_entry(&self, reservation_id: &str) {
        let mut entries = self.entries.lock();
        match entries.entry(reservation_id.to_string()) {
            Entry::Occupied(entry) => {
                if let PrefillReservationEntry::Pending { cancellation, .. } = entry.get() {
                    cancellation.cancel();
                }
            }
            Entry::Vacant(entry) => {
                entry.insert(PrefillReservationEntry::Cancelled {
                    created_at: Instant::now(),
                });
            }
        }
    }

    fn expire_pending(&self, now: Instant, retention: Duration) {
        let mut entries = self.entries.lock();
        for entry in entries.values_mut() {
            let cancellation = match entry {
                PrefillReservationEntry::Pending {
                    cancellation,
                    created_at,
                    ..
                } if now.saturating_duration_since(*created_at) >= retention => {
                    Some(cancellation.clone())
                }
                _ => None,
            };
            if let Some(cancellation) = cancellation {
                cancellation.cancel();
                *entry = PrefillReservationEntry::Cancelled { created_at: now };
            }
        }
    }

    fn get_active(&self, reservation_id: &str) -> Option<ActivePrefillReservation> {
        match self.entries.lock().get(reservation_id) {
            Some(PrefillReservationEntry::Active(reservation)) => Some(reservation.clone()),
            Some(PrefillReservationEntry::Pending { .. })
            | Some(PrefillReservationEntry::Cancelled { .. })
            | None => None,
        }
    }

    fn remove_if_scheduler_id(&self, reservation_id: &str, scheduler_id: &str) {
        let mut entries = self.entries.lock();
        if entries.get(reservation_id).is_some_and(|entry| {
            matches!(entry, PrefillReservationEntry::Active(active) if active.scheduler_id == scheduler_id)
        }) {
            entries.remove(reservation_id);
        }
    }

    // Claim the least-attempted generation so retained failures cannot starve the backlog.
    fn claim_expired_active_ids(
        &self,
        now: Instant,
        retention: Duration,
        limit: usize,
    ) -> Vec<String> {
        if limit == 0 {
            return Vec::new();
        }

        let mut entries = self.entries.lock();
        let minimum_attempts = entries
            .values()
            .filter_map(|entry| match entry {
                PrefillReservationEntry::Active(active)
                    if now.saturating_duration_since(active.created_at) >= retention =>
                {
                    Some(active.reap_attempts)
                }
                _ => None,
            })
            .min();
        let Some(minimum_attempts) = minimum_attempts else {
            return Vec::new();
        };

        entries
            .iter_mut()
            .filter_map(|(reservation_id, entry)| match entry {
                PrefillReservationEntry::Active(active)
                    if now.saturating_duration_since(active.created_at) >= retention
                        && active.reap_attempts == minimum_attempts =>
                {
                    active.reap_attempts = active.reap_attempts.saturating_add(1);
                    Some(reservation_id.clone())
                }
                _ => None,
            })
            .take(limit)
            .collect()
    }

    fn remove_expired_cancellations(&self, now: Instant, retention: Duration) {
        self.entries.lock().retain(|_, entry| {
            !matches!(entry, PrefillReservationEntry::Cancelled { created_at }
                if now.saturating_duration_since(*created_at) >= retention)
        });
    }
}

fn reservation_retention_from_expiry(active_request_expiry: Duration) -> Duration {
    active_request_expiry.saturating_add(RESERVATION_EXPIRY_GRACE)
}

fn reservation_retention() -> Duration {
    reservation_retention_from_expiry(ACTIVE_REQUEST_EXPIRY_DURATION)
}

fn scheduler_id(reservation_id: &str) -> String {
    format!("{PREFILL_SCHEDULER_ID_PREFIX}{reservation_id}")
}

fn ignore_missing_request(result: std::result::Result<(), SequenceError>) -> Result<()> {
    match result {
        Ok(()) | Err(SequenceError::RequestNotFound { .. }) => Ok(()),
        Err(error) => Err(error.into()),
    }
}

impl EppReservationManager {
    /// Register the pending state before potentially slow request preprocessing.
    ///
    /// Calling this repeatedly for the same in-flight booking is idempotent. A
    /// cancellation that arrived first is observed here and never queues work.
    pub fn begin(&self, router: &PrefillRouter, reservation_id: &str) -> Result<()> {
        if reservation_id.is_empty() {
            anyhow::bail!("prefill reservation ID must not be empty");
        }
        if router.lifecycle_state() != PrefillLifecycleState::Active {
            return Err(anyhow::anyhow!(PrefillError::NotActivated));
        }
        if router.prefill_router.get().is_none() {
            return Err(anyhow::anyhow!(PrefillError::NotActivated));
        }

        match self.begin_entry(reservation_id) {
            BeginReservation::Pending(_) => Ok(()),
            BeginReservation::Cancelled => {
                anyhow::bail!("prefill reservation {reservation_id:?} was cancelled")
            }
            BeginReservation::AlreadyExists => {
                anyhow::bail!("prefill reservation {reservation_id:?} already exists")
            }
        }
    }

    /// Drop an unclaimed pending reservation when preprocessing fails before scheduler admission.
    pub fn abort(&self, reservation_id: &str) {
        self.abort_unclaimed(reservation_id);
    }

    /// Atomically select and reserve a prefill worker for an externally dispatched request.
    ///
    /// The scheduler observes the booking before selecting the next request. The caller owns
    /// the returned reservation until first output, terminal completion, or cancellation.
    #[expect(clippy::too_many_arguments)]
    pub async fn reserve(
        &self,
        router: &PrefillRouter,
        reservation_id: &str,
        token_ids: &[u32],
        block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
        lora_name: Option<String>,
        priority_jump: f64,
        strict_priority: u32,
        allowed_worker_ids: Option<std::collections::HashSet<WorkerId>>,
        routing_constraints: RoutingConstraints,
    ) -> Result<PrefillQueryOutcome> {
        if reservation_id.is_empty() {
            anyhow::bail!("prefill reservation ID must not be empty");
        }
        let cancellation = match self.claim_pending(reservation_id) {
            BeginReservation::Pending(cancellation) => cancellation,
            BeginReservation::Cancelled => {
                anyhow::bail!("prefill reservation {reservation_id:?} was cancelled")
            }
            BeginReservation::AlreadyExists => {
                anyhow::bail!("prefill reservation {reservation_id:?} already exists")
            }
        };
        if router.lifecycle_state() != PrefillLifecycleState::Active {
            self.remove_pending(reservation_id);
            return Err(anyhow::anyhow!(PrefillError::NotActivated));
        }
        let Some(inner) = router.prefill_router.get() else {
            self.remove_pending(reservation_id);
            return Err(anyhow::anyhow!(PrefillError::NotActivated));
        };
        let InnerPrefillRouter::KvRouter(router) = inner else {
            self.remove_pending(reservation_id);
            return Err(anyhow::anyhow!(PrefillError::NotActivated));
        };
        let chooser = router.chooser.clone();
        let scheduler_id = scheduler_id(reservation_id);

        let outcome = tokio::select! {
            biased;
            _ = cancellation.cancelled() => {
                self.remove_pending(reservation_id);
                anyhow::bail!("prefill reservation {reservation_id:?} was cancelled")
            }
            outcome = chooser.find_best_match_details(
                Some(&scheduler_id),
                token_ids,
                block_mm_infos,
                None,
                true,
                false,
                lora_name,
                priority_jump,
                strict_priority,
                None,
                None,
                allowed_worker_ids,
                routing_constraints,
            ) => outcome,
        };
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(error) => {
                self.remove_pending(reservation_id);
                return Err(error);
            }
        };

        match outcome {
            crate::kv_router::FindBestMatchOutcome::Routed { worker, .. } => {
                let reservation = ActivePrefillReservation {
                    chooser: chooser.clone(),
                    scheduler_id: scheduler_id.clone(),
                    created_at: Instant::now(),
                    reap_attempts: 0,
                };
                match self.activate(reservation_id, reservation) {
                    Activation::Active => Ok(PrefillQueryOutcome::Routed {
                        worker_id: worker.worker_id,
                        dp_rank: Some(worker.dp_rank),
                    }),
                    Activation::Cancelled => {
                        if let Err(error) = self.release(reservation_id).await {
                            tracing::warn!(
                                %reservation_id,
                                %error,
                                "Failed to release cancelled EPP prefill reservation; retaining it for reaper retry"
                            );
                        }
                        anyhow::bail!("prefill reservation {reservation_id:?} was cancelled");
                    }
                    Activation::Lost => {
                        if let Err(error) =
                            ignore_missing_request(chooser.free(&scheduler_id).await)
                        {
                            tracing::warn!(
                                %reservation_id,
                                %error,
                                "Failed to release EPP prefill reservation with lost ownership"
                            );
                        }
                        anyhow::bail!("prefill reservation {reservation_id:?} lost ownership");
                    }
                }
            }
            crate::kv_router::FindBestMatchOutcome::QueueRejected { rejection } => {
                self.remove_pending(reservation_id);
                Ok(PrefillQueryOutcome::QueueRejected { rejection })
            }
        }
    }

    /// Cancel a pending prefill reservation without waiting for scheduler cleanup.
    ///
    /// An active reservation remains owned by the normal booking lifecycle. If cancellation
    /// races with admission, the Go caller drains the reserve result and releases that booking.
    pub fn cancel(&self, reservation_id: &str) {
        if !reservation_id.is_empty() {
            self.cancel_entry(reservation_id);
        }
    }

    /// Release a prefill reservation. Missing reservations are idempotent no-ops.
    pub async fn release(&self, reservation_id: &str) -> Result<()> {
        let Some(reservation) = self.get_active(reservation_id) else {
            return Ok(());
        };

        ignore_missing_request(reservation.chooser.free(&reservation.scheduler_id).await)?;
        self.remove_if_scheduler_id(reservation_id, &reservation.scheduler_id);
        Ok(())
    }

    async fn release_expired_batch(self: &Arc<Self>, reservation_ids: Vec<String>) {
        stream::iter(reservation_ids)
            .for_each_concurrent(RESERVATION_REAPER_CONCURRENCY, |reservation_id| {
                let manager = Arc::clone(self);
                async move {
                    match tokio::time::timeout(
                        RESERVATION_RELEASE_TIMEOUT,
                        manager.release(&reservation_id),
                    )
                    .await
                    {
                        Ok(Ok(())) => {}
                        Ok(Err(error)) => {
                            tracing::warn!(
                                %reservation_id,
                                %error,
                                "Failed to expire stale EPP prefill reservation"
                            );
                        }
                        Err(_) => {
                            tracing::warn!(
                                %reservation_id,
                                timeout_secs = RESERVATION_RELEASE_TIMEOUT.as_secs(),
                                "Timed out expiring stale EPP prefill reservation"
                            );
                        }
                    }
                }
            })
            .await;
    }

    /// Reap stale EPP-owned bookings. The task only weakly holds the manager,
    /// so destroying the C router handle stops it without affecting the
    /// prefill router lifecycle.
    pub fn spawn_reaper(manager: &Arc<Self>) {
        let manager: Weak<Self> = Arc::downgrade(manager);

        tokio::spawn(async move {
            let retention = reservation_retention();
            let mut interval = tokio::time::interval_at(
                Instant::now() + RESERVATION_REAPER_INTERVAL,
                RESERVATION_REAPER_INTERVAL,
            );
            interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

            loop {
                interval.tick().await;
                let Some(manager) = manager.upgrade() else {
                    return;
                };
                let now = Instant::now();
                manager.expire_pending(now, retention);
                manager.remove_expired_cancellations(now, CANCELLED_RESERVATION_RETENTION);
                let expired =
                    manager.claim_expired_active_ids(now, retention, RESERVATION_REAPER_BATCH_SIZE);
                manager.release_expired_batch(expired).await;
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancellation_before_reservation_is_observed() {
        let registry = EppReservationManager::default();
        registry.cancel_entry("reservation-1");

        assert!(matches!(
            registry.begin_entry("reservation-1"),
            BeginReservation::Cancelled
        ));
        assert!(matches!(
            registry.begin_entry("reservation-1"),
            BeginReservation::Pending(_)
        ));
    }

    #[test]
    fn pre_registered_pending_reservation_survives_tombstone_reaping() {
        let registry = EppReservationManager::default();
        let BeginReservation::Pending(cancellation) = registry.begin_entry("reservation-1") else {
            panic!("expected pending reservation");
        };

        registry.cancel_entry("reservation-1");
        registry.remove_expired_cancellations(
            Instant::now() + CANCELLED_RESERVATION_RETENTION + Duration::from_secs(1),
            CANCELLED_RESERVATION_RETENTION,
        );

        assert!(cancellation.is_cancelled());
        let BeginReservation::Pending(existing) = registry.begin_entry("reservation-1") else {
            panic!("expected pending reservation to remain registered");
        };
        assert!(existing.is_cancelled());
    }

    #[test]
    fn only_one_reserve_claims_pending_reservation() {
        let registry = EppReservationManager::default();
        assert!(matches!(
            registry.begin_entry("reservation-1"),
            BeginReservation::Pending(_)
        ));
        assert!(matches!(
            registry.claim_pending("reservation-1"),
            BeginReservation::Pending(_)
        ));
        assert!(matches!(
            registry.claim_pending("reservation-1"),
            BeginReservation::AlreadyExists
        ));
    }

    #[test]
    fn expired_pending_reservation_becomes_cancelled_tombstone() {
        let registry = EppReservationManager::default();
        let BeginReservation::Pending(cancellation) = registry.begin_entry("reservation-1") else {
            panic!("expected pending reservation");
        };

        registry.expire_pending(
            Instant::now() + Duration::from_secs(61),
            Duration::from_secs(60),
        );

        assert!(cancellation.is_cancelled());
        assert!(matches!(
            registry.claim_pending("reservation-1"),
            BeginReservation::Cancelled
        ));
        assert!(matches!(
            registry.begin_entry("reservation-1"),
            BeginReservation::Pending(_)
        ));
    }

    #[test]
    fn cancellation_signals_pending_reservation() {
        let registry = EppReservationManager::default();
        let BeginReservation::Pending(cancellation) = registry.begin_entry("reservation-1") else {
            panic!("expected pending reservation");
        };

        registry.cancel_entry("reservation-1");
        assert!(cancellation.is_cancelled());
        registry.remove_pending("reservation-1");
    }

    #[test]
    fn reservation_retention_tracks_active_request_expiry() {
        assert_eq!(
            reservation_retention_from_expiry(Duration::from_secs(30)),
            Duration::from_secs(90)
        );
        assert_eq!(
            reservation_retention_from_expiry(Duration::from_secs(3600)),
            Duration::from_secs(3660)
        );
    }
}
