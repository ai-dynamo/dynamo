// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, hash_map::Entry},
    sync::{Arc, Weak},
    time::Duration,
};

use anyhow::Result;
use dynamo_kv_router::{
    protocols::{BlockExtraInfo, RoutingConstraints, WorkerId},
    sequence::DEFAULT_ACTIVE_REQUEST_EXPIRY_DURATION,
};
use parking_lot::Mutex;
use tokio::time::{Instant, MissedTickBehavior};
use tokio_util::sync::CancellationToken;

use super::{
    InnerPrefillRouter, PrefillError, PrefillLifecycleState, PrefillQueryOutcome, PrefillRouter,
};
use crate::kv_router::{KvRouter, sequence::SequenceError};

const PREFILL_SCHEDULER_ID_PREFIX: &str = "epp-prefill/";
const ACTIVE_REQUEST_EXPIRY_ENV: &str = "DYN_ROUTER_ACTIVE_REQUEST_EXPIRY_SECS";
const RESERVATION_EXPIRY_GRACE: Duration = Duration::from_secs(60);
const CANCELLED_RESERVATION_RETENTION: Duration = Duration::from_secs(60);
const RESERVATION_REAPER_INTERVAL: Duration = Duration::from_secs(30);

struct ActivePrefillReservation {
    chooser: Arc<KvRouter>,
    scheduler_id: String,
    created_at: Instant,
}

impl Clone for ActivePrefillReservation {
    fn clone(&self) -> Self {
        Self {
            chooser: self.chooser.clone(),
            scheduler_id: self.scheduler_id.clone(),
            created_at: self.created_at,
        }
    }
}

enum PrefillReservationEntry {
    /// Cancelling this token drops the selection future. Release/1.3.0 keeps a
    /// queued entry until its next dequeue, where the closed response receiver
    /// prevents scheduler booking.
    Pending { cancellation: CancellationToken },
    Active(ActivePrefillReservation),
    /// Preserve a cancellation that reaches Rust before the blocking reserve
    /// call creates the pending entry.
    Cancelled { created_at: Instant },
}

enum BeginReservation {
    Pending(CancellationToken),
    Cancelled,
    AlreadyExists,
}

pub(super) struct PrefillReservationRegistry {
    entries: Mutex<HashMap<String, PrefillReservationEntry>>,
}

impl Default for PrefillReservationRegistry {
    fn default() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }
}

impl PrefillReservationRegistry {
    fn begin(&self, reservation_id: &str) -> BeginReservation {
        let mut entries = self.entries.lock();
        match entries.entry(reservation_id.to_string()) {
            Entry::Vacant(entry) => {
                let cancellation = CancellationToken::new();
                entry.insert(PrefillReservationEntry::Pending {
                    cancellation: cancellation.clone(),
                });
                BeginReservation::Pending(cancellation)
            }
            Entry::Occupied(entry) => match entry.get() {
                PrefillReservationEntry::Cancelled { .. } => {
                    entry.remove();
                    BeginReservation::Cancelled
                }
                PrefillReservationEntry::Pending { .. } | PrefillReservationEntry::Active(_) => {
                    BeginReservation::AlreadyExists
                }
            },
        }
    }

    fn activate(&self, reservation_id: &str, reservation: ActivePrefillReservation) -> bool {
        let mut entries = self.entries.lock();
        let Some(entry) = entries.get(reservation_id) else {
            return false;
        };

        let PrefillReservationEntry::Pending { cancellation } = entry else {
            return false;
        };
        if cancellation.is_cancelled() {
            entries.remove(reservation_id);
            return false;
        }

        entries.insert(
            reservation_id.to_string(),
            PrefillReservationEntry::Active(reservation),
        );
        true
    }

    fn remove_pending(&self, reservation_id: &str) {
        let mut entries = self.entries.lock();
        if matches!(
            entries.get(reservation_id),
            Some(PrefillReservationEntry::Pending { .. })
        ) {
            entries.remove(reservation_id);
        }
    }

    fn cancel(&self, reservation_id: &str) {
        let mut entries = self.entries.lock();
        match entries.entry(reservation_id.to_string()) {
            Entry::Occupied(entry) => {
                if let PrefillReservationEntry::Pending { cancellation } = entry.get() {
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

    fn expired_active_ids(&self, now: Instant, retention: Duration) -> Vec<String> {
        self.entries
            .lock()
            .iter()
            .filter(|(_, entry)| {
                matches!(entry, PrefillReservationEntry::Active(active)
                    if now.saturating_duration_since(active.created_at) >= retention)
            })
            .map(|(reservation_id, _)| reservation_id.clone())
            .collect()
    }

    fn remove_expired_cancellations(&self, now: Instant, retention: Duration) {
        self.entries.lock().retain(|_, entry| {
            !matches!(entry, PrefillReservationEntry::Cancelled { created_at }
                if now.saturating_duration_since(*created_at) >= retention)
        });
    }
}

fn reservation_retention() -> Duration {
    let configured = std::env::var(ACTIVE_REQUEST_EXPIRY_ENV)
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .map(Duration::from_secs)
        .unwrap_or(DEFAULT_ACTIVE_REQUEST_EXPIRY_DURATION);
    configured
        .max(DEFAULT_ACTIVE_REQUEST_EXPIRY_DURATION)
        .saturating_add(RESERVATION_EXPIRY_GRACE)
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

impl PrefillRouter {
    /// Atomically select and reserve a prefill worker for an externally dispatched request.
    ///
    /// The scheduler observes the booking before selecting the next request. The caller owns
    /// the returned reservation until first output, terminal completion, or cancellation.
    #[expect(clippy::too_many_arguments)]
    pub async fn reserve_prefill_worker(
        &self,
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
        if self.lifecycle_state() != PrefillLifecycleState::Active {
            return Err(anyhow::anyhow!(PrefillError::NotActivated));
        }
        let inner = self
            .prefill_router
            .get()
            .ok_or_else(|| anyhow::anyhow!(PrefillError::NotActivated))?;
        let InnerPrefillRouter::KvRouter(router) = inner else {
            return Err(anyhow::anyhow!(PrefillError::NotActivated));
        };
        let chooser = router.chooser.clone();
        let scheduler_id = scheduler_id(reservation_id);
        let cancellation = match self.reservations.begin(reservation_id) {
            BeginReservation::Pending(cancellation) => cancellation,
            BeginReservation::Cancelled => {
                anyhow::bail!("prefill reservation {reservation_id:?} was cancelled")
            }
            BeginReservation::AlreadyExists => {
                anyhow::bail!("prefill reservation {reservation_id:?} already exists")
            }
        };

        let outcome = tokio::select! {
            biased;
            _ = cancellation.cancelled() => {
                self.reservations.remove_pending(reservation_id);
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
                self.reservations.remove_pending(reservation_id);
                return Err(error);
            }
        };

        match outcome {
            crate::kv_router::FindBestMatchOutcome::Routed { worker, .. } => {
                let reservation = ActivePrefillReservation {
                    chooser: chooser.clone(),
                    scheduler_id: scheduler_id.clone(),
                    created_at: Instant::now(),
                };
                if !self.reservations.activate(reservation_id, reservation) {
                    ignore_missing_request(chooser.free(&scheduler_id).await)?;
                    anyhow::bail!("prefill reservation {reservation_id:?} was cancelled");
                }
                Ok(PrefillQueryOutcome::Routed {
                    worker_id: worker.worker_id,
                    dp_rank: Some(worker.dp_rank),
                })
            }
            crate::kv_router::FindBestMatchOutcome::QueueRejected { rejection } => {
                self.reservations.remove_pending(reservation_id);
                Ok(PrefillQueryOutcome::QueueRejected { rejection })
            }
        }
    }

    /// Cancel a pending prefill reservation without waiting for scheduler cleanup.
    ///
    /// An active reservation remains owned by the normal booking lifecycle. If cancellation
    /// races with admission, the Go caller drains the reserve result and releases that booking.
    pub fn cancel_prefill_reservation(&self, reservation_id: &str) {
        if !reservation_id.is_empty() {
            self.reservations.cancel(reservation_id);
        }
    }

    /// Release a prefill reservation. Missing reservations are idempotent no-ops.
    pub async fn release_prefill_reservation(&self, reservation_id: &str) -> Result<()> {
        let Some(reservation) = self.reservations.get_active(reservation_id) else {
            return Ok(());
        };

        ignore_missing_request(reservation.chooser.free(&reservation.scheduler_id).await)?;
        self.reservations
            .remove_if_scheduler_id(reservation_id, &reservation.scheduler_id);
