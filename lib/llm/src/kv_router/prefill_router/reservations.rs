// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, hash_map::Entry},
    sync::{Arc, Weak},
    time::Duration,
};

use anyhow::Result;
use dynamo_kv_router::{
    protocols::{BlockExtraInfo, RoutingConstraints, WorkerId, WorkerWithDpRank},
    sequence::DEFAULT_ACTIVE_REQUEST_EXPIRY_DURATION,
};
use parking_lot::Mutex;
use tokio::time::{Instant, MissedTickBehavior};

use super::{
    InnerPrefillRouter, PrefillError, PrefillLifecycleState, PrefillQueryOutcome, PrefillRouter,
};
use crate::kv_router::{KvRouter, sequence::SequenceError};

const PREFILL_SCHEDULER_ID_PREFIX: &str = "epp-prefill/";
const ACTIVE_REQUEST_EXPIRY_ENV: &str = "DYN_ROUTER_ACTIVE_REQUEST_EXPIRY_SECS";
const RESERVATION_EXPIRY_GRACE: Duration = Duration::from_secs(60);
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

pub(super) struct PrefillReservationRegistry {
    entries: Mutex<HashMap<String, ActivePrefillReservation>>,
}

impl Default for PrefillReservationRegistry {
    fn default() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }
}

impl PrefillReservationRegistry {
    fn insert(&self, reservation_id: &str, reservation: ActivePrefillReservation) -> bool {
        match self.entries.lock().entry(reservation_id.to_string()) {
            Entry::Vacant(entry) => {
                entry.insert(reservation);
                true
            }
            Entry::Occupied(_) => false,
        }
    }

    fn get(&self, reservation_id: &str) -> Option<ActivePrefillReservation> {
        self.entries.lock().get(reservation_id).cloned()
    }

    fn remove_if_scheduler_id(&self, reservation_id: &str, scheduler_id: &str) {
        let mut entries = self.entries.lock();
        if entries
            .get(reservation_id)
            .is_some_and(|entry| entry.scheduler_id == scheduler_id)
        {
            entries.remove(reservation_id);
        }
    }

    fn expired_ids(&self, now: Instant, retention: Duration) -> Vec<String> {
        self.entries
            .lock()
            .iter()
            .filter(|(_, entry)| now.saturating_duration_since(entry.created_at) >= retention)
            .map(|(reservation_id, _)| reservation_id.clone())
            .collect()
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
        let outcome = chooser
            .find_best_match_details(
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
            )
            .await?;

        match outcome {
            crate::kv_router::FindBestMatchOutcome::Routed { worker, .. } => {
                let reservation = ActivePrefillReservation {
                    chooser: chooser.clone(),
                    scheduler_id: scheduler_id.clone(),
                    created_at: Instant::now(),
                };
                if !self.reservations.insert(reservation_id, reservation) {
                    ignore_missing_request(chooser.free(&scheduler_id).await)?;
                    anyhow::bail!("prefill reservation {reservation_id:?} already exists");
                }
                Ok(PrefillQueryOutcome::Routed {
                    worker_id: worker.worker_id,
                    dp_rank: Some(worker.dp_rank),
                })
            }
            crate::kv_router::FindBestMatchOutcome::QueueRejected { rejection } => {
                Ok(PrefillQueryOutcome::QueueRejected { rejection })
            }
        }
    }

    /// Release a prefill reservation. Missing reservations are idempotent no-ops.
    ///
    /// The registry entry remains present until scheduler cleanup succeeds, so a timeout or
    /// transient router error can be retried safely.
    pub async fn release_prefill_reservation(&self, reservation_id: &str) -> Result<()> {
        let Some(reservation) = self.reservations.get(reservation_id) else {
            return Ok(());
        };

        ignore_missing_request(reservation.chooser.free(&reservation.scheduler_id).await)?;
        self.reservations
            .remove_if_scheduler_id(reservation_id, &reservation.scheduler_id);
        Ok(())
    }

    pub(super) fn spawn_reservation_reaper(router: &Arc<Self>) {
        let router: Weak<Self> = Arc::downgrade(router);
        let cancellation = router
            .upgrade()
            .expect("router must be alive while starting reservation reaper")
            .cancel_token
            .child_token();

        tokio::spawn(async move {
            let retention = reservation_retention();
            let mut interval = tokio::time::interval_at(
                Instant::now() + RESERVATION_REAPER_INTERVAL,
                RESERVATION_REAPER_INTERVAL,
            );
            interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

            loop {
                tokio::select! {
                    _ = cancellation.cancelled() => return,
                    _ = interval.tick() => {
                        let Some(router) = router.upgrade() else {
                            return;
                        };
                        let expired = router.reservations.expired_ids(Instant::now(), retention);
                        for reservation_id in expired {
                            if let Err(error) = router.release_prefill_reservation(&reservation_id).await {
                                tracing::warn!(
                                    %reservation_id,
                                    %error,
                                    "Failed to expire stale EPP prefill reservation"
