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
    selector::WorkerSelector,
    sequences::active_request_expiry_duration,
};
use parking_lot::Mutex;
use tokio::time::{Instant, MissedTickBehavior};
use tokio_util::sync::CancellationToken;

use super::{
    InnerPrefillRouter, PrefillError, PrefillLifecycleState, PrefillQueryOutcome, PrefillRouter,
};
use crate::{
    kv_router::{KvRouter, sequence::SequenceError},
    local_model::runtime_config::ModelRuntimeConfig,
};

const PREFILL_SCHEDULER_ID_PREFIX: &str = "epp-prefill/";
const RESERVATION_EXPIRY_GRACE: Duration = Duration::from_secs(60);
const CANCELLED_RESERVATION_RETENTION: Duration = Duration::from_secs(60);
const RESERVATION_REAPER_INTERVAL: Duration = Duration::from_secs(30);
const RESERVATION_RELEASE_TIMEOUT: Duration = Duration::from_secs(5);

struct ActivePrefillReservation<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    chooser: Arc<KvRouter<Sel>>,
    scheduler_id: String,
    worker: WorkerWithDpRank,
    created_at: Instant,
}

impl<Sel> Clone for ActivePrefillReservation<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    fn clone(&self) -> Self {
        Self {
            chooser: self.chooser.clone(),
            scheduler_id: self.scheduler_id.clone(),
            worker: self.worker,
            created_at: self.created_at,
        }
    }
}

enum PrefillReservationEntry<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    /// A scheduler admission is in flight. Cancelling this token drops the
    /// lifecycle-tracked scheduler future and retracts any queued admission.
    Pending {
        cancellation: CancellationToken,
        created_at: Instant,
        claimed: bool,
    },
    Active(ActivePrefillReservation<Sel>),
    /// A cancellation can arrive from Go before the blocking reserve call
    /// reaches Rust. Preserve that cancellation briefly so the later reserve
    /// observes it instead of queueing.
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

/// EPP-owned prefill reservation state.
///
/// This lives alongside the C router handle rather than in [`PrefillRouter`].
/// It receives a router reference only while taking an admission snapshot, and
/// retains the resulting chooser in active entries so later cleanup is safe
/// across router rebinds.
pub struct EppReservationManager<Sel = dynamo_kv_router::selector::DefaultWorkerSelector>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    entries: Mutex<HashMap<String, PrefillReservationEntry<Sel>>>,
}

impl<Sel> Default for EppReservationManager<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    fn default() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }
}

impl<Sel> EppReservationManager<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    /// Construct an empty manager. Call [`Self::spawn_reaper`] after placing
    /// the manager in an [`Arc`] owned by the external router handle.
    pub fn new() -> Self {
        Self::default()
    }

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

    fn activate(
        &self,
        reservation_id: &str,
        reservation: ActivePrefillReservation<Sel>,
    ) -> Activation {
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

    fn get_active(&self, reservation_id: &str) -> Option<ActivePrefillReservation<Sel>> {
        match self.entries.lock().get(reservation_id) {
            Some(PrefillReservationEntry::Active(reservation)) => Some(reservation.clone()),
            Some(PrefillReservationEntry::Pending { .. })
            | Some(PrefillReservationEntry::Cancelled { .. })
            | None => None,
        }
    }

    fn remove_if_scheduler_id(&self, reservation_id: &str, scheduler_id: &str) {
        let mut entries = self.entries.lock();
        if entries
            .get(reservation_id)
            .is_some_and(|entry| {
                matches!(entry, PrefillReservationEntry::Active(active) if active.scheduler_id == scheduler_id)
            })
        {
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

fn reservation_retention_from_expiry(active_request_expiry: Duration) -> Duration {
    active_request_expiry.saturating_add(RESERVATION_EXPIRY_GRACE)
}

fn reservation_retention() -> Duration {
    reservation_retention_from_expiry(active_request_expiry_duration())
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

impl<Sel> EppReservationManager<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    /// Register the pending state before potentially slow request preprocessing.
    ///
    /// Calling this repeatedly for the same in-flight booking is idempotent. A
    /// cancellation that arrived first is observed here and never queues work.
    pub fn begin(&self, router: &PrefillRouter<Sel>, reservation_id: &str) -> Result<()> {
        if reservation_id.is_empty() {
            anyhow::bail!("prefill reservation ID must not be empty");
        }
        if router.lifecycle_state() != PrefillLifecycleState::Active {
            return Err(anyhow::anyhow!(PrefillError::NotActivated));
        }
        if router.binding.load_full().is_none() {
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
        router: &PrefillRouter<Sel>,
        reservation_id: &str,
        token_ids: &[u32],
        block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
        lora_name: Option<String>,
        cache_namespace: Option<String>,
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
        let Some(binding) = router.binding.load_full() else {
            self.remove_pending(reservation_id);
            return Err(anyhow::anyhow!(PrefillError::NotActivated));
        };

        let InnerPrefillRouter::KvRouter(router) = &binding.router else {
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
            outcome = chooser.find_best_match_details_with_lifecycle(
                Some(&scheduler_id),
                token_ids,
                block_mm_infos,
                None,
                true,
                false,
                lora_name,
                cache_namespace,
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
                    worker,
                    created_at: Instant::now(),
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
                        if let Err(error) = ignore_missing_request(
                            chooser.free_if_worker(&scheduler_id, worker).await,
                        ) {
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

    /// Cancel a pending prefill reservation without waiting for router cleanup.
    ///
    /// An active reservation remains owned by the normal booking lifecycle. If cancellation
    /// races with admission, the Go caller drains the reserve result and releases that booking.
    pub fn cancel(&self, reservation_id: &str) {
        if !reservation_id.is_empty() {
            self.cancel_entry(reservation_id);
        }
    }

    /// Release a prefill reservation. Missing reservations are idempotent no-ops.
    ///
    /// The registry entry remains present until scheduler cleanup succeeds, so a timeout or
    /// transient router error can be retried safely.
    pub async fn release(&self, reservation_id: &str) -> Result<()> {
        let Some(reservation) = self.get_active(reservation_id) else {
            return Ok(());
        };

        ignore_missing_request(
            reservation
                .chooser
                .free_if_worker(&reservation.scheduler_id, reservation.worker)
                .await,
        )?;
        self.remove_if_scheduler_id(reservation_id, &reservation.scheduler_id);
        Ok(())
    }

    /// Reap stale EPP-owned bookings. The task only weakly holds the manager,
    /// so destroying the C router handle stops it without affecting the
    /// prefill router's lifecycle.
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
                let expired = manager.expired_active_ids(now, retention);
                for reservation_id in expired {
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
            }
        });
    }
}
#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        sync::{Arc, atomic::Ordering},
    };

    use dynamo_kv_router::{config::KvRouterConfig, selector::DefaultWorkerSelector};
    use dynamo_runtime::{
        DistributedRuntime, Runtime,
        distributed::DistributedConfig,
        pipeline::{PushRouter, RouterMode},
        protocols::annotated::Annotated,
    };
    use tokio::sync::watch;

    use super::*;
    use crate::{
        discovery::ModelManager,
        kv_router::{KvPushRouter, prefill_router::PrefillBinding},
        protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest},
        worker_type::WorkerType,
    };

    async fn tracked_binding(
        label: &str,
    ) -> (
        Arc<PrefillBinding<DefaultWorkerSelector>>,
        Arc<KvRouter<DefaultWorkerSelector>>,
    ) {
        let runtime = Runtime::from_current().unwrap();
        let distributed = DistributedRuntime::new(runtime, DistributedConfig::process_local())
            .await
            .unwrap();
        let component = distributed
            .namespace(format!("epp-prefill-{label}-{}", uuid::Uuid::new_v4()))
            .unwrap()
            .component("workers".to_string())
            .unwrap();
        let endpoint = component.endpoint("generate");
        let endpoint_id = endpoint.id();
        let client = endpoint.client().await.unwrap();
        let (_workers_tx, workers_rx) = watch::channel(HashMap::from([
            (7, ModelRuntimeConfig::default()),
            (8, ModelRuntimeConfig::default()),
        ]));
        let config = KvRouterConfig {
            overlap_score_credit: 0.0,
            router_temperature: 0.0,
            use_kv_events: false,
            router_track_active_blocks: false,
            router_track_prefill_tokens: true,
            skip_initial_worker_wait: true,
            ..Default::default()
        };
        let chooser = Arc::new(
            KvRouter::new_with_worker_role(
                endpoint,
                client.clone(),
                workers_rx,
                None,
                16,
                DefaultWorkerSelector::new(Some(config.clone()), "prefill"),
                Some(config),
                None,
                Some(WorkerType::Prefill),
                "prefill",
                Some(format!("epp-prefill-{label}")),
                false,
                None,
                None,
            )
            .await
            .unwrap(),
        );
        let push_router =
            PushRouter::<PreprocessedRequest, Annotated<LLMEngineOutput>>::from_client(
                client,
                RouterMode::KV,
            )
            .await
            .unwrap();
        let binding = Arc::new(PrefillBinding {
            endpoint_id,
            router: InnerPrefillRouter::KvRouter(Arc::new(KvPushRouter::new_with_coordinator(
                push_router,
                chooser.clone(),
                None,
            ))),
        });
        (binding, chooser)
    }

    async fn tracked_prefill_router() -> (
        Arc<PrefillRouter<DefaultWorkerSelector>>,
        Arc<KvRouter<DefaultWorkerSelector>>,
    ) {
        let (binding, chooser) = tracked_binding("primary").await;
        let router = PrefillRouter::disabled(Arc::new(ModelManager::new()), RouterMode::KV, None);
        router.binding.store(Some(binding));
        router
            .lifecycle
            .store(PrefillLifecycleState::Active as u8, Ordering::Release);
        (router, chooser)
    }

    fn routed_worker(outcome: PrefillQueryOutcome) -> WorkerWithDpRank {
        match outcome {
            PrefillQueryOutcome::Routed {
                worker_id,
                dp_rank: Some(dp_rank),
            } => WorkerWithDpRank::new(worker_id, dp_rank),
            _ => panic!("expected routed prefill worker with a DP rank"),
        }
    }

    async fn reserve(
        manager: &EppReservationManager<DefaultWorkerSelector>,
        router: &PrefillRouter<DefaultWorkerSelector>,
        id: &str,
    ) -> WorkerWithDpRank {
        routed_worker(
            manager
                .reserve(
                    router,
                    id,
                    &[1u32; 64],
                    None,
                    None,
                    None,
                    0.0,
                    0,
                    None,
                    RoutingConstraints::default(),
                )
                .await
                .unwrap(),
        )
    }

    #[tokio::test]
    async fn reservations_affect_selection_and_release_load() {
        let (router, chooser) = tracked_prefill_router().await;
        let manager = EppReservationManager::default();

        let first = reserve(&manager, &router, "reservation-1").await;
        let second = reserve(&manager, &router, "reservation-2").await;

        assert_ne!(
            first, second,
            "the second request should select the idle worker"
        );
        let loads = chooser
            .get_potential_loads(&[], None, None, None, None)
            .await
            .unwrap();
        assert_eq!(loads.len(), 2);
        assert!(loads.iter().all(|load| load.potential_prefill_tokens == 64));

        manager.release("reservation-1").await.unwrap();
        manager.release("reservation-2").await.unwrap();
        let loads = chooser
            .get_potential_loads(&[], None, None, None, None)
            .await
            .unwrap();
        assert!(loads.iter().all(|load| load.potential_prefill_tokens == 0));
    }

    #[tokio::test]
    async fn release_uses_the_binding_that_created_the_reservation() {
        let (router, original_chooser) = tracked_prefill_router().await;
        let manager = EppReservationManager::default();
        reserve(&manager, &router, "reservation-before-rebind").await;

        let (replacement, _) = tracked_binding("replacement").await;
        router.binding.store(Some(replacement));
        manager.release("reservation-before-rebind").await.unwrap();

        let loads = original_chooser
            .get_potential_loads(&[], None, None, None, None)
            .await
            .unwrap();
        assert!(loads.iter().all(|load| load.potential_prefill_tokens == 0));
    }

    #[test]
    fn scheduler_ids_are_role_scoped() {
        assert_eq!(scheduler_id("booking-id"), "epp-prefill/booking-id");
    }

    #[test]
    fn cancellation_before_reservation_is_observed() {
        let registry = EppReservationManager::<DefaultWorkerSelector>::default();
        registry.cancel("reservation-1");

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
    fn reservation_retention_preserves_short_and_long_expiry_overrides() {
        assert_eq!(
            reservation_retention_from_expiry(Duration::from_secs(30)),
            Duration::from_secs(90)
        );
        assert_eq!(
            reservation_retention_from_expiry(Duration::from_secs(3600)),
            Duration::from_secs(3660)
        );
    }

    #[test]
    fn pre_registered_pending_reservation_survives_tombstone_reaping() {
        let registry = EppReservationManager::<DefaultWorkerSelector>::default();
        let BeginReservation::Pending(cancellation) = registry.begin_entry("reservation-1") else {
            panic!("expected pending reservation");
        };

        registry.cancel("reservation-1");
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
        let registry = EppReservationManager::<DefaultWorkerSelector>::default();
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
        let registry = EppReservationManager::<DefaultWorkerSelector>::default();
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

    #[tokio::test]
    async fn cancelled_activation_retains_active_reservation_for_retry() {
        let (_, chooser) = tracked_prefill_router().await;
        let registry = EppReservationManager::default();
        let BeginReservation::Pending(cancellation) = registry.claim_pending("reservation-1")
        else {
            panic!("expected claimed pending reservation");
        };
        cancellation.cancel();

        let activation = registry.activate(
            "reservation-1",
            ActivePrefillReservation {
                chooser,
                scheduler_id: scheduler_id("reservation-1"),
                worker: WorkerWithDpRank::new(7, 0),
                created_at: Instant::now(),
            },
        );
        assert!(matches!(activation, Activation::Cancelled));
        assert!(registry.get_active("reservation-1").is_some());
    }

    #[test]
    fn cancellation_signals_pending_reservation() {
        let registry = EppReservationManager::<DefaultWorkerSelector>::default();
        let BeginReservation::Pending(cancellation) = registry.begin_entry("reservation-1") else {
            panic!("expected pending reservation");
        };

        registry.cancel("reservation-1");
        assert!(cancellation.is_cancelled());
        registry.remove_pending("reservation-1");
    }
}
