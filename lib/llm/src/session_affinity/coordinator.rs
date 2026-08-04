// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::HashSet,
    pin::Pin,
    sync::{
        Arc, OnceLock, Weak,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    task::{Context, Poll},
    time::Duration,
};

use dashmap::{DashMap, mapref::entry::Entry};
use dynamo_runtime::{
    engine::{AsyncEngineContext, AsyncEngineContextProvider},
    error::{DynamoError, ErrorType},
    pipeline::{Data, Error, ManyOut, ResponseStream},
};
use futures::Stream;
use tokio::{sync::Notify, time::Instant};
use tokio_util::sync::CancellationToken;

#[cfg(test)]
use super::replica_sync::SessionAffinityUpdate;
use super::{
    MAX_SESSION_AFFINITY_ENTRIES, MAX_SESSION_AFFINITY_ID_BYTES, MAX_SESSION_AFFINITY_TTL_SECS,
    ScaleUpMigrationTracker, ScaleUpSnapshot, replica_sync::ReplicaSyncRuntime,
};
use crate::{
    preprocessor::PreprocessedRequest,
    protocols::common::{
        extensions::{SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId},
        timing::RequestPhase,
    },
};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct AffinityTarget {
    pub worker_id: u64,
    pub dp_rank: Option<u32>,
}

enum AffinityEntry {
    Initializing {
        revision: u64,
        notify: Arc<Notify>,
    },
    Bound {
        target: AffinityTarget,
        revision: u64,
        active_leases: usize,
        idle_deadline: Instant,
        scale_snapshot: Option<Arc<ScaleUpSnapshot>>,
        migration_generation: u64,
    },
    Migrating {
        old_target: AffinityTarget,
        revision: u64,
        notify: Arc<Notify>,
    },
}

pub(super) struct AffinityCoordinatorInner {
    entries: DashMap<String, AffinityEntry>,
    ttl: Duration,
    max_entries: usize,
    max_session_id_bytes: usize,
    entry_count: AtomicUsize,
    next_revision: AtomicU64,
    scale_up: Option<ScaleUpMigrationTracker>,
    cancel: CancellationToken,
    replica: OnceLock<ReplicaSyncRuntime>,
    #[cfg(test)]
    reaper_started: Arc<Notify>,
    #[cfg(test)]
    waiter_observed: Arc<Notify>,
}

impl Drop for AffinityCoordinatorInner {
    fn drop(&mut self) {
        self.cancel.cancel();
        if let Some(replica) = self.replica.get_mut() {
            replica.shutdown_now();
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ReplicaApplyOutcome {
    Inserted,
    Refreshed,
    ReplacedExpired,
    ReboundMigration,
    IgnoredInitializing,
    IgnoredConflict,
    RejectedSessionId,
    RejectedCapacity,
}

#[derive(Clone)]
pub struct AffinityCoordinator {
    inner: Arc<AffinityCoordinatorInner>,
}

impl AffinityCoordinator {
    pub fn new(ttl: Duration) -> Result<Self, Error> {
        Self::new_with_options(
            ttl,
            MAX_SESSION_AFFINITY_ENTRIES,
            MAX_SESSION_AFFINITY_ID_BYTES,
            None,
        )
    }

    pub(crate) fn new_with_scale_up(
        ttl: Duration,
        scale_up: ScaleUpMigrationTracker,
    ) -> Result<Self, Error> {
        Self::new_with_options(
            ttl,
            MAX_SESSION_AFFINITY_ENTRIES,
            MAX_SESSION_AFFINITY_ID_BYTES,
            Some(scale_up),
        )
    }

    #[cfg(test)]
    fn new_with_limits(
        ttl: Duration,
        max_entries: usize,
        max_session_id_bytes: usize,
    ) -> Result<Self, Error> {
        Self::new_with_options(ttl, max_entries, max_session_id_bytes, None)
    }

    fn new_with_options(
        ttl: Duration,
        max_entries: usize,
        max_session_id_bytes: usize,
        scale_up: Option<ScaleUpMigrationTracker>,
    ) -> Result<Self, Error> {
        if !(Duration::from_secs(1)..=Duration::from_secs(MAX_SESSION_AFFINITY_TTL_SECS))
            .contains(&ttl)
        {
            return Err(invalid_argument(format!(
                "session affinity TTL must be between 1 and {MAX_SESSION_AFFINITY_TTL_SECS} seconds"
            )));
        }
        let inner = Arc::new(AffinityCoordinatorInner {
            entries: DashMap::new(),
            ttl,
            max_entries,
            max_session_id_bytes,
            entry_count: AtomicUsize::new(0),
            next_revision: AtomicU64::new(1),
            scale_up,
            cancel: CancellationToken::new(),
            replica: OnceLock::new(),
            #[cfg(test)]
            reaper_started: Arc::new(Notify::new()),
            #[cfg(test)]
            waiter_observed: Arc::new(Notify::new()),
        });
        Self::spawn_reaper(&inner);
        tracing::info!(
            ttl_secs = ttl.as_secs(),
            max_entries,
            "session affinity enabled"
        );
        Ok(Self { inner })
    }

    fn spawn_reaper(inner: &Arc<AffinityCoordinatorInner>) {
        let weak = Arc::downgrade(inner);
        let cancel = inner.cancel.clone();
        let period = inner.ttl.min(Duration::from_secs(30));
        #[cfg(test)]
        let reaper_started = inner.reaper_started.clone();
        tokio::spawn(async move {
            #[cfg(test)]
            reaper_started.notify_one();
            loop {
                tokio::select! {
                    _ = cancel.cancelled() => return,
                    _ = tokio::time::sleep(period) => {}
                }
                let Some(inner) = weak.upgrade() else {
                    return;
                };
                let now = Instant::now();
                let mut removed = 0;
                inner.entries.retain(|_, entry| {
                    let retain = !matches!(
                        entry,
                        AffinityEntry::Bound {
                            active_leases: 0,
                            idle_deadline,
                            ..
                        } if *idle_deadline <= now
                    );
                    removed += usize::from(!retain);
                    retain
                });
                inner.entry_count.fetch_sub(removed, Ordering::Relaxed);
            }
        });
    }

    pub(crate) async fn enable_replica_sync(
        &self,
        client: dynamo_runtime::component::Client,
    ) -> Result<(), Error> {
        let replica =
            ReplicaSyncRuntime::start(client, Arc::downgrade(&self.inner), &self.inner.cancel)
                .await?;
        self.inner
            .replica
            .set(replica)
            .map_err(|_| anyhow::anyhow!("session affinity replica sync already enabled"))
    }

    #[cfg(test)]
    pub(crate) async fn acquire(
        &self,
        session_id: &SessionAffinityId,
        requested_target: Option<AffinityTarget>,
    ) -> Result<AffinityAcquire, Error> {
        self.acquire_inner(session_id, requested_target, None).await
    }

    pub(crate) async fn acquire_with_context(
        &self,
        session_id: &SessionAffinityId,
        requested_target: Option<AffinityTarget>,
        request_context: &dyn AsyncEngineContext,
    ) -> Result<AffinityAcquire, Error> {
        self.acquire_inner(session_id, requested_target, Some(request_context))
            .await
    }

    async fn acquire_inner(
        &self,
        session_id: &SessionAffinityId,
        requested_target: Option<AffinityTarget>,
        request_context: Option<&dyn AsyncEngineContext>,
    ) -> Result<AffinityAcquire, Error> {
        self.validate_session_id(session_id)?;
        let session_id = session_id.as_str().to_string();

        loop {
            let now = Instant::now();
            match self.inner.entries.entry(session_id.clone()) {
                Entry::Vacant(entry) => {
                    self.reserve_entry()?;
                    let scale_snapshot = self
                        .inner
                        .scale_up
                        .as_ref()
                        .map(ScaleUpMigrationTracker::snapshot);
                    tracing::debug!(
                        session_id = %session_id,
                        "session affinity miss: new session, pinning after worker selection"
                    );
                    return Ok(AffinityAcquire::Initialize(entry.insert_initializing(
                        &self.inner,
                        session_id,
                        requested_target,
                        scale_snapshot,
                    )));
                }
                Entry::Occupied(mut entry) => match entry.get_mut() {
                    AffinityEntry::Initializing { notify, .. }
                    | AffinityEntry::Migrating { notify, .. } => {
                        #[cfg(test)]
                        self.inner.waiter_observed.notify_one();
                        let notified = notify.clone().notified_owned();
                        tokio::pin!(notified);
                        notified.as_mut().enable();
                        drop(entry);
                        if let Some(context) = request_context {
                            tokio::select! {
                                biased;
                                _ = context.stopped() => {
                                    return Err(cancelled(context.id()));
                                }
                                _ = context.killed() => {
                                    return Err(cancelled(context.id()));
                                }
                                _ = notified => {}
                            }
                        } else {
                            notified.await;
                        }
                    }
                    AffinityEntry::Bound {
                        target: _,
                        revision,
                        active_leases,
                        idle_deadline,
                        ..
                    } if *active_leases == 0 && *idle_deadline <= now => {
                        tracing::debug!(
                            session_id = %session_id,
                            "session affinity miss: pin expired (idle past TTL), re-selecting worker"
                        );
                        let revision = self.inner.next_revision.fetch_add(1, Ordering::Relaxed);
                        let notify = Arc::new(Notify::new());
                        *entry.get_mut() = AffinityEntry::Initializing {
                            revision,
                            notify: notify.clone(),
                        };
                        drop(entry);
                        return Ok(AffinityAcquire::Initialize(AffinityInitialization {
                            coordinator: Arc::downgrade(&self.inner),
                            session_id,
                            revision,
                            notify,
                            requested_target,
                            scale_snapshot: self
                                .inner
                                .scale_up
                                .as_ref()
                                .map(ScaleUpMigrationTracker::snapshot),
                            active: true,
                        }));
                    }
                    AffinityEntry::Bound {
                        target,
                        revision,
                        active_leases,
                        scale_snapshot,
                        migration_generation,
                        ..
                    } => {
                        if requested_target.is_none()
                            && *active_leases == 0
                            && let (Some(scale_up), Some(previous_snapshot)) =
                                (&self.inner.scale_up, scale_snapshot.as_ref())
                        {
                            let evaluation = scale_up.evaluate(
                                &SessionAffinityId::new(session_id.clone()),
                                previous_snapshot,
                            );
                            if let Some(migration_workers) = evaluation.migration_workers {
                                let old_target = *target;
                                let previous_snapshot = previous_snapshot.clone();
                                let next_snapshot = evaluation.snapshot;
                                let old_migration_generation = *migration_generation;
                                let revision =
                                    self.inner.next_revision.fetch_add(1, Ordering::Relaxed);
                                let notify = Arc::new(Notify::new());
                                *entry.get_mut() = AffinityEntry::Migrating {
                                    old_target,
                                    revision,
                                    notify: notify.clone(),
                                };
                                drop(entry);
                                return Ok(AffinityAcquire::Migrate(AffinityMigration {
                                    coordinator: Arc::downgrade(&self.inner),
                                    session_id,
                                    revision,
                                    notify,
                                    old_target,
                                    old_migration_generation,
                                    previous_snapshot,
                                    next_snapshot,
                                    migration_workers,
                                    active: true,
                                }));
                            }
                            *scale_snapshot = Some(evaluation.snapshot);
                        }
                        validate_bound_target(&session_id, *target, requested_target)?;
                        tracing::debug!(
                            session_id = %session_id,
                            worker_id = target.worker_id,
                            dp_rank = ?target.dp_rank,
                            active_leases = *active_leases + 1,
                            "session affinity hit: reusing pinned worker"
                        );
                        *active_leases += 1;
                        let lease = AffinityLease {
                            coordinator: Arc::downgrade(&self.inner),
                            session_id,
                            revision: *revision,
                            migration_generation: (*migration_generation > 0)
                                .then_some(*migration_generation),
                            active: true,
                        };
                        return Ok(AffinityAcquire::Bound {
                            target: *target,
                            lease,
                        });
                    }
                },
            }
        }
    }

    pub fn query_target(
        &self,
        session_id: &SessionAffinityId,
        requested_target: Option<AffinityTarget>,
    ) -> Result<Option<AffinityTarget>, Error> {
        self.validate_session_id(session_id)?;
        let Some(entry) = self.inner.entries.get(session_id.as_str()) else {
            return Ok(None);
        };
        let target = match entry.value() {
            AffinityEntry::Bound {
                target,
                active_leases,
                idle_deadline,
                ..
            } => {
                if *active_leases == 0 && *idle_deadline <= Instant::now() {
                    return Ok(None);
                }
                *target
            }
            // Query-only requests do not initiate migration. While another
            // request moves the session, a concurrent query keeps the old pin.
            AffinityEntry::Migrating { old_target, .. } => *old_target,
            AffinityEntry::Initializing { .. } => return Ok(None),
        };
        validate_bound_target(session_id.as_str(), target, requested_target)?;
        tracing::debug!(
            session_id = %session_id.as_str(),
            worker_id = target.worker_id,
            dp_rank = ?target.dp_rank,
            "session affinity hit: reusing pinned worker"
        );

        Ok(Some(target))
    }

    #[cfg(test)]
    pub(super) fn entry_count(&self) -> usize {
        self.inner.entry_count.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(super) fn cancellation_token(&self) -> CancellationToken {
        self.inner.cancel.clone()
    }

    #[cfg(test)]
    pub(super) async fn wait_for_reaper(&self) {
        self.inner.reaper_started.notified().await;
    }

    #[cfg(test)]
    pub(super) async fn wait_for_initializing_waiter(&self) {
        self.inner.waiter_observed.notified().await;
    }

    #[cfg(test)]
    pub(super) fn expire_for_test(&self, session_id: &SessionAffinityId) {
        let Some(mut entry) = self.inner.entries.get_mut(session_id.as_str()) else {
            panic!("session affinity entry missing");
        };
        let AffinityEntry::Bound {
            active_leases,
            idle_deadline,
            ..
        } = entry.value_mut()
        else {
            panic!("session affinity entry is not bound");
        };
        assert_eq!(*active_leases, 0);
        *idle_deadline = Instant::now();
    }

    #[cfg(test)]
    pub(super) fn with_test_limits(max_entries: usize, max_session_id_bytes: usize) -> Self {
        Self::new_with_limits(Duration::from_secs(10), max_entries, max_session_id_bytes).unwrap()
    }

    #[cfg(test)]
    pub(super) fn enable_test_replica(
        &self,
        router_id: u64,
        capacity: usize,
    ) -> tokio::sync::mpsc::Receiver<SessionAffinityUpdate> {
        let (replica, rx) = ReplicaSyncRuntime::for_test(router_id, capacity);
        self.inner
            .replica
            .set(replica)
            .unwrap_or_else(|_| panic!("session affinity test replica already enabled"));
        rx
    }

    #[cfg(test)]
    pub(super) fn apply_replica_update_for_test(
        &self,
        session_id: impl Into<String>,
        target: AffinityTarget,
    ) -> ReplicaApplyOutcome {
        self.inner
            .apply_replica_update(session_id.into(), target, None)
    }

    #[cfg(test)]
    pub(super) fn apply_replica_migration_for_test(
        &self,
        session_id: impl Into<String>,
        target: AffinityTarget,
        generation: u64,
    ) -> ReplicaApplyOutcome {
        self.inner
            .apply_replica_update(session_id.into(), target, Some(generation))
    }

    fn validate_session_id(&self, session_id: &SessionAffinityId) -> Result<(), Error> {
        if session_id.as_str().len() > self.inner.max_session_id_bytes {
            return Err(invalid_argument(format!(
                "session affinity ID must not exceed {} bytes",
                self.inner.max_session_id_bytes
            )));
        }
        Ok(())
    }

    fn reserve_entry(&self) -> Result<(), Error> {
        self.inner
            .reserve_entry()
            .then_some(())
            .ok_or_else(|| resource_exhausted("session affinity entry limit reached"))
    }
}

impl AffinityCoordinatorInner {
    fn reserve_entry(&self) -> bool {
        self.entry_count
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| {
                (count < self.max_entries).then_some(count + 1)
            })
            .is_ok()
    }

    fn publish_replica_update(
        &self,
        session_id: &str,
        target: AffinityTarget,
        migration_generation: Option<u64>,
    ) {
        if let Some(replica) = self.replica.get() {
            replica.publish(session_id, target, migration_generation);
        }
    }

    pub(super) fn apply_replica_update(
        &self,
        session_id: String,
        target: AffinityTarget,
        migration_generation: Option<u64>,
    ) -> ReplicaApplyOutcome {
        if session_id.len() > self.max_session_id_bytes {
            return ReplicaApplyOutcome::RejectedSessionId;
        }

        let now = Instant::now();
        let incoming_generation = migration_generation.unwrap_or(0);
        let scale_snapshot = self
            .scale_up
            .as_ref()
            .map(ScaleUpMigrationTracker::snapshot);
        match self.entries.entry(session_id) {
            Entry::Vacant(entry) => {
                if !self.reserve_entry() {
                    return ReplicaApplyOutcome::RejectedCapacity;
                }
                let revision = self.next_revision.fetch_add(1, Ordering::Relaxed);
                entry.insert(AffinityEntry::Bound {
                    target,
                    revision,
                    active_leases: 0,
                    idle_deadline: now + self.ttl,
                    scale_snapshot,
                    migration_generation: incoming_generation,
                });
                ReplicaApplyOutcome::Inserted
            }
            Entry::Occupied(mut entry) => match entry.get_mut() {
                AffinityEntry::Initializing { .. } | AffinityEntry::Migrating { .. } => {
                    ReplicaApplyOutcome::IgnoredInitializing
                }
                AffinityEntry::Bound {
                    active_leases,
                    idle_deadline,
                    ..
                } if *active_leases == 0 && *idle_deadline <= now => {
                    let revision = self.next_revision.fetch_add(1, Ordering::Relaxed);
                    *entry.get_mut() = AffinityEntry::Bound {
                        target,
                        revision,
                        active_leases: 0,
                        idle_deadline: now + self.ttl,
                        scale_snapshot,
                        migration_generation: incoming_generation,
                    };
                    ReplicaApplyOutcome::ReplacedExpired
                }
                AffinityEntry::Bound {
                    target: existing,
                    idle_deadline,
                    scale_snapshot: existing_snapshot,
                    migration_generation: existing_generation,
                    ..
                } if *existing == target => {
                    *idle_deadline = now + self.ttl;
                    if incoming_generation > *existing_generation {
                        *existing_generation = incoming_generation;
                        *existing_snapshot = scale_snapshot;
                    }
                    ReplicaApplyOutcome::Refreshed
                }
                AffinityEntry::Bound {
                    migration_generation: existing_generation,
                    ..
                } if migration_generation.is_some()
                    && incoming_generation > *existing_generation =>
                {
                    let revision = self.next_revision.fetch_add(1, Ordering::Relaxed);
                    *entry.get_mut() = AffinityEntry::Bound {
                        target,
                        revision,
                        active_leases: 0,
                        idle_deadline: now + self.ttl,
                        scale_snapshot,
                        migration_generation: incoming_generation,
                    };
                    ReplicaApplyOutcome::ReboundMigration
                }
                AffinityEntry::Bound { .. } => ReplicaApplyOutcome::IgnoredConflict,
            },
        }
    }
}

trait VacantEntryExt {
    fn insert_initializing(
        self,
        inner: &Arc<AffinityCoordinatorInner>,
        session_id: String,
        requested_target: Option<AffinityTarget>,
        scale_snapshot: Option<Arc<ScaleUpSnapshot>>,
    ) -> AffinityInitialization;
}

impl<'a> VacantEntryExt for dashmap::mapref::entry::VacantEntry<'a, String, AffinityEntry> {
    fn insert_initializing(
        self,
        inner: &Arc<AffinityCoordinatorInner>,
        session_id: String,
        requested_target: Option<AffinityTarget>,
        scale_snapshot: Option<Arc<ScaleUpSnapshot>>,
    ) -> AffinityInitialization {
        let revision = inner.next_revision.fetch_add(1, Ordering::Relaxed);
        let notify = Arc::new(Notify::new());
        self.insert(AffinityEntry::Initializing {
            revision,
            notify: notify.clone(),
        });
        AffinityInitialization {
            coordinator: Arc::downgrade(inner),
            session_id,
            revision,
            notify,
            requested_target,
            scale_snapshot,
            active: true,
        }
    }
}

pub(crate) enum AffinityAcquire {
    Initialize(AffinityInitialization),
    Migrate(AffinityMigration),
    Bound {
        target: AffinityTarget,
        lease: AffinityLease,
    },
}

impl AffinityAcquire {
    pub(crate) fn action_name(&self) -> &'static str {
        match self {
            Self::Initialize(_) => "initialize",
            Self::Migrate(_) => "migrate",
            Self::Bound { .. } => "reuse",
        }
    }

    pub(crate) fn target(&self) -> Option<AffinityTarget> {
        match self {
            Self::Initialize(_) | Self::Migrate(_) => None,
            Self::Bound { target, .. } => Some(*target),
        }
    }

    /// Newly added workers eligible for this session's lazy scale-up move.
    /// The underlying token or text selector still chooses the exact rank.
    pub(crate) fn migration_worker_ids(&self) -> Option<&HashSet<u64>> {
        match self {
            Self::Migrate(migration) => Some(&migration.migration_workers),
            Self::Initialize(_) | Self::Bound { .. } => None,
        }
    }

    pub(crate) fn into_stream<U: Data>(
        self,
        selected_target: AffinityTarget,
        stream: ManyOut<U>,
    ) -> Result<ManyOut<U>, Error> {
        match self {
            Self::Initialize(initialization) => {
                let lease = initialization.commit(selected_target)?;
                lease.publish(selected_target);
                Ok(lease.into_stream(stream))
            }
            Self::Migrate(migration) => {
                let lease = migration.commit(selected_target)?;
                lease.publish(selected_target);
                Ok(lease.into_stream(stream))
            }
            Self::Bound { target, mut lease } => {
                if let Err(error) = validate_bound_target("session", target, Some(selected_target))
                {
                    lease.invalidate();
                    return Err(error);
                }
                lease.publish(target);
                Ok(lease.into_stream(stream))
            }
        }
    }

    pub(crate) fn invalidate(self) {
        match self {
            Self::Bound { mut lease, .. } => lease.invalidate(),
            Self::Initialize(_) | Self::Migrate(_) => {
                // Dropping either transactional operation performs rollback.
            }
        }
    }
}

pub(crate) struct AffinityInitialization {
    coordinator: Weak<AffinityCoordinatorInner>,
    session_id: String,
    revision: u64,
    notify: Arc<Notify>,
    requested_target: Option<AffinityTarget>,
    scale_snapshot: Option<Arc<ScaleUpSnapshot>>,
    active: bool,
}

impl AffinityInitialization {
    pub(crate) fn commit(mut self, target: AffinityTarget) -> Result<AffinityLease, Error> {
        validate_bound_target(&self.session_id, target, self.requested_target)?;
        let Some(inner) = self.coordinator.upgrade() else {
            return Err(anyhow::anyhow!("session affinity coordinator dropped"));
        };
        let Some(mut entry) = inner.entries.get_mut(&self.session_id) else {
            return Err(invalid_argument(
                "session affinity initialization was cancelled",
            ));
        };
        if !matches!(
            entry.value(),
            AffinityEntry::Initializing { revision, .. } if *revision == self.revision
        ) {
            return Err(invalid_argument("session affinity initialization changed"));
        }
        *entry = AffinityEntry::Bound {
            target,
            revision: self.revision,
            active_leases: 1,
            idle_deadline: Instant::now() + inner.ttl,
            // If topology changed while selection was in flight, bind against
            // the latest view rather than immediately migrating next turn.
            scale_snapshot: inner
                .scale_up
                .as_ref()
                .map(ScaleUpMigrationTracker::snapshot)
                .or_else(|| self.scale_snapshot.clone()),
            migration_generation: 0,
        };
        drop(entry);
        self.active = false;
        self.notify.notify_waiters();
        Ok(AffinityLease {
            coordinator: Arc::downgrade(&inner),
            session_id: self.session_id.clone(),
            revision: self.revision,
            migration_generation: None,
            active: true,
        })
    }
}

impl Drop for AffinityInitialization {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let Some(inner) = self.coordinator.upgrade() else {
            return;
        };
        let removed = inner.entries.remove_if(&self.session_id, |_, entry| {
            matches!(
                entry,
                AffinityEntry::Initializing { revision, .. } if *revision == self.revision
            )
        });
        if removed.is_some() {
            inner.entry_count.fetch_sub(1, Ordering::Relaxed);
        }
        self.notify.notify_waiters();
    }
}

pub(crate) struct AffinityMigration {
    coordinator: Weak<AffinityCoordinatorInner>,
    session_id: String,
    revision: u64,
    notify: Arc<Notify>,
    old_target: AffinityTarget,
    old_migration_generation: u64,
    previous_snapshot: Arc<ScaleUpSnapshot>,
    next_snapshot: Arc<ScaleUpSnapshot>,
    migration_workers: Arc<HashSet<u64>>,
    active: bool,
}

impl AffinityMigration {
    fn commit(mut self, target: AffinityTarget) -> Result<AffinityLease, Error> {
        if !self.migration_workers.contains(&target.worker_id) {
            return Err(invalid_argument(format!(
                "session {} scale-up migration selected worker {}, which is not newly added",
                self.session_id, target.worker_id
            )));
        }
        let Some(inner) = self.coordinator.upgrade() else {
            return Err(anyhow::anyhow!("session affinity coordinator dropped"));
        };
        let Some(mut entry) = inner.entries.get_mut(&self.session_id) else {
            return Err(invalid_argument("session affinity migration was cancelled"));
        };
        if !matches!(
            entry.value(),
            AffinityEntry::Migrating { revision, .. } if *revision == self.revision
        ) {
            return Err(invalid_argument("session affinity migration changed"));
        }
        let generation = self.next_snapshot.generation();
        *entry = AffinityEntry::Bound {
            target,
            revision: self.revision,
            active_leases: 1,
            idle_deadline: Instant::now() + inner.ttl,
            scale_snapshot: Some(self.next_snapshot.clone()),
            migration_generation: generation,
        };
        drop(entry);
        self.active = false;
        self.notify.notify_waiters();
        tracing::debug!(
            session_id = %self.session_id,
            generation,
            old_worker_id = self.old_target.worker_id,
            old_dp_rank = ?self.old_target.dp_rank,
            new_worker_id = target.worker_id,
            new_dp_rank = ?target.dp_rank,
            "Committed affinity scale-up migration"
        );
        Ok(AffinityLease {
            coordinator: Arc::downgrade(&inner),
            session_id: self.session_id.clone(),
            revision: self.revision,
            migration_generation: Some(generation),
            active: true,
        })
    }

    fn rollback(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;
        let Some(inner) = self.coordinator.upgrade() else {
            return;
        };
        let Some(mut entry) = inner.entries.get_mut(&self.session_id) else {
            return;
        };
        if !matches!(
            entry.value(),
            AffinityEntry::Migrating { revision, .. } if *revision == self.revision
        ) {
            return;
        }
        *entry = AffinityEntry::Bound {
            target: self.old_target,
            revision: self.revision,
            active_leases: 0,
            idle_deadline: Instant::now() + inner.ttl,
            scale_snapshot: Some(self.previous_snapshot.clone()),
            migration_generation: self.old_migration_generation,
        };
        drop(entry);
        self.notify.notify_waiters();
        tracing::debug!(
            session_id = %self.session_id,
            worker_id = self.old_target.worker_id,
            dp_rank = ?self.old_target.dp_rank,
            "Rolled back affinity scale-up migration"
        );
    }
}

impl Drop for AffinityMigration {
    fn drop(&mut self) {
        self.rollback();
    }
}

pub(crate) struct AffinityLease {
    coordinator: Weak<AffinityCoordinatorInner>,
    session_id: String,
    revision: u64,
    migration_generation: Option<u64>,
    active: bool,
}

impl AffinityLease {
    fn publish(&self, target: AffinityTarget) {
        if let Some(inner) = self.coordinator.upgrade() {
            inner.publish_replica_update(&self.session_id, target, self.migration_generation);
        }
    }

    pub(crate) fn into_stream<U: Data>(self, stream: ManyOut<U>) -> ManyOut<U> {
        let context = stream.context();
        ResponseStream::new(
            Box::pin(AffinityTrackedStream {
                stream,
                lease: Some(self),
            }),
            context,
        )
    }

    fn release(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;
        let Some(inner) = self.coordinator.upgrade() else {
            return;
        };
        let target = {
            let Some(mut entry) = inner.entries.get_mut(&self.session_id) else {
                return;
            };
            let AffinityEntry::Bound {
                target,
                revision,
                active_leases,
                idle_deadline,
                ..
            } = entry.value_mut()
            else {
                return;
            };
            if *revision != self.revision || *active_leases == 0 {
                return;
            }
            *active_leases -= 1;
            *idle_deadline = Instant::now() + inner.ttl;
            *target
        };
        inner.publish_replica_update(&self.session_id, target, self.migration_generation);
    }

    fn invalidate(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;
        let Some(inner) = self.coordinator.upgrade() else {
            return;
        };
        let removed = inner.entries.remove_if(&self.session_id, |_, entry| {
            matches!(
                entry,
                AffinityEntry::Bound { revision, .. } if *revision == self.revision
            )
        });
        if removed.is_some() {
            inner.entry_count.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

impl Drop for AffinityLease {
    fn drop(&mut self) {
        self.release();
    }
}

struct AffinityTrackedStream<U: Data> {
    stream: ManyOut<U>,
    lease: Option<AffinityLease>,
}

impl<U: Data> Stream for AffinityTrackedStream<U> {
    type Item = U;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.stream).poll_next(cx) {
            Poll::Ready(None) => {
                drop(self.lease.take());
                Poll::Ready(None)
            }
            Poll::Ready(Some(item)) => Poll::Ready(Some(item)),
            poll => poll,
        }
    }
}

pub fn affinity_id<T: Send + Sync + 'static>(
    request: &dynamo_runtime::pipeline::SingleIn<T>,
) -> Result<Option<Arc<SessionAffinityId>>, Error> {
    request
        .get_optional::<SessionAffinityId>(SESSION_AFFINITY_CONTEXT_KEY)
        .map_err(|message| invalid_argument(format!("invalid session affinity context: {message}")))
}

pub fn explicit_target(
    request: &PreprocessedRequest,
    phase: RequestPhase,
) -> Result<Option<AffinityTarget>, Error> {
    let Some(routing) = request.routing.as_ref() else {
        return Ok(None);
    };
    let (worker_id, dp_rank) = match phase {
        RequestPhase::Prefill => (
            routing.prefill_worker_id.or(routing.backend_instance_id),
            routing.prefill_dp_rank.or(routing.dp_rank),
        ),
        RequestPhase::Decode => (
            routing.decode_worker_id.or(routing.backend_instance_id),
            routing.dp_rank,
        ),
        RequestPhase::Aggregated => (
            routing.decode_worker_id.or(routing.backend_instance_id),
            routing.dp_rank,
        ),
    };
    if worker_id.is_none() && dp_rank.is_some() {
        return Err(invalid_argument(
            "DP rank requires an explicit worker for session affinity",
        ));
    }
    Ok(worker_id.map(|worker_id| AffinityTarget { worker_id, dp_rank }))
}

fn validate_bound_target(
    session_id: &str,
    bound: AffinityTarget,
    requested: Option<AffinityTarget>,
) -> Result<(), Error> {
    let Some(requested) = requested else {
        return Ok(());
    };
    if bound.worker_id != requested.worker_id {
        return Err(invalid_argument(format!(
            "session {session_id} is bound to worker {}, not {}",
            bound.worker_id, requested.worker_id
        )));
    }
    match (bound.dp_rank, requested.dp_rank) {
        (Some(bound), Some(requested)) if bound != requested => Err(invalid_argument(format!(
            "session {session_id} is bound to DP rank {bound}, not {requested}"
        ))),
        (None, Some(requested)) => Err(invalid_argument(format!(
            "session {session_id} has worker-only affinity and cannot add DP rank {requested}"
        ))),
        _ => Ok(()),
    }
}

pub(crate) fn invalid_argument(message: impl Into<String>) -> Error {
    DynamoError::builder()
        .error_type(ErrorType::InvalidArgument)
        .message(message.into())
        .build()
        .into()
}

fn resource_exhausted(message: impl Into<String>) -> Error {
    DynamoError::builder()
        .error_type(ErrorType::ResourceExhausted)
        .message(message.into())
        .build()
        .into()
}

fn cancelled(context_id: &str) -> Error {
    DynamoError::builder()
        .error_type(ErrorType::Cancelled)
        .message(format!(
            "request {context_id} was cancelled while waiting for session affinity"
        ))
        .build()
        .into()
}
