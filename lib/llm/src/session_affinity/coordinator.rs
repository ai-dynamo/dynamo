// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    sync::{
        Arc, OnceLock,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::Duration,
};

use dashmap::{DashMap, mapref::entry::Entry};
use dynamo_runtime::{
    engine::AsyncEngineContext,
    error::{DynamoError, ErrorType},
    pipeline::Error,
};
use tokio::{sync::Notify, time::Instant};
use tokio_util::sync::CancellationToken;

#[cfg(test)]
use super::replica_sync::SessionAffinityUpdate;
use super::{
    AffinityTarget, MAX_SESSION_AFFINITY_ENTRIES, MAX_SESSION_AFFINITY_ID_BYTES,
    MAX_SESSION_AFFINITY_TTL_SECS,
    lifecycle::{AffinityAcquire, AffinityInitialization, AffinityLease, VacantEntryExt},
    replica_sync::ReplicaSyncRuntime,
    state::{
        AffinityRevision, ReplicaApplyOutcome, ReplicaBinding, apply_replica_binding,
        revision_timestamp,
    },
};
use crate::{
    preprocessor::PreprocessedRequest,
    protocols::common::{
        extensions::{SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId},
        timing::RequestPhase,
    },
};

pub(super) enum AffinityEntry {
    Initializing {
        revision: AffinityRevision,
        generation: u64,
        notify: Arc<Notify>,
        pending_replica: Option<ReplicaBinding>,
    },
    Bound {
        target: AffinityTarget,
        revision: AffinityRevision,
        generation: u64,
        active_leases: usize,
        idle_deadline: Instant,
    },
}

pub(super) struct AffinityCoordinatorInner {
    pub(super) entries: DashMap<String, AffinityEntry>,
    pub(super) ttl: Duration,
    max_entries: usize,
    max_session_id_bytes: usize,
    pub(super) entry_count: AtomicUsize,
    next_revision: AtomicU64,
    next_generation: AtomicU64,
    router_id: AtomicU64,
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

#[derive(Clone)]
pub struct AffinityCoordinator {
    inner: Arc<AffinityCoordinatorInner>,
}

impl AffinityCoordinator {
    pub fn new(ttl: Duration) -> Result<Self, Error> {
        Self::new_with_limits(
            ttl,
            MAX_SESSION_AFFINITY_ENTRIES,
            MAX_SESSION_AFFINITY_ID_BYTES,
        )
    }

    fn new_with_limits(
        ttl: Duration,
        max_entries: usize,
        max_session_id_bytes: usize,
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
            next_revision: AtomicU64::new(revision_timestamp()),
            next_generation: AtomicU64::new(1),
            router_id: AtomicU64::new(0),
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
            .router_id
            .store(replica.router_id(), Ordering::Relaxed);
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
                    tracing::debug!(
                        session_id = %session_id,
                        "session affinity miss: binding after worker selection"
                    );
                    return Ok(AffinityAcquire::Initialize(entry.insert_initializing(
                        &self.inner,
                        session_id,
                        requested_target,
                    )));
                }
                Entry::Occupied(mut entry) => match entry.get_mut() {
                    AffinityEntry::Initializing { notify, .. } => {
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
                            "session affinity miss: binding expired (idle past TTL), re-selecting worker"
                        );
                        let revision = self.inner.next_revision();
                        let generation = self.inner.next_generation();
                        let notify = Arc::new(Notify::new());
                        *entry.get_mut() = AffinityEntry::Initializing {
                            revision,
                            generation,
                            notify: notify.clone(),
                            pending_replica: None,
                        };
                        drop(entry);
                        return Ok(AffinityAcquire::Initialize(AffinityInitialization {
                            coordinator: Arc::downgrade(&self.inner),
                            session_id,
                            revision,
                            generation,
                            notify,
                            requested_target,
                            active: true,
                        }));
                    }
                    AffinityEntry::Bound {
                        target,
                        revision,
                        generation,
                        active_leases,
                        ..
                    } => {
                        tracing::debug!(
                            session_id = %session_id,
                            worker_id = target.worker_id,
                            dp_rank = ?target.dp_rank,
                            active_leases = *active_leases + 1,
                            "session affinity hit: using preferred worker"
                        );
                        *active_leases += 1;
                        let lease = AffinityLease {
                            coordinator: Arc::downgrade(&self.inner),
                            session_id,
                            revision: *revision,
                            generation: *generation,
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
        _requested_target: Option<AffinityTarget>,
    ) -> Result<Option<AffinityTarget>, Error> {
        self.validate_session_id(session_id)?;
        let now = Instant::now();
        let Some(entry) = self.inner.entries.get(session_id.as_str()) else {
            return Ok(None);
        };
        let AffinityEntry::Bound {
            target,
            active_leases,
            idle_deadline,
            ..
        } = entry.value()
        else {
            return Ok(None);
        };
        if *active_leases == 0 && *idle_deadline <= now {
            return Ok(None);
        }
        tracing::debug!(
            session_id = %session_id.as_str(),
            worker_id = target.worker_id,
            dp_rank = ?target.dp_rank,
            "session affinity hit: using preferred worker"
        );

        Ok(Some(*target))
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
        self.inner.router_id.store(router_id, Ordering::Relaxed);
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
        self.apply_replica_revision_for_test(session_id, target, 1, 99)
    }

    #[cfg(test)]
    pub(super) fn apply_replica_revision_for_test(
        &self,
        session_id: impl Into<String>,
        target: AffinityTarget,
        sequence: u64,
        router_id: u64,
    ) -> ReplicaApplyOutcome {
        self.inner.apply_replica_update(
            session_id.into(),
            target,
            AffinityRevision {
                sequence,
                router_id,
            },
        )
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
    pub(super) fn next_revision(&self) -> AffinityRevision {
        let mut previous = self.next_revision.load(Ordering::Relaxed);
        let sequence = loop {
            let next = previous.saturating_add(1);
            match self.next_revision.compare_exchange_weak(
                previous,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break next,
                Err(observed) => previous = observed,
            }
        };
        AffinityRevision {
            sequence,
            router_id: self.router_id.load(Ordering::Relaxed),
        }
    }

    fn observe_revision(&self, revision: AffinityRevision) {
        self.next_revision
            .fetch_max(revision.sequence, Ordering::Relaxed);
    }

    pub(super) fn next_generation(&self) -> u64 {
        self.next_generation.fetch_add(1, Ordering::Relaxed)
    }

    fn reserve_entry(&self) -> bool {
        self.entry_count
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| {
                (count < self.max_entries).then_some(count + 1)
            })
            .is_ok()
    }

    pub(super) fn publish_replica_update(
        &self,
        session_id: &str,
        target: AffinityTarget,
        revision: AffinityRevision,
    ) {
        if let Some(replica) = self.replica.get() {
            replica.publish(session_id, target, revision);
        }
    }

    pub(super) fn apply_replica_update(
        &self,
        session_id: String,
        target: AffinityTarget,
        revision: AffinityRevision,
    ) -> ReplicaApplyOutcome {
        if session_id.len() > self.max_session_id_bytes {
            return ReplicaApplyOutcome::RejectedSessionId;
        }

        self.observe_revision(revision);
        let now = Instant::now();
        match self.entries.entry(session_id) {
            Entry::Vacant(entry) => {
                if !self.reserve_entry() {
                    return ReplicaApplyOutcome::RejectedCapacity;
                }
                entry.insert(AffinityEntry::Bound {
                    target,
                    revision,
                    generation: self.next_generation(),
                    active_leases: 0,
                    idle_deadline: now + self.ttl,
                });
                ReplicaApplyOutcome::Inserted
            }
            Entry::Occupied(mut entry) => match entry.get_mut() {
                AffinityEntry::Initializing {
                    pending_replica, ..
                } => {
                    let outcome = match pending_replica {
                        Some(binding) => apply_replica_binding(binding, target, revision),
                        None => {
                            *pending_replica = Some(ReplicaBinding { target, revision });
                            ReplicaApplyOutcome::Inserted
                        }
                    };
                    match outcome {
                        ReplicaApplyOutcome::IgnoredStale => outcome,
                        _ => ReplicaApplyOutcome::DeferredInitializing,
                    }
                }
                AffinityEntry::Bound {
                    target: existing_target,
                    revision: existing_revision,
                    active_leases,
                    idle_deadline,
                    ..
                } => {
                    let mut binding = ReplicaBinding {
                        target: *existing_target,
                        revision: *existing_revision,
                    };
                    let outcome = apply_replica_binding(&mut binding, target, revision);
                    if outcome != ReplicaApplyOutcome::IgnoredStale {
                        *existing_target = binding.target;
                        *existing_revision = binding.revision;
                        if *active_leases == 0 {
                            *idle_deadline = now + self.ttl;
                        }
                    }
                    outcome
                }
            },
        }
    }
}

pub fn affinity_id(
    request: &dynamo_runtime::pipeline::SingleIn<PreprocessedRequest>,
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

pub(super) fn validate_bound_target(
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
