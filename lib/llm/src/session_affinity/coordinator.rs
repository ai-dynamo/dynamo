// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    pin::Pin,
    sync::{
        Arc, Weak,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    task::{Context, Poll},
    time::Duration,
};

use dashmap::{DashMap, mapref::entry::Entry};
use dynamo_runtime::{
    engine::{AsyncEngineContext, AsyncEngineContextProvider},
    error::{DynamoError, ErrorType},
    pipeline::{Error, ManyOut, ResponseStream},
};
use futures::Stream;
use tokio::{sync::Notify, time::Instant};
use tokio_util::sync::CancellationToken;

use super::{
    LlmResponse, MAX_SESSION_AFFINITY_ENTRIES, MAX_SESSION_AFFINITY_ID_BYTES,
    MAX_SESSION_AFFINITY_TTL_SECS,
};
use crate::{
    preprocessor::PreprocessedRequest,
    protocols::common::{
        extensions::{SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId},
        timing::RequestPhase,
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    },
}

struct AffinityCoordinatorInner {
    entries: DashMap<String, AffinityEntry>,
    ttl: Duration,
    max_entries: usize,
    max_session_id_bytes: usize,
    entry_count: AtomicUsize,
    next_revision: AtomicU64,
    cancel: CancellationToken,
    #[cfg(test)]
    reaper_started: Arc<Notify>,
    #[cfg(test)]
    waiter_observed: Arc<Notify>,
}

impl Drop for AffinityCoordinatorInner {
    fn drop(&mut self) {
        self.cancel.cancel();
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
            next_revision: AtomicU64::new(1),
            cancel: CancellationToken::new(),
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
                    tracing::info!(
                        session_id = %session_id,
                        "session affinity miss: new session, pinning after worker selection"
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
                    } if *active_leases == 0 && *idle_deadline <= now => {
                        tracing::info!(
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
                            active: true,
                        }));
                    }
                    AffinityEntry::Bound {
                        target,
                        revision,
                        active_leases,
                        ..
                    } => {
                        validate_bound_target(&session_id, *target, requested_target)?;
                        tracing::info!(
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
        let AffinityEntry::Bound {
            target,
            active_leases,
            idle_deadline,
            ..
        } = entry.value()
        else {
            return Ok(None);
        };
        if *active_leases == 0 && *idle_deadline <= Instant::now() {
            return Ok(None);
        }
        validate_bound_target(session_id.as_str(), *target, requested_target)?;
        tracing::info!(
            session_id = %session_id.as_str(),
            worker_id = target.worker_id,
            dp_rank = ?target.dp_rank,
            active_leases = *active_leases + 1,
            "session affinity hit: reusing pinned worker"
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
            .entry_count
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| {
                (count < self.inner.max_entries).then_some(count + 1)
            })
            .map(|_| ())
            .map_err(|_| resource_exhausted("session affinity entry limit reached"))
    }
}

trait VacantEntryExt {
    fn insert_initializing(
        self,
        inner: &Arc<AffinityCoordinatorInner>,
        session_id: String,
        requested_target: Option<AffinityTarget>,
    ) -> AffinityInitialization;
}

impl<'a> VacantEntryExt for dashmap::mapref::entry::VacantEntry<'a, String, AffinityEntry> {
    fn insert_initializing(
        self,
        inner: &Arc<AffinityCoordinatorInner>,
        session_id: String,
        requested_target: Option<AffinityTarget>,
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
            active: true,
        }
    }
}

pub(crate) enum AffinityAcquire {
    Initialize(AffinityInitialization),
    Bound {
        target: AffinityTarget,
        lease: AffinityLease,
    },
}

impl AffinityAcquire {
    pub(crate) fn target(&self) -> Option<AffinityTarget> {
        match self {
            Self::Initialize(_) => None,
            Self::Bound { target, .. } => Some(*target),
        }
    }

    pub(crate) fn into_stream(
        self,
        selected_target: AffinityTarget,
        stream: ManyOut<LlmResponse>,
    ) -> Result<ManyOut<LlmResponse>, Error> {
        match self {
            Self::Initialize(initialization) => {
                Ok(initialization.commit(selected_target)?.into_stream(stream))
            }
            Self::Bound { target, mut lease } => {
                if let Err(error) = validate_bound_target("session", target, Some(selected_target))
                {
                    lease.invalidate();
                    return Err(error);
                }
                Ok(lease.into_stream(stream))
            }
        }
    }

    pub(crate) fn invalidate(self) {
        if let Self::Bound { mut lease, .. } = self {
            lease.invalidate();
        }
    }
}

pub(crate) struct AffinityInitialization {
    coordinator: Weak<AffinityCoordinatorInner>,
    session_id: String,
    revision: u64,
    notify: Arc<Notify>,
    requested_target: Option<AffinityTarget>,
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
        };
        drop(entry);
        self.active = false;
        self.notify.notify_waiters();
        Ok(AffinityLease {
            coordinator: Arc::downgrade(&inner),
            session_id: self.session_id.clone(),
            revision: self.revision,
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

pub(crate) struct AffinityLease {
    coordinator: Weak<AffinityCoordinatorInner>,
    session_id: String,
    revision: u64,
    active: bool,
}

impl AffinityLease {
    pub(crate) fn into_stream(self, stream: ManyOut<LlmResponse>) -> ManyOut<LlmResponse> {
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
        let Some(mut entry) = inner.entries.get_mut(&self.session_id) else {
            return;
        };
        let AffinityEntry::Bound {
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

struct AffinityTrackedStream {
    stream: ManyOut<LlmResponse>,
    lease: Option<AffinityLease>,
}

impl Stream for AffinityTrackedStream {
    type Item = LlmResponse;

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
