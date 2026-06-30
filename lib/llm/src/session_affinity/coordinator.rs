// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Session affinity uses a process-local cache in front of immutable shared claims.
//! Cache hits exact-route without distributed I/O. On a cache miss, the discovery
//! backend reads the claim first and evaluates the query-only routing proposal only
//! when no claim exists. The payload returned by claim arbitration is authoritative:
//! distributed mode caches only `Created` or `Existing` payloads, and racing losers
//! discard their proposal, cache the winner, and dispatch to it. Explicit worker and
//! rank headers are proposals only while no binding exists.
//!
//! Shared claim deletion invalidates local caches eventually through `Delete` and
//! `Reset` events. The configured affinity TTL evicts only local cache entries; it
//! does not expire or replace a shared claim. Bindings are never rebound in v1. If a
//! bound worker disappears, exact dispatch fails without fallback and the caller must
//! use a new session ID. Explicit close requires no concurrent active requests, is
//! terminal, and the closed session ID must not be reused.

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
    discovery::{ClaimEvent, ClaimOutcome, ClaimPayloadFuture, Discovery},
    engine::{AsyncEngineContext, AsyncEngineContextProvider},
    error::{DynamoError, ErrorType},
    pipeline::{Error, ManyOut, ResponseStream},
};
use futures::Stream;
use serde::{Deserialize, Serialize};
use tokio::{
    sync::{Notify, broadcast},
    time::Instant,
};
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

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
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
    claims: ClaimCoordination,
    ttl: Duration,
    max_entries: usize,
    max_session_id_bytes: usize,
    entry_count: AtomicUsize,
    next_revision: AtomicU64,
    cancel: CancellationToken,
    #[cfg(test)]
    probe: AffinityCoordinatorProbe,
}

#[derive(Clone)]
struct ClaimCoordination {
    scope: String,
    discovery: Option<Arc<dyn Discovery>>,
}

impl ClaimCoordination {
    fn key(&self, session_id: &SessionAffinityId) -> String {
        format!(
            "{}/{}",
            self.scope,
            blake3::hash(session_id.as_str().as_bytes()).to_hex()
        )
    }

    fn subscribe(&self) -> Option<broadcast::Receiver<ClaimEvent>> {
        self.discovery
            .as_ref()
            .and_then(|discovery| discovery.subscribe_claim_events())
    }

    async fn resolve(
        &self,
        key: &str,
        proposed_payload: &mut ClaimPayloadFuture<'_>,
    ) -> Result<(serde_json::Value, bool), Error> {
        let Some(discovery) = self.discovery.as_ref() else {
            return Ok((proposed_payload.as_mut().await?, true));
        };

        match discovery.create_or_get_claim(key, proposed_payload).await? {
            ClaimOutcome::Created(payload) => Ok((payload, true)),
            ClaimOutcome::Existing(payload) => Ok((payload, false)),
            ClaimOutcome::Unsupported => Ok((proposed_payload.as_mut().await?, true)),
        }
    }

    async fn close(&self, key: &str) -> anyhow::Result<()> {
        let Some(discovery) = self.discovery.as_ref() else {
            return Ok(());
        };
        discovery.close_claim(key).await?;
        Ok(())
    }
}

#[cfg(test)]
struct AffinityCoordinatorProbe {
    reaper_started: Arc<Notify>,
    waiter_observed: Arc<Notify>,
}

#[cfg(test)]
impl AffinityCoordinatorProbe {
    fn new() -> Self {
        Self {
            reaper_started: Arc::new(Notify::new()),
            waiter_observed: Arc::new(Notify::new()),
        }
    }
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
            "local".to_string(),
            None,
        )
    }

    pub(crate) fn new_distributed(
        ttl: Duration,
        claim_scope: String,
        discovery: Arc<dyn Discovery>,
    ) -> Result<Self, Error> {
        Self::new_with_limits(
            ttl,
            MAX_SESSION_AFFINITY_ENTRIES,
            MAX_SESSION_AFFINITY_ID_BYTES,
            claim_scope,
            Some(discovery),
        )
    }

    fn new_with_limits(
        ttl: Duration,
        max_entries: usize,
        max_session_id_bytes: usize,
        claim_scope: String,
        discovery: Option<Arc<dyn Discovery>>,
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
            claims: ClaimCoordination {
                scope: claim_scope,
                discovery,
            },
            ttl,
            max_entries,
            max_session_id_bytes,
            entry_count: AtomicUsize::new(0),
            next_revision: AtomicU64::new(1),
            cancel: CancellationToken::new(),
            #[cfg(test)]
            probe: AffinityCoordinatorProbe::new(),
        });
        Self::spawn_reaper(&inner);
        Self::spawn_claim_listener(&inner);
        Ok(Self { inner })
    }

    fn spawn_claim_listener(inner: &Arc<AffinityCoordinatorInner>) {
        let Some(mut events) = inner.claims.subscribe() else {
            return;
        };
        let weak = Arc::downgrade(inner);
        let cancel = inner.cancel.clone();

        tokio::spawn(async move {
            loop {
                let event = tokio::select! {
                    _ = cancel.cancelled() => return,
                    event = events.recv() => event,
                };
                let Some(inner) = weak.upgrade() else {
                    return;
                };
                match event {
                    Ok(ClaimEvent::Delete(key)) => Self::evict_key(&inner, &key),
                    Ok(ClaimEvent::Reset) | Err(broadcast::error::RecvError::Lagged(_)) => {
                        Self::clear_entries(&inner);
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        Self::clear_entries(&inner);
                        return;
                    }
                }
            }
        });
    }

    fn evict_key(inner: &AffinityCoordinatorInner, key: &str) {
        let Some((_, entry)) = inner.entries.remove(key) else {
            return;
        };
        if let AffinityEntry::Initializing { notify, .. } = entry {
            notify.notify_waiters();
        }
        Self::decrement_entry_count(inner, 1);
        tracing::debug!(claim_key = key, "evicted session affinity cache entry");
    }

    fn clear_entries(inner: &AffinityCoordinatorInner) {
        for entry in inner.entries.iter() {
            if let AffinityEntry::Initializing { notify, .. } = entry.value() {
                notify.notify_waiters();
            }
        }
        inner.entries.clear();
        inner.entry_count.store(0, Ordering::Relaxed);
        tracing::debug!("cleared session affinity cache after claim watcher reset");
    }

    fn decrement_entry_count(inner: &AffinityCoordinatorInner, removed: usize) {
        let _ = inner
            .entry_count
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| {
                Some(count.saturating_sub(removed))
            });
    }

    fn spawn_reaper(inner: &Arc<AffinityCoordinatorInner>) {
        let weak = Arc::downgrade(inner);
        let cancel = inner.cancel.clone();
        let period = inner.ttl.min(Duration::from_secs(30));
        #[cfg(test)]
        let reaper_started = inner.probe.reaper_started.clone();
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
                Self::decrement_entry_count(&inner, removed);
            }
        });
    }

    #[cfg(test)]
    pub(crate) async fn acquire(
        &self,
        session_id: &SessionAffinityId,
    ) -> Result<AffinityAcquire, Error> {
        self.acquire_inner(session_id, None).await
    }

    pub(crate) async fn acquire_with_context(
        &self,
        session_id: &SessionAffinityId,
        request_context: &dyn AsyncEngineContext,
    ) -> Result<AffinityAcquire, Error> {
        self.acquire_inner(session_id, Some(request_context)).await
    }

    async fn acquire_inner(
        &self,
        session_id: &SessionAffinityId,
        request_context: Option<&dyn AsyncEngineContext>,
    ) -> Result<AffinityAcquire, Error> {
        self.validate_session_id(session_id)?;
        let claim_key = self.inner.claims.key(session_id);

        loop {
            let now = Instant::now();
            match self.inner.entries.entry(claim_key.clone()) {
                Entry::Vacant(entry) => {
                    self.reserve_entry()?;
                    return Ok(AffinityAcquire::Initialize(
                        entry.insert_initializing(&self.inner, claim_key),
                    ));
                }
                Entry::Occupied(mut entry) => match entry.get_mut() {
                    AffinityEntry::Initializing { notify, .. } => {
                        #[cfg(test)]
                        self.inner.probe.waiter_observed.notify_one();
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
                        let revision = self.inner.next_revision.fetch_add(1, Ordering::Relaxed);
                        let notify = Arc::new(Notify::new());
                        *entry.get_mut() = AffinityEntry::Initializing {
                            revision,
                            notify: notify.clone(),
                        };
                        drop(entry);
                        return Ok(AffinityAcquire::Initialize(AffinityInitialization {
                            coordinator: Arc::downgrade(&self.inner),
                            claim_key,
                            revision,
                            notify,
                            active: true,
                        }));
                    }
                    AffinityEntry::Bound {
                        target,
                        revision,
                        active_leases,
                        ..
                    } => {
                        *active_leases += 1;
                        let lease = AffinityLease {
                            coordinator: Arc::downgrade(&self.inner),
                            claim_key,
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
    ) -> Result<Option<AffinityTarget>, Error> {
        self.validate_session_id(session_id)?;
        let claim_key = self.inner.claims.key(session_id);
        let Some(entry) = self.inner.entries.get(&claim_key) else {
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
        Ok(Some(*target))
    }

    #[cfg(test)]
    pub(super) fn entry_count(&self) -> usize {
        self.inner.entry_count.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(super) fn claim_key_for_test(&self, session_id: &SessionAffinityId) -> String {
        self.inner.claims.key(session_id)
    }

    #[cfg(test)]
    pub(super) fn cancellation_token(&self) -> CancellationToken {
        self.inner.cancel.clone()
    }

    #[cfg(test)]
    pub(super) async fn wait_for_reaper(&self) {
        self.inner.probe.reaper_started.notified().await;
    }

    #[cfg(test)]
    pub(super) async fn wait_for_initializing_waiter(&self) {
        self.inner.probe.waiter_observed.notified().await;
    }

    #[cfg(test)]
    pub(super) fn expire_for_test(&self, session_id: &SessionAffinityId) {
        let claim_key = self.inner.claims.key(session_id);
        let Some(mut entry) = self.inner.entries.get_mut(&claim_key) else {
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
        Self::new_with_limits(
            Duration::from_secs(10),
            max_entries,
            max_session_id_bytes,
            "local".to_string(),
            None,
        )
        .unwrap()
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
        claim_key: String,
    ) -> AffinityInitialization;
}

impl<'a> VacantEntryExt for dashmap::mapref::entry::VacantEntry<'a, String, AffinityEntry> {
    fn insert_initializing(
        self,
        inner: &Arc<AffinityCoordinatorInner>,
        claim_key: String,
    ) -> AffinityInitialization {
        let revision = inner.next_revision.fetch_add(1, Ordering::Relaxed);
        let notify = Arc::new(Notify::new());
        self.insert(AffinityEntry::Initializing {
            revision,
            notify: notify.clone(),
        });
        AffinityInitialization {
            coordinator: Arc::downgrade(inner),
            claim_key,
            revision,
            notify,
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
    pub(crate) async fn resolve(
        self,
        proposed_payload: ClaimPayloadFuture<'_>,
    ) -> Result<ResolvedAffinity, Error> {
        match self {
            Self::Initialize(initialization) => initialization.resolve(proposed_payload).await,
            Self::Bound { target, lease } => Ok(ResolvedAffinity {
                target,
                lease,
                created: false,
            }),
        }
    }
}

pub(crate) struct AffinityInitialization {
    coordinator: Weak<AffinityCoordinatorInner>,
    claim_key: String,
    revision: u64,
    notify: Arc<Notify>,
    active: bool,
}

impl AffinityInitialization {
    async fn resolve(
        self,
        mut proposed_payload: ClaimPayloadFuture<'_>,
    ) -> Result<ResolvedAffinity, Error> {
        let Some(inner) = self.coordinator.upgrade() else {
            return Err(anyhow::anyhow!("session affinity coordinator dropped"));
        };

        let (payload, created) = inner
            .claims
            .resolve(&self.claim_key, &mut proposed_payload)
            .await?;
        let target: AffinityTarget = serde_json::from_value(payload)
            .map_err(|err| anyhow::anyhow!("invalid session affinity claim payload: {err}"))?;
        let lease = self.commit(target)?;
        Ok(ResolvedAffinity {
            target,
            lease,
            created,
        })
    }

    fn commit(mut self, target: AffinityTarget) -> Result<AffinityLease, Error> {
        let Some(inner) = self.coordinator.upgrade() else {
            return Err(anyhow::anyhow!("session affinity coordinator dropped"));
        };
        let Some(mut entry) = inner.entries.get_mut(&self.claim_key) else {
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
            claim_key: self.claim_key.clone(),
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
        let removed = inner.entries.remove_if(&self.claim_key, |_, entry| {
            matches!(
                entry,
                AffinityEntry::Initializing { revision, .. } if *revision == self.revision
            )
        });
        if removed.is_some() {
            AffinityCoordinator::decrement_entry_count(&inner, 1);
        }
        self.notify.notify_waiters();
    }
}

pub(crate) struct AffinityLease {
    coordinator: Weak<AffinityCoordinatorInner>,
    claim_key: String,
    revision: u64,
    active: bool,
}

impl AffinityLease {
    fn release(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;
        let Some(inner) = self.coordinator.upgrade() else {
            return;
        };
        let Some(mut entry) = inner.entries.get_mut(&self.claim_key) else {
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
}

impl Drop for AffinityLease {
    fn drop(&mut self) {
        self.release();
    }
}

pub(crate) struct ResolvedAffinity {
    target: AffinityTarget,
    lease: AffinityLease,
    created: bool,
}

impl ResolvedAffinity {
    pub(crate) fn target(&self) -> AffinityTarget {
        self.target
    }

    pub(crate) fn was_created(&self) -> bool {
        self.created
    }

    pub(crate) fn into_stream(
        self,
        stream: ManyOut<LlmResponse>,
        close_on_finish: bool,
    ) -> ManyOut<LlmResponse> {
        let context = stream.context();
        let close = close_on_finish.then(|| CloseAction {
            coordinator: self.lease.coordinator.clone(),
            claims: self
                .lease
                .coordinator
                .upgrade()
                .map(|inner| inner.claims.clone()),
            claim_key: self.lease.claim_key.clone(),
        });
        ResponseStream::new(
            Box::pin(AffinityTrackedStream {
                stream,
                lease: Some(self.lease),
                close,
            }),
            context,
        )
    }
}

struct CloseAction {
    coordinator: Weak<AffinityCoordinatorInner>,
    claims: Option<ClaimCoordination>,
    claim_key: String,
}

impl CloseAction {
    fn run(self) {
        if let Some(inner) = self.coordinator.upgrade() {
            AffinityCoordinator::evict_key(&inner, &self.claim_key);
        }
        let Some(claims) = self.claims else {
            return;
        };
        let claim_key = self.claim_key;
        tokio::spawn(async move {
            if let Err(error) = claims.close(&claim_key).await {
                tracing::error!(%claim_key, %error, "failed to close session affinity claim");
            }
        });
    }
}

struct AffinityTrackedStream {
    stream: ManyOut<LlmResponse>,
    lease: Option<AffinityLease>,
    close: Option<CloseAction>,
}

impl AffinityTrackedStream {
    fn finish(&mut self) {
        drop(self.lease.take());
        if let Some(close) = self.close.take() {
            close.run();
        }
    }
}

impl Stream for AffinityTrackedStream {
    type Item = LlmResponse;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.stream).poll_next(cx) {
            Poll::Ready(None) => {
                self.finish();
                Poll::Ready(None)
            }
            Poll::Ready(Some(item)) => Poll::Ready(Some(item)),
            poll => poll,
        }
    }
}

impl Drop for AffinityTrackedStream {
    fn drop(&mut self) {
        self.finish();
    }
}

pub fn affinity_id(
    request: &dynamo_runtime::pipeline::SingleIn<PreprocessedRequest>,
) -> Result<Option<Arc<SessionAffinityId>>, Error> {
    request
        .get_optional::<SessionAffinityId>(SESSION_AFFINITY_CONTEXT_KEY)
        .map_err(|message| invalid_argument(format!("invalid session affinity context: {message}")))
}

pub fn session_final(request: &PreprocessedRequest) -> bool {
    request
        .agent_context
        .as_ref()
        .is_some_and(|context| context.session_final == Some(true))
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
