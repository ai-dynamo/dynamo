// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Session affinity: pin a session id to the worker (and optionally the DP
//! rank) that served it, for as long as requests keep arriving within the TTL.
//!
//! [`SessionAffinity`] is the table. A host acquires a session before
//! selecting a worker: a new session yields an [`AffinityInitialization`] the
//! host commits with the worker it dispatched to; a bound session yields the
//! target and an [`AffinityLease`] that keeps the binding alive until the
//! response stream ends. Bindings are versioned so replicas can exchange them
//! through an [`AffinityReplicaSink`]; the transport is the host's.

use std::pin::Pin;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use dashmap::{DashMap, mapref::entry::Entry};
use serde::{Deserialize, Serialize};
use tokio::sync::Notify;
use tokio::sync::futures::OwnedNotified;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

pub use crate::protocols::WorkerAffinityTarget as AffinityTarget;

pub const MAX_SESSION_AFFINITY_TTL_SECS: u64 = 31_536_000;
pub const MAX_SESSION_AFFINITY_ENTRIES: usize = 65_536;
pub const MAX_SESSION_AFFINITY_ID_BYTES: usize = 256;

/// How a bound session treats a dispatch that landed elsewhere.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionAffinityMode {
    /// The binding is exact: dispatching to another worker or rank is an error.
    #[default]
    Hard,
    /// The binding follows the dispatch: the session rebinds to where it ran.
    Soft,
}

impl std::str::FromStr for SessionAffinityMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "hard" => Ok(Self::Hard),
            "soft" => Ok(Self::Soft),
            _ => Err(format!(
                "invalid session affinity mode {value:?}; expected 'hard' or 'soft'"
            )),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AffinityError {
    #[error("{0}")]
    InvalidArgument(String),
    #[error("{0}")]
    ResourceExhausted(String),
    #[error("session affinity table dropped")]
    Dropped,
}

/// Ordering for replicated bindings: higher wins, ties broken by writer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct AffinityVersion {
    pub sequence: u64,
    pub writer_id: u64,
}

/// Receives every binding this table publishes for its replicas.
pub trait AffinityReplicaSink: Send + Sync {
    fn publish(&self, session_id: &str, target: AffinityTarget, version: AffinityVersion);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplicaApplyOutcome {
    Inserted,
    Refreshed,
    ReplacedExpired,
    ReplacedNewer,
    IgnoredInitializing,
    IgnoredConflict,
    RejectedSessionId,
    RejectedCapacity,
}

enum AffinityEntry {
    Initializing {
        revision: u64,
        notify: Arc<Notify>,
    },
    Bound {
        target: AffinityTarget,
        revision: u64,
        version: AffinityVersion,
        active_leases: usize,
        idle_deadline: Instant,
    },
}

struct Inner {
    entries: DashMap<String, AffinityEntry>,
    ttl: Duration,
    max_entries: usize,
    max_session_id_bytes: usize,
    entry_count: AtomicUsize,
    next_revision: AtomicU64,
    next_sequence: AtomicU64,
    writer_id: AtomicU64,
    cancel: CancellationToken,
    replica: OnceLock<Arc<dyn AffinityReplicaSink>>,
    reaper_started: Arc<Notify>,
    waiter_observed: Arc<Notify>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

/// The session table. Cheap to clone; the last clone stops the reaper.
#[derive(Clone)]
pub struct SessionAffinity {
    inner: Arc<Inner>,
}

/// A non-owning handle for replica appliers, so a sink task never keeps the
/// table alive.
#[derive(Clone)]
pub struct WeakSessionAffinity {
    inner: Weak<Inner>,
}

impl WeakSessionAffinity {
    pub fn upgrade(&self) -> Option<SessionAffinity> {
        self.inner.upgrade().map(|inner| SessionAffinity { inner })
    }
}

/// A session acquired for one request.
pub enum Acquired {
    /// New (or expired) session: select a worker, then `commit` it.
    Initialize(AffinityInitialization),
    /// Bound session: route to `target`; the lease releases on drop.
    Bound {
        target: AffinityTarget,
        lease: AffinityLease,
    },
}

/// One step of acquiring a session.
pub enum AcquireStep {
    /// New (or expired) session: select a worker, then `commit` it.
    Initialize(AffinityInitialization),
    /// Bound session: route to `target`; the lease releases on drop.
    Bound {
        target: AffinityTarget,
        lease: AffinityLease,
    },
    /// Another request is initializing this session; await, then retry.
    Wait(Pin<Box<OwnedNotified>>),
}

impl SessionAffinity {
    pub fn new(ttl: Duration) -> Result<Self, AffinityError> {
        Self::new_with_limits(
            ttl,
            MAX_SESSION_AFFINITY_ENTRIES,
            MAX_SESSION_AFFINITY_ID_BYTES,
        )
    }

    pub fn new_with_limits(
        ttl: Duration,
        max_entries: usize,
        max_session_id_bytes: usize,
    ) -> Result<Self, AffinityError> {
        if !(Duration::from_secs(1)..=Duration::from_secs(MAX_SESSION_AFFINITY_TTL_SECS))
            .contains(&ttl)
        {
            return Err(AffinityError::InvalidArgument(format!(
                "session affinity TTL must be between 1 and {MAX_SESSION_AFFINITY_TTL_SECS} seconds"
            )));
        }
        let inner = Arc::new(Inner {
            entries: DashMap::new(),
            ttl,
            max_entries,
            max_session_id_bytes,
            entry_count: AtomicUsize::new(0),
            next_revision: AtomicU64::new(1),
            next_sequence: AtomicU64::new(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64,
            ),
            writer_id: AtomicU64::new(0),
            cancel: CancellationToken::new(),
            replica: OnceLock::new(),
            reaper_started: Arc::new(Notify::new()),
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

    fn spawn_reaper(inner: &Arc<Inner>) {
        let weak = Arc::downgrade(inner);
        let cancel = inner.cancel.clone();
        let period = inner.ttl.min(Duration::from_secs(30));
        let reaper_started = inner.reaper_started.clone();
        tokio::spawn(async move {
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

    pub fn downgrade(&self) -> WeakSessionAffinity {
        WeakSessionAffinity {
            inner: Arc::downgrade(&self.inner),
        }
    }

    /// Install the replica sink and this replica's writer id. Returns `false`
    /// when a sink is already installed.
    pub fn enable_replication(&self, writer_id: u64, sink: Arc<dyn AffinityReplicaSink>) -> bool {
        self.inner.writer_id.store(writer_id, Ordering::Relaxed);
        self.inner.replica.set(sink).is_ok()
    }

    pub fn ttl(&self) -> Duration {
        self.inner.ttl
    }

    /// Acquire `session_id` for a request. `requested_target` is an explicit
    /// pin the request carries; a bound session must agree with it.
    pub fn try_acquire(
        &self,
        session_id: &str,
        requested_target: Option<AffinityTarget>,
    ) -> Result<AcquireStep, AffinityError> {
        self.validate_session_id(session_id)?;
        let now = Instant::now();
        match self.inner.entries.entry(session_id.to_string()) {
            Entry::Vacant(entry) => {
                self.reserve_entry()?;
                tracing::debug!(
                    session_id,
                    "session affinity miss: new session, pinning after worker selection"
                );
                let revision = self.inner.next_revision.fetch_add(1, Ordering::Relaxed);
                let notify = Arc::new(Notify::new());
                entry.insert(AffinityEntry::Initializing {
                    revision,
                    notify: notify.clone(),
                });
                Ok(AcquireStep::Initialize(AffinityInitialization {
                    table: Arc::downgrade(&self.inner),
                    session_id: session_id.to_string(),
                    revision,
                    notify,
                    requested_target,
                    active: true,
                }))
            }
            Entry::Occupied(mut entry) => match entry.get_mut() {
                AffinityEntry::Initializing { notify, .. } => {
                    self.inner.waiter_observed.notify_one();
                    // Register before releasing the entry so a commit or a
                    // dropped initialization between the two cannot be missed.
                    let mut notified = Box::pin(notify.clone().notified_owned());
                    notified.as_mut().enable();
                    Ok(AcquireStep::Wait(notified))
                }
                AffinityEntry::Bound {
                    active_leases,
                    idle_deadline,
                    ..
                } if *active_leases == 0 && *idle_deadline <= now => {
                    tracing::debug!(
                        session_id,
                        "session affinity miss: pin expired (idle past TTL), re-selecting worker"
                    );
                    let revision = self.inner.next_revision.fetch_add(1, Ordering::Relaxed);
                    let notify = Arc::new(Notify::new());
                    *entry.get_mut() = AffinityEntry::Initializing {
                        revision,
                        notify: notify.clone(),
                    };
                    Ok(AcquireStep::Initialize(AffinityInitialization {
                        table: Arc::downgrade(&self.inner),
                        session_id: session_id.to_string(),
                        revision,
                        notify,
                        requested_target,
                        active: true,
                    }))
                }
                AffinityEntry::Bound {
                    target,
                    revision,
                    version,
                    active_leases,
                    ..
                } => {
                    validate_bound_target(session_id, *target, requested_target)?;
                    tracing::debug!(
                        session_id,
                        worker_id = target.worker_id,
                        dp_rank = ?target.dp_rank,
                        active_leases = *active_leases + 1,
                        "session affinity hit: reusing pinned worker"
                    );
                    *active_leases += 1;
                    Ok(AcquireStep::Bound {
                        target: *target,
                        lease: AffinityLease {
                            table: Arc::downgrade(&self.inner),
                            session_id: session_id.to_string(),
                            revision: *revision,
                            version: *version,
                            active: true,
                        },
                    })
                }
            },
        }
    }

    /// Acquire, waiting out another request's initialization. Hosts that need
    /// to abandon the wait on their own signal drive [`Self::try_acquire`].
    pub async fn acquire(
        &self,
        session_id: &str,
        requested_target: Option<AffinityTarget>,
    ) -> Result<Acquired, AffinityError> {
        loop {
            match self.try_acquire(session_id, requested_target)? {
                AcquireStep::Wait(notified) => notified.await,
                AcquireStep::Initialize(init) => return Ok(Acquired::Initialize(init)),
                AcquireStep::Bound { target, lease } => {
                    return Ok(Acquired::Bound { target, lease });
                }
            }
        }
    }

    /// The bound target, without taking a lease. Read-only.
    pub fn query_target(
        &self,
        session_id: &str,
        requested_target: Option<AffinityTarget>,
    ) -> Result<Option<AffinityTarget>, AffinityError> {
        self.validate_session_id(session_id)?;
        let Some(entry) = self.inner.entries.get(session_id) else {
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
        validate_bound_target(session_id, *target, requested_target)?;
        tracing::debug!(
            session_id,
            worker_id = target.worker_id,
            dp_rank = ?target.dp_rank,
            "session affinity hit: reusing pinned worker"
        );
        Ok(Some(*target))
    }

    /// Advance the local sequence past one seen from a replica.
    pub fn observe_replica_sequence(&self, sequence: u64) {
        self.inner.observe_replica_sequence(sequence);
    }

    /// Apply a binding published by a replica.
    pub fn apply_replica_update(
        &self,
        session_id: String,
        target: AffinityTarget,
        version: AffinityVersion,
    ) -> ReplicaApplyOutcome {
        self.inner.apply_replica_update(session_id, target, version)
    }

    pub fn entry_count(&self) -> usize {
        self.inner.entry_count.load(Ordering::Relaxed)
    }

    /// Cancelled when the last handle drops.
    pub fn cancellation_token(&self) -> CancellationToken {
        self.inner.cancel.clone()
    }

    /// Test hook: the reaper task has started.
    pub async fn wait_for_reaper(&self) {
        self.inner.reaper_started.notified().await;
    }

    /// Test hook: a request observed an initializing session and is waiting.
    pub async fn wait_for_initializing_waiter(&self) {
        self.inner.waiter_observed.notified().await;
    }

    /// Test hook: expire an idle bound session now.
    pub fn expire_for_test(&self, session_id: &str) {
        let Some(mut entry) = self.inner.entries.get_mut(session_id) else {
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

    pub fn next_version(&self) -> AffinityVersion {
        self.inner.next_version()
    }

    fn validate_session_id(&self, session_id: &str) -> Result<(), AffinityError> {
        if session_id.len() > self.inner.max_session_id_bytes {
            return Err(AffinityError::InvalidArgument(format!(
                "session affinity ID must not exceed {} bytes",
                self.inner.max_session_id_bytes
            )));
        }
        Ok(())
    }

    fn reserve_entry(&self) -> Result<(), AffinityError> {
        self.inner.reserve_entry().then_some(()).ok_or_else(|| {
            AffinityError::ResourceExhausted("session affinity entry limit reached".to_string())
        })
    }
}

impl Inner {
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
        version: AffinityVersion,
    ) {
        if let Some(replica) = self.replica.get() {
            replica.publish(session_id, target, version);
        }
    }

    fn next_version(&self) -> AffinityVersion {
        AffinityVersion {
            sequence: self.next_sequence.fetch_add(1, Ordering::Relaxed),
            writer_id: self.writer_id.load(Ordering::Relaxed),
        }
    }

    fn observe_replica_sequence(&self, sequence: u64) {
        self.next_sequence
            .fetch_max(sequence.saturating_add(1), Ordering::Relaxed);
    }

    fn apply_replica_update(
        &self,
        session_id: String,
        target: AffinityTarget,
        version: AffinityVersion,
    ) -> ReplicaApplyOutcome {
        if session_id.len() > self.max_session_id_bytes {
            return ReplicaApplyOutcome::RejectedSessionId;
        }
        self.observe_replica_sequence(version.sequence);

        let now = Instant::now();
        match self.entries.entry(session_id) {
            Entry::Vacant(entry) => {
                if !self.reserve_entry() {
                    return ReplicaApplyOutcome::RejectedCapacity;
                }
                let revision = self.next_revision.fetch_add(1, Ordering::Relaxed);
                entry.insert(AffinityEntry::Bound {
                    target,
                    revision,
                    version,
                    active_leases: 0,
                    idle_deadline: now + self.ttl,
                });
                ReplicaApplyOutcome::Inserted
            }
            Entry::Occupied(mut entry) => match entry.get_mut() {
                AffinityEntry::Initializing { .. } => ReplicaApplyOutcome::IgnoredInitializing,
                AffinityEntry::Bound {
                    active_leases,
                    idle_deadline,
                    ..
                } if *active_leases == 0 && *idle_deadline <= now => {
                    let revision = self.next_revision.fetch_add(1, Ordering::Relaxed);
                    *entry.get_mut() = AffinityEntry::Bound {
                        target,
                        revision,
                        version,
                        active_leases: 0,
                        idle_deadline: now + self.ttl,
                    };
                    ReplicaApplyOutcome::ReplacedExpired
                }
                AffinityEntry::Bound {
                    target: existing,
                    version: existing_version,
                    idle_deadline,
                    ..
                } if *existing == target && version >= *existing_version => {
                    *existing_version = version;
                    *idle_deadline = now + self.ttl;
                    ReplicaApplyOutcome::Refreshed
                }
                AffinityEntry::Bound {
                    target: existing,
                    version: existing_version,
                    idle_deadline,
                    ..
                } if version > *existing_version => {
                    *existing = target;
                    *existing_version = version;
                    *idle_deadline = now + self.ttl;
                    ReplicaApplyOutcome::ReplacedNewer
                }
                AffinityEntry::Bound { .. } => ReplicaApplyOutcome::IgnoredConflict,
            },
        }
    }
}

/// A session this request is the first to bind. Dropping it uncommitted
/// releases the slot and wakes waiters so they re-acquire.
pub struct AffinityInitialization {
    table: Weak<Inner>,
    session_id: String,
    revision: u64,
    notify: Arc<Notify>,
    requested_target: Option<AffinityTarget>,
    active: bool,
}

impl AffinityInitialization {
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Bind the session to the worker the request was dispatched to.
    pub fn commit(mut self, target: AffinityTarget) -> Result<AffinityLease, AffinityError> {
        validate_bound_target(&self.session_id, target, self.requested_target)?;
        let Some(inner) = self.table.upgrade() else {
            return Err(AffinityError::Dropped);
        };
        let Some(mut entry) = inner.entries.get_mut(&self.session_id) else {
            return Err(AffinityError::InvalidArgument(
                "session affinity initialization was cancelled".to_string(),
            ));
        };
        if !matches!(
            entry.value(),
            AffinityEntry::Initializing { revision, .. } if *revision == self.revision
        ) {
            return Err(AffinityError::InvalidArgument(
                "session affinity initialization changed".to_string(),
            ));
        }
        let version = inner.next_version();
        *entry = AffinityEntry::Bound {
            target,
            revision: self.revision,
            version,
            active_leases: 1,
            idle_deadline: Instant::now() + inner.ttl,
        };
        drop(entry);
        self.active = false;
        self.notify.notify_waiters();
        Ok(AffinityLease {
            table: Arc::downgrade(&inner),
            session_id: self.session_id.clone(),
            revision: self.revision,
            version,
            active: true,
        })
    }
}

impl Drop for AffinityInitialization {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let Some(inner) = self.table.upgrade() else {
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

/// A live use of a bound session. Dropping it releases the use and refreshes
/// the idle deadline.
pub struct AffinityLease {
    table: Weak<Inner>,
    session_id: String,
    revision: u64,
    version: AffinityVersion,
    active: bool,
}

impl AffinityLease {
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Publish the binding to replicas.
    pub fn publish(&self, target: AffinityTarget) {
        if let Some(inner) = self.table.upgrade() {
            inner.publish_replica_update(&self.session_id, target, self.version);
        }
    }

    /// Move the binding from `expected` to `target` if nothing changed it since
    /// this lease was taken.
    pub fn rebind(&mut self, expected: AffinityTarget, target: AffinityTarget) -> bool {
        let Some(inner) = self.table.upgrade() else {
            return false;
        };
        let Some(mut entry) = inner.entries.get_mut(&self.session_id) else {
            return false;
        };
        let AffinityEntry::Bound {
            target: current,
            revision,
            version,
            ..
        } = entry.value_mut()
        else {
            return false;
        };
        if *revision != self.revision || *version != self.version || *current != expected {
            return false;
        }
        let next_version = inner.next_version();
        *current = target;
        *version = next_version;
        self.version = next_version;
        true
    }

    /// Soft mode: where a bound session that was dispatched to `dispatched`
    /// should now point. A worker-only binding stays worker-only; a ranked
    /// binding follows the dispatched rank when there is one.
    pub fn rebound_target(bound: AffinityTarget, dispatched: AffinityTarget) -> AffinityTarget {
        match bound.dp_rank {
            None => AffinityTarget::new(dispatched.worker_id, None),
            Some(_) => dispatched.dp_rank.map_or(bound, |dp_rank| {
                AffinityTarget::new(dispatched.worker_id, Some(dp_rank))
            }),
        }
    }

    fn release(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;
        let Some(inner) = self.table.upgrade() else {
            return;
        };
        let (target, version) = {
            let Some(mut entry) = inner.entries.get_mut(&self.session_id) else {
                return;
            };
            let AffinityEntry::Bound {
                target,
                revision,
                version,
                active_leases,
                idle_deadline,
            } = entry.value_mut()
            else {
                return;
            };
            if *revision != self.revision || *active_leases == 0 {
                return;
            }
            *active_leases -= 1;
            if *version != self.version {
                return;
            }
            *idle_deadline = Instant::now() + inner.ttl;
            (*target, *version)
        };
        inner.publish_replica_update(&self.session_id, target, version);
    }

    /// Drop the binding this lease holds (the bound worker is gone). A binding
    /// already replaced by a newer version is left alone and this use released.
    pub fn invalidate(&mut self) {
        if !self.active {
            return;
        }
        let Some(inner) = self.table.upgrade() else {
            self.active = false;
            return;
        };
        let removed = inner.entries.remove_if(&self.session_id, |_, entry| {
            matches!(
                entry,
                AffinityEntry::Bound { revision, version, .. }
                    if *revision == self.revision && *version == self.version
            )
        });
        match removed {
            Some((_, AffinityEntry::Bound { target, .. })) => {
                tracing::debug!(
                    session_id = %self.session_id,
                    worker_id = target.worker_id,
                    dp_rank = ?target.dp_rank,
                    "invalidated current session affinity binding"
                );
                self.active = false;
                inner.entry_count.fetch_sub(1, Ordering::Relaxed);
            }
            Some((_, AffinityEntry::Initializing { .. })) => {
                unreachable!("bound lease removed an initializing entry")
            }
            None => self.release(),
        }
    }
}

impl Drop for AffinityLease {
    fn drop(&mut self) {
        self.release();
    }
}

/// A bound session must agree with the target the request explicitly asked for.
pub fn validate_bound_target(
    session_id: &str,
    bound: AffinityTarget,
    requested: Option<AffinityTarget>,
) -> Result<(), AffinityError> {
    let Some(requested) = requested else {
        return Ok(());
    };
    if bound.worker_id != requested.worker_id {
        return Err(AffinityError::InvalidArgument(format!(
            "session {session_id} is bound to worker {}, not {}",
            bound.worker_id, requested.worker_id
        )));
    }
    match (bound.dp_rank, requested.dp_rank) {
        (Some(bound), Some(requested)) if bound != requested => {
            Err(AffinityError::InvalidArgument(format!(
                "session {session_id} is bound to DP rank {bound}, not {requested}"
            )))
        }
        (None, Some(requested)) => Err(AffinityError::InvalidArgument(format!(
            "session {session_id} has worker-only affinity and cannot add DP rank {requested}"
        ))),
        _ => Ok(()),
    }
}

/// Hard mode: a bound session must have been dispatched where it is bound.
pub fn validate_dispatch_target(
    session_id: &str,
    bound: AffinityTarget,
    dispatched: AffinityTarget,
) -> Result<(), AffinityError> {
    if bound.worker_id != dispatched.worker_id {
        return Err(AffinityError::InvalidArgument(format!(
            "session {session_id} is bound to worker {}, not {}",
            bound.worker_id, dispatched.worker_id
        )));
    }
    if let Some(bound_rank) = bound.dp_rank {
        match dispatched.dp_rank {
            Some(dispatched_rank) if dispatched_rank == bound_rank => {}
            Some(dispatched_rank) => {
                return Err(AffinityError::InvalidArgument(format!(
                    "session {session_id} is bound to DP rank {bound_rank}, not {dispatched_rank}"
                )));
            }
            None => {
                return Err(AffinityError::InvalidArgument(format!(
                    "session {session_id} is bound to DP rank {bound_rank}, but dispatch did not select a DP rank"
                )));
            }
        }
    }
    Ok(())
}
