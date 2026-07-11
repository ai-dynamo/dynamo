// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    future::Future,
    sync::{
        Arc, Weak,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::Duration,
};

use dashmap::{DashMap, mapref::entry::Entry};
use thiserror::Error;
use tokio::{sync::Notify, time::Instant};
use tokio_util::sync::CancellationToken;

const MIN_IDLE_TTL: Duration = Duration::from_secs(1);
const MAX_IDLE_TTL: Duration = Duration::from_secs(31_536_000);

/// Resource and lifetime limits for session placement state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionPlacementConfig {
    /// Time a committed placement remains after its last active request.
    pub idle_ttl: Duration,
    /// Maximum time an initialization owner may hold a provisional placement.
    ///
    /// `None` preserves the owner until it commits or is dropped.
    pub initialization_timeout: Option<Duration>,
    /// Maximum number of initializing and committed entries.
    pub max_entries: usize,
    /// Maximum encoded placement-key size.
    pub max_key_bytes: usize,
}

/// Failures produced while configuring or acquiring session placement state.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SessionPlacementError {
    #[error("invalid session placement configuration: {0}")]
    InvalidConfig(&'static str),

    #[error("session placement key is {actual_bytes} bytes; maximum is {max_bytes}")]
    KeyTooLong {
        actual_bytes: usize,
        max_bytes: usize,
    },

    #[error("session placement entry limit of {max_entries} reached")]
    Capacity { max_entries: usize },

    #[error("session placement coordinator was dropped")]
    CoordinatorDropped,

    #[error("session placement requires a Tokio runtime")]
    RuntimeUnavailable,

    #[error("session placement initialization was cancelled")]
    InitializationCancelled,

    #[error("session placement initialization changed")]
    InitializationChanged,

    #[error("session placement acquisition was cancelled")]
    AcquireCancelled,
}

enum PlacementEntry<T> {
    Initializing {
        revision: u64,
        notify: Arc<Notify>,
        deadline: Option<Instant>,
    },
    Bound {
        target: T,
        revision: u64,
        active_leases: usize,
        idle_deadline: Instant,
    },
}

struct SessionPlacementInner<T> {
    entries: DashMap<String, PlacementEntry<T>>,
    config: SessionPlacementConfig,
    entry_count: AtomicUsize,
    next_revision: AtomicU64,
    cancel: CancellationToken,
    #[cfg(test)]
    reaper_started: Arc<Notify>,
    #[cfg(test)]
    waiter_observed: Arc<Notify>,
}

impl<T> Drop for SessionPlacementInner<T> {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

/// Coordinates bounded soft placement state for arbitrary cloneable routing targets.
pub struct SessionPlacement<T> {
    inner: Arc<SessionPlacementInner<T>>,
}

impl<T> Clone for SessionPlacement<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T> SessionPlacement<T>
where
    T: Clone + Send + Sync + 'static,
{
    /// Creates a coordinator and starts its idle-state reaper on the current Tokio runtime.
    pub fn new(config: SessionPlacementConfig) -> Result<Self, SessionPlacementError> {
        validate_config(config)?;
        tokio::runtime::Handle::try_current()
            .map_err(|_| SessionPlacementError::RuntimeUnavailable)?;
        let inner = Arc::new(SessionPlacementInner {
            entries: DashMap::new(),
            config,
            entry_count: AtomicUsize::new(0),
            next_revision: AtomicU64::new(1),
            cancel: CancellationToken::new(),
            #[cfg(test)]
            reaper_started: Arc::new(Notify::new()),
            #[cfg(test)]
            waiter_observed: Arc::new(Notify::new()),
        });
        Self::spawn_reaper(&inner);
        Ok(Self { inner })
    }

    /// Acquires an existing placement or ownership of the key's provisional initialization.
    pub async fn acquire(&self, key: &str) -> Result<PlacementAcquire<T>, SessionPlacementError> {
        self.acquire_inner(key, std::future::pending()).await
    }

    pub(crate) async fn acquire_with_cancellation<F>(
        &self,
        key: &str,
        cancellation: F,
    ) -> Result<PlacementAcquire<T>, SessionPlacementError>
    where
        F: Future<Output = ()>,
    {
        self.acquire_inner(key, cancellation).await
    }

    async fn acquire_inner<F>(
        &self,
        key: &str,
        cancellation: F,
    ) -> Result<PlacementAcquire<T>, SessionPlacementError>
    where
        F: Future<Output = ()>,
    {
        self.validate_key(key)?;
        let key = key.to_owned();
        tokio::pin!(cancellation);

        loop {
            let now = Instant::now();
            match self.inner.entries.entry(key.clone()) {
                Entry::Vacant(entry) => {
                    self.reserve_entry()?;
                    let revision = self.next_revision();
                    let notify = Arc::new(Notify::new());
                    entry.insert(PlacementEntry::Initializing {
                        revision,
                        notify: notify.clone(),
                        deadline: self
                            .inner
                            .config
                            .initialization_timeout
                            .map(|timeout| now + timeout),
                    });
                    return Ok(PlacementAcquire::Initialize(PlacementInitialization {
                        placement: Arc::downgrade(&self.inner),
                        key,
                        revision,
                        notify,
                        active: true,
                    }));
                }
                Entry::Occupied(mut entry) => match entry.get_mut() {
                    PlacementEntry::Initializing {
                        notify, deadline, ..
                    } if deadline.as_ref().is_some_and(|deadline| *deadline <= now) => {
                        let stale_notify = notify.clone();
                        let revision = self.next_revision();
                        let notify = Arc::new(Notify::new());
                        *entry.get_mut() = PlacementEntry::Initializing {
                            revision,
                            notify: notify.clone(),
                            deadline: self
                                .inner
                                .config
                                .initialization_timeout
                                .map(|timeout| now + timeout),
                        };
                        drop(entry);
                        stale_notify.notify_waiters();
                        return Ok(PlacementAcquire::Initialize(PlacementInitialization {
                            placement: Arc::downgrade(&self.inner),
                            key,
                            revision,
                            notify,
                            active: true,
                        }));
                    }
                    PlacementEntry::Initializing {
                        notify, deadline, ..
                    } => {
                        #[cfg(test)]
                        self.inner.waiter_observed.notify_one();
                        let deadline = *deadline;
                        let notified = notify.clone().notified_owned();
                        tokio::pin!(notified);
                        notified.as_mut().enable();
                        drop(entry);
                        let timeout = async move {
                            match deadline {
                                Some(deadline) => tokio::time::sleep_until(deadline).await,
                                None => std::future::pending::<()>().await,
                            }
                        };
                        tokio::pin!(timeout);
                        tokio::select! {
                            biased;
                            _ = cancellation.as_mut() => {
                                return Err(SessionPlacementError::AcquireCancelled);
                            }
                            _ = notified => {}
                            _ = timeout.as_mut() => {}
                        }
                    }
                    PlacementEntry::Bound {
                        active_leases,
                        idle_deadline,
                        ..
                    } if *active_leases == 0 && *idle_deadline <= now => {
                        let revision = self.next_revision();
                        let notify = Arc::new(Notify::new());
                        *entry.get_mut() = PlacementEntry::Initializing {
                            revision,
                            notify: notify.clone(),
                            deadline: self
                                .inner
                                .config
                                .initialization_timeout
                                .map(|timeout| now + timeout),
                        };
                        drop(entry);
                        return Ok(PlacementAcquire::Initialize(PlacementInitialization {
                            placement: Arc::downgrade(&self.inner),
                            key,
                            revision,
                            notify,
                            active: true,
                        }));
                    }
                    PlacementEntry::Bound {
                        target,
                        revision,
                        active_leases,
                        ..
                    } => {
                        *active_leases += 1;
                        return Ok(PlacementAcquire::Bound {
                            target: target.clone(),
                            lease: PlacementLease {
                                placement: Arc::downgrade(&self.inner),
                                key,
                                revision: *revision,
                                active: true,
                            },
                        });
                    }
                },
            }
        }
    }

    /// Returns a valid committed target without creating state or acquiring a request lease.
    pub fn query(&self, key: &str) -> Result<Option<T>, SessionPlacementError> {
        self.validate_key(key)?;
        let Some(entry) = self.inner.entries.get(key) else {
            return Ok(None);
        };
        let PlacementEntry::Bound {
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
        Ok(Some(target.clone()))
    }

    fn spawn_reaper(inner: &Arc<SessionPlacementInner<T>>) {
        let weak = Arc::downgrade(inner);
        let cancel = inner.cancel.clone();
        let period = inner
            .config
            .initialization_timeout
            .map_or(inner.config.idle_ttl, |timeout| {
                inner.config.idle_ttl.min(timeout)
            })
            .min(Duration::from_secs(30));
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
                inner.entries.retain(|_, entry| {
                    let retain = match entry {
                        PlacementEntry::Initializing {
                            notify, deadline, ..
                        } if deadline.as_ref().is_some_and(|deadline| *deadline <= now) => {
                            notify.notify_waiters();
                            false
                        }
                        PlacementEntry::Bound {
                            active_leases: 0,
                            idle_deadline,
                            ..
                        } if *idle_deadline <= now => false,
                        _ => true,
                    };
                    if !retain {
                        inner.entry_count.fetch_sub(1, Ordering::Relaxed);
                    }
                    retain
                });
            }
        });
    }

    fn validate_key(&self, key: &str) -> Result<(), SessionPlacementError> {
        if key.len() > self.inner.config.max_key_bytes {
            return Err(SessionPlacementError::KeyTooLong {
                actual_bytes: key.len(),
                max_bytes: self.inner.config.max_key_bytes,
            });
        }
        Ok(())
    }

    fn reserve_entry(&self) -> Result<(), SessionPlacementError> {
        self.inner
            .entry_count
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| {
                (count < self.inner.config.max_entries).then_some(count + 1)
            })
            .map(|_| ())
            .map_err(|_| SessionPlacementError::Capacity {
                max_entries: self.inner.config.max_entries,
            })
    }

    fn next_revision(&self) -> u64 {
        self.inner.next_revision.fetch_add(1, Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn entry_count(&self) -> usize {
        self.inner.entry_count.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn cancellation_token(&self) -> CancellationToken {
        self.inner.cancel.clone()
    }

    #[cfg(test)]
    pub(crate) async fn wait_for_reaper(&self) {
        self.inner.reaper_started.notified().await;
    }

    #[cfg(test)]
    pub(crate) async fn wait_for_initializing_waiter(&self) {
        self.inner.waiter_observed.notified().await;
    }

    #[cfg(test)]
    pub(crate) fn expire_for_test(&self, key: &str) {
        let Some(mut entry) = self.inner.entries.get_mut(key) else {
            panic!("session placement entry missing");
        };
        let PlacementEntry::Bound {
            active_leases,
            idle_deadline,
            ..
        } = entry.value_mut()
        else {
            panic!("session placement entry is not bound");
        };
        assert_eq!(*active_leases, 0);
        *idle_deadline = Instant::now();
    }
}

/// Result of acquiring placement state for a session key.
pub enum PlacementAcquire<T> {
    /// The caller owns provisional initialization and must commit or drop it.
    Initialize(PlacementInitialization<T>),
    /// The session already has a committed target and an active-request lease.
    Bound { target: T, lease: PlacementLease<T> },
}

/// Exclusive ownership of a provisional placement decision.
pub struct PlacementInitialization<T> {
    placement: Weak<SessionPlacementInner<T>>,
    key: String,
    revision: u64,
    notify: Arc<Notify>,
    active: bool,
}

impl<T> PlacementInitialization<T> {
    /// Commits the target after the request plane confirms that dispatch was accepted.
    pub fn commit(mut self, target: T) -> Result<PlacementLease<T>, SessionPlacementError> {
        let Some(inner) = self.placement.upgrade() else {
            return Err(SessionPlacementError::CoordinatorDropped);
        };
        let Some(mut entry) = inner.entries.get_mut(&self.key) else {
            return Err(SessionPlacementError::InitializationCancelled);
        };
        let now = Instant::now();
        let initialization_is_current = matches!(
            entry.value(),
            PlacementEntry::Initializing {
                revision,
                deadline,
                ..
            } if *revision == self.revision
                && deadline
                    .as_ref()
                    .is_none_or(|deadline| *deadline > now)
        );
        if !initialization_is_current {
            let initialization_expired = matches!(
                entry.value(),
                PlacementEntry::Initializing {
                    revision,
                    deadline,
                    ..
                } if *revision == self.revision
                    && deadline
                        .as_ref()
                        .is_some_and(|deadline| *deadline <= now)
            );
            drop(entry);
            if initialization_expired {
                let removed = inner.entries.remove_if(&self.key, |_, entry| {
                    matches!(
                        entry,
                        PlacementEntry::Initializing { revision, .. }
                            if *revision == self.revision
                    )
                });
                if removed.is_some() {
                    inner.entry_count.fetch_sub(1, Ordering::Relaxed);
                }
                self.notify.notify_waiters();
                self.active = false;
                return Err(SessionPlacementError::InitializationCancelled);
            }
            return Err(SessionPlacementError::InitializationChanged);
        }
        *entry = PlacementEntry::Bound {
            target,
            revision: self.revision,
            active_leases: 1,
            idle_deadline: Instant::now() + inner.config.idle_ttl,
        };
        drop(entry);
        self.active = false;
        self.notify.notify_waiters();
        Ok(PlacementLease {
            placement: Arc::downgrade(&inner),
            key: self.key.clone(),
            revision: self.revision,
            active: true,
        })
    }
}

impl<T> Drop for PlacementInitialization<T> {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let Some(inner) = self.placement.upgrade() else {
            return;
        };
        let removed = inner.entries.remove_if(&self.key, |_, entry| {
            matches!(
                entry,
                PlacementEntry::Initializing { revision, .. } if *revision == self.revision
            )
        });
        if removed.is_some() {
            inner.entry_count.fetch_sub(1, Ordering::Relaxed);
        }
        self.notify.notify_waiters();
    }
}

/// Active-request lease that prevents a committed placement from expiring.
pub struct PlacementLease<T> {
    placement: Weak<SessionPlacementInner<T>>,
    key: String,
    revision: u64,
    active: bool,
}

impl<T> PlacementLease<T> {
    /// Removes the matching placement, without affecting a newer revision.
    pub fn invalidate(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;
        let Some(inner) = self.placement.upgrade() else {
            return;
        };
        let removed = inner.entries.remove_if(&self.key, |_, entry| {
            matches!(
                entry,
                PlacementEntry::Bound { revision, .. } if *revision == self.revision
            )
        });
        if removed.is_some() {
            inner.entry_count.fetch_sub(1, Ordering::Relaxed);
        }
    }

    pub(crate) fn abandon(&mut self) {
        self.release(false);
    }

    fn release(&mut self, refresh_ttl: bool) {
        if !self.active {
            return;
        }
        self.active = false;
        let Some(inner) = self.placement.upgrade() else {
            return;
        };
        let Some(mut entry) = inner.entries.get_mut(&self.key) else {
            return;
        };
        let PlacementEntry::Bound {
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
        if refresh_ttl {
            *idle_deadline = Instant::now() + inner.config.idle_ttl;
        }
    }
}

impl<T> Drop for PlacementLease<T> {
    fn drop(&mut self) {
        self.release(true);
    }
}

fn validate_config(config: SessionPlacementConfig) -> Result<(), SessionPlacementError> {
    if !(MIN_IDLE_TTL..=MAX_IDLE_TTL).contains(&config.idle_ttl) {
        return Err(SessionPlacementError::InvalidConfig(
            "idle_ttl must be between 1 second and 1 year",
        ));
    }
    if let Some(initialization_timeout) = config.initialization_timeout
        && !(MIN_IDLE_TTL..=MAX_IDLE_TTL).contains(&initialization_timeout)
    {
        return Err(SessionPlacementError::InvalidConfig(
            "initialization_timeout must be between 1 second and 1 year when configured",
        ));
    }
    if config.max_entries == 0 {
        return Err(SessionPlacementError::InvalidConfig(
            "max_entries must be greater than zero",
        ));
    }
    if config.max_key_bytes == 0 {
        return Err(SessionPlacementError::InvalidConfig(
            "max_key_bytes must be greater than zero",
        ));
    }
    Ok(())
}
