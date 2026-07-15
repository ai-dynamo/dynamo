// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    future::Future,
    sync::{
        Arc, Weak,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    time::Duration,
};

use dashmap::{DashMap, mapref::entry::Entry};
use parking_lot::Mutex;
use thiserror::Error;
use tokio::{runtime::Handle, sync::Notify, task::JoinHandle, time::Instant};
use tokio_util::sync::CancellationToken;

const MIN_IDLE_TTL: Duration = Duration::from_secs(1);
const MAX_IDLE_TTL: Duration = Duration::from_secs(31_536_000);
// DashMap removal and the global count update are separate atomic operations. Retry a bounded
// number of times when a concurrent removal has exposed a map slot but not yet released its count.
const CAPACITY_CONTENTION_RETRIES: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SessionPlacementConfig {
    pub(crate) idle_ttl: Duration,
    pub(crate) initialization_timeout: Option<Duration>,
    pub(crate) max_entries: usize,
    pub(crate) max_key_bytes: usize,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub(crate) enum SessionPlacementError {
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

    #[error(
        "session placement dispatch {attempt_id} for target generation {target_generation} has an ambiguous outcome"
    )]
    DispatchAmbiguous {
        attempt_id: u64,
        target_generation: u64,
    },

    #[error(
        "session placement target generation changed from {expected_generation} to {actual_generation}"
    )]
    TargetGenerationChanged {
        expected_generation: u64,
        actual_generation: u64,
    },

    #[error("session placement acquisition was cancelled")]
    AcquireCancelled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PlacementAttemptId(u64);

impl PlacementAttemptId {
    pub(crate) fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct TargetGeneration(u64);

impl TargetGeneration {
    pub(crate) const UNVERSIONED: Self = Self(0);

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "versioned targets are consumed by the global-router integration"
        )
    )]
    pub(crate) fn new(generation: u64) -> Self {
        Self(generation)
    }

    pub(crate) fn get(self) -> u64 {
        self.0
    }
}

struct VersionedTargetInner<T> {
    target: T,
    generation: TargetGeneration,
}

pub(crate) struct VersionedTarget<T> {
    inner: Arc<VersionedTargetInner<T>>,
}

impl<T> VersionedTarget<T> {
    fn new(target: T, generation: TargetGeneration) -> Self {
        Self {
            inner: Arc::new(VersionedTargetInner { target, generation }),
        }
    }

    pub(crate) fn target(&self) -> &T {
        &self.inner.target
    }

    pub(crate) fn generation(&self) -> TargetGeneration {
        self.inner.generation
    }

    fn same_as(&self, other: &Self) -> bool {
        self.generation() == other.generation() && Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl<T> Clone for VersionedTarget<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

// A reservation may be replaced before dispatch begins. Once a target is Dispatching, every
// non-definitive outcome is quarantined as Ambiguous so another owner cannot silently replay it.
enum PlacementEntry<T> {
    Reserved {
        attempt: PlacementAttemptId,
        notify: Arc<Notify>,
        deadline: Option<Instant>,
    },
    Dispatching {
        attempt: PlacementAttemptId,
        candidate: VersionedTarget<T>,
        notify: Arc<Notify>,
        deadline: Option<Instant>,
    },
    Ambiguous {
        attempt: PlacementAttemptId,
        candidate: VersionedTarget<T>,
    },
    Bound {
        target: VersionedTarget<T>,
        revision: PlacementAttemptId,
        active_leases: usize,
        idle_deadline: Instant,
    },
}

struct SessionPlacementInner<T> {
    entries: DashMap<String, PlacementEntry<T>>,
    config: SessionPlacementConfig,
    entry_count: AtomicUsize,
    next_attempt: AtomicU64,
    cancel: CancellationToken,
    reaper_running: AtomicBool,
    reaper: Mutex<Option<JoinHandle<()>>>,
    #[cfg(test)]
    reaper_started: Arc<Notify>,
    #[cfg(test)]
    reaper_completed: Arc<Notify>,
    #[cfg(test)]
    waiter_observed: Arc<Notify>,
}

struct ReaperRunning<T> {
    inner: Weak<SessionPlacementInner<T>>,
}

impl<T> Drop for ReaperRunning<T> {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.upgrade() {
            inner.reaper_running.store(false, Ordering::Release);
        }
    }
}

impl<T> Drop for SessionPlacementInner<T> {
    fn drop(&mut self) {
        self.cancel.cancel();
        if let Some(reaper) = self.reaper.get_mut().take() {
            reaper.abort();
        }
    }
}

pub(crate) struct SessionPlacement<T> {
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
    T: Send + Sync + 'static,
{
    pub(crate) fn new(config: SessionPlacementConfig) -> Result<Self, SessionPlacementError> {
        validate_config(config)?;
        let handle =
            Handle::try_current().map_err(|_| SessionPlacementError::RuntimeUnavailable)?;
        let inner = Arc::new(SessionPlacementInner {
            entries: DashMap::new(),
            config,
            entry_count: AtomicUsize::new(0),
            next_attempt: AtomicU64::new(1),
            cancel: CancellationToken::new(),
            reaper_running: AtomicBool::new(false),
            reaper: Mutex::new(None),
            #[cfg(test)]
            reaper_started: Arc::new(Notify::new()),
            #[cfg(test)]
            reaper_completed: Arc::new(Notify::new()),
            #[cfg(test)]
            waiter_observed: Arc::new(Notify::new()),
        });
        *inner.reaper.lock() = Some(Self::spawn_reaper_task(&inner, &handle));
        Ok(Self { inner })
    }

    pub(crate) async fn acquire(
        &self,
        key: &str,
    ) -> Result<PlacementAcquire<T>, SessionPlacementError> {
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
        self.ensure_reaper();
        let key = key.to_owned();
        let mut reaped_for_capacity = false;
        let mut capacity_contention_retries = 0;
        tokio::pin!(cancellation);

        loop {
            let now = Instant::now();
            match self.inner.entries.entry(key.clone()) {
                Entry::Vacant(entry) => {
                    if let Err(error) = self.reserve_entry() {
                        drop(entry);
                        if reaped_for_capacity {
                            if capacity_contention_retries < CAPACITY_CONTENTION_RETRIES
                                && self.inner.entries.len() < self.inner.config.max_entries
                            {
                                capacity_contention_retries += 1;
                                tokio::task::yield_now().await;
                                continue;
                            }
                            return Err(error);
                        }
                        self.reap_expired_async().await;
                        reaped_for_capacity = true;
                        continue;
                    }
                    let attempt = self.next_attempt();
                    let notify = Arc::new(Notify::new());
                    entry.insert(PlacementEntry::Reserved {
                        attempt,
                        notify: notify.clone(),
                        deadline: self.initialization_deadline(now),
                    });
                    return Ok(PlacementAcquire::Initialize(PlacementInitialization {
                        placement: Arc::downgrade(&self.inner),
                        key,
                        attempt,
                        notify,
                        active: true,
                    }));
                }
                Entry::Occupied(mut entry) => match entry.get_mut() {
                    PlacementEntry::Reserved {
                        notify, deadline, ..
                    } if deadline.as_ref().is_some_and(|deadline| *deadline <= now) => {
                        let stale_notify = notify.clone();
                        let attempt = self.next_attempt();
                        let notify = Arc::new(Notify::new());
                        *entry.get_mut() = PlacementEntry::Reserved {
                            attempt,
                            notify: notify.clone(),
                            deadline: self.initialization_deadline(now),
                        };
                        drop(entry);
                        stale_notify.notify_waiters();
                        return Ok(PlacementAcquire::Initialize(PlacementInitialization {
                            placement: Arc::downgrade(&self.inner),
                            key,
                            attempt,
                            notify,
                            active: true,
                        }));
                    }
                    PlacementEntry::Reserved {
                        notify, deadline, ..
                    } => {
                        let deadline = *deadline;
                        let notified = notify.clone().notified_owned();
                        self.wait_for_initialization(
                            entry,
                            notified,
                            deadline,
                            cancellation.as_mut(),
                        )
                        .await?;
                    }
                    PlacementEntry::Dispatching {
                        attempt,
                        candidate,
                        notify,
                        deadline,
                    } if deadline.as_ref().is_some_and(|deadline| *deadline <= now) => {
                        let attempt = *attempt;
                        let candidate = candidate.clone();
                        let notify = notify.clone();
                        *entry.get_mut() = PlacementEntry::Ambiguous {
                            attempt,
                            candidate: candidate.clone(),
                        };
                        drop(entry);
                        notify.notify_waiters();
                        return Err(dispatch_ambiguous(attempt, candidate.generation()));
                    }
                    PlacementEntry::Dispatching {
                        notify, deadline, ..
                    } => {
                        let deadline = *deadline;
                        let notified = notify.clone().notified_owned();
                        self.wait_for_initialization(
                            entry,
                            notified,
                            deadline,
                            cancellation.as_mut(),
                        )
                        .await?;
                    }
                    PlacementEntry::Ambiguous { attempt, candidate } => {
                        return Err(dispatch_ambiguous(*attempt, candidate.generation()));
                    }
                    PlacementEntry::Bound {
                        target,
                        active_leases,
                        idle_deadline,
                        ..
                    } if *active_leases == 0 && *idle_deadline <= now => {
                        let stale_target = target.clone();
                        let attempt = self.next_attempt();
                        let notify = Arc::new(Notify::new());
                        *entry.get_mut() = PlacementEntry::Reserved {
                            attempt,
                            notify: notify.clone(),
                            deadline: self.initialization_deadline(now),
                        };
                        drop(entry);
                        drop(stale_target);
                        return Ok(PlacementAcquire::Initialize(PlacementInitialization {
                            placement: Arc::downgrade(&self.inner),
                            key,
                            attempt,
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

    async fn wait_for_initialization<F>(
        &self,
        entry: dashmap::mapref::entry::OccupiedEntry<'_, String, PlacementEntry<T>>,
        notified: tokio::sync::futures::OwnedNotified,
        deadline: Option<Instant>,
        mut cancellation: std::pin::Pin<&mut F>,
    ) -> Result<(), SessionPlacementError>
    where
        F: Future<Output = ()>,
    {
        #[cfg(test)]
        self.inner.waiter_observed.notify_one();
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
            _ = cancellation.as_mut() => Err(SessionPlacementError::AcquireCancelled),
            _ = notified => Ok(()),
            _ = timeout.as_mut() => Ok(()),
        }
    }

    pub(crate) fn query(
        &self,
        key: &str,
    ) -> Result<Option<VersionedTarget<T>>, SessionPlacementError> {
        self.validate_key(key)?;
        self.ensure_reaper();
        let Some(entry) = self.inner.entries.get(key) else {
            return Ok(None);
        };
        match entry.value() {
            PlacementEntry::Ambiguous { attempt, candidate } => {
                Err(dispatch_ambiguous(*attempt, candidate.generation()))
            }
            PlacementEntry::Bound {
                target,
                active_leases,
                idle_deadline,
                ..
            } if *active_leases > 0 || *idle_deadline > Instant::now() => Ok(Some(target.clone())),
            _ => Ok(None),
        }
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "ambiguous dispatch recovery is consumed by the global-router integration"
        )
    )]
    pub(crate) fn resolve_ambiguous(
        &self,
        key: &str,
        attempt: PlacementAttemptId,
        generation: TargetGeneration,
        resolution: AmbiguousResolution,
    ) -> Result<Option<PlacementLease<T>>, SessionPlacementError> {
        self.validate_key(key)?;
        match resolution {
            AmbiguousResolution::Accepted => {
                let Some(mut entry) = self.inner.entries.get_mut(key) else {
                    return Err(SessionPlacementError::InitializationCancelled);
                };
                let PlacementEntry::Ambiguous {
                    attempt: current,
                    candidate,
                } = entry.value()
                else {
                    return Err(SessionPlacementError::InitializationChanged);
                };
                validate_attempt(*current, candidate, attempt, generation)?;
                let target = candidate.clone();
                *entry = PlacementEntry::Bound {
                    target,
                    revision: attempt,
                    active_leases: 1,
                    idle_deadline: Instant::now() + self.inner.config.idle_ttl,
                };
                drop(entry);
                Ok(Some(PlacementLease {
                    placement: Arc::downgrade(&self.inner),
                    key: key.to_owned(),
                    revision: attempt,
                    active: true,
                }))
            }
            AmbiguousResolution::DefinitelyNotAccepted => {
                let removed = self.inner.entries.remove_if(key, |_, entry| {
                    matches!(
                        entry,
                        PlacementEntry::Ambiguous {
                            attempt: current,
                            candidate,
                        } if *current == attempt && candidate.generation() == generation
                    )
                });
                if removed.is_none() {
                    return self.ambiguous_resolution_error(key, attempt, generation);
                }
                self.inner.entry_count.fetch_sub(1, Ordering::Relaxed);
                drop(removed);
                Ok(None)
            }
        }
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "only used by the deferred global-router recovery path"
        )
    )]
    fn ambiguous_resolution_error(
        &self,
        key: &str,
        attempt: PlacementAttemptId,
        generation: TargetGeneration,
    ) -> Result<Option<PlacementLease<T>>, SessionPlacementError> {
        let Some(entry) = self.inner.entries.get(key) else {
            return Err(SessionPlacementError::InitializationCancelled);
        };
        let PlacementEntry::Ambiguous {
            attempt: current,
            candidate,
        } = entry.value()
        else {
            return Err(SessionPlacementError::InitializationChanged);
        };
        validate_attempt(*current, candidate, attempt, generation)?;
        Err(SessionPlacementError::InitializationChanged)
    }

    fn spawn_reaper_task(inner: &Arc<SessionPlacementInner<T>>, handle: &Handle) -> JoinHandle<()> {
        let weak = Arc::downgrade(inner);
        let running = ReaperRunning {
            inner: weak.clone(),
        };
        inner.reaper_running.store(true, Ordering::Release);
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
        #[cfg(test)]
        let reaper_completed = inner.reaper_completed.clone();
        handle.spawn(async move {
            let _running = running;
            let sleep = tokio::time::sleep(period);
            tokio::pin!(sleep);
            #[cfg(test)]
            reaper_started.notify_one();
            loop {
                tokio::select! {
                    _ = cancel.cancelled() => return,
                    _ = sleep.as_mut() => {}
                }
                let Some(inner) = weak.upgrade() else {
                    return;
                };
                let cleanup = tokio::task::spawn_blocking(move || {
                    inner.reap_expired(Instant::now());
                });
                match cleanup.await {
                    Ok(()) => {
                        #[cfg(test)]
                        reaper_completed.notify_one();
                    }
                    Err(error) => {
                        tracing::warn!(?error, "session placement cleanup task failed");
                    }
                }
                sleep.as_mut().reset(Instant::now() + period);
            }
        })
    }

    fn ensure_reaper(&self) {
        if self.inner.reaper_running.load(Ordering::Acquire) {
            return;
        }
        let mut reaper = self.inner.reaper.lock();
        if self.inner.reaper_running.load(Ordering::Acquire) {
            return;
        }
        if let Ok(handle) = Handle::try_current() {
            *reaper = Some(Self::spawn_reaper_task(&self.inner, &handle));
        }
    }

    async fn reap_expired_async(&self) {
        let inner = self.inner.clone();
        if let Err(error) =
            tokio::task::spawn_blocking(move || inner.reap_expired(Instant::now())).await
        {
            tracing::warn!(?error, "session placement capacity cleanup task failed");
        }
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

    fn next_attempt(&self) -> PlacementAttemptId {
        PlacementAttemptId(self.inner.next_attempt.fetch_add(1, Ordering::Relaxed))
    }

    fn initialization_deadline(&self, now: Instant) -> Option<Instant> {
        self.inner
            .config
            .initialization_timeout
            .map(|timeout| now + timeout)
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
    pub(crate) async fn wait_for_reap(&self) {
        self.inner.reaper_completed.notified().await;
    }

    #[cfg(test)]
    pub(crate) async fn stop_reaper_for_test(&self) {
        let reaper = {
            let mut reaper = self.inner.reaper.lock();
            reaper.take()
        };
        if let Some(reaper) = reaper {
            reaper.abort();
            let _ = reaper.await;
        }
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

impl<T> SessionPlacementInner<T>
where
    T: Send + Sync + 'static,
{
    fn reap_expired(&self, now: Instant) {
        let keys: Vec<String> = self
            .entries
            .iter()
            .filter_map(|entry| {
                let is_expired = match entry.value() {
                    PlacementEntry::Reserved { deadline, .. }
                    | PlacementEntry::Dispatching { deadline, .. } => {
                        deadline.as_ref().is_some_and(|deadline| *deadline <= now)
                    }
                    PlacementEntry::Bound {
                        active_leases: 0,
                        idle_deadline,
                        ..
                    } => *idle_deadline <= now,
                    _ => false,
                };
                is_expired.then(|| entry.key().clone())
            })
            .collect();

        for key in keys {
            let mut transitioned_notify = None;
            if let Some(mut entry) = self.entries.get_mut(&key)
                && let PlacementEntry::Dispatching {
                    attempt,
                    candidate,
                    notify,
                    deadline,
                } = entry.value_mut()
                && deadline.as_ref().is_some_and(|deadline| *deadline <= now)
            {
                let attempt = *attempt;
                let candidate = candidate.clone();
                transitioned_notify = Some(notify.clone());
                *entry = PlacementEntry::Ambiguous { attempt, candidate };
            }
            if let Some(notify) = transitioned_notify {
                notify.notify_waiters();
                continue;
            }

            let removed = self.entries.remove_if(&key, |_, entry| {
                matches!(
                    entry,
                    PlacementEntry::Reserved {
                        deadline: Some(deadline),
                        ..
                    } if *deadline <= now
                ) || matches!(
                    entry,
                    PlacementEntry::Bound {
                        active_leases: 0,
                        idle_deadline,
                        ..
                    } if *idle_deadline <= now
                )
            });
            let Some((_key, removed_entry)) = removed else {
                continue;
            };
            self.entry_count.fetch_sub(1, Ordering::Relaxed);
            if let PlacementEntry::Reserved { notify, .. } = &removed_entry {
                notify.notify_waiters();
            }
            drop(removed_entry);
        }
    }
}

pub(crate) enum PlacementAcquire<T> {
    Initialize(PlacementInitialization<T>),
    Bound {
        target: VersionedTarget<T>,
        lease: PlacementLease<T>,
    },
}

pub(crate) struct PlacementInitialization<T> {
    placement: Weak<SessionPlacementInner<T>>,
    key: String,
    attempt: PlacementAttemptId,
    notify: Arc<Notify>,
    active: bool,
}

impl<T> PlacementInitialization<T>
where
    T: Send + Sync + 'static,
{
    pub(crate) fn begin_dispatch(
        mut self,
        target: T,
        generation: TargetGeneration,
    ) -> Result<PlacementDispatch<T>, SessionPlacementError> {
        let Some(inner) = self.placement.upgrade() else {
            return Err(SessionPlacementError::CoordinatorDropped);
        };
        let Some(mut entry) = inner.entries.get_mut(&self.key) else {
            return Err(SessionPlacementError::InitializationCancelled);
        };
        let PlacementEntry::Reserved {
            attempt,
            notify,
            deadline,
        } = entry.value()
        else {
            return Err(SessionPlacementError::InitializationChanged);
        };
        if *attempt != self.attempt {
            return Err(SessionPlacementError::InitializationChanged);
        }
        if deadline
            .as_ref()
            .is_some_and(|deadline| *deadline <= Instant::now())
        {
            drop(entry);
            return Err(SessionPlacementError::InitializationCancelled);
        }
        let notify = notify.clone();
        let deadline = *deadline;
        let candidate = VersionedTarget::new(target, generation);
        *entry = PlacementEntry::Dispatching {
            attempt: self.attempt,
            candidate: candidate.clone(),
            notify,
            deadline,
        };
        drop(entry);
        self.active = false;
        Ok(PlacementDispatch {
            placement: Arc::downgrade(&inner),
            key: self.key.clone(),
            attempt: self.attempt,
            candidate,
            notify: self.notify.clone(),
            active: true,
        })
    }

    pub(crate) fn commit_already_accepted(
        self,
        target: T,
        generation: TargetGeneration,
    ) -> Result<PlacementLease<T>, SessionPlacementError> {
        // Compatibility only: callers must not use this around an in-flight dispatch. New
        // forwarding paths must call `begin_dispatch` before sending and finish with the
        // transport's explicit outcome.
        self.begin_dispatch(target, generation)?
            .finish(PlacementDispatchOutcome::Accepted)?
            .ok_or(SessionPlacementError::InitializationChanged)
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
                PlacementEntry::Reserved { attempt, .. } if *attempt == self.attempt
            )
        });
        if removed.is_some() {
            inner.entry_count.fetch_sub(1, Ordering::Relaxed);
        }
        drop(removed);
        self.notify.notify_waiters();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PlacementDispatchOutcome {
    Accepted,
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "transport outcome mapping is added with the global-router integration"
        )
    )]
    DefinitelyNotAccepted,
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "transport outcome mapping is added with the global-router integration"
        )
    )]
    Ambiguous,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "ambiguous dispatch recovery is consumed by the global-router integration"
    )
)]
pub(crate) enum AmbiguousResolution {
    Accepted,
    DefinitelyNotAccepted,
}

pub(crate) struct PlacementDispatch<T> {
    placement: Weak<SessionPlacementInner<T>>,
    key: String,
    attempt: PlacementAttemptId,
    candidate: VersionedTarget<T>,
    notify: Arc<Notify>,
    active: bool,
}

impl<T> PlacementDispatch<T>
where
    T: Send + Sync + 'static,
{
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "attempt IDs are consumed by the global-router recovery path"
        )
    )]
    pub(crate) fn attempt_id(&self) -> PlacementAttemptId {
        self.attempt
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "the global-router dispatch path reads the reserved target"
        )
    )]
    pub(crate) fn target(&self) -> &VersionedTarget<T> {
        &self.candidate
    }

    pub(crate) fn finish(
        mut self,
        outcome: PlacementDispatchOutcome,
    ) -> Result<Option<PlacementLease<T>>, SessionPlacementError> {
        let Some(inner) = self.placement.upgrade() else {
            return Err(SessionPlacementError::CoordinatorDropped);
        };
        match outcome {
            PlacementDispatchOutcome::Accepted => {
                let Some(mut entry) = inner.entries.get_mut(&self.key) else {
                    return Err(SessionPlacementError::InitializationCancelled);
                };
                let notify = match entry.value() {
                    PlacementEntry::Dispatching {
                        attempt,
                        candidate,
                        notify,
                        ..
                    } => {
                        validate_attempt(
                            *attempt,
                            candidate,
                            self.attempt,
                            self.candidate.generation(),
                        )?;
                        Some(notify.clone())
                    }
                    PlacementEntry::Ambiguous { attempt, candidate } => {
                        validate_attempt(
                            *attempt,
                            candidate,
                            self.attempt,
                            self.candidate.generation(),
                        )?;
                        None
                    }
                    _ => return Err(SessionPlacementError::InitializationChanged),
                };
                *entry = PlacementEntry::Bound {
                    target: self.candidate.clone(),
                    revision: self.attempt,
                    active_leases: 1,
                    idle_deadline: Instant::now() + inner.config.idle_ttl,
                };
                drop(entry);
                self.active = false;
                if let Some(notify) = notify {
                    notify.notify_waiters();
                }
                Ok(Some(PlacementLease {
                    placement: Arc::downgrade(&inner),
                    key: self.key.clone(),
                    revision: self.attempt,
                    active: true,
                }))
            }
            PlacementDispatchOutcome::DefinitelyNotAccepted => {
                let removed = inner.entries.remove_if(&self.key, |_, entry| {
                    matches!(
                        entry,
                        PlacementEntry::Dispatching {
                            attempt,
                            candidate,
                            ..
                        } | PlacementEntry::Ambiguous {
                            attempt,
                            candidate,
                        } if *attempt == self.attempt && candidate.same_as(&self.candidate)
                    )
                });
                if removed.is_none() {
                    return Err(SessionPlacementError::InitializationChanged);
                }
                inner.entry_count.fetch_sub(1, Ordering::Relaxed);
                let notify = removed.as_ref().and_then(|(_, entry)| match entry {
                    PlacementEntry::Dispatching { notify, .. } => Some(notify.clone()),
                    _ => None,
                });
                drop(removed);
                self.active = false;
                if let Some(notify) = notify {
                    notify.notify_waiters();
                }
                Ok(None)
            }
            PlacementDispatchOutcome::Ambiguous => {
                self.mark_ambiguous(&inner)?;
                self.active = false;
                Ok(None)
            }
        }
    }

    fn mark_ambiguous(
        &self,
        inner: &Arc<SessionPlacementInner<T>>,
    ) -> Result<(), SessionPlacementError> {
        let Some(mut entry) = inner.entries.get_mut(&self.key) else {
            return Err(SessionPlacementError::InitializationCancelled);
        };
        match entry.value() {
            PlacementEntry::Dispatching {
                attempt, candidate, ..
            } => validate_attempt(
                *attempt,
                candidate,
                self.attempt,
                self.candidate.generation(),
            )?,
            PlacementEntry::Ambiguous { attempt, candidate } => {
                return validate_attempt(
                    *attempt,
                    candidate,
                    self.attempt,
                    self.candidate.generation(),
                );
            }
            _ => return Err(SessionPlacementError::InitializationChanged),
        }
        *entry = PlacementEntry::Ambiguous {
            attempt: self.attempt,
            candidate: self.candidate.clone(),
        };
        drop(entry);
        self.notify.notify_waiters();
        Ok(())
    }
}

impl<T> Drop for PlacementDispatch<T> {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let Some(inner) = self.placement.upgrade() else {
            return;
        };
        let Some(mut entry) = inner.entries.get_mut(&self.key) else {
            return;
        };
        let PlacementEntry::Dispatching {
            attempt, candidate, ..
        } = entry.value()
        else {
            return;
        };
        if *attempt != self.attempt || !candidate.same_as(&self.candidate) {
            return;
        }
        *entry = PlacementEntry::Ambiguous {
            attempt: self.attempt,
            candidate: self.candidate.clone(),
        };
        drop(entry);
        self.notify.notify_waiters();
    }
}

pub(crate) struct PlacementLease<T> {
    placement: Weak<SessionPlacementInner<T>>,
    key: String,
    revision: PlacementAttemptId,
    active: bool,
}

impl<T> PlacementLease<T> {
    pub(crate) fn invalidate(&mut self) {
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
        drop(removed);
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

fn validate_attempt<T>(
    current_attempt: PlacementAttemptId,
    current_target: &VersionedTarget<T>,
    expected_attempt: PlacementAttemptId,
    expected_generation: TargetGeneration,
) -> Result<(), SessionPlacementError> {
    if current_attempt != expected_attempt {
        return Err(SessionPlacementError::InitializationChanged);
    }
    let actual_generation = current_target.generation();
    if actual_generation != expected_generation {
        return Err(SessionPlacementError::TargetGenerationChanged {
            expected_generation: expected_generation.get(),
            actual_generation: actual_generation.get(),
        });
    }
    Ok(())
}

fn dispatch_ambiguous(
    attempt: PlacementAttemptId,
    generation: TargetGeneration,
) -> SessionPlacementError {
    SessionPlacementError::DispatchAmbiguous {
        attempt_id: attempt.get(),
        target_generation: generation.get(),
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
