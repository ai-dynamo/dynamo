// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generational event system for coordinating local awaiters with minimal overhead.

mod local;
use dynamo_identity::WorkerId;
pub use local::{LocalEvent, LocalEventSystem};

use anyhow::{Result, anyhow, bail};
use dashmap::DashMap;
use parking_lot::Mutex as ParkingMutex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
use std::fmt::{self, Display, Formatter};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::task::{Context, Poll, Waker};
use tracing::{error, trace};

/// Alias for event generation counters.
pub type Generation = u32;

const WORKER_BITS: u32 = 64;
const LOCAL_BITS: u32 = 32;
const GENERATION_BITS: u32 = 32;

const LOCAL_SHIFT: u32 = GENERATION_BITS;
const WORKER_SHIFT: u32 = LOCAL_SHIFT + LOCAL_BITS;

const WORKER_MASK: u128 = ((1u128 << WORKER_BITS) - 1) << WORKER_SHIFT;
const LOCAL_MASK: u128 = ((1u128 << LOCAL_BITS) - 1) << LOCAL_SHIFT;
const GENERATION_MASK: u128 = (1u128 << GENERATION_BITS) - 1;
const MAX_GENERATION: Generation = ((1u64 << GENERATION_BITS) - 1) as Generation;
const MAX_LOCAL_INDEX: u32 = u32::MAX;

pub type EventManager = Arc<LocalEventSystem>;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct EventKey {
    worker: u64,
    index: u32,
}

impl EventKey {
    pub(crate) fn new(worker: u64, index: u32) -> Self {
        Self { worker, index }
    }

    pub(crate) fn from_handle(handle: EventHandle) -> Self {
        Self {
            worker: handle.owner_worker().as_u64(),
            index: handle.local_index(),
        }
    }

    fn handle(&self, generation: Generation) -> Result<EventHandle> {
        EventHandle::new(self.worker, self.index, generation)
    }
}

impl Display for EventKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "EventKey(owner={}, local={})", self.worker, self.index)
    }
}

/// Public event handle encoded in a single u128 value.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EventHandle(u128);

impl EventHandle {
    pub fn new(owner_worker: u64, local_index: u32, generation: Generation) -> Result<Self> {
        // Note: MAX_LOCAL_INDEX is u32::MAX and MAX_GENERATION is u32::MAX
        // so overflow checks are not needed for these u32 parameters

        let raw = ((owner_worker as u128) << WORKER_SHIFT)
            | ((local_index as u128) << LOCAL_SHIFT)
            | (generation as u128);
        Ok(Self(raw))
    }

    pub fn from_raw(raw: u128) -> Self {
        Self(raw)
    }

    pub fn raw(&self) -> u128 {
        self.0
    }

    pub fn as_u128(&self) -> u128 {
        self.0
    }

    pub fn owner_worker(&self) -> WorkerId {
        WorkerId::from_u64(((self.0 & WORKER_MASK) >> WORKER_SHIFT) as u64)
    }

    pub fn local_index(&self) -> u32 {
        ((self.0 & LOCAL_MASK) >> LOCAL_SHIFT) as u32
    }

    pub fn generation(&self) -> Generation {
        (self.0 & GENERATION_MASK) as Generation
    }

    fn key(&self) -> EventKey {
        EventKey::from_handle(*self)
    }

    pub fn with_generation(&self, generation: Generation) -> Result<Self> {
        EventHandle::new(self.owner_worker().as_u64(), self.local_index(), generation)
    }
}

impl Display for EventHandle {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EventHandle(owner={}, local={}, gen={})",
            self.owner_worker(),
            self.local_index(),
            self.generation()
        )
    }
}

/// Status returned from non-blocking event queries.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EventStatus {
    Pending,
    Ready,
    Poisoned,
}

/// Describes a poisoned event generation.
#[derive(Clone, Debug)]
pub struct EventPoison {
    handle: EventHandle,
    reason: Arc<str>,
}

impl EventPoison {
    pub fn new(handle: EventHandle, reason: impl Into<String>) -> Self {
        Self {
            handle,
            reason: Arc::<str>::from(reason.into()),
        }
    }

    pub fn from_shared(handle: EventHandle, reason: Arc<str>) -> Self {
        Self { handle, reason }
    }

    pub fn handle(&self) -> EventHandle {
        self.handle
    }

    pub fn reason(&self) -> &str {
        &self.reason
    }
}

impl Display for EventPoison {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Event {} poisoned: {}", self.handle, self.reason())
    }
}

impl std::error::Error for EventPoison {}

type PoisonArc = Arc<EventPoison>;

#[derive(Clone, Debug)]
pub(crate) enum CompletionKind {
    Triggered,
    Poisoned(PoisonArc),
}

impl CompletionKind {
    fn as_result(&self) -> Result<(), EventPoison> {
        match self {
            Self::Triggered => Ok(()),
            Self::Poisoned(poison) => Err((**poison).clone()),
        }
    }
}
#[derive(Clone)]
struct ActiveSlot {
    state: Arc<ActiveSlotState>,
}

pub(crate) struct ActiveSlotState {
    // Combine completion and wakers under a single lock to prevent lost wakeups
    inner: ParkingMutex<SlotStateInner>,
    completed: AtomicBool,
    generation: AtomicU64,
    waiter_count: AtomicU32,
}

struct SlotStateInner {
    completion: Option<Arc<CompletionKind>>,
    wakers: Vec<Waker>,
}

impl ActiveSlot {
    fn new() -> Self {
        Self {
            state: Arc::new(ActiveSlotState {
                inner: ParkingMutex::new(SlotStateInner {
                    completion: None,
                    wakers: Vec::with_capacity(2), // Optimize for common case of 1-2 waiters
                }),
                completed: AtomicBool::new(false),
                generation: AtomicU64::new(0),
                waiter_count: AtomicU32::new(0),
            }),
        }
    }

    fn waiter(&self) -> LocalEventWaiter {
        let observed_generation = self.state.generation.load(Ordering::Acquire);
        LocalEventWaiter::pending(Arc::clone(&self.state), observed_generation)
    }

    fn begin_generation(&self) -> u64 {
        self.state.begin_generation()
    }

    fn complete(&self, value: Arc<CompletionKind>, generation: u64) {
        self.state.complete(value, generation);
    }

    fn complete_triggered(&self, generation: u64) {
        self.state.complete(Self::triggered_arc(), generation);
    }

    fn triggered_arc() -> Arc<CompletionKind> {
        static TRIGGERED: OnceLock<Arc<CompletionKind>> = OnceLock::new();
        Arc::clone(TRIGGERED.get_or_init(|| Arc::new(CompletionKind::Triggered)))
    }
}

impl ActiveSlotState {
    fn begin_generation(&self) -> u64 {
        let next = self.generation.fetch_add(1, Ordering::AcqRel) + 1;
        self.completed.store(false, Ordering::Release);

        // Only clear completion if no waiters are using it
        if self.waiter_count.load(Ordering::Acquire) == 0 {
            let mut guard = self.inner.lock();
            guard.completion = None;
            // Retain capacity for next generation, but clear items
            guard.wakers.clear();
        }

        next
    }

    fn complete(&self, value: Arc<CompletionKind>, generation: u64) {
        let current = self.generation.load(Ordering::Acquire);
        if current != generation {
            return;
        }
        if self.completed.swap(true, Ordering::AcqRel) {
            return;
        }

        let wakers = {
            let mut guard = self.inner.lock();
            guard.completion = Some(value);
            // Drain wakers to notify them outside the lock
            std::mem::take(&mut guard.wakers)
        };

        for waker in wakers {
            waker.wake();
        }
    }
}

/// Future that waits for an event to complete.
///
/// This can be used in `tokio::select!` and polled multiple times efficiently.
/// The waiter creates a fresh notification registration on each poll to ensure
/// proper wakeup semantics.
pub struct LocalEventWaiter {
    state: Option<Arc<ActiveSlotState>>,
    observed_generation: u64,
    immediate_result: Option<Arc<CompletionKind>>,
}

impl LocalEventWaiter {
    /// Creates a waiter that immediately resolves with the given result.
    #[allow(private_interfaces)]
    pub(crate) fn immediate(result: Arc<CompletionKind>) -> Self {
        Self {
            state: None,
            observed_generation: 0,
            immediate_result: Some(result),
        }
    }

    /// Creates a waiter that will wait for completion from the active slot.
    #[allow(private_interfaces)]
    pub(crate) fn pending(state: Arc<ActiveSlotState>, observed_generation: u64) -> Self {
        // Increment waiter count to prevent completion from being cleared
        state.waiter_count.fetch_add(1, Ordering::AcqRel);
        Self {
            state: Some(state),
            observed_generation,
            immediate_result: None,
        }
    }
}

impl Future for LocalEventWaiter {
    type Output = Result<()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self;

        // Check for immediate result first
        if let Some(result) = &this.immediate_result {
            return Poll::Ready(result.as_ref().as_result().map_err(anyhow::Error::new));
        }

        let state = this
            .state
            .as_ref()
            .expect("LocalEventWaiter with no state or immediate_result");

        // Acquire lock to check completion and register waker atomically
        let mut inner = state.inner.lock();
        let current = state.generation.load(Ordering::Acquire);

        // 1. Check generation freshness
        if current != this.observed_generation {
            if let Some(value) = &inner.completion {
                return Poll::Ready(value.as_ref().as_result().map_err(anyhow::Error::new));
            }
            return Poll::Ready(Err(anyhow!(EventPoison::new(
                EventHandle::from_raw(0),
                format!(
                    "generation expired: observed {}, current {}",
                    this.observed_generation, current
                ),
            ))));
        }

        // 2. Check for completion
        if let Some(value) = &inner.completion {
            return Poll::Ready(value.as_ref().as_result().map_err(anyhow::Error::new));
        }

        // 3. Register waker with deduplication
        // This is critical for performance in select! loops
        let waker = cx.waker();
        if let Some(existing) = inner.wakers.iter_mut().find(|w| w.will_wake(waker)) {
            // Update existing waker in case the task moved to a different thread
            existing.clone_from(waker);
        } else {
            inner.wakers.push(waker.clone());
        }

        Poll::Pending
    }
}

impl Drop for LocalEventWaiter {
    fn drop(&mut self) {
        if let Some(state) = &self.state {
            state.waiter_count.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

enum WaitRegistration {
    Ready,
    Pending(LocalEventWaiter),
    Poisoned(PoisonArc),
}

#[derive(Debug)]
enum EventEntryError {
    ActiveGeneration {
        key: EventKey,
        active: Generation,
    },
    GenerationOverflow {
        key: EventKey,
    },
    InvalidGeneration {
        key: EventKey,
        requested: Generation,
        active: Option<Generation>,
    },
    MissingSlot {
        key: EventKey,
        generation: Generation,
    },
}

impl Display for EventEntryError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::ActiveGeneration { key, active } => {
                write!(f, "Event {} already has active generation {}", key, active)
            }
            Self::GenerationOverflow { key } => {
                write!(
                    f,
                    "Event {} exhausted generation space ({} bits)",
                    key, GENERATION_BITS
                )
            }
            Self::InvalidGeneration {
                key,
                requested,
                active,
            } => match active {
                Some(current) => write!(
                    f,
                    "Invalid generation {} for event {}; active generation {}",
                    requested, key, current
                ),
                None => write!(
                    f,
                    "Invalid generation {} for event {}; no active generation",
                    requested, key
                ),
            },
            Self::MissingSlot { key, generation } => {
                write!(
                    f,
                    "Missing slot for event {} generation {}",
                    key, generation
                )
            }
        }
    }
}

impl std::error::Error for EventEntryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

type EventEntryResult<T> = std::result::Result<T, EventEntryError>;

/// Owner-side event entry reused across generations.
struct EventEntry {
    key: EventKey,
    state: ParkingMutex<EventState>,
}

impl EventEntry {
    fn new(key: EventKey) -> Self {
        Self {
            key,
            state: ParkingMutex::new(EventState::default()),
        }
    }

    fn key(&self) -> EventKey {
        self.key
    }

    fn begin_generation(&self) -> EventEntryResult<Generation> {
        let mut state = self.state.lock();
        if let Some(active) = state.active_generation {
            return Err(EventEntryError::ActiveGeneration {
                key: self.key,
                active,
            });
        }
        if state.last_triggered == MAX_GENERATION || state.retired {
            return Err(EventEntryError::GenerationOverflow { key: self.key });
        }
        let next = state
            .last_triggered
            .checked_add(1)
            .expect("checked for overflow above");
        let slot = state.active_slot.get_or_insert_with(ActiveSlot::new);
        let slot_gen = slot.begin_generation();
        state.slot_generation = slot_gen;
        state.active_generation = Some(next);
        Ok(next)
    }

    fn status_for(&self, generation: Generation) -> EventStatus {
        let state = self.state.lock();
        if generation <= state.last_triggered {
            if state.poisoned.contains_key(&generation) {
                EventStatus::Poisoned
            } else {
                EventStatus::Ready
            }
        } else {
            EventStatus::Pending
        }
    }

    fn register_local_waiter(&self, generation: Generation) -> EventEntryResult<WaitRegistration> {
        let state = self.state.lock();
        if generation <= state.last_triggered {
            if let Some(poison) = state.poisoned.get(&generation) {
                return Ok(WaitRegistration::Poisoned(poison.clone()));
            }
            return Ok(WaitRegistration::Ready);
        }

        match state.active_generation {
            Some(active) if active == generation => {
                let slot = state
                    .active_slot
                    .as_ref()
                    .ok_or(EventEntryError::MissingSlot {
                        key: self.key,
                        generation,
                    })?;
                Ok(WaitRegistration::Pending(slot.waiter()))
            }
            Some(active) => Err(EventEntryError::InvalidGeneration {
                key: self.key,
                requested: generation,
                active: Some(active),
            }),
            None => Err(EventEntryError::InvalidGeneration {
                key: self.key,
                requested: generation,
                active: None,
            }),
        }
    }

    fn finalize_completion(
        &self,
        generation: Generation,
        completion: CompletionKind,
    ) -> EventEntryResult<()> {
        let mut state = self.state.lock();
        if state.active_generation != Some(generation) {
            return Err(EventEntryError::InvalidGeneration {
                key: self.key,
                requested: generation,
                active: state.active_generation,
            });
        }

        let slot = state
            .active_slot
            .as_ref()
            .ok_or(EventEntryError::MissingSlot {
                key: self.key,
                generation,
            })?
            .clone();
        let slot_gen = state.slot_generation;

        state.last_triggered = generation;
        state.active_generation = None;

        match &completion {
            CompletionKind::Poisoned(poison) => {
                state.poisoned.insert(generation, poison.clone());
            }
            CompletionKind::Triggered => {
                state.poisoned.remove(&generation);
            }
        }

        drop(state);

        match completion {
            CompletionKind::Triggered => slot.complete_triggered(slot_gen),
            CompletionKind::Poisoned(_) => {
                let completion_arc = Arc::new(completion.clone());
                slot.complete(completion_arc, slot_gen);
            }
        }

        Ok(())
    }

    fn retire(&self) {
        let mut state = self.state.lock();
        state.retired = true;
        state.active_generation = None;
        state.active_slot = None;
    }

    fn is_retired(&self) -> bool {
        let state = self.state.lock();
        state.retired
    }

    fn active_handle(&self) -> Option<EventHandle> {
        let generation = {
            let state = self.state.lock();
            if state.retired {
                return None;
            }
            state.active_generation
        }?;
        self.key.handle(generation).ok()
    }

    pub(crate) fn poison_reason(&self, generation: Generation) -> Option<Arc<str>> {
        let state = self.state.lock();
        state
            .poisoned
            .get(&generation)
            .map(|p| Arc::<str>::from(p.reason().to_string()))
    }
}

#[derive(Default)]
struct EventState {
    last_triggered: Generation,
    active_generation: Option<Generation>,
    active_slot: Option<ActiveSlot>,
    poisoned: BTreeMap<Generation, PoisonArc>,
    slot_generation: u64,
    retired: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use tokio::task::yield_now;

    const TEST_WORKER: u64 = 0x42;

    fn create_system() -> Arc<LocalEventSystem> {
        LocalEventSystem::new(TEST_WORKER)
    }

    #[tokio::test]
    async fn wait_resolves_after_trigger() -> Result<()> {
        let system = create_system();
        let event = system.new_event()?;
        let handle = event.handle();

        let waiter = {
            let system = Arc::clone(&system);
            tokio::spawn(async move { system.awaiter(handle)?.await })
        };

        yield_now().await;
        event.trigger()?;
        waiter.await??;
        Ok(())
    }

    #[tokio::test]
    async fn wait_ready_if_triggered_first() -> Result<()> {
        let system = create_system();
        let event = system.new_event()?;
        let handle = event.handle();

        event.trigger()?;
        system.awaiter(handle)?.await?;
        Ok(())
    }

    #[tokio::test]
    async fn poison_is_visible() -> Result<()> {
        let system = create_system();
        let event = system.new_event()?;
        let handle = event.handle();

        system.poison(handle, "boom")?;
        let err = system.awaiter(handle)?.await.unwrap_err();
        let poison = err.downcast::<EventPoison>().unwrap();
        assert_eq!(poison.reason(), "boom");
        Ok(())
    }

    #[tokio::test]
    async fn entry_reused_after_completion() -> Result<()> {
        let system = create_system();
        let event = system.new_event()?;
        let handle = event.handle();
        let index = handle.local_index();
        let generation = handle.generation();

        event.trigger()?;
        system.awaiter(handle)?.await?;

        let next = system.new_event()?;
        let next_handle = next.handle();
        assert_eq!(next_handle.local_index(), index);
        assert_eq!(next_handle.generation(), generation + 1);
        Ok(())
    }

    #[tokio::test]
    async fn multiple_waiters_wake() -> Result<()> {
        let system = create_system();
        let event = system.new_event()?;
        let handle = event.handle();

        let mut waiters = Vec::new();
        for _ in 0..8 {
            let system_clone = Arc::clone(&system);
            waiters.push(tokio::spawn(
                async move { system_clone.awaiter(handle)?.await },
            ));
        }

        yield_now().await;
        event.trigger()?;
        for waiter in waiters {
            waiter.await??;
        }
        Ok(())
    }

    #[tokio::test]
    async fn merge_triggers_after_dependencies() -> Result<()> {
        let system = create_system();
        let first = system.new_event()?;
        let second = system.new_event()?;

        let merged = system.merge_events(vec![first.handle(), second.handle()])?;

        first.trigger()?;
        second.trigger()?;

        system.awaiter(merged)?.await?;
        Ok(())
    }

    #[tokio::test]
    async fn merge_poison_accumulates_reasons() -> Result<()> {
        let system = create_system();
        let first = system.new_event()?;
        let second = system.new_event()?;

        let merged = system.merge_events(vec![first.handle(), second.handle()])?;

        system.poison(first.handle(), "first failed")?;
        system.poison(second.handle(), "second failed")?;

        let err = system.awaiter(merged)?.await.unwrap_err();
        let poison = err.downcast::<EventPoison>().unwrap();
        assert!(poison.reason().contains("first failed"));
        assert!(poison.reason().contains("second failed"));
        Ok(())
    }

    #[tokio::test]
    async fn force_shutdown_poison_pending() -> Result<()> {
        let system = create_system();
        let event = system.new_event()?;
        let handle = event.handle();

        let waiter = {
            let system = system.clone();
            tokio::spawn(async move { system.awaiter(handle)?.await })
        };

        yield_now().await;
        system.force_shutdown("shutdown");

        let err = waiter.await.unwrap().unwrap_err();
        let poison = err.downcast::<EventPoison>().unwrap();
        assert_eq!(poison.reason(), "shutdown");
        Ok(())
    }

    #[tokio::test]
    async fn new_event_fails_after_force_shutdown() -> Result<()> {
        let system = create_system();
        let event = system.new_event()?;
        system.force_shutdown("shutdown");

        let err = match system.new_event() {
            Ok(_) => panic!("expected shutdown to block new events"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("shutdown"));

        let err = system.awaiter(event.handle())?.await.unwrap_err();
        assert!(err.downcast::<EventPoison>().is_ok());
        Ok(())
    }

    #[tokio::test]
    async fn force_shutdown_is_idempotent() -> Result<()> {
        let system = create_system();
        let _ = system.new_event()?;
        system.force_shutdown("shutdown");
        system.force_shutdown("shutdown");
        assert!(system.new_event().is_err());
        Ok(())
    }
}
