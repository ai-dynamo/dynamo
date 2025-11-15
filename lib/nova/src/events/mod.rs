// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generational event system for coordinating local awaiters with minimal overhead.

mod local;
pub use local::{LocalEvent, LocalEventSystem};

use anyhow::{Result, anyhow, bail};
use dashmap::DashMap;
use parking_lot::Mutex as ParkingMutex;
use std::collections::{BTreeMap, VecDeque};
use std::fmt::{self, Display, Formatter};
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use tokio::sync::Notify;
use tokio_util::task::TaskTracker;
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
struct EventKey {
    worker: u64,
    index: u32,
}

impl EventKey {
    fn new(worker: u64, index: u32) -> Self {
        Self { worker, index }
    }

    fn from_handle(handle: EventHandle) -> Self {
        Self {
            worker: handle.owner_worker(),
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
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
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

    pub fn owner_worker(&self) -> u64 {
        ((self.0 & WORKER_MASK) >> WORKER_SHIFT) as u64
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
        EventHandle::new(self.owner_worker(), self.local_index(), generation)
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
enum CompletionKind {
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

struct ActiveSlotState {
    notify: Notify,
    completion: ParkingMutex<Option<Arc<CompletionKind>>>,
    completed: AtomicBool,
    generation: AtomicU64,
}

impl ActiveSlot {
    fn new() -> Self {
        Self {
            state: Arc::new(ActiveSlotState {
                notify: Notify::new(),
                completion: ParkingMutex::new(None),
                completed: AtomicBool::new(false),
                generation: AtomicU64::new(0),
            }),
        }
    }

    fn waiter(&self) -> LocalWaiter {
        LocalWaiter {
            state: Arc::clone(&self.state),
            observed_generation: self.state.generation.load(Ordering::Acquire),
        }
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
    fn clone_completion(&self) -> Option<Arc<CompletionKind>> {
        let guard = self.completion.lock();
        guard.as_ref().cloned()
    }

    fn begin_generation(&self) -> u64 {
        let next = self.generation.fetch_add(1, Ordering::AcqRel) + 1;
        self.completed.store(false, Ordering::Release);
        {
            let mut guard = self.completion.lock();
            guard.take();
        }
        next
    }

    fn complete(&self, value: Arc<CompletionKind>, generation: u64) {
        let current = self.generation.load(Ordering::Acquire);
        if current != generation {
            return;
        }
        if self.completed.swap(true, Ordering::SeqCst) {
            return;
        }
        {
            let mut guard = self.completion.lock();
            *guard = Some(value);
        }
        self.notify.notify_waiters();
    }
}

#[derive(Clone)]
struct LocalWaiter {
    state: Arc<ActiveSlotState>,
    observed_generation: u64,
}

impl LocalWaiter {
    async fn wait(self) -> Arc<CompletionKind> {
        loop {
            let current = self.state.generation.load(Ordering::Acquire);
            if current != self.observed_generation {
                // stale waiter; treat as completion missed
                if let Some(value) = self.state.clone_completion() {
                    return value;
                }
            } else if let Some(value) = self.state.clone_completion() {
                return value;
            }
            self.state.notify.notified().await;
        }
    }
}

enum WaitRegistration {
    Ready,
    Pending(LocalWaiter),
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
            tokio::spawn(async move { system.wait(handle).await })
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
        system.wait(handle).await?;
        Ok(())
    }

    #[tokio::test]
    async fn poison_is_visible() -> Result<()> {
        let system = create_system();
        let event = system.new_event()?;
        let handle = event.handle();

        system.poison(handle, "boom")?;
        let err = system.wait(handle).await.unwrap_err();
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
        system.wait(handle).await?;

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
            waiters.push(tokio::spawn(async move { system_clone.wait(handle).await }));
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

        system.wait(merged).await?;
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

        let err = system.wait(merged).await.unwrap_err();
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
            tokio::spawn(async move { system.wait(handle).await })
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

        let err = system.wait(event.handle()).await.unwrap_err();
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
