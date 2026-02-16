// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::anyhow;
use parking_lot::Mutex as ParkingMutex;
use std::collections::BTreeMap;
use std::fmt::{self, Display, Formatter};
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use super::completion::{CompletionKind, PoisonArc, WaitRegistration};
use crate::handle::EventHandle;
use crate::status::{EventStatus, Generation};

const MAX_GENERATION: Generation = Generation::MAX;
const GENERATION_BITS: u32 = 32;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct EventKey {
    index: u32,
}

impl EventKey {
    pub(crate) fn new(index: u32) -> Self {
        Self { index }
    }

    pub(crate) fn from_handle(handle: EventHandle) -> Self {
        Self {
            index: handle.local_index(),
        }
    }

    pub(crate) fn handle(&self, system_id: u64, generation: Generation) -> EventHandle {
        EventHandle::new(system_id, self.index, generation)
    }
}

impl Display for EventKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "EventKey(index={})", self.index)
    }
}

#[derive(Debug)]
pub(crate) enum EventEntryError {
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
    AlreadyCompleted {
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
            Self::AlreadyCompleted { key, generation } => {
                write!(
                    f,
                    "Event {} generation {} already completed successfully",
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

pub(crate) type EventEntryResult<T> = std::result::Result<T, EventEntryError>;

/// Outcome of an atomic try-to-poison operation.
#[derive(Debug)]
pub(crate) enum PoisonOutcome {
    /// Successfully poisoned. Caller must recycle the entry.
    Poisoned,
    /// Already poisoned (idempotent success). No recycling needed.
    AlreadyPoisoned,
}

/// Owner-side event entry reused across generations.
///
/// All state mutations are serialized through a single `ParkingMutex<EventState>`,
/// eliminating the stale-completion race present in the original `slot` module.
pub(crate) struct EventEntry {
    key: EventKey,
    state: ParkingMutex<EventState>,
}

impl EventEntry {
    pub(crate) fn new(key: EventKey) -> Self {
        Self {
            key,
            state: ParkingMutex::new(EventState::default()),
        }
    }

    pub(crate) fn key(&self) -> EventKey {
        self.key
    }

    /// Advance to the next generation.
    ///
    /// Flushes any stale wakers from the previous generation so they re-poll
    /// and resolve via the `observed_generation <= last_triggered` check.
    pub(crate) fn begin_generation(&self) -> EventEntryResult<Generation> {
        let stale_wakers;
        let next;
        {
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
            next = state
                .last_triggered
                .checked_add(1)
                .expect("checked for overflow above");

            // Flush stale wakers from the previous generation.
            stale_wakers = std::mem::take(&mut state.wakers);

            state.active_generation = Some(next);
        }

        // Wake stale wakers outside lock to reduce contention.
        for waker in stale_wakers {
            waker.wake();
        }

        Ok(next)
    }

    pub(crate) fn status_for(&self, generation: Generation) -> EventStatus {
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

    pub(crate) fn register_local_waiter(
        &self,
        generation: Generation,
    ) -> EventEntryResult<WaitRegistration> {
        let state = self.state.lock();
        if generation <= state.last_triggered {
            if let Some(poison) = state.poisoned.get(&generation) {
                return Ok(WaitRegistration::Poisoned(poison.clone()));
            }
            return Ok(WaitRegistration::Ready);
        }

        match state.active_generation {
            Some(active) if active == generation => Ok(WaitRegistration::Pending),
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

    /// Complete the current generation with the given result.
    ///
    /// Stores poison history (if applicable) and wakes all registered waiters.
    /// Both the state update and waker drain happen under the same lock
    /// acquisition, preventing the stale-completion race (Race 1) and the
    /// drop-then-signal fragility (Race 2) present in the original `slot` module.
    pub(crate) fn finalize_completion(
        &self,
        generation: Generation,
        completion: CompletionKind,
    ) -> EventEntryResult<()> {
        let wakers;
        {
            let mut state = self.state.lock();
            if state.active_generation != Some(generation) {
                return Err(EventEntryError::InvalidGeneration {
                    key: self.key,
                    requested: generation,
                    active: state.active_generation,
                });
            }

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

            wakers = std::mem::take(&mut state.wakers);
        }

        // Wake all registered waiters outside the lock.
        for waker in wakers {
            waker.wake();
        }

        Ok(())
    }

    /// Atomically attempt to poison the given generation.
    ///
    /// Holds the entry lock across both the status check and the state
    /// transition, eliminating the TOCTOU window present when `status_for`
    /// and `finalize_completion` are called separately.
    pub(crate) fn try_to_poison(
        &self,
        generation: Generation,
        poison: PoisonArc,
    ) -> EventEntryResult<PoisonOutcome> {
        let wakers;
        {
            let mut state = self.state.lock();

            if generation <= state.last_triggered {
                return if state.poisoned.contains_key(&generation) {
                    Ok(PoisonOutcome::AlreadyPoisoned)
                } else {
                    Err(EventEntryError::AlreadyCompleted {
                        key: self.key,
                        generation,
                    })
                };
            }

            if state.active_generation != Some(generation) {
                return Err(EventEntryError::InvalidGeneration {
                    key: self.key,
                    requested: generation,
                    active: state.active_generation,
                });
            }

            // Transition to poisoned (same mutations as finalize_completion)
            state.last_triggered = generation;
            state.active_generation = None;
            state.poisoned.insert(generation, poison);
            wakers = std::mem::take(&mut state.wakers);
        }

        for waker in wakers {
            waker.wake();
        }

        Ok(PoisonOutcome::Poisoned)
    }

    /// Poll for waiter resolution, called by [`super::waiter::EventAwaiter::poll`].
    ///
    /// Checks the entry state under lock and either returns a result or
    /// registers the provided waker for future notification.
    pub(crate) fn poll_waiter(
        &self,
        observed_generation: Generation,
        cx: &mut Context<'_>,
    ) -> Poll<anyhow::Result<()>> {
        let mut state = self.state.lock();

        // Check if our generation has completed.
        if observed_generation <= state.last_triggered {
            if let Some(poison) = state.poisoned.get(&observed_generation) {
                return Poll::Ready(Err(anyhow::Error::new((**poison).clone())));
            }
            return Poll::Ready(Ok(()));
        }

        // Generation not yet completed — check if still active.
        if state.active_generation.is_none() {
            return Poll::Ready(Err(anyhow!("generation expired without completion")));
        }

        // Register waker with deduplication (critical for select! loops).
        let waker = cx.waker();
        if let Some(existing) = state.wakers.iter_mut().find(|w| w.will_wake(waker)) {
            existing.clone_from(waker);
        } else {
            state.wakers.push(waker.clone());
        }

        Poll::Pending
    }

    pub(crate) fn retire(&self) {
        let wakers;
        {
            let mut state = self.state.lock();
            debug_assert!(
                state.wakers.is_empty(),
                "retire() called with {} registered wakers on {:?}",
                state.wakers.len(),
                self.key,
            );
            state.retired = true;
            state.active_generation = None;
            wakers = std::mem::take(&mut state.wakers);
        }
        for waker in wakers {
            waker.wake();
        }
    }

    pub(crate) fn is_retired(&self) -> bool {
        let state = self.state.lock();
        state.retired
    }

    pub(crate) fn active_handle(&self, system_id: u64) -> Option<EventHandle> {
        let generation = {
            let state = self.state.lock();
            if state.retired {
                return None;
            }
            state.active_generation
        }?;
        Some(self.key.handle(system_id, generation))
    }

    #[allow(dead_code)]
    pub(crate) fn poison_reason(&self, generation: Generation) -> Option<Arc<str>> {
        let state = self.state.lock();
        state
            .poisoned
            .get(&generation)
            .map(|p| Arc::<str>::from(p.reason().to_string()))
    }
}

/// Per-entry state protected by a single mutex.
///
/// All fields are read and written under the same lock, which structurally
/// prevents the races present in the original two-lock (`EventState` +
/// `SlotStateInner`) design.
struct EventState {
    /// Highest generation that has completed (triggered or poisoned).
    last_triggered: Generation,
    /// Currently pending generation, if any.
    active_generation: Option<Generation>,
    /// Registered wakers from pending `EventAwaiter` futures.
    wakers: Vec<Waker>,
    /// Poison history keyed by generation.
    poisoned: BTreeMap<Generation, PoisonArc>,
    /// Permanently unusable (generation space exhausted).
    retired: bool,
}

impl Default for EventState {
    fn default() -> Self {
        Self {
            last_triggered: 0,
            active_generation: None,
            wakers: Vec::with_capacity(2),
            poisoned: BTreeMap::new(),
            retired: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Wake, Waker};

    fn make_entry(index: u32) -> EventEntry {
        EventEntry::new(EventKey::new(index))
    }

    #[test]
    fn entry_error_active_generation() {
        let entry = make_entry(0);
        entry.begin_generation().unwrap(); // generation 1 now active
        let err = entry.begin_generation().unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("already has active generation"));
    }

    #[test]
    fn entry_error_generation_overflow() {
        let entry = make_entry(1);
        entry.retire();
        let err = entry.begin_generation().unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("exhausted generation space"));
    }

    #[test]
    fn entry_error_invalid_generation_waiter() {
        let entry = make_entry(2);
        let generation = entry.begin_generation().unwrap();
        // Request a waiter for a generation that doesn't match
        match entry.register_local_waiter(generation + 99) {
            Err(err) => {
                let msg = format!("{}", err);
                assert!(msg.contains("Invalid generation"));
                assert!(msg.contains("active generation"));
            }
            Ok(_) => panic!("expected InvalidGeneration error"),
        }
    }

    #[test]
    fn entry_error_invalid_generation_no_active() {
        let entry = make_entry(3);
        // No active generation at all
        match entry.register_local_waiter(1) {
            Err(err) => {
                let msg = format!("{}", err);
                assert!(msg.contains("Invalid generation"));
                assert!(msg.contains("no active generation"));
            }
            Ok(_) => panic!("expected InvalidGeneration error"),
        }
    }

    #[test]
    fn entry_key_display() {
        let key = EventKey::new(42);
        let display = format!("{}", key);
        assert!(display.contains("EventKey"));
        assert!(display.contains("42"));
    }

    #[test]
    fn entry_active_handle_when_retired() {
        let entry = make_entry(4);
        entry.begin_generation().unwrap();
        entry.retire();
        assert!(entry.active_handle(0).is_none());
        assert!(entry.is_retired());
    }

    #[test]
    fn entry_active_handle_when_active() {
        let entry = make_entry(5);
        let generation = entry.begin_generation().unwrap();
        let handle = entry.active_handle(0);
        assert!(handle.is_some());
        assert_eq!(handle.unwrap().generation(), generation);
    }

    #[test]
    fn entry_error_source() {
        let entry = make_entry(6);
        entry.begin_generation().unwrap();
        let err = entry.begin_generation().unwrap_err();
        assert!(std::error::Error::source(&err).is_none());
    }

    #[test]
    fn entry_status_for_pending_and_ready() {
        let entry = make_entry(7);
        let generation = entry.begin_generation().unwrap();
        assert_eq!(entry.status_for(generation), EventStatus::Pending);

        // Trigger it
        entry
            .finalize_completion(generation, CompletionKind::Triggered)
            .unwrap();
        assert_eq!(entry.status_for(generation), EventStatus::Ready);

        // Past generations are Ready
        assert_eq!(entry.status_for(0), EventStatus::Ready);
    }

    #[test]
    fn entry_status_for_poisoned() {
        let entry = make_entry(8);
        let generation = entry.begin_generation().unwrap();
        let handle = entry.key().handle(0, generation);
        let poison = Arc::new(crate::status::EventPoison::new(handle, "test"));
        entry
            .finalize_completion(generation, CompletionKind::Poisoned(poison))
            .unwrap();
        assert_eq!(entry.status_for(generation), EventStatus::Poisoned);
    }

    #[test]
    fn entry_poison_reason() {
        let entry = make_entry(9);
        let generation = entry.begin_generation().unwrap();
        let handle = entry.key().handle(0, generation);
        let poison = Arc::new(crate::status::EventPoison::new(handle, "oops"));
        entry
            .finalize_completion(generation, CompletionKind::Poisoned(poison))
            .unwrap();
        let reason = entry.poison_reason(generation);
        assert_eq!(&*reason.unwrap(), "oops");
    }

    #[derive(Default)]
    struct CountingWake {
        count: AtomicUsize,
    }

    impl Wake for CountingWake {
        fn wake(self: Arc<Self>) {
            self.count.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn counting_waker() -> (Arc<CountingWake>, Waker) {
        let state = Arc::new(CountingWake::default());
        let waker = Waker::from(Arc::clone(&state));
        (state, waker)
    }

    #[test]
    fn poll_waiter_deduplicates_waker_registrations() {
        let entry = make_entry(10);
        let generation = entry.begin_generation().unwrap();

        let (wake_state, waker) = counting_waker();
        let mut cx = Context::from_waker(&waker);

        assert!(entry.poll_waiter(generation, &mut cx).is_pending());
        assert!(entry.poll_waiter(generation, &mut cx).is_pending());

        {
            let state = entry.state.lock();
            assert_eq!(state.wakers.len(), 1, "waker should be deduplicated");
        }

        entry
            .finalize_completion(generation, CompletionKind::Triggered)
            .unwrap();

        assert_eq!(wake_state.count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn finalize_completion_wakes_all_distinct_waiters() {
        let entry = make_entry(11);
        let generation = entry.begin_generation().unwrap();

        let (first_state, first_waker) = counting_waker();
        let (second_state, second_waker) = counting_waker();

        let mut first_cx = Context::from_waker(&first_waker);
        let mut second_cx = Context::from_waker(&second_waker);

        assert!(entry.poll_waiter(generation, &mut first_cx).is_pending());
        assert!(entry.poll_waiter(generation, &mut second_cx).is_pending());

        entry
            .finalize_completion(generation, CompletionKind::Triggered)
            .unwrap();

        assert_eq!(first_state.count.load(Ordering::SeqCst), 1);
        assert_eq!(second_state.count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn begin_generation_flushes_stale_wakers() {
        let entry = make_entry(12);
        let generation = entry.begin_generation().unwrap();

        entry
            .finalize_completion(generation, CompletionKind::Triggered)
            .unwrap();

        let (wake_state, stale_waker) = counting_waker();
        {
            let mut state = entry.state.lock();
            state.wakers.push(stale_waker);
        }

        let next_generation = entry.begin_generation().unwrap();
        assert_eq!(next_generation, generation + 1);
        assert_eq!(wake_state.count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn try_to_poison_pending_succeeds() {
        let entry = make_entry(14);
        let generation = entry.begin_generation().unwrap();
        let handle = entry.key().handle(0, generation);
        let poison = Arc::new(crate::status::EventPoison::new(handle, "boom"));
        match entry.try_to_poison(generation, poison).unwrap() {
            PoisonOutcome::Poisoned => {}
            PoisonOutcome::AlreadyPoisoned => panic!("expected Poisoned"),
        }
        assert_eq!(entry.status_for(generation), EventStatus::Poisoned);
    }

    #[test]
    fn try_to_poison_already_poisoned_is_idempotent() {
        let entry = make_entry(15);
        let generation = entry.begin_generation().unwrap();
        let handle = entry.key().handle(0, generation);
        let poison = Arc::new(crate::status::EventPoison::new(handle, "first"));
        match entry.try_to_poison(generation, poison).unwrap() {
            PoisonOutcome::Poisoned => {}
            PoisonOutcome::AlreadyPoisoned => panic!("expected Poisoned on first call"),
        }
        let poison2 = Arc::new(crate::status::EventPoison::new(handle, "second"));
        match entry.try_to_poison(generation, poison2).unwrap() {
            PoisonOutcome::AlreadyPoisoned => {}
            PoisonOutcome::Poisoned => panic!("expected AlreadyPoisoned on second call"),
        }
    }

    #[test]
    fn try_to_poison_already_triggered_returns_error() {
        let entry = make_entry(16);
        let generation = entry.begin_generation().unwrap();
        entry
            .finalize_completion(generation, CompletionKind::Triggered)
            .unwrap();
        let handle = entry.key().handle(0, generation);
        let poison = Arc::new(crate::status::EventPoison::new(handle, "too late"));
        let err = entry.try_to_poison(generation, poison).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("already completed successfully"), "got: {msg}");
    }

    #[test]
    fn try_to_poison_invalid_generation() {
        let entry = make_entry(17);
        let _generation = entry.begin_generation().unwrap();
        let handle = entry.key().handle(0, 999);
        let poison = Arc::new(crate::status::EventPoison::new(handle, "wrong gen"));
        let err = entry.try_to_poison(999, poison).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("Invalid generation"), "got: {msg}");
    }

    #[test]
    fn try_to_poison_wakes_waiters() {
        let entry = make_entry(18);
        let generation = entry.begin_generation().unwrap();

        let (wake_state, waker) = counting_waker();
        let mut cx = Context::from_waker(&waker);
        assert!(entry.poll_waiter(generation, &mut cx).is_pending());

        let handle = entry.key().handle(0, generation);
        let poison = Arc::new(crate::status::EventPoison::new(handle, "wake test"));
        entry.try_to_poison(generation, poison).unwrap();

        assert_eq!(wake_state.count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn retire_wakes_registered_wakers() {
        let entry = make_entry(13);
        let generation = entry.begin_generation().unwrap();

        let (wake_state, waker) = counting_waker();
        let mut cx = Context::from_waker(&waker);

        // Register a waker by polling the pending generation.
        assert!(entry.poll_waiter(generation, &mut cx).is_pending());

        // Retire the entry — in debug builds the debug_assert fires (catching
        // the invariant violation), in release builds the wakers are defensively
        // drained and woken.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            entry.retire();
        }));

        if cfg!(debug_assertions) {
            assert!(result.is_err(), "debug_assert should fire when wakers are registered");
        } else {
            result.expect("retire() should not panic in release");
            assert_eq!(wake_state.count.load(Ordering::SeqCst), 1);
            assert!(entry.is_retired());
        }
    }
}
