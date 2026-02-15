// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use parking_lot::Mutex as ParkingMutex;
use std::collections::BTreeMap;
use std::fmt::{self, Display, Formatter};
use std::sync::Arc;

use super::active::ActiveSlot;
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

pub(crate) type EventEntryResult<T> = std::result::Result<T, EventEntryError>;

/// Owner-side event entry reused across generations.
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

    pub(crate) fn begin_generation(&self) -> EventEntryResult<Generation> {
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

    pub(crate) fn finalize_completion(
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
            poisoned @ CompletionKind::Poisoned(_) => {
                let completion_arc = Arc::new(poisoned);
                slot.complete(completion_arc, slot_gen);
            }
        }

        Ok(())
    }

    pub(crate) fn retire(&self) {
        let mut state = self.state.lock();
        state.retired = true;
        state.active_generation = None;
        state.active_slot = None;
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

#[cfg(test)]
mod tests {
    use super::*;

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
