// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Remote event state management for subscriber-side tracking.

use parking_lot::Mutex;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use velo_events::{EventAwaiter, EventHandle, EventManager, EventPoison, EventStatus};

/// Identifies a remote event by its (system_id, local_index) pair.
/// Used as DashMap key to group generations of the same remote event.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct RemoteEventKey {
    pub system_id: u64,
    pub local_index: u32,
}

impl RemoteEventKey {
    pub fn from_handle(handle: EventHandle) -> Self {
        Self {
            system_id: handle.system_id(),
            local_index: handle.local_index(),
        }
    }
}

/// Tracks whether an event completed via trigger or poison.
#[derive(Clone, Debug)]
pub(crate) enum CompletionKind {
    Triggered,
    Poisoned(Arc<EventPoison>),
}

/// Cached information for completed remote events stored in the LRU cache.
pub(crate) struct CompletedEventInfo {
    pub highest_generation: u32,
    pub poisoned_generations: BTreeSet<u32>,
}

/// Per-remote-handle state on the subscriber side.
///
/// Creates proxy local events that waiters can subscribe to. When completion
/// notifications arrive from the remote owner, the proxy events are completed.
pub(crate) struct RemoteEvent {
    /// Event manager for creating and completing proxy events
    proxy_manager: EventManager,
    /// Fast-path: highest generation known to be complete
    known_triggered: AtomicU32,
    /// Active proxy event handles mapped by generation
    proxy_handles: Mutex<BTreeMap<u32, EventHandle>>,
    /// Completion history for resolved generations
    completions: Mutex<BTreeMap<u32, Arc<CompletionKind>>>,
    /// Deduplication: generations for which we've sent a subscription request
    pending: Mutex<BTreeSet<u32>>,
}

pub(crate) enum WaitRegistration {
    Ready,
    Pending(EventAwaiter),
    Poisoned(Arc<EventPoison>),
}

impl RemoteEvent {
    pub fn new(proxy_manager: EventManager) -> Self {
        Self {
            proxy_manager,
            known_triggered: AtomicU32::new(0),
            proxy_handles: Mutex::new(BTreeMap::new()),
            completions: Mutex::new(BTreeMap::new()),
            pending: Mutex::new(BTreeSet::new()),
        }
    }

    pub fn known_generation(&self) -> u32 {
        self.known_triggered.load(Ordering::Acquire)
    }

    fn update_known_generation(&self, generation: u32) {
        let _ = self
            .known_triggered
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                (generation > current).then_some(generation)
            });
    }

    pub fn register_waiter(&self, generation: u32) -> WaitRegistration {
        // Fast path: already complete
        if generation <= self.known_generation() {
            let completions = self.completions.lock();
            return match completions.get(&generation) {
                Some(completion) => match completion.as_ref() {
                    CompletionKind::Triggered => WaitRegistration::Ready,
                    CompletionKind::Poisoned(p) => WaitRegistration::Poisoned(p.clone()),
                },
                None => WaitRegistration::Ready,
            };
        }

        // Create or reuse a proxy event for this generation
        let proxy_handle = {
            let mut proxy_handles = self.proxy_handles.lock();
            *proxy_handles.entry(generation).or_insert_with(|| {
                let event = self
                    .proxy_manager
                    .new_event()
                    .expect("Failed to create proxy event for remote subscription");
                event.into_handle()
            })
        };

        // Create an awaiter on the proxy event
        match self.proxy_manager.awaiter(proxy_handle) {
            Ok(waiter) => WaitRegistration::Pending(waiter),
            Err(_) => WaitRegistration::Ready,
        }
    }

    pub fn add_pending(&self, generation: u32) -> bool {
        self.pending.lock().insert(generation)
    }

    pub fn complete_generation(&self, generation: u32, completion: CompletionKind) {
        self.update_known_generation(generation);

        // Store completion with bounded history
        {
            let mut completions = self.completions.lock();
            completions.insert(generation, Arc::new(completion.clone()));

            const MAX_COMPLETION_HISTORY: u32 = 100;
            let known_gen = self.known_generation();
            if known_gen > MAX_COMPLETION_HISTORY {
                completions.retain(|&g, _| g > known_gen - MAX_COMPLETION_HISTORY);
            }
        }

        // Trigger or poison all proxy events for generations <= this one
        let handles_to_complete = {
            let mut proxy_handles = self.proxy_handles.lock();
            let gens_to_wake: Vec<u32> = proxy_handles
                .range(..=generation)
                .map(|(g, _)| *g)
                .collect();

            let mut handles = Vec::new();
            for generation in gens_to_wake {
                if let Some(handle) = proxy_handles.remove(&generation) {
                    handles.push(handle);
                }
            }
            handles
        };

        // Complete proxy events outside lock
        for handle in handles_to_complete {
            match &completion {
                CompletionKind::Triggered => {
                    let _ = self.proxy_manager.trigger(handle);
                }
                CompletionKind::Poisoned(p) => {
                    let _ = self.proxy_manager.poison(handle, p.reason());
                }
            }
        }

        // Clean up pending
        let mut pending = self.pending.lock();
        let to_remove: Vec<u32> = pending
            .iter()
            .copied()
            .take_while(|g| *g <= generation)
            .collect();
        for g in to_remove {
            pending.remove(&g);
        }
    }

    pub fn status_for(&self, generation: u32) -> EventStatus {
        if generation <= self.known_generation() {
            let completions = self.completions.lock();
            match completions.get(&generation) {
                Some(completion) => match completion.as_ref() {
                    CompletionKind::Triggered => EventStatus::Ready,
                    CompletionKind::Poisoned(_) => EventStatus::Poisoned,
                },
                None => EventStatus::Ready,
            }
        } else {
            EventStatus::Pending
        }
    }

    pub fn is_cacheable(&self) -> bool {
        let pending = self.pending.lock();
        let proxy_handles = self.proxy_handles.lock();
        pending.is_empty() && proxy_handles.is_empty()
    }
}
