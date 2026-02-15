// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use parking_lot::Mutex as ParkingMutex;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::task::Waker;

use super::completion::CompletionKind;
use super::waiter::EventAwaiter;

#[derive(Clone)]
pub(crate) struct ActiveSlot {
    state: Arc<ActiveSlotState>,
}

pub(crate) struct ActiveSlotState {
    // Combine completion and wakers under a single lock to prevent lost wakeups
    pub(crate) inner: ParkingMutex<SlotStateInner>,
    pub(crate) completed: AtomicBool,
    pub(crate) generation: AtomicU64,
    pub(crate) waiter_count: AtomicU32,
}

pub(crate) struct SlotStateInner {
    pub(crate) completion: Option<Arc<CompletionKind>>,
    pub(crate) wakers: Vec<Waker>,
}

impl ActiveSlot {
    pub(crate) fn new() -> Self {
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

    pub(crate) fn waiter(&self) -> EventAwaiter {
        let observed_generation = self.state.generation.load(Ordering::Acquire);
        EventAwaiter::pending(Arc::clone(&self.state), observed_generation)
    }

    pub(crate) fn begin_generation(&self) -> u64 {
        self.state.begin_generation()
    }

    pub(crate) fn complete(&self, value: Arc<CompletionKind>, generation: u64) {
        self.state.complete(value, generation);
    }

    pub(crate) fn complete_triggered(&self, generation: u64) {
        self.state.complete(Self::triggered_arc(), generation);
    }

    fn triggered_arc() -> Arc<CompletionKind> {
        static TRIGGERED: OnceLock<Arc<CompletionKind>> = OnceLock::new();
        Arc::clone(TRIGGERED.get_or_init(|| Arc::new(CompletionKind::Triggered)))
    }
}

impl ActiveSlotState {
    pub(crate) fn begin_generation(&self) -> u64 {
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

    pub(crate) fn complete(&self, value: Arc<CompletionKind>, generation: u64) {
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
