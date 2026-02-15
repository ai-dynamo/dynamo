// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::anyhow;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::task::{Context, Poll};

use super::active::ActiveSlotState;
use super::completion::CompletionKind;
use crate::handle::EventHandle;
use crate::status::EventPoison;

/// Future that waits for an event to complete.
///
/// This can be used in `tokio::select!` and polled multiple times efficiently.
/// The waiter creates a fresh notification registration on each poll to ensure
/// proper wakeup semantics.
pub struct EventAwaiter {
    state: Option<Arc<ActiveSlotState>>,
    observed_generation: u64,
    immediate_result: Option<Arc<CompletionKind>>,
}

impl EventAwaiter {
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

impl Future for EventAwaiter {
    type Output = anyhow::Result<()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self;

        // Check for immediate result first
        if let Some(result) = &this.immediate_result {
            return Poll::Ready(result.as_ref().as_result().map_err(anyhow::Error::new));
        }

        let state = this
            .state
            .as_ref()
            .expect("EventAwaiter with no state or immediate_result");

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

impl Drop for EventAwaiter {
    fn drop(&mut self) {
        if let Some(state) = &self.state {
            state.waiter_count.fetch_sub(1, Ordering::AcqRel);
        }
    }
}
