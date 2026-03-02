// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use super::completion::CompletionKind;
use super::entry::EventEntry;
use crate::status::Generation;

/// Future that waits for an event to complete.
///
/// This can be used in `tokio::select!` and polled multiple times efficiently.
/// Waker deduplication inside the entry lock prevents unbounded growth.
pub struct EventAwaiter {
    entry: Option<Arc<EventEntry>>,
    observed_generation: Generation,
    immediate_result: Option<Arc<CompletionKind>>,
}

impl EventAwaiter {
    /// Creates a waiter that immediately resolves with the given result.
    #[allow(private_interfaces)]
    pub(crate) fn immediate(result: Arc<CompletionKind>) -> Self {
        Self {
            entry: None,
            observed_generation: 0,
            immediate_result: Some(result),
        }
    }

    /// Creates a waiter that will poll the entry for completion.
    #[allow(private_interfaces)]
    pub(crate) fn pending(entry: Arc<EventEntry>, generation: Generation) -> Self {
        Self {
            entry: Some(entry),
            observed_generation: generation,
            immediate_result: None,
        }
    }
}

impl Future for EventAwaiter {
    type Output = anyhow::Result<()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();

        // Fast path: immediate result (already-completed event)
        if let Some(result) = &this.immediate_result {
            return Poll::Ready(result.as_ref().as_result().map_err(anyhow::Error::new));
        }

        let entry = this
            .entry
            .as_ref()
            .expect("EventAwaiter with no entry or immediate_result");

        entry.poll_waiter(this.observed_generation, cx)
    }
}
