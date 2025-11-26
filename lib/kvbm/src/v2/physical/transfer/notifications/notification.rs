// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transfer completion notification handle.

use anyhow::Result;
use dynamo_nova::{
    am::SyncResult,
    events::{LocalEvent, LocalEventSystem, LocalEventWaiter},
};
use futures::future::{Either, Ready, ready};
use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

pub enum TransferAwaiter {
    Local(LocalEventWaiter),
    Sync(SyncResult),
}

impl std::future::Future for TransferAwaiter {
    type Output = Result<()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.get_mut() {
            Self::Local(waiter) => Pin::new(waiter).poll(cx),
            Self::Sync(sync) => Pin::new(sync).poll(cx),
        }
    }
}

/// Notification handle for an in-progress transfer.
///
/// This object can be awaited to block until the transfer completes.
/// The transfer is tracked by a background handler that polls for completion
/// or processes notification events.
///
/// Uses `futures::Either` to avoid event system overhead for synchronous completions.
/// Pending transfers use `LocalEventWaiter` which avoids heap allocation and repeated
/// DashMap lookups when awaiting.
pub struct TransferCompleteNotification {
    awaiter: Either<Ready<Result<()>>, TransferAwaiter>,
}

impl TransferCompleteNotification {
    /// Create a notification that is already completed (for synchronous transfers).
    ///
    /// This is useful for transfers that complete immediately without needing
    /// background polling, such as memcpy operations.
    ///
    /// This is extremely efficient - no allocations, locks, or event system overhead.
    pub fn completed() -> Self {
        Self {
            awaiter: Either::Left(ready(Ok(()))),
        }
    }

    /// Create a notification from a `LocalEventWaiter`.
    ///
    /// This is the primary way to construct a notification when you already
    /// have an event waiter from the event system.
    pub fn from_awaiter(awaiter: LocalEventWaiter) -> Self {
        Self {
            awaiter: Either::Right(TransferAwaiter::Local(awaiter)),
        }
    }

    /// Create a notification from a synchronous active message result.
    pub fn from_sync_result(sync: SyncResult) -> Self {
        Self {
            awaiter: Either::Right(TransferAwaiter::Sync(sync)),
        }
    }

    /// Check if the notification can yield the current task.
    ///
    /// The internal ::Left arm is guaranteed to be ready, while the ::Right arm is not.
    pub fn could_yield(&self) -> bool {
        matches!(self.awaiter, Either::Right(_))
    }

    /// Aggregate multiple notifications into one that completes when all are done.
    ///
    /// This is useful when a transfer is split across multiple workers and you want
    /// to wait for all of them to complete.
    ///
    /// # Arguments
    /// * `notifications` - The notifications to aggregate
    /// * `events` - The event system to create the aggregate event
    /// * `runtime` - The tokio runtime handle to spawn the aggregation task
    ///
    /// # Behavior
    /// - If the list is empty, returns an already-completed notification
    /// - If there's only one, returns it directly
    /// - Otherwise, creates a new event and spawns a task to await all notifications
    pub fn aggregate(
        notifications: Vec<Self>,
        events: &Arc<LocalEventSystem>,
        runtime: &tokio::runtime::Handle,
    ) -> Result<Self> {
        if notifications.is_empty() {
            return Ok(Self::completed());
        }
        if notifications.len() == 1 {
            return Ok(notifications.into_iter().next().unwrap());
        }

        // Check if all notifications are already complete (no yielding needed)
        if notifications.iter().all(|n| !n.could_yield()) {
            return Ok(Self::completed());
        }

        // Create a new event for the aggregate completion
        let event = events.new_event()?;
        let awaiter = events.awaiter(event.handle())?;

        // Spawn task that awaits all notifications and triggers/poisons the event
        runtime.spawn(await_all_notifications(notifications, event));

        Ok(Self::from_awaiter(awaiter))
    }
}

/// Awaits all transfer notifications and signals completion via the event.
///
/// This function awaits ALL notifications regardless of individual failures,
/// then triggers the event on success or poisons it with error details on failure.
async fn await_all_notifications(
    notifications: Vec<TransferCompleteNotification>,
    local_event: LocalEvent,
) {
    // Await all notifications, collecting results
    let results: Vec<Result<()>> =
        futures::future::join_all(notifications.into_iter().map(|n| n.into_future())).await;

    // Check for any failures
    let errors: Vec<_> = results.into_iter().filter_map(|r| r.err()).collect();

    if errors.is_empty() {
        // Ignore trigger error - if event system is shutdown, nothing to do
        let _ = local_event.trigger();
    } else {
        let error_msg = errors
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("; ");
        // Ignore poison error - if event system is shutdown, nothing to do
        let _ = local_event.poison(error_msg);
    }
}

impl std::future::IntoFuture for TransferCompleteNotification {
    type Output = Result<()>;
    type IntoFuture = Either<Ready<Result<()>>, TransferAwaiter>;

    fn into_future(self) -> Self::IntoFuture {
        self.awaiter
    }
}
