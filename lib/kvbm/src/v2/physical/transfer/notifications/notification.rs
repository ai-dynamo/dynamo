// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transfer completion notification handle.

use anyhow::Result;
use dynamo_nova::{am::SyncResult, events::LocalEventWaiter};
use futures::future::{Either, Ready, ready};
use std::{
    pin::Pin,
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
}

impl std::future::IntoFuture for TransferCompleteNotification {
    type Output = Result<()>;
    type IntoFuture = Either<Ready<Result<()>>, TransferAwaiter>;

    fn into_future(self) -> Self::IntoFuture {
        self.awaiter
    }
}
