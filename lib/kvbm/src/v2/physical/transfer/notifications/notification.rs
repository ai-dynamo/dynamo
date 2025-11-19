// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transfer completion notification handle.

use anyhow::Result;
use dynamo_nova::events::LocalEventWaiter;
use futures::future::{Either, Ready, ready};

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
    awaiter: Either<Ready<Result<()>>, LocalEventWaiter>,
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
            awaiter: Either::Right(awaiter),
        }
    }
}

impl std::future::IntoFuture for TransferCompleteNotification {
    type Output = Result<()>;
    type IntoFuture = Either<Ready<Result<()>>, LocalEventWaiter>;

    fn into_future(self) -> Self::IntoFuture {
        self.awaiter
    }
}
