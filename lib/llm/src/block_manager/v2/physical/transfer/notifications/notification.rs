// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transfer completion notification handle.

use anyhow::Result;
use tokio::sync::oneshot;

/// Notification handle for an in-progress transfer.
///
/// This object can be awaited to block until the transfer completes.
/// The transfer is tracked by a background handler that polls for completion
/// or processes notification events.
pub struct TransferCompleteNotification {
    pub(crate) status: oneshot::Receiver<Result<()>>,
}

impl TransferCompleteNotification {
    /// Create a notification that is already completed (for synchronous transfers).
    ///
    /// This is useful for transfers that complete immediately without needing
    /// background polling, such as memcpy operations.
    pub fn completed() -> Self {
        let (tx, rx) = oneshot::channel();
        // Signal completion immediately
        let _ = tx.send(Ok(()));
        Self { status: rx }
    }

    /// Wait for the transfer to complete (blocking).
    ///
    /// This method blocks the current thread until the transfer completes.
    /// Use `.await` for async contexts.
    ///
    /// Returns `Ok(())` when the transfer successfully completes, or an error
    /// if the background handler was dropped before completion or if the transfer failed.
    pub fn wait(self) -> Result<()> {
        self.status
            .blocking_recv()
            .map_err(|_| anyhow::anyhow!("Transfer handler dropped before completion"))?
    }
}

impl std::future::Future for TransferCompleteNotification {
    type Output = Result<()>;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        use std::pin::Pin;
        Pin::new(&mut self.status).poll(cx).map(|result| {
            result
                .map_err(|_| anyhow::anyhow!("Transfer handler dropped before completion"))
                .and_then(|r| r)
        })
    }
}
