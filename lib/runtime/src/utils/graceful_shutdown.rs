// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::Notify;

/// Tracks graceful shutdown state for endpoints
pub struct GracefulShutdownTracker {
    active_endpoints: AtomicUsize,
    shutdown_complete: Notify,
}

impl std::fmt::Debug for GracefulShutdownTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GracefulShutdownTracker")
            .field(
                "active_endpoints",
                &self.active_endpoints.load(Ordering::SeqCst),
            )
            .finish()
    }
}

impl GracefulShutdownTracker {
    pub(crate) fn new() -> Self {
        Self {
            active_endpoints: AtomicUsize::new(0),
            shutdown_complete: Notify::new(),
        }
    }

    pub(crate) fn register_endpoint(&self) {
        let count = self.active_endpoints.fetch_add(1, Ordering::SeqCst);
        tracing::debug!(
            "Endpoint registered, total active: {} -> {}",
            count,
            count + 1
        );
    }

    pub(crate) fn unregister_endpoint(&self) {
        let prev = self.active_endpoints.fetch_sub(1, Ordering::SeqCst);
        tracing::debug!(
            "Endpoint unregistered, remaining active: {} -> {}",
            prev,
            prev - 1
        );
        if prev == 1 {
            // Last endpoint completed
            tracing::info!("Last endpoint completed, notifying all waiters");
            self.shutdown_complete.notify_waiters();
        }
    }

    /// Get the current count of active endpoints
    pub(crate) fn get_count(&self) -> usize {
        self.active_endpoints.load(Ordering::Acquire)
    }

    pub(crate) async fn wait_for_completion(&self) {
        loop {
            // Create the waiter BEFORE checking the condition
            let notified = self.shutdown_complete.notified();

            let count = self.active_endpoints.load(Ordering::SeqCst);
            tracing::trace!("Checking completion status, active endpoints: {}", count);

            if count == 0 {
                tracing::debug!("All endpoints completed");
                break;
            }

            // Only wait if there are still active endpoints
            tracing::debug!("Waiting for {} endpoints to complete", count);
            notified.await;
            tracing::trace!("Received notification, rechecking...");
        }
    }

    // This method is no longer needed since we can access the tracker directly
}
