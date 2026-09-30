// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod nats_server;
pub mod push_endpoint;
pub mod push_handler;
pub mod shared_tcp_endpoint;
pub mod unified_server;

use super::*;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::sync::Notify;

/// Wait for inflight requests to drain, bounded by `timeout`. Returns the count
/// still inflight when the wait ends: `0` on a clean drain, `>0` if the timeout
/// fired (a stuck request must not wedge teardown). Shared by the NATS push and
/// TCP request planes.
async fn drain_inflight(
    inflight: Arc<AtomicU64>,
    notify: Arc<Notify>,
    endpoint_name: &str,
    timeout: Duration,
) -> u64 {
    let inflight_count = inflight.load(Ordering::SeqCst);
    if inflight_count == 0 {
        return 0;
    }

    tracing::info!(
        endpoint_name,
        inflight_count,
        "Waiting for inflight requests to complete"
    );

    let wait = async {
        while inflight.load(Ordering::SeqCst) > 0 {
            notify.notified().await;
        }
    };

    match tokio::time::timeout(timeout, wait).await {
        Ok(()) => {
            tracing::info!(endpoint_name, "All inflight requests completed");
            0
        }
        Err(_) => {
            let remaining = inflight.load(Ordering::SeqCst);
            tracing::warn!(
                endpoint_name,
                timeout_secs = timeout.as_secs(),
                remaining,
                "Timed out waiting for inflight requests to drain; proceeding with shutdown"
            );
            remaining
        }
    }
}

#[cfg(test)]
mod drain_tests {
    use super::*;

    /// Drain returns within the bound even if the request never completes.
    #[tokio::test(start_paused = true)]
    async fn drain_inflight_is_bounded_when_request_never_completes() {
        let inflight = Arc::new(AtomicU64::new(1));
        let notify = Arc::new(Notify::new());

        let remaining = tokio::time::timeout(
            Duration::from_secs(3600),
            drain_inflight(inflight, notify, "test-endpoint", Duration::from_secs(5)),
        )
        .await
        .expect("drain_inflight must be bounded; it hung past the outer guard");

        assert_eq!(
            remaining, 1,
            "the stuck inflight request should still be counted as remaining"
        );
    }

    /// A clean drain (request completes) returns 0 promptly.
    #[tokio::test(start_paused = true)]
    async fn drain_inflight_returns_zero_when_requests_complete() {
        let inflight = Arc::new(AtomicU64::new(1));
        let notify = Arc::new(Notify::new());

        let inflight_clone = inflight.clone();
        let notify_clone = notify.clone();
        tokio::spawn(async move {
            inflight_clone.fetch_sub(1, Ordering::SeqCst);
            notify_clone.notify_one();
        });

        let remaining =
            drain_inflight(inflight, notify, "test-endpoint", Duration::from_secs(5)).await;

        assert_eq!(remaining, 0, "a completed request should drain cleanly");
    }
}
