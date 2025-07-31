// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # HTTP Service Request Throttling Service
//!
//! This module provides adaptive request throttling for HTTP services based on backend worker capacity.
//! It integrates with the KV router scheduler to automatically throttle incoming requests when
//! all workers are busy, providing system-wide back-pressure and preventing cascade failures.
//!
//! ## Architecture Overview
//!
//! The throttling service operates using an event-driven architecture:
//!
//! ```text
//! ┌─────────────────┐    KVAllWorkersBusyEvent    ┌─────────────────────┐
//! │   KV Scheduler  │ ──────────────────────────► │  Throttling Service │
//! │                 │                             │                     │
//! │ - Monitors      │                             │ - Subscribes to     │
//! │   worker queue  │                             │   busy events       │
//! │   depths        │                             │ - Sets temporary    │
//! │ - Publishes     │                             │   throttling        │
//! │   busy events   │                             │ - Auto-expires      │
//! └─────────────────┘                             └─────────────────────┘
//!                                                            │
//!                                                            ▼
//! ┌─────────────────┐    is_throttled()            ┌────────────────────┐
//! │ HTTP Service    │ ◄─────────────────────────── │   HTTP Handlers    │
//! │                 │                              │                    │
//! │ - Checks        │                              │ - Check throttling │
//! │   throttling    │                              │   before processing│
//! │   status        │                              │ - Reject with      │
//! │ - Returns 503   │                              │   503 if throttled │
//! │   if throttled  │                              │                    │
//! └─────────────────┘                              └────────────────────┘
//! ```
//!
//! ## Key Components
//!
//! ### ThrottlingState
//! Thread-safe state management for throttling with automatic expiration:
//! - Tracks when throttling was activated
//! - Configurable throttling duration
//! - Automatic expiration based on elapsed time
//!
//! ### HttpServiceRequestThrottler
//! Main throttling service that integrates with the event system:
//! - Subscribes to `KV_ALL_WORKERS_BUSY_SUBJECT` events from the scheduler
//! - Manages request throttling state transitions
//! - Provides thread-safe access for HTTP handlers
//!
//! ## Integration with KV Scheduler
//!
//! The throttling service responds to events from [`crate::kv_router::scheduler::KvScheduler`]:
//!
//! 1. **Event Trigger**: When the scheduler's request queue exceeds the configured threshold,
//!    it publishes a [`KVAllWorkersBusyEvent`]
//!
//! 2. **Throttling Activation**: The throttling service receives this event and immediately
//!    activates throttling for the configured duration
//!
//! 3. **Request Rejection**: HTTP handlers check [`HttpServiceRequestThrottler::is_throttled()`]
//!    and reject new requests with HTTP 429 (Too Many Requests)
//!
//! 4. **Automatic Recovery**: Request throttling automatically expires after the configured
//!    duration, allowing normal request processing to resume
//!
//! ## Configuration
//!
//! Request throttling behavior is controlled by:
//!
//! - **Enable/Disable**: Pass `Some(duration)` to enable, `None` to disable
//! - **Request Throttle Duration**: How long to reject requests after receiving a busy event
//! - **Queue Threshold**: Configured in the scheduler to determine when to trigger events

use dynamo_runtime::{component::Namespace, traits::events::EventSubscriber, CancellationToken};
use futures::StreamExt;

use crate::kv_router::{scheduler::KVAllWorkersBusyEvent, KV_ALL_WORKERS_BUSY_SUBJECT};
use anyhow::Result;
use std::{
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};

/// Request throttling state that can be shared across request handlers
#[derive(Debug, Clone)]
pub struct ThrottlingState {
    rate_limit_start: Option<Instant>,
    rate_limit_duration: Duration,
}

impl Default for ThrottlingState {
    fn default() -> Self {
        Self {
            rate_limit_start: None,
            rate_limit_duration: Duration::from_secs(5),
        }
    }
}

impl ThrottlingState {
    pub fn new(rate_limit_duration: Duration) -> Self {
        ThrottlingState {
            rate_limit_duration,
            ..Self::default()
        }
    }

    /// Check if state is currently request throttled
    pub fn is_throttled(&self) -> bool {
        match self.rate_limit_start {
            Some(start_time) => start_time.elapsed() < self.rate_limit_duration,
            None => false,
        }
    }

    /// Set request throttling state to true and record the start time
    pub fn set_throttling(&mut self) {
        let now = Instant::now();
        self.rate_limit_start = Some(now);
    }

    /// Set request throttling state to false and clear the start time
    pub fn clear_throttling(&mut self) {
        self.rate_limit_start = None;
    }
}

#[derive(Clone)]
pub struct HttpServiceRequestThrottler {
    is_enabled: bool,
    rate_limit_state: Arc<RwLock<ThrottlingState>>,
}

impl HttpServiceRequestThrottler {
    pub fn new(all_workers_busy_rejection_time_window: Option<Duration>) -> Self {
        Self {
            is_enabled: all_workers_busy_rejection_time_window.is_some(),
            rate_limit_state: Arc::new(RwLock::new(
                all_workers_busy_rejection_time_window
                    .map(ThrottlingState::new)
                    .unwrap_or_default(),
            )),
        }
    }

    /// Check if the current underlying state is throttled
    pub fn is_throttled(&self) -> bool {
        if !self.is_enabled {
            return false;
        }

        let state = self.rate_limit_state.read().unwrap();
        state.is_throttled()
    }

    /// Start monitoring the request throttling state
    ///
    /// This function will spawn a new task that will monitor the throttling state and set the throttling state to true if the all workers busy event is received.
    /// It will also clear the throttling state after the throttling duration has passed.
    pub fn start_monitoring(
        &self,
        namespace: Namespace,
        cancellation_token: CancellationToken,
    ) -> Result<()> {
        if !self.is_enabled {
            tracing::warn!("Throttling service is disabled, skipping monitoring");
            return Ok(());
        }

        let rate_limit_duration = { self.rate_limit_state.read().unwrap().rate_limit_duration };
        let rate_limiter = self.rate_limit_state.clone();

        tokio::spawn(async move {
            let mut all_workers_busy_event_rx = match namespace
                .subscribe(KV_ALL_WORKERS_BUSY_SUBJECT)
                .await
            {
                Ok(rx) => rx,
                Err(e) => {
                    tracing::error!("We can't monitor the request throttling state, and therefore this will be disabled. Failed to subscribe to all workers busy event: {}.", e);
                    return;
                }
            };

            let sleep = tokio::time::sleep(Duration::MAX);
            tokio::pin!(sleep);

            loop {
                tokio::select! {
                    Some(msg) = all_workers_busy_event_rx.next() => {
                        let kv_all_workers_busy_event: KVAllWorkersBusyEvent = match serde_json::from_slice(&msg.payload) {
                            Ok(event) => event,
                            Err(e) => {
                                tracing::error!("Failed to deserialize all workers busy event: {}", e);
                                continue;
                            }
                        };
                        let timestamp = kv_all_workers_busy_event.timestamp;
                        tracing::info!(
                            "Received all workers busy event with queue_depth={} at timestamp={}",
                            kv_all_workers_busy_event.max_queue_depth,
                            timestamp
                        );

                        {
                            let mut state = rate_limiter.write().unwrap();
                            state.set_throttling();
                        };

                        sleep.as_mut().reset(tokio::time::Instant::now() + rate_limit_duration);
                    }
                    _ = &mut sleep => {
                        {
                            let mut state = rate_limiter.write().unwrap();
                            state.clear_throttling();
                        }
                        sleep.as_mut().reset(tokio::time::Instant::now() + Duration::MAX);
                    }
                    _ = cancellation_token.cancelled() => {
                        tracing::debug!("Throttling service monitoring task shutting down");
                        break;
                    }
                }
            }
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::time::sleep;

    #[test]
    fn test_request_throttling_state_default() {
        let state = ThrottlingState::default();
        assert_eq!(state.rate_limit_start, None);
        assert_eq!(state.rate_limit_duration, Duration::from_secs(5));
    }

    #[test]
    fn test_request_throttling_state_new() {
        let duration = Duration::from_secs(10);
        let state = ThrottlingState::new(duration);
        assert_eq!(state.rate_limit_start, None);
        assert_eq!(state.rate_limit_duration, duration);
    }

    #[test]
    fn test_request_throttling_state_is_request_throttled_when_not_set() {
        let state = ThrottlingState::default();
        assert!(!state.is_throttled());
    }

    #[test]
    fn test_request_throttling_state_is_request_throttled_when_set() {
        let mut state = ThrottlingState::default();
        state.set_throttling();
        assert!(state.is_throttled());
    }

    #[tokio::test]
    async fn test_request_throttling_state_is_request_throttled_after_expiry() {
        let mut state = ThrottlingState::new(Duration::from_millis(10));
        state.set_throttling();
        assert!(state.is_throttled());

        // Wait for the request throttling to expire
        sleep(Duration::from_millis(21)).await;
        assert!(!state.is_throttled());
    }

    #[test]
    fn test_request_throttling_state_clear_request_throttling() {
        let mut state = ThrottlingState::default();
        state.set_throttling();
        assert!(state.is_throttled());

        state.clear_throttling();
        assert!(!state.is_throttled());
        assert_eq!(state.rate_limit_start, None);
    }

    #[test]
    fn test_http_service_throttling_service_new_enabled() {
        let duration = Duration::from_secs(10);
        let throttling_service = HttpServiceRequestThrottler::new(Some(duration));
        assert!(throttling_service.is_enabled);
        assert_eq!(
            throttling_service
                .rate_limit_state
                .read()
                .unwrap()
                .rate_limit_duration,
            duration
        );
    }

    #[test]
    fn test_http_service_throttling_service_new_disabled() {
        let throttling_service = HttpServiceRequestThrottler::new(None);
        assert!(!throttling_service.is_enabled);
        assert_eq!(
            throttling_service
                .rate_limit_state
                .read()
                .unwrap()
                .rate_limit_duration,
            Duration::from_secs(5)
        );
    }

    #[test]
    fn test_http_service_request_throttler_is_not_request_throttled_when_disabled() {
        let rate_limiter = HttpServiceRequestThrottler::new(None);
        assert!(!rate_limiter.is_throttled());
    }

    #[test]
    fn test_http_service_request_throttler_is_not_request_throttled_when_enabled_and_not_set() {
        let rate_limiter = HttpServiceRequestThrottler::new(Some(Duration::from_secs(5)));
        assert!(!rate_limiter.is_throttled());
    }

    #[test]
    fn test_http_service_request_throttler_is_request_throttled_when_enabled_and_set() {
        let rate_limiter = HttpServiceRequestThrottler::new(Some(Duration::from_secs(5)));
        {
            let mut state = rate_limiter.rate_limit_state.write().unwrap();
            state.set_throttling();
        }
        assert!(rate_limiter.is_throttled());
    }

    #[tokio::test]
    async fn test_http_service_request_throttler_is_request_throttled_after_expiry() {
        let rate_limiter = HttpServiceRequestThrottler::new(Some(Duration::from_millis(10)));
        {
            let mut state = rate_limiter.rate_limit_state.write().unwrap();
            state.set_throttling();
        }
        assert!(rate_limiter.is_throttled());

        // Wait for the request throttling to expire
        sleep(Duration::from_millis(21)).await;
        assert!(!rate_limiter.is_throttled());
    }

    #[tokio::test]
    async fn test_http_service_updates_request_throttling_duration() {
        let rate_limiter = HttpServiceRequestThrottler::new(Some(Duration::from_secs(1)));
        {
            let mut state = rate_limiter.rate_limit_state.write().unwrap();
            state.set_throttling();
        }
        assert!(rate_limiter.is_throttled());

        sleep(Duration::from_millis(550)).await;

        {
            let mut state = rate_limiter.rate_limit_state.write().unwrap();
            state.set_throttling();
        }
        assert!(rate_limiter.is_throttled());

        sleep(Duration::from_millis(550)).await;

        assert!(rate_limiter.is_throttled());
    }
}
