// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # HTTP Service Rate Limiter
//!
//! This module provides adaptive rate limiting for HTTP services based on backend worker capacity.
//! It integrates with the KV router scheduler to automatically throttle incoming requests when
//! all workers are busy, providing system-wide back-pressure and preventing cascade failures.
//!
//! ## Architecture Overview
//!
//! The rate limiter operates using an event-driven architecture:
//!
//! ```text
//! ┌─────────────────┐    KVAllWorkersBusyEvent    ┌──────────────────┐
//! │   KV Scheduler  │ ──────────────────────────► │  Rate Limiter    │
//! │                 │                             │                  │
//! │ - Monitors      │                             │ - Subscribes to  │
//! │   worker queue  │                             │   busy events    │
//! │   depths        │                             │ - Sets temporary │
//! │ - Publishes     │                             │   rate limit     │
//! │   busy events   │                             │ - Auto-expires   │
//! └─────────────────┘                             └──────────────────┘
//!                                                            │
//!                                                            ▼
//! ┌─────────────────┐    is_rate_limited()        ┌──────────────────┐
//! │ OpenAI Service  │ ◄────────────────────────── │   HTTP Handlers  │
//! │                 │                             │                  │
//! │ - Checks rate   │                             │ - Check limiter  │
//! │   limit status  │                             │   before         │
//! │ - Returns 429   │                             │   processing     │
//! │   if limited    │                             │ - Reject with    │
//! │                 │                             │   429 if limited │
//! └─────────────────┘                             └──────────────────┘
//! ```
//!
//! ## Key Components
//!
//! ### RateLimitState
//! Thread-safe state management for rate limiting with automatic expiration:
//! - Tracks when rate limiting was activated
//! - Configurable rate limit duration
//! - Automatic expiration based on elapsed time
//!
//! ### HttpServiceRateLimiter
//! Main rate limiter that integrates with the event system:
//! - Subscribes to `KV_ALL_WORKERS_BUSY_SUBJECT` events from the scheduler
//! - Manages rate limit state transitions
//! - Provides thread-safe access for HTTP handlers
//!
//! ## Integration with KV Scheduler
//!
//! The rate limiter responds to events from [`crate::kv_router::scheduler::KvScheduler`]:
//!
//! 1. **Event Trigger**: When the scheduler's request queue exceeds the configured threshold,
//!    it publishes a [`KVAllWorkersBusyEvent`]
//!
//! 2. **Rate Limit Activation**: The rate limiter receives this event and immediately
//!    activates rate limiting for the configured duration
//!
//! 3. **Request Rejection**: HTTP handlers check [`HttpServiceRateLimiter::is_rate_limited()`]
//!    and reject new requests with HTTP 429 (Too Many Requests)
//!
//! 4. **Automatic Recovery**: Rate limiting automatically expires after the configured
//!    duration, allowing normal request processing to resume
//!
//! ## Configuration
//!
//! Rate limiting behavior is controlled by:
//!
//! - **Enable/Disable**: Pass `Some(duration)` to enable, `None` to disable
//! - **Rate Limit Duration**: How long to reject requests after receiving a busy event
//! - **Queue Threshold**: Configured in the scheduler to determine when to trigger events

use dynamo_runtime::{component::Namespace, traits::events::EventSubscriber, CancellationToken};
use futures::StreamExt;

use crate::kv_router::{scheduler::KVAllWorkersBusyEvent, KV_ALL_WORKERS_BUSY_SUBJECT};
use anyhow::Result;
use std::{
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};

/// Rate limiting state that can be shared across request handlers
#[derive(Debug, Clone)]
pub struct RateLimitState {
    rate_limit_start: Option<Instant>,
    rate_limit_duration: Duration,
}

impl Default for RateLimitState {
    fn default() -> Self {
        Self {
            rate_limit_start: None,
            rate_limit_duration: Duration::from_secs(5),
        }
    }
}

impl RateLimitState {
    pub fn new(rate_limit_duration: Duration) -> Self {
        RateLimitState {
            rate_limit_duration,
            ..Self::default()
        }
    }

    /// Check if state is currently rate limited
    pub fn is_rate_limited(&self) -> bool {
        match self.rate_limit_start {
            Some(start_time) => start_time.elapsed() < self.rate_limit_duration,
            None => false,
        }
    }

    /// Set rate limit state to true and record the start time
    pub fn set_rate_limit(&mut self) {
        let now = Instant::now();
        self.rate_limit_start = Some(now);
    }

    /// Set rate limit state to false and clear the start time
    pub fn clear_rate_limit(&mut self) {
        self.rate_limit_start = None;
    }
}

#[derive(Clone)]
pub struct HttpServiceRateLimiter {
    is_enabled: bool,
    rate_limit_state: Arc<RwLock<RateLimitState>>,
}

impl HttpServiceRateLimiter {
    pub fn new(all_workers_busy_rejection_time_window: Option<Duration>) -> Self {
        Self {
            is_enabled: all_workers_busy_rejection_time_window.is_some(),
            rate_limit_state: Arc::new(RwLock::new(
                all_workers_busy_rejection_time_window
                    .map(RateLimitState::new)
                    .unwrap_or_default(),
            )),
        }
    }

    /// Check if the current underlying state is rate limited
    pub fn is_rate_limited(&self) -> bool {
        if !self.is_enabled {
            return false;
        }

        let state = self.rate_limit_state.read().unwrap();
        state.is_rate_limited()
    }

    /// Start monitoring the rate limit state
    ///
    /// This function will spawn a new task that will monitor the rate limit state and set the rate limit state to true if the all workers busy event is received.
    /// It will also clear the rate limit state after the rate limit duration has passed.
    pub fn start_monitoring(
        &self,
        namespace: Namespace,
        cancellation_token: CancellationToken,
    ) -> Result<()> {
        if !self.is_enabled {
            tracing::warn!("Rate limiter is disabled, skipping monitoring");
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
                    tracing::error!("We can't monitor the rate limit state, and therefore this will be disabled. Failed to subscribe to all workers busy event: {}.", e);
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
                            state.set_rate_limit();
                        };

                        sleep.as_mut().reset(tokio::time::Instant::now() + rate_limit_duration);
                    }
                    _ = &mut sleep => {
                        {
                            let mut state = rate_limiter.write().unwrap();
                            state.clear_rate_limit();
                        }
                        sleep.as_mut().reset(tokio::time::Instant::now() + Duration::MAX);
                    }
                    _ = cancellation_token.cancelled() => {
                        tracing::debug!("Rate limiter monitoring task shutting down");
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
    fn test_rate_limit_state_default() {
        let state = RateLimitState::default();
        assert_eq!(state.rate_limit_start, None);
        assert_eq!(state.rate_limit_duration, Duration::from_secs(5));
    }

    #[test]
    fn test_rate_limit_state_new() {
        let duration = Duration::from_secs(10);
        let state = RateLimitState::new(duration);
        assert_eq!(state.rate_limit_start, None);
        assert_eq!(state.rate_limit_duration, duration);
    }

    #[test]
    fn test_rate_limit_state_is_rate_limited_when_not_set() {
        let state = RateLimitState::default();
        assert!(!state.is_rate_limited());
    }

    #[test]
    fn test_rate_limit_state_is_rate_limited_when_set() {
        let mut state = RateLimitState::default();
        state.set_rate_limit();
        assert!(state.is_rate_limited());
    }

    #[tokio::test]
    async fn test_rate_limit_state_is_rate_limited_after_expiry() {
        let mut state = RateLimitState::new(Duration::from_millis(10));
        state.set_rate_limit();
        assert!(state.is_rate_limited());

        // Wait for the rate limit to expire
        sleep(Duration::from_millis(21)).await;
        assert!(!state.is_rate_limited());
    }

    #[test]
    fn test_rate_limit_state_clear_rate_limit() {
        let mut state = RateLimitState::default();
        state.set_rate_limit();
        assert!(state.is_rate_limited());

        state.clear_rate_limit();
        assert!(!state.is_rate_limited());
        assert_eq!(state.rate_limit_start, None);
    }

    #[test]
    fn test_http_service_rate_limiter_new_enabled() {
        let duration = Duration::from_secs(10);
        let rate_limiter = HttpServiceRateLimiter::new(Some(duration));
        assert!(rate_limiter.is_enabled);
        assert_eq!(
            rate_limiter
                .rate_limit_state
                .read()
                .unwrap()
                .rate_limit_duration,
            duration
        );
    }

    #[test]
    fn test_http_service_rate_limiter_new_disabled() {
        let rate_limiter = HttpServiceRateLimiter::new(None);
        assert!(!rate_limiter.is_enabled);
        assert_eq!(
            rate_limiter
                .rate_limit_state
                .read()
                .unwrap()
                .rate_limit_duration,
            Duration::from_secs(5)
        );
    }

    #[test]
    fn test_http_service_rate_limiter_is_not_rate_limited_when_disabled() {
        let rate_limiter = HttpServiceRateLimiter::new(None);
        assert!(!rate_limiter.is_rate_limited());
    }

    #[test]
    fn test_http_service_rate_limiter_is_not_rate_limited_when_enabled_and_not_set() {
        let rate_limiter = HttpServiceRateLimiter::new(Some(Duration::from_secs(5)));
        assert!(!rate_limiter.is_rate_limited());
    }

    #[test]
    fn test_http_service_rate_limiter_is_rate_limited_when_enabled_and_set() {
        let rate_limiter = HttpServiceRateLimiter::new(Some(Duration::from_secs(5)));
        {
            let mut state = rate_limiter.rate_limit_state.write().unwrap();
            state.set_rate_limit();
        }
        assert!(rate_limiter.is_rate_limited());
    }

    #[tokio::test]
    async fn test_http_service_rate_limiter_is_rate_limited_after_expiry() {
        let rate_limiter = HttpServiceRateLimiter::new(Some(Duration::from_millis(10)));
        {
            let mut state = rate_limiter.rate_limit_state.write().unwrap();
            state.set_rate_limit();
        }
        assert!(rate_limiter.is_rate_limited());

        // Wait for the rate limit to expire
        sleep(Duration::from_millis(21)).await;
        assert!(!rate_limiter.is_rate_limited());
    }

    #[tokio::test]
    async fn test_http_service_updates_rate_limit_time_window() {
        let rate_limiter = HttpServiceRateLimiter::new(Some(Duration::from_secs(1)));
        {
            let mut state = rate_limiter.rate_limit_state.write().unwrap();
            state.set_rate_limit();
        }
        assert!(rate_limiter.is_rate_limited());

        sleep(Duration::from_millis(550)).await;

        {
            let mut state = rate_limiter.rate_limit_state.write().unwrap();
            state.set_rate_limit();
        }
        assert!(rate_limiter.is_rate_limited());

        sleep(Duration::from_millis(550)).await;

        assert!(rate_limiter.is_rate_limited());
    }
}
