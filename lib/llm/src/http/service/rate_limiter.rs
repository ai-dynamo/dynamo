use dynamo_runtime::{component::Namespace, traits::events::EventSubscriber};
use futures::StreamExt;

use crate::kv_router::{scheduler::KVAllWorkersBusyEvent, KV_ALL_WORKERS_BUSY_SUBJECT};
use anyhow::{Context, Error, Result};
use std::{
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};

/// Rate limiting state that can be shared across request handlers
#[derive(Debug, Clone)]
pub struct RateLimitState {
    pub is_rate_limited: bool,
    pub rate_limit_start: Option<Instant>,
    pub rate_limit_duration: Duration,
}

impl Default for RateLimitState {
    fn default() -> Self {
        Self {
            is_rate_limited: false,
            rate_limit_start: None,
            rate_limit_duration: Duration::from_secs(5),
        }
    }
}

impl RateLimitState {
    pub fn new(rate_limit_duration: Duration) -> Self {
        let mut this = Self::default();
        this.rate_limit_duration = rate_limit_duration;
        this
    }

    /// Check if state is currently rate limited
    pub fn is_rate_limited(&self) -> bool {
        if !self.is_rate_limited {
            return false;
        }

        // Check if rate limit duration has expired
        if let Some(start_time) = self.rate_limit_start {
            if start_time.elapsed() > self.rate_limit_duration {
                return false;
            }
        }
        true
    }

    /// Set rate limit state to true and record the start time
    pub fn set_rate_limit(&mut self) {
        let now = Instant::now();
        self.is_rate_limited = true;
        self.rate_limit_start = Some(now);
    }

    /// Set rate limit state to false and clear the start time
    pub fn clear_rate_limit(&mut self) {
        self.is_rate_limited = false;
        self.rate_limit_start = None;
    }
}

#[derive(Clone)]
pub struct HttpServiceRateLimiter {
    pub is_enabled: bool,
    pub rate_limit_state: Arc<RwLock<RateLimitState>>,
}

impl HttpServiceRateLimiter {
    pub fn new(all_workers_busy_rejection_time_window: Option<Duration>) -> Self {
        Self {
            is_enabled: all_workers_busy_rejection_time_window.is_some(),
            rate_limit_state: Arc::new(RwLock::new(
                all_workers_busy_rejection_time_window
                    .map(|duration| RateLimitState::new(duration))
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
    pub fn start_monitoring(&self, namespace: Namespace) -> Result<()> {
        tracing::info!(
            "Starting rate limit monitoring for namespace={}",
            namespace.name()
        );
        let rate_limit_duration = { self.rate_limit_state.read().unwrap().rate_limit_duration };
        let rate_limiter = self.rate_limit_state.clone();
        tokio::spawn(async move {
            let mut all_workers_busy_event_rx = namespace
                .subscribe(KV_ALL_WORKERS_BUSY_SUBJECT)
                .await
                .context("Failed to subscribe to all workers busy event")?;
            while let Some(msg) = all_workers_busy_event_rx.next().await {
                let kv_all_workers_busy_event: KVAllWorkersBusyEvent =
                    serde_json::from_slice(&msg.payload)
                        .context("Failed to deserialize all workers busy event")?;
                let timestamp = kv_all_workers_busy_event.timestamp;
                tracing::info!(
                    "Received all workers busy event with queue_depth={} at timestamp={}",
                    kv_all_workers_busy_event.max_queue_depth,
                    timestamp
                );

                let mut state = rate_limiter.write().unwrap();
                state.set_rate_limit();
                drop(state);

                // Schedule clearing the rate limit after the rate limit duration
                let rate_limiter_clone = rate_limiter.clone();
                tokio::spawn(async move {
                    tokio::time::sleep(rate_limit_duration).await;
                    let mut state = rate_limiter_clone.write().unwrap();
                    state.clear_rate_limit();
                });
            }
            Ok::<(), Error>(())
        });
        Ok(())
    }
}
