// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LORA Load Estimator
//!
//! Tracks LORA adapter usage over time to estimate load for allocation decisions.
//! Supports single-router (polling) and multi-router (event-based) modes.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use dynamo_runtime::component::Component;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::EventSubscriber;
use tokio::sync::RwLock;

use crate::kv_router::ACTIVE_SEQUENCES_SUBJECT;
use crate::kv_router::protocols::{ActiveSequenceEvent, ActiveSequenceEventData};
use crate::kv_router::scheduler::KvScheduler;

/// Time-series sample of LORA load
#[derive(Debug, Clone)]
pub struct LoadSample {
    pub timestamp: Instant,
    pub active_count: usize,
}

/// Configuration for load estimation
#[derive(Debug, Clone)]
pub struct LoadEstimatorConfig {
    /// How often to poll for load updates (single-router mode)
    pub poll_interval: Duration,

    /// Maximum number of samples to keep per LORA
    pub max_samples: usize,
}

impl Default for LoadEstimatorConfig {
    fn default() -> Self {
        Self {
            poll_interval: Duration::from_secs(5),
            max_samples: 1000,
        }
    }
}

/// Estimates LORA load based on active request counts over time
pub struct LoadEstimator {
    /// Current active counts per LORA (latest snapshot)
    active_counts: Arc<DashMap<String, usize>>,

    /// Historical load samples per LORA
    history: Arc<RwLock<HashMap<String, VecDeque<LoadSample>>>>,

    /// Configuration
    config: LoadEstimatorConfig,
}

impl LoadEstimator {
    /// Create a new load estimator with default configuration
    pub fn new() -> Self {
        Self::with_config(LoadEstimatorConfig::default())
    }

    /// Create a new load estimator with custom configuration
    pub fn with_config(config: LoadEstimatorConfig) -> Self {
        Self {
            active_counts: Arc::new(DashMap::new()),
            history: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Start polling the scheduler for LORA load (single-router mode)
    pub fn start_polling(
        self: Arc<Self>,
        scheduler: Arc<KvScheduler>,
        component: Component,
    ) -> tokio::task::JoinHandle<()> {
        let cancel_token = component.drt().child_token();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(self.config.poll_interval);
            tracing::info!("Started LORA load polling");

            loop {
                tokio::select! {
                    _ = cancel_token.cancelled() => {
                        tracing::debug!("LORA load polling task cancelled");
                        break;
                    }
                    _ = interval.tick() => {
                        // Poll scheduler for current LORA counts
                        let lora_counts = scheduler.get_active_lora_counts();

                        // Update load estimates
                        if let Err(e) = self.update_from_counts(lora_counts).await {
                            tracing::warn!("Failed to update load estimates: {}", e);
                        }
                    }
                }
            }
        })
    }

    /// Start subscribing to ActiveSequenceEvent for LORA load (multi-router mode)
    pub fn start_event_subscription(
        self: Arc<Self>,
        component: Component,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            if let Err(e) = self.subscribe_to_events(component).await {
                tracing::error!("Error in LORA load event subscription: {}", e);
            }
        })
    }

    /// Subscribe to ActiveSequenceEvent and update load tracking
    async fn subscribe_to_events(&self, component: Component) -> anyhow::Result<()> {
        let cancel_token = component.drt().child_token();
        let mut subscriber = EventSubscriber::for_component(&component, ACTIVE_SEQUENCES_SUBJECT)
            .await?
            .typed::<ActiveSequenceEvent>();

        tracing::info!("Started LORA load event subscription");

        loop {
            tokio::select! {
                _ = cancel_token.cancelled() => {
                    tracing::debug!("LORA load event subscription cancelled");
                    break;
                }
                result = subscriber.next() => {
                    match result {
                        Some(Ok((_envelope, event))) => {
                            self.handle_event(event).await;
                        }
                        Some(Err(e)) => {
                            tracing::warn!("Error receiving LORA load event: {}", e);
                        }
                        None => {
                            tracing::warn!("LORA load event stream ended");
                            break;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Handle an ActiveSequenceEvent and update load tracking
    async fn handle_event(&self, event: ActiveSequenceEvent) {
        if let Some(lora_name) = event.lora_name {
            match event.data {
                ActiveSequenceEventData::AddRequest { .. } => {
                    // Increment load for this LORA
                    if let Err(e) = self.increment_load(&lora_name).await {
                        tracing::warn!("Failed to increment load for {}: {}", lora_name, e);
                    }
                }
                ActiveSequenceEventData::Free => {
                    // Decrement load for this LORA
                    if let Err(e) = self.decrement_load(&lora_name).await {
                        tracing::warn!("Failed to decrement load for {}: {}", lora_name, e);
                    }
                }
                ActiveSequenceEventData::MarkPrefillCompleted => {
                    // No load change for prefill completion
                }
            }
        }
    }

    /// Increment load count for a LORA and record sample
    async fn increment_load(&self, lora_name: &str) -> anyhow::Result<()> {
        let now = Instant::now();

        // Increment active count
        let new_count = *self
            .active_counts
            .entry(lora_name.to_string())
            .and_modify(|count| *count += 1)
            .or_insert(1);

        // Add sample to history
        let mut history = self.history.write().await;
        let samples = history
            .entry(lora_name.to_string())
            .or_insert_with(VecDeque::new);

        samples.push_back(LoadSample {
            timestamp: now,
            active_count: new_count,
        });

        self.trim_samples(samples, now);

        Ok(())
    }

    /// Decrement load count for a LORA and record sample
    async fn decrement_load(&self, lora_name: &str) -> anyhow::Result<()> {
        let now = Instant::now();

        // Atomically decrement and conditionally remove
        let new_count = match self.active_counts.entry(lora_name.to_string()) {
            dashmap::mapref::entry::Entry::Occupied(mut entry) => {
                let count = entry.get_mut();
                *count = count.saturating_sub(1);
                let new_val = *count;
                if new_val == 0 {
                    entry.remove_entry();
                }
                new_val
            }
            dashmap::mapref::entry::Entry::Vacant(_) => {
                // No entry exists, treat as 0
                0
            }
        };

        // Add sample to history
        let mut history = self.history.write().await;
        let samples = history
            .entry(lora_name.to_string())
            .or_insert_with(VecDeque::new);

        samples.push_back(LoadSample {
            timestamp: now,
            active_count: new_count,
        });

        self.trim_samples(samples, now);

        Ok(())
    }

    /// Update load estimates from a snapshot of LORA counts
    async fn update_from_counts(&self, lora_counts: HashMap<String, usize>) -> anyhow::Result<()> {
        let now = Instant::now();
        let mut history = self.history.write().await;

        // Update active counts
        for (lora_name, count) in &lora_counts {
            self.active_counts.insert(lora_name.clone(), *count);
        }

        // Remove LORAs that are no longer active
        self.active_counts
            .retain(|lora_name, _| lora_counts.contains_key(lora_name));

        // Add samples to history
        for (lora_name, count) in lora_counts {
            let samples = history.entry(lora_name).or_insert_with(VecDeque::new);

            samples.push_back(LoadSample {
                timestamp: now,
                active_count: count,
            });

            // Trim old samples
            self.trim_samples(samples, now);
        }

        // Clean up LORAs with no recent activity
        history.retain(|_, samples| !samples.is_empty());

        Ok(())
    }

    /// Trim samples to fixed-size window
    fn trim_samples(&self, samples: &mut VecDeque<LoadSample>, _now: Instant) {
        // Enforce max samples limit
        while samples.len() > self.config.max_samples {
            samples.pop_front();
        }
    }

    /// Get current active counts
    pub fn get_current_load(&self) -> HashMap<String, usize> {
        self.active_counts
            .iter()
            .map(|entry| (entry.key().clone(), *entry.value()))
            .collect()
    }

    /// Get time series samples for all LORAs (oldest -> newest)
    pub async fn get_time_series(&self) -> HashMap<String, Vec<LoadSample>> {
        let history = self.history.read().await;
        history
            .iter()
            .map(|(lora, samples)| (lora.clone(), samples.iter().cloned().collect()))
            .collect()
    }
}

impl Default for LoadEstimator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_load_estimator_time_series() {
        let estimator = Arc::new(LoadEstimator::new());

        // Simulate updates
        let mut counts = HashMap::new();
        counts.insert("lora-math".to_string(), 5);
        counts.insert("lora-code".to_string(), 3);

        estimator.update_from_counts(counts).await.unwrap();

        let all_series = estimator.get_time_series().await;
        let series_math = all_series.get("lora-math").unwrap();
        let series_code = all_series.get("lora-code").unwrap();

        assert_eq!(series_math.len(), 1);
        assert_eq!(series_math[0].active_count, 5);
        assert_eq!(series_code.len(), 1);
        assert_eq!(series_code[0].active_count, 3);
        assert!(!all_series.contains_key("lora-xyz"));
    }

    #[tokio::test]
    async fn test_load_estimator_max_samples() {
        let config = LoadEstimatorConfig {
            max_samples: 2,
            ..Default::default()
        };
        let estimator = Arc::new(LoadEstimator::with_config(config));

        for count in [1, 2, 3] {
            let mut counts = HashMap::new();
            counts.insert("lora-math".to_string(), count);
            estimator.update_from_counts(counts).await.unwrap();
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        let all_series = estimator.get_time_series().await;
        let series = all_series.get("lora-math").unwrap();
        assert_eq!(series.len(), 2);
        assert_eq!(series[0].active_count, 2);
        assert_eq!(series[1].active_count, 3);
    }
}
