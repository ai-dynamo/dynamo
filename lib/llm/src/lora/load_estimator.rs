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

use crate::kv_router::ACTIVE_SEQUENCES_SUBJECT;
use crate::kv_router::protocols::{ActiveSequenceEvent, ActiveSequenceEventData};
use crate::kv_router::scheduler::KvScheduler;

/// Time-series sample of LORA load
#[derive(Debug, Clone)]
pub struct LoadSample {
    pub timestamp: Instant,
    pub active_count: usize,
}

/// Per-LORA load data combining active count and history
#[derive(Debug, Clone, Default)]
struct LoraLoadData {
    /// Current active request count
    active_count: usize,
    /// Historical load samples
    samples: VecDeque<LoadSample>,
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
    /// Per-LORA load data (active count + history) with atomic updates
    data: DashMap<String, LoraLoadData>,

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
            data: DashMap::new(),
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
                        self.update_from_counts(lora_counts);
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
                            self.handle_event(event);
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
    fn handle_event(&self, event: ActiveSequenceEvent) {
        if let Some(lora_name) = event.lora_name {
            match event.data {
                ActiveSequenceEventData::AddRequest { .. } => {
                    // Increment load for this LORA
                    self.increment_load(&lora_name);
                }
                ActiveSequenceEventData::Free => {
                    // Decrement load for this LORA
                    self.decrement_load(&lora_name);
                }
                ActiveSequenceEventData::MarkPrefillCompleted => {
                    // No load change for prefill completion
                }
            }
        }
    }

    /// Increment load count for a LORA and record sample (atomic)
    fn increment_load(&self, lora_name: &str) {
        let now = Instant::now();
        let max_samples = self.config.max_samples;

        self.data
            .entry(lora_name.to_string())
            .and_modify(|data| {
                data.active_count += 1;
                data.samples.push_back(LoadSample {
                    timestamp: now,
                    active_count: data.active_count,
                });
                // Trim old samples
                while data.samples.len() > max_samples {
                    data.samples.pop_front();
                }
            })
            .or_insert_with(|| {
                let mut data = LoraLoadData {
                    active_count: 1,
                    samples: VecDeque::new(),
                };
                data.samples.push_back(LoadSample {
                    timestamp: now,
                    active_count: 1,
                });
                data
            });
    }

    /// Decrement load count for a LORA and record sample (atomic)
    fn decrement_load(&self, lora_name: &str) {
        let now = Instant::now();
        let max_samples = self.config.max_samples;

        // Update existing entry or ignore if not present
        if let Some(mut entry) = self.data.get_mut(lora_name) {
            let data = entry.value_mut();
            data.active_count = data.active_count.saturating_sub(1);
            data.samples.push_back(LoadSample {
                timestamp: now,
                active_count: data.active_count,
            });
            // Trim old samples
            while data.samples.len() > max_samples {
                data.samples.pop_front();
            }
        }
    }

    /// Update load estimates from a snapshot of LORA counts
    fn update_from_counts(&self, lora_counts: HashMap<String, usize>) {
        let now = Instant::now();
        let max_samples = self.config.max_samples;

        // Update or insert entries for all LORAs in the snapshot
        for (lora_name, count) in &lora_counts {
            self.data
                .entry(lora_name.clone())
                .and_modify(|data| {
                    data.active_count = *count;
                    data.samples.push_back(LoadSample {
                        timestamp: now,
                        active_count: *count,
                    });
                    // Trim old samples
                    while data.samples.len() > max_samples {
                        data.samples.pop_front();
                    }
                })
                .or_insert_with(|| {
                    let mut data = LoraLoadData {
                        active_count: *count,
                        samples: VecDeque::new(),
                    };
                    data.samples.push_back(LoadSample {
                        timestamp: now,
                        active_count: *count,
                    });
                    data
                });
        }

        // Remove LORAs that are no longer active (set count to 0, keep history)
        for mut entry in self.data.iter_mut() {
            if !lora_counts.contains_key(entry.key()) {
                let data = entry.value_mut();
                if data.active_count > 0 {
                    data.active_count = 0;
                    data.samples.push_back(LoadSample {
                        timestamp: now,
                        active_count: 0,
                    });
                    // Trim old samples
                    while data.samples.len() > max_samples {
                        data.samples.pop_front();
                    }
                }
            }
        }
    }

    /// Get current active counts
    pub fn get_current_load(&self) -> HashMap<String, usize> {
        self.data
            .iter()
            .filter(|entry| entry.value().active_count > 0)
            .map(|entry| (entry.key().clone(), entry.value().active_count))
            .collect()
    }

    /// Get time series samples for all LORAs (oldest -> newest)
    pub fn get_time_series(&self) -> HashMap<String, Vec<LoadSample>> {
        self.data
            .iter()
            .map(|entry| {
                (
                    entry.key().clone(),
                    entry.value().samples.iter().cloned().collect(),
                )
            })
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

    #[test]
    fn test_load_estimator_time_series() {
        let estimator = LoadEstimator::new();

        // Simulate updates
        let mut counts = HashMap::new();
        counts.insert("lora-math".to_string(), 5);
        counts.insert("lora-code".to_string(), 3);

        estimator.update_from_counts(counts);

        let all_series = estimator.get_time_series();
        let series_math = all_series.get("lora-math").unwrap();
        let series_code = all_series.get("lora-code").unwrap();

        assert_eq!(series_math.len(), 1);
        assert_eq!(series_math[0].active_count, 5);
        assert_eq!(series_code.len(), 1);
        assert_eq!(series_code[0].active_count, 3);
        assert!(!all_series.contains_key("lora-xyz"));
    }

    #[test]
    fn test_load_estimator_max_samples() {
        let config = LoadEstimatorConfig {
            max_samples: 2,
            ..Default::default()
        };
        let estimator = LoadEstimator::with_config(config);

        for count in [1, 2, 3] {
            let mut counts = HashMap::new();
            counts.insert("lora-math".to_string(), count);
            estimator.update_from_counts(counts);
        }

        let all_series = estimator.get_time_series();
        let series = all_series.get("lora-math").unwrap();
        assert_eq!(series.len(), 2);
        assert_eq!(series[0].active_count, 2);
        assert_eq!(series[1].active_count, 3);
    }

    #[test]
    fn test_increment_decrement_atomicity() {
        let estimator = LoadEstimator::new();

        // Increment twice
        estimator.increment_load("lora-test");
        estimator.increment_load("lora-test");

        let load = estimator.get_current_load();
        assert_eq!(load.get("lora-test"), Some(&2));

        // Decrement once
        estimator.decrement_load("lora-test");

        let load = estimator.get_current_load();
        assert_eq!(load.get("lora-test"), Some(&1));

        // Check history has all samples
        let series = estimator.get_time_series();
        let samples = series.get("lora-test").unwrap();
        assert_eq!(samples.len(), 3);
        assert_eq!(samples[0].active_count, 1);
        assert_eq!(samples[1].active_count, 2);
        assert_eq!(samples[2].active_count, 1);
    }
}
