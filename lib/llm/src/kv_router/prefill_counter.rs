// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use dynamo_runtime::component::Component;
use dynamo_runtime::traits::events::{EventPublisher, EventSubscriber};
use futures::StreamExt;
use std::sync::Arc;
use uuid::Uuid;

use super::protocols::{PrefillEvent, PrefillEventData};
use crate::kv_router::PREFILL_SUBJECT;
use dashmap::DashMap;
use std::collections::HashMap;
use std::hash::Hash;

pub fn get_snapshot<K, V>(state: &DashMap<K, V>) -> HashMap<K, V>
where
    K: Clone + Hash + Eq,
    V: Copy,
{
    state
        .iter()
        .map(|entry| (entry.key().clone(), *entry.value()))
        .collect()
}

#[derive(Default)]
struct PrefillCounterState {
    tokens_map: HashMap<String, usize>, // Plain HashMap
    running_sum: usize,                 // Plain usize
}

impl PrefillCounterState {
    fn insert(&mut self, key: String, value: usize) -> Option<usize> {
        // Takes &mut self
        let old_value = self.tokens_map.insert(key, value);

        if let Some(old) = old_value {
            self.running_sum -= old;
            self.running_sum += value;
        } else {
            self.running_sum += value;
        }

        old_value
    }

    fn remove(&mut self, key: &str) -> Option<usize> {
        // Takes &mut self
        let removed = self.tokens_map.remove(key);

        if let Some(value) = removed {
            self.running_sum -= value;
        }

        removed
    }

    fn running_sum(&self) -> usize {
        self.running_sum
    }
}

/// A counter that tracks pending prefill tokens for each request.
///
/// This struct maintains a local hashmap of request_id to token count,
/// and a running sum of all tokens. It no longer handles its own subscriptions.
#[derive(Default)] // Removed Clone
pub struct PrefillCounter {
    state: PrefillCounterState, // No Arc, direct ownership
}

impl PrefillCounter {
    // Internal methods for direct state manipulation (no publishing)
    fn insert_direct(&mut self, request_id: String, tokens: usize) -> Option<usize> {
        // Takes &mut self
        self.state.insert(request_id, tokens)
    }

    fn remove_direct(&mut self, request_id: &str) -> Option<usize> {
        // Takes &mut self
        self.state.remove(request_id)
    }

    #[allow(dead_code)]
    fn update_direct(&mut self, request_id: String, new_tokens: usize) {
        // Takes &mut self
        if let Some(old_tokens) = self.state.tokens_map.get(&request_id).copied() {
            let delta = new_tokens as isize - old_tokens as isize;
            self.state.running_sum = (self.state.running_sum as isize + delta) as usize;
            self.state.tokens_map.insert(request_id, new_tokens);
        }
    }

    pub fn get(&self, request_id: &str) -> Option<usize> {
        self.state.tokens_map.get(request_id).copied()
    }

    pub fn running_sum(&self) -> usize {
        self.state.running_sum()
    }

    pub fn len(&self) -> usize {
        self.state.tokens_map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.state.tokens_map.is_empty()
    }
}

/// A collection of PrefillCounters for multiple workers with centralized event handling
pub struct PrefillCountersMultiWorker {
    pub counters: Arc<DashMap<i64, PrefillCounter>>,
    pub request_to_workers: Arc<DashMap<String, i64>>,
    component: Component,
    router_id: Uuid,
}

impl PrefillCountersMultiWorker {
    // Helper function to handle new prefill logic
    fn handle_new_prefill(
        counters: &Arc<DashMap<i64, PrefillCounter>>,
        request_to_workers: &Arc<DashMap<String, i64>>,
        request_id: &str,
        worker_id: i64,
        tokens: usize,
    ) {
        // Check if request already exists
        if let Some(existing_worker_id) = request_to_workers.get(request_id) {
            tracing::warn!(
                "Request {} already exists for worker {}, but trying to add to worker {}",
                request_id,
                *existing_worker_id,
                worker_id
            );
        }

        // Update mapping
        request_to_workers.insert(request_id.to_string(), worker_id);

        // Get or create counter and insert using get_mut
        if let Some(mut counter) = counters.get_mut(&worker_id) {
            counter.insert_direct(request_id.to_string(), tokens);
        } else {
            tracing::warn!(
                "Worker {} does not exist, creating new PrefillCounter",
                worker_id
            );
            let mut new_counter = PrefillCounter::default();
            new_counter.insert_direct(request_id.to_string(), tokens);
            counters.insert(worker_id, new_counter);
        };
    }

    // Helper function to handle complete prefill logic
    fn handle_complete_prefill(
        counters: &Arc<DashMap<i64, PrefillCounter>>,
        request_to_workers: &Arc<DashMap<String, i64>>,
        request_id: &str,
    ) -> Option<usize> {
        // Remove from request_to_workers and get the worker_id
        let Some((_, worker_id)) = request_to_workers.remove(request_id) else {
            tracing::warn!("Request {} not found in request_to_workers", request_id);
            return None;
        };

        // Use the worker_id from request_to_workers with get_mut
        let Some(mut counter) = counters.get_mut(&worker_id) else {
            tracing::warn!(
                "No counter found for worker {} for request {}",
                worker_id,
                request_id
            );
            return None;
        };

        let removed_tokens = counter.remove_direct(request_id);
        if removed_tokens.is_none() {
            tracing::warn!("Attempted to remove non-existent request: {}", request_id);
        }

        removed_tokens
    }

    pub fn new(component: Component) -> Self {
        let counters = Arc::new(DashMap::new());
        let request_to_workers = Arc::new(DashMap::new());
        let router_id = Uuid::new_v4();

        let multi_worker = Self {
            counters: counters.clone(),
            request_to_workers: request_to_workers.clone(),
            component: component.clone(),
            router_id,
        };

        // Start the subscription loop
        let counters_clone = counters.clone();
        let request_to_workers_clone = request_to_workers.clone();
        let component_clone = component.clone();
        let router_id_clone = router_id;

        tokio::spawn(async move {
            if let Err(e) = Self::subscribe_to_events(
                counters_clone,
                request_to_workers_clone,
                component_clone,
                router_id_clone,
            )
            .await
            {
                tracing::error!("Error in prefill events subscription: {}", e);
            }
        });

        multi_worker
    }

    /// Background task to subscribe to prefill events and update all counters
    async fn subscribe_to_events(
        counters: Arc<DashMap<i64, PrefillCounter>>,
        request_to_workers: Arc<DashMap<String, i64>>,
        component: Component,
        router_id: Uuid,
    ) -> Result<()> {
        let mut subscriber = component
            .subscribe_with_type::<PrefillEvent>(PREFILL_SUBJECT)
            .await?;

        while let Some(result) = subscriber.next().await {
            let Ok(event) = result else {
                tracing::error!("Error receiving prefill event: {}", result.unwrap_err());
                continue;
            };

            // Skip events emitted by itself
            if event.router_id == router_id {
                continue;
            }

            match event.data {
                PrefillEventData::NewPrefill(tokens) => {
                    Self::handle_new_prefill(
                        &counters,
                        &request_to_workers,
                        &event.request_id,
                        event.worker_id,
                        tokens,
                    );
                }
                PrefillEventData::UpdatePrefill(_) => {
                    // Do nothing for now
                    continue;
                }
                PrefillEventData::CompletePrefill => {
                    Self::handle_complete_prefill(
                        &counters,
                        &request_to_workers,
                        &event.request_id,
                    );
                }
            }
        }

        Ok(())
    }

    pub async fn add_prefill(
        &self,
        worker_id: i64,
        request_id: String,
        new_tokens: usize,
    ) -> Result<()> {
        let event = PrefillEvent {
            request_id: request_id.clone(),
            worker_id,
            data: PrefillEventData::NewPrefill(new_tokens),
            router_id: self.router_id,
        };
        self.component.publish(PREFILL_SUBJECT, &event).await?;

        // Use the helper function
        Self::handle_new_prefill(
            &self.counters,
            &self.request_to_workers,
            &request_id,
            worker_id,
            new_tokens,
        );

        Ok(())
    }

    pub async fn remove_prefill(&self, request_id: &str) -> Result<Option<usize>> {
        // Send the event first with dummy worker_id
        let event = PrefillEvent {
            request_id: request_id.to_string(),
            worker_id: 0, // Dummy worker_id
            data: PrefillEventData::CompletePrefill,
            router_id: self.router_id,
        };
        self.component.publish(PREFILL_SUBJECT, &event).await?;

        // Use the helper function
        Ok(Self::handle_complete_prefill(
            &self.counters,
            &self.request_to_workers,
            request_id,
        ))
    }

    /// Get the running sums for all workers as a HashMap<i64, usize>
    pub async fn running_sums(&self) -> HashMap<i64, usize> {
        self.counters
            .iter()
            .map(|entry| (*entry.key(), entry.value().running_sum()))
            .collect()
    }

    /// Get a specific counter's running sum
    pub async fn get_worker_sum(&self, worker_id: i64) -> Option<usize> {
        self.counters.get(&worker_id).map(|c| c.running_sum())
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use dynamo_runtime::{DistributedRuntime, Runtime};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use tokio::time::Duration;

    #[test]
    #[ignore]
    fn test_prefill_counter_multiworker_synchronization() -> Result<()> {
        // Initialize logging once
        dynamo_runtime::logging::init();

        let worker_id_1 = 1;
        let worker_id_2 = 2;
        let tokens_per_request = 100;
        let requests_per_worker = 10;

        // Shared state for collecting results from both threads
        let results1 = Arc::new(Mutex::new(None));
        let results2 = Arc::new(Mutex::new(None));
        let final_results1 = Arc::new(Mutex::new(None));
        let final_results2 = Arc::new(Mutex::new(None));

        let results1_clone = results1.clone();
        let results2_clone = results2.clone();
        let final_results1_clone = final_results1.clone();
        let final_results2_clone = final_results2.clone();

        // Thread 1: First distributed runtime with multi_worker1
        let handle1 = thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();

            rt.block_on(async {
                // Create runtime and distributed runtime
                let runtime = Runtime::from_current()?;
                let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;

                // Create namespace and components with same names
                let namespace = distributed.namespace("test_prefill_multiworker")?;
                let component = namespace
                    .component("counters")?
                    .service_builder()
                    .create()
                    .await?;

                // Create first PrefillCountersMultiWorker instance
                let multi_worker1 = PrefillCountersMultiWorker::new(component);

                // Give some time for subscribers to initialize
                tokio::time::sleep(Duration::from_millis(3000)).await;

                // Send requests to multi_worker1's worker
                for i in 0..requests_per_worker {
                    let request_id = format!("mw1_request_{}", i);
                    multi_worker1
                        .add_prefill(worker_id_1, request_id, tokens_per_request)
                        .await?;
                }

                // Wait for synchronization
                tokio::time::sleep(Duration::from_millis(1000)).await;

                // Get running sums after additions
                let sums1 = multi_worker1.running_sums().await;
                *results1_clone.lock().unwrap() = Some(sums1);

                // Wait for other thread to add its requests
                tokio::time::sleep(Duration::from_millis(2000)).await;

                // Remove all requests from multi_worker1
                for i in 0..requests_per_worker {
                    let request_id = format!("mw1_request_{}", i);
                    multi_worker1.remove_prefill(&request_id).await?;
                }

                // Wait for removal synchronization
                tokio::time::sleep(Duration::from_millis(1000)).await;

                // Get final running sums
                let final_sums1 = multi_worker1.running_sums().await;
                *final_results1_clone.lock().unwrap() = Some(final_sums1);

                // Keep runtime alive a bit longer for synchronization
                tokio::time::sleep(Duration::from_millis(1000)).await;

                // Shutdown runtime
                runtime.shutdown();

                Ok::<(), anyhow::Error>(())
            })
        });

        // Thread 2: Second distributed runtime with multi_worker2
        let handle2 = thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();

            rt.block_on(async {
                // Create runtime and distributed runtime
                let runtime = Runtime::from_current()?;
                let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;

                // Create namespace and components with same names
                let namespace = distributed.namespace("test_prefill_multiworker")?;
                let component = namespace
                    .component("counters")?
                    .service_builder()
                    .create()
                    .await?;

                // Create second PrefillCountersMultiWorker instance
                let multi_worker2 = PrefillCountersMultiWorker::new(component);

                // Give some time for subscribers to initialize
                tokio::time::sleep(Duration::from_millis(3000)).await;

                // Wait a bit to ensure multi_worker1 has started
                tokio::time::sleep(Duration::from_millis(500)).await;

                // Send requests to multi_worker2's worker
                for i in 0..requests_per_worker {
                    let request_id = format!("mw2_request_{}", i);
                    multi_worker2
                        .add_prefill(worker_id_2, request_id, tokens_per_request)
                        .await?;
                }

                // Wait for synchronization
                tokio::time::sleep(Duration::from_millis(1000)).await;

                // Get running sums after additions
                let sums2 = multi_worker2.running_sums().await;
                *results2_clone.lock().unwrap() = Some(sums2);

                // Wait for other thread to remove its requests
                tokio::time::sleep(Duration::from_millis(2000)).await;

                // Remove all requests from multi_worker2
                for i in 0..requests_per_worker {
                    let request_id = format!("mw2_request_{}", i);
                    multi_worker2.remove_prefill(&request_id).await?;
                }

                // Wait for removal synchronization
                tokio::time::sleep(Duration::from_millis(1000)).await;

                // Get final running sums
                let final_sums2 = multi_worker2.running_sums().await;
                *final_results2_clone.lock().unwrap() = Some(final_sums2);

                // Keep runtime alive a bit longer for synchronization
                tokio::time::sleep(Duration::from_millis(1000)).await;

                // Shutdown runtime
                runtime.shutdown();

                Ok::<(), anyhow::Error>(())
            })
        });

        // Wait for both threads to complete
        handle1.join().unwrap()?;
        handle2.join().unwrap()?;

        // Extract results
        let sums1 = results1.lock().unwrap().take().unwrap();
        let sums2 = results2.lock().unwrap().take().unwrap();
        let final_sums1 = final_results1.lock().unwrap().take().unwrap();
        let final_sums2 = final_results2.lock().unwrap().take().unwrap();

        // Verify both multi-workers see all requests
        assert_eq!(
            sums1.get(&worker_id_1),
            Some(&(requests_per_worker * tokens_per_request)),
            "MultiWorker1 should see worker 1's requests"
        );
        assert_eq!(
            sums1.get(&worker_id_2),
            Some(&(requests_per_worker * tokens_per_request)),
            "MultiWorker1 should see worker 2's requests"
        );
        assert_eq!(
            sums2.get(&worker_id_1),
            Some(&(requests_per_worker * tokens_per_request)),
            "MultiWorker2 should see worker 1's requests"
        );
        assert_eq!(
            sums2.get(&worker_id_2),
            Some(&(requests_per_worker * tokens_per_request)),
            "MultiWorker2 should see worker 2's requests"
        );

        // Verify both multi-workers show zero sums after removal
        assert_eq!(
            final_sums1.get(&worker_id_1).copied().unwrap_or(0),
            0,
            "MultiWorker1 should show zero for worker 1"
        );
        assert_eq!(
            final_sums1.get(&worker_id_2).copied().unwrap_or(0),
            0,
            "MultiWorker1 should show zero for worker 2"
        );
        assert_eq!(
            final_sums2.get(&worker_id_1).copied().unwrap_or(0),
            0,
            "MultiWorker2 should show zero for worker 1"
        );
        assert_eq!(
            final_sums2.get(&worker_id_2).copied().unwrap_or(0),
            0,
            "MultiWorker2 should show zero for worker 2"
        );

        Ok(())
    }
}
