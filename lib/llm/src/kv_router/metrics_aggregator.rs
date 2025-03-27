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

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use dynamo_runtime::component::Component;
use dynamo_runtime::{service::EndpointInfo, utils::Duration, Result};
use tokio_util::sync::CancellationToken;
use tracing::debug;

use crate::kv_router::indexer::{KvIndexer, KvIndexerInterface, WorkerId};
use crate::kv_router::protocols::ForwardPassMetrics;
use crate::kv_router::scheduler::Endpoint;
use crate::kv_router::ProcessedEndpoints;
use crate::kv_router::KV_METRICS_ENDPOINT;

pub struct KvMetricsAggregator {
    pub service_name: String,
    pub endpoints: Arc<Mutex<ProcessedEndpoints>>,
    pub kv_block_size: usize,
    indexer: KvIndexer,
}

impl KvMetricsAggregator {
    /// Creates a new KvMetricsAggregator with KvIndexer
    ///
    /// # Arguments
    /// * `component` - Component information
    /// * `cancellation_token` - Token for cancelling background tasks
    /// * `kv_block_size` - Size of the KV blocks
    ///
    /// # Returns
    /// * `Result<Self, anyhow::Error>` - New KvMetricsAggregator instance or an error
    pub async fn new(
        component: Component,
        cancellation_token: CancellationToken,
        kv_block_size: usize,
    ) -> Self {
        let (ep_tx, mut ep_rx) = tokio::sync::mpsc::channel(128);

        tokio::spawn(collect_endpoints_task(
            component.clone(),
            ep_tx,
            cancellation_token.clone(),
        ));

        let endpoints = Arc::new(Mutex::new(ProcessedEndpoints::default()));
        let endpoints_clone = endpoints.clone();
        tokio::spawn(async move {
            tracing::debug!("scheduler background task started");
            loop {
                match ep_rx.recv().await {
                    Some(endpoints) => match endpoints_clone.lock() {
                        Ok(mut shared_endpoint) => {
                            *shared_endpoint = endpoints;
                        }
                        Err(e) => {
                            tracing::error!("Failed to acquire lock on endpoints: {:?}", e);
                        }
                    },
                    None => {
                        tracing::warn!("endpoint subscriber shutdown");
                        break;
                    }
                };
            }

            tracing::trace!("background endpoint subscriber shutting down");
        });
        let indexer = KvIndexer::new(cancellation_token.clone(), kv_block_size);

        Self {
            service_name: component.service_name(),
            endpoints,
            indexer,
            kv_block_size,
        }
    }

    pub fn get_endpoints(&self) -> ProcessedEndpoints {
        match self.endpoints.lock() {
            Ok(endpoints) => endpoints.clone(),
            Err(e) => {
                tracing::error!("Failed to acquire lock on endpoints: {:?}", e);
                ProcessedEndpoints::default()
            }
        }
    }

    /// Finds the best worker based on overlap scores, GPU cache usage, and request queue metrics
    ///
    /// # Arguments
    /// * `tokens` - The tokens to match
    /// * `block_size` - The size of the block
    ///
    /// # Returns
    /// * `Result<String, anyhow::Error>` - The ID of the best worker or an error
    pub async fn find_best_worker(&self, tokens: &[u32]) -> Result<WorkerId, anyhow::Error> {
        // Get overlap scores from the indexer
        let overlap_scores = self.indexer.find_matches_for_request(tokens).await?;

        // Get endpoint metrics
        let endpoints = self.get_endpoints();
        let mut endpoint_metrics: HashMap<WorkerId, &ForwardPassMetrics> = HashMap::new();
        for endpoint in &endpoints.endpoints {
            endpoint_metrics.insert(endpoint.worker_id(), &endpoint.data);
        }

        let mut max_requests_waiting = 0;
        for endpoint in &endpoints.endpoints {
            max_requests_waiting = max_requests_waiting.max(endpoint.data.num_requests_waiting);
        }
        let max_requests_waiting = max_requests_waiting.max(1);

        // Compute cost for each worker
        let mut worker_costs: HashMap<WorkerId, f64> = HashMap::new();
        for (worker_id, metrics) in &endpoint_metrics {
            let overlap_score = overlap_scores.scores.get(worker_id).copied().unwrap_or(0);
            let cost = calculate_worker_cost(
                overlap_score,
                self.kv_block_size,
                tokens.len(),
                metrics.gpu_cache_usage_perc,
                metrics.num_requests_waiting,
                max_requests_waiting,
            );
            worker_costs.insert(*worker_id, cost);
        }

        if worker_costs.is_empty() {
            return Err(anyhow::anyhow!("No valid workers found"));
        }

        // Find workers with maximum cost
        let max_cost = worker_costs
            .values()
            .map(|cost| *cost)
            .fold(f64::NEG_INFINITY, f64::max);
        let best_workers: Vec<WorkerId> = worker_costs
            .into_iter()
            .filter(|(_, cost)| (cost - max_cost).abs() < f64::EPSILON)
            .map(|(id, _)| id)
            .collect();

        // Randomly select one of the best workers
        let index = rand::random::<u32>() % (best_workers.len() as u32);
        let index = index as usize;
        match best_workers.get(index) {
            Some(worker_id) => {
                debug!("Selected worker {} with cost {}", worker_id, max_cost);
                Ok(*worker_id)
            }
            None => Err(anyhow::anyhow!("Failed to select a worker")),
        }
    }
}

/// Calculate the cost function for a worker based on the formula:
/// 2 * overlap * block_size / len_of_tokens - gpu_cache_usage_perc - num_requests_waiting / max(num_requests_waiting)
fn calculate_worker_cost(
    overlap: u32,
    block_size: usize,
    tokens_len: usize,
    gpu_cache_usage_perc: f32,
    num_requests_waiting: u64,
    max_requests_waiting: u64,
) -> f64 {
    let overlap_term = (overlap as f64) * (block_size as f64) / (tokens_len as f64);
    let gpu_term = gpu_cache_usage_perc as f64;
    let queue_term = (num_requests_waiting as f64) / (max_requests_waiting as f64);

    2.0 * overlap_term - gpu_term - queue_term
}

/// [gluo TODO] 'collect_endpoints' is from component/metrics,
/// should consolidate these functions into generic metrics aggregator
/// functions and shared by KvMetricsAggregator and component/metrics.
/// Collect endpoints from a component
pub async fn collect_endpoints(
    component: &Component,
    subject: &str,
    timeout: Duration,
) -> Result<Vec<EndpointInfo>> {
    // Collect stats from each backend
    let stream = component.scrape_stats(timeout).await?;

    // Filter the stats by the service subject
    let endpoints = stream
        .into_endpoints()
        .filter(|e| e.subject.starts_with(subject))
        .collect::<Vec<_>>();
    tracing::debug!("Endpoints: {endpoints:?}");

    if endpoints.is_empty() {
        tracing::warn!("No endpoints found matching subject {subject}");
    }

    Ok(endpoints)
}

pub async fn collect_endpoints_task(
    component: Component,
    ep_tx: tokio::sync::mpsc::Sender<ProcessedEndpoints>,
    cancel: CancellationToken,
) {
    let backoff_delay = Duration::from_millis(100);
    let scrape_timeout = Duration::from_millis(300);
    let endpoint = component.endpoint(KV_METRICS_ENDPOINT);
    let service_subject = endpoint.subject();

    loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                tracing::debug!("cancellation token triggered");
                break;
            }
            _ = tokio::time::sleep(backoff_delay) => {
                tracing::trace!("collecting endpoints for service: {}", service_subject);
                let unfiltered_endpoints =
                    match collect_endpoints(&component, &service_subject, scrape_timeout).await
                    {
                        Ok(v) => v,
                        Err(e) => {
                            tracing::warn!("Failed to retrieve endpoints for {}: {:?}", service_subject, e);
                            continue;
                        }
                    };
                tracing::debug!("unfiltered endpoints: {:?}", unfiltered_endpoints);

                let endpoints: Vec<Endpoint> = unfiltered_endpoints
                    .into_iter()
                    .filter(|s| s.data.is_some())
                    .filter_map(|s|
                        match s.data.unwrap().decode::<ForwardPassMetrics>() {
                            Ok(data) => Some(Endpoint {
                                name: s.name,
                                subject: s.subject,
                                data,
                            }),
                            Err(e) => {
                                tracing::debug!("skip endpoint data that can't be parsed as ForwardPassMetrics: {:?}", e);
                                None
                            }
                        }
                    )
                    .collect();
                tracing::debug!("endpoints: {:?}", endpoints);

                tracing::trace!(
                    "found {} endpoints for service: {}",
                    endpoints.len(),
                    service_subject
                );

                let processed = ProcessedEndpoints::new(endpoints);
                if ep_tx.send(processed).await.is_err() {
                    tracing::trace!("failed to send processed endpoints; shutting down");
                    break;
                }
            }
        }
    }
}
