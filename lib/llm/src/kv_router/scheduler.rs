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

use dynamo_runtime::component::Namespace;
use dynamo_runtime::traits::events::EventPublisher;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::borrow::BorrowMut;
use std::collections::HashMap;

use super::protocols::WorkerSelectionResult;
use super::WorkerSelector;
use crate::kv_router::indexer::OverlapScores;
pub use crate::kv_router::protocols::ForwardPassMetrics;
use crate::kv_router::scoring::ProcessedEndpoints;
use crate::kv_router::KvRouterConfig;
use crate::kv_router::KV_HIT_RATE_SUBJECT;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KVHitRateEvent {
    pub worker_id: i64,
    pub isl_blocks: usize,
    pub overlap_blocks: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum KvSchedulerError {
    #[error("no endpoints aviailable to route work")]
    NoEndpoints,

    #[error("all workers busy")]
    AllWorkersBusy,

    #[error("endpoint subscriber shutdown")]
    SubscriberShutdown,
}

/// [gluo FIXME] exactly the same as EndpointInfo except that 'data'
/// is cleaned (not optional)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Endpoint {
    pub name: String,
    pub subject: String,
    pub data: ForwardPassMetrics,
}

impl Endpoint {
    pub fn worker_id(&self) -> i64 {
        i64::from_str_radix(
            self.subject
                .split("-")
                .last()
                .expect("invalid subject")
                .to_string()
                .as_str(),
            16,
        )
        .expect("invalid worker id")
    }
}

pub struct SchedulingRequest {
    pub isl_tokens: usize,
    pub overlap: OverlapScores,
    resp_tx: tokio::sync::oneshot::Sender<i64>,
}

impl SchedulingRequest {
    pub fn respond(self, worker_id: i64) {
        if self.resp_tx.send(worker_id).is_err() {
            tracing::trace!("failed to send response to requestor");
        }
    }
}

pub struct KvScheduler {
    request_tx: tokio::sync::mpsc::Sender<SchedulingRequest>,
}

impl KvScheduler {
    pub async fn start(
        ns: Namespace,
        block_size: usize,
        endpoints_rx: tokio::sync::watch::Receiver<ProcessedEndpoints>,
        selector: Option<Box<dyn WorkerSelector + Send + Sync>>,
    ) -> Result<Self, KvSchedulerError> {
        let selector = selector.unwrap_or(Box::new(DefaultWorkerSelector::default()));
        let mut endpoints_rx = endpoints_rx;
        let mut endpoints: ProcessedEndpoints = endpoints_rx.borrow_and_update().clone();

        let (event_tx, event_rx) = tokio::sync::mpsc::unbounded_channel::<KVHitRateEvent>();
        tokio::spawn(async move {
            let mut event_rx = event_rx;
            while let Some(event) = event_rx.recv().await {
                if let Err(e) = ns.publish(KV_HIT_RATE_SUBJECT, &event).await {
                    tracing::warn!("Failed to publish KV hit rate event: {:?}", e);
                }
            }
        });

        // Channel to accept new scheduling requests
        let (request_tx, request_rx) = tokio::sync::mpsc::channel::<SchedulingRequest>(1024);
        // Background task to handle scheduling requests
        tokio::spawn(async move {
            let mut request: SchedulingRequest;
            let mut request_rx = request_rx;
            tracing::trace!("scheduler background task started");

            'outer: loop {
                request = tokio::select! {
                    biased;

                    new_request = request_rx.recv() => {
                        match new_request {
                            Some(new_request) => {
                                tracing::trace!("received request to be scheduled");
                                new_request
                            },
                            None => {
                                tracing::trace!("scheduler shutdown");
                                break 'outer;
                            }
                        }
                    }

                    _ = endpoints_rx.changed() => {
                        endpoints = endpoints_rx.borrow_and_update().clone();
                        continue 'outer;
                    }
                };
                loop {
                    match selector.select_worker(&endpoints, &request, block_size) {
                        Ok(selection) => {
                            let worker_id = process_worker_selection(
                                endpoints.borrow_mut(),
                                selection,
                                &event_tx,
                            );
                            request.respond(worker_id);
                            continue 'outer;
                        }
                        Err(KvSchedulerError::AllWorkersBusy) => {
                            tracing::trace!("all workers busy; waiting for more capacity");
                            match endpoints_rx.changed().await {
                                Ok(_) => {}
                                Err(e) => {
                                    tracing::error!("error waiting for endpoints change: {:?}", e);
                                    break 'outer;
                                }
                            };
                            endpoints = endpoints_rx.borrow_and_update().clone();
                        }
                        Err(e) => {
                            tracing::error!("error scheduling request: {:?}", e);
                            break 'outer;
                        }
                    }
                }
            }

            tracing::trace!("background endpoint subscriber shutting down");
        });

        Ok(KvScheduler { request_tx })
    }

    pub async fn schedule(
        &self,
        overlap: OverlapScores,
        isl_tokens: usize,
    ) -> Result<i64, KvSchedulerError> {
        let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
        let request = SchedulingRequest {
            isl_tokens,
            overlap,
            resp_tx,
        };
        self.request_tx
            .send(request)
            .await
            .map_err(|_| KvSchedulerError::SubscriberShutdown)?;
        let res = resp_rx
            .await
            .map_err(|_| KvSchedulerError::SubscriberShutdown)?;
        Ok(res)
    }
}

// This becomes the driver function that handles the selection result
pub fn process_worker_selection(
    workers: &mut ProcessedEndpoints,
    selection: WorkerSelectionResult,
    event_tx: &tokio::sync::mpsc::UnboundedSender<KVHitRateEvent>,
) -> i64 {
    let worker = workers
        .endpoints
        .get_mut(&selection.worker_id)
        .expect("worker not found");

    // Update worker state predictively
    // Will be overwritten on next polling of metrics
    worker.data.num_requests_waiting += 1;
    // Assumes radix attention so KV load is only incremented by uncached blocks
    // overlap_blocks can be bigger than required_blocks. I don't know if that's a bug or not.
    worker.data.kv_active_blocks += selection
        .required_blocks
        .saturating_sub(selection.overlap_blocks as u64);

    // Emit event
    if let Err(e) = event_tx.send(KVHitRateEvent {
        worker_id: selection.worker_id,
        isl_blocks: selection.required_blocks as usize,
        overlap_blocks: selection.overlap_blocks,
    }) {
        tracing::warn!("Failed to send KV hit rate event: {:?}", e);
    }

    selection.worker_id
}

// Default implementation matching the Python _cost_function
#[derive(Debug, Clone, Default)]
pub struct DefaultWorkerSelector {
    pub kv_router_config: KvRouterConfig,
}

impl DefaultWorkerSelector {
    pub fn new(kv_router_config: Option<KvRouterConfig>) -> Self {
        Self {
            kv_router_config: kv_router_config.unwrap_or_default(),
        }
    }
}

impl WorkerSelector for DefaultWorkerSelector {
    fn select_worker(
        &self,
        workers: &ProcessedEndpoints,
        request: &SchedulingRequest,
        block_size: usize,
    ) -> Result<WorkerSelectionResult, KvSchedulerError> {
        assert!(request.isl_tokens > 0);

        if workers.endpoints.is_empty() {
            return Err(KvSchedulerError::NoEndpoints);
        }

        let mut worker_scores = HashMap::new();
        let mut max_waiting = 0.0;

        // Calculate worker scores and find max waiting requests
        for (worker_id, ep) in workers.endpoints.iter() {
            // Calculate score similar to Python version
            if let Some(score) = request.overlap.scores.get(worker_id) {
                let score = *score as f64 * block_size as f64 / request.isl_tokens as f64;
                worker_scores.insert(worker_id, score);
            }

            // Track max waiting requests
            max_waiting = f64::max(max_waiting, ep.data.num_requests_waiting as f64);
        }

        // make immutable
        let worker_scores = worker_scores;
        let max_waiting = max_waiting;

        // Calculate logits for each worker
        let mut best_logit = f64::NEG_INFINITY;
        let mut best_workers = Vec::new();

        for (worker_id, ep) in workers.endpoints.iter() {
            let worker_id = *worker_id;

            // Get score or default to 0.0
            let score = worker_scores.get(&worker_id).copied().unwrap_or(0.0);

            // Calculate normalized metrics
            let gpu_cache_usage = ep.data.gpu_cache_usage_perc as f64;
            let normalized_waiting = if max_waiting > 0.0 {
                ep.data.num_requests_waiting as f64 / max_waiting
            } else {
                0.0
            };

            // Calculate logit using same formula as Python
            let logit = self.kv_router_config.overlap_score_weight * score
                - self.kv_router_config.gpu_cache_usage_weight * gpu_cache_usage
                - self.kv_router_config.waiting_requests_weight * normalized_waiting;

            tracing::trace!(
                "Formula for {worker_id}: {logit:.3} = {:.1} * {score:.3} - {:.1} * {gpu_cache_usage:.3} - {:.1} * {normalized_waiting:.3}",
                self.kv_router_config.overlap_score_weight,
                self.kv_router_config.gpu_cache_usage_weight,
                self.kv_router_config.waiting_requests_weight,
            );

            // Track best workers
            match logit.partial_cmp(&best_logit) {
                Some(std::cmp::Ordering::Greater) => {
                    best_logit = logit;
                    best_workers.clear();
                    best_workers.push(worker_id);
                }
                Some(std::cmp::Ordering::Equal) => {
                    best_workers.push(worker_id);
                }
                _ => {}
            }
        }

        // Return early if no valid workers found
        if best_workers.is_empty() {
            return Err(KvSchedulerError::NoEndpoints);
        } else if best_logit == 0.0 {
            tracing::debug!("best worker logit is 0");
        }

        let worker_id = if best_workers.len() == 1 {
            best_workers[0]
        } else {
            // Randomly select from best workers
            let mut rng = rand::rng();
            best_workers[rng.random_range(0..best_workers.len())]
        };

        // Lower to trace level eventually. Nice to see KV routing working for now.
        tracing::debug!("Selected worker: {worker_id}, logit: {best_logit:.3}");

        // Log selection metrics
        let total_blocks = std::cmp::max(request.isl_tokens / block_size, 1) as u64;
        let overlap_blocks = request.overlap.scores.get(&worker_id).copied().unwrap_or(0) as usize;

        Ok(WorkerSelectionResult {
            worker_id,
            required_blocks: total_blocks,
            overlap_blocks,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a worker endpoint
    fn create_endpoint(
        worker_id: i64,
        gpu_cache_usage_perc: f32,
        num_requests_waiting: u64,
    ) -> Endpoint {
        Endpoint {
            name: format!("worker-{}", worker_id),
            subject: format!("worker-subject-{:x}", worker_id),
            data: ForwardPassMetrics {
                gpu_cache_usage_perc,
                num_requests_waiting,
                // Other fields can be default initialized for this test
                ..Default::default()
            },
        }
    }

    // Helper to create ProcessedEndpoints
    struct WorkerInfo {
        id: i64,
        usage: f32,
        waiting: u64,
    }
    fn create_workers(workers: Vec<WorkerInfo>) -> ProcessedEndpoints {
        let mut endpoints = HashMap::new();
        for worker in workers {
            endpoints.insert(
                worker.id,
                create_endpoint(worker.id, worker.usage, worker.waiting),
            );
        }
        ProcessedEndpoints {
            endpoints,
            load_avg: 0.0,
            load_std: 0.0,
        }
    }

    // Helper to create a scheduling request
    struct WorkerOverlap {
        worker_id: i64,
        overlap_blocks: u32,
    }
    fn create_request(overlaps: Vec<WorkerOverlap>, isl_tokens: usize) -> SchedulingRequest {
        SchedulingRequest {
            isl_tokens,
            overlap: OverlapScores {
                scores: overlaps
                    .into_iter()
                    .map(|wo| (wo.worker_id, wo.overlap_blocks))
                    .collect(),
                frequencies: vec![],
            },
            resp_tx: tokio::sync::oneshot::channel().0,
        }
    }

    #[test]
    fn test_select_worker_basic() {
        // Setup workers
        let workers = create_workers(vec![
            WorkerInfo {
                id: 1,
                usage: 0.50,
                waiting: 1,
            },
            WorkerInfo {
                id: 2,
                usage: 0.80,
                waiting: 0,
            },
        ]);

        // Setup request: 100 tokens, block_size=20 (5 blocks)
        let request = create_request(
            vec![
                WorkerOverlap {
                    worker_id: 1,
                    overlap_blocks: 3,
                },
                WorkerOverlap {
                    worker_id: 2,
                    overlap_blocks: 4,
                },
            ],
            100,
        );
        let selector = DefaultWorkerSelector::new(None);
        let block_size = 20;

        // Execute selection
        let result = selector
            .select_worker(&workers, &request, block_size)
            .expect("Should select a worker");
        // Worker 2 should win because:
        // Worker1: 2.0 * 0.600 - 1.0 * 0.500 - 1.0 * 1.000 = -0.3
        // Worker2: 2.0 * 0.800 - 1.0 * 0.800 - 1.0 * 0.000 = 0.8
        assert_eq!(result.worker_id, 2);
        assert_eq!(result.required_blocks, 5); // 100 tokens / 20 block_size
        assert_eq!(result.overlap_blocks, 4);
    }

    #[test]
    fn test_no_endpoints() {
        let workers = create_workers(vec![]);
        let request = create_request(vec![], 100);
        let selector = DefaultWorkerSelector::new(None);
        let block_size = 20;

        match selector.select_worker(&workers, &request, block_size) {
            Err(KvSchedulerError::NoEndpoints) => {} // Expected
            _ => panic!("Should return NoEndpoints error"),
        }
    }

    #[test]
    fn test_no_overlap_scores() {
        // Workers exist but request has no overlap scores
        let workers = create_workers(vec![WorkerInfo {
            id: 1,
            usage: 0.50,
            waiting: 1,
        }]);
        let request = create_request(vec![], 100); // No overlaps
        let selector = DefaultWorkerSelector::new(None);
        let block_size = 20;

        let result = selector
            .select_worker(&workers, &request, block_size)
            .expect("Should fallback to selecting worker");

        // Worker1 should be selected with 0 overlap
        assert_eq!(result.worker_id, 1);
        assert_eq!(result.overlap_blocks, 0);
    }

    #[test]
    fn test_custom_weights() {
        // Setup workers
        let workers = create_workers(vec![
            WorkerInfo {
                id: 1,
                usage: 0.50,
                waiting: 1,
            },
            WorkerInfo {
                id: 2,
                usage: 0.80,
                waiting: 0,
            },
        ]);

        // Custom config with high priority on GPU usage
        let config = KvRouterConfig {
            gpu_cache_usage_weight: 10.0, // Very high weight
            overlap_score_weight: 2.0,    // just current defaults
            waiting_requests_weight: 1.0,
        };
        let selector = DefaultWorkerSelector::new(Some(config));
        let request = create_request(
            vec![
                WorkerOverlap {
                    worker_id: 1,
                    overlap_blocks: 3,
                },
                WorkerOverlap {
                    worker_id: 2,
                    overlap_blocks: 4,
                },
            ],
            100,
        );
        let block_size = 20;

        let result = selector
            .select_worker(&workers, &request, block_size)
            .expect("Should select worker");

        assert_eq!(result.worker_id, 1);
    }
}
