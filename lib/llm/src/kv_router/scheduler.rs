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
        block_size: u32,
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

    worker.data.kv_stats.kv_active_blocks += selection
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

// Helper function for softmax sampling
fn softmax_sample(logits: &HashMap<i64, f64>, temperature: f64) -> i64 {
    if logits.is_empty() {
        panic!("Empty logits for softmax sampling");
    }

    let keys: Vec<_> = logits.keys().copied().collect();
    let values: Vec<_> = logits.values().copied().collect();

    // Find min and max for normalization
    let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let probabilities = if min_val == max_val {
        // All values are the same, uniform probability
        vec![1.0 / keys.len() as f64; keys.len()]
    } else {
        // Normalize values
        let normalized: Vec<_> = values
            .iter()
            .map(|&v| {
                let norm = v / (max_val - min_val);
                // Lower is better, so negate
                -norm
            })
            .collect();

        // Apply temperature and softmax
        let scaled: Vec<_> = normalized.iter().map(|&v| v / temperature).collect();

        let max_scaled = scaled.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_values: Vec<_> = scaled.iter().map(|&v| (v - max_scaled).exp()).collect();

        let sum_exp: f64 = exp_values.iter().sum();
        exp_values.iter().map(|&v| v / sum_exp).collect()
    };

    // Sample from the probability distribution
    let mut rng = rand::rng();
    let sample: f64 = rng.random();

    let mut cumsum = 0.0;
    for (i, &prob) in probabilities.iter().enumerate() {
        cumsum += prob;
        if sample <= cumsum {
            return keys[i];
        }
    }

    // Fallback to last key (shouldn't normally reach here)
    keys[keys.len() - 1]
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
        block_size: u32,
    ) -> Result<WorkerSelectionResult, KvSchedulerError> {
        assert!(request.isl_tokens > 0);

        if workers.endpoints.is_empty() {
            return Err(KvSchedulerError::NoEndpoints);
        }

        let request_blocks = request.isl_tokens.div_ceil(block_size as usize);
        let mut worker_logits = HashMap::new();

        // Calculate logits for each worker
        for (worker_id, ep) in workers.endpoints.iter() {
            let worker_id = *worker_id;

            // Get overlap blocks for this worker
            let overlap_blocks =
                request.overlap.scores.get(&worker_id).copied().unwrap_or(0) as f64;
            let new_blocks = request_blocks as f64 - overlap_blocks;

            let kv_total_blocks = ep.data.kv_total_blocks as f64;
            assert!(kv_total_blocks > 0.0);

            let normalized_new_blocks = new_blocks / kv_total_blocks;
            let gpu_cache_usage = (ep.data.kv_stats.kv_active_blocks as f64) / kv_total_blocks;
            let num_requests_waiting = ep.data.worker_stats.num_requests_waiting as f64;

            // Calculate logit (lower is better)
            let logit = self.kv_router_config.overlap_score_weight * normalized_new_blocks
                + self.kv_router_config.gpu_cache_usage_weight * gpu_cache_usage
                + self.kv_router_config.waiting_requests_weight * num_requests_waiting;

            worker_logits.insert(worker_id, logit);

            tracing::info!(
                "Formula for {worker_id}: {logit:.3} = {:.1} * {normalized_new_blocks:.3} + {:.1} * {gpu_cache_usage:.3} + {:.1} * {num_requests_waiting:.3}",
                self.kv_router_config.overlap_score_weight,
                self.kv_router_config.gpu_cache_usage_weight,
                self.kv_router_config.waiting_requests_weight,
            );
        }

        // Return early if no valid workers found
        if worker_logits.is_empty() || worker_logits.values().all(|&v| v == 0.0) {
            tracing::warn!("All worker logits are zero. Fallback to random routing.");
            // Pick random worker
            let mut rng = rand::rng();
            let worker_ids: Vec<_> = workers.endpoints.keys().copied().collect();
            let worker_id = worker_ids[rng.random_range(0..worker_ids.len())];
            let overlap_blocks =
                request.overlap.scores.get(&worker_id).copied().unwrap_or(0) as usize;
            return Ok(WorkerSelectionResult {
                worker_id,
                required_blocks: request_blocks as u64,
                overlap_blocks,
            });
        }

        // Use softmax sampling to select worker
        let temperature = 1.0; // You can make this configurable if needed
        let best_worker_id = softmax_sample(&worker_logits, temperature);

        let overlap_blocks = request
            .overlap
            .scores
            .get(&best_worker_id)
            .copied()
            .unwrap_or(0) as usize;
        let best_logit = worker_logits[&best_worker_id];

        tracing::info!(
            "Selected worker: {}, logit: {:.3}",
            best_worker_id,
            best_logit
        );

        Ok(WorkerSelectionResult {
            worker_id: best_worker_id,
            required_blocks: request_blocks as u64,
            overlap_blocks,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_sample_single_key() {
        // Test that with a single key, softmax_sample always returns that key
        let mut logits = HashMap::new();
        let worker_id = 42;
        logits.insert(worker_id, 0.5); // The value doesn't matter

        // Test with different temperatures
        for temperature in &[0.1, 1.0, 10.0] {
            let result = softmax_sample(&logits, *temperature);
            assert_eq!(result, worker_id, "Should return the only available worker");
        }

        // Test with different logit values
        logits.clear();
        logits.insert(worker_id, -100.0); // Very negative value
        assert_eq!(softmax_sample(&logits, 1.0), worker_id);

        logits.clear();
        logits.insert(worker_id, 100.0); // Very positive value
        assert_eq!(softmax_sample(&logits, 1.0), worker_id);

        logits.clear();
        logits.insert(worker_id, 0.0); // Zero value
        assert_eq!(softmax_sample(&logits, 1.0), worker_id);
    }

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

    // #[test]
    // fn test_select_worker_basic() {
    //     // Setup workers
    //     let workers = create_workers(vec![
    //         WorkerInfo {
    //             id: 1,
    //             usage: 0.50,
    //             waiting: 1,
    //         },
    //         WorkerInfo {
    //             id: 2,
    //             usage: 0.80,
    //             waiting: 0,
    //         },
    //     ]);

    //     // Setup request: 100 tokens, block_size=20 (5 blocks)
    //     let request = create_request(
    //         vec![
    //             WorkerOverlap {
    //                 worker_id: 1,
    //                 overlap_blocks: 3,
    //             },
    //             WorkerOverlap {
    //                 worker_id: 2,
    //                 overlap_blocks: 4,
    //             },
    //         ],
    //         100,
    //     );
    //     let selector = DefaultWorkerSelector::new(None);
    //     let block_size = 20;

    //     // Execute selection
    //     let result = selector
    //         .select_worker(&workers, &request, block_size)
    //         .expect("Should select a worker");
    //     // Worker 2 should win because:
    //     // Worker1: 2.0 * 0.600 - 1.0 * 0.500 - 1.0 * 1.000 = -0.3
    //     // Worker2: 2.0 * 0.800 - 1.0 * 0.800 - 1.0 * 0.000 = 0.8
    //     assert_eq!(result.worker_id, 2);
    //     assert_eq!(result.required_blocks, 5); // 100 tokens / 20 block_size
    //     assert_eq!(result.overlap_blocks, 4);
    // }

    // #[test]
    // fn test_no_overlap_scores() {
    //     // Workers exist but request has no overlap scores
    //     let workers = create_workers(vec![WorkerInfo {
    //         id: 1,
    //         usage: 0.50,
    //         waiting: 1,
    //     }]);
    //     let request = create_request(vec![], 100); // No overlaps
    //     let selector = DefaultWorkerSelector::new(None);
    //     let block_size = 20;

    //     let result = selector
    //         .select_worker(&workers, &request, block_size)
    //         .expect("Should fallback to selecting worker");

    //     // Worker1 should be selected with 0 overlap
    //     assert_eq!(result.worker_id, 1);
    //     assert_eq!(result.overlap_blocks, 0);
    // }

    // #[test]
    // fn test_custom_weights() {
    //     // Setup workers
    //     let workers = create_workers(vec![
    //         WorkerInfo {
    //             id: 1,
    //             usage: 0.50,
    //             waiting: 1,
    //         },
    //         WorkerInfo {
    //             id: 2,
    //             usage: 0.80,
    //             waiting: 0,
    //         },
    //     ]);

    //     // Custom config with high priority on GPU usage
    //     let config = KvRouterConfig {
    //         gpu_cache_usage_weight: 10.0, // Very high weight
    //         overlap_score_weight: 2.0,    // just current defaults
    //         waiting_requests_weight: 1.0,
    //     };
    //     let selector = DefaultWorkerSelector::new(Some(config));
    //     let request = create_request(
    //         vec![
    //             WorkerOverlap {
    //                 worker_id: 1,
    //                 overlap_blocks: 3,
    //             },
    //             WorkerOverlap {
    //                 worker_id: 2,
    //                 overlap_blocks: 4,
    //             },
    //         ],
    //         100,
    //     );
    //     let block_size = 20;

    //     let result = selector
    //         .select_worker(&workers, &request, block_size)
    //         .expect("Should select worker");

    //     assert_eq!(result.worker_id, 1);
    // }
}
