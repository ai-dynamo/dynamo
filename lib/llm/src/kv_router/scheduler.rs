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
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

use super::protocols::WorkerSelectionResult;
use super::WorkerSelector;
use crate::kv_router::indexer::OverlapScores;
use crate::kv_router::indexer::WorkerId;
use crate::kv_router::protocols::LoadMetrics;
use crate::kv_router::scoring::ProcessedEndpoints;
use crate::kv_router::sequence::ActiveSequencesMultiWorker;
use crate::kv_router::KvRouterConfig;
use crate::kv_router::KV_HIT_RATE_SUBJECT;
use crate::tokens::TokenBlockSequence;

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
    pub data: LoadMetrics,
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
    sequences: Arc<Mutex<ActiveSequencesMultiWorker>>,
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

        let sequences = Arc::new(Mutex::new(ActiveSequencesMultiWorker::new(
            block_size as usize,
            endpoints.worker_ids(),
        )));
        let sequences_clone = sequences.clone();

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
                        let mut sequences_guard = sequences_clone.lock().await;
                        endpoints.update_active_blocks_all(sequences_guard.update_workers(endpoints.worker_ids()));
                        continue 'outer;
                    }
                };
                loop {
                    match selector.select_worker(&endpoints, &request, block_size) {
                        Ok(selection) => {
                            if let Err(e) = event_tx.send(KVHitRateEvent {
                                worker_id: selection.worker_id,
                                isl_blocks: selection.required_blocks as usize,
                                overlap_blocks: selection.overlap_blocks,
                            }) {
                                tracing::warn!("Failed to send KV hit rate event: {:?}", e);
                            }
                            request.respond(selection.worker_id);
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

        Ok(KvScheduler {
            request_tx,
            sequences,
        })
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

    /// Add a new request with its initial tokens to a specific worker
    pub async fn add_request(
        &self,
        request_id: String,
        token_sequence: TokenBlockSequence,
        worker_id: WorkerId,
    ) -> usize {
        let mut sequences = self.sequences.lock().await;
        sequences.add_request(request_id, token_sequence, worker_id)
    }

    /// Push a token to a specific request's sequence
    pub async fn push(&self, request_id: &String, token: u32) -> usize {
        let mut sequences = self.sequences.lock().await;
        sequences.push(request_id, token)
    }

    /// Free all blocks associated with a request
    pub async fn free(&self, request_id: &String) -> usize {
        let mut sequences = self.sequences.lock().await;
        sequences.free(request_id)
    }
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
        let potential_active_blocks = workers.active_blocks();

        let mut worker_logits = HashMap::new();
        let mut max_logit = f64::NEG_INFINITY;

        // Calculate logits for each worker
        for (worker_id, _) in workers.endpoints.iter() {
            let cached_blocks = request.overlap.scores.get(worker_id).copied().unwrap_or(0) as f64;
            let prefill_blocks = request_blocks as f64 - cached_blocks;

            let decode_blocks = *potential_active_blocks.get(worker_id).unwrap() as f64;

            // Calculate logit (lower is better)
            let logit = self.kv_router_config.overlap_score_weight * prefill_blocks + decode_blocks;
            max_logit = max_logit.max(logit);

            worker_logits.insert(*worker_id, logit);

            tracing::info!(
                "Formula for {worker_id}: {logit:.3} = {:.1} * {prefill_blocks:.3} + {decode_blocks:.3}",
                self.kv_router_config.overlap_score_weight,
            );
        }

        // Normalize by dividing by max value
        for logit in worker_logits.values_mut() {
            *logit /= max_logit;
        }

        // Use softmax sampling to select worker
        let temperature = self.kv_router_config.temperature; // You can make this configurable if needed
        let best_worker_id = softmax_sample(&worker_logits, temperature);

        let overlap_blocks = request
            .overlap
            .scores
            .get(&best_worker_id)
            .copied()
            .unwrap_or(0) as usize;
        let best_logit = worker_logits[&best_worker_id];

        tracing::info!(
            "Selected worker: {}, normalized logit: {:.3}",
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
}
