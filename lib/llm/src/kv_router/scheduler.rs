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

use crate::kv_router::indexer::OverlapScores;
pub use crate::kv_router::protocols::ForwardPassMetrics;
use crate::kv_router::scoring::ProcessedEndpoints;
use crate::kv_router::KV_HIT_RATE_SUBJECT;

use super::protocols::WorkerSelectionResult;
use super::WorkerSelector;

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
        let selector = selector.unwrap_or(Box::new(DefaultWorkerSelector));
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
        tracing::debug!("scheduler starting");
        // Background task to handle scheduling requests
        tokio::spawn(async move {
            let mut request: SchedulingRequest;
            let mut request_rx = request_rx;
            tracing::debug!("scheduler background task started");

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
                tracing::debug!("selected");
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
        tracing::debug!("before sending request");
        self.request_tx
            .send(request)
            .await
            .map_err(|_| KvSchedulerError::SubscriberShutdown)?;
        tracing::debug!("after sending request");

        let res = resp_rx
            .await
            .map_err(|_| KvSchedulerError::SubscriberShutdown)?;
        tracing::debug!("after receiving response");
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

    // Update worker state
    worker.data.request_active_slots += 1;
    worker.data.kv_active_blocks += selection.required_blocks - selection.overlap_blocks as u64;

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
#[derive(Default)]
pub struct DefaultWorkerSelector;

impl WorkerSelector for DefaultWorkerSelector {
    fn select_worker(
        &self,
        workers: &ProcessedEndpoints,
        request: &SchedulingRequest,
        block_size: usize,
    ) -> Result<WorkerSelectionResult, KvSchedulerError> {
        // 1. Handle edge cases
        if workers.endpoints.is_empty() {
            // No endpoints at all
            return Err(KvSchedulerError::NoEndpoints);
        }

        // If block_size == 0 or isl_tokens == 0, we consider that
        // no blocks are required. We'll still pick a worker if possible,
        // but the overlap-based score effectively becomes 0.
        let zero_tokens_or_blocks = (request.isl_tokens == 0) || (block_size == 0);
        let block_size_f = block_size as f64;
        let isl_tokens_f = request.isl_tokens as f64;

        // 2. Find the maximum request_active_slots to normalize usage
        let max_active = workers
            .endpoints
            .values()
            .map(|ep| ep.data.request_active_slots)
            .max()
            .unwrap_or(0);

        // If all endpoints have 0 active slots, we can either:
        // - Return an error (like the original code), or
        // - Still proceed with the logic (since we do have endpoints).
        if max_active == 0 {
            // Decide how you want to handle this scenario.
            // We'll return an error as the original code does:
            return Err(KvSchedulerError::NoEndpoints);
        }

        // 3. In a single pass, determine the best worker(s) based on the logit formula
        let mut best_logit = f64::NEG_INFINITY;
        let mut best_workers = Vec::new();

        for (worker_id, ep) in &workers.endpoints {
            // Overlap score (defaults to 0 if missing)
            let overlap_score = request.overlap.scores.get(worker_id).copied().unwrap_or(0) as f64;

            // If we have zero tokens or block size, we treat the effective 'score' as 0
            // because we can't allocate blocks or meaningfully normalize the overlap.
            let score = if zero_tokens_or_blocks {
                0.0
            } else {
                overlap_score * block_size_f / isl_tokens_f
            };

            // Avoid division by zero; skip or handle endpoints that have zero kv_total_blocks
            if ep.data.kv_total_blocks == 0 {
                continue;
            }

            let gpu_cache_usage = ep.data.kv_active_blocks as f64 / ep.data.kv_total_blocks as f64;
            let normalized_active = ep.data.request_active_slots as f64 / max_active as f64;

            let logit = 2.0 * score - gpu_cache_usage - normalized_active;

            tracing::info!(
                "Formula for {}: {:.3} = 2.0 * {:.3} - {:.3} - {:.3}",
                worker_id,
                logit,
                score,
                gpu_cache_usage,
                normalized_active
            );

            // Track best logit
            match logit.partial_cmp(&best_logit) {
                Some(std::cmp::Ordering::Greater) => {
                    best_logit = logit;
                    best_workers.clear();
                    best_workers.push(*worker_id);
                }
                Some(std::cmp::Ordering::Equal) => {
                    best_workers.push(*worker_id);
                }
                _ => {}
            }
        }

        // 4. Check if we found at least one valid endpoint
        if best_workers.is_empty() {
            return Err(KvSchedulerError::NoEndpoints);
        }

        // 5. If more than one, pick randomly among the best
        let worker_id = if best_workers.len() == 1 {
            best_workers[0]
        } else {
            let mut rng = rand::rng();
            best_workers[rng.random_range(0..best_workers.len())]
        };

        // 6. Calculate required blocks for the chosen endpoint
        //    If tokens == 0 or block_size == 0, the result is 0 blocks.
        let required_blocks = if zero_tokens_or_blocks {
            0
        } else {
            // As in the original code: `min(request.isl_tokens / block_size, 1) as u64`
            std::cmp::min(request.isl_tokens / block_size, 1) as u64
        };

        // Overlap blocks, if any
        let overlap_blocks = request
            .overlap
            .scores
            .get(&worker_id)
            .copied()
            .unwrap_or(0) as usize;

        tracing::info!("Selected worker: {}, logit: {:.3}", worker_id, best_logit);

        Ok(WorkerSelectionResult {
            worker_id,
            required_blocks,
            overlap_blocks,
        })
    }
}
