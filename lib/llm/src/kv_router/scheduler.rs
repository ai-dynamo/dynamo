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
use std::borrow::BorrowMut;
use std::collections::HashMap;

use crate::kv_router::indexer::OverlapScores;
pub use crate::kv_router::protocols::ForwardPassMetrics;
use crate::kv_router::scoring::ProcessedEndpoints;
use crate::kv_router::KV_HIT_RATE_SUBJECT;

use super::protocols::{KVHitRateEvent, WorkerDp, WorkerSelectionResult};
use super::WorkerSelector;

#[derive(Debug, thiserror::Error)]
pub enum KvSchedulerError {
    #[error("no endpoints aviailable to route work")]
    NoEndpoints,

    #[error("all workers busy")]
    AllWorkersBusy,

    #[error("endpoint subscriber shutdown")]
    SubscriberShutdown,
}

pub struct SchedulingRequest {
    pub isl_tokens: usize,
    pub overlap: OverlapScores<WorkerDp>,
    resp_tx: tokio::sync::oneshot::Sender<WorkerDp>,
}

impl SchedulingRequest {
    pub fn respond(self, identifier: WorkerDp) {
        if self.resp_tx.send(identifier).is_err() {
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

        let (event_tx, event_rx) =
            tokio::sync::mpsc::unbounded_channel::<KVHitRateEvent<WorkerDp>>();
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
        overlap: OverlapScores<WorkerDp>,
        isl_tokens: usize,
    ) -> Result<WorkerDp, KvSchedulerError> {
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
    selection: WorkerSelectionResult<WorkerDp>,
    event_tx: &tokio::sync::mpsc::UnboundedSender<KVHitRateEvent<WorkerDp>>,
) -> WorkerDp {
    let worker = workers
        .endpoints
        .get_mut(&selection.worker.worker_id)
        .expect("worker not found");

    let dp_rank = selection.worker.dp_rank.unwrap_or(0) as usize;

    // Update worker state predictively
    // Will be overwritten on next polling of metrics
    worker.data[dp_rank].num_requests_waiting += 1;
    // Assumes radix attention so KV load is only incremented by uncached blocks
    // overlap_blocks can be bigger than required_blocks. I don't know if that's a bug or not.
    worker.data[dp_rank].kv_active_blocks += selection
        .required_blocks
        .saturating_sub(selection.overlap_blocks as u64);

    // Emit event
    if let Err(e) = event_tx.send(KVHitRateEvent {
        worker: selection.worker,
        isl_blocks: selection.required_blocks as usize,
        overlap_blocks: selection.overlap_blocks,
    }) {
        tracing::warn!("Failed to send KV hit rate event: {:?}", e);
    }

    selection.worker
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
    ) -> Result<WorkerSelectionResult<WorkerDp>, KvSchedulerError> {
        assert!(request.isl_tokens > 0);

        if workers.endpoints.is_empty() {
            return Err(KvSchedulerError::NoEndpoints);
        }

        let mut worker_scores = HashMap::new();
        let mut max_waiting = 0.0;

        // Calculate worker scores and find max waiting requests
        for (worker_id, ep) in workers.endpoints.iter() {
            for dp_rank in ep.data.iter().map(|metrics| metrics.data_parallel_rank) {
                let worker_with_dp_rank = WorkerDp {
                    worker_id: *worker_id,
                    dp_rank,
                };
                if let Some(score) = request.overlap.scores.get(&worker_with_dp_rank) {
                    let score = *score as f64 * block_size as f64 / request.isl_tokens as f64;
                    worker_scores.insert(worker_with_dp_rank, score);
                }
                // Track max waiting requests
                max_waiting = f64::max(
                    max_waiting,
                    ep.data[dp_rank.unwrap_or(0) as usize].num_requests_waiting as f64,
                );
            }
        }

        // make immutable
        let worker_scores = worker_scores;
        let max_waiting = max_waiting;

        // Calculate logits for each worker
        let mut best_logit = f64::NEG_INFINITY;
        let mut best_workers = Vec::new();

        for (worker_id, ep) in workers.endpoints.iter() {
            let worker_id = *worker_id;
            for fwd_pass_metrics in ep.data.iter() {
                let dp_rank = fwd_pass_metrics.data_parallel_rank;
                let worker_with_dp_rank = WorkerDp { worker_id, dp_rank };

                // Get score or default to 0.0
                let score = worker_scores
                    .get(&worker_with_dp_rank)
                    .copied()
                    .unwrap_or(0.0);

                // Calculate normalized metrics
                let gpu_cache_usage =
                    ep.data[dp_rank.unwrap_or(0) as usize].gpu_cache_usage_perc as f64;
                let normalized_waiting = if max_waiting > 0.0 {
                    ep.data[dp_rank.unwrap_or(0) as usize].num_requests_waiting as f64 / max_waiting
                } else {
                    0.0
                };

                // Calculate logit using same formula as Python
                let logit = 2.0 * score - gpu_cache_usage - normalized_waiting;

                tracing::trace!(
                    "Formula for {worker_id}: {logit:.3} = 2.0 * {score:.3} - {gpu_cache_usage:.3} - {normalized_waiting:.3}",
                );

                // Track best workers
                match logit.partial_cmp(&best_logit) {
                    Some(std::cmp::Ordering::Greater) => {
                        best_logit = logit;
                        best_workers.clear();
                        best_workers.push(worker_with_dp_rank);
                    }
                    Some(std::cmp::Ordering::Equal) => {
                        best_workers.push(worker_with_dp_rank);
                    }
                    _ => {}
                }
            }
        }

        // Return early if no valid workers found
        if best_workers.is_empty() {
            return Err(KvSchedulerError::NoEndpoints);
        } else if best_logit == 0.0 {
            tracing::debug!("best worker logit is 0");
        }

        let best_worker_and_dp = if best_workers.len() == 1 {
            best_workers[0]
        } else {
            // Randomly select from best workers
            let mut rng = rand::rng();
            best_workers[rng.random_range(0..best_workers.len())]
        };

        // Lower to trace level eventually. Nice to see KV routing working for now.
        tracing::debug!("Selected worker: {best_worker_and_dp:?}, logit: {best_logit:.3}");

        // Log selection metrics
        let total_blocks = std::cmp::max(request.isl_tokens / block_size, 1) as u64;
        let overlap_blocks = request
            .overlap
            .scores
            .get(&best_worker_and_dp)
            .copied()
            .unwrap_or(0) as usize;

        Ok(WorkerSelectionResult {
            worker: best_worker_and_dp,
            required_blocks: total_blocks,
            overlap_blocks,
        })
    }
}
