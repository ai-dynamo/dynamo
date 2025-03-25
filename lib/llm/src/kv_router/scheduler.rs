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
use serde::{Deserialize, Serialize};
use std::borrow::BorrowMut;
use std::cmp::min;

use crate::kv_router::indexer::OverlapScores;
pub use crate::kv_router::protocols::ForwardPassMetrics;
use crate::kv_router::scoring::ProcessedEndpoints;
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
    isl_tokens: usize,
    overlap: OverlapScores,
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
        endpoints_rx: tokio::sync::mpsc::Receiver<ProcessedEndpoints>,
        ns: Namespace,
        kv_block_size: usize,
    ) -> Result<Self, KvSchedulerError> {
        let mut endpoints_rx = endpoints_rx;

        tracing::trace!("awaiting the start of the background endpoint subscriber");
        let mut endpoints = match endpoints_rx.recv().await {
            Some(endpoints) => endpoints,
            None => {
                return Err(KvSchedulerError::SubscriberShutdown);
            }
        };

        // Channel to asynchronously publish metric events on
        let (event_tx, event_rx) = tokio::sync::mpsc::unbounded_channel::<KVHitRateEvent>();

        // Publisher task
        tokio::spawn(async move {
            let mut event_rx = event_rx;
            while let Some(event) = event_rx.recv().await {
                if let Err(e) = ns.publish(KV_HIT_RATE_SUBJECT, &event).await {
                    tracing::warn!("Failed to publish KV hit rate event: {:?}", e);
                }
            }
        });

        // Channel to accept new scheduling requests
        let (request_tx, request_rx) = tokio::sync::mpsc::channel::<SchedulingRequest>(16);
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

                    new_endpoints = endpoints_rx.recv() => {
                        match new_endpoints {
                            Some(new_endpoints) => {
                                tracing::trace!("updated endpoints");
                                endpoints = new_endpoints;
                                continue 'outer;
                            }
                            None => {
                                tracing::trace!("endpoint subscriber shutdown");
                                break 'outer;
                            }
                        }
                    }
                };
                tracing::debug!("selected");
                loop {
                    match select_worker(endpoints.borrow_mut(), &request, &event_tx, kv_block_size)
                    {
                        Ok(worker_id) => {
                            request.respond(worker_id);
                            continue 'outer;
                        }
                        Err(KvSchedulerError::AllWorkersBusy) => {
                            tracing::trace!("all workers busy; waiting for more capacity");
                            endpoints = match endpoints_rx.recv().await {
                                Some(endpoints) => endpoints,
                                None => {
                                    tracing::trace!("endpoint subscriber shutdown");
                                    break 'outer;
                                }
                            };
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

pub fn select_worker(
    workers: &mut ProcessedEndpoints,
    request: &SchedulingRequest,
    event_tx: &tokio::sync::mpsc::UnboundedSender<KVHitRateEvent>,
    kv_block_size: usize,
) -> Result<i64, KvSchedulerError> {
    // balance mode prioritizes balancing load across workers
    let balance_threshold: f64 = 0.1;
    let balance_mode = workers.load_std > balance_threshold * workers.load_avg;

    // Determine alpha based on mode
    let alpha = if balance_mode { 0.7 } else { 0.3 };
    let gamma = 0.1; // example tuning param

    // Compute each worker's score
    let mut best_index = None;
    let mut best_cost = f64::INFINITY;
    // [FIXME] REMOVE ONLY FOR TESTING
    if workers.endpoints.is_empty() {
        return Err(KvSchedulerError::NoEndpoints);
    }

    for (i, w) in workers.endpoints.iter().enumerate() {
        // Exclude workers that are at capacity
        if w.data.request_active_slots >= w.data.request_total_slots
            || w.data.kv_active_blocks >= w.data.kv_total_blocks
        {
            continue;
        }

        let kv_load_ratio = w.data.kv_active_blocks as f64 / w.data.kv_total_blocks as f64;
        let load_deviation = kv_load_ratio - workers.load_avg;

        // [FIXME] multiple endpoints of the same worker cause out of bound error
        let worker_id = workers.worker_ids[i];
        let overlap_score = request.overlap.scores.get(&worker_id).map_or(0, |x| *x);
        let overlap_score = overlap_score as usize * kv_block_size;

        let new_tokens = request.isl_tokens.saturating_sub(overlap_score);
        let normalized_new_tokens = new_tokens as f64 / request.isl_tokens as f64;

        let request_load_ratio =
            w.data.request_active_slots as f64 / w.data.request_total_slots as f64;

        // cost = alpha * load_deviation + (1 - alpha)*normalized_new_tokens + gamma * request_load_ratio
        let cost = alpha * load_deviation
            + (1.0 - alpha) * normalized_new_tokens
            + gamma * request_load_ratio;

        tracing::debug!("worker: {}; load_deviation: {}; normalized new blocks: {}; request_load_ratio: {} cost: {}",
                worker_id,
                load_deviation,
                normalized_new_tokens,
                request_load_ratio,
                cost
            );

        if cost < best_cost {
            best_cost = cost;
            best_index = Some(i);
        }
    }

    if let Some(best_index) = best_index {
        let total_blocks = min(request.isl_tokens / kv_block_size, 1);

        workers.endpoints[best_index].data.request_active_slots += 1;
        workers.endpoints[best_index].data.kv_active_blocks += total_blocks as u64;

        // Optimization - pass this to a channel for emitting events, async task, etc. to avoid blocking the scheduler
        let best_worker_id = workers.endpoints[best_index].worker_id();
        let isl_blocks = request.isl_tokens / kv_block_size;
        let overlap_blocks = request
            .overlap
            .scores
            .get(&best_worker_id)
            .copied()
            .unwrap_or(0);
        if let Err(e) = event_tx.send(KVHitRateEvent {
            worker_id: best_worker_id,
            isl_blocks,
            overlap_blocks: overlap_blocks as usize,
        }) {
            tracing::warn!("Failed to send KV hit rate event: {:?}", e);
        }
    }

    match best_index {
        Some(i) => {
            tracing::info!(
                "selected worker: {}; cost: {}",
                workers.endpoints[i].worker_id(),
                best_cost
            );
            Ok(workers.endpoints[i].worker_id())
        }
        None => {
            tracing::debug!("all workers busy");
            Err(KvSchedulerError::AllWorkersBusy)
        }
    }
}
