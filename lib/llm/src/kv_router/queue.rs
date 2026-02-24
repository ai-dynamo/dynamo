// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::Mutex;

use super::WorkerSelector;
use super::protocols::WorkerWithDpRank;
use super::scheduler::{SchedulingRequest, SchedulingResponse};
use super::sequence::{ActiveSequencesMultiWorker, SequenceRequest};
use crate::discovery::RuntimeConfigWatch;

/// Large default for max_num_batched_tokens when not configured (effectively disables queueing for that worker)
const DEFAULT_MAX_BATCHED_TOKENS: u64 = 10_000_000;

/// Entry in the priority queue, ordered by effective arrival time (lower = higher priority).
/// Effective arrival = elapsed time since queue start minus `priority_jump`.
struct QueueEntry {
    effective_offset: Duration,
    request: SchedulingRequest,
}

impl Eq for QueueEntry {}

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.effective_offset == other.effective_offset
    }
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap; reverse so lower effective_offset = higher priority
        other.effective_offset.cmp(&self.effective_offset)
    }
}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Queue that gates scheduling requests behind a capacity check.
/// When all workers exceed `threshold_frac` utilisation the request is parked in `pending`.
/// When capacity frees up (`update()`), pending requests are scheduled in priority order.
/// If queueing is disabled (threshold_frac is None), requests are scheduled immediately.
pub struct SchedulerQueue {
    pending: Mutex<BinaryHeap<QueueEntry>>,
    slots: Arc<ActiveSequencesMultiWorker>,
    workers_with_configs: RuntimeConfigWatch,
    /// Cached threshold fraction; None means queueing is disabled.
    threshold_frac: Option<f64>,
    /// Reference instant for computing arrival offsets.
    start_time: Instant,
    block_size: u32,
    selector: Box<dyn WorkerSelector + Send + Sync>,
}

impl SchedulerQueue {
    pub fn new(
        slots: Arc<ActiveSequencesMultiWorker>,
        workers_with_configs: RuntimeConfigWatch,
        threshold_frac: Option<f64>,
        block_size: u32,
        selector: Box<dyn WorkerSelector + Send + Sync>,
    ) -> Self {
        if let Some(frac) = threshold_frac {
            tracing::info!("Router queue enabled with threshold fraction {frac}");
        }
        Self {
            pending: Mutex::new(BinaryHeap::new()),
            slots,
            workers_with_configs,
            threshold_frac,
            start_time: Instant::now(),
            block_size,
            selector,
        }
    }

    /// Build a QueueEntry for a request, computing its effective arrival offset.
    fn make_entry(&self, request: SchedulingRequest) -> QueueEntry {
        let arrival_offset = self.start_time.elapsed();
        let jump = Duration::from_secs_f64(request.priority_jump.max(0.0));
        let effective_offset = arrival_offset.saturating_sub(jump);
        QueueEntry {
            effective_offset,
            request,
        }
    }

    /// Enqueue a new request.
    /// If queueing is disabled or workers have capacity, schedule immediately.
    /// Otherwise park in the pending heap.
    pub async fn enqueue(&self, request: SchedulingRequest) {
        let Some(threshold) = self.threshold_frac else {
            self.schedule(request).await;
            return;
        };

        if self.all_workers_busy(threshold).await {
            tracing::debug!("all workers busy, queueing request");
            let entry = self.make_entry(request);
            self.pending.lock().await.push(entry);
        } else {
            self.schedule(request).await;
        }
    }

    /// Called on prefill_complete/free. Drains pending requests while workers have capacity.
    /// Each scheduled request updates active_tokens via add_request, so the busy check
    /// sees fresh state on the next iteration.
    pub async fn update(&self) {
        let Some(threshold) = self.threshold_frac else {
            return;
        };

        loop {
            if self.all_workers_busy(threshold).await {
                break;
            }
            let Some(entry) = self.pending.lock().await.pop() else {
                break;
            };
            tracing::debug!("scheduling request from pending queue");
            self.schedule(entry.request).await;
        }
    }

    /// Run the full scheduling pipeline for a single request:
    /// compute potential load → select worker → respond → book via add_request.
    async fn schedule(&self, mut request: SchedulingRequest) {
        let (decode_blocks, prefill_tokens) = self
            .slots
            .potential_blocks_and_tokens(
                request.token_seq.clone(),
                request.isl_tokens,
                request.overlaps.clone(),
            )
            .await;
        request.decode_blocks = decode_blocks;
        request.prefill_tokens = prefill_tokens;

        let selection = {
            let workers = self.workers_with_configs.borrow();
            self.selector
                .select_worker(&workers, &request, self.block_size)
        };

        let selection = match selection {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("scheduling failed: {e}");
                request.respond(Err(e));
                return;
            }
        };

        request.respond(Ok(SchedulingResponse {
            best_worker: selection.worker,
            overlap_blocks: selection.overlap_blocks,
        }));

        if !request.update_states {
            return;
        }

        let Some(request_id) = request.maybe_request_id else {
            tracing::error!("No request_id provided to add_request to the slot tracker");
            return;
        };

        if let Err(e) = self
            .slots
            .add_request(SequenceRequest {
                request_id: request_id.clone(),
                token_sequence: request.token_seq,
                isl: request.isl_tokens,
                overlap: selection.overlap_blocks,
                expected_output_tokens: None,
                worker: selection.worker,
                lora_name: request.lora_name.clone(),
            })
            .await
        {
            tracing::warn!("Failed to add request {request_id}: {e}");
        }
    }

    /// Check if all workers are busy based on threshold.
    /// Returns true only if ALL workers exceed the threshold (no worker has capacity).
    async fn all_workers_busy(&self, threshold: f64) -> bool {
        let active_tokens = self.slots.active_tokens().await;
        let configs = self.workers_with_configs.borrow();

        for (&worker_id, config) in configs.iter() {
            let dp_size = config.data_parallel_size;
            let max_batched = config
                .max_num_batched_tokens
                .unwrap_or(DEFAULT_MAX_BATCHED_TOKENS);

            for dp_rank in 0..dp_size {
                let worker = WorkerWithDpRank::new(worker_id, dp_rank);
                let tokens = active_tokens.get(&worker).copied().unwrap_or(0);
                if (tokens as f64) <= threshold * (max_batched as f64) {
                    return false;
                }
            }
        }
        true
    }
}
