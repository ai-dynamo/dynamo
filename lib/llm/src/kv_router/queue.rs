// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dynamo_runtime::config::environment_names::llm::kv_router as env_kv_router;
use tokio::sync::{Mutex, Notify};

use crate::discovery::RuntimeConfigWatch;

use super::protocols::WorkerWithDpRank;
use super::scheduler::SchedulingRequest;
use super::sequence::ActiveSequencesMultiWorker;

/// Large default for max_num_batched_tokens when not configured (effectively disables queueing for that worker)
const DEFAULT_MAX_BATCHED_TOKENS: u64 = 10_000_000;

/// Returns the queue threshold fraction if set, None if queueing is disabled
fn queue_threshold_frac() -> Option<f64> {
    let val = std::env::var(env_kv_router::DYN_ROUTER_QUEUE_THRESHOLD_FRAC).ok()?;
    let Ok(frac) = val.parse::<f64>() else {
        tracing::warn!(
            "{} set to invalid value '{val}', ignoring",
            env_kv_router::DYN_ROUTER_QUEUE_THRESHOLD_FRAC
        );
        return None;
    };
    if frac.is_nan() || frac <= 0.0 {
        tracing::warn!(
            "{} must be > 0 (got {frac}), ignoring",
            env_kv_router::DYN_ROUTER_QUEUE_THRESHOLD_FRAC
        );
        return None;
    }
    Some(frac)
}

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

/// Queue for managing scheduling requests with interior mutability.
/// Requests are held in `pending` when all workers are busy, and moved to `ready` when capacity frees up.
/// If queueing is disabled (env var not set), all requests go directly to `ready`.
/// Requests are ordered by effective arrival time: arrival_offset - priority_jump.
pub struct SchedulerQueue {
    pending: Mutex<BinaryHeap<QueueEntry>>,
    ready: Mutex<VecDeque<SchedulingRequest>>,
    slots: Arc<ActiveSequencesMultiWorker>,
    workers_with_configs: RuntimeConfigWatch,
    ready_notify: Arc<Notify>,
    /// Cached threshold fraction; None means queueing is disabled.
    threshold_frac: Option<f64>,
    /// Reference instant for computing arrival offsets.
    start_time: Instant,
}

impl SchedulerQueue {
    pub fn new(
        slots: Arc<ActiveSequencesMultiWorker>,
        workers_with_configs: RuntimeConfigWatch,
        ready_notify: Arc<Notify>,
    ) -> Self {
        let threshold_frac = queue_threshold_frac();
        if let Some(frac) = threshold_frac {
            tracing::info!("Router queue enabled with threshold fraction {frac}");
        }
        Self {
            pending: Mutex::new(BinaryHeap::new()),
            ready: Mutex::new(VecDeque::new()),
            slots,
            workers_with_configs,
            ready_notify,
            threshold_frac,
            start_time: Instant::now(),
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
    /// If queueing is disabled (env var not set), fast-track to ready.
    /// Otherwise, check busy condition and place in ready or pending.
    pub async fn enqueue(&self, request: SchedulingRequest) {
        let Some(threshold) = self.threshold_frac else {
            self.ready.lock().await.push_back(request);
            return;
        };

        if self.all_workers_busy(threshold).await {
            tracing::debug!("all workers busy, queueing request");
            let entry = self.make_entry(request);
            self.pending.lock().await.push(entry);
        } else {
            self.ready.lock().await.push_back(request);
        }
    }

    /// Try to dequeue the highest-priority request from the ready queue.
    pub async fn try_dequeue(&self) -> Option<SchedulingRequest> {
        self.ready.lock().await.pop_front()
    }

    /// Called on prefill_complete/free. Re-checks pending requests and moves eligible to ready.
    /// Notifies scheduler loop if any requests were moved.
    pub async fn update(&self) {
        let Some(threshold) = self.threshold_frac else {
            return;
        };

        let mut moved = false;
        loop {
            if self.pending.lock().await.is_empty() {
                break;
            }
            if self.all_workers_busy(threshold).await {
                break;
            }
            let entry = self.pending.lock().await.pop();
            if let Some(entry) = entry {
                tracing::debug!("moving request from pending to ready");
                self.ready.lock().await.push_back(entry.request);
                moved = true;
            } else {
                break;
            }
        }
        if moved {
            self.ready_notify.notify_one();
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
