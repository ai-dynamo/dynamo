// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Result, anyhow};
use tokio::sync::Notify;
use tokio::time::Instant;
use uuid::Uuid;

use crate::common::protocols::DirectRequest;
use crate::loadgen::WorkloadDriver;

#[derive(Clone, Copy, Debug)]
pub(super) enum LiveReplayMode {
    Trace,
    Concurrency { max_in_flight: usize },
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct LiveRuntimeStats {
    pub(super) dispatch_history: Vec<usize>,
    pub(super) max_in_flight_seen: usize,
    pub(super) prefill_marked_count: usize,
    pub(super) freed_count: usize,
    pub(super) vllm_preemptions_total: u64,
}

#[derive(Default)]
pub(super) struct SharedLiveRuntimeStats {
    dispatch_history: Mutex<Vec<usize>>,
    current_in_flight: AtomicUsize,
    max_in_flight_seen: AtomicUsize,
    prefill_marked_count: AtomicUsize,
    freed_count: AtomicUsize,
}

impl SharedLiveRuntimeStats {
    pub(super) fn record_dispatch(&self, worker_idx: usize) {
        self.dispatch_history.lock().unwrap().push(worker_idx);
        let current = self.current_in_flight.fetch_add(1, Ordering::AcqRel) + 1;
        self.max_in_flight_seen.fetch_max(current, Ordering::AcqRel);
    }

    pub(super) fn record_completion(&self) {
        self.current_in_flight.fetch_sub(1, Ordering::AcqRel);
    }

    pub(super) fn record_prefill_marked(&self) {
        self.prefill_marked_count.fetch_add(1, Ordering::AcqRel);
    }

    pub(super) fn record_freed(&self) {
        self.freed_count.fetch_add(1, Ordering::AcqRel);
    }

    pub(super) fn snapshot(&self, vllm_preemptions_total: u64) -> LiveRuntimeStats {
        LiveRuntimeStats {
            dispatch_history: self.dispatch_history.lock().unwrap().clone(),
            max_in_flight_seen: self.max_in_flight_seen.load(Ordering::Acquire),
            prefill_marked_count: self.prefill_marked_count.load(Ordering::Acquire),
            freed_count: self.freed_count.load(Ordering::Acquire),
            vllm_preemptions_total,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct ArrivalEvent {
    pub(super) uuid: Uuid,
    pub(super) at_ms: f64,
    pub(super) input_tokens: usize,
    pub(super) output_tokens: usize,
}

pub(super) struct WorkloadDispatchState {
    pub(super) driver: Mutex<WorkloadDriver>,
    pub(super) wakeup: Notify,
    pub(super) start: Instant,
}

pub(super) fn now_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

pub(super) fn request_uuid(request: &DirectRequest) -> Result<Uuid> {
    request
        .uuid
        .ok_or_else(|| anyhow!("online replay requires requests to have stable UUIDs"))
}

pub(super) fn arrival_event(request: &DirectRequest, arrival_at_ms: f64) -> Result<ArrivalEvent> {
    let uuid = request_uuid(request)?;
    Ok(ArrivalEvent {
        uuid,
        at_ms: arrival_at_ms,
        input_tokens: request.tokens.len(),
        output_tokens: request.max_output_tokens,
    })
}
