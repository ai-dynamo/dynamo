// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use anyhow::{Result, anyhow};
use dashmap::DashMap;
use tokio::sync::{Notify, mpsc};
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

    pub(super) fn snapshot(&self) -> LiveRuntimeStats {
        LiveRuntimeStats {
            dispatch_history: self.dispatch_history.lock().unwrap().clone(),
            max_in_flight_seen: self.max_in_flight_seen.load(Ordering::Acquire),
            prefill_marked_count: self.prefill_marked_count.load(Ordering::Acquire),
            freed_count: self.freed_count.load(Ordering::Acquire),
        }
    }
}

#[derive(Default)]
pub(super) struct RequestState {
    first_token_seen: AtomicBool,
    completed_seen: AtomicBool,
    completion_notify: Notify,
}

impl RequestState {
    pub(super) fn mark_first_token_once(&self) -> bool {
        !self.first_token_seen.swap(true, Ordering::AcqRel)
    }

    pub(super) fn mark_completed_once(&self) -> bool {
        !self.completed_seen.swap(true, Ordering::AcqRel)
    }

    pub(super) fn notify_completion(&self) {
        self.completion_notify.notify_waiters();
    }

    pub(super) async fn wait_for_completion(&self) {
        loop {
            let notified = self.completion_notify.notified();
            if self.completed_seen.load(Ordering::Acquire) {
                return;
            }
            notified.await;
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

pub(super) type RequestRegistry = Arc<DashMap<Uuid, Arc<RequestState>>>;

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

pub(super) fn record_arrival(
    arrival_tx: &mpsc::UnboundedSender<ArrivalEvent>,
    request: &DirectRequest,
    arrival_at_ms: f64,
) -> Result<Uuid> {
    let uuid = request_uuid(request)?;
    let input_tokens = request.tokens.len();
    let output_tokens = request.max_output_tokens;
    arrival_tx
        .send(ArrivalEvent {
            uuid,
            at_ms: arrival_at_ms,
            input_tokens,
            output_tokens,
        })
        .map_err(|_| anyhow!("online replay arrival channel closed"))?;
    Ok(uuid)
}
