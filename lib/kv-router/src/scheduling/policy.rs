// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use ordered_float::OrderedFloat;

use super::config::RouterQueuePolicy;
use super::types::SchedulingRequest;

/// Pluggable scheduling policy that determines queue ordering.
/// Monomorphized for zero-cost inlining on the hot comparison path.
///
/// Higher key = higher priority (natural max-heap ordering).
pub trait SchedulingPolicy: Send + Sync + 'static {
    /// Priority key stored in each queue entry.
    type Key: Ord + Eq + Clone + Send + 'static;

    /// Compute priority key at enqueue time.
    fn enqueue_key(&self, arrival_offset: Duration, request: &SchedulingRequest) -> Self::Key;

    /// Recompute priority key during update(). Default: return old key unchanged.
    fn rekey(&self, _now: Duration, old_key: &Self::Key, _req: &SchedulingRequest) -> Self::Key {
        old_key.clone()
    }

    /// When true, queue rebuilds heap via rekey() on each update() call.
    /// When false (default), rekey path is compiled out entirely.
    const DYNAMIC: bool = false;
}

/// FCFS with priority bumps: key = priority_jump - arrival_offset.
/// Earlier arrival or higher priority_jump produces a higher key, scheduled first.
///
/// Optimizes for tail TTFT — no request waits longer than necessary,
/// since ordering is purely by (adjusted) arrival time.
pub struct FcfsPolicy;

impl SchedulingPolicy for FcfsPolicy {
    type Key = OrderedFloat<f64>;

    fn enqueue_key(&self, arrival_offset: Duration, request: &SchedulingRequest) -> Self::Key {
        OrderedFloat(request.priority_jump.max(0.0) - arrival_offset.as_secs_f64())
    }
}

/// Weighted Shortest Processing Time (Smith's rule):
/// key = (1 + priority_jump) / isl_tokens.
/// Higher ratio (shorter job relative to its weight) produces a higher key, scheduled first.
///
/// Optimizes for average TTFT — minimizes total weighted completion time
/// (Smith 1956). Short or high-priority requests are scheduled before
/// long low-priority ones, reducing mean latency across the batch.
pub struct WsptPolicy;

impl SchedulingPolicy for WsptPolicy {
    type Key = OrderedFloat<f64>;

    fn enqueue_key(&self, _arrival_offset: Duration, request: &SchedulingRequest) -> Self::Key {
        let weight = 1.0 + request.priority_jump.max(0.0);
        let processing_time = request.isl_tokens.max(1) as f64;
        OrderedFloat(weight / processing_time)
    }
}

/// Runtime-dispatched scheduling policy selected via configuration.
/// Delegates to the concrete policy variant; the branch is fully predictable
/// since the variant is fixed at queue construction time.
pub enum RouterSchedulingPolicy {
    Fcfs(FcfsPolicy),
    Wspt(WsptPolicy),
}

impl From<RouterQueuePolicy> for RouterSchedulingPolicy {
    fn from(kind: RouterQueuePolicy) -> Self {
        match kind {
            RouterQueuePolicy::Fcfs => Self::Fcfs(FcfsPolicy),
            RouterQueuePolicy::Wspt => Self::Wspt(WsptPolicy),
        }
    }
}

impl SchedulingPolicy for RouterSchedulingPolicy {
    type Key = OrderedFloat<f64>;

    fn enqueue_key(&self, arrival_offset: Duration, request: &SchedulingRequest) -> Self::Key {
        match self {
            Self::Fcfs(p) => p.enqueue_key(arrival_offset, request),
            Self::Wspt(p) => p.enqueue_key(arrival_offset, request),
        }
    }
}
