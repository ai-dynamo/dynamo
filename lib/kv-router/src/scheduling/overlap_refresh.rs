// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Re-query overlap scores at dequeue time.
//!
//! When a request has been parked in the scheduler queue, the KV cache state on each worker
//! may have changed significantly while it waited. The `tier_overlap_blocks` and
//! `effective_overlap_blocks` computed at enqueue time can be stale enough to pick the
//! wrong worker on dispatch.
//!
//! [`SchedulerQueue`](super::queue::SchedulerQueue) holds an optional
//! `Arc<dyn OverlapScoresRefresh>`. When set, the queue calls
//! [`OverlapScoresRefresh::refresh`] for any request that waited longer than the configured
//! threshold, replacing the per-tier and effective-overlap fields on the request with a fresh
//! read from the indexer.
//!
//! Refresh failures are non-fatal: an implementation can return `None` and the queue will
//! dispatch with the (stale) original scores rather than dropping the request.

use std::collections::HashMap;

use async_trait::async_trait;

use crate::protocols::{LocalBlockHash, WorkerWithDpRank};

use super::types::TierOverlapBlocks;

/// Result of a successful overlap refresh.
///
/// Carries everything required to overwrite the overlap-related fields on a
/// [`SchedulingRequest`](super::types::SchedulingRequest) at dequeue time.
#[derive(Debug, Clone, Default)]
pub struct RefreshedOverlap {
    pub tier_overlap_blocks: TierOverlapBlocks,
    pub effective_overlap_blocks: HashMap<WorkerWithDpRank, f64>,
    pub effective_cached_tokens: HashMap<WorkerWithDpRank, usize>,
}

/// Re-query overlap scores for a request that has been waiting in the scheduler queue.
///
/// Implementations are expected to be cheap to clone (typically `Arc`-wrapped) and to never
/// panic. Returning `None` indicates the refresh failed; the queue will then dispatch with
/// the original scores.
#[async_trait]
pub trait OverlapScoresRefresh: Send + Sync {
    async fn refresh(&self, block_hashes: &[LocalBlockHash]) -> Option<RefreshedOverlap>;
}

/// Default no-op refresher used when dequeue-time overlap refresh is not configured.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopOverlapScoresRefresh;

#[async_trait]
impl OverlapScoresRefresh for NoopOverlapScoresRefresh {
    async fn refresh(&self, _block_hashes: &[LocalBlockHash]) -> Option<RefreshedOverlap> {
        None
    }
}
