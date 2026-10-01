// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Experimental aggregated pool mapping of Dynamo's default device-cache cost.
//!
//! Pool work is the sum of Frontend scheduler tokens/blocks across all ranks.
//! A CKF hit estimates a prefix available somewhere in the pool; local KV routing
//! still chooses its worker. This is not a worker-level placement guarantee.
//! Homogeneous block sizes/capacities are required for this first comparison.
//! Worker scheduling can project additional unique active blocks using its local
//! sequence trie. Pool CKFs cannot distinguish active from retained blocks, so
//! the incoming prompt's full block count is conservatively added to decode
//! work. That term is identical across homogeneous candidates and does not
//! change their ordering; this mapping does not reproduce worker placement.

use std::collections::HashMap;
use std::sync::Arc;

use dynamo_custom_policy_builtin::credit::aggregated_cost;
use dynamo_kv_router::global_view::PoolId;
use dynamo_kv_router::global_view::eligibility::eligible_pools;
use dynamo_kv_router::global_view::overlap::KvOverlapScorer;
use dynamo_kv_router::global_view::selection::{LoadBasis, PoolDecision};
use dynamo_kv_router::global_view::state::{
    FreshnessPolicy, PoolRole, PoolStateRepository, SignalState, SignalStatus,
};
use parking_lot::Mutex;

use super::scheduler_metrics::SchedulerLoadRepository;

#[derive(Default)]
struct Assignments {
    epoch: u64,
    prefill_tokens: u64,
    decode_blocks: u64,
    count: u64,
}

pub struct CreditPoolSelector {
    repository: Arc<dyn PoolStateRepository>,
    scheduler: Arc<dyn SchedulerLoadRepository>,
    overlap: Arc<dyn KvOverlapScorer>,
    freshness: FreshnessPolicy,
    credit: f64,
    assignments: Mutex<HashMap<(PoolId, String), Assignments>>,
}

impl CreditPoolSelector {
    pub fn new(
        repository: Arc<dyn PoolStateRepository>,
        scheduler: Arc<dyn SchedulerLoadRepository>,
        overlap: Arc<dyn KvOverlapScorer>,
        freshness: FreshnessPolicy,
        credit: f64,
    ) -> Self {
        Self {
            repository,
            scheduler,
            overlap,
            freshness,
            credit,
            assignments: Mutex::new(HashMap::new()),
        }
    }

    /// Missing scheduler coverage returns None so the host can apply its existing
    /// load policy to ALL eligible pools. Never mix missing work with zero work.
    /// Missing/stale CKF omits just that credit, as for the existing pool policy.
    pub fn select(&self, model: &str, tokens: &[u32], now_ms: u64) -> Option<PoolDecision> {
        let pools: Vec<_> =
            eligible_pools(self.repository.as_ref(), model, now_ms, &self.freshness)
                .into_iter()
                .filter(|p| p.descriptors.roles.contains(&PoolRole::Aggregated))
                .collect();
        let candidates: Option<Vec<_>> = pools
            .iter()
            .map(|pool| {
                let load = self.scheduler.snapshot(&pool.pool_id, model, now_ms)?;
                let overlap = matches!(
                    pool.signal_status.kv_overlap.state,
                    SignalState::Complete | SignalState::Degraded
                )
                .then(|| {
                    self.overlap
                        .estimate_matched_prefix_tokens(&pool.pool_id, model, tokens)
                })
                .flatten()
                .map(|v| v.min(tokens.len() as u64));
                Some((pool, load, overlap))
            })
            .collect();
        let candidates = candidates?;
        let first = candidates.first()?;
        let block_size = first.1.block_size;
        if block_size == 0
            || !first.0.capacity.live_workers.is_some_and(|v| v > 0)
            || !first.0.capacity.max_concurrency.is_some_and(|v| v > 0)
        {
            return None;
        }
        // First experiment is homogeneous. An unsupported topology falls back.
        if candidates.iter().any(|(pool, load, _)| {
            load.block_size != first.1.block_size
                || pool.capacity.live_workers != first.0.capacity.live_workers
                || pool.capacity.max_concurrency != first.0.capacity.max_concurrency
        }) {
            return None;
        }
        let mut assignments = self.assignments.lock();
        let mut best: Option<PoolDecision> = None;
        for (pool, load, overlap) in candidates {
            let local = assignments
                .entry((pool.pool_id.clone(), model.to_owned()))
                .or_default();
            if local.epoch != load.collected_at_unix_ms {
                *local = Assignments {
                    epoch: load.collected_at_unix_ms,
                    ..Default::default()
                };
            }
            let cost = pool_cost(
                load.prefill_tokens.saturating_add(local.prefill_tokens),
                load.decode_blocks.saturating_add(local.decode_blocks),
                tokens.len() as u64,
                load.block_size,
                overlap.unwrap_or(0),
                self.credit,
            );
            let decision = PoolDecision {
                pool_id: pool.pool_id.clone(),
                region: pool.descriptors.location.region.clone(),
                frontend_endpoint: pool.descriptors.frontend_endpoint.clone()?,
                basis: LoadBasis::Observed,
                observed_requests: None,
                assignments_since_sample: local.count,
                normalized_by_max_concurrency: false,
                cost,
                load_status: SignalStatus {
                    state: SignalState::Complete,
                    received_at_unix_ms: Some(load.collected_at_unix_ms),
                    ..Default::default()
                },
                matched_prefix_tokens: overlap,
            };
            if best.as_ref().is_none_or(|b| {
                (decision.cost, decision.pool_id.as_str()) < (b.cost, b.pool_id.as_str())
            }) {
                best = Some(decision);
            }
        }
        let selected = best?;
        let local = assignments.get_mut(&(selected.pool_id.clone(), model.to_owned()))?;
        // Conservative unobserved work until the next scrape. Keep booking
        // identical between credit=0 and credit=1, isolating the scoring weight.
        // These are router-local estimates, not authoritative worker bookings.
        local.prefill_tokens = local.prefill_tokens.saturating_add(tokens.len() as u64);
        local.decode_blocks = local
            .decode_blocks
            .saturating_add((tokens.len() as u64).div_ceil(u64::from(block_size)));
        local.count = local.count.saturating_add(1);
        Some(selected)
    }
}

fn pool_cost(
    prefill_tokens: u64,
    decode_blocks: u64,
    prompt_tokens: u64,
    block_size: u32,
    overlap_tokens: u64,
    credit: f64,
) -> f64 {
    aggregated_cost(
        prefill_tokens.saturating_add(prompt_tokens) as f64 / f64::from(block_size),
        credit * (overlap_tokens as f64 / f64::from(block_size)),
        1.0, // Dynamo default prefill_load_scale.
        decode_blocks.saturating_add(prompt_tokens.div_ceil(u64::from(block_size))) as f64,
        0.0, // Dynamo default decode_active_request_weight; device-only CKF.
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cache_saves_prefill_but_busy_cache_can_lose() {
        let cold_idle = pool_cost(0, 0, 1024, 64, 0, 1.0);
        assert_eq!(cold_idle, 32.0);
        assert_eq!(pool_cost(0, 0, 1024, 64, 1024, 1.0), 16.0);
        assert!(pool_cost(0, 20, 1024, 64, 1024, 1.0) > cold_idle);
        assert_eq!(pool_cost(64, 2, 1024, 64, 1024, 0.0), 35.0);
        assert_eq!(pool_cost(64, 2, 1024, 64, 1024, 1.0), 19.0);
    }
}
