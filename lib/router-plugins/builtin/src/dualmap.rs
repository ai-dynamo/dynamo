// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DualMap: two hash-mapped candidates per prompt prefix, chosen by cache reuse under a load cap.
//!
//! Ported from "DualMap: Enabling Both Cache Affinity and Load Balancing for Distributed LLM
//! Serving" (<https://arxiv.org/abs/2602.06502>). Two independent hashes of the prompt prefix map
//! every request to two candidate workers, so requests sharing a prefix share candidates. The
//! request goes to the candidate with more cached prefix unless its pending prefill, including
//! this request, would exceed `pending_prefill_token_budget`, the backlog a worker can clear
//! within the TTFT SLO; then it goes to the candidate with less pending prefill. Equal cache
//! reuse also picks the less loaded candidate.
//!
//! A prefix is hot when its share of the last `window_requests` requests exceeds `2 / n` for `n`
//! workers. Hot prefixes hash a longer key, doubling from `hash_prefix_blocks`, so their requests
//! spread over more candidate pairs.
//!
//! This port uses rendezvous hashing, which keeps the mapping stable as workers join and leave
//! like the paper's consistent-hash rings. The paper also migrates queued requests between the
//! two candidates; Dynamo places each request once, so that step is omitted. Requests without
//! prefix hashes, such as those in disaggregated prefill pools, go to the worker with the least
//! pending prefill.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use crate::signals::{device_overlap_blocks, rendezvous, uncached_prompt_tokens};
use dynamo_kv_router::KvRouterConfig;
use dynamo_kv_router::plugins::worker_selection::{
    WorkerInputView, WorkerInputs, WorkerPicker, WorkerSelectionContext, WorkerSelectionPolicy,
    WorkerSelectionPolicyError, WorkerSelectionPolicyFactory,
};
use dynamo_kv_router::plugins::{
    RouterPluginRegistry, WorkerSelectionPolicyParameters, WorkerSelectionPolicyProviderError,
    WorkerSelectionPolicyRegistryError,
};

/// Policy type selected by `worker_selection.instances[].type`.
pub const POLICY_TYPE: &str = "dualmap";

/// Prefix depths tracked per request: `hash_prefix_blocks` doubled up to seven times.
const MAX_LEVELS: usize = 8;

#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
struct Parameters {
    hash_prefix_blocks: usize,
    pending_prefill_token_budget: usize,
    window_requests: usize,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            hash_prefix_blocks: 4,
            pending_prefill_token_budget: 65_536,
            window_requests: 1_000,
        }
    }
}

impl Parameters {
    fn validate(&self) -> Result<(), WorkerSelectionPolicyProviderError> {
        if self.hash_prefix_blocks == 0
            || self.pending_prefill_token_budget == 0
            || self.window_requests == 0
        {
            return Err(WorkerSelectionPolicyProviderError::new(
                "hash_prefix_blocks, pending_prefill_token_budget, and window_requests must be positive",
            ));
        }
        Ok(())
    }
}

struct DualMapPicker {
    parameters: Parameters,
    /// Prefix keys at every tracked depth for each request in the window.
    window: VecDeque<([u64; MAX_LEVELS], usize)>,
    arrivals: HashMap<u64, usize>,
}

impl DualMapPicker {
    fn new(parameters: Parameters) -> Self {
        Self {
            parameters,
            window: VecDeque::with_capacity(parameters.window_requests),
            arrivals: HashMap::new(),
        }
    }

    /// Record this request's prefix keys, then return the shallowest key that is not hot.
    fn hash_key(&mut self, hashes: &[u64], workers: usize) -> Option<u64> {
        let mut keys = [0; MAX_LEVELS];
        let mut levels = 0;
        let mut depth = self.parameters.hash_prefix_blocks;
        while levels < MAX_LEVELS && depth <= hashes.len() {
            keys[levels] = hashes[depth - 1];
            levels += 1;
            depth *= 2;
        }
        if levels == 0 {
            return None;
        }
        if self.window.len() == self.parameters.window_requests
            && let Some((expired, expired_levels)) = self.window.pop_front()
        {
            for key in &expired[..expired_levels] {
                if let Some(count) = self.arrivals.get_mut(key) {
                    *count -= 1;
                    if *count == 0 {
                        self.arrivals.remove(key);
                    }
                }
            }
        }
        for key in &keys[..levels] {
            *self.arrivals.entry(*key).or_default() += 1;
        }
        self.window.push_back((keys, levels));
        let hot_share = 2.0 / workers as f64;
        let window_len = self.window.len() as f64;
        keys[..levels]
            .iter()
            .copied()
            .find(|key| self.arrivals[key] as f64 / window_len <= hot_share)
            .or(Some(keys[levels - 1]))
    }
}

impl WorkerPicker for DualMapPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD
    }

    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let candidates = input.candidates();
        let cache = input
            .cache()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("cache input unavailable"))?;
        let load = input
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        let pending = |row: usize| (load[row].active_prefill_tokens(), candidates[row].worker());
        let least_pending = |rows: &mut dyn Iterator<Item = usize>| {
            rows.min_by_key(|&row| pending(row))
                .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))
        };

        let key = match context.prefix_hashes() {
            Some(hashes) if candidates.len() > 1 => self.hash_key(hashes, candidates.len()),
            _ => None,
        };
        let Some(key) = key else {
            return least_pending(&mut (0..candidates.len()));
        };
        let draw = |seed: u64, skip: Option<usize>| {
            (0..candidates.len())
                .filter(|&row| Some(row) != skip)
                .max_by_key(|&row| rendezvous(key, candidates[row].worker(), seed))
        };
        let (Some(first), Some(second)) = (draw(1, None), draw(2, draw(1, None))) else {
            return least_pending(&mut (0..candidates.len()));
        };

        let cached = |row: usize| cache.get(row).map_or(0.0, device_overlap_blocks);
        let (warm, cold) = match cached(first).total_cmp(&cached(second)) {
            std::cmp::Ordering::Equal => return least_pending(&mut [first, second].into_iter()),
            std::cmp::Ordering::Greater => (first, second),
            std::cmp::Ordering::Less => (second, first),
        };
        let warm_backlog = load[warm].active_prefill_tokens()
            + cache.get(warm).map_or(context.prompt_tokens(), |cache| {
                uncached_prompt_tokens(context, cache)
            });
        if warm_backlog > self.parameters.pending_prefill_token_budget {
            return least_pending(&mut [warm, cold].into_iter());
        }
        Ok(warm)
    }
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let parameters: Parameters = parameters.deserialize()?;
    parameters.validate()?;
    Ok(Arc::new(
        move |config: &KvRouterConfig, worker_type, _partition| {
            WorkerSelectionPolicy::new(
                config.clone(),
                worker_type.as_str(),
                Vec::new(),
                Box::new(DualMapPicker::new(parameters)),
            )
        },
    ))
}

pub fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register_worker_selection(POLICY_TYPE, Arc::new(provider))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{Worker, request, select};
    use dynamo_kv_router::protocols::WorkerWithDpRank;

    fn policy(parameters: Parameters) -> WorkerSelectionPolicy {
        WorkerSelectionPolicy::new(
            KvRouterConfig::default(),
            "test",
            Vec::new(),
            Box::new(DualMapPicker::new(parameters)),
        )
    }

    fn idle(count: u64) -> Vec<Worker> {
        (0..count).map(Worker::new).collect()
    }

    #[test]
    fn a_prefix_maps_to_the_same_two_candidates() {
        let policy = policy(Parameters::default());
        let workers = idle(8);
        let picks: std::collections::HashSet<_> = (0..20)
            .map(|_| select(&policy, request(8, 3), &workers))
            .collect();
        // Idle, uncached candidates tie; the same prefix keeps landing on one worker.
        assert_eq!(picks.len(), 1);
    }

    #[test]
    fn prefers_the_warmer_candidate_until_its_backlog_exceeds_the_budget() {
        let mut probe = DualMapPicker::new(Parameters::default());
        let hashes = request(8, 3).token_seq.unwrap();
        let key = probe.hash_key(&hashes, 8).unwrap();
        let rank = |id| WorkerWithDpRank::from_worker_id(id);
        let ranked = |seed| {
            (0..8)
                .max_by_key(|&id| rendezvous(key, rank(id), seed))
                .unwrap()
        };
        let first = ranked(1);
        let second = (0..8)
            .filter(|&id| id != first)
            .max_by_key(|&id| rendezvous(key, rank(id), 2))
            .unwrap();

        let mut workers = idle(8);
        workers[second as usize] = Worker::new(second).cached(8);
        let budget = Parameters {
            pending_prefill_token_budget: 1_000,
            ..Parameters::default()
        };
        assert_eq!(select(&policy(budget), request(8, 3), &workers), second);
        workers[second as usize] = workers[second as usize].prefill(1_001);
        assert_eq!(select(&policy(budget), request(8, 3), &workers), first);
    }

    #[test]
    fn a_hot_prefix_spreads_over_more_candidates() {
        // With four workers a prefix is hot above half of the window. Every request shares the
        // first 4 blocks but differs at block 8, so the hot short key gives way to distinct keys.
        let policy = policy(Parameters {
            window_requests: 16,
            ..Parameters::default()
        });
        let workers = idle(4);
        let picks: std::collections::HashSet<_> = (0..16)
            .map(|suffix| {
                let mut request = request(8, 3);
                let hashes = request.token_seq.as_mut().unwrap();
                hashes[7] = 10_000 + suffix;
                select(&policy, request, &workers)
            })
            .collect();
        assert!(picks.len() > 2, "hot prefix stayed on {picks:?}");
    }
}
