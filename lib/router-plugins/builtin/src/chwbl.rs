// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Consistent hashing with bounded loads (CHWBL) on the prompt prefix.
//!
//! Ported from KubeAI's `balance_chwbl.go`, with defaults from SGLang's `prefix_hash` gateway
//! policy, both after "Consistent Hashing with Bounded Loads" (Mirrokni et al., SODA 2018).
//! Requests sharing their first `prefix_tokens` map to the same worker. A worker accepts a
//! request while its active requests stay within `load_factor` times the fleet average, counting
//! the new request; otherwise the request walks to the next worker in hash order. When every
//! worker is over the bound, the first worker in hash order takes it.
//!
//! The policy reads no cache state, so it routes identically with or without KV events. The key
//! is block aligned: it covers the prompt's first `prefix_tokens / block_size` full blocks, or
//! all of them for a shorter prompt. Prompts without a full block, and requests without prefix
//! hashes such as those in disaggregated prefill pools, go to the least-loaded worker.
//! Rendezvous hashing stands in for the hash ring.

use std::sync::Arc;

use dynamo_kv_router::KvRouterConfig;
use dynamo_kv_router::plugins::worker_selection::{
    WorkerInputView, WorkerInputs, WorkerPicker, WorkerSelectionContext, WorkerSelectionPolicy,
    WorkerSelectionPolicyError, WorkerSelectionPolicyFactory,
};
use dynamo_kv_router::plugins::{
    RouterPluginRegistry, WorkerSelectionPolicyParameters, WorkerSelectionPolicyProviderError,
    WorkerSelectionPolicyRegistryError,
};

use crate::signals::rendezvous;

/// Policy type selected by `worker_selection.instances[].type`.
pub const POLICY_TYPE: &str = "chwbl";

#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
struct Parameters {
    prefix_tokens: usize,
    load_factor: f64,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            prefix_tokens: 256,
            load_factor: 1.25,
        }
    }
}

impl Parameters {
    fn validate(&self) -> Result<(), WorkerSelectionPolicyProviderError> {
        if self.prefix_tokens == 0 {
            return Err(WorkerSelectionPolicyProviderError::new(
                "prefix_tokens must be positive",
            ));
        }
        if !(self.load_factor.is_finite() && self.load_factor >= 1.0) {
            return Err(WorkerSelectionPolicyProviderError::new(
                "load_factor must be a finite number of at least 1.0",
            ));
        }
        Ok(())
    }
}

struct ChwblPicker {
    parameters: Parameters,
    order: Vec<usize>,
}

impl WorkerPicker for ChwblPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD
    }

    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let candidates = input.candidates();
        let load = input
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        let requests = |row: usize| load[row].active_requests();
        let key_blocks =
            (self.parameters.prefix_tokens / context.block_size().max(1) as usize).max(1);
        let key = context
            .prefix_hashes()
            .and_then(|hashes| hashes.get(key_blocks.min(hashes.len()).checked_sub(1)?))
            .copied();
        let Some(key) = key else {
            return (0..candidates.len())
                .min_by_key(|&row| (requests(row), candidates[row].worker()))
                .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"));
        };

        self.order.clear();
        self.order.extend(0..candidates.len());
        self.order.sort_unstable_by_key(|&row| {
            std::cmp::Reverse(rendezvous(key, candidates[row].worker(), 0))
        });
        let total: usize = (0..candidates.len()).map(requests).sum();
        let bound = (total + 1) as f64 / candidates.len() as f64 * self.parameters.load_factor;
        let within_bound = |row: usize| total == 0 || requests(row) as f64 <= bound;
        self.order
            .iter()
            .copied()
            .find(|&row| within_bound(row))
            .or_else(|| self.order.first().copied())
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))
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
                Box::new(ChwblPicker {
                    parameters,
                    order: Vec::new(),
                }),
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

    fn policy() -> WorkerSelectionPolicy {
        WorkerSelectionPolicy::new(
            KvRouterConfig::default(),
            "test",
            Vec::new(),
            Box::new(ChwblPicker {
                parameters: Parameters::default(),
                order: Vec::new(),
            }),
        )
    }

    fn idle(count: u64) -> Vec<Worker> {
        (0..count).map(Worker::new).collect()
    }

    #[test]
    fn a_prefix_keeps_its_worker_and_prefixes_spread() {
        let policy = policy();
        let workers = idle(8);
        let owner = select(&policy, request(32, 5), &workers);
        assert!((0..5).all(|_| select(&policy, request(32, 5), &workers) == owner));
        let owners: std::collections::HashSet<_> = (0..32)
            .map(|prefix| select(&policy, request(32, prefix), &workers))
            .collect();
        assert!(owners.len() > 4, "prefixes collapsed onto {owners:?}");
    }

    #[test]
    fn an_overloaded_owner_passes_the_request_on() {
        let policy = policy();
        let mut workers = idle(4);
        let owner = select(&policy, request(32, 5), &workers) as usize;
        // Six requests on the owner exceed (6 + 1) / 4 × 1.25 ≈ 2.2.
        workers[owner] = workers[owner].requests(6);
        assert_ne!(select(&policy, request(32, 5), &workers) as usize, owner);
        // The bound grows with the fleet's load: with 6 + 3 × 5 = 21 requests in flight it is
        // 22 / 4 × 1.25 ≈ 6.9, which admits the owner again.
        for (id, worker) in workers.iter_mut().enumerate() {
            if id != owner {
                *worker = worker.requests(5);
            }
        }
        assert_eq!(select(&policy, request(32, 5), &workers) as usize, owner);
    }
}
