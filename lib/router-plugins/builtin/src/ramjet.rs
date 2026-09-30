// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ramjet: capped prefix affinity minus weighted load.
//!
//! Ported from Helix's Ramjet load balancer (<https://github.com/helixml/ramjet>, `src/router.rs`
//! at `5319354c`). Each worker scores `affinity − alpha × load_units`; the highest score wins,
//! then the deeper raw overlap, then a rotating tie-break.
//!
//! - Affinity counts cached prompt in Ramjet's 2 KB fingerprint blocks, about 512 tokens, and is
//!   capped at `max_affinity_blocks`. The cap lets a large load gap override even a
//!   multi-megabyte cached prefix.
//! - The `relative` basis, used by Ramjet's published Dynamo comparison, credits the warmest
//!   worker's capped overlap minus how far a worker trails it, capped. Differences near the
//!   leader then survive the cap, which keeps long sessions sticky. `marginal` credits overlap
//!   above the coldest worker; `absolute` credits capped raw overlap.
//! - Ramjet charges each in-flight request `clamp(ceil(uncached_prompt_bytes / 32 KB), 1, 8)`
//!   load units. Its non-default phase-aware mode keeps only one unit once the first token
//!   streams. Dynamo exposes load per worker, not per request, so this port approximates the
//!   phase-aware mode with `active_requests + active_prefill_tokens / load_unit_tokens`: up to one
//!   more unit per prefilling request than Ramjet's rounding, and no per-request cap.
//!
//! Ramjet builds its own approximate prefix index from served responses, with an eviction
//! horizon to age out stale blocks. This port reads Dynamo's KV-event index instead, which
//! observes evictions directly.
//!
//! TODO(worker-taints): port Ramjet's long-prompt lane, which confines very long prompts to
//! designated replicas. It needs worker taints on policy candidates: a filter sees one worker at
//! a time, so it cannot fall back to every worker when no lane replica is eligible.

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

use crate::signals::{device_overlap_blocks, rotate_by_worker};

/// Policy type selected by `worker_selection.instances[].type`.
pub const POLICY_TYPE: &str = "ramjet";

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum AffinityBasis {
    Absolute,
    Marginal,
    Relative,
}

/// Defaults follow Ramjet's, except `basis`, which follows its published Dynamo comparison, and
/// the load model, which approximates the phase-aware mode.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
struct Parameters {
    alpha: f64,
    affinity_block_tokens: usize,
    max_affinity_blocks: usize,
    load_unit_tokens: usize,
    basis: AffinityBasis,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            alpha: 4.0,
            affinity_block_tokens: 512,
            max_affinity_blocks: 32,
            load_unit_tokens: 8_192,
            basis: AffinityBasis::Relative,
        }
    }
}

impl Parameters {
    fn validate(&self) -> Result<(), WorkerSelectionPolicyProviderError> {
        if !self.alpha.is_finite() || self.alpha < 0.0 {
            return Err(WorkerSelectionPolicyProviderError::new(
                "alpha must be a finite non-negative number",
            ));
        }
        if self.affinity_block_tokens == 0
            || self.max_affinity_blocks == 0
            || self.load_unit_tokens == 0
        {
            return Err(WorkerSelectionPolicyProviderError::new(
                "affinity_block_tokens, max_affinity_blocks, and load_unit_tokens must be positive",
            ));
        }
        Ok(())
    }

    fn credited_affinity(&self, overlap: usize, coldest: usize, warmest: usize) -> usize {
        let cap = self.max_affinity_blocks;
        match self.basis {
            AffinityBasis::Absolute => overlap.min(cap),
            AffinityBasis::Marginal => overlap.saturating_sub(coldest).min(cap),
            AffinityBasis::Relative => warmest
                .min(cap)
                .saturating_sub(warmest.saturating_sub(overlap).min(cap)),
        }
    }
}

struct RamjetPicker {
    parameters: Parameters,
    rotation: usize,
    overlaps: Vec<usize>,
    tied: Vec<usize>,
}

impl WorkerPicker for RamjetPicker {
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
        let parameters = self.parameters;
        let block_size = f64::from(context.block_size());
        self.overlaps.clear();
        self.overlaps.extend(cache.iter().map(|cache| {
            (device_overlap_blocks(cache) * block_size / parameters.affinity_block_tokens as f64)
                .max(0.0) as usize
        }));
        let coldest = self.overlaps.iter().copied().min().unwrap_or(0);
        let warmest = self.overlaps.iter().copied().max().unwrap_or(0);
        let score = |row: usize| {
            let load = &load[row];
            let load_units = load.active_requests() as f64
                + load.active_prefill_tokens() as f64 / parameters.load_unit_tokens as f64;
            let overlap = self.overlaps[row];
            let affinity = parameters.credited_affinity(overlap, coldest, warmest);
            (affinity as f64 - parameters.alpha * load_units, overlap)
        };
        let best = (0..candidates.len())
            .map(score)
            .max_by(|left, right| left.0.total_cmp(&right.0).then(left.1.cmp(&right.1)))
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))?;
        self.tied.clear();
        self.tied
            .extend((0..candidates.len()).filter(|&row| score(row) == best));
        let row = rotate_by_worker(
            &mut self.tied,
            |row| candidates[row].worker(),
            self.rotation,
        )
        .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))?;
        self.rotation = self.rotation.wrapping_add(1);
        Ok(row)
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
                Box::new(RamjetPicker {
                    parameters,
                    rotation: 0,
                    overlaps: Vec::new(),
                    tied: Vec::new(),
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

    fn policy(parameters: Parameters) -> WorkerSelectionPolicy {
        WorkerSelectionPolicy::new(
            KvRouterConfig::default(),
            "test",
            Vec::new(),
            Box::new(RamjetPicker {
                parameters,
                rotation: 0,
                overlaps: Vec::new(),
                tied: Vec::new(),
            }),
        )
    }

    /// Affinity blocks of 16 tokens make one Dynamo block one Ramjet block.
    fn unit_blocks(basis: AffinityBasis) -> Parameters {
        Parameters {
            affinity_block_tokens: 16,
            basis,
            ..Parameters::default()
        }
    }

    #[test]
    fn capped_affinity_yields_to_a_large_load_gap() {
        let policy = policy(unit_blocks(AffinityBasis::Absolute));
        // 40 cached blocks cap at 32: 32 − 4 × 7 = 4 beats the idle worker's 0.
        let workers = [Worker::new(0), Worker::new(1).cached(40).requests(7)];
        assert_eq!(select(&policy, request(64, 1), &workers), 1);
        // A ninth request tips it: 32 − 4 × 9 = −4.
        let workers = [Worker::new(0), Worker::new(1).cached(40).requests(9)];
        assert_eq!(select(&policy, request(64, 1), &workers), 0);
    }

    #[test]
    fn relative_basis_keeps_differences_past_the_cap() {
        // Worker 1 holds a 100-block session, worker 0 the first 90 blocks of it. Absolute
        // affinity caps both at 32, so load decides: 32 − 4 = 28 beats 32 − 8 = 24. Relative
        // affinity credits worker 0 with 32 − 10 = 22, and 22 − 4 = 18 loses to 24.
        let workers = [
            Worker::new(0).cached(90).requests(1),
            Worker::new(1).cached(100).requests(2),
        ];
        let select_with = |basis| select(&policy(unit_blocks(basis)), request(128, 1), &workers);
        assert_eq!(select_with(AffinityBasis::Absolute), 0);
        assert_eq!(select_with(AffinityBasis::Relative), 1);
    }

    #[test]
    fn marginal_basis_discounts_a_prefix_every_worker_caches() {
        // A 40-block prefix shared by both workers saturates the cap, so absolute affinity sees
        // no difference and load decides: 32 beats 32 − 4. Marginal affinity credits only the
        // 8 blocks beyond the shared prefix, and 8 − 4 beats 0.
        let workers = [
            Worker::new(0).cached(40),
            Worker::new(1).cached(48).requests(1),
        ];
        let select_with = |basis| select(&policy(unit_blocks(basis)), request(64, 1), &workers);
        assert_eq!(select_with(AffinityBasis::Absolute), 0);
        assert_eq!(select_with(AffinityBasis::Marginal), 1);
    }

    #[test]
    fn prefill_backlog_counts_as_fractional_load() {
        let policy = policy(unit_blocks(AffinityBasis::Absolute));
        // Worker 1 alone would score 10 − 4 = 6. Its 16k queued prefill tokens add two load
        // units, and 10 − 4 × 3 loses to the idle worker's 0.
        let workers = [
            Worker::new(0),
            Worker::new(1).cached(10).requests(1).prefill(16_384),
        ];
        assert_eq!(select(&policy, request(64, 1), &workers), 0);
    }

    #[test]
    fn ties_rotate_across_workers() {
        let policy = policy(Parameters::default());
        let workers = [Worker::new(0), Worker::new(1), Worker::new(2)];
        let picks: Vec<_> = (0..3)
            .map(|_| select(&policy, request(4, 1), &workers))
            .collect();
        assert_eq!(picks, [0, 1, 2]);
    }
}
