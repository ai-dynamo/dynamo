// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use dynamo_kv_router::indexer::cuckoo::ProducerIdentity;
use dynamo_kv_router::protocols::{ActiveLoad, WorkerId, WorkerWithDpRank};

use crate::local_model::runtime_config::ModelRuntimeConfig;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct LoadCapacity {
    total_kv_blocks: Option<u64>,
    max_num_batched_tokens: Option<u64>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct LoadObservation {
    kv_used_blocks: Option<u64>,
    active_decode_blocks: Option<u64>,
    active_prefill_tokens: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct PoolLoadState {
    capacities: HashMap<WorkerWithDpRank, LoadCapacity>,
    observations: HashMap<WorkerWithDpRank, LoadObservation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolLoadSnapshot {
    pub producer: ProducerIdentity,
    pub kv_used_blocks: u64,
    pub total_kv_blocks: u64,
    pub kv_observed_ranks: usize,
    pub kv_expected_ranks: usize,
    pub active_decode_blocks: u64,
    pub decode_observed_ranks: usize,
    pub decode_expected_ranks: usize,
    pub active_prefill_tokens: u64,
    pub prefill_token_capacity: u64,
    pub prefill_observed_ranks: usize,
    pub prefill_expected_ranks: usize,
}

impl PoolLoadSnapshot {
    pub fn has_degraded_coverage(self) -> bool {
        if self.kv_expected_ranks == 0
            && self.decode_expected_ranks == 0
            && self.prefill_expected_ranks == 0
        {
            return true;
        }
        self.kv_observed_ranks < self.kv_expected_ranks
            || self.decode_observed_ranks < self.decode_expected_ranks
            || self.prefill_observed_ranks < self.prefill_expected_ranks
    }
}

impl PoolLoadState {
    pub(super) fn from_runtime_configs(
        runtime_configs: &HashMap<WorkerId, ModelRuntimeConfig>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            capacities: load_ranks_from_configs(runtime_configs)?,
            observations: HashMap::new(),
        })
    }

    pub(super) fn replace_capacity(
        &mut self,
        runtime_configs: &HashMap<WorkerId, ModelRuntimeConfig>,
    ) -> anyhow::Result<bool> {
        let capacities = load_ranks_from_configs(runtime_configs)?;
        if self.capacities == capacities {
            return Ok(false);
        }
        self.observations
            .retain(|rank, _| capacities.contains_key(rank));
        self.capacities = capacities;
        Ok(true)
    }

    pub(super) fn observe(&mut self, load: ActiveLoad) -> bool {
        let rank = WorkerWithDpRank::new(load.worker_id, load.dp_rank);
        if !self.capacities.contains_key(&rank) {
            return false;
        }
        if load.kv_used_blocks.is_none()
            && load.active_decode_blocks.is_none()
            && load.active_prefill_tokens.is_none()
        {
            return true;
        }
        let observation = self.observations.entry(rank).or_default();
        if let Some(value) = load.kv_used_blocks {
            observation.kv_used_blocks = Some(value);
        }
        if let Some(value) = load.active_decode_blocks {
            observation.active_decode_blocks = Some(value);
        }
        if let Some(value) = load.active_prefill_tokens {
            observation.active_prefill_tokens = Some(value);
        }
        true
    }

    pub(super) fn clear_observations(&mut self) -> bool {
        if self.observations.is_empty() {
            return false;
        }
        self.observations.clear();
        true
    }

    pub(super) fn snapshot(&self, producer: ProducerIdentity) -> PoolLoadSnapshot {
        let mut snapshot = PoolLoadSnapshot {
            producer,
            kv_used_blocks: 0,
            total_kv_blocks: 0,
            kv_observed_ranks: 0,
            kv_expected_ranks: 0,
            active_decode_blocks: 0,
            decode_observed_ranks: 0,
            decode_expected_ranks: 0,
            active_prefill_tokens: 0,
            prefill_token_capacity: 0,
            prefill_observed_ranks: 0,
            prefill_expected_ranks: 0,
        };
        for (rank, capacity) in &self.capacities {
            let observation = self.observations.get(rank);
            if let Some(total) = capacity.total_kv_blocks {
                snapshot.kv_expected_ranks = snapshot.kv_expected_ranks.saturating_add(1);
                snapshot.decode_expected_ranks = snapshot.decode_expected_ranks.saturating_add(1);
                snapshot.total_kv_blocks = snapshot.total_kv_blocks.saturating_add(total);
                if let Some(value) = observation.and_then(|value| value.kv_used_blocks) {
                    snapshot.kv_observed_ranks = snapshot.kv_observed_ranks.saturating_add(1);
                    snapshot.kv_used_blocks = snapshot.kv_used_blocks.saturating_add(value);
                }
                if let Some(value) = observation.and_then(|value| value.active_decode_blocks) {
                    snapshot.decode_observed_ranks =
                        snapshot.decode_observed_ranks.saturating_add(1);
                    snapshot.active_decode_blocks =
                        snapshot.active_decode_blocks.saturating_add(value);
                }
            }
            if let Some(total) = capacity.max_num_batched_tokens {
                snapshot.prefill_expected_ranks = snapshot.prefill_expected_ranks.saturating_add(1);
                snapshot.prefill_token_capacity =
                    snapshot.prefill_token_capacity.saturating_add(total);
                if let Some(value) = observation.and_then(|value| value.active_prefill_tokens) {
                    snapshot.prefill_observed_ranks =
                        snapshot.prefill_observed_ranks.saturating_add(1);
                    snapshot.active_prefill_tokens =
                        snapshot.active_prefill_tokens.saturating_add(value);
                }
            }
        }
        snapshot
    }
}

const MAX_LOAD_RANKS_PER_WORKER: u32 = 4096;

fn load_ranks_from_configs(
    runtime_configs: &HashMap<WorkerId, ModelRuntimeConfig>,
) -> anyhow::Result<HashMap<WorkerWithDpRank, LoadCapacity>> {
    let mut ranks = HashMap::new();
    for (&worker_id, config) in runtime_configs {
        anyhow::ensure!(
            config.data_parallel_size != 0,
            "worker {worker_id} has zero data_parallel_size"
        );
        anyhow::ensure!(
            config.data_parallel_size <= MAX_LOAD_RANKS_PER_WORKER,
            "worker {worker_id} declares {} data-parallel ranks, above the supported {}",
            config.data_parallel_size,
            MAX_LOAD_RANKS_PER_WORKER
        );
        let end = config
            .data_parallel_start_rank
            .checked_add(config.data_parallel_size)
            .ok_or_else(|| {
                anyhow::anyhow!("worker {worker_id} data-parallel rank range overflow")
            })?;
        // vLLM's Ray data-parallel backend cannot propagate num_gpu_blocks to the
        // registering process and publishes total_kv_blocks=0 in its place, so a zero
        // total is "capacity unknown", not a zero-block worker. Advertising it as real
        // capacity would put the ranks in kv_expected_ranks with an unreachable total.
        let total_kv_blocks = config.total_kv_blocks.filter(|&total| total != 0);
        for dp_rank in config.data_parallel_start_rank..end {
            ranks.insert(
                WorkerWithDpRank::new(worker_id, dp_rank),
                LoadCapacity {
                    total_kv_blocks,
                    max_num_batched_tokens: config.max_num_batched_tokens,
                },
            );
        }
    }
    Ok(ranks)
}

#[cfg(test)]
mod tests {
    use dynamo_kv_router::identity::{
        CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, PoolId, RoutingScopeId,
    };
    use dynamo_kv_router::indexer::cuckoo::{CkfConfig, DcCkfState};

    use super::*;

    fn producer() -> ProducerIdentity {
        let format = DcCkfState::new(CkfConfig::new(32))
            .expect("fixture state")
            .format();
        ProducerIdentity::new(
            PoolId::new(
                IndexerDomainId::new(
                    CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
                    RoutingScopeId::new([2; 16], IdentitySource::Explicit),
                ),
                DcId::new(3),
            ),
            7,
            11,
            format,
        )
    }

    fn config(
        start_rank: u32,
        rank_count: u32,
        kv_blocks: Option<u64>,
        prefill_tokens: Option<u64>,
    ) -> ModelRuntimeConfig {
        ModelRuntimeConfig {
            data_parallel_start_rank: start_rank,
            data_parallel_size: rank_count,
            total_kv_blocks: kv_blocks,
            max_num_batched_tokens: prefill_tokens,
            ..ModelRuntimeConfig::default()
        }
    }

    fn load(worker_id: WorkerId, dp_rank: u32) -> ActiveLoad {
        ActiveLoad {
            worker_id,
            dp_rank,
            ..ActiveLoad::default()
        }
    }

    #[test]
    fn kv_only_capacity_reaches_full_coverage_without_prefill_ranks() {
        let mut state = PoolLoadState::from_runtime_configs(&HashMap::from([(
            9,
            config(0, 1, Some(100), None),
        )]))
        .unwrap();
        let mut report = load(9, 0);
        report.kv_used_blocks = Some(40);
        report.active_decode_blocks = Some(10);
        assert!(state.observe(report));

        let snapshot = state.snapshot(producer());
        assert_eq!(snapshot.prefill_expected_ranks, 0);
        assert!(!snapshot.has_degraded_coverage());
    }

    #[test]
    fn ray_dp_zero_kv_total_is_unknown_capacity_not_zero_blocks() {
        let mut state = PoolLoadState::from_runtime_configs(&HashMap::from([(
            9,
            config(0, 2, Some(0), Some(2_048)),
        )]))
        .unwrap();
        let mut report = load(9, 0);
        report.kv_used_blocks = Some(40);
        report.active_prefill_tokens = Some(512);
        assert!(state.observe(report));

        let snapshot = state.snapshot(producer());
        assert_eq!(snapshot.kv_expected_ranks, 0);
        assert_eq!(snapshot.total_kv_blocks, 0);
        assert_eq!(snapshot.kv_used_blocks, 0);
        assert_eq!(snapshot.prefill_expected_ranks, 2);
        assert_eq!(snapshot.active_prefill_tokens, 512);
    }

    #[test]
    fn oversized_data_parallel_declaration_is_rejected() {
        let error = PoolLoadState::from_runtime_configs(&HashMap::from([(
            9,
            config(0, MAX_LOAD_RANKS_PER_WORKER + 1, Some(100), Some(2_048)),
        )]))
        .unwrap_err();
        assert!(error.to_string().contains("data-parallel ranks"));
    }

    #[test]
    fn partial_reports_preserve_independent_latest_values() {
        let mut state = PoolLoadState::from_runtime_configs(&HashMap::from([(
            9,
            config(0, 2, Some(100), Some(2_048)),
        )]))
        .unwrap();
        let mut first = load(9, 0);
        first.kv_used_blocks = Some(40);
        first.active_prefill_tokens = Some(512);
        assert!(state.observe(first));
        let mut second = load(9, 0);
        second.active_decode_blocks = Some(30);
        assert!(state.observe(second));
        let mut replacement = load(9, 0);
        replacement.kv_used_blocks = Some(42);
        assert!(state.observe(replacement));

        let snapshot = state.snapshot(producer());
        assert_eq!(snapshot.kv_used_blocks, 42);
        assert_eq!(snapshot.active_decode_blocks, 30);
        assert_eq!(snapshot.active_prefill_tokens, 512);
        assert_eq!(snapshot.total_kv_blocks, 200);
        assert_eq!(snapshot.prefill_token_capacity, 4_096);
        assert_eq!(snapshot.kv_observed_ranks, 1);
        assert_eq!(snapshot.kv_expected_ranks, 2);
        assert!(snapshot.has_degraded_coverage());
    }

    #[test]
    fn unknown_ranks_are_ignored_and_disconnect_clears_observations() {
        let mut state = PoolLoadState::from_runtime_configs(&HashMap::from([(
            9,
            config(0, 1, Some(100), Some(2_048)),
        )]))
        .unwrap();
        let mut unknown = load(9, 1);
        unknown.kv_used_blocks = Some(40);
        assert!(!state.observe(unknown));
        let mut known = load(9, 0);
        known.kv_used_blocks = Some(40);
        assert!(state.observe(known));
        assert_eq!(state.snapshot(producer()).kv_observed_ranks, 1);
        assert!(state.clear_observations());
        assert_eq!(state.snapshot(producer()).kv_observed_ranks, 0);
    }

    #[test]
    fn capacity_change_drops_departed_rank_observations() {
        let mut state = PoolLoadState::from_runtime_configs(&HashMap::from([(
            9,
            config(0, 2, Some(100), Some(2_048)),
        )]))
        .unwrap();
        let mut departed = load(9, 1);
        departed.kv_used_blocks = Some(40);
        assert!(state.observe(departed));
        assert!(
            state
                .replace_capacity(&HashMap::from([(9, config(0, 1, Some(100), Some(2_048)),)]))
                .unwrap()
        );
        let snapshot = state.snapshot(producer());
        assert_eq!(snapshot.kv_expected_ranks, 1);
        assert_eq!(snapshot.kv_observed_ranks, 0);
    }
}
