// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use dynamo_kv_router::indexer::cuckoo::ProducerIdentity;
use dynamo_kv_router::protocols::{ActiveLoad, WorkerId, WorkerWithDpRank};

use crate::local_model::runtime_config::ModelRuntimeConfig;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct LoadCapacity {
    total_kv_blocks: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct PoolLoadState {
    capacities: HashMap<WorkerWithDpRank, LoadCapacity>,
    observations: HashMap<WorkerWithDpRank, u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LoadObservationOutcome {
    UnknownRank,
    IgnoredAdvisory,
    Updated,
}

/// DC-wide load derived from worker-authoritative `kv_used_blocks` reports.
///
/// Router-local active decode and prefill lanes are intentionally excluded until
/// their events carry publisher identity and can be aggregated across replicas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolLoadSnapshot {
    pub producer: ProducerIdentity,
    /// Aggregate authoritative KV usage, available only after every declared rank
    /// has reported at least once for the current capacity generation.
    pub kv_used_blocks: Option<u64>,
    /// Aggregate KV capacity, available only when every declared rank publishes a
    /// non-zero capacity.
    pub total_kv_blocks: Option<u64>,
    /// Declared ranks that have published authoritative KV usage.
    pub kv_observed_ranks: usize,
    /// Declared ranks whose KV capacity is known and non-zero.
    pub kv_capacity_ranks: usize,
    /// Worker ranks declared by the current runtime configuration.
    pub kv_expected_ranks: usize,
}

impl PoolLoadSnapshot {
    pub fn has_degraded_coverage(self) -> bool {
        self.kv_expected_ranks == 0
            || self.kv_observed_ranks < self.kv_expected_ranks
            || self.kv_capacity_ranks < self.kv_expected_ranks
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
        let capacities = match load_ranks_from_configs(runtime_configs) {
            Ok(capacities) => capacities,
            Err(error) => {
                // Never leave the previously authoritative snapshot live after an
                // invalid capacity refresh. The registry publishes this empty state
                // before returning the error to its caller.
                self.capacities.clear();
                self.observations.clear();
                return Err(error);
            }
        };
        if self.capacities == capacities {
            return Ok(false);
        }
        self.observations.retain(|rank, _| {
            self.capacities
                .get(rank)
                .is_some_and(|previous| capacities.get(rank) == Some(previous))
        });
        self.capacities = capacities;
        Ok(true)
    }

    pub(super) fn observe(&mut self, load: ActiveLoad) -> LoadObservationOutcome {
        let rank = WorkerWithDpRank::new(load.worker_id, load.dp_rank);
        if !self.capacities.contains_key(&rank) {
            return LoadObservationOutcome::UnknownRank;
        }
        let Some(kv_used_blocks) = load.kv_used_blocks else {
            // active_decode_blocks and active_prefill_tokens can be emitted by
            // multiple router replicas without publisher identity. They are not a
            // globally authoritative DC load signal and are intentionally ignored.
            // Return a distinct accepted outcome so the collector does not
            // misdiagnose an advisory report as an unknown-rank event.
            return LoadObservationOutcome::IgnoredAdvisory;
        };
        self.observations.insert(rank, kv_used_blocks);
        LoadObservationOutcome::Updated
    }

    pub(super) fn clear_observations(&mut self) -> bool {
        if self.observations.is_empty() {
            return false;
        }
        self.observations.clear();
        true
    }

    pub(super) fn snapshot(&self, producer: ProducerIdentity) -> PoolLoadSnapshot {
        let mut kv_used_blocks = 0_u64;
        let mut total_kv_blocks = 0_u64;
        let mut snapshot = PoolLoadSnapshot {
            producer,
            kv_used_blocks: None,
            total_kv_blocks: None,
            kv_observed_ranks: 0,
            kv_capacity_ranks: 0,
            kv_expected_ranks: 0,
        };
        for (rank, capacity) in &self.capacities {
            snapshot.kv_expected_ranks = snapshot.kv_expected_ranks.saturating_add(1);
            if let Some(total) = capacity.total_kv_blocks {
                snapshot.kv_capacity_ranks = snapshot.kv_capacity_ranks.saturating_add(1);
                total_kv_blocks = total_kv_blocks.saturating_add(total);
            }
            if let Some(value) = self.observations.get(rank) {
                snapshot.kv_observed_ranks = snapshot.kv_observed_ranks.saturating_add(1);
                kv_used_blocks = kv_used_blocks.saturating_add(*value);
            }
        }
        if snapshot.kv_expected_ranks != 0
            && snapshot.kv_observed_ranks == snapshot.kv_expected_ranks
        {
            snapshot.kv_used_blocks = Some(kv_used_blocks);
        }
        if snapshot.kv_expected_ranks != 0
            && snapshot.kv_capacity_ranks == snapshot.kv_expected_ranks
        {
            snapshot.total_kv_blocks = Some(total_kv_blocks);
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
        // registering process and uses zero as an unknown-capacity sentinel. Runtime
        // config does not carry backend identity, and zero is never a usable pressure
        // denominator for any backend, so normalize it fail-closed for every engine.
        let total_kv_blocks = config.total_kv_blocks.filter(|&total| total != 0);
        for dp_rank in config.data_parallel_start_rank..end {
            ranks.insert(
                WorkerWithDpRank::new(worker_id, dp_rank),
                LoadCapacity { total_kv_blocks },
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
    fn authoritative_kv_reaches_full_coverage() {
        let mut state = PoolLoadState::from_runtime_configs(&HashMap::from([(
            9,
            config(0, 1, Some(100), None),
        )]))
        .unwrap();
        let mut report = load(9, 0);
        report.kv_used_blocks = Some(40);
        report.active_decode_blocks = Some(10);
        assert_eq!(state.observe(report), LoadObservationOutcome::Updated);

        let snapshot = state.snapshot(producer());
        assert_eq!(snapshot.kv_used_blocks, Some(40));
        assert_eq!(snapshot.total_kv_blocks, Some(100));
        assert_eq!(snapshot.kv_observed_ranks, 1);
        assert_eq!(snapshot.kv_capacity_ranks, 1);
        assert_eq!(snapshot.kv_expected_ranks, 1);
        assert!(!snapshot.has_degraded_coverage());
    }

    #[test]
    fn unknown_capacity_preserves_observations_but_degrades_coverage() {
        let mut state = PoolLoadState::from_runtime_configs(&HashMap::from([(
            9,
            config(0, 2, Some(0), Some(2_048)),
        )]))
        .unwrap();
        let mut report = load(9, 0);
        report.kv_used_blocks = Some(40);
        report.active_prefill_tokens = Some(512);
        assert_eq!(state.observe(report), LoadObservationOutcome::Updated);
        let mut second_report = load(9, 1);
        second_report.kv_used_blocks = Some(30);
        assert_eq!(
            state.observe(second_report),
            LoadObservationOutcome::Updated
        );

        let snapshot = state.snapshot(producer());
        assert_eq!(snapshot.kv_expected_ranks, 2);
        assert_eq!(snapshot.kv_observed_ranks, 2);
        assert_eq!(snapshot.kv_capacity_ranks, 0);
        assert_eq!(snapshot.kv_used_blocks, Some(70));
        assert_eq!(snapshot.total_kv_blocks, None);
        assert!(snapshot.has_degraded_coverage());
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
    fn partial_reports_do_not_expose_partial_aggregate() {
        let mut state = PoolLoadState::from_runtime_configs(&HashMap::from([(
            9,
            config(0, 2, Some(100), Some(2_048)),
        )]))
        .unwrap();
        let mut first = load(9, 0);
        first.kv_used_blocks = Some(40);
        first.active_prefill_tokens = Some(512);
        assert_eq!(state.observe(first), LoadObservationOutcome::Updated);
        let mut scheduler_only = load(9, 0);
        scheduler_only.active_prefill_tokens = Some(512);
        scheduler_only.active_decode_blocks = Some(30);
        assert_eq!(
            state.observe(scheduler_only),
            LoadObservationOutcome::IgnoredAdvisory
        );
        let mut second = load(9, 0);
        second.active_decode_blocks = Some(30);
        assert_eq!(
            state.observe(second),
            LoadObservationOutcome::IgnoredAdvisory
        );
        let mut replacement = load(9, 0);
        replacement.kv_used_blocks = Some(42);
        assert_eq!(state.observe(replacement), LoadObservationOutcome::Updated);

        let snapshot = state.snapshot(producer());
        assert_eq!(snapshot.kv_used_blocks, None);
        assert_eq!(snapshot.total_kv_blocks, Some(200));
        assert_eq!(snapshot.kv_observed_ranks, 1);
        assert_eq!(snapshot.kv_capacity_ranks, 2);
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
        assert_eq!(state.observe(unknown), LoadObservationOutcome::UnknownRank);
        let mut known = load(9, 0);
        known.kv_used_blocks = Some(40);
        assert_eq!(state.observe(known), LoadObservationOutcome::Updated);
        assert_eq!(state.snapshot(producer()).kv_used_blocks, Some(40));
        assert_eq!(state.snapshot(producer()).kv_observed_ranks, 1);
        assert!(state.clear_observations());
        assert_eq!(state.snapshot(producer()).kv_used_blocks, None);
        assert_eq!(state.snapshot(producer()).kv_observed_ranks, 0);
    }

    #[test]
    fn capacity_change_clears_only_affected_rank_observations() {
        let mut state = PoolLoadState::from_runtime_configs(&HashMap::from([
            (9, config(0, 1, Some(100), Some(2_048))),
            (10, config(0, 1, Some(100), Some(2_048))),
        ]))
        .unwrap();
        let mut changed = load(9, 0);
        changed.kv_used_blocks = Some(40);
        assert_eq!(state.observe(changed), LoadObservationOutcome::Updated);
        let mut unchanged = load(10, 0);
        unchanged.kv_used_blocks = Some(30);
        assert_eq!(state.observe(unchanged), LoadObservationOutcome::Updated);
        assert_eq!(state.snapshot(producer()).kv_used_blocks, Some(70));

        assert!(
            state
                .replace_capacity(&HashMap::from([
                    (9, config(0, 1, Some(200), Some(2_048))),
                    (10, config(0, 1, Some(100), Some(2_048))),
                ]))
                .unwrap()
        );
        let snapshot = state.snapshot(producer());
        assert_eq!(snapshot.kv_expected_ranks, 2);
        assert_eq!(snapshot.kv_observed_ranks, 1);
        assert_eq!(snapshot.kv_used_blocks, None);
        assert_eq!(snapshot.total_kv_blocks, Some(300));
        assert!(snapshot.has_degraded_coverage());
    }

    #[test]
    fn invalid_capacity_clears_previous_authoritative_state() {
        let mut state = PoolLoadState::from_runtime_configs(&HashMap::from([(
            9,
            config(0, 1, Some(100), None),
        )]))
        .unwrap();
        let mut report = load(9, 0);
        report.kv_used_blocks = Some(40);
        assert_eq!(state.observe(report), LoadObservationOutcome::Updated);
        assert!(!state.snapshot(producer()).has_degraded_coverage());

        let error = state
            .replace_capacity(&HashMap::from([(9, config(0, 0, Some(100), None))]))
            .unwrap_err();
        assert!(error.to_string().contains("zero data_parallel_size"));

        let snapshot = state.snapshot(producer());
        assert_eq!(snapshot.kv_used_blocks, None);
        assert_eq!(snapshot.total_kv_blocks, None);
        assert_eq!(snapshot.kv_observed_ranks, 0);
        assert_eq!(snapshot.kv_capacity_ranks, 0);
        assert_eq!(snapshot.kv_expected_ranks, 0);
        assert!(snapshot.has_degraded_coverage());
    }
}
