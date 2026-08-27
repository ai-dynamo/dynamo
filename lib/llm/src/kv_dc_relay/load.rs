// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::time::{Duration, Instant};

use dynamo_kv_router::indexer::cuckoo::ProducerIdentity;
use dynamo_kv_router::protocols::{ActiveLoad, WorkerId, WorkerWithDpRank};
use dynamo_runtime::transports::event_plane::EventEnvelope;

use crate::local_model::runtime_config::ModelRuntimeConfig;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct LoadCapacity {
    total_kv_blocks: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct PoolLoadState {
    capacities: HashMap<WorkerWithDpRank, LoadCapacity>,
    observations: HashMap<WorkerWithDpRank, LoadObservation>,
    publisher_sequences: HashMap<u64, u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ObservedValue {
    value: u64,
    publisher_id: u64,
    received_at: Instant,
    published_at_unix_ms: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct LoadObservation {
    kv_used_blocks: Option<ObservedValue>,
    active_decode_blocks: Option<ObservedValue>,
    active_prefill_tokens: Option<ObservedValue>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LoadObservationOutcome {
    UnknownRank,
    IgnoredStale,
    Updated,
}

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
    pub active_decode_blocks: Option<u64>,
    pub active_prefill_tokens: Option<u64>,
    pub source_observed_at_unix_ms: u64,
    complete: bool,
}

impl PoolLoadSnapshot {
    pub fn has_degraded_coverage(self) -> bool {
        self.kv_expected_ranks == 0
            || self.kv_observed_ranks < self.kv_expected_ranks
            || self.kv_capacity_ranks < self.kv_expected_ranks
    }

    pub fn is_complete(self) -> bool {
        self.complete
    }
}

const LOAD_FRESHNESS: Duration = Duration::from_secs(5);

impl PoolLoadState {
    pub(super) fn from_runtime_configs(
        runtime_configs: &HashMap<WorkerId, ModelRuntimeConfig>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            capacities: load_ranks_from_configs(runtime_configs)?,
            observations: HashMap::new(),
            publisher_sequences: HashMap::new(),
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
                self.clear_source_state();
                return Err(error);
            }
        };
        if self.capacities == capacities {
            return Ok(false);
        }
        self.capacities = capacities;
        self.clear_source_state();
        Ok(true)
    }

    pub(super) fn observe(
        &mut self,
        envelope: &EventEnvelope,
        load: ActiveLoad,
    ) -> LoadObservationOutcome {
        self.observe_at(envelope, load, Instant::now())
    }

    fn observe_at(
        &mut self,
        envelope: &EventEnvelope,
        load: ActiveLoad,
        received_at: Instant,
    ) -> LoadObservationOutcome {
        let rank = WorkerWithDpRank::new(load.worker_id, load.dp_rank);
        if !self.capacities.contains_key(&rank) {
            return LoadObservationOutcome::UnknownRank;
        }

        if let Some(previous) = self.publisher_sequences.get(&envelope.publisher_id) {
            if envelope.sequence <= *previous {
                return LoadObservationOutcome::IgnoredStale;
            }
            if envelope.sequence > previous.saturating_add(1) {
                for observation in self.observations.values_mut() {
                    clear_publisher_values(observation, envelope.publisher_id);
                }
            }
        }
        self.publisher_sequences
            .insert(envelope.publisher_id, envelope.sequence);

        let observed = |value: Option<u64>| {
            value.map(|value| ObservedValue {
                value,
                publisher_id: envelope.publisher_id,
                received_at,
                published_at_unix_ms: envelope.published_at,
            })
        };
        let observation = self.observations.entry(rank).or_default();
        if load.kv_used_blocks.is_some() {
            observation.kv_used_blocks = observed(load.kv_used_blocks);
        }
        if load.active_decode_blocks.is_some() {
            observation.active_decode_blocks = observed(load.active_decode_blocks);
        }
        if load.active_prefill_tokens.is_some() {
            observation.active_prefill_tokens = observed(load.active_prefill_tokens);
        }
        LoadObservationOutcome::Updated
    }

    pub(super) fn clear_observations(&mut self) -> bool {
        if self.observations.is_empty() && self.publisher_sequences.is_empty() {
            return false;
        }
        self.clear_source_state();
        true
    }

    fn clear_source_state(&mut self) {
        self.observations.clear();
        self.publisher_sequences.clear();
    }

    pub(super) fn snapshot(&self, producer: ProducerIdentity, now: Instant) -> PoolLoadSnapshot {
        let mut kv_used_blocks = Some(0_u64);
        let mut total_kv_blocks = Some(0_u64);
        let mut active_decode_blocks = Some(0_u64);
        let mut active_prefill_tokens = Some(0_u64);
        let mut active_decode_ranks = 0_usize;
        let mut active_prefill_ranks = 0_usize;
        let mut source_observed_at_unix_ms = None;
        let mut valid = true;
        let mut snapshot = PoolLoadSnapshot {
            producer,
            kv_used_blocks: None,
            total_kv_blocks: None,
            kv_observed_ranks: 0,
            kv_capacity_ranks: 0,
            kv_expected_ranks: 0,
            active_decode_blocks: None,
            active_prefill_tokens: None,
            source_observed_at_unix_ms: 0,
            complete: false,
        };
        for (rank, capacity) in &self.capacities {
            snapshot.kv_expected_ranks = snapshot.kv_expected_ranks.saturating_add(1);
            if let Some(total) = capacity.total_kv_blocks {
                snapshot.kv_capacity_ranks = snapshot.kv_capacity_ranks.saturating_add(1);
                valid &= checked_add(&mut total_kv_blocks, total);
            }
            let Some(observation) = self.observations.get(rank) else {
                active_decode_blocks = None;
                active_prefill_tokens = None;
                continue;
            };
            if let Some(value) = observation
                .kv_used_blocks
                .filter(|value| observation_is_fresh(*value, now))
                .filter(|value| value.published_at_unix_ms > 0)
                .filter(|value| {
                    capacity
                        .total_kv_blocks
                        .is_none_or(|total| value.value <= total)
                })
            {
                snapshot.kv_observed_ranks = snapshot.kv_observed_ranks.saturating_add(1);
                valid &= checked_add(&mut kv_used_blocks, value.value);
                source_observed_at_unix_ms = Some(
                    source_observed_at_unix_ms
                        .map_or(value.published_at_unix_ms, |current: u64| {
                            current.min(value.published_at_unix_ms)
                        }),
                );
            }
            if let Some(value) = observation
                .active_decode_blocks
                .filter(|value| observation_is_fresh(*value, now))
            {
                active_decode_ranks = active_decode_ranks.saturating_add(1);
                let _ = checked_add(&mut active_decode_blocks, value.value);
            }
            if let Some(value) = observation
                .active_prefill_tokens
                .filter(|value| observation_is_fresh(*value, now))
            {
                active_prefill_ranks = active_prefill_ranks.saturating_add(1);
                let _ = checked_add(&mut active_prefill_tokens, value.value);
            }
        }
        if snapshot.kv_expected_ranks != 0
            && snapshot.kv_observed_ranks == snapshot.kv_expected_ranks
        {
            snapshot.kv_used_blocks = kv_used_blocks;
        }
        if snapshot.kv_expected_ranks != 0
            && snapshot.kv_capacity_ranks == snapshot.kv_expected_ranks
        {
            snapshot.total_kv_blocks = total_kv_blocks;
        }
        if snapshot.kv_expected_ranks != 0 && active_decode_ranks == snapshot.kv_expected_ranks {
            snapshot.active_decode_blocks = active_decode_blocks;
        }
        if snapshot.kv_expected_ranks != 0 && active_prefill_ranks == snapshot.kv_expected_ranks {
            snapshot.active_prefill_tokens = active_prefill_tokens;
        }
        snapshot.source_observed_at_unix_ms = source_observed_at_unix_ms.unwrap_or_default();
        snapshot.complete = valid
            && !snapshot.has_degraded_coverage()
            && snapshot.kv_used_blocks.is_some()
            && snapshot.total_kv_blocks.is_some()
            && snapshot.source_observed_at_unix_ms > 0;
        snapshot
    }
}

fn clear_publisher_values(observation: &mut LoadObservation, publisher_id: u64) {
    if observation
        .kv_used_blocks
        .is_some_and(|value| value.publisher_id == publisher_id)
    {
        observation.kv_used_blocks = None;
    }
    if observation
        .active_decode_blocks
        .is_some_and(|value| value.publisher_id == publisher_id)
    {
        observation.active_decode_blocks = None;
    }
    if observation
        .active_prefill_tokens
        .is_some_and(|value| value.publisher_id == publisher_id)
    {
        observation.active_prefill_tokens = None;
    }
}

fn observation_is_fresh(value: ObservedValue, now: Instant) -> bool {
    now.saturating_duration_since(value.received_at) <= LOAD_FRESHNESS
}

fn checked_add(total: &mut Option<u64>, value: u64) -> bool {
    *total = (*total).and_then(|total| total.checked_add(value));
    total.is_some()
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
    use dynamo_runtime::transports::event_plane::EventEnvelope;

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

    fn envelope(publisher_id: u64, sequence: u64, published_at: u64) -> EventEnvelope {
        EventEnvelope {
            publisher_id,
            sequence,
            published_at,
            topic: String::new(),
            payload: Default::default(),
        }
    }

    fn observe(state: &mut PoolLoadState, load: ActiveLoad) -> LoadObservationOutcome {
        let sequence = state.publisher_sequences.get(&1).copied().unwrap_or(0) + 1;
        state.observe(&envelope(1, sequence, sequence), load)
    }

    fn snapshot(state: &PoolLoadState) -> PoolLoadSnapshot {
        state.snapshot(producer(), Instant::now())
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
        assert_eq!(observe(&mut state, report), LoadObservationOutcome::Updated);

        let snapshot = snapshot(&state);
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
        assert_eq!(observe(&mut state, report), LoadObservationOutcome::Updated);
        let mut second_report = load(9, 1);
        second_report.kv_used_blocks = Some(30);
        assert_eq!(
            observe(&mut state, second_report),
            LoadObservationOutcome::Updated
        );

        let snapshot = snapshot(&state);
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
        assert_eq!(observe(&mut state, first), LoadObservationOutcome::Updated);
        let mut scheduler_only = load(9, 0);
        scheduler_only.active_prefill_tokens = Some(512);
        scheduler_only.active_decode_blocks = Some(30);
        assert_eq!(
            observe(&mut state, scheduler_only),
            LoadObservationOutcome::Updated
        );
        let mut second = load(9, 0);
        second.active_decode_blocks = Some(30);
        assert_eq!(observe(&mut state, second), LoadObservationOutcome::Updated);
        let mut replacement = load(9, 0);
        replacement.kv_used_blocks = Some(42);
        assert_eq!(
            observe(&mut state, replacement),
            LoadObservationOutcome::Updated
        );

        let snapshot = snapshot(&state);
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
        assert_eq!(
            observe(&mut state, unknown),
            LoadObservationOutcome::UnknownRank
        );
        let mut known = load(9, 0);
        known.kv_used_blocks = Some(40);
        assert_eq!(observe(&mut state, known), LoadObservationOutcome::Updated);
        assert_eq!(snapshot(&state).kv_used_blocks, Some(40));
        assert_eq!(snapshot(&state).kv_observed_ranks, 1);
        assert!(state.clear_observations());
        assert_eq!(snapshot(&state).kv_used_blocks, None);
        assert_eq!(snapshot(&state).kv_observed_ranks, 0);
    }

    #[test]
    fn capacity_change_requires_fresh_observations() {
        let mut state = PoolLoadState::from_runtime_configs(&HashMap::from([
            (9, config(0, 1, Some(100), Some(2_048))),
            (10, config(0, 1, Some(100), Some(2_048))),
        ]))
        .unwrap();
        let mut changed = load(9, 0);
        changed.kv_used_blocks = Some(40);
        assert_eq!(
            observe(&mut state, changed),
            LoadObservationOutcome::Updated
        );
        let mut unchanged = load(10, 0);
        unchanged.kv_used_blocks = Some(30);
        assert_eq!(
            observe(&mut state, unchanged),
            LoadObservationOutcome::Updated
        );
        assert_eq!(snapshot(&state).kv_used_blocks, Some(70));

        assert!(
            state
                .replace_capacity(&HashMap::from([
                    (9, config(0, 1, Some(200), Some(2_048))),
                    (10, config(0, 1, Some(100), Some(2_048))),
                ]))
                .unwrap()
        );
        let snapshot = snapshot(&state);
        assert_eq!(snapshot.kv_expected_ranks, 2);
        assert_eq!(snapshot.kv_observed_ranks, 0);
        assert_eq!(snapshot.kv_used_blocks, None);
        assert_eq!(snapshot.total_kv_blocks, Some(300));
        assert!(snapshot.has_degraded_coverage());
    }

    #[test]
    fn stale_and_gapped_publishers_cannot_leave_complete_load() {
        let mut state = PoolLoadState::from_runtime_configs(&HashMap::from([
            (9, config(0, 1, Some(100), None)),
            (10, config(0, 1, Some(100), None)),
        ]))
        .unwrap();
        let now = Instant::now();

        let mut first = load(9, 0);
        first.kv_used_blocks = Some(40);
        assert_eq!(
            state.observe_at(&envelope(1, 1, 10), first, now),
            LoadObservationOutcome::Updated
        );
        let mut second = load(10, 0);
        second.kv_used_blocks = Some(30);
        assert_eq!(
            state.observe_at(&envelope(1, 2, 20), second, now),
            LoadObservationOutcome::Updated
        );
        assert!(state.snapshot(producer(), now).is_complete());

        let mut stale = load(9, 0);
        stale.kv_used_blocks = Some(99);
        assert_eq!(
            state.observe_at(&envelope(1, 2, 30), stale, now),
            LoadObservationOutcome::IgnoredStale
        );
        assert_eq!(state.snapshot(producer(), now).kv_used_blocks, Some(70));

        let mut after_gap = load(9, 0);
        after_gap.kv_used_blocks = Some(41);
        assert_eq!(
            state.observe_at(&envelope(1, 4, 40), after_gap, now),
            LoadObservationOutcome::Updated
        );
        let snapshot = state.snapshot(producer(), now);
        assert!(!snapshot.is_complete());
        assert_eq!(snapshot.kv_observed_ranks, 1);

        let stale_snapshot =
            state.snapshot(producer(), now + LOAD_FRESHNESS + Duration::from_nanos(1));
        assert!(!stale_snapshot.is_complete());
        assert_eq!(stale_snapshot.kv_observed_ranks, 0);
    }

    #[test]
    fn scheduler_facts_replace_per_rank_without_affecting_kv_ownership() {
        let mut state = PoolLoadState::from_runtime_configs(&HashMap::from([(
            9,
            config(0, 1, Some(100), None),
        )]))
        .unwrap();
        let now = Instant::now();

        let mut worker = load(9, 0);
        worker.kv_used_blocks = Some(40);
        state.observe_at(&envelope(1, 1, 10), worker, now);
        let mut router = load(9, 0);
        router.active_decode_blocks = Some(12);
        router.active_prefill_tokens = Some(34);
        state.observe_at(&envelope(2, 1, 20), router, now);

        let snapshot = state.snapshot(producer(), now);
        assert!(snapshot.is_complete());
        assert_eq!(snapshot.kv_used_blocks, Some(40));
        assert_eq!(snapshot.active_decode_blocks, Some(12));
        assert_eq!(snapshot.active_prefill_tokens, Some(34));
        assert_eq!(snapshot.source_observed_at_unix_ms, 10);
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
        assert_eq!(observe(&mut state, report), LoadObservationOutcome::Updated);
        assert!(!snapshot(&state).has_degraded_coverage());

        let error = state
            .replace_capacity(&HashMap::from([(9, config(0, 0, Some(100), None))]))
            .unwrap_err();
        assert!(error.to_string().contains("zero data_parallel_size"));

        let snapshot = snapshot(&state);
        assert_eq!(snapshot.kv_used_blocks, None);
        assert_eq!(snapshot.total_kv_blocks, None);
        assert_eq!(snapshot.kv_observed_ranks, 0);
        assert_eq!(snapshot.kv_capacity_ranks, 0);
        assert_eq!(snapshot.kv_expected_ranks, 0);
        assert!(snapshot.has_degraded_coverage());
    }
}
