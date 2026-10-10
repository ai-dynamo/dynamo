// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use dynamo_kv_router::indexer::cuckoo::ProducerIdentity;
use dynamo_kv_router::protocols::{
    ActiveLoad, SchedulerGroup, SchedulerLoad, WorkerId, WorkerWithDpRank,
};
use dynamo_runtime::transports::event_plane::EventEnvelope;

use super::serving_load::{SOURCE_RETENTION, freshness};
use crate::kv_router::sequence::SCHEDULER_LOAD_PUBLISH_INTERVAL;
use crate::local_model::runtime_config::ModelRuntimeConfig;

/// How long a router scheduler's view counts.
const SCHEDULER_FRESHNESS: Duration = freshness(SCHEDULER_LOAD_PUBLISH_INTERVAL);

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
/// Router scheduler views (`active_decode_blocks`, `active_prefill_tokens`) are
/// not part of this snapshot; [`SchedulerLoadState`] aggregates them from the
/// scheduler-load subject, where each view carries its scheduler group.
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
            // active_decode_blocks and active_prefill_tokens on this subject are
            // emitted by router replicas without a scheduler group, so they cannot
            // be aggregated; SchedulerLoadState consumes the scheduler-load subject.
            // Return a distinct accepted outcome so the collector does not
            // misdiagnose an advisory report as an unknown-rank event.
            return LoadObservationOutcome::IgnoredAdvisory;
        };
        self.observations.insert(rank, kv_used_blocks);
        LoadObservationOutcome::Updated
    }

    pub(super) fn declares(&self, rank: &WorkerWithDpRank) -> bool {
        self.capacities.contains_key(rank)
    }

    pub(super) fn declared_ranks(&self) -> usize {
        self.capacities.len()
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
        // TODO(rank-aware-kv-capacity): resolve exact/conservative capacity per rank and carry
        // quality into the pool snapshot. Estimated fallbacks must not count as authoritative
        // rank coverage merely because every rank received a scalar.
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

/// How much of an aggregate's expected sources reported recently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Coverage<T> {
    /// Every expected source reported; the aggregate is exact.
    Complete(T),
    /// Some expected sources reported. A partial sum would read as lower load,
    /// so no aggregate is exposed.
    Partial,
    /// No expected source reported.
    Missing,
}

/// One pool's in-flight load as the routers' schedulers see it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct SchedulerTotals {
    pub(super) active_decode_blocks: u64,
    pub(super) active_prefill_tokens: u64,
}

impl SchedulerTotals {
    fn of(load: &SchedulerLoad) -> Self {
        Self {
            active_decode_blocks: load.active_decode_blocks,
            active_prefill_tokens: load.active_prefill_tokens,
        }
    }

    fn max(self, other: Self) -> Self {
        Self {
            active_decode_blocks: self.active_decode_blocks.max(other.active_decode_blocks),
            active_prefill_tokens: self.active_prefill_tokens.max(other.active_prefill_tokens),
        }
    }

    fn saturating_add(self, other: Self) -> Self {
        Self {
            active_decode_blocks: self
                .active_decode_blocks
                .saturating_add(other.active_decode_blocks),
            active_prefill_tokens: self
                .active_prefill_tokens
                .saturating_add(other.active_prefill_tokens),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SchedulerLoadSnapshot {
    pub(super) coverage: Coverage<SchedulerTotals>,
    /// Oldest publish time among the counted views; 0 when none.
    pub(super) source_observed_ms: u64,
}

#[derive(Debug, Clone)]
struct SchedulerView {
    load: SchedulerLoad,
    published_at_ms: u64,
    received_at: Instant,
}

#[derive(Debug, Clone, Copy)]
struct PublisherProgress {
    last_sequence: u64,
    last_seen: Instant,
}

/// Router scheduler views of one pool's worker ranks, from the scheduler-load
/// subject. Views expire after [`SCHEDULER_FRESHNESS`]; routers republish every
/// view each publish interval, so a lost report is repaired by the next one.
#[derive(Debug, Default)]
pub(super) struct SchedulerLoadState {
    /// Latest view per rank and publishing router.
    views: HashMap<(WorkerWithDpRank, u64), SchedulerView>,
    /// Kept for [`SOURCE_RETENTION`] so duplicate or reordered deliveries from a
    /// briefly silent router are still recognized.
    publishers: HashMap<u64, PublisherProgress>,
}

impl SchedulerLoadState {
    /// Records one view. Duplicate and out-of-order deliveries from a publisher
    /// are ignored; gaps are accepted.
    pub(super) fn observe(
        &mut self,
        envelope: &EventEnvelope,
        load: SchedulerLoad,
        received_at: Instant,
    ) {
        let progress = PublisherProgress {
            last_sequence: envelope.sequence,
            last_seen: received_at,
        };
        match self.publishers.entry(envelope.publisher_id) {
            Entry::Occupied(entry) if envelope.sequence <= entry.get().last_sequence => return,
            Entry::Occupied(mut entry) => {
                entry.insert(progress);
            }
            Entry::Vacant(entry) => {
                entry.insert(progress);
            }
        }
        let rank = WorkerWithDpRank::new(load.worker_id, load.dp_rank);
        self.views.insert(
            (rank, envelope.publisher_id),
            SchedulerView {
                load,
                published_at_ms: envelope.published_at,
                received_at,
            },
        );
    }

    /// Forgets every view, e.g. after the subscription failed and reports may
    /// have been lost. Publisher progress is kept: publisher ids belong to one
    /// publisher incarnation, so resubscribing never rewinds a sequence.
    pub(super) fn clear(&mut self) {
        self.views.clear();
    }

    /// Drops expired views and publishers silent for [`SOURCE_RETENTION`].
    pub(super) fn prune(&mut self, now: Instant) {
        self.views
            .retain(|_, view| is_fresh(view.received_at, SCHEDULER_FRESHNESS, now));
        self.publishers
            .retain(|_, progress| is_fresh(progress.last_seen, SOURCE_RETENTION, now));
    }

    /// Views of one replica group overlap, so each rank takes the group's
    /// maximum. A standalone scheduler that reconnected publishes under a new
    /// publisher id while its previous view is still fresh, so each rank takes
    /// its latest view. Groups are disjoint, so the pool total sums every group
    /// of every rank. A missing scheduler would silently undercount, so the
    /// total is exposed only when every expected scheduler (`discovered` plus
    /// any with a fresh view) has a fresh view of every rank `declared` by the
    /// pool. Failed discovery (`None`) cannot prove that.
    pub(super) fn snapshot(
        &self,
        declared: &PoolLoadState,
        discovered: Option<&HashSet<u64>>,
        now: Instant,
    ) -> SchedulerLoadSnapshot {
        let mut per_group =
            HashMap::<(WorkerWithDpRank, &SchedulerGroup), (SchedulerTotals, u64)>::new();
        let mut ranks_per_publisher = HashMap::<u64, usize>::new();
        let mut oldest = None::<u64>;
        for ((rank, publisher), view) in &self.views {
            if !declared.declares(rank) || !is_fresh(view.received_at, SCHEDULER_FRESHNESS, now) {
                continue;
            }
            *ranks_per_publisher.entry(*publisher).or_default() += 1;
            oldest = Some(oldest.map_or(view.published_at_ms, |at| at.min(view.published_at_ms)));
            let load = SchedulerTotals::of(&view.load);
            let group = &view.load.group;
            per_group
                .entry((*rank, group))
                .and_modify(|(total, latest)| match group {
                    SchedulerGroup::ReplicaGroup { .. } => *total = total.max(load),
                    SchedulerGroup::Standalone { .. } => {
                        if view.published_at_ms > *latest {
                            (*total, *latest) = (load, view.published_at_ms);
                        }
                    }
                })
                .or_insert((load, view.published_at_ms));
        }
        let Some(source_observed_ms) = oldest else {
            return SchedulerLoadSnapshot {
                coverage: Coverage::Missing,
                source_observed_ms: 0,
            };
        };
        let ranks = declared.declared_ranks();
        let complete = discovered.is_some_and(|discovered| {
            discovered
                .iter()
                .chain(ranks_per_publisher.keys())
                .all(|publisher| ranks_per_publisher.get(publisher) == Some(&ranks))
        });
        let coverage = if complete {
            Coverage::Complete(
                per_group
                    .into_values()
                    .map(|(total, _)| total)
                    .fold(SchedulerTotals::default(), SchedulerTotals::saturating_add),
            )
        } else {
            Coverage::Partial
        };
        SchedulerLoadSnapshot {
            coverage,
            source_observed_ms,
        }
    }
}

fn is_fresh(at: Instant, window: Duration, now: Instant) -> bool {
    now.saturating_duration_since(at) <= window
}

/// Worker facts from a pool's runtime configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PoolDeployment {
    pub(super) live_workers: u64,
    /// Sum over workers of `max_num_seqs * data_parallel_size`; `None` while a
    /// worker does not declare `max_num_seqs`.
    pub(super) max_concurrency: Option<u64>,
}

impl PoolDeployment {
    pub(super) fn from_runtime_configs(
        runtime_configs: &HashMap<WorkerId, ModelRuntimeConfig>,
    ) -> Self {
        Self {
            live_workers: runtime_configs.len() as u64,
            max_concurrency: runtime_configs.values().try_fold(0_u64, |total, config| {
                let per_worker = config
                    .max_num_seqs?
                    .saturating_mul(u64::from(config.data_parallel_size));
                Some(total.saturating_add(per_worker))
            }),
        }
    }
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

    fn envelope(publisher_id: u64, sequence: u64, published_at: u64) -> EventEnvelope {
        EventEnvelope {
            publisher_id,
            sequence,
            published_at,
            topic: String::new(),
            payload: Default::default(),
        }
    }

    fn standalone(scheduler_id: u64) -> SchedulerGroup {
        SchedulerGroup::Standalone { scheduler_id }
    }

    fn replicas(group_id: &str) -> SchedulerGroup {
        SchedulerGroup::ReplicaGroup {
            group_id: group_id.to_string(),
        }
    }

    fn view(dp_rank: u32, group: SchedulerGroup, decode: u64, prefill: u64) -> SchedulerLoad {
        SchedulerLoad {
            worker_id: 9,
            dp_rank,
            active_decode_blocks: decode,
            active_prefill_tokens: prefill,
            group,
        }
    }

    /// Delivers `views` from `publisher`, one sequence each, starting at 1.
    fn publish(
        state: &mut SchedulerLoadState,
        publisher: u64,
        views: impl IntoIterator<Item = SchedulerLoad>,
        now: Instant,
    ) {
        for (sequence, load) in (1..).zip(views) {
            state.observe(&envelope(publisher, sequence, publisher * 10), load, now);
        }
    }

    fn ranks(rank_count: u32) -> PoolLoadState {
        PoolLoadState::from_runtime_configs(&HashMap::from([(
            9,
            config(0, rank_count, Some(100), None),
        )]))
        .unwrap()
    }

    fn totals(decode: u64, prefill: u64) -> Coverage<SchedulerTotals> {
        Coverage::Complete(SchedulerTotals {
            active_decode_blocks: decode,
            active_prefill_tokens: prefill,
        })
    }

    #[test]
    fn scheduler_load_takes_the_rank_maximum_within_a_group_and_sums_groups_and_ranks() {
        let now = Instant::now();
        let declared = ranks(2);
        let mut state = SchedulerLoadState::default();
        // Replicas 1 and 2 share group "a"; each lags on something.
        publish(
            &mut state,
            1,
            [view(0, replicas("a"), 12, 30), view(1, replicas("a"), 4, 5)],
            now,
        );
        publish(
            &mut state,
            2,
            [view(0, replicas("a"), 10, 34), view(1, replicas("a"), 6, 7)],
            now,
        );
        // A standalone router only knows the requests it routed itself.
        publish(
            &mut state,
            3,
            [view(0, standalone(3), 1, 2), view(1, standalone(3), 3, 4)],
            now,
        );

        let snapshot = state.snapshot(&declared, Some(&HashSet::from([1, 2, 3])), now);
        assert_eq!(snapshot.coverage, totals(12 + 6 + 1 + 3, 34 + 7 + 2 + 4));
        assert_eq!(snapshot.source_observed_ms, 10);
    }

    #[test]
    fn reconnected_standalone_scheduler_counts_only_its_latest_view() {
        let now = Instant::now();
        let mut state = SchedulerLoadState::default();
        // Router 1 reconnected as publisher 2 while its old view is still fresh.
        publish(&mut state, 1, [view(0, standalone(1), 30, 40)], now);
        publish(&mut state, 2, [view(0, standalone(1), 3, 4)], now);
        assert_eq!(
            state
                .snapshot(&ranks(1), Some(&HashSet::from([2])), now)
                .coverage,
            totals(3, 4)
        );
    }

    #[test]
    fn scheduler_load_is_partial_until_every_expected_scheduler_covers_every_rank() {
        let now = Instant::now();
        let declared = ranks(2);
        let mut state = SchedulerLoadState::default();
        let discovered = HashSet::from([1, 2]);
        assert_eq!(
            state.snapshot(&declared, Some(&discovered), now).coverage,
            Coverage::Missing
        );

        publish(
            &mut state,
            1,
            [view(0, standalone(1), 1, 1), view(1, standalone(1), 1, 1)],
            now,
        );
        // Router 2 is discovered but silent.
        assert_eq!(
            state.snapshot(&declared, Some(&discovered), now).coverage,
            Coverage::Partial
        );
        // Router 2 reports only one of the two ranks.
        publish(&mut state, 2, [view(0, standalone(2), 1, 1)], now);
        assert_eq!(
            state.snapshot(&declared, Some(&discovered), now).coverage,
            Coverage::Partial
        );
        publish(&mut state, 2, [view(1, standalone(2), 1, 1)], now);
        // Sequence 1 again: the duplicate is dropped, so rank 1 is still missing.
        assert_eq!(
            state.snapshot(&declared, Some(&discovered), now).coverage,
            Coverage::Partial
        );
        state.observe(&envelope(2, 2, 20), view(1, standalone(2), 1, 1), now);
        assert_eq!(
            state.snapshot(&declared, Some(&discovered), now).coverage,
            totals(4, 4)
        );
        // Failed discovery cannot prove completeness.
        assert_eq!(
            state.snapshot(&declared, None, now).coverage,
            Coverage::Partial
        );
    }

    #[test]
    fn undiscovered_schedulers_fall_back_to_recent_publishers() {
        // Broker-mode publishers skip discovery registration.
        let now = Instant::now();
        let declared = ranks(1);
        let mut state = SchedulerLoadState::default();
        publish(&mut state, 1, [view(0, standalone(1), 12, 34)], now);
        publish(&mut state, 2, [view(0, standalone(2), 5, 7)], now);
        let nobody = HashSet::new();
        assert_eq!(
            state.snapshot(&declared, Some(&nobody), now).coverage,
            totals(17, 41)
        );

        // A router that stops publishing ages out of the expected set.
        let later = now + SCHEDULER_FRESHNESS + Duration::from_millis(1);
        state.observe(&envelope(1, 2, 30), view(0, standalone(1), 12, 34), later);
        assert_eq!(
            state.snapshot(&declared, Some(&nobody), later).coverage,
            totals(12, 34)
        );
    }

    #[test]
    fn scheduler_views_expire_and_undeclared_ranks_do_not_count() {
        let now = Instant::now();
        let mut state = SchedulerLoadState::default();
        publish(
            &mut state,
            1,
            [
                view(0, standalone(1), 12, 34),
                view(5, standalone(1), 99, 99),
            ],
            now,
        );
        let discovered = HashSet::from([1]);
        assert_eq!(
            state.snapshot(&ranks(1), Some(&discovered), now).coverage,
            totals(12, 34)
        );

        let expired = now + SCHEDULER_FRESHNESS + Duration::from_millis(1);
        assert_eq!(
            state.snapshot(&ranks(1), Some(&discovered), expired),
            SchedulerLoadSnapshot {
                coverage: Coverage::Missing,
                source_observed_ms: 0,
            }
        );

        // Long-silent publishers are forgotten, so a restarted router's
        // sequence space is accepted again.
        let retired = now + SOURCE_RETENTION + Duration::from_secs(1);
        state.prune(retired);
        assert!(state.views.is_empty() && state.publishers.is_empty());
        publish(&mut state, 1, [view(0, standalone(1), 1, 1)], retired);
        assert_eq!(
            state
                .snapshot(&ranks(1), Some(&discovered), retired)
                .coverage,
            totals(1, 1)
        );
        state.clear();
        assert_eq!(
            state
                .snapshot(&ranks(1), Some(&discovered), retired)
                .coverage,
            Coverage::Missing
        );
    }

    #[test]
    fn deployment_counts_workers_and_requires_every_concurrency_limit() {
        let limited = |rank_count| ModelRuntimeConfig {
            max_num_seqs: Some(8),
            ..config(0, rank_count, Some(100), None)
        };
        assert_eq!(
            PoolDeployment::from_runtime_configs(&HashMap::from([
                (9, limited(2)),
                (10, limited(1))
            ])),
            PoolDeployment {
                live_workers: 2,
                max_concurrency: Some(24),
            }
        );
        assert_eq!(
            PoolDeployment::from_runtime_configs(&HashMap::from([
                (9, limited(2)),
                (10, config(0, 1, Some(100), None)),
            ]))
            .max_concurrency,
            None
        );
    }
}
