// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Global Router's policies as router core plugins.
//!
//! Each policy is one [`WorkerPicker`] and no scorers, registered in a
//! [`RouterPluginRegistry`] under its name, which [`policy_config`] then
//! selects for the aggregated worker pool. The core creates one picker per
//! partition, so a picker's cursor is already per model.
//!
//! The core's built-in selector weighs KV overlap and projected load; it has
//! no plain round-robin or fewest-in-flight picker, so [`RoundRobinPicker`]
//! and [`LeastInFlightPicker`] (counting the Global Frontends' requests in
//! flight) are written here. [`HeadroomPicker`] is the affinity rule of
//! [`crate::AffinitySelector`] (`placement::headroom`), reading the same
//! [`PoolSignals`]; the session pin it defaults to comes from the core's
//! session affinity table as the request's affinity target.

use std::collections::HashMap;
use std::sync::Arc;

use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::plugins::worker_selection::{
    ScoredWorkerCandidate, WorkerInputView, WorkerPicker, WorkerSelectionContext,
    WorkerSelectionPolicy, WorkerSelectionPolicyError,
};
use dynamo_kv_router::services::selection::{
    RouterPluginRegistry, WorkerSelectionPolicyFactory, WorkerSelectionPolicyParameters,
    WorkerSelectionPolicyProvider, WorkerSelectionPolicyProviderError,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::{RoutingPartitionRef, WorkerId, WorkerType};

use super::directory::{PoolDirectory, PoolEntry};
use super::{AffinityConfig, CorePolicy, LeastInFlightConfig};
use crate::frontend_signals::FrontendSignals;
use crate::placement::{self, Placed};
use crate::selector::in_flight_load;
use crate::signals::{PoolOccupancy, PoolSignals};

impl CorePolicy {
    /// The name the policy is registered and selected under.
    pub fn name(&self) -> &'static str {
        match self {
            Self::RoundRobin => "global-round-robin",
            Self::LeastInFlight(_) => "global-least-in-flight",
            Self::Affinity(_) => "global-affinity",
        }
    }
}

/// A registry with every policy of [`CorePolicy`]. The least-in-flight and
/// affinity pickers are registered only with their configuration, which
/// `policy` carries.
pub fn registry(
    policy: &CorePolicy,
    directory: Arc<PoolDirectory>,
) -> Result<RouterPluginRegistry, WorkerSelectionPolicyRegistryError> {
    let mut registry = RouterPluginRegistry::default();
    registry.register_worker_selection(
        CorePolicy::RoundRobin.name(),
        provider(RoundRobinPicker::default),
    )?;
    match policy {
        CorePolicy::LeastInFlight(config) => {
            let config = config.clone();
            registry.register_worker_selection(
                policy.name(),
                provider(move || LeastInFlightPicker::new(&config, directory.clone())),
            )?;
        }
        CorePolicy::Affinity(affinity) => {
            let affinity = affinity.clone();
            registry.register_worker_selection(
                policy.name(),
                provider(move || HeadroomPicker::new(&affinity, directory.clone())),
            )?;
        }
        CorePolicy::RoundRobin => {}
    }
    Ok(registry)
}

/// The router policy configuration that selects `policy` for the
/// aggregated pool, as the core reads it from YAML.
pub fn policy_config(policy: &CorePolicy) -> String {
    let name = policy.name();
    format!(
        "worker_selection:\n  aggregated: {name}\n  instances:\n    - name: {name}\n      type: {name}\n"
    )
}

/// A provider for a policy of one picker and no scorers, with a fresh
/// picker from `picker` for each partition. The policies take no
/// parameters.
fn provider<P, F>(picker: F) -> WorkerSelectionPolicyProvider
where
    P: WorkerPicker + 'static,
    F: Fn() -> P + Send + Sync + 'static,
{
    let picker = Arc::new(picker);
    Arc::new(
        move |_parameters: &WorkerSelectionPolicyParameters| -> Result<
            WorkerSelectionPolicyFactory,
            WorkerSelectionPolicyProviderError,
        > {
            let picker = picker.clone();
            Ok(Arc::new(
                move |config: &KvRouterConfig, worker_type: WorkerType, _: RoutingPartitionRef<'_>| {
                    WorkerSelectionPolicy::new(
                        config.clone(),
                        worker_type.as_str(),
                        Vec::new(),
                        Box::new(picker()),
                    )
                },
            ))
        },
    )
}

/// Picks the candidates in turn, in the order they come, so each gets an
/// equal share. When the candidates change, the cursor carries on over the
/// new list, as [`crate::RoundRobinSelector`]'s does.
#[derive(Clone, Debug, Default)]
pub struct RoundRobinPicker {
    turn: usize,
}

impl RoundRobinPicker {
    /// The index in `workers` of the one whose turn it is.
    fn choose(&mut self, workers: impl IntoIterator<Item = WorkerId>) -> Option<usize> {
        let turn = self.turn;
        self.turn = self.turn.wrapping_add(1);
        nth_row(0..workers.into_iter().count(), turn)
    }
}

impl WorkerPicker for RoundRobinPicker {
    fn pick(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let workers = input
            .candidates()
            .iter()
            .map(|candidate| candidate.worker().worker_id);
        self.choose(workers)
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no candidates"))
    }
}

/// Picks the nearest candidate with the fewest requests in flight, with the
/// rule of [`crate::LeastInFlightSelector`] (affinity's placement on
/// requests in flight over each pool's capacity), counting the Global
/// Frontends' own requests ([`FrontendSignals`]). Ties rotate among the tied
/// candidates.
///
/// It reads none of the core's inputs, so the core books nothing for it: no
/// reservation, no sequence tracking, nothing to free and nothing replicated
/// per request. A candidate the directory does not know comes last.
///
/// Its pick counts as pending on the chosen pool until the request takes it
/// over ([`FrontendSignals::take_pending`]): the core runs a model's picks
/// one at a time, so the next pick sees this one before the request is
/// forwarded.
#[derive(Clone)]
pub struct LeastInFlightPicker {
    counts: Arc<FrontendSignals>,
    signals: Arc<PoolSignals>,
    region: String,
    headroom_factor: f64,
    directory: Arc<PoolDirectory>,
    ties: usize,
}

impl LeastInFlightPicker {
    pub fn new(config: &LeastInFlightConfig, directory: Arc<PoolDirectory>) -> Self {
        Self {
            counts: config.counts.clone(),
            signals: config.signals.clone(),
            region: config.region.clone(),
            headroom_factor: config.headroom_factor,
            directory,
            ties: 0,
        }
    }

    /// The index in `workers` of the pool to send the request to.
    fn choose(&mut self, workers: &[WorkerId]) -> Option<usize> {
        let pools = self.directory.snapshot();
        let round_trips = self.signals.region_round_trips();
        // Read each count once: requests start and finish while this runs.
        let mut placed = workers
            .iter()
            .enumerate()
            .map(|(row, worker)| match pools.get(worker) {
                Some(entry) => {
                    let in_flight = self.counts.in_flight(&entry.id);
                    Placed {
                        item: row,
                        load: in_flight_load(in_flight, self.signals.capacity(&entry.id)),
                        distance: placement::distance(&entry.region, &self.region, &round_trips),
                        in_flight,
                    }
                }
                None => Placed {
                    item: row,
                    load: None,
                    distance: None,
                    in_flight: u64::MAX,
                },
            })
            .collect::<Vec<_>>();
        placement::order(&mut placed);
        let tied = placement::leading_ties(&placed);
        if tied == 0 {
            return None;
        }
        let default = self.ties % tied;
        self.ties = self.ties.wrapping_add(1);
        Some(placed[placement::headroom(&placed, default, self.headroom_factor)].item)
    }
}

impl WorkerPicker for LeastInFlightPicker {
    fn pick(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let workers = input
            .candidates()
            .iter()
            .map(|candidate| candidate.worker().worker_id)
            .collect::<Vec<_>>();
        let row = self
            .choose(&workers)
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no candidates"))?;
        self.counts.add_pending(workers[row]);
        Ok(row)
    }
}

/// Keeps each session on one pool and moves it only for a pool with much
/// more room: the rule of [`crate::AffinitySelector`], over the same
/// signals, with the core's session table holding the pin.
///
/// Candidates are ordered nearest region first, then less full, then by
/// worker id; unknown distances and loads come last. The default is the
/// session's bound pool (the request's affinity target), else the first
/// candidate. The request leaves the default only for the first candidate
/// with more than `headroom_factor` times the default's spare capacity; a
/// full default, or one whose load is unknown, leaves for the first
/// candidate with any room, and stays if none has. The core then binds the
/// session to the pool picked (soft affinity), so the pin moves with it.
pub struct HeadroomPicker {
    signals: Arc<PoolSignals>,
    directory: Arc<PoolDirectory>,
    region: String,
    headroom_factor: f64,
}

/// A candidate row, with what the pick log describes.
struct Row<'a> {
    row: usize,
    worker: WorkerId,
    /// The pool, if the directory knows the worker.
    entry: Option<&'a PoolEntry>,
    /// How full the pool is; `None` without a fresh report.
    occupancy: Option<PoolOccupancy>,
}

/// The candidate in one word for the pick log: region, slots and KV
/// occupancy, and the round trip in milliseconds; `?` when unknown.
fn describe(candidate: &Placed<Row<'_>>) -> String {
    let region = candidate
        .item
        .entry
        .map_or("?", |entry| entry.region.as_str());
    let (slots, kv) = match candidate.item.occupancy {
        Some(occupancy) => (
            format!("{:.2}", occupancy.slots),
            occupancy.kv.map_or("?".to_owned(), |kv| format!("{kv:.2}")),
        ),
        None => ("?".to_owned(), "?".to_owned()),
    };
    let rtt = candidate
        .distance
        .map_or("?".to_owned(), |rtt| rtt.as_millis().to_string());
    format!("{region}:slots={slots},kv={kv},rtt={rtt}ms")
}

impl HeadroomPicker {
    pub fn new(config: &AffinityConfig, directory: Arc<PoolDirectory>) -> Self {
        Self {
            signals: config.signals.clone(),
            directory,
            region: config.region.clone(),
            headroom_factor: config.headroom_factor,
        }
    }

    /// The candidates from least to most expensive ([`placement::order`]),
    /// equal ones by worker id.
    fn place<'a>(
        &self,
        pools: &'a HashMap<WorkerId, PoolEntry>,
        rows: &[ScoredWorkerCandidate],
    ) -> Vec<Placed<Row<'a>>> {
        let round_trips = self.signals.region_round_trips();
        let mut placed = rows
            .iter()
            .enumerate()
            .map(|(row, candidate)| {
                let worker = candidate.worker().worker_id;
                let entry = pools.get(&worker);
                let occupancy = entry.and_then(|entry| self.signals.occupancy(&entry.id));
                Placed {
                    item: Row {
                        row,
                        worker,
                        entry,
                        occupancy,
                    },
                    load: occupancy.map(|occupancy| occupancy.load),
                    distance: entry.and_then(|entry| {
                        placement::distance(&entry.region, &self.region, &round_trips)
                    }),
                    in_flight: 0,
                }
            })
            .collect::<Vec<_>>();
        placed.sort_by_key(|candidate| candidate.item.worker);
        placement::order(&mut placed);
        placed
    }
}

impl WorkerPicker for HeadroomPicker {
    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let pools = self.directory.snapshot();
        let placed = self.place(&pools, input.candidates());
        if placed.is_empty() {
            return Err(WorkerSelectionPolicyError::failed("no candidates"));
        }
        let bound = context.affinity_target().map(|target| target.worker_id);
        let default = bound
            .and_then(|worker| {
                placed
                    .iter()
                    .position(|candidate| candidate.item.worker == worker)
            })
            .unwrap_or(0);
        let choice = placement::headroom(&placed, default, self.headroom_factor);
        tracing::trace!(
            candidates = ?placed.iter().map(describe).collect::<Vec<_>>(),
            default = describe(&placed[default]),
            bound = bound.is_some(),
            choice = describe(&placed[choice]),
            "the affinity picker placed the pools"
        );
        Ok(placed[choice].item.row)
    }
}

/// The row that is `turn`th, modulo their number, in the order given: the
/// candidates' order, as the current selectors rotate. `None` without rows.
fn nth_row(rows: impl IntoIterator<Item = usize>, turn: usize) -> Option<usize> {
    let rows = rows.into_iter().collect::<Vec<_>>();
    rows.get(turn % rows.len().max(1)).copied()
}
