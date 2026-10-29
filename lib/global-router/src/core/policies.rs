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
//! no plain "fewest active requests" picker, so [`LeastRequestsPicker`] is
//! written here. [`HeadroomPicker`] is the affinity rule of
//! [`crate::AffinitySelector`] (`placement::headroom`), reading the same
//! [`PoolSignals`]; the session pin it defaults to comes from the core's
//! session affinity table as the request's affinity target.

use std::cmp::Ordering;
use std::sync::Arc;
use std::time::Duration;

use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::plugins::worker_selection::{
    ScoredWorkerCandidate, WorkerInputView, WorkerInputs, WorkerPicker, WorkerSelectionContext,
    WorkerSelectionPolicy, WorkerSelectionPolicyError,
};
use dynamo_kv_router::services::selection::{
    RouterPluginRegistry, WorkerSelectionPolicyFactory, WorkerSelectionPolicyParameters,
    WorkerSelectionPolicyProvider, WorkerSelectionPolicyProviderError,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::{RoutingPartitionRef, WorkerId, WorkerType};

use super::directory::PoolDirectory;
use super::{AffinityConfig, CorePolicy};
use crate::signals::{PoolOccupancy, PoolSignals};

impl CorePolicy {
    /// The name the policy is registered and selected under.
    pub fn name(&self) -> &'static str {
        match self {
            Self::RoundRobin => "global-round-robin",
            Self::LeastRequests => "global-least-requests",
            Self::Affinity(_) => "global-affinity",
        }
    }
}

/// A registry with every policy of [`CorePolicy`]. The affinity picker is
/// registered only with its configuration, which `policy` carries.
pub fn registry(
    policy: &CorePolicy,
    directory: Arc<PoolDirectory>,
) -> Result<RouterPluginRegistry, WorkerSelectionPolicyRegistryError> {
    let mut registry = RouterPluginRegistry::default();
    registry.register_worker_selection(
        CorePolicy::RoundRobin.name(),
        provider(RoundRobinPicker::default),
    )?;
    registry.register_worker_selection(
        CorePolicy::LeastRequests.name(),
        provider(LeastRequestsPicker::default),
    )?;
    if let CorePolicy::Affinity(affinity) = policy {
        let affinity = affinity.clone();
        registry.register_worker_selection(
            policy.name(),
            provider(move || HeadroomPicker::new(&affinity, directory.clone())),
        )?;
    }
    Ok(registry)
}

/// Whether `policy`'s picker reads the core's bookings, as the picker itself
/// declares it: a picker that asks for [`WorkerInputs::LOAD`] ranks by the
/// requests booked on each worker, so every request it places must be
/// booked. Any other picker's picks book nothing.
pub fn reads_bookings(policy: &CorePolicy, directory: &Arc<PoolDirectory>) -> bool {
    let inputs = match policy {
        CorePolicy::RoundRobin => RoundRobinPicker::default().required_worker_inputs(),
        CorePolicy::LeastRequests => LeastRequestsPicker::default().required_worker_inputs(),
        CorePolicy::Affinity(affinity) => {
            HeadroomPicker::new(affinity, directory.clone()).required_worker_inputs()
        }
    };
    inputs.contains(WorkerInputs::LOAD)
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

/// Picks the candidates in turn, in ascending worker id, so each gets an
/// equal share. When the candidates change, the cursor carries on over the
/// new list, as [`crate::RoundRobinSelector`]'s does.
#[derive(Debug, Default)]
pub struct RoundRobinPicker {
    turn: usize,
}

impl WorkerPicker for RoundRobinPicker {
    fn pick(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let rows = input.candidates();
        let turn = self.turn;
        self.turn = self.turn.wrapping_add(1);
        nth_by_worker(rows.iter().enumerate(), turn)
    }
}

/// Picks the candidate with the fewest active requests, counting the core's
/// own bookings, as [`crate::LeastInFlightSelector`] counts this process's
/// requests. Ties rotate among the tied candidates.
#[derive(Debug, Default)]
pub struct LeastRequestsPicker {
    ties: usize,
}

impl WorkerPicker for LeastRequestsPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD
    }

    fn pick(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let rows = input.candidates();
        let loads = input
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load inputs were not supplied"))?;
        let requests = |row: usize| loads.get(row).map_or(0, |load| load.active_requests());
        let fewest = (0..rows.len())
            .map(requests)
            .min()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no candidates"))?;
        let tied = rows
            .iter()
            .enumerate()
            .filter(|(row, _)| requests(*row) == fewest);
        let turn = self.ties;
        self.ties = self.ties.wrapping_add(1);
        nth_by_worker(tied, turn)
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

struct Candidate {
    row: usize,
    worker: WorkerId,
    /// The pool's region, if the directory knows the worker.
    region: Option<String>,
    /// How full the pool is; `None` without a fresh report.
    occupancy: Option<PoolOccupancy>,
    /// Occupancy, 1 being full; `None` without a fresh report.
    load: Option<f64>,
    /// Round trip to the pool's region; zero for this region.
    distance: Option<Duration>,
}

impl Candidate {
    /// The candidate in one word for the pick log: region, slots and KV
    /// occupancy, and the round trip in milliseconds; `?` when unknown.
    fn describe(&self) -> String {
        let region = self.region.as_deref().unwrap_or("?");
        let (slots, kv) = match self.occupancy {
            Some(occupancy) => (
                format!("{:.2}", occupancy.slots),
                occupancy.kv.map_or("?".to_owned(), |kv| format!("{kv:.2}")),
            ),
            None => ("?".to_owned(), "?".to_owned()),
        };
        let rtt = self
            .distance
            .map_or("?".to_owned(), |rtt| rtt.as_millis().to_string());
        format!("{region}:slots={slots},kv={kv},rtt={rtt}ms")
    }
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

    /// The candidates from least to most expensive.
    fn place(&self, rows: &[ScoredWorkerCandidate]) -> Vec<Candidate> {
        let pools = self.directory.snapshot();
        let round_trips = self.signals.region_round_trips();
        let mut placed = rows
            .iter()
            .enumerate()
            .map(|(row, candidate)| {
                let worker = candidate.worker().worker_id;
                let entry = pools.get(&worker);
                let occupancy = entry.and_then(|entry| self.signals.occupancy(&entry.id));
                Candidate {
                    row,
                    worker,
                    region: entry.map(|entry| entry.region.clone()),
                    occupancy,
                    load: occupancy.map(|occupancy| occupancy.load),
                    distance: entry.and_then(|entry| {
                        if entry.region == self.region {
                            Some(Duration::ZERO)
                        } else {
                            round_trips.get(&entry.region).copied()
                        }
                    }),
                }
            })
            .collect::<Vec<_>>();
        placed.sort_by(|left, right| {
            known_first(left.distance, right.distance)
                .then_with(|| known_first_f64(left.load, right.load))
                .then_with(|| left.worker.cmp(&right.worker))
        });
        placed
    }
}

impl WorkerPicker for HeadroomPicker {
    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let placed = self.place(input.candidates());
        let Some(first) = placed.first() else {
            return Err(WorkerSelectionPolicyError::failed("no candidates"));
        };
        let bound = context.affinity_target().map(|target| target.worker_id);
        let default = bound
            .and_then(|worker| placed.iter().find(|candidate| candidate.worker == worker))
            .unwrap_or(first);
        let need = default
            .load
            .map_or(0.0, |load| (self.headroom_factor * (1.0 - load)).max(0.0));
        let choice = placed
            .iter()
            .filter(|candidate| candidate.worker != default.worker)
            .find(|candidate| candidate.load.is_some_and(|load| 1.0 - load > need))
            .unwrap_or(default);
        tracing::debug!(
            candidates = ?placed.iter().map(Candidate::describe).collect::<Vec<_>>(),
            default = default.describe(),
            bound = bound.is_some(),
            need,
            choice = choice.describe(),
            "the affinity picker placed the pools"
        );
        Ok(choice.row)
    }
}

/// Smaller first; unknown last.
fn known_first<T: Ord>(left: Option<T>, right: Option<T>) -> Ordering {
    match (left, right) {
        (Some(left), Some(right)) => left.cmp(&right),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

fn known_first_f64(left: Option<f64>, right: Option<f64>) -> Ordering {
    match (left, right) {
        (Some(left), Some(right)) => left.total_cmp(&right),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

/// The row that is `turn`th, modulo their number, once `rows` are in
/// ascending worker id: a stable order whatever order the core lists them
/// in.
fn nth_by_worker<'a>(
    rows: impl IntoIterator<Item = (usize, &'a ScoredWorkerCandidate)>,
    turn: usize,
) -> Result<usize, WorkerSelectionPolicyError> {
    let mut rows: Vec<(usize, WorkerId)> = rows
        .into_iter()
        .map(|(row, candidate)| (row, candidate.worker().worker_id))
        .collect();
    if rows.is_empty() {
        return Err(WorkerSelectionPolicyError::failed("no candidates"));
    }
    rows.sort_unstable_by_key(|(_, worker)| *worker);
    let (row, _) = rows[turn % rows.len()];
    Ok(row)
}
