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
//! flight) are written here.

use std::sync::Arc;

use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::plugins::worker_selection::{
    WorkerInputView, WorkerPicker, WorkerSelectionContext, WorkerSelectionPolicy,
    WorkerSelectionPolicyError,
};
use dynamo_kv_router::services::selection::{
    RouterPluginRegistry, WorkerSelectionPolicyFactory, WorkerSelectionPolicyParameters,
    WorkerSelectionPolicyProvider, WorkerSelectionPolicyProviderError,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::{RoutingPartitionRef, WorkerId, WorkerType};

use super::directory::PoolDirectory;
use super::{CorePolicy, LeastInFlightConfig};
use crate::frontend_signals::FrontendSignals;
use crate::placement::{self, Placed};
use crate::selector::in_flight_load;
use crate::signals::PoolSignals;

impl CorePolicy {
    /// The name the policy is registered and selected under.
    pub fn name(&self) -> &'static str {
        match self {
            Self::RoundRobin => "global-round-robin",
            Self::LeastInFlight(_) => "global-least-in-flight",
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

/// The row that is `turn`th, modulo their number, in the order given: the
/// candidates' order, as the current selectors rotate. `None` without rows.
fn nth_row(rows: impl IntoIterator<Item = usize>, turn: usize) -> Option<usize> {
    let rows = rows.into_iter().collect::<Vec<_>>();
    rows.get(turn % rows.len().max(1)).copied()
}
